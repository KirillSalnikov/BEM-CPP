"""Exact low-polynomial coarse columns; two-level correction without BEM calls."""
import ctypes as ct
import time
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from solve import lib,address,check

lib.dirac_submatrix.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p]
if hasattr(lib,'dirac_dense_gemv_subtract'):
    lib.dirac_dense_gemv_subtract.argtypes=[ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]
if hasattr(lib,'dirac_dense_gemv_subtract_f'):
    lib.dirac_dense_gemv_subtract_f.argtypes=[ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]
if hasattr(lib,'dirac_transformed_columns'):
    lib.dirac_transformed_columns.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]

def coarse_columns_many(g,modes,coarse_basis=1,chunk=8):
    d=g.d;nb=d.basis_count;reps=g.representatives
    if coarse_basis not in (1,3,6,10) or coarse_basis>nb:raise ValueError('coarse polynomial space')
    outputs=[np.zeros((g.size,len(reps)*coarse_basis*8),complex,order='F') for _ in modes]
    phases=[]
    for p,l,m in modes:
        phases.append([np.exp(2j*np.pi*(m%g.order)*(j%g.order)/g.order)*
                       ((-1)**(p+l+m+1) if j>=g.order else 1) for j in range(g.group_size)])
    for start in range(0,len(reps),chunk):
        chosen=reps[start:start+chunk];end=start+len(chosen)
        views=[out[:,start*coarse_basis*8:end*coarse_basis*8] for out in outputs]
        for j,t in enumerate(g.transforms):
            # The complete degree-0/1/2/3 polynomial space is invariant under
            # triangle vertex permutations; no higher modes enter its lift.
            assert np.max(abs(t.basis[chosen,coarse_basis:,:coarse_basis]),initial=0)<2e-12
            if getattr(d,'_geometry_only',False):
                from sector_fmm import lib as _typed_library
                ids=np.ascontiguousarray(t.image[chosen],np.int32);targets=np.ascontiguousarray(reps,np.int32)
                basis=np.ascontiguousarray(t.basis[chosen,:coarse_basis,:coarse_basis]);components=np.ascontiguousarray(t.components[chosen])
                if coarse_basis!=d.basis_count:raise ValueError('direct coarse geometry must match requested basis')
                lifted=np.empty(views[0].shape,complex)
                check(lib.dirac_direct_transformed_columns(d.p,len(targets),address(targets),len(ids),address(ids),address(basis),address(components),address(lifted)))
            elif hasattr(lib,'dirac_transformed_columns'):
                ids=np.ascontiguousarray(t.image[chosen],np.int32)
                basis=np.ascontiguousarray(t.basis[chosen,:coarse_basis,:coarse_basis])
                components=np.ascontiguousarray(t.components[chosen])
                lifted=np.empty(views[0].shape,complex)
                check(lib.dirac_transformed_columns(d.p,len(g.rows),address(g.rows),len(ids),address(ids),coarse_basis,address(basis),address(components),address(lifted)))
            else:
                columns=np.ascontiguousarray((t.image[chosen,None]*nb+np.arange(coarse_basis)).ravel(),np.int32)
                block=np.empty((g.size,len(columns)*8),complex)
                check(lib.dirac_submatrix(d.p,len(g.rows),address(g.rows),len(columns),address(columns),address(block)))
                lifted=np.einsum('ijbc,jba,jcd->ijad',block.reshape(g.size,len(chosen),coarse_basis,8),
                                 t.basis[chosen,:coarse_basis,:coarse_basis],t.components[chosen],optimize=True)
            lifted=np.asfortranarray(lifted.reshape(views[0].shape))
            for view,phase in zip(views,phases):view+=phase[j]*lifted
    return outputs

def coarse_columns(g,mode,coarse_basis=1,chunk=8):
    return coarse_columns_many(g,[mode],coarse_basis,chunk)[0]

class CoarseCorrection:
    def __init__(self,g,coarse_basis=1,budget_bytes=8<<30,kernel='simd',workspace_bytes=256<<20):
        if coarse_basis not in (1,3,6,10) or coarse_basis>g.d.basis_count:raise ValueError('coarse polynomial space')
        if budget_bytes<0:raise ValueError('coarse cache budget must be nonnegative')
        self.g=g;self.coarse_basis=coarse_basis;self.cache={};self.statistics={}
        self.budget_bytes=budget_bytes;self.bytes=0;self.setup_s=0.;self.peak_bytes=0;self.evictions=0
        self.workspace_bytes=int(workspace_bytes)
        if self.workspace_bytes<=0:raise ValueError('coarse assembly workspace must be positive')
        if kernel not in ('blas','simd'):raise ValueError('coarse correction kernel')
        self.kernel=kernel
        if kernel=='simd' and not hasattr(lib,'dirac_dense_gemv_subtract_f'):raise RuntimeError('selected library lacks column-major coarse SIMD action')
        self.factors=np.ascontiguousarray(g.d._dense_factors[0][g.representatives])
        self.pivots=np.ascontiguousarray(g.d._dense_factors[1][g.representatives])
        self.width=g.d.basis_count*8
        self.indices=np.arange(g.size).reshape(-1,g.d.basis_count,8)[:,:coarse_basis].ravel()

    def memory_plan(self):
        nc=len(self.indices);nf=self.g.size
        # Three lifted/phase tile buffers, bounded independently of sector size.
        column_tile_bytes=3*16*nf*self.coarse_basis*8
        chunk=min(8,max(1,self.workspace_bytes//column_tile_bytes))
        assembly=column_tile_bytes*chunk
        # Full-space reference audit vectors/transforms and panel LU scratch.
        finish=16*(12*self.g.d.size+12*nf+3*nc+self.width*nc)
        return dict(persistent_bytes=16*(nf*nc+nc*nc)+4*nc,
                    workspace_bytes=max(assembly,finish),assembly_chunk=chunk,
                    panel_factors_bytes=self.factors.nbytes+self.pivots.nbytes)

    def release_except(self,modes=()):
        """Call only between completed GMRES batches; no disk spill or rebuild loop."""
        keep={self.key(mode) for mode in modes}
        for key in list(self.cache):
            if key in keep:continue
            matrix,factor=self.cache.pop(key)
            self.bytes-=matrix.nbytes+factor[0].nbytes+factor[1].nbytes;self.evictions+=1

    def key(self,mode):
        p,l,m=mode
        return m%self.g.order,(-1)**(p+l+m+1) if self.g.horizontal else 1

    def prepare(self,mode,audit=True):
        self.prepare_many([mode],audit)

    def prepare_many(self,modes,audit=True):
        missing={self.key(mode):mode for mode in modes if self.key(mode) not in self.cache}
        if not missing:return
        plan=self.memory_plan();required=plan['persistent_bytes']
        peak=self.bytes+required*len(missing)+plan['workspace_bytes']+plan['panel_factors_bytes']
        if peak>self.budget_bytes:raise MemoryError('coarse cache and assembly workspace exceed budget')
        start=time.perf_counter()
        # Finish one sector before allocating the next. Never retain the list
        # of all AZ arrays alongside all transformed W arrays.
        for key,mode in missing.items():
            begin=time.perf_counter()
            az=coarse_columns(self.g,mode,self.coarse_basis,chunk=plan['assembly_chunk'])
            self._finish(key,mode,az,time.perf_counter()-begin,audit)
            del az
        self.peak_bytes=max(self.peak_bytes,self.bytes)
        self.setup_s+=time.perf_counter()-start

    def _finish(self,key,mode,az,assembled,audit):
        start=time.perf_counter();nc=len(self.indices)
        coarse=np.empty((nc,nc),complex,order='F')
        # Advanced indexing followed by order conversion used to allocate two
        # complete Ac arrays. Copy panel rows directly into the final LU buffer.
        cbw=self.coarse_basis*8
        for i in range(len(self.g.representatives)):
            coarse[i*cbw:(i+1)*cbw]=az[i*self.width:i*self.width+cbw]
        if not np.isfinite(coarse).all():raise RuntimeError('nonfinite coarse matrix')
        factor=lu_factor(coarse,overwrite_a=True,check_finite=False)
        audit_error=None
        if audit:
            rng=np.random.default_rng(867);q=rng.normal(size=nc)+1j*rng.normal(size=nc)
            lifted=np.zeros(self.g.size,complex);lifted[self.indices]=q
            exact=self.g.restrict(self.g.d.apply_reference(self.g.expand_mode(lifted,mode)))
            audit_error=float(np.linalg.norm(az@q-exact)/np.linalg.norm(exact))
            if audit_error>2e-11:raise RuntimeError('coarse columns fail independent full-space action')
            coarse_action=az@q
        # M = B + (Z - B A Z) (R A Z)^-1 R. No full BEM action per inverse.
        for i in range(len(self.g.representatives)):
            sl=slice(i*self.width,(i+1)*self.width)
            az[sl]=lu_solve((self.factors[i],self.pivots[i]),np.array(az[sl],order='F'),overwrite_b=True)
        az[self.indices,np.arange(nc)]-=1.
        self.cache[key]=(az,factor);self.bytes+=az.nbytes+factor[0].nbytes+factor[1].nbytes
        coarse_exactness=None
        if audit:
            coarse_exactness=float(np.linalg.norm(self.apply(coarse_action,mode)-lifted)/np.linalg.norm(lifted))
            if coarse_exactness>2e-10:raise RuntimeError('two-level inverse fails coarse range exactness')
        self.statistics[str(key)]=dict(setup_s=time.perf_counter()-start+assembled,assembly_s=assembled,
             fine_size=self.g.size,coarse_size=nc,bytes=az.nbytes+factor[0].nbytes+factor[1].nbytes,
             column_error=audit_error,coarse_range_error=coarse_exactness)

    def apply(self,x,mode):
        matrix,factor=self.cache[self.key(mode)]
        x=np.ascontiguousarray(x);out=np.empty_like(x)
        lib.dirac_block_lu(len(self.g.representatives),self.width,address(self.factors),address(self.pivots),address(x),address(out))
        q=lu_solve(factor,x[self.indices],check_finite=False)
        if self.kernel=='blas':out-=matrix@q
        else:check(lib.dirac_dense_gemv_subtract_f(len(x),len(q),address(matrix),address(np.ascontiguousarray(q)),address(out)))
        return out
