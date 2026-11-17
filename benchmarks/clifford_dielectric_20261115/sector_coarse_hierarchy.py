"""P3 -> local P1 ILU -> bounded global P0/P1 patch correction.

Global patch count is capped explicitly. Its correction has
8*aggregate_basis*patch_count columns on the coarse or full level:
O(N) storage for a fixed cap. Full-level application needs no FMM call.
"""
import ctypes as ct
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.linalg import lu_factor,lu_solve
from solve import lib,address,check
from sector_coarse_sparse import SparseCoarse

class HierarchicalCoarse(SparseCoarse):
    def __init__(self,*args,aggregate_groups=64,aggregate_basis=1,outer=False,storage='fp64',lu_threads=1,batch_correction=False,eliminate_full_coarse=False,**kwargs):
        # ka40/ref4 has 992 representative panels. The former 512 cap
        # prevented a full P1 correction even when compressed storage fits.
        # The existing allocation budget still governs preparation.
        if not isinstance(aggregate_groups,int) or not 1<=aggregate_groups<=4096:raise ValueError('aggregate cap must be in 1..4096')
        if storage not in ('fp64','fp32'):raise ValueError('global correction storage must be fp64 or fp32')
        if storage=='fp32' and not hasattr(lib,'dirac_dense_gemv_subtract_cf'):raise RuntimeError('rebuild Dirac library for compressed coarse correction')
        self.storage=storage;self.storage_dtype=np.dtype('complex64' if storage=='fp32' else 'complex128')
        self.batch_correction=bool(batch_correction)
        if self.batch_correction:
            if not outer or storage!='fp32':raise ValueError('batched correction requires full-level FP32 storage')
            if not hasattr(lib,'dirac_dense_gemm_subtract_cf'):raise RuntimeError('rebuild Dirac library for batched coarse correction')
            lib.dirac_dense_gemm_subtract_cf.argtypes=[ct.c_int,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]
        if not isinstance(lu_threads,int) or not 1<=lu_threads<=64:raise ValueError('coarse LU threads must be in 1..64')
        self.lu_threads=lu_threads;self.blas_limits=None
        if lu_threads>1:
            try:from threadpoolctl import threadpool_limits
            except ImportError as e:raise RuntimeError('parallel coarse LU requires optional threadpoolctl') from e
            self.blas_limits=threadpool_limits
        if storage=='fp32':lib.dirac_dense_gemv_subtract_cf.argtypes=[ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]
        super().__init__(*args,**kwargs)
        if aggregate_basis not in (1,3) or aggregate_basis>self.coarse_basis:raise ValueError('global patch basis must be P0/P1 within the coarse space')
        lib.dirac_direct_transformed_columns_basis.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]
        self.aggregate_groups=aggregate_groups;self.aggregate_basis=aggregate_basis;self.outer=bool(outer);self.global_cache={};self.global_sizes={}
        g=self.low_geometry;d=self.low;reps=g.representatives
        # Never put two different physical faces in one patch. Recursively
        # bisect the longest spatial extent within each face.
        normals=np.round(d.frame[reps,0],10)
        _,face=np.unique(normals,axis=0,return_inverse=True)
        groups=[np.flatnonzero(face==i) for i in range(face.max()+1)]
        if len(groups)>aggregate_groups:raise ValueError('aggregate cap smaller than representative face count')
        while len(groups)<min(aggregate_groups,len(reps)):
            eligible=[j for j,v in enumerate(groups) if len(v)>1]
            if not eligible:break
            j=max(eligible,key=lambda k:np.max(np.ptp(d.xyz[reps[groups[k]]],axis=0)))
            ids=groups.pop(j);axis=np.argmax(np.ptp(d.xyz[reps[ids]],axis=0))
            ids=ids[np.argsort(d.xyz[reps[ids],axis],kind='stable')];mid=len(ids)//2
            groups.extend([ids[:mid],ids[mid:]])
        self.patch=np.empty(len(reps),np.int32);self.rotation=np.zeros((len(reps),8,8));self.weights=np.empty(len(reps))
        for j,ids in enumerate(groups):
            self.patch[ids]=j;self.weights[ids]=1/np.sqrt(len(ids))
            rotation=d.frame[reps[ids]]@d.frame[reps[ids[0]]].T
            self.rotation[ids,0,0]=1;self.rotation[ids,4,4]=1
            self.rotation[ids,1:4,1:4]=rotation;self.rotation[ids,5:8,5:8]=rotation
        coefficients=np.zeros((len(reps),self.coarse_basis,aggregate_basis))
        if aggregate_basis==1:coefficients[:,0,0]=self.weights
        else:
            from dirac_symmetry import polynomial
            vertices=np.empty((d.count,3,3));lib.dirac_vertices(d.p,address(vertices))
            inverse=np.linalg.inv(polynomial(np.eye(3))[:,:3])
            for ids in groups:
                anchor=reps[ids[0]];center=d.xyz[reps[ids]].mean(axis=0)
                coordinates=(vertices[reps[ids]]-center)@d.frame[anchor,1:3].T
                values=np.concatenate([np.ones((*coordinates.shape[:2],1)),coordinates],axis=2)
                local=inverse@values
                orthogonal,_=np.linalg.qr(local.reshape(-1,3),mode='reduced')
                coefficients[ids]=orthogonal.reshape(len(ids),3,3)
        self.ng=8*aggregate_basis*len(groups);width=self.coarse_basis*8;gw=8*aggregate_basis
        self.Pblocks=np.einsum('pab,pcd->pacbd',coefficients,self.rotation).reshape(len(reps),width,gw)
        rows=np.broadcast_to(np.arange(len(reps))[:,None,None]*width+np.arange(width)[None,:,None],self.Pblocks.shape).ravel()
        cols=np.broadcast_to(self.patch[:,None,None]*gw+np.arange(gw)[None,None,:],self.Pblocks.shape).ravel()
        self.P=coo_matrix((self.Pblocks.ravel(),(rows,cols)),shape=(len(self.indices),self.ng)).tocsc()
        self.P.eliminate_zeros();self.R=self.P.conjugate().T
        p=self.P.tocoo()
        self.transfer=coo_matrix((p.data,(self.indices[p.row],p.col)),shape=(self.g.size,self.ng)).tocsc() if self.outer else self.P
        self.restriction=self.transfer.conjugate().T
        # With a complete orthonormal low space Z, R=Z*, Ac=R A Z,
        # the residual r=x-AZ Ac^-1 R x has zero low coefficients. Thus
        # Mx = B_panel r + Z Ac^-1 R x: the low ILU cancels exactly.
        # Store only (AZ)_high and apply the panel inverse AFTER subtraction,
        # retaining its high-to-low coupling. No sparse factor or BAZ needed.
        self.eliminate_full_coarse=bool(eliminate_full_coarse)
        if self.eliminate_full_coarse and not (self.outer and storage=='fp32' and self.ng==len(self.indices) and self.g.size>self.ng):
            raise ValueError('coarse elimination requires full low space, full-level FP32 storage, and higher fine modes')
        self.correction_rows=np.setdiff1d(np.arange(self.g.size),self.indices) if self.eliminate_full_coarse else None
        self.low_diagonal=None
        if self.outer and not self.eliminate_full_coarse:
            cbw=8*self.coarse_basis;unit=np.eye(self.width,dtype=complex)[:,:cbw]
            self.low_diagonal=np.asarray([lu_solve((f,p),unit,check_finite=False)[:cbw] for f,p in zip(self.factors,self.pivots)])

    def memory_plan(self):
        plan=super().memory_plan();nc=len(self.indices)
        rows=self.g.size if self.outer else nc;assembly_rows=rows
        if self.eliminate_full_coarse:
            rows=len(self.correction_rows);plan['persistent_bytes']=0
        w_bytes=self.storage_dtype.itemsize*rows*self.ng
        extra=w_bytes+16*self.ng**2+4*self.ng
        plan['persistent_bytes']+=extra
        row_reserve=704 if self.storage=='fp32' else 640
        # The streamed FP32 path creates one Fortran Ac, already included in
        # persistent storage, and factor_global enforces its in-place LU.
        # No second/third ng**2 array coexists with it. Reserve BLAS workspace
        # and the existing bounded assembly/conversion tiles separately.
        lu_work=max(128<<20,self.lu_threads*(32<<20)) if self.storage=='fp32' else 32*self.ng**2
        plan['workspace_bytes']+=lu_work+16*assembly_rows*row_reserve+3*sum(x.nbytes for x in (self.P.data,self.P.indices,self.P.indptr,self.patch,self.rotation,self.weights,self.Pblocks))
        if self.low_diagonal is not None:plan['workspace_bytes']+=self.low_diagonal.nbytes
        plan.update(aggregate_count=self.ng//(8*self.aggregate_basis),aggregate_basis=self.aggregate_basis,global_columns=self.ng,global_correction_bytes=extra,
                    stored_W_bytes=w_bytes,stored_fine_W_bytes=w_bytes if self.outer else 0,dense_Ac_bytes=16*self.ng**2,
                    global_storage=self.storage,global_assembly_streamed=self.storage=='fp32',global_lu_threads=self.lu_threads,
                    global_lu_inplace_required=self.storage=='fp32',global_lu_workspace_reserve_bytes=lu_work,
                    global_space_is_full_coarse=self.ng==nc,dense_P1_Ac_bytes=16*self.ng**2 if self.ng==nc and self.coarse_basis==3 else 0,outer_correction=self.outer,
                    eliminated_full_coarse=self.eliminate_full_coarse,stored_correction_rows=rows,
                    assembly=('exact low-space LU plus high AZ rows; redundant ILU eliminated' if self.eliminate_full_coarse else 'local sparse ILU plus exact bounded-column global polynomial patches'),
                    asymptotic_scope='O(N) factor/correction planning storage at fixed radius, fill and aggregate cap; assembly time still quadratic')
        return plan

    def global_columns(self,mode):
        return next(self.global_column_tiles(mode,patches=self.ng//(8*self.aggregate_basis)))[1]

    def global_column_tiles(self,mode,patches=8):
        """Exact FP64 AZ in bounded patch-column tiles; no full AZ allocation."""
        if patches<1:raise ValueError('patch tile must be positive')
        lg=self.g if self.outer else self.low_geometry;d=lg.d;reps=np.ascontiguousarray(lg.representatives,np.int32)
        p,l,m=mode;hp=(-1)**(p+l+m+1);nc=lg.size;cb=self.aggregate_basis;gw=8*cb
        for first in range(0,self.ng//gw,patches):
            last=min(first+patches,self.ng//gw)
            selected=np.flatnonzero((self.patch>=first)&(self.patch<last))
            az=np.zeros((nc,(last-first)*gw),complex,order='F')
            for begin in range(0,len(selected),8):
                ids=selected[begin:begin+8];chosen=reps[ids];width=gw*len(chosen)
                block=np.zeros((nc,width),complex)
                for j,t in enumerate(lg.transforms):
                    sources=np.ascontiguousarray(t.image[chosen],np.int32)
                    basis=np.ascontiguousarray(t.basis[chosen,:cb,:cb]);components=np.ascontiguousarray(t.components[chosen])
                    part=np.empty_like(block)
                    check(lib.dirac_direct_transformed_columns_basis(d.p,len(reps),address(reps),len(sources),address(sources),cb,address(basis),address(components),address(part)))
                    phase=np.exp(2j*np.pi*(m%lg.order)*(j%lg.order)/lg.order)*(hp if j>=lg.order else 1)
                    part*=phase;block+=part
                for j,k in enumerate(ids):
                    dest=slice(gw*(self.patch[k]-first),gw*(self.patch[k]-first)+gw)
                    az[:,dest]+=block[:,gw*j:gw*j+gw]@self.Pblocks[k,:gw]
            yield first*gw,az

    def project_global(self,az,columns=32):
        # SciPy CSR @ multivector calls other.ravel() in C order. Applied to
        # the complete Fortran AZ this copies the entire multi-GiB correction.
        # Restrict bounded RHS tiles directly into the final Fortran LU buffer.
        matrix=np.empty((self.ng,self.ng),complex,order='F')
        for first in range(0,self.ng,columns):
            last=min(first+columns,self.ng)
            matrix[:,first:last]=self.restriction@az[:,first:last]
        return matrix

    def factor_global(self,matrix):
        """Temporarily use BLAS threads for LU only, outside any OpenMP team."""
        if self.storage=='fp32' and (matrix.dtype!=np.complex128 or not matrix.flags.f_contiguous or not matrix.flags.writeable):
            raise ValueError('streamed coarse LU requires writable Fortran complex128 Ac')
        begin=time.perf_counter();threads=self.lu_threads if matrix.shape[0]>=1024 else 1
        if threads>1:
            with self.blas_limits(limits=threads,user_api='blas'):
                factor=lu_factor(matrix,overwrite_a=True,check_finite=False)
        else:factor=lu_factor(matrix,overwrite_a=True,check_finite=False)
        if self.storage=='fp32' and not np.shares_memory(matrix,factor[0]):
            raise MemoryError('streamed coarse LU copied Ac outside its allocation contract')
        return factor,time.perf_counter()-begin,threads

    def solve_coarse(self,x,mode):
        y=super().solve_coarse(x,mode)
        if self.outer or self.key(mode) not in self.global_cache:return y
        w,factor=self.global_cache[self.key(mode)];q=lu_solve(factor,self.R@x,check_finite=False)
        self.subtract_correction(w,q,y)
        return y

    def subtract_correction(self,w,q,out):
        apply=lib.dirac_dense_gemv_subtract_cf if w.dtype==np.complex64 else lib.dirac_dense_gemv_subtract_f
        check(apply(len(out),len(q),address(w),address(np.ascontiguousarray(q)),address(out)))

    def smooth(self,x,mode):
        x=np.ascontiguousarray(x);out=np.empty_like(x)
        lib.dirac_block_lu(len(self.g.representatives),self.width,address(self.factors),address(self.pivots),address(x),address(out))
        if self.eliminate_full_coarse:return out
        low=x[self.indices];local=np.einsum('pab,pb->pa',self.low_diagonal,low.reshape(len(self.g.representatives),-1),optimize=True).ravel()
        out[self.indices]+=self.cache[self.key(mode)].solve(low)-local
        return out

    def apply_batch(self,x,modes):
        if not self.outer:return super().apply_batch(x,modes)
        if self.eliminate_full_coarse:return self.apply_eliminated(x,modes)
        if self.batch_correction:
            x=np.ascontiguousarray(x,dtype=complex)
            if x.shape!=(len(modes),self.g.size) or not 1<=len(modes)<=16:raise ValueError('coarse correction batch dimensions')
            begin=time.perf_counter();out=np.empty_like(x);groups={}
            for j,mode in enumerate(modes):
                out[j]=self.smooth(x[j],mode)
                groups.setdefault(self.key(mode),[]).append(j)
            for key,indices in groups.items():
                w,factor=self.global_cache[key]
                rhs=np.asfortranarray(self.restriction@x[indices].T)
                q=np.ascontiguousarray(lu_solve(factor,rhs,check_finite=False).T)
                local=np.ascontiguousarray(out[indices])
                check(lib.dirac_dense_gemm_subtract_cf(w.shape[0],self.ng,len(indices),address(w),address(q),address(local)))
                out[indices]=local
            self.action_s+=time.perf_counter()-begin;self.applications+=len(x)
            return out
        begin=time.perf_counter();out=np.empty_like(x)
        for j,mode in enumerate(modes):
            out[j]=self.smooth(x[j],mode);w,factor=self.global_cache[self.key(mode)]
            q=lu_solve(factor,self.restriction@x[j],check_finite=False)
            self.subtract_correction(w,q,out[j])
        self.action_s+=time.perf_counter()-begin;self.applications+=len(x);return out

    def apply_eliminated(self,x,modes):
        """Full low space: B_panel (I-AZ Ac^-1 R)x + Z Ac^-1 R x."""
        x=np.ascontiguousarray(x,dtype=complex)
        if x.shape!=(len(modes),self.g.size) or not 1<=len(modes)<=16:raise ValueError('eliminated correction batch dimensions')
        begin=time.perf_counter();out=np.empty_like(x);groups={}
        for j,mode in enumerate(modes):groups.setdefault(self.key(mode),[]).append(j)
        for key,indices in groups.items():
            w,factor=self.global_cache[key]
            rhs=np.asfortranarray(self.restriction@x[indices].T)
            q=np.ascontiguousarray(lu_solve(factor,rhs,check_finite=False).T)
            high=np.ascontiguousarray(x[indices][:,self.correction_rows])
            if self.batch_correction:
                check(lib.dirac_dense_gemm_subtract_cf(w.shape[0],self.ng,len(indices),address(w),address(q),address(high)))
            else:
                for j in range(len(indices)):self.subtract_correction(w,q[j],high[j])
            residual=np.zeros(self.g.size,complex)
            for j,dest in enumerate(indices):
                residual[self.correction_rows]=high[j]
                lib.dirac_block_lu(len(self.g.representatives),self.width,address(self.factors),address(self.pivots),address(residual),address(out[dest]))
                out[dest,self.indices]+=self.P@q[j]
        self.action_s+=time.perf_counter()-begin;self.applications+=len(x)
        return out

    def release_except(self,modes=()):
        keep={self.key(mode) for mode in modes}
        for key in list(self.global_cache):
            if key not in keep:
                del self.global_cache[key];self.bytes-=self.global_sizes.pop(key)
        super().release_except(modes)

    def prepare_many(self,modes,audit=True):
        if any(self.key(m) in self.cache and self.key(m) not in self.global_cache for m in modes):
            raise RuntimeError('incomplete global correction from a failed preparation; recreate preconditioner')
        missing={self.key(m):m for m in modes if self.key(m) not in self.cache}
        if not missing:return
        if self.eliminate_full_coarse:
            plan=self.memory_plan()
            if self.bytes+len(missing)*plan['persistent_bytes']+plan['workspace_bytes']+plan['panel_factors_bytes']>self.budget_bytes:
                raise MemoryError('eliminated coarse factors and workspace exceed budget')
            for key in missing:
                self.cache[key]=None;self.sizes[key]=0
                self.statistics[str(key)]=dict(bytes=0,setup_s=0.,local_ILU_eliminated=True)
        else:super().prepare_many(list(missing.values()),audit)
        start=time.perf_counter()
        for key,mode in missing.items():
            if self.storage=='fp32':
                self.prepare_compressed(key,mode,audit)
                continue
            begin=time.perf_counter();az=self.global_columns(mode);assembly=time.perf_counter()-begin
            rng=np.random.default_rng(47);q=rng.normal(size=self.ng)+1j*rng.normal(size=self.ng)
            lifted=self.transfer@q;aq=az@q;error=None
            if audit:
                if self.outer:z=lifted
                else:z=np.zeros(self.g.size,complex);z[self.indices]=lifted
                exact=self.g.restrict(self.g.d.apply_reference(self.g.expand_mode(z,mode)))
                if not self.outer:exact=exact[self.indices]
                error=float(np.linalg.norm(aq-exact)/np.linalg.norm(exact))
                if error>max(2e-11,100*self.g.d.strict_eps):raise RuntimeError('global patch columns fail independent full-space action')
            matrix=self.project_global(az)
            factor,lu_seconds,lu_threads=self.factor_global(matrix)
            # In-place C*AZ - P in small RHS tiles; global rank is capped.
            for first in range(0,self.ng,16):
                if self.outer:
                    last=min(first+16,self.ng);tile=np.empty((self.g.size,last-first),complex,order='F')
                    for j in range(last-first):
                        lib.dirac_block_lu(len(self.g.representatives),self.width,address(self.factors),address(self.pivots),address(az[:,first+j]),address(tile[:,j]))
                    low=np.asfortranarray(az[self.indices,first:last])
                    local=np.einsum('pab,pbr->par',self.low_diagonal,low.reshape(len(self.g.representatives),8*self.coarse_basis,last-first),optimize=True).reshape(low.shape)
                    tile[self.indices]+=self.cache[key].solve(low)-local;az[:,first:last]=tile
                else:az[:,first:first+16]=self.cache[key].solve(np.asfortranarray(az[:,first:first+16]))
            for j in range(self.ng):
                lo,hi=self.transfer.indptr[j:j+2];az[self.transfer.indices[lo:hi],j]-=self.transfer.data[lo:hi]
            self.global_cache[key]=(az,factor)
            size=az.nbytes+factor[0].nbytes+factor[1].nbytes;self.global_sizes[key]=size;self.bytes+=size
            range_error=float(np.linalg.norm((self.apply(aq,mode) if self.outer else self.solve_coarse(aq,mode))-lifted)/np.linalg.norm(lifted)) if audit else None
            if range_error is not None and (not np.isfinite(range_error) or range_error>1e-8):raise RuntimeError('global patch correction range audit failed')
            self.statistics[str(key)].update(global_setup_s=time.perf_counter()-begin,global_assembly_s=assembly,
                global_lu_s=lu_seconds,global_lu_threads=lu_threads,
                global_columns=self.ng,global_correction_bytes=size,global_column_error=error,global_range_error=range_error,
                stored_W_bytes=az.nbytes,stored_fine_W_bytes=az.nbytes if self.outer else 0,dense_Ac_bytes=factor[0].nbytes,
                global_space_is_full_coarse=self.ng==len(self.indices),dense_P1_Ac_bytes=factor[0].nbytes if self.ng==len(self.indices) and self.coarse_basis==3 else 0,outer_correction=self.outer)
            self.statistics[str(key)]['bytes']+=size
        self.peak_bytes=max(self.peak_bytes,self.bytes);self.setup_s+=time.perf_counter()-start

    def prepare_compressed(self,key,mode,audit):
        """Compute exact AZ/Ac and smoother in FP64, store only W in FP32."""
        begin=time.perf_counter();rows=self.g.size if self.outer else len(self.indices)
        w=np.empty((len(self.correction_rows) if self.eliminate_full_coarse else rows,self.ng),np.complex64,order='F')
        matrix=np.empty((self.ng,self.ng),complex,order='F')
        rng=np.random.default_rng(47);q=rng.normal(size=self.ng)+1j*rng.normal(size=self.ng)
        aq=np.zeros(rows,complex);assembly=0.;conversion_error2=0.;exact_w_norm2=0.
        tiles=self.global_column_tiles(mode)
        while True:
            t=time.perf_counter()
            try:first,az=next(tiles)
            except StopIteration:break
            assembly+=time.perf_counter()-t;last=first+az.shape[1]
            # Projection must precede smoothing/quantization. Restrict at most
            # 32 RHS at once; never cast the complete multi-GiB W to FP64.
            for col in range(0,az.shape[1],32):
                end=min(col+32,az.shape[1]);matrix[:,first+col:first+end]=self.restriction@az[:,col:end]
            if audit:aq+=az@q[first:last]
            for col in range(0,az.shape[1],16):
                end=min(col+16,az.shape[1])
                if self.eliminate_full_coarse:
                    tile=np.asfortranarray(az[self.correction_rows,col:end])
                elif self.outer:
                    tile=np.empty((rows,end-col),complex,order='F')
                    for j in range(end-col):
                        lib.dirac_block_lu(len(self.g.representatives),self.width,address(self.factors),address(self.pivots),address(az[:,col+j]),address(tile[:,j]))
                    low=np.asfortranarray(az[self.indices,col:end])
                    local=np.einsum('pab,pbr->par',self.low_diagonal,low.reshape(len(self.g.representatives),8*self.coarse_basis,end-col),optimize=True).reshape(low.shape)
                    tile[self.indices]+=self.cache[key].solve(low)-local
                else:tile=self.cache[key].solve(np.asfortranarray(az[:,col:end]))
                if not self.eliminate_full_coarse:
                    for j in range(end-col):
                        lo,hi=self.transfer.indptr[first+col+j:first+col+j+2]
                        tile[self.transfer.indices[lo:hi],j]-=self.transfer.data[lo:hi]
                if not np.isfinite(tile).all():raise RuntimeError('nonfinite global correction before FP32 storage')
                w[:,first+col:first+end]=tile
                stored=w[:,first+col:first+end]
                if not np.isfinite(stored).all():raise RuntimeError('global correction overflows FP32 storage')
                # Bounded FP64 roundtrip audit of the actual stored coefficients.
                exact_w_norm2+=float(np.vdot(tile,tile).real)
                delta=tile-stored;conversion_error2+=float(np.vdot(delta,delta).real)
            del az
        lifted=self.transfer@q;error=None
        if audit:
            if self.outer:z=lifted
            else:z=np.zeros(self.g.size,complex);z[self.indices]=lifted
            exact=self.g.restrict(self.g.d.apply_reference(self.g.expand_mode(z,mode)))
            if not self.outer:exact=exact[self.indices]
            error=float(np.linalg.norm(aq-exact)/np.linalg.norm(exact))
            if error>max(2e-11,100*self.g.d.strict_eps):raise RuntimeError('streamed global columns fail independent full-space action')
        if not np.isfinite(matrix).all():raise RuntimeError('nonfinite projected global matrix')
        factor,lu_seconds,lu_threads=self.factor_global(matrix)
        self.global_cache[key]=(w,factor)
        size=w.nbytes+factor[0].nbytes+factor[1].nbytes;self.global_sizes[key]=size;self.bytes+=size
        range_error=float(np.linalg.norm((self.apply(aq,mode) if self.outer else self.solve_coarse(aq,mode))-lifted)/np.linalg.norm(lifted)) if audit else None
        # Only the approximate preconditioner loses FP64 range exactness.
        # The GMRES operator and every final strict residual keep their tolerances.
        range_limit=5e-6
        if range_error is not None and (not np.isfinite(range_error) or range_error>range_limit):
            # Keep the sparse-only state visibly incomplete after a failed
            # audit, so a retry cannot silently reuse the rejected correction.
            del self.global_cache[key];self.bytes-=self.global_sizes.pop(key)
            raise RuntimeError('compressed global correction range audit failed')
        self.statistics[str(key)].update(global_setup_s=time.perf_counter()-begin,global_assembly_s=assembly,
            global_lu_s=lu_seconds,global_lu_threads=lu_threads,
            global_columns=self.ng,global_correction_bytes=size,global_column_error=error,global_range_error=range_error,
            global_range_limit=range_limit,global_storage=self.storage,global_assembly_streamed=True,
            storage_relative_error=float(np.sqrt(conversion_error2/max(exact_w_norm2,np.finfo(float).tiny))),
            stored_W_bytes=w.nbytes,stored_fine_W_bytes=w.nbytes if self.outer else 0,dense_Ac_bytes=factor[0].nbytes,
            global_space_is_full_coarse=self.ng==len(self.indices),dense_P1_Ac_bytes=factor[0].nbytes if self.ng==len(self.indices) and self.coarse_basis==3 else 0,outer_correction=self.outer)
        self.statistics[str(key)]['bytes']+=size
