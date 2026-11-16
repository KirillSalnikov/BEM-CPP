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
    def __init__(self,*args,aggregate_groups=64,aggregate_basis=1,outer=False,**kwargs):
        if not isinstance(aggregate_groups,int) or not 1<=aggregate_groups<=512:raise ValueError('aggregate cap must be in 1..512')
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
        self.low_diagonal=None
        if self.outer:
            cbw=8*self.coarse_basis;unit=np.eye(self.width,dtype=complex)[:,:cbw]
            self.low_diagonal=np.asarray([lu_solve((f,p),unit,check_finite=False)[:cbw] for f,p in zip(self.factors,self.pivots)])

    def memory_plan(self):
        plan=super().memory_plan();nc=len(self.indices)
        rows=self.g.size if self.outer else nc
        extra=16*(rows*self.ng+self.ng**2)+4*self.ng
        plan['persistent_bytes']+=extra
        plan['workspace_bytes']+=16*(2*self.ng**2+rows*640)+3*sum(x.nbytes for x in (self.P.data,self.P.indices,self.P.indptr,self.patch,self.rotation,self.weights,self.Pblocks))
        if self.low_diagonal is not None:plan['workspace_bytes']+=self.low_diagonal.nbytes
        plan.update(aggregate_count=self.ng//(8*self.aggregate_basis),aggregate_basis=self.aggregate_basis,global_columns=self.ng,global_correction_bytes=extra,
                    stored_W_bytes=16*rows*self.ng,stored_fine_W_bytes=16*rows*self.ng if self.outer else 0,dense_Ac_bytes=16*self.ng**2,
                    global_space_is_full_coarse=self.ng==nc,dense_P1_Ac_bytes=16*self.ng**2 if self.ng==nc and self.coarse_basis==3 else 0,outer_correction=self.outer,
                    assembly='local sparse ILU plus exact bounded-column global polynomial patches',
                    asymptotic_scope='O(N) factor/correction planning storage at fixed radius, fill and aggregate cap; assembly time still quadratic')
        return plan

    def global_columns(self,mode):
        lg=self.g if self.outer else self.low_geometry;d=lg.d;reps=np.ascontiguousarray(lg.representatives,np.int32)
        p,l,m=mode;hp=(-1)**(p+l+m+1);nc=lg.size;cb=self.aggregate_basis;gw=8*cb
        az=np.zeros((nc,self.ng),complex,order='F')
        for begin in range(0,len(reps),8):
            end=min(begin+8,len(reps));chosen=reps[begin:end];width=gw*len(chosen)
            block=np.zeros((nc,width),complex)
            for j,t in enumerate(lg.transforms):
                sources=np.ascontiguousarray(t.image[chosen],np.int32)
                basis=np.ascontiguousarray(t.basis[chosen,:cb,:cb]);components=np.ascontiguousarray(t.components[chosen])
                part=np.empty_like(block)
                check(lib.dirac_direct_transformed_columns_basis(d.p,len(reps),address(reps),len(sources),address(sources),cb,address(basis),address(components),address(part)))
                phase=np.exp(2j*np.pi*(m%lg.order)*(j%lg.order)/lg.order)*(hp if j>=lg.order else 1)
                block+=phase*part
            for j,k in enumerate(range(begin,end)):
                dest=slice(gw*self.patch[k],gw*self.patch[k]+gw)
                az[:,dest]+=block[:,gw*j:gw*j+gw]@self.Pblocks[k,:gw]
        return az

    def project_global(self,az,columns=32):
        # SciPy CSR @ multivector calls other.ravel() in C order. Applied to
        # the complete Fortran AZ this copies the entire multi-GiB correction.
        # Restrict bounded RHS tiles directly into the final Fortran LU buffer.
        matrix=np.empty((self.ng,self.ng),complex,order='F')
        for first in range(0,self.ng,columns):
            last=min(first+columns,self.ng)
            matrix[:,first:last]=self.restriction@az[:,first:last]
        return matrix

    def solve_coarse(self,x,mode):
        y=super().solve_coarse(x,mode)
        if self.outer or self.key(mode) not in self.global_cache:return y
        w,factor=self.global_cache[self.key(mode)];q=lu_solve(factor,self.R@x,check_finite=False)
        check(lib.dirac_dense_gemv_subtract_f(len(y),len(q),address(w),address(np.ascontiguousarray(q)),address(y)))
        return y

    def smooth(self,x,mode):
        x=np.ascontiguousarray(x);out=np.empty_like(x)
        lib.dirac_block_lu(len(self.g.representatives),self.width,address(self.factors),address(self.pivots),address(x),address(out))
        low=x[self.indices];local=np.einsum('pab,pb->pa',self.low_diagonal,low.reshape(len(self.g.representatives),-1),optimize=True).ravel()
        out[self.indices]+=self.cache[self.key(mode)].solve(low)-local
        return out

    def apply_batch(self,x,modes):
        if not self.outer:return super().apply_batch(x,modes)
        begin=time.perf_counter();out=np.empty_like(x)
        for j,mode in enumerate(modes):
            out[j]=self.smooth(x[j],mode);w,factor=self.global_cache[self.key(mode)]
            q=lu_solve(factor,self.restriction@x[j],check_finite=False)
            check(lib.dirac_dense_gemv_subtract_f(out.shape[1],len(q),address(w),address(np.ascontiguousarray(q)),address(out[j])))
        self.action_s+=time.perf_counter()-begin;self.applications+=len(x);return out

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
        super().prepare_many(list(missing.values()),audit)
        start=time.perf_counter()
        for key,mode in missing.items():
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
            factor=lu_factor(matrix,overwrite_a=True,check_finite=False)
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
                global_columns=self.ng,global_correction_bytes=size,global_column_error=error,global_range_error=range_error,
                stored_W_bytes=az.nbytes,stored_fine_W_bytes=az.nbytes if self.outer else 0,dense_Ac_bytes=factor[0].nbytes,
                global_space_is_full_coarse=self.ng==len(self.indices),dense_P1_Ac_bytes=factor[0].nbytes if self.ng==len(self.indices) and self.coarse_basis==3 else 0,outer_correction=self.outer)
            self.statistics[str(key)]['bytes']+=size
        self.peak_bytes=max(self.peak_bytes,self.bytes);self.setup_s+=time.perf_counter()-start
