"""Two-level P3/P1 correction using bounded local blocks and incomplete LU.

Only the preconditioner drops far couplings. The GMRES operator and final
full-space FMM residual remain unchanged. No dense Ac or stored W.
"""
import ctypes as ct
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import spilu
from solve import lib,address,check
from sector_coarse_fmm import MatrixFreeCoarse

class SparseCoarse(MatrixFreeCoarse):
    def __init__(self,g,coarse_basis=3,budget_bytes=4<<30,radius=1.5,drop_tol=1e-3,fill_factor=4):
        if not np.isfinite(radius) or not 0<radius<=8:raise ValueError('coarse radius must be in (0,8]')
        if not 0<drop_tol<1 or not np.isfinite(fill_factor) or not 1<=fill_factor<=20:raise ValueError('ILU drop/fill controls')
        super().__init__(g,coarse_basis,budget_bytes)
        lib.dirac_direct_panel_pairs.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]
        self.radius=radius;self.drop_tol=drop_tol;self.fill_factor=fill_factor
        self.sizes={};d=self.low;lg=self.low_geometry;reps=lg.representatives
        vertices=np.empty((d.count,3,3));lib.dirac_vertices(d.p,address(vertices))
        diam=np.max(np.linalg.norm(vertices-np.roll(vertices,1,axis=1),axis=2),axis=1)
        tree=cKDTree(d.xyz);targets=[];sources=[];offsets=[0]
        for row in reps:
            candidates=np.asarray(sorted(tree.query_ball_point(d.xyz[row],radius*diam.max()*(1+1e-12))),dtype=np.int32)
            distance=np.linalg.norm(d.xyz[candidates]-d.xyz[row],axis=1)
            keep=candidates[distance<=radius*np.maximum(diam[row],diam[candidates])*(1+1e-12)]
            targets.extend([row]*len(keep));sources.extend(keep.tolist());offsets.append(len(sources))
            if 16*len(sources)*(coarse_basis*8)**2>budget_bytes:raise MemoryError('sparse panel blocks alone exceed budget')
        self.targets=np.asarray(targets,np.int32);self.sources=np.asarray(sources,np.int32);self.offsets=np.asarray(offsets,np.int32)
        self.orbit=np.empty(d.count,np.int32);self.group=np.empty(d.count,np.int32)
        for j,t in enumerate(lg.transforms):self.orbit[t.image[reps]]=np.arange(len(reps));self.group[t.image[reps]]=j

    def memory_plan(self):
        nc=len(self.indices);width=self.coarse_basis*8;nnz=len(self.sources)*width**2
        pattern=sum(x.nbytes for x in (self.targets,self.sources,self.offsets,self.orbit,self.group))
        # SuperLU's fill_factor is an upper fill target, not a process RSS cap.
        # Reserve matrix/BSR conversions and an additional factor-sized workspace.
        factors=int((20*nnz+8*nc)*self.fill_factor)+16*nc+8
        workspace=32*nnz+2*factors+16*(14*self.g.d.size+20*self.g.size)+(32<<20)
        workspace+=sum(t.basis.nbytes+t.components.nbytes+t.image.nbytes for t in self.low_geometry.transforms)
        return dict(persistent_bytes=factors,workspace_bytes=workspace+pattern,
            panel_factors_bytes=self.factors.nbytes+self.pivots.nbytes,stored_W_bytes=0,
            dense_Ac_bytes=0,assembly='local exact panel pairs with symmetry phases; sparse ILU',
            block_pairs=len(self.sources),upper_matrix_nnz=nnz,fill_factor=self.fill_factor,
            scope='ILU fill and workspace planning reserve, actual factor.nnz checked before caching')

    def matrix(self,mode):
        d=self.low;lg=self.low_geometry;w=self.coarse_basis*8;p,l,m=mode
        hp=(-1)**(p+l+m+1);data=np.empty((len(self.sources),w,w),complex)
        for begin in range(0,len(data),256):
            end=min(begin+256,len(data));target=self.targets[begin:end];source=self.sources[begin:end]
            block=data[begin:end]
            check(lib.dirac_direct_panel_pairs(d.p,len(source),address(target),address(source),address(block)))
            groups=self.group[source];reps=lg.representatives[self.orbit[source]]
            for group in np.unique(groups):
                mask=groups==group;t=lg.transforms[group];ids=reps[mask]
                change=np.einsum('pab,pcd->pacbd',t.basis[ids],t.components[ids]).reshape(len(ids),w,w)
                phase=np.exp(2j*np.pi*(m%lg.order)*(group%lg.order)/lg.order)*(hp if group>=lg.order else 1)
                block[mask]=phase*(block[mask]@change)
        # sum_duplicates compacts BSR indptr in place. Keep the geometric pair
        # offsets immutable across sectors and repeated assemblies.
        matrix=bsr_matrix((data,self.orbit[self.sources],self.offsets.copy()),shape=(len(self.indices),)*2)
        matrix.sum_duplicates();matrix=matrix.tocsc();matrix.data[abs(matrix.data)<1e-15]=0;matrix.eliminate_zeros()
        return matrix

    def solve_coarse(self,x,mode):return self.cache[self.key(mode)].solve(x)

    def release_except(self,modes=()):
        keep={self.key(mode) for mode in modes}
        for key in list(self.cache):
            if key not in keep:
                del self.cache[key];self.bytes-=self.sizes.pop(key);self.evictions+=1

    def prepare_many(self,modes,audit=True):
        missing={self.key(m):m for m in modes if self.key(m) not in self.cache}
        if not missing:return
        plan=self.memory_plan();required=self.bytes+len(missing)*plan['persistent_bytes']+plan['workspace_bytes']+plan['panel_factors_bytes']
        if required>self.budget_bytes:raise MemoryError('sparse coarse factors and workspace exceed budget')
        start=time.perf_counter()
        for key,mode in missing.items():
            begin=time.perf_counter();matrix=self.matrix(mode);assembly=time.perf_counter()-begin
            factor=spilu(matrix,drop_tol=self.drop_tol,fill_factor=self.fill_factor,permc_spec='COLAMD')
            size=20*factor.nnz+16*len(self.indices)+8
            if size>plan['persistent_bytes']:raise MemoryError('actual sparse factor exceeded planned fill budget')
            error=None
            if audit:
                rng=np.random.default_rng(21);q=rng.normal(size=matrix.shape[0])+1j*rng.normal(size=matrix.shape[0])
                rhs=matrix@q;answer=factor.solve(rhs)
                error=float(np.linalg.norm(matrix@answer-rhs)/np.linalg.norm(rhs))
                if not np.isfinite(answer).all() or not np.isfinite(error):raise RuntimeError('nonfinite sparse coarse inverse')
            self.cache[key]=factor;self.sizes[key]=size;self.bytes+=size
            self.statistics[str(key)]=dict(setup_s=time.perf_counter()-begin,assembly_s=assembly,
                coarse_size=len(self.indices),matrix_nnz=matrix.nnz,factor_nnz=factor.nnz,bytes=size,
                local_inverse_residual=error,approximate_coarse=True,bytes_scope='estimated complex/index/pointer/permutation payload from SuperLU nnz',radius=self.radius,drop_tol=self.drop_tol,fill_factor=self.fill_factor,
                stored_W_bytes=0,dense_Ac_bytes=0)
        self.peak_bytes=max(self.peak_bytes,self.bytes);self.setup_s+=time.perf_counter()-start
