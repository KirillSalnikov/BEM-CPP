"""Same two-level formula, storing only Ac LU; B A Z q is applied via FMM."""
import time
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from solve import Dirac,lib,address
from dirac_symmetry import CyclicGeometry
from sector_coarse import CoarseCorrection,coarse_columns

class MatrixFreeCoarse(CoarseCorrection):
    def __init__(self,g,coarse_basis=3,budget_bytes=4<<30):
        if not getattr(g.d,'_matrix_free',False):raise ValueError('matrix-free coarse requires sector FMM')
        if coarse_basis not in (1,3):raise ValueError('P0/P1 coarse space required')
        super().__init__(g,coarse_basis,budget_bytes)
        self.low=Dirac(**g.d.parameters,degree={1:0,3:1}[coarse_basis],geometry_only=True)
        self.low_geometry=CyclicGeometry(self.low,order=g.order,horizontal=g.horizontal)
        if not np.array_equal(self.low_geometry.representatives,g.representatives):raise ValueError('coarse and fine panel maps differ')
        self.action_s=0.;self.applications=0

    def close(self):self.low.close()

    def memory_plan(self):
        nc=len(self.indices);nf=self.g.size
        geometry=sum(t.basis.nbytes+t.components.nbytes+t.image.nbytes for t in self.low_geometry.transforms)
        return dict(persistent_bytes=16*nc*nc+4*nc,
            workspace_bytes=max(3*16*nc*self.coarse_basis*8*8,16*(14*self.g.d.size+20*nf))+geometry,
            assembly_chunk=8,panel_factors_bytes=self.factors.nbytes+self.pivots.nbytes,
            stored_W_bytes=0,assembly='direct exact coarse tiles, no pair cache')

    def release_except(self,modes=()):
        keep={self.key(mode) for mode in modes}
        for key in list(self.cache):
            if key in keep:continue
            factor=self.cache.pop(key);self.bytes-=factor[0].nbytes+factor[1].nbytes;self.evictions+=1

    def prepare_many(self,modes,audit=True):
        missing={self.key(mode):mode for mode in modes if self.key(mode) not in self.cache}
        if not missing:return
        plan=self.memory_plan();required=self.bytes+len(missing)*plan['persistent_bytes']+plan['workspace_bytes']+plan['panel_factors_bytes']
        if required>self.budget_bytes:raise MemoryError('matrix-free coarse LU and workspace exceed budget')
        begin=time.perf_counter()
        for key,mode in missing.items():
            t=time.perf_counter();matrix=coarse_columns(self.low_geometry,mode,self.coarse_basis,plan['assembly_chunk']);assembly=time.perf_counter()-t
            column_error=range_error=None
            if audit:
                rng=np.random.default_rng(867);q=rng.normal(size=len(self.indices))+1j*rng.normal(size=len(self.indices))
                z=np.zeros(self.g.size,complex);z[self.indices]=q
                exact=self.g.restrict(self.g.d.apply_reference(self.g.expand_mode(z,mode)))
                column_error=float(np.linalg.norm(matrix@q-exact[self.indices])/np.linalg.norm(exact[self.indices]))
                if column_error>max(2e-11,100*self.g.d.strict_eps):raise RuntimeError('direct coarse matrix fails strict FMM projection audit')
            if not np.isfinite(matrix).all():raise RuntimeError('nonfinite coarse matrix')
            factor=lu_factor(matrix,overwrite_a=True,check_finite=False);self.cache[key]=factor
            size=factor[0].nbytes+factor[1].nbytes;self.bytes+=size
            if audit:
                range_error=float(np.linalg.norm(self.apply(exact,mode)-z)/np.linalg.norm(z))
                if range_error>max(2e-10,100*self.g.d.eps):raise RuntimeError('matrix-free coarse range audit failed')
            self.statistics[str(key)]=dict(setup_s=time.perf_counter()-t,assembly_s=assembly,
                fine_size=self.g.size,coarse_size=len(self.indices),bytes=size,stored_W_bytes=0,
                column_error=column_error,coarse_range_error=range_error)
        self.peak_bytes=max(self.peak_bytes,self.bytes);self.setup_s+=time.perf_counter()-begin

    def apply(self,x,mode):return self.apply_batch(np.asarray(x)[None,:],[mode])[0]

    def solve_coarse(self,x,mode):
        return lu_solve(self.cache[self.key(mode)],x,check_finite=False)

    def apply_batch(self,x,modes):
        begin=time.perf_counter();x=np.ascontiguousarray(x,complex)
        z=np.zeros_like(x)
        for j,mode in enumerate(modes):z[j,self.indices]=self.solve_coarse(x[j,self.indices],mode)
        full=np.ascontiguousarray([self.g.expand_mode(v,m) for v,m in zip(z,modes)])
        az=self.g.d.apply_sector_batch(full);residual=np.ascontiguousarray(x-az);out=np.empty_like(x)
        for j in range(len(x)):
            lib.dirac_block_lu(len(self.g.representatives),self.width,address(self.factors),address(self.pivots),address(residual[j]),address(out[j]))
        out+=z;self.action_s+=time.perf_counter()-begin;self.applications+=len(x);return out
