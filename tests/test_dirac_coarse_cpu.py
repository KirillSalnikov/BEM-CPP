"""Opt-in two-level sector preconditioner checks against full FP64 actions."""
import os,sys,unittest
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('BEM_DIRAC_LIBRARY'),'requires selected Dirac CPU library')
class CoarseDiracCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0]=[str(R/'benchmarks/clifford_dielectric_20261115'),str(R/'scripts')]
        from solve import Dirac
        cls.d=Dirac('prism',0,3,index=1.3116+.12j,degree=3,order=24,target_order=13,
                    near_subdiv=1,dense_kernel='simd',block_panels=1,dense_cache='orbits12')
        cls.g=cls.d._compact_geometry
    @classmethod
    def tearDownClass(cls):cls.d.close()
    def test_shared_assembly_parities_and_reuse(self):
        from sector_coarse import CoarseCorrection,coarse_columns
        from dirac_symmetry import solve_modal_batch
        modes=[(0,1,-1),(1,1,-1),(0,2,0),(1,3,2)]
        pre=CoarseCorrection(self.g,3,budget_bytes=64<<20)
        pre.prepare_many(modes)
        self.assertEqual(len(pre.cache),4)
        self.assertLess(max(s['column_error'] for s in pre.statistics.values()),2e-11)
        self.assertLess(max(s['coarse_range_error'] for s in pre.statistics.values()),2e-10)
        before=pre.bytes;elapsed=pre.setup_s
        pre.prepare_many(modes+[(0,3,-1)])
        self.assertEqual(pre.bytes,before);self.assertEqual(pre.setup_s,elapsed)
        f=[self.d.modal_incident(m) for m in modes]
        original,_=solve_modal_batch(self.d,self.g,f,modes)
        new,stats=solve_modal_batch(self.d,self.g,f,modes,preconditioner=pre)
        self.assertLess(np.linalg.norm(new-original)/np.linalg.norm(original),1e-6)
        for h,incident,s in zip(new,f,stats):
            rhs=(incident.reshape(-1,8)*2*self.d.scaling[2]).ravel()
            self.assertLess(np.linalg.norm(self.d.apply_reference(h)-rhs)/np.linalg.norm(rhs),1e-8)
            self.assertEqual(s['sector_coarse_basis'],3)
    def test_budget_and_space_reject_before_allocation(self):
        from sector_coarse import CoarseCorrection
        pre=CoarseCorrection(self.g,1,budget_bytes=1)
        with self.assertRaises(MemoryError):pre.prepare((0,1,0))
        self.assertEqual(pre.bytes,0);self.assertEqual(len(pre.cache),0)
        for invalid in [0,2,11]:
            with self.assertRaises(ValueError):CoarseCorrection(self.g,invalid)
    def test_parallel_gemv_against_blas(self):
        from solve import lib,address,check
        import sector_coarse
        rng=np.random.default_rng(599)
        for nr,nc in [(37,29),(1,1),(19,80),(137,91),(64,70),(129,7)]:
            a=np.ascontiguousarray(rng.normal(size=(nr,nc))+1j*rng.normal(size=(nr,nc)))
            x=np.ascontiguousarray(rng.normal(size=nc)+1j*rng.normal(size=nc))
            y=np.ascontiguousarray(rng.normal(size=nr)+1j*rng.normal(size=nr));expected=y-a@x
            check(lib.dirac_dense_gemv_subtract(nr,nc,address(a),address(x),address(y)))
            self.assertLess(np.linalg.norm(y-expected)/np.linalg.norm(expected),2e-12)
            af=np.asfortranarray(a);y=expected+a@x
            check(lib.dirac_dense_gemv_subtract_f(nr,nc,address(af),address(x),address(y)))
            self.assertLess(np.linalg.norm(y-expected)/np.linalg.norm(expected),2e-12)
        self.assertNotEqual(lib.dirac_dense_gemv_subtract(0,nc,address(a),address(x),address(y)),0)
        self.assertNotEqual(lib.dirac_dense_gemv_subtract_f(0,nc,address(a),address(x),address(y)),0)

    def test_sector_eviction_preserves_operator_and_bounds_live_cache(self):
        from sector_coarse import CoarseCorrection
        modes=[(0,1,-1),(1,1,-1),(0,2,0),(1,3,2)]
        reference=CoarseCorrection(self.g,3,kernel='blas');reference.prepare_many(modes)
        pre=CoarseCorrection(self.g,3)
        plan=pre.memory_plan()
        pre.budget_bytes=sum(plan[k] for k in ('persistent_bytes','workspace_bytes','panel_factors_bytes'))
        with self.assertRaises(MemoryError):pre.prepare_many(modes)
        self.assertEqual(pre.bytes,0)
        rng=np.random.default_rng(118);x=rng.normal(size=self.g.size)+1j*rng.normal(size=self.g.size)
        for mode in modes:
            pre.release_except([mode]);pre.prepare(mode)
            self.assertEqual(len(pre.cache),1)
            self.assertEqual(pre.bytes,plan['persistent_bytes'])
            self.assertTrue(pre.cache[pre.key(mode)][0].flags.f_contiguous)
            np.testing.assert_allclose(pre.apply(x,mode),reference.apply(x,mode),rtol=2e-12,atol=2e-12)
        self.assertEqual(pre.peak_bytes,plan['persistent_bytes'])
        pre.release_except();self.assertEqual(pre.bytes,0);self.assertEqual(pre.evictions,4)
if __name__=='__main__':unittest.main()
