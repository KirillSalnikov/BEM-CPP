"""Exact assembly oracles and strict solves for bounded coarse preconditioners."""
import os,sys,unittest
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('BEM_DIRAC_LIBRARY'),'requires current Dirac library and FMM3D')
class HierarchyCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0]=[str(R/'benchmarks/clifford_dielectric_20261115'),str(R/'scripts')]
        from sector_fmm import SectorFMM
        cls.d=SectorFMM(ref=0,ka=3,index=1.3116+.12j,degree=3,near_subdiv=1,eps=1e-8)

    @classmethod
    def tearDownClass(cls):cls.d.close()

    def test_sparse_all_pairs_equal_dense_in_multiple_sectors(self):
        from sector_coarse_sparse import SparseCoarse
        from sector_coarse import coarse_columns
        c=SparseCoarse(self.d._compact_geometry,3,radius=8)
        offsets=c.offsets.copy()
        for mode in [(0,2,-1),(1,2,-1),(1,3,2),(0,2,-1)]:
            sparse=c.matrix(mode).toarray();dense=coarse_columns(c.low_geometry,mode,3)
            self.assertLess(np.linalg.norm(sparse-dense)/np.linalg.norm(dense),2e-12)
            np.testing.assert_array_equal(c.offsets,offsets)
        c.close()

    def test_global_transfer_columns_range_and_sector_eviction(self):
        from sector_coarse_hierarchy import HierarchicalCoarse
        from sector_coarse import coarse_columns
        c=HierarchicalCoarse(self.d._compact_geometry,3,radius=1,aggregate_groups=3,aggregate_basis=3)
        np.testing.assert_allclose((c.R@c.P).toarray(),np.eye(c.ng),atol=2e-14)
        for mode in [(0,2,-1),(1,2,-1)]:
            dense=coarse_columns(c.low_geometry,mode,3)
            actual=c.global_columns(mode);expected=dense@c.P
            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),2e-12)
            c.release_except([mode]);c.prepare(mode)
            self.assertEqual(len(c.cache),1);self.assertEqual(len(c.global_cache),1)
            self.assertLess(c.statistics[str(c.key(mode))]['global_range_error'],1e-8)
        c.release_except();self.assertEqual(c.bytes,0);self.assertEqual(len(c.global_cache),0);c.close()

    def test_both_polarizations_against_strict_full_operator(self):
        from sector_coarse_hierarchy import HierarchicalCoarse
        from dirac_symmetry import solve_modal_batch
        c=HierarchicalCoarse(self.d._compact_geometry,3,radius=1.5,aggregate_groups=3)
        modal=[(0,2,-1),(1,1,-1)];incidents=[self.d.modal_incident(m) for m in modal]
        h,stats=solve_modal_batch(self.d,self.d._compact_geometry,incidents,modal,preconditioner=c)
        for v,inc,s in zip(h,incidents,stats):
            rhs=(inc.reshape(-1,8)*2*self.d.scaling[2]).ravel()
            self.assertLess(np.linalg.norm(self.d.apply_reference(v)-rhs)/np.linalg.norm(rhs),1.01e-8)
            self.assertEqual(s['info'],0)
        c.close()

    def test_small_budget_rejects_before_coarse_allocation(self):
        from sector_coarse_hierarchy import HierarchicalCoarse
        c=HierarchicalCoarse(self.d._compact_geometry,3,radius=1.5,aggregate_groups=3)
        c.budget_bytes=1
        with self.assertRaises(MemoryError):c.prepare((0,2,-1))
        self.assertEqual(c.bytes,0);self.assertFalse(c.cache);self.assertFalse(c.global_cache)
        c.close()

    def test_outer_full_P1_matches_exact_coarse_and_uses_no_FMM(self):
        from sector_coarse_hierarchy import HierarchicalCoarse
        from sector_coarse_fmm import MatrixFreeCoarse
        g=self.d._compact_geometry;mode=(0,2,-1)
        c=HierarchicalCoarse(g,3,radius=1.5,aggregate_groups=len(g.representatives),aggregate_basis=3,outer=True)
        exact=MatrixFreeCoarse(g,3);c.prepare(mode);exact.prepare(mode)
        rng=np.random.default_rng(140);x=rng.normal(size=g.size)+1j*rng.normal(size=g.size)
        expected=exact.apply(x,mode);before=self.d.calls
        actual=c.apply(x,mode)
        self.assertEqual(self.d.calls,before)
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),2e-9)
        c.close();exact.close()

    def test_projection_does_not_copy_the_full_correction(self):
        from sector_coarse_hierarchy import HierarchicalCoarse
        c=HierarchicalCoarse(self.d._compact_geometry,3,radius=1.5,aggregate_groups=3,aggregate_basis=3,outer=True)
        rng=np.random.default_rng(19);az=np.asfortranarray(rng.normal(size=(c.g.size,c.ng))+1j*rng.normal(size=(c.g.size,c.ng)))
        expected=c.restriction@az;original=c.restriction
        class Bounded:
            def __matmul__(self,x):
                if x.shape[1]>32:raise AssertionError('full correction copy would be made')
                return original@x
        c.restriction=Bounded();actual=c.project_global(az)
        self.assertTrue(actual.flags.f_contiguous)
        np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=2e-14);c.close()

    def test_close_releases_panel_factors_without_waiting_for_gc(self):
        import weakref
        from solve import Dirac
        d=Dirac('prism',0,1,degree=1,target_order=13,dense_kernel='simd',block_panels=1)
        factors=weakref.ref(d._dense_factors[0])
        self.assertIsNotNone(factors())
        d.close()
        self.assertIsNone(factors())
        self.assertIsNone(d._dense_inverse)
        self.assertIsNone(d._compact_geometry)
        d.close()  # Idempotent cleanup after successful or interrupted work.

if __name__=='__main__':unittest.main()
