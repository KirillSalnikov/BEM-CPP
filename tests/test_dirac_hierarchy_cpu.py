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

    def test_full_p1_above_former_cap_keeps_memory_guard(self):
        from sector_coarse_hierarchy import HierarchicalCoarse
        c=HierarchicalCoarse(self.d._compact_geometry,3,radius=1.5,
            aggregate_groups=1024,aggregate_basis=3,outer=True,storage='fp32')
        try:
            self.assertEqual(c.ng,len(c.indices))
            self.assertTrue(c.memory_plan()['global_space_is_full_coarse'])
            np.testing.assert_allclose((c.R@c.P).toarray(),np.eye(c.ng),atol=2e-14)
            c.budget_bytes=1
            with self.assertRaises(MemoryError):c.prepare((0,2,-1))
            self.assertFalse(c.cache)
            self.assertFalse(c.global_cache)
        finally:c.close()

    def test_dense_LU_controls_do_not_reset_hierarchical_LU(self):
        from sector_coarse_hierarchy import HierarchicalCoarse
        from sector_coarse import CoarseCorrection
        c=HierarchicalCoarse(self.d._compact_geometry,3,aggregate_groups=3,lu_threads=4)
        try:
            self.assertEqual(c.lu_threads,4)
            self.assertIsNotNone(c.blas_limits)
            dense=CoarseCorrection(self.d._compact_geometry,3,lu_threads=4)
            self.assertEqual(dense.dense_lu_threads,4)
            self.assertIsNotNone(dense.dense_blas_limits)
        finally:c.close()

    def test_full_low_space_elimination_matches_exact_inverse(self):
        from sector_coarse_hierarchy import HierarchicalCoarse
        from dirac_symmetry import solve_modal_batch
        from unittest.mock import patch
        g=self.d._compact_geometry;modes=[(0,2,-1),(1,1,-1)]
        options=dict(aggregate_groups=1024,aggregate_basis=3,outer=True)
        exact=HierarchicalCoarse(g,3,**options)
        reduced=HierarchicalCoarse(g,3,storage='fp32',batch_correction=True,eliminate_full_coarse=True,**options)
        try:
            exact.prepare_many(modes)
            with patch.object(reduced,'matrix',side_effect=AssertionError('redundant sparse ILU built')):
                reduced.prepare_many(modes)
            rng=np.random.default_rng(142);x=rng.normal(size=(2,g.size))+1j*rng.normal(size=(2,g.size))
            expected=exact.apply_batch(x,modes);before=self.d.calls;actual=reduced.apply_batch(x,modes)
            self.assertEqual(self.d.calls,before)
            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),1e-6)
            for key,(w,factor) in reduced.global_cache.items():
                self.assertEqual(w.shape,(g.size-len(reduced.indices),reduced.ng))
                self.assertIsNone(reduced.cache[key])
            reduced.batch_correction=False;single=reduced.apply_batch(x,modes)
            self.assertLess(np.linalg.norm(single-actual)/np.linalg.norm(actual),1e-12)
            reduced.batch_correction=True
            incidents=[self.d.modal_incident(m) for m in modes]
            h,stats=solve_modal_batch(self.d,g,incidents,modes,preconditioner=reduced)
            for v,inc,s in zip(h,incidents,stats):
                rhs=(inc.reshape(-1,8)*2*self.d.scaling[2]).ravel()
                self.assertLess(np.linalg.norm(self.d.apply_reference(v)-rhs)/np.linalg.norm(rhs),1.01e-8)
                self.assertEqual(s['info'],0)
            reduced.release_except();self.assertEqual(reduced.bytes,0)
        finally:exact.close();reduced.close()

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

    def test_compressed_storage_streamed_assembly_and_strict_solves(self):
        from sector_coarse_hierarchy import HierarchicalCoarse
        from dirac_symmetry import solve_modal_batch
        from unittest.mock import patch
        g=self.d._compact_geometry;modal=[(0,2,-1),(1,1,-1)]
        for outer in (False,True):
            options=dict(radius=1.5,aggregate_groups=3,aggregate_basis=3,outer=outer)
            full=HierarchicalCoarse(g,3,**options)
            compressed=HierarchicalCoarse(g,3,storage='fp32',**options)
            try:
                exact_columns=full.global_columns(modal[0])
                tiled=np.empty_like(exact_columns)
                for first,block in compressed.global_column_tiles(modal[0],patches=1):tiled[:,first:first+block.shape[1]]=block
                np.testing.assert_allclose(tiled,exact_columns,rtol=2e-12,atol=2e-12)
                full.prepare_many(modal)
                # A compressed preparation must never request the full AZ array.
                with patch.object(compressed,'global_columns',side_effect=AssertionError('unbounded FP64 AZ')):
                    compressed.prepare_many(modal)
                key=compressed.key(modal[0]);cw,cf=compressed.global_cache[key];fw,ff=full.global_cache[key]
                self.assertEqual(cw.dtype,np.complex64);self.assertTrue(cw.flags.f_contiguous)
                self.assertEqual(cw.nbytes*2,fw.nbytes)
                self.assertEqual(cf[0].dtype,np.complex128)
                np.testing.assert_allclose(cf[0],ff[0],rtol=3e-11,atol=3e-11)
                rng=np.random.default_rng(741);x=rng.normal(size=g.size)+1j*rng.normal(size=g.size)
                expected=full.apply(x,modal[0]);before=self.d.calls
                actual=compressed.apply(x,modal[0]);self.assertEqual(self.d.calls,before if outer else before+1)
                self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),2e-6)
                self.assertLess(compressed.statistics[str(key)]['storage_relative_error'],6e-8)
                incidents=[self.d.modal_incident(m) for m in modal]
                h,stats=solve_modal_batch(self.d,g,incidents,modal,preconditioner=compressed)
                for v,inc,s in zip(h,incidents,stats):
                    rhs=(inc.reshape(-1,8)*2*self.d.scaling[2]).ravel()
                    self.assertLess(np.linalg.norm(self.d.apply_reference(v)-rhs)/np.linalg.norm(rhs),1.01e-8)
                    self.assertEqual(s['info'],0)
                compressed.release_except();self.assertEqual(compressed.bytes,0)
            finally:full.close();compressed.close()

    def test_mixed_gemv_accumulates_fp64_with_original_complex_input(self):
        from solve import lib,address,check
        import ctypes as ct
        lib.dirac_dense_gemv_subtract_cf.argtypes=[ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]
        rng=np.random.default_rng(113);rows,cols=135,71
        a=np.asfortranarray(rng.normal(size=(rows,cols))+1j*rng.normal(size=(rows,cols)),dtype=np.complex64)
        x=(rng.normal(size=cols)+1j*rng.normal(size=cols))*1e-90
        y=(rng.normal(size=rows)+1j*rng.normal(size=rows))*1e-90
        expected=y-a.astype(complex)@x;actual=y.copy()
        check(lib.dirac_dense_gemv_subtract_cf(rows,cols,address(a),address(x),address(actual)))
        self.assertLess(np.linalg.norm((actual-expected)/1e-90)/np.linalg.norm(expected/1e-90),3e-14)

    def test_batched_mixed_product_and_tails(self):
        from solve import lib,address,check
        import ctypes as ct
        if not hasattr(lib,'dirac_dense_gemm_subtract_cf'):self.skipTest('requires rebuilt batch library')
        lib.dirac_dense_gemm_subtract_cf.argtypes=[ct.c_int,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]
        rng=np.random.default_rng(176);rows,cols=135,71
        a=np.asfortranarray(rng.normal(size=(rows,cols))+1j*rng.normal(size=(rows,cols)),dtype=np.complex64)
        for nrhs in (1,2,4,16):
            x=(rng.normal(size=(nrhs,cols))+1j*rng.normal(size=(nrhs,cols)))*1e-90
            y=(rng.normal(size=(nrhs,rows))+1j*rng.normal(size=(nrhs,rows)))*1e-90
            expected=y-x@a.astype(complex).T
            check(lib.dirac_dense_gemm_subtract_cf(rows,cols,nrhs,address(a),address(x),address(y)))
            self.assertLess(np.linalg.norm((y-expected)/1e-90)/np.linalg.norm(expected/1e-90),3e-14)

    def test_batched_preconditioner_keeps_sectors_and_strict_solve(self):
        from sector_coarse_hierarchy import HierarchicalCoarse
        from dirac_symmetry import solve_modal_batch
        from solve import lib
        if not hasattr(lib,'dirac_dense_gemm_subtract_cf'):self.skipTest('requires rebuilt batch library')
        c=HierarchicalCoarse(self.d._compact_geometry,3,radius=1.5,aggregate_groups=1024,
            aggregate_basis=3,outer=True,storage='fp32',batch_correction=True)
        modes=[(0,2,-1),(1,1,-1),(1,2,-1)]
        try:
            c.prepare_many(modes)
            rng=np.random.default_rng(578);x=rng.normal(size=(3,c.g.size))+1j*rng.normal(size=(3,c.g.size))
            c.batch_correction=False;expected=c.apply_batch(x,modes)
            c.batch_correction=True;before=self.d.calls;actual=c.apply_batch(x,modes)
            self.assertEqual(self.d.calls,before)
            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),2e-12)
            inc=[self.d.modal_incident(m) for m in modes[:2]]
            h,stats=solve_modal_batch(self.d,c.g,inc,modes[:2],preconditioner=c)
            for v,f,s in zip(h,inc,stats):
                rhs=(f.reshape(-1,8)*2*self.d.scaling[2]).ravel()
                self.assertLess(np.linalg.norm(self.d.apply_reference(v)-rhs)/np.linalg.norm(rhs),1.01e-8)
                self.assertEqual(s['info'],0)
        finally:c.close()

    def test_failed_compressed_audit_cannot_be_reused(self):
        from sector_coarse_hierarchy import HierarchicalCoarse
        from unittest.mock import patch
        c=HierarchicalCoarse(self.d._compact_geometry,3,radius=1.5,aggregate_groups=3,aggregate_basis=3,outer=True,storage='fp32')
        mode=(0,2,-1)
        try:
            with patch.object(c,'subtract_correction',side_effect=lambda w,q,out:out.fill(0)):
                with self.assertRaisesRegex(RuntimeError,'range audit'):c.prepare(mode)
            self.assertFalse(c.global_cache)
            with self.assertRaisesRegex(RuntimeError,'incomplete'):c.prepare(mode)
            c.release_except();self.assertEqual(c.bytes,0)
        finally:c.close()

    def test_streamed_lu_reuses_Ac_and_rejects_hidden_copy(self):
        from sector_coarse_hierarchy import HierarchicalCoarse
        from unittest.mock import patch
        from scipy.linalg import lu_solve
        c=HierarchicalCoarse(self.d._compact_geometry,3,aggregate_groups=3,storage='fp32')
        rng=np.random.default_rng(743)
        a=np.asfortranarray(np.eye(96)+(rng.normal(size=(96,96))+1j*rng.normal(size=(96,96)))/100)
        x=rng.normal(size=96)+1j*rng.normal(size=96);rhs=a@x
        try:
            factor,_,_=c.factor_global(a)
            self.assertTrue(np.shares_memory(a,factor[0]))
            self.assertLess(np.linalg.norm(lu_solve(factor,rhs)-x)/np.linalg.norm(x),2e-13)
            with self.assertRaisesRegex(ValueError,'Fortran'):c.factor_global(np.eye(96,dtype=complex))
            with patch('sector_coarse_hierarchy.lu_factor',return_value=(a.copy(),np.arange(96))):
                with self.assertRaisesRegex(MemoryError,'copied Ac'):c.factor_global(a)
        finally:c.close()

    def test_parallel_global_lu_restores_blas_threads(self):
        try:from threadpoolctl import threadpool_info
        except ImportError:self.skipTest('optional threadpoolctl unavailable')
        from sector_coarse_hierarchy import HierarchicalCoarse
        from scipy.linalg import lu_factor,lu_solve
        from unittest.mock import patch
        c=HierarchicalCoarse(self.d._compact_geometry,3,aggregate_groups=3,lu_threads=4)
        pools=lambda:[v['num_threads'] for v in threadpool_info() if v['user_api']=='blas']
        before=pools();observed=[];rng=np.random.default_rng(842);n=1024
        matrix=np.asfortranarray(np.eye(n)+(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))/100)
        x=rng.normal(size=n)+1j*rng.normal(size=n);rhs=matrix@x
        def checked(*a,**kw):observed.extend(pools());return lu_factor(*a,**kw)
        try:
            with patch('sector_coarse_hierarchy.lu_factor',side_effect=checked):factor,seconds,threads=c.factor_global(matrix)
            self.assertEqual(threads,4);self.assertTrue(observed);self.assertEqual(set(observed),{4})
            self.assertEqual(pools(),before)
            self.assertLess(np.linalg.norm(lu_solve(factor,rhs)-x)/np.linalg.norm(x),2e-13)
            with patch('sector_coarse_hierarchy.lu_factor',side_effect=RuntimeError('injected LU failure')):
                with self.assertRaisesRegex(RuntimeError,'injected'):c.factor_global(matrix)
            self.assertEqual(pools(),before)
        finally:c.close()

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
