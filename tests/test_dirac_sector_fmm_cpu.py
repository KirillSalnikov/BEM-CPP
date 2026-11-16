"""Opt-in independent dense checks for the representative-target FMM path."""
import os,sys,unittest
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('BEM_DIRAC_LIBRARY'), 'requires selected Dirac library and FMM3D')
class SectorFMMCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0]=[str(R/'benchmarks/clifford_dielectric_20261115'),str(R/'scripts')]
        from sector_fmm import SectorFMM
        from solve import Dirac
        kw=dict(shape='prism',ref=0,ka=3,index=1.3116+.12j,degree=3,order=24,target_order=13,near_subdiv=1)
        cls.f=SectorFMM(**kw,fmm_packet=2)
        cls.d=Dirac(**kw,dense_kernel='simd',block_panels=1,dense_cache='orbits12')

    @classmethod
    def tearDownClass(cls):cls.f.close();cls.d.close()

    def test_full_and_selected_arbitrary_packet(self):
        f=self.f;g=f._compact_geometry;rng=np.random.default_rng(144)
        x=rng.normal(size=(3,f.size))+1j*rng.normal(size=(3,f.size))
        expected=self.d.apply_reference_batch(x)
        for packet in (1,2,4):
            f.fmm_packet=packet
            y=f.apply_reference_batch(x);z=f.apply_sector_batch(x)
            self.assertLess(np.linalg.norm(y-expected)/np.linalg.norm(expected),2e-10)
            self.assertLess(np.linalg.norm(z-g.restrict(expected))/np.linalg.norm(g.restrict(expected)),2e-9)
        self.assertLess(np.linalg.norm(f._dense_inverse(x[0])-self.d._dense_inverse(x[0]))/np.linalg.norm(self.d._dense_inverse(x[0])),2e-12)
        self.assertLess(g.audit_operator(),2e-11)

    def test_P1_coarse_inverse_and_both_polarizations(self):
        from sector_coarse_fmm import MatrixFreeCoarse
        from sector_coarse import CoarseCorrection
        from dirac_symmetry import solve_modal_batch
        f=self.f;f.fmm_packet=2;g=f._compact_geometry
        coarse=MatrixFreeCoarse(g,3);dense=CoarseCorrection(self.d._compact_geometry,3)
        modal=[(0,2,-1),(1,1,-1)] # same rotational and horizontal sector
        coarse.prepare_many(modal);dense.prepare_many(modal)
        rng=np.random.default_rng(99);x=rng.normal(size=(2,g.size))+1j*rng.normal(size=(2,g.size))
        expected=np.array([dense.apply(v,m) for v,m in zip(x,modal)])
        actual=coarse.apply_batch(x,modal)
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),2e-9)
        incidents=np.array([f.modal_incident(m) for m in modal])
        h,stats=solve_modal_batch(f,g,incidents,modal,preconditioner=coarse)
        for v,inc,s in zip(h,incidents,stats):
            rhs=(inc.reshape(-1,8)*(2*f.scaling[2])).ravel()
            self.assertLess(np.linalg.norm(self.d.apply_reference(v)-rhs)/np.linalg.norm(rhs),1.01e-8)
            self.assertEqual(s['info'],0)
        self.assertEqual(coarse.memory_plan()['stored_W_bytes'],0)
        coarse.release_except();self.assertEqual(coarse.bytes,0);coarse.close()

    def test_near_budget_is_enforced(self):
        from sector_fmm import SectorFMM
        with self.assertRaisesRegex(RuntimeError,'budget'):
            SectorFMM(ref=0,degree=0,max_bytes=1,audit=False)

    def test_P0_without_horizontal_reduction(self):
        from sector_fmm import SectorFMM
        from solve import Dirac
        kw=dict(shape='prism',ref=0,ka=2,degree=0,order=14,target_order=13,near_subdiv=0)
        f=SectorFMM(**kw,horizontal=False,audit=False);d=Dirac(**kw)
        rng=np.random.default_rng(31);x=rng.normal(size=(1,f.size))+1j*rng.normal(size=(1,f.size))
        expected=d.apply_reference_batch(x);g=f._compact_geometry
        self.assertEqual(g.group_size,6)
        self.assertLess(np.linalg.norm(f.apply_reference_batch(x)-expected)/np.linalg.norm(expected),2e-10)
        self.assertLess(np.linalg.norm(f.apply_sector_batch(x)-g.restrict(expected))/np.linalg.norm(g.restrict(expected)),2e-9)
        f.close();d.close()

if __name__=='__main__':unittest.main()
