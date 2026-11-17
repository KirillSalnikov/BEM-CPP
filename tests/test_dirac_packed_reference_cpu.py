"""Independent full dense oracles for symmetry-packing of strict actions."""
import os,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
R=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('BEM_DIRAC_LIBRARY'),'requires selected Dirac and FMM3D libraries')
class PackedReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0]=[str(R/'benchmarks/clifford_dielectric_20261115'),str(R/'scripts')]
        from solve import Dirac
        from sector_fmm import SectorFMM
        from tmatrix_reference import modes
        from packed_reference import key
        kw=dict(ref=0,ka=8,index=1.3116+.3j,degree=3,order=24,target_order=13,near_subdiv=1)
        cls.d=SectorFMM(**kw);cls.g=cls.d._compact_geometry;cls.g.audit_operator()
        cls.oracle=Dirac(shape='prism',**kw,dense_kernel='simd',block_panels=1)
        chosen={}
        for mode in modes(6):chosen.setdefault(key(cls.g,mode),mode)
        cls.modes=list(chosen.values());assert len(cls.modes)==12
        rng=np.random.default_rng(72)
        cls.x=rng.normal(size=(12,cls.g.size))+1j*rng.normal(size=(12,cls.g.size))
        cls.x*=10.**np.linspace(-6,6,12)[:,None]
        cls.expected=np.array([cls.g.restrict(cls.oracle.apply_reference(cls.g.expand_mode(x,m))) for x,m in zip(cls.x,cls.modes)])

    @classmethod
    def tearDownClass(cls):cls.d.close();cls.oracle.close()

    def test_all_sectors_and_unequal_norms_against_full_dense(self):
        from packed_reference import apply_packed
        got,stats=apply_packed(self.g,self.x,self.modes)
        errors=np.linalg.norm(got-self.expected,axis=1)/np.linalg.norm(self.expected,axis=1)
        self.assertLess(max(errors),3e-10);self.assertEqual(stats['full_operator_calls'],1)
        self.assertEqual(stats['packed_sizes'],[12])

    def test_duplicates_and_zero_cannot_cancel(self):
        from packed_reference import apply_packed
        x=np.array([self.x[0],-self.x[0],self.x[2],np.zeros(self.g.size)])
        modes=[self.modes[0],self.modes[0],self.modes[2],self.modes[4]]
        got,stats=apply_packed(self.g,x,modes)
        self.assertEqual(stats['full_operator_calls'],2)
        self.assertEqual(np.linalg.norm(got[3]),0.)
        for i,expected in enumerate([self.expected[0],-self.expected[0],self.expected[2]]):
            self.assertLess(np.linalg.norm(got[i]-expected)/np.linalg.norm(expected),3e-10)

    def test_projection_and_outside_row_corruption(self):
        from packed_reference import apply_packed,project_modes,add_expanded
        x=self.x[5];mode=self.modes[5];full=np.zeros(self.d.size,complex)
        add_expanded(self.g,full,x,mode)
        self.assertLess(np.linalg.norm(full-self.g.expand_mode(x,mode))/np.linalg.norm(full),1e-14)
        recovered=project_modes(self.g,full,self.modes)
        recovered[5]-=x
        self.assertLess(np.linalg.norm(recovered)/np.linalg.norm(x),1e-13)
        original=self.d.apply_reference
        absent=next(p for p in range(self.d.count) if p not in self.g.representatives)
        def faulty(v):
            y=original(v);y[absent*self.d.basis_count*8]+=1.;return y
        with patch.object(self.d,'apply_reference',faulty):
            with self.assertRaisesRegex(RuntimeError,'violates covariance'):apply_packed(self.g,self.x[:2],self.modes[:2])

    def test_without_covariance_audit_rejects(self):
        from packed_reference import apply_packed
        with patch.object(self.g,'operator_error',np.inf):
            with self.assertRaisesRegex(ValueError,'covariance audit'):apply_packed(self.g,self.x,self.modes)

if __name__=='__main__':unittest.main()
