"""Independent full action, individual stopping gates and mirror failure tests."""
import os,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
R=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('BEM_DIRAC_LIBRARY'),'requires selected Dirac library')
class BatchChecksCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0]=[str(R/'benchmarks/clifford_dielectric_20261115'),str(R/'scripts')]
        from solve import Dirac
        from dirac_symmetry import SurfaceTransform,solve_modal_batch
        options=dict(shape='prism',ref=0,ka=3,index=1.3116+.12j,degree=2,order=24,
            target_order=13,near_subdiv=1,dense_kernel='simd',block_panels=1)
        cls.full=Dirac(**options);cls.d=Dirac(**options,dense_cache='orbits12')
        cls.g=cls.d._compact_geometry;cls.mirror=SurfaceTransform(cls.d,np.diag([1.,-1.,1.]))
        cls.modes=[(0,1,-1),(1,2,-1),(0,3,-2)]
        cls.f=np.array([cls.d.modal_incident(mode) for mode in cls.modes])
        cls.h,cls.stats=solve_modal_batch(cls.d,cls.g,cls.f,cls.modes)
    @classmethod
    def tearDownClass(cls):cls.full.close();cls.d.close()

    def test_arbitrary_vectors_full_and_compact(self):
        rng=np.random.default_rng(141)
        for count in [1,2,3,8,9,16]:
            x=rng.normal(size=(count,self.d.size))+1j*rng.normal(size=(count,self.d.size))
            expected=np.array([self.full.apply_reference(v) for v in x])
            for d in [self.full,self.d]:
                actual=d.apply_reference_batch(x)
                for a,b in zip(actual,expected):self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(b),2e-11)
        with self.assertRaises(ValueError):self.d.apply_reference_batch(x[0])
        with self.assertRaises(ValueError):self.d.apply_reference_batch(np.zeros((17,self.d.size)))

    def test_every_modal_residual_against_independent_full_cache(self):
        for h,f,stats in zip(self.h,self.f,self.stats):
            rhs=(f.reshape(-1,8)*(2*self.full.scaling[2])).ravel()
            self.assertLess(np.linalg.norm(self.full.apply_reference(h)-rhs)/np.linalg.norm(rhs),1e-8)
            self.assertEqual(stats['reference_batch_size'],3)
            self.assertGreater(stats['full_reference_s'],0.)

    def test_corruption_of_one_direct_column_rejects_entire_batch(self):
        from dirac_symmetry import solve_modal_batch
        original=self.d.apply_reference_batch
        def faulty(x):
            value=original(x);value[1,0]+=1.;return value
        with patch.object(self.d,'apply_reference_batch',faulty):
            with self.assertRaisesRegex(RuntimeError,'full FP64 audit failed'):
                solve_modal_batch(self.d,self.g,self.f,self.modes)

    def test_corruption_outside_representative_rows_is_also_rejected(self):
        from dirac_symmetry import solve_modal_batch
        absent=next(p for p in range(self.d.count) if p not in self.g.representatives)
        index=absent*self.d.basis_count*8;original=self.d.apply_reference_batch
        def faulty(x):
            value=original(x);value[1,index]+=1.;return value
        with patch.object(self.d,'apply_reference_batch',faulty):
            with self.assertRaisesRegex(RuntimeError,'full FP64 cyclic residual failed'):
                solve_modal_batch(self.d,self.g,self.f,self.modes)

    def test_zero_rhs_alongside_nonzero_rhs(self):
        incident=self.full.incident((0,0,1),(1,0,0))
        h,stats=self.full.solve_incidents(np.array([np.zeros_like(incident),incident]))
        self.assertEqual(np.linalg.norm(h[0]),0.)
        self.assertEqual(stats[0]['residual'],0.)
        self.assertLess(stats[1]['residual'],1e-8)
        self.assertEqual(stats[1]['reference_batch_size'],2)

    def test_mirror_batch_full_residual_and_one_bad_partner(self):
        from batch_checks import audit_mirror_batch
        from tmatrix_reference import modes
        from tmatrix_symmetry import vertical_mirror_plan
        selected=modes(3);lookup={tuple(m):i for i,m in enumerate(selected)}
        flip,signs,_,_=vertical_mirror_plan(3)
        requests=[]
        for mode,f,h in zip(self.modes,self.f,self.h):
            j=lookup[mode];requests.append((j,int(flip[j]),selected[int(flip[j])],signs[j],f,h))
        checked=audit_mirror_batch(self.d,self.mirror,requests,1e-8)
        for request in requests:
            j,partner,mode,sign,f,h=request;pf=self.full.modal_incident(mode)
            rhs=(pf.reshape(-1,8)*(2*self.full.scaling[2])).ravel()
            residual=np.linalg.norm(self.full.apply_reference(sign*self.mirror.apply(h))-rhs)/np.linalg.norm(rhs)
            self.assertLess(residual,1e-8);self.assertEqual(checked[j][0],partner)
            self.assertAlmostEqual(checked[j][1]['residual'],residual,delta=2e-12)
        original=self.d.apply_reference_batch
        def faulty(x):
            value=original(x);value[-1,7]+=1.;return value
        with patch.object(self.d,'apply_reference_batch',faulty):
            with self.assertRaisesRegex(RuntimeError,'mirror residual failed'):
                audit_mirror_batch(self.d,self.mirror,requests,1e-8)

if __name__=='__main__':unittest.main()
