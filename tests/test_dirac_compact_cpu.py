"""Opt-in exact symmetry storage regression, including nonmodal full vectors."""
import os, sys, unittest
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('BEM_DIRAC_LIBRARY'), 'requires selected Dirac library')
class CompactDiracCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(R/'benchmarks/clifford_dielectric_20261115'), str(R/'scripts')]
        from solve import Dirac
        from dirac_symmetry import CyclicGeometry
        options = dict(shape='prism', ref=0, ka=3, index=1.3116+.12j, degree=2,
                       order=24, target_order=13, near_subdiv=1, dense_kernel='simd', block_panels=1)
        cls.full = Dirac(**options)
        cls.compact = Dirac(**options, dense_cache='orbits12', max_bytes=8<<20)
        cls.g = cls.compact._compact_geometry
        cls.full_g = CyclicGeometry(cls.full, horizontal=True)

    @classmethod
    def tearDownClass(cls):
        cls.full.close(); cls.compact.close()

    def test_arbitrary_full_action_and_inverse_transform(self):
        d=self.compact;g=self.g;rng=np.random.default_rng(97)
        x=rng.normal(size=(3,d.size))+1j*rng.normal(size=(3,d.size))
        expected=np.array([self.full.apply_reference(v) for v in x])
        actual=np.array([d.apply_reference(v) for v in x])
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),2e-11)
        self.assertLess(np.linalg.norm(d.apply_batch(x)-expected)/np.linalg.norm(expected),2e-11)
        # The density deliberately has no definite rotational or mirror parity.
        projected=g.expand_mode(g.restrict(x[0]),(0,1,0))
        self.assertGreater(np.linalg.norm(projected-x[0])/np.linalg.norm(x[0]),.1)
        for transform in g.transforms:
            self.assertLess(np.linalg.norm(transform.inverse(transform.apply(x))-x)/np.linalg.norm(x),2e-12)
        self.assertLess(np.linalg.norm(d._dense_inverse(x[0])-self.full._dense_inverse(x[0]))/np.linalg.norm(self.full._dense_inverse(x[0])),2e-12)
        self.assertEqual(self.full.cache_info['kernel_bytes'],12*d.cache_info['kernel_bytes'])
        self.assertLess(d.cache_info['direct_audit']['relative_error'],2e-11)

    def test_horizontal_parity_M_N_and_full_residual(self):
        from dirac_symmetry import solve_modal_batch
        modes=[(p,l,m) for p in (0,1) for l,m in [(1,-1),(1,0),(2,1),(3,-2)]]
        incident=np.array([self.compact.modal_incident(mode) for mode in modes])
        for f,mode in zip(incident,modes):
            p,l,m=mode
            self.assertLess(np.linalg.norm(self.g.transforms[6].apply(f)-(-1)**(p+l+m+1)*f)/np.linalg.norm(f),2e-12)
        h,stats=solve_modal_batch(self.compact,self.g,incident,modes)
        reference,reference_stats=solve_modal_batch(self.full,self.full_g,incident,modes)
        self.assertLess(np.linalg.norm(h-reference)/np.linalg.norm(reference),1e-7)
        for f,v,stat in zip(incident,h,stats):
            rhs=(f.reshape(-1,8)*(2*self.full.scaling[2])).ravel()
            self.assertLess(np.linalg.norm(self.full.apply_reference(v)-rhs)/np.linalg.norm(rhs),1e-8)
            self.assertLess(stat['residual'],1e-8)

    def test_unsupported_operations_reject(self):
        from solve import Dirac
        from dirac_symmetry import CyclicGeometry
        with self.assertRaises(ValueError):self.g.expand(np.zeros(self.g.size),1)
        with self.assertRaises(ValueError):CyclicGeometry(self.full,order=0)
        with self.assertRaises(ValueError):Dirac(dense_cache='orbits12')
        with self.assertRaisesRegex(RuntimeError,'full dense cache'):
            self.compact.cauchy(np.zeros(self.compact.size,complex))

if __name__=='__main__':unittest.main()
