"""Fast radiation agrees with the existing full eight-component native sum."""
import os,sys,unittest
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('BEM_DIRAC_LIBRARY'),'requires Dirac and optional FINUFFT')
class Radiation(unittest.TestCase):
    def test_all_components_and_incident_subtraction(self):
        sys.path[:0]=[str(R/'benchmarks/clifford_dielectric_20261115'),str(R/'scripts')]
        from solve import Dirac
        from radiation import DiracRadiation,load_finufft
        try:load_finufft()
        except ImportError:self.skipTest('optional FINUFFT is unavailable')
        rng=np.random.default_rng(18);directions=rng.normal(size=(170,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
        for degree in (0,1,2,3):
            d=Dirac(shape='prism',ref=1,ka=10,index=1.3116+.2j,degree=degree,target_order=13,geometry_only=True)
            p=DiracRadiation(d,directions,threads=2)
            h=rng.normal(size=d.size)+1j*rng.normal(size=d.size);f=d.incident([0,0,1],[1,0,0])
            for incident in (None,f):
                expected=d.field(h,directions,far=True,incident=incident)
                actual=p.apply(h,incident)
                self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),5e-11)
            self.assertEqual(np.linalg.norm(p.apply(np.zeros_like(h))),0.)
            p.close();d.close()
if __name__=='__main__':unittest.main()
