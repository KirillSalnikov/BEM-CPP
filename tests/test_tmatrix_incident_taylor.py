import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tmatrix_incident import regular_fields
from tmatrix_incident_taylor import RegularTaylorFieldCache,regular_field_moments
from tmatrix_size_taylor import polynomial


class IncidentTaylorTests(unittest.TestCase):
    def test_complex_contour_and_nearby_fields(self):
        rng=np.random.default_rng(46)
        points=np.r_[rng.uniform(-1,1,(30,3)),[[0,0,0],[0,0,.7],[0,0,-.7],[1e-9,0,0]]]
        central_error=coefficient_error=nearby_error=0.
        for k in (.3,3.,20.,40.):
            radius=min(.3,.1*k);cache=RegularTaylorFieldCache(points,k,6)
            for mode in ((0,1,-1),(1,1,0),(0,2,2),(1,6,-3),(0,20,20),(1,40,-1)):
                a=np.empty_like(cache.base);cache.fill(mode,a)
                ref=regular_fields(points,mode,k);norm=max(np.linalg.norm(ref),1e-280)
                central_error=max(central_error,np.linalg.norm(a[0]-ref)/norm)
                samples=np.array([regular_fields(points,mode,k+radius*np.exp(2j*np.pi*q/64)) for q in range(64)])
                coefficients=np.fft.fft(samples,axis=0)[:7]/64
                scaled=a*radius**np.arange(7)[:,None,None]
                coefficient_error=max(coefficient_error,np.linalg.norm(scaled-coefficients)/norm)
                for delta in (-radius/100,radius/100):
                    exact=regular_fields(points,mode,k+delta)
                    nearby_error=max(nearby_error,np.linalg.norm(polynomial(a,delta)-exact)/max(np.linalg.norm(exact),1e-280))
                swapped=np.empty_like(a);cache.fill((1-mode[0],mode[1],mode[2]),swapped)
                direct=regular_field_moments(points,(1-mode[0],mode[1],mode[2]),k,6)
                np.testing.assert_array_equal(swapped,direct)
        print(dict(central_error=central_error,contour_scaled_error=coefficient_error,nearby_error=nearby_error),flush=True)
        self.assertLess(central_error,3e-14)
        self.assertLess(coefficient_error,5e-12)
        self.assertLess(nearby_error,5e-12)

    def test_invalid_inputs(self):
        points=np.array([[1.,0,0]])
        for k,n in [(0,4),(float('nan'),4),(3,17),(3,-1)]:
            with self.assertRaises(ValueError):RegularTaylorFieldCache(points,k,n)
        cache=RegularTaylorFieldCache(points,3,0);out=np.empty_like(cache.base)
        with self.assertRaises(ValueError):cache.fill((0,0,0),out)
        with self.assertRaises(ValueError):cache.fill((0,1,2),out)
        with self.assertRaises(ValueError):cache.fill((0,1,0),np.empty((1,1,6)))

if __name__=='__main__':unittest.main()
