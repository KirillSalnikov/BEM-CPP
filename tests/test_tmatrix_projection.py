import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tmatrix_projection import SphericalProjector
from tmatrix_reference import transverse_harmonics


class ProjectionTest(unittest.TestCase):
    def test_fft_matches_direct_integral_for_arbitrary_complex_fields(self):
        rng=np.random.default_rng(47)
        for order in (1,5,12):
            band=2*order+4
            z,w=np.polynomial.legendre.leggauss(band)
            zz,pp=np.meshgrid(z,np.arange(2*band)*np.pi/band,indexing='ij')
            directions=np.stack((np.sqrt(1-zz*zz)*np.cos(pp),np.sqrt(1-zz*zz)*np.sin(pp),zz),axis=-1).reshape(-1,3)
            b=transverse_harmonics(directions,order)
            weights=np.repeat(np.repeat(w,2*band)*np.pi/band,2)
            fields=rng.normal(size=len(b))+1j*rng.normal(size=len(b))
            expected=b.conj().T@(fields*weights)
            np.testing.assert_allclose(SphericalProjector(order,z,w)@fields,expected,rtol=3e-12,atol=3e-14)

    def test_projection_recovers_modal_amplitudes(self):
        order=7; band=order+3
        z,w=np.polynomial.legendre.leggauss(band)
        zz,pp=np.meshgrid(z,np.arange(2*band)*np.pi/band,indexing='ij')
        directions=np.stack((np.sqrt(1-zz*zz)*np.cos(pp),np.sqrt(1-zz*zz)*np.sin(pp),zz),axis=-1).reshape(-1,3)
        b=transverse_harmonics(directions,order)
        rng=np.random.default_rng(17); amplitudes=rng.normal(size=b.shape[1])+1j*rng.normal(size=b.shape[1])
        np.testing.assert_allclose(SphericalProjector(order,z,w)@(b@amplitudes),amplitudes,rtol=3e-12,atol=3e-13)


if __name__=='__main__': unittest.main()
