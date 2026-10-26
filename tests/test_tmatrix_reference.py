#!/usr/bin/env python3
"""Independent quadrature, orthonormality, and Mie convention checks."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
from tmatrix_reference import (modes, transverse_harmonics, analytic_mueller,
                               numerical_so3_mueller, clebsch)
from verify_mie import mie_mueller, mie_coefficients


class TMatrixReferenceTest(unittest.TestCase):
    def test_harmonic_orthonormality(self):
        z, w = np.polynomial.legendre.leggauss(8)
        phi = np.arange(16)*2*np.pi/16
        zz, pp = np.meshgrid(z,phi,indexing="ij")
        dirs = np.stack((np.sqrt(1-zz*zz)*np.cos(pp),np.sqrt(1-zz*zz)*np.sin(pp),zz),axis=-1).reshape(-1,3)
        v = transverse_harmonics(dirs,4)
        ww = np.repeat(np.repeat(w,16)*2*np.pi/16,2)
        np.testing.assert_allclose(v.conj().T@(ww[:,None]*v),np.eye(v.shape[1]),atol=3e-14)

    def test_general_complex_matrix_so3(self):
        rng = np.random.default_rng(37)
        for order in (1,2):
            q = len(modes(order))
            kernel = rng.normal(size=(q,q))+1j*rng.normal(size=(q,q))
            theta = [0,37,90,151,180]
            analytical = analytic_mueller(kernel,order,theta)
            independent = numerical_so3_mueller(kernel,order,theta)
            np.testing.assert_allclose(analytical,independent,atol=2e-14,rtol=2e-13)

    def test_mie_full_mueller_convention(self):
        order, ka, ri = 5, 1., 1.5
        a,b,_ = mie_coefficients(ri,ka)
        t = np.diag([-b[l] if p==0 else -a[l] for p,l,m in modes(order)])
        kernel = t*(4*np.pi/(-1j*ka))
        theta = np.linspace(0,180,19)
        actual = ka**2*analytic_mueller(kernel,order,theta)
        reference = np.moveaxis(np.asarray(mie_mueller(theta.tolist(),ri,ka)),-1,0)
        self.assertLess(np.linalg.norm(actual-reference)/np.linalg.norm(reference),1e-7)

    def test_known_clebsch_coefficients(self):
        self.assertAlmostEqual(clebsch(1,0,1,0,0,0),-1/np.sqrt(3))
        self.assertAlmostEqual(clebsch(1,1,1,1,2,2),1.)
        self.assertEqual(clebsch(1,1,1,1,1,0),0.)

    def test_integrated_scattering_trace(self):
        order=2
        rng=np.random.default_rng(42)
        q=len(modes(order))
        kernel=rng.normal(size=(q,q))+1j*rng.normal(size=(q,q))
        z,w=np.polynomial.legendre.leggauss(2*order+1)
        mueller=analytic_mueller(kernel,order,np.rad2deg(np.arccos(z)))
        integrated=2*np.pi*np.dot(w,mueller[:,0,0])
        trace=np.linalg.norm(kernel)**2/(8*np.pi)
        self.assertLess(abs(integrated/trace-1),2e-13)


if __name__ == "__main__":
    unittest.main()
