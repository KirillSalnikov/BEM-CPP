import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tmatrix_size_taylor import moments, polynomial, radiation_moments, matrix_pade, evaluate_pade


class TaylorTests(unittest.TestCase):
    def test_moments_predict_independent_systems(self):
        rng=np.random.default_rng(381);n=7
        a=np.zeros((7,n,n),complex);b=np.zeros((7,n,3),complex)
        a[0]=4*np.eye(n)+.1*rng.normal(size=(n,n));a[1]=.1*rng.normal(size=(n,n));a[2]=.03j*rng.normal(size=(n,n))
        b[:3]=rng.normal(size=(3,n,3))+1j*rng.normal(size=(3,n,3))
        x,_=moments(a,b)
        for delta in [-.3,.19]:
            exact=np.linalg.solve(polynomial(a,delta),polynomial(b,delta))
            error=np.linalg.norm(polynomial(x,delta)-exact)/np.linalg.norm(exact)
            exact_half=np.linalg.solve(polynomial(a,delta/2),polynomial(b,delta/2))
            error_half=np.linalg.norm(polynomial(x,delta/2)-exact_half)/np.linalg.norm(exact_half)
            self.assertLess(error,1e-7)
            self.assertGreater(error/error_half,80)  # sixth degree: remainder O(delta^7)

    def test_radiation_includes_prefactor_derivative(self):
        x=np.zeros((4,2,2),complex);d=x.copy();x[0]=np.eye(2);d[0]=2*np.eye(2)
        phase=np.array([1,1j]);c=radiation_moments(d,x,3.,phase)
        np.testing.assert_allclose(polynomial(c,.2),-1j*3.2/(4*np.pi)*2*np.eye(2))

    def test_pade_recovers_known_matrix_resonance(self):
        basis=np.array([[1,2j],[-3,4]],complex)
        c=np.array([basis*(.2+.1j)**j for j in range(7)])
        model=matrix_pade(c,1)
        for delta in [-1.,.5,1.]:
            np.testing.assert_allclose(evaluate_pade(model,delta),basis/(1-(.2+.1j)*delta),rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()
