import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.linalg import qr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tmatrix_size_rom import current_basis,project_bem,separate_central_currents,ka_scaled_equation
from tmatrix_size_taylor import moments,polynomial


class ReducedBEMTests(unittest.TestCase):
    def test_invariant_current_space_predicts_unseen_sizes(self):
        rng=np.random.default_rng(713)
        q,_=qr(rng.normal(size=(15,15))+1j*rng.normal(size=(15,15)))
        a=np.zeros((4,15,15),complex)
        a[0]=q@np.diag(np.linspace(2,4,15))@q.conj().T
        a[1]=q@np.diag(.2j*np.arange(15))@q.conj().T
        a[2]=.04*np.eye(15)
        b=np.zeros((4,15,3),complex);b[0]=q[:,:3];b[1]=.2j*b[0]
        d=np.zeros((4,3,15),complex);d[0]=rng.normal(size=(3,15));d[1]=.07j*d[0]
        currents,_=moments(a,b)
        basis,info=current_basis(currents,max_rank=3)
        self.assertLess(info['orthogonality_error'],1e-12)
        for testing in ['galerkin','central_inverse']:
            model,_,_=project_bem(a,b,d,basis,np.array([1,1j,-1]),3.,testing=testing)
            for delta in [-1.2,.3,1.5]:
                predicted,y=model.evaluate(3+delta)
                exact=np.linalg.solve(polynomial(a,delta),polynomial(b,delta))
                np.testing.assert_allclose(basis@y,exact,rtol=1e-12,atol=1e-12)
                phase=model.phase
                target=-1j*(3+delta)/(4*np.pi)*phase[:,None]*(polynomial(d,delta)@exact)*phase[None,:].conj()
                np.testing.assert_allclose(predicted,target,rtol=1e-12,atol=1e-12)
                scaled,_=ka_scaled_equation(model).evaluate(3+delta)
                np.testing.assert_allclose(scaled,target,rtol=1e-12,atol=1e-12)

    def test_randomized_basis_and_guards(self):
        rng=np.random.default_rng(17)
        u=rng.normal(size=(120,4))+1j*rng.normal(size=(120,4))
        x=np.array([u@(rng.normal(size=(4,40))+1j*rng.normal(size=(4,40))) for _ in range(3)])
        q,info=current_basis(x,max_rank=4,oversample=4)
        self.assertLess(max(info['current_coefficient_projection_errors']),1e-12)
        with self.assertRaises(ValueError):current_basis(x,window=-1)

    def test_separate_current_spaces_do_not_tie_electric_to_magnetic(self):
        rng=np.random.default_rng(61)
        x=rng.normal(size=(40,6))+1j*rng.normal(size=(40,6))
        q,info=separate_central_currents(x,12)
        self.assertLess(info['current_projection_error'],1e-12)
        self.assertLess(info['orthogonality_error'],1e-12)
        independent=np.vstack((2*x[:20],-3j*x[20:]))
        np.testing.assert_allclose(q@(q.conj().T@independent),independent,atol=1e-12)


if __name__=='__main__':unittest.main()
