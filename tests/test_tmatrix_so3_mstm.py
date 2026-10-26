import os
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from tmatrix_reference import modes,analytic_mueller,transverse_harmonics,coherency_to_mueller
from tmatrix_so3_mstm import analytic_mueller_tmatrix


def reciprocal_t(order, seed=23):
    selected=modes(order);q=len(selected)
    lookup={tuple(mode):i for i,mode in enumerate(selected)}
    flip=np.array([lookup[p,l,-m] for p,l,m in selected])
    signs=np.array([(-1.)**m for p,l,m in selected])
    rng=np.random.default_rng(seed)
    t=rng.normal(size=(q,q))+1j*rng.normal(size=(q,q))
    return .5*(t+t.T[np.ix_(flip,flip)]*signs[:,None]*signs[None,:])


@unittest.skipUnless((ROOT/'bin/tmatrix_so3_mstm').exists(),'build the analytic Fortran SO3 backend')
class FullAnalyticTest(unittest.TestCase):
    def test_all_sixteen_entries_and_threaded_reduction(self):
        for order in (1,3,6):
            t=reciprocal_t(order)
            phase=np.array([1j**(l-p) for p,l,m in modes(order)])
            kernel=4j*np.pi*phase.conj()[:,None]*t*phase[None,:]
            angles=np.array([0.,17.,83.,149.,180.])
            expected=analytic_mueller(kernel,order,angles)
            for threads in (1,4):
                observed,meta=analytic_mueller_tmatrix(t,order,angles,threads=threads)
                self.assertEqual(meta['Euler_quadrature_nodes'],0)
                np.testing.assert_allclose(observed,expected,rtol=3e-11,atol=np.max(abs(expected))*3e-13)

    def test_cyclic_selection(self):
        order=4;t=reciprocal_t(order)
        m=np.array([x[2] for x in modes(order)])
        t[(m[:,None]-m[None,:])%3!=0]=0
        angles=np.linspace(0,180,9)
        a,_=analytic_mueller_tmatrix(t,order,angles,cyclic_order=1)
        b,meta=analytic_mueller_tmatrix(t,order,angles,cyclic_order=3)
        self.assertEqual(meta['cyclic_relative_defect'],0)
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=np.max(abs(a))*1e-13)

    def test_large_order_isotropic_matrix_against_direct_fields(self):
        # A rotationally invariant matrix needs no orientation integration;
        # compare its direct Jones covariance against the full contractions.
        order=16;selected=modes(order)
        t=np.diag([(.03+.07j)*np.exp(-l/5)*(1+.2*p) for p,l,m in selected])
        phase=np.array([1j**(l-p) for p,l,m in selected])
        kernel=4j*np.pi*phase.conj()[:,None]*t*phase[None,:]
        angles=np.array([0.,17.,83.,149.,180.]);r=np.deg2rad(angles)
        directions=np.stack((np.sin(r),np.zeros_like(r),np.cos(r)),axis=1)
        incoming=transverse_harmonics(np.array([[0.,0.,1.]]),order)
        outgoing=transverse_harmonics(directions,order).reshape(-1,2,len(selected))
        expected=[]
        for out in outgoing:
            amplitude=(out@kernel@incoming.conj().T).ravel()
            expected.append(coherency_to_mueller(np.outer(amplitude,amplitude.conj())))
        observed,_=analytic_mueller_tmatrix(t,order,angles)
        np.testing.assert_allclose(observed,expected,rtol=2e-10,atol=np.max(abs(observed))*1e-12)

    def test_arbitrary_nonreciprocal_matrix_matches_all_sixteen_entries(self):
        order=3;t=reciprocal_t(order);t[0,1]+=10+3j;t[3,4]-=7j
        phase=np.array([1j**(l-p) for p,l,m in modes(order)])
        kernel=4j*np.pi*phase.conj()[:,None]*t*phase[None,:]
        angles=np.array([0.,17.,83.,149.,180.])
        expected=analytic_mueller(kernel,order,angles)
        observed,meta=analytic_mueller_tmatrix(t,order,angles)
        self.assertEqual(meta['reciprocal_partner_passes'],2)
        self.assertFalse(meta['reciprocity_projection_applied'])
        np.testing.assert_allclose(observed,expected,rtol=3e-11,atol=np.max(abs(expected))*3e-13)

    @unittest.skipUnless((ROOT/'bin/tmatrix_so3_mstm_fp32').exists(),'build fp32 T storage backend')
    def test_single_precision_T_storage_keeps_double_contractions(self):
        t=reciprocal_t(5);angles=np.linspace(0,180,19)
        expected,_=analytic_mueller_tmatrix(t,5,angles,precision='fp64')
        observed,meta=analytic_mueller_tmatrix(t,5,angles,precision='fp32')
        self.assertEqual(meta['contraction_precision'],'fp64')
        self.assertLess(np.linalg.norm(observed-expected)/np.linalg.norm(expected),2e-7)


if __name__=='__main__':unittest.main()
