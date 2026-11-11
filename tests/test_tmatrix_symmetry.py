import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from tmatrix_incident import regular_fields
from tmatrix_symmetry import c2x_plan, c2x_partner_column, vertical_mirror_plan
from tmatrix_reference import modes


class C2Test(unittest.TestCase):
    def test_vertical_reflection_polar_e_axial_h(self):
        points=np.random.default_rng(91).normal(size=(24,3))*.4
        mirror=np.array([1.,-1.,1.])
        for l in (1,2,5,10):
            for m in {-l,-1,0,1,l}:
                for p in (0,1):
                    fields=regular_fields(points*mirror,(p,l,m),4.).reshape(-1,2,3)
                    fields[:,0]*=mirror
                    fields[:,1]*=-mirror
                    reference=(-1.)**(p+m+1)*regular_fields(points,(p,l,-m),4.)
                    np.testing.assert_allclose(fields.reshape(-1,6),reference,rtol=2e-11,atol=1e-13)

    def test_vertical_plan_and_shell_do_not_impose_c2_or_horizontal(self):
        from tmatrix_order_shell import shell_schedule
        modal=modes(5);flip,signs,reps,audits=vertical_mirror_plan(5)
        plan=shell_schedule(5,vertical=True,c6=True)
        np.testing.assert_array_equal(plan['signs'],signs)
        self.assertEqual(plan['representatives'],reps)
        self.assertEqual(len(reps),5*(5+3))
        rng=np.random.default_rng(6);n=len(modal)
        t=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        t=.5*(t+signs[:,None]*t[np.ix_(flip,flip)]*signs[None,:])
        for i in reps:
            np.testing.assert_array_equal(c2x_partner_column(t[:,i],i,flip,signs),t[:,flip[i]])
        _,c2signs,_,_=c2x_plan(5)
        self.assertGreater(np.linalg.norm(t-c2signs[:,None]*t[np.ix_(flip,flip)]*c2signs[None,:]),1.)
        with self.assertRaises(ValueError):shell_schedule(5,c2=True,vertical=True)

    def test_regular_fields_transform_as_a_proper_rotation(self):
        rng = np.random.default_rng(78)
        points = rng.normal(size=(24, 3))*.4
        rotation = np.array([1., -1., -1.])
        for l in (1, 2, 5, 10):
            for m in {-l, -1, 0, 1, l}:
                for p in (0, 1):
                    # E and H both transform as vectors under a proper rotation.
                    a = regular_fields(points*rotation, (p, l, m), 4.)
                    b = regular_fields(points, (p, l, -m), 4.)
                    rotated = a.reshape(-1, 2, 3)*rotation
                    np.testing.assert_allclose(rotated.reshape(a.shape), (-1.)**l*b,
                                               rtol=2e-11, atol=1e-13)

    def test_column_reconstruction_and_plan(self):
        order = 5
        permutation, signs, representatives, audits = c2x_plan(order)
        n = len(modes(order))
        self.assertEqual(len(representatives), order*(order+3))
        self.assertTrue(all(modes(order)[i][2] < 0 for i in audits))
        rng = np.random.default_rng(4)
        t = rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        transformed = signs[:,None]*t[np.ix_(permutation,permutation)]*signs[None,:]
        t = .5*(t+transformed)
        for i in representatives:
            np.testing.assert_array_equal(c2x_partner_column(t[:,i],i,permutation,signs),t[:,permutation[i]])


if __name__ == '__main__':
    unittest.main()
