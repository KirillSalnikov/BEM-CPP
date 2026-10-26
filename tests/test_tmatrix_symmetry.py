import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from tmatrix_incident import regular_fields
from tmatrix_symmetry import c2x_plan, c2x_partner_column
from tmatrix_reference import modes


class C2Test(unittest.TestCase):
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
