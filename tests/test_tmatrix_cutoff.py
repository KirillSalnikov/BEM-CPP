import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from study_tmatrix_cutoff import cutoff_indices, error_metrics, stable_minimum
from tmatrix_reference import modes
from tmatrix_so3_mstm import analytic_mueller_tmatrix


class CutoffTest(unittest.TestCase):
    def test_both_polarizations_keep_physical_mode_addresses(self):
        original = modes(7)
        indices = cutoff_indices(7, 3)
        self.assertEqual([original[i] for i in indices], modes(3))
        self.assertNotEqual(list(indices), list(range(len(indices))))

    @unittest.skipUnless((ROOT/'bin/tmatrix_so3_mstm').exists(), 'build analytic backend')
    def test_reduced_matrix_and_zero_padded_matrix_have_same_average(self):
        rng = np.random.default_rng(92)
        indices = cutoff_indices(3, 2)
        t = rng.normal(size=(len(indices), len(indices)))+1j*rng.normal(size=(len(indices), len(indices)))
        padded = np.zeros((len(modes(3)), len(modes(3))), complex)
        padded[np.ix_(indices, indices)] = t
        theta = np.array([0., 39., 90., 131., 180.])
        a, _ = analytic_mueller_tmatrix(t, 2, theta, threads=1)
        b, _ = analytic_mueller_tmatrix(padded, 3, theta, threads=1)
        np.testing.assert_allclose(a, b, atol=np.max(abs(a))*1e-12, rtol=1e-11)

    def test_mueller_zeros_and_nonmonotone_convergence(self):
        reference = np.zeros((3, 4, 4)); reference[:, 0, 0] = [2., 1., .5]
        candidate = reference.copy(); candidate[2, 1, 2] = .005
        result = error_metrics(candidate, reference, np.array([0, 90, 180]))
        self.assertAlmostEqual(result['maximum_all16_error_over_reference_M11'], .01)
        self.assertEqual(result['worst_entry'], 'M23')
        records = [{'order': i+1, 'acceptance_error': error} for i, error in enumerate([.2, .005, .02, .004, 0.])]
        self.assertEqual(stable_minimum(records, .01), 4)


if __name__ == '__main__': unittest.main()
