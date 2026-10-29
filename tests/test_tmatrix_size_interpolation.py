import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from tmatrix_size_interpolation import fit_aaa, RationalMatrix, validate_family


class SizeInterpolationTest(unittest.TestCase):
    def test_recovers_complex_matrix_common_poles_on_unseen_points(self):
        rng = np.random.default_rng(71)
        a, b, c = (rng.normal(size=(3, 4))+1j*rng.normal(size=(3, 4)) for _ in range(3))
        def exact(x): return a+b/(x-(.4+.3j))+c/(x-(-.5-.2j))
        nodes = np.linspace(-1, 1, 11)
        matrices = [exact(x) for x in nodes]
        small, info = fit_aaa(nodes, matrices, max_terms=5, tolerance=1e-12, chunk_entries=2)
        large, _ = fit_aaa(nodes, matrices, max_terms=5, tolerance=1e-12, chunk_entries=1000)
        self.assertEqual(len(info['support_indices']), 3)
        for x in [-.91, -.53, -.11, .33, .87]:
            np.testing.assert_allclose(small.evaluate(x), exact(x), rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(small.evaluate(x), large.evaluate(x), rtol=1e-11, atol=1e-11)
        for x, f in zip(small.nodes, small.samples):
            actual = small.evaluate(x)
            np.testing.assert_array_equal(actual, f)
            self.assertFalse(np.shares_memory(actual, f))
        fortran, _ = fit_aaa(nodes, [np.asfortranarray(a) for a in matrices], max_terms=5, tolerance=1e-12)
        np.testing.assert_allclose(fortran.evaluate(.27), exact(.27), rtol=1e-11, atol=1e-11)
        self.assertTrue(fortran.evaluate(.27).flags.f_contiguous)

    def test_constant_zero_and_memmap_output(self):
        import tempfile
        for value in [0j, 2+3j]:
            samples = [np.full((3, 3), value) for _ in range(3)]
            model, _ = fit_aaa([0, .5, 1], samples)
            with tempfile.TemporaryDirectory() as directory:
                out = np.lib.format.open_memmap(Path(directory)/'out.npy', mode='w+', dtype=complex, shape=(3, 3))
                self.assertIs(model.evaluate(.23, out), out)
                np.testing.assert_allclose(out, value)

    def test_guards(self):
        samples = [np.ones((2, 2), complex), np.eye(2, dtype=complex)]
        with self.assertRaises(ValueError): fit_aaa([0, 0], samples)
        with self.assertRaises(ValueError): fit_aaa([0, 1], [samples[0], samples[1]*np.nan])
        with self.assertRaises(ValueError): fit_aaa([0, 1], samples, max_terms=2)
        model = RationalMatrix(np.array([0, 1]), tuple(samples), np.ones(2), (0, 1))
        with self.assertRaises(ValueError): model.evaluate(.5)
        with self.assertRaises(ValueError): model.evaluate(1.01)
        with self.assertRaises(ValueError): model.evaluate(np.nan)
        with self.assertRaises(ValueError): model.evaluate(0, samples[0])
        validate_family([{'ka': 1, 'ref': 3}, {'ka': 2, 'ref': 3}])
        with self.assertRaises(ValueError): validate_family([{'ka': 1, 'ref': 3}, {'ka': 2, 'ref': 4}])


if __name__ == '__main__': unittest.main()
