import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tmatrix_reference import modes
from tmatrix_reciprocal_extension import complete_order_extension, extension_layout, extension_work_counts


class ReciprocalExtensionTests(unittest.TestCase):
    @staticmethod
    def reciprocal_fixture(order):
        # Scalar indexing oracle independent of the vectorized completion.
        basis = modes(order)
        lookup = {mode: i for i, mode in enumerate(basis)}
        rng = np.random.default_rng(91)
        raw = rng.normal(size=(len(basis), len(basis))) + 1j*rng.normal(size=(len(basis), len(basis)))
        result = raw.copy()
        for i, (pi, li, mi) in enumerate(basis):
            for j, (pj, lj, mj) in enumerate(basis):
                result[i, j] = .5*(raw[i, j]+(-1.)**(mi+mj)*raw[lookup[pj, lj, -mj], lookup[pi, li, -mi]])
        return result

    def test_signed_transpose_and_polarization_embedding(self):
        full = self.reciprocal_fixture(4)
        for old in (1, 2, 3):
            low, high, *_ = extension_layout(old, 4)
            actual = complete_order_extension(full[np.ix_(low, low)], full[:, high], old, 4)
            np.testing.assert_array_equal(actual, full)
            # The electric low-order block is not in a leading square slice.
            self.assertGreater(low[len(low)//2], len(low)//2)

    def test_nonreciprocal_known_entries_are_preserved_not_projected(self):
        full = self.reciprocal_fixture(4)
        low, high, *_ = extension_layout(2, 4)
        full[low[0], low[1]] += 7j
        full[low[2], high[3]] += 3+4j
        source = full.copy()
        old, new = full[np.ix_(low, low)].copy(), full[:, high].copy()
        actual = complete_order_extension(old, new, 2, 4)
        np.testing.assert_array_equal(actual[np.ix_(low, low)], old)
        np.testing.assert_array_equal(actual[:, high], new)
        np.testing.assert_array_equal(full, source)
        self.assertGreater(np.linalg.norm(actual-full), 1.)
        self.assertGreater(np.linalg.norm(actual[np.ix_(high, low)].imag), 1.)

    def test_missing_rows_are_not_read_from_reference(self):
        full = self.reciprocal_fixture(3)
        low, high, *_ = extension_layout(1, 3)
        expected = complete_order_extension(full[np.ix_(low, low)], full[:, high], 1, 3)
        full[np.ix_(high, low)] = np.nan
        observed = complete_order_extension(full[np.ix_(low, low)], full[:, high], 1, 3)
        np.testing.assert_array_equal(observed, expected)

    def test_invalid_inputs_and_alias_rejected(self):
        full = self.reciprocal_fixture(3)
        low, high, *_ = extension_layout(1, 3)
        a, b = full[np.ix_(low, low)], full[:, high]
        for orders in [(0, 3), (3, 3), (4, 3), (1.5, 3), (True, 3), (1, 257)]:
            with self.subTest(orders=orders), self.assertRaises(ValueError):
                extension_layout(*orders)
        with self.assertRaises(ValueError):
            complete_order_extension(a, b[:, :-1], 1, 3)
        bad = b.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            complete_order_extension(a, bad, 1, 3)
        with self.assertRaises(ValueError):
            complete_order_extension(a, b, 1, 3, out=np.empty_like(full, dtype=np.complex64))
        # All high columns form two bands; a view with the required shape can alias out.
        alias = full[:, :len(high)]
        with self.assertRaises(ValueError):
            complete_order_extension(a, alias, 1, 3, out=full)

    def test_cost_is_only_mode_count(self):
        counts = extension_work_counts(64, 68)
        self.assertEqual(counts['old_modes'], 8448)
        self.assertEqual(counts['new_incident_modes'], 1072)
        self.assertEqual(counts['full_modes'], 9520)
        self.assertAlmostEqual(counts['avoided_column_fraction'], 8448/9520)


if __name__ == '__main__':
    unittest.main()
