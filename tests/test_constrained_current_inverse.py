import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from constrained_current_inverse import ConstrainedCurrentInverse


class ConstrainedInverseTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12026)

    def complex(self, *shape):
        return self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape)

    def test_prior_recovers_nonradiating_currents_and_affine_constraints(self):
        # R has a 15-dimensional nullspace. Ordinary minimum-norm inversion
        # cannot recover its hidden current; the physical prior can.
        r, c, x = self.complex(9, 24), self.complex(4, 24), self.complex(24, 3)
        metric = np.exp(self.rng.uniform(-2, 2, 24))
        weights = np.exp(self.rng.uniform(-2, 2, 9))
        truth = x @ self.complex(3, 2)
        inverse = ConstrainedCurrentInverse(r, c, x, current_metric=metric, field_weights=weights)
        got, report = inverse.solve(r @ truth, c @ truth)
        np.testing.assert_allclose(got, truth, rtol=2e-11, atol=2e-11)
        ordinary = ConstrainedCurrentInverse(r, c, np.empty((24, 0)), radiation_rcond=1e-12,
                                            current_metric=metric, field_weights=weights)
        minimum, _ = ordinary.solve(r @ truth, c @ truth)
        self.assertGreater(np.linalg.norm(minimum - truth) / np.linalg.norm(truth), .1)
        self.assertLess(max(report['constraint_relative_residual']), 1e-12)
        # Same full-rank physical subspace with a complex invertible basis.
        change = self.complex(3, 3) + 4 * np.eye(3)
        altered = ConstrainedCurrentInverse(r, c, x @ change, current_metric=metric, field_weights=weights)
        again, _ = altered.solve(r @ truth, c @ truth)
        np.testing.assert_allclose(again, got, rtol=2e-11, atol=2e-11)

    def test_complement_and_complex_phase(self):
        r, c, x = self.complex(8, 20), self.complex(3, 20), self.complex(20, 2)
        inverse = ConstrainedCurrentInverse(r, c, x, radiation_rcond=1e-12)
        field, d = self.complex(8, 2), self.complex(3, 2)
        got, _ = inverse.solve(field, d)
        np.testing.assert_allclose(r @ got, field, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(c @ got, d, rtol=1e-12, atol=1e-12)
        rotated, _ = inverse.solve(1j * field, 1j * d)
        np.testing.assert_allclose(rotated, 1j * got, rtol=1e-12, atol=1e-12)
        # Independent equality-constrained weighted minimum-norm solution.
        empty = ConstrainedCurrentInverse(r, c, np.empty((20, 0)), radiation_rcond=1e-12)
        actual, _ = empty.solve(field, d)
        reference = np.linalg.pinv(np.r_[c, r]) @ np.r_[d, field]
        np.testing.assert_allclose(actual, reference, rtol=1e-11, atol=1e-11)

    def test_dependent_constraints_and_zero_boundary_rhs(self):
        r = self.complex(8, 20)
        c = self.complex(2, 20)
        c = np.r_[c, c[:1] * (2 + 3j)]
        inverse = ConstrainedCurrentInverse(r, c, np.empty((20, 0)), radiation_rcond=1e-12)
        field = self.complex(8)
        got, _ = inverse.solve(field, np.zeros(3))
        np.testing.assert_allclose(c @ got, 0, atol=1e-12)
        np.testing.assert_allclose(r @ got, field, rtol=1e-12, atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            inverse.solve(field, [0, 0, 1])

    def test_no_constraints_and_truncated_singular_directions(self):
        r = np.diag([1., .1, 1e-8]).astype(complex)
        inv = ConstrainedCurrentInverse(r, np.empty((0, 3)), np.empty((3, 0)), radiation_rcond=1e-3)
        got, _ = inv.solve(np.ones(3, complex), np.empty(0))
        np.testing.assert_allclose(got, [1, 10, 0])
        self.assertEqual(inv.report['radiation']['rank'], 2)
        # Retained physical prior restores that weak direction without
        # asserting that the truncated unconstrained inverse is exact.
        inv = ConstrainedCurrentInverse(r, np.empty((0, 3)), np.eye(3)[:, 2:])
        got, _ = inv.solve([0, 0, 1e-8j], np.empty(0))
        np.testing.assert_allclose(got, [0, 0, 1j])
        invisible = ConstrainedCurrentInverse(np.zeros((2, 3)), np.empty((0, 3)), np.eye(3))
        got, _ = invisible.solve(np.ones(2), np.empty(0))
        np.testing.assert_array_equal(got, np.zeros(3))
        self.assertFalse(invisible.report['full_column_rank_prior'])

    def test_invalid_data(self):
        r, c, x = np.eye(3), np.empty((0, 3)), np.empty((3, 0))
        for kw in [dict(field_weights=[1, 0, 1]), dict(current_metric=[1j, 1, 1]),
                   dict(radiation_rcond=-1), dict(prior_rcond=float('nan'))]:
            with self.assertRaises(ValueError):
                ConstrainedCurrentInverse(r, c, x, **kw)
        inv = ConstrainedCurrentInverse(r, c, x)
        for f, d in [([1, 2], []), ([1, 2, np.nan], []), ([1, 2, 3], [1])]:
            with self.assertRaises(ValueError):
                inv.solve(f, d)


if __name__ == '__main__':
    unittest.main()
