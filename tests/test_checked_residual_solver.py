"""A converged approximate system must not authorize the physical solution."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from checked_residual_solver import solve_refined


class CheckedResidualTests(unittest.TestCase):
    def test_biased_inner_operator_is_corrected_by_separate_check(self):
        diagonal = np.array([1.+.1j, 2.-.3j, 3.+.2j, 4.-.1j])
        b = np.array([1., 2j, -3., 4j])
        x, row = solve_refined(b, lambda v: 1.03*diagonal*v,
                               lambda v: diagonal*v, None, tolerance=1e-8)
        self.assertEqual(row['status'], 'complete')
        self.assertGreater(row['checks'], 1)
        self.assertGreater(row['corrections'][0]['residual'], .01)
        self.assertLess(np.linalg.norm(diagonal*x-b)/np.linalg.norm(b), 1e-8)

    def test_exhausted_corrections_do_not_report_convergence(self):
        _, row = solve_refined(np.ones(4), lambda v: 2*v, lambda v: v,
                               None, max_outer=1)
        self.assertEqual(row['status'], 'failed')
        self.assertGreater(row['residual'], .1)

    def test_nonfinite_checker_fails_and_zero_rhs_needs_no_operator(self):
        def forbidden(_):
            raise AssertionError('A zero RHS needs no action')
        x, row = solve_refined(np.zeros(4), forbidden, forbidden, None)
        np.testing.assert_array_equal(x, np.zeros(4))
        self.assertEqual(row['checks'], 0)
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite checked'):
            solve_refined(np.ones(4), lambda v: v,
                          lambda v: np.full_like(v, np.nan), None)


if __name__ == '__main__':
    unittest.main()
