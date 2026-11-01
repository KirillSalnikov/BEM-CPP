import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tmatrix_size_taylor import pade_interval_guard, evaluate_pade, evaluate_pade_checked
from tmatrix_continuation_policy import select_window, accept_holdouts


def model(q, p=None, scale=1.):
    return np.asarray(p if p is not None else [[[1.]]], complex), np.asarray(q, complex), dict(scale=scale, rank=len(q)-1)


class PoleTests(unittest.TestCase):
    def test_interior_pole_cannot_hide_behind_finite_endpoint(self):
        m = model([1., -2.])
        self.assertTrue(np.isfinite(evaluate_pade(m, 1.)).all())
        self.assertFalse(pade_interval_guard(m, (0., 1.))['passed'])
        with self.assertRaises(ValueError): evaluate_pade_checked(m, 1.)

    def test_removable_pole_is_flagged_not_silently_cancelled(self):
        m = model([1., -2.], [[[1.]], [[-2.]]])
        g = pade_interval_guard(m, (0., 1.))
        self.assertTrue(g['poles'][0]['possible_removable_pole'])
        self.assertFalse(g['passed'])

    def test_nonreal_near_pole_and_scale(self):
        pole = .5 + .001j
        g = pade_interval_guard(model([1., -1 / pole], scale=.2), (0., .2))
        self.assertFalse(g['passed'])
        self.assertAlmostEqual(g['poles'][0]['delta_real'], .1)
        self.assertAlmostEqual(g['poles'][0]['distance_to_segment'], .0002)

    def test_remote_poles_pass_and_rank_deficiency_does_not(self):
        m = model([1., -.1j, .02])
        self.assertTrue(pade_interval_guard(m, (-.5, .5))['passed'])
        np.testing.assert_allclose(evaluate_pade_checked(m, .3), evaluate_pade(m, .3))
        m[2]['rank'] = 1
        self.assertFalse(pade_interval_guard(m, (-.5, .5))['passed'])


class PolicyTests(unittest.TestCase):
    def fixture(self):
        models = [dict(name='low', degree=2), dict(name='high', degree=4)]
        radii = [.125, .25, .5]
        rows = [dict(model=m['name'], radius=r, sign=s, pole_guard_passed=True,
                     acceptance_error=(r / (.25 if m['name']=='low' else .5))**4 * .01)
                for m in models for r in radii for s in (-1, 1)]
        return rows, radii, models

    def test_select_then_fresh_holdouts_before_center_move(self):
        rows, radii, models = self.fixture()
        s = select_window(rows, radii, models)
        self.assertEqual(s['selection']['model'], 'high')
        self.assertEqual(s['selection']['radius'], .25)
        holdouts = [dict(delta=sign*fraction*.25, acceptance_error=.004, independent_reference=True, pole_guard_passed=True)
                    for sign in (-1,1) for fraction in (.25,.75,.9)]
        receipt = accept_holdouts(s, holdouts, 3., 4.)
        self.assertEqual(receipt['next_center'], 3.25)
        holdouts[-1]['acceptance_error'] = .011
        self.assertFalse(accept_holdouts(s, holdouts, 3., 4.)['passed'])
        holdouts[-1].update(acceptance_error=None, pole_guard_passed=False)
        rejected = accept_holdouts(s, holdouts, 3., 4.)
        self.assertFalse(rejected['passed'])
        self.assertIsNone(rejected['next_center'])

    def test_no_acceptance_after_failed_inner_radius(self):
        rows, radii, models = self.fixture()
        for r in rows:
            if r['model']=='high':r['acceptance_error'] = .02 if r['radius']==.125 else 0.
        s = select_window(rows, radii, models)
        self.assertEqual(s['selection']['model'], 'low')
        self.assertEqual(s['selection']['radius'], .125)

    def test_no_incomplete_nonfinite_or_fake_holdouts(self):
        rows, radii, models = self.fixture()
        with self.assertRaises(ValueError):select_window(rows[:-1], radii, models)
        rows[0]['acceptance_error'] = float('nan')
        with self.assertRaises(ValueError):select_window(rows, radii, models)
        rows, radii, models = self.fixture();s=select_window(rows,radii,models)
        with self.assertRaises(ValueError):accept_holdouts(s, [], 3., 4.)


if __name__ == '__main__':
    unittest.main()
