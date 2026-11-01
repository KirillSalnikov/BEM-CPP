import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from tmatrix_size_loewner import fit_loewner


class LoewnerTest(unittest.TestCase):
    def test_independent_mimo_resolvent_and_affine_part(self):
        rng = np.random.default_rng(763)
        c = rng.normal(size=(3,4))+1j*rng.normal(size=(3,4))
        b = rng.normal(size=(4,3))+1j*rng.normal(size=(4,3))
        d = rng.normal(size=(3,3)); slope = rng.normal(size=(3,3))
        poles = np.array([-.4+.2j, .7-.3j, 1.3+.4j, -.8-.5j])
        def exact(x): return d+x*slope+(c/(x-poles))@b
        nodes = np.linspace(-1, 1, 9)
        model, info = fit_loewner(nodes, [exact(x) for x in nodes], rtol=1e-12)
        self.assertLess(info['maximum_training_relative_error'], 1e-10)
        for x in [-.91, -.37, .13, .62, .97]:
            np.testing.assert_allclose(model.evaluate(x), exact(x), rtol=1e-10, atol=1e-10)

    def test_constant_and_guards(self):
        sample = np.array([[2+1j]])
        model, info = fit_loewner([-1,0,1], [sample]*3)
        np.testing.assert_array_equal(model.evaluate(.2), sample)
        self.assertEqual(info['rank'], 0)
        with self.assertRaises(ValueError): model.evaluate(1.01)
        with self.assertRaises(ValueError): fit_loewner([0,0,1], [sample]*3)
        with self.assertRaises(ValueError): fit_loewner([0,1], [sample]*2)


if __name__ == '__main__': unittest.main()
