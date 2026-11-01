"""Physics identities, cheap center control, and separate error tolerances."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_tmatrix_neighbors import Pipeline, THETA, acceptance, check_family, mbs_command, read, windows
from tmatrix_size_interpolation import RationalMatrix


class NeighborTests(unittest.TestCase):
    def test_no_validation_only_full_solves(self):
        sequence = windows(40.)
        self.assertEqual((sequence[0]['lo'], sequence[0]['hi']), (39.75, 40.25))
        fitting = set()
        for window in sequence:
            for stage in window['stages']:
                self.assertEqual(stage['validation'], [40.])
                self.assertNotIn(40., stage['training'])
                fitting.update(stage['training'])
        self.assertEqual(fitting, {39.5, 39.75, 40.25, 40.5})

    def test_center_is_not_used_to_construct_linear_model(self):
        slope = np.array([[1+2j, -.2j], [.7, -3j]])
        offset = np.array([[.5, -.9j], [2j, 3.]])
        nodes = np.array([39.75, 40.25])
        samples = tuple(offset + (ka-40)*slope for ka in nodes)
        model = RationalMatrix(nodes, samples, [-1., 1.], (39.75, 40.25))
        np.testing.assert_allclose(model.evaluate(40.), offset, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(model.evaluate(39.875), offset-.125*slope,
                                   rtol=1e-14, atol=1e-14)
        with self.assertRaises(ValueError):
            model.evaluate(40.5)

    def test_four_percent_is_only_mbs(self):
        plan = dict(validation_mode='reuse_completed_center', bem_target=.01,
                    mbs_tolerance=.04, mbs_binary='/frozen/mbs', ri=1.3116, wavelength=.532)
        self.assertTrue(acceptance(.009, .0024, plan))
        self.assertFalse(acceptance(.02, .0024, plan))
        self.assertFalse(acceptance(float('nan'), .0024, plan))
        for mode in ('coherent', 'incoherent'):
            cmd = mbs_command(plan, 40.25, {'particle_L_over_D_vertex': (10/7)*np.sqrt(3)/2},
                              Path('/new/result'), mode)
            self.assertEqual(float(cmd[cmd.index('--fullauto')+1]), .04)
            self.assertEqual('--incoherent' in cmd, mode == 'incoherent')
            self.assertNotIn('--fullauto-final-checks', cmd)

    def test_absorption_reaches_native_mbs_command(self):
        plan = dict(mbs_tolerance=.04, mbs_binary='/frozen/mbs', ri=1.3116,
                    ri_imag=10., wavelength=.532)
        cmd = mbs_command(plan, 20., {'particle_L_over_D_vertex': 10/7*np.sqrt(3)/2},
                          Path('/absorbing/result'), 'coherent')
        start = cmd.index('--refractive-index')
        self.assertEqual(list(map(float, cmd[start+1:start+3])), [1.3116, 10.])

    def test_geometry_and_basis_cannot_be_mixed(self):
        plan = dict(ref=6, order=58, ri=1.3116, wavelength=.532, aspect=10/7)
        parameters = dict(ka=40.25, ref=6, order=58, ri=1.3116, ri_imag=0.,
                          wavelength=.532, prism_aspect=10/7, shape='prism')
        check_family(parameters, 40.25, plan)
        for key, value in [('prism_aspect', 1.), ('order', 64), ('ka', 40.), ('ref', 5)]:
            with self.assertRaises(ValueError):
                check_family(dict(parameters, **{key: value}), 40.25, plan)

    def test_saved_analytic_average_preserves_backend_metadata_and_layout(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.plan = {'order': 1}
        mean = np.arange(len(THETA)*16).reshape(len(THETA), 4, 4)
        metadata = dict(order=1, Csca=2., Cext=3., Euler_quadrature_nodes=0)
        pipeline.average = Mock(return_value=(mean, metadata))
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)/'result'
            pipeline.save_average(dest, np.eye(6, dtype=complex), 1., {'order': 1})
            saved = read(dest/'average.json')
            self.assertEqual(saved['order'], 1)
            self.assertEqual(saved['Euler_quadrature_nodes'], 0)
            np.testing.assert_array_equal(np.moveaxis(saved['mueller'], -1, 0), mean)
            self.assertEqual(read(dest/'summary.json')['scattering_cross_section'], 2.)


if __name__ == '__main__':
    unittest.main()
