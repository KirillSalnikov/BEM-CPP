"""Blind reciprocal shell completion and rejection of invalid provenance."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from complete_tmatrix_order_shell import AcceptedColumns, assemble, matching_parameters
from tmatrix_reciprocal_extension import extension_layout
from tmatrix_reference import modes


def write(path, data):
    path.write_text(json.dumps(data))


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='bem-order-completion-')
        self.root = Path(self.temp.name)
        self.source, self.shell = self.root/'old', self.root/'shell'
        self.source.mkdir(); self.shell.mkdir(); (self.shell/'columns').mkdir()
        self.old_order, self.order, self.ka = 2, 4, 3.
        self.low, self.high, flip, sign, _ = extension_layout(2, 4)
        q = len(modes(4)); rng = np.random.default_rng(34)
        raw = rng.normal(size=(q, q))+1j*rng.normal(size=(q, q))
        self.full = .5*(raw+sign[:, None]*raw[np.ix_(flip, flip)].T*sign[None, :])
        self.parameters = dict(shape='prism', ref=1, ka=self.ka, ri=1.3116, ri_imag=0.,
            prism_aspect=10/7, wavelength=.532, geometry='prism', geometry_digest=None,
            binary_digest='worker-sha', incident_digest='incident-sha', projection_digest='projection-sha',
            harmonics_digest='harmonics-sha', BEM_environment={}, tol=2e-6, order=2,
            c6_driver_digest='old-driver')
        self.new_parameters = dict(self.parameters, order=4, incident_min_order=3,
                                   c6_driver_digest='new-driver', order_shell_digest='shell-sha')
        self.old_manifest = dict(signature='old-signature', parameters=self.parameters, modes=modes(2))
        self.new_manifest = dict(signature='new-signature', parameters=self.new_parameters, modes=modes(4))
        write(self.source/'manifest.json', self.old_manifest)
        write(self.shell/'manifest.json', self.new_manifest)
        write(self.source/'summary.json', dict(self.old_manifest, complete=True))
        write(self.shell/'summary.json', dict(self.new_manifest, complete=False, requested_columns_complete=True,
            incident_order_shell=dict(requested_columns=self.high.tolist())))
        np.save(self.source/'tmatrix.npy', self.full[np.ix_(self.low, self.low)])
        modal = modes(4); phase = np.array([1j**(l-p) for p,l,m in modal])
        controls = [modal.index((p, 1, m)) for p in (0, 1) for m in (-1, 1)]
        for i in [*self.high, *controls]:
            column = self.full[:, i]/(-1j*self.ka/(4*np.pi)*phase*phase[i].conj())
            np.save(self.shell/'columns'/f'{i:06d}.npy', column)
            write(self.shell/'columns'/f'{i:06d}.json', dict(mode=modal[i], scale=1.,
                  status=dict(strict_residual_x=1e-9, strict_residual_y=1e-9)))
        self.control = controls[0]

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def average(matrix, order, theta, **kwargs):
        # Control-flow fixture only: the physical/analytic averages are
        # checked by the separately queued full direct T comparison.
        m = np.zeros((len(theta), 4, 4)); m[:, 0, 0] = np.vdot(matrix, matrix).real
        return m, dict(Euler_quadrature_nodes=0, reciprocity_projection_applied=False)

    def test_blind_extension_preserves_inputs_and_does_not_claim_direct_full_T(self):
        before = {str(p):p.read_bytes() for folder in [self.source, self.shell] for p in folder.rglob('*') if p.is_file()}
        with patch('complete_tmatrix_order_shell.analytic_mueller_tmatrix', self.average):
            report = assemble(self.source, self.shell, self.root/'out', ntheta=181)
        np.testing.assert_allclose(np.load(self.root/'out/tmatrix.npy'), self.full, rtol=1e-14, atol=1e-14)
        self.assertTrue(report['sampled_controls_passed'])
        self.assertFalse(report['complete_direct_T'])
        self.assertFalse(report['completion_proven'])
        self.assertFalse((self.root/'out/summary.json').exists())
        for path, data in before.items(): self.assertEqual(Path(path).read_bytes(), data)

    def test_material_worker_operator_and_convention_mismatches_reject(self):
        matching_parameters(self.parameters, self.new_parameters)
        for key, value in [('ri',1.3), ('ka',3.1), ('ref',2), ('prism_aspect',1.),
                           ('binary_digest','changed'), ('harmonics_digest','changed'),
                           ('BEM_environment',{'BEM_TMATRIX_DUFFY_ORDER':'4'}), ('incident_min_order',2)]:
            target = copy.deepcopy(self.new_parameters); target[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                matching_parameters(self.parameters, target)

    def test_residual_gate_blocks_postprocessing(self):
        path = self.shell/'columns'/f'{self.high[0]:06d}.json'
        d = json.loads(path.read_text()); d['status']['strict_residual_x'] = 1e-3; write(path, d)
        with patch('complete_tmatrix_order_shell.analytic_mueller_tmatrix') as average:
            with self.assertRaisesRegex(ValueError, 'residual gate'):
                assemble(self.source, self.shell, self.root/'out')
            average.assert_not_called()

    def test_wrong_reciprocity_or_projection_rejected_by_independent_control(self):
        path = self.shell/'columns'/f'{self.control:06d}.npy'
        value = np.load(path); value[self.high[0]] += 10.; np.save(path, value)
        with patch('complete_tmatrix_order_shell.analytic_mueller_tmatrix') as average:
            with self.assertRaisesRegex(ValueError, 'controls reject'):
                assemble(self.source, self.shell, self.root/'out')
            average.assert_not_called()
        self.assertFalse((self.root/'out/tmatrix.npy').exists())
        self.assertFalse(json.loads((self.root/'out/report.json').read_text())['sampled_controls_passed'])

    def test_uncommitted_c6_and_unknown_inferred_entries_rejected(self):
        path = self.shell/'columns'/f'{self.high[0]:06d}.json'
        d = json.loads(path.read_text()); d['derived_by'] = 'reciprocity'; write(path, d)
        reader = AcceptedColumns(self.shell, json.loads((self.shell/'manifest.json').read_text()))
        with self.assertRaisesRegex(ValueError, 'Unknown or inferred'):
            reader.get(int(self.high[0]))
        d['derived_by'] = 'C6_packed_RHS'; d['c6_batch'] = 'f'*24; write(path, d)
        reader.parameters['c6_pack'] = True
        with self.assertRaises(FileNotFoundError):
            reader.get(int(self.high[0]))

    def test_source_overwrite_and_changed_modes_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            assemble(self.source, self.shell, self.source/'overwrite')
        manifest = json.loads((self.shell/'manifest.json').read_text())
        manifest['modes'][0], manifest['modes'][1] = manifest['modes'][1], manifest['modes'][0]
        with self.assertRaisesRegex(ValueError, 'modal order'):
            AcceptedColumns(self.shell, manifest)


if __name__ == '__main__':
    unittest.main()
