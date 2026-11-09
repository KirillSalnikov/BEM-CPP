"""Scheduling and process-level shell controls; fake worker is not physical BEM."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from tmatrix_reference import modes
from tmatrix_order_shell import shell_schedule
from tmatrix_c6 import audit_packets, production_tasks
from tmatrix_symmetry import c2x_plan
from complete_tmatrix_order_shell import AcceptedColumns


class ShellScheduleTests(unittest.TestCase):
    def test_full_build_keeps_original_controls_and_work_order(self):
        for order in (1, 4, 7, 64):
            for mirror in (False, True):
                plan = shell_schedule(order, c2=True, c6=True, mirror=mirror)
                flip, _, reps, audits = c2x_plan(order)
                control = [i for j in audits for i in (int(flip[j]), j)]
                self.assertEqual(plan['requested'], list(range(len(modes(order)))))
                self.assertEqual(plan['work'], list(dict.fromkeys(control+reps)))
                self.assertEqual(plan['c6_groups'], audit_packets(modes(order), order, mirror))

    def test_shell_covers_both_polarizations_and_mirrors_without_low_production(self):
        modal = modes(68)
        plan = shell_schedule(68, 65, c2=True, c6=True, mirror=True)
        self.assertEqual(len(plan['requested']), 1072)
        self.assertEqual(set(plan['permutation'][plan['requested']]), set(plan['requested']))
        controls = set(plan['audit_work']) | {i for group in plan['c6_groups'] for i in group}
        production = production_tasks(modal, [i for i in plan['work'] if i not in controls], (), True)
        self.assertTrue(production)
        self.assertTrue(all(modal[i][1] >= 65 for _, group in production for i in group))
        audited = {modal[i][1] for group in plan['c6_groups'] for i in group}
        self.assertTrue({5, 34, 65, 66, 68} <= audited)
        covered = controls | {i for _, group in production for i in group}
        covered |= {int(plan['permutation'][i]) for i in covered}
        self.assertTrue(set(plan['requested']) <= covered)

    def test_invalid_ranges_rejected(self):
        for order, minimum in [(8, 0), (8, 9), (257, 2), (8., 2), (8, True)]:
            with self.assertRaises(ValueError):
                shell_schedule(order, minimum)


class ShellStreamingTests(unittest.TestCase):
    def test_shell_resume_matches_direct_columns_and_does_not_claim_full_T(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('BEM_')}
        env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', BEM_TMATRIX_D6H_MESH='1')
        command = [sys.executable, str(ROOT/'scripts/build_fmm_tmatrix.py'),
                   '--shape', 'prism', '--ka', '3', '--ri', '1.3', '--ref', '1',
                   '--order', '8', '--tol', '1e-6', '--no-average',
                   '--worker-binary', str(ROOT/'tests/c6_fake_worker.py')]
        with tempfile.TemporaryDirectory(prefix='bem-shell-check-') as temp:
            base = Path(temp)
            def run(name, flags, success=True):
                r = subprocess.run(command+['--out', str(base/name)]+flags, env=env,
                                   text=True, capture_output=True, timeout=60)
                self.assertEqual(r.returncode == 0, success, r.stdout+r.stderr)
                return r
            run('direct', [])
            flags = ['--incident-min-order', '7', '--c2-reuse', '--c6-pack', '--mirror-pack']
            run('shell', flags+['--max-pairs-this-run', '2'])
            self.assertFalse(json.loads((base/'shell/summary.json').read_text())['requested_columns_complete'])
            run('shell', flags)
            report = json.loads((base/'shell/summary.json').read_text())
            self.assertFalse(report['complete'])
            self.assertTrue(report['requested_columns_complete'])
            self.assertTrue(report['symmetry_reuse']['passed'])
            self.assertTrue(report['c6_packing']['independent_reference_audits_complete'])
            self.assertFalse((base/'shell/tmatrix.npy').exists())
            requested = shell_schedule(8, 7)['requested']
            self.assertEqual(report['requested_columns'], len(requested))
            direct = np.load(base/'direct/kernel.npy')
            reader = AcceptedColumns(base/'shell', json.loads((base/'shell/manifest.json').read_text()))
            for i in requested:
                np.testing.assert_allclose(np.load(base/f'shell/columns/{i:06d}.npy'),
                                           direct[:, i], rtol=1e-8, atol=1e-10)
                accepted, _ = reader.get(i)
                np.testing.assert_allclose(accepted, direct[:, i], rtol=1e-8, atol=1e-10)
            saved = {p.name: p.read_bytes() for p in (base/'shell/columns').glob('*.npy')}
            run('shell', flags)
            self.assertEqual(json.loads((base/'shell/summary.json').read_text())['pairs_this_run'], [])
            self.assertEqual(saved, {p.name:p.read_bytes() for p in (base/'shell/columns').glob('*.npy')})
            changed = flags.copy(); changed[1] = '6'
            rejected = run('shell', changed, False)
            self.assertIn('different operator/order', rejected.stderr)


if __name__ == '__main__':
    unittest.main()
