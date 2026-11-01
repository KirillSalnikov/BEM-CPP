"""Protocol, failed-packet quarantine and checkpoint tests, with a known linear map."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_large_tmatrix import task_counts


class AdaptiveToleranceTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
        self.env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',BEM_TEST_FAKE_HYBRID='1')

    def tearDown(self):
        self.temp.cleanup()

    def run_case(self,name,adaptive=True,extra=(),success=True):
        out=self.root/name
        cmd=[sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),
            '--shape','prism','--ref','1','--ka','1','--ri','1.3','--order','6',
            '--tol','3e-5' if adaptive else '2e-6','--c2-reuse','--c6-pack','--mirror-pack',
            '--incident-prefetch','--no-average','--worker-binary',str(ROOT/'tests/c6_fake_worker.py'),
            '--out',str(out)]
        if adaptive:cmd+=['--adaptive-tight-tol','2e-6']
        result=subprocess.run(cmd+list(extra),cwd=ROOT,env=self.env,text=True,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
        self.assertEqual(result.returncode==0,success,result.stdout)
        return out

    def test_failed_loose_audit_retries_before_commit_and_survives_restart(self):
        baseline=self.run_case('reference',adaptive=False)
        self.env['BEM_TEST_FAKE_CROSS_TALK']='adaptive'
        stop=task_counts(6)['independent_control_pairs']+1
        out=self.run_case('adaptive',extra=['--max-pairs-this-run',str(stop)])
        self.assertFalse(json.loads((out/'summary.json').read_text())['complete'])
        self.assertTrue(list((out/'adaptive_tolerance').glob('*.json')))
        self.assertEqual(list((out/'c6_batches').glob('*.json')),[])
        self.run_case('adaptive')
        summary=json.loads((out/'summary.json').read_text())
        self.assertTrue(summary['complete'])
        # Lower degrees may still legitimately try the loose limit. The
        # previously rejected audit must resume directly with the tight one.
        audits=[p for p in summary['pairs_this_run'] if p['rhs_kind']=='audit']
        self.assertTrue(audits)
        self.assertTrue(all(p['requested_tolerance']==2e-6 for p in audits))
        self.assertTrue(summary['c6_packing']['independent_reference_audits_complete'])
        np.testing.assert_allclose(np.load(baseline/'tmatrix.npy'),np.load(out/'tmatrix.npy'),atol=1e-10,rtol=1e-8)
        self.run_case('adaptive')
        self.assertEqual(json.loads((out/'summary.json').read_text())['pairs_this_run'],[])

    def test_tight_control_rejects_residual_below_loose_ceiling(self):
        self.env.update(BEM_TEST_FAKE_BAD_RESIDUAL='1',BEM_TEST_FAKE_BAD_RESIDUAL_VALUE='1e-5')
        out=self.run_case('bad',success=False)
        self.assertTrue((out/'rejected_solve.json').exists())
        self.assertEqual(list((out/'columns').glob('*.npy')),[])

    def test_old_protocol_fails_before_saving_columns(self):
        self.env['BEM_TEST_FAKE_OLD_PROTOCOL']='1'
        out=self.run_case('old',success=False)
        self.assertEqual(list((out/'columns').glob('*.npy')),[])

    def test_failure_at_tight_tolerance_remains_fatal(self):
        self.env['BEM_TEST_FAKE_CROSS_TALK']='audit'
        out=self.run_case('broken',success=False)
        self.assertTrue(list((out/'adaptive_tolerance').glob('*.json')))
        self.assertEqual(list((out/'c6_batches').glob('*.json')),[])


if __name__=='__main__':unittest.main()
