"""C6v T builds: independent references, checkpoint migration and corruption."""
import hashlib
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
from tmatrix_checkpoint_migration import physical_parameters


class VerticalReuseTest(unittest.TestCase):
    def test_bullet_auto_reconstruction_resume_and_wrong_covariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
            env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',BEM_TEST_FAKE_C6V='1')
            base=[sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),
                  '--particle','2','1','1','--ka','2','--ref','0','--order','4',
                  '--tol','1e-6','--no-average','--worker-binary',str(ROOT/'tests/c6_fake_worker.py')]
            def run(name,extra=(),environment=env,success=True):
                p=subprocess.run(base+['--out',str(root/name)]+list(extra),env=environment,
                    cwd=ROOT,capture_output=True,text=True,timeout=60)
                self.assertEqual(p.returncode==0,success,p.stdout+p.stderr)
                return p
            run('direct',['--particle-symmetry','off'])
            run('auto')
            report=json.loads((root/'auto/summary.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['symmetry_reuse']['method'],'sigma_v_xz')
            self.assertTrue(report['symmetry_reuse']['passed'])
            self.assertFalse(report['parameters'].get('c2_reuse',False))
            self.assertFalse(report['c6_packing']['horizontal_reflection_parity'])
            self.assertTrue(report['c6_packing']['vertical_reflection_reuse'])
            self.assertGreater(report['reconstructed_columns'],0)
            np.testing.assert_allclose(np.load(root/'auto/tmatrix.npy'),np.load(root/'direct/tmatrix.npy'),
                                       rtol=2e-10,atol=1e-12)
            run('auto')
            self.assertEqual(json.loads((root/'auto/summary.json').read_text())['pairs_this_run'],[])
            bad=run('wrong',environment=dict(env,BEM_TEST_FAKE_C6V='0'),success=False)
            self.assertIn('sigma_v_xz column audit failed',bad.stderr)
            self.assertFalse(any(json.loads(p.read_text()).get('derived_by')=='sigma_v_xz'
                                 for p in (root/'wrong/columns').glob('*.json')))

    def test_add_vertical_reuse_to_existing_c6_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'old';target=root/'new'
            env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
            env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',BEM_TEST_FAKE_C6V='1')
            base=[sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),
                  '--particle','2','1','1','--particle-symmetry','off','--c6-pack',
                  '--ka','2','--ref','0','--order','4','--tol','1e-6','--no-average',
                  '--worker-binary',str(ROOT/'tests/c6_fake_worker.py')]
            def run(out,extra=()):
                p=subprocess.run(base+['--out',str(out)]+list(extra),env=env,cwd=ROOT,
                                 capture_output=True,text=True,timeout=60)
                self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            run(source)
            m=json.loads((source/'manifest.json').read_text());params=m['parameters']
            proof=dict(source_binary_digest=params['binary_digest'],target_binary_digest=params['binary_digest'],
                physical_parameters_sha256=hashlib.sha256(json.dumps(physical_parameters(params),sort_keys=True).encode()
                    +(source/'directions.bin').read_bytes()).hexdigest(),validated=True,
                physical_source_hashes_before={'operator':'unchanged'},physical_source_hashes_after={'operator':'unchanged'},
                maximum_field_relative_difference=0.,maximum_strict_residual=1e-12,
                equivalence_kind='identical_worker_and_incident_inputs',identical_incident_fields=True,
                reuse_identical_solver_audits=True)
            proof_path=root/'proof.json';proof_path.write_text(json.dumps(proof))
            extra=['--vertical-reuse','--solver-restart-from',str(source),'--solver-restart-proof',str(proof_path)]
            run(target,extra)
            report=json.loads((target/'solver_migration.json').read_text())
            self.assertEqual(report['copied_columns'],len(m['modes']))
            self.assertTrue(report['identical_solver_coded_audits_reused'])
            summary=json.loads((target/'summary.json').read_text())
            self.assertTrue(summary['symmetry_reuse']['passed'])
            self.assertEqual(summary['pairs_this_run'],[])
            np.testing.assert_array_equal(np.load(target/'tmatrix.npy'),np.load(source/'tmatrix.npy'))


if __name__=='__main__':unittest.main()
