"""Particle CLI, symmetry dispatch and persistent failed-audit fallback."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


class ParticleDispatchTests(unittest.TestCase):
    def test_failed_auto_packing_keeps_direct_columns_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
            env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',BEM_TEST_FAKE_CROSS_TALK='audit')
            base=[sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),'--particle','1','1',str(2/np.sqrt(3)),
                  '--ka','.7','--ref','0','--order','4','--tol','1e-6','--no-average',
                  '--worker-binary',str(ROOT/'tests/c6_fake_worker.py')]
            def run(out,extra=()):
                p=subprocess.run(base+['--out',str(out)]+list(extra),env=env,cwd=ROOT,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
                self.assertEqual(p.returncode,0,p.stdout)
                return json.loads((out/'summary.json').read_text())
            reference=root/'direct';actual=root/'auto'
            run(reference,['--particle-symmetry','off'])
            report=run(actual)
            self.assertTrue(report['c6_packing']['individual_fallback_after_failed_audit'])
            self.assertFalse(report['c6_packing']['independent_reference_audits_complete'])
            self.assertEqual(report['c6_packing']['packed_columns'],0)
            np.testing.assert_allclose(np.load(actual/'tmatrix.npy'),np.load(reference/'tmatrix.npy'),rtol=1e-10,atol=1e-12)
            self.assertEqual(run(actual)['pairs_this_run'],[])

    def test_incompatible_explicit_symmetry_is_rejected_before_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ('auto','off'):
                p=subprocess.run([sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),
                    '--particle','2','2','1','--particle-symmetry',mode,'--c2-reuse','--out',temporary],
                    cwd=ROOT,text=True,capture_output=True,timeout=30)
                self.assertNotEqual(p.returncode,0)
                self.assertIn('verified C2x symmetry',p.stderr)

    def test_generated_mesh_identity_survives_output_directory_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';target=root/'target'
            env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
            env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            base=[sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),'--particle','2','2','1',
                '--particle-symmetry','off','--ka','1','--ref','0','--order','2','--no-average',
                '--worker-binary',str(ROOT/'tests/c6_fake_worker.py')]
            for out,extra in ((source,['--columns','0','2']),(target,['--seed-from',str(source)])):
                p=subprocess.run(base+['--out',str(out)]+extra,env=env,cwd=ROOT,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
                self.assertEqual(p.returncode,0,p.stdout)
            a=json.loads((source/'manifest.json').read_text())['parameters']
            b=json.loads((target/'manifest.json').read_text())['parameters']
            self.assertEqual(a['geometry'],b['geometry'])
            self.assertTrue(a['geometry'].startswith('particle_obj_sha256:'))
            self.assertEqual(json.loads((target/'seed_report.json').read_text())['copied_columns'],2)


if __name__=='__main__':unittest.main()
