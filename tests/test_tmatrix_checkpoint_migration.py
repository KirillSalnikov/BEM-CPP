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


class MigrationTest(unittest.TestCase):
    def test_complete_packed_migration_and_rejected_operator_or_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);old=p/'old';new=p/'new'
            worker=p/'new_worker.py';worker.write_bytes((ROOT/'tests/c6_fake_worker.py').read_bytes()+b'\n# same fake operator, new solver build\n');worker.chmod(0o755)
            env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
            env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(ROOT/'scripts'))
            base=[sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),'--shape','prism','--ka','1',
                  '--ri','1.3','--ref','1','--order','7','--tol','1e-6','--no-average',
                  '--c2-reuse','--c6-pack','--mirror-pack']
            def run(out,extra=(),environment=env,success=True):
                r=subprocess.run(base+['--out',str(out)]+list(extra),cwd=ROOT,env=environment,text=True,
                                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
                self.assertEqual(r.returncode==0,success,r.stdout);return r
            run(old,['--worker-binary',str(ROOT/'tests/c6_fake_worker.py')])
            m=json.loads((old/'manifest.json').read_text());params=m['parameters']
            proof={'source_binary_digest':params['binary_digest'],
                   'target_binary_digest':hashlib.sha256(worker.read_bytes()).hexdigest(),
                   'physical_parameters_sha256':hashlib.sha256(json.dumps(physical_parameters(params),sort_keys=True).encode()+(old/'directions.bin').read_bytes()).hexdigest(),
                   'validated':True,'physical_source_hashes_before':{'operator':'unchanged'},
                   'physical_source_hashes_after':{'operator':'unchanged'},
                   'maximum_field_relative_difference':0.,'maximum_strict_residual':1e-12}
            proof_path=p/'proof.json';proof_path.write_text(json.dumps(proof))
            same_proof=dict(proof,target_binary_digest=params['binary_digest'],
                equivalence_kind='identical_worker_and_incident_inputs',identical_incident_fields=True,
                reuse_identical_solver_audits=True)
            same_path=p/'same_proof.json';same_path.write_text(json.dumps(same_proof))
            same_extra=['--worker-binary',str(ROOT/'tests/c6_fake_worker.py'),'--incident-prefetch',
                '--solver-restart-from',str(old),'--solver-restart-proof',str(same_path)]
            run(p/'same_worker',same_extra)
            same_report=json.loads((p/'same_worker/solver_migration.json').read_text())
            self.assertTrue(same_report['identical_solver_coded_audits_reused'])
            self.assertFalse(same_report['new_coded_audits_required'])
            self.assertEqual(json.loads((p/'same_worker/summary.json').read_text())['pairs_this_run'],[])
            np.testing.assert_array_equal(np.load(p/'same_worker/tmatrix.npy'),np.load(old/'tmatrix.npy'))
            bad=run(p/'different_solver_setting',same_extra,dict(env,BEM_HOST_GMRES_STRICT_INTERVAL='8'),False)
            self.assertIn('audit reuse requires identical',bad.stdout)
            extra=['--worker-binary',str(worker),'--solver-restart-from',str(old),'--solver-restart-proof',str(proof_path)]
            candidate=dict(env,BEM_TMATRIX_SECTOR_GMRES='1',BEM_TMATRIX_ORBIT_NODES='50')
            run(new,extra,candidate)
            report=json.loads((new/'solver_migration.json').read_text())
            self.assertGreater(report['copied_packets'],0)
            self.assertEqual(report['copied_columns'],len(m['modes']))
            np.testing.assert_array_equal(np.load(new/'tmatrix.npy'),np.load(old/'tmatrix.npy'))
            summary=json.loads((new/'summary.json').read_text())
            self.assertTrue(summary['c6_packing']['independent_reference_audits_complete'])
            self.assertTrue(summary['pairs_this_run'])
            self.assertTrue(all(row['rhs_kind']=='audit' for row in summary['pairs_this_run']))
            run(new,extra,candidate)  # idempotent migration and normal resume
            self.assertEqual(json.loads((new/'summary.json').read_text())['pairs_this_run'],[])
            result=run(p/'bad_geometry',extra+['--prism-aspect','1.4'],candidate,False)
            self.assertIn('changes the physical operator',result.stdout)
            result=run(p/'bad_quadrature',extra,dict(candidate,BEM_TMATRIX_DUFFY_ORDER='8'),False)
            self.assertIn('changes the physical operator',result.stdout)
            receipt=json.loads(next((old/'c6_batches').glob('*.json')).read_text())
            path=old/'columns'/f'{receipt["columns"][0]:06d}.npy';original=path.read_bytes()
            damaged=np.load(path);damaged[0]+=.1;np.save(path,damaged)
            result=run(p/'bad_data',extra,candidate,False)
            self.assertIn('checksum mismatch',result.stdout)
            path.write_bytes(original)
            proof['maximum_field_relative_difference']=.01;proof_path.write_text(json.dumps(proof))
            result=run(p/'bad_proof',extra,candidate,False)
            self.assertIn('validation proof',result.stdout)


if __name__=='__main__':unittest.main()
