"""File-content identity for resume; the fake worker is not a physics oracle."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class PhysicalRitzCheckpointTests(unittest.TestCase):
    def test_changed_prior_at_same_path_cannot_resume(self):
        with tempfile.TemporaryDirectory(prefix='bem-ritz-resume-') as temporary:
            base=Path(temporary);prior=base/'prior.bin';prior.write_bytes(b'first physical prior fixture')
            env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
            env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',BEM_TMATRIX_RITZ_PRIOR=str(prior))
            command=[sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),
                '--shape','prism','--ka','3','--ri','1.3','--ref','1','--order','3',
                '--tol','1e-6','--no-average','--columns','0','2',
                '--worker-binary',str(ROOT/'tests/c6_fake_worker.py'),'--out',str(base/'output')]
            def run():return subprocess.run(command,env=env,capture_output=True,text=True,timeout=60)
            first=run();self.assertEqual(first.returncode,0,first.stdout+first.stderr)
            manifest=json.loads((base/'output/manifest.json').read_text())
            self.assertEqual(manifest['parameters']['physical_ritz_prior_digest'],hashlib.sha256(prior.read_bytes()).hexdigest())
            columns={p.name:p.read_bytes() for p in (base/'output/columns').glob('*.npy')}
            same=run();self.assertEqual(same.returncode,0,same.stdout+same.stderr)
            prior.write_bytes(b'other physical prior fixture')
            changed=run();self.assertNotEqual(changed.returncode,0)
            self.assertIn('different operator/order',changed.stderr)
            self.assertEqual(columns,{p.name:p.read_bytes() for p in (base/'output/columns').glob('*.npy')})
            self.assertEqual(manifest,json.loads((base/'output/manifest.json').read_text()))


if __name__=='__main__':unittest.main()
