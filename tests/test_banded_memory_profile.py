"""GPU regression: minimal FMM bands and standalone radiation projection."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('BEM_TEST_GPU')=='1','set BEM_TEST_GPU=1 for CUDA integration')
class BandedMemoryTest(unittest.TestCase):
    def test_operator_and_radiation_agree_with_full_workspaces(self):
        with tempfile.TemporaryDirectory(prefix='bem-banded-test-') as temporary:
            out=Path(temporary)
            incident=out/'incident.txt';incident.write_text('0 57 17 1\n')
            env=os.environ.copy();env.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',
                BEM_FMM_BANDED_SPLIT_DEPTH='1',BEM_FMM_BANDED_COARSE_MAX_LEAF='4096',
                BEM_FMM_BANDED_MIDDLE_MAX_LEAF='512',BEM_FMM_PAIR_CURRENTS='0',
                BEM_FMM_PHASE_CACHE='0',BEM_MIXED_ITERATIVE_REFINEMENT='1',
                BEM_FARFIELD_ANALYTIC_ALPHA='1',BEM_FARFIELD_SPECTRAL_ALPHA='48')
            reference=None
            for minimal,assembly in (('1','0'),('0','0'),('1','1')):
                env.update(BEM_FMM_BANDED_MINIMAL_WORKSPACE=minimal,BEM_MULLER_GPU_ASSEMBLY=assembly)
                result=out/f'result_{minimal}_{assembly}.json'
                command=[str(ROOT/'bin/muller_nodal_fmm_demo_fp32'),
                    '--shape','prism','--ref','2','--ka','5','--ri','1.3','--edge-mode','hdiv',
                    '--quad','7','--duffy-order','4','--digits','5','--max-leaf','64',
                    '--fmm-near-radius','3','--tol','1e-6','--max-iters','200','--gmres-restart','60',
                    '--mbj-only','--mbj-nodes','12','--no-dense-validation','--no-checkpoint',
                    '--near-correction-cache',str(out/'operator.near'),'--mbj-cache',str(out/'operator.mbj'),
                    '--out',str(result),'--ntheta','37','--orient-average','128','1','1',
                    '--orient-file',str(incident)]
                process=subprocess.run(command,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                self.assertEqual(process.returncode,0,process.stdout)
                data=json.loads(result.read_text());values=np.asarray(data['mueller'])
                self.assertLess(data['iterations']['maximum_residual'],1e-6)
                if reference is None: reference=values
                else: self.assertLess(np.linalg.norm(values-reference)/np.linalg.norm(reference),1e-10)


if __name__=='__main__': unittest.main()
