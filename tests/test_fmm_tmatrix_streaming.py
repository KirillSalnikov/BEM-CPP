#!/usr/bin/env python3
"""Optional GPU integration test for partial-column restart correctness."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('BEM_TEST_GPU')=='1','set BEM_TEST_GPU=1 for the CUDA streaming test')
class StreamingTest(unittest.TestCase):
    def test_multilevel_device_builder_matches_host_builder(self):
        with tempfile.TemporaryDirectory(prefix='bem-tm-ml-device-') as temporary:
            directory=Path(temporary)
            base=[sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),
                  '--shape','prism','--ref','4','--ka','12','--ri','1.3','--order','8',
                  '--tol','1e-5','--columns','0','2','--no-average']
            env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
            env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',BEM_FMM_MULTILEVEL='1',
                BEM_FMM_BANDED_SPLIT_DEPTH='3',BEM_FMM_BANDED_COARSE_MAX_LEAF='512',
                BEM_FMM_PAIR_CURRENTS='0',BEM_FMM_PHASE_CACHE='0',BEM_FMM_STRICT_PAIR_WORKSPACE='0',
                BEM_TMATRIX_MBJ_NODES='50')
            for name in ('host','device','fast_device','fast_tight'):
                local=env.copy();local.update(BEM_MULLER_GPU_ASSEMBLY='0' if name=='host' else '1',
                                             BEM_FMM_MULTILEVEL_GPU_ASSEMBLY='1')
                extra=['--fast-mlfma'] if name.startswith('fast_') else []
                if name=='fast_tight':
                    extra+=['--tol','1e-8']
                process=subprocess.run(base+extra+['--out',str(directory/name)],cwd=ROOT,env=local,
                    text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
                self.assertEqual(process.returncode,0,process.stdout)
                report=json.loads((directory/name/'summary.json').read_text())
                expected='host_gmres_gpu_banded_fmm' if name=='host' else 'paired_device_gmres'
                self.assertEqual(report['operator']['solver_backend'],expected)
                self.assertLess(report['maximum_strict_residual'],1e-5)
                if name=='fast_tight':
                    self.assertLess(report['maximum_strict_residual'],1e-8)
                    self.assertIn('tightening inner target after FP64 residual rejection',
                                  (directory/name/'worker.log').read_text())
            for index in range(2):
                a=np.load(directory/'host/columns'/f'{index:06d}.npy')
                for name in ('device','fast_device','fast_tight'):
                    b=np.load(directory/name/'columns'/f'{index:06d}.npy')
                    self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(a),1e-4,(name,index))

    def test_c2_reuse_matches_independent_columns_and_rejects_bad_audit(self):
        with tempfile.TemporaryDirectory(prefix='bem-tm-c2-test-') as temporary:
            directory=Path(temporary)
            base=[sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),
                  '--shape','prism','--ref','3','--ka','3','--ri','1.3','--order','3',
                  '--tol','1e-4','--c2-audit-tol','2e-4','--no-average']
            env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
            env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
            partial=subprocess.run(base+['--c2-reuse','--max-pairs-this-run','1','--out',str(directory/'c2')],
                                   cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
            self.assertEqual(partial.returncode,0,partial.stdout)
            incomplete=json.loads((directory/'c2/summary.json').read_text())
            self.assertFalse(incomplete['complete'])
            self.assertEqual(incomplete['independently_solved_columns'],2)
            self.assertEqual(incomplete['reconstructed_columns'],0)
            for name,extra in [('full',[]),('c2',['--c2-reuse'])]:
                process=subprocess.run(base+extra+['--out',str(directory/name)],cwd=ROOT,env=env,
                                       text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
                self.assertEqual(process.returncode,0,process.stdout)
            a=np.load(directory/'full/tmatrix.npy');b=np.load(directory/'c2/tmatrix.npy')
            self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(a),1e-4)
            metadata=json.loads((directory/'c2/summary.json').read_text())
            self.assertTrue(metadata['symmetry_reuse']['passed'])
            self.assertGreater(metadata['reconstructed_columns'],0)
            self.assertLess(metadata['independently_solved_columns'],len(a))
            self.assertEqual(metadata['maximum_strict_residual_applies_to'],'independently_solved_columns_only')
            before=(directory/'c2/tmatrix.npy').read_bytes()
            resumed=subprocess.run(base+['--c2-reuse','--out',str(directory/'c2')],cwd=ROOT,env=env,
                                   text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
            self.assertEqual(resumed.returncode,0,resumed.stdout)
            self.assertEqual(before,(directory/'c2/tmatrix.npy').read_bytes())
            failed=subprocess.run(base+['--c2-reuse','--c2-audit-tol','1e-14','--out',str(directory/'reject')],
                                  cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn('C2 column audit failed',failed.stdout)
            self.assertFalse((directory/'reject/tmatrix.npy').exists())
            for path in (directory/'reject/columns').glob('*.json'):
                self.assertNotIn('derived_by',json.loads(path.read_text()))

    def test_banded_host_builder_matches_device_builder(self):
        with tempfile.TemporaryDirectory(prefix='bem-tm-banded-test-') as directory:
            directory=Path(directory)
            base=[sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),
                  '--shape','prism','--ref','1','--ka','1','--ri','1.3','--order','2',
                  '--tol','1e-5','--no-average']
            env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
            for name in ('device','banded'):
                local=env.copy()
                if name=='banded':
                    local.update(BEM_FMM_BANDED_SPLIT_DEPTH='1',BEM_FMM_BANDED_COARSE_MAX_LEAF='256',
                                 BEM_MULLER_GPU_ASSEMBLY='0',BEM_FMM_PAIR_CURRENTS='0',BEM_FMM_PHASE_CACHE='0',
                                 BEM_TMATRIX_MBJ_NODES='12')
                process=subprocess.run(base+['--out',str(directory/name)],cwd=ROOT,env=local,
                                       text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                self.assertEqual(process.returncode,0,process.stdout)
            a=np.load(directory/'device/tmatrix.npy');b=np.load(directory/'banded/tmatrix.npy')
            self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(a),1e-4)
            metadata=json.loads((directory/'banded/summary.json').read_text())
            self.assertEqual(metadata['operator']['solver_backend'],'host_gmres_gpu_banded_fmm')
            self.assertLess(metadata['maximum_strict_residual'],1e-5)

    def test_partial_restart_and_operator_guard(self):
        with tempfile.TemporaryDirectory(prefix='bem-tm-test-') as directory:
            path=Path(directory)
            base=[sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),
                  '--shape','sphere','--ref','0','--ka','1','--ri','1.5','--order','1',
                  '--tol','1e-5','--no-average','--out',str(path/'resumed')]
            env=os.environ.copy(); env['OPENBLAS_NUM_THREADS']='1'; env['OMP_NUM_THREADS']='4'
            def run(command,success=True):
                result=subprocess.run(command,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                if success:
                    self.assertEqual(result.returncode,0,result.stdout)
                return result
            run(base+['--columns','0','2'])
            report=json.loads((path/'resumed/summary.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual(report['completed_columns'],2)
            self.assertFalse((path/'resumed/kernel.npy').exists())
            column=path/'resumed/columns/000000.npy'
            previous=column.read_bytes()
            run(base)
            self.assertEqual(column.read_bytes(),previous)
            report=json.loads((path/'resumed/summary.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['completed_columns'],6)
            self.assertFalse(report['dense_matrix_formed'])
            continuous=base.copy(); continuous[-1]=str(path/'continuous')
            run(continuous)
            np.testing.assert_allclose(np.load(path/'resumed/tmatrix.npy'),np.load(path/'continuous/tmatrix.npy'),
                                       rtol=2e-5,atol=2e-8)
            mismatch=base.copy(); mismatch[mismatch.index('--ri')+1]='1.6'
            result=run(mismatch,False)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('different operator/order',result.stdout)


if __name__=='__main__':
    unittest.main()
