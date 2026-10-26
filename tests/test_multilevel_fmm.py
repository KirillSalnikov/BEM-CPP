"""Optional CUDA regressions against independent point evaluation at every band."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('BEM_TEST_GPU') == '1', 'set BEM_TEST_GPU=1')
class MultilevelTest(unittest.TestCase):
    def test_m2l_cache_budget_fallback(self):
        with tempfile.TemporaryDirectory(prefix='bem-cache-budget-') as temporary:
            directory=Path(temporary)
            env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
            env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',BEM_FMM_MULTILEVEL='1',
                BEM_FMM_BANDED_SPLIT_DEPTH='3',BEM_FMM_BANDED_COARSE_MAX_LEAF='512',
                BEM_FMM_MULTILEVEL_M2L_FP32='1',BEM_FMM_MULTILEVEL_M2L_CACHE='1',
                BEM_FMM_PAIR_CURRENTS='0',BEM_MULLER_GPU_ASSEMBLY='0',BEM_FMM_PHASE_CACHE='0',
                BEM_FMM_STRICT_PAIR_WORKSPACE='0',BEM_PROBE_SEED='83')
            for name,reserve in (('cached','2048'),('fallback','1000000')):
                env['BEM_FMM_MULTILEVEL_CACHE_RESERVE_MB']=reserve
                command=[str(ROOT/'bin/muller_fmm_action_probe'),'4','12',
                         str(directory/'operator'),str(directory/name)]
                result=subprocess.run(command,cwd=ROOT,env=env,text=True,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout)
                self.assertIn('FP32 M2L read cache' if name=='cached' else 'M2L read cache skipped',result.stdout)
            a=np.fromfile(directory/'cached.bin',np.complex128)
            b=np.fromfile(directory/'fallback.bin',np.complex128)
            self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(a),1e-12)

    def test_two_and_three_levels_real_and_absorbing_media(self):
        with tempfile.TemporaryDirectory(prefix='bem-multilevel-') as temporary:
            out = Path(temporary)
            for ref, ka, coarse, middle, absorption in (
                (4, 12, 512, 0, 0.), (4, 12, 512, 0, .03), (5, 25, 1024, 128, 0.)
            ):
                env = os.environ.copy()
                # Isolate the diagnostic from any caller's production knobs.
                env = {k: v for k, v in env.items() if not k.startswith('BEM_')}
                env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                    BEM_FMM_BANDED_SPLIT_DEPTH='3', BEM_FMM_BANDED_COARSE_MAX_LEAF=str(coarse),
                    BEM_FMM_BANDED_MIDDLE_MAX_LEAF=str(middle), BEM_FMM_PAIR_CURRENTS='0',
                    BEM_MULLER_GPU_ASSEMBLY='0', BEM_FMM_PHASE_CACHE='0', BEM_FMM_FLAT_NEAR_SOURCES='0',
                    BEM_FMM_STRICT_PAIR_WORKSPACE='0', BEM_FMM_BANDED_MINIMAL_WORKSPACE='1',
                    BEM_PROBE_RI_IMAG=str(absorption), BEM_PROBE_SEED='71', BEM_PROBE_STRICT='1')
                case = out / f'{ref}_{ka}_{absorption}'
                case.mkdir()
                for enabled in (0, 1, 2, 3):
                    env['BEM_FMM_MULTILEVEL'] = str(int(enabled>0))
                    if enabled==2:
                        # Even with every mixed inner option requested, strict
                        # actions must bypass them and retain FP64 arithmetic.
                        env.update(BEM_MULLER_GPU_ASSEMBLY='1',BEM_FMM_MULTILEVEL_GPU_ASSEMBLY='1',
                            BEM_FMM_MULTILEVEL_PAIR='1',BEM_FMM_MULTILEVEL_M2L_FP32='1',
                            BEM_FMM_MULTILEVEL_L2P_FP32='1',BEM_FMM_MULTILEVEL_INTERPOLATION_FP32='1',
                            BEM_FMM_MULTILEVEL_P2M_FP32='1',
                            BEM_FMM_MULTILEVEL_SHIFT_CACHE='1',BEM_FMM_MULTILEVEL_M2L_CACHE='1',
                            BEM_FMM_MULTILEVEL_FLAT_NEAR='1',BEM_PROBE_WARM_MIXED='1',
                            BEM_FMM_MULTILEVEL_CHUNK='256')
                    if enabled==3:
                        env.update(BEM_MULLER_GPU_ASSEMBLY='0',BEM_FMM_MULTILEVEL_L2P_WARP='1')
                    prefix = case / str(enabled)
                    command = [str(ROOT/'bin/muller_fmm_action_probe'), str(ref), str(ka),
                               str(case/'operator'), str(prefix)]
                    process = subprocess.run(command, env=env, cwd=ROOT, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
                    self.assertEqual(process.returncode, 0, process.stdout)
                    self.assertTrue(json.loads(prefix.with_suffix('.json').read_text())['strict'])
                    if enabled:
                        self.assertIn('[MLFMA]', process.stdout)
                        if ref == 5: self.assertIn('level 4 -> 3', process.stdout)
                    if enabled>=2:
                        self.assertIn('FP32 M2L read cache',process.stdout)
                        self.assertIn('FP64 shift cache',process.stdout)
                a = np.fromfile(case/'0.bin', np.complex128)
                for candidate in (1,2,3):
                    b = np.fromfile(case/f'{candidate}.bin', np.complex128)
                    error = np.linalg.norm(a-b)/np.linalg.norm(a)
                    self.assertTrue(np.isfinite(b).all())
                    self.assertLess(error, 1e-9, (ref, ka, absorption, candidate, error))


if __name__ == '__main__':
    unittest.main()
