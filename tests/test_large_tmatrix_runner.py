import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_large_tmatrix as runner


class LargeRunTest(unittest.TestCase):
    def test_freeze_keeps_explicit_validated_worker_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';target=root/'runtime';worker=root/'old_worker'
            for folder,names in (('scripts',runner.SCRIPTS),('bin',runner.BINARIES)):
                (source/folder).mkdir(parents=True)
                for name in names:(source/folder/name).write_text('current '+name)
            worker.write_bytes(b'validated old worker')
            runner.frozen_runtime(target,source,worker)
            self.assertEqual((target/'bin/muller_tmatrix_fmm_server').read_bytes(),worker.read_bytes())
            runner.frozen_runtime(target,source,worker)
            worker.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'requested worker differs'):
                runner.frozen_runtime(target,source,worker)

    def test_solver_migration_and_nested_fft_survive_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'old';proof=Path(temp)/'proof.json'
            args=['runner','--out',temp,'--order','4','--fft-inner','8','--fft-radius','.5','--fft-memory','host','--mbj-overlap','8',
                  '--strict-interval','8','--incident-prefetch','--worker-binary',str(source/'worker'),
                  '--solver-restart-from',str(source),'--solver-restart-proof',str(proof)]
            with patch.object(sys,'argv',args), patch.object(runner,'frozen_runtime'), \
                 patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=1024*2**30)), \
                 patch.object(runner.psutil,'virtual_memory',return_value=SimpleNamespace(available=60*2**30)), \
                 patch.object(runner,'run',return_value={'state':'complete'}) as stage:
                self.assertEqual(runner.main(),0)
            plan=json.loads((Path(temp)/'plan.json').read_text())
            for command in (plan['command'],plan['resume_command']):
                self.assertEqual(command[command.index('--solver-restart-from')+1],str(source.resolve()))
                self.assertEqual(command[command.index('--solver-restart-proof')+1],str(proof.resolve()))
            command=plan['resume_command']
            self.assertEqual(command[command.index('--fft-inner')+1],'8')
            self.assertEqual(command[command.index('--fft-radius')+1],'0.5')
            self.assertEqual(command[command.index('--fft-memory')+1],'host')
            self.assertEqual(command[command.index('--strict-interval')+1],'8')
            self.assertEqual(command[command.index('--worker-binary')+1],str(source/'worker'))
            self.assertIn('--incident-prefetch',command)
            self.assertIn('--incident-prefetch',plan['command'])
            for call in stage.call_args_list:
                self.assertEqual(call.kwargs['env']['BEM_TMATRIX_FFT_INNER'],'8')
                self.assertEqual(call.kwargs['env']['BEM_TMATRIX_MBJ_OVERLAP'],'8')
                self.assertEqual(call.kwargs['env']['BEM_PFFT_CORR_MEMORY'],'host')
                self.assertEqual(call.kwargs['env']['BEM_HOST_GMRES_STRICT_INTERVAL'],'8')
                self.assertNotIn('BEM_TMATRIX_CIRCULANT_BLEND',call.kwargs['env'])

    def test_cutoff_controls_are_sequential_and_unlimited(self):
        with tempfile.TemporaryDirectory() as temp:
            args=['runner','--out',temp,'--order','4','--cutoff-orders','2','3','4']
            with patch.object(sys,'argv',args), patch.object(runner,'frozen_runtime'), \
                 patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=1024*2**30)), \
                 patch.object(runner.psutil,'virtual_memory',return_value=SimpleNamespace(available=60*2**30)), \
                 patch.object(runner,'run',return_value={'state':'complete'}) as stage:
                self.assertEqual(runner.main(),0)
            self.assertEqual(stage.call_count,3)
            self.assertTrue(stage.call_args.args[0][1].endswith('study_tmatrix_cutoff.py'))
            self.assertEqual(stage.call_args.args[0][-3:],['2','3','4'])
            self.assertTrue(all(c.kwargs['wall_limit_s'] is None for c in stage.call_args_list))

    def test_aspect_and_wavelength_survive_frozen_launch_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            args=['runner','--out',temp,'--ka','40','--ref','6','--order','64',
                  '--ri','1.3116','--prism-aspect',str(1/.7),'--wavelength','.532','--restart','32']
            with patch.object(sys,'argv',args), patch.object(runner,'frozen_runtime'), \
                 patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=1024*2**30)), \
                 patch.object(runner.psutil,'virtual_memory',return_value=SimpleNamespace(available=60*2**30)), \
                 patch.object(runner,'run',return_value={'state':'complete'}) as stage:
                self.assertEqual(runner.main(),0)
            plan=json.loads((Path(temp)/'plan.json').read_text())
            for command in (plan['command'],plan['resume_command']):
                self.assertEqual(float(command[command.index('--prism-aspect')+1]),1/.7)
                self.assertEqual(command[command.index('--wavelength')+1],'0.532')
                self.assertEqual(command[command.index('--restart')+1],'32')
            d=plan['physical_dimensions']
            self.assertAlmostEqual(d['height_um']/d['width_across_flats_um'],1/.7)
            self.assertIsNone(plan['time_limit'])
            self.assertIsNone(stage.call_args.kwargs['wall_limit_s'])

    def test_live_iteration_parser_keeps_latest_rhs(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp)/'worker.log'
            log.write_text('[T host GMRES 70] projected 1e-5 elapsed_s=700.000\n'
                           '[T host GMRES 1] projected 0.1 elapsed_s=11.000\n'
                           '[T host GMRES 2] projected 0.01 elapsed_s=23.000\n')
            data = runner.iteration_sample(log)
            self.assertEqual(data['current_rhs_iteration'], 2)
            self.assertEqual(data['seconds_per_iteration_recent'], 12.)

    def test_sector_solver_survives_frozen_launch_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            args=['runner','--out',temp,'--order','4','--sector-gmres','--orbit-nodes','100','--circulant-blend','.25']
            with patch.object(sys,'argv',args), patch.object(runner,'frozen_runtime'), \
                 patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=1024*2**30)), \
                 patch.object(runner.psutil,'virtual_memory',return_value=SimpleNamespace(available=60*2**30)), \
                 patch.object(runner,'run',return_value={'state':'complete'}) as stage:
                self.assertEqual(runner.main(),0)
            plan=json.loads((Path(temp)/'plan.json').read_text())
            self.assertTrue(plan['parameters']['sector_gmres'])
            self.assertIn('--sector-gmres',plan['resume_command'])
            self.assertEqual(plan['resume_command'][plan['resume_command'].index('--orbit-nodes')+1],'100')
            self.assertEqual(stage.call_args.kwargs['env']['BEM_TMATRIX_SECTOR_GMRES'],'1')
            self.assertEqual(stage.call_args.kwargs['env']['BEM_TMATRIX_ORBIT_NODES'],'100')
            log=Path(temp)/'worker.log'
            log.write_text('[T host GMRES 12] projected 1e-5 elapsed_s=12.000\n'
                           '[T sector GMRES 1] projected 0.1 elapsed_s=1.000\n'
                           '[T sector GMRES 2] projected 0.01 elapsed_s=3.000\n')
            sample=runner.iteration_sample(log)
            self.assertEqual(sample['current_rhs_iteration'],2)
            self.assertEqual(sample['seconds_per_iteration_recent'],2.)

    def test_complete_run_has_zero_remaining_eta(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); base = out/'main'; base.mkdir()
            (base/'progress.json').write_text(json.dumps({'state':'complete'}))
            runner.estimate(out, base, {'initial_worker_pairs_including_audits':1970}, 'complete')
            data = json.loads((out/'eta.json').read_text())
            self.assertEqual(data['remaining_worker_pairs_before_future_fallback'], 0)
            self.assertTrue(all(v==0 for v in data['scenario_days_excluding_setup_field_preparation_and_postprocessing'].values()))

    def test_pilot_continues_automatically_but_failure_stops_full_stage(self):
        for state in ('complete', 'failed'):
            with tempfile.TemporaryDirectory() as temp:
                args = ['run_large_tmatrix.py','--out',temp,'--order','4']
                with patch.object(sys, 'argv', args), patch.object(runner, 'frozen_runtime'), \
                     patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=1024*2**30)), \
                     patch.object(runner.psutil, 'virtual_memory', return_value=SimpleNamespace(available=60*2**30)), \
                     patch.object(runner, 'run', return_value={'state':state}) as stage:
                    result = runner.main()
                self.assertEqual(result, 0 if state=='complete' else 1)
                self.assertEqual(stage.call_count, 2 if state=='complete' else 1)
                self.assertIn('--max-pairs-this-run', stage.call_args_list[0].args[0])
                self.assertIsNone(stage.call_args_list[0].kwargs['wall_limit_s'])
                self.assertEqual(stage.call_args_list[0].kwargs['env']['BEM_HOST_GMRES_STRICT_INTERVAL'], '4')
                self.assertNotIn('BEM_TMATRIX_RECYCLE_RANK', stage.call_args_list[0].kwargs['env'])
                if state=='complete': self.assertNotIn('--max-pairs-this-run', stage.call_args_list[1].args[0])


if __name__ == '__main__': unittest.main()
