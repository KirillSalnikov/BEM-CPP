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
    def test_hybrid_mode_and_thread_profile_survive_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            args=['runner','--out',temp,'--order','4','--sector-gmres','--hybrid-cpu','--prepare-only',
                  '--tol','3e-5','--adaptive-tight-tol','2e-6']
            with patch.object(sys,'argv',args),patch.object(runner,'frozen_runtime'), \
                 patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=1024*2**30)), \
                 patch.object(runner.psutil,'virtual_memory',return_value=SimpleNamespace(available=60*2**30)), \
                 patch.dict(runner.os.environ,{'BEM_CPU_THREADS':'16','BEM_CIRCULANT_CPU_FFT':'auto'}):
                self.assertEqual(runner.main(),0)
            plan=json.loads((Path(temp)/'plan.json').read_text())
            self.assertTrue(plan['parameters']['hybrid_cpu'])
            for command in (plan['command'],plan['resume_command']):
                self.assertIn('--hybrid-cpu',command)
                self.assertEqual(float(command[command.index('--adaptive-tight-tol')+1]),2e-6)
            self.assertEqual(plan['parameters']['adaptive_tight_tol'],2e-6)
            self.assertEqual(plan['profile']['BEM_TMATRIX_HYBRID'],'1')
            self.assertEqual(plan['profile']['BEM_TMATRIX_CPU_STRICT'],'1')
            self.assertEqual(plan['profile']['BEM_CPU_THREADS'],'16')
            self.assertEqual(plan['profile']['BEM_CIRCULANT_CPU_FFT'],'auto')
            self.assertEqual(plan['profile']['OMP_WAIT_POLICY'],runner.os.environ.get('OMP_WAIT_POLICY','PASSIVE'))
            self.assertEqual(plan['profile']['GOMP_SPINCOUNT'],runner.os.environ.get('GOMP_SPINCOUNT','0'))

    def test_hybrid_eta_counts_shared_batch_time_once(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);base=out/'main';base.mkdir()
            pair={'total_worker_s':100.,'hybrid_accounted_worker_s':50.,'incident_wait_s':2.}
            (base/'operator_metadata.json').write_text(json.dumps({'hybrid_batch_capacity':2}))
            (base/'progress.json').write_text(json.dumps({'remaining_worker_pairs':10,'last_pair':pair}))
            (base/'summary.json').write_text(json.dumps({'pairs_this_run':[pair,pair]}))
            (base/'worker.log').write_text('[T sector GMRES 2] projected 0.01 elapsed_s=100.000\n')
            runner.estimate(out,base,{'initial_worker_pairs_including_audits':12},'main')
            data=json.loads((out/'eta.json').read_text())
            self.assertEqual(data['iteration_sample'],{})
            self.assertEqual(data['pilot_mean_pair_s'],52.)
            self.assertEqual(data['last_completed_pair_s'],52.)
            self.assertAlmostEqual(data['pilot_extrapolated_remaining_days'],520/86400)

    def test_projection_and_operator_options_survive_frozen_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            args=['runner','--out',temp,'--order','14','--angular-band','18',
                  '--prism-aspect',str(10/7),'--sector-gmres','--strict-face-fft',
                  '--gpu-assembly','--gpu-host-correction','--prepare-only']
            with patch.object(sys,'argv',args),patch.object(runner,'frozen_runtime'), \
                 patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=1024*2**30)), \
                 patch.object(runner.psutil,'virtual_memory',return_value=SimpleNamespace(available=60*2**30)):
                self.assertEqual(runner.main(),0)
            plan=json.loads((Path(temp)/'plan.json').read_text())
            self.assertEqual(plan['parameters']['angular_band'],18)
            for command in [plan['command'],plan['resume_command']]:
                self.assertEqual(command[command.index('--angular-band')+1],'18')
            self.assertIn('--strict-face-fft',plan['resume_command'])
            self.assertIn('--gpu-assembly',plan['resume_command'])
            self.assertIn('--gpu-host-correction',plan['resume_command'])
            self.assertEqual(plan['profile']['BEM_TMATRIX_SECTOR_GMRES'],'1')
            self.assertEqual(plan['profile']['BEM_MULLER_GPU_ASSEMBLY'],'1')
            self.assertEqual(plan['profile']['BEM_FACE_FFT_STRICT_ONLY'],'1')
            self.assertEqual(plan['profile']['BEM_FACE_FFT_REUSE_CROSS'],'1')
            self.assertEqual(plan['profile']['BEM_MULLER_GPU_STRICT_ONLY'],'1')
            self.assertEqual(plan['profile']['BEM_FACE_FFT_SMOOTH_PADDING'],'1')
            self.assertEqual(plan['profile']['BEM_FACE_FFT_DEDUP_KERNELS'],'1')
            self.assertEqual(plan['profile']['BEM_MULLER_GPU_HOST_CORRECTION'],'1')

    def test_prepare_preserves_real_memory_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            args=['runner','--out',temp,'--order','4','--prepare-only']
            with patch.object(sys,'argv',args),patch.object(runner,'frozen_runtime'), \
                 patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=1024*2**30)), \
                 patch.object(runner.psutil,'virtual_memory',return_value=SimpleNamespace(available=8*2**30)), \
                 patch.object(runner,'run') as stage:
                with self.assertRaisesRegex(RuntimeError,'32 GiB available'):
                    runner.main()
                stage.assert_not_called()

    def test_freeze_keeps_explicit_validated_worker_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';target=root/'runtime';worker=root/'old_worker'
            for folder,names in (('scripts',runner.SCRIPTS),('bin',runner.BINARIES),('reference/particles',runner.GEOMETRY_DATA)):
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
            self.assertEqual(stage.call_count,1)
            self.assertNotIn('--max-pairs-this-run',stage.call_args.args[0])
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
            args=['runner','--out',temp,'--order','4','--sector-gmres','--orbit-nodes','100','--circulant-blend','.25',
                  '--sector-recycle-rank','8','--sector-recycle-mask','130','--vertical-preconditioner',
                  '--opposite-face-weight','.5','--opposite-face-mask','2',
                  '--inner-local-coordinates','--sector-projected-factor','.25','--sector-first-pass-factor','8']
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
            self.assertEqual(stage.call_args.kwargs['env']['BEM_TMATRIX_SECTOR_RECYCLE_RANK'],'8')
            self.assertEqual(stage.call_args.kwargs['env']['BEM_TMATRIX_SECTOR_RECYCLE_MASK'],'130')
            self.assertEqual(plan['parameters']['sector_recycle_rank'],8)
            self.assertEqual(plan['resume_command'][plan['resume_command'].index('--sector-recycle-rank')+1],'8')
            self.assertEqual(plan['resume_command'][plan['resume_command'].index('--sector-recycle-mask')+1],'130')
            self.assertIn('--vertical-preconditioner',plan['resume_command'])
            self.assertIn('--inner-local-coordinates',plan['resume_command'])
            self.assertEqual(plan['resume_command'][plan['resume_command'].index('--sector-projected-factor')+1],'0.25')
            self.assertEqual(stage.call_args.kwargs['env']['BEM_TMATRIX_VERTICAL_PRECONDITIONER'],'1')
            self.assertEqual(stage.call_args.kwargs['env']['BEM_FMM_P2P_LOCAL_COORDINATES'],'1')
            self.assertEqual(stage.call_args.kwargs['env']['BEM_SECTOR_PROJECTED_FACTOR'],'0.25')
            self.assertEqual(stage.call_args.kwargs['env']['BEM_SECTOR_FIRST_PASS_FACTOR'],'8.0')
            self.assertEqual(plan['parameters']['sector_first_pass_factor'],8.)
            self.assertEqual(plan['resume_command'][plan['resume_command'].index('--sector-first-pass-factor')+1],'8.0')
            self.assertEqual(stage.call_args.kwargs['env']['BEM_CIRCULANT_OPPOSITE_WEIGHT'],'0.5')
            self.assertEqual(stage.call_args.kwargs['env']['BEM_CIRCULANT_OPPOSITE_MASK'],'2')
            self.assertEqual(plan['parameters']['opposite_face_weight'],.5)
            self.assertEqual(plan['resume_command'][plan['resume_command'].index('--opposite-face-weight')+1],'0.5')
            self.assertEqual(plan['resume_command'][plan['resume_command'].index('--opposite-face-mask')+1],'2')
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

    def test_checkpoint_resume_keeps_one_worker_and_its_audits(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/'main_ref6_L4';(base/'columns').mkdir(parents=True)
            (base/'manifest.json').write_text('{}');(base/'columns/0.npy').write_bytes(b'validated by builder')
            args=['runner','--out',temp,'--order','4','--ref','6']
            with patch.object(sys,'argv',args),patch.object(runner,'frozen_runtime'), \
                 patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=1024*2**30)), \
                 patch.object(runner.psutil,'virtual_memory',return_value=SimpleNamespace(available=60*2**30)), \
                 patch.object(runner,'run',return_value={'state':'complete'}) as stage:
                self.assertEqual(runner.main(),0)
            self.assertEqual(stage.call_count,1)
            self.assertNotIn('--max-pairs-this-run',stage.call_args.args[0])
            self.assertIsNone(stage.call_args.kwargs['wall_limit_s'])


if __name__ == '__main__': unittest.main()
