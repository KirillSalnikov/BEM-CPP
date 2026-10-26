"""Orchestrator budget/resume checks; the BEM worker is never started."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_tmatrix_day as day


class BudgetTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='bem-day-test-')
        self.directory=Path(self.temporary.name)
        self.source=self.directory/'source'
        for folder,names in (('scripts',day.SCRIPTS),('bin',day.BINARIES)):
            (self.source/folder).mkdir(parents=True)
            for name in names:
                target=self.source/folder/name
                if folder=='scripts':
                    shutil.copy2(ROOT/folder/name,target)
                else:
                    target.write_bytes(b'CPU fixture; not an executable')
        self.out=self.directory/'run'

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self,extra):
        with patch.object(day,'ROOT',self.source), \
             patch.object(sys,'argv',['run_tmatrix_day','--out',str(self.out)]+extra), \
             patch.object(day,'run',return_value={'state':'failed','reason':'CPU test sentinel'}) as worker, \
             patch.object(day.psutil,'virtual_memory',return_value=SimpleNamespace(available=16*2**30)), \
             patch.object(day.shutil,'disk_usage',return_value=SimpleNamespace(free=128*2**30)), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                day.main()
            self.assertEqual(stopped.exception.code,2)
            return worker

    def test_unlimited_reaches_supervisor_without_deadline(self):
        worker=self.invoke(['--no-time-limit','--duffy-order','6'])
        self.assertIsNone(worker.call_args.kwargs['wall_limit_s'])
        self.assertEqual(worker.call_args.kwargs['env']['BEM_TMATRIX_DUFFY_ORDER'],'6')
        plan=json.loads((self.out/'plan.json').read_text())
        self.assertIsNone(plan['time_limit_hours_this_attempt'])
        self.assertIn('--no-time-limit',plan['resume_command'])
        self.assertNotIn('--deadline-utc',plan['resume_command'])
        status=json.loads((self.out/'status.json').read_text())
        self.assertIsNone(status['deadline_utc'])
        self.assertFalse(status['time_limit_enabled'])

    def test_bounded_mode_keeps_deadline(self):
        worker=self.invoke(['--hours','.5'])
        self.assertTrue(1790<worker.call_args.kwargs['wall_limit_s']<=1800)
        plan=json.loads((self.out/'plan.json').read_text())
        self.assertIn('--deadline-utc',plan['resume_command'])
        self.assertNotIn('--no-time-limit',plan['resume_command'])

    def test_new_controller_preserves_frozen_operator_runtime(self):
        old=self.directory/'old'
        shutil.copytree(self.source,old)
        (old/'scripts/run_tmatrix_day.py').write_text('# previous orchestration version\n')
        runtime=self.out/'runtime'
        day.freeze_runtime(runtime,old)
        before={str(p.relative_to(runtime)):p.read_bytes() for p in runtime.rglob('*') if p.is_file()}
        self.invoke(['--no-time-limit'])
        after={str(p.relative_to(runtime)):p.read_bytes() for p in runtime.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        plan=json.loads((self.out/'plan.json').read_text())
        controller=Path(plan['resume_command'][1])
        self.assertTrue(controller.is_relative_to(self.out/'controllers'))
        self.assertEqual(controller.read_bytes(),(self.source/'scripts/run_tmatrix_day.py').read_bytes())

    def test_conflicting_deadline_is_rejected(self):
        with patch.object(sys,'argv',['day','--out',str(self.out),'--no-time-limit',
                                      '--deadline-utc','2099-01-01T00:00:00+00:00']), \
             contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit):
                day.main()
        self.assertIn('cannot be combined',error.getvalue())
        self.assertFalse(self.out.exists())


if __name__=='__main__':
    unittest.main()
