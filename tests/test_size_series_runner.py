import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_size_series as runner


class SizeSeriesTest(unittest.TestCase):
    def test_profiles_match_three_band_uniform_tree_depths(self):
        for ref in range(3,7):
            seg=2**ref;nz=math.ceil(math.sqrt(3)*seg);nz+=nz%2
            points=7*12*seg*(seg+nz);env=runner.profile(ref)
            depths=[math.ceil(math.log(2*points/int(env[k]),8)) for k in
                    ('BEM_TMATRIX_MAX_LEAF','BEM_FMM_BANDED_MIDDLE_MAX_LEAF','BEM_FMM_BANDED_COARSE_MAX_LEAF')]
            self.assertEqual(depths,[depths[0],depths[0]-1,depths[0]-2])
            self.assertEqual(int(env['BEM_FMM_BANDED_SPLIT_DEPTH']),depths[-1])
        self.assertEqual(runner.profile(4)['BEM_FMM_BANDED_SPLIT_DEPTH'],'2')

    def test_large_sizes_have_bounded_ref_and_extra_modal_margin(self):
        for ka in [10,15,20,30,40,50]:
            ref,order=runner.selection(ka)
            self.assertLessEqual(ref,6)
            self.assertGreaterEqual(order,1.291648635563488*ka+8)

    def test_expensive_full_stage_is_deferred_after_checkpointed_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            series=runner.Series(out,{'deadline_unix':time.time()+1000,'ri':1.3116})
            def fake_run(command,supervisor,**kwargs):
                self.assertLess(kwargs['wall_limit_s'],1000)
                self.assertIn('--max-pairs-this-run',command)
                dest=Path(command[command.index('--out')+1]);dest.mkdir(exist_ok=True)
                (dest/'summary.json').write_text(json.dumps({'complete':False,'pairs_this_run':[{'total_worker_s':100}]*4}))
                return {'state':'complete'}
            try:
                with patch.object(runner,'run',side_effect=fake_run) as run:
                    with self.assertRaises(runner.Deferred):series.build(10,3,21)
                    self.assertEqual(run.call_count,1)
                self.assertFalse(runner.read(out/'ka10_ref3_L21'/'summary.json')['complete'])
            finally:series.done.set();series.thread.join()

    def test_deadline_prevents_new_solver(self):
        with tempfile.TemporaryDirectory() as directory:
            series=runner.Series(Path(directory),{'deadline_unix':time.time()-1,'ri':1.3116})
            try:
                with patch.object(runner,'run') as run:
                    with self.assertRaises(runner.EndSeries):series.build(10,3,21)
                    run.assert_not_called()
            finally:series.done.set();series.thread.join()


if __name__=='__main__':unittest.main()
