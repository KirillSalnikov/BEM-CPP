import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import benchmark_gpu_thermal as thermal

def valid():
    return dict(passed=True,score_candidate=True,warmup_s=120.,host_device_relative_error=1e-7,mixed_strict_relative_error=1e-6,linearity_relative_error=1e-7,zero_input_output_max=0.,trials=[dict(wall_s=90.,output_relative_error=1e-7,gpu_utilization_median=100.,gpu_utilization_p10=95.) for _ in range(3)])
class ThermalScoreTests(unittest.TestCase):
    def test_complete_hot_gpu_work_passes(self):thermal.validate_thermal_score(valid())
    def test_short_failed_or_partial_work_cannot_be_ranked(self):
        for key,value in [('passed',False),('score_candidate',False),('warmup_s',119.)]:
            d=valid();d[key]=value
            with self.assertRaises(RuntimeError):thermal.validate_thermal_score(d)
        d=valid();d['trials']=d['trials'][:2]
        with self.assertRaises(RuntimeError):thermal.validate_thermal_score(d)
        d=valid()
        for r in d['trials']:r['wall_s']=1.
        with self.assertRaises(RuntimeError):thermal.validate_thermal_score(d)
    def test_low_or_unknown_gpu_load_is_not_a_gpu_score(self):
        for key,value in [('gpu_utilization_median',80.),('gpu_utilization_median',None),('gpu_utilization_p10',50.)]:
            d=valid();d['trials'][0][key]=value
            with self.assertRaises(RuntimeError):thermal.validate_thermal_score(d)
    def test_bad_or_nonfinite_operator_output_is_rejected(self):
        for key,value in [('host_device_relative_error',.01),('mixed_strict_relative_error',float('nan')),('zero_input_output_max',1e-6)]:
            d=valid();d[key]=value
            with self.assertRaises(RuntimeError):thermal.validate_thermal_score(d)
if __name__=='__main__':unittest.main()
