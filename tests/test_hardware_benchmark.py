import importlib.util,json,math,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('bench',Path(__file__).resolve().parents[1]/'scripts/benchmark_hardware.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def matrix():return [[0.]+[1.]+[0.]*15,[180.]+[2.]+[0.]*15]
def report(seconds,key='same',passed=True,source='abc'):
    return dict(passed=passed,hardware=dict(label='GPU'),build=dict(numerical_source_sha256=source,host_compiler_build_ids=['GCC 14'],installation=dict(nvcc='CUDA 13.4')),cases=[dict(id='test',comparison_key=key,statistics=dict(wall_s=dict(median=seconds,n=5,relative_MAD=.01)))])

class BenchmarkTests(unittest.TestCase):
    def test_scaled_error_does_not_divide_by_weak_elements(self):
        ref=matrix();actual=matrix();actual[0][2]=1e-7
        error=b.matrix_error(actual,ref)
        self.assertEqual(error['max_scaled_M11'],1e-7);self.assertGreater(error['relative_L2'],0)

    def test_truncated_nonfinite_and_wrong_angle_rejected(self):
        for value in [matrix()[:1],[[0.]+[float('nan')]+[0.]*15,matrix()[1]],[[1.]+matrix()[0][1:],matrix()[1]]]:
            with self.assertRaises(ValueError):b.validate_matrix(value,2)

    def test_median_and_MAD_are_robust_to_one_slow_run(self):
        stats=b.statistics_of([1.,1.,1.,1.,100.]);self.assertEqual(stats['median'],1);self.assertEqual(stats['MAD'],0);self.assertEqual(stats['max'],100)

    def test_incomparable_and_unvalidated_reports_cannot_be_ranked(self):
        for candidate in [report(1,key='different-threads'),report(1,passed=False)]:
            with self.assertRaises(ValueError):b.compare_reports(report(2),candidate)
        candidate=report(1);candidate['cases'][0]['statistics']['wall_s']['n']=1
        with self.assertRaises(ValueError):b.compare_reports(report(2),candidate)

    def test_speedup_and_software_change_labels(self):
        row=b.compare_reports(report(8),report(4))[0];self.assertEqual(row['speedup_baseline_over_candidate'],2);self.assertEqual(row['scope'],'hardware/platform comparison')
        row=b.compare_reports(report(8),report(4,source='changed'))[0];self.assertEqual(row['scope'],'software/toolchain and hardware comparison')

    def test_failed_solver_is_not_a_timing_result(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(b,'gpu_sample',return_value=None):
            with self.assertRaises(RuntimeError):b.timed_process([sys.executable,'-c','raise SystemExit(3)'],dict(__import__('os').environ),Path(temp)/'run',0,5)

    def test_bem_residual_gate_rejects_fast_unconverged_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'stdout.log').write_text('')
            u=[[[1.,1.] for j in range(4)] for i in range(4)]
            d=dict(physical=dict(mueller=u,theta_degrees=[0.,180.],parallel_fmm_residual=.1),mbj=dict(fmm_residual_verified=True,fmm_residual=.1))
            (root/'result.json').write_text(json.dumps(d))
            with self.assertRaises(ValueError):b.read_result(dict(kind='bem_fixed',angles=2,residual_limit=1e-5),root,'mixed')

    def test_partial_T_matrix_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'stdout.log').write_text('');(root/'result').mkdir()
            (root/'result/summary.json').write_text(json.dumps(dict(complete=False,completed_columns=1,mode_count=48)))
            (root/'result/average.json').write_text(json.dumps(dict(theta=[0.,180.],mueller=[[[1.,1.] for j in range(4)] for i in range(4)])))
            with self.assertRaises(ValueError):b.read_result(dict(kind='bem_tmatrix',angles=2,work=dict(columns=48)),root,'mixed')

    def test_native_binary_is_pinned_in_command(self):
        case=dict(kind='mbs',arguments=['--threads','{threads}'])
        command=b.command_for(case,dict(native_binary='/immutable/solver',executable='/mutable/wrapper'),'gpu-fp32',8,Path('/fresh/run'))
        self.assertEqual(command[0],'/immutable/solver');self.assertIn('8',command);self.assertEqual(command[-2],'/fresh/run/result')

    def test_missing_gpu_counters_remain_unknown(self):
        with patch.object(b,'smi_path',return_value='/smi'),patch.object(b,'capture',return_value='N/A, N/A, N/A, N/A, N/A, N/A'):
            sample=b.gpu_sample(0);self.assertIsNone(sample['power.draw']);self.assertIsNone(sample['memory.used'])

if __name__=='__main__':unittest.main()
