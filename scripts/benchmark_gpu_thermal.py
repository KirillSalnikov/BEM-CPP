#!/usr/bin/env python3
"""Prepared GPU Muller operator: long thermal warmup, fixed work, load gates."""
import argparse,json,statistics,time,math,sys,os,threading
from pathlib import Path
import benchmark_hardware as b

SUITE=Path(__file__).resolve().parents[1]/'benchmarks/gpu_thermal_v1'
def validate_thermal_score(result):
    trials=result['trials']
    if not result.get('passed') or not result.get('score_candidate') or len(trials)<3:
        raise RuntimeError('Incomplete/unvalidated native work cannot be ranked')
    if result['warmup_s']<120 or sum(t['wall_s'] for t in trials)<180:
        raise RuntimeError('Warmup/measurement duration is too short for a thermal score')
    gates={'host_device_relative_error':2e-6,'mixed_strict_relative_error':2e-4,'linearity_relative_error':2e-6,'zero_input_output_max':0.}
    for key,limit in gates.items():
        value=result[key]
        if not math.isfinite(value) or value<0 or value>limit:raise RuntimeError('Operator validation failed: '+key)
    for trial in trials:
        if not math.isfinite(trial['wall_s']) or not math.isfinite(trial['output_relative_error']) or trial['output_relative_error']>2e-6:
            raise RuntimeError('Invalid trial output/timing')
        if trial['gpu_utilization_median'] is None or trial['gpu_utilization_p10'] is None or trial['gpu_utilization_median']<95 or trial['gpu_utilization_p10']<90:
            raise RuntimeError('GPU load gate failed; this is not a GPU-dominated score')

def compare_reports(baseline,candidate):
    if not baseline.get('passed') or not candidate.get('passed'):raise ValueError('Unvalidated thermal results cannot be ranked')
    if baseline.get('comparison_key')!=candidate.get('comparison_key'):raise ValueError('Different thermal workload/protocol')
    same_source=baseline['build']['numerical_source_sha256']==candidate['build']['numerical_source_sha256'] and baseline['build'].get('thermal_source_sha256')==candidate['build'].get('thermal_source_sha256')
    same_compilers=baseline['build']['host_compiler_build_ids']==candidate['build']['host_compiler_build_ids'] and baseline['build']['installation']['nvcc']==candidate['build']['installation']['nvcc']
    return dict(speedup_baseline_over_candidate=baseline['statistics']['wall_s']['median']/candidate['statistics']['wall_s']['median'],same_numerical_and_workload_source=same_source,same_recorded_toolchain=same_compilers,scope='hardware/platform comparison' if same_source and same_compilers else 'software/toolchain and hardware comparison')

def main():
    if len(sys.argv)>1 and sys.argv[1]=='compare':
        p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);a=p.parse_args(sys.argv[2:]);print(json.dumps(compare_reports(json.loads(a.baseline.read_text()),json.loads(a.candidate.read_text())),indent=2));return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--prefix',type=Path)
    p.add_argument('--paused-processes',type=Path,help='Recorded PID/start-time lease for deliberately suspended GPU jobs')
    p.add_argument('--pilot',action='store_true',help='Unscored setup/32-action calibration, not a hardware result')
    p.add_argument('--repeats',type=int,default=3,help='At least three; increase on faster GPUs to span >=180 measured seconds')
    p.add_argument('--device',type=int,default=0);p.add_argument('--timeout',type=int,default=3600)
    a=p.parse_args();suite=json.loads((SUITE/'suite.json').read_text());case=suite['case']
    if a.device<0 or a.timeout<1 or a.repeats<3:p.error('Invalid device/timeout')
    a.output=a.output.expanduser().absolute()
    if a.output.exists():p.error('Use a fresh output directory')
    a.profile='mixed';a.threads=8;a.binary=a.python=None;a.label='GPU thermal operator'
    install_suite=json.loads((b.SUITE/'suite.json').read_text());build=b.build_info(a,install_suite)
    prefix=(a.prefix or Path.home()/'.local/opt/bem-cpp').expanduser().absolute()
    native,exports=b.wrapper(prefix/'bin/bem-gpu-thermal');build['thermal_component']=dict(path=str(native),sha256=b.sha(native));build['thermal_source_sha256']=b.sha(Path(build['source_root'])/'tools/benchmark_gpu_thermal.cpp')
    if build['installation']['binaries'].get('bem-gpu-thermal')!=str(native):raise RuntimeError('Thermal component must match installed provenance')
    hardware=b.hardware(a,build)
    if hardware['before'] and (hardware['before'].get('utilization.gpu') or 0)>10:raise RuntimeError('GPU busy before test')
    a.output.mkdir(parents=True);env=b.environment(a,build);env.update(exports);env['BEM_MULLER_GPU_ASSEMBLY']='1'
    destination=a.output/'native.json'
    count=32 if a.pilot else case['iterations_per_repeat'];warm=0 if a.pilot else 120;repeats=1 if a.pilot else a.repeats
    command=[str(native),str(case['ref']),str(case['ka']),str(count),str(warm),str(repeats),str(destination)]
    lease=json.loads(a.paused_processes.read_text()) if a.paused_processes else {'processes':[]}
    allowed_paused={str(v['pid']):str(v['start_ticks']) for v in lease['processes']}
    guard=[];stop=threading.Event()
    def monitor_processes():
        while not stop.is_set():
            text=b.capture([b.smi_path(),'-i',str(a.device),'--query-compute-apps=pid','--format=csv,noheader,nounits'])
            entry=dict(time=time.time(),processes=[],forbidden=[])
            if text is None:entry['error']='GPU process query unavailable'
            for token in (text or '').splitlines():
                pid=token.strip()
                if not pid.isdigit():
                    entry['error']='Unrecognized GPU process query output'
                    continue
                try:
                    status=Path('/proc/'+pid+'/stat').read_text().split();executable=str(Path('/proc/'+pid+'/exe').resolve())
                    ours=executable==str(native) and int(status[3])==os.getpid()
                    paused=status[2]=='T' and allowed_paused.get(pid)==status[21]
                    entry['processes'].append(dict(pid=int(pid),state=status[2],start_ticks=status[21],benchmark=ours,leased_pause=paused))
                    if not ours and not paused:entry['forbidden'].append(int(pid))
                except FileNotFoundError:continue
                except (OSError,ValueError,IndexError) as error:
                    entry['error']='Cannot identify GPU PID '+pid+': '+str(error)
            guard.append(entry)
            stop.wait(1)
    thread=threading.Thread(target=monitor_processes,daemon=True);thread.start()
    try:run=b.timed_process(command,env,a.output/'execution',a.device,a.timeout)
    finally:stop.set();thread.join(timeout=35)
    if thread.is_alive():guard.append(dict(error='GPU process monitor did not terminate',forbidden=[]))

    result=json.loads(destination.read_text());assert result['passed'] and result['system_dofs']==case['system_dofs']
    assert result['mesh_triangles']==case['mesh_triangles'] and result['iterations_per_repeat']==count
    if b.sha(native)!=build['thermal_component']['sha256']:raise RuntimeError('Binary changed')
    report=dict(schema=1,suite=suite['id'],profile='mixed',threads=8,case=case,
        runner_sha256=b.sha(Path(__file__)),build={k:v for k,v in build.items() if k!='exports'},
        hardware=hardware,command=command,process=run,native=result,pilot=a.pilot,passed=False,compute_process_guard=guard,paused_process_lease=lease,telemetry_driver_sha256=b.sha(Path(b.__file__)))
    for trial in result['trials']:
        samples=[s for s in run['telemetry'] if trial['start_unix_s']+2<=s['time']<=trial['start_unix_s']+trial['wall_s']-2]
        util=sorted(s['utilization.gpu'] for s in samples if s.get('utilization.gpu') is not None)
        temperatures=[s['temperature.gpu'] for s in samples if s.get('temperature.gpu') is not None]
        trial['gpu_utilization_median']=statistics.median(util) if util else None
        trial['gpu_utilization_p10']=util[int(.1*(len(util)-1))] if util else None
        trial['temperature_range_C']=[min(temperatures),max(temperatures)] if temperatures else None
    b.write(a.output/'results.json',report)
    if a.pilot:print('UNSCORED PILOT',result['trials'],flush=True);return
    if not guard or any(v.get('error') or v['forbidden'] for v in guard):raise RuntimeError('Concurrent/unidentified GPU compute process; hardware score rejected')
    validate_thermal_score(result)
    stats=b.statistics_of([t['wall_s'] for t in result['trials']]);report.update(passed=True,statistics=dict(wall_s=stats,matvec_per_s=case['iterations_per_repeat']/stats['median']),timing_stable=stats['relative_MAD']<=.1,
        comparison_key=b.digest([suite,120,3,'prepared resident operator; no setup/solver timing; exclusive compute process guard v1']))
    b.write(a.output/'results.json',report);print('VALIDATED',stats,flush=True)
if __name__=='__main__':main()
