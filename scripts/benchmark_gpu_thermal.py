#!/usr/bin/env python3
"""Prepared GPU Muller operator: long thermal warmup, fixed work, load gates."""
import argparse,json,statistics,time
from pathlib import Path
import benchmark_hardware as b

SUITE=Path(__file__).resolve().parents[1]/'benchmarks/gpu_thermal_v1'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--prefix',type=Path)
    p.add_argument('--pilot',action='store_true',help='Unscored setup/32-action calibration, not a hardware result')
    p.add_argument('--device',type=int,default=0);p.add_argument('--timeout',type=int,default=3600)
    a=p.parse_args();suite=json.loads((SUITE/'suite.json').read_text());case=suite['case']
    if a.device<0 or a.timeout<1:p.error('Invalid device/timeout')
    a.output=a.output.expanduser().absolute()
    if a.output.exists():p.error('Use a fresh output directory')
    a.profile='mixed';a.threads=8;a.binary=a.python=None;a.label='GPU thermal operator'
    install_suite=json.loads((b.SUITE/'suite.json').read_text());build=b.build_info(a,install_suite)
    prefix=(a.prefix or Path.home()/'.local/opt/bem-cpp').expanduser().absolute()
    native,exports=b.wrapper(prefix/'bin/bem-gpu-thermal');build['thermal_component']=dict(path=str(native),sha256=b.sha(native))
    if build['installation']['binaries'].get('bem-gpu-thermal')!=str(native):raise RuntimeError('Thermal component must match installed provenance')
    hardware=b.hardware(a,build)
    if hardware['before'] and (hardware['before'].get('utilization.gpu') or 0)>10:raise RuntimeError('GPU busy before test')
    a.output.mkdir(parents=True);env=b.environment(a,build);env.update(exports);env['BEM_MULLER_GPU_ASSEMBLY']='1'
    destination=a.output/'native.json'
    count=32 if a.pilot else case['iterations_per_repeat'];warm=0 if a.pilot else 120;repeats=1 if a.pilot else 3
    command=[str(native),str(case['ref']),str(case['ka']),str(count),str(warm),str(repeats),str(destination)]
    run=b.timed_process(command,env,a.output/'execution',a.device,a.timeout)
    result=json.loads(destination.read_text());assert result['passed'] and result['system_dofs']==case['system_dofs']
    assert result['mesh_triangles']==case['mesh_triangles'] and result['iterations_per_repeat']==count
    if b.sha(native)!=build['thermal_component']['sha256']:raise RuntimeError('Binary changed')
    report=dict(schema=1,suite=suite['id'],profile='mixed',threads=8,case=case,
        runner_sha256=b.sha(Path(__file__)),build={k:v for k,v in build.items() if k!='exports'},
        hardware=hardware,command=command,process=run,native=result,pilot=a.pilot,passed=False)
    for trial in result['trials']:
        samples=[s for s in run['telemetry'] if trial['start_unix_s']+2<=s['time']<=trial['start_unix_s']+trial['wall_s']-2]
        util=sorted(s['utilization.gpu'] for s in samples if s.get('utilization.gpu') is not None)
        temperatures=[s['temperature.gpu'] for s in samples if s.get('temperature.gpu') is not None]
        trial['gpu_utilization_median']=statistics.median(util) if util else None
        trial['gpu_utilization_p10']=util[int(.1*(len(util)-1))] if util else None
        trial['temperature_range_C']=[min(temperatures),max(temperatures)] if temperatures else None
    b.write(a.output/'results.json',report)
    if a.pilot:print('UNSCORED PILOT',result['trials'],flush=True);return
    if not result['score_candidate'] or result['warmup_s']<120 or sum(t['wall_s'] for t in result['trials'])<180:
        raise RuntimeError('Warmup/measurement duration is too short for a thermal score')
    if any(t['gpu_utilization_median'] is None or t['gpu_utilization_median']<95 or t['gpu_utilization_p10']<90 for t in result['trials']):
        raise RuntimeError('GPU load gate failed; this is not a GPU-dominated score')
    stats=b.statistics_of([t['wall_s'] for t in result['trials']]);report.update(passed=True,statistics=dict(wall_s=stats,matvec_per_s=case['iterations_per_repeat']/stats['median']),timing_stable=stats['relative_MAD']<=.1,
        comparison_key=b.digest([suite,120,3,'prepared resident operator; no setup/solver timing']))
    b.write(a.output/'results.json',report);print('VALIDATED',stats,flush=True)
if __name__=='__main__':main()
