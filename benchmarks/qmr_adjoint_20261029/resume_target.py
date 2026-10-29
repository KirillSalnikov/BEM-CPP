"""Resume the unchanged frozen production run after transpose/adjoint and classical QMR controls."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'runs/ka40_n1p3116_l0p532_aspect10over7_pipeline_20261029'
bench=ROOT/'benchmarks/qmr_adjoint_20261029'
for process in psutil.process_iter(['pid','name','cmdline']):
 if process.info['name'] in ('muller_solver_f','muller_tmatrix_') or any(Path(a).name in ('muller_solver_fft_probe','muller_tmatrix_fmm_server') for a in (process.info['cmdline'] or [])):
  raise RuntimeError('GPU worker still running: '+str(process.info['pid']))
stop=json.loads((bench/'checkpoint_stop.json').read_text())
base=out/'main_ref6_L64'
sha=hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest()
assert sha==stop['source_manifest_sha256']
assert len(list((base/'columns').glob('*.npy')))==stop['column_files']
command=json.loads((out/'plan.json').read_text())['resume_command']
with (out/'supervisor_adjoint_resume.log').open('a') as log:
 process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'supervisor_pid':process.pid,'resumed_unix':time.time(),'command':command,'retained_columns':stop['column_files'],
        'retained_packets':stop['committed_packets'],'manifest_sha256':sha,'worker_and_solver_settings_unchanged':True,
        'wall_time_limit':None,'incident_prefetch_enabled':True}
(bench/'production_resumed.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
