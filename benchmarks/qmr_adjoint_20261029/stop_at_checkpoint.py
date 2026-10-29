import hashlib,json,os,signal,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'runs/ka40_n1p3116_l0p532_aspect10over7_pipeline_20261029'
progress=out/'main_ref6_L64/progress.json'
initial=json.loads(progress.read_text())['pairs_this_run']
supervisor=int((out/'supervisor.pid').read_text())
print('Waiting for next committed packet after',initial,'supervisor',supervisor,flush=True)
while True:
 p=json.loads(progress.read_text())
 if p['state']=='solving' and p['pairs_this_run']>initial:break
 if not psutil.pid_exists(supervisor):raise RuntimeError('supervisor exited before boundary')
 time.sleep(.1)
root=psutil.Process(supervisor);children=root.children(recursive=True)
record={'reason':'User-requested true transpose/adjoint and classical QMR validation at committed boundary','progress':p,
        'supervisor_pid':supervisor,'child_pids':[q.pid for q in children],'stopped_at_unix':time.time()}
os.kill(supervisor,signal.SIGTERM)
_,alive=psutil.wait_procs(children+[root],timeout=30)
alive=[q for q in alive if q.is_running() and q.status()!=psutil.STATUS_ZOMBIE]
if alive:raise RuntimeError('old processes remain alive: '+str([q.pid for q in alive]))
base=out/'main_ref6_L64'
record['column_files']=len(list((base/'columns').glob('*.npy')))
record['committed_packets']=len(list((base/'c6_batches').glob('*.json')))
record['source_manifest_sha256']=hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest()
(out/'checkpoint_adjoint_handover.json').write_text(json.dumps(record,indent=2)+'\n')
(ROOT/'benchmarks/qmr_adjoint_20261029/checkpoint_stop.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
