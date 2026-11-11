"""Bounded GPU measurement with guaranteed production SIGCONT on normal failures."""
import argparse,datetime,json,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
B=Path(__file__).resolve().parent;R=B.parents[1]
p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--q',type=int,default=3)
p.add_argument('--timeout',type=int,default=300);p.add_argument('--variants',default='baseline,refine,nested4,nested8,baseline_repeat');p.add_argument('--maxiter',type=int,default=240)
p.add_argument('--strict-interval',type=int,default=1)
p.add_argument('--production',type=Path,default=R/'runs/scice_regular_shapes_ka20_20261111/Bullet/bem',
    help='Exact production output directory to pause; use the current directory after a checkpoint migration')
a=p.parse_args();out=a.case.resolve();c=json.loads((out/'case.json').read_text())
production=a.production.resolve()
launch=production/'launch.json'
pid=json.loads(launch.read_text())['pid'] if launch.exists() else None;parent=None
try:
 if pid is not None:
  candidate=psutil.Process(pid)
  if candidate.is_running() and str(production) in ' '.join(candidate.cmdline()):parent=candidate
except psutil.NoSuchProcess:pass
created=parent.create_time() if parent else None
if parent and os.getpgid(pid)!=pid:raise RuntimeError('Production process group mismatch')
# Without a verified production owner, obey the shared GPU queue instead of
# silently competing with a different worker after a campaign migration.
gpu_locks=[]
if parent is None:
 import fcntl
 for suffix in ('','.1','.2'):
  lock=open('/home/kirill/neuro/agent-sandbox/.gpu.lock'+suffix,'a')
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_locks.append(lock)
env={k:v for k,v in os.environ.items() if not k.startswith(('BEM_','MBS_'))}
env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',
    BEM_MULLER_GPU_ASSEMBLY='0',BEM_FMM_MULTILEVEL_GPU_ASSEMBLY='0',BEM_FMM_MULTILEVEL='1',
    BEM_FMM_BANDED_SPLIT_DEPTH=str(3 if c['ref']==5 else 2),
    BEM_FMM_BANDED_COARSE_MAX_LEAF='3696',BEM_FMM_BANDED_MIDDLE_MAX_LEAF='462',
    BEM_FMM_BANDED_COARSE_ORDER_REFERENCE_DEPTH='3',BEM_FMM_ORDER_REFERENCE_DEPTH='5',
    BEM_FMM_ALLOW_DEPTH6='1',BEM_FMM_MULTILEVEL_COMPACT_TEMP='1',BEM_FMM_BANDED_MINIMAL_WORKSPACE='1',
    BEM_FMM_CONCURRENT_MEDIA='1',BEM_FMM_MULTILEVEL_M2L_FP32='1',BEM_FMM_MULTILEVEL_P2M_FP32='1',
    BEM_FMM_MULTILEVEL_L2P_FP32='1',BEM_FMM_MULTILEVEL_INTERPOLATION_FP32='1',
    BEM_FMM_MULTILEVEL_SHIFT_CACHE='1',BEM_FMM_MULTILEVEL_M2L_CACHE='1',BEM_FMM_MULTILEVEL_FLAT_NEAR='1',
    BEM_FMM_MULTILEVEL_PAIR='1',BEM_FMM_MULTILEVEL_P2M_SHARED='1',BEM_FMM_M2L_DIRECTION_BLOCKS='1',
    BEM_FMM_P2P_SHARED_SOURCES='1',BEM_FMM_P2P_THREADS='128',BEM_FMM_P2P_SHARED_REAL='1',
    BEM_FMM_P2P_DOUBLE_SINGLE='1',BEM_FMM_P2P_DS_COMPENSATED='1',BEM_FMM_P2P_DS_FACTORED='1',
    BEM_FMM_P2P_DS_THREADS='64',BEM_FMM_P2P_DS_LEAF_SPLIT='32',
    BEM_HOST_GMRES_STRICT_INTERVAL=str(a.strict_interval),BEM_MIXED_ADAPTIVE_INNER='1',RQ_VARIANTS=a.variants,RQ_MAXITER=str(a.maxiter))
command=[str(B/'probe'),'solve',c['shape'],str(c['ref']),str(c['ka']),str(a.q),str(out)]
start=time.time();child=None;code=None;timeout=False;resumed=False
# A separate watchdog restores the production group even if this controller dies.
watchdog=None
if parent:
 watchdog_code='''import os,signal,time,psutil,sys
time.sleep(float(sys.argv[3]))
try:
 p=psutil.Process(int(sys.argv[1]))
 if p.create_time()==float(sys.argv[2]):os.killpg(p.pid,signal.SIGCONT)
except ProcessLookupError:pass
'''
 watchdog=subprocess.Popen([sys.executable,'-c',watchdog_code,str(pid),str(created),str(a.timeout+20)],start_new_session=True)
 os.killpg(pid,signal.SIGSTOP)
try:
 time.sleep(.3)
 with (out/f'q{a.q}.log').open('w') as log:
  child=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  print('probe_pid',child.pid,'production_paused',pid if parent else None,flush=True)
  try:code=child.wait(timeout=a.timeout)
  except subprocess.TimeoutExpired:
   timeout=True;os.killpg(child.pid,signal.SIGTERM)
   try:code=child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
finally:
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGKILL);child.wait()
 if parent:
  try:
   if parent.is_running() and parent.create_time()==created:os.killpg(pid,signal.SIGCONT);resumed=True
  except psutil.NoSuchProcess:pass
 if watchdog:watchdog.terminate();watchdog.wait()
 receipt=dict(command=command,environment={k:v for k,v in env.items() if k.startswith(('BEM_','OMP_','RQ_'))},returncode=code,
    timeout=timeout,started_unix=start,elapsed_s=time.time()-start,production_pid=pid if parent else None,production_resumed=resumed)
 (out/f'q{a.q}.run.json').write_text(json.dumps(receipt,indent=2)+'\n')
 with (B/'production_pauses.jsonl').open('a') as f:f.write(json.dumps(dict(case=str(out),q=a.q,paused_s=time.time()-start,resumed=resumed))+'\n')
print(json.dumps(receipt),flush=True)
