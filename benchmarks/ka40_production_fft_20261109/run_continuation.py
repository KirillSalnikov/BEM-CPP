"""Own GPU locks for the accelerated ka40 continuation and SO(3) average."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root=Path(__file__).resolve().parents[2]
base=Path(__file__).resolve().parent
target=root/'runs/scice_ka40_compressed_20261109'
prepared=json.loads((target/'preparation.json').read_text())
assert prepared['state']=='validated_ready'
source=Path(prepared['source']);runtime=Path(prepared['runtime'])
out=target/'solution'
locks=[]
for suffix in ('','.1','.2'):
    handle=open('/home/kirill/neuro/agent-sandbox/.gpu.lock'+suffix,'a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
for name,expected in json.loads((runtime/'sha256.json').read_text()).items():
    assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==expected,name
manifest=json.loads((source/'manifest.json').read_text())
env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
env.update(manifest['parameters']['BEM_environment'])
(target/'scratch').mkdir(exist_ok=True)
env.update(OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',
           BEM_FMM_P2P_DOUBLE_SINGLE='1',BEM_TMATRIX_COMPRESSED_FFT_CASE=str(target/'assets'),
           BEM_SO3_TMPDIR=str(target/'scratch'))
out.mkdir(exist_ok=True)
if not (out/'operator.near').exists():
    subprocess.run(['cp','--reflink=auto',str(source/'operator.near'),str(out/'operator.near')],check=True)
stopped=json.loads((root/'benchmarks/ka40_symmetric_fft_20261108/production_user_paused.json').read_text())
old_command=stopped['restart_argv']['1149381']
command=[sys.executable,str(runtime/'scripts/build_fmm_tmatrix.py'),*old_command[2:]]
command[command.index('--out')+1]=str(out)
command[command.index('--so3-threads')+1]='16'
command+=['--worker-binary',str(runtime/'bin/muller_tmatrix_fmm_server'),
          '--solver-restart-from',str(source),'--solver-restart-proof',prepared['proof']]
record=dict(state='starting',time=time.time(),launcher_pid=os.getpid(),command=command,
            automatic_old_worker_resume=False,wall_time_limit=None,
            environment={k:v for k,v in env.items() if k.startswith(('BEM_','OMP_','OPENBLAS_','GOMP_'))})
def save():
    temporary=target/'launch.json.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n')
    temporary.replace(target/'launch.json')
save()
with (target/'driver.log').open('a') as log:
    process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(state='running',driver_pid=process.pid);save()
    code=process.wait()
record.update(state='complete' if code==0 else 'failed',exit_code=code,finished=time.time());save()
sys.exit(code)
