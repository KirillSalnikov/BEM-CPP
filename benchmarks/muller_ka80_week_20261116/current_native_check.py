"""Measure the current complete Muller action against saved ka80 controls."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

B=Path(__file__).resolve().parent
R=B.parents[1]
case=B/'native_ka80'
binary=B/'source/bin/native'
started=time.monotonic()
report=dict(status='waiting',records=[])
def save():
    p=B/'current_native.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(B/'current_native.json')
save()
while True:
    if (B/'STOP').exists():raise RuntimeError('STOP requested')
    build=json.loads((B/'worker_build.json').read_text())
    selection=json.loads((B/'depth_control_selection.json').read_text())
    lazy=json.loads((B/'lazy_check_run/status.json').read_text())
    if build['status']=='failed' or selection['status']=='failed' or lazy['state']=='failed':
        raise RuntimeError('required earlier build/check failed; inspect reports')
    if build['status']=='complete' and selection['status']=='complete' and lazy['state']=='complete':break
    time.sleep(3)

locks=[]
while True:
    if (B/'STOP').exists():raise RuntimeError('STOP requested')
    try:
        for suffix in ('','.1','.2'):
            f=open('/home/kirill/neuro/agent-sandbox/.gpu.lock'+suffix,'a');locks.append(f)
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if not busy:break
    except BlockingIOError:pass
    for f in locks:f.close()
    locks=[];time.sleep(3)

old=json.loads((B/'depth_screen.json').read_text())
assert old['ka']==80 and old['system_dofs']==4552704
assert hashlib.sha256((B/'depth_screen.json').read_bytes()).hexdigest()==selection['source_sha256']
report.update(status='running',binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
              scope='Complete strict Muller actions, current consistent build vs legacy ka80 controls; not full T timing')
save()
base={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
for item in old['records']:
    if item['depth'] not in selection['accepted_depths'] or item['status']!='complete':continue
    depth=item['depth']
    # A rejected deeper tree is not retried just to obtain a faster timing.
    if depth==7 and max(a['relative_depth_difference'] for a in item['actions'])>=1e-8:continue
    env=dict(base,**item['environment'])
    env.update(BEM_FMM_P2P_DS_COMPENSATED='1',BEM_FMM_P2P_DS_FACTORED='1',
               BEM_FMM_P2P_DS_THREADS='128',BEM_FMM_P2P_DS_LEAF_SPLIT='32',BEM_FMM_FUSE_MEDIA='0')
    # Export the newly computed native correction once for the ka80 FFT case.
    # The C++ writer streams chunks; no second operator initialization is needed.
    if depth==6:env.pop('BEM_TEST_SKIP_EXPORT',None)
    row=dict(depth=depth,status='preparing',actions=[],environment={k:v for k,v in env.items() if k.startswith(('BEM_','OMP_','GOMP_','OPENBLAS_'))})
    report['records'].append(row);save()
    with (B/f'current_depth{depth}.log').open('w') as log:
        p=subprocess.Popen([str(binary),'7','80','0',str(case),'serve'],env=env,
                           stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        t=time.monotonic()
        try:
            reply=p.stdout.readline().strip()
            if reply!='READY 4552704':raise RuntimeError('native setup: '+reply)
            row.update(status='actions',setup_s=time.monotonic()-t);save()
            for i,label in enumerate(('random','smooth')):
                source=case/(label+'.bin');target=case/f'current_depth{depth}_{label}.bin'
                p.stdin.write(f'AD {i} 1 {source} {target}\n');p.stdin.flush()
                reply=p.stdout.readline().strip()
                if not reply.startswith(f'DONE {i} '):raise RuntimeError('native action: '+reply)
                a=np.fromfile(target,np.complex128);b=np.fromfile(case/f'depth6_{label}.bin',np.complex128)
                error=float(np.linalg.norm(a-b)/np.linalg.norm(b));del a,b
                result=dict(input=label,seconds=float(reply.split()[2]),relative_legacy_depth6_error=error)
                row['actions'].append(result);save();print(json.dumps(result),flush=True)
                if not np.isfinite(error) or error>1e-8:raise RuntimeError('current operator differs from legacy control')
            p.stdin.write('QUIT\n');p.stdin.flush();p.wait(timeout=30)
            if p.returncode:raise RuntimeError('native nonzero exit')
            row['status']='complete';save()
        except BaseException as e:
            row.update(status='failed',error=repr(e));report['status']='failed';save();raise
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait()
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
report.update(status='complete',wait_included_wall_s=time.monotonic()-started,
              near_export={name:dict(bytes=(case/name).stat().st_size,sha256=digest(case/name))
                           for name in ('native.near','near_rows.bin','near_cols.bin','near_values.bin')})
save()
