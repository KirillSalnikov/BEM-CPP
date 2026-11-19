"""First complete ka80/ref7 modal pair after new FFT/native validation.

One independent electric/magnetic pair at l=80, m=0. It establishes actual
solver convergence and combined resource use before a packed-T throughput
pilot. This pair alone cannot certify the runtime/accuracy of the full T.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

B=Path(__file__).resolve().parent
R=B.parents[1]
runtime=B/'runtime_v2'
sys.path.insert(0,str(runtime/'scripts'))
from tmatrix_reference import modes

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-name', default='cached')
parser.add_argument('--first-pass-factor', type=float, default=0)
parser.add_argument('--restart', type=int, default=64, choices=(64, 96, 128))
args=parser.parse_args()
if not args.run_name.replace('_','').isalnum():
    parser.error('run-name must contain only letters, digits, underscores')
if not 0 <= args.first_pass_factor <= 64:
    parser.error('first-pass-factor must be between 0 and 64')
out=B/f'first_modal_pair_{args.run_name}'
receipt=B/('first_modal_pilot.json' if args.run_name=='cached' else f'first_modal_pilot_{args.run_name}.json')
out.mkdir(exist_ok=True)
report=dict(status='waiting_for_new_ka80_factors',scope=__doc__)
def save():
    p=receipt.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2)+'\n');p.replace(receipt)
save()
while True:
    if (B/'STOP').exists():raise RuntimeError('STOP requested')
    state=json.loads((B/'complete_fft_check_run/status.json').read_text())
    if state['state']=='failed':raise RuntimeError('complete FFT action check failed')
    if state['state']=='complete':break
    time.sleep(3)
native=json.loads((B/'current_native.json').read_text())
assert native['status']=='complete'
choice=min(native['records'],key=lambda r:sum(a['seconds'] for a in r['actions']))
assert choice['status']=='complete'
env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
env.update(choice['environment'])
leaf=env['BEM_TEST_MAX_LEAF']
for key in list(env):
    if key.startswith('BEM_TEST_'):del env[key]
env.update(OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',
    BEM_TMATRIX_MAX_LEAF=leaf,BEM_FMM_P2P_DOUBLE_SINGLE='1',BEM_FMM_P2P_DS_COMPENSATED='1',
    BEM_FMM_P2P_DS_FACTORED='1',BEM_FMM_P2P_DS_THREADS='128',
    BEM_TMATRIX_SECTOR_GMRES='1',BEM_TMATRIX_ORBIT_NODES='50',BEM_TMATRIX_SECTOR_RECYCLE_RANK='16',
    BEM_TMATRIX_VERTICAL_PRECONDITIONER='1',BEM_TMATRIX_CIRCULANT_BLEND='.5',
    BEM_FARFIELD_BACKEND='face_nufft',BEM_FARFIELD_CPU_THREADS='8',BEM_FARFIELD_EPS='1e-10',
    BEM_FARFIELD_PARALLEL_FACES='1',BEM_TMATRIX_COMPRESSED_FFT_CASE=str(B/'fft_ka80'),
    BEM_TMATRIX_COMPRESSED_FFT_WIRE_FP32='1',BEM_TMATRIX_CHECKER_COMPARE='0')
env['BEM_SECTOR_FIRST_PASS_FACTOR']=str(args.first_pass_factor)
env['LD_LIBRARY_PATH']=str(runtime/'lib')+(os.pathsep+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
modal=modes(137)
indices=[modal.index((p,80,0)) for p in (0,1)]
command=[sys.executable,str(runtime/'scripts/build_fmm_tmatrix.py'),
    '--shape','prism','--ref','7','--ka','80','--ri','1.3116','--wavelength','.532',
    '--prism-aspect','1.6495721976846451','--order','137','--projection-method','fft',
    '--tol','2e-6','--restart',str(args.restart),'--max-iters','1000','--column-indices',*map(str,indices),
    '--max-pairs-this-run','1','--no-average','--no-incident-cache','--release-incident-buffer',
    '--worker-binary',str(runtime/'bin/muller_tmatrix_fmm_server'),
    '--operator-cache-prefix',str(B/'fft_ka80/native'),'--out',str(out)]
report.update(status='waiting_for_gpu',selected_depth=choice['depth'],indices=indices,command=command,
              environment={k:v for k,v in env.items() if k.startswith(('BEM_','OMP_','OPENBLAS_','GOMP_'))})
save()
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
report['status']='running';save();start=time.monotonic()
try:
    subprocess.run(command,cwd=R,env=env,check=True)
    rows=[]
    for index in indices:
        row=json.loads((out/'columns'/f'{index:06d}.json').read_text())
        if max(row['status']['strict_residual_x'],row['status']['strict_residual_y'])>2e-6:
            raise RuntimeError('modal pair failed strict residual')
        rows.append(row)
    report.update(status='complete',columns=rows,elapsed_s=time.monotonic()-start,
                  operator=json.loads((out/'operator_metadata.json').read_text()))
except BaseException as e:
    report.update(status='failed',error=repr(e));raise
finally:save()
