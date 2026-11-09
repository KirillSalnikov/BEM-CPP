"""One real T-matrix packet on ref7, after the independent memory/action audit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import fcntl,hashlib,json,subprocess,sys,time,threading
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1];CASE=BASE/'variants/recompressed_packed_0';OUT=BASE/'tmatrix_pilot_compressed'
RUNTIME=BASE/'ref7_runtime'
sys.path.insert(0,str(RUNTIME/'scripts'))
while True:
    path=BASE.parent/'ref7_memory_20261109/combined_ref7_report.json'
    if path.exists():
        audit=json.loads(path.read_text())
        if audit['status']=='failed':raise RuntimeError('Coexistence audit failed; pilot is blocked')
        if audit['status']=='complete':break
    time.sleep(3)
if max(r['relative_error'] for r in audit['fft_actions'])>5e-4:raise RuntimeError('FFT does not meet the native startup tolerance')
checked=json.loads((BASE/'actions.json').read_text())
assert checked['status']=='complete'
assert all(r['passes_original_action_gate'] for r in checked['records'])
assert (CASE/'service_assets.json').exists()
from run_large_tmatrix import frozen_runtime
frozen_runtime(RUNTIME,worker_binary=BASE.parent/'ref7_memory_20261109/bin_lazy/muller_tmatrix_fmm_server')
queue_path=BASE/'tmatrix_pilot_compressed_report.json'
queue=dict(status='waiting_for_gpu_locks',pid=os.getpid(),case=str(CASE),columns=[817,5172],
           order=65,strict_tolerance=2e-6,full_T_build=False)
queue_path.write_text(json.dumps(queue,indent=2)+'\n')
print('Waiting for all GPU slots; two-column ref7 pilot only.',flush=True)
locks=[]
for suffix in ('','.1','.2'):
    handle=open('/home/kirill/neuro/agent-sandbox/.gpu.lock'+suffix,'a');fcntl.flock(handle,fcntl.LOCK_EX);locks.append(handle)
if (OUT/'manifest.json').exists():raise RuntimeError('Preserve the previous pilot before restarting')
OUT.mkdir(exist_ok=True)
from compressed_fft_service import prepare_manifest
# The variant manifest exists; the native companion rechecks all hashes.
source=ROOT/'runs/scice_ka40_checker_tuned_20261109/solution'
manifest=json.loads((source/'manifest.json').read_text())
environment={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
environment.update(manifest['parameters']['BEM_environment'])
environment.update({k:v for k,v in audit['native_environment'].items() if k.startswith(('BEM_FMM_','BEM_MULLER_'))})
environment.update(OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',
    BEM_TMATRIX_MAX_LEAF=audit['native_environment']['BEM_TEST_MAX_LEAF'],BEM_FMM_P2P_DOUBLE_SINGLE='1',
    BEM_TMATRIX_COMPRESSED_FFT_CASE=str(CASE),BEM_TMATRIX_COMPRESSED_FFT_WIRE_FP32='1',BEM_TMATRIX_CHECKER_COMPARE='0')
subprocess.run(['cp','--reflink=auto',str(CASE/'native.near'),str(OUT/'operator.near')],check=True)
binary=RUNTIME/'bin/muller_tmatrix_fmm_server'
command=[sys.executable,str(RUNTIME/'scripts/build_fmm_tmatrix.py'),
    '--shape','prism','--ref','7','--ka','40','--ri','1.3116','--wavelength','.532',
    '--prism-aspect','1.6495721976846451','--order','65','--projection-method','fft',
    '--tol','2e-6','--restart','64','--max-iters','1000','--column-indices','817','5172',
    '--no-average','--no-incident-cache','--release-incident-buffer','--worker-binary',str(binary),'--out',str(OUT)]
report=dict(status='starting',command=command,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    environment={k:v for k,v in environment.items() if k.startswith(('BEM_','OMP_','OPENBLAS_','GOMP_'))},samples=[])
def save():
    temp=BASE/'tmatrix_pilot_compressed_report.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(BASE/'tmatrix_pilot_compressed_report.json')
save();started=time.perf_counter();stop=threading.Event()
def monitor():
    while not stop.is_set():
        try:
            gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            available=int(next(v for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')).split()[1])
            report['samples'].append(dict(seconds=time.perf_counter()-started,gpu_MiB=gpu,available_KiB=available))
        except (OSError,ValueError,subprocess.CalledProcessError):pass
        stop.wait(3)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:
    with (BASE/'tmatrix_pilot_compressed_driver.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('T pilot failed; see driver and worker logs')
    rows=[]
    for index in (817,5172):
        status=json.loads((OUT/'columns'/f'{index:06d}.json').read_text())['status']
        residual=max(status['strict_residual_x'],status['strict_residual_y'])
        if residual>2e-6:raise RuntimeError(f'Unconverged T column {index}: {residual}')
        new=np.load(OUT/'columns'/f'{index:06d}.npy');old=np.load(source/'columns'/f'{index:06d}.npy')
        rows.append(dict(index=index,status=status,relative_ref6_difference=float(np.linalg.norm(new-old)/np.linalg.norm(old))))
    report.update(status='complete',columns=rows,wall_s=time.perf_counter()-started,
        operator=json.loads((OUT/'operator_metadata.json').read_text()),fft=json.loads((OUT/'external_fft.json').read_text()))
except BaseException as error:
    report.update(status='failed',error=repr(error));raise
finally:
    stop.set();thread.join();report['GPU_peak_MiB']=max((r['gpu_MiB'] for r in report['samples']),default=0)
    report['RAM_min_available_KiB']=min((r['available_KiB'] for r in report['samples']),default=0);save()
