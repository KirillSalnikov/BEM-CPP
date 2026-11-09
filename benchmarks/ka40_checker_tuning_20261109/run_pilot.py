"""One real ka40/ref6 native-sector solve with the compressed FFT companion."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[2]
base=Path(__file__).resolve().parent
source=root/'runs/scice_ka40_compressed_20261109/solution'
case=root/'runs/scice_ka40_compressed_20261109/assets'
out=base/'pilot'
locks=[]
for suffix in ('','.1','.2'):
    handle=open('/home/kirill/neuro/agent-sandbox/.gpu.lock'+suffix,'a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
if (out/'manifest.json').exists():raise RuntimeError('Preserve the previous pilot before relaunching')
out.mkdir(exist_ok=True)
manifest=json.loads((source/'manifest.json').read_text())
env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
env.update(manifest['parameters']['BEM_environment'])
env.update(OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',
           BEM_FMM_P2P_DOUBLE_SINGLE='1',BEM_TMATRIX_COMPRESSED_FFT_CASE=str(case),
           BEM_FMM_P2P_DS_THREADS='64',BEM_FMM_P2P_DS_LEAF_SPLIT='32',
           BEM_FMM_P2P_DS_FACTORED='1',BEM_TMATRIX_CHECKER_COMPARE='1')
subprocess.run(['cp','--reflink=auto',str(source/'operator.near'),str(out/'operator.near')],check=True)
binary=base/'bin/muller_tmatrix_fmm_server'
command=[sys.executable,str(root/'scripts/build_fmm_tmatrix.py'),
         '--shape','prism','--ref','6','--ka','40','--ri','1.3116','--wavelength','.532',
         '--prism-aspect','1.6495721976846451','--order','65','--projection-method','fft',
         '--tol','2e-6','--restart','64','--max-iters','1000','--column-indices','817','5172',
         '--no-average','--worker-binary',str(binary),'--out',str(out)]
(base/'pilot_launch.json').write_text(json.dumps(dict(command=command,
    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    environment={k:v for k,v in env.items() if k.startswith(('BEM_','OMP_','OPENBLAS_','GOMP_'))}),indent=2)+'\n')
with (base/'pilot_driver.log').open('w') as log:
    result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:raise RuntimeError('Native FFT pilot failed; inspect pilot_driver.log and pilot/worker.log')
import numpy as np
records=[]
for index in (817,5172):
    old=np.load(source/'columns'/f'{index:06d}.npy')
    new=np.load(out/'columns'/f'{index:06d}.npy')
    error=float(np.linalg.norm(new-old)/np.linalg.norm(old))
    records.append(dict(column=index,relative_difference=error))
    assert error<1e-4,records[-1]
report=dict(status='complete',columns=records,
            maximum_column_difference=max(r['relative_difference'] for r in records),
            operator=json.loads((out/'operator_metadata.json').read_text()),
            solution=json.loads((out/'columns/000817.json').read_text())['status'])
(base/'pilot_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
