import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import fcntl,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'scripts'))
from compressed_fft_service import Companion
locks=[]
for suffix in ('','.1','.2'):
    handle=open('/home/kirill/neuro/agent-sandbox/.gpu.lock'+suffix,'a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
with (base/'expanded_accuracy.log').open('w') as log:
    subprocess.run([str(root/'tests/p2p_double_single_check')],stdout=log,stderr=subprocess.STDOUT,check=True)
case=root/'runs/scice_ka40_compressed_20261109/assets'
out=base/'wire_probe';out.mkdir(exist_ok=True)
currents=np.load(root/'benchmarks/ka40_symmetric_fft_20261108/ka40_ref6/ds_currents.npy')
assert currents.shape==(6,1142784)
currents.tofile(out/'currents.bin')
with tempfile.TemporaryDirectory(prefix='bem-wire-') as temporary:
    socket=Path(temporary)/'fft.sock'
    with Companion(case,socket,out):
        with (out/'comparison.jsonl').open('w') as log:
            subprocess.run([str(base/'wire_probe_bin'),str(socket),str(out/'currents.bin')],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
rows=[json.loads(line) for line in (out/'comparison.jsonl').read_text().splitlines()]
baseline=[r['seconds'] for r in rows if r['trial']==3]
candidate=[r['seconds'] for r in rows if r['trial']==2]
report=dict(status='complete',baseline_median_s=float(np.median(baseline)),
    candidate_median_s=float(np.median(candidate)),speedup=float(np.median(baseline)/np.median(candidate)),
    maximum_action_relative_difference=max(r['relative_difference'] for r in rows),
    vectors=6,calls=len(rows),rows=rows)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
