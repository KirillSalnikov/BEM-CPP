"""Screen rank schedules against independent full ka80 FMM actions before solver use."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='8',OMP_WAIT_POLICY='PASSIVE')
import numpy as np
import torch
base = Path(__file__).resolve().parent
sys.path.insert(0, str(base/'zero_rank_runtime/scripts'))
from compressed_fft_service import validate_manifest, load_operator

candidate_dir=base.parent/'muller_rotation_20261120'
if (candidate_dir/'enable_gpu_trial').exists():
    import subprocess
    start_ns=time.time_ns()
    screen=subprocess.run([sys.executable,str(candidate_dir/'gpu_screen.py')])
    if screen.returncode:
        print(f'Rotation pilot exited {screen.returncode}; continuing the rank screen',flush=True)
    comparison=candidate_dir/'ka80_pair_comparison.json'
    if screen.returncode==0 and comparison.exists() and comparison.stat().st_mtime_ns>=start_ns:
        finished=json.loads(comparison.read_text())
        if finished['passed'] and finished['speedup']>1.05:
            raise SystemExit(0)

case = base/'fft_ka80'
validate_manifest(case)
op = load_operator(case)
screens = json.loads((base/'spectral_traffic_screen.json').read_text())
data = {}
for label in ['smooth','random']:
    x = np.fromfile(base/'native_ka80'/f'{label}.bin',dtype=np.complex128)
    expected = np.fromfile(base/'native_ka80'/f'current_depth6_{label}.bin',dtype=np.complex128)
    data[label] = torch.as_tensor(x.astype(np.complex64),device='cuda'),expected
rows = []
def measure(cutoff,label):
    for i, screen in screens.items():
        block = op.groups[int(i)]['streamed']
        if cutoff:
            record = next(r for r in screen['rows'] if r['tol']==cutoff)
            ranks = record['batch_ranks']
            assert len(ranks)==(block.nf+block.batch-1)//block.batch
            block.batch_ranks = [max(1,r) for r in ranks]
        else:
            block.batch_ranks = None
    tensor,expected = data[label]
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.inference_mode():
        actual = op(tensor).cpu().numpy()
    elapsed = time.monotonic()-start
    error = np.linalg.norm(actual-expected)/np.linalg.norm(expected)
    return dict(cutoff=cutoff,input=label,seconds=elapsed,relative_native_error=float(error),passed=bool(error<=5e-4))
measure(0,'random')
for cut in [0,1e-5,1e-4,5e-4,1e-3,1e-2,0]:
    for label in data:
        row = measure(cut,label)
        rows.append(row)
        tmp = base/'rank_full_action_screen.tmp'
        tmp.write_text(json.dumps(rows,indent=2))
        tmp.replace(base/'rank_full_action_screen.json')
        print(row,flush=True)
        if cut==0 and not row['passed']:
            raise RuntimeError('baseline failed')
# Release CUDA and all mapped factors before starting the solver under the same device lease.
os.execv(sys.executable,[sys.executable,str(base/'rank_modal_followup.py')])
