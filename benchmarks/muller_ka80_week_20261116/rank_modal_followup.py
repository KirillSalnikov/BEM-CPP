"""Run the fastest passing FFT schedule on the saved reference pair."""
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
import numpy as np

base = Path(__file__).resolve().parent
rows = json.loads((base/'rank_full_action_screen.json').read_text())
baseline = statistics.mean(r['seconds'] for r in rows if r['cutoff']==0)
def eligible(cut):
    subset = [r for r in rows if r['cutoff']==cut]
    return bool(cut and {r['input'] for r in subset}=={'random','smooth'} and all(r['passed'] for r in subset))
times = {cut:statistics.mean(r['seconds'] for r in rows if r['cutoff']==cut)
         for cut in {r['cutoff'] for r in rows} if eligible(cut)}
if not times or min(times.values())>=baseline*.95:
    print('No passing candidate saved 5% of the complete action.',flush=True)
    raise SystemExit(0)
cut = min(times,key=times.get)
ranks = {}
for i,data in json.loads((base/'spectral_traffic_screen.json').read_text()).items():
    row = next(r for r in data['rows'] if r['tol']==cut)
    ranks[int(i)] = [max(1,r) for r in row['batch_ranks']]
runtime = base/'rank_modal_runtime'
shutil.copytree(base/'runtime_v2',runtime,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','sha256.json'))
for name in ('compressed_fft_service.py','symmetric_galerkin_fft.py','bounded_spectral_storage.py'):
    shutil.copy2(base/'zero_rank_runtime/scripts'/name,runtime/'scripts'/name)
p = runtime/'scripts/compressed_fft_service.py'
source=p.read_text()
anchor='    torch.cuda.empty_cache()\n    return result'
assert source.count(anchor)==1
p.write_text(source.replace(anchor,'    for index, ranks in '+repr(ranks)+'.items():\n        result.groups[index]["streamed"].batch_ranks = ranks\n'+anchor))
receipt = json.loads((base/'first_modal_pilot_cpudone.json').read_text())
out = base/'rank_modal_pair'
command = receipt['command'].copy()
command[1]=str(runtime/'scripts/build_fmm_tmatrix.py')
command[command.index('--out')+1]=str(out)
env = {k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
env.update(receipt['environment'])
env['LD_LIBRARY_PATH']=str(runtime/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
subprocess.run(command,env=env,cwd=base.parents[1],check=True)
errors = []
for i in receipt['indices']:
    name=f'{i:06d}.npy'
    x, y = [np.load(d/'columns'/name) for d in (base/'first_modal_pair_cpudone',out)]
    errors.append(float(np.linalg.norm(x-y)/np.linalg.norm(x)))
status=json.loads((out/'columns'/f"{receipt['indices'][0]:06d}.json").read_text())['status']
ref=receipt['columns'][0]['status']
answer=dict(cutoff=cut,strict_residuals=[status['strict_residual_x'],status['strict_residual_y']],relative_column_errors=errors,
    candidate_worker_s=status['total_worker_s'],baseline_worker_s=ref['total_worker_s'],candidate_iterations=[status['host_iterations_x'],status['host_iterations_y']])
answer['passed']=bool(np.all(np.array(errors)<=1e-3) and np.all(np.array(answer['strict_residuals'])<=2e-6))
(base/'rank_modal_comparison.json').write_text(json.dumps(answer,indent=2))
