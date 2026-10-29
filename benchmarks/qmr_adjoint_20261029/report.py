import json,csv,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent;runs=ROOT/'runs/qmr_adjoint_20261029'
rows=[]
for p in sorted(runs.iterdir()):
 if not (p/'results.jsonl').exists():continue
 data=[json.loads(line)for line in (p/'results.jsonl').read_text().splitlines()];meta=data[0]
 identity=json.loads((p/'supervisor/command.json').read_text())
 for result in data[1:]:
  row={'run':p.name,'ref':meta['ref'],'ka':meta['ka'],'system_dofs':meta['system_dofs'],'binary_sha256':identity['binary_sha256'],**result}
  if 'converged' in result and result['converged']:
   baseline=p/('farfield.circ0.25.rhs'+str(result['rhs'])+'.bin')
   trial=p/('farfield.'+result['method']+'.rhs'+str(result['rhs'])+'.bin')
   if baseline.exists() and trial.exists():
    a=np.fromfile(baseline,dtype=np.complex128);b=np.fromfile(trial,dtype=np.complex128)
    row['farfield_relative_l2']=float(np.linalg.norm(a-b)/np.linalg.norm(a))
  rows.append(row)
summary={'scope':'Transpose of discrete FMM/Galerkin operator; QMR is without look-ahead; FP64 acceptance retained',
 'records':rows,'api':['matvec_transpose','matvec_adjoint','matvec_transpose_strict','matvec_adjoint_strict'],
 'production_solver_changed':False,'source_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
 ['src/fmm_transpose.cuh','src/muller_fmm_transpose.inc','src/streaming_qmr.h','src/muller_mbj.cpp','src/block_circulant_fft.cu','src/prism_circulant_inverse.cpp']}}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
keys=['run','ref','ka','system_dofs','method','fp32','transpose_error','adjoint_error','dense_transpose_error','forward_s','transpose_s','adjoint_s','iterations','seconds','strict_residual','converged','farfield_relative_l2','forward_actions','adjoint_actions']
with (out/'results.csv').open('w')as f:
 w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
