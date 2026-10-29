import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];runs=ROOT/'runs/symmetry_solver_20261029';out=Path(__file__).parent
records=[]
reference=runs/'ref5_packet_l36'
for run in sorted(runs.iterdir()):
 file=run/'results.jsonl'
 if not file.exists():continue
 data=[json.loads(line)for line in file.read_text().splitlines()];meta=data[0]
 record={'run':run.name,'configuration':meta,'results':data[1:]}
 identity=run/'supervisor/command.json'
 if identity.exists():record['binary_sha256']=json.loads(identity.read_text())['binary_sha256']
 methods={}
 for item in record['results']:
  if 'converged' not in item:continue
  method=item['method'];methods.setdefault(method,[]).append(item)
  ref=(run if (run/'farfield.circ0.25.rhs0.bin').exists() else reference)/f'farfield.circ0.25.rhs{item["rhs"]}.bin'
  field=run/f'farfield.{method}.rhs{item["rhs"]}.bin'
  if ref.exists() and field.exists():
   a=np.fromfile(ref,dtype=np.complex128);b=np.fromfile(field,dtype=np.complex128)
   item['farfield_relative_l2']=float(np.linalg.norm(a-b)/np.linalg.norm(a))
 if meta.get('rhs_kind')=='C6_mirror_packet':
  record['complete_packets']={method:{'seconds':sum(x['seconds']for x in pair),'fmm_actions':sum(x['fmm_actions']for x in pair),
    'strict_actions':sum(x['strict_actions']for x in pair),'max_strict_residual':max(x['strict_residual']for x in pair),
    'max_farfield_relative_l2':max((x.get('farfield_relative_l2',0)for x in pair))}
    for method,pair in methods.items() if len(pair)==2 and all(x['converged']for x in pair)}
 records.append(record)
summary={'scope':'C6 x horizontal reflection; independent sector Krylov with original full FP64 residual gate','records':records,
 'production_changes_enabled':(out/'production_launched.json').exists(),'source_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in
 ['src/prism_sector_solver.h','tools/muller_tmatrix_fmm_server.cpp','src/prism_symmetric_circulant.h','src/prism_symmetry_sectors.h','src/prism_orbit_inverse.h','src/prism_orbit_inverse.cpp','src/streaming_sector_gmres.h','tools/muller_solver_fft_probe.cpp']}}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for record in records:print(record['run'],json.dumps(record.get('complete_packets',{})))
