import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from tmatrix_c6 import encode_fields,sector
p=argparse.ArgumentParser();p.add_argument('--ref',type=int,default=5);p.add_argument('--ka',type=float,default=40);p.add_argument('--degree',type=int,default=36)
a=p.parse_args();out=ROOT/f'runs/symmetry_solver_20261029/packet_ref{a.ref}_ka{a.ka:g}_l{a.degree}';out.mkdir(parents=True,exist_ok=True);prefix=out/'packet'
cmd=[str(ROOT/'bin/prism_packet_rhs'),str(a.ref),str(10/7),str(prefix)]
start=time.monotonic();subprocess.run(cmd+['positions'],check=True)
positions=np.fromfile(str(prefix)+'.positions.bin',dtype=np.float64).reshape(-1,3)
modes=[(p,a.degree,m) for m in range(6) for p in (0,1)]
fields=np.memmap(str(prefix)+'.fields.bin',mode='w+',dtype=np.complex128,shape=(2,len(positions),6))
scales,combined=encode_fields(positions,modes,list(range(12)),a.ka,fields,reflection=True)
fields.flush();del fields
subprocess.run(cmd+['assemble'],check=True)
(out/'packet.json').write_text(json.dumps({'ref':a.ref,'ka':a.ka,'degree':a.degree,'modes':modes,'sectors':[sector(m,True)for m in modes],'scales':np.asarray(scales).tolist(),'combined_scales':np.asarray(combined).tolist(),'rhs_points':len(positions),'preparation_s':time.monotonic()-start},indent=2)+'\n')
print(prefix,flush=True)
