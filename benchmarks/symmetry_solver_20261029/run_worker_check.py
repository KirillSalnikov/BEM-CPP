import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from bounded_bem_run import run
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--ref',type=int,default=6);p.add_argument('--degree',type=int,default=36);p.add_argument('--old',action='store_true');p.add_argument('--reference',type=Path)
a=p.parse_args();ka=40 if a.ref in (5,6)else 8
out=ROOT/'runs/symmetry_solver_20261029'/a.name
packet=ROOT/f'runs/symmetry_solver_20261029/packet_ref{a.ref}_ka{ka}_l{a.degree}/packet'
env={k:v for k,v in os.environ.items()if not k.startswith('BEM_')}
env.update(json.loads((ROOT/'runs/symmetry_solver_20261029/ref6_packet_l36_fast_symcirc/environment.json').read_text()))
env.update(BEM_FMM_BANDED_SPLIT_DEPTH=str(max(2,a.ref-2)),BEM_FMM_ORDER_REFERENCE_DEPTH=str(a.ref))
binary=ROOT/'runs/ka40_n1p3116_l0p532_aspect10over7_20261029/runtime/bin/muller_tmatrix_fmm_server'if a.old else ROOT/'bin/muller_tmatrix_fmm_server'
cmd=[sys.executable,str(ROOT/'benchmarks/symmetry_solver_20261029/worker_packet_check.py'),'--out',str(out),'--packet',str(packet),'--binary',str(binary)]
if not a.old:cmd+=['--sector']
if a.reference:cmd+=['--reference',str(a.reference.resolve())]
s=run(cmd,out/'supervisor',env=env,wall_limit_s=None);print(s,flush=True)
raise SystemExit(0 if s['state']=='complete'else 1)
