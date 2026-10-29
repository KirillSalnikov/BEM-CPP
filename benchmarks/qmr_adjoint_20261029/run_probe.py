import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from bounded_bem_run import run
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--ref',type=int,default=3);p.add_argument('--ka',type=float,default=8)
p.add_argument('--methods',default='adjoint');p.add_argument('--banded',action='store_true');p.add_argument('--absorption',type=float,default=0)
p.add_argument('--verify-interval',type=int,default=128);p.add_argument('--limit',type=int,default=500)
p.add_argument('--overlap',type=int,default=0);p.add_argument('--rhs',type=int,default=1)
a=p.parse_args();out=ROOT/'runs/qmr_adjoint_20261029'/a.name;out.mkdir(parents=True,exist_ok=True)
env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
env.update(json.loads((ROOT/'runs/qmr_hotspot_20261029/ref5_gmres32_tfqmr/environment.json').read_text()))
env.update(BEM_SOLVER_METHODS=a.methods,BEM_SOLVER_VERIFY_INTERVAL=str(a.verify_interval),BEM_SOLVER_LIMIT=str(a.limit),
 BEM_SOLVER_RHS_COUNT=str(a.rhs),BEM_PROBE_MBJ_OVERLAP=str(a.overlap),BEM_PROBE_IMAG_INDEX=str(a.absorption),
 BEM_PROBE_FARFIELD_PREFIX=str(out/'farfield'),BEM_FMM_BANDED_SPLIT_DEPTH=str(max(2,a.ref-2) if a.banded else 0),
 BEM_FMM_ORDER_REFERENCE_DEPTH=str(max(1,a.ref)),BEM_FMM_P2P_LOCAL_COORDINATES='0',BEM_FMM_P2P_SHARED_FAST_TRIG='0')
cache=out/'operator'
if a.ref in (5,6) and a.ka==40 and not a.absorption:
 cache=ROOT/f'runs/circulant_20261029/ref{a.ref}_ka40/operator'
(out/'environment.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith('BEM_') or k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},indent=2))
cmd=[str(ROOT/'bin/muller_solver_fft_probe'),str(a.ref),str(a.ka),str(cache),str(out/'results.jsonl')]
s=run(cmd,out/'supervisor',env=env,wall_limit_s=None);print(s,flush=True)
raise SystemExit(0 if s['state']=='complete' else 1)
