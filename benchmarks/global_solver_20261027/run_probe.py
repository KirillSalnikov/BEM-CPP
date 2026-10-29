#!/usr/bin/env python3
"""Run bounded global-preconditioner probes, never the production T build."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from bounded_bem_run import run
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('name');p.add_argument('--small',action='store_true')
p.add_argument('--rank',type=int,default=16);p.add_argument('--train',type=int,default=64)
p.add_argument('--steps',type=int,default=32)
a=p.parse_args()
profile=next(ast.literal_eval(n.value) for n in ast.parse((ROOT/'scripts/run_large_tmatrix.py').read_text()).body
             if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROFILE' for t in n.targets))
env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')};env.update(profile)
for suffix in ['M2L_FP32','P2M_FP32','L2P_FP32','INTERPOLATION_FP32','SHIFT_CACHE','M2L_CACHE','FLAT_NEAR']:
    env['BEM_FMM_MULTILEVEL_'+suffix]='1'
env.update(BEM_FMM_MULTILEVEL_CHUNK='256',BEM_PROBE_MAX_LEAF='29',BEM_PROBE_MBJ_NODES='12',
           BEM_MIXED_ADAPTIVE_INNER='1',BEM_GLOBAL_RANK=str(a.rank),BEM_GLOBAL_TRAIN_STEPS=str(a.train),
           BEM_GLOBAL_TEST_STEPS=str(a.steps),BEM_PROBE_TOL='1e-5')
out=ROOT/'runs/global_solver_20261027';out.mkdir(exist_ok=True)
ref,ka=7,112
cache=ROOT/'runs/ka112_d6h_L149_20261027_v2/main_ref7_L149/operator'
if a.small:
    ref,ka=2,3;cache=out/'small_operator';env['BEM_FMM_BANDED_SPLIT_DEPTH']='0';env['OMP_NUM_THREADS']='4'
else:
    free=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<22500:raise RuntimeError('Need 22500 MiB free before the full-size diagnostic')
binary=ROOT/'bin/muller_recycle_fmm_probe'
command=[str(binary),str(ref),str(ka),str(cache),str(out/a.name)]
identity={'command':command,'env':{k:v for k,v in env.items() if k.startswith('BEM_')},
          'sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
env['BEM_GLOBAL_SIGNATURE']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
result=run(command,out/a.name,env=env,rss_limit_gib=46,available_gib=6,disk_reserve_gib=40,
           wall_limit_s=180 if a.small else 2700)
print(json.dumps(result,indent=2));raise SystemExit(0 if result['state']=='complete' else 1)
