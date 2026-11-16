"""Streaming sector order: mirror checkpoints, positive holes, policy change."""
import os,sys,subprocess,json,hashlib
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;R=B.parents[1];D=B/os.environ.get('DIRAC_RESUME_CASE','resume')
env=dict(os.environ,OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',
    BEM_DIRAC_LIBRARY=str(B/'libdirac_hierarchy.so'))
base=[sys.executable,str(R/'benchmarks/clifford_dielectric_20261115/tmatrix.py'),
    '--shape','prism','--ref','0','--ka','1','--degree','1','--L','3','--order','14',
    '--target-order','13','--near-subdiv','1','--backend','fmm-sector','--fmm-packet','1','--block-panels','1',
    '--batch-size','8','--sector-coarse-basis','3','--coarse-factorization','hierarchical','--coarse-aggregate-groups','64','--coarse-aggregate-basis','3','--coarse-global-scope','full','--symmetry','c6hv',
    '--so3','mstm','--out',str(D)]
def run(name,args,fail=False):
    r=subprocess.run(base+args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,text=True)
    (B/(name+'.log')).write_text(r.stdout)
    assert bool(r.returncode)==fail,r.stdout[-3000:]
    return r.stdout
def digest():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (D/'columns').glob('*.npz')}
run('resume_first',['--max-columns','9'])
first=digest();assert 0<len(first)<=9
run('resume_complete',['--resume','--coarse-budget-gib','1','--coarse-cache-policy','all'])
complete=digest();assert all(complete[k]==v for k,v in first.items())
reference=np.load(D/'solution.npz')['T'].copy()
mirrored=[]
for p in (D/'columns').glob('*.npz'):
    with np.load(p) as z:
        s=json.loads(str(z['statistics']))
        if 'mirror_from' in s:mirrored.append(p)
assert mirrored
hole=sorted(mirrored)[0];hole.unlink();before=digest()
run('resume_hole',['--resume'])
after=digest();assert all(after[k]==v for k,v in before.items())
with np.load(hole) as z:assert 'mirror_from' not in json.loads(str(z['statistics']))
error=float(np.linalg.norm(np.load(D/'solution.npz')['T']-reference)/np.linalg.norm(reference))
assert error<1e-7
for arg,value in [('--aspect','1.6495721976846451'),('--coarse-aggregate-groups','32'),('--fmm-packet','2'),('--fmm-strict-eps','1e-13'),('--coarse-global-scope','coarse')]:
    assert 'checkpoint numerical parameters or code changed' in run('reject_'+arg[2:],['--resume',arg,value],True)
    assert digest()==after
# An impossible global RAM budget is rejected with no dense matrix allocation.
output=run('reject_ram',['--resume','--ram-budget-gib','.001'],True)
assert 'no near cache allocated' in output
result=dict(first_columns=len(first),preserved=True,cache_policy_and_budget_change_allowed=True,
    positive_hole_recomputed=True,T_hole_error=error,changed_aspect_rejected=True,
    changed_coarse_rejected=True,RAM_rejected_before_near_allocation=True)
(B/'resume_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
