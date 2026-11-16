"""Bounded two-mode experiment, strict residual unchanged, memory measured fresh."""
import os,sys,argparse,json,time,resource
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parents[1]
sys.path[:0]=[str(R/'benchmarks/clifford_dielectric_20261115'),str(R/'scripts')]
import numpy as np
from sector_fmm import SectorFMM
from sector_coarse_sparse import SparseCoarse
from sector_coarse_fmm import MatrixFreeCoarse
from dirac_symmetry import solve_modal_batch
p=argparse.ArgumentParser();p.add_argument('--ka',type=int,default=10);p.add_argument('--ref',type=int,default=2)
p.add_argument('--radius',type=float,default=1.5);p.add_argument('--drop',type=float,default=1e-3);p.add_argument('--fill',type=float,default=4)
p.add_argument('--exact',action='store_true');p.add_argument('--max-solve-seconds',type=float,default=180)
p.add_argument('--groups',type=int,default=0)
p.add_argument('--aggregate-basis',type=int,default=1)
p.add_argument('--outer',action='store_true')
p.add_argument('--budget-gib',type=float,default=8)
p.add_argument('--out',type=Path,required=True);a=p.parse_args();begin=time.perf_counter()
d=SectorFMM(ref=a.ref,ka=a.ka,aspect=1.6495721976846451,degree=3,near_subdiv=1,eps=1e-8,strict_eps=1e-12,fmm_packet=1)
g=d._compact_geometry;setup=time.perf_counter()-begin
print(json.dumps(dict(stage='operator_ready',seconds=setup)),flush=True)
budget=int(a.budget_gib*(1<<30))
if a.exact:c=MatrixFreeCoarse(g,3,budget)
elif a.groups:
    from sector_coarse_hierarchy import HierarchicalCoarse
    c=HierarchicalCoarse(g,3,budget,radius=a.radius,drop_tol=a.drop,fill_factor=a.fill,aggregate_groups=a.groups,aggregate_basis=a.aggregate_basis,outer=a.outer)
else:c=SparseCoarse(g,3,budget,radius=a.radius,drop_tol=a.drop,fill_factor=a.fill)
modal=[(0,2,-1),(1,1,-1)];incidents=[d.modal_incident(m) for m in modal]
t=time.perf_counter();c.prepare_many(modal);prepare=time.perf_counter()-t
print(json.dumps(dict(stage='coarse_ready',seconds=prepare,bytes=c.bytes)),flush=True)
real_apply=c.apply_batch;solve_begin=time.perf_counter()
def limited(x,modes):
    if time.perf_counter()-solve_begin>a.max_solve_seconds:raise TimeoutError('bounded screening solve limit reached')
    return real_apply(x,modes)
c.apply_batch=limited
result=dict(ka=a.ka,ref=a.ref,radius=a.radius,drop=a.drop,fill=a.fill,exact=a.exact,aggregate_groups=a.groups,aggregate_basis=a.aggregate_basis,outer=a.outer,
    operator_setup_s=setup,coarse_setup_s=prepare,coarse_bytes=c.bytes,coarse_memory_plan=c.memory_plan(),coarse_statistics=c.statistics)
try:
    h,stats=solve_modal_batch(d,g,incidents,modal,preconditioner=c)
    result.update(status='converged',iterations=[s['iterations'] for s in stats],residuals=[s['residual'] for s in stats])
    np.savez(a.out.with_suffix('.npz'),h=h)
except (RuntimeError,TimeoutError) as error:result.update(status='failed',error=str(error)[-1200:])
result.update(solve_s=time.perf_counter()-solve_begin,wall_s=time.perf_counter()-begin,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    preconditioner_action_s=c.action_s,applications=c.applications,fmm_s=d.fmm_s,near_s=d.near_s)
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
c.close();d.close()
