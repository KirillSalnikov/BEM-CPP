"""Fixed-incidence Dirac solve with the same lab frame as the Muller driver."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/clifford_dielectric_20261115'))
from solve import Dirac
from scattering_frame import frame,mueller,save_mueller


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape',choices=('sphere','prism'),required=True)
    p.add_argument('--ka',type=float,required=True);p.add_argument('--index',type=complex,required=True)
    p.add_argument('--aspect',type=float,required=True);p.add_argument('--ref',type=int,required=True)
    p.add_argument('--degree',type=int,required=True);p.add_argument('--order',type=int,default=24)
    p.add_argument('--target-order',type=int,default=13);p.add_argument('--near-subdiv',type=int,default=1)
    p.add_argument('--tol',type=float,default=1e-8);p.add_argument('--backend',choices=('dense','fmm-sector'),required=True)
    p.add_argument('--ram-budget-gib',type=float,required=True);p.add_argument('--kernel-budget-gib',type=float,required=True)
    p.add_argument('--symmetry',choices=('none','c6','c6v','d6h'),default='none')
    p.add_argument('--orientation',type=float,nargs=3,default=(0,0,0));p.add_argument('--ntheta',type=int,default=181)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    # Full-space plane waves are not single C6 modal sectors. Reserve full
    # Krylov storage even when the operator itself reuses symmetry rows.
    options=vars(a).copy();options.update(L=1,block_panels=1,sector_coarse_basis=0,coarse_cache_policy='sector',
        coarse_budget_gib=a.ram_budget_gib,so3='none',so3_threads=1,batch_size=2,fmm_packet=1,
        dense_cache='orbits12' if a.symmetry=='d6h' else 'full',symmetry='c6hv' if a.symmetry=='d6h' else a.symmetry)
    from memory_plan import dense_memory_plan,sector_fmm_memory_plan
    plan=(dense_memory_plan if a.backend=='dense' else sector_fmm_memory_plan)(SimpleNamespace(**options))
    full,sector=plan['full_dofs'],plan['sector_dofs']
    extra=16*66*2*(full-sector)
    plan['full_incidence_extra_workspace_bytes']=extra
    if plan['estimated_peak_bytes']+extra>plan['effective_budget_bytes']:
        raise MemoryError('Full-incidence Krylov space exceeds RAM budget before operator allocation')
    (a.out/'memory_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    begin=time.perf_counter()
    kw=dict(shape=a.shape,ref=a.ref,ka=a.ka,index=a.index,aspect=a.aspect,degree=a.degree,
            order=a.order,target_order=a.target_order,near_subdiv=a.near_subdiv,max_bytes=int(a.kernel_budget_gib*2**30))
    if a.backend=='dense':d=Dirac(**kw,dense_kernel='simd',block_panels=1,dense_cache=options['dense_cache'])
    else:
        from sector_fmm import SectorFMM
        d=SectorFMM(**kw,eps=min(1e-8,a.tol),strict_eps=min(1e-12,a.tol/100),horizontal=a.symmetry=='d6h')
    try:
        theta=np.linspace(0,180,a.ntheta);basis,obs,et,ep=frame(a.orientation,theta)
        inc=[d.incident(basis[2],polar) for polar in basis[:2]]
        values,stats=d.solve_incidents(inc,tol=a.tol)
        if any(s['info'] or not np.isfinite(s['residual']) or s['residual']>a.tol*1.01 for s in stats):
            raise RuntimeError('Fixed-incidence solve failed strict residual check')
        far=np.array([d.field(h,obs,far=True,incident=f)[:,5:8] for h,f in zip(values,inc)])
        save_mueller(a.out,theta,mueller(far,et,ep,a.ka))
        report=dict(complete=True,method='dirac',task='fixed',ka=a.ka,index=str(a.index),
                    euler_degrees=a.orientation,statistics=stats,elapsed_s=time.perf_counter()-begin,
                    dofs=d.size,memory_plan=plan,maximum_strict_residual=max(s['residual'] for s in stats),
                    Mueller_units='dimensionless; multiply by (wavelength/(2*pi))^2 for differential cross section')
        (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:d.close()


if __name__=='__main__':main()
