#!/usr/bin/env python3
"""Separate current-basis error from operator Taylor error using exact A(k).

The reduced current/test bases are frozen in a prior central-only experiment.
Full small solves here are independent validation and timing references only.
No production or GPU T-matrix build is launched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.linalg import lu_factor,lu_solve,solve
from bounded_bem_run import atomic_json
from tmatrix_reference import modes,transverse_harmonics
from tmatrix_so3_mstm import analytic_mueller_tmatrix
from study_tmatrix_cutoff import error_metrics,cutoff_indices

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--offsets',type=float,nargs='+',default=[-1.,-.5,.5,1.])
    p.add_argument('--ranks',type=int,nargs='+')
    args=p.parse_args();modeldir=args.model.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if not (modeldir/'frozen.json').exists():raise RuntimeError('central model is not frozen')
    definition=json.loads((modeldir/'plan.json').read_text());plan=definition['parameters'];source=Path(definition['source'])
    ka0=plan['ka0'];order=plan['order'];basis=np.load(modeldir/'basis.npy');max_rank=basis.shape[1]
    with np.load(modeldir/'central_inverse.npz') as data:test=data['test']
    ranks=args.ranks or [max_rank]
    if max(ranks)>max_rank:raise ValueError('rank exceeds frozen basis')
    binary=ROOT/'bin/muller_tmatrix_taylor_probe'
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=plan['binary_sha256']:raise RuntimeError('assembler changed')
    atomic_json(out/'plan.json',dict(model=str(modeldir),offsets=args.offsets,ranks=ranks,
        parameters=plan,frozen_model_sha256=hashlib.sha256((modeldir/'central_inverse.npz').read_bytes()).hexdigest(),
        training_ka=[ka0],off_center_currents_used_for_training=False,
        operator='exact at each control ka; no Taylor truncation',new_solves='small dense CPU reference only'))
    band=2*order+4;z,w=np.polynomial.legendre.leggauss(band)
    zz,pp=np.meshgrid(z,np.arange(2*band)*np.pi/band,indexing='ij')
    directions=np.stack((np.sqrt(1-zz*zz)*np.cos(pp),np.sqrt(1-zz*zz)*np.sin(pp),zz),axis=-1).reshape(-1,3)
    weights=np.repeat(w,2*band)*np.pi/band
    wv=np.repeat(weights,2)[:,None]*transverse_harmonics(directions,order)
    phase=np.array([1j**(l-pol) for pol,l,m in modes(order)])
    theta=np.linspace(0,180,1801);records=[]
    for offset in args.offsets:
        ka=ka0+offset;folder=out/f'ka{ka:.8f}';folder.mkdir(exist_ok=True);prefix=folder/'operator'
        command=[str(binary),str(plan['ref']),str(ka),str(plan['ri']),str(plan['ri_imag']),str(plan['prism_aspect']),
            str(source/'grid.bin'),str(prefix),str(plan['quad']),str(plan['duffy']),'0','0']
        start=time.perf_counter()
        with (folder/'worker.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        assembly_wall=time.perf_counter()-start
        meta=json.loads(prefix.with_suffix('.json').read_text());n=meta['dofs'];channels=2*meta['points']
        a=np.memmap(str(prefix)+'.A.bin',dtype=np.complex128,mode='r',shape=(n,n))
        braw=np.memmap(str(prefix)+'.B.bin',dtype=np.complex128,mode='r',shape=(n,channels))
        draw=np.memmap(str(prefix)+'.D.bin',dtype=np.complex128,mode='r',shape=(channels,n))
        start=time.perf_counter();b=braw@wv;d=wv.conj().T@draw;modal_s=time.perf_counter()-start
        start=time.perf_counter();aq=a@basis;ar=test.conj().T@aq;br=test.conj().T@b;dr=d@basis;projection_s=time.perf_counter()-start
        np.savez(folder/'reduced_exact.npz',A=ar,B=br,D=dr,ka=ka,phase=phase)
        start=time.perf_counter();factor=lu_factor(a);direct_x=lu_solve(factor,b);direct_solve_s=time.perf_counter()-start
        direct_residual=float(np.linalg.norm(a@direct_x-b)/np.linalg.norm(b))
        if direct_residual>1e-10:raise RuntimeError('direct residual failed')
        start=time.perf_counter();direct=-1j*ka/(4*np.pi)*phase[:,None]*(d@direct_x)*phase[None,:].conj();direct_radiation_s=time.perf_counter()-start
        np.save(folder/'T.npy',direct)
        direct_mean,dm=analytic_mueller_tmatrix(direct,order,theta,precision='fp64',threads=2,cyclic_order=1)
        np.savez_compressed(folder/'direct_average.npz',theta_deg=theta,mueller=direct_mean)
        index=cutoff_indices(order,order-2)
        truncated,_=analytic_mueller_tmatrix(direct[np.ix_(index,index)],order-2,theta,precision='fp64',threads=2,cyclic_order=1)
        tail=error_metrics(truncated,direct_mean,theta)['maximum_all16_error_over_reference_M11']
        for rank in ranks:
            start=time.perf_counter();y=solve(ar[:rank,:rank],br[:rank]);pred=-1j*ka/(4*np.pi)*phase[:,None]*(dr[:,:rank]@y)*phase[None,:].conj();reduced_solve_s=time.perf_counter()-start
            mean,pm=analytic_mueller_tmatrix(pred,order,theta,precision='fp64',threads=2,cyclic_order=1)
            assert pm['Euler_quadrature_nodes']==0
            metrics=error_metrics(mean,direct_mean,theta)
            csca=float(abs(np.vdot(pred,pred).real/np.vdot(direct,direct).real-1));cext=float(abs(np.trace(pred).real/np.trace(direct).real-1))
            record=dict(ka=ka,offset=offset,rank=rank,**metrics,Csca_relative_error=csca,Cext_relative_error=cext,
                acceptance_error=max(metrics['maximum_all16_error_over_reference_M11'],csca,cext),
                T_relative_error=float(np.linalg.norm(pred-direct)/np.linalg.norm(direct)),
                direct_residual=direct_residual,exact_operator_residual=float(np.linalg.norm(aq[:,:rank]@y-b)/np.linalg.norm(b)),
                modal_tail_error=tail,assembly_wall_s=assembly_wall,assembly_compute_s=meta['assembly_s']+meta['incident_radiation_s'],
                modal_projection_s=modal_s,reduced_operator_projection_s=projection_s,
                direct_solve_s=direct_solve_s,direct_radiation_s=direct_radiation_s,reduced_solve_s=reduced_solve_s,
                direct_SO3_s=dm['end_to_end_s'],reduced_SO3_s=pm['end_to_end_s'])
            records.append(record);np.savez_compressed(folder/f'rank{rank}_comparison.npz',theta_deg=theta,predicted=mean,direct=direct_mean)
            print(json.dumps(dict(ka=ka,rank=rank,maximum_percent=100*record['acceptance_error'],exact_residual=record['exact_operator_residual'],
                projection_s=projection_s,direct_solve_s=direct_solve_s,reduced_solve_s=reduced_solve_s)),flush=True)
        atomic_json(out/'progress.json',records)
        del a,braw,draw,aq,factor,direct_x,b,d
        for suffix in ['A','B','D']:Path(str(prefix)+f'.{suffix}.bin').unlink()
    atomic_json(out/'report.json',dict(status='complete',records=records,model=str(modeldir),
        cost_scope='Exact operator assembly and projection performed at each ka; setup not amortized here.',
        new_small_direct_solves=len(args.offsets),new_large_T_runs=0,mesh_error_certified=False))


if __name__=='__main__':main()
