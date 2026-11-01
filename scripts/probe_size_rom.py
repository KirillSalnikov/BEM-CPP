#!/usr/bin/env python3
"""Current-moment reduced BEM experiment; only small saved reference families."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.linalg import lu_factor
from bounded_bem_run import atomic_json
from tmatrix_reference import modes
from tmatrix_size_rom import current_basis,separate_central_currents,project_bem
from tmatrix_size_taylor import polynomial
from tmatrix_so3_mstm import analytic_mueller_tmatrix
from study_tmatrix_cutoff import error_metrics

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ka',type=float,required=True)
    p.add_argument('--ref',type=int,default=2)
    p.add_argument('--ranks',type=int,nargs='+',default=[128,256,512,768])
    p.add_argument('--offsets',type=float,nargs='+',default=[-1.,-.75,-.5,-.25,.25,.5,.75,1.])
    p.add_argument('--basis-kind',choices=['moments','separate_central_currents'],default='moments')
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    source=ROOT/f'benchmarks/size_taylor_20261031/ka{args.ka:g}_ref{args.ref}'
    plan=json.loads((source/'plan.json').read_text());oldmodel=json.loads((source/'model.json').read_text())
    meta=json.loads((source/'center/operator.json').read_text());n=meta['dofs'];degree=plan['degree'];order=plan['order']
    def status(phase,**kw):atomic_json(out/'status.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**kw))
    atomic_json(out/'plan.json',dict(source=str(source),parameters=plan,ranks=args.ranks,offsets=args.offsets,
        window=1.,training_ka=[args.ka],training=args.basis_kind,target_solutions_used_for_training=False,
        full_operator_dofs=n,theta_step_deg=.1,mesh_error_certified=False,
        currents_sha256=hashlib.sha256((source/'current_coefficients.npy').read_bytes()).hexdigest(),
        code_sha256={name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
                    for name in ['tmatrix_size_rom.py','probe_size_rom.py']}))
    a=np.memmap(source/'center/operator.A.bin',dtype=np.complex128,mode='r',shape=(degree+1,n,n))
    with np.load(source/'center/projected.npz') as arrays:b=arrays['B'];d=arrays['D']
    currents=np.load(source/'current_coefficients.npy')
    status('build_basis');start=time.perf_counter()
    if args.basis_kind=='moments':basis,binfo=current_basis(currents,window=1.,max_rank=max(args.ranks))
    else:basis,binfo=separate_central_currents(currents[0],max(args.ranks))
    basis_s=time.perf_counter()-start
    np.save(out/'basis.npy',basis);atomic_json(out/'basis.json',dict(**binfo,seconds=basis_s))
    print(json.dumps(dict(phase='basis',ka=args.ka,ref=args.ref,rank=binfo['rank'],seconds=basis_s)),flush=True)
    phase=np.array([1j**(l-pol) for pol,l,m in modes(order)])
    start=time.perf_counter();factor=lu_factor(a[0]);factor_s=time.perf_counter()-start
    models={};setup={};aq=None
    for method in ['galerkin','central_inverse']:
        status('project',method=method);start=time.perf_counter()
        model,test,aq=project_bem(a,b,d,basis,phase,args.ka,testing=method,factor=factor)
        projection_s=time.perf_counter()-start;models[method]=model
        setup[method]=dict(basis_s=basis_s,projection_s=projection_s,extra_factor_s=factor_s if method=='central_inverse' else 0.,
            existing_data_preparation_s=basis_s+projection_s+(factor_s if method=='central_inverse' else 0.),
            historical_central_data_acquisition_s=sum(oldmodel['timings']['bridge'][k] for k in ['assembly_s','incident_radiation_s','modal_projection_s'])+oldmodel['timings']['moment_s'])
        np.savez(out/f'{method}.npz',A=model.A,B=model.B,D=model.D,phase=model.phase,ka0=model.ka0,test=test)
        atomic_json(out/f'{method}_setup.json',setup[method])
        print(json.dumps(dict(phase='projected',method=method,seconds=projection_s)),flush=True)
    # The models are frozen before reading any off-center reference T.
    atomic_json(out/'frozen.json',dict(training_ka=[args.ka],models=['galerkin','central_inverse'],frozen_unix=time.time()))
    theta=np.linspace(0,180,1801);records=[]
    references={}
    if args.ref==2:
        for r in json.loads((ROOT/f'benchmarks/interpolation_step_20261031/ka{args.ka:g}/references.json').read_text()):references[r['ka']]=r['T']
    for offset in args.offsets:
        ka=args.ka+offset
        path=Path(references.get(ka,str(source/f'control_{ka:.8f}/T.npy')))
        if not path.exists():raise RuntimeError(f'missing same-mesh direct control: {path}')
        direct=np.load(path);direct_mean,reference_info=analytic_mueller_tmatrix(direct,order,theta,precision='fp64',threads=2,cyclic_order=1)
        apq=polynomial(aq,offset);bp=polynomial(b,offset)
        for method,model in models.items():
            for rank in args.ranks:
                status('validate',ka=ka,method=method,rank=rank)
                start=time.perf_counter();pred,y=model.evaluate(ka,rank=rank);evaluation_s=time.perf_counter()-start
                mean,info=analytic_mueller_tmatrix(pred,order,theta,precision='fp64',threads=2,cyclic_order=1)
                assert info['Euler_quadrature_nodes']==0
                metrics=error_metrics(mean,direct_mean,theta)
                csca=float(abs(np.vdot(pred,pred).real/np.vdot(direct,direct).real-1))
                cext=float(abs(np.trace(pred).real/np.trace(direct).real-1))
                record=dict(ka=ka,offset=offset,method=method,rank=rank,**metrics,
                    acceptance_error=max(metrics['maximum_all16_error_over_reference_M11'],csca,cext),
                    Csca_relative_error=csca,Cext_relative_error=cext,T_relative_error=float(np.linalg.norm(pred-direct)/np.linalg.norm(direct)),
                    full_Taylor_operator_residual=float(np.linalg.norm(apq[:,:rank]@y-bp)/np.linalg.norm(bp)),
                    evaluation_s=evaluation_s,analytic_SO3_s=info['end_to_end_s'],
                    direct_T_path=str(path),direct_T_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                records.append(record)
                np.savez_compressed(out/f'{method}_r{rank}_ka{ka:.8f}.npz',theta_deg=theta,predicted=mean,direct=direct_mean)
                atomic_json(out/'progress.json',records)
                print(json.dumps(dict(phase='control',ka=ka,method=method,rank=rank,percent=100*record['acceptance_error'],evaluation_s=evaluation_s)),flush=True)
    atomic_json(out/'report.json',dict(records=records,setup=setup,order=order,dofs=n,degree=degree,
        basis=binfo,source=str(source),status='complete',new_Maxwell_solves=0,
        limitations=['Same-mesh error only.','Degree-6 operator error and basis error are combined here.',
            'Dense CPU timings are not a forecast for production FMM/GPU.']))
    status('complete')


if __name__=='__main__':main()
