#!/usr/bin/env python3
"""Extend only operator/RHS/radiation derivatives of a frozen current ROM.

No additional current derivatives or off-center training solves are required.
The separately built dense assembler supports order 12; original binary and
saved central data are preserved. Raw temporary operators are deleted after
successful coefficient checks and projection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from bounded_bem_run import atomic_json
from tmatrix_reference import transverse_harmonics
from tmatrix_size_rom import ReducedBEM
from tmatrix_so3_mstm import analytic_mueller_tmatrix
from study_tmatrix_cutoff import error_metrics

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ranks',type=int,nargs='+',required=True)
    args=p.parse_args();old=args.model.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    definition=json.loads((old/'plan.json').read_text());plan=definition['parameters'];source=Path(definition['source'])
    if not (old/'frozen.json').exists():raise RuntimeError('current model not frozen')
    basis=np.load(old/'basis.npy')[:,:max(args.ranks)]
    with np.load(old/'central_inverse.npz') as data:
        test=data['test'][:,:max(args.ranks)];phase=data['phase'];aold=data['A'][:,:max(args.ranks),:max(args.ranks)];bold=data['B'][:,:max(args.ranks)];dold=data['D'][:,:,:max(args.ranks)]
    with np.load(source/'center/projected.npz') as data:old_b=data['B'];old_d=data['D']
    n=basis.shape[0];order=plan['order'];degree=12;ka0=plan['ka0'];binary=ROOT/'bin/muller_tmatrix_rom_operator_probe'
    command=[str(binary),str(plan['ref']),str(ka0),str(plan['ri']),str(plan['ri_imag']),str(plan['prism_aspect']),
        str(source/'grid.bin'),str(out/'operator'),str(plan['quad']),str(plan['duffy']),str(degree),'0']
    atomic_json(out/'plan.json',dict(model=str(old),parameters=plan,degree=degree,ranks=args.ranks,command=command,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),basis_sha256=hashlib.sha256((old/'basis.npy').read_bytes()).hexdigest(),
        off_center_training_solves=0,new_current_derivatives=0))
    start=time.perf_counter()
    with (out/'assembly.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    assembly_wall=time.perf_counter()-start
    meta=json.loads((out/'operator.json').read_text());channels=2*meta['points']
    a=np.memmap(out/'operator.A.bin',dtype=np.complex128,mode='r',shape=(degree+1,n,n))
    b=np.memmap(out/'operator.B.bin',dtype=np.complex128,mode='r',shape=(degree+1,n,channels))
    d=np.memmap(out/'operator.D.bin',dtype=np.complex128,mode='r',shape=(degree+1,channels,n))
    acheck=np.memmap(source/'center/operator.A.bin',dtype=np.complex128,mode='r',shape=(7,n,n))
    band=2*order+4;z,w=np.polynomial.legendre.leggauss(band);zz,pp=np.meshgrid(z,np.arange(2*band)*np.pi/band,indexing='ij')
    directions=np.stack((np.sqrt(1-zz*zz)*np.cos(pp),np.sqrt(1-zz*zz)*np.sin(pp),zz),axis=-1).reshape(-1,3)
    weights=np.repeat(w,2*band)*np.pi/band;wv=np.repeat(weights,2)[:,None]*transverse_harmonics(directions,order)
    ar=[];br=[];dr=[];checks=[];modal_s=project_s=0.
    for j in range(degree+1):
        start=time.perf_counter();bm=b[j]@wv;dm=wv.conj().T@d[j];modal_s+=time.perf_counter()-start
        if j<7:
            check=dict(degree=j,A_error=float(np.linalg.norm(a[j]-acheck[j])/max(np.linalg.norm(acheck[j]),1e-300)),
                B_error=float(np.linalg.norm(bm-old_b[j])/max(np.linalg.norm(old_b[j]),1e-300)),
                D_error=float(np.linalg.norm(dm-old_d[j])/max(np.linalg.norm(old_d[j]),1e-300)))
            if max(check[k] for k in ['A_error','B_error','D_error'])>1e-10:raise RuntimeError('original derivative coefficient mismatch')
            checks.append(check);ar.append(aold[j]);br.append(bold[j]);dr.append(dold[j])
        else:
            start=time.perf_counter();ar.append(test.conj().T@(a[j]@basis));br.append(test.conj().T@bm);dr.append(dm@basis);project_s+=time.perf_counter()-start
        print(json.dumps(dict(phase='coefficient',degree=j)),flush=True)
    model=ReducedBEM(np.array(ar),np.array(br),np.array(dr),phase,ka0)
    np.savez(out/'model.npz',A=model.A,B=model.B,D=model.D,phase=phase,ka0=ka0)
    atomic_json(out/'setup.json',dict(assembly_wall_s=assembly_wall,assembly_compute_s=meta['assembly_s']+meta['incident_radiation_s'],
        modal_projection_s=modal_s,additional_reduced_projection_s=project_s,original_coefficients_checks=checks,
        original_data_cost_is_separate=True))
    del a,b,d,acheck,bm,dm
    for suffix in ['A','B','D']:(out/f'operator.{suffix}.bin').unlink()
    # Freeze before loading independent control matrices.
    atomic_json(out/'frozen.json',dict(training_ka=[ka0],degree=degree,frozen_unix=time.time()))
    refs=json.loads((ROOT/f'benchmarks/interpolation_step_20261031/ka{ka0:g}/references.json').read_text())
    theta=np.linspace(0,180,1801);records=[]
    for ref in refs:
        ka=ref['ka']
        if ka==ka0:continue
        path=Path(ref['T'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:raise RuntimeError('reference changed')
        direct=np.load(path)
        cached=ROOT/f'benchmarks/interpolation_step_20261031/ka{ka0:g}/direct_ka{ka:.8f}/average.npz'
        with np.load(cached) as data:direct_mean=data['mueller']
        for rank in args.ranks:
            start=time.perf_counter();pred,y=model.evaluate(ka,rank=rank);evaluation_s=time.perf_counter()-start
            mean,info=analytic_mueller_tmatrix(pred,order,theta,precision='fp64',threads=2,cyclic_order=1)
            assert info['Euler_quadrature_nodes']==0
            metrics=error_metrics(mean,direct_mean,theta)
            csca=float(abs(np.vdot(pred,pred).real/np.vdot(direct,direct).real-1));cext=float(abs(np.trace(pred).real/np.trace(direct).real-1))
            r=dict(ka=ka,offset=ka-ka0,rank=rank,**metrics,Csca_relative_error=csca,Cext_relative_error=cext,
                acceptance_error=max(metrics['maximum_all16_error_over_reference_M11'],csca,cext),
                evaluation_s=evaluation_s,analytic_SO3_s=info['end_to_end_s'])
            records.append(r)
            np.savez_compressed(out/f'rank{rank}_ka{ka:.8f}.npz',theta_deg=theta,predicted=mean,direct=direct_mean)
        atomic_json(out/'progress.json',records)
        print(json.dumps(dict(phase='control',ka=ka,errors={r['rank']:100*r['acceptance_error'] for r in records if r['ka']==ka})),flush=True)
    atomic_json(out/'report.json',dict(status='complete',records=records,model=str(old),degree=degree,
        new_current_solves=0,off_center_training_Ts=0,new_large_T_runs=0,mesh_accuracy_certified=False))


if __name__=='__main__':main()
