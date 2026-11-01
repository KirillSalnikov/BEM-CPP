#!/usr/bin/env python3
"""Single-center analytic size derivatives versus independent small BEM solves.

Training: one central operator and its analytic derivatives. References at
other sizes are assembled/solved only AFTER the Taylor/Padé model is frozen.
Times use the same new dense assembler for both methods, avoiding an unfair
comparison with the slower legacy scalar assembly implementation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from bounded_bem_run import atomic_json
from tmatrix_reference import modes,tangent_frame,transverse_harmonics
from tmatrix_size_taylor import moments,polynomial,radiation_moments,matrix_pade,evaluate_pade
from tmatrix_so3_mstm import analytic_mueller_tmatrix
from study_tmatrix_cutoff import cutoff_indices,error_metrics

ROOT=Path(__file__).resolve().parents[1]


def elapsed(start):return time.perf_counter()-start
def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ka',type=float,required=True)
    p.add_argument('--ref',type=int,default=1)
    p.add_argument('--order',type=int,default=12)
    p.add_argument('--degree',type=int,default=6)
    p.add_argument('--duffy',type=int,default=4)
    p.add_argument('--offsets',type=float,nargs='+',default=[-.05,.05,-.125,.125,-.25,.25,-.5,.5,-1,1])
    p.add_argument('--verify-library',action='store_true')
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    binary=ROOT/'bin/muller_tmatrix_taylor_probe'
    plan=dict(ka0=args.ka,ref=args.ref,order=args.order,degree=args.degree,quad=7,duffy=args.duffy,
        ri=1.3116,ri_imag=0.,prism_aspect=10/7,wavelength=.532,offsets=args.offsets,
        one_central_LU=True,off_center_training_Ts=0,operator_derivatives='analytic, one quadrature traversal',
        coefficients='derivative / factorial',verify_library=args.verify_library,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:
        raise RuntimeError('benchmark definition changed')
    atomic_json(out/'plan.json',plan)
    def status(phase,**extra):atomic_json(out/'status.json',dict(state='running',phase=phase,pid=os.getpid(),updated_unix=time.time(),**extra))
    band=2*args.order+4
    z,w=np.polynomial.legendre.leggauss(band)
    zz,pp=np.meshgrid(z,np.arange(2*band)*np.pi/band,indexing='ij')
    directions=np.stack((np.sqrt(1-zz*zz)*np.cos(pp),np.sqrt(1-zz*zz)*np.sin(pp),zz),axis=-1).reshape(-1,3)
    weights=np.repeat(w,2*band)*np.pi/band
    _,_,et,ep=tangent_frame(directions)
    np.concatenate((directions,et,ep),axis=1).tofile(out/'grid.bin')
    v=transverse_harmonics(directions,args.order);wv=np.repeat(weights,2)[:,None]*v
    phase=np.array([1j**(l-pol) for pol,l,m in modes(args.order)])
    gram=relative(v.conj().T@wv,np.eye(v.shape[1]))
    if gram>1e-11:raise RuntimeError('harmonic projection quadrature failed')
    theta=np.linspace(0,180,721)
    def bridge(name,ka,degree,verify=False):
        folder=out/name;folder.mkdir(exist_ok=True);prefix=folder/'operator'
        cmd=[str(binary),str(args.ref),str(ka),str(plan['ri']),'0',str(plan['prism_aspect']),
            str(out/'grid.bin'),str(prefix),'7',str(args.duffy),str(degree),'1' if verify else '0']
        if not prefix.with_suffix('.json').exists() or not all(Path(str(prefix)+f'.{k}.bin').exists() for k in ['A','B','D']):
            atomic_json(folder/'command.json',dict(command=cmd))
            with (folder/'worker.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        meta=json.loads(prefix.with_suffix('.json').read_text());n=meta['dofs'];channels=2*meta['points']
        a=np.memmap(str(prefix)+'.A.bin',mode='r',dtype=np.complex128,shape=(degree+1,n,n))
        b=np.memmap(str(prefix)+'.B.bin',mode='r',dtype=np.complex128,shape=(degree+1,n,channels))
        d=np.memmap(str(prefix)+'.D.bin',mode='r',dtype=np.complex128,shape=(degree+1,channels,n))
        t=time.perf_counter()
        modal_b=np.array([bj@wv for bj in b]);modal_d=np.array([wv.conj().T@dj for dj in d])
        meta['modal_projection_s']=elapsed(t);del b,d
        # Projected data suffice; no enormous raw incident/radiation archive.
        np.savez(folder/'projected.npz',B=modal_b,D=modal_d)
        return a,modal_b,modal_d,meta
    status('central_analytic_derivatives')
    a,b,d,meta=bridge('center',args.ka,args.degree,args.verify_library)
    if args.verify_library and max(meta['matrix_library_error'],meta['rhs_library_error'],meta['radiation_library_error'])>1e-10:
        raise RuntimeError('independent scalar bridge verification failed')
    status('one_LU_and_current_moments')
    t=time.perf_counter();x,factor=moments(a,b);moment_seconds=elapsed(t)
    t=time.perf_counter();coeff=radiation_moments(d,x,args.ka,phase);tmatrix_seconds=elapsed(t)
    t=time.perf_counter();pade=matrix_pade(coeff,2,.5);pade_seconds=elapsed(t)
    np.save(out/'T_coefficients.npy',coeff);np.save(out/'current_coefficients.npy',x)
    np.save(out/'central_T.npy',coeff[0])
    model_hash=hashlib.sha256((out/'T_coefficients.npy').read_bytes()).hexdigest()
    setup_s=meta['assembly_s']+meta['incident_radiation_s']+meta['modal_projection_s']+moment_seconds+tmatrix_seconds+pade_seconds
    central_residual=relative(a[0]@x[0],b[0])
    if central_residual>1e-10:raise RuntimeError('central residual failed')
    atomic_json(out/'model.json',dict(frozen_before_off_center_solves=True,training_ka=[args.ka],
        T_coefficients_sha256=model_hash,pade=pade[2],central_residual=central_residual,
        timings=dict(bridge=meta,moment_s=moment_seconds,T_series_s=tmatrix_seconds,pade_s=pade_seconds,
            total_setup_s=setup_s),note='One LU; degree+1 block backsolves. Existing T alone is insufficient.'))
    print(json.dumps(dict(phase='model_ready',ka=args.ka,ref=args.ref,setup_s=setup_s,meta=meta)),flush=True)
    def average(tmatrix,order=args.order):
        mean,info=analytic_mueller_tmatrix(tmatrix,order,theta,precision='fp64',threads=2,cyclic_order=1)
        if info['Euler_quadrature_nodes']!=0:raise RuntimeError('SO3 must be analytic')
        return mean,info
    records=[];curves=[];t0mu,_=average(coeff[0])
    for delta in args.offsets:
        ka=args.ka+delta
        if ka<=0:raise ValueError('ka must be positive')
        status('independent_off_center_reference',ka=ka,completed=len(records))
        at,bt,dt,mt=bridge(f'control_{ka:.8f}',ka,0)
        t=time.perf_counter();lu=lu_factor(at[0]);xt=lu_solve(lu,bt[0]);direct_solve_s=elapsed(t)
        residual=relative(at[0]@xt,bt[0])
        if residual>1e-10:raise RuntimeError('reference residual failed')
        t=time.perf_counter();direct=-1j*ka/(4*np.pi)*phase[:,None]*(dt[0]@xt)*phase[None,:].conj();direct_transform_s=elapsed(t)
        direct_mean,info=average(direct)
        indices=cutoff_indices(args.order,args.order-2)
        truncated,_=average(direct[np.ix_(indices,indices)],args.order-2)
        tail=error_metrics(truncated,direct_mean,theta)
        model_matrix_error=relative(polynomial(a,delta),at[0])
        record=dict(ka=ka,delta=delta,ref=args.ref,direct_residual=residual,
            operator_Taylor_relative_error=model_matrix_error,
            central_current_predictor_residual=relative(at[0]@polynomial(x,delta),bt[0]),
            modal_tail_error=tail['maximum_all16_error_over_reference_M11'],
            direct_setup_s=mt['assembly_s']+mt['incident_radiation_s']+mt['modal_projection_s']+direct_solve_s+direct_transform_s,
            direct_analytic_SO3_s=info['end_to_end_s'],methods={})
        arrays=dict(theta_deg=theta,direct=direct_mean)
        candidates={'frozen_T':lambda:coeff[0]}
        for order in [1,2,4,args.degree]:candidates[f'Taylor_{order}']=lambda order=order:polynomial(coeff[:order+1],delta)
        candidates['Pade_4_2']=lambda:evaluate_pade(pade,delta)
        for name,build in candidates.items():
            t=time.perf_counter();prediction=build();evaluation_s=elapsed(t)
            mean,md=average(prediction)
            metrics=error_metrics(mean,direct_mean,theta)
            metrics.update(T_relative_error=relative(prediction,direct),evaluation_s=evaluation_s,analytic_SO3_s=md['end_to_end_s'])
            record['methods'][name]=metrics;arrays[name]=mean
        np.save(out/f'control_{ka:.8f}/T.npy',direct)
        np.savez_compressed(out/f'control_{ka:.8f}/comparison.npz',**arrays)
        atomic_json(out/f'control_{ka:.8f}/comparison.json',record)
        records.append(record);curves.append(arrays)
        atomic_json(out/'progress.json',dict(records=records,model_sha256=model_hash))
        print(json.dumps(dict(ka=ka,delta=delta,ref=args.ref,tail=record['modal_tail_error'],
            errors={k:v['maximum_all16_error_over_reference_M11'] for k,v in record['methods'].items()},
            direct_s=record['direct_setup_s'])),flush=True)
        # Direct control raw operators are reproducible and are not training data.
        del at,bt,dt,xt,lu
        for suffix in ['A','B','D']:(out/f'control_{ka:.8f}/operator.{suffix}.bin').unlink()
    if hashlib.sha256((out/'T_coefficients.npy').read_bytes()).hexdigest()!=model_hash:raise RuntimeError('model changed during controls')
    direct_total=sum(r['direct_setup_s']+r['direct_analytic_SO3_s'] for r in records)
    totals={}
    for name in records[0]['methods']:
        prediction_total=sum(r['methods'][name]['evaluation_s']+r['methods'][name]['analytic_SO3_s'] for r in records)
        accepted=[r for r in records if r['methods'][name]['maximum_all16_error_over_reference_M11']<=.01]
        accepted_direct=sum(r['direct_setup_s']+r['direct_analytic_SO3_s'] for r in accepted)
        accepted_model=sum(r['methods'][name]['evaluation_s']+r['methods'][name]['analytic_SO3_s'] for r in accepted)
        totals[name]=dict(incremental_targets_s=prediction_total,setup_included_total_s=setup_s+prediction_total,
            direct_total_s=direct_total,speedup_including_setup=direct_total/(setup_s+prediction_total),
            accepted_target_count=len(accepted),accepted_direct_s=accepted_direct,
            accepted_model_with_setup_s=setup_s+accepted_model,
            accepted_speedup_including_setup=accepted_direct/(setup_s+accepted_model),
            timing_scope='Computation incl. analytic SO3; excludes validation and artifact serialization.')
    report=dict(plan=plan,gram_error=gram,model_sha256=model_hash,records=records,timings=totals,
        tolerance=.01,Euler_quadrature_nodes=0,mesh_error_certified=False,
        limitations=['Continuation error is measured against the same discretized BEM, not exact Maxwell.',
            'Dense CPU timings do not predict FMM/GPU savings.',
            'One central LU and derivative currents are required; saved T0 alone cannot create this model.',
            'CPU proof of concept stores dense derivative operators; production needs matrix-free derivative actions.'])
    atomic_json(out/'report.json',report)
    plot(out,report,curves)
    atomic_json(out/'status.json',dict(state='complete',pid=os.getpid(),updated_unix=time.time(),report=str(out/'report.json')))


def plot(out,report,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=report['records'];plan=report['plan'];order=np.argsort([r['delta'] for r in records])
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for method in records[0]['methods']:
        axes[0].semilogy([records[i]['delta'] for i in order],
            [100*records[i]['methods'][method]['maximum_all16_error_over_reference_M11'] for i in order],'.-',label=method)
    axes[0].axhline(1,color='red',ls='--',lw=1);axes[0].set(xlabel='delta ka',ylabel='max |delta Mij|/M11 direct, %',title='Analytic SO(3): all 16 entries')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    methods=['Taylor_6','Pade_4_2']
    # Do not advertise speed obtained by including inaccurate target sizes.
    speedups=[];labels=[]
    for method in methods:
        accepted=[r for r in records if r['methods'][method]['maximum_all16_error_over_reference_M11']<=.01]
        direct=sum(r['direct_setup_s']+r['direct_analytic_SO3_s'] for r in accepted)
        incremental=sum(r['methods'][method]['evaluation_s']+r['methods'][method]['analytic_SO3_s'] for r in accepted)
        total=report['timings'][method]
        setup=total['setup_included_total_s']-total['incremental_targets_s']
        speedups.append(direct/(setup+incremental));labels.append(f'{method}\n{len(accepted)} passing targets')
    axes[1].bar(labels,speedups)
    axes[1].axhline(1,color='.5',lw=.8);axes[1].set(ylabel='Direct CPU time / model time',title='Only targets passing 1%; setup included\nCompute time incl. SO(3), excluding artifact writes')
    fig.suptitle(f"One central BEM, ka0={plan['ka0']:g}, ref={plan['ref']}; n=1.3116, H/width=10/7\nSame-mesh continuation test; no off-center T training")
    for ext in ['png','pdf']:fig.savefig(out/f'convergence_and_cost.{ext}',dpi=160)
    plt.close(fig)
    selected=min(range(len(records)),key=lambda k:abs(records[k]['delta']-.25))
    record=records[selected];data=curves[selected]
    fig,axes=plt.subplots(4,4,figsize=(15,12),layout='constrained')
    for i in range(4):
        for j in range(4):
            ax=axes[i,j]
            for name,color,style in [('direct','#1765a3','-'),('frozen_T','#888888',':'),('Taylor_6','#21864b','--'),('Pade_4_2','#dc672a','-.')]:
                y=data[name][:,i,j]/(data['direct'][:,0,0] if (i,j)!=(0,0) else 1)
                ax.plot(data['theta_deg'],y,style,color=color,label=name,lw=1.2)
            if (i,j)==(0,0):ax.set_yscale('log')
            ax.set(xlim=(0,180),xlabel='theta, degrees',title=f'M{i+1}{j+1}'+(' / direct M11' if (i,j)!=(0,0) else ''))
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f"ka0={plan['ka0']:g} -> ka={record['ka']:g}; ref={plan['ref']}; analytic SO(3)\nDerivative continuation vs independent same-mesh BEM")
    for ext in ['png','pdf']:fig.savefig(out/f'all16_delta_{record["delta"]:g}.{ext}',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    try:main()
    except Exception as error:
        import sys
        if '--out' in sys.argv:
            folder=Path(sys.argv[sys.argv.index('--out')+1]);folder.mkdir(parents=True,exist_ok=True)
            atomic_json(folder/'status.json',dict(state='failed',error=repr(error),pid=os.getpid(),updated_unix=time.time()))
        raise
