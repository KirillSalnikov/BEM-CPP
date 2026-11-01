#!/usr/bin/env python3
"""Measure spacing of actual T-matrix samples, distinct from Taylor continuation.

The existing small-size reference family is reused. Extra small direct solves
provide independent midpoint controls. No production ka40 T is constructed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.linalg import lu_factor,lu_solve
from bounded_bem_run import atomic_json
from tmatrix_reference import modes,tangent_frame,transverse_harmonics
from tmatrix_size_interpolation import RationalMatrix,fit_aaa
from tmatrix_so3_mstm import analytic_mueller_tmatrix
from study_tmatrix_cutoff import error_metrics,cutoff_indices

ROOT=Path(__file__).resolve().parents[1]


def polynomial(nodes,values,x):
    z=np.asarray(nodes,dtype=float)
    weights=np.array([1/np.prod(z[i]-np.delete(z,i)) for i in range(len(z))])
    return RationalMatrix(z,tuple(values),weights,(float(z.min()),float(z.max()))).evaluate(x)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ka',type=float,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    source=ROOT/f'benchmarks/size_taylor_20261031/ka{args.ka:g}_ref2'
    old=json.loads((source/'plan.json').read_text());L=old['order'];ref=old['ref'];ka0=args.ka
    binary=ROOT/'bin/muller_tmatrix_taylor_probe'
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=old['binary_sha256']:raise RuntimeError('direct assembler changed')
    positive=[.025,.05,.0625,.125,.1875,.25,.3125,.375,.4375,.5,.625,.75,.875,1.]
    offsets=sorted([-x for x in positive]+[0.]+positive)
    definitions=[]
    for h in [.5,.25,.125]:
        # Fixed [-1,1] domain for the first two, [-.5,.5] for the finer local window.
        nodes=np.arange(-2 if h==.5 else -4,(2 if h==.5 else 4)+1)*h
        controls=(nodes[:-1]+nodes[1:])/2
        for method in ['linear','local_cubic','AAA']:
            definitions.append(dict(method=method,step=float(h),nodes=nodes.tolist(),controls=controls.tolist()))
    for h in [.1,.05]:definitions.append(dict(method='linear',step=h,nodes=[-h/2,h/2],controls=[0.]))
    plan=dict(ka0=ka0,ref=ref,order=L,ri=old['ri'],aspect=old['prism_aspect'],wavelength=.532,
        theta_step_deg=.1,tolerance=.01,source=str(source),direct_assembler_sha256=old['binary_sha256'],
        offsets=offsets,models=definitions,
        scope='Actual complex-T sample interpolation, no Taylor coefficients used; same-mesh error only.')
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:raise RuntimeError('experiment definition changed')
    atomic_json(out/'plan.json',plan)
    def status(phase,**kwargs):atomic_json(out/'status.json',dict(state='running',phase=phase,pid=os.getpid(),updated_unix=time.time(),**kwargs))
    band=2*L+4;z,w=np.polynomial.legendre.leggauss(band)
    zz,pp=np.meshgrid(z,np.arange(2*band)*np.pi/band,indexing='ij')
    directions=np.stack((np.sqrt(1-zz*zz)*np.cos(pp),np.sqrt(1-zz*zz)*np.sin(pp),zz),axis=-1).reshape(-1,3)
    weights=np.repeat(w,2*band)*np.pi/band
    v=transverse_harmonics(directions,L);wv=np.repeat(weights,2)[:,None]*v
    phase=np.array([1j**(l-pol) for pol,l,m in modes(L)])
    theta=np.linspace(0,180,1801);samples={};averages={};references=[]
    def average(t):
        mean,info=analytic_mueller_tmatrix(t,L,theta,precision='fp64',threads=2,cyclic_order=1)
        assert info['Euler_quadrature_nodes']==0
        return mean
    for delta in offsets:
        ka=ka0+delta;folder=out/f'direct_ka{ka:.8f}';folder.mkdir(exist_ok=True)
        status('direct_controls',ka=ka,completed=len(samples))
        saved=source/'central_T.npy' if delta==0 else source/f'control_{ka:.8f}/T.npy'
        local=folder/'T.npy';new=False;residual=None
        if saved.exists():path=saved
        elif local.exists():path=local
        else:
            new=True;prefix=folder/'operator';started=time.perf_counter()
            command=[str(binary),str(ref),str(ka),str(old['ri']),'0',str(old['prism_aspect']),
                str(source/'grid.bin'),str(prefix),'7',str(old['duffy']),'0','0']
            with (folder/'worker.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            meta=json.loads(prefix.with_suffix('.json').read_text());n=meta['dofs'];channels=2*meta['points']
            a=np.memmap(str(prefix)+'.A.bin',mode='r',dtype=np.complex128,shape=(n,n))
            b=np.memmap(str(prefix)+'.B.bin',mode='r',dtype=np.complex128,shape=(n,channels))
            d=np.memmap(str(prefix)+'.D.bin',mode='r',dtype=np.complex128,shape=(channels,n))
            bm=b@wv;dm=wv.conj().T@d
            factor=lu_factor(a);x=lu_solve(factor,bm)
            residual=float(np.linalg.norm(a@x-bm)/np.linalg.norm(bm))
            if not np.isfinite(residual) or residual>1e-10:raise RuntimeError('direct solve residual failed')
            t=-1j*ka/(4*np.pi)*phase[:,None]*(dm@x)*phase[None,:].conj()
            np.save(local,t);path=local
            atomic_json(folder/'direct.json',dict(ka=ka,residual=residual,wall_s=time.perf_counter()-started,command=command))
            del a,b,d,bm,dm,factor,x,t
            for suffix in ['A','B','D']:Path(str(prefix)+f'.{suffix}.bin').unlink()
        sample=np.load(path,mmap_mode='r');samples[delta]=sample
        if not (folder/'average.npz').exists():np.savez_compressed(folder/'average.npz',theta_deg=theta,mueller=average(sample))
        with np.load(folder/'average.npz') as cache:averages[delta]=cache['mueller']
        references.append(dict(ka=ka,delta=delta,T=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),new_solve=new,residual=residual))
        print(json.dumps(dict(phase='direct_ready',ka=ka,new_solve=new,residual=residual)),flush=True)
    atomic_json(out/'references.json',references)
    records=[]
    for index,definition in enumerate(definitions):
        method=definition['method'];h=definition['step'];nodes=np.array(definition['nodes']);controls=definition['controls']
        if set(nodes)&set(controls):raise RuntimeError('training/validation overlap')
        train=[samples[x] for x in nodes];model_dir=out/f'{index:02d}_{method}_h{h:g}';model_dir.mkdir(exist_ok=True)
        status('interpolation_controls',method=method,step=h)
        start=time.perf_counter();diagnostics={}
        if method=='AAA':model,diagnostics=fit_aaa(nodes,train,tolerance=1e-6)
        atomic_json(model_dir/'model.json',dict(**definition,diagnostics=diagnostics,fit_s=time.perf_counter()-start,
            training_sha256=[next(r['sha256'] for r in references if r['delta']==x) for x in nodes],
            holdouts_excluded_from_fit=True))
        checks=[]
        for delta in controls:
            if method=='AAA':predicted=model.evaluate(delta);support=model.nodes.tolist()
            else:
                count=2 if method=='linear' else 4
                left=int(np.searchsorted(nodes,delta))-1-(1 if count==4 else 0)
                left=max(0,min(left,len(nodes)-count));support=nodes[left:left+count].tolist()
                predicted=polynomial(support,[samples[x] for x in support],delta)
            mean=average(predicted);direct=averages[delta];metrics=error_metrics(mean,direct,theta)
            actual=samples[delta];csca=float(abs(np.vdot(predicted,predicted).real/np.vdot(actual,actual).real-1))
            cext=float(abs((np.trace(predicted).real-np.trace(actual).real)/np.trace(actual).real))
            metric=max(metrics['maximum_all16_error_over_reference_M11'],csca,cext)
            check=dict(ka=ka0+delta,delta=delta,support_ka=[ka0+x for x in support],**metrics,
                T_relative_error=float(np.linalg.norm(predicted-actual)/np.linalg.norm(actual)),
                Csca_relative_error=csca,Cext_relative_error=cext,acceptance_error=metric)
            checks.append(check)
            np.savez_compressed(model_dir/f'control_{ka0+delta:.8f}.npz',theta_deg=theta,predicted=mean,direct=direct)
        record=dict(**definition,ka0=ka0,training_count=len(nodes),control_count=len(checks),checks=checks,
            maximum_error=max(r['acceptance_error'] for r in checks),
            maximum_mueller_error=max(r['maximum_all16_error_over_reference_M11'] for r in checks))
        record['passes_1pct']=record['maximum_error']<=.01
        records.append(record);atomic_json(model_dir/'report.json',record);atomic_json(out/'progress.json',records)
        print(json.dumps(dict(phase='model_result',method=method,step=h,maximum_percent=100*record['maximum_error'],passes=record['passes_1pct'])),flush=True)
    # Modal control at every actual midpoint, independent of interpolation.
    indices=cutoff_indices(L,L-2);tail=[]
    for delta in sorted({x for d in definitions for x in d['controls']}):
        t=samples[delta];truncated,info=analytic_mueller_tmatrix(t[np.ix_(indices,indices)],L-2,theta,precision='fp64',threads=2,cyclic_order=1)
        tail.append(dict(ka=ka0+delta,error=error_metrics(truncated,averages[delta],theta)['maximum_all16_error_over_reference_M11']))
    report=dict(plan=plan,records=records,modal_tail_controls=tail,new_direct_solve_count=sum(r['new_solve'] for r in references),
        Euler_quadrature_nodes=0,mesh_accuracy_certified=False,ka40_certified=False)
    atomic_json(out/'report.json',report)
    atomic_json(out/'status.json',dict(state='complete',pid=os.getpid(),updated_unix=time.time()))


if __name__=='__main__':main()
