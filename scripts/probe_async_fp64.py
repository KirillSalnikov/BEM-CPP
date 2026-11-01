#!/usr/bin/env python3
"""Real small BEM packets: GPU FP32 solves with CPU/GPU FP64 residual gates.

The dense operator is preassembled and immutable. This is a scheduling and
refinement prototype, not a CPU port of the production FFT/FMM operator.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from async_fp64_residual import AsyncFP64Residual
from bounded_bem_run import atomic_json
from tmatrix_reference import modes
from tmatrix_so3_mstm import analytic_mueller_tmatrix
from study_tmatrix_cutoff import error_metrics

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cpu-threads',type=int,default=16);args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    source=ROOT/'benchmarks/size_taylor_20261031/ka6_ref2';n=4032
    a=np.array(np.memmap(source/'center/operator.A.bin',dtype=np.complex128,mode='r',shape=(7,n,n))[0]);a.setflags(write=False)
    with np.load(source/'center/projected.npz') as data:b=data['B'][0];d=data['D'][0]
    q=b.shape[1];ga=torch.tensor(a,device='cuda',dtype=torch.complex64)
    ga64=torch.tensor(a,device='cuda',dtype=torch.complex128)
    torch.cuda.synchronize();start=time.perf_counter();lu,piv=torch.linalg.lu_factor(ga);torch.cuda.synchronize();factor_s=time.perf_counter()-start
    phase=np.array([1j**(l-pol) for pol,l,m in modes(16)])
    reference_t=np.load(ROOT/'benchmarks/cpu_gpu_dense_20261031/full_dense_cpu_fp64_t2.npy')
    theta=np.linspace(0,180,1801)
    reference_mean,_=analytic_mueller_tmatrix(reference_t,16,theta,precision='fp64',threads=2,cyclic_order=1)
    tolerance=2e-6;records=[]
    def gpu_solve(rhs):
        t=torch.tensor(rhs,device='cuda',dtype=torch.complex64)
        return torch.linalg.lu_solve(lu,piv,t).cpu().numpy().astype(np.complex128)
    def solve(mode,width,depth,tol=tolerance,columns=q):
        ready=deque((lo,min(lo+width,columns),0,None,None) for lo in range(0,columns,width))
        pending=deque();answer=np.zeros((n,columns),complex);timelines=[];residuals=[];rounds=[]
        start=time.perf_counter();check_s=0.;gpu_s=0.;wait_s=0.
        checker=AsyncFP64Residual(lambda x:a@x,workers=1,capacity=depth) if mode!='gpu_strict' else None
        try:
            while ready or pending:
                while ready and (mode=='gpu_strict' or len(pending)<depth):
                    lo,hi,version,old_x,residual=ready.popleft();rhs=b[:,lo:hi]
                    gs=time.perf_counter();correction=gpu_solve(rhs if old_x is None else residual)
                    x=correction if old_x is None else old_x+correction
                    ge=time.perf_counter();gpu_s+=ge-gs;timelines.append(dict(kind='GPU_FP32',start=gs-start,end=ge-start,lo=lo,version=version))
                    if mode=='gpu_strict':
                        cs=time.perf_counter()
                        gx=torch.tensor(x,device='cuda',dtype=torch.complex128)
                        gb=torch.tensor(rhs,device='cuda',dtype=torch.complex128)
                        r=(gb-ga64@gx).cpu().numpy()
                        maximum=float(np.max(np.linalg.norm(r,axis=0)/np.linalg.norm(rhs,axis=0)))
                        ce=time.perf_counter();check_s+=ce-cs;timelines.append(dict(kind='GPU_FP64',start=cs-start,end=ce-start,lo=lo,version=version))
                        if maximum<=tol:answer[:,lo:hi]=x;residuals.append(maximum);rounds.append(version)
                        elif version<3:ready.appendleft((lo,hi,version+1,x,r))
                        else:raise RuntimeError('GPU strict refinement failed')
                    else:
                        ticket=checker.submit(x,rhs,label=f'{lo}:{hi}',version=version)
                        pending.append((ticket,lo,hi))
                if pending:
                    ticket,lo,hi=pending.popleft();before=time.perf_counter();verdict=checker.take(ticket);wait_s+=time.perf_counter()-before
                    check_s+=verdict.seconds;timelines.append(dict(kind='CPU_FP64',start=verdict.started-start,end=verdict.finished-start,lo=lo,version=ticket.version))
                    if verdict.accepts(tol):
                        answer[:,lo:hi]=verdict.candidate;residuals.append(verdict.maximum_relative);rounds.append(ticket.version)
                    elif ticket.version<3:ready.appendleft((lo,hi,ticket.version+1,verdict.candidate,verdict.residual))
                    else:raise RuntimeError('CPU strict refinement failed')
        finally:
            if checker:checker.close()
        wall=time.perf_counter()-start
        overlap=sum(max(0,min(g['end'],c['end'])-max(g['start'],c['start']))
            for g in timelines if g['kind']=='GPU_FP32' for c in timelines if c['kind']=='CPU_FP64')
        return answer,dict(mode=mode,packet_width=width,queue_depth=depth,wall_s=wall,
            gpu_FP32_s=gpu_s,strict_s=check_s,waiting_for_CPU_s=wait_s,CPU_GPU_overlap_s=overlap,
            maximum_accepted_relative_residual=max(residuals),refinement_rounds=rounds,timeline=timelines)
    # Warm both GPU arithmetic and CPU BLAS outside recorded trials.
    x=gpu_solve(b[:,:32]);a@x
    for trial in range(4):
        configs=[('gpu_strict',32,1),('cpu_serial',32,1),('cpu_pipeline',32,2),
                 ('gpu_strict',64,1),('cpu_serial',64,1),('cpu_pipeline',64,2)]
        if trial%2:configs=configs[::-1]
        for mode,width,depth in configs:
            x,r=solve(mode,width,depth);r['trial']=trial
            if trial==0:
                # Independent whole-block FP64 residual; never a norm-only GPU check.
                r['independent_CPU_max_residual']=float(np.max(np.linalg.norm(b-a@x,axis=0)/np.linalg.norm(b,axis=0)))
                if r['independent_CPU_max_residual']>tolerance*(1+1e-8):raise RuntimeError('independent residual rejected')
                t=-1j*6/(4*np.pi)*phase[:,None]*(d@x)*phase[None,:].conj()
                mean,info=analytic_mueller_tmatrix(t,16,theta,precision='fp64',threads=2,cyclic_order=1)
                r['T_difference_vs_CPU64']=float(np.linalg.norm(t-reference_t)/np.linalg.norm(reference_t))
                r['maximum_all16_error_over_CPU64_M11']=error_metrics(mean,reference_mean,theta)['maximum_all16_error_over_reference_M11']
            records.append(r);atomic_json(out/'progress.json',records)
            print(json.dumps({k:v for k,v in r.items() if k not in ['timeline','refinement_rounds']}),flush=True)
    # Tight tolerance forces true mixed-precision refinement on an independent block.
    x,tight=solve('cpu_pipeline',32,2,tol=1e-10,columns=64)
    tight['independent_CPU_max_residual']=float(np.max(np.linalg.norm(b[:,:64]-a@x,axis=0)/np.linalg.norm(b[:,:64],axis=0)))
    if tight['independent_CPU_max_residual']>1e-10:raise RuntimeError('tight CPU residual failed')
    summaries=[]
    for mode,width,depth in [('gpu_strict',32,1),('cpu_serial',32,1),('cpu_pipeline',32,2),('gpu_strict',64,1),('cpu_serial',64,1),('cpu_pipeline',64,2)]:
        rows=[r for r in records if r['mode']==mode and r['packet_width']==width]
        summaries.append(dict(mode=mode,packet_width=width,queue_depth=depth,
            median_s=statistics.median(r['wall_s'] for r in rows),
            median_overlap_s=statistics.median(r['CPU_GPU_overlap_s'] for r in rows),
            max_residual=max(r['maximum_accepted_relative_residual'] for r in rows),
            maximum_refinement_rounds=max(max(r['refinement_rounds']) for r in rows)))
    atomic_json(out/'report.json',dict(status='complete',summaries=summaries,records=records,tight_control=tight,
        GPU=torch.cuda.get_device_name(0),CPU_threads=args.cpu_threads,factor_once_s=factor_s,dofs=n,RHS=q,tolerance=tolerance,
        operator='preassembled dense BEM FP64, ka6 ref2 n1.3116 aspect10/7',
        included='FP32 GPU packet solves, all FP64 residual gates and corrections, CPU/GPU transfers; GPU LU setup excluded and shared',
        excluded=['operator assembly','T radiation and SO3 postprocessing'],
        snapshots_immutable=True,gate='max per-column true FP64 residual',
        production_FMM_CPU_operator_available=False,production_modified=False,new_physical_cases=0))


if __name__=='__main__':main()
