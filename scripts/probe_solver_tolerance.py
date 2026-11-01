#!/usr/bin/env python3
"""Same-mesh full-T tolerance study using saved small BEM systems.

Independent restarted right-preconditioned GMRES recurrences are batched on
the GPU. Every column is accepted only against its own original FP64 RHS.
This dense diagnostic measures observable errors; its timings are not a
forecast for the production sector/FFT solver.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from bounded_bem_run import atomic_json
from study_tmatrix_cutoff import error_metrics
from tmatrix_reference import modes
from tmatrix_so3_mstm import analytic_mueller_tmatrix

ROOT=Path(__file__).resolve().parents[1]


class BatchedGMRES:
    def __init__(self,a,group_edges=24):
        self.n=len(a)
        half=self.n//2
        # Interleave J/M coefficients per BDM edge, then use exact local
        # diagonal blocks. No inverse of the full system is used by GMRES.
        self.permutation=np.array([[i,i+1,half+i,half+i+1] for i in range(0,half,2)]).ravel()
        self.block=4*group_edges
        if self.n%self.block:raise ValueError('block size must divide this diagnostic system')
        self.a64=torch.tensor(a[np.ix_(self.permutation,self.permutation)],device='cuda',dtype=torch.complex128)
        self.a=self.a64.to(torch.complex64)
        blocks=torch.stack([self.a64[i:i+self.block,i:i+self.block] for i in range(0,self.n,self.block)])
        self.inv=torch.linalg.inv(blocks).to(torch.complex64)

    def inverse(self,v):
        return (self.inv@v.reshape(-1,self.block,v.shape[1])).reshape(v.shape)

    def packet(self,b,tolerance,restart=64,limit=400):
        b=torch.tensor(b[self.permutation],device='cuda',dtype=torch.complex128)
        scales=torch.linalg.vector_norm(b,dim=0)
        if bool(torch.any(scales==0)):raise ValueError('zero modal RHS')
        b=b/scales
        n,q=b.shape
        x=torch.zeros_like(b);r=b.clone()
        accepted=torch.zeros(q,dtype=torch.bool,device='cuda')
        iterations=torch.zeros(q,dtype=torch.int64,device='cuda')
        checks=torch.zeros_like(iterations)
        residual=torch.ones(q,dtype=torch.float64,device='cuda')
        torch.cuda.synchronize();start=time.perf_counter()
        cycles=0
        while not bool(torch.all(accepted)):
            cycles+=1
            if int(torch.max(iterations))>=limit:raise RuntimeError(f'GMRES limit: residual {residual.max().item()}')
            active=~accepted
            r[:,accepted]=0
            norm=torch.linalg.vector_norm(r,dim=0)
            v=torch.zeros((restart+1,n,q),device='cuda',dtype=torch.complex64)
            v[0]=(r/norm.clamp_min(1e-300)).to(torch.complex64)
            h=torch.zeros((restart+1,restart,q),device='cuda',dtype=torch.complex128)
            c=torch.zeros((restart,q),device='cuda',dtype=torch.complex128)
            s=torch.zeros_like(c)
            g=torch.zeros((restart+1,q),device='cuda',dtype=torch.complex128);g[0]=norm
            for col in range(restart):
                w=self.a@self.inverse(v[col])
                for _ in range(2):
                    projection=torch.einsum('knb,nb->kb',v[:col+1].conj(),w)
                    h[:col+1,col]+=projection
                    w-=torch.einsum('knb,kb->nb',v[:col+1],projection)
                next_norm=torch.linalg.vector_norm(w,dim=0)
                h[col+1,col]=next_norm
                v[col+1]=w/next_norm.clamp_min(1e-30)
                for row in range(col):
                    ha=h[row,col].clone();hb=h[row+1,col].clone()
                    h[row,col]=c[row].conj()*ha+s[row].conj()*hb
                    h[row+1,col]=-s[row]*ha+c[row]*hb
                ha=h[col,col].clone();hb=h[col+1,col].clone()
                d=torch.sqrt(torch.abs(ha)**2+torch.abs(hb)**2)
                c[col]=torch.where(d>1e-30,ha/d.clamp_min(1e-30),torch.ones_like(ha))
                s[col]=torch.where(d>1e-30,hb/d.clamp_min(1e-30),torch.zeros_like(hb))
                h[col,col]=torch.where(active,d.to(h.dtype),torch.ones_like(ha));h[col+1,col]=0
                g[col]=torch.where(active,g[col],torch.zeros_like(g[col]))
                g[col+1]=-s[col]*g[col];g[col]=c[col].conj()*g[col]
                iterations+=active
                done=active & ((torch.abs(g[col+1])<=.5*tolerance)|(next_norm<1e-30))
                active=active & ~done
                v[col+1,:,~active]=0
                used=col+1
                if not bool(torch.any(active)):break
            coefficients=torch.linalg.solve_triangular(h[:used,:used].permute(2,0,1),g[:used].T[:,:,None],upper=True)[:,:,0].T
            correction=torch.einsum('knb,kb->nb',v[:used],coefficients.to(v.dtype))
            x+=self.inverse(correction).to(x.dtype)
            r=b-self.a64@x
            residual=torch.linalg.vector_norm(r,dim=0)
            if not bool(torch.all(torch.isfinite(residual))):raise RuntimeError('nonfinite FP64 residual')
            checks+=~accepted
            accepted=residual<=tolerance
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        original=np.empty((n,q),np.complex128)
        original[self.permutation]=(x*scales).cpu().numpy()
        return original,dict(seconds=seconds,iterations=iterations.cpu().tolist(),strict_checks=checks.cpu().tolist(),
                             residuals=residual.cpu().tolist(),cycles=cycles)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ka',type=float,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',type=Path,help='prepared saved-system directory')
    parser.add_argument('--tolerances',type=float,nargs='+',default=[1e-2,3e-3,1e-3,3e-4,1e-4,1e-5,2e-6])
    parser.add_argument('--packet',type=int,default=32)
    parser.add_argument('--column-limit',type=int,default=0,help='only a solver smoke test; no T average')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    source=args.source.resolve() if args.source else ROOT/f'benchmarks/size_taylor_20261031/ka{args.ka:g}_ref2'
    plan=json.loads((source/'plan.json').read_text());n=json.loads((source/'center/operator.json').read_text())['dofs']
    order=plan['order'];q=2*order*(order+2)
    a=np.array(np.memmap(source/'center/operator.A.bin',dtype=np.complex128,mode='r',shape=(plan['degree']+1,n,n))[0])
    with np.load(source/'center/projected.npz') as data:b=data['B'][0];radiation=data['D'][0]
    exact_x=np.load(source/'current_coefficients.npy',mmap_mode='r')[0]
    phase=np.array([1j**(l-p) for p,l,m in modes(order)])
    to_t=lambda x:-1j*args.ka/(4*np.pi)*phase[:,None]*(radiation@x)*phase[None,:].conj()
    exact_t=to_t(exact_x)
    theta=np.linspace(0,180,1801)
    exact_mean,reference_info=analytic_mueller_tmatrix(exact_t,order,theta,precision='fp64',threads=2,cyclic_order=1)
    atomic_json(out/'plan.json',dict(parameters=plan,original_rhs_norm_per_column=True,all16_Mueller_normalization='reference M11 at each angle',
        preconditioner='exact diagonal blocks of 24 contiguous BDM edges, right preconditioning',
        timing_scope='dense independent GMRES packets, not production FMM/sector GMRES',tolerances=args.tolerances,packet=args.packet,
        discretization_error_measured=False,source=str(source)))
    solver=BatchedGMRES(a)
    # Warm up allocations and mixed/strict kernels outside the timed packets.
    solver.packet(b[:,:2],1e-3)
    records=[]
    for tol in args.tolerances:
        packets=[];x=np.empty_like(b);columns=min(args.column_limit,q) if args.column_limit else q
        for lo in range(0,columns,args.packet):
            hi=min(lo+args.packet,columns)
            x[:,lo:hi],info=solver.packet(b[:,lo:hi],tol)
            packets.append(info)
            atomic_json(out/'progress.json',dict(ka=args.ka,tolerance=tol,columns_completed=hi,total_columns=columns,
                solve_s=sum(p['seconds'] for p in packets),max_residual=max(max(p['residuals']) for p in packets)))
            print(json.dumps(dict(ka=args.ka,tol=tol,columns=hi,seconds=info['seconds'],max_iterations=max(info['iterations']))),flush=True)
        residual=np.linalg.norm(a@x[:,:columns]-b[:,:columns],axis=0)/np.linalg.norm(b[:,:columns],axis=0)
        if not np.isfinite(residual).all() or np.max(residual)>tol:raise RuntimeError('independent CPU residual rejected')
        record=dict(ka=args.ka,tolerance=tol,solve_s=sum(p['seconds'] for p in packets),
            max_strict_residual=float(np.max(residual)),median_strict_residual=float(np.median(residual)),
            total_iterations=sum(sum(p['iterations']) for p in packets),maximum_iterations=max(max(p['iterations']) for p in packets),
            mean_iterations=float(np.mean([i for p in packets for i in p['iterations']])),
            current_relative_error=float(np.linalg.norm(x[:,:columns]-exact_x[:,:columns])/np.linalg.norm(exact_x[:,:columns])))
        if columns==q:
            t=to_t(x);mean,info=analytic_mueller_tmatrix(t,order,theta,precision='fp64',threads=2,cyclic_order=1)
            assert info['Euler_quadrature_nodes']==0
            metrics=error_metrics(mean,exact_mean,theta)
            csca=float(abs(np.vdot(t,t).real/np.vdot(exact_t,exact_t).real-1))
            cext=float(abs(np.trace(t).real/np.trace(exact_t).real-1))
            record.update(**metrics,Csca_relative_error=csca,Cext_relative_error=cext,
                acceptance_error=max(metrics['maximum_all16_error_over_reference_M11'],csca,cext),
                T_relative_error=float(np.linalg.norm(t-exact_t)/np.linalg.norm(exact_t)),analytic_average_s=info['end_to_end_s'])
            np.save(out/f'T_tol{tol:.0e}.npy',t)
            np.savez_compressed(out/f'Mueller_tol{tol:.0e}.npz',theta=theta,reference=exact_mean,candidate=mean)
        records.append(record);atomic_json(out/'results.json',records)
        print(json.dumps(dict(phase='tolerance_complete',**record)),flush=True)
    atomic_json(out/'complete.json',dict(complete=True,full_T=not bool(args.column_limit),records=records))


if __name__=='__main__':main()
