"""Cheap intermediate residual updates, strict acceptance of the final answer.

After a strict residual replacement r=b-A*x, updates use r <- r-A_mixed*dx.
The approximation therefore acts on the correction, not the whole solution.
An approximate residual NEVER authorizes acceptance.
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres

def solve_checked(rhs,inner,mixed,strict,preconditioner,*,tolerance=2e-6,restart=80,max_cycles=6,max_outer=8,progress=None):
    b=np.asarray(rhs,dtype=complex);n=len(b);bnorm=float(np.linalg.norm(b))
    x=np.zeros(n,complex);r=b.copy();ratio=1.;stagnant=0
    record=dict(iterations=0,solver_s=0.,mixed_residual_s=0.,strict_check_s=0.,mixed_calls=0,strict_calls=0,corrections=[])
    if bnorm==0:
        record.update(strict=0.,status='zero_rhs');return x,record
    A=LinearOperator((n,n),matvec=inner,dtype=complex)
    for outer in range(max_outer):
        history=[];last=[time.perf_counter()]
        def callback(value):
            history.append(float(value))
            if progress and time.perf_counter()-last[0]>20:
                progress(dict(outer=outer,iterations=len(history),inner=float(value)));last[0]=time.perf_counter()
        start=time.perf_counter()
        dx,info=gmres(A,r,M=preconditioner,rtol=1e-4 if outer==0 else min(.05,.15*tolerance/max(ratio,1e-300)),atol=0,restart=restart,maxiter=max_cycles,callback=callback,callback_type='pr_norm')
        record['solver_s']+=time.perf_counter()-start;record['iterations']+=len(history);x+=dx
        start=time.perf_counter();r-=mixed(dx);record['mixed_residual_s']+=time.perf_counter()-start;record['mixed_calls']+=1
        previous=ratio;ratio=float(np.linalg.norm(r)/bnorm)
        if not np.isfinite(ratio):raise RuntimeError('Nonfinite intermediate residual')
        stagnant=stagnant+1 if ratio>.8*previous else 0
        row=dict(iterations=len(history),info=int(info),proxy_residual=ratio)
        if ratio<=.25*tolerance or stagnant>=2 or outer==max_outer-1:
            start=time.perf_counter();r=b-strict(x);record['strict_check_s']+=time.perf_counter()-start;record['strict_calls']+=1
            ratio=float(np.linalg.norm(r)/bnorm);row['strict_residual']=ratio;stagnant=0
            if not np.isfinite(ratio):raise RuntimeError('Nonfinite strict residual')
            if ratio<=tolerance:
                record['corrections'].append(row);record.update(strict=ratio,status='complete');return x,record
        record['corrections'].append(row)
        if progress:progress(dict(outer=outer,**row))
    record.update(strict=ratio,status='failed');return x,record
