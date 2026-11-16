"""Full eight-density Dirac action: FMM3D plus exact sparse near corrections."""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'deps'))
sys.path.insert(0,str(B.parent/'clifford_dielectric_20261115'))
from solve import Dirac,lib,address,check

class DiracFMM(Dirac):
    def __init__(self,*args,eps=1e-8,engine='native',**kwargs):
        if not 1e-13<=eps<=1e-3:raise ValueError('FMM precision range')
        if engine not in ('native','wheel'):raise ValueError('FMM engine must be native or wheel')
        kwargs['correction_only']=True;kwargs.setdefault('target_order',13)
        super().__init__(*args,**kwargs)
        self.eps=eps
        if engine=='native':
            from native import NativeFMM
            self.engine=NativeFMM()
        elif engine=='wheel':
            import fmm3dpy
            self.engine=fmm3dpy
        self.engine_name=engine
        self.points=np.asfortranarray(self.quadrature()[:,:,:3].reshape(-1,3).T)
        self.near_s=0.;self.fmm_s=0.;self.projection_s=0.
    def cauchy(self,x,medium=0):
        if medium not in (0,1):raise ValueError('medium')
        x=np.ascontiguousarray(x,dtype=complex).reshape(-1)
        if x.size!=self.size:raise ValueError('density dimensions')
        q=np.empty((13*self.count,8),complex);lib.dirac_quadrature_charges(self.p,address(x),address(q))
        k=self.ka*(self.index if medium else 1)
        before=time.perf_counter()
        v=self.engine.hfmm3d(eps=self.eps,zk=k,sources=self.points,charges=np.asfortranarray(q.T),pg=2,nd=8)
        self.fmm_s+=time.perf_counter()-before
        if v.ier:raise RuntimeError('FMM3D error '+str(v.ier))
        pot=v.pot;g=-2*v.grad;s=2j*k*pot
        f=np.empty((13*self.count,8),complex)
        f[:,0]=g[0,0]+g[1,1]+g[2,2]+s[7]
        f[:,4]=-g[4,0]-g[5,1]-g[6,2]+s[3]
        f[:,1]=g[2,1]-g[1,2]-g[3,0]+s[4]
        f[:,2]=g[0,2]-g[2,0]-g[3,1]+s[5]
        f[:,3]=g[1,0]-g[0,1]-g[3,2]+s[6]
        f[:,5]=g[6,1]-g[5,2]-g[7,0]-s[0]
        f[:,6]=g[4,2]-g[6,0]-g[7,1]-s[1]
        f[:,7]=g[5,0]-g[4,1]-g[7,2]-s[2]
        before=time.perf_counter();result=np.empty(self.size,complex)
        lib.dirac_project_cauchy(self.p,address(f),address(result));self.projection_s+=time.perf_counter()-before
        before=time.perf_counter();near=np.empty_like(result)
        check(lib.dirac_near_correction(self.p,medium,address(x),address(near)));self.near_s+=time.perf_counter()-before
        return result+near
    def apply(self,x):
        x=np.ascontiguousarray(x,dtype=complex).reshape(-1,8);start=time.perf_counter()
        e=self.cauchy(x*self.scaling[3],0).reshape(-1,8)
        i=self.cauchy(x*self.scaling[1],1).reshape(-1,8)
        y=x+self.scaling[0]*i-self.scaling[2]*e
        self.calls+=1;self.action_s+=time.perf_counter()-start
        return y.ravel()

    def solve_refined(self,incident,tol=1e-8,strict_eps=1e-10,inner_tol=1e-3,max_corrections=6):
        """Cheap FMM GMRES corrections with a strict, independent final residual."""
        from scipy.sparse.linalg import LinearOperator,gmres
        if not 0<tol<1 or not 0<inner_tol<1 or not 1e-13<=strict_eps<=self.eps:raise ValueError('refinement tolerances')
        f=np.ascontiguousarray(incident,dtype=complex).reshape(-1)
        if f.size!=self.size or not np.all(np.isfinite(f)):raise ValueError('incident dimensions or values')
        b=np.empty_like(f);lib.dirac_rhs(self.p,address(f),address(b));bnorm=np.linalg.norm(b)
        if bnorm==0:return np.zeros_like(b),dict(seconds=0.,action_s=0.,calls=0,iterations=0,info=0,residual=0.,history=[],strict_history=[])
        h=np.zeros_like(b);residual=b.copy();history=[];strict_history=[];start=time.perf_counter();before=self.action_s;calls=self.calls
        low_eps=self.eps
        for correction in range(max_corrections):
            local=[]
            update,info=gmres(LinearOperator((self.size,self.size),matvec=self.apply,dtype=complex),residual,
                rtol=inner_tol if correction==0 else .02,atol=0,restart=60,maxiter=10,
                callback=local.append,callback_type='pr_norm')
            history.extend(float(v) for v in local)
            if info:raise RuntimeError('inner Dirac GMRES failed: '+str(info))
            h+=update
            try:
                self.eps=strict_eps;residual=b-self.apply(h)
            finally:self.eps=low_eps
            true_residual=float(np.linalg.norm(residual)/bnorm);strict_history.append(true_residual)
            if not np.isfinite(true_residual):raise RuntimeError('nonfinite strict Dirac residual')
            if true_residual<=tol:
                return h,dict(seconds=time.perf_counter()-start,action_s=self.action_s-before,calls=self.calls-calls,
                    iterations=len(history),info=0,residual=true_residual,history=history,strict_history=strict_history,
                    refinement_steps=correction+1,inner_fmm_eps=low_eps,strict_fmm_eps=strict_eps)
        raise RuntimeError('strict Dirac refinement did not converge: '+str(strict_history))
