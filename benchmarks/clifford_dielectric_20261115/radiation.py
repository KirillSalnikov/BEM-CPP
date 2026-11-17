"""Reusable FP64 planar-face Fourier radiation for the Dirac P0..P3 basis.

The same 13-point native quadrature is transformed; no density components or
panel moments are dropped. Curved surfaces use the independent direct backend.
"""
import sys,time
import numpy as np
from solve import R,lib,address

def load_finufft():
    try:
        import finufft
    except ImportError:
        local=R/'benchmarks/gpu_media_radiation_20261115/python_deps'
        if not local.exists():raise
        sys.path.insert(0,str(local))
        import finufft
    return finufft

class DiracRadiation:
    def __init__(self,d,directions,eps=1e-14,threads=8):
        finufft=load_finufft()
        start=time.perf_counter();self.d=d;self.directions=np.ascontiguousarray(directions,float)
        if self.directions.ndim!=2 or self.directions.shape[1]!=3 or len(self.directions)==0 or not np.isfinite(self.directions).all() or np.max(abs(np.linalg.norm(self.directions,axis=1)-1))>1e-12:
            raise ValueError('radiation requires finite unit directions')
        if not 1e-14<=eps<=1e-8 or not 1<=threads<=64:raise ValueError('radiation tolerance/threads')
        points=d.points.T if hasattr(d,'points') and d.points is not None else d.quadrature()[:,:,:3].reshape(-1,3)
        normal=d.frame[:,0];center=d.xyz-d.xyz.mean(axis=0)
        scale=max(np.ptp(d.xyz,axis=0).max(),1e-100)
        keys=np.column_stack((normal,np.sum(center*normal,axis=1)/scale))
        _,groups=np.unique(np.round(keys,10),axis=0,return_inverse=True)
        if groups.max()>=128:raise ValueError('radiation needs planar patches; use direct for curved surfaces')
        self.faces=[];self.owned_bytes=0
        for group in range(groups.max()+1):
            panels=np.flatnonzero(groups==group);ids=(13*panels[:,None]+np.arange(13)).ravel()
            position=points[ids]
            # A mean of millions of repeated coordinates can drift off the
            # plane by 1e-11 and introduce a spurious ka-dependent phase.
            # An actual quadrature point is on the plane by construction.
            origin=position[0].copy();position-=origin;n=normal[panels[0]]
            if np.max(abs(position@n))>2e-11*scale:raise ValueError('nonplanar radiation patch')
            axis=np.eye(3)[np.argmin(abs(n))];u=np.cross(n,axis);u/=np.linalg.norm(u);v=np.cross(n,u)
            x=np.ascontiguousarray(position@u);y=np.ascontiguousarray(position@v)
            s=np.ascontiguousarray(d.ka*(self.directions@u));t=np.ascontiguousarray(d.ka*(self.directions@v))
            plan=finufft.Plan(3,2,n_trans=8,eps=eps,isign=-1,dtype='complex128',nthreads=threads)
            plan.setpts(x,y,s=s,t=t)
            phase=np.exp(-1j*d.ka*(self.directions@origin))
            self.faces.append((ids,phase,plan));self.owned_bytes+=ids.nbytes+phase.nbytes+x.nbytes+y.nbytes+s.nbytes+t.nbytes
        self.setup_s=time.perf_counter()-start;self.action_s=0.;self.calls=0;self.eps=eps

    def apply(self,density,incident=None):
        start=time.perf_counter();d=self.d;h=np.asarray(density,complex).reshape(d.count,d.basis_count,8)
        if not np.isfinite(h).all():raise ValueError('nonfinite radiation density')
        if incident is not None:h=h-np.asarray(incident).reshape(h.shape)/d.scaling[3]
        q=np.ascontiguousarray(h*d.scaling[3]);charges=np.empty((13*d.count,8),complex)
        lib.dirac_quadrature_charges(d.p,address(q),address(charges));del q
        total=np.zeros((8,len(self.directions)),complex)
        for ids,phase,plan in self.faces:
            total+=plan.execute(np.ascontiguousarray(charges[ids].T))*phase
        total=total.T/(4*np.pi);n=self.directions;h=total[:,:3];e=total[:,4:7];ik=1j*d.ka
        out=np.empty_like(total)
        out[:,0]=ik*(-np.sum(n*h,axis=1)+total[:,7]);out[:,4]=ik*(np.sum(n*e,axis=1)+total[:,3])
        out[:,1:4]=ik*(-np.cross(n,h)+n*total[:,3,None]+e)
        out[:,5:8]=ik*(-np.cross(n,e)+n*total[:,7,None]-h)
        self.calls+=1;self.action_s+=time.perf_counter()-start;return out

    def close(self):self.faces=[];self.d=None
