"""Exact SO(3) averaging with covariance factors and batched angular GEMMs."""
import numpy as np
from tmatrix_reference import irreducible_blocks,clebsch,transverse_harmonics,coherency_to_mueller


def analytic_mueller_factored(kernel,order,theta_degrees,angle_batch=32,max_workspace_bytes=512<<20):
    kernel=np.asarray(kernel,complex);n=2*order*(order+2)
    if kernel.shape!=(n,n) or angle_batch<1:raise ValueError('SO3 dimensions or angle batch')
    incoming=transverse_harmonics(np.array([[0.,0.,1.]]),order)
    angle=np.deg2rad(np.atleast_1d(theta_degrees));directions=np.stack((np.sin(angle),np.zeros_like(angle),np.cos(angle)),axis=-1)
    output=transverse_harmonics(directions,order).reshape(-1,2,n)
    coherence=np.zeros((len(angle),4,4),complex);blocks=irreducible_blocks(order)
    for total in range(2*order+1):
        k=2*total+1
        if 16*n*k*2*k+16*angle_batch*k*4*k>max_workspace_bytes:raise MemoryError('factored SO3 workspace exceeds explicit budget')
        # H[output_mode, projection, incoming_pol, covariance_factor].
        # Contract the fixed incoming basis once, then share H over angles.
        h=np.zeros((n,k,2,k),complex)
        for _,l,first in blocks:
            for _,lp,second in blocks:
                if not abs(l-lp)<=total<=l+lp:continue
                g=np.zeros(k,complex);coupling=[]
                for mi,m in enumerate(range(-l,l+1)):
                    for ni,q in enumerate(range(-lp,lp+1)):
                        projection=m-q
                        if abs(projection)>total:continue
                        c=(-1)**(lp-q)*clebsch(l,m,lp,-q,total,projection)
                        g[projection+total]+=c*kernel[first.start+mi,second.start+ni]
                        if np.any(incoming[:,second.start+ni]):coupling.append((mi,projection+total,ni,c))
                g/=np.sqrt(k)
                for mi,projection,ni,c in coupling:
                    h[first.start+mi,projection]+=c*incoming[:,second.start+ni].conj()[:,None]*g[None,:]
        for begin in range(0,len(angle),angle_batch):
            stop=min(begin+angle_batch,len(angle))
            p=(output[begin:stop].reshape(-1,n)@h.reshape(n,-1)).reshape(stop-begin,2,k,2,k).transpose(0,2,1,3,4).reshape(stop-begin,k,4,k)
            coherence[begin:stop]+=np.einsum('amif,amjf->aij',p,p.conj(),optimize=True)
    return np.array([coherency_to_mueller(c) for c in coherence])
