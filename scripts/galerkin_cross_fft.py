"""Experimental finite-face Muller action in local BDM1 edge moments.

Exact quadrature contraction and zero-padded FFT along a common lattice axis.
Optional frequency-wise SVD is an approximation, intended first for mixed
iterations/preconditioning with the original strict residual retained.
No global prism incidence map, mass matrix or Duffy correction is replaced.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.fft import next_fast_len, fft, ifft
from scipy.linalg import svd

@dataclass
class Face:
    origin: np.ndarray
    transverse: np.ndarray
    common: np.ndarray
    nx: int
    ny: int
    patterns: tuple = (0,1,2,3)

    @property
    def normal(self):
        n=np.cross(self.transverse,self.common)
        return n/np.linalg.norm(n)

def cross_matrix(n):
    x,y,z=n
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def derivatives(r,k):
    """Independent analytic point Green derivatives, all complex128/float64."""
    r2=np.einsum('...a,...a->...',r,r)
    active=r2>=1e-24
    inv=np.where(active,1/np.sqrt(np.where(active,r2,1.)),0.)
    radius=np.where(active,r2*inv,1.)
    g=np.where(active,np.exp(1j*k*radius)*inv/(4*np.pi),0.)
    first=g*(1j*k-inv)*inv
    a=g*(3*inv**2-3j*k*inv-k*k)*inv**2
    b=g*(inv**2-1j*k*inv)
    h=a[...,None,None]*r[..., :,None]*r[...,None,:]-b[...,None,None]*np.eye(3)
    return g,first[...,None]*r,h

class CrossGalerkinFFT:
    def __init__(self,fixture,target,source,ka,ri):
        if not np.allclose(target.common,source.common,rtol=0,atol=1e-15):
            raise ValueError('Common lattice steps must agree')
        if min(target.nx,target.ny,source.nx,source.ny)<=0 or ka<=0:
            raise ValueError('Invalid finite patch')
        if min(np.linalg.norm(np.cross(f.transverse,f.common)) for f in (target,source))<=0:
            raise ValueError('Degenerate lattice')
        self.target,self.source,self.ka,self.ri=target,source,float(ka),complex(ri)
        self.nfft=next_fast_len(target.ny+source.ny-1)
        self.delta=np.arange(-(source.ny-1),target.ny)
        self.patterns=fixture['patterns']
        self.tc=target.nx*len(target.patterns)*6
        self.sc=source.nx*len(source.patterns)*6
        self.symbol=None;self.point_symbol=None;self.spatial=None

    def pattern(self,face,index):
        data=self.patterns[index]
        axes=np.stack((face.transverse,face.common))
        return np.asarray(data['points'])@axes,np.asarray(data['weighted_basis'])@axes

    def projected_blocks(self,pt,ps,deltas=None):
        t,s=self.target,self.source
        deltas=self.delta if deltas is None else np.asarray(deltas)
        qt,bt=self.pattern(t,pt);qs,bs=self.pattern(s,ps)
        offset=(np.asarray(t.origin)-np.asarray(s.origin)+
            np.arange(t.nx)[:,None,None,None]*t.transverse-
            np.arange(s.nx)[None,:,None,None]*s.transverse+
            deltas[None,None,:,None]*t.common)
        r=offset[...,None,None,:]+qt[:,None,:]-qs[None,:,:]
        g0,grad0,h0=derivatives(r,self.ka)
        g1,grad1,h1=derivatives(r,self.ka*self.ri)
        h=h0-h1+(self.ka**2*g0-(self.ka*self.ri)**2*g1)[...,None,None]*np.eye(3)
        n=t.normal
        grad_e=grad0-self.ri**2*grad1;grad_m=grad0-grad1
        ke=np.einsum('...a,a->...',grad_e,n)[...,None,None]*np.eye(3)-grad_e[..., :,None]*n
        km=np.einsum('...a,a->...',grad_m,n)[...,None,None]*np.eye(3)-grad_m[..., :,None]*n
        def flatten(a):return a.transpose(0,1,2,3,5,4,6).reshape(t.nx,s.nx,len(deltas),21,21)
        p=bt.transpose(0,2,1).reshape(21,6);q=bs.transpose(0,2,1).reshape(21,6)
        rotated=np.cross(bt,n).transpose(0,2,1).reshape(21,6)
        projected=(rotated.T@flatten(h)@q,p.T@flatten(ke)@q,p.T@flatten(km)@q)
        # Point representation uses unrotated test vectors; rotate its K1 field.
        raw=(flatten(np.einsum('ab,...bc->...ac',cross_matrix(n),h)),flatten(ke),flatten(km))
        return projected,raw

    def projection(self,face):
        local=face.nx*len(face.patterns)*6;points=face.nx*len(face.patterns)*21
        matrix=np.zeros((2*points,2*local))
        for i in range(face.nx):
            for p,kind in enumerate(face.patterns):
                _,basis=self.pattern(face,kind);block=basis.transpose(0,2,1).reshape(21,6)
                a=(i*len(face.patterns)+p)*21;b=(i*len(face.patterns)+p)*6
                for current in range(2):matrix[current*points+a:current*points+a+21,current*local+b:current*local+b+6]=block
        return matrix

    def build(self,point_reference=False,retain_spatial=False):
        start=time.perf_counter();t,s=self.target,self.source
        kernel=np.zeros((self.nfft,2*self.tc,2*self.sc),complex)
        tq=t.nx*len(t.patterns)*21;sq=s.nx*len(s.patterns)*21
        raw=np.zeros((self.nfft,2*tq,2*sq),complex) if point_reference else None
        def put(array,a,b,n,m,blocks):
            for current in range(2):
                factor=(1 if current==0 else -1)*1j/self.ka
                array[self.delta%self.nfft,a+current*n:a+current*n+blocks[0].shape[-2],b+current*m:b+current*m+blocks[0].shape[-1]]=factor*blocks[0]
            array[self.delta%self.nfft,a:a+blocks[1].shape[-2],m+b:m+b+blocks[1].shape[-1]]=blocks[1]
            array[self.delta%self.nfft,n+a:n+a+blocks[2].shape[-2],b:b+blocks[2].shape[-1]]=blocks[2]
        for p,pt in enumerate(t.patterns):
            for q,ps in enumerate(s.patterns):
                projected,point=self.projected_blocks(pt,ps)
                for i in range(t.nx):
                    for j in range(s.nx):
                        a=(i*len(t.patterns)+p);b=(j*len(s.patterns)+q)
                        put(kernel,a*6,b*6,self.tc,self.sc,tuple(x[i,j] for x in projected))
                        if raw is not None:put(raw,a*21,b*21,tq,sq,tuple(x[i,j] for x in point))
        self.spatial=kernel if retain_spatial else None
        self.symbol=fft(kernel,axis=0)
        if raw is not None:self.point_symbol=fft(raw,axis=0)
        self.preparation_s=time.perf_counter()-start
        return self

    def apply(self,coefficients,mask_target=None,mask_source=None):
        x=np.asarray(coefficients,complex).reshape(2*self.sc,self.source.ny)
        if mask_source is not None:x=x*mask_source
        spectrum=fft(x,n=self.nfft,axis=1).T
        yhat=np.einsum('fij,fj->fi',self.symbol,spectrum)
        y=ifft(yhat,axis=0)[:self.target.ny].T
        return y if mask_target is None else y*mask_target

    def apply_point(self,x):
        if self.point_symbol is None:raise ValueError('Point reference not built')
        bs=np.stack([self.pattern(self.source,p)[1] for p in self.source.patterns])
        bt=np.stack([self.pattern(self.target,p)[1] for p in self.target.patterns])
        shaped=x.reshape(2,self.source.nx,len(self.source.patterns),6,self.source.ny)
        q=np.einsum('pqda,cxpds->cxpqas',bs,shaped).reshape(-1,self.source.ny)
        spectrum=fft(q,n=self.nfft,axis=1).T
        yhat=np.einsum('fij,fj->fi',self.point_symbol,spectrum)
        field=ifft(yhat,axis=0)[:self.target.ny].T.reshape(2,self.target.nx,len(self.target.patterns),7,3,self.target.ny)
        return np.einsum('pqda,cxpqas->cxpds',bt,field).reshape(2*self.tc,self.target.ny)

    def apply_direct(self,x):
        if self.spatial is None:raise ValueError('Spatial reference not retained')
        out=np.zeros((2*self.tc,self.target.ny),complex)
        for jt in range(self.target.ny):
            for js in range(self.source.ny):out[:,jt]+=self.spatial[(jt-js)%self.nfft]@x[:,js]
        return out

    def factorize(self):
        start=time.perf_counter()
        self.svds=[svd(a,full_matrices=False,check_finite=False,lapack_driver='gesdd') for a in self.symbol]
        self.factorization_s=time.perf_counter()-start

    def compressed(self,tolerance):
        if not 0<tolerance<1:raise ValueError('Invalid SVD tolerance')
        if not hasattr(self,'svds'):self.factorize()
        factors=[];ranks=[];entries=0
        for f,(u,s,vh) in enumerate(self.svds):
            rank=int(np.count_nonzero(s>tolerance*s[0])) if s[0] else 0
            ranks.append(rank)
            if rank*(u.shape[0]+vh.shape[1])>=u.shape[0]*vh.shape[1]:
                factors.append((self.symbol[f],None));entries+=self.symbol[f].size
            else:
                left=u[:,:rank]*s[:rank];right=vh[:rank,:]
                factors.append((left,right));entries+=left.size+right.size
        info=dict(tolerance=tolerance,ranks=ranks,stored_bytes=entries*16,
                  dense_bytes=self.symbol.nbytes,storage_ratio=self.symbol.nbytes/max(1,entries*16))
        return factors,info

    def apply_compressed(self,x,factors):
        spectrum=fft(x,n=self.nfft,axis=1).T
        yhat=np.empty((self.nfft,2*self.tc),complex)
        for f,(u,v) in enumerate(factors):yhat[f]=u@spectrum[f] if v is None else u@(v@spectrum[f])
        return ifft(yhat,axis=0)[:self.target.ny].T
