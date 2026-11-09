"""Generate spectral cross-face tiles without materializing the whole symbol.

Geometry is evaluated in FP64, Green derivatives and contractions in FP32.
This approximate inner operator always requires an independent strict residual.
"""
import numpy as np
import torch
import time
from pathlib import Path
from dataclasses import replace
from galerkin_cross_fft import CrossGalerkinFFT

class GPUCrossTiles:
    def __init__(self,block,device='cuda',*,precision=32,fft64=False):
        self.block=block;self.device=device;self.dtype=torch.complex128 if precision==64 else torch.complex64;self.fft64=fft64
        self.data={}
        for name,face in (('t',block.target),('s',block.source)):
            patterns=[]
            for p in face.patterns:
                q,b=block.pattern(face,p)
                patterns.append((torch.as_tensor(q,device=device,dtype=torch.float64),torch.as_tensor(b,device=device,dtype=self.dtype)))
            self.data[name]=patterns

    def symbol_tile(self,first_t,last_t,first_s,last_s):
        b=self.block;t,s=b.target,b.source;dev=self.device
        ni,nj=last_t-first_t,last_s-first_s;nt,ns=len(t.patterns),len(s.patterns)
        def real(x):return torch.as_tensor(x,device=dev,dtype=torch.float64)
        offset=(real(t.origin-s.origin)[None,None,None,:]+real(np.arange(first_t,last_t))[:,None,None,None]*real(t.transverse)-real(np.arange(first_s,last_s))[None,:,None,None]*real(s.transverse)+real(b.delta)[None,None,:,None]*real(t.common))
        normal=real(t.normal).to(self.dtype);eye=torch.eye(3,device=dev,dtype=self.dtype)
        kernel=torch.zeros((b.nfft,2*ni*nt*6,2*nj*ns*6),device=dev,dtype=self.dtype)
        dd=torch.as_tensor(b.delta%b.nfft,device=dev)
        def derivatives(r,k):
            r2=torch.sum(r*r,dim=-1);active=r2>=1e-24
            inv=torch.where(active,torch.rsqrt(torch.where(active,r2,torch.ones_like(r2))),torch.zeros_like(r2))
            radius=torch.where(active,r2*inv,torch.ones_like(r2))
            g=torch.exp((1j*k)*radius)*inv/(4*np.pi)
            grad=(g*(1j*k-inv)*inv)[...,None]*r
            a=g*(3*inv.square()-3j*k*inv-k*k)*inv.square();c=g*(inv.square()-1j*k*inv)
            h=a[...,None,None]*r[..., :,None]*r[...,None,:]-c[...,None,None]*eye
            return g,grad,h
        def flatten(a):return a.permute(0,1,2,3,5,4,6).reshape(ni,nj,len(b.delta),21,21)
        for p,(qt,bt) in enumerate(self.data['t']):
            for q,(qs,bs) in enumerate(self.data['s']):
                r=(offset[...,None,None,:]+qt[:,None,:]-qs[None,:,:]).to(torch.float64 if self.dtype==torch.complex128 else torch.float32)
                g0,d0,h0=derivatives(r,b.ka);g1,d1,h1=derivatives(r,b.ka*b.ri)
                h=h0-h1+(b.ka**2*g0-(b.ka*b.ri)**2*g1)[...,None,None]*eye
                de=d0-b.ri**2*d1;dm=d0-d1
                ke=torch.sum(de*normal,dim=-1)[...,None,None]*eye-de[..., :,None]*normal
                km=torch.sum(dm*normal,dim=-1)[...,None,None]*eye-dm[..., :,None]*normal
                left=bt.permute(0,2,1).reshape(21,6);right=bs.permute(0,2,1).reshape(21,6)
                rotated=torch.linalg.cross(bt,normal.expand_as(bt),dim=-1).permute(0,2,1).reshape(21,6)
                blocks=(rotated.T@flatten(h)@right,left.T@flatten(ke)@right,left.T@flatten(km)@right)
                rows=(torch.arange(ni,device=dev)[:,None]*nt*6+p*6+torch.arange(6,device=dev)).ravel()
                cols=(torch.arange(nj,device=dev)[:,None]*ns*6+q*6+torch.arange(6,device=dev)).ravel()
                v=[x.permute(2,0,3,1,4).reshape(len(b.delta),ni*6,nj*6) for x in blocks]
                for rr,cc,value in ((rows,cols,1j/b.ka*v[0]),(rows+ni*nt*6,cols+nj*ns*6,-1j/b.ka*v[0]),(rows,cols+nj*ns*6,v[1]),(rows+ni*nt*6,cols,v[2])):
                    kernel[dd[:,None,None],rr[None,:,None],cc[None,None,:]]=value
        rows=np.r_[np.arange(first_t*nt*6,last_t*nt*6),b.tc+np.arange(first_t*nt*6,last_t*nt*6)]
        cols=np.r_[np.arange(first_s*ns*6,last_s*ns*6),b.sc+np.arange(first_s*ns*6,last_s*ns*6)]
        return torch.fft.fft(kernel.to(torch.complex128) if self.fft64 else kernel,dim=0),rows,cols

def compress_streamed(block,directory,rank,*,tile=4,tolerance=1e-4,device='cuda',progress=None,reference=None,precision=32):
    """One-pass bilateral sketch, with independent held-out RHS checks.

    Only spectral tiles and sketches reside on the device. Factors are written
    into mmap files one frequency batch at a time. The full symbol is never
    allocated. A holdout is a diagnostic, not an operator-norm certificate.
    """
    start=time.perf_counter();directory=Path(directory);directory.mkdir(exist_ok=True,parents=True)
    nf,nr,nc=block.nfft,2*block.tc,2*block.sc;rank=min(rank,nr,nc);ell=min(rank+64,nr)
    gen=torch.Generator(device=device);gen.manual_seed(840400+nr+nc)
    rand=lambda shape:torch.randn(shape,device=device,dtype=torch.complex64,generator=gen)
    omega=rand((nc,rank));psi=rand((nr,ell));test=rand((nc,8))
    y=torch.zeros((nf,nr,rank),device=device,dtype=torch.complex64)
    z=torch.zeros((nf,ell,nc),device=device,dtype=torch.complex64)
    held=torch.zeros((nf,nr,8),device=device,dtype=torch.complex64)
    builder=GPUCrossTiles(block,device,precision=precision);done=0
    total=((block.target.nx+tile-1)//tile)*((block.source.nx+tile-1)//tile)
    for i in range(0,block.target.nx,tile):
        for j in range(0,block.source.nx,tile):
            a,rows,cols=builder.symbol_tile(i,min(i+tile,block.target.nx),j,min(j+tile,block.source.nx))
            a=a.to(torch.complex64)
            rr=torch.as_tensor(rows,device=device);cc=torch.as_tensor(cols,device=device)
            y[:,rr,:]+=a@omega[cc];z[:,:,cc]+=psi[rr].mH@a;held[:,rr,:]+=a@test[cc]
            done+=1
        if progress:progress(done,total,time.perf_counter()-start)
    del a,builder
    left=np.lib.format.open_memmap(directory/'left.npy',mode='w+',dtype=np.complex64,shape=(nf,nr,rank))
    right=np.lib.format.open_memmap(directory/'right.npy',mode='w+',dtype=np.complex64,shape=(nf,rank,nc))
    errors=[];reference_errors=[]
    for first in range(0,nf,4):
        q,_=torch.linalg.qr(y[first:first+4],mode='reduced')
        b=torch.linalg.lstsq(psi.mH@q,z[first:first+4],driver='gels').solution
        delta=q@(b@test)-held[first:first+4]
        norm=lambda v:torch.sum(v.abs().double().square(),dim=(1,2))
        errors.extend(torch.sqrt(norm(delta)/norm(held[first:first+4])).cpu().tolist())
        if reference is not None:
            exact=torch.as_tensor(reference[first:first+4],device=device,dtype=torch.complex64)
            reference_errors.extend(torch.sqrt(norm(q@b-exact)/norm(exact)).cpu().tolist())
        left[first:first+len(q)]=q.cpu().numpy();right[first:first+len(q)]=b.cpu().numpy()
    left.flush();right.flush()
    report=dict(rank=rank,kernel_precision=precision,shape=[nf,nr,nc],factor_bytes=left.nbytes+right.nbytes,preparation_s=time.perf_counter()-start,holdout_errors=errors,holdout_max=max(errors),reference_errors=reference_errors,reference_max=max(reference_errors) if reference_errors else None,tolerance=tolerance,criterion='Eight held-out random RHS at every frequency; not an operator norm bound.')
    if max(errors)>tolerance or (reference_errors and max(reference_errors)>tolerance):
        report['status']='failed_accuracy'
    else:report['status']='complete'
    return (left,right),report

def parallel_symbol_gpu(block,device='cuda',tile=4,precision=32):
    """2D parallel-face symbol, contracting a single source transverse cell."""
    t,s=block.target,block.source
    dx=np.arange(-(s.nx-1),t.nx)
    target=replace(t,origin=t.origin-(s.nx-1)*t.transverse,nx=len(dx))
    helper=CrossGalerkinFFT(block.fixture,target,replace(s,nx=1),block.ka,block.ri)
    builder=GPUCrossTiles(helper,device,precision=precision)
    symbol=torch.zeros((block.nfx,block.nfy,block.tc,block.sc),device=device,dtype=builder.dtype)
    for first in range(0,len(dx),tile):
        last=min(first+tile,len(dx));ni=last-first
        value,_,_=builder.symbol_tile(first,last,0,1)
        value=value.reshape(block.nfy,2,ni,block.tc//2,2,1,block.sc//2).permute(2,0,1,3,4,5,6).reshape(ni,block.nfy,block.tc,block.sc)
        symbol[torch.as_tensor(dx[first:last]%block.nfx,device=device)]=value
    return torch.fft.fft(symbol,dim=0).cpu().numpy()
