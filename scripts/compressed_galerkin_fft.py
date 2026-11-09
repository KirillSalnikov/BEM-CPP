"""Validated frequency-wise random range compression of cross-face symbols.

Preparation currently requires the CPU dense symbols. Bounded GPU batches
avoid uploading the whole symbol, but this is NOT yet a streamed ref6 builder.
Every frequency is checked against its complete dense matrix, not just a sketch.
"""
import time
import numpy as np
from symmetric_galerkin_fft import ParallelGalerkinFFT

def compress_cross_symbols(cpu,tolerance=1e-4,device='cuda',batch=4,progress=None):
    import torch
    if not 0<tolerance<1 or batch<1:raise ValueError('Invalid compression parameters')
    factors={};records=[];start=time.perf_counter()
    generator=torch.Generator(device=device);generator.manual_seed(840190)
    for index,block in enumerate(cpu.canonical):
        if isinstance(block,ParallelGalerkinFFT):continue
        nf,nr,nc=block.symbol.shape;rank=min(96,min(nr,nc))
        while True:
            left=np.empty((nf,nr,rank),np.complex64);right=np.empty((nf,rank,nc),np.complex64);errors=[]
            for first in range(0,nf,batch):
                a=torch.as_tensor(block.symbol[first:first+batch],device=device,dtype=torch.complex64)
                omega=torch.randn((len(a),nc,rank),device=device,dtype=torch.complex64,generator=generator)
                q,_=torch.linalg.qr(a@omega,mode='reduced');b=q.mH@a
                # Full residual, FP64 accumulation: subtracting squared norms
                # would lose all significant digits at a 1e-4 relative target.
                residual=a-q@b
                err=torch.sqrt(torch.sum(residual.abs().double().square(),dim=(1,2))/torch.sum(a.abs().double().square(),dim=(1,2)))
                errors.extend(err.cpu().tolist());left[first:first+len(a)]=q.cpu().numpy();right[first:first+len(a)]=b.cpu().numpy()
            if max(errors)<=tolerance:break
            rank=min(rank+32,min(nr,nc))
            if rank*(nr+nc)>=nr*nc:
                left=right=None;break
        row=dict(index=index,shape=list(block.symbol.shape),rank=rank,compressed=left is not None,max_frobenius_error=max(errors),frequency_errors=errors)
        if left is not None:
            factors[index]=(left,right);row['bytes_fp32']=left.nbytes+right.nbytes
        else:row['bytes_fp32']=block.symbol.size*8
        records.append(row)
        if progress:progress(row)
    return factors,dict(tolerance=tolerance,preparation_s=time.perf_counter()-start,records=records)
