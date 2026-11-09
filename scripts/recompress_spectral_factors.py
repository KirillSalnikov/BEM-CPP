"""Reduce existing frequency factors using two thin QR factorizations and SVD.

Only the small rank-by-rank core is decomposed. This changes an approximate
inner action; original-operator checks remain required. The reported error
concerns the supplied factors, not their error relative to the BEM kernel.
"""
import json
import math
from pathlib import Path
import tempfile
import time
import numpy as np
from scipy.linalg import qr,svd


def recompress_frequency(left,right,tolerance):
    """Balanced complex64 factors with a controlled relative Frobenius tail."""
    u=np.asarray(left,np.complex128);v=np.asarray(right,np.complex128)
    if not np.isfinite(u).all() or not np.isfinite(v).all():raise ValueError('Nonfinite factors')
    qu,ru=qr(u,mode='economic',check_finite=False)
    qv,rv=qr(v.conj().T,mode='economic',check_finite=False)
    p,s,h=svd(ru@rv.conj().T,full_matrices=False,check_finite=False)
    tail=np.r_[np.cumsum(s[::-1]**2)[::-1],0.];norm=float(np.sqrt(tail[0]))
    rank=max(1,int(np.flatnonzero(tail<=(.9*tolerance*norm)**2)[0]))
    while True:
        a=(qu@p[:,:rank])*np.sqrt(s[:rank])
        b=np.sqrt(s[:rank,None])*(h[:rank]@qv.conj().T)
        ac=a.astype(np.complex64);bc=b.astype(np.complex64)
        da=float(np.linalg.norm(ac.astype(np.complex128)-a))
        db=float(np.linalg.norm(bc.astype(np.complex128)-b))
        scale=float(np.sqrt(s[0]))
        rounding=(da*scale+(scale+da)*db)/norm if norm else 0.
        truncation=float(np.sqrt(tail[rank])/norm) if norm else 0.
        if truncation+rounding<=tolerance:break
        rank+=1
        if rank>len(s):raise ValueError('Requested tolerance is below complex64 factor accuracy')
    return ac,bc,dict(rank=rank,tail_relative_error=truncation,
        fp32_rounding_bound=rounding,combined_relative_error_bound=truncation+rounding)


def recompress_factors(left,right,directory,*,tolerance=1e-5,rank_multiple=8,progress=None):
    if not np.isfinite(tolerance) or not 0<tolerance<1 or rank_multiple<1:
        raise ValueError('Invalid recompression tolerance/alignment')
    if left.ndim!=3 or right.ndim!=3 or left.shape[0]!=right.shape[0] or left.shape[2]!=right.shape[1]:
        raise ValueError('Incompatible factor shapes')
    nf,nr,old_rank=left.shape;nc=right.shape[2]
    if min(nf,nr,nc,old_rank)<1 or old_rank>min(nr,nc):raise ValueError('Invalid factor dimensions')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    def identity(value):
        result=dict(shape=list(value.shape),dtype=str(value.dtype))
        if isinstance(value,np.memmap):
            source=Path(value.filename).resolve();stat=source.stat()
            result.update(path=str(source),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                          ctime_ns=stat.st_ctime_ns,inode=stat.st_ino)
        return result
    signature=dict(left=identity(left),right=identity(right),tolerance=tolerance,rank_multiple=rank_multiple)
    report_path=directory/'recompression.json'
    if isinstance(left,np.memmap) and isinstance(right,np.memmap) and report_path.exists():
        report=json.loads(report_path.read_text())
        if report.get('status')=='complete' and report.get('signature')==signature:
            a=np.load(directory/'left.npy',mmap_mode='r');b=np.load(directory/'right.npy',mmap_mode='r')
            if a.shape==(nf,nr,report['rank']) and b.shape==(nf,report['rank'],nc):return (a,b),report
    started=time.perf_counter();records=[]
    with tempfile.TemporaryDirectory(prefix='factor-packing-',dir=directory) as tmp:
        tmp=Path(tmp)
        with (tmp/'left.bin').open('wb') as la,(tmp/'right.bin').open('wb') as rb:
            for frequency in range(nf):
                begin=time.perf_counter()
                a,b,record=recompress_frequency(left[frequency],right[frequency],tolerance)
                a.tofile(la);b.tofile(rb)
                record.update(frequency=frequency,seconds=time.perf_counter()-begin);records.append(record)
                if progress:progress(record,nf)
        rank=min(old_rank,int(math.ceil(max(r['rank'] for r in records)/rank_multiple))*rank_multiple)
        a=np.lib.format.open_memmap(directory/'left.tmp.npy',mode='w+',dtype=np.complex64,shape=(nf,nr,rank))
        b=np.lib.format.open_memmap(directory/'right.tmp.npy',mode='w+',dtype=np.complex64,shape=(nf,rank,nc))
        la=np.memmap(tmp/'left.bin',dtype=np.complex64,mode='r')
        rb=np.memmap(tmp/'right.bin',dtype=np.complex64,mode='r');lo=ro=0
        for frequency,record in enumerate(records):
            count=record['rank'];a[frequency]=0;b[frequency]=0
            a[frequency,:,:count]=la[lo:lo+nr*count].reshape(nr,count)
            b[frequency,:count]=rb[ro:ro+count*nc].reshape(count,nc)
            lo+=nr*count;ro+=count*nc
        assert lo==len(la) and ro==len(rb)
        a.flush();b.flush();del a,b,la,rb
        (directory/'left.tmp.npy').replace(directory/'left.npy')
        (directory/'right.tmp.npy').replace(directory/'right.npy')
    report=dict(status='complete',signature=signature,old_rank=old_rank,rank=rank,
        old_bytes=left.nbytes+right.nbytes,new_bytes=nf*rank*(nr+nc)*8,
        maximum_relative_error_bound=max(r['combined_relative_error_bound'] for r in records),
        seconds=time.perf_counter()-started,records=records,
        criterion='Relative Frobenius SVD tail plus complex64 factor rounding bound; excludes FP64 roundoff, original factorization error and subsequent FP16 encoding.')
    temp=report_path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(report_path)
    return (np.load(directory/'left.npy',mmap_mode='r'),np.load(directory/'right.npy',mmap_mode='r')),report
