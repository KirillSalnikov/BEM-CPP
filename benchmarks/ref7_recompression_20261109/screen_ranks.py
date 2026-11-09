"""CPU QR/SVD screen of existing ref7 factors; no kernel rebuild or GPU use."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import qr,svd

BASE=Path(__file__).resolve().parent
CASE=BASE.parent/'ref7_memory_20261109/geometry_ref7'
records=[]
for row in json.loads((CASE/'streamed_preparation.json').read_text())['records']:
    if row['kind']=='2D':continue
    left=np.load(CASE/row['directory']/'left.npy',mmap_mode='r')
    right=np.load(CASE/row['directory']/'right.npy',mmap_mode='r')
    for frequency in sorted(set((0,1,len(left)//8,len(left)//4,len(left)//2,len(left)-1))):
        start=time.perf_counter()
        u=np.asarray(left[frequency],np.complex128)
        v=np.asarray(right[frequency],np.complex128)
        qu,ru=qr(u,mode='economic',check_finite=False)
        qv,rv=qr(v.conj().T,mode='economic',check_finite=False)
        p,s,h=svd(ru@rv.conj().T,full_matrices=False,check_finite=False)
        tail=np.r_[np.cumsum(s[::-1]**2)[::-1],0.]
        ranks={str(tol):int(np.flatnonzero(tail<=tol**2*tail[0])[0])
               for tol in (1e-6,3e-6,1e-5,3e-5,1e-4)}
        rank=ranks['1e-05']
        a=(qu@p[:,:rank])*np.sqrt(s[:rank])
        b=np.sqrt(s[:rank,None])*(h[:rank]@qv.conj().T)
        rng=np.random.default_rng(9609+frequency)
        x=rng.normal(size=(v.shape[1],4))+1j*rng.normal(size=(v.shape[1],4))
        expected=u@(v@x)
        def encoded(value):
            scale=np.max(np.abs(value)) or 1.
            return ((value.real/scale).astype(np.float16).astype(np.float32)+
                    1j*(value.imag/scale).astype(np.float16).astype(np.float32))*scale
        output=encoded(a)@(encoded(b)@x.astype(np.complex64))
        records.append(dict(block=row['index'],frequency=frequency,old_rank=left.shape[2],ranks=ranks,
                            recompressed_action_error=float(np.linalg.norm(a@(b@x)-expected)/np.linalg.norm(expected)),
                            half_action_error=float(np.linalg.norm(output-expected)/np.linalg.norm(expected)),
                            seconds=time.perf_counter()-start))
        (BASE/'rank_screen.json').write_text(json.dumps(dict(status='running',records=records),indent=2)+'\n')
        print(json.dumps(records[-1]),flush=True)
(BASE/'rank_screen.json').write_text(json.dumps(dict(status='complete',records=records),indent=2)+'\n')
