"""Operator-only compact inverse with the full D6h symmetry of a prism.

Callbacks act on matrices whose columns are vectors. Neither right-hand sides
nor reference solutions enter construction. This module does not alter A.
"""
import time
import numpy as np
from scipy.linalg import qr,svd,lu_factor,lu_solve

class D6hMaps:
    def __init__(self,maps):
        self.maps=np.asarray(maps);self.n=self.maps.shape[-1]
        if self.maps.shape!=(3,2,self.n):raise ValueError('three signed generators required')
        for indices,signs in self.maps:
            if not np.array_equal(np.sort(indices),np.arange(self.n)) or not np.all(abs(signs)==1):raise ValueError('invalid signed permutation')
    def transform(self,x,g):
        indices,signs=self.maps[g];y=np.empty_like(x);y[indices]=signs[:,None]*x;return y
    def project(self,x,h,m,v=0):
        x=(x+h*self.transform(x,1))/2
        y=np.zeros_like(x);r=x
        for j in range(6):
            y+=np.exp(-2j*np.pi*m*j/6)*r/6;r=self.transform(r,0)
        if v:y=(y+v*self.transform(y,2))/2
        return y

class CompactInverse:
    def __init__(self,Q,C):
        self.Q=Q;self.C=C
        H=np.eye(Q.shape[1])+C@Q
        self.small_condition=float(np.linalg.cond(H));self.lu=lu_factor(H,check_finite=False)
    @property
    def bytes(self):return self.Q.nbytes+self.C.nbytes+self.lu[0].nbytes+self.lu[1].nbytes
    def apply(self,x):return x-self.Q@lu_solve(self.lu,self.C@x,check_finite=False)

def build_blocks(K,KH,group,directory,max_rank=512,oversample=8,seed=2026110703,progress=print):
    if max_rank%16:raise ValueError('ranks must preserve D6h multiplicities')
    keep=max_rank//16;sample=keep+oversample;blocks=[]
    for h in [1,-1]:
        for m in [0,1,2,3]:
            for v in ([1,-1] if m in [0,3] else [0]):
                rng=np.random.default_rng(np.random.SeedSequence([seed,h+1,m,v+1]))
                path=directory/f'irrep_h{h}_m{m}_v{v}.npz'
                start=time.perf_counter()
                if path.exists():
                    old=np.load(path);q=old['Q'];c=old['C'];elapsed=float(old['seconds']);singular=old['singular']
                else:
                    omega=(rng.normal(size=(group.n,sample))+1j*rng.normal(size=(group.n,sample)))/np.sqrt(2*group.n)
                    omega=group.project(omega,h,m,v)
                    y=group.project(K(omega),h,m,v)
                    q,_=qr(y,mode='economic',check_finite=False)
                    c=group.project(KH(q),h,m,v).conj().T
                    u,singular,vh=svd(c,full_matrices=False,check_finite=False)
                    q=(q@u)[:,:keep];c=singular[:keep,None]*vh[:keep]
                    elapsed=time.perf_counter()-start
                    np.savez_compressed(path,Q=q,C=c,seconds=elapsed,singular=singular)
                blocks.append({'h':h,'m':m,'v':v,'Q':q,'C':c,'seconds':elapsed,'singular':singular})
                progress(f'irrep h={h} m={m} v={v}: {elapsed:.3f} s',flush=True)
    return blocks

def model_for_rank(blocks,group,rank):
    if rank%16:raise ValueError('rank must preserve D6h multiplicities')
    r=rank//16;q=[];c=[]
    for b in blocks:
        Q=b['Q'][:,:r];C=b['C'][:r];q.append(Q);c.append(C)
        if b['m'] in [1,2]:
            q.append(group.transform(Q,2));c.append(group.transform(C.conj().T,2).conj().T)
    return CompactInverse(np.concatenate(q,axis=1),np.concatenate(c,axis=0))
