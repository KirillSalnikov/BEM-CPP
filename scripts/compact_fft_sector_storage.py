"""Store a D6h compact inverse in C6 x horizontal sector coordinates.

The vertical mirror is already respected by paired blocks and by the ± parity
blocks at m=0,3. This is an algebraic storage conversion, not a new fit.
"""
import numpy as np
from scipy.sparse import csc_matrix,hstack
from scipy.linalg import lu_factor,lu_solve

def sector_transform(group):
    n=group.n;idx=np.arange(n);sgn=np.ones(n,int);maps=[]
    rotation,rs=group.maps[0];horizontal,hs=group.maps[1]
    for j in range(6):
        maps.append((idx.copy(),sgn.copy()));sgn=rs[idx]*sgn;idx=rotation[idx]
    maps += [(horizontal[i],hs[i]*s) for i,s in maps.copy()]
    visited=np.zeros(n,bool);rows=[[] for _ in range(12)];cols=[[] for _ in range(12)];vals=[[] for _ in range(12)];counts=np.zeros(12,int)
    for seed in range(n):
        if visited[seed]:continue
        orbit=np.unique([i[seed] for i,_ in maps]);visited[orbit]=True
        for h in [1,-1]:
            for m in range(6):
                v=np.zeros(len(orbit),complex)
                for j,(indices,signs) in enumerate(maps):
                    k=np.searchsorted(orbit,indices[seed]);v[k]+=signs[seed]*np.exp(-2j*np.pi*m*(j%6)/6)*(h if j>=6 else 1)/12
                length=np.linalg.norm(v)
                if length<1e-10:continue
                sector=(0 if h==1 else 6)+m;coordinate=counts[sector];counts[sector]+=1
                rows[sector].extend(orbit);cols[sector].extend([coordinate]*len(orbit));vals[sector].extend(v/length)
    blocks=[csc_matrix((v,(r,c)),shape=(n,int(count))) for v,r,c,count in zip(vals,rows,cols,counts)]
    U=hstack(blocks,format='csr');assert U.shape==(n,n)
    return U,np.r_[0,np.cumsum(counts)]

class SectorCompactInverse:
    def __init__(self,blocks,group,rank):
        if rank%16:raise ValueError('rank must preserve symmetry multiplicities')
        self.U,self.offsets=sector_transform(group);self.blocks=[];keep=rank//16
        for b in blocks:
            Q=b['Q'][:,:keep];C=b['C'][:keep]
            variants=[(b['m'],Q,C)]
            if b['m'] in [1,2]:variants.append((6-b['m'],group.transform(Q,2),group.transform(C.conj().T,2).conj().T))
            for m,Q,C in variants:
                sector=(0 if b['h']==1 else 6)+m;sl=slice(self.offsets[sector],self.offsets[sector+1]);u=self.U[:,sl]
                q=np.asarray(u.conj().T@Q);c=np.asarray((u.conj().T@C.conj().T).conj().T)
                if np.linalg.norm(u@q-Q)/np.linalg.norm(Q)>1e-10:raise ValueError('basis is not in its declared sector')
                self.blocks.append((sl,q,c,lu_factor(np.eye(keep)+c@q,check_finite=False)))
    @property
    def bytes(self):
        return self.U.data.nbytes+self.U.indices.nbytes+self.U.indptr.nbytes+self.offsets.nbytes+sum(q.nbytes+c.nbytes+lu[0].nbytes+lu[1].nbytes for _,q,c,lu in self.blocks)
    def apply(self,x):
        z=self.U.conj().T@x;y=z.copy()
        for sl,q,c,lu in self.blocks:y[sl]-=q@lu_solve(lu,c@z[sl],check_finite=False)
        return self.U@y
