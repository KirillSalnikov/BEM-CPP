"""Single-C6-mode restriction of equivalent face-pair applications."""
import numpy as np


def rotation_tables(cpu):
    n = cpu.geometry.N
    perm = np.full(n, -1, np.int32)
    sign = np.zeros(n, np.int8)
    plans = []
    for index in range(len(cpu.canonical)):
        aliases = [a for k,a in cpu.aliases if k == index]
        families = {}
        for j,a in enumerate(aliases):
            if a["vertical"] is None or a["horizontal"] is None:
                raise ValueError("rotation folding requires the verified regular prism face maps")
            family = families.setdefault((a["vertical"],a["horizontal"]), [])
            family.append(j)
        reps = []
        which = np.empty(len(aliases),np.int64)
        shifts = np.zeros(len(aliases),np.int64)
        for family in families.values():
            family.sort(key=lambda j: aliases[j]["rotation"])
            complete = [aliases[j]["rotation"] for j in family] == list(range(6))
            for j in family:
                if not complete or j == family[0]:
                    reps.append(j)
                which[j] = len(reps)-1
                if not complete:
                    continue
                rot = aliases[j]["rotation"]
                shifts[j] = rot
                a,b = aliases[j],aliases[family[(rot+1)%6]]
                sa,sb = a["source"],b["source"]
                oa,ob = np.argsort(sa["local"]),np.argsort(sb["local"])
                if not np.array_equal(sa["local"][oa],sb["local"][ob]):
                    raise ValueError("rotated local support mismatch")
                src,dst = sa["global_"][oa],sb["global_"][ob]
                signs = (a["source_sign"][oa]*b["source_sign"][ob]).astype(np.int8)
                if np.any((perm[src]>=0)&((perm[src]!=dst)|(sign[src]!=signs))):
                    raise ValueError("inconsistent edge rotation")
                perm[src],sign[src] = dst,signs
        plans.append((reps,which,shifts))
    ids = np.arange(n,dtype=np.int32)
    if (perm<0).any() or np.unique(perm).size != n or not np.all(np.abs(sign)==1):
        raise ValueError("incomplete signed edge rotation")
    p = ids
    product = np.ones(n,np.int8)
    for _ in range(6):
        product *= sign[p]
        p = perm[p]
    if not np.array_equal(ids,p) or np.any(product!=1):
        raise ValueError("rotation does not close")
    return perm,sign,plans


def local_rotation_rows(perm,sign):
    n=len(perm)//2
    if not np.array_equal(perm[:n]+n,perm[n:]) or not np.array_equal(sign[:n],sign[n:]):
        raise ValueError("current rotations differ")
    p=perm[:n]
    ids=np.arange(n,dtype=np.int32)
    owner=ids.copy()
    step=ids
    for _ in range(5):
        step=p[step]
        np.minimum(owner,step,out=owner)
    reps=np.flatnonzero(ids==owner).astype(np.int32)
    if 6*len(reps)!=n:
        raise ValueError("short rotation orbit in local matrix")
    phase_id=np.empty(n,np.uint8)
    step=reps
    direction=np.ones_like(reps,dtype=np.int8)
    for j in range(6):
        phase_id[step]=j+6*(direction<0)
        direction*=sign[step]
        step=p[step]
    roots=np.exp(2j*np.pi*np.arange(6)[:,None]*np.arange(6)/6)
    phases=np.ascontiguousarray(np.concatenate((roots,-roots),axis=1),np.complex64)
    return reps,owner,phase_id,phases


class SingleRotationMode:
    def __init__(self,cpu,device,dtype):
        import torch
        perm,sign,plans = rotation_tables(cpu)
        self.perm = torch.as_tensor(perm,device=device)
        self.sign = torch.as_tensor(sign,device=device)
        roots = np.exp(2j*np.pi*np.arange(6)/6)
        self.roots = torch.as_tensor(roots,device=device,dtype=dtype)
        self.plans = []
        self.bytes = self.perm.numel()*self.perm.element_size()+self.sign.numel()*self.sign.element_size()
        for reps,which,shifts in plans:
            self.plans.append(None if len(reps)==len(which) else (
                torch.tensor(reps,device=device),torch.as_tensor(which,device=device),
                torch.as_tensor(roots[(np.arange(6)[:,None]*shifts)%6],device=device,dtype=dtype)))
        self.bytes += self.roots.numel()*self.roots.element_size()
        self.bytes += sum(t.numel()*t.element_size() for plan in self.plans if plan for t in plan)
        self.tolerance = 2e-7 if dtype==torch.complex64 else 2e-13
        self.hits = self.fallbacks = 0

    def detect(self,x):
        import torch
        # Infer the character, then check its residual explicitly. Test the
        # entire incident current, including edge moments shared by two faces.
        rotated = x[self.perm]*self.sign
        correlation = torch.vdot(x,rotated)
        mode = (self.roots.conj()*correlation).real.argmax()
        defect = torch.linalg.vector_norm(rotated-self.roots[mode]*x)
        norm = torch.linalg.vector_norm(x)
        valid = (norm>0) & torch.isfinite(norm) & torch.isfinite(correlation)
        selected = int(torch.where(valid & (defect<=self.tolerance*norm),mode,-1).item())
        if selected<0:
            self.fallbacks+=1
        else:
            self.hits+=1
        return selected
