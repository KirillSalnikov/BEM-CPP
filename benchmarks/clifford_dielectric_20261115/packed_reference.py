"""Experimental full-space FMM audits packed in distinct symmetry sectors.

For a covariant operator A, P_s A sum_t x_t = A x_s. The full operator is
still evaluated at strict precision. Packing is conditional on an independent
covariance audit; it is not an independent arbitrary-matrix audit of each x_s.
Duplicate irreducible sectors are put into separate passes. Input currents and
output actions are stored only on representatives, bounding memory by K/G of
the unreduced batch. No production acceptance policy is changed by this module.
"""
import time
import numpy as np


def key(g,mode):
    p,l,m=mode
    return (m%g.order,(-1)**(p+l+m+1) if g.horizontal else 1)


def characters(g,mode):
    m,h=key(g,mode)
    return np.array([np.exp(2j*np.pi*m*(j%g.order)/g.order)*(h if j>=g.order else 1)
                     for j in range(g.group_size)])


def add_expanded(g,out,x,mode,scale=1.):
    """Add a sector to a full vector without allocating another full vector."""
    dst=np.asarray(out).reshape(g.d.count,g.d.basis_count,8)
    x=np.asarray(x).reshape(len(g.representatives),g.d.basis_count,8)
    for phase,t in zip(characters(g,mode),g.transforms):
        for first in range(0,len(g.representatives),4096):
            ids=g.representatives[first:first+4096]
            dst[t.image[ids]]+=(scale*phase)*np.einsum('pab,pbc,pdc->pad',
                t.basis[ids],x[first:first+4096],t.components[ids],optimize=True)


def project_modes(g,x,modes):
    """Orthogonal character projections using every panel, not selected rows."""
    x=np.asarray(x).reshape(g.d.count,g.d.basis_count,8)
    weights=np.conjugate(np.array([characters(g,m) for m in modes]))/g.group_size
    result=np.zeros((len(modes),len(g.representatives),g.d.basis_count,8),complex)
    for j,t in enumerate(g.transforms):
        for first in range(0,len(g.representatives),4096):
            ids=g.representatives[first:first+4096]
            pulled=np.einsum('pba,pbc,pcd->pad',t.basis[ids],x[t.image[ids]],t.components[ids],optimize=True)
            for k in range(len(modes)):result[k,first:first+len(ids)]+=weights[k,j]*pulled
    return result.reshape(len(modes),g.size)


def apply_packed(g,currents,modes):
    """Return reduced strict actions and explicit covariance-dependent scope."""
    if not np.isfinite(getattr(g,'operator_error',np.inf)) or g.operator_error>2e-11:
        raise ValueError('packed reference requires an independent covariance audit')
    currents=np.asarray(currents,complex)
    if currents.shape!=(len(modes),g.size) or not np.isfinite(currents).all():
        raise ValueError('packed reference current shape/finiteness')
    if not 1<=len(modes)<=32:raise ValueError('packed reference supports 1..32 candidates')
    start=time.perf_counter();packs=[];keys=[];scales=np.linalg.norm(currents,axis=1)
    result=np.zeros_like(currents)
    for index,mode in enumerate(modes):
        if scales[index]==0:continue
        sector=key(g,mode)
        for pack,used in zip(packs,keys):
            if sector not in used:pack.append(index);used.add(sector);break
        else:packs.append([index]);keys.append({sector})
    leakage=[];operator_s=0.
    for pack in packs:
        combined=np.zeros(g.d.size,complex)
        for j in pack:add_expanded(g,combined,currents[j],modes[j],1/scales[j])
        begin=time.perf_counter();value=g.d.apply_reference(combined);operator_s+=time.perf_counter()-begin
        selected=[modes[j] for j in pack];part=project_modes(g,value,selected)
        # Detect output outside the packed sum of sectors, including errors on
        # panels that are absent from the representative-row restriction.
        combined.fill(0)
        for v,mode in zip(part,selected):add_expanded(g,combined,v,mode)
        denominator=np.linalg.norm(value);combined-=value
        error=float(np.linalg.norm(combined)/denominator) if denominator else float(np.linalg.norm(combined))
        threshold=max(5e-11,100*getattr(g.d,'strict_eps',0.))
        if not np.isfinite(error) or error>threshold:
            raise RuntimeError(f'packed reference output violates covariance: {error}')
        leakage.append(error)
        for j,v in zip(pack,part):result[j]=v*scales[j]
    return result,dict(full_operator_calls=len(packs),packed_sizes=[len(p) for p in packs],
        output_leakage=leakage,operator_s=operator_s,wall_s=time.perf_counter()-start,
        covariance_error=g.operator_error,
        scope='Full strict action of normalized sums; individual actions recovered using audited operator covariance. Not independent per-column arbitrary-operator checks.')
