"""Projection ROM of a size-dependent BEM system, using current moments.

The basis uses central solution derivatives only. Reduced systems are solved
at each target size; no target T or target current is used to train the model.
Operator-series error and projection error must be validated separately.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import lu_factor, lu_solve, svd, qr, solve
from tmatrix_size_taylor import polynomial


def current_basis(currents, *, window=1., max_rank=None, seed=932, oversample=24):
    x = np.asarray(currents)
    if x.ndim != 3 or window <= 0 or not np.isfinite(x).all():
        raise ValueError('finite current coefficient tensor and positive window required')
    snapshots = np.hstack([window**j*v for j,v in enumerate(x)])
    cap = min(snapshots.shape) if max_rank is None else min(int(max_rank), *snapshots.shape)
    if cap < 1: raise ValueError('positive rank required')
    if cap+oversample >= min(snapshots.shape):
        u,s,_ = svd(snapshots,full_matrices=False,check_finite=False)
        basis = u[:,:cap]
    else:
        rng=np.random.default_rng(seed)
        omega=rng.normal(size=(snapshots.shape[1],cap+oversample))+1j*rng.normal(size=(snapshots.shape[1],cap+oversample))
        q,_=qr(snapshots@omega,mode='economic')
        # Reorthogonalized subspace iteration, not an unbounded power product.
        for _ in range(2):
            z,_=qr(snapshots.conj().T@q,mode='economic')
            q,_=qr(snapshots@z,mode='economic')
        u,s,_=svd(q.conj().T@snapshots,full_matrices=False,check_finite=False)
        basis=q@u[:,:cap]
    norm=float(np.linalg.norm(snapshots))
    errors=[]
    for j,v in enumerate(x):
        errors.append(float(np.linalg.norm(v-basis@(basis.conj().T@v))/max(np.linalg.norm(v),1e-300)))
    return basis, dict(rank=cap,snapshot_shape=list(snapshots.shape),window=window,seed=seed,
        singular_values=s.tolist(),snapshot_norm=norm,current_coefficient_projection_errors=errors,
        orthogonality_error=float(np.linalg.norm(basis.conj().T@basis-np.eye(cap))))


def separate_central_currents(current, max_rank):
    """Independent electric/magnetic spaces from one central block solve.

    Columns alternate the two spaces so every even prefix is a valid smaller
    basis. This relaxes the central-frequency relation between J and M while
    retaining their resolved spatial shapes; no frequency derivatives are used.
    """
    x=np.asarray(current);n=x.shape[0];half=n//2
    if x.ndim!=2 or n%2 or max_rank%2 or not 2<=max_rank<=2*min(half,x.shape[1]):
        raise ValueError('even system and basis ranks required')
    basis=np.zeros((n,max_rank),complex);spectra=[]
    for p,block in enumerate([x[:half],x[half:]]):
        u,s,_=svd(block,full_matrices=False,check_finite=True)
        basis[p*half:(p+1)*half,p::2]=u[:,:max_rank//2];spectra.append(s.tolist())
    return basis,dict(rank=max_rank,source='central J and M independently; no current derivatives',
        component_singular_values=spectra,
        current_projection_error=float(np.linalg.norm(x-basis@(basis.conj().T@x))/np.linalg.norm(x)),
        orthogonality_error=float(np.linalg.norm(basis.conj().T@basis-np.eye(max_rank))))


@dataclass
class ReducedBEM:
    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    phase: np.ndarray
    ka0: float

    def evaluate(self, ka, *, rank=None):
        r=self.A.shape[1] if rank is None else int(rank)
        if not np.isfinite(ka) or ka <= 0 or not 1 <= r <= self.A.shape[1]:
            raise ValueError('invalid ka or reduced rank')
        delta=ka-self.ka0
        a=polynomial(self.A[:,:r,:r],delta)
        b=polynomial(self.B[:,:r],delta)
        d=polynomial(self.D[:,:,:r],delta)
        y=solve(a,b,check_finite=True)
        t=-1j*ka/(4*np.pi)*self.phase[:,None]*(d@y)*self.phase[None,:].conj()
        return t,y


def ka_scaled_equation(model):
    """Taylor-expand (ka/ka0)*A and (ka/ka0)*B to the same degree.

    The exact BEM equation is equivalent for ka>0. This removes the known
    1/ka factor in part of A before truncation, without fitting target data.
    """
    a=model.A.copy();b=model.B.copy()
    a[1:]+=model.A[:-1]/model.ka0;b[1:]+=model.B[:-1]/model.ka0
    return ReducedBEM(a,b,model.D.copy(),model.phase.copy(),model.ka0)


def project_bem(a,b,d,basis,phase,ka0,*,testing='central_inverse',factor=None):
    if testing == 'galerkin':
        test=basis
    elif testing == 'central_inverse':
        if factor is None:factor=lu_factor(a[0])
        test=lu_solve(factor,basis,trans=2)
    else:raise ValueError('unknown test basis')
    aq=np.array([v@basis for v in a])
    model=ReducedBEM(np.array([test.conj().T@v for v in aq]),
                     np.array([test.conj().T@v for v in b]),
                     np.array([v@basis for v in d]),np.asarray(phase),float(ka0))
    return model,test,aq
