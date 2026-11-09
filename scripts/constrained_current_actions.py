"""Constrained coherent-field inversion from forward and adjoint actions.

No full radiation matrix or selected BEM rows are assembled here. A thin
current basis spans sqrt(G) X, G^-1/2 C*, and G^-1/2 R*. The dense research
inverse is evaluated only in that basis. With an exact spanning basis its
particular, prior and complementary currents are the same as in the full
space; this is not a prior-only approximation. Numerical basis deflation is
checked separately from the physical radiation regularization.

Storage is O(N*(prior_columns+constraint_rows+field_channels)), not constant
in angular resolution. Setup requires adjoint as well as forward actions.
This module returns initial guesses, never accepted BEM solutions.
"""
import time
import numpy as np
from scipy.linalg import svd, qr
from constrained_current_inverse import ConstrainedCurrentInverse, _matrix, _positive_weights


class ActionConstrainedCurrentInverse:
    def __init__(self, dofs, field_channels, constraint_rows, radiation, radiation_adjoint,
                 constraints, constraints_adjoint, prior, *, field_weights=None, current_metric=None,
                 block_size=8, basis_rtol=1e-12, adjoint_tolerance=1e-10,
                 memory_limit_bytes=4*1024**3, **inverse_options):
        started = time.perf_counter()
        values = (dofs, field_channels, constraint_rows, block_size, memory_limit_bytes)
        if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) for v in values):
            raise ValueError('dimensions, block size and memory limit must be integers')
        if dofs < 1 or field_channels < 1 or constraint_rows < 0 or block_size < 1 or memory_limit_bytes < 1:
            raise ValueError('invalid action-inverse dimensions or memory limit')
        if not np.isfinite(basis_rtol) or not 0 < basis_rtol < 1e-4:
            raise ValueError('basis_rtol must be finite and in (0, 1e-4)')
        if not np.isfinite(adjoint_tolerance) or not 0 < adjoint_tolerance < 1:
            raise ValueError('invalid adjoint tolerance')
        x = _matrix(prior, 'prior')
        if x.shape[0] != dofs:
            raise ValueError('prior current dimension mismatch')
        self.sqrt_g = np.sqrt(_positive_weights(current_metric, dofs, 'current metric'))
        weights = _positive_weights(field_weights, field_channels, 'field weights')
        cap = min(dofs, x.shape[1]+field_channels+constraint_rows)
        # Conservative budget for owned basis, prior, block temporaries,
        # reduced matrices and the original inverse's SVD work arrays.
        estimate = 16*(dofs*(cap+2*x.shape[1]+8*block_size) +
                       12*(field_channels+constraint_rows+cap)*cap)
        if estimate > memory_limit_bytes:
            raise MemoryError(f'action inverse needs an estimated {estimate} bytes; budget {memory_limit_bytes}')
        self.counts = {name:dict(calls=0, columns=0, seconds=0.) for name in ('R', 'RH', 'C', 'CH')}
        self._actions = dict(R=radiation, RH=radiation_adjoint, C=constraints, CH=constraints_adjoint)
        self._dimensions = dict(R=(field_channels,dofs), RH=(dofs,field_channels),
                                C=(constraint_rows,dofs), CH=(dofs,constraint_rows))
        def action(name, value):
            value = np.asarray(value, complex)
            rows, cols = self._dimensions[name]
            if value.ndim != 2 or value.shape[0] != cols or not np.isfinite(value).all():
                raise ValueError('invalid input to '+name)
            before = time.perf_counter()
            result = np.asarray(self._actions[name](value), complex)
            self.counts[name]['calls'] += 1
            self.counts[name]['columns'] += value.shape[1]
            self.counts[name]['seconds'] += time.perf_counter()-before
            if result.shape != (rows, value.shape[1]) or not np.isfinite(result).all():
                raise ValueError('invalid action result from '+name)
            return result
        self._apply = action
        # Complex bilinear tests catch accidental transposes and inconsistent
        # current/equation metrics before building the inverse.
        rng = np.random.default_rng(12012026)
        probe = rng.normal(size=(dofs,2))+1j*rng.normal(size=(dofs,2))
        adjoint_errors = {}
        for name, adj in [('R','RH'), ('C','CH')]:
            rows = self._dimensions[name][0]
            if not rows: continue
            dual = rng.normal(size=(rows,2))+1j*rng.normal(size=(rows,2))
            forward, backward = action(name,probe), action(adj,dual)
            left = np.sum(dual.conj()*forward,axis=0)
            right = np.sum(backward.conj()*probe,axis=0)
            scale = np.maximum(np.linalg.norm(dual,axis=0)*np.linalg.norm(forward,axis=0),
                               np.linalg.norm(backward,axis=0)*np.linalg.norm(probe,axis=0))
            error = float(np.max(np.abs(left-right)/np.maximum(scale,np.finfo(float).tiny)))
            if error > adjoint_tolerance:
                raise ValueError(f'{name} adjoint consistency failed: {error:g}')
            adjoint_errors[name] = error
        storage = np.empty((dofs,cap),complex,order='F')
        rank = 0
        dropped = 0
        def append(block):
            nonlocal rank, dropped
            norms = np.linalg.norm(block,axis=0)
            if not np.isfinite(norms).all(): raise FloatingPointError('nonfinite basis column norm')
            active = norms > np.finfo(float).tiny
            dropped += int((~active).sum())
            if not active.any(): return
            value = np.array(block[:,active]/norms[active],order='F')
            q = storage[:,:rank]
            for _ in range(2): value -= q@(q.conj().T@value)
            u, s, _ = svd(value,full_matrices=False,check_finite=False)
            keep = s > basis_rtol
            if rank == cap: keep[:] = False
            added = int(keep.sum())
            dropped += value.shape[1]-added
            if not added: return
            if rank+added > cap: raise FloatingPointError('current span exceeds its mathematical dimension')
            value = u[:,keep]
            for _ in range(2): value -= q@(q.conj().T@value)
            value, _ = qr(value,mode='economic',check_finite=False)
            storage[:,rank:rank+added] = value
            rank += added
        for first in range(0,x.shape[1],block_size):
            append(self.sqrt_g[:,None]*x[:,first:first+block_size])
        for name, size in [('CH',constraint_rows), ('RH',field_channels)]:
            for first in range(0,size,block_size):
                indices = np.arange(first,min(first+block_size,size))
                basis = np.zeros((size,len(indices)),complex)
                basis[indices,np.arange(len(indices))] = 1.
                append(action(name,basis)/self.sqrt_g[:,None])
        self.basis = storage[:,:rank]
        defect = float(np.linalg.norm(self.basis.conj().T@self.basis-np.eye(rank)))
        if not np.isfinite(defect) or defect > 1e-10*max(1,np.sqrt(rank)):
            raise FloatingPointError('reduced current basis lost orthogonality')
        reduced_r = np.empty((field_channels,rank),complex)
        reduced_c = np.empty((constraint_rows,rank),complex)
        for first in range(0,rank,block_size):
            physical = self.basis[:,first:first+block_size]/self.sqrt_g[:,None]
            reduced_r[:,first:first+block_size] = action('R',physical)
            if constraint_rows: reduced_c[:,first:first+block_size] = action('C',physical)
        projected_prior = self.basis.conj().T@(self.sqrt_g[:,None]*x)
        # If all actions and priors vanish, a one-dimensional zero model
        # preserves the original dense inverse's boundary consistency checks.
        model_r, model_c, model_x = reduced_r,reduced_c,projected_prior
        if not rank:
            model_r=np.zeros((field_channels,1),complex)
            model_c=np.zeros((constraint_rows,1),complex)
            model_x=np.zeros((1,x.shape[1]),complex)
        self.inverse = ConstrainedCurrentInverse(model_r,model_c,model_x,field_weights=weights,**inverse_options)
        # Forward checks of the represented adjoints are performed on new
        # random dual vectors, not the vectors used to form the basis.
        span_errors = {}
        for name, reduced in [('RH',reduced_r), ('CH',reduced_c)]:
            size=self._dimensions[name][1]
            if not size: continue
            dual = rng.normal(size=(size,2))+1j*rng.normal(size=(size,2))
            truth = action(name,dual)/self.sqrt_g[:,None]
            approximated = self.basis@(reduced.conj().T@dual)
            error = np.linalg.norm(approximated-truth,axis=0)/np.maximum(np.linalg.norm(truth,axis=0),np.finfo(float).tiny)
            span_errors[name]=float(error.max())
            if not np.isfinite(error).all() or error.max() > max(adjoint_tolerance,100*basis_rtol):
                raise ValueError('reduced current span failed independent '+name+' control')
        self.report = dict(method='actions_on_complete_current_span',dofs=dofs,field_channels=field_channels,
            constraint_rows=constraint_rows,current_basis_rank=rank,capacity=cap,basis_rtol=basis_rtol,
            basis_orthogonality_defect=defect,dropped_basis_directions=dropped,
            estimated_owned_peak_bytes=estimate,basis_allocated_bytes=storage.nbytes,
            reduced_operator_bytes=reduced_r.nbytes+reduced_c.nbytes,
            full_radiation_or_constraint_matrices_assembled=False,adjoint_errors=adjoint_errors,
            independent_span_errors=span_errors,inverse=self.inverse.report,
            setup_s=time.perf_counter()-started,setup_action_counts={k:dict(v) for k,v in self.counts.items()},
            original_BEM_residual_still_required=True)

    def solve(self, field, boundary_rhs):
        reduced, diagnostics = self.inverse.solve(field,boundary_rhs)
        physical = (self.basis@reduced if self.basis.shape[1] else
                    np.zeros((len(self.sqrt_g),*reduced.shape[1:]),complex))
        physical /= self.sqrt_g if physical.ndim==1 else self.sqrt_g[:,None]
        return physical, diagnostics
