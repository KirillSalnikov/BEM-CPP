"""Shared-denominator rational approximation of complex T matrices.

Matrix-valued AAA with relative Frobenius row scaling. Training samples must
have the same basis, geometry and material. This is an interpolator, not a
Maxwell solver or an error certificate. Validate on independent direct solves.
Loewner QR and matrix evaluation stream over entries; samples may be memmaps.
The caller must keep the sample arrays unchanged while the model is in use.
"""
from dataclasses import dataclass
import numpy as np


def _samples(nodes, samples):
    z = np.array(nodes, dtype=float, copy=True)
    f = tuple(np.asarray(a) for a in samples)
    if z.ndim != 1 or not len(z) or len(z) != len(f):
        raise ValueError('one real node per sample is required')
    if not np.isfinite(z).all() or len(np.unique(z)) != len(z):
        raise ValueError('nodes must be finite and distinct')
    order = 'C' if f[0].flags.c_contiguous else 'F'
    if any(a.shape != f[0].shape or not a.size or not (a.flags.c_contiguous if order == 'C' else a.flags.f_contiguous)
           or a.dtype.kind not in 'fc' for a in f):
        raise ValueError('samples must be nonempty floating arrays of identical shape and contiguous storage order')
    return z, f


@dataclass
class RationalMatrix:
    nodes: np.ndarray
    samples: tuple
    weights: np.ndarray
    domain: tuple
    chunk_entries: int = 8192

    def __post_init__(self):
        self.nodes, self.samples = _samples(self.nodes, self.samples)
        self.weights = np.array(self.weights, dtype=complex, copy=True)
        if self.weights.shape != self.nodes.shape or not np.isfinite(self.weights).all() or not np.any(self.weights):
            raise ValueError('finite nonzero barycentric weights are required')
        if (len(self.domain) != 2 or not np.isfinite(self.domain).all()
                or self.domain[0] > self.nodes.min() or self.domain[1] < self.nodes.max()
                or self.chunk_entries < 1):
            raise ValueError('invalid domain or chunk size')
        self.weights /= np.max(abs(self.weights))

    def coefficients(self, x):
        x = float(x)
        if not np.isfinite(x) or not self.domain[0] <= x <= self.domain[1]:
            raise ValueError('extrapolation outside the training interval is disabled')
        delta = x-self.nodes
        hit = np.flatnonzero(delta == 0)
        if hit.size:
            c = np.zeros(len(delta), complex); c[hit[0]] = 1
            return c
        terms = self.weights*(np.min(abs(delta))/delta)
        denominator = np.sum(terms)
        if abs(denominator) <= 64*np.finfo(float).eps*np.sum(abs(terms)):
            raise ValueError('rational denominator is singular or ill-conditioned')
        return terms/denominator

    def evaluate(self, x, out=None):
        c = self.coefficients(x)
        order = 'C' if self.samples[0].flags.c_contiguous else 'F'
        if out is None:
            out = np.empty(self.samples[0].shape, dtype=np.complex128, order=order)
        if (out.shape != self.samples[0].shape or out.dtype != np.complex128
                or not (out.flags.c_contiguous if order == 'C' else out.flags.f_contiguous) or not out.flags.writeable
                or any(np.shares_memory(out, a) for a in self.samples)):
            raise ValueError('output must be a separate writable contiguous complex128 array')
        target = out.reshape(-1, order=order)
        flat = [a.reshape(-1, order=order) for a in self.samples]
        for start in range(0, target.size, self.chunk_entries):
            sl = slice(start, start+self.chunk_entries)
            target[sl] = 0
            for weight, sample in zip(c, flat):
                if weight != 0:
                    target[sl] += weight*sample[sl]
            if not np.isfinite(target[sl]).all():
                raise ValueError('nonfinite interpolated matrix')
        return out


def _loewner_weights(z, flat, support, remaining, scales, chunk_entries):
    """QR of stacked Loewner blocks, avoiding normal equations and a huge SVD."""
    r = np.empty((0, len(support)), complex)
    for start in range(0, flat[0].size, chunk_entries):
        sl = slice(start, start+chunk_entries)
        for i in remaining:
            a = np.column_stack([(flat[i][sl]-flat[j][sl])/
                                 ((z[i]-z[j])*scales[i]) for j in support])
            r = np.linalg.qr(np.vstack((r, a)), mode='r')
    # full_matrices also supplies null vectors for very small scalar problems.
    _, _, vh = np.linalg.svd(r, full_matrices=True)
    return vh[-1].conj()


def fit_aaa(nodes, samples, *, max_terms=None, tolerance=1e-6, chunk_entries=8192):
    """Return (model, diagnostics); *every* supplied sample is a training solve.

    The max relative Frobenius error selects support nodes. All matrix entries
    share scalar complex weights. A support cap below the training count leaves
    a nonempty Loewner problem. No independent validation data belong here.
    """
    z, f = _samples(nodes, samples)
    n = len(z)
    max_terms = n-1 if max_terms is None else max_terms
    if n < 2 or not 1 <= max_terms < n or chunk_entries < 1 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('need >=2 samples, 1<=max_terms<sample count and a valid tolerance/chunk size')
    order = 'C' if f[0].flags.c_contiguous else 'F'
    flat = [a.reshape(-1, order=order) for a in f]
    norm2, initial2 = np.zeros(n), np.zeros(n)
    for start in range(0, flat[0].size, chunk_entries):
        block = np.array([a[start:start+chunk_entries] for a in flat], dtype=complex)
        if not np.isfinite(block).all():
            raise ValueError('nonfinite training matrix')
        norm2 += np.sum(abs(block)**2, axis=1)
        initial2 += np.sum(abs(block-np.mean(block, axis=0))**2, axis=1)
    if not np.isfinite(norm2).all() or not np.isfinite(initial2).all():
        raise ValueError('training matrix norm overflow')
    norms = np.sqrt(norm2)
    scales = np.maximum(norms, max(float(np.max(norms))*1e-14, 1e-150))
    residuals = np.sqrt(initial2)/scales
    support, history = [], []
    for _ in range(max_terms):
        remaining = [i for i in range(n) if i not in support]
        support.append(max(remaining, key=lambda i: residuals[i]))
        remaining = [i for i in range(n) if i not in support]
        weights = _loewner_weights(z, flat, support, remaining, scales, chunk_entries)
        model = RationalMatrix(z[support], tuple(f[i] for i in support), weights,
                               (float(z.min()), float(z.max())), chunk_entries)
        residuals = np.zeros(n)
        for i in remaining:
            try:
                c = model.coefficients(z[i])
            except ValueError:
                residuals[i] = np.inf
                continue
            err2 = 0.
            for start in range(0, flat[0].size, chunk_entries):
                sl = slice(start, start+chunk_entries)
                delta = sum(w*flat[j][sl] for w, j in zip(c, support))-flat[i][sl]
                err2 += float(np.vdot(delta, delta).real)
            residuals[i] = np.sqrt(err2)/scales[i]
        history.append({'terms': len(support), 'max_training_relative_frobenius_error': float(max(residuals))})
        if max(residuals) <= tolerance:
            break
    return model, {'algorithm': 'matrix-valued AAA, relative Frobenius row scaling, streamed QR/SVD',
                   'training_count': n, 'support_indices': support,
                   'training_relative_frobenius_errors': residuals.tolist(),
                   'history': history, 'tolerance': tolerance,
                   'training_tolerance_met': bool(max(residuals) <= tolerance)}


def validate_family(parameters):
    """Reject inconsistent physical/numerical definitions before fitting T(ka)."""
    ignored = {'ka'}
    if not parameters:
        raise ValueError('empty family')
    first = {k: v for k, v in parameters[0].items() if k not in ignored}
    for p in parameters[1:]:
        if {k: v for k, v in p.items() if k not in ignored} != first:
            raise ValueError('T-matrix family differs in parameters other than ka')
