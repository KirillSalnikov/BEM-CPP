"""Experimental full-block Loewner interpolation of sampled complex matrices.

Unlike scalar-weight AAA, this realizes a matrix-valued resolvent. A sample at
the center is extracted exactly, and divided differences use every other
sample. Truncation uses training data only. Independent validation is required;
neither passivity nor accuracy between training nodes is guaranteed.

Reference: Mayo & Antoulas (2007), "A framework for the solution of the
generalized realization problem", https://www.sciencedirect.com/science/article/pii/S0024379507001280.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve, svd


@dataclass
class LoewnerMatrix:
    center: float
    scale: float
    central_sample: np.ndarray
    loewner: np.ndarray
    shifted: np.ndarray
    left: np.ndarray
    right: np.ndarray
    domain: tuple

    def evaluate(self, x):
        if not np.isfinite(x) or not self.domain[0] <= x <= self.domain[1]:
            raise ValueError('extrapolation is disabled')
        z = (float(x)-self.center)/self.scale
        if z == 0 or not self.loewner.size:
            return self.central_sample.copy()
        return self.central_sample + z*(self.left@solve(self.shifted-z*self.loewner, self.right))


def fit_loewner(nodes, samples, *, rtol=1e-10, center_index=None):
    z = np.asarray(nodes, dtype=float)
    f = [np.asarray(t, dtype=complex) for t in samples]
    if (z.ndim != 1 or len(z) != len(f) or len(z) < 3 or len(z) % 2 != 1
            or not np.isfinite(z).all() or len(set(z)) != len(z)
            or not 0 < rtol < 1 or any(t.ndim != 2 or t.shape != f[0].shape
            or not np.isfinite(t).all() for t in f)):
        raise ValueError('need an odd number >=3 of finite equal-shaped matrix samples')
    if center_index is None:
        center_index = int(np.argmin(abs(z-(z.min()+z.max())/2)))
    if not 0 <= center_index < len(z): raise ValueError('invalid center index')
    center = float(z[center_index]); scale = float(max(abs(z-center)))
    t0 = f[center_index].copy()
    indices = [int(i) for i in np.argsort(z) if i != center_index]
    x = (z-center)/scale
    divided = {i: (f[i]-t0)/x[i] for i in indices}
    li, ri = indices[::2], indices[1::2]
    loewner = np.block([[(divided[i]-divided[j])/(x[i]-x[j]) for j in ri] for i in li])
    shifted = np.block([[(x[i]*divided[i]-x[j]*divided[j])/(x[i]-x[j]) for j in ri] for i in li])
    # A nonreal pencil point avoids choosing a physical real-frequency pole.
    # It also retains polynomial descriptor components that L alone can miss.
    u, singular, vh = svd(shifted-1j*loewner, full_matrices=False, check_finite=True)
    rank = int(np.count_nonzero(singular > rtol*singular[0])) if singular[0] > 0 else 0
    y = u[:, :rank].conj().T; v = vh[:rank].conj().T
    model = LoewnerMatrix(center, scale, t0, y@loewner@v, y@shifted@v,
                          np.hstack([divided[j] for j in ri])@v,
                          y@np.vstack([divided[i] for i in li]),
                          (float(z.min()), float(z.max())))
    errors = [float(np.linalg.norm(model.evaluate(k)-t)/max(np.linalg.norm(t), 1e-300)) for k,t in zip(z,f)]
    info = dict(algorithm='full-block Loewner, central divided differences, SVD of shifted-i*L',
                center=center, scale=scale, training_count=len(z), rank=rank,
                pencil_shape=list(loewner.shape), rtol=rtol, singular_values=singular.tolist(),
                left_indices=li, right_indices=ri, training_relative_errors=errors,
                maximum_training_relative_error=max(errors))
    return model, info
