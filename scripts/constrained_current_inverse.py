"""Small research inverse for coherent radiation and physical current priors.

Implements Citraro et al. (2024), eq. 26, in the nullspace of C x = d.
C must describe consistent boundary equations, e.g. rows of the original
Muller system. It must NOT assert zero physical field inside a dielectric.
This dense, factored-SVD prototype is not a production FMM preconditioner.
The returned current is only a seed: the original BEM residual must decide
acceptance after correction. Truncated radiation SVD is regularization,
not an exact generalized inverse of the untruncated radiation operator.
"""
import numpy as np
from scipy.linalg import svd


def _matrix(value, name):
    result = np.asarray(value, dtype=np.complex128)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite complex matrix')
    return result


def _positive_weights(value, size, name):
    if value is None:
        return np.ones(size)
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f'{name} must be real')
    result = np.asarray(raw, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError(f'{name} must contain {size} positive finite weights')
    return result


class _SVD:
    """Thin factors only; neither normal equations nor a dense inverse."""
    def __init__(self, matrix, rcond):
        if not np.isfinite(rcond) or not 0 <= rcond < 1:
            raise ValueError('SVD rcond must be finite and in [0, 1)')
        if min(matrix.shape) == 0:
            self.u = np.empty((matrix.shape[0], 0), complex)
            self.s = np.empty(0)
            self.v = np.empty((matrix.shape[1], 0), complex)
            self.original_rank_bound = 0
            return
        u, s, vh = svd(matrix, full_matrices=False, check_finite=False)
        keep = s > rcond * s[0]
        self.u = u[:, keep].copy()
        self.s = s[keep].copy()
        self.v = vh[keep].conj().T.copy()
        self.original_rank_bound = len(s)

    def solve(self, rhs):
        return self.v @ ((self.u.conj().T @ rhs) / self.s[:, None])

    def report(self):
        return dict(rank=len(self.s), rank_bound=self.original_rank_bound,
                    retained_condition=float(self.s[0] / self.s[-1]) if len(self.s) else None)


class ConstrainedCurrentInverse:
    """Minimize weighted coherent-field error while enforcing C x = d.

For y = sqrt(G) x, let P project onto ker(C / sqrt(G)), U=P sqrt(G) X,
and H=sqrt(W) R / sqrt(G). With y_p=(C/sqrt(G))^+ d and g=sqrt(W) f-H y_p,
the homogeneous part is U (H U)^+ g + (H P)^+ (g-H U (H U)^+ g).
P is applied through thin singular vectors; no n-by-n projector is formed.
Exact prior recovery requires H U to have full column rank and no discarded
prior singular values. Nonradiating prior columns cannot be distinguished.
"""
    def __init__(self, radiation, constraints, prior, *, field_weights=None,
                 current_metric=None, radiation_rcond=1e-3, prior_rcond=1e-12,
                 constraint_rcond=1e-13, constraint_tolerance=2e-10):
        r = _matrix(radiation, 'radiation')
        c = _matrix(constraints, 'constraints')
        x = _matrix(prior, 'prior')
        if r.shape[0] == 0 or r.shape[1] == 0 or c.shape[1] != r.shape[1] or x.shape[0] != r.shape[1]:
            raise ValueError('incompatible radiation, constraint and prior dimensions')
        if not np.isfinite(constraint_tolerance) or constraint_tolerance <= 0:
            raise ValueError('positive finite constraint tolerance required')
        self.sqrt_w = np.sqrt(_positive_weights(field_weights, r.shape[0], 'field weights'))
        self.sqrt_g = np.sqrt(_positive_weights(current_metric, r.shape[1], 'current metric'))
        self.r = self.sqrt_w[:, None] * r / self.sqrt_g[None, :]
        self.c = c / self.sqrt_g[None, :]
        self.constraints = _SVD(self.c, constraint_rcond)
        self.prior = self._project(self.sqrt_g[:, None] * x)
        self.prior_field = self.r @ self.prior
        self.prior_inverse = _SVD(self.prior_field, prior_rcond)
        v = self.constraints.v
        self.complement = _SVD(self.r - (self.r @ v) @ v.conj().T, radiation_rcond)
        self.constraint_tolerance = constraint_tolerance
        self.report = dict(constraints=self.constraints.report(), prior=self.prior_inverse.report(),
                           radiation=self.complement.report(), prior_columns=x.shape[1],
                           full_column_rank_prior=len(self.prior_inverse.s) == x.shape[1],
                           radiation_rcond=radiation_rcond, prior_rcond=prior_rcond,
                           constraint_rcond=constraint_rcond)

    def _project(self, vector):
        v = self.constraints.v
        return vector - v @ (v.conj().T @ vector)

    def solve(self, field, boundary_rhs):
        f = np.asarray(field, dtype=complex)
        d = np.asarray(boundary_rhs, dtype=complex)
        vector = f.ndim == 1
        if vector:
            f, d = f[:, None], d[:, None] if d.ndim == 1 else d
        if (f.ndim != 2 or d.ndim != 2 or f.shape[0] != self.r.shape[0]
                or d.shape != (self.c.shape[0], f.shape[1])
                or not np.isfinite(f).all() or not np.isfinite(d).all()):
            raise ValueError('invalid coherent field or boundary RHS')
        particular = self.constraints.solve(d)
        # Relative to each supplied RHS, not to an arbitrarily scaled operator.
        dnorm = np.linalg.norm(d, axis=0)
        floor = np.finfo(float).tiny
        inconsistent = np.linalg.norm(self.c @ particular - d, axis=0) / np.maximum(dnorm, floor)
        if np.any(inconsistent > self.constraint_tolerance):
            raise ValueError('boundary RHS is inconsistent with retained constraint space')
        weighted_field = self.sqrt_w[:, None] * f
        remaining = weighted_field - self.r @ particular
        coefficient = self.prior_inverse.solve(remaining)
        prior_part = self.prior @ coefficient
        correction = self._project(self.complement.solve(remaining - self.prior_field @ coefficient))
        y = particular + self._project(prior_part + correction)
        error = np.linalg.norm(self.c @ y - d, axis=0)
        # Homogeneous constraints need an absolute backward-error scale.
        denominator = np.maximum(dnorm, np.linalg.norm(self.c) * np.linalg.norm(y, axis=0) * np.finfo(float).eps)
        relative_constraint = error / np.maximum(denominator, floor)
        # For d=0, use the ordinary normwise backward error instead.
        homogeneous = dnorm == 0
        relative_constraint[homogeneous] = error[homogeneous] / np.maximum(
            np.linalg.norm(self.c) * np.linalg.norm(y[:, homogeneous], axis=0), floor)
        if np.any(relative_constraint > self.constraint_tolerance):
            raise FloatingPointError('reconstructed seed violates the boundary constraint gate')
        diagnostics = dict(weighted_field_relative_residual=(np.linalg.norm(self.r @ y - weighted_field, axis=0)
                / np.maximum(np.linalg.norm(weighted_field, axis=0), floor)).tolist(),
            constraint_relative_residual=relative_constraint.tolist(),
            current_metric_norm=np.linalg.norm(y, axis=0).tolist(),
            prior_metric_norm=np.linalg.norm(prior_part, axis=0).tolist(),
            complement_metric_norm=np.linalg.norm(correction, axis=0).tolist())
        result = y / self.sqrt_g[:, None]
        return (result[:, 0] if vector else result), diagnostics
