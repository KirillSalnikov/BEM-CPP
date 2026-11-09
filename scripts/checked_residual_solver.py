"""Iterative refinement with a separately supplied accurate residual operator.

The inner operator may be compressed or low precision. Its own GMRES residual
never authorizes acceptance: every outer correction is checked by ``check``.
Double-single and original FP64 checks can use the same solve policy.
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


def solve_refined(rhs, inner, check, preconditioner, *, tolerance=2e-6,
                  restart=80, max_cycles=6, max_outer=6, progress=None):
    b = np.asarray(rhs, dtype=np.complex128)
    if (b.ndim != 1 or not np.all(np.isfinite(b))
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Expected a finite RHS vector and positive tolerance')
    bnorm = float(np.linalg.norm(b))
    x = np.zeros_like(b)
    record = dict(iterations=0, solver_s=0., check_s=0., checks=0,
                  corrections=[], status='running')
    if bnorm == 0:
        record.update(residual=0., status='zero_rhs')
        return x, record
    A = LinearOperator((len(b), len(b)), matvec=inner, dtype=np.complex128)
    residual, ratio = b.copy(), 1.
    for outer in range(max_outer):
        history = []
        last_progress = time.perf_counter()

        def callback(value):
            nonlocal last_progress
            history.append(float(value))
            if progress and time.perf_counter() - last_progress > 20:
                progress(dict(outer=outer, iterations=len(history), inner=float(value)))
                last_progress = time.perf_counter()

        start = time.perf_counter()
        delta, info = gmres(
            A, residual, M=preconditioner,
            rtol=1e-4 if outer == 0 else min(.05, .15*tolerance/ratio),
            atol=0., restart=restart, maxiter=max_cycles,
            callback=callback, callback_type='pr_norm')
        record['solver_s'] += time.perf_counter() - start
        record['iterations'] += len(history)
        x += delta
        if not np.all(np.isfinite(x)):
            raise RuntimeError('Nonfinite current from inner solve')
        start = time.perf_counter()
        residual = b - check(x)
        ratio = float(np.linalg.norm(residual) / bnorm)
        record['check_s'] += time.perf_counter() - start
        record['checks'] += 1
        row = dict(iterations=len(history), info=int(info), residual=ratio)
        record['corrections'].append(row)
        if progress:
            progress(dict(outer=outer, **row))
        if not np.isfinite(ratio):
            raise RuntimeError('Nonfinite checked residual')
        if ratio <= tolerance:
            record.update(residual=ratio, status='complete')
            return x, record
    record.update(residual=ratio, status='failed')
    return x, record
