"""Independent SciPy GMRES solves with coordinated native operator batches.

Worker threads only run the existing SciPy Krylov recurrence. All native
operators and block inverses execute on the calling thread, preventing nested
OpenMP teams. Each column retains its own stopping test and scalar FP64 audit.
"""
from concurrent.futures import Future
from queue import Queue
from threading import Thread
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


def solve_incidents(d, incidents, tol=1e-8, maxiter=10, tile=256):
    from solve import lib, address
    f = np.ascontiguousarray(incidents, dtype=complex)
    if f.ndim != 2 or f.shape[1] != d.size or not 1 <= len(f) <= 16:
        raise ValueError('incident batch dimensions')
    if not np.all(np.isfinite(f)) or not 0 < tol < 1 or maxiter < 1:
        raise ValueError('invalid batch RHS or stopping controls')
    if not getattr(d,'_matrix_free',False) and (d._correction_only or d._dense_kernel != 'simd'):
        raise ValueError('batched GMRES requires dense FP64 SIMD')
    if not hasattr(lib, 'dirac_apply_batch') or not 16 <= tile <= 16384:
        raise ValueError('batch implementation or source tile unavailable')
    if hasattr(d, '_batch_rhs'):
        b = d._batch_rhs(f)
    else:
        b = np.empty_like(f)
        for j in range(len(f)):
            lib.dirac_rhs(d.p, address(f[j]), address(b[j]))
    queue = Queue()
    output = [None] * len(f)
    start = time.perf_counter()

    def worker(j):
        history = []
        counters = dict(calls=0, action_s=0., inverse_s=0.)

        def request(kind, vector):
            reply = Future()
            queue.put((kind, j, vector, reply))
            value, seconds = reply.result()
            if kind == 'inverse':
                counters['inverse_s'] += seconds
            else:
                counters['calls'] += 1
                counters['action_s'] += seconds
            return value

        try:
            norm = np.linalg.norm(b[j])
            if norm == 0:
                h = np.zeros(d.size, complex)
                info, residual = 0, 0.
            else:
                operator = LinearOperator((d.size, d.size), dtype=complex,
                                          matvec=lambda x: request('apply', x))
                inverse = (LinearOperator((d.size, d.size), dtype=complex,
                                          matvec=lambda x: request('inverse', x))
                           if d._dense_inverse else None)
                h, info = gmres(operator, b[j], M=inverse, rtol=tol, atol=0,
                                restart=60, maxiter=maxiter,
                                callback=history.append, callback_type='pr_norm')
                # The calling thread audits ALL final densities together after
                # the workers finish. Nothing is returned/accepted before then.
                residual = None
            stats = dict(counters, iterations=len(history), info=int(info),
                         residual=residual, history=[float(v) for v in history],
                         strict_history=[], dense_kernel=d._dense_kernel,
                         block_panels=d.acceleration['block_panels'])
            if info:
                raise RuntimeError('batched Dirac GMRES failed ' + str(stats))
            queue.put(('done', j, (h, stats), None))
        except BaseException as error:
            queue.put(('failed', j, error, None))

    threads = [Thread(target=worker, args=(j,), daemon=True) for j in range(len(f))]
    for thread in threads:
        thread.start()
    active, pending, failure = len(f), [], None
    batch_sizes = []
    while active:
        # A thread is either finished or blocked at a native operation. Waiting
        # for all active threads avoids races and makes batches reproducible.
        while len(pending) < active:
            kind, j, value, reply = queue.get()
            if kind in ('done', 'failed'):
                active -= 1
                if kind == 'failed':
                    failure = failure or value
                else:
                    output[j] = value
            else:
                pending.append((kind, j, value, reply))
        if not pending:
            continue
        replies = []
        if failure is None:
            try:
                for kind in ('inverse', 'apply', 'reference'):
                    group = [p for p in pending if p[0] == kind]
                    if not group:
                        continue
                    t = time.perf_counter()
                    if kind == 'apply':
                        vectors = np.stack([p[2] for p in group])
                        result = (d._batch_apply(vectors, [p[1] for p in group], tile)
                                  if hasattr(d, '_batch_apply') else d.apply_batch(vectors, tile))
                        batch_sizes.append(len(group))
                    elif kind == 'inverse':
                        result = (d._batch_inverses(np.stack([p[2] for p in group]),[p[1] for p in group]) if hasattr(d,'_batch_inverses') else
                                  [(d._batch_inverse(p[2], p[1]) if hasattr(d, '_batch_inverse') else d._dense_inverse(p[2])) for p in group])
                    else:
                        result = [(d._batch_reference(p[2], p[1]) if hasattr(d, '_batch_reference')
                                   else d.apply_reference(p[2])) for p in group]
                    share = (time.perf_counter() - t) / len(group)
                    replies.extend((p[3], (v, share)) for p, v in zip(group, result))
            except BaseException as error:
                failure = error
        for _, _, _, reply in pending:
            if failure is not None:
                reply.set_exception(failure)
        if failure is None:
            for reply, value in replies:
                reply.set_result(value)
        pending = []
    for thread in threads:
        thread.join()
    if failure is not None:
        raise failure
    final=np.stack([v[0] for v in output]);audit_start=time.perf_counter()
    if hasattr(d,'_batch_references'):
        checked=d._batch_references(final,list(range(len(final))))
    elif hasattr(d,'apply_reference_batch'):
        checked=d.apply_reference_batch(final)
    else:checked=np.array([d.apply_reference(v) for v in final])
    audit_seconds=time.perf_counter()-audit_start
    for j,(_,stats) in enumerate(output):
        norm=np.linalg.norm(b[j]);defect=np.linalg.norm(checked[j]-b[j])
        residual=float(defect/norm) if norm else (0. if defect==0 else float('inf'))
        stats.update(residual=residual,strict_history=[residual],full_reference_s=audit_seconds/len(final),reference_batch_size=len(final))
        stats['calls']+=1;stats['action_s']+=audit_seconds/len(final)
        if not np.isfinite(residual) or residual>tol*1.01:
            raise RuntimeError('batched Dirac full FP64 audit failed '+str(stats))
    wall = time.perf_counter() - start
    for _, stats in output:
        stats.update(seconds=wall / len(f), batch_wall_s=wall, batch_size=len(f),
                     batch_source_tile=tile, batch_action_sizes=batch_sizes,
                     seconds_scope='equal share of measured batch wall time; not independent column latency')
    return np.stack([v[0] for v in output]), [v[1] for v in output]
