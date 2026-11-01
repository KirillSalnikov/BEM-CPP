"""Experimental order extension from an old square T and new incident columns.

Only the missing high-outgoing/low-incident block is inferred. This is a
signed transpose, not an adjoint, in the regular/outgoing convention used
by tmatrix_so3_mstm. A discrete BEM T need not be exactly reciprocal: the
result is an approximation until checked against independent direct data.
This module does not certify an order, modify a checkpoint, or run a solver.
"""
import numpy as np
from tmatrix_reference import modes


def extension_layout(low_order, full_order):
    if (not isinstance(low_order, (int, np.integer)) or isinstance(low_order, bool) or
        not isinstance(full_order, (int, np.integer)) or isinstance(full_order, bool) or
        not 1 <= low_order < full_order <= 256):
        raise ValueError('require integer orders 1 <= low < full <= 256')
    modal = modes(full_order)
    lookup = {mode: i for i, mode in enumerate(modal)}
    low = np.array([i for i, (_, degree, _) in enumerate(modal) if degree <= low_order])
    high = np.array([i for i, (_, degree, _) in enumerate(modal) if degree > low_order])
    flip = np.array([lookup[p, degree, -m] for p, degree, m in modal])
    signs = np.array([(-1.)**m for _, _, m in modal])
    high_lookup = {int(index): j for j, index in enumerate(high)}
    high_flip_columns = np.array([high_lookup[int(flip[index])] for index in high])
    return low, high, flip, signs, high_flip_columns


def complete_order_extension(low_t, high_incident_columns, low_order, full_order, *, out=None):
    """Columns must follow the l>low subset of modes(full_order), all rows kept.

    The old block uses modes(low_order), whose two polarizations are embedded
    separately. The caller is responsible for matching geometry, frequency,
    normalization and operator between the two directly computed inputs.
    """
    low, high, flip, signs, high_flip_columns = extension_layout(low_order, full_order)
    q = len(low) + len(high)
    a, b = np.asarray(low_t), np.asarray(high_incident_columns)
    if a.shape != (len(low), len(low)) or b.shape != (q, len(high)):
        raise ValueError('input dimensions do not match the stated modal bases')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite direct T input')
    if out is None:
        out = np.empty((q, q), dtype=np.complex128)
    if out.shape != (q, q) or out.dtype != np.complex128 or not out.flags.writeable:
        raise ValueError('output must be a writable complex128 square matrix')
    if np.shares_memory(out, a) or np.shares_memory(out, b):
        raise ValueError('output must not alias either direct input')
    out[np.ix_(low, low)] = a
    out[:, high] = b
    # B[j_flip, i_flip] is in a directly computed new incident column.
    # No conjugation and no averaging of already computed entries.
    for start in range(0, len(high), 128):
        selected = high[start:start+128]
        block = b[np.ix_(flip[low], high_flip_columns[start:start+128])].T
        out[np.ix_(selected, low)] = block * signs[selected, None] * signs[low][None, :]
    return out


def extension_work_counts(low_order, full_order):
    low, high, _, _, _ = extension_layout(low_order, full_order)
    return dict(old_modes=len(low), new_incident_modes=len(high), full_modes=len(low)+len(high),
                old_columns_not_resolved=len(low),
                avoided_column_fraction=len(low)/(len(low)+len(high)),
                inferred_complex_entries=len(low)*len(high),
                scope='Mode counts only; not measured wall time or a solver-work guarantee.')
