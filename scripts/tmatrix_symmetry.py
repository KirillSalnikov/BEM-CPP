"""C2x covariance in the repository's regular/outgoing vector-wave basis.

For a proper half turn about x, U|p,l,m> = (-1)^l |p,l,-m>.
Thus T[:, -m] = (-1)^l U T[:, m] for a C2x-invariant particle.
Numerical BEM symmetry must be checked independently before reusing columns.
"""
import numpy as np
from tmatrix_reference import modes


def c2x_plan(order):
    modal = modes(order)
    lookup = {mode: index for index, mode in enumerate(modal)}
    permutation = np.array([lookup[(p, l, -m)] for p, l, m in modal])
    signs = np.array([(-1.)**l for p, l, m in modal])
    representatives = [i for i, (_, _, m) in enumerate(modal) if m >= 0]
    audits = sorted({lookup[(p, l, -m)] for p in range(2)
                    for l in {1, max(1, order//2), order}
                    for m in {1, l}})
    return permutation, signs, representatives, audits


def c2x_partner_column(column, source, permutation, signs):
    return signs[source]*signs*np.asarray(column)[permutation]


def vertical_mirror_plan(order):
    """Reflection in xz: E is polar, H is axial.

    U|p,l,m> = (-1)^(p+m+1) |p,l,-m>. This holds for Cnv
    particles, including a bullet without C2x or a horizontal mirror.
    The spatial mirror must be measured and the column relation audited.
    """
    permutation, _, representatives, audits = c2x_plan(order)
    signs = np.array([(-1.)**(p+m+1) for p, l, m in modes(order)])
    return permutation, signs, representatives, audits
