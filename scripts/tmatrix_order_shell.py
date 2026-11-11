"""Incident-order shells with the same independent C2/C6 controls as full builds.

Outgoing projection always retains the full order. Lower incident degrees
are solved only when selected as independent symmetry controls. A completed
shell is not a complete T matrix and must never be used as one implicitly.
"""
import numpy as np
from tmatrix_reference import modes
from tmatrix_symmetry import c2x_plan, vertical_mirror_plan
from tmatrix_c6 import audit_packets


def shell_schedule(order, minimum=1, *, c2=False, c6=False, mirror=False, vertical=False):
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool)
           for v in (order, minimum)) or not 1 <= minimum <= order <= 256:
        raise ValueError('require integer orders 1 <= minimum <= order <= 256')
    if mirror and not c6:
        raise ValueError('mirror packing requires C6')
    if c2 and vertical:
        raise ValueError('choose C2x or vertical reflection for negative-m reuse')
    modal = modes(order)
    lookup = {mode: i for i, mode in enumerate(modal)}
    requested = [i for i, (_, degree, _) in enumerate(modal) if degree >= minimum]
    permutation, signs, all_representatives, audits = (vertical_mirror_plan if vertical else c2x_plan)(order)
    representatives = [i for i in all_representatives if modal[i][1] >= minimum]
    groups = audit_packets(modal, order, mirror) if c6 else []
    if minimum > 1:
        # Retain the original low/intermediate controls, and cover the first,
        # middle and last degrees actually being added. Both polarizations
        # and the two extreme nonzero m values are checked independently.
        shell_degrees = sorted({minimum, (minimum+order)//2, order})
        audits = sorted(set(audits) | {lookup[p, degree, -m]
                        for degree in shell_degrees for p in (0, 1) for m in {1, degree}})
        if c6:
            for degree in shell_degrees:
                additions = ([[lookup[p, degree, m] for m in range(min(degree, 5)+1) for p in (0, 1)]]
                             if mirror else [[lookup[p, degree, m] for m in range(min(degree, 5)+1)]
                                             for p in (0, 1)])
                for group in additions:
                    if group not in groups:
                        groups.append(group)
    reuse = c2 or vertical
    audit_work = [i for negative in audits for i in (int(permutation[negative]), negative)] if reuse else []
    work = list(dict.fromkeys(audit_work+representatives)) if reuse else requested
    return dict(requested=requested, work=work, representatives=representatives,
                c2_audits=audits if reuse else [], c6_groups=groups,
                audit_work=audit_work, permutation=permutation, signs=signs)
