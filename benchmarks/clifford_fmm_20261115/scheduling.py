"""Validated local scheduling patch for the pinned FMM3D source.

Only independent target-box work distribution changes. Arithmetic, precision,
tree occupancy and the order of accumulation within each target stay unchanged.
"""
import hashlib

REVISION = 'direct-near-dynamic-1-v1'
UPSTREAM_SHA256 = '6be43894d9bb7f078fb79dea3f566dd0b9ff676b97890c5e3de97b252d05072f'
OLD = 'C$OMP PARALLEL DO DEFAULT(SHARED)'
NEW = OLD + ' SCHEDULE(DYNAMIC,1)'


def patch_source(path):
    current = path.read_text()
    # Accept precisely upstream or our own patch; never overwrite other edits.
    original = current.replace(NEW, OLD)
    if hashlib.sha256(original.encode()).hexdigest() != UPSTREAM_SHA256:
        raise RuntimeError('Unrecognized hfmm3d.f; preserve local edits and inspect before patching')
    start = original.index('      if(ifnear.eq.0) goto 1000')
    stop = original.index('      timeinfo(6)', start)
    part = original[start:stop]
    if part.count(OLD) != 13:
        raise RuntimeError('Unexpected direct-near OpenMP loops')
    patched = original[:start] + part.replace(OLD, NEW) + original[stop:]
    if current not in (original, patched):
        raise RuntimeError('Partial or unrelated scheduling patch')
    if current != patched:
        path.write_text(patched)
    return dict(revision=REVISION, changed_loops=13,
                upstream_sha256=UPSTREAM_SHA256,
                patched_sha256=hashlib.sha256(patched.encode()).hexdigest())
