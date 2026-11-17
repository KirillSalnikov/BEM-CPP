"""Hash-checked FP64 direct-gradient optimization for pinned FMM3D."""
import hashlib
from pathlib import Path

REVISION='blocked-radial-gradient-fp64-poly-v2'
UPSTREAM_SHA256='7ccde0b435533c48eff41a12062476f5b647c05be4296500d756f67f727be12c'
INCLUDE='#include <template-kernels.hpp>\n'
PATCHED_INCLUDE=INCLUDE+'#include <bem_direct.hpp>\n'
OLD='  h3ddirectcg_vec_cpp<double,VECDIM>(nd, zk, sources, charge, ns, ztarg, nt, pot, grad, thresh);'
NEW='  if (!bem_fmm::direct<VECDIM>(nd, zk, sources, charge, ns, ztarg, nt, pot, grad, thresh))\n'+OLD


def patch_source(root):
    root=Path(root);path=root/'vec-kernels/src/libkernels.cpp'
    current=path.read_text()
    original=current.replace(PATCHED_INCLUDE,INCLUDE).replace(NEW,OLD)
    if hashlib.sha256(original.encode()).hexdigest()!=UPSTREAM_SHA256:
        raise RuntimeError('Unrecognized FMM3D libkernels.cpp; preserve local edits before patching')
    if original.count(OLD)!=1:raise RuntimeError('Unexpected FMM3D direct-gradient entry point')
    patched=original.replace(INCLUDE,PATCHED_INCLUDE).replace(OLD,NEW)
    if current not in (original,patched):raise RuntimeError('Partial FMM3D direct-gradient patch')
    header=Path(__file__).with_name('bem_direct.hpp').read_bytes()
    target=root/'vec-kernels/include/bem_direct.hpp'
    changed=current!=patched or not target.exists() or target.read_bytes()!=header
    if current!=patched:path.write_text(patched)
    if changed:
        target.write_bytes(header)
        # Upstream's make rule does not track header dependencies.
        (root/'vec-kernels/src/libkernels.o').unlink(missing_ok=True)
    return dict(revision=REVISION,upstream_sha256=UPSTREAM_SHA256,
                patched_sha256=hashlib.sha256(patched.encode()).hexdigest(),
                header_sha256=hashlib.sha256(header).hexdigest())
