"""ABBA on the same full-size modal current; native FMM projection is the reference."""
import json
import statistics
import time

import numpy as np
import torch

from ka80_action import B, HERE, load_cpu, project

torch.set_num_threads(4)
op=load_cpu()
fold=op.rotation_folding
perm,sign=fold.perm.numpy(),fold.sign.numpy()
x,y=[torch.tensor(project(np.fromfile(B/"native_ka80"/(k+".bin"),complex),perm,sign,1),dtype=torch.complex64)
     for k in ("random","current_depth6_random")]
screens=json.loads((B/"spectral_traffic_screen.json").read_text())
rows=[]
for enabled in [False,True,True,False]:
    op.rotation_folding=fold if enabled else None
    for i,screen in screens.items():
        rank=next(r["batch_ranks"] for r in screen["rows"] if r["tol"]==.0005)
        op.groups[int(i)]["streamed"].ranks=rank if enabled else None
    start=time.perf_counter()
    z=op(x)
    seconds=time.perf_counter()-start
    err=float(torch.linalg.vector_norm(z-y)/torch.linalg.vector_norm(y))
    if not np.isfinite(err) or err>5e-4:raise RuntimeError(err)
    rows.append(dict(candidate=enabled,seconds=seconds,relative_native_error=err))
    print(rows[-1],flush=True)
    (HERE/"ka80_cpu_repeat.json").write_text(json.dumps(rows,indent=2)+"\n")
