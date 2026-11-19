"""Interleave CPU thread counts on the same full ka80 regular and near-field action."""
import json
import tempfile
from time import perf_counter
import numpy as np
import torch

from ka80_action import B,HERE,load_cpu,project
from ka80_stencil import native_panel
from ka80_transverse import CoarseBlock


def main():
    torch.set_num_threads(4)
    op=load_cpu()
    ranks=json.loads((HERE/'rank_plan.json').read_text())['ranks']
    for k,r in ranks.items():
        g=op.groups[int(k)]
        block=CoarseBlock(g['streamed'],g['op'],r,7,8,2,32)
        block.lift,block.restrict=block.lift.to_sparse_csr(),block.restrict.to_sparse_csr()
        g['streamed']=block
    dense=[(g,g['streamed']) for i,g in enumerate(op.groups) if str(i) not in ranks]
    perm,sign=op.rotation_folding.perm.numpy(),op.rotation_folding.sign.numpy()
    x,y=[torch.as_tensor(project(np.fromfile(B/'native_ka80'/(k+'.bin'),complex),perm,sign,1),dtype=op.dtype)
         for k in ('random','current_depth6_random')]
    out=[]
    with tempfile.TemporaryDirectory() as folder:
        kernels={threads:native_panel(folder,threads) for threads in (4,8,16)}
        for threads in (4,8,16,16,8,4):
            torch.set_num_threads(threads)
            mul,current=kernels[threads]
            for k in ranks:op.groups[int(k)]['streamed'].direct=mul
            for group,old in dense:group['streamed']=lambda x,rows,old=old:current(*old.arr,x)
            t=perf_counter()
            with torch.inference_mode():z=op(x)
            elapsed=perf_counter()-t
            error=float(torch.linalg.vector_norm(z-y)/torch.linalg.vector_norm(y))
            if not error<=5e-4:raise RuntimeError(error)
            out.append(dict(threads=threads,seconds=elapsed,native_error=error))
            print(out[-1],flush=True)
            (HERE/'ka80_panel_threads.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
