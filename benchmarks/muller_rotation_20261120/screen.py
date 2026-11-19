"""Interleaved full-vs-folded physical actions on an existing SciIce prism."""
import argparse
import json
import pickle
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"scripts"))
from symmetric_galerkin_fft import TorchSymmetricAction


def project(v,perm,sign,m):
    y=v.copy()
    step=v
    for j in range(1,6):
        step=step[perm]*sign
        y+=np.exp(-2j*np.pi*m*j/6)*step
    return y/6


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--case",choices=["ka10_ref2","ka20_ref3"],default="ka10_ref2")
    p.add_argument("--half",action="store_true")
    a=p.parse_args()
    base=HERE.parent/"symmetric_galerkin_fft_20261108"
    with (base/(a.case+".pkl")).open("rb") as f: cpu=pickle.load(f)
    cpu.geometry.directory=base/a.case
    cpu.load_correction()
    torch.set_num_threads(2)
    op=TorchSymmetricAction(cpu,a.device,torch.complex64,local_storage="csr",
                            symbol_storage="half" if a.half else "full")
    op.enable_rotation_folding(cpu)
    fold=op.rotation_folding
    rng=np.random.default_rng(8141)
    v=rng.normal(size=op.N)+1j*rng.normal(size=op.N)
    perm=fold.perm.cpu().numpy()
    sign=fold.sign.cpu().numpy()
    report=dict(case=a.case,device=a.device,half=a.half,groups=[
        [g["count"],len(plan[0]) if plan else g["count"]] for g,plan in zip(op.groups,fold.plans)],rows=[])
    for m in [-1,0,1,2,3,4,5]:
        z=v if m==-1 else project(v,perm,sign,m)
        x=torch.as_tensor(z,device=a.device,dtype=op.dtype)
        samples=[[],[]]
        detected=fold.detect(x)
        if detected!=m:
            raise ValueError(f"rotation detection {m} => {detected}")
        for i,flag in enumerate([0,1,1,0,0,1,1,0]):
            op.rotation_folding=fold if flag else None
            if a.device=="cuda":torch.cuda.synchronize()
            start=perf_counter()
            y=op(x)
            if a.device=="cuda":torch.cuda.synchronize()
            elapsed=perf_counter()-start
            if i==0: reference=y.clone()
            if i>=2:samples[flag].append(elapsed)
            if flag:
                error=(torch.linalg.vector_norm(y-reference)/torch.linalg.vector_norm(reference)).item()
                if not np.isfinite(error) or error>3e-6:raise ValueError((m,error))
        full,folded=(float(np.median(z)) for z in samples)
        report["rows"].append(dict(mode=m,relative_error=error,seconds_full=full,
                                   seconds_folded=folded,speedup=full/folded))
        print(report["rows"][-1],flush=True)
    (HERE/f"{a.case}_{a.device}_{'half' if a.half else 'full'}.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
