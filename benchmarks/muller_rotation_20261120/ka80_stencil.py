"""ABBA comparison on identical fully prepared factors and physical ka80 currents."""
import argparse
import ctypes
import json
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ka80_action import B,HERE,load_cpu,project
from ka80_transverse import CoarseBlock


def native_panel(folder,threads):
    library=Path(folder)/f'half_panel_{threads}.so'
    subprocess.run(['g++','-O3','-mavx2','-mf16c','-mfma','-shared','-fPIC','-fopenmp',
                    str(HERE/'half_panel_cpu.cpp'),'-o',str(library)],check=True)
    fun=ctypes.CDLL(str(library)).panel_dot
    fun.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*3+[ctypes.c_int]
    fun.restype=None
    def apply(a,x):
        f,r,k,_=a.shape
        if not a.is_contiguous() or x.shape[:2]!=(f,k):
            raise ValueError('Invalid CPU factor or input')
        data=x.permute(0,2,1).contiguous()
        y=torch.empty((f,r,x.shape[-1]),dtype=x.dtype)
        fun(f,r,k,x.shape[-1],a.data_ptr(),data.data_ptr(),y.data_ptr(),threads)
        return y
    packed=ctypes.CDLL(str(library)).packed_dot
    packed.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*4+[ctypes.c_int]
    packed.restype=None
    def currents(a,scale,x):
        f,_,r,k,_=a.shape
        data=x.permute(0,2,1).contiguous()
        y=torch.empty((f,2*r,x.shape[-1]),dtype=x.dtype)
        packed(f,r,k,x.shape[-1],a.data_ptr(),scale.data_ptr(),data.data_ptr(),y.data_ptr(),threads)
        return y
    return apply,currents


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--direct',action='store_true')
    p.add_argument('--currents',action='store_true')
    args=p.parse_args()
    temporary=tempfile.TemporaryDirectory()
    native,currents=native_panel(temporary.name,4) if args.direct else (None,None)
    torch.set_num_threads(4)
    op=load_cpu()
    ranks=json.loads((HERE/'rank_plan.json').read_text())['ranks']
    panels=[]
    for k,r in ranks.items():
        group=op.groups[int(k)]
        block=CoarseBlock(group['streamed'],group['op'],r,7,8,2,32)
        group['streamed']=block
        dense=(block.lift,block.restrict)
        panels.append((block,dense,tuple(m.to_sparse_csr() for m in dense)))
    dense_groups=[(g,g["streamed"]) for i,g in enumerate(op.groups) if str(i) not in ranks]
    report=[]
    perm,sign=op.rotation_folding.perm.numpy(),op.rotation_folding.sign.numpy()
    for label,m in [('random',1),('smooth',0 if args.direct else -1)]:
        x,y=[np.fromfile(B/'native_ka80'/(k+'.bin'),complex) for k in (label,'current_depth6_'+label)]
        if m>=0:x,y=project(x,perm,sign,m),project(y,perm,sign,m)
        x,y=[torch.as_tensor(v,dtype=op.dtype) for v in (x,y)]
        reference=None
        for sparse in (False,True,True,False):
            for block,dense,csr in panels:
                block.lift,block.restrict=csr if sparse else dense
                block.direct=native if sparse else None
            for group,old in dense_groups:
                group["streamed"]=(lambda x,rows,old=old:currents(*old.arr,x)) if args.currents and sparse else old
            t=perf_counter()
            with torch.inference_mode():z=op(x)
            elapsed=perf_counter()-t
            if reference is None:reference=z
            error=float(torch.linalg.vector_norm(z-y)/torch.linalg.vector_norm(y))
            delta=float(torch.linalg.vector_norm(z-reference)/torch.linalg.vector_norm(reference))
            row=dict(input=label,m=m,direct=sparse and args.direct,sparse=sparse,seconds=elapsed,native_error=error,relative_change=delta)
            print(row,flush=True)
            report.append(row)
            (HERE/('ka80_direct_panels.json' if args.direct and not args.currents else ('ka80_all_panels.json' if args.currents else 'ka80_stencil.json'))).write_text(json.dumps(report,indent=2)+'\n')
            if not (error<=5e-4 and delta<2e-6):raise RuntimeError(row)


if __name__=='__main__':main()
