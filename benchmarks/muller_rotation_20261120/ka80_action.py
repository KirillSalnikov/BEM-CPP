"""Independent full ka80 screen using the existing encoded spectra in bounded batches."""
import ctypes as ct
import json
import os
import pickle
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"scripts"))
from symmetric_galerkin_fft import TorchSymmetricAction
from bounded_spectral_storage import decode_half_symbol
from screen import project

B=HERE.parent/"muller_ka80_week_20261116"
CASE=B/"fft_ka80"


class Local:
    def __init__(self):
        local=CASE/"packed_local"
        self.rows,self.cols,self.values=[np.load(local/(key+".npy"),mmap_mode="r") for key in ("rows","columns","coefficients")]
        self.lib=ct.CDLL(str(next(local.glob("host_*.so"))))
        self.apply=self.lib.bem_packed_current_cpu
        self.apply.argtypes=[ct.c_int,*([ct.c_void_p]*5),ct.c_int]
        self.apply.restype=None

    def __call__(self,x):
        y=torch.empty_like(x)
        self.apply(x.numel()//2,self.rows.ctypes.data,self.cols.ctypes.data,self.values.ctypes.data,
                   x.data_ptr(),y.data_ptr(),4)
        return y


class Block:
    def __init__(self,index):
        directory=CASE/".spectral-cache"/str(index)
        self.factor=(directory/"left.npy").is_file()
        keys=("left","right","left_scale","right_scale") if self.factor else ("symbol_packed","symbol_packed_scale")
        self.arr=[torch.from_numpy(np.load(directory/(k+".npy"),mmap_mode="c")) for k in keys]
        self.ranks=None

    def __call__(self,spectrum,rows):
        out=torch.empty((len(spectrum),rows,spectrum.shape[-1]),dtype=spectrum.dtype)
        batch=8 if self.factor else 512
        for i in range(0,len(out),batch):
            ix=slice(i,i+batch)
            if self.factor:
                l,r,ls,rs=self.arr
                rank=l.shape[2] if self.ranks is None else max(1,self.ranks[i//batch])
                left=torch.view_as_complex(l[ix,:,:rank].float())
                right=torch.view_as_complex(r[ix,:rank].float())
                out[ix]=torch.bmm(left,torch.bmm(right,spectrum[ix]))*(ls[ix]*rs[ix])
            else:
                symbol,scale=self.arr
                out[ix]=torch.bmm(decode_half_symbol(symbol[ix]),spectrum[ix])*scale[ix]
        return out


def load_cpu():
    with (CASE/"bound_operator.pkl").open("rb") as f:cpu=pickle.load(f)
    factors={}
    for row in json.loads((CASE/"streamed_preparation.json").read_text())["records"]:
        index=row["index"]
        if row["kind"]=="2D":cpu.canonical[index].symbol=np.load(CASE/row["file"],mmap_mode="c")
        else:factors[index]=tuple(np.load(CASE/row["directory"]/name,mmap_mode="c") for name in ("left.npy","right.npy"))
    cpu.local=sp.csr_matrix((cpu.geometry.N,cpu.geometry.N),dtype=np.complex64)
    op=TorchSymmetricAction(cpu,"cpu",torch.complex64,factors=factors,compact_maps=True,local_storage="csr")
    op.local=Local()
    op.local_storage="packed"
    for i,g in enumerate(op.groups):g["streamed"]=Block(i)
    op.enable_rotation_folding(cpu)
    return op


def main():
    torch.set_num_threads(4)
    op=load_cpu()
    fold=op.rotation_folding
    perm,sign=fold.perm.numpy(),fold.sign.numpy()
    schedules=json.loads((B/"spectral_traffic_screen.json").read_text())
    results=[]
    for name,m in [("random",1),("smooth",0),("random",-1),("smooth",-1)]:
        x,y=[np.fromfile(B/"native_ka80"/(key+".bin"),complex) for key in (name,"current_depth6_"+name)]
        if m>=0:x,y=project(x,perm,sign,m),project(y,perm,sign,m)
        x=torch.as_tensor(x,dtype=torch.complex64)
        if fold.detect(x)!=m:
            print("measured mode",fold.detect(x),"input",name,m,flush=True)
        y=torch.as_tensor(y,dtype=torch.complex64)
        reference=None
        for cut in [0,.0001,.0005,.001,.01]:
            for i,d in schedules.items():
                block=op.groups[int(i)]["streamed"]
                block.ranks=next(t["batch_ranks"] for t in d["rows"] if t["tol"]==cut) if cut else None
            start=perf_counter()
            z=op(x)
            elapsed=perf_counter()-start
            if reference is None:reference=z
            rel=lambda a,b:float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))
            row=dict(input=name,m=m,detected=fold.detect(x),cut=cut,seconds_cpu=elapsed,
                     relative_native_error=rel(z,y),change_vs_untrimmed=rel(z,reference))
            print(row,flush=True)
            if cut==0 and row["relative_native_error"]>5e-4:raise RuntimeError("reference gate")
            results.append(row)
            (HERE/"ka80_ref7_cpu.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":main()
