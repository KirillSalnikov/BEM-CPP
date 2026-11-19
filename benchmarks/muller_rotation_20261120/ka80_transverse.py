"""Screen transverse compression against the full target-size FFT and native FMM actions."""
import argparse
import json
from time import perf_counter
import numpy as np
import torch

from ka80_action import HERE, B, load_cpu, project
from transverse_fft_interpolation import TransverseLines


class CoarseBlock:
    def __init__(self,block,op,ranks,degree,edge,stride,guard):
        self.left = TransverseLines(op.target.nx,2*op.tc,degree=degree,edge=edge,stride=stride)
        self.right = TransverseLines(op.source.nx,2*op.sc,degree=degree,edge=edge,stride=stride)
        self.lift = torch.from_numpy(self.left.matrix)
        self.restrict = torch.from_numpy(self.right.matrix.T.copy())
        self.direct=None
        self.panels=[]
        self.compact=[]
        self.saved=0
        l,r,ls,rs=block.arr
        for j,rank in enumerate(ranks):
            sl=slice(j*8,(j+1)*8)
            compact=self.left.compact_batch(j*8,8,len(l),guard)
            self.compact.append(compact)
            a,b=l[sl,:,:rank],r[sl,:rank]
            if compact:
                a=a.index_select(1,torch.from_numpy(self.left.indices))
                b=b.index_select(2,torch.from_numpy(self.right.indices))
            else:
                a=a.contiguous()
                b=b.contiguous()
            scale=ls[sl]*rs[sl]
            self.panels.append((a,b,scale))
            self.saved+=(l[sl,:,:rank].numel()+r[sl,:rank].numel()-a.numel()-b.numel())*a.element_size()

    def __call__(self,spectrum,rows):
        result=torch.empty((len(spectrum),rows,spectrum.shape[-1]),dtype=spectrum.dtype)
        for j,(a,b,scale) in enumerate(self.panels):
            sl=slice(j*8,(j+1)*8)
            x=spectrum[sl]
            if self.compact[j]:x=self.right.apply(x,self.restrict,self.right.channels)
            if self.direct is not None and x.shape[-1]<=2:
                compressed=self.direct(a,self.direct(b,x))*scale
            else:
                left,right=(torch.view_as_complex(t.float()) for t in (a,b))
                compressed=torch.bmm(left,torch.bmm(right,x))*scale
            result[sl]=self.left.apply(compressed,self.lift,self.left.channels) if self.compact[j] else compressed
        return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--degree',type=int,default=7)
    p.add_argument('--edge',type=int,default=8)
    p.add_argument('--stride',type=int,default=2)
    p.add_argument('--guard',type=int,default=0)
    p.add_argument('--groups',type=int,nargs='+',default=[1,2,4,5])
    a=p.parse_args()
    torch.set_num_threads(4)
    op=load_cpu()
    fold=op.rotation_folding
    perm,sign=fold.perm.numpy(),fold.sign.numpy()
    rank=json.loads((HERE/'rank_plan.json').read_text())['ranks']
    original={i:op.groups[i]['streamed'] for i in map(int,rank)}
    for i,block in original.items():
        block.ranks=rank[str(i)]
    start=perf_counter()
    compact={i:CoarseBlock(original[i],op.groups[i]['op'],rank[str(i)],a.degree,a.edge,a.stride,a.guard) for i in a.groups}
    saved=sum(block.saved for block in compact.values())
    old=json.loads((HERE/'transfer_summary.json').read_text())['stream_bytes_candidate']
    report=dict(parameters=vars(a),setup_seconds=perf_counter()-start,stream_bytes_previous=old,
                stream_bytes_candidate=old-saved,rows=[])
    path=HERE/f"transverse_{a.stride}_{a.degree}_{a.edge}_{''.join(map(str,a.groups))}{'_g'+str(a.guard) if a.guard else ''}.json"
    rng=np.random.default_rng(33271)
    for label,m in [('smooth',0),('smooth',-1),('random',1),('random',-1),('holdout',3),('holdout',-1)]:
        if label=='holdout':
            x=rng.standard_normal(op.N)+1j*rng.standard_normal(op.N)
            native=None
        else:
            x,native=[np.fromfile(B/'native_ka80'/(k+'.bin'),complex) for k in (label,'current_depth6_'+label)]
            if m>=0:native=project(native,perm,sign,m)
            native=torch.as_tensor(native,dtype=op.dtype)
        if m>=0:x=project(x,perm,sign,m)
        x=torch.as_tensor(x,dtype=op.dtype)
        norms=lambda y,ref:float(torch.linalg.vector_norm(y-ref)/torch.linalg.vector_norm(ref))
        row=dict(input=label,m=m)
        for enabled in (False,True):
            for i,small in compact.items():op.groups[i]['streamed']=small if enabled else original[i]
            t=perf_counter()
            with torch.inference_mode(): y=op(x)
            row['cpu_s_'+str(enabled)]=perf_counter()-t
            if enabled:
                row['relative_change']=norms(y,reference)
                if native is not None:row['native_error']=norms(y,native)
            else:
                reference=y
                if native is not None:row['previous_native_error']=norms(y,native)
        report['rows'].append(row)
        path.write_text(json.dumps(report,indent=2)+'\n')
        print(row,flush=True)

if __name__=='__main__':main()
