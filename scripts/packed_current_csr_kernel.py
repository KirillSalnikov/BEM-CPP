"""Fused complex CSR action; J/M share one index and one diagonal coefficient."""
import torch
import triton
import triton.language as tl

@triton.jit
def _action(R,C,V,X,Y,N:tl.constexpr,B:tl.constexpr,ROWS:tl.constexpr):
    row=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
    begin=tl.load(R+row,mask=row<N,other=0)
    end=tl.load(R+row+1,mask=row<N,other=0)
    index=begin[:,None]+tl.arange(0,B)[None,:]
    valid=(row[:,None]<N)&(index<end[:,None])
    col=tl.load(C+index,mask=valid,other=0)
    ar=tl.load(V+6*index,mask=valid,other=0);ai=tl.load(V+6*index+1,mask=valid,other=0)
    br=tl.load(V+6*index+2,mask=valid,other=0);bi=tl.load(V+6*index+3,mask=valid,other=0)
    cr=tl.load(V+6*index+4,mask=valid,other=0);ci=tl.load(V+6*index+5,mask=valid,other=0)
    jr=tl.load(X+2*col,mask=valid,other=0);ji=tl.load(X+2*col+1,mask=valid,other=0)
    mr=tl.load(X+2*(col+N),mask=valid,other=0);mi=tl.load(X+2*(col+N)+1,mask=valid,other=0)
    yjr=tl.sum(ar*jr-ai*ji+br*mr-bi*mi,axis=1)
    yji=tl.sum(ar*ji+ai*jr+br*mi+bi*mr,axis=1)
    ymr=tl.sum(cr*jr-ci*ji-ar*mr+ai*mi,axis=1)
    ymi=tl.sum(cr*ji+ci*jr-ar*mi-ai*mr,axis=1)
    tl.store(Y+2*row,yjr,mask=row<N);tl.store(Y+2*row+1,yji,mask=row<N)
    tl.store(Y+2*(row+N),ymr,mask=row<N);tl.store(Y+2*(row+N)+1,ymi,mask=row<N)

def action(operator,x):
    y=torch.empty_like(x);block=triton.next_power_of_2(max(1,operator.meta['max_row_entries']))
    _action[(triton.cdiv(operator.n,4),)](operator.rows,operator.columns,
        torch.view_as_real(operator.values),torch.view_as_real(x),torch.view_as_real(y),
        operator.n,block,4,num_warps=4,enable_fp_fusion=True)
    return y
