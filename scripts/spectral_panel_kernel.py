"""Direct FP32 spectral actions on the immutable FP16 encoding for thin RHS panels."""
import numpy as np
import torch
import triton
import triton.language as tl


def stencil_table(matrix,device):
    """Pack each row of P or its transpose. Both kernels gather from the input."""
    rows,cols=np.nonzero(matrix)
    counts=np.bincount(rows,minlength=len(matrix))
    width=int(counts.max())
    offsets=np.arange(len(rows))-np.repeat(np.cumsum(counts)-counts,counts)
    index=np.zeros((len(matrix),width),np.int32)
    weight=np.zeros_like(index,np.float32)
    index[rows,offsets]=cols
    weight[rows,offsets]=matrix[rows,cols]
    return torch.as_tensor(index,device=device),torch.as_tensor(weight,device=device)


@triton.jit
def _interpolate(X,I,W,Y,N,SRC:tl.constexpr,DST:tl.constexpr,LANES:tl.constexpr,
                 TERMS:tl.constexpr,BLOCK:tl.constexpr):
    out=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=out//LANES%DST
    base=out//(DST*LANES)*SRC*LANES+out%LANES
    result=tl.full((BLOCK,),0,tl.float32)
    for term in tl.static_range(TERMS):
        index=tl.load(I+row*TERMS+term)
        weight=tl.load(W+row*TERMS+term)
        value=tl.load(X+base+index*LANES,mask=(out<N)&(weight!=0.),other=0.)
        result=tl.fma(weight,value,result)
    tl.store(Y+out,result,mask=out<N)


def interpolate(x,table,channels):
    nf,dimension,rhs=x.shape
    indices,weights=table
    source=dimension//(2*channels)
    dest,terms=indices.shape
    output=torch.empty((nf,2*dest*channels,rhs),device=x.device,dtype=x.dtype)
    count=output.numel()*2
    _interpolate[(triton.cdiv(count,128),)](torch.view_as_real(x.contiguous()),indices,weights,
        torch.view_as_real(output),count,source,dest,channels*rhs*2,terms,128)
    return output


@triton.jit
def _multiply(A,X,Y,ROWS:tl.constexpr,COLS:tl.constexpr,RHS:tl.constexpr,
              SPLITS:tl.constexpr,TILE_R:tl.constexpr,TILE_K:tl.constexpr):
    # Each CTA owns its output rows and a contiguous portion of the dot
    # product. Long right factors use a small FP32 reduction across tiles.
    row=tl.program_id(0)*TILE_R+tl.arange(0,TILE_R)
    f=tl.program_id(1)
    split=tl.program_id(2)
    k=split*TILE_K+tl.arange(0,TILE_K)
    a=2*((f*ROWS+row[:,None])*COLS+k[None,:])
    valid=(row[:,None]<ROWS)&(k[None,:]<COLS)
    ar=tl.load(A+a,mask=valid,other=0.).to(tl.float32)
    ai=tl.load(A+a+1,mask=valid,other=0.).to(tl.float32)
    for h in tl.static_range(RHS):
        src=2*((f*COLS+k)*RHS+h)
        xr=tl.load(X+src,mask=k<COLS,other=0.)
        xi=tl.load(X+src+1,mask=k<COLS,other=0.)
        re=tl.sum(ar*xr[None,:]-ai*xi[None,:],axis=1)
        im=tl.sum(ar*xi[None,:]+ai*xr[None,:],axis=1)
        dst=2*(((f*ROWS+row)*SPLITS+split)*RHS+h)
        tl.store(Y+dst,re,mask=row<ROWS)
        tl.store(Y+dst+1,im,mask=row<ROWS)


def half_product(factor,x):
    nf,rows,cols,_=factor.shape
    rhs=x.shape[-1]
    if (x.shape[:2]!=(nf,cols) or x.dtype!=torch.complex64 or factor.dtype!=torch.float16
            or rhs not in (1,2) or not factor.is_contiguous() or factor.shape[-1]!=2):
        raise ValueError('Invalid encoded spectral factor or thin complex RHS')
    tile_k=min(256,triton.next_power_of_2(cols))
    splits=triton.cdiv(cols,tile_k)
    tile_r=4 if splits>1 else 8
    parts=torch.empty((nf,rows,splits,rhs),device=x.device,dtype=x.dtype)
    _multiply[(triton.cdiv(rows,tile_r),nf,splits)](factor,torch.view_as_real(x.contiguous()),
        torch.view_as_real(parts),rows,cols,rhs,splits,tile_r,tile_k,num_warps=4)
    return parts[:,:,0] if splits==1 else parts.sum(dim=2)


@triton.jit
def _current_blocks(A,S,X,Y,ROWS:tl.constexpr,COLS:tl.constexpr,RHS:tl.constexpr,
                    TILE_R:tl.constexpr,TILE_K:tl.constexpr):
    f=tl.program_id(1)
    row=tl.program_id(0)*TILE_R+tl.arange(0,TILE_R)
    k=tl.arange(0,TILE_K)
    offset=2*((3*f*ROWS+row[:,None])*COLS+k[None,:])
    step=2*ROWS*COLS
    valid=(row[:,None]<ROWS)&(k[None,:]<COLS)
    ar=tl.load(A+offset,mask=valid,other=0.).to(tl.float32)
    ai=tl.load(A+offset+1,mask=valid,other=0.).to(tl.float32)
    br=tl.load(A+offset+step,mask=valid,other=0.).to(tl.float32)
    bi=tl.load(A+offset+step+1,mask=valid,other=0.).to(tl.float32)
    cr=tl.load(A+offset+2*step,mask=valid,other=0.).to(tl.float32)
    ci=tl.load(A+offset+2*step+1,mask=valid,other=0.).to(tl.float32)
    scale=tl.load(S+f)
    for h in tl.static_range(RHS):
        j=2*((f*2*COLS+k)*RHS+h)
        m=j+2*COLS*RHS
        jr=tl.load(X+j,mask=k<COLS,other=0.)
        ji=tl.load(X+j+1,mask=k<COLS,other=0.)
        mr=tl.load(X+m,mask=k<COLS,other=0.)
        mi=tl.load(X+m+1,mask=k<COLS,other=0.)
        top_re=tl.sum(ar*jr[None,:]-ai*ji[None,:]+br*mr[None,:]-bi*mi[None,:],axis=1)
        top_im=tl.sum(ar*ji[None,:]+ai*jr[None,:]+br*mi[None,:]+bi*mr[None,:],axis=1)
        bot_re=tl.sum(cr*jr[None,:]-ci*ji[None,:]-ar*mr[None,:]+ai*mi[None,:],axis=1)
        bot_im=tl.sum(cr*ji[None,:]+ci*jr[None,:]-ar*mi[None,:]-ai*mr[None,:],axis=1)
        top=2*((f*2*ROWS+row)*RHS+h)
        bot=top+2*ROWS*RHS
        tl.store(Y+top,scale*top_re,mask=row<ROWS)
        tl.store(Y+top+1,scale*top_im,mask=row<ROWS)
        tl.store(Y+bot,scale*bot_re,mask=row<ROWS)
        tl.store(Y+bot+1,scale*bot_im,mask=row<ROWS)


def current_product(a,scale,x):
    nf,blocks,rows,cols,pair=a.shape
    rhs=x.shape[-1]
    if (blocks!=3 or pair!=2 or x.shape[:2]!=(nf,2*cols) or rhs not in (1,2)
            or scale.numel()!=nf or not a.is_contiguous() or a.dtype!=torch.float16
            or scale.dtype!=torch.float32 or x.dtype!=torch.complex64):
        raise ValueError('Invalid packed current symbol or RHS')
    y=torch.empty((nf,2*rows,rhs),device=x.device,dtype=x.dtype)
    # Bound the contiguous input scratch for strided resident FFT spectra.
    for start in range(0,nf,4096):
        stop=min(nf,start+4096)
        _current_blocks[(triton.cdiv(rows,8),stop-start)](a[start:stop],scale[start:stop].contiguous(),
            torch.view_as_real(x[start:stop].contiguous()),torch.view_as_real(y[start:stop]),
            rows,cols,rhs,8,triton.next_power_of_2(cols),num_warps=4)
    return y
