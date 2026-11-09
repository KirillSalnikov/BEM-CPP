"""Finite-face FFT with 2D parallel blocks and verified point-group reuse.

Experimental. This module leaves production binaries and checkpoints alone.
All near corrections and the mass matrix still come from the native operator.
"""
import time
from dataclasses import replace
import numpy as np
from scipy.fft import next_fast_len,fftn,ifftn
from galerkin_cross_fft import CrossGalerkinFFT
from closed_galerkin_fft import ClosedGalerkinFFT
from surface_symmetry import PointGroup

class ParallelGalerkinFFT:
    def __init__(self,fixture,target,source,ka,ri):
        if not np.allclose(target.transverse,source.transverse,rtol=0,atol=1e-14) or not np.allclose(target.common,source.common,rtol=0,atol=1e-14):
            raise ValueError('2D convolution requires identical lattice axes')
        self.target,self.source,self.ka,self.ri=target,source,ka,ri
        self.fixture=fixture;self.nfx=next_fast_len(target.nx+source.nx-1);self.nfy=next_fast_len(target.ny+source.ny-1)
        self.tc=12*len(target.patterns);self.sc=12*len(source.patterns)

    def build(self):
        start=time.perf_counter();t,s=self.target,self.source
        dx=np.arange(-(s.nx-1),t.nx);dy=np.arange(-(s.ny-1),t.ny)
        # Contract local quadrature only once per displacement, not per cell pair.
        shifted=replace(t,origin=t.origin-(s.nx-1)*t.transverse,nx=len(dx))
        single=replace(s,nx=1)
        helper=CrossGalerkinFFT(self.fixture,shifted,single,self.ka,self.ri)
        spatial=np.zeros((self.nfx,self.nfy,self.tc,self.sc),complex)
        for p,pt in enumerate(t.patterns):
            for q,ps in enumerate(s.patterns):
                # Bound temporary Green/Hessian storage even on larger grids.
                for first in range(0,len(dy),8):
                    d=dy[first:first+8];blocks,_=helper.projected_blocks(pt,ps,d)
                    k1,ke,km=(v[:,0] for v in blocks)
                    ix=(dx%self.nfx)[:,None];iy=(d%self.nfy)[None,:];a=p*6;b=q*6
                    spatial[ix,iy,a:a+6,b:b+6]=1j/self.ka*k1
                    spatial[ix,iy,a+self.tc//2:a+self.tc//2+6,b+self.sc//2:b+self.sc//2+6]=-1j/self.ka*k1
                    spatial[ix,iy,a:a+6,b+self.sc//2:b+self.sc//2+6]=ke
                    spatial[ix,iy,a+self.tc//2:a+self.tc//2+6,b:b+6]=km
        self.symbol=fftn(spatial,axes=(0,1));self.preparation_s=time.perf_counter()-start
        return self

    def apply(self,x):
        s,t=self.source,self.target
        shaped=np.asarray(x).reshape(2,s.nx,self.sc//2,s.ny).transpose(0,2,1,3).reshape(self.sc,s.nx,s.ny)
        transformed=fftn(shaped,s=(self.nfx,self.nfy),axes=(1,2)).transpose(1,2,0)
        spectral=np.einsum('xyij,xyj->xyi',self.symbol,transformed)
        result=ifftn(spectral,axes=(0,1))[:t.nx,:t.ny]
        return result.reshape(t.nx,t.ny,2,self.tc//2).transpose(2,0,3,1).reshape(2*t.nx*(self.tc//2),t.ny)

def prism_group():
    result=[]
    for horizontal in (False,True):
        for vertical in (False,True):
            for j in range(6):
                a=j*np.pi/3;c,s=np.cos(a),np.sin(a)
                rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
                reflection=np.diag([-1. if vertical else 1.,1.,-1. if horizontal else 1.])
                result.append((rotation@reflection,(-1)**(int(horizontal)+int(vertical)),j,vertical,horizontal))
    return result

class SymmetricGalerkinFFT(ClosedGalerkinFFT):
    def __init__(self,geometry,progress=None,*,symmetry=True,two_dimensional=True,cross_factory=None,block_factory=None,group=None):
        self.geometry=geometry;self.canonical=[];self.aliases=[];self.local=None
        self.symmetry=symmetry;self.two_dimensional=two_dimensional
        self.group_checks=[];start=time.perf_counter();assigned=set()
        normals=np.array([geometry.normals[f[0]] for f in geometry.faces]);face_count=len(geometry.faces)
        if group is not None and not symmetry:raise ValueError('Explicit group conflicts with symmetry=False')
        if group is None:
            from regular_surface_geometry import RegularSurfaceGeometry
            operations=prism_group() if symmetry and not isinstance(geometry,RegularSurfaceGeometry) else [(np.eye(3),1,0,False,False)]
            group=PointGroup([row[0] for row in operations])
        else:
            operations=[(R,int(det),i,None,None) for i,(R,det) in enumerate(zip(group.matrices,group.determinants))]
        face_maps=group.face_maps(geometry)
        self.point_group=group;self.face_maps=face_maps
        for t in range(face_count):
            for s in range(face_count):
                if (t,s) in assigned:continue
                parallel=np.linalg.norm(np.cross(normals[t],normals[s]))<1e-10
                u,v,c=geometry.axes(t,s)
                use_2d=parallel and two_dimensional
                if use_2d:
                    try:geometry.layout(s,u,c);v=u.copy()
                    except ValueError:use_2d=False
                target=geometry.layout(t,u,c);source=geometry.layout(s,v,c)
                kind=ParallelGalerkinFFT if use_2d else (cross_factory or CrossGalerkinFFT)
                op=(block_factory(geometry.fixture,target['face'],source['face'],geometry.meta['ka'],complex(*geometry.meta['ri']),use_2d) if block_factory else kind(geometry.fixture,target['face'],source['face'],geometry.meta['ka'],complex(*geometry.meta['ri'])).build())
                index=len(self.canonical);self.canonical.append(op);aliases=[]
                for operation,(R,det,j,vertical,horizontal) in enumerate(operations):
                    tt,ss=face_maps[operation,[t,s]]
                    if (tt,ss) in assigned:continue
                    if max(np.linalg.norm(normals[tt]-R@normals[t]),np.linalg.norm(normals[ss]-R@normals[s]))>1e-10:raise ValueError('Face group map')
                    lt=geometry.layout(tt,R@u,R@c);ls=geometry.layout(ss,R@v,R@c)
                    for old,new in ((target,lt),(source,ls)):
                        f,h=old['face'],new['face']
                        defect=np.linalg.norm(h.origin-group.transform(f.origin,operation))
                        if old['shape']!=new['shape'] or f.patterns!=h.patterns or defect>1e-10:
                            raise ValueError(f'Point-group lattice mismatch: {t,s}->{tt,ss}, {defect}')
                    # Under det(R)=-1, K1 flips sign; K2eps/K2mu do not.
                    # diag(1,det) A diag(det,1) has precisely this property.
                    source_sign=ls['signs']*np.where(ls['global_']<geometry.n,det,1)
                    target_sign=lt['signs']*np.where(lt['global_']>=geometry.n,det,1)
                    a=dict(target=lt,source=ls,source_sign=source_sign,target_sign=target_sign,
                           faces=(tt,ss),det=det,rotation=j,vertical=vertical,horizontal=horizontal)
                    aliases.append(a);self.aliases.append((index,a));assigned.add((tt,ss))
                self.group_checks.append(dict(canonical=(t,s),pair_count=len(aliases),two_dimensional=isinstance(op,ParallelGalerkinFFT),
                                              bytes=op.symbol.nbytes if op.symbol is not None else op.factor_bytes,reflected=sum(a['det']==-1 for a in aliases)))
                if progress:progress(len(assigned),sum(x.symbol.nbytes if x.symbol is not None else x.factor_bytes for x in self.canonical))
        if len(assigned)!=face_count**2:raise ValueError('Incomplete face coverage')
        self.preparation_s=time.perf_counter()-start
        self.regular_bytes=sum(x.symbol.nbytes if x.symbol is not None else x.factor_bytes for x in self.canonical)

    def apply(self,x):
        y=np.zeros(self.geometry.N,complex)
        for index,a in self.aliases:
            op=self.canonical[index];s=a['source'];t=a['target']
            z=np.zeros(np.prod(s['shape']),complex);z[s['local']]=a['source_sign']*x[s['global_']]
            result=op.apply(z.reshape(s['shape'])).ravel()
            np.add.at(y,t['global_'],a['target_sign']*result[t['local']])
        if self.local is not None:y+=self.local@x
        return y

    def torch_operator(self,device='cuda',dtype=None,*,capture=False):
        import torch
        operator=TorchSymmetricAction(self,device,torch.complex128 if dtype is None else dtype)
        return TorchCapturedAction(operator) if capture else operator

def torch_local_matrix(matrix, device, dtype, storage):
    """Create the local sparse operator and count actual index/value bytes."""
    import torch
    if storage not in ('coo', 'csr'):
        raise ValueError('Local storage must be coo or csr')
    csr = matrix.tocsr(copy=False)
    if not csr.has_canonical_format:
        csr.sum_duplicates()
    if not csr.has_sorted_indices:
        csr.sort_indices()
    values = torch.as_tensor(csr.data, device=device, dtype=dtype)
    if storage == 'csr':
        index_dtype = (torch.int32 if max(*csr.shape, csr.nnz) <= np.iinfo(np.int32).max
                       else torch.int64)
        rows = torch.as_tensor(csr.indptr, device=device, dtype=index_dtype)
        columns = torch.as_tensor(csr.indices, device=device, dtype=index_dtype)
        local = torch.sparse_csr_tensor(rows, columns, values, size=csr.shape)
        size = sum(a.numel()*a.element_size() for a in (rows, columns, values))
    else:
        coo = csr.tocoo(copy=False)
        indices = torch.as_tensor(np.stack((coo.row, coo.col)), device=device)
        local = torch.sparse_coo_tensor(indices, values, csr.shape, is_coalesced=True)
        size = local._nnz()*(16+local.element_size())
    return local, size


class TorchSymmetricAction:
    """One resident symbol per orbit, with face aliases as multiple RHS."""
    def __init__(self,cpu,device,dtype,*,factors=None,factor_storage='full',local_storage='coo'):
        import torch
        if factor_storage not in ('full','half'):raise ValueError('Invalid factor storage')
        if factor_storage=='half' and dtype!=torch.complex64:raise ValueError('Half factor storage uses FP32 arithmetic')
        def encode_half(values):
            encoded=torch.empty((*values.shape,2),device=device,dtype=torch.float16)
            scales=np.empty((len(values),1,1),np.float32)
            for first in range(0,len(values),16):
                value=np.asarray(values[first:first+16]);scale=np.max(np.abs(value),axis=(1,2),keepdims=True).astype(np.float32)
                scale=np.where(scale>0,scale,1);half=np.empty((*value.shape,2),np.float16)
                half[...,0]=value.real/scale;half[...,1]=value.imag/scale
                encoded[first:first+len(value)].copy_(torch.from_numpy(half));scales[first:first+len(value)]=scale
            return encoded,torch.as_tensor(scales,device=device)
        self.device,self.dtype,self.N=device,dtype,cpu.geometry.N;self.groups=[];self.bytes=0
        for index,op in enumerate(cpu.canonical):
            aliases=[a for i,a in cpu.aliases if i==index];count=len(aliases)
            sc,sny=aliases[0]['source']['shape'];tc,tny=aliases[0]['target']['shape']
            indices=np.zeros((count,sc*sny),np.int64);signs=np.zeros_like(indices,dtype=float)
            target_local=[];target_global=[];target_sign=[]
            for j,a in enumerate(aliases):
                s,t=a['source'],a['target'];indices[j,s['local']]=s['global_'];signs[j,s['local']]=a['source_sign']
                target_local.extend(j*tc*tny+t['local']);target_global.extend(t['global_']);target_sign.extend(a['target_sign'])
            group=dict(op=op,count=count,sc=sc,sny=sny,tc=tc,tny=tny,
                indices=torch.as_tensor(indices.reshape(count,sc,sny),device=device),signs=torch.as_tensor(signs.reshape(count,sc,sny),device=device,dtype=dtype),
                target_local=torch.as_tensor(np.array(target_local),device=device),target_global=torch.as_tensor(np.array(target_global),device=device),
                target_sign=torch.as_tensor(np.array(target_sign),device=device,dtype=dtype))
            if factors is not None and index in factors:
                if isinstance(op,ParallelGalerkinFFT):raise ValueError('Only cross-face symbols support low-rank factors')
                u,v=factors[index]
                if u.shape[0]!=op.nfft or v.shape[0]!=op.nfft or u.shape[1]!=2*op.tc or v.shape[2]!=2*op.sc or u.shape[2]!=v.shape[1]:raise ValueError('Invalid cross-symbol factors')
                if factor_storage=='half':
                    left,ls=encode_half(u);right,rs=encode_half(v);group.update(left_half=left,right_half=right,factor_scale=ls*rs)
                else:group.update(left=torch.as_tensor(u,device=device,dtype=dtype),right=torch.as_tensor(v,device=device,dtype=dtype))
            else:group['symbol']=torch.as_tensor(op.symbol,device=device,dtype=dtype)
            self.bytes+=sum(v.numel()*v.element_size() for v in group.values() if torch.is_tensor(v));self.groups.append(group)
        self.local,self.local_bytes=torch_local_matrix(cpu.local,device,dtype,local_storage)
        self.local_storage=local_storage
        self.bytes+=self.local_bytes

    def __call__(self,x):
        import torch
        y=torch.zeros(self.N,device=self.device,dtype=self.dtype)
        for g in self.groups:
            op=g['op'];count=g['count'];values=x[g['indices']]*g['signs']
            if isinstance(op,ParallelGalerkinFFT):
                values=values.reshape(count,2,op.source.nx,op.sc//2,op.source.ny).permute(0,1,3,2,4).reshape(count,op.sc,op.source.nx,op.source.ny)
                spectrum=torch.fft.fftn(values,s=(op.nfx,op.nfy),dim=(2,3)).permute(2,3,1,0).reshape(op.nfx*op.nfy,op.sc,count)
                result=torch.bmm(g['symbol'].reshape(op.nfx*op.nfy,op.tc,op.sc),spectrum).reshape(op.nfx,op.nfy,op.tc,count)
                values=torch.fft.ifftn(result,dim=(0,1))[:op.target.nx,:op.target.ny]
                values=values.reshape(op.target.nx,op.target.ny,2,op.tc//2,count).permute(4,2,0,3,1).reshape(-1)
            else:
                spectrum=torch.fft.fft(values,n=op.nfft,dim=2).permute(2,1,0)
                if 'left_half' in g:
                    # Decode only 16 frequencies at a time; full-factor decoding
                    # would erase the memory saving on the production mesh.
                    result=torch.empty((op.nfft,2*op.tc,count),device=self.device,dtype=self.dtype)
                    for first in range(0,op.nfft,16):
                        last=min(first+16,op.nfft)
                        left=torch.view_as_complex(g['left_half'][first:last].float())
                        right=torch.view_as_complex(g['right_half'][first:last].float())
                        result[first:last]=torch.bmm(left,torch.bmm(right,spectrum[first:last]))*g['factor_scale'][first:last]
                else:result=torch.bmm(g['left'],torch.bmm(g['right'],spectrum)) if 'left' in g else torch.bmm(g['symbol'],spectrum)
                values=torch.fft.ifft(result,dim=0)[:op.target.ny].permute(2,1,0).reshape(-1)
            y.index_add_(0,g['target_global'],g['target_sign']*values[g['target_local']])
        return y+torch.sparse.mm(self.local,x[:,None])[:,0]

class TorchCapturedAction:
    """Replay a fixed-shape operator with CUDA Graphs; coefficients unchanged.

    Input/output buffers are owned by this wrapper. The returned device result
    is overwritten on the next call; clients must consume it before replaying.
    """
    def __init__(self,operator):
        import torch
        if torch.device(operator.device).type!='cuda':raise ValueError('CUDA Graph requires CUDA')
        self.operator=operator;self.device=operator.device;self.dtype=operator.dtype;self.N=operator.N
        self.input=torch.zeros(self.N,device=self.device,dtype=self.dtype)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.output=operator(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.output=operator(self.input)
        self.bytes=operator.bytes+self.input.numel()*self.input.element_size()+self.output.numel()*self.output.element_size()

    def __call__(self,x):
        if x.shape!=self.input.shape or x.dtype!=self.dtype or x.device!=self.input.device:
            raise ValueError('Captured operator input shape, dtype or device mismatch')
        self.input.copy_(x);self.graph.replay();return self.output
