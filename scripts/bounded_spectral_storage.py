"""Bounded, double-buffered CPU/GPU storage for immutable spectral coefficients.

Cross factors use the same per-frequency FP16 encoding as the resident action.
Only storage changes; FFT sizes, ranks, arithmetic and validation are unchanged.
"""
import json
import os
from pathlib import Path
import numpy as np


class SharedStreamBuffers:
    """One double buffer for sequential spectral blocks of different shapes.

    Register every block before the first action. Events remain attached to
    the underlying slots when views change, so a later block cannot overwrite
    coefficients still used by a preceding CUDA product.
    """
    def __init__(self,device):
        import torch
        self.device=device;self.capacity=0;self.slots=None
        self.copy_stream=torch.cuda.Stream(device=device)

    def register(self,arrays,batch):
        if self.slots is not None:raise RuntimeError('Register all streamed blocks before the first action')
        offset=0;spec=[]
        for a in arrays:
            shape=(min(batch,len(a)),*a.shape[1:]);count=int(np.prod(shape));offset=(offset+255)//256*256
            spec.append((offset,count,shape,a.dtype));offset+=count*a.dtype.itemsize
        self.capacity=max(self.capacity,offset)
        return spec

    @property
    def bytes(self):return 2*self.capacity

    @property
    def host_bytes(self):return self.bytes

    def views(self,spec):
        import torch
        if self.slots is None:
            self.slots=[dict(host_raw=torch.empty(self.capacity,dtype=torch.uint8,pin_memory=True),
                             gpu_raw=torch.empty(self.capacity,dtype=torch.uint8,device=self.device),
                             ready=torch.cuda.Event(),consumed=torch.cuda.Event(),used=False) for _ in range(2)]
            # The allocator may reuse scratch whose last operation is still
            # queued on the allocating stream (e.g. the preceding FFT). Its
            # first H2D write must not race those operations on copy_stream.
            self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
            for slot in self.slots:slot['gpu_raw'].record_stream(self.copy_stream)
        for slot in self.slots:
            slot['gpu_raw'].record_stream(torch.cuda.current_stream(self.device))
            for side in ('host','gpu'):
                slot[side]=[slot[side+'_raw'][offset:offset+count*dtype.itemsize].view(
                    torch.from_numpy(np.empty(0,dtype=dtype)).dtype).reshape(shape)
                    for offset,count,shape,dtype in spec]
        return self.slots


def half_factors(values, directory, name,*,packed_currents=False):
    """Build a reusable mmap encoding without making a full anonymous copy."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    source=Path(values.filename).resolve() if isinstance(values,np.memmap) else None
    signature=dict(shape=list(values.shape),dtype=str(values.dtype))
    if packed_currents:
        if values.ndim!=3 or values.shape[1]%2 or values.shape[2]%2:
            raise ValueError('Current-block symbol must have even row and column counts')
        signature['packing']='three_current_blocks'
    if source is not None:
        stat=source.stat()
        signature.update(source=str(source),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    metadata=directory/f'{name}.json';data=directory/f'{name}.npy';scales=directory/f'{name}_scale.npy'
    if source is not None and metadata.exists() and data.exists() and scales.exists():
        if json.loads(metadata.read_text())==signature:
            return np.load(data,mmap_mode='c'),np.load(scales,mmap_mode='c')
    temporary=directory/f'{name}.{os.getpid()}.tmp.npy'
    encoded_shape=((len(values),3,values.shape[1]//2,values.shape[2]//2,2)
                   if packed_currents else (*values.shape,2))
    encoded=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float16,shape=encoded_shape)
    scale=np.empty((len(values),1,1),np.float32)
    for first in range(0,len(values),16):
        value=np.asarray(values[first:first+16])
        if packed_currents:
            nr,nc=value.shape[1]//2,value.shape[2]//2
            a=value[:,:nr,:nc]
            if not np.array_equal(value[:,nr:,nc:],-a):raise ValueError('Current diagonal blocks are not exact negatives')
            value=np.stack((a,value[:,:nr,nc:],value[:,nr:,:nc]),axis=1)
        norm=np.max(np.abs(value),axis=tuple(range(1,value.ndim)),keepdims=True).astype(np.float32)
        norm=np.where(norm>0,norm,1);scale[first:first+len(value)]=norm.reshape(-1,1,1)
        encoded[first:first+len(value),...,0]=value.real/norm
        encoded[first:first+len(value),...,1]=value.imag/norm
    encoded.flush();del encoded;temporary.replace(data)
    np.save(scales,scale)
    tempmeta=metadata.with_suffix('.tmp');tempmeta.write_text(json.dumps(signature)+'\n');tempmeta.replace(metadata)
    return np.load(data,mmap_mode='c'),np.load(scales,mmap_mode='c')


def decode_half_symbol(encoded):
    """Restore a full matrix in bounded scratch, sharing the diagonal block."""
    import torch
    value=torch.view_as_complex(encoded.float())
    if value.ndim==3:return value
    if value.ndim!=4 or value.shape[1]!=3:raise ValueError('Invalid packed current symbol')
    top=torch.cat((value[:,0],value[:,1]),dim=2)
    bottom=torch.cat((value[:,2],-value[:,0]),dim=2)
    return torch.cat((top,bottom),dim=1)


def exact_factor_batch_ranks(left,right,batch,cache_path=None):
    """Trim ONLY exact zero outer-product tails, retaining complete batches.

    Zero padding survives half encoding exactly. No SVD tolerance, norm cutoff
    or assumption about a recompression receipt is used here.
    """
    if batch<1 or left.ndim!=4 or right.ndim!=4 or left.shape[-1]!=2 or right.shape[-1]!=2 or \
       left.shape[0]!=right.shape[0] or left.shape[2]!=right.shape[1] or not left.shape[2]:
        raise ValueError('Invalid encoded factor dimensions/batch')
    signature=None;maximum=left.shape[2]
    if cache_path is not None and all(isinstance(a,np.memmap) for a in (left,right)):
        def identity(a):
            source=Path(a.filename).resolve();stat=source.stat()
            return dict(source=str(source),shape=list(a.shape),dtype=str(a.dtype),
                        size=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns,inode=stat.st_ino)
        signature=dict(format=1,batch=batch,left=identity(left),right=identity(right))
        cache_path=Path(cache_path)
        if cache_path.exists():
            saved=json.loads(cache_path.read_text());ranks=saved.get('ranks',[])
            if saved.get('signature')==signature and len(ranks)==(len(left)+batch-1)//batch and \
               all(type(r) is int and 1<=r<=maximum for r in ranks):return ranks
    ranks=[]
    for first in range(0,len(left),batch):
        if not np.isfinite(left[first:first+batch]).all() or not np.isfinite(right[first:first+batch]).all():
            raise ValueError('Nonfinite encoded spectral factors')
        a=np.any(left[first:first+batch]!=0,axis=(1,3))
        b=np.any(right[first:first+batch]!=0,axis=(2,3))
        active=np.flatnonzero(np.any(a & b,axis=0))
        rank=int(active[-1])+1 if len(active) else 1
        ranks.append(min(maximum,(rank+7)//8*8))
    if signature is not None:
        cache_path.parent.mkdir(exist_ok=True,parents=True)
        temporary=cache_path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(dict(signature=signature,ranks=ranks))+'\n');temporary.replace(cache_path)
    return ranks


def factor_batch_views(arrays,first,count,rank):
    """Physical slices to pack into contiguous pinned/device slot prefixes."""
    left,right,scale=arrays
    if not 1<=rank<=left.shape[2] or first<0 or count<1 or first+count>len(left):
        raise ValueError('Invalid factor batch slice')
    return (left[first:first+count,:,:rank,:],right[first:first+count,:rank,:,:],scale[first:first+count])


class StreamedSpectralBlock:
    """Two bounded pinned/device slots, with H2D copies overlapping products."""
    def __init__(self, arrays, *, device, batch, half=False, half_matrix=False,pool=None,trim_zero_rank=False,rank_cache=None):
        import torch
        if torch.device(device).type!='cuda':raise ValueError('Streaming requires CUDA')
        if half and half_matrix:raise ValueError('Select factored or dense half storage')
        if batch<1 or not arrays or any(len(a)!=len(arrays[0]) for a in arrays):raise ValueError('Invalid streamed batch/arrays')
        self.arrays=arrays;self.device=device;self.batch=batch;self.half=half;self.half_matrix=half_matrix
        self.nf=len(arrays[0]);self.pool=pool;self.pinned_batches=None
        self.transverse=None
        self.direct_panels=False
        if trim_zero_rank and not half:raise ValueError('Zero-rank trimming requires encoded factors')
        self.batch_ranks=exact_factor_batch_ranks(*arrays[:2],batch,cache_path=rank_cache) if trim_zero_rank else None
        self.transfer_bytes=sum(a.nbytes for a in arrays)
        if self.batch_ranks is not None:
            self.transfer_bytes=sum(sum(a.nbytes for a in factor_batch_views(arrays,first,min(batch,self.nf-first),rank))
                                    for first,rank in zip(range(0,self.nf,batch),self.batch_ranks))
        if pool is not None:
            if torch.device(pool.device)!=torch.device(device):raise ValueError('Shared buffers are on another device')
            self.spec=pool.register(arrays,batch);self.copy_stream=pool.copy_stream
            self.bytes=self.host_bytes=0  # Accounted once by the pool owner.
            return
        self.copy_stream=torch.cuda.Stream(device=device)
        self.slots=[]
        for _ in range(2):
            host=[torch.empty((min(batch,self.nf),*a.shape[1:]),dtype=torch.from_numpy(a[:1]).dtype,pin_memory=True) for a in arrays]
            gpu=[torch.empty_like(a,device=device) for a in host]
            self.slots.append(dict(host=host,gpu=gpu,ready=torch.cuda.Event(),consumed=torch.cuda.Event(),used=False))
        self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        for slot in self.slots:
            for value in slot['gpu']:value.record_stream(self.copy_stream)
        self.bytes=sum(t.numel()*t.element_size() for slot in self.slots for t in slot['gpu'])
        self.host_bytes=sum(t.numel()*t.element_size() for slot in self.slots for t in slot['host'])

    def batch_sources(self,first):
        count=min(self.batch,self.nf-first)
        if self.batch_ranks is None:
            return [a[first:first+count] for a in self.arrays]
        return factor_batch_views(self.arrays,first,count,self.batch_ranks[first//self.batch])

    def set_batch_ranks(self,ranks):
        if self.pinned_batches is not None or getattr(self,'transverse',None) or not self.half or len(ranks)!=(self.nf+self.batch-1)//self.batch:
            raise ValueError('Invalid factor rank schedule')
        if any(type(r) is not int or not 1<=r<=self.arrays[0].shape[2] for r in ranks):
            raise ValueError('Invalid retained rank')
        self.batch_ranks=list(ranks)
        self.transfer_bytes=sum(a.nbytes for first in range(0,self.nf,self.batch) for a in self.batch_sources(first))

    def set_transverse_lines(self,target,source,guard=0):
        """Restrict both stored factors before pinning their immutable panels."""
        import torch
        if self.pinned_batches is not None or not self.half or self.batch_ranks is None or getattr(self,'transverse',None):
            raise ValueError('Transverse compression requires a fresh ranked factor block')
        left,right,scale=self.arrays
        if target.nx*target.channels*2!=left.shape[1] or source.nx*source.channels*2!=right.shape[2]:
            raise ValueError('Transverse layout does not match factor dimensions')
        if type(guard) is not int or not 0<=guard<=self.nf//2:
            raise ValueError('Invalid transverse frequency guard')
        self.compact_panels=[target.compact_batch(first,self.batch,self.nf,guard)
                             for first in range(0,self.nf,self.batch)]
        self.transverse=(target,source)
        self.lift=torch.as_tensor(target.matrix,device=self.device)
        self.restrict=torch.as_tensor(source.matrix.T.copy(),device=self.device)
        self.bytes+=self.lift.numel()*self.lift.element_size()+self.restrict.numel()*self.restrict.element_size()
        self.transfer_bytes=scale.nbytes
        for j,(rank,compact) in enumerate(zip(self.batch_ranks,self.compact_panels)):
            dimension=len(target.indices)+len(source.indices) if compact else left.shape[1]+right.shape[2]
            self.transfer_bytes+=min(self.batch,self.nf-j*self.batch)*rank*dimension*2*left.itemsize

    def enable_direct_panels(self):
        """Decode streamed factors in registers for one or two right-hand sides."""
        if not self.half and not (self.half_matrix and self.arrays is not None and self.arrays[0].ndim==5):
            raise ValueError('Direct panels require encoded factors or packed current blocks')
        if self.direct_panels:return
        from spectral_panel_kernel import stencil_table
        if self.transverse:
            target,source=self.transverse
            self.direct_lift=stencil_table(target.matrix,self.device)
            self.direct_restrict=stencil_table(source.matrix.T,self.device)
            self.bytes+=sum(t.numel()*t.element_size() for t in self.direct_lift+self.direct_restrict)
        self.direct_panels=True

    def pin_coefficients(self):
        """Pack immutable coefficient panels once, then DMA directly from them."""
        import torch
        if self.pinned_batches is not None:
            raise ValueError('Spectral coefficients already pinned')
        raw=torch.empty(self.transfer_bytes,dtype=torch.uint8,pin_memory=True)
        batches=[]
        offset=0
        for first in range(0,self.nf,self.batch):
            panels=[]
            sources=self.batch_sources(first)
            if getattr(self,'transverse',None) and self.compact_panels[first//self.batch]:
                left,right,scale=sources
                target,source_lines=self.transverse
                sources=(np.take(left,target.indices,axis=1),np.take(right,source_lines.indices,axis=2),scale)
            for a in sources:
                source=torch.from_numpy(a)
                target=raw[offset:offset+a.nbytes].view(source.dtype).view(a.shape)
                target.copy_(source)
                panels.append(target)
                offset+=a.nbytes
            batches.append(panels)
        if offset!=self.transfer_bytes:
            raise ValueError('Pinned coefficient length mismatch')
        self.pinned_batches=batches
        self.host_bytes+=offset
        self.arrays=None

    def __call__(self, spectrum, rows):
        import torch
        if len(spectrum)!=self.nf:raise ValueError('Spectral frequency count mismatch')
        if self.transverse and self.pinned_batches is None:
            raise RuntimeError('Pin the compact transverse factors before applying them')
        current=torch.cuda.current_stream(self.device)
        slots=self.pool.views(self.spec) if self.pool is not None else self.slots
        if self.pool is None:
            for slot in slots:
                for value in slot['gpu']:value.record_stream(current)
        result=torch.empty((self.nf,rows,spectrum.shape[-1]),dtype=spectrum.dtype,device=self.device)
        def load(slot,first):
            count=min(self.batch,self.nf-first)
            if self.pinned_batches is not None:
                host=self.pinned_batches[first//self.batch]
                gpu=[a.reshape(-1)[:s.numel()].view(s.shape) for a,s in zip(slot['gpu'],host)]
                with torch.cuda.stream(self.copy_stream):
                    if slot['used']:self.copy_stream.wait_event(slot['consumed'])
                    for target,source in zip(gpu,host):target.copy_(source,non_blocking=True)
                    slot['ready'].record(self.copy_stream)
                slot['active_gpu']=gpu
                return count
            if slot['used']:slot['consumed'].synchronize()
            if self.batch_ranks is None:
                sources=[source[first:first+count] for source in self.arrays]
                host=[a[:count] for a in slot['host']];gpu=[a[:count] for a in slot['gpu']]
            else:
                sources=factor_batch_views(self.arrays,first,count,self.batch_ranks[first//self.batch])
                # Pack into contiguous prefixes; sliced rank axes in mmap
                # factors are strided and must not trigger strided H2D copies.
                host=[a.reshape(-1)[:s.size].reshape(s.shape) for a,s in zip(slot['host'],sources)]
                gpu=[a.reshape(-1)[:s.size].reshape(s.shape) for a,s in zip(slot['gpu'],sources)]
            for target,source in zip(host,sources):
                target.copy_(torch.from_numpy(source))
            with torch.cuda.stream(self.copy_stream):
                for target,source in zip(gpu,host):target.copy_(source,non_blocking=True)
                slot['ready'].record(self.copy_stream)
            slot['active_gpu']=gpu
            return count
        count=load(slots[0],0)
        for number,first in enumerate(range(0,self.nf,self.batch)):
            slot=slots[number%2];next_first=first+count
            if next_first<self.nf:next_count=load(slots[(number+1)%2],next_first)
            current.wait_event(slot['ready']);data=slot['active_gpu']
            if self.half:
                source=spectrum[first:first+count]
                direct=self.direct_panels and source.shape[-1]<=2
                if direct:
                    from spectral_panel_kernel import half_product,interpolate
                compact=self.transverse and self.compact_panels[number]
                if compact:
                    target,source_lines=self.transverse
                    source=(interpolate(source,self.direct_restrict,source_lines.channels) if direct else
                            source_lines.apply(source,self.restrict,source_lines.channels))
                if direct:
                    product=half_product(data[0],half_product(data[1],source))*data[2]
                else:
                    left,right=[torch.view_as_complex(a.float()) for a in data[:2]]
                    product=torch.bmm(left,torch.bmm(right,source))*data[2]
                if compact:
                    product=(interpolate(product,self.direct_lift,target.channels) if direct else
                             target.apply(product,self.lift,target.channels))
                result[first:first+count]=product
            elif self.half_matrix:
                if self.direct_panels and spectrum.shape[-1]<=2:
                    from spectral_panel_kernel import current_product
                    result[first:first+count]=current_product(data[0],data[1],spectrum[first:first+count])
                else:
                    symbol=decode_half_symbol(data[0])
                    result[first:first+count]=torch.bmm(symbol,spectrum[first:first+count])*data[1]
            else:torch.bmm(data[0],spectrum[first:first+count],out=result[first:first+count])
            slot['consumed'].record(current);slot['used']=True
            if next_first<self.nf:count=next_count
        return result
