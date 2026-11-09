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


class StreamedSpectralBlock:
    """Two bounded pinned/device slots, with H2D copies overlapping products."""
    def __init__(self, arrays, *, device, batch, half=False, half_matrix=False,pool=None):
        import torch
        if torch.device(device).type!='cuda':raise ValueError('Streaming requires CUDA')
        if half and half_matrix:raise ValueError('Select factored or dense half storage')
        if batch<1 or not arrays or any(len(a)!=len(arrays[0]) for a in arrays):raise ValueError('Invalid streamed batch/arrays')
        self.arrays=arrays;self.device=device;self.batch=batch;self.half=half;self.half_matrix=half_matrix
        self.nf=len(arrays[0]);self.pool=pool
        self.transfer_bytes=sum(a.nbytes for a in arrays)
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

    def __call__(self, spectrum, rows):
        import torch
        if len(spectrum)!=self.nf:raise ValueError('Spectral frequency count mismatch')
        current=torch.cuda.current_stream(self.device)
        slots=self.pool.views(self.spec) if self.pool is not None else self.slots
        if self.pool is None:
            for slot in slots:
                for value in slot['gpu']:value.record_stream(current)
        result=torch.empty((self.nf,rows,spectrum.shape[-1]),dtype=spectrum.dtype,device=self.device)
        def load(slot,first):
            if slot['used']:slot['consumed'].synchronize()
            count=min(self.batch,self.nf-first)
            for target,source in zip(slot['host'],self.arrays):
                target[:count].copy_(torch.from_numpy(source[first:first+count]))
            with torch.cuda.stream(self.copy_stream):
                for target,source in zip(slot['gpu'],slot['host']):target[:count].copy_(source[:count],non_blocking=True)
                slot['ready'].record(self.copy_stream)
            return count
        count=load(slots[0],0)
        for number,first in enumerate(range(0,self.nf,self.batch)):
            slot=slots[number%2];next_first=first+count
            if next_first<self.nf:next_count=load(slots[(number+1)%2],next_first)
            current.wait_event(slot['ready']);data=[a[:count] for a in slot['gpu']]
            if self.half:
                left,right=[torch.view_as_complex(a.float()) for a in data[:2]]
                result[first:first+count]=torch.bmm(left,torch.bmm(right,spectrum[first:first+count]))*data[2]
            elif self.half_matrix:
                symbol=decode_half_symbol(data[0])
                result[first:first+count]=torch.bmm(symbol,spectrum[first:first+count])*data[1]
            else:torch.bmm(data[0],spectrum[first:first+count],out=result[first:first+count])
            slot['consumed'].record(current);slot['used']=True
            if next_first<self.nf:count=next_count
        return result
