"""Bounded, double-buffered CPU/GPU storage for immutable spectral coefficients.

Cross factors use the same per-frequency FP16 encoding as the resident action.
Only storage changes; FFT sizes, ranks, arithmetic and validation are unchanged.
"""
import json
import os
from pathlib import Path
import numpy as np


def half_factors(values, directory, name):
    """Build a reusable mmap encoding without making a full anonymous copy."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    source=Path(values.filename).resolve() if isinstance(values,np.memmap) else None
    signature=dict(shape=list(values.shape),dtype=str(values.dtype))
    if source is not None:
        stat=source.stat()
        signature.update(source=str(source),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    metadata=directory/f'{name}.json';data=directory/f'{name}.npy';scales=directory/f'{name}_scale.npy'
    if source is not None and metadata.exists() and data.exists() and scales.exists():
        if json.loads(metadata.read_text())==signature:
            return np.load(data,mmap_mode='c'),np.load(scales,mmap_mode='c')
    temporary=directory/f'{name}.{os.getpid()}.tmp.npy'
    encoded=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float16,shape=(*values.shape,2))
    scale=np.empty((len(values),1,1),np.float32)
    for first in range(0,len(values),16):
        value=np.asarray(values[first:first+16])
        norm=np.max(np.abs(value),axis=(1,2),keepdims=True).astype(np.float32)
        norm=np.where(norm>0,norm,1);scale[first:first+len(value)]=norm
        encoded[first:first+len(value),...,0]=value.real/norm
        encoded[first:first+len(value),...,1]=value.imag/norm
    encoded.flush();del encoded;temporary.replace(data)
    np.save(scales,scale)
    tempmeta=metadata.with_suffix('.tmp');tempmeta.write_text(json.dumps(signature)+'\n');tempmeta.replace(metadata)
    return np.load(data,mmap_mode='c'),np.load(scales,mmap_mode='c')


class StreamedSpectralBlock:
    """Two bounded pinned/device slots, with H2D copies overlapping products."""
    def __init__(self, arrays, *, device, batch, half=False, half_matrix=False):
        import torch
        if torch.device(device).type!='cuda':raise ValueError('Streaming requires CUDA')
        if half and half_matrix:raise ValueError('Select factored or dense half storage')
        self.arrays=arrays;self.device=device;self.batch=batch;self.half=half;self.half_matrix=half_matrix
        self.nf=len(arrays[0]);self.copy_stream=torch.cuda.Stream(device=device)
        self.slots=[]
        for _ in range(2):
            host=[torch.empty((min(batch,self.nf),*a.shape[1:]),dtype=torch.from_numpy(a[:1]).dtype,pin_memory=True) for a in arrays]
            gpu=[torch.empty_like(a,device=device) for a in host]
            self.slots.append(dict(host=host,gpu=gpu,ready=torch.cuda.Event(),consumed=torch.cuda.Event(),used=False))
        self.bytes=sum(t.numel()*t.element_size() for slot in self.slots for t in slot['gpu'])
        self.host_bytes=sum(t.numel()*t.element_size() for slot in self.slots for t in slot['host'])
        self.transfer_bytes=sum(a.nbytes for a in arrays)

    def __call__(self, spectrum, rows):
        import torch
        if len(spectrum)!=self.nf:raise ValueError('Spectral frequency count mismatch')
        current=torch.cuda.current_stream(self.device)
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
        count=load(self.slots[0],0)
        for number,first in enumerate(range(0,self.nf,self.batch)):
            slot=self.slots[number%2];next_first=first+count
            if next_first<self.nf:next_count=load(self.slots[(number+1)%2],next_first)
            current.wait_event(slot['ready']);data=[a[:count] for a in slot['gpu']]
            if self.half:
                left,right=[torch.view_as_complex(a.float()) for a in data[:2]]
                result[first:first+count]=torch.bmm(left,torch.bmm(right,spectrum[first:first+count]))*data[2]
            elif self.half_matrix:
                symbol=torch.view_as_complex(data[0].float())
                result[first:first+count]=torch.bmm(symbol,spectrum[first:first+count])*data[1]
            else:torch.bmm(data[0],spectrum[first:first+count],out=result[first:first+count])
            slot['consumed'].record(current);slot['used']=True
            if next_first<self.nf:count=next_count
        return result
