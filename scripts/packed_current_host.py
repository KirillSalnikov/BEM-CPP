"""CPU local action overlapped with the GPU regular FFT action."""
import ctypes
import hashlib
import json
import os
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np


class PackedHostCurrentCSR:
    def __init__(self,directory,device='cuda',threads=16):
        import torch
        directory=Path(directory);self.meta=json.loads((directory/'metadata.json').read_text())
        self.n=self.meta['current_dofs'];self.device=device;self.threads=threads
        if threads<1:raise ValueError('CPU thread count must be positive')
        self.rows=np.load(directory/'rows.npy',mmap_mode='r')
        self.columns=np.load(directory/'columns.npy',mmap_mode='r')
        self.values=np.load(directory/'coefficients.npy',mmap_mode='r')
        source=Path(__file__).with_name('packed_current_cpu.cpp')
        digest=hashlib.sha256(source.read_bytes()+platform.node().encode()+platform.machine().encode()).hexdigest()[:20]
        library=directory/f'host_{digest}.so'
        if not library.exists():
            temporary=library.with_name(library.name+f'.{os.getpid()}.tmp')
            subprocess.run(['g++','-O3','-march=native','-std=c++11','-fopenmp','-shared','-fPIC',str(source),'-o',str(temporary)],check=True,capture_output=True)
            temporary.replace(library)
        self.library=ctypes.CDLL(str(library));self.function=self.library.bem_packed_current_cpu
        self.function.argtypes=[ctypes.c_int,*([ctypes.c_void_p]*5),ctypes.c_int];self.function.restype=None
        self.input=torch.empty(2*self.n,dtype=torch.complex64,pin_memory=True)
        self.output=torch.empty_like(self.input,pin_memory=True)
        self.gpu=torch.empty_like(self.input,device=device)
        self.bytes=self.gpu.numel()*self.gpu.element_size()
        self.host_bytes=self.rows.nbytes+self.columns.nbytes+self.values.nbytes+2*self.input.numel()*self.input.element_size()
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='bem-local');self.pending=None
        self.copied=torch.cuda.Event();self.used=False

    def enable_rotation(self,perm,sign):
        from prism_fft_rotation import local_rotation_rows
        self.rotation_rows=local_rotation_rows(perm,sign)
        self.host_bytes+=sum(table.nbytes for table in self.rotation_rows)
        self.rotated=self.library.bem_packed_current_rotated
        self.rotated.argtypes=self.function.argtypes+[ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*3
        self.rotated.restype=None

    def begin(self,x,mode=-1,host_input=None):
        if self.pending is not None:raise RuntimeError('Host local action is already running')
        if x.shape!=self.gpu.shape or x.dtype!=self.gpu.dtype:raise ValueError('Host local input mismatch')
        if self.used:self.copied.synchronize()
        if host_input is None:
            self.input.copy_(x,non_blocking=False)
            host_input=self.input
        elif host_input.device.type!='cpu' or host_input.shape!=x.shape or host_input.dtype!=x.dtype or not host_input.is_contiguous():
            raise ValueError('Invalid CPU mirror of local action input')
        self.input_owner=host_input
        args=(self.n,self.rows.ctypes.data,self.columns.ctypes.data,self.values.ctypes.data,
              host_input.data_ptr(),self.output.data_ptr(),self.threads)
        if mode<0:
            self.pending=self.pool.submit(self.function,*args)
        else:
            reps,owner,ids,phases=self.rotation_rows
            self.pending=self.pool.submit(self.rotated,*args,reps.ctypes.data,len(reps),owner.ctypes.data,
                                         ids.ctypes.data,phases[mode].ctypes.data)

    def finish(self):
        if self.pending is None:raise RuntimeError('Host local action was not started')
        self.pending.result();self.pending=None;self.input_owner=None
        self.gpu.copy_(self.output,non_blocking=True);self.copied.record();self.used=True
        return self.gpu
