"""Experimental GPU FMM adapter with the same density/output layout as FMM3D."""
import ctypes as ct
from pathlib import Path
from types import SimpleNamespace
import numpy as np

class GPUFMM:
    def __init__(self,sources,targets,k,digits=6,leaf=128,depth=0):
        self.p=None
        self.lib=ct.CDLL(str(Path(__file__).with_name('libgpu_fmm.so')))
        self.lib.bem_gpu_create.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_double,ct.c_double,ct.c_int,ct.c_int,ct.c_int]
        self.lib.bem_gpu_create.restype=ct.c_void_p
        self.lib.bem_gpu_destroy.argtypes=[ct.c_void_p]
        self.lib.bem_gpu_apply.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p]
        self.lib.bem_gpu_info.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.bem_gpu_error.restype=ct.c_char_p
        s=np.asfortranarray(sources,dtype=float);t=np.asfortranarray(targets,dtype=float)
        if s.ndim!=2 or t.ndim!=2 or s.shape[0]!=3 or t.shape[0]!=3:raise ValueError('source/target layout')
        self.capacity=3 if depth else 4
        self.ns=s.shape[1];self.nt=t.shape[1];self.p=self.lib.bem_gpu_create(s.ctypes.data,self.ns,t.ctypes.data,self.nt,complex(k).real,complex(k).imag,digits,leaf,depth)
        if not self.p:raise RuntimeError(self.lib.bem_gpu_error().decode())
        info=np.empty(7,np.int64);self.lib.bem_gpu_info(self.p,info.ctypes.data)
        self.info=dict(zip(('sources','targets','nodes','directions','order','device_bytes','actual_depth'),map(int,info)))
    def apply(self,charges):
        if not self.p:raise RuntimeError('GPU FMM plan is closed')
        q=np.asfortranarray(charges,dtype=complex)
        if q.ndim!=2 or q.shape[1]!=self.ns or not 1<=q.shape[0]<=16:raise ValueError('charge layout')
        nd=q.shape[0]
        p=np.empty((nd,self.nt),complex,order='F');g=np.empty((nd,3,self.nt),complex,order='F')
        for first in range(0,nd,self.capacity):
            last=min(nd,first+self.capacity);part=np.asfortranarray(q[first:last])
            pp=np.empty((last-first,self.nt),complex,order='F');gg=np.empty((last-first,3,self.nt),complex,order='F')
            if self.lib.bem_gpu_apply(self.p,part.ctypes.data,last-first,pp.ctypes.data,gg.ctypes.data):raise RuntimeError(self.lib.bem_gpu_error().decode())
            p[first:last]=pp;g[first:last]=gg
        info=np.empty(7,np.int64);self.lib.bem_gpu_info(self.p,info.ctypes.data)
        self.info.update(dict(zip(('sources','targets','nodes','directions','order','device_bytes','actual_depth'),map(int,info))))
        return SimpleNamespace(pot=p,grad=g,ier=0)
    def close(self):
        if self.p:self.lib.bem_gpu_destroy(self.p);self.p=None
