"""Small typed ctypes bridge to the documented FMM3D Fortran vector ABI."""
import ctypes as ct
from pathlib import Path
from types import SimpleNamespace
import numpy as np

class NativeFMM:
    def __init__(self,path=None):
        self.path=Path(path or Path(__file__).resolve().parent/'FMM3D/lib/libfmm3d.so')
        self.lib=ct.CDLL(str(self.path))
        self.fn=self.lib.hfmm3d_s_c_g_vec_
        self.fn.argtypes=[ct.c_void_p]*9;self.fn.restype=None
        self.target_fn=self.lib.hfmm3d_t_c_g_vec_
        self.target_fn.argtypes=[ct.c_void_p]*11;self.target_fn.restype=None
    def targets(self,*,eps,zk,sources,charges,targets):
        sources=np.asfortranarray(sources,dtype=np.float64);charges=np.asfortranarray(charges,dtype=np.complex128)
        targets=np.asfortranarray(targets,dtype=np.float64)
        if sources.ndim!=2 or sources.shape[0]!=3 or targets.ndim!=2 or targets.shape[0]!=3 or charges.ndim!=2 or charges.shape[1]!=sources.shape[1]:raise ValueError('FMM target shape')
        nd=charges.shape[0];count=targets.shape[1]
        n=ct.c_int64(sources.shape[1]);nt=ct.c_int64(count);dens=ct.c_int64(nd);tol=ct.c_double(eps);ier=ct.c_int64(-1)
        wave=np.array([zk],complex);pot=np.empty((nd,count),complex,order='F');grad=np.empty((nd,3,count),complex,order='F')
        ptr=lambda a:a.ctypes.data_as(ct.c_void_p)
        self.target_fn(ct.byref(dens),ct.byref(tol),ptr(wave),ct.byref(n),ptr(sources),ptr(charges),ct.byref(nt),ptr(targets),ptr(pot),ptr(grad),ct.byref(ier))
        return SimpleNamespace(pot=pot,grad=grad,ier=ier.value)
    def hfmm3d(self,*,eps,zk,sources,charges,pg,nd):
        if pg!=2:raise ValueError('native Dirac bridge computes potential and gradient')
        sources=np.asfortranarray(sources,dtype=np.float64)
        charges=np.asfortranarray(charges,dtype=np.complex128)
        if sources.ndim!=2 or sources.shape[0]!=3 or charges.shape!=(nd,sources.shape[1]):raise ValueError('FMM shape')
        count=sources.shape[1];n=ct.c_int64(count);dens=ct.c_int64(nd);tol=ct.c_double(eps);ier=ct.c_int64(-1)
        wave=np.array([zk],np.complex128);pot=np.empty((nd,count),np.complex128,order='F');grad=np.empty((nd,3,count),np.complex128,order='F')
        ptr=lambda a:a.ctypes.data_as(ct.c_void_p)
        self.fn(ct.byref(dens),ct.byref(tol),ptr(wave),ct.byref(n),ptr(sources),ptr(charges),ptr(pot),ptr(grad),ct.byref(ier))
        return SimpleNamespace(pot=pot,grad=grad,ier=ier.value)
