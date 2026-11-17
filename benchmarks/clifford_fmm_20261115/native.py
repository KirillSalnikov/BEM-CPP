"""Small typed ctypes bridge to the documented FMM3D Fortran vector ABI."""
import ctypes as ct
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np

class NativeFMM:
    def __init__(self,path=None,ndiv=None):
        self.path=Path(path or os.environ.get('BEM_FMM3D_LIBRARY') or Path(__file__).resolve().parent/'FMM3D/lib/libfmm3d.so')
        self.lib=ct.CDLL(str(self.path))
        self.fn=self.lib.hfmm3d_s_c_g_vec_
        self.fn.argtypes=[ct.c_void_p]*9;self.fn.restype=None
        self.target_fn=self.lib.hfmm3d_t_c_g_vec_
        self.target_fn.argtypes=[ct.c_void_p]*11;self.target_fn.restype=None
        if ndiv is not None and (not isinstance(ndiv,int) or ndiv<0):raise ValueError('FMM leaf size must be a nonnegative integer or None')
        self.ndiv=ndiv;self.last_profile=None
        self.ndiv_fn=self.lib.hfmm3d_ndiv_;self.ndiv_fn.argtypes=[ct.c_void_p]*25;self.ndiv_fn.restype=None
        self.div_fn=self.lib.hndiv_;self.div_fn.argtypes=[ct.c_void_p]*10;self.div_fn.restype=None
    def targets(self,*,eps,zk,sources,charges,targets):
        sources=np.asfortranarray(sources,dtype=np.float64);charges=np.asfortranarray(charges,dtype=np.complex128)
        targets=np.asfortranarray(targets,dtype=np.float64)
        if sources.ndim!=2 or sources.shape[0]!=3 or targets.ndim!=2 or targets.shape[0]!=3 or charges.ndim!=2 or charges.shape[1]!=sources.shape[1]:raise ValueError('FMM target shape')
        if self.ndiv is not None:return self._evaluate(eps,zk,sources,charges,targets)
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
        if self.ndiv is not None:return self._evaluate(eps,zk,sources,charges,None)
        count=sources.shape[1];n=ct.c_int64(count);dens=ct.c_int64(nd);tol=ct.c_double(eps);ier=ct.c_int64(-1)
        wave=np.array([zk],np.complex128);pot=np.empty((nd,count),np.complex128,order='F');grad=np.empty((nd,3,count),np.complex128,order='F')
        ptr=lambda a:a.ctypes.data_as(ct.c_void_p)
        self.fn(ct.byref(dens),ct.byref(tol),ptr(wave),ct.byref(n),ptr(sources),ptr(charges),ptr(pot),ptr(grad),ct.byref(ier))
        return SimpleNamespace(pot=pot,grad=grad,ier=ier.value)

    def _evaluate(self,eps,zk,sources,charges,targets):
        """Pinned FMM3D expert ABI: tune tree occupancy without changing eps."""
        import time
        start=time.perf_counter();at_sources=targets is None
        nd=ct.c_int64(charges.shape[0]);ns=ct.c_int64(sources.shape[1]);nt=ct.c_int64(0 if at_sources else targets.shape[1])
        tol=ct.c_double(eps);wave=np.array([zk],complex)
        yes=ct.c_int64(1);no=ct.c_int64(0);source_flag=ct.c_int64(2 if at_sources else 0);target_flag=ct.c_int64(0 if at_sources else 2)
        division=ct.c_int64(self.ndiv);divflag=ct.c_int64(0);periodic=ct.c_int64(0);ier=ct.c_int64(-1)
        dummy=np.zeros(1,complex);dummy_points=np.zeros((3,1),order='F');timing=np.zeros(6)
        ptr=lambda a:a.ctypes.data_as(ct.c_void_p)
        if not division.value:
            lo=sources.min(axis=1);hi=sources.max(axis=1)
            if not at_sources:
                lo=np.minimum(lo,targets.min(axis=1));hi=np.maximum(hi,targets.max(axis=1))
            box=ct.c_double(float(np.real(zk)*np.max(hi-lo)/(2*np.pi)))
            self.div_fn(ct.byref(box),ct.byref(tol),ct.byref(ns),ct.byref(nt),ct.byref(yes),ct.byref(no),ct.byref(source_flag),ct.byref(target_flag),ct.byref(division),ct.byref(divflag))
        count=ns.value if at_sources else nt.value
        pot=np.empty((nd.value,count),complex,order='F');grad=np.empty((nd.value,3,count),complex,order='F')
        self.ndiv_fn(ct.byref(nd),ct.byref(tol),ptr(wave),ct.byref(ns),ptr(sources),ct.byref(yes),ptr(charges),
            ct.byref(no),ptr(dummy),ct.byref(periodic),ct.byref(source_flag),ptr(pot if at_sources else dummy),
            ptr(grad if at_sources else dummy),ptr(dummy),ct.byref(nt),ptr(dummy_points if at_sources else targets),
            ct.byref(target_flag),ptr(dummy if at_sources else pot),ptr(dummy if at_sources else grad),ptr(dummy),
            ct.byref(division),ct.byref(divflag),ct.byref(yes),ptr(timing),ct.byref(ier))
        self.last_profile=dict(leaf_size=division.value,wall_s=time.perf_counter()-start,stage_seconds=timing.tolist(),
            stages=['form_multipoles','upward','translations','downward','evaluate_expansions','direct_near'])
        return SimpleNamespace(pot=pot,grad=grad,ier=ier.value,profile=self.last_profile)
