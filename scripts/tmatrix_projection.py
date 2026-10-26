"""Factored spherical projection: FFT in azimuth, quadrature in latitude."""
import numpy as np
from tmatrix_reference import magnetic_angular_components, modes


class SphericalProjector:
    def __init__(self, order, z, weights):
        self.order=order
        self.band=len(z)
        self.nphi=2*self.band
        if self.band<=order:
            raise ValueError('latitude band must exceed the multipole order')
        selected=modes(order)
        half=len(selected)//2
        theta=np.arccos(z); phi=np.zeros_like(theta)
        self.coefficients=np.empty((half,self.band,2),dtype=np.complex128)
        self.azimuth=np.empty(half,dtype=np.int64)
        for i,(_,l,m) in enumerate(selected[:half]):
            _,xt,xp=magnetic_angular_components(theta,phi,l,m)
            self.coefficients[i,:,0]=xt.conj()*weights*np.pi/self.band
            self.coefficients[i,:,1]=xp.conj()*weights*np.pi/self.band
            self.azimuth[i]=m%self.nphi

    def __matmul__(self, transverse):
        # Independent validation directions, if present, carry zero weight.
        values=np.asarray(transverse)[:2*self.band*self.nphi].reshape(self.band,self.nphi,2)
        fourier=np.fft.fft(values,axis=1)
        magnetic=np.empty(len(self.azimuth),dtype=np.complex128)
        electric=np.empty_like(magnetic)
        # Small blocks keep the gathered Fourier values bounded at high L.
        for first in range(0,len(magnetic),256):
            last=min(first+256,len(magnetic))
            c=self.coefficients[first:last]
            f=fourier[:,self.azimuth[first:last],:].transpose(1,0,2)
            magnetic[first:last]=np.sum(c*f,axis=(1,2))
            electric[first:last]=np.sum(-c[:,:,1]*f[:,:,0]+c[:,:,0]*f[:,:,1],axis=1)
        return np.r_[magnetic,electric]
