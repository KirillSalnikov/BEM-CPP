"""Analytic ka derivatives of regular vector spherical incident waves.

The radial ODE generates derivative/factorial coefficients. Geometry,
material and angular modes stay fixed. No finite differences, fitting, or
angular synthesis of small incident waves is used.
"""
import numpy as np
from scipy.special import spherical_jn
from tmatrix_reference import magnetic_angular_components, tangent_frame


class RegularTaylorFieldCache:
    def __init__(self, positions, wave_number, degree):
        points=np.asarray(positions,dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
            raise ValueError('invalid surface positions')
        if not np.isfinite(wave_number) or wave_number<=0 or int(degree)!=degree or not 0<=degree<=16:
            raise ValueError('positive real wave number and degree 0..16 required')
        self.k=float(wave_number);self.degree=int(degree)
        self.radius=np.linalg.norm(points,axis=1)
        self.direction=np.divide(points,self.radius[:,None],
            out=np.broadcast_to([0.,0.,1.],points.shape).copy(),where=self.radius[:,None]!=0)
        self.theta,self.phi,self.et,self.ep=tangent_frame(self.direction)
        self.radial_degree=None;self.angular_mode=None
        self.base=np.empty((self.degree+1,len(points),6),complex)

    def _radial(self,l):
        r=self.radius;k=self.k;z=k*r
        a=np.empty((self.degree+2,len(r)),dtype=float)
        a[0]=spherical_jn(l,z);a[1]=r*spherical_jn(l,z,derivative=True)
        # k² g'' + 2 k g' + (k² r² - l(l+1)) g = 0.
        for q in range(self.degree):
            value=2*k*(q+1)**2*a[q+1]+(q*(q+1)+z*z-l*(l+1))*a[q]
            if q: value+=2*k*r*r*a[q-1]
            if q>1:value+=r*r*a[q-2]
            a[q+2]=-value/(k*k*(q+2)*(q+1))
        over=np.zeros((self.degree+1,len(r)))
        derivative=np.zeros_like(over)
        for q in range(self.degree+1):
            over[q]=np.divide(a[q],r,out=np.zeros_like(r),where=r!=0)
            if q:over[q]-=over[q-1]
            over[q]/=k
            derivative[q]=np.divide((q+1)*a[q+1],r,out=np.zeros_like(r),where=r!=0)
        if l==1:
            over[0,r==0]=1/3;derivative[0,r==0]=1/3
        self.j=a[:self.degree+1];self.j_over_z=over;self.jp=derivative
        self.radial_degree=l;self.angular_mode=None

    def fill(self,mode,destination):
        p,l,m=mode
        if p not in (0,1) or int(l)!=l or l<1 or int(m)!=m or abs(m)>l:
            raise ValueError('invalid regular wave mode')
        if destination.shape!=self.base.shape or destination.dtype!=np.complex128:
            raise ValueError('Taylor field destination shape/dtype')
        if self.radial_degree!=l:self._radial(l)
        if self.angular_mode!=(l,m):
            y,xt,xp=magnetic_angular_components(self.theta,self.phi,l,m)
            x=xt[:,None]*self.et+xp[:,None]*self.ep
            cross=np.cross(self.direction,x)
            radial=(1j*np.sqrt(l*(l+1))*y)[:,None]*self.direction
            self.base[:,:,:3]=self.j[:,:,None]*x[None,:,:]
            self.base[:,:,3:]=-1j*((self.j_over_z+self.jp)[:,:,None]*cross[None,:,:]
                +self.j_over_z[:,:,None]*radial[None,:,:])
            self.angular_mode=(l,m)
        if p==0:destination[:]=self.base
        else:
            destination[:,:,:3]=1j*self.base[:,:,3:]
            destination[:,:,3:]=-1j*self.base[:,:,:3]


def regular_field_moments(positions,mode,wave_number,degree):
    cache=RegularTaylorFieldCache(positions,wave_number,degree)
    result=np.empty_like(cache.base);cache.fill(mode,result)
    return result
