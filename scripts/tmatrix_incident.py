"""Regular vector spherical waves for streaming modal BEM right-hand sides."""
import numpy as np
from scipy.special import spherical_jn
from tmatrix_reference import magnetic_angular_components, tangent_frame


def regular_fields(positions, mode, wave_number):
    """Return [Ex,Ey,Ez,Hx,Hy,Hz], eta_ext=1, exp(-i omega t).

    M=j_l X_lm, N=curl(M)/k; E_M=M, H_M=-i N and vice versa.
    Fields are evaluated directly, avoiding cancellation in angular synthesis
    for high-order weakly excited regular waves.
    """
    polarization, l, m = mode
    radius = np.linalg.norm(positions, axis=1)
    direction = np.divide(positions, radius[:, None],
                          out=np.broadcast_to([0., 0., 1.], positions.shape).copy(),
                          where=radius[:, None] != 0)
    theta, phi, et, ep = tangent_frame(direction)
    y,xt,xp=magnetic_angular_components(theta,phi,l,m)
    x=xt[:,None]*et+xp[:,None]*ep
    z=wave_number*radius
    j=spherical_jn(l,z)
    jp=spherical_jn(l,z,derivative=True)
    j_over_z=np.divide(j,z,out=np.full_like(z,1/3 if l==1 else 0.),where=z!=0)
    magnetic=j[:,None]*x
    electric=(j_over_z+jp)[:,None]*np.cross(direction,x)+(
        1j*np.sqrt(l*(l+1))*j_over_z*y)[:,None]*direction
    e,h=(magnetic,-1j*electric) if polarization==0 else (electric,-1j*magnetic)
    return np.concatenate((e,h),axis=1)


class RegularFieldCache:
    """Cache surface frames, one radial degree and one angular mode.

    Electric/magnetic incident polarizations share the same M/N fields.
    Storage is O(surface points), independent of the number of T columns.
    """
    def __init__(self, positions, wave_number, chunk_size=65536):
        self.radius=np.linalg.norm(positions,axis=1)
        self.direction=np.divide(positions,self.radius[:,None],
            out=np.broadcast_to([0.,0.,1.],positions.shape).copy(),where=self.radius[:,None]!=0)
        self.theta,self.phi,self.et,self.ep=tangent_frame(self.direction)
        self.z=wave_number*self.radius
        self.chunk_size=chunk_size
        self.degree=None
        self.angular_mode=None
        self.base=np.empty((len(positions),6),complex)

    def fill(self, mode, destination):
        p,l,m=mode
        if self.degree!=l:
            self.j=spherical_jn(l,self.z)
            self.jp=spherical_jn(l,self.z,derivative=True)
            self.j_over_z=np.divide(self.j,self.z,out=np.full_like(self.z,1/3 if l==1 else 0.),where=self.z!=0)
            self.degree=l
            self.angular_mode=None
        if self.angular_mode!=(l,m):
            for start in range(0,len(self.z),self.chunk_size):
                sl=slice(start,start+self.chunk_size)
                y,xt,xp=magnetic_angular_components(self.theta[sl],self.phi[sl],l,m)
                x=xt[:,None]*self.et[sl]+xp[:,None]*self.ep[sl]
                magnetic=self.j[sl,None]*x
                electric=(self.j_over_z[sl,None]+self.jp[sl,None])*np.cross(self.direction[sl],x)+(
                    1j*np.sqrt(l*(l+1))*self.j_over_z[sl]*y)[:,None]*self.direction[sl]
                self.base[sl,:3]=magnetic
                self.base[sl,3:]=-1j*electric
            self.angular_mode=(l,m)
        if p==0:
            destination[:]=self.base
        else:
            destination[:,:3]=1j*self.base[:,3:]
            destination[:,3:]=-1j*self.base[:,:3]
