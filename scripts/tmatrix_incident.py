"""Regular vector spherical waves for streaming modal BEM right-hand sides."""
from concurrent.futures import ThreadPoolExecutor
import os

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


def fill_regular_field(positions, mode, ka, destination, chunk_size=65536,
                       field_function=regular_fields):
    """Write disjoint surface chunks directly to the packet scratch array."""
    threads = int(os.environ.get('BEM_INCIDENT_THREADS', '1'))
    if threads < 1 or chunk_size < 1:
        raise ValueError('incident threads and chunk size must be positive')
    chunks = range(0, len(positions), chunk_size)

    def fill(start):
        part = slice(start, start + chunk_size)
        destination[part] = field_function(positions[part], mode, ka)

    if threads == 1 or len(chunks) <= 1:
        for start in chunks:
            fill(start)
    else:
        with ThreadPoolExecutor(max_workers=min(threads, len(chunks))) as pool:
            for _ in pool.map(fill, chunks):
                pass


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


class SeparableFieldCache:
    """Evaluate radial and polar special functions once per distinct FP64 argument."""

    def __init__(self, positions, wave_number, chunk_size=65536):
        self.chunk_size = chunk_size
        self.threads = int(os.environ.get('BEM_INCIDENT_THREADS', '1'))
        if self.threads < 1 or chunk_size < 1:
            raise ValueError('incident threads and chunk size must be positive')
        radius = np.linalg.norm(positions, axis=1)
        cos_theta = np.divide(positions[:, 2], radius, out=np.ones_like(radius), where=radius != 0)
        theta = np.arccos(np.clip(cos_theta, -1., 1.))
        self.positions = positions
        self.radii, radial = np.unique(radius, return_inverse=True)
        self.z = wave_number*self.radii
        self.radial_index = radial.astype(np.int32 if len(self.z) <= np.iinfo(np.int32).max else np.intp)
        del radial
        self.theta, polar = np.unique(theta, return_inverse=True)
        self.polar_index = polar.astype(np.int32 if len(self.theta) <= np.iinfo(np.int32).max else np.intp)
        self.phi = np.arctan2(positions[:, 1], positions[:, 0])
        self.phi[radius==0] = 0.
        self.degree = None
        self.angular_mode = None

    def _chunks(self, count, function):
        starts = range(0, count, self.chunk_size)
        def apply(start):
            function(slice(start, start+self.chunk_size))
        if self.threads == 1 or len(starts) < 2:
            for start in starts:
                apply(start)
        else:
            with ThreadPoolExecutor(max_workers=min(self.threads, len(starts))) as pool:
                for _ in pool.map(apply, starts):
                    pass

    def fill(self, mode, destination):
        polarization, l, m = mode
        if self.degree != l:
            self.j, self.tangential, self.radial = np.empty((3,len(self.z)))
            def radial(sl):
                z = self.z[sl]
                j = spherical_jn(l,z)
                j_over_z = np.divide(j,z,out=np.full_like(z,1/3 if l==1 else 0.),where=z!=0)
                self.j[sl] = j
                self.tangential[sl] = spherical_jn(l,z,derivative=True)+j_over_z
                self.radial[sl] = np.sqrt(l*(l+1))*j_over_z
            self._chunks(len(self.z), radial)
            self.degree = l
        if self.angular_mode != (l,m):
            # At zero azimuth Y and X_theta are real, X_phi is imaginary.
            # The common exp(i*m*phi) is restored after the meridian evaluation.
            self.y, self.xt, self.xp = np.empty((3,len(self.theta)))
            def angular(sl):
                t = self.theta[sl]
                y,xt,xp = magnetic_angular_components(t,np.zeros_like(t),l,m)
                self.y[sl],self.xt[sl],self.xp[sl] = y.real,xt.real,xp.imag
            self._chunks(len(self.theta), angular)
            self.angular_mode = l,m

        def reconstruct(sl):
            ri,ti = self.radial_index[sl],self.polar_index[sl]
            theta,phi = self.theta[ti],self.phi[sl]
            st,ct,sp,cp = np.sin(theta),np.cos(theta),np.sin(phi),np.cos(phi)
            phase = np.exp(1j*m*phi) if m else 1.
            a,b = self.xt[ti], self.xp[ti]
            mag = self.j[ri]*phase
            tangential,radial = self.tangential[ri], self.radial[ri]*self.y[ti]
            # arccos can round very near a pole. The reference uses the
            # original Cartesian direction for the cross product and radial term.
            positions = self.positions[sl]
            radius = self.radii[ri]
            rho = np.divide(positions[:,0]*cp + positions[:,1]*sp,radius,
                            out=np.zeros_like(radius),where=radius!=0)
            uz = np.divide(positions[:,2],radius,out=np.ones_like(radius),where=radius!=0)
            electric = 1j*(radial*rho-tangential*b*uz)
            azimuth = tangential*a*(uz*ct+rho*st)
            field = destination[sl]
            magnetic_field,electric_field = (field[:,:3],field[:,3:]) if polarization==0 else (field[:,3:],field[:,:3])
            electric_phase = -1j*phase if polarization==0 else phase
            if polarization:
                mag = -1j*mag
            magnetic_field[:,0] = mag*(a*ct*cp-1j*b*sp)
            magnetic_field[:,1] = mag*(a*ct*sp+1j*b*cp)
            magnetic_field[:,2] = -mag*a*st
            electric_field[:,0] = electric_phase*(electric*cp-azimuth*sp)
            electric_field[:,1] = electric_phase*(electric*sp+azimuth*cp)
            electric_field[:,2] = electric_phase*(1j*(radial*uz+tangential*b*rho))
        self._chunks(len(self.phi), reconstruct)
