"""Small-order T-matrix reference and exact SO(3) Mueller averaging.

The angular scattering kernel G uses orthonormal transverse vector spherical
harmonics X_lm and r-hat cross X_lm. The conventional regular/outgoing-wave
T matrix is (-i*k/4pi) U G U^H, U_M=i^l, U_N=i^(l-1), for
the repository's global radiation sign (S=-i*k*F, opposite the usual
Mie amplitude sign). The sign is checked against the Mie T diagonal.
SO(3) averaging uses irreducible tensor operators and Clebsch-Gordan
orthogonality. It averages Jones products, never the Jones matrix itself.
This reference implementation targets small L, not large production systems.
"""
from functools import lru_cache
import math
import numpy as np
try:
    from scipy.special import sph_harm_y
except ImportError:  # scipy 1.10--1.14, supported by requirements-analysis.txt
    from scipy.special import sph_harm
    def sph_harm_y(l, m, theta, phi):
        return sph_harm(m, l, phi, theta)


def modes(order):
    return [(p, l, m) for p in range(2) for l in range(1, order+1) for m in range(-l, l+1)]


def tangent_frame(directions):
    r = np.asarray(directions, dtype=float)
    theta = np.arccos(np.clip(r[:, 2], -1, 1))
    phi = np.arctan2(r[:, 1], r[:, 0])
    et = np.stack((np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)), axis=-1)
    ep = np.stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)), axis=-1)
    return theta, phi, et, ep


def magnetic_angular_components(theta, phi, l, m):
    y = sph_harm_y(l, m, theta, phi)
    derivative = np.zeros_like(y)
    if m < l:
        derivative += .5*np.sqrt((l-m)*(l+m+1))*np.exp(-1j*phi)*sph_harm_y(l,m+1,theta,phi)
    if m > -l:
        derivative -= .5*np.sqrt((l+m)*(l-m+1))*np.exp(1j*phi)*sph_harm_y(l,m-1,theta,phi)
    s = np.sin(theta)
    pole = np.abs(s)<1e-10
    xt = -m*y/(np.sqrt(l*(l+1))*np.where(pole,1.,s))
    xp = -1j*derivative/np.sqrt(l*(l+1))
    xt[pole]=0; xp[pole]=0
    if abs(m)==1 and pole.any():
        south=theta[pole]>np.pi/2
        value=np.sqrt((2*l+1)/(16*np.pi))*np.exp(1j*m*phi[pole])*np.where(south,(-1)**(l+m),1)
        xt[pole]=value
        xp[pole]=1j*m*value*np.where(south,-1,1)
    return y, xt, xp


def vector_harmonics(directions, order):
    theta, phi, et, ep = tangent_frame(directions)
    basis=[]
    for l in range(1,order+1):
        for m in range(-l,l+1):
            _,xt,xp=magnetic_angular_components(theta,phi,l,m)
            basis.append(xt[:,None]*et+xp[:,None]*ep)
    magnetic=np.stack(basis,axis=-1)
    electric=np.cross(np.asarray(directions)[:,:,None],magnetic,axis=1)
    return np.concatenate((magnetic,electric),axis=-1)


def transverse_harmonic_block(directions, selected_modes):
    theta, phi, _, _ = tangent_frame(directions)
    result=np.empty((2*len(directions),len(selected_modes)),dtype=np.complex128)
    for column,(p,l,m) in enumerate(selected_modes):
        _,xt,xp=magnetic_angular_components(theta,phi,l,m)
        result[0::2,column]=xt if p==0 else -xp
        result[1::2,column]=xp if p==0 else xt
    return result


def transverse_harmonics(directions, order):
    return transverse_harmonic_block(directions,modes(order))


@lru_cache(None)
def clebsch(j1, m1, j2, m2, total, projection):
    if m1+m2 != projection or not abs(j1-j2) <= total <= j1+j2:
        return 0.0
    # Exact integer factorial Racah sum, evaluated in double at the end.
    # Small-order prototype only; avoids an optional symbolic dependency.
    f = math.factorial
    if abs(m1)>j1 or abs(m2)>j2 or abs(projection)>total:
        return 0.0
    prefactor = math.sqrt(
        (2*total+1)*f(total+j1-j2)*f(total-j1+j2)*f(j1+j2-total)/f(j1+j2+total+1)
        * f(total+projection)*f(total-projection)*f(j1-m1)*f(j1+m1)*f(j2-m2)*f(j2+m2))
    lower = max(0, j2-total-m1, j1+m2-total)
    upper = min(j1+j2-total, j1-m1, j2+m2)
    value = 0.0
    for k in range(lower, upper+1):
        value += (-1)**k/(f(k)*f(j1+j2-total-k)*f(j1-m1-k)*f(j2+m2-k)
                          * f(total-j2+m1+k)*f(total-j1-m2+k))
    return prefactor*value


def irreducible_blocks(order):
    blocks = []
    index = 0
    for p in range(2):
        for l in range(1, order+1):
            blocks.append((p, l, slice(index, index+2*l+1)))
            index += 2*l+1
    return blocks


def prepare_so3_average(kernel, order):
    """Build rotationally invariant covariances of irreducible T blocks."""
    blocks = irreducible_blocks(order)
    result = []
    for total in range(2*order+1):
        entries, coefficients = [], []
        for _, l, first in blocks:
            for _, lp, second in blocks:
                if not abs(l-lp) <= total <= l+lp:
                    continue
                coupling = np.zeros((2*total+1, 2*l+1, 2*lp+1))
                for mi, m in enumerate(range(-l, l+1)):
                    for ni, n in enumerate(range(-lp, lp+1)):
                        projection = m-n
                        if abs(projection) <= total:
                            coupling[projection+total, mi, ni] = (
                                (-1)**(lp-n)*clebsch(l, m, lp, -n, total, projection))
                entries.append((first, second, coupling))
                coefficients.append(np.einsum('kmn,mn->k', coupling, kernel[first, second]))
        g = np.asarray(coefficients)
        covariance = g @ g.conj().T/(2*total+1)
        result.append((entries, covariance))
    return result


def averaged_coherency(prepared, output_basis, input_basis):
    """Exact uniform SO(3) mean for a finite multipole expansion."""
    result = np.zeros((4, 4), dtype=complex)
    for entries, covariance in prepared:
        amplitude = np.stack([
            np.einsum('um,kmn,vn->kuv', output_basis[:, first], coupling,
                      input_basis[:, second].conj()).reshape(coupling.shape[0], 4)
            for first, second, coupling in entries], axis=-1)
        result += np.einsum('mib,bc,mjc->ij', amplitude, covariance, amplitude.conj(), optimize=True)
    return result


def coherency_to_mueller(coherency):
    # Incoming/outgoing basis is (parallel, perpendicular); reorder Jones
    # entries [F00,F01,F10,F11] to the repository's [s1,s2,s3,s4] convention.
    c = np.asarray(coherency)[np.ix_([3, 0, 1, 2], [3, 0, 1, 2])]
    a1, a2, a3, a4 = c.diagonal().real
    s2s3, s1s4, s2s4, s1s3, s1s2, s3s4 = c[1,2], c[0,3], c[1,3], c[0,2], c[0,1], c[2,3]
    return np.array([
        [.5*(a1+a2+a3+a4), .5*(a2-a1+a4-a3), (s2s3+s1s4).real, (s2s3-s1s4).imag],
        [.5*(a2-a1-a4+a3), .5*(a2+a1-a4-a3), (s2s3-s1s4).real, (s2s3+s1s4).imag],
        [(s2s4+s1s3).real, (s2s4-s1s3).real, (s1s2+s3s4).real, -(s1s2+s3s4).imag],
        [(-s2s4+s1s3).imag, (-s2s4-s1s3).imag, (s1s2-s3s4).imag, (s1s2-s3s4).real]])


def analytic_mueller(kernel, order, theta_degrees):
    prepared = prepare_so3_average(kernel, order)
    input_basis = transverse_harmonics(np.array([[0., 0., 1.]]), order)
    angle = np.deg2rad(theta_degrees)
    directions = np.stack((np.sin(angle), np.zeros_like(angle), np.cos(angle)), axis=-1)
    output = transverse_harmonics(directions, order).reshape(-1, 2, len(modes(order)))
    return np.stack([coherency_to_mueller(averaged_coherency(prepared, out, input_basis)) for out in output])


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def numerical_so3_mueller(kernel, order, theta_degrees, beta_nodes=None, periodic_nodes=None):
    """Independent direct SO(3) cubature, exact at sufficient finite bandwidth."""
    beta_nodes = beta_nodes or (2*order+1)
    periodic_nodes = periodic_nodes or (4*order+1)
    mu, weights = np.polynomial.legendre.leggauss(beta_nodes)
    angles = np.deg2rad(theta_degrees)
    outgoing = np.stack((np.sin(angles), np.zeros_like(angles), np.cos(angles)), axis=-1)
    incoming = np.array([0., 0., 1.])
    out_theta = np.stack((np.cos(angles), np.zeros_like(angles), -np.sin(angles)), axis=-1)
    out_phi = np.array([0., 1., 0.])
    rotations, rotation_weights = [], []
    for z, weight in zip(mu, weights):
        s = np.sqrt(1-z*z)
        ry = np.array([[z, 0, s], [0, 1., 0], [-s, 0, z]])
        for alpha in np.arange(periodic_nodes)*2*np.pi/periodic_nodes:
            for gamma in np.arange(periodic_nodes)*2*np.pi/periodic_nodes:
                rotations.append(rotation_z(alpha) @ ry @ rotation_z(gamma))
                rotation_weights.append(weight/(2*periodic_nodes**2))
    rotations = np.asarray(rotations)
    rotation_weights = np.asarray(rotation_weights)
    particle_in = np.einsum('rji,j->ri', rotations, incoming)
    vin = vector_harmonics(particle_in, order)
    # Rotated laboratory x/y polarization vectors in the particle frame.
    # Compute the same basis rotation explicitly: lab = R * particle.
    input_basis = np.einsum('rij,rjq->riq', rotations, vin)[:, :2, :]
    result = []
    for direction, et in zip(outgoing, out_theta):
        particle_out = np.einsum('rji,j->ri', rotations, direction)
        vout = np.einsum('rij,rjq->riq', rotations, vector_harmonics(particle_out, order))
        output_basis = np.stack((np.einsum('j,rjq->rq', et, vout),
                                 np.einsum('j,rjq->rq', out_phi, vout)), axis=1)
        amplitude = np.einsum('rua,ab,rvb->ruv', output_basis, kernel, input_basis.conj(), optimize=True).reshape(-1, 4)
        covariance = np.einsum('r,ri,rj->ij', rotation_weights, amplitude, amplitude.conj())
        result.append(coherency_to_mueller(covariance))
    return np.asarray(result)
