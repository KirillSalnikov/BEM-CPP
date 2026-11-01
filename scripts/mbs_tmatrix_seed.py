"""Project coherent fixed-orientation fields into a body-frame T seed.

Angular integration here constructs the linear scattering operator. It does
not average Mueller matrices over orientations; final SO(3) is a separate
analytic contraction. No intensity-derived phase or amplitude fitting.
"""
import numpy as np
from tmatrix_reference import modes,magnetic_angular_components


def rotated_harmonics(directions,order):
    """Stable Cartesian harmonics when a rotated quadrature point hits a pole.

    atan2(hypot(x,y),z) preserves tiny transverse coordinates; acos(z)
    would turn a unit-vector roundoff of 1e-16 into an angle of 1e-8.
    """
    r=np.asarray(directions,float)
    norm=np.linalg.norm(r,axis=1)
    if r.ndim!=2 or r.shape[1]!=3 or np.max(np.abs(norm-1))>1e-12:
        raise ValueError('unit directions required')
    r=r/norm[:,None];theta=np.arctan2(np.hypot(r[:,0],r[:,1]),r[:,2]);phi=np.arctan2(r[:,1],r[:,0])
    et=np.stack((np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-np.sin(theta)),axis=-1)
    ep=np.stack((-np.sin(phi),np.cos(phi),np.zeros_like(phi)),axis=-1)
    columns=[]
    for l in range(1,order+1):
        for m in range(-l,l+1):
            _,xt,xp=magnetic_angular_components(theta,phi,l,m)
            columns.append(xt[:,None]*et+xp[:,None]*ep)
    magnetic=np.stack(columns,axis=-1)
    return np.concatenate((magnetic,np.cross(r[:,:,None],magnetic,axis=1)),axis=-1)


def body_rotation(beta_degrees,gamma_degrees):
    """BEM body -> MBS laboratory; native hexagon has a 30-degree offset.

    Particle::SetRotateMatrix(beta,gamma,0) is Ry(beta) Rz(gamma).
    The BEM prism vertices start at pi/6, the native MBS vertices at zero.
    """
    b,g=np.deg2rad([beta_degrees,gamma_degrees-30.])
    c,s=np.cos(b),np.sin(b);cg,sg=np.cos(g),np.sin(g)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])@np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])


def native_sphere(band):
    if not isinstance(band,int) or band<2:raise ValueError('positive sphere quadrature band')
    z,w=np.polynomial.legendre.leggauss(band)
    theta=np.rad2deg(np.arccos(z))[::-1];w=w[::-1]
    phi=np.arange(2*band)*180./band
    tt,ff=np.meshgrid(np.deg2rad(theta),np.deg2rad(phi),indexing='ij')
    direction=np.stack((np.sin(tt)*np.cos(ff),np.sin(tt)*np.sin(ff),-np.cos(tt)),axis=-1).reshape(-1,3)
    vf=np.stack((-np.sin(ff),np.cos(ff),np.zeros_like(ff)),axis=-1).reshape(-1,3)
    vt=np.cross(vf,direction)
    weights=np.repeat(w,2*band)*np.pi/band
    return theta,phi,direction,np.stack((vt,vf),axis=1),weights


def orientation_projection(jones_bem,beta,gamma,lab_directions,lab_axes,out_weights,order):
    """Return outgoing modal fields and incoming basis for both polarizations."""
    j=np.asarray(jones_bem,complex);directions=np.asarray(lab_directions,float);axes=np.asarray(lab_axes,float)
    weights=np.asarray(out_weights,float)
    if (j.shape!=(len(directions),2,2) or axes.shape!=(len(directions),2,3)
            or weights.shape!=(len(directions),) or not np.isfinite(j).all()):
        raise ValueError('inconsistent fixed-orientation amplitude data')
    rotation=body_rotation(beta,gamma)
    body_directions=directions@rotation;body_axes=axes@rotation
    outgoing=rotated_harmonics(body_directions,order)
    fields=np.einsum('poc,poi->pci',body_axes,j)
    modal=np.einsum('pcq,pci,p->qi',outgoing.conj(),fields,weights,optimize=True)
    incident=np.array([[0.,0.,-1.]])@rotation
    input_axes=np.array([[-1.,0.,0.],[0.,1.,0.]])@rotation
    incoming=input_axes@rotated_harmonics(incident,order)[0]
    return modal,incoming


def kernel_to_t(kernel,ka,order):
    phase=np.array([1j**(l-p) for p,l,m in modes(order)])
    return -1j*ka/(4*np.pi)*phase[:,None]*np.asarray(kernel)*phase[None,:].conj()


def t_to_kernel(t,ka,order):
    phase=np.array([1j**(l-p) for p,l,m in modes(order)])
    return (4*np.pi/(-1j*ka))*phase[:,None].conj()*np.asarray(t)*phase[None,:]


def field_symmetry_matrix(transform,directions,weights,order):
    """Polar-vector angular representation, including improper D6h actions."""
    s=np.asarray(transform,float)
    if s.shape!=(3,3) or np.linalg.norm(s.T@s-np.eye(3))>1e-12:raise ValueError('orthogonal transform required')
    basis=rotated_harmonics(directions,order)
    transformed=np.einsum('ij,pjq->piq',s,rotated_harmonics(np.asarray(directions)@s,order))
    return np.einsum('piq,pir,p->qr',basis.conj(),transformed,weights,optimize=True)
