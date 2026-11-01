import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mbs_tmatrix_seed import body_rotation,native_sphere,orientation_projection,kernel_to_t,field_symmetry_matrix
from tmatrix_reference import modes


class ProjectionTests(unittest.TestCase):
    def test_native_rotation_and_hexagon_frame(self):
        np.testing.assert_allclose(body_rotation(0,30),np.eye(3),atol=1e-15)
        r=body_rotation(37,49)
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=5e-16)
        # Explicit native Ry(beta)Rz(gamma) action on its zero-azimuth vertex
        # must match our BEM vertex at pi/6 after body rotation.
        b,g=np.deg2rad([37,49]);native=np.array([np.cos(b)*np.cos(g),np.sin(g),-np.sin(b)*np.cos(g)])
        np.testing.assert_allclose(r@np.array([np.sqrt(3)/2,.5,0]),native,atol=5e-16)

    def test_isotropic_electric_dipole_has_known_T_eigenvalue(self):
        order,band,ka=3,5,2.7;strength=.2+.13j
        theta,phi,directions,axes,weights=native_sphere(band)
        z,w=np.polynomial.legendre.leggauss(band);kernel=np.zeros((len(modes(order)),)*2,complex)
        for mu,weight in zip(z,w):
            beta=np.rad2deg(np.arccos(mu))
            for azimuth in phi:
                gamma=30+azimuth;r=body_rotation(beta,gamma)
                body=directions@r;input_axes=np.array([[-1.,0,0],[0,1.,0]])@r
                # Independent Cartesian electric dipole radiation.
                field=strength*(input_axes.T[None,:,:]-body[:,:,None]*np.einsum('pc,ic->pi',body,input_axes)[:,None,:])
                jones=np.einsum('poc,pci->poi',axes@r,field)
                outgoing,incoming=orientation_projection(jones,beta,gamma,directions,axes,weights,order)
                kernel+=weight*np.pi/band*(outgoing@incoming)
        t=kernel_to_t(kernel,ka,order);expected=np.zeros_like(t)
        for i,(pol,l,m) in enumerate(modes(order)):
            if pol==1 and l==1:expected[i,i]=-2j*ka*strength/3
        np.testing.assert_allclose(t,expected,rtol=2e-13,atol=2e-14)

    def test_polar_vector_D6h_representation(self):
        _,_,directions,_,weights=native_sphere(6);angle=np.pi/3
        rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        mirror=np.diag([1.,-1.,1.]);horizontal=np.diag([1.,1.,-1.])
        q=[field_symmetry_matrix(s,directions,weights,3) for s in (rotation,mirror,horizontal)]
        identity=np.eye(len(modes(3)))
        for item in q:np.testing.assert_allclose(item.conj().T@item,identity,atol=2e-14)
        np.testing.assert_allclose(np.linalg.matrix_power(q[0],6),identity,atol=3e-14)
        np.testing.assert_allclose(q[1]@q[0]@q[1],q[0].conj().T,atol=3e-14)
        for item in q[1:]:np.testing.assert_allclose(item@item,identity,atol=2e-14)


if __name__=='__main__':unittest.main()
