import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mbs_coherent_field import native_jones_to_bem


class CoherentFieldTests(unittest.TestCase):
    def test_independent_electric_dipole_radiation_sign(self):
        # Exact outgoing dipole E/H on a unit sphere with time factor e^-iwt.
        # Integrating its equivalent physical J=n x H, M=-n x E with the
        # repository's radiation prefactor gives MINUS its physical far E.
        # This establishes the sign without fitting any BEM/MBS comparison.
        z,w=np.polynomial.legendre.leggauss(40)
        zz,pp=np.meshgrid(z,np.arange(80)*2*np.pi/80,indexing='ij')
        n=np.stack((np.sqrt(1-zz*zz)*np.cos(pp),np.sqrt(1-zz*zz)*np.sin(pp),zz),axis=-1).reshape(-1,3)
        weight=np.repeat(w,80)*2*np.pi/80
        p=np.array([1.,.3j,-.2]);s=np.array([[1.,0,0],[0,1.,0],[0,0,1.],[.6,.8,0]])
        for k in [.3,3.,10.]:
            dot=n@p;transverse=p-n*dot[:,None]
            e=np.exp(1j*k)/(4*np.pi)*(k*k*transverse+(3*n*dot[:,None]-p)*(1-1j*k))
            h=k*k*np.exp(1j*k)/(4*np.pi)*(1+1j/k)*np.cross(n,p)
            j,m=np.cross(n,h),-np.cross(n,e)
            phase=np.exp(-1j*k*s@n.T)*weight;fj,fm=phase@j,phase@m
            repository_far=-1j*k/(4*np.pi)*(fj-s*np.sum(s*fj,axis=1)[:,None]-np.cross(s,fm))
            physical_far=k*k/(4*np.pi)*(p-s*(s@p)[:,None])
            self.assertLess(np.linalg.norm(repository_far+physical_far)/np.linalg.norm(physical_far),1e-12)
            self.assertGreater(np.linalg.norm(repository_far-physical_far)/np.linalg.norm(physical_far),1.99)

    def test_units_complex_phase_and_mueller_invariance(self):
        j=np.array([[1+.2j, .3j],[.1-.4j, 2-.3j]])
        a=20*.532/(2*np.pi)
        converted=native_jones_to_bem(j,20,.532)
        np.testing.assert_allclose(converted,-j/a)
        np.testing.assert_allclose(native_jones_to_bem(2*j,20,1.064),converted)
        np.testing.assert_allclose(native_jones_to_bem(1j*j,20,.532),1j*converted)
        np.testing.assert_allclose(abs(converted)**2,abs(j/a)**2)
        for bad in [0,float('nan'),-1]:
            with self.assertRaises(ValueError):native_jones_to_bem(j,bad,.532)


if __name__=='__main__':unittest.main()
