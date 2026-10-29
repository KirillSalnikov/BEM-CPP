#!/usr/bin/env python3
"""Check regular-wave E/H against independent plane-wave synthesis."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from tmatrix_incident import regular_fields,RegularFieldCache
from tmatrix_reference import vector_harmonics,modes,magnetic_angular_components,sph_harm_y


class IncidentTest(unittest.TestCase):
    def test_one_neighbor_derivative_matches_two_neighbor_reference(self):
        rng=np.random.default_rng(79)
        near=np.r_[0.,np.geomspace(1e-14,1e-2,24),np.pi/2,np.pi-np.geomspace(1e-14,1e-2,24),np.pi]
        theta=np.r_[np.arccos(rng.uniform(-1,1,301)),near]
        phi=rng.uniform(-np.pi,np.pi,len(theta))
        for l in (1,2,20,44,112,149,256,400):
            for m in sorted({-l,-l+1,-l//2,-1,0,1,l//2,l-1,l}):
                y=sph_harm_y(l,m,theta,phi)
                derivative=np.zeros_like(y)
                if m<l:derivative+=.5*np.sqrt((l-m)*(l+m+1))*np.exp(-1j*phi)*sph_harm_y(l,m+1,theta,phi)
                if m>-l:derivative-=.5*np.sqrt((l+m)*(l-m+1))*np.exp(1j*phi)*sph_harm_y(l,m-1,theta,phi)
                sine=np.sin(theta);pole=np.abs(sine)<1e-10
                xt=-m*y/(np.sqrt(l*(l+1))*np.where(pole,1.,sine))
                xp=-1j*derivative/np.sqrt(l*(l+1));xt[pole]=0;xp[pole]=0
                if abs(m)==1 and pole.any():
                    south=theta[pole]>np.pi/2
                    value=np.sqrt((2*l+1)/(16*np.pi))*np.exp(1j*m*phi[pole])*np.where(south,(-1)**(l+m),1)
                    xt[pole]=value;xp[pole]=1j*m*value*np.where(south,-1,1)
                actual=magnetic_angular_components(theta,phi,l,m)
                for a,b in zip(actual,(y,xt,xp)):
                    self.assertTrue(np.isfinite(a).all())
                    # The old two-neighbor identity loses several 1e-12
                    # near poles at l=256..400; compare the independent
                    # differentiated recurrence below as well.
                    self.assertLess(np.max(np.abs(a-b))/max(np.max(np.abs(b)),1.),1e-11,(l,m))
                try:
                    _,direct=sph_harm_y(l,m,theta,phi,diff_n=1)
                except TypeError:  # Legacy SciPy wrapper has no derivatives.
                    continue
                direct_xp=-1j*direct[:,0]/np.sqrt(l*(l+1))
                self.assertLess(np.max(np.abs(actual[2][~pole]-direct_xp[~pole]))/
                                max(np.max(np.abs(direct_xp[~pole])),1.),2e-12,(l,m))

    def test_cached_fields_match_at_poles_origin_and_high_orders(self):
        rng=np.random.default_rng(42)
        points=np.vstack((rng.normal(size=(80,3)),[[0,0,0],[0,0,1],[0,0,-1]]))
        cache=RegularFieldCache(points,20.,chunk_size=13)
        output=np.empty((len(points),6),complex)
        for l in (1,5,22,44,1):
            for m in {0,1,l,-l}:
                for p in (0,1):
                    cache.fill((p,l,m),output)
                    np.testing.assert_allclose(output,regular_fields(points,(p,l,m),20.),rtol=1e-13,atol=1e-15)

    def test_reflection_with_polar_E_and_axial_H(self):
        points=np.random.default_rng(55).normal(size=(51,3))
        for l in (1,5,22,44):
            for m in (0,1,l):
                for p in (0,1):
                    expected=regular_fields(points,(p,l,m),20.)*(-1)**(l+m+p+1)
                    actual=regular_fields(points*[1,1,-1],(p,l,m),20.)*[1,1,-1,-1,-1,1]
                    self.assertLess(np.linalg.norm(expected-actual)/np.linalg.norm(expected),1e-11)

    def test_plane_wave_synthesis(self):
        z,w=np.polynomial.legendre.leggauss(16)
        zz,pp=np.meshgrid(z,np.arange(32)*2*np.pi/32,indexing='ij')
        directions=np.stack((np.sqrt(1-zz*zz)*np.cos(pp),np.sqrt(1-zz*zz)*np.sin(pp),zz),axis=-1).reshape(-1,3)
        weights=np.repeat(w,32)*2*np.pi/32
        points=np.array([[0,0,0],[0,0,.7],[0,0,-.4],[.2,-.3,.1],[-.5,.1,.7]],dtype=float)
        k=2.
        phase=np.exp(1j*k*(points@directions.T))*weights
        basis=vector_harmonics(directions,3)
        for index,mode in enumerate(modes(3)):
            p,l,m=mode
            electric=basis[:,:,index]
            magnetic=np.cross(directions,electric)
            scale=4*np.pi*1j**(l-p)
            reference=np.concatenate((phase@electric,phase@magnetic),axis=1)/scale
            np.testing.assert_allclose(regular_fields(points,mode,k),reference,atol=2e-14,rtol=2e-12)

    def test_weak_high_order_modes_are_finite(self):
        points=np.array([[.1,.2,.3],[0,0,1.]],dtype=float)
        for p in (0,1):
            field=regular_fields(points,(p,64,32),1.)
            self.assertTrue(np.isfinite(field).all())
            self.assertGreater(np.linalg.norm(field),0.)


if __name__=='__main__':
    unittest.main()
