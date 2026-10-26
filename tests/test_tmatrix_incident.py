#!/usr/bin/env python3
"""Check regular-wave E/H against independent plane-wave synthesis."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from tmatrix_incident import regular_fields,RegularFieldCache
from tmatrix_reference import vector_harmonics,modes


class IncidentTest(unittest.TestCase):
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
