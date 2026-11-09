"""Native MBS geometry fixtures, actual mesh symmetries and invalid inputs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from mbs_particle_geometry import builtin,prepare,read_particle,triangulate,write_obj
from surface_symmetry import PointGroup

SPECS=([1,2,1],[2,2,1],[3,2,1],[4,1],[10,2,1,20],[12,2,1,2],[999,1])


class MbsParticleGeometryTests(unittest.TestCase):
    def test_all_native_mbs_geometry_and_symmetry(self):
        expected=(24,12,8,24,24,4,1)
        components=(1,1,6,1,1,2,8)
        for spec,order,parts in zip(SPECS,expected,components):
            with self.subTest(particle=spec):
                reference=read_particle(ROOT/f'reference/particles/native_type{spec[0]}.particle')
                a,b=np.vstack(builtin(spec)),np.vstack(reference)
                scale=np.linalg.norm(np.ptp(a,axis=0))
                self.assertLess(cKDTree(a).query(b)[0].max()/scale,1e-12)
                self.assertLess(cKDTree(b).query(a)[0].max()/scale,1e-12)
                vertices,triangles,report=prepare(spec)
                _,_,native=triangulate(reference)
                self.assertAlmostEqual(report['volume']/native['volume'],1.,places=12)
                self.assertEqual(report['symmetry']['order'],order)
                self.assertEqual(report['components'],parts)
                group=PointGroup.detect(vertices,triangles)
                self.assertEqual(len(group.matrices),order)
                # The native file header is advisory: geometry determines the group.
                _,_,loaded=prepare(particle_file=ROOT/f'reference/particles/native_type{spec[0]}.particle')
                self.assertEqual(loaded['symmetry']['order'],order)

    def test_bullet_does_not_inherit_prism_horizontal_mirror(self):
        _,_,r=prepare([2,2,1]);s=r['solver_symmetries']
        self.assertTrue(s['c6_z']);self.assertTrue(s['vertical_mirror'])
        self.assertFalse(s['horizontal_mirror']);self.assertFalse(s['c2_x']);self.assertFalse(s['d6h_z'])

    def test_geometry_scale_and_legacy_droxtal_parameters(self):
        a,_,r=prepare([4,1]);b,_,s=prepare([4,91,73,2])
        np.testing.assert_allclose(b,2*a,atol=1e-14)
        self.assertAlmostEqual(s['volume']/r['volume'],8.)
        _,_,r=prepare([999,7]);self.assertAlmostEqual(r['dmax'],7.,places=12)

    def test_rotated_translated_mesh_and_refinement(self):
        vertices,triangles,r=prepare([4,1]);a=.371;c,s=np.cos(a),np.sin(a)
        Q=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        center=np.array([.53,-.27,.19]);v=vertices@Q.T+center
        group=PointGroup.detect(v,triangles)
        self.assertEqual(len(group.matrices),24)
        np.testing.assert_allclose(group.center,center,atol=1e-13)
        points=list(v);edges={};refined=[]
        def midpoint(a,b):
            key=tuple(sorted((a,b)))
            if key not in edges:edges[key]=len(points);points.append((v[a]+v[b])/2)
            return edges[key]
        for a,b,c in triangles:
            ab,bc,ca=midpoint(a,b),midpoint(b,c),midpoint(c,a)
            refined.extend(((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)))
        self.assertEqual(len(PointGroup.detect(points,refined).matrices),24)
        v[0]+=[.001,.002,.004]
        self.assertEqual(len(PointGroup.detect(v,triangles).matrices),1)

    def test_invalid_parameters_and_boundaries(self):
        for spec in ([0,1,1],[1,0,1],[2,1],[3,1,1,-1],[4,0],[10,2,1,70],[12,2,1,3],[999,-1],[3,2,100,1]):
            with self.subTest(spec=spec),self.assertRaises(ValueError):prepare(spec)
        with self.assertRaisesRegex(ValueError,'closed manifold'):triangulate([np.eye(3)])
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'bad.particle';path.write_text('0\n0\n90 60\n\n0 0 nan\n1 0 0\n0 1 0\n')
            with self.assertRaises(ValueError):prepare(particle_file=path)


if __name__=='__main__':unittest.main()
