"""Independent native BDM/Green checks on cube, tetrahedron and C1 geometry."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from surface_symmetry import PointGroup
from regular_surface_geometry import RegularSurfaceGeometry
from symmetric_galerkin_fft import SymmetricGalerkinFFT
from checked_residual_solver import solve_refined
try:
    import torch
except ImportError:
    torch=None


def tetrahedron():
    points=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)
    triangles=[]
    for indices in itertools.combinations(range(4),3):
        v=points[list(indices)]
        if np.dot(np.cross(v[1]-v[0],v[2]-v[0]),v.mean(axis=0))<0:v=v[[0,2,1]]
        triangles.append(v)
    return np.array(triangles)


def cube():
    triangles=[]
    for axis in range(3):
        for sign in (-1,1):
            normal=np.eye(3)[axis]*sign;u=np.eye(3)[(axis+1)%3];v=np.cross(normal,u)
            for i,j in itertools.product(range(2),repeat=2):
                corners=np.array([normal+(i-1+a)*u+(j-1+b)*v for a,b in ((0,0),(1,0),(1,1),(0,1))])
                pattern=((0,1,2),(0,2,3)) if (i+j)%2==0 else ((0,1,3),(1,2,3))
                triangles.extend(corners[list(t)] for t in pattern)
    return np.array(triangles)


def nonconvex_column():
    cells={(0,0),(1,0),(0,1)};triangles=[]
    for i,j in sorted(cells):
        for triangle in cube()*.5+np.array([i+.5,j+.5,0.]):
            normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
            axis=int(np.argmax(abs(normal)));sign=int(np.sign(normal[axis]))
            neighbor=(i+sign,j) if axis==0 else (i,j+sign)
            if axis!=2 and neighbor in cells:continue
            triangles.append(triangle)
    return np.array(triangles)


def write_obj(path,triangles):
    points=[];indices=[]
    for triangle in triangles:
        face=[]
        for x in triangle:
            index=next((i for i,p in enumerate(points) if np.linalg.norm(x-p)<1e-12),None)
            if index is None:index=len(points);points.append(x)
            face.append(index+1)
        indices.append(face)
    path.write_text(''.join('v '+' '.join(map(str,p))+'\n' for p in points)+
                    ''.join('f '+' '.join(map(str,t))+'\n' for t in indices))


class SurfaceSymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.directory=Path(cls.temp.name)
        includes=['-I'+str(ROOT/'src'),'-I'+str(ROOT/'tests/host_cuda_stubs')]
        sources=[str(ROOT/'src'/name) for name in ('muller_nodal.cpp','muller_duffy.cpp','mesh.cpp')]
        for name,extra in (('export_bdm_fft_patterns',[]),('export_regular_surface_fixture',[str(ROOT/'src/muller_dense.cpp')])):
            subprocess.run(['g++','-O2','-std=c++11',*includes,str(ROOT/'tools'/f'{name}.cpp'),*sources,*extra,
                            '-o',str(cls.directory/name)],check=True,capture_output=True)
        cls.fixture=json.loads(subprocess.check_output([str(cls.directory/'export_bdm_fft_patterns')],text=True))
        cls.cases={};cls.results=[]

    @classmethod
    def tearDownClass(cls):
        print('SURFACE_FFT_RESULTS='+json.dumps(cls.results,sort_keys=True))
        cls.temp.cleanup()

    def geometry(self,name,triangles,material=(1.3116,0.)):
        if name in self.cases:return self.cases[name]
        case=self.directory/name;case.mkdir();obj=case/'shape.obj';write_obj(obj,triangles)
        subprocess.run([str(self.directory/'export_regular_surface_fixture'),str(obj),str(case),'3',*map(str,material)],
                       check=True,capture_output=True)
        # Axes are prescribed lattice steps. For squares choose the two shortest
        # independent edges; tetrahedra use their own edge lattice.
        vertices=np.fromfile(case/'vertices.bin',float).reshape(-1,3,3)
        normals=np.fromfile(case/'normals.bin',float).reshape(-1,3)
        remaining=set(range(len(vertices)));axes=[]
        while remaining:
            t=min(remaining);face=[i for i in remaining if np.linalg.norm(normals[i]-normals[t])<1e-10 and
                                  np.max(abs((vertices[i]-vertices[t,0])@normals[t]))<1e-10]
            edges=(vertices[face]-np.roll(vertices[face],1,axis=1)).reshape(-1,3)
            edges=edges[np.argsort(np.linalg.norm(edges,axis=1))];a=edges[0]
            b=next(e for e in edges if np.linalg.norm(np.cross(a,e))>1e-10)
            axes.append([a,b]);remaining.difference_update(face)
        result=RegularSurfaceGeometry(case,self.fixture,axes);self.cases[name]=result;return result

    def check_action(self,name,geometry,group):
        operator=SymmetricGalerkinFFT(geometry,group=group)
        reference=np.fromfile(geometry.directory/'regular.bin',complex).reshape(geometry.N,geometry.N)
        rng=np.random.default_rng(431);error=0.
        for _ in range(3):
            x=rng.normal(size=geometry.N)+1j*rng.normal(size=geometry.N)
            y=operator.apply(x);z=reference@x
            error=max(error,float(np.linalg.norm(y-z)/np.linalg.norm(z)))
        self.assertLess(error,2e-11)
        self.assertEqual(len(operator.aliases),len(geometry.faces)**2)
        self.results.append(dict(case=name,group_order=len(group.matrices),faces=len(geometry.faces),
            canonical_blocks=len(operator.canonical),full_blocks=len(operator.aliases),native_regular_action_error=error))
        return operator,reference

    def test_cube_noncommuting_group_and_subgroups(self):
        geometry=self.geometry('cube',cube())
        cycle=np.array([[0,1,0],[0,0,1],[1,0,0]])
        swap=np.array([[0,1,0],[1,0,0],[0,0,1]])
        groups={'Oh':PointGroup.from_generators([cycle,swap,np.diag([-1,1,1])]),
                'Cs':PointGroup.from_generators([np.diag([1,1,-1])]),
                'C4':PointGroup.from_generators([[[0,-1,0],[1,0,0],[0,0,1]]]),
                'C1':PointGroup([np.eye(3)])}
        for name,group in groups.items():
            with self.subTest(group=name):self.check_action(name,geometry,group)

    def test_tetrahedron_reflections_and_checked_full_operator(self):
        center=np.array([.37,-.21,.14]);geometry=self.geometry('tetra',tetrahedron()+center,(1.3116,.2))
        cycle=np.array([[0,1,0],[0,0,1],[1,0,0]])
        swap=np.array([[0,1,0],[1,0,0],[0,0,1]])
        group=PointGroup.from_generators([cycle,swap,np.diag([-1,-1,1])],center)
        self.assertEqual(len(group.matrices),24)
        operator,regular=self.check_action('Td_absorbing_translated',geometry,group)
        dense=np.fromfile(geometry.directory/'dense.bin',complex).reshape(geometry.N,geometry.N)
        operator.local=sparse.csr_matrix(dense-regular)  # independent native singular/mass correction, test only
        rng=np.random.default_rng(37);b=rng.normal(size=geometry.N)+1j*rng.normal(size=geometry.N)
        solution,row=solve_refined(b,operator.apply,lambda x:dense@x,None,tolerance=1e-9,restart=geometry.N)
        self.assertEqual(row['status'],'complete')
        solution_error=float(np.linalg.norm(solution-np.linalg.solve(dense,b))/np.linalg.norm(solution))
        self.assertLess(solution_error,1e-8)
        self.results[-1].update(full_native_residual=row['residual'],full_native_solution_error=solution_error)

    @unittest.skipUnless(torch is not None,'Optional PyTorch dependency is not installed')
    def test_torch_general_group_action(self):
        center=np.array([.37,-.21,.14]);geometry=self.geometry('tetra',tetrahedron()+center,(1.3116,.2))
        group=PointGroup.from_generators([[[0,1,0],[0,0,1],[1,0,0]],
            [[0,1,0],[1,0,0],[0,0,1]],np.diag([-1,-1,1])],center)
        operator=SymmetricGalerkinFFT(geometry,group=group)
        regular=np.fromfile(geometry.directory/'regular.bin',complex).reshape(geometry.N,geometry.N)
        dense=np.fromfile(geometry.directory/'dense.bin',complex).reshape(geometry.N,geometry.N)
        operator.local=sparse.csr_matrix(dense-regular)
        rng=np.random.default_rng(37);b=rng.normal(size=geometry.N)+1j*rng.normal(size=geometry.N)
        torch.set_num_threads(1)
        for dtype in (torch.complex64,torch.complex128):
            resident=operator.torch_operator('cpu',dtype=dtype)
            value=resident(torch.as_tensor(b,dtype=dtype)).numpy()
            self.assertLess(np.linalg.norm(value-dense@b)/np.linalg.norm(dense@b),2e-6 if dtype==torch.complex64 else 2e-12)

    def test_nonconvex_column_with_two_reflections(self):
        geometry=self.geometry('nonconvex',nonconvex_column())
        group=PointGroup.from_generators([[[0,1,0],[1,0,0],[0,0,1]],np.diag([1,1,-1])])
        self.check_action('C2v_nonconvex_column',geometry,group)

    def test_rotated_symmetry_axes(self):
        angle=.371;c,s=np.cos(angle),np.sin(angle)
        Q=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        center=np.array([.31,.27,-.16])
        geometry=self.geometry('rotated_cube',cube()@Q.T+center)
        generators=[np.array([[0,1,0],[0,0,1],[1,0,0]]),np.array([[0,1,0],[1,0,0],[0,0,1]]),np.diag([-1,1,1])]
        group=PointGroup.from_generators([Q@R@Q.T for R in generators],center)
        self.check_action('Oh_rotated_translated',geometry,group)

    def test_asymmetric_geometry_and_invalid_symmetry(self):
        triangles=tetrahedron().copy();triangles[:,:,0]*=1.3;triangles[:,:,1]+=.17*triangles[:,:,2]
        geometry=self.geometry('asymmetric',triangles)
        self.check_action('C1_skew_tetra',geometry,PointGroup([np.eye(3)]))
        with self.assertRaisesRegex(ValueError,'preserve|changes'):
            SymmetricGalerkinFFT(geometry,group=PointGroup.from_generators([[[0,1,0],[0,0,1],[1,0,0]]]))

    def test_invalid_groups_and_mesh_symmetry(self):
        with self.assertRaisesRegex(ValueError,'orthogonal'):PointGroup([np.diag([2,1,1])])
        with self.assertRaisesRegex(ValueError,'closed'):PointGroup([np.eye(3),[[0,-1,0],[1,0,0],[0,0,1]]])
        with self.assertRaisesRegex(ValueError,'unique'):PointGroup([np.eye(3),np.eye(3)])
        angle=np.sqrt(2);R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        with self.assertRaisesRegex(ValueError,'close'):PointGroup.from_generators([R],max_order=12)


if __name__=='__main__':unittest.main()
