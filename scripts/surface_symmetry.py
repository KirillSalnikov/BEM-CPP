"""Finite O(3) groups and verified actions on triangulated surfaces.

This is an orbit reduction of physical blocks, not a C6 Fourier-sector
decomposition. Noncommuting groups and improper rotations are supported.
"""
import numpy as np
from scipy.spatial import cKDTree


class PointGroup:
    def __init__(self, matrices, center=(0., 0., 0.), tolerance=1e-10):
        self.matrices=np.asarray(matrices,dtype=float)
        self.center=np.asarray(center,dtype=float)
        self.tolerance=float(tolerance)
        if (self.matrices.ndim!=3 or self.matrices.shape[1:]!=(3,3) or not len(self.matrices)
                or self.center.shape!=(3,) or not np.isfinite(self.matrices).all()
                or not np.isfinite(self.center).all() or not 0<self.tolerance<=1e-6):
            raise ValueError('Invalid finite point group')
        a=self.matrices;tol=self.tolerance
        if np.max(abs(a@a.transpose(0,2,1)-np.eye(3)))>tol:
            raise ValueError('Group matrices must be orthogonal')
        self.determinants=np.rint(np.linalg.det(a)).astype(int)
        def locate(x):
            found=np.flatnonzero(np.max(abs(a-x),axis=(1,2))<=tol)
            if len(found)!=1:raise ValueError('Group must be closed, unique and contain identity')
            return int(found[0])
        self.identity=locate(np.eye(3))
        self.products=np.array([[locate(x@y) for y in a] for x in a])

    @classmethod
    def from_generators(cls, generators, center=(0.,0.,0.), max_order=240):
        generators=np.asarray(generators,float).reshape(-1,3,3)
        if not np.isfinite(generators).all() or (len(generators) and
                np.max(abs(generators@generators.transpose(0,2,1)-np.eye(3)))>1e-10):
            raise ValueError('Generators must be finite orthogonal matrices')
        matrices=[np.eye(3)];i=0
        while i<len(matrices):
            for generator in generators:
                candidate=matrices[i]@generator
                if all(np.max(abs(candidate-old))>1e-10 for old in matrices):
                    if len(matrices)>=max_order:raise ValueError('Generators do not close within max_order')
                    matrices.append(candidate)
            i+=1
        return cls(matrices,center)

    def transform(self, points, index):
        return (np.asarray(points)-self.center)@self.matrices[index].T+self.center

    def face_maps(self, geometry):
        """Check triangle connectivity, vertex locations and outward normals."""
        vertices=np.asarray(geometry.vertices,float)
        normals=np.asarray(geometry.normals,float)
        centers=vertices.mean(axis=1)
        tree=cKDTree(centers)
        scale=max(float(np.max(np.linalg.norm(vertices-self.center,axis=2))),1.)
        tol=self.tolerance*scale
        owner=np.full(len(vertices),-1,dtype=int)
        for index,face in enumerate(geometry.faces):
            if np.any(owner[face]!=-1):raise ValueError('Overlapping face partition')
            owner[face]=index
        if np.any(owner<0):raise ValueError('Incomplete face partition')
        maps=[]
        for index,R in enumerate(self.matrices):
            distance,permutation=tree.query(self.transform(centers,index))
            if np.max(distance)>tol or len(np.unique(permutation))!=len(vertices):
                raise ValueError('Point group does not preserve the triangulated mesh')
            moved=self.transform(vertices,index)
            distances=np.linalg.norm(moved[:,:,None,:]-vertices[permutation,None,:,:],axis=-1)
            if (np.max(np.min(distances,axis=2))>tol or
                    np.max(np.min(distances,axis=1))>tol or
                    np.max(np.linalg.norm(normals@R.T-normals[permutation],axis=1))>self.tolerance):
                raise ValueError('Point group changes triangle vertices or outward normals')
            mapping=[]
            for face in geometry.faces:
                targets=np.unique(owner[permutation[face]])
                if len(targets)!=1 or len(face)!=len(geometry.faces[targets[0]]):
                    raise ValueError('Point group does not preserve the face partition')
                mapping.append(int(targets[0]))
            maps.append(mapping)
        maps=np.array(maps)
        if any(not np.array_equal(maps[i,maps[j]],maps[self.products[i,j]])
               for i in range(len(maps)) for j in range(len(maps))):
            raise ValueError('Inconsistent face group action')
        return maps
