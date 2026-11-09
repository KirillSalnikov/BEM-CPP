"""Finite O(3) groups and verified actions on triangulated surfaces.

This is an orbit reduction of physical blocks, not a C6 Fourier-sector
decomposition. Noncommuting groups and improper rotations are supported.
"""
import numpy as np
from scipy.spatial import cKDTree


class PointGroup:
    @classmethod
    def detect(cls, vertices, triangles=None, tolerance=1e-9):
        """Find all finite orthogonal symmetries of a 3-D vertex/triangle set.

        Three independent anchor images determine an isometry. Gram-matrix
        filtering avoids assuming an axis, particle name or rotation order.
        Triangle connectivity is checked when supplied, not just the envelope.
        """
        points=np.asarray(vertices,float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
            raise ValueError('Expected finite 3-D vertices')
        if len(points)<4:raise ValueError('A closed 3-D surface needs four vertices')
        center=points.mean(axis=0);x=points-center
        scale=np.max(np.linalg.norm(x,axis=1))
        if not scale>0:raise ValueError('Degenerate point set')
        x=x/scale;r2=np.sum(x*x,axis=1);tree=cKDTree(x)
        a=int(np.argmax(r2));b=int(np.argmax(np.linalg.norm(np.cross(x[a],x),axis=1)))
        c=int(np.argmax(abs(x@np.cross(x[a],x[b]))))
        anchors=x[[a,b,c]].T
        if abs(np.linalg.det(anchors))<1e-12:raise ValueError('Symmetry detection requires a 3-D surface')
        inv=np.linalg.inv(anchors);gram=anchors.T@anchors
        candidates=[np.flatnonzero(abs(r2-gram[j,j])<4*tolerance) for j in range(3)]
        faces=None if triangles is None else np.asarray(triangles,int)
        if faces is not None and (faces.ndim!=2 or faces.shape[1]!=3 or faces.min()<0 or faces.max()>=len(x)):
            raise ValueError('Invalid symmetry triangle indices')
        keys=None if faces is None else {tuple(sorted(t)) for t in faces}
        result=[]
        for i in candidates[0]:
            for j in candidates[1][abs(x[candidates[1]]@x[i]-gram[0,1])<4*tolerance]:
                possible=candidates[2]
                possible=possible[(abs(x[possible]@x[i]-gram[0,2])<4*tolerance)&
                                  (abs(x[possible]@x[j]-gram[1,2])<4*tolerance)]
                for k in possible:
                    R=x[[i,j,k]].T@inv
                    if np.max(abs(R.T@R-np.eye(3)))>8*tolerance:continue
                    u,_,vh=np.linalg.svd(R);R=u@vh
                    distance,permutation=tree.query(x@R.T)
                    if distance.max()>tolerance or len(np.unique(permutation))!=len(x):continue
                    if keys is not None and {tuple(sorted(t)) for t in permutation[faces]}!=keys:continue
                    if all(np.max(abs(R-old))>tolerance for old in result):result.append(R)
        if not result:raise ValueError('No identity symmetry found')
        # Identity first; stable ordering makes metadata/checkpoints reproducible.
        result.sort(key=lambda R:(np.max(abs(R-np.eye(3)))>tolerance,tuple(np.round(R,10).ravel())))
        return cls(result,center,tolerance=min(1e-6,16*tolerance))

    def summary(self):
        proper=self.matrices[self.determinants==1]
        orders=[]
        for R in proper:
            angle=np.arccos(np.clip((np.trace(R)-1)/2,-1.,1.))
            orders.append(1 if angle<1e-7 else int(round(2*np.pi/angle)))
        mirrors=sum(int(det==-1 and abs(np.trace(R)-1)<1e-7)
                    for R,det in zip(self.matrices,self.determinants))
        return dict(order=len(self.matrices),proper_rotations=len(proper),
                    improper_operations=int(np.sum(self.determinants<0)),mirror_planes=mirrors,
                    inversion=bool(np.any(np.max(abs(self.matrices+np.eye(3)),axis=(1,2))<self.tolerance)),
                    maximum_rotation_order=max(orders),center=self.center.tolist(),
                    matrices=self.matrices.tolist(),tolerance=self.tolerance)

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
