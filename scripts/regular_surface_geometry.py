"""BDM layouts for regular planar faces without a prescribed particle shape.

Axes are explicit lattice steps, one pair per planar face. Unsupported face
pairs fail rather than silently replacing the physical operator. The ordinary
FMM solver remains the fallback for irregular or curved meshes.
"""
import json
from pathlib import Path
import numpy as np
from closed_galerkin_fft import Geometry


class RegularSurfaceGeometry(Geometry):
    def __init__(self,directory,fixture,face_axes):
        self.directory=Path(directory);self.fixture=fixture
        self.meta=json.loads((self.directory/'geometry.json').read_text())
        self.n=self.meta['current_dofs'];self.N=2*self.n
        def read(name,dtype,shape):return np.fromfile(self.directory/(name+'.bin'),dtype).reshape(shape)
        self.vertices=read('vertices',float,(-1,3,3));self.points=read('points',float,(-1,7,3))
        self.basis=read('basis',float,(-1,7,6,3));self.normals=read('normals',float,(-1,3))
        self.dofs=read('dofs',np.int32,(-1,6))
        if not all(np.isfinite(a).all() for a in (self.vertices,self.points,self.basis,self.normals)):
            raise ValueError('Nonfinite surface geometry')
        if len(self.normals)!=len(self.vertices) or self.dofs.shape!=(len(self.vertices),6):
            raise ValueError('Inconsistent surface arrays')
        if self.dofs.min()<0 or self.dofs.max()>=self.n or not np.all(np.bincount(self.dofs.ravel(),minlength=self.n)==2):
            raise ValueError('Closed BDM edges must have two triangle incidences')
        self.faces=[];remaining=set(range(len(self.vertices)))
        scale=max(1.,float(np.max(np.linalg.norm(self.vertices,axis=2))))
        while remaining:
            t=min(remaining);normal=self.normals[t];origin=self.vertices[t,0]
            indices=np.array(sorted(i for i in remaining if
                np.linalg.norm(self.normals[i]-normal)<1e-10 and
                np.max(abs((self.vertices[i]-origin)@normal))<1e-10*scale))
            if not len(indices):raise ValueError('Degenerate/nonplanar triangle')
            self.faces.append(indices);remaining.difference_update(indices)
        self.face_axes=np.asarray(face_axes,float)
        if self.face_axes.shape!=(len(self.faces),2,3) or not np.isfinite(self.face_axes).all():
            raise ValueError('Provide two lattice steps per planar face, in first-triangle order')
        self.layouts={};self.max_mapping_error=0.;self.max_lattice_error=0.;self.pair_axes={}
        for face,(u,c) in enumerate(self.face_axes):self.layout(face,u,c)

    def axes(self,t,s):
        if (t,s) in self.pair_axes:return self.pair_axes[t,s]
        def candidates(index):
            a,b=self.face_axes[index]
            return [a,b,-a,-b,a+b,a-b,-a+b,-a-b]
        target,source=candidates(t),candidates(s)
        for common in target:
            if not any(np.allclose(common,c,rtol=0,atol=1e-12) for c in source):continue
            for u in target:
                if np.linalg.norm(np.cross(u,common))<1e-12:continue
                for v in source:
                    if np.linalg.norm(np.cross(v,common))<1e-12:continue
                    try:self.layout(t,u,common);self.layout(s,v,common)
                    except ValueError:continue
                    self.pair_axes[t,s]=(u,v,common)
                    return u,v,common
        raise ValueError(f'Faces {t},{s} have no supported common lattice: use FMM for this mesh')
