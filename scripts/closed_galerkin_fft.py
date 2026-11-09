"""Experimental closed regular-prism action, with exact global BDM incidence.

The spectral part replaces regular quadrature only. Near corrections are
exported independently by MullerFmmOperator; the 13-point mass stays explicit.
No fitted solutions or dense reference matrix enter construction.
"""
from dataclasses import dataclass
import json
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from galerkin_cross_fft import Face, CrossGalerkinFFT

TRIANGLES=(((0,0),(1,0),(1,1)),((0,0),(1,1),(0,1)),
           ((0,0),(1,0),(0,1)),((1,0),(1,1),(0,1)))

@dataclass
class OrientedFace(Face):
    outward: object = None
    @property
    def normal(self):
        return self.outward

class Geometry:
    def __init__(self,directory,fixture):
        self.directory=Path(directory);self.fixture=fixture
        self.meta=json.loads((self.directory/'geometry.json').read_text())
        self.n=self.meta['current_dofs'];self.N=2*self.n
        def read(name,dtype,shape):return np.fromfile(self.directory/(name+'.bin'),dtype).reshape(shape)
        self.vertices=read('vertices',float,(-1,3,3));self.points=read('points',float,(-1,7,3))
        self.basis=read('basis',float,(-1,7,6,3));self.normals=read('normals',float,(-1,3))
        self.dofs=read('dofs',np.int32,(-1,6))
        self.faces=[];remaining=set(range(len(self.vertices)))
        while remaining:
            t=min(remaining);indices=np.array(sorted(i for i in remaining if np.linalg.norm(self.normals[i]-self.normals[t])<1e-10))
            self.faces.append(indices);remaining.difference_update(indices)
        if len(self.faces)!=8:raise ValueError('Expected six sides and two caps')
        edges=(self.vertices-np.roll(self.vertices,1,axis=1)).reshape(-1,3)
        horizontal=edges[(abs(edges[:,2])<1e-12)&(np.linalg.norm(edges[:,:2],axis=1)>1e-12)]
        self.h=float(np.min(np.linalg.norm(horizontal,axis=1)))
        zs=np.unique(np.round(self.vertices[:,:,2].ravel(),12));self.dz=float(np.min(np.diff(zs)))
        # Use the true (unrounded) minimum positive vertical edge length.
        self.dz=float(np.min(abs(edges[abs(edges[:,2])>1e-10,2])))
        self.layouts={};self.max_mapping_error=0.;self.max_lattice_error=0.
        if not np.all(np.bincount(self.dofs.ravel(),minlength=self.n)==2):
            raise ValueError('Closed BDM edges must have two triangle incidences')

    def axes(self,t,s):
        nt=self.normals[self.faces[t][0]];ns=self.normals[self.faces[s][0]]
        ct=abs(nt[2])>.5;cs=abs(ns[2])>.5
        z=np.array([0.,0.,self.dz])
        def side(n):return np.cross([0.,0.,1.],n)*self.h
        if not ct and not cs:return side(nt),side(ns),z
        if ct and cs:
            side_face=next(f for f in self.faces if abs(self.normals[f[0],2])<.5)
            common=side(self.normals[side_face[0]])
            other=np.array([.5*common[0]-np.sqrt(3)*.5*common[1],np.sqrt(3)*.5*common[0]+.5*common[1],0.])
            return other,other,common
        common=side(ns if ct else nt)
        other=np.array([.5*common[0]-np.sqrt(3)*.5*common[1],np.sqrt(3)*.5*common[0]+.5*common[1],0.])
        return (other,z,common) if ct else (z,other,common)

    def layout(self,face_id,transverse,common):
        key=(face_id,tuple(np.round(transverse,12)),tuple(np.round(common,12)))
        if key in self.layouts:return self.layouts[key]
        triangles=self.faces[face_id];vertices=self.vertices[triangles]
        axes=np.stack((transverse,common));origin=vertices[0,0].copy()
        coords=(vertices-origin)@np.linalg.pinv(axes)
        rounded=np.rint(coords).astype(int);err=float(np.max(abs(coords-rounded)))
        if err>1e-8:raise ValueError(f'Nonlattice face {face_id}: {err}')
        self.max_lattice_error=max(self.max_lattice_error,err)
        lower=rounded.min(axis=(0,1));rounded-=lower;origin+=lower@axes
        nx,ny=rounded.max(axis=(0,1));records=[];patterns=[]
        for t,cv in zip(triangles,rounded):
            cell=cv.min(axis=0);local=cv-cell;matched=[p for p,v in enumerate(TRIANGLES) if set(map(tuple,local))==set(v)]
            if len(matched)!=1:raise ValueError(f'Invalid triangle {local}')
            p=matched[0];patterns.append(p)
            points=np.asarray(self.fixture['patterns'][p]['points'])@axes+origin+cell@axes
            distance=np.linalg.norm(points[:,None,:]-self.points[t][None,:,:],axis=2)
            perm=distance.argmin(axis=1)
            if len(set(perm))!=7 or distance[np.arange(7),perm].max()>1e-9:raise ValueError('Quadrature permutation')
            wc=(np.asarray(self.fixture['patterns'][p]['weighted_basis'])@axes).transpose(0,2,1).reshape(21,6)
            wa=self.basis[t,perm].transpose(0,2,1).reshape(21,6)
            matrix=np.linalg.lstsq(wc,wa,rcond=None)[0]
            nearest=np.rint(matrix);error=float(max(np.linalg.norm(wc@nearest-wa)/np.linalg.norm(wa),np.max(abs(matrix-nearest))))
            self.max_mapping_error=max(self.max_mapping_error,error)
            if error>1e-10 or not np.all(abs(nearest).sum(axis=0)==1) or not np.all(abs(nearest).sum(axis=1)==1):
                raise ValueError(f'BDM map is not a signed permutation: {error}')
            columns=abs(nearest).argmax(axis=1);sign=nearest[np.arange(6),columns]
            records.append((t,cell,p,self.dofs[t,columns],sign))
        patterns=tuple(sorted(set(patterns)))
        face=OrientedFace(origin,transverse,common,int(nx),int(ny),patterns,self.normals[triangles[0]])
        local=[];global_=[];signs=[];tc=int(nx)*len(patterns)*6
        for t,(i,j),p,dofs,sign in records:
            channels=(i*len(patterns)+patterns.index(p))*6+np.arange(6)
            for current in range(2):
                local.extend((channels+current*tc)*ny+j);global_.extend(dofs+current*self.n);signs.extend(sign)
        layout=dict(face=face,local=np.array(local,dtype=np.int64),global_=np.array(global_,dtype=np.int64),
                    signs=np.array(signs),records=records,shape=(2*tc,int(ny)))
        if len(set(local))!=len(local):raise ValueError('Duplicate local triangle slot')
        self.layouts[key]=layout;return layout

class ClosedGalerkinFFT:
    def __init__(self,geometry,progress=None):
        self.geometry=geometry;self.pairs=[];self.preparation_s=0.
        start=time.perf_counter()
        for t in range(len(geometry.faces)):
            for s in range(len(geometry.faces)):
                u,v,c=geometry.axes(t,s);target=geometry.layout(t,u,c);source=geometry.layout(s,v,c)
                op=CrossGalerkinFFT(geometry.fixture,target['face'],source['face'],geometry.meta['ka'],complex(*geometry.meta['ri'])).build()
                self.pairs.append((op,target,source))
            if progress:progress(t+1,sum(op.symbol.nbytes for op,_,_ in self.pairs))
        self.preparation_s=time.perf_counter()-start
        self.regular_bytes=sum(op.symbol.nbytes for op,_,_ in self.pairs)
        self.local=None

    def load_correction(self):
        g=self.geometry;d=g.directory;n=g.n;k=g.meta['ka'];ri=complex(*g.meta['ri'])
        rows=np.fromfile(d/'near_rows.bin',np.int32);cols=np.fromfile(d/'near_cols.bin',np.int32)
        values=np.fromfile(d/'near_values.bin',complex).reshape(-1,3)
        matrices=[sparse.csr_matrix((values[:,j],cols,rows),shape=(n,n)) for j in range(3)]
        if (d/'mass_values.bin').exists():
            mass=sparse.coo_matrix((np.fromfile(d/'mass_values.bin',float),
                (np.fromfile(d/'mass_rows.bin',np.int32),np.fromfile(d/'mass_cols.bin',np.int32))),shape=(n,n)).tocsr()
        else:
            mass=sparse.csr_matrix(np.fromfile(d/'mass.bin',float).reshape(n,n))
        self.local=sparse.bmat([[1j/k*matrices[0],matrices[1]+.5*(ri*ri+1)*mass],
                               [matrices[2]+mass,-1j/k*matrices[0]]],format='csr')
        self.local.eliminate_zeros()

    def apply(self,x):
        g=self.geometry;y=np.zeros(g.N,complex);gathered={}
        for op,t,s in self.pairs:
            key=id(s)
            if key not in gathered:
                v=np.zeros(np.prod(s['shape']),complex);v[s['local']]=s['signs']*x[s['global_']]
                gathered[key]=v.reshape(s['shape'])
            output=op.apply(gathered[key]).ravel()
            np.add.at(y,t['global_'],t['signs']*output[t['local']])
        if self.local is not None:y+=self.local@x
        return y

    def torch_operator(self,device='cuda',dtype=None):
        import torch
        dtype=torch.complex128 if dtype is None else dtype
        return TorchClosedAction(self,device,dtype)

class TorchClosedAction:
    def __init__(self,cpu,device,dtype):
        import torch
        self.cpu,self.device,self.dtype=cpu,device,dtype;self.N=cpu.geometry.N
        self.layouts={};self.pairs=[]
        for op,t,s in cpu.pairs:
            for layout in (t,s):
                if id(layout) not in self.layouts:
                    self.layouts[id(layout)]={name:torch.as_tensor(layout[name],device=device) for name in ('local','global_','signs')}
                    self.layouts[id(layout)]['signs']=self.layouts[id(layout)]['signs'].to(dtype)
                    self.layouts[id(layout)]['shape']=layout['shape']
            self.pairs.append((torch.as_tensor(op.symbol,device=device,dtype=dtype),op.nfft,self.layouts[id(t)],self.layouts[id(s)]))
        coo=cpu.local.tocoo()
        # PyTorch CUDA sparse complex CSR support varies; COO sparse.mm is portable.
        self.local=torch.sparse_coo_tensor(torch.as_tensor(np.stack((coo.row,coo.col)),device=device),
                    torch.as_tensor(coo.data,device=device,dtype=dtype),(self.N,self.N)).coalesce()
        self.bytes=sum(a.numel()*a.element_size() for a,_,_,_ in self.pairs)+self.local._nnz()*(16+self.local.element_size())

    def __call__(self,x):
        import torch
        y=torch.zeros(self.N,device=self.device,dtype=self.dtype);gathered={};transforms={}
        for symbol,nfft,t,s in self.pairs:
            key=id(s)
            if key not in gathered:
                v=torch.zeros(s['shape'][0]*s['shape'][1],device=self.device,dtype=self.dtype)
                v[s['local']]=s['signs']*x[s['global_']];gathered[key]=v.reshape(s['shape'])
            fkey=(key,nfft)
            if fkey not in transforms:transforms[fkey]=torch.fft.fft(gathered[key],n=nfft,dim=1).transpose(0,1).unsqueeze(-1)
            out=torch.fft.ifft(torch.bmm(symbol,transforms[fkey]).squeeze(-1),dim=0)[:t['shape'][1]].transpose(0,1).reshape(-1)
            y.index_add_(0,t['global_'],t['signs']*out[t['local']])
        return y+torch.sparse.mm(self.local,x[:,None])[:,0]

class TorchBatchedClosedAction:
    """Batch equal-sized face pairs: typically four groups instead of 64 launches.

    This only changes execution order. It uses every original face-pair symbol,
    both currents, and the same finite masks and global edge assembly.
    """
    def __init__(self,cpu,device,dtype,compressed=False):
        import torch
        self.device,self.dtype,self.N=device,dtype,cpu.geometry.N
        groups={}
        for op,t,s in cpu.pairs:
            factors=getattr(op,'gpu_factors',None) if compressed else None
            rank=factors[0].shape[-1] if factors is not None else 0
            key=(op.nfft,op.symbol.shape[1],op.symbol.shape[2],t['shape'][1],s['shape'][1],rank)
            groups.setdefault(key,[]).append((op,t,s))
        self.groups=[];self.bytes=0
        for (nfft,tc,sc,tny,sny,rank),pairs in groups.items():
            count=len(pairs);indices=np.zeros((count,sc*sny),np.int64);signs=np.zeros((count,sc*sny))
            target_local=[];target_global=[];target_sign=[]
            for i,(op,t,s) in enumerate(pairs):
                indices[i,s['local']]=s['global_'];signs[i,s['local']]=s['signs']
                target_local.extend(i*tc*tny+t['local']);target_global.extend(t['global_']);target_sign.extend(t['signs'])
            symbol=torch.empty((count,nfft,tc,rank or sc),device=device,dtype=dtype)
            right=torch.empty((count,nfft,rank,sc),device=device,dtype=dtype) if rank else None
            for i,(op,_,_) in enumerate(pairs):
                symbol[i].copy_(torch.as_tensor(op.gpu_factors[0] if rank else op.symbol,device=device,dtype=dtype))
                if rank:right[i].copy_(torch.as_tensor(op.gpu_factors[1],device=device,dtype=dtype))
            group=dict(symbol=symbol.reshape(count*nfft,tc,rank or sc),right=right.reshape(count*nfft,rank,sc) if rank else None,
                       nfft=nfft,tc=tc,sc=sc,tny=tny,sny=sny,count=count,
                       indices=torch.as_tensor(indices.reshape(count,sc,sny),device=device),
                       signs=torch.as_tensor(signs.reshape(count,sc,sny),device=device,dtype=dtype),
                       target_local=torch.as_tensor(np.array(target_local),device=device),
                       target_global=torch.as_tensor(np.array(target_global),device=device),
                       target_sign=torch.as_tensor(np.array(target_sign),device=device,dtype=dtype))
            self.bytes+=sum(v.numel()*v.element_size() for v in group.values() if torch.is_tensor(v));self.groups.append(group)
        coo=cpu.local.tocoo()
        self.local=torch.sparse_coo_tensor(torch.as_tensor(np.stack((coo.row,coo.col)),device=device),
                    torch.as_tensor(coo.data,device=device,dtype=dtype),(self.N,self.N)).coalesce()
        self.bytes+=self.local._nnz()*(16+self.local.element_size())

    def __call__(self,x):
        import torch
        y=torch.zeros(self.N,device=self.device,dtype=self.dtype)
        for g in self.groups:
            values=x[g['indices']]*g['signs']
            transformed=torch.fft.fft(values,n=g['nfft'],dim=2).transpose(1,2).reshape(g['count']*g['nfft'],g['sc'],1)
            if g['right'] is not None:transformed=torch.bmm(g['right'],transformed)
            spectral=torch.bmm(g['symbol'],transformed).reshape(g['count'],g['nfft'],g['tc'])
            values=torch.fft.ifft(spectral,dim=1)[:,:g['tny']].transpose(1,2).reshape(-1)
            y.index_add_(0,g['target_global'],g['target_sign']*values[g['target_local']])
        return y+torch.sparse.mm(self.local,x[:,None])[:,0]
