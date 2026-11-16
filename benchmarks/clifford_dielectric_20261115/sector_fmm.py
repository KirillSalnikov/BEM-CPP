"""Symmetry-reduced FMM with exact selected near blocks, no dense far cache."""
import ctypes as ct
import sys,time
import numpy as np
from solve import Dirac,lib,address,check,B
from dirac_symmetry import CyclicGeometry
sys.path.insert(0,str(B.parent/'clifford_fmm_20261115'))
from native import NativeFMM

lib.dirac_build_near_rows.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_size_t]
lib.dirac_near_rows.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p]
lib.dirac_self_blocks_direct.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p]
lib.dirac_direct_transformed_columns.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p]

class SectorFMM(Dirac):
    def __init__(self,shape='prism',ref=2,ka=10,index=1.3116,aspect=1.1547005383792517,
                 degree=3,order=24,target_order=13,near_subdiv=1,max_bytes=8<<30,
                 eps=1e-10,strict_eps=1e-12,fmm_packet=1,horizontal=True,audit=True):
        if not 1e-13<=strict_eps<=eps<=1e-7 or fmm_packet not in (1,2,4):raise ValueError('FMM accuracy or packet controls')
        if shape!='prism' or target_order!=13:raise ValueError('sector FMM requires a symmetric prism and target13')
        start=time.perf_counter()
        super().__init__(shape,ref,ka,index,aspect,order,target_order,max_bytes,degree,near_subdiv=near_subdiv,geometry_only=True)
        self._matrix_free=True;self.eps=eps;self.strict_eps=strict_eps;self.fmm_packet=fmm_packet
        self.parameters=dict(shape=shape,ref=ref,ka=ka,index=index,aspect=aspect,order=order,target_order=target_order,near_subdiv=near_subdiv)
        self._compact_geometry=CyclicGeometry(self,horizontal=horizontal);g=self._compact_geometry
        self.ids=np.ascontiguousarray(g.representatives,np.int32)
        check(lib.dirac_build_near_rows(self.p,len(self.ids),address(self.ids),max_bytes))
        self._geometry_only=False;self._correction_only=True
        self.engine=NativeFMM();q=self.quadrature();self.points=np.asfortranarray(q[:,:,:3].reshape(-1,3).T)
        self.target_slots=(self.ids[:,None]*13+np.arange(13)).ravel()
        self.targets=np.asfortranarray(self.points[:,self.target_slots])
        self.fmm_s=0.;self.near_s=0.;self.projection_s=0.
        self.configure_dense('simd',1);self._dense_kernel='fmm';self.acceleration['kernel']='fmm_exact_near'
        self.cache_info=dict(policy='near_orbits',group_size=g.group_size,representative_panels=len(self.ids),
            kernel_bytes=int(lib.dirac_bytes(self.p)),dense_far_bytes=0,point_sources=self.points.shape[1])
        if audit:self.cache_info['direct_audit']=g.audit_direct()
        self.setup_s=time.perf_counter()-start;self.calls=0;self.action_s=0.

    def self_blocks_selected(self,ids,out):
        check(lib.dirac_self_blocks_direct(self.p,len(ids),address(ids),address(out)))

    def _near_rows(self,x,medium):
        x=np.ascontiguousarray(x);out=np.empty((len(x),self._compact_geometry.size),complex)
        for j,v in enumerate(x):check(lib.dirac_near_rows(self.p,medium,address(v),len(self.ids),address(self.ids),address(out[j])))
        return out

    def _near_full(self,x,medium):
        g=self._compact_geometry;out=np.empty((len(x),self.count,self.basis_count,8),complex)
        for transform in g.transforms:
            part=self._near_rows(transform.inverse(x),medium).reshape(len(x),len(self.ids),self.basis_count,8)
            out[:,transform.image[self.ids]]=np.einsum('pab,rpbc,pdc->rpad',transform.basis[self.ids],part,transform.components[self.ids],optimize=True)
        return out.reshape(len(x),self.size)

    def _cauchy_packet(self,x,medium,eps,selected):
        n=len(x);charges=[]
        for v in x:
            q=np.empty((13*self.count,8),complex);lib.dirac_quadrature_charges(self.p,address(np.ascontiguousarray(v)),address(q));charges.append(q.T)
        q=np.asfortranarray(np.asarray(charges).reshape(n*8,-1));k=self.ka*(self.index if medium else 1)
        begin=time.perf_counter()
        if selected:v=self.engine.targets(eps=eps,zk=k,sources=self.points,charges=q,targets=self.targets)
        else:v=self.engine.hfmm3d(eps=eps,zk=k,sources=self.points,charges=q,pg=2,nd=n*8)
        self.fmm_s+=time.perf_counter()-begin
        if v.ier:raise RuntimeError('FMM3D error '+str(v.ier))
        pot=v.pot.reshape(n,8,-1);g=-2*v.grad.reshape(n,8,3,-1);s=2j*k*pot
        f=np.empty((n,pot.shape[-1],8),complex)
        f[:,:,0]=g[:,0,0]+g[:,1,1]+g[:,2,2]+s[:,7]
        f[:,:,4]=-g[:,4,0]-g[:,5,1]-g[:,6,2]+s[:,3]
        f[:,:,1]=g[:,2,1]-g[:,1,2]-g[:,3,0]+s[:,4]
        f[:,:,2]=g[:,0,2]-g[:,2,0]-g[:,3,1]+s[:,5]
        f[:,:,3]=g[:,1,0]-g[:,0,1]-g[:,3,2]+s[:,6]
        f[:,:,5]=g[:,6,1]-g[:,5,2]-g[:,7,0]-s[:,0]
        f[:,:,6]=g[:,4,2]-g[:,6,0]-g[:,7,1]-s[:,1]
        f[:,:,7]=g[:,5,0]-g[:,4,1]-g[:,7,2]-s[:,2]
        begin=time.perf_counter();projected=np.empty((n,self.size),complex)
        for j in range(n):
            if selected:
                full=np.zeros((13*self.count,8),complex);full[self.target_slots]=f[j]
            else:full=np.ascontiguousarray(f[j])
            lib.dirac_project_cauchy(self.p,address(full),address(projected[j]))
        if selected:projected=self._compact_geometry.restrict(projected)
        self.projection_s+=time.perf_counter()-begin
        begin=time.perf_counter();near=self._near_rows(x,medium) if selected else self._near_full(x,medium)
        self.near_s+=time.perf_counter()-begin
        return projected+near

    def _run(self,x,selected,eps):
        x=np.ascontiguousarray(x,dtype=complex)
        if x.ndim!=2 or x.shape[1]!=self.size or not 1<=len(x)<=16:raise ValueError('FMM batch dimensions')
        begin=time.perf_counter();out=[]
        for start in range(0,len(x),self.fmm_packet):
            part=x[start:start+self.fmm_packet];local=part.reshape(len(part),-1,8)
            e=self._cauchy_packet((local*self.scaling[3]).reshape(part.shape),0,eps,selected).reshape(len(part),-1,8)
            i=self._cauchy_packet((local*self.scaling[1]).reshape(part.shape),1,eps,selected).reshape(len(part),-1,8)
            identity=self._compact_geometry.restrict(part).reshape(len(part),-1,8) if selected else local
            out.append((identity+self.scaling[0]*i-self.scaling[2]*e).reshape(len(part),-1))
        self.action_s+=time.perf_counter()-begin;self.calls+=len(x);return np.concatenate(out)

    def apply(self,x):return self._run(np.asarray(x).reshape(1,-1),False,self.eps)[0]
    def apply_batch(self,x,tile=256):return self._run(x,False,self.eps)
    def apply_sector_batch(self,x):return self._run(x,True,self.eps)
    def apply_reference(self,x):return self._run(np.asarray(x).reshape(1,-1),False,self.strict_eps)[0]
    def apply_reference_batch(self,x):return self._run(x,False,self.strict_eps)
    def cauchy(self,x,medium=0):return self._cauchy_packet(np.ascontiguousarray(x).reshape(1,-1),medium,self.strict_eps,False)[0]
