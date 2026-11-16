"""CPU screening CLI for the complete eight-density dielectric Dirac equation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import ctypes as ct
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

B=Path(__file__).resolve().parent
R=B.parents[1]
sys.path.insert(0,str(R))
from verify_mie import mie_mueller

LIBRARY=Path(os.environ.get('BEM_DIRAC_LIBRARY',str(B/'libdirac.so'))).resolve()
lib=ct.CDLL(str(LIBRARY))
ptr=ct.c_void_p
lib.dirac_create.argtypes=[ct.c_int,ct.c_int,ct.c_double,ct.c_double,ct.c_double,ct.c_double,ct.c_int,ct.c_size_t,ct.c_int,ct.c_int]
lib.dirac_create.restype=ptr
lib.dirac_create_near.argtypes=lib.dirac_create.argtypes;lib.dirac_create_near.restype=ptr
lib.dirac_create_near_ex.argtypes=lib.dirac_create.argtypes+[ct.c_int];lib.dirac_create_near_ex.restype=ptr
if hasattr(lib,'dirac_create_dense_ex'):
    lib.dirac_create_dense_ex.argtypes=lib.dirac_create.argtypes+[ct.c_int];lib.dirac_create_dense_ex.restype=ptr
lib.dirac_near_correction.argtypes=[ptr,ct.c_int,ptr,ptr]
lib.dirac_quadrature_charges.argtypes=[ptr,ptr,ptr]
lib.dirac_project_cauchy.argtypes=[ptr,ptr,ptr]
lib.dirac_destroy.argtypes=[ptr]
lib.dirac_size.argtypes=[ptr];lib.dirac_size.restype=ct.c_int
lib.dirac_basis_count.argtypes=[ptr];lib.dirac_basis_count.restype=ct.c_int
lib.dirac_bytes.argtypes=[ptr];lib.dirac_bytes.restype=ct.c_size_t
lib.dirac_geometry.argtypes=[ptr,ptr]
lib.dirac_scaling.argtypes=[ptr,ptr]
lib.dirac_quadrature.argtypes=[ptr,ptr]
lib.dirac_apply.argtypes=[ptr,ptr,ptr]
if hasattr(lib,'dirac_apply_simd'):
    lib.dirac_apply_simd.argtypes=[ptr,ptr,ptr]
    lib.dirac_apply_mixed.argtypes=[ptr,ptr,ptr]
    lib.dirac_prepare_mixed.argtypes=[ptr,ct.c_size_t]
    lib.dirac_diagonal_blocks.argtypes=[ptr,ct.c_int,ptr]
    lib.dirac_block_lu.argtypes=[ct.c_int,ct.c_int,ptr,ptr,ptr,ptr]
if hasattr(lib,'dirac_apply_batch'):
    lib.dirac_apply_batch.argtypes=[ptr,ct.c_int,ptr,ptr,ct.c_int]
if hasattr(lib,'dirac_create_geometry'):
    lib.dirac_create_geometry.argtypes=lib.dirac_create.argtypes+[ct.c_int];lib.dirac_create_geometry.restype=ptr
    lib.dirac_build_rows.argtypes=[ptr,ct.c_int,ptr,ct.c_size_t]
    lib.dirac_apply_rows_reference.argtypes=[ptr,ptr,ct.c_int,ptr,ptr]
    lib.dirac_apply_direct_panels.argtypes=[ptr,ptr,ct.c_int,ptr,ptr]
    lib.dirac_self_blocks_selected.argtypes=[ptr,ct.c_int,ptr,ptr]
    lib.dirac_apply_rows.argtypes=[ptr,ct.c_int,ptr,ct.c_int,ptr,ptr,ct.c_int]
lib.dirac_rhs.argtypes=[ptr,ptr,ptr]
lib.dirac_project_fields.argtypes=[ptr,ptr,ptr]
lib.dirac_incident.argtypes=[ptr,ptr,ptr,ptr]
lib.dirac_cauchy.argtypes=[ptr,ct.c_int,ptr,ptr]
lib.dirac_field.argtypes=[ptr,ct.c_int,ptr,ct.c_int,ptr,ptr,ct.c_int]
lib.dirac_error.restype=ct.c_char_p
if hasattr(lib,'dirac_apply_rows_reference_batch'):
    lib.dirac_apply_rows_reference_batch.argtypes=[ptr,ct.c_int,ptr,ct.c_int,ptr,ptr]
def address(x):return x.ctypes.data_as(ptr)
def check(rc):
    if rc:raise RuntimeError(lib.dirac_error().decode())

class Dirac:
    def __init__(self,shape='sphere',ref=2,ka=1.,index=1.3116,aspect=1.1547005383792517,order=14,target_order=1,max_bytes=8<<30,degree=0,correction_only=False,near_subdiv=0,dense_kernel='scalar',block_panels=0,mixed_cache_bytes=4<<30,dense_cache='full',reference_checks='batch',geometry_only=False):
        start=time.perf_counter()
        if reference_checks not in ('scalar','batch'):raise ValueError('reference check policy')
        self.reference_checks=reference_checks
        if dense_cache not in ('full','orbits12'):raise ValueError('dense cache policy')
        if dense_cache!='full' and (correction_only or dense_kernel!='simd' or block_panels!=1):raise ValueError('orbit cache requires dense SIMD and one-panel LU')
        arguments=[int(shape=='prism'),ref,aspect,ka,complex(index).real,complex(index).imag,order,max_bytes,target_order,degree]
        if dense_cache!='full' or geometry_only:
            if not hasattr(lib,'dirac_create_geometry'):raise RuntimeError('selected library lacks orbit cache')
            self.p=lib.dirac_create_geometry(*arguments,near_subdiv)
        elif correction_only:self.p=lib.dirac_create_near_ex(*arguments,near_subdiv)
        else:
            if near_subdiv:self.p=lib.dirac_create_dense_ex(*arguments,near_subdiv)
            else:self.p=lib.dirac_create(*arguments)
        if not self.p:raise RuntimeError(lib.dirac_error().decode())
        self.size=lib.dirac_size(self.p);self.basis_count=lib.dirac_basis_count(self.p);self.count=self.size//(8*self.basis_count);self.ka=ka;self.index=complex(index)
        self.target_order=target_order
        self.geometry=np.empty((self.count,13));lib.dirac_geometry(self.p,address(self.geometry))
        self.xyz=self.geometry[:,:3];self.frame=self.geometry[:,3:12].reshape(-1,3,3)
        self.scaling=np.empty((4,8),complex);lib.dirac_scaling(self.p,address(self.scaling))
        self._correction_only=correction_only;self._dense_kernel='scalar';self._dense_inverse=None
        self._compact_geometry=None;self.calls=0;self.action_s=0.;self.cache_info=dict(policy=dense_cache)
        self._geometry_only=bool(geometry_only);self._dense_factors=None
        if geometry_only:
            if correction_only:raise ValueError('geometry-only constructor cannot also build a correction cache')
            self.acceleration=dict(kernel='geometry_only',block_panels=0,setup_s=0.,preconditioner_bytes=0,inverse_s=0.)
            self.setup_s=time.perf_counter()-start;return
        if dense_cache!='full':
            from dirac_symmetry import CyclicGeometry
            g=CyclicGeometry(self,horizontal=True);ids=np.ascontiguousarray(g.representatives,dtype=np.int32)
            check(lib.dirac_build_rows(self.p,len(ids),address(ids),max_bytes));self._compact_geometry=g
            self.cache_info.update(representative_panels=len(ids),group_size=g.group_size)
        self.acceleration=dict(kernel='scalar',block_panels=0,setup_s=0.,preconditioner_bytes=0,inverse_s=0.)
        self.configure_dense(dense_kernel,block_panels,mixed_cache_bytes)
        if self._compact_geometry is not None:self.cache_info['direct_audit']=self._compact_geometry.audit_direct()
        self.cache_info['kernel_bytes']=int(lib.dirac_bytes(self.p))
        self.setup_s=time.perf_counter()-start;self.calls=0;self.action_s=0
    def close(self):
        if getattr(self,'p',None):lib.dirac_destroy(self.p);self.p=None
        # The inverse closure owns the panel factors and captures self. Merely
        # destroying the native pair cache kept these arrays alive during SO(3).
        self._dense_inverse=None;self._dense_factors=None;self._compact_geometry=None
        for name in ('geometry','xyz','frame','points','targets','target_slots'):
            if hasattr(self,name):setattr(self,name,None)
    def __del__(self):self.close()
    def apply(self,x):
        if self._compact_geometry is not None:return self._compact_geometry.full_action(x,reference=False)
        function={'scalar':'dirac_apply','simd':'dirac_apply_simd','mixed':'dirac_apply_mixed'}[self._dense_kernel]
        return self._apply_native(x,function)
    def apply_reference(self,x):
        # FMM subclasses own their full operator; a near-only cache is never
        # used as a substitute for a strict residual.
        if self._correction_only:return self.apply(x)
        if self._compact_geometry is not None:return self._compact_geometry.full_action(x,reference=True)
        return self._apply_native(x,'dirac_apply')
    def apply_reference_batch(self,x):
        x=np.ascontiguousarray(x,dtype=complex)
        if x.ndim!=2 or x.shape[1]!=self.size or not 1<=len(x)<=16:raise ValueError('reference batch dimensions')
        if self.reference_checks=='scalar' or self._correction_only or not hasattr(lib,'dirac_apply_rows_reference_batch'):
            return np.array([self.apply_reference(v) for v in x])
        if self._compact_geometry is not None:return self._compact_geometry.full_action(x,reference=True)
        rows=np.arange(self.size//8,dtype=np.int32);out=np.empty_like(x);start=time.perf_counter()
        check(lib.dirac_apply_rows_reference_batch(self.p,len(x),address(x),len(rows),address(rows),address(out)))
        self.action_s+=time.perf_counter()-start;self.calls+=len(x);return out
    def apply_batch(self,x,tile=256):
        if self._correction_only or self._dense_kernel!='simd':raise ValueError('batch action requires dense FP64 SIMD')
        x=np.ascontiguousarray(x,dtype=complex)
        if x.ndim!=2 or x.shape[1]!=self.size or not 1<=len(x)<=16:raise ValueError('batch dimensions')
        if not 16<=tile<=16384:raise ValueError('batch source tile')
        if not hasattr(lib,'dirac_apply_batch'):raise RuntimeError('selected library lacks batch action')
        if self._compact_geometry is not None:return self._compact_geometry.full_action(x,tile=tile,reference=False)
        y=np.empty_like(x);start=time.perf_counter()
        check(lib.dirac_apply_batch(self.p,len(x),address(x),address(y),tile))
        self.action_s+=time.perf_counter()-start;self.calls+=len(x);return y
    def solve_incidents(self,incidents,tol=1e-8,maxiter=10,tile=256):
        from batch_gmres import solve_incidents
        return solve_incidents(self,incidents,tol,maxiter,tile)
    def _apply_native(self,x,function):
        x=np.ascontiguousarray(x,dtype=complex).reshape(-1);y=np.empty_like(x);assert len(x)==self.size
        start=time.perf_counter();check(getattr(lib,function)(self.p,address(x),address(y)))
        self.action_s+=time.perf_counter()-start;self.calls+=1;return y
    def configure_dense(self,kernel='scalar',block_panels=0,mixed_cache_bytes=4<<30):
        if kernel not in ('scalar','simd','mixed') or not 0<=block_panels<=8:raise ValueError('dense kernel or block group')
        if self._compact_geometry is not None and (kernel!='simd' or block_panels!=1):raise ValueError('orbit cache requires SIMD and one-panel LU')
        if self._correction_only and not getattr(self,'_matrix_free',False) and (kernel!='scalar' or block_panels):raise ValueError('dense acceleration cannot be applied to a near-only FMM cache')
        if (kernel!='scalar' or block_panels) and not hasattr(lib,'dirac_apply_simd'):raise RuntimeError('selected library lacks dense acceleration')
        start=time.perf_counter();self._dense_inverse=None;self._dense_factors=None
        self.acceleration=dict(kernel=kernel,block_panels=block_panels,setup_s=0.,preconditioner_bytes=0,inverse_s=0.)
        if kernel=='mixed':
            if mixed_cache_bytes<0:raise ValueError('negative additional mixed-cache budget')
            check(lib.dirac_prepare_mixed(self.p,int(mixed_cache_bytes)))
        if block_panels:
            from scipy.linalg import lu_factor
            width=8*self.basis_count*block_panels;count=(self.count+block_panels-1)//block_panels
            if self._compact_geometry is None:
                factors=np.empty((count,width,width),complex);check(lib.dirac_diagonal_blocks(self.p,block_panels,address(factors)))
            else:factors=self._compact_geometry.self_blocks()
            pivots=np.empty((count,width),np.int32)
            for j in range(count):factors[j],pivots[j]=lu_factor(factors[j],check_finite=True)
            if not np.all(np.isfinite(factors)):raise RuntimeError('nonfinite block factors')
            self._dense_factors=(factors,pivots)
            self.acceleration['preconditioner_bytes']=factors.nbytes+pivots.nbytes
            def inverse(x):
                x=np.ascontiguousarray(x,dtype=complex).ravel()
                if x.size!=self.size:raise ValueError('block inverse dimensions')
                padded=x if len(x)==count*width else np.pad(x,(0,count*width-len(x)))
                out=np.empty_like(padded);t=time.perf_counter()
                lib.dirac_block_lu(count,width,address(factors),address(pivots),address(padded),address(out))
                self.acceleration['inverse_s']+=time.perf_counter()-t;return out[:self.size]
            self._dense_inverse=inverse
        self._dense_kernel=kernel;self.acceleration['setup_s']=time.perf_counter()-start
    def cauchy(self,x,medium=0):
        x=np.ascontiguousarray(x,dtype=complex).reshape(-1);y=np.empty_like(x)
        check(lib.dirac_cauchy(self.p,medium,address(x),address(y)));return y
    def incident(self,direction,polar):
        d=np.asarray(direction,float);e=np.asarray(polar,float)
        if abs(np.linalg.norm(d)-1)>1e-12 or abs(d@e)>1e-12:raise ValueError('plane wave geometry')
        f=np.zeros(self.size,complex);check(lib.dirac_incident(self.p,address(d),address(e),address(f)))
        return f.ravel()
    def solve_incident(self,incident,tol=1e-8,maxiter=10,initial=None):
        f=np.ascontiguousarray(incident,dtype=complex).reshape(-1);b=np.empty_like(f)
        assert len(f)==self.size
        if not np.all(np.isfinite(f)) or not 0<tol<1:raise ValueError('invalid RHS or tolerance')
        lib.dirac_rhs(self.p,address(f),address(b));history=[]
        bnorm=np.linalg.norm(b)
        if bnorm==0:return np.zeros_like(b),dict(seconds=0.,action_s=0.,calls=0,iterations=0,info=0,residual=0.,history=[])
        start=time.perf_counter();before=self.action_s;calls=self.calls;before_inverse=self.acceleration['inverse_s'];strict_history=[]
        operator=LinearOperator((self.size,self.size),matvec=self.apply,dtype=complex)
        inverse=LinearOperator((self.size,self.size),matvec=self._dense_inverse,dtype=complex) if self._dense_inverse else None
        if self._dense_kernel=='mixed':
            h=np.zeros_like(b) if initial is None else np.array(initial,dtype=complex,copy=True).reshape(-1)
            defect=b.copy() if initial is None else b-self.apply_reference(h)
            residual=float(np.linalg.norm(defect)/bnorm);info=0
            for correction in range(6):
                if residual<=tol:break
                local=[];update,info=gmres(operator,defect,M=inverse,rtol=min(.1,.25*tol/residual),atol=0,restart=60,maxiter=maxiter,callback=local.append,callback_type='pr_norm')
                history.extend(local);h+=update;defect=b-self.apply_reference(h)
                residual=float(np.linalg.norm(defect)/bnorm);strict_history.append(residual)
                if residual<=tol:info=0;break
                if info or not np.isfinite(residual):break
        else:
            h,info=gmres(operator,b,M=inverse,x0=initial,rtol=tol,atol=0,restart=60,maxiter=maxiter,callback=history.append,callback_type='pr_norm')
            residual=np.linalg.norm(self.apply_reference(h)-b)/bnorm;strict_history.append(float(residual))
        stats=dict(seconds=time.perf_counter()-start,action_s=self.action_s-before,calls=self.calls-calls,
            iterations=len(history),info=int(info),residual=float(residual),history=[float(v) for v in history],
            strict_history=strict_history,dense_kernel=self._dense_kernel,block_panels=self.acceleration['block_panels'],
            inverse_s=self.acceleration['inverse_s']-before_inverse)
        if info or not np.isfinite(residual) or residual>tol*1.01:raise RuntimeError('Dirac GMRES failed '+str(stats))
        return h,stats
    def solve(self,direction=(0,0,1),polar=(1,0,0),tol=1e-8):return self.solve_incident(self.incident(direction,polar),tol)
    def quadrature(self):
        points=np.empty((self.count,13,4));lib.dirac_quadrature(self.p,address(points));return points
    def modal_incident(self,mode):
        from tmatrix_incident import regular_fields
        if self.target_order==1:field=regular_fields(self.xyz,mode,self.ka)
        else:
            q=self.quadrature();f=np.ascontiguousarray(regular_fields(q[:,:,:3].reshape(-1,3),mode,self.ka))
            result=np.empty(self.size,complex);lib.dirac_project_fields(self.p,address(f),address(result));return result
        result=np.zeros((self.count,8),complex)
        result[:,1:4]=np.einsum('nij,nj->ni',self.frame,field[:,3:])
        result[:,5:8]=np.einsum('nij,nj->ni',self.frame,field[:,:3]);return result.ravel()
    def field(self,h,targets,medium=0,far=False,incident=None,project=False):
        h=np.ascontiguousarray(h,dtype=complex);x=np.ascontiguousarray(targets,dtype=float).reshape(-1,3)
        if incident is not None:
            if medium!=0:raise ValueError('incident subtraction is for exterior regular incident fields')
            # The exterior Cauchy representation of an entire incident Maxwell
            # field is exactly zero. Subtract its density analytically; this
            # avoids a large discretization leakage when scattering is weak.
            h=np.ascontiguousarray(h.reshape(-1,8)-np.asarray(incident).reshape(-1,8)/self.scaling[3]).ravel()
        if project:
            # Eq. (34): Calderon-project the layer density. This is an
            # optional independent discretization check, not a new solve.
            scale=self.scaling[1 if medium else 3]
            q=np.ascontiguousarray(h.reshape(-1,8)*scale).ravel()
            projected=.5*(q+(1 if medium else -1)*self.cauchy(q,medium))
            h=np.ascontiguousarray(projected.reshape(-1,8)/scale).ravel()
        y=np.empty((len(x),8),complex)
        check(lib.dirac_field(self.p,medium,address(h),len(x),address(x),address(y),int(far)));return y

sigma=np.array([np.eye(2),[[1,0],[0,-1]],[[0,1],[1,0]],[[0,-1j],[1j,0]]],complex)
def mueller(fields,theta):
    et=np.c_[np.cos(theta),np.zeros(len(theta)),-np.sin(theta)]
    ep=np.tile([0,1,0],(len(theta),1))
    j=np.stack((np.einsum('pac,ac->ap',fields,et),np.einsum('pac,ac->ap',fields,ep)),axis=1)
    return .5*np.einsum('iab,nbc,jcd,nad->nij',sigma,j,sigma,j.conj()).real

def main():
    p=argparse.ArgumentParser();p.add_argument('--shape',choices=['sphere','prism'],default='sphere')
    p.add_argument('--ref',type=int,default=2);p.add_argument('--ka',type=float,default=1.)
    p.add_argument('--index',type=complex,default=1.3116);p.add_argument('--aspect',type=float,default=1.1547005383792517)
    p.add_argument('--order',type=int,default=14);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--target-order',type=int,default=1)
    p.add_argument('--degree',type=int,default=0)
    args=p.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    d=Dirac(args.shape,args.ref,args.ka,args.index,args.aspect,args.order,args.target_order,degree=args.degree)
    report=dict(shape=args.shape,ref=args.ref,ka=args.ka,index=str(args.index),panels=d.count,dofs=d.size,
        setup_s=d.setup_s,kernel_bytes=lib.dirac_bytes(d.p),operator='Helsing/Karlsson/Rosen eight-density Dirac',degree=args.degree,target_order=args.target_order,solves=[])
    report['exterior_regular_incident_subtracted']=True
    print('setup',report,flush=True)
    incident=d.incident((0,0,1),(1,0,0))
    report['plane_Cauchy_trace_error']=float(np.linalg.norm(d.cauchy(incident)-incident)/np.linalg.norm(incident))
    theta=np.linspace(0,np.pi,181);directions=np.c_[np.sin(theta),np.zeros(181),np.cos(theta)]
    fields=[];densities=[];scalar=[];longitudinal=[]
    for polar in [(1,0,0),(0,1,0)]:
        h,stats=d.solve(polar=polar);report['solves'].append(stats);densities.append(h)
        far=d.field(h,directions,far=True,incident=d.incident((0,0,1),polar));fields.append(far[:,5:8]);scalar.append(float(np.linalg.norm(far[:,[0,4]])/max(np.linalg.norm(far[:,5:8]),1e-300)))
        longitudinal.append(float(np.linalg.norm(np.einsum('ij,ij->i',far[:,5:8],directions))/max(np.linalg.norm(far[:,5:8]),1e-300)))
        print('solve',polar,{k:v for k,v in stats.items() if k!='history'},flush=True)
    fields=np.asarray(fields);mu=mueller(fields,theta)*args.ka**2
    report['far_scalar_relative_norm']=scalar;report['far_longitudinal_relative_norm']=longitudinal
    if args.shape=='sphere' and abs(args.index-1)>1e-12:
        ref=np.moveaxis(np.asarray(mie_mueller(np.degrees(theta),args.index,args.ka)),-1,0)
        report['mie_all16_max_over_M11']=float(np.max(abs(mu-ref)/ref[:,0,0,None,None]))
        report['mie_M11_relative_L2']=float(np.linalg.norm(mu[:,0,0]-ref[:,0,0])/np.linalg.norm(ref[:,0,0]))
        report['mie_forward_ratio']=float(mu[0,0,0]/ref[0,0,0]);report['mie_backward_ratio']=float(mu[-1,0,0]/ref[-1,0,0])
    np.savez(args.out.with_suffix('.npz'),theta=theta,mueller=mu,fields=fields,densities=densities,geometry=d.geometry)
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='solves'},indent=2),flush=True)

if __name__=='__main__':main()
