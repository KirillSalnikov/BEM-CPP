"""Measured panel/basis transformations for the eight-density Dirac system."""
import ctypes as ct
import numpy as np
from scipy.spatial import cKDTree
from solve import lib, address


def polynomial(bary):
    xi,eta=bary[...,1],bary[...,2];s=xi+eta;d=xi-eta
    return np.stack([np.ones_like(s),np.sqrt(6)*d,np.sqrt(2)*(2-3*s),
        np.sqrt(3)*(10*s*s-12*s+3),3*d*(5*s-4),np.sqrt(15)*(xi*xi-4*xi*eta+eta*eta),
        2*(35*s**3-60*s*s+30*s-4),2*np.sqrt(3)*d*(21*s*s-30*s+10),
        np.sqrt(5)*(3*d*d-s*s)*(7*s-6),np.sqrt(7)*d*(5*d*d-3*s*s)],axis=-1)


class SurfaceTransform:
    def __init__(self,d,matrix):
        self.d=d;matrix=np.asarray(matrix,float);self.matrix=matrix
        if matrix.shape!=(3,3) or np.linalg.norm(matrix.T@matrix-np.eye(3))>1e-12:raise ValueError('orthogonal spatial transform required')
        parity=float(round(np.linalg.det(matrix)))
        lib.dirac_vertices.argtypes=[ct.c_void_p,ct.c_void_p]
        vertices=np.empty((d.count,3,3));lib.dirac_vertices(d.p,address(vertices))
        scale=max(np.linalg.norm(vertices,axis=-1).max(),1.)
        distances,image=cKDTree(d.xyz).query(d.xyz@matrix.T)
        if distances.max()>2e-11*scale or len(set(image))!=d.count:raise ValueError('surface panels do not possess requested symmetry')
        self.image=image;self.geometry_error=float(distances.max()/scale)
        moved=vertices@matrix.T
        delta=np.linalg.norm(moved[:,:,None,:]-vertices[image][:,None,:,:],axis=-1)
        permutations=np.argmin(delta,axis=2)
        if np.max(np.min(delta,axis=2))>2e-11*scale or any(len(set(v))!=3 for v in permutations):raise ValueError('surface triangles do not map under symmetry')
        degree={1:0,3:1,6:2,10:3}[d.basis_count]
        grid=np.array([[1/3,1/3,1/3]]) if not degree else np.array([[i/degree,j/degree,1-(i+j)/degree] for i in range(degree+1) for j in range(degree+1-i)])
        v=polynomial(grid)[:,:d.basis_count];cache={};basis=[]
        for perm in permutations:
            key=tuple(perm)
            if key not in cache:cache[key]=np.linalg.solve(v,polynomial(grid[:,perm])[:,:d.basis_count])
            basis.append(cache[key])
        self.basis=np.array(basis)
        frames=np.einsum('pij,jk,plk->pil',d.frame[image],matrix,d.frame)
        self.components=np.zeros((d.count,8,8));self.components[:,0,0]=1;self.components[:,4,4]=parity
        self.components[:,1:4,1:4]=parity*frames;self.components[:,5:8,5:8]=frames
        self.orthogonality_error=float(np.max(abs(np.einsum('pji,pjk->pik',self.basis,self.basis)-np.eye(d.basis_count))))
        if self.orthogonality_error>2e-11:raise ValueError('polynomial symmetry transform is not orthogonal')

    def apply(self,x):
        x=np.asarray(x,complex);shape=x.shape;x=x.reshape(*shape[:-1],self.d.count,self.d.basis_count,8)
        out=np.empty_like(x)
        out[...,self.image,:,:]=np.einsum('pab,...pbc,pdc->...pad',self.basis,x,self.components,optimize=True)
        return out.reshape(shape)

    def inverse(self,x):
        x=np.asarray(x,complex);shape=x.shape;x=x.reshape(*shape[:-1],self.d.count,self.d.basis_count,8)
        out=np.einsum('pba,...pbc,pcd->...pad',self.basis,x[...,self.image,:,:],self.components,optimize=True)
        return out.reshape(shape)


class CyclicGeometry:
    def __init__(self,d,order=6,horizontal=False):
        if not isinstance(order,int) or order<2:raise ValueError('cyclic order must be an integer >= 2')
        self.d=d;self.order=order;self.horizontal=bool(horizontal);self.transforms=[]
        self.group_size=order*(2 if horizontal else 1)
        for g in range(self.group_size):
            a=2*np.pi*(g%order)/order;c,s=np.cos(a),np.sin(a)
            self.transforms.append(SurfaceTransform(d,[[c,-s,0],[s,c,0],[0,0,-1 if g>=order else 1]]))
        seen=set();representatives=[]
        for j in range(d.count):
            if j in seen:continue
            orbit={int(t.image[j]) for t in self.transforms}
            if len(orbit)!=self.group_size or orbit & seen:raise ValueError('symmetry panels must have disjoint free orbits')
            representatives.append(j);seen.update(orbit)
        self.representatives=np.asarray(representatives);self.rows=(self.representatives[:,None]*d.basis_count+np.arange(d.basis_count)).ravel().astype(np.int32)
        self.size=len(self.rows)*8

    def restrict(self,x):return np.asarray(x).reshape(-1,self.d.count,self.d.basis_count,8)[:,self.representatives].reshape(*np.asarray(x).shape[:-1],self.size)

    def expand(self,x,m,horizontal_parity=None):
        if self.horizontal and horizontal_parity not in (-1,1):raise ValueError('horizontal parity must be explicitly specified')
        x=np.asarray(x).reshape(len(self.representatives),self.d.basis_count,8)
        out=np.empty((self.d.count,self.d.basis_count,8),complex)
        for g,t in enumerate(self.transforms):
            reps=self.representatives
            phase=np.exp(2j*np.pi*(m%self.order)*(g%self.order)/self.order)
            if g>=self.order:phase*=horizontal_parity
            out[t.image[reps]]=phase*np.einsum('pab,pbc,pdc->pad',t.basis[reps],x,t.components[reps],optimize=True)
        return out.ravel()

    def expand_mode(self,x,mode):
        p,l,m=mode
        return self.expand(x,m,(-1)**(p+l+m+1))

    def full_action(self,x,reference=False,tile=256):
        """Full-space action for arbitrary densities, without assuming a mode.

        The scalar reference traverses representative kernel rows independently
        of the SIMD iteration and modal lifting. Direct quadratures audit the
        geometric row reconstruction when the compact cache is created.
        """
        import time
        from solve import check
        x=np.ascontiguousarray(x,dtype=complex);shape=x.shape
        if x.ndim not in (1,2) or x.shape[-1]!=self.d.size:raise ValueError('full reconstructed action dimensions')
        x=x.reshape(-1,self.d.size)
        if not 1<=len(x)<=16:raise ValueError('full reconstructed batch dimensions')
        out=np.empty((len(x),self.d.count,self.d.basis_count,8),complex);reps=self.representatives;t0=time.perf_counter()
        for transform in self.transforms:
            pulled=np.ascontiguousarray(transform.inverse(x));part=np.empty((len(x),self.size),complex)
            if reference:
                if len(x)>1 and self.d.reference_checks=='batch' and hasattr(lib,'dirac_apply_rows_reference_batch'):
                    check(lib.dirac_apply_rows_reference_batch(self.d.p,len(x),address(pulled),len(self.rows),address(self.rows),address(part)))
                else:
                    for j in range(len(x)):check(lib.dirac_apply_rows_reference(self.d.p,address(pulled[j]),len(self.rows),address(self.rows),address(part[j])))
            else:check(lib.dirac_apply_rows(self.d.p,len(x),address(pulled),len(self.rows),address(self.rows),address(part),tile))
            part=part.reshape(len(x),len(reps),self.d.basis_count,8)
            out[:,transform.image[reps]]=np.einsum('pab,rpbc,pdc->rpad',transform.basis[reps],part,transform.components[reps],optimize=True)
        self.d.action_s+=time.perf_counter()-t0;self.d.calls+=len(x)
        return out.reshape(shape)

    def self_blocks(self):
        from solve import check
        width=self.d.basis_count*8;reps=self.representatives
        ids=np.ascontiguousarray(reps,dtype=np.int32);blocks=np.empty((len(ids),width,width),complex)
        if getattr(self.d,'_matrix_free',False):self.d.self_blocks_selected(ids,blocks)
        else:check(lib.dirac_self_blocks_selected(self.d.p,len(ids),address(ids),address(blocks)))
        result=np.empty((self.d.count,width,width),complex)
        for t in self.transforms:
            change=np.einsum('pab,pcd->pacbd',t.basis[reps],t.components[reps]).reshape(len(reps),width,width)
            result[t.image[reps]]=change@blocks@change.transpose(0,2,1)
        return result

    def audit_direct(self,seed=371):
        """Independent quadratures on representatives and unretained rows."""
        import time
        from solve import check
        start=time.perf_counter();rng=np.random.default_rng(seed)
        x=np.ascontiguousarray(rng.normal(size=self.d.size)+1j*rng.normal(size=self.d.size))
        chosen=np.array(sorted({int(t.image[p]) for t in self.transforms for p in self.representatives[[0,-1]]}),np.int32)
        direct=np.empty((len(chosen),self.d.basis_count,8),complex)
        check(lib.dirac_apply_direct_panels(self.d.p,address(x),len(chosen),address(chosen),address(direct)))
        predicted=(self.d.apply_reference(x) if getattr(self.d,'_matrix_free',False) else self.full_action(x,reference=True)).reshape(self.d.count,self.d.basis_count,8)[chosen]
        error=float(np.linalg.norm(predicted-direct)/np.linalg.norm(direct))
        if error>2e-11:raise RuntimeError(f'compact cache fails independent direct quadrature audit: {error}')
        return dict(relative_error=error,panels=chosen.tolist(),seconds=time.perf_counter()-start)

    def audit_operator(self,seed=513):
        rng=np.random.default_rng(seed);x=rng.normal(size=self.d.size)+1j*rng.normal(size=self.d.size)
        ax=self.d.apply_reference(x);norm=np.linalg.norm(ax)
        generators=[1,self.order] if self.horizontal else [1]
        self.operator_errors=[float(np.linalg.norm(self.d.apply_reference(self.transforms[g].apply(x))-self.transforms[g].apply(ax))/norm) for g in generators]
        self.operator_error=max(self.operator_errors)
        if self.operator_error>2e-11:raise ValueError(f'cyclic discretized operator fails covariance: {self.operator_error}')
        return self.operator_error


def solve_modal_batch(parent,geometry,incidents,modal,tol=1e-8,tile=256,preconditioner=None):
    """Independent cyclic sectors, audited against the full original system."""
    import time
    from batch_gmres import solve_incidents
    from solve import check
    if (not getattr(parent,'_matrix_free',False) and (parent._dense_kernel!='simd' or parent._correction_only)) or parent.acceleration['block_panels']!=1:
        raise ValueError('cyclic solves require full FP64 SIMD and one-panel LU')
    if not hasattr(geometry,'operator_error'):geometry.audit_operator()
    f=np.asarray(incidents,complex)
    if f.shape!=(len(modal),parent.size):raise ValueError('modal batch dimensions')
    reduced=geometry.restrict(f)
    lift_errors=[]
    for j,mode in enumerate(modal):
        err=float(np.linalg.norm(geometry.expand_mode(reduced[j],mode)-f[j])/max(np.linalg.norm(f[j]),1e-300))
        if err>2e-11:raise ValueError(f'incident is outside the claimed cyclic sector: {err}')
        lift_errors.append(err)
    coarse_setup_s=0.
    if preconditioner is not None:
        t=time.perf_counter();preconditioner.prepare_many(modal);coarse_setup_s=time.perf_counter()-t
    class Sector:
        def __init__(self):
            self.size=geometry.size;self._dense_kernel=parent._dense_kernel;self._correction_only=False;self._matrix_free=getattr(parent,'_matrix_free',False)
            self.acceleration=dict(parent.acceleration);self.acceleration['inverse_s']=0.
            factors,pivots=parent._dense_factors
            self.lu=np.ascontiguousarray(factors[geometry.representatives]);self.piv=np.ascontiguousarray(pivots[geometry.representatives])
            self.cache={};self._dense_inverse=self.inverse
        def _batch_rhs(self,f):return (f.reshape(len(f),-1,8)*(2*parent.scaling[2])).reshape(f.shape)
        def inverse(self,x):
            x=np.ascontiguousarray(x);out=np.empty_like(x);t=time.perf_counter()
            lib.dirac_block_lu(len(geometry.representatives),8*parent.basis_count,address(self.lu),address(self.piv),address(x),address(out))
            self.acceleration['inverse_s']+=time.perf_counter()-t;return out
        def _batch_inverse(self,x,j):
            return self.inverse(x) if preconditioner is None else preconditioner.apply(x,modal[j])
        def _batch_inverses(self,x,indices):
            if preconditioner is not None and hasattr(preconditioner,'apply_batch'):
                return preconditioner.apply_batch(x,[modal[j] for j in indices])
            return np.asarray([self._batch_inverse(v,j) for v,j in zip(x,indices)])
        def _batch_apply(self,x,indices,tile):
            full=np.ascontiguousarray([geometry.expand_mode(v,modal[j]) for v,j in zip(x,indices)])
            if getattr(parent,'_matrix_free',False):return parent.apply_sector_batch(full)
            out=np.empty((len(x),self.size),complex)
            check(lib.dirac_apply_rows(parent.p,len(x),address(full),len(geometry.rows),address(geometry.rows),address(out),tile))
            return out
        def _batch_reference(self,x,j):
            self.cache[j]=parent.apply_reference(geometry.expand_mode(x,modal[j]));return geometry.restrict(self.cache[j])
        def _batch_references(self,x,indices):
            full=np.ascontiguousarray([geometry.expand_mode(v,modal[j]) for v,j in zip(x,indices)])
            checked=parent.apply_reference_batch(full)
            for j,value in zip(indices,checked):self.cache[j]=value
            return geometry.restrict(checked)
    lib.dirac_apply_rows.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int]
    op=Sector();start=time.perf_counter();x,stats=solve_incidents(op,reduced,tol=tol,tile=tile)
    result=np.array([geometry.expand_mode(v,mode) for v,mode in zip(x,modal)])
    for j,s in enumerate(stats):
        rhs=(f[j].reshape(-1,8)*(2*parent.scaling[2])).ravel();norm=np.linalg.norm(rhs)
        residual=float(np.linalg.norm(op.cache[j]-rhs)/norm) if norm else 0.
        s['reduced_residual']=s['residual'];s['residual']=residual;s['strict_history'].append(residual)
        s.update(cyclic_order=geometry.order,reduced_dofs=op.size,full_dofs=parent.size,modal_m=modal[j][2],lift_error=lift_errors[j],operator_covariance=geometry.operator_error)
        s.update(horizontal_parity=(-1)**sum([modal[j][0],modal[j][1],modal[j][2],1]) if geometry.horizontal else None,group_size=geometry.group_size)
        s['residual_scope']=('full-space FMM at strict epsilon plus exact near corrections; direct quadrature audited' if getattr(parent,'_matrix_free',False) else
            'independent full-space FP64; orbit cache audited against direct quadratures' if parent._compact_geometry is not None else 'full original stored FP64 operator')
        s.update(sector_coarse_basis=preconditioner.coarse_basis if preconditioner is not None else 0,
                 coarse_setup_s=coarse_setup_s/len(stats))
        if not np.isfinite(residual) or residual>tol*1.01:raise RuntimeError('full FP64 cyclic residual failed '+str(s))
    wall=time.perf_counter()-start
    for s in stats:s.update(seconds=wall/len(stats),batch_wall_s=wall)
    return result,stats
