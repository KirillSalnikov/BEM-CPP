"""Build regular M/N T columns with the dielectric Dirac solver; exact SO(3)."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,sys,time,hashlib,shutil
from pathlib import Path
import numpy as np
from solve import Dirac,R,B,LIBRARY
sys.path.insert(0,str(R/'scripts'))
from tmatrix_reference import modes,tangent_frame,transverse_harmonics,analytic_mueller
from tmatrix_projection import SphericalProjector
from verify_mie import mie_coefficients,mie_mueller

def main():
    p=argparse.ArgumentParser();p.add_argument('--shape',choices=['sphere','prism'],default='sphere')
    p.add_argument('--ref',type=int,default=3);p.add_argument('--ka',type=float,default=1)
    p.add_argument('--index',type=complex,default=1.3116);p.add_argument('--L',type=int,default=3)
    p.add_argument('--aspect',type=float,default=1.1547005383792517,help='prism height / width across flats; SciIce Column: 1.6495721976846451')
    p.add_argument('--degree',type=int,choices=[0,1,2,3],default=0)
    p.add_argument('--target-order',type=int);p.add_argument('--order',type=int,default=14)
    p.add_argument('--backend',choices=['dense','fmm','fmm-sector'],default='dense');p.add_argument('--fmm-eps',type=float)
    p.add_argument('--fmm-strict-eps',type=float,default=1e-12)
    p.add_argument('--fmm-packet',type=int,choices=[1,2,4],default=1)
    p.add_argument('--near-subdiv',type=int,choices=[0,1,2],default=0)
    p.add_argument('--solver',choices=['standard','refined'],default='standard')
    p.add_argument('--tol',type=float,default=1e-8)
    p.add_argument('--dense-kernel',choices=['scalar','simd','mixed'],default='scalar')
    p.add_argument('--dense-cache',choices=['full','orbits12'],default='full')
    p.add_argument('--reference-checks',choices=['scalar','batch'],default='batch')
    p.add_argument('--block-panels',type=int,choices=range(9),default=0)
    p.add_argument('--batch-size',type=int,choices=range(1,17),default=1)
    p.add_argument('--batch-tile',type=int,default=256)
    p.add_argument('--symmetry',choices=['none','c6','c6v','c6hv'],default='none')
    p.add_argument('--sector-coarse-basis',type=int,choices=[0,1,3],default=0)
    p.add_argument('--coarse-budget-gib',type=float,default=8)
    p.add_argument('--coarse-cache-policy',choices=['sector','all'],default='sector')
    p.add_argument('--coarse-factorization',choices=['dense','sparse','hierarchical'],default='dense')
    p.add_argument('--coarse-near-radius',type=float,default=1.5)
    p.add_argument('--coarse-ilu-drop',type=float,default=1e-3)
    p.add_argument('--coarse-ilu-fill',type=float,default=4)
    p.add_argument('--coarse-aggregate-groups',type=int,default=128)
    p.add_argument('--coarse-aggregate-basis',type=int,choices=[1,3],default=3)
    p.add_argument('--coarse-global-scope',choices=['coarse','full'],default='full')
    p.add_argument('--so3',choices=['reference','factored','mstm'],default='reference')
    p.add_argument('--no-average',action='store_true',help='Save the complete T; skip SO(3) contraction')
    p.add_argument('--ntheta',type=int,default=181,help='Scattering angles in the analytic average')
    p.add_argument('--so3-threads',type=int,default=4)
    p.add_argument('--mixed-cache-gib',type=float,default=4)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-columns',type=int,default=0)
    p.add_argument('--matrix-budget-gib',type=float,default=4);p.add_argument('--kernel-budget-gib',type=float,default=8)
    p.add_argument('--ram-budget-gib',type=float,default=48,help='dense-driver peak planning budget, including workspace')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.fmm_eps is None:a.fmm_eps=1e-10 if a.backend=='fmm-sector' else 1e-8
    if not 1<=a.L<=128:raise ValueError('L must be in 1..128; convergence must be checked separately')
    if not 2<=a.ntheta<=100001:raise ValueError('ntheta must be in 2..100001')
    if a.max_columns<0:raise ValueError('max-columns must be nonnegative')
    if not 0<a.tol<1:raise ValueError('solver tolerance must be in (0,1)')
    if not np.isfinite(a.aspect) or a.aspect<=0:raise ValueError('aspect must be positive')
    if not 1<=a.so3_threads<=64:raise ValueError('SO3 threads must be in 1..64')
    if a.solver=='refined' and a.backend!='fmm':raise ValueError('refined solver requires FMM')
    if a.backend=='fmm' and (a.dense_kernel!='scalar' or a.block_panels):raise ValueError('legacy FMM does not support dense acceleration')
    if a.backend=='fmm-sector' and (a.shape!='prism' or a.symmetry not in ('c6v','c6hv') or a.block_panels!=1 or a.dense_kernel!='scalar'):
        raise ValueError('sector FMM requires prism, c6v/c6hv, --block-panels 1 and no dense kernel flag')
    if a.backend=='fmm-sector' and not (1e-13<=a.fmm_strict_eps<=a.fmm_eps<=min(1e-7,a.tol) and a.fmm_strict_eps<=a.tol/100):
        raise ValueError('sector FMM requires strict eps <= tol/100 and eps <= min(1e-7,tol); every column still passes a strict full-space audit')
    if not 16<=a.batch_tile<=16384:raise ValueError('batch tile must be in 16..16384')
    if a.batch_size>1 and a.backend!='fmm-sector' and (a.backend!='dense' or a.dense_kernel!='simd'):raise ValueError('batch solves require dense FP64 SIMD or sector FMM')
    if a.symmetry!='none' and a.backend!='fmm-sector' and (a.backend!='dense' or a.dense_kernel!='simd' or a.block_panels!=1):raise ValueError('cyclic symmetry requires dense SIMD or sector FMM and one-panel LU')
    if a.dense_cache!='full' and (a.backend!='dense' or a.symmetry!='c6hv'):raise ValueError('orbit cache requires dense backend and c6hv symmetry')
    if a.sector_coarse_basis and a.symmetry=='none':raise ValueError('coarse correction requires cyclic sectors')
    if not np.isfinite(a.coarse_budget_gib) or a.coarse_budget_gib<=0:raise ValueError('coarse cache budget must be positive')
    if a.sector_coarse_basis>[1,3,6,10][a.degree]:raise ValueError('coarse degree exceeds surface degree')
    if a.coarse_factorization!='dense':
        if a.backend!='fmm-sector' or not a.sector_coarse_basis:raise ValueError('sparse/hierarchical coarse requires fmm-sector and a coarse basis')
        if not 0<a.coarse_near_radius<=8 or not 0<a.coarse_ilu_drop<1 or not 1<=a.coarse_ilu_fill<=20 or not 2<=a.coarse_aggregate_groups<=512:raise ValueError('invalid sparse coarse controls')
        if a.coarse_factorization=='hierarchical' and a.coarse_aggregate_basis>a.sector_coarse_basis:raise ValueError('aggregate basis exceeds coarse basis')
    if not np.isfinite(a.mixed_cache_gib) or a.mixed_cache_gib<0:raise ValueError('additional mixed-cache budget must be nonnegative')
    count=2*a.L*(a.L+2)
    if shutil.disk_usage(a.out).free<6*16*count*count+(1<<30):
        raise MemoryError('Insufficient disk for mapped T/kernel, checkpoints and atomic output')
    if not np.isfinite(a.ram_budget_gib) or a.ram_budget_gib<=0:raise ValueError('RAM budget must be positive')
    if 2*count*count*16>a.matrix_budget_gib*(1<<30):raise ValueError('T/kernel matrices exceed the explicit memory budget')
    start=time.perf_counter()
    if a.target_order is None:a.target_order=13 if a.backend!='dense' else 7
    memory_plan={}
    if a.backend=='dense':
        from memory_plan import dense_memory_plan
        memory_plan=dense_memory_plan(a)
    elif a.backend=='fmm-sector':
        from memory_plan import sector_fmm_memory_plan
        memory_plan=sector_fmm_memory_plan(a)
    signatures={str(path.relative_to(R)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [R/'src/dirac_transmission.cpp',B/'solve.py',Path(__file__),R/'scripts/tmatrix_reference.py',R/'scripts/tmatrix_incident.py',R/'scripts/tmatrix_projection.py']}
    signatures['selected_Dirac_library']=hashlib.sha256(LIBRARY.read_bytes()).hexdigest()
    signatures['matrix_storage.py']=hashlib.sha256((B/'matrix_storage.py').read_bytes()).hexdigest()
    signatures['batch_checks.py']=hashlib.sha256((B/'batch_checks.py').read_bytes()).hexdigest()
    if a.batch_size>1:signatures['batch_gmres.py']=hashlib.sha256((B/'batch_gmres.py').read_bytes()).hexdigest()
    if a.symmetry!='none':
        for name in ['dirac_symmetry.py','batch_gmres.py']:signatures[name]=hashlib.sha256((B/name).read_bytes()).hexdigest()
        signatures['tmatrix_symmetry.py']=hashlib.sha256((R/'scripts/tmatrix_symmetry.py').read_bytes()).hexdigest()
    if a.so3=='factored':signatures['so3_factored.py']=hashlib.sha256((B/'so3_factored.py').read_bytes()).hexdigest()
    if a.so3=='mstm' and not a.no_average:
        for path in [R/'scripts/tmatrix_so3_mstm.py',Path(os.environ.get('BEM_SO3_BINARY_FP64',str(R/'bin/tmatrix_so3_mstm')))]:
            signatures[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    if a.sector_coarse_basis:signatures['sector_coarse.py']=hashlib.sha256((B/'sector_coarse.py').read_bytes()).hexdigest()
    if a.backend in ('fmm','fmm-sector'):
        for name in ['fmm.py','native.py','FMM3D/lib/libfmm3d.so']:
            path=B.parent/'clifford_fmm_20261115'/name;signatures[str(path.relative_to(R))]=hashlib.sha256(path.read_bytes()).hexdigest()
    if a.backend=='fmm-sector':
        for name in ['sector_fmm.py','sector_coarse_fmm.py']:signatures[name]=hashlib.sha256((B/name).read_bytes()).hexdigest()
        if a.coarse_factorization!='dense':
            for name in ['sector_coarse_sparse.py','sector_coarse_hierarchy.py']:signatures[name]=hashlib.sha256((B/name).read_bytes()).hexdigest()
    config=dict(shape=a.shape,ref=a.ref,ka=a.ka,index=[complex(a.index).real,complex(a.index).imag],L=a.L,degree=a.degree,target_order=a.target_order,near_order=a.order,near_subdiv=a.near_subdiv,backend=a.backend,fmm_eps=a.fmm_eps,solver=a.solver,tolerance=a.tol,sha256=signatures)
    config.update(dense_kernel=a.dense_kernel,block_panels=a.block_panels,mixed_cache_gib=a.mixed_cache_gib,batch_size=a.batch_size,batch_tile=a.batch_tile)
    config.update(symmetry=a.symmetry,so3=a.so3,dense_cache=a.dense_cache)
    config.update(sector_coarse_basis=a.sector_coarse_basis)
    config.update(aspect=a.aspect)
    config.update(reference_checks=a.reference_checks)
    if a.backend=='fmm-sector':config.update(fmm_strict_eps=a.fmm_strict_eps,fmm_packet=a.fmm_packet)
    if a.coarse_factorization!='dense':config.update(coarse_factorization=a.coarse_factorization,
        coarse_near_radius=a.coarse_near_radius,coarse_ilu_drop=a.coarse_ilu_drop,coarse_ilu_fill=a.coarse_ilu_fill,coarse_aggregate_groups=a.coarse_aggregate_groups,coarse_aggregate_basis=a.coarse_aggregate_basis,coarse_global_scope=a.coarse_global_scope)
    config_path=a.out/'configuration.json';columns=a.out/'columns'
    if config_path.exists():
        if not a.resume:raise ValueError('output already initialized; use --resume to validate and continue')
        if json.loads(config_path.read_text())!=config:raise ValueError('checkpoint numerical parameters or code changed')
    else:
        if a.resume:raise ValueError('no checkpoint configuration to resume')
        config_path.write_text(json.dumps(config,indent=2)+'\n')
    columns.mkdir(exist_ok=True)
    (a.out/'memory_plan.json').write_text(json.dumps(memory_plan,indent=2)+'\n')
    if a.backend=='fmm':
        sys.path.insert(0,str(B.parent/'clifford_fmm_20261115'));from fmm import DiracFMM
        d=DiracFMM(a.shape,a.ref,a.ka,a.index,aspect=a.aspect,target_order=a.target_order,order=a.order,degree=a.degree,eps=a.fmm_eps,max_bytes=int(a.kernel_budget_gib*(1<<30)),near_subdiv=a.near_subdiv)
    elif a.backend=='fmm-sector':
        from sector_fmm import SectorFMM
        d=SectorFMM(a.shape,a.ref,a.ka,a.index,aspect=a.aspect,target_order=a.target_order,order=a.order,degree=a.degree,eps=a.fmm_eps,strict_eps=a.fmm_strict_eps,fmm_packet=a.fmm_packet,max_bytes=int(a.kernel_budget_gib*(1<<30)),near_subdiv=a.near_subdiv,horizontal=a.symmetry=='c6hv')
    else:d=Dirac(a.shape,a.ref,a.ka,a.index,aspect=a.aspect,target_order=a.target_order,order=a.order,degree=a.degree,max_bytes=int(a.kernel_budget_gib*(1<<30)),near_subdiv=a.near_subdiv,dense_kernel=a.dense_kernel,block_panels=a.block_panels,mixed_cache_bytes=int(a.mixed_cache_gib*(1<<30)),dense_cache=a.dense_cache)
    d.reference_checks=a.reference_checks
    solve=lambda f:(d.solve_refined(f,tol=a.tol) if a.solver=='refined' else d.solve_incident(f,tol=a.tol))
    cyclic=None;mirror=None;symmetry_info={};symmetry_start=time.perf_counter()
    if a.symmetry!='none':
        from dirac_symmetry import CyclicGeometry,SurfaceTransform,solve_modal_batch
        cyclic=d._compact_geometry or CyclicGeometry(d,horizontal=a.symmetry=='c6hv');cyclic.audit_operator()
        symmetry_info.update(cyclic_order=6,group_size=cyclic.group_size,reduced_dofs=cyclic.size,operator_covariance=cyclic.operator_error,horizontal=a.symmetry=='c6hv')
    if a.symmetry in ('c6v','c6hv'):
        from tmatrix_symmetry import vertical_mirror_plan
        mirror=SurfaceTransform(d,np.diag([1.,-1.,1.]));flip,mirror_signs,_,_=vertical_mirror_plan(a.L)
        rng=np.random.default_rng(543);x=rng.normal(size=d.size)+1j*rng.normal(size=d.size);ax=d.apply_reference(x)
        err=float(np.linalg.norm(d.apply_reference(mirror.apply(x))-mirror.apply(ax))/np.linalg.norm(ax))
        if err>2e-11:raise ValueError('mirror operator covariance failed')
        symmetry_info['mirror_covariance']=err
    symmetry_info['setup_audit_s']=time.perf_counter()-symmetry_start
    coarse=None
    if a.sector_coarse_basis:
        from sector_coarse import CoarseCorrection
        if a.backend=='fmm-sector':
            from sector_coarse_fmm import MatrixFreeCoarse
            options=dict(radius=a.coarse_near_radius,drop_tol=a.coarse_ilu_drop,fill_factor=a.coarse_ilu_fill)
            if a.coarse_factorization=='hierarchical':
                from sector_coarse_hierarchy import HierarchicalCoarse
                coarse=HierarchicalCoarse(cyclic,a.sector_coarse_basis,int(a.coarse_budget_gib*(1<<30)),aggregate_groups=a.coarse_aggregate_groups,aggregate_basis=a.coarse_aggregate_basis,outer=a.coarse_global_scope=='full',**options)
            elif a.coarse_factorization=='sparse':
                from sector_coarse_sparse import SparseCoarse
                coarse=SparseCoarse(cyclic,a.sector_coarse_basis,int(a.coarse_budget_gib*(1<<30)),**options)
            else:coarse=MatrixFreeCoarse(cyclic,a.sector_coarse_basis,int(a.coarse_budget_gib*(1<<30)))
        else:coarse=CoarseCorrection(cyclic,a.sector_coarse_basis,int(a.coarse_budget_gib*(1<<30)))
        states=1 if a.coarse_cache_policy=='sector' else len({coarse.key(mode) for mode in modes(a.L)})
        memory=coarse.memory_plan()
        required=states*memory['persistent_bytes']+memory['workspace_bytes']+memory['panel_factors_bytes']
        if required>coarse.budget_bytes:
            raise MemoryError(f'coarse cache and workspace for {states} resident sectors need {required/(1<<30):.3f} GiB; increase --coarse-budget-gib')
    modal=modes(a.L);band=2*a.L+4;z,w=np.polynomial.legendre.leggauss(band)
    zz,phi=np.meshgrid(z,np.arange(2*band)*np.pi/band,indexing='ij')
    directions=np.stack((np.sqrt(1-zz*zz)*np.cos(phi),np.sqrt(1-zz*zz)*np.sin(phi),zz),axis=-1).reshape(-1,3)
    _,_,et,ep=tangent_frame(directions);projector=SphericalProjector(a.L,z,w)
    from matrix_storage import convert_projected,release_pages,norm_squared
    phase=np.asarray([1j**(l-p) for p,l,m in modal])
    # Columns remain the authoritative checkpoints. This disposable mapping is
    # rebuilt from them on resume, including after an interrupted conversion.
    projected=np.lib.format.open_memmap(a.out/'kernel.npy',mode='w+',dtype=np.complex128,
                                       shape=(len(modal),len(modal)),fortran_order=True)
    statistics=[None]*len(modal);rhs_s=0;far_s=0;new_columns=0;resumed_columns=0;pending_solutions={};pending_mirrors={};created=set()
    saved_indices={int(p.stem) for p in columns.glob('*.npz')}
    column_order=list(range(len(modal)))
    if coarse is not None and a.coarse_cache_policy=='sector':
        # All negative sources precede positive mirror partners. Positive holes
        # on resume are solved directly at the end, grouped by their own sector.
        column_order.sort(key=lambda j:(bool(mirror is not None and modal[j][2]>0),coarse.key(modal[j]),j))
    def save_column(index,v,stats):
        nonlocal new_columns
        if not np.all(np.isfinite(v)):raise RuntimeError('nonfinite projected T column')
        checkpoint=columns/f'{index:06d}.npz';temporary=checkpoint.with_suffix('.tmp')
        with temporary.open('wb') as target:
            v=np.array(v,copy=True);stats_text=json.dumps(stats)
            np.savez(target,projected=v,mode=modal[index],statistics=stats_text,sha256=hashlib.sha256(v.tobytes()).hexdigest(),statistics_sha256=hashlib.sha256(stats_text.encode()).hexdigest())
            target.flush();os.fsync(target.fileno())
        os.replace(temporary,checkpoint);created.add(index);saved_indices.add(index);new_columns+=1
    def progress(status):
        value=dict(status=status,completed_columns=len(saved_indices),total_columns=len(modal),new_columns=new_columns,resumed_columns=resumed_columns,wall_s=time.perf_counter()-start)
        value.update(operator_setup_s=d.setup_s,coarse_setup_s=coarse.setup_s if coarse else 0.,coarse_bytes=coarse.bytes if coarse else 0.,coarse_sectors=len(coarse.cache) if coarse else 0,
                     coarse_peak_bytes=coarse.peak_bytes if coarse else 0,coarse_cache_policy=a.coarse_cache_policy)
        temporary=a.out/'progress.json.tmp';temporary.write_text(json.dumps(value)+'\n');os.replace(temporary,a.out/'progress.json')
        return value
    for position,j in enumerate(column_order):
        mode=modal[j]
        checkpoint=columns/f'{j:06d}.npz'
        if checkpoint.exists():
            with np.load(checkpoint,allow_pickle=False) as saved:
                v=saved['projected'];stats=json.loads(str(saved['statistics']));saved_mode=tuple(saved['mode'])
                if saved_mode!=tuple(mode) or v.shape!=(len(modal),) or not np.all(np.isfinite(v)) or not np.isfinite(stats['residual']) or not 0<=stats['residual']<=1.01*a.tol or stats['info']!=0:raise ValueError('invalid saved T column')
                if hashlib.sha256(v.tobytes()).hexdigest()!=str(saved['sha256']):raise ValueError('saved T column checksum mismatch')
                if hashlib.sha256(str(saved['statistics']).encode()).hexdigest()!=str(saved['statistics_sha256']):raise ValueError('saved T statistics checksum mismatch')
                projected[:,j]=v;statistics[j]=stats;resumed_columns+=int(j not in created)
                rhs_s+=stats.get('rhs_s',0.);far_s+=stats.get('far_projection_s',0.)
            continue
        if a.max_columns and new_columns>=a.max_columns:
            print(json.dumps(progress('checkpoint')));d.close();return
        if j not in pending_solutions:
            indices=[];reserved=0
            for k in column_order[position:]:
                if k in saved_indices:continue
                if coarse is not None and a.coarse_cache_policy=='sector' and coarse.key(modal[k])!=coarse.key(mode):continue
                # Positive partners are written immediately with their source.
                # A positive hole after resume is solved directly if necessary.
                if mirror is not None and modal[k][2]>0 and int(flip[k]) not in saved_indices:continue
                cost=2 if mirror is not None and modal[k][2]<0 and int(flip[k]) not in saved_indices else 1
                if a.max_columns and new_columns+reserved+cost>a.max_columns:break
                indices.append(k);reserved+=cost
                if len(indices)>=a.batch_size:break
            if not indices:
                print(json.dumps(progress('checkpoint')));d.close();return
            if coarse is not None and a.coarse_cache_policy=='sector':coarse.release_except([modal[k] for k in indices])
            incident_batch=[];rhs_times=[]
            for k in indices:
                t=time.perf_counter();incident_batch.append(d.modal_incident(modal[k]));rhs_times.append(time.perf_counter()-t)
            if cyclic is not None:solutions,batch_stats=solve_modal_batch(d,cyclic,incident_batch,[modal[k] for k in indices],tol=a.tol,tile=a.batch_tile,preconditioner=coarse)
            elif a.batch_size>1:solutions,batch_stats=d.solve_incidents(incident_batch,tol=a.tol,tile=a.batch_tile)
            else:
                h,s=solve(incident_batch[0]);solutions=[h];batch_stats=[s]
            if mirror is not None:
                from batch_checks import audit_mirror_batch
                requests=[(k,int(flip[k]),modal[int(flip[k])],mirror_signs[k],f,h)
                    for k,f,h in zip(indices,incident_batch,solutions)
                    if modal[k][2]<0 and int(flip[k]) not in saved_indices]
                pending_mirrors.update(audit_mirror_batch(d,mirror,requests,a.tol))
            for k,f,h,s,rt in zip(indices,incident_batch,solutions,batch_stats,rhs_times):pending_solutions[k]=(f,h,s,rt)
        f,h,stats,rhs_time=pending_solutions.pop(j);rhs_s+=rhs_time;statistics[j]=stats
        t=time.perf_counter();field=d.field(h,directions,far=True,incident=f)[:,5:8]
        transverse=np.stack((np.einsum('ij,ij->i',field,et),np.einsum('ij,ij->i',field,ep)),axis=1).ravel()
        projected[:,j]=projector@transverse;far_time=time.perf_counter()-t;far_s+=far_time
        stats['rhs_s']=rhs_time;stats['far_projection_s']=far_time
        save_column(j,projected[:,j],stats)
        if mirror is not None and mode[2]<0 and int(flip[j]) not in saved_indices:
            partner,ps=pending_mirrors.pop(j)
            pv=mirror_signs[j]*mirror_signs*projected[flip,j]
            save_column(partner,pv,ps)
        progress('running');print(j+1,len(modal),mode,stats['iterations'],stats['residual'],flush=True)
    # Physical Cauchy radiation convention. No fitted sign or normalization.
    T=convert_projected(projected,phase,a.ka,a.out/'tmatrix.npy');kernel=projected
    theta=np.linspace(0,180,a.ntheta)
    # The coarse cache is no longer needed once all columns are written.
    coarse_report=dict(coarse_basis=0);coarse_action_s=0.;stored_fine_W_bytes=0
    if coarse is not None:
        coarse.release_except()
        coarse_plan=coarse.memory_plan();coarse_action_s=getattr(coarse,'action_s',0.)
        stored_fine_W_bytes=coarse_plan.get('stored_fine_W_bytes',0)
        coarse_report=dict(coarse_basis=a.sector_coarse_basis,setup_s=coarse.setup_s,bytes=coarse.bytes,
            peak_cache_bytes=coarse.peak_bytes,evictions=coarse.evictions,cache_policy=a.coarse_cache_policy,
            memory_plan=coarse_plan,sectors=coarse.statistics)
        if hasattr(coarse,'close'):coarse.close()
        coarse=None
    # Drop the last packet's full-space currents, including references in the
    # mirror request list. These are not needed for independent plane checks.
    incident_batch=solutions=requests=f=h=None
    checks=[]
    # Direct independent plane wave, absent from the outgoing projection grid.
    inc=np.array([[0.,0.,1.]])
    obs=np.array([[.36,.48,.8],[-.8,.6,0.],[0.,0.,-1.]])
    vout=transverse_harmonics(obs,a.L);vin=transverse_harmonics(inc,a.L)
    _,_,ot,op=tangent_frame(obs)
    direct=np.empty((2*len(obs),2),complex)
    direct_incidents=[d.incident(inc[0],polar) for polar in ((1.,0.,0.),(0.,1.,0.))]
    if a.batch_size>1:direct_solutions,_=d.solve_incidents(direct_incidents,tol=a.tol,tile=a.batch_tile)
    else:direct_solutions=[solve(f)[0] for f in direct_incidents]
    for pol,(f,h) in enumerate(zip(direct_incidents,direct_solutions)):
        field=d.field(h,obs,far=True,incident=f)[:,5:8]
        direct[:,pol]=np.stack((np.einsum('ij,ij->i',field,ot),np.einsum('ij,ij->i',field,op)),axis=1).ravel()
    predicted=vout@kernel@vin.conj().T
    csca=2*np.pi/a.ka**2*norm_squared(T);cext=-2*np.pi/a.ka**2*np.trace(T).real
    from solve import lib
    kernel_bytes=int(lib.dirac_bytes(d.p));d.close()
    cyclic=mirror=None
    direct_incidents=direct_solutions=f=h=None
    import gc
    gc.collect()
    # Native pair cache released before the analytic contraction subprocess.
    t=time.perf_counter();so3_metadata={}
    release_pages(kernel);release_pages(T)
    if a.no_average:
        mu=np.empty((0,4,4));theta=np.empty(0)
    elif a.so3=='mstm':
        from tmatrix_so3_mstm import analytic_mueller_tmatrix
        mu,so3_metadata=analytic_mueller_tmatrix(T,a.L,theta,precision='fp64',threads=a.so3_threads)
    elif a.so3=='factored':
        from so3_factored import analytic_mueller_factored
        mu=a.ka**2*analytic_mueller_factored(kernel,a.L,theta)
    else:mu=a.ka**2*analytic_mueller(kernel,a.L,theta)
    so3_s=time.perf_counter()-t
    report=dict(shape=a.shape,ka=a.ka,index=str(a.index),ref=a.ref,L=a.L,degree=a.degree,backend=a.backend,solver=a.solver,tolerance=a.tol,fmm_eps=a.fmm_eps if a.backend!='dense' else None,target_order=a.target_order,near_order=a.order,near_subdiv=a.near_subdiv,columns=len(modal),new_columns=new_columns,resumed_columns=resumed_columns,dofs=d.size,
        kernel_bytes=kernel_bytes,setup_s=d.setup_s,rhs_s=rhs_s,solve_s=sum(r['seconds'] for r in statistics),
        far_projection_s=far_s,analytic_SO3_s=so3_s,wall_s=time.perf_counter()-start,wall_scope='current invocation including preparation; per-column solve/RHS/radiation totals include resumed columns',Euler_quadrature_nodes=0,
        iterations=[r['iterations'] for r in statistics],max_residual=max(r['residual'] for r in statistics),
        held_out_reconstruction_error=float(np.linalg.norm(predicted-direct)/np.linalg.norm(direct)),
        csca=float(csca),cext=float(cext),absorption_cross_section=float(cext-csca),
        lossless_energy_error=float(abs(cext-csca)/csca) if abs(complex(a.index).imag)<1e-14 else None)
    report['dense_acceleration']=d.acceleration
    report['dense_cache']=d.cache_info
    if a.backend=='fmm-sector':report['FMM']=dict(eps=a.fmm_eps,strict_eps=a.fmm_strict_eps,packet=a.fmm_packet,fmm_s=d.fmm_s,near_s=d.near_s,projection_s=d.projection_s,
        reference_scope='Full-space FMM at strict epsilon plus exact near corrections, audited by independent direct panel quadrature',
        coarse_action_s=coarse_action_s,stored_fine_W_bytes=stored_fine_W_bytes)
    report['sector_preconditioner']=coarse_report
    report.update(aspect=a.aspect,analytic_SO3_metadata=so3_metadata,memory_plan=memory_plan)
    report['coarse_factorization']=a.coarse_factorization
    report.update(reference_checks=a.reference_checks,column_reference_s=sum(s.get('full_reference_s',0.) for s in statistics),
                  column_reference_scope='Direct and mirrored final column audits; excludes setup/coarse/covariance audits')
    report.update(batch_size=a.batch_size,batch_tile=a.batch_tile,symmetry=a.symmetry,symmetry_info=symmetry_info,so3=a.so3,
        independent_columns=sum('mirror_from' not in s for s in statistics),mirror_columns=sum('mirror_from' in s for s in statistics))
    if a.shape=='sphere' and not a.no_average:
        aa,bb,last=mie_coefficients(a.index,a.ka);reference=np.diag([-bb[l] if p==0 else -aa[l] for p,l,m in modal])
        exact=np.moveaxis(np.asarray(mie_mueller(theta,a.index,a.ka)),-1,0)
        report['Mie_T_relative_error']=float(np.linalg.norm(T-reference)/np.linalg.norm(reference))
        report['Mie_all16_max_over_M11']=float(np.max(abs(mu-exact)/exact[:,0,0,None,None]))
    report.update(average_computed=not a.no_average,T_storage='mapped tmatrix.npy and kernel.npy; 32-column conversion',
                  matrix_conversion_buffer_bytes=32*len(modal)*16)
    temporary=a.out/'solution.tmp'
    with temporary.open('wb') as stream:
        np.savez(stream,T=T,kernel=kernel,theta=theta,mueller=mu,modes=modal)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,a.out/'solution.npz')
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    (a.out/'progress.json').write_text(json.dumps(dict(status='complete',completed_columns=len(modal),total_columns=len(modal)))+'\n')
    d.close()

if __name__=='__main__':main()
