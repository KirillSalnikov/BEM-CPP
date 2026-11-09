#!/usr/bin/env python3
"""Stream regular spherical-wave RHS through a persistent GPU/FMM operator.

Only two surface solutions reside at once. Completed modal columns are
atomic checkpoints. --columns START COUNT permits bounded/chunked builds.
The full kernel is written as a disk-backed array only after completion.
"""
import argparse
import contextlib
import hashlib
import json
import os
import shutil
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time
from collections import deque
import numpy as np
from tmatrix_projection import SphericalProjector
from tmatrix_so3_mstm import analytic_mueller_tmatrix
from tmatrix_symmetry import c2x_plan, c2x_partner_column
import tmatrix_c6
import tmatrix_pipeline
from tmatrix_order_shell import shell_schedule
from tmatrix_reference import (modes, tangent_frame, transverse_harmonics, transverse_harmonic_block,
                               analytic_mueller)

ROOT=Path(__file__).resolve().parents[1]


def receive_into(connection, array):
    view=memoryview(array).cast('B')
    while view:
        count=connection.recv_into(view)
        if count==0:
            raise RuntimeError("FMM worker exited; inspect worker.log")
        view=view[count:]


def receive_message(connection):
    length=np.empty(1,dtype=np.uint64)
    receive_into(connection,length)
    if length[0]>1024*1024:
        raise RuntimeError("invalid FMM worker message length")
    content=np.empty(int(length[0]),dtype=np.uint8)
    receive_into(connection,content)
    return json.loads(content.tobytes())


def atomic_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


def atomic_array(path,array):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.save(stream,array)
    temporary.replace(path)


def main():
    process_start=time.monotonic()
    # Use accurate residual replacement for mixed near-field iterations.
    # The final independent strict-residual gate remains active in all modes.
    os.environ.setdefault('BEM_MIXED_ITERATIVE_REFINEMENT','1')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape',choices=['sphere','prism','tetra'],default='sphere')
    parser.add_argument('--obj',type=Path)
    parser.add_argument('--particle','-p',nargs='+',type=float,
                        help='MBS-fast geometry: TYPE parameters; types 1,2,3,4,10,12,999')
    parser.add_argument('--particle-file','--pf',type=Path,help='Native MBS-fast polygon file')
    parser.add_argument('--particle-symmetry',choices=['auto','off'],default='auto',
                        help='Automatically use measured compatible particle symmetries; off is an independent reference path')
    parser.add_argument('--ref',type=int,default=3)
    parser.add_argument('--ka',type=float,default=1.)
    parser.add_argument('--ri',type=float,default=1.5)
    parser.add_argument('--ri-imag',type=float,default=0.)
    parser.add_argument('--prism-aspect',type=float,default=1.,help='Height / width across opposite faces')
    parser.add_argument('--wavelength',type=float,help='Physical vacuum wavelength in micrometres (scale metadata)')
    parser.add_argument('--order',type=int,default=5)
    parser.add_argument('--incident-min-order',type=int,default=1,
                        help='Experimental: compute only this incident degree and above, plus independent symmetry controls; keep all outgoing rows')
    parser.add_argument('--angular-band',type=int,default=0)
    parser.add_argument('--projection-method',choices=['fft','dense'],default='fft')
    parser.add_argument('--tol',type=float,default=1e-5)
    parser.add_argument('--adaptive-tight-tol',type=float,
                        help='Opt-in C6 policy: strict controls, retry failed loose packets and tighten higher degrees')
    parser.add_argument('--restart',type=int,default=80)
    parser.add_argument('--max-iters',type=int,default=500)
    parser.add_argument('--fast-mlfma',action='store_true',
                        help='Use mixed inner MLFMA kernels with FP64 residual checks; retain the configured band layout and assembly backend')
    parser.add_argument('--columns',type=int,nargs=2,metavar=('START','COUNT'))
    parser.add_argument('--column-indices',type=int,nargs='+',
                        help='Independently solve these modal indices in one worker session')
    parser.add_argument('--max-pairs-this-run',type=int,
                        help='Stop after this many worker solve pairs, including C6 audits/fallback; retain checkpoints')
    parser.add_argument('--c2-reuse',action='store_true',
                        help='For the built-in prism, audit C2x covariance then reconstruct negative-m columns')
    parser.add_argument('--c2-audit-tol',type=float,default=1e-4)
    parser.add_argument('--c6-pack',action='store_true',
                        help='Pack distinct C6 sectors using two phase codes, independent audits and fallback')
    parser.add_argument('--c6-audit-tol',type=float,default=1e-4)
    parser.add_argument('--mirror-pack',action='store_true',
                        help='Use horizontal-reflection parity with C6; select a D6h-invariant prism mesh')
    parser.add_argument('--hybrid-cpu',action='store_true',help='Overlap independent T packets with CPU FP64 gates; requires sector GMRES')
    parser.add_argument('--incident-prefetch',action='store_true',help='Prepare the next packet while the worker solves')
    parser.add_argument('--no-incident-cache',action='store_true',help='Disable shared geometry/radial incident-field cache')
    parser.add_argument('--seed-from',type=Path,
                        help='Copy compatible individual/C2 checkpoints into a new output directory')
    parser.add_argument('--solver-restart-from',type=Path,help='Audit and import accepted checkpoints after a solver-only change')
    parser.add_argument('--solver-restart-proof',type=Path,help='Physical equivalence and strict-residual validation receipt')
    parser.add_argument('--ntheta',type=int,default=37)
    parser.add_argument('--no-average',action='store_true')
    parser.add_argument('--average-backend',choices=['mstm','reference'],default='mstm')
    parser.add_argument('--so3-threads',type=int,default=4)
    parser.add_argument('--so3-precision',choices=['auto','fp32','fp64'],default='auto')
    parser.add_argument('--plane-only',action='store_true',help='Measure one direct plane-wave pair with the same resident operator')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--worker-binary',type=Path,help='Pin a worker executable for long builds')
    parser.add_argument('--neural-prior',type=Path,
                        help='Checked T-specific physical current prior; one shared D6h correction for all modal RHS')
    parser.add_argument('--operator-cache-prefix',type=Path,help='Reuse validated near/MBJ caches from an existing operator')
    args=parser.parse_args()
    particle_info=None
    if args.particle is not None or args.particle_file is not None:
        if args.obj is not None or (args.particle is not None and args.particle_file is not None):
            parser.error('Choose exactly one of --obj, --particle or --particle-file')
        from mbs_particle_geometry import prepare,write_obj
        try:
            vertices,triangles,particle_info=prepare(args.particle,args.particle_file)
        except (ValueError,OSError) as error:parser.error(str(error))
        # Content-address the generated surface; a rejected resume must never
        # overwrite geometry belonging to an existing checkpoint.
        geometry_hash=hashlib.sha256(vertices.tobytes()+triangles.tobytes()).hexdigest()
        geometry_directory=args.out/'geometry';geometry_directory.mkdir(parents=True,exist_ok=True)
        generated=geometry_directory/(geometry_hash+'.obj')
        if not generated.exists():write_obj(generated,vertices,triangles)
        atomic_json(generated.with_suffix('.json'),particle_info)
        if args.particle is not None and args.particle[0]==1:
            # Preserve the established full D6h prism implementation, including
            # its quality-controlled mesh and specialized preconditioner.
            args.shape='prism'
            args.prism_aspect=args.particle[1]/(np.sqrt(3)*args.particle[2]/2)
            os.environ.setdefault('BEM_TMATRIX_D6H_MESH','1')
        else:
            args.shape='particle';args.obj=generated
        sym=particle_info['solver_symmetries']
        if args.c2_reuse and not sym['c2_x']:parser.error('--c2-reuse requires verified C2x symmetry')
        if args.c6_pack and not sym['c6_z']:parser.error('--c6-pack requires verified C6z symmetry')
        if args.mirror_pack and not (sym['c6_z'] and sym['horizontal_mirror']):
            parser.error('--mirror-pack requires verified C6z and horizontal reflection symmetry')
        if args.particle_symmetry=='auto':
            full=not (args.columns or args.column_indices or args.plane_only)
            if full:
                args.c2_reuse=sym['c2_x'];args.c6_pack=sym['c6_z']
                args.mirror_pack=sym['c6_z'] and sym['horizontal_mirror']
            args.incident_prefetch=True
            # Small-ka higher multipoles radiate by cancellation. Keep the
            # radiation integral accurate; this does not replace covariance
            # checks of the complete discretized equation.
            if args.ka<=1:
                os.environ.setdefault('BEM_FARFIELD_FAST_PHASE','0')
                os.environ.setdefault('BEM_FARFIELD_FP32_ACCUM','0')
            for key,value in {'BEM_FMM_P2P_DOUBLE_SINGLE':'1','BEM_FMM_P2P_DS_FACTORED':'1',
                'BEM_FMM_P2P_DS_THREADS':'64','BEM_FMM_P2P_DS_LEAF_SPLIT':'32'}.items():
                os.environ.setdefault(key,value)
            if sym['d6h_z']:
                if args.obj is not None:os.environ['BEM_TMATRIX_GEOMETRY_D6H']='1'
                for key,value in {'BEM_TMATRIX_SECTOR_GMRES':'1','BEM_TMATRIX_ORBIT_NODES':'50',
                    'BEM_TMATRIX_SECTOR_RECYCLE_RANK':'8','BEM_TMATRIX_VERTICAL_PRECONDITIONER':'1',
                    'BEM_TMATRIX_SYMMETRY_WARM_START':'1'}.items():os.environ.setdefault(key,value)
            else:
                os.environ.setdefault('BEM_MULLER_GPU_ASSEMBLY','0')
                if os.environ['BEM_MULLER_GPU_ASSEMBLY']=='0':os.environ.setdefault('BEM_TMATRIX_RECYCLE_RANK','8')
    builtin_prism=args.shape=='prism' and args.obj is None
    symmetry_allowed=particle_info['solver_symmetries'] if particle_info else {}
    compressed_case=os.environ.get('BEM_TMATRIX_COMPRESSED_FFT_CASE')
    if compressed_case:
        compressed_case=Path(compressed_case).resolve()
        asset_path=compressed_case/'service_assets.json'
        if args.shape!='prism' or args.obj is not None or not asset_path.is_file():
            parser.error('compressed FFT requires a built-in prism and a prepared service_assets.json')
    neural_prior_info=None
    if args.neural_prior is not None:
        from tmatrix_neural_prior import configure
        neural_prior_info=configure(args.neural_prior,shape=args.shape,obj=args.obj,ref=args.ref,
            ka=args.ka,ri=args.ri,ri_imag=args.ri_imag,aspect=args.prism_aspect)
    if args.adaptive_tight_tol is not None and (
            not np.isfinite(args.adaptive_tight_tol) or not 0<args.adaptive_tight_tol<=args.tol
            or not args.c6_pack or args.plane_only):
        parser.error('--adaptive-tight-tol requires C6 packing, 0 < tight <= tol, and modal solves')
    if args.hybrid_cpu:
        os.environ['BEM_TMATRIX_HYBRID']='1'
    if os.environ.get('BEM_TMATRIX_HYBRID','0')!='0':
        os.environ.setdefault('BEM_TMATRIX_CPU_STRICT','1')
        # Idle OpenMP teams must yield cores to the other packet/checker.
        os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')
        os.environ.setdefault('GOMP_SPINCOUNT','0')
        local_fftw=Path.home()/'.local/lib/libfftw3.so.3'
        if local_fftw.exists():os.environ.setdefault('BEM_CPU_FFTW_LIBRARY',str(local_fftw))
    if bool(args.solver_restart_from)!=bool(args.solver_restart_proof) or (args.seed_from and args.solver_restart_from):
        parser.error('solver restart requires source and proof, and cannot be combined with seed-from')
    if not np.isfinite(args.prism_aspect) or args.prism_aspect<=0:
        parser.error('--prism-aspect must be finite and positive')
    if args.wavelength is not None and (not np.isfinite(args.wavelength) or args.wavelength<=0):
        parser.error('--wavelength must be finite and positive')
    if args.shape=='prism' and args.obj is None:
        os.environ['BEM_TMATRIX_PRISM_ASPECT']=str(args.prism_aspect)
    if args.mirror_pack:
        if not args.c6_pack:
            parser.error('--mirror-pack requires --c6-pack')
        if not builtin_prism and not (symmetry_allowed.get('c6_z') and symmetry_allowed.get('horizontal_mirror')):
            parser.error('--mirror-pack requires measured C6 and horizontal reflection symmetry')
        os.environ['BEM_TMATRIX_D6H_MESH']='1'
        if not builtin_prism and symmetry_allowed.get('d6h_z'):os.environ['BEM_TMATRIX_GEOMETRY_D6H']='1'
    if args.fast_mlfma:
        if int(os.environ.get('BEM_FMM_BANDED_SPLIT_DEPTH','0'))<=0:
            parser.error('--fast-mlfma requires BEM_FMM_BANDED_SPLIT_DEPTH>0; choose a band layout for the mesh')
        os.environ['BEM_FMM_MULTILEVEL']='1'
        for key,value in {
            'BEM_FMM_MULTILEVEL_PAIR':'1',
            'BEM_FMM_MULTILEVEL_M2L_FP32':'1',
            'BEM_FMM_M2L_DIRECTION_BLOCKS':'1',
            'BEM_FMM_MULTILEVEL_P2M_FP32':'1',
            'BEM_FMM_MULTILEVEL_P2M_SHARED':'1',
            'BEM_FMM_MULTILEVEL_L2P_FP32':'1',
            'BEM_FMM_MULTILEVEL_INTERPOLATION_FP32':'1',
            'BEM_FMM_MULTILEVEL_COMPACT_TEMP':'1',
            'BEM_FMM_P2P_SHARED_SOURCES':'1',
            'BEM_FMM_MULTILEVEL_SHIFT_CACHE':'1',
            'BEM_FMM_MULTILEVEL_M2L_CACHE':'1',
            'BEM_FMM_MULTILEVEL_FLAT_NEAR':'1',
            'BEM_FMM_MULTILEVEL_CHUNK':'256',
            'BEM_MIXED_ADAPTIVE_INNER':'1',
        }.items():
            os.environ.setdefault(key,value)
    if args.max_pairs_this_run is not None and args.max_pairs_this_run<1:
        parser.error('--max-pairs-this-run must be positive')
    if args.columns and args.column_indices:
        parser.error('choose --columns or --column-indices')
    if not 1<=args.incident_min_order<=args.order:
        parser.error('--incident-min-order must be between 1 and --order')
    if args.incident_min_order>1 and (args.columns or args.column_indices or args.plane_only or not args.no_average):
        parser.error('an incident-order shell requires --no-average and cannot be combined with other column selectors or --plane-only')
    if args.c2_reuse and (not (builtin_prism or symmetry_allowed.get('c2_x')) or args.columns or args.column_indices or args.plane_only):
        parser.error('--c2-reuse requires a complete build with verified C2x symmetry')
    if not np.isfinite(args.c2_audit_tol) or args.c2_audit_tol<=0:
        parser.error('--c2-audit-tol must be finite and positive')
    if args.c6_pack and (not (builtin_prism or symmetry_allowed.get('c6_z')) or args.columns or args.column_indices or args.plane_only):
        parser.error('--c6-pack requires a complete build with verified C6z symmetry')
    if not np.isfinite(args.c6_audit_tol) or args.c6_audit_tol<=0:
        parser.error('--c6-audit-tol must be finite and positive')
    if args.so3_threads<1 or args.order<1 or args.order>256 or args.ref<0 or args.ka<=0 or args.tol<=0 or args.ntheta<2:
        parser.error('invalid order, mesh, wavenumber, tolerance, or angular count')
    if not args.no_average and args.average_backend=='reference' and args.order>8:
        parser.error('the reference SO3 implementation is limited to L<=8; use --average-backend mstm')
    if not args.no_average and args.average_backend=='mstm':
        precision=('fp64' if args.order<=32 else 'fp32') if args.so3_precision=='auto' else args.so3_precision
        post_binary=ROOT/'bin'/('tmatrix_so3_mstm' if precision=='fp64' else 'tmatrix_so3_mstm_fp32')
        if not post_binary.exists(): parser.error(f'build {post_binary.name} before starting the T build')
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    column_dir=out/'columns'; column_dir.mkdir(exist_ok=True)
    modal_list=modes(args.order); count=len(modal_list)
    first,number=args.columns or (0,count)
    if first<0 or number<1 or first+number>count:
        parser.error('column range must lie inside the complete T matrix')
    if args.column_indices and any(i<0 or i>=count for i in args.column_indices):
        parser.error('column indices must lie inside the complete T matrix')
    band=args.angular_band or (2*args.order+4)
    if band<=args.order:
        parser.error('angular band must exceed multipole order')
    projection_bytes=count*2*(2*band*band+10)*16 if args.projection_method=='dense' else 0
    kernel_bytes=count*count*16
    estimated_disk=projection_bytes+3*kernel_bytes
    if estimated_disk>0.8*shutil.disk_usage(out).free:
        raise RuntimeError(f'T/projection storage needs about {estimated_disk/2**30:.1f} GiB; insufficient free disk space')
    z,w=np.polynomial.legendre.leggauss(band)
    zz,pp=np.meshgrid(z,np.arange(2*band)*np.pi/band,indexing='ij')
    directions=np.stack((np.sqrt(1-zz*zz)*np.cos(pp),np.sqrt(1-zz*zz)*np.sin(pp),zz),axis=-1).reshape(-1,3)
    weights=np.repeat(w,2*band)*np.pi/band
    projection_points=len(directions)
    rng=np.random.default_rng(193)
    checks=rng.normal(size=(9,3)); checks/=np.linalg.norm(checks,axis=1)[:,None]
    directions=np.vstack((directions,checks,[[0.,0.,1.]]))
    weights=np.r_[weights,np.zeros(10)]
    directions.tofile(out/'directions.bin')
    check_basis=transverse_harmonics(directions[projection_points:],args.order)
    _,_,et,ep=tangent_frame(directions)
    geometry=str(args.obj.resolve()) if args.obj else args.shape
    binary=args.worker_binary.resolve() if args.worker_binary else ROOT/'bin/muller_tmatrix_fmm_server'
    fingerprint={k:v for k,v in vars(args).items() if k not in ('out','columns','column_indices','max_pairs_this_run','fast_mlfma','ntheta','no_average','obj','plane_only','operator_cache_prefix','average_backend','so3_threads','so3_precision','worker_binary','seed_from','solver_restart_from','solver_restart_proof')}
    fingerprint.pop('particle');fingerprint.pop('particle_file');fingerprint.pop('particle_symmetry')
    if particle_info is not None:
        fingerprint['particle_geometry']=particle_info
        fingerprint['particle_generator_digest']=hashlib.sha256((ROOT/'scripts/mbs_particle_geometry.py').read_bytes()).hexdigest()
    fingerprint.pop('neural_prior',None)
    if neural_prior_info is not None:
        fingerprint['neural_tmatrix_prior']=neural_prior_info
    if args.incident_min_order==1:
        fingerprint.pop('incident_min_order')
    else:
        fingerprint['order_shell_digest']=hashlib.sha256((ROOT/'scripts/tmatrix_order_shell.py').read_bytes()).hexdigest()
    if args.adaptive_tight_tol is None:
        fingerprint.pop('adaptive_tight_tol')
    if args.c6_pack:
        fingerprint['c6_implementation_digest']=hashlib.sha256((ROOT/'scripts/tmatrix_c6.py').read_bytes()).hexdigest()
        fingerprint['c6_driver_digest']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    else:
        fingerprint.pop('c6_pack'); fingerprint.pop('c6_audit_tol')
    if not args.mirror_pack:
        fingerprint.pop('mirror_pack')
    fingerprint.pop('hybrid_cpu')  # Captured by BEM_environment and worker digest.
    fingerprint.pop('incident_prefetch')
    fingerprint['incident_schedule_digest']=hashlib.sha256((ROOT/'scripts/tmatrix_pipeline.py').read_bytes()).hexdigest()
    fingerprint.pop('no_incident_cache')
    if not args.c2_reuse:
        # Preserve existing non-symmetry checkpoint identities.
        fingerprint.pop('c2_reuse'); fingerprint.pop('c2_audit_tol')
    fingerprint['geometry']=geometry
    fingerprint['geometry_digest']=hashlib.sha256(args.obj.read_bytes()).hexdigest() if args.obj else None
    if particle_info is not None and args.obj is not None:
        fingerprint['geometry']='particle_obj_sha256:'+fingerprint['geometry_digest']
    fingerprint['binary_digest']=hashlib.sha256(binary.read_bytes()).hexdigest()
    if compressed_case:
        fingerprint['compressed_fft_assets_digest']=hashlib.sha256((compressed_case/'service_assets.json').read_bytes()).hexdigest()
        code=hashlib.sha256()
        for name in ('compressed_fft_service.py','symmetric_galerkin_fft.py','closed_galerkin_fft.py','galerkin_cross_fft.py','streamed_galerkin_fft.py','tiled_galerkin_fft.py','surface_symmetry.py','regular_surface_geometry.py'):
            code.update((ROOT/'scripts'/name).read_bytes())
        fingerprint['compressed_fft_code_digest']=code.hexdigest()
    if os.environ.get('BEM_TMATRIX_RITZ_PRIOR'):
        prior=Path(os.environ['BEM_TMATRIX_RITZ_PRIOR']).resolve()
        digest=hashlib.sha256()
        with prior.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        fingerprint['physical_ritz_prior_digest']=digest.hexdigest()
        os.environ['BEM_TMATRIX_RITZ_PRIOR']=str(prior)
    fingerprint['incident_digest']=hashlib.sha256((ROOT/'scripts/tmatrix_incident.py').read_bytes()).hexdigest()
    fingerprint['projection_digest']=hashlib.sha256((ROOT/'scripts/tmatrix_projection.py').read_bytes()).hexdigest()
    fingerprint['harmonics_digest']=hashlib.sha256((ROOT/'scripts/tmatrix_reference.py').read_bytes()).hexdigest()
    if args.c2_reuse:
        fingerprint['symmetry_digest']=hashlib.sha256((ROOT/'scripts/tmatrix_symmetry.py').read_bytes()).hexdigest()
    fingerprint['BEM_environment']={k:v for k,v in os.environ.items() if k.startswith('BEM_')}
    signature=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()+directions.tobytes()).hexdigest()
    manifest_path=out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['signature']!=signature:
        raise RuntimeError('output columns belong to a different operator/order; use another output directory')
    atomic_json(manifest_path,{'signature':signature,'parameters':fingerprint,'modes':modal_list})
    if args.solver_restart_from:
        from tmatrix_checkpoint_migration import migrate
        migrated=migrate(args.solver_restart_from,out,fingerprint,directions,modal_list,args.tol,signature,args.solver_restart_proof)
        print(f'Migrated {migrated["copied_columns"]} verified columns and {migrated["copied_packets"]} packed receipts',flush=True)
    if args.seed_from:
        seeded=tmatrix_c6.seed_columns(args.seed_from,out,fingerprint,directions,modal_list,args.tol)
        atomic_json(out/'seed_report.json',seeded)
        print(f'Seeded {seeded["copied_columns"]} compatible checkpoints',flush=True)
    # O(q^2) angular projection lives on disk and is generated in fixed-size
    # mode blocks. No dense surface matrix or surface-current slab is formed.
    projection_path=out/'projection.npy'
    projection_marker=out/'projection.signature'
    projection_signature=hashlib.sha256(directions.tobytes()+str(args.order).encode()+
                                       fingerprint['harmonics_digest'].encode()).hexdigest()
    if args.projection_method=='dense' and not (projection_marker.exists() and projection_marker.read_text().strip()==projection_signature):
        temporary_projection=out/'projection.tmp.npy'
        projection=np.lib.format.open_memmap(temporary_projection,mode='w+',dtype=np.complex128,
                                            shape=(count,2*len(directions)))
        for begin in range(0,count,32):
            selected=modal_list[begin:begin+32]
            block=transverse_harmonic_block(directions,selected)
            projection[begin:begin+len(selected)]=block.conj().T*np.repeat(weights,2)[None,:]
        projection.flush(); del projection
        temporary_projection.replace(projection_path)
        projection_marker.write_text(projection_signature+'\n')
    atomic_json(out/'progress.json',{'state':'preparing_projection','pid':os.getpid(),'mode_count':count,'projection_method':args.projection_method})
    print(f'Preparing {args.projection_method} projection for {count} T modes',flush=True)
    weighted_projection=(np.load(projection_path,mmap_mode='r') if args.projection_method=='dense'
                         else SphericalProjector(args.order,z,w))
    schedule=shell_schedule(args.order,args.incident_min_order,c2=args.c2_reuse,c6=args.c6_pack,mirror=args.mirror_pack)
    requested=(schedule['requested'] if args.incident_min_order>1 else
               list(dict.fromkeys(args.column_indices)) if args.column_indices else list(range(first,first+number)))
    def valid_column(index):
        metadata=column_dir/f'{index:06d}.json'
        data=column_dir/f'{index:06d}.npy'
        if not metadata.exists() or not data.exists():
            return False
        try:
            information=json.loads(metadata.read_text())
            stored=information['status']
            residual=max(stored['strict_residual_x'],stored['strict_residual_y'])
            if args.adaptive_tight_tol is not None:
                limit=stored.get('requested_tolerance')
                if limit not in (args.tol,args.adaptive_tight_tol) or residual>limit:
                    return False
                if 'c6_batch' not in information and limit!=args.adaptive_tight_tol:
                    return False
            values=np.load(data,mmap_mode='r')
            valid=(np.isfinite([stored['strict_residual_x'],stored['strict_residual_y']]).all() and residual<=args.tol and
                   information['mode']==list(modal_list[index]) and
                   values.shape==(count,) and np.isfinite(values).all())
            if not valid:
                return False
            if information.get('derived_by')=='C2x_half_turn':
                origin=information['source_column']
                if (not args.c2_reuse or not isinstance(origin,int) or not 0<=origin<count
                        or modal_list[origin][2]<=0 or int(c2_permutation[origin])!=index
                        or not valid_column(origin)):
                    return False
                original=np.load(column_dir/f'{origin:06d}.npy')
                if not np.array_equal(values,c2x_partner_column(original,origin,c2_permutation,c2_signs)):
                    return False
            if 'c6_batch' in information:
                if not args.c6_pack:
                    return False
                batch=information['c6_batch']
                if not isinstance(batch,str) or len(batch)!=24 or any(c not in '0123456789abcdef' for c in batch):
                    return False
                receipt=json.loads((out/'c6_batches'/f'{batch}.json').read_text())
                origin=information.get('source_column',index)
                if not (receipt['passed'] and receipt['signature']==signature and origin in receipt['columns']):
                    return False
                if origin!=index:
                    parent=json.loads((column_dir/f'{origin:06d}.json').read_text())
                    return parent.get('c6_batch')==batch and valid_column(origin)
                if receipt['column_sha256'][str(index)]!=hashlib.sha256(data.read_bytes()).hexdigest():
                    return False
            return True
        except (OSError,ValueError,KeyError):
            return False
    symmetry_report=None
    c2_verified=False
    if args.c2_reuse:
        c2_permutation,c2_signs=schedule['permutation'],schedule['signs']
        c2_representatives,c2_audits=schedule['representatives'],schedule['c2_audits']
        # Audits come first, so an incompatible numerical operator is rejected
        # before spending time on the remaining representative columns.
        audit_work=schedule['audit_work']
        work_order=schedule['work']
    else:
        work_order=requested
    def update_c2_columns(new_sources=None):
        nonlocal symmetry_report,c2_verified
        if not args.c2_reuse:
            return
        became_verified=not c2_verified
        if not c2_verified:
            if not all(valid_column(i) and valid_column(int(c2_permutation[i])) for i in c2_audits):
                return
            comparisons=[]
            for target in c2_audits:
                source=int(c2_permutation[target])
                original=np.load(column_dir/f'{source:06d}.npy')
                measured=np.load(column_dir/f'{target:06d}.npy')
                predicted=c2x_partner_column(original,source,c2_permutation,c2_signs)
                error=float(np.linalg.norm(predicted-measured)/max(np.linalg.norm(measured),1e-300))
                comparisons.append({'source':source,'target':target,'relative_column_error':error})
            maximum=max(item['relative_column_error'] for item in comparisons)
            symmetry_report={'method':'C2x_half_turn','audit_columns':comparisons,
                'maximum_audit_relative_error':maximum,'tolerance':args.c2_audit_tol,
                'passed':bool(np.isfinite(maximum) and maximum<=args.c2_audit_tol),
                'audit_is_sampled_not_an_operator_norm_bound':True,
                'reconstructed_columns_have_no_independent_residual_check':True}
            atomic_json(out/'c2_audit.json',symmetry_report)
            if not symmetry_report['passed']:
                raise RuntimeError(f'C2 column audit failed: {maximum:.3e}; no symmetry-derived columns accepted')
            c2_verified=True
            print(f'C2 covariance accepted on {len(comparisons)} independent column pairs: {maximum:.3e}',flush=True)
        # Scan all representatives once on resume / first acceptance. During
        # a long build only inspect the newly solved pair (never O(q^3) I/O).
        sources=c2_representatives if became_verified or new_sources is None else new_sources
        for source in sources:
            if modal_list[source][2]<0:
                continue
            target=int(c2_permutation[source])
            if target==source or valid_column(target) or not valid_column(source):
                continue
            original=np.load(column_dir/f'{source:06d}.npy')
            values=c2x_partner_column(original,source,c2_permutation,c2_signs)
            metadata=json.loads((column_dir/f'{source:06d}.json').read_text())
            atomic_array(column_dir/f'{target:06d}.npy',values)
            provenance={'c6_batch':metadata['c6_batch']} if 'c6_batch' in metadata else {}
            atomic_json(column_dir/f'{target:06d}.json',{
                'mode':modal_list[target],'status':metadata['status'],
                'scale':metadata['scale'],'derived_by':'C2x_half_turn','source_column':source,
                'strict_residual_measured_for_this_column':False,**provenance})
    update_c2_columns()
    pending=[i for i in work_order if not valid_column(i)]
    if args.plane_only:
        pending=[]
    tasks=deque()
    tight_from_degree=None
    adaptive_dir=out/'adaptive_tolerance'
    if args.adaptive_tight_tol is not None:
        adaptive_dir.mkdir(exist_ok=True)
        for path in adaptive_dir.glob('*.json'):
            prior=json.loads(path.read_text())
            if prior.get('signature')==signature:
                degree=int(prior['tight_from_degree'])
                tight_from_degree=degree if tight_from_degree is None else min(degree,tight_from_degree)
    def task_tolerance(kind,group):
        if args.adaptive_tight_tol is None:
            return args.tol
        if kind=='single' or (tight_from_degree is not None and
                             max(modal_list[i][1] for i in group)>=tight_from_degree):
            return args.adaptive_tight_tol
        return args.tol
    def single_tasks(indices):
        return [('single',indices[i:i+2]) for i in range(0,len(indices),2)]
    c6_reference_groups=[]
    c6_fallback_path=out/'c6_individual_fallback.json'
    c6_individual_fallback=False
    if particle_info is not None and args.particle_symmetry=='auto' and c6_fallback_path.exists():
        c6_individual_fallback=json.loads(c6_fallback_path.read_text()).get('signature')==signature
    if args.c6_pack:
        for name in ('c6_audits','c6_batches','c6_rejections'):
            (out/name).mkdir(exist_ok=True)
        c6_reference_groups=schedule['c6_groups']
    if args.c6_pack and not c6_individual_fallback:
        controls=list(dict.fromkeys((audit_work if args.c2_reuse else [])+
                      [index for group in c6_reference_groups for index in group]))
        missing_controls=[i for i in controls if not valid_column(i)]
        tasks.extend(single_tasks(missing_controls))
        for group in c6_reference_groups:
            audit_path=out/'c6_audits'/f'{tmatrix_c6.packet_id(group)}.json'
            try:
                audit=json.loads(audit_path.read_text())
                reusable=(audit['passed'] and audit['signature']==signature
                          and all(valid_column(i) for i in group))
            except (OSError,ValueError,KeyError):
                reusable=False
            if not reusable:
                tasks.append(('audit',group))
        rejections=[]
        for path in (out/'c6_rejections').glob('*.json'):
            rejection=json.loads(path.read_text())
            if rejection.get('signature')==signature:
                rejections.append(rejection)
        remaining=[i for i in pending if i not in set(controls)]
        tasks.extend(tmatrix_c6.production_tasks(modal_list,remaining,rejections,args.mirror_pack))
    else:
        tasks.extend(single_tasks(pending))
    print(f'FMM T matrix: {count} modes; requested {len(requested)}, pending {len(pending)}',flush=True)
    timings=[]
    atomic_json(out/'progress.json',{'state':'starting','pid':os.getpid(),'mode_count':count,'requested_columns':len(requested),'pending_columns':len(pending),'planned_worker_pairs':len(tasks),'disk_estimate_gib':estimated_disk/2**30})
    startup=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='bem-tm-') as temporary, (out/'worker.log').open('a') as log, contextlib.ExitStack() as companions:
        socket_path=str(Path(temporary)/'worker.sock')
        command=[str(binary),socket_path,geometry,str(args.ref),str(args.ka),str(args.ri),str(args.ri_imag),
                 str(out/'directions.bin'),str(args.operator_cache_prefix.resolve() if args.operator_cache_prefix else out/'operator'),str(args.tol),str(args.restart),str(args.max_iters)]
        env=os.environ.copy(); env.setdefault('OMP_NUM_THREADS','8')
        # Worker-only values: derive AFTER fingerprinting to avoid signing the
        # signature itself. Never share a recycle space across operators/runs.
        env['BEM_TMATRIX_RECYCLE_SIGNATURE']=signature
        env['BEM_TMATRIX_RECYCLE_PATH']=str(out/'krylov_recycle.bin')
        if compressed_case:
            from compressed_fft_service import Companion
            fft_socket=Path(temporary)/'compressed_fft.sock'
            companions.enter_context(Companion(compressed_case,fft_socket,out))
            # A temporary transport path is worker-only, never part of the
            # persistent physical/checkpoint fingerprint.
            env['BEM_TMATRIX_COMPRESSED_FFT_SOCKET']=str(fft_socket)
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        preparation=None
        try:
            while True:
                try:
                    connection.connect(socket_path); break
                except (FileNotFoundError,ConnectionRefusedError):
                    if process.poll() is not None:
                        raise RuntimeError('FMM worker startup failed; inspect worker.log')
                    if time.monotonic()-startup>600:
                        raise RuntimeError('FMM worker startup timed out')
                    time.sleep(.05)
            hello=receive_message(connection)
            if args.adaptive_tight_tol is not None and not hello.get('per_packet_tolerance'):
                raise RuntimeError('adaptive tolerance requires a worker with per-packet tolerance support')
            if neural_prior_info is not None:
                from tmatrix_neural_prior import validate_worker
                validate_worker(neural_prior_info,hello)
            if args.shape=='prism' and args.obj is None:
                actual_aspect=hello.get('prism_height_to_width_across_flats',1.)
                if not np.isfinite(actual_aspect) or not np.isclose(actual_aspect,args.prism_aspect,rtol=1e-13,atol=0):
                    raise RuntimeError('worker prism aspect does not match requested height/width')
            recycle_rank=int(hello.get('krylov_recycle_capacity',0))
            if recycle_rank:
                # U and C, plus the temporary file during atomic replacement.
                recycle_disk_peak=2*(104+2*recycle_rank*hello['system_dofs']*16)
                estimated_disk+=recycle_disk_peak
                hello['recycle_checkpoint_peak_gib']=recycle_disk_peak/2**30
                if estimated_disk>0.8*shutil.disk_usage(out).free:
                    raise RuntimeError('insufficient free disk for T/projection and atomic recycle checkpoints')
            if args.mirror_pack and not (hello.get('prism_dihedral_mesh') or hello.get('geometry_d6h_verified')):
                raise RuntimeError('mirror packing requires a worker with a verified D6h-invariant mesh')
            positions=np.empty((hello['rhs_points'],3),dtype=np.float64)
            receive_into(connection,positions)
            atomic_json(out/'operator_metadata.json',hello)
            print(json.dumps(hello),flush=True)
            far=np.empty((2,len(directions),3),dtype=np.complex128)

            def send_operation(operation,limit):
                connection.sendall(struct.pack('=iid',5,operation,limit) if args.adaptive_tight_tol is not None
                                   else struct.pack('=i',operation))

            def result(limit=None):
                limit=args.tol if limit is None else limit
                status=receive_message(connection)
                receive_into(connection,far)
                if args.adaptive_tight_tol is not None and status.get('requested_tolerance')!=limit:
                    raise RuntimeError('worker did not honor the requested packet tolerance')
                residual=max(status['strict_residual_x'],status['strict_residual_y'])
                if not np.isfinite([status['strict_residual_x'],status['strict_residual_y']]).all() or residual>limit:
                    atomic_json(out/'rejected_solve.json',status)
                    raise RuntimeError(f'modal solve rejected: strict residual {residual}; no column checkpoint written')
                return status

            fields=np.empty((2,len(positions),6),dtype=np.complex128)
            preparation=tmatrix_pipeline.IncidentPreparation(positions,modal_list,args.ka,fields,
                cache=not args.no_incident_cache and not args.plane_only,prefetch=args.incident_prefetch)
            plane_control=None
            if args.plane_only:
                plane=np.array([1.,0.,0.,0.,1.,0.,0.,0.,1.],dtype=np.float64)
                connection.sendall(struct.pack('=i',2)); connection.sendall(memoryview(plane).cast('B'))
                plane_control=result()
                atomic_json(out/'plane_control.json',plane_control)
            inflight=deque()
            capacity=int(hello.get('hybrid_batch_capacity',1))
            if capacity not in (1,2):
                raise RuntimeError('unsupported worker batch capacity')
            while tasks or inflight:
                if not inflight:
                    if args.max_pairs_this_run is not None and len(timings)>=args.max_pairs_this_run:
                        break
                    batch=[]
                    while tasks and len(batch)<capacity:
                        if args.max_pairs_this_run is not None and len(timings)+len(batch)>=args.max_pairs_this_run:
                            break
                        # An audit depends on previously saved reference columns.
                        # Never send it before all earlier results are committed.
                        if batch and (tasks[0][0]=='audit' or batch[0][0]=='audit'):
                            break
                        kind,group=tasks.popleft()
                        if kind!='audit':
                            group=[i for i in group if not valid_column(i)]
                            if not group:
                                continue
                            if len(group)==1:
                                kind='single'
                        batch.append((kind,group))
                    if not batch:
                        continue
                    if len(batch)==2:
                        connection.sendall(struct.pack('=ii',4,2))
                    for kind,group in batch:
                        packing_reflection=args.mirror_pack and kind!='packed_rotation'
                        scales,combined,field_seconds,field_wait,prefetched=preparation.get(kind,group,packing_reflection)
                        operation=3 if (kind in ('packed','packed_rotation') and
                                        (hello.get('symmetry_warm_start',False) or
                                         hello.get('krylov_recycle_capacity',0)>0 or
                                         hello.get('sector_recycle_capacity',0)>0)) else 1
                        limit=task_tolerance(kind,group)
                        send_operation(operation,limit)
                        connection.sendall(memoryview(fields).cast('B'))
                        inflight.append((kind,group,packing_reflection,scales,combined,
                                         field_seconds,field_wait,prefetched,limit))
                    # Both payloads have been consumed; one shared field buffer
                    # can now prepare the next packet while this batch solves.
                    if tasks and (args.max_pairs_this_run is None or len(timings)+len(inflight)<args.max_pairs_this_run):
                        next_kind,next_group=tasks[0]
                        preparation.prefetch(next_kind,next_group,args.mirror_pack and next_kind!='packed_rotation')
                kind,group,packing_reflection,scales,combined,field_seconds,field_wait,prefetched,limit=inflight.popleft()
                pair=(group if len(group)==2 else group*2) if kind=='single' else group
                # result() verifies the original FP64 residual BEFORE any
                # column, symmetry receipt, or checkpoint is written.
                status=result(limit);status['incident_field_s']=field_seconds
                status['incident_wait_s']=field_wait
                status['incident_prefetched']=prefetched
                status['rhs_kind']=kind
                timings.append(status)
                projected=[]
                for slot in range(2):
                    transverse=np.stack((np.einsum('ni,ni->n',et,far[slot]),
                                         np.einsum('ni,ni->n',ep,far[slot])),axis=1).reshape(-1)
                    projected.append(weighted_projection@transverse)
                saved=[]
                if kind=='single':
                    for slot,index in enumerate(pair):
                        if slot and pair[slot]==pair[0]:
                            continue
                        p,l,m=modal_list[index]
                        kernel_column=projected[slot]*(4*np.pi*1j**(l-p)*scales[slot])
                        atomic_array(column_dir/f'{index:06d}.npy',kernel_column)
                        atomic_json(column_dir/f'{index:06d}.json',{
                            'mode':modal_list[index],'status':status,'scale':scales[slot]})
                        saved.append(index)
                else:
                    columns,check=tmatrix_c6.decode_columns(projected,modal_list,group,scales,combined,args.c6_audit_tol,packing_reflection)
                    batch=tmatrix_c6.packet_id(group)
                    check.update(signature=signature,status=status)
                    if kind=='audit':
                        references={i:np.load(column_dir/f'{i:06d}.npy') for i in group}
                        check['independent_reference_check']=tmatrix_c6.compare_references(columns,references,args.c6_audit_tol)
                        check['passed']=check['passed'] and check['independent_reference_check']['passed']
                    retry_tighter=(not check['passed'] and args.adaptive_tight_tol is not None
                                   and limit>args.adaptive_tight_tol)
                    if retry_tighter:
                        degree=max(modal_list[i][1] for i in group)
                        tight_from_degree=degree if tight_from_degree is None else min(degree,tight_from_degree)
                        check['tight_from_degree']=tight_from_degree
                        atomic_json(adaptive_dir/f'{kind}_{batch}.json',check)
                        tasks.appendleft((kind,group))
                        print(f'C6 packet failed at {limit:g}; retry at {args.adaptive_tight_tol:g}, '
                              f'tighten degrees >= {tight_from_degree}',flush=True)
                    elif kind=='audit':
                        atomic_json(out/'c6_audits'/f'{batch}.json',check)
                        if not check['passed']:
                            if particle_info is None or args.particle_symmetry!='auto' or inflight:
                                raise RuntimeError('C6 packed RHS audit failed; no packed checkpoints accepted')
                            # Geometry can have exact C6 while a coarse quadrature
                            # or weak multipole fails the numerical packing gate.
                            # Keep independently solved columns, discard this
                            # decoded packet and persist a direct-solve fallback.
                            c6_individual_fallback=True
                            atomic_json(c6_fallback_path,dict(signature=signature,
                                reason='packed audit failed',audit_packet=batch,
                                packed_columns_accepted=False))
                            tasks.clear()
                            tasks.extend(single_tasks([i for i in work_order if not valid_column(i)]))
                            print('C6 numerical audit rejected packing; continuing with individual columns',flush=True)
                    elif not check['passed']:
                        # Remember the rejection across interruption. Never retry
                        # the same failed symmetry level on restart.
                        atomic_json(out/'c6_rejections'/f'{batch}.json',check)
                        fallback=tmatrix_c6.production_tasks(modal_list,group,[check],args.mirror_pack)
                        tasks.extendleft(reversed(fallback))
                        strategy='retrying rotational sectors' if packing_reflection else 'solving individually'
                        print(f'C6 packet {group} failed cross-talk check; {strategy}',flush=True)
                    else:
                        hashes={}
                        for j,index in enumerate(group):
                            data_path=column_dir/f'{index:06d}.npy'
                            atomic_array(data_path,columns[index])
                            hashes[str(index)]=hashlib.sha256(data_path.read_bytes()).hexdigest()
                            atomic_json(column_dir/f'{index:06d}.json',{
                                'mode':modal_list[index],'status':status,'scale':float(scales[j]),
                                'derived_by':'C6_packed_RHS','c6_batch':batch,
                                'strict_residual_measured_for_this_column':False})
                        # Receipt commits the whole packet. Stray column files
                        # after a crash are ignored until a receipt exists.
                        check['column_sha256']=hashes
                        atomic_json(out/'c6_batches'/f'{batch}.json',check)
                        saved=group
                update_c2_columns(saved)
                atomic_json(out/'progress.json',{'state':'solving','pid':os.getpid(),'mode_count':count,
                            'last_columns':saved,'last_attempted_columns':group,'rhs_kind':kind,
                            'pairs_this_run':len(timings),'remaining_worker_pairs':len(tasks)+len(inflight),'last_pair':status,
                            'elapsed_s':time.monotonic()-process_start})
                print(f'{kind} columns {group}: iterations={status["iterations"]}, solve={status["solve_s"]:.3f}s, '
                      f'strict residual={max(status["strict_residual_x"],status["strict_residual_y"]):.2e}',flush=True)
            complete=all(valid_column(i) for i in range(count))
            validation=None
            if complete:
                kernel=np.lib.format.open_memmap(out/'kernel.npy',mode='w+',dtype=np.complex128,shape=(count,count),fortran_order=True)
                for index in range(count):
                    kernel[:,index]=np.load(column_dir/f'{index:06d}.npy')
                kernel.flush()
                errors=[]; norms=[]
                # New incident directions excluded from the spherical projection.
                for i in range(3):
                    direction_index=projection_points+i
                    plane=np.r_[et[direction_index],ep[direction_index],directions[direction_index]].astype(np.float64)
                    plane_tolerance=args.adaptive_tight_tol or args.tol
                    send_operation(2,plane_tolerance); connection.sendall(memoryview(plane).cast('B'))
                    result(plane_tolerance)
                    observed=np.stack((np.einsum('ni,pni->np',et[projection_points:],far[:,projection_points:]),
                                       np.einsum('ni,pni->np',ep[projection_points:],far[:,projection_points:])),axis=1).reshape(-1,2)
                    predicted=check_basis@kernel@check_basis[2*i:2*i+2].conj().T
                    errors.append(np.linalg.norm(predicted-observed)**2); norms.append(np.linalg.norm(observed)**2)
                validation=float(np.sqrt(sum(errors)/sum(norms)))
                print(f'new-direction field error vs direct FMM: {validation:.3e}',flush=True)
            connection.sendall(struct.pack('=i',0)); connection.close()
            process.wait(timeout=60)
            if process.returncode:
                raise RuntimeError('FMM worker failed during shutdown')
        finally:
            if preparation is not None:preparation.close()
            connection.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
    total_seconds=time.monotonic()-startup
    all_status=[json.loads((column_dir/f'{i:06d}.json').read_text()) for i in range(count) if valid_column(i)]
    phase=np.array([1j**(l-p) for p,l,m in modal_list])
    column_physics=[]
    for index in requested:
        if not (column_dir/f'{index:06d}.json').exists():
            continue
        column=np.load(column_dir/f'{index:06d}.npy')
        t_column=-1j*args.ka/(4*np.pi)*phase*column*phase[index].conj()
        scattered=float(np.linalg.norm(t_column)**2)
        extinct=float(-t_column[index].real)
        check={'column':index,'mode':modal_list[index],
               'scattered_modal_power':scattered,'extinction_modal_power':extinct,
               'energy_balance_relative_error':abs(extinct-scattered)/max(scattered,1e-300)}
        if args.shape=='prism' and args.obj is None:
            mask=np.array([(mode[2]-modal_list[index][2])%6!=0 for mode in modal_list])
            check['sixfold_forbidden_mode_leakage']=float(np.linalg.norm(t_column[mask])/max(np.linalg.norm(t_column),1e-300))
        column_physics.append(check)
    report={'signature':signature,'parameters':fingerprint,'operator':hello,'mode_count':count,
            'completed_columns':len(all_status),'complete':complete,'elapsed_this_run_s':total_seconds,
            'requested_columns':len(requested),'requested_columns_complete':all(valid_column(i) for i in requested),
            'new_direction_field_error':validation,'pairs_this_run':timings,
            'maximum_strict_residual':max((max(d['status']['strict_residual_x'],d['status']['strict_residual_y']) for d in all_status),default=None),
            'dense_matrix_formed':False,'resident_solution_columns':hello.get('resident_solution_columns',2),
            'column_physics_checks':column_physics,'plane_control':plane_control}
    physical_scale={'wavelength_um':args.wavelength,
                    'volume_equivalent_radius_um':args.ka*args.wavelength/(2*np.pi) if args.wavelength else None,
                    'prism_height_to_width_across_flats':args.prism_aspect}
    if particle_info is not None:
        if not builtin_prism:physical_scale.pop('prism_height_to_width_across_flats')
        physical_scale.update(particle_name=particle_info['name'],
            particle_input_equivalent_radius=particle_info['equivalent_radius'],
            particle_input_to_um=(physical_scale['volume_equivalent_radius_um']/particle_info['equivalent_radius']
                                  if args.wavelength else None),
            body_origin_in_particle_input=particle_info['symmetry']['center'])
    report['physical_scale']=physical_scale
    if args.incident_min_order>1:
        requested_set=set(requested)
        report['incident_order_shell']={
            'minimum':args.incident_min_order,'maximum':args.order,'outgoing_order':args.order,
            'requested_columns':requested,
            'additional_control_columns':sorted(int(p.stem) for p in column_dir.glob('*.json')
                if int(p.stem) not in requested_set and valid_column(int(p.stem))),
            'is_complete_T_matrix':complete,'reciprocity_inferred_entries':0}
    if args.c2_reuse:
        report['symmetry_reuse']=symmetry_report
        report['independently_solved_columns']=sum('derived_by' not in item for item in all_status)
        report['reconstructed_columns']=sum(item.get('derived_by')=='C2x_half_turn' for item in all_status)
        report['maximum_strict_residual_applies_to']='independently_solved_columns_only'
    if args.c6_pack:
        audits=[]
        for group in c6_reference_groups:
            path=out/'c6_audits'/f'{tmatrix_c6.packet_id(group)}.json'
            if path.exists():
                audits.append(json.loads(path.read_text()))
        report['c6_packing']={
            'version':tmatrix_c6.VERSION,'phase_codes_per_packet':2,'maximum_columns_per_packet':12 if args.mirror_pack else 6,
            'horizontal_reflection_parity':args.mirror_pack,
            'vertical_reflection_from_C2_and_horizontal':bool(args.mirror_pack and args.c2_reuse),
            'shared_incident_cache':not args.no_incident_cache,
            'independent_reference_audits_complete':len(audits)==len(c6_reference_groups) and all(a['passed'] for a in audits),
            'individual_fallback_after_failed_audit':c6_individual_fallback,
            'packed_columns':sum(item.get('derived_by')=='C6_packed_RHS' for item in all_status),
            'individual_columns':sum('derived_by' not in item for item in all_status),
            'rejected_packets':len(list((out/'c6_rejections').glob('*.json'))),
            'packed_pairs_this_run':sum(t.get('rhs_kind') in ('packed','packed_rotation') for t in timings),
            'rotation_only_fallback_pairs_this_run':sum(t.get('rhs_kind')=='packed_rotation' for t in timings),
            'audit_pairs_this_run':sum(t.get('rhs_kind')=='audit' for t in timings),
            'individual_pairs_this_run':sum(t.get('rhs_kind')=='single' for t in timings),
            'individual_surface_residuals_measured_for_packed_columns':False}
        report['maximum_strict_residual_applies_to']='individual_and_combined_RHS; not_each_demultiplexed_column'
    if args.adaptive_tight_tol is not None:
        report['adaptive_tolerance']={
            'loose':args.tol,'tight':args.adaptive_tight_tol,'tight_from_degree':tight_from_degree,
            'independent_controls_always_tight':True,'audit_threshold_changed':False,
            'loose_attempts_this_run':sum(t['requested_tolerance']==args.tol for t in timings),
            'tight_attempts_this_run':sum(t['requested_tolerance']==args.adaptive_tight_tol for t in timings),
            'retries_recorded':len(list(adaptive_dir.glob('*.json')))}
    if complete:
        kernel=np.load(out/'kernel.npy',mmap_mode='r')
        t=np.lib.format.open_memmap(out/'tmatrix.npy',mode='w+',dtype=np.complex128,shape=(count,count),fortran_order=True)
        for column in range(count):
            t[:,column]=-1j*args.ka/(4*np.pi)*phase*kernel[:,column]*phase[column].conj()
        t.flush()
        power=sum(float(np.vdot(t[:,column],t[:,column]).real) for column in range(count))
        report['scattering_cross_section']=float(2*np.pi/args.ka**2*power)
        report['extinction_cross_section']=float(-2*np.pi/args.ka**2*np.trace(t).real)
        if not args.no_average:
            theta=np.linspace(0,180,args.ntheta)
            start=time.perf_counter()
            if args.average_backend=='mstm':
                import mmap
                kernel._mmap.madvise(mmap.MADV_DONTNEED)
                mean,post=analytic_mueller_tmatrix(t,args.order,theta,threads=args.so3_threads,
                    precision=args.so3_precision,cyclic_order=6 if (builtin_prism or symmetry_allowed.get('c6_z'))
                    and not c6_individual_fallback and (particle_info is None or args.particle_symmetry=='auto') else 1)
            else:
                mean=args.ka**2*analytic_mueller(kernel,args.order,theta)
                post={'method':'analytic_full_SO3_reference','Euler_quadrature_nodes':0}
            report['analytic_so3_s']=time.perf_counter()-start
            report['analytic_so3']=post
            atomic_json(out/'average.json',{'solver':'muller_fmm_tmatrix_so3','ka':args.ka,'order':args.order,
                                          'physical_scale':physical_scale,
                                          **post,'theta':theta.tolist(),'mueller':np.moveaxis(mean,0,-1).tolist()})
    report['end_to_end_s']=time.monotonic()-process_start
    atomic_json(out/'summary.json',report)
    atomic_json(out/'progress.json',{'state':'complete' if complete else 'partial','mode_count':count,'completed_columns':len(all_status),'elapsed_s':report['end_to_end_s']})
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs_this_run','column_physics_checks')},indent=2),flush=True)
    print(out/'summary.json',flush=True)


if __name__=='__main__':
    main()
