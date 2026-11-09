#!/usr/bin/env python3
"""Launch a frozen, resumable large-prism T build and record live ETA scenarios.

The first independently solved pair is retained as a timing pilot. On success
the full symmetry-audited build continues automatically with the same columns.
No wall-clock deadline is imposed. Numerical and resource gates remain active.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time

import psutil
from bounded_bem_run import atomic_json, run
from tmatrix_reference import modes
from tmatrix_symmetry import c2x_plan
from tmatrix_c6 import audit_packets, packets, production_tasks

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ('run_large_tmatrix.py', 'bounded_bem_run.py', 'build_fmm_tmatrix.py', 'tmatrix_neural_prior.py',
           'tmatrix_reference.py', 'tmatrix_incident.py', 'tmatrix_projection.py',
           'tmatrix_so3_mstm.py', 'tmatrix_symmetry.py', 'tmatrix_c6.py', 'study_tmatrix_cutoff.py', 'select_tmatrix_order.py', 'tmatrix_checkpoint_migration.py','tmatrix_pipeline.py', 'tmatrix_order_shell.py', 'tmatrix_reciprocal_extension.py', 'complete_tmatrix_order_shell.py')
SCRIPTS += ('compressed_fft_service.py','symmetric_galerkin_fft.py','closed_galerkin_fft.py',
            'galerkin_cross_fft.py','streamed_galerkin_fft.py','tiled_galerkin_fft.py',
            'compressed_galerkin_fft.py','surface_symmetry.py','regular_surface_geometry.py')
SCRIPTS += ('mbs_particle_geometry.py',)
BINARIES = ('muller_tmatrix_fmm_server', 'tmatrix_so3_mstm', 'tmatrix_so3_mstm_fp32')
GEOMETRY_DATA = ('mbs_fixed_aggregate.json',)
PROFILE = {
    'OMP_NUM_THREADS': '16', 'OPENBLAS_NUM_THREADS': '1',
    'BEM_FMM_BANDED_SPLIT_DEPTH': '4', 'BEM_FMM_BANDED_COARSE_MAX_LEAF': '2048',
    'BEM_FMM_BANDED_MIDDLE_MAX_LEAF': '230',
    'BEM_FMM_BANDED_COARSE_ORDER_REFERENCE_DEPTH': '3',
    'BEM_FMM_ALLOW_DEPTH6': '1', 'BEM_TMATRIX_MAX_LEAF': '29',
    # Match the validated compact depth-six operator action. The worker's
    # implicit depth-five floor otherwise doubles the finest-band storage.
    'BEM_FMM_ORDER_REFERENCE_DEPTH': '6',
    'BEM_FMM_MULTILEVEL': '1', 'BEM_MULLER_GPU_ASSEMBLY': '0',
    'BEM_FMM_MULTILEVEL_GPU_ASSEMBLY': '0', 'BEM_TMATRIX_MBJ_NODES': '12',
    'BEM_FMM_PAIR_CURRENTS': '0', 'BEM_FMM_PHASE_CACHE': '0',
    'BEM_FMM_FLAT_NEAR_SOURCES': '0', 'BEM_FMM_STRICT_PAIR_WORKSPACE': '0',
    'BEM_FMM_L2P_FP32': '0', 'BEM_MIXED_ITERATIVE_REFINEMENT': '1',
    'BEM_FMM_BANDED_MINIMAL_WORKSPACE': '1', 'BEM_FMM_CONCURRENT_MEDIA': '1',
    'BEM_FMM_MULTILEVEL_COMPACT_TEMP': '1', 'BEM_FMM_P2P_SHARED_SOURCES': '1',
    # Full ka=100/ref=7 ABBA control: 4.70 -> 4.36 s/operator action.
    # Keep the established 32 leaf partitions; change launch occupancy only.
    'BEM_FMM_P2P_THREADS': '128',
    'BEM_FMM_P2P_SHARED_REAL': '1',
    # Share source loads and leaf-local coordinate conversion across P2M
    # directions. Only the mixed P2M action selects this kernel.
    'BEM_FMM_MULTILEVEL_P2M_SHARED': '1',
    # Independent M2L direction blocks preserve the source summation order.
    'BEM_FMM_M2L_DIRECTION_BLOCKS': '1',
    'BEM_TMATRIX_D6H_MESH': '1', 'BEM_TMATRIX_DUFFY_ORDER': '6',
    'BEM_TMATRIX_SYMMETRY_WARM_START': '1',
    # Carry the inner Arnoldi residual through unproductive restart boundaries.
    # Target/limit exits and every fourth cycle still use the full FP64 action.
    'BEM_HOST_GMRES_STRICT_INTERVAL': '4',
}


def solver_profile(ref, mbj_nodes=None):
    profile = dict(PROFILE)
    if ref < 7:
        profile.update(BEM_FMM_BANDED_SPLIT_DEPTH=str(max(1, ref-2)),
                       BEM_FMM_BANDED_COARSE_MAX_LEAF='3696',
                       BEM_FMM_BANDED_MIDDLE_MAX_LEAF='462',
                       BEM_FMM_ORDER_REFERENCE_DEPTH=str(ref),
                       BEM_TMATRIX_MAX_LEAF='64', BEM_TMATRIX_MBJ_NODES='50')
    if mbj_nodes is not None:
        profile['BEM_TMATRIX_MBJ_NODES'] = str(mbj_nodes)
    return profile


def stamp():
    return datetime.now(timezone.utc).isoformat()


def frozen_runtime(destination, source=ROOT, worker_binary=None):
    if not destination.exists():
        temporary = destination.with_name(destination.name+'.preparing')
        temporary.mkdir(parents=True, exist_ok=True)
        for directory, names in (('scripts', SCRIPTS), ('bin', BINARIES),('reference/particles',GEOMETRY_DATA)):
            (temporary/directory).mkdir(parents=True,exist_ok=True)
            for name in names:
                src=worker_binary if worker_binary is not None and name=='muller_tmatrix_fmm_server' else source/directory/name
                shutil.copy2(src, temporary/directory/name)
        hashes = {str(p.relative_to(temporary)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in temporary.rglob('*') if p.is_file()}
        atomic_json(temporary/'sha256.json', hashes)
        temporary.rename(destination)
    for name, expected in json.loads((destination/'sha256.json').read_text()).items():
        if hashlib.sha256((destination/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'frozen runtime changed: {name}')
    if worker_binary is not None and hashlib.sha256(worker_binary.read_bytes()).digest()!=hashlib.sha256((destination/'bin/muller_tmatrix_fmm_server').read_bytes()).digest():
        raise RuntimeError('requested worker differs from frozen runtime')


def task_counts(order):
    modal = modes(order)
    permutation, _, representatives, audits = c2x_plan(order)
    audit_work = [i for negative in audits for i in (int(permutation[negative]), negative)]
    groups = audit_packets(modal, order, True)
    controls = list(dict.fromkeys(audit_work+[i for group in groups for i in group]))
    work = list(dict.fromkeys(audit_work+representatives))
    control_set = set(controls)
    production = production_tasks(modal, [i for i in work if i not in control_set], (), True)
    return {'mode_count': len(modal), 'C2_representatives': len(representatives),
            'ideal_D6h_packets': len(packets(modal, representatives, True)),
            'independent_control_pairs': (len(controls)+1)//2,
            'coded_audit_pairs': len(groups), 'production_pairs': len(production),
            'initial_worker_pairs_including_audits': (len(controls)+1)//2+len(groups)+len(production)}


def read_json(path):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return {}


def iteration_sample(log):
    try:
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-512*1024))
            lines = stream.read().decode(errors='replace')
    except OSError:
        return {}
    samples = []
    for match in re.finditer(r'\[T (?:host|sector) GMRES (\d+)\] projected ([\deE+.\-]+) elapsed_s=([\d.]+)', lines):
        point = (int(match[1]), float(match[2]), float(match[3]))
        if samples and point[0] <= samples[-1][0]: samples = []
        samples.append(point)
    if not samples: return {}
    window = samples[-12:]
    first, last = window[0], window[-1]
    seconds = ((last[2]-first[2])/(last[0]-first[0]) if last[0] > first[0]
               else last[2]/last[0])
    return {'current_rhs_iteration': last[0], 'projected_residual': last[1],
            'current_rhs_elapsed_s': last[2], 'seconds_per_iteration_recent': seconds,
            'iteration_intervals_measured': max(0, len(window)-1)}


def estimate(out, base, counts, phase):
    progress = read_json(base/'progress.json')
    sample = iteration_sample(base/'worker.log')
    hybrid=int(read_json(base/'operator_metadata.json').get('hybrid_batch_capacity',1))>1
    if hybrid:sample={}  # Interleaved lanes do not define one iteration clock.
    total = counts['initial_worker_pairs_including_audits']
    remaining = progress.get('remaining_worker_pairs', progress.get('planned_worker_pairs', total))
    if progress.get('state') == 'complete': remaining = 0
    seconds = sample.get('seconds_per_iteration_recent')
    result = {'updated_utc': stamp(), 'phase': phase, 'progress': progress,
              'iteration_sample': sample,
              'rate_source': 'current_host_GMRES' if sample else 'waiting_for_current_run_measurement',
              'remaining_worker_pairs_before_future_fallback': remaining,
              'scenario_days_excluding_setup_field_preparation_and_postprocessing': {
                  str(iterations): (0 if remaining==0 else remaining*2*iterations*seconds/86400 if seconds is not None else None)
                  for iterations in (50, 100, 200, 500)},
              'scenario_definition': 'assumed iterations per RHS; two sequential RHS per host-worker pair',
              'not_a_convergence_or_finish_time_guarantee': True}
    pilot = read_json(base/'summary.json')
    pairs = pilot.get('pairs_this_run', [])
    if pairs:
        durations = [p.get('hybrid_accounted_worker_s',p['total_worker_s'])+p.get('incident_wait_s',p.get('incident_field_s', 0.)) for p in pairs]
        result['completed_pilot_pairs'] = len(durations)
        result['pilot_mean_pair_s'] = sum(durations)/len(durations)
        result['pilot_extrapolated_remaining_days'] = remaining*result['pilot_mean_pair_s']/86400
        result['pilot_limitation'] = 'initial independent modes; coded high-degree packets and fallbacks can cost differently'
    if 'last_pair' in progress:
        last = progress['last_pair']
        duration = last.get('hybrid_accounted_worker_s',last['total_worker_s'])+last.get('incident_wait_s',last.get('incident_field_s', 0.))
        result['last_completed_pair_s'] = duration
        result['last_pair_extrapolated_remaining_days'] = remaining*duration/86400
    atomic_json(out/'eta.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ka', type=float, default=112.)
    parser.add_argument('--ref', type=int, default=7)
    parser.add_argument('--order', type=int, default=149)
    parser.add_argument('--order-study',type=Path,help='Select guarded L from complete same-task analytic truncation controls')
    parser.add_argument('--order-error-budget',type=float,default=.01)
    parser.add_argument('--order-guard',type=int,default=2)
    parser.add_argument('--angular-band', type=int, default=0,
                        help='T projection latitude nodes; 0 retains 2*L+4. Reduced grids require convergence checks.')
    parser.add_argument('--ri', type=float, default=1.3)
    parser.add_argument('--prism-aspect', type=float, default=1.)
    parser.add_argument('--wavelength', type=float, help='Vacuum wavelength in micrometres')
    parser.add_argument('--restart', type=int, default=80)
    parser.add_argument('--strict-interval', type=int, default=4)
    parser.add_argument('--incident-prefetch', action='store_true')
    parser.add_argument('--hybrid-cpu', action='store_true', help='Overlap two T packets with independent CPU FP64 gates')
    parser.add_argument('--gpu-assembly', action='store_true', help='Assemble operator actions on GPU, including with host sector GMRES')
    parser.add_argument('--gpu-host-correction', action='store_true', help='Keep local corrections in host RAM while using GPU assembly; requires a host solver')
    parser.add_argument('--strict-face-fft', action='store_true', help='Experimental face FFT for FP64 residual actions; mixed actions remain FMM')
    parser.add_argument('--worker-binary',type=Path,help='Freeze this validated worker instead of the current build')
    parser.add_argument('--max-iters', type=int, default=1000)
    parser.add_argument('--mbj-nodes', type=int)
    parser.add_argument('--mbj-overlap', type=int, default=0)
    parser.add_argument('--wave-rank', type=int, default=0)
    parser.add_argument('--fft-inner', type=int, default=0)
    parser.add_argument('--fft-radius', type=float, default=1.)
    parser.add_argument('--fft-memory', choices=('device','host'), default='device')
    parser.add_argument('--solver-restart-from', type=Path)
    parser.add_argument('--solver-restart-proof', type=Path)
    parser.add_argument('--circulant-blend', type=float, default=0.)
    parser.add_argument('--opposite-face-weight',type=float,default=0.,help='Weight of opposite-face FFT coupling, 0 disables')
    parser.add_argument('--opposite-face-mask',type=int,default=3,help='Coupled FFT banks: 1 sides, 2 caps, 3 both')
    parser.add_argument('--sector-gmres', action='store_true', help='Independent C6/reflection Krylov sectors with orbit-local blocks')
    parser.add_argument('--orbit-nodes', type=int, default=50)
    parser.add_argument('--sector-recycle-rank',type=int,default=0,help='RAM-only recycled directions per symmetry sector')
    parser.add_argument('--sector-recycle-mask',type=int,default=4095,help='Bit mask of sectors eligible for recycling')
    parser.add_argument('--vertical-preconditioner',action='store_true',help='Complete the inverse symmetry to D6h')
    parser.add_argument('--inner-local-coordinates',action='store_true',help='Use local coordinates in mixed near interactions')
    parser.add_argument('--sector-projected-factor',type=float,default=.5)
    parser.add_argument('--sector-first-pass-factor',type=float,default=0,help='Coarser first inner target; zero preserves the original target')
    parser.add_argument('--cutoff-orders', type=int, nargs='+', help='Analytic truncation controls after completion')
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--adaptive-tight-tol',type=float,help='Opt-in strict controls and tighter retries for C6 packets')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    order_evidence=None
    if args.order_study:
        from select_tmatrix_order import audit_and_select,validate_target
        try:
            order_evidence=audit_and_select(args.order_study,args.order_error_budget,args.order_guard)
            args.order=validate_target(order_evidence,dict(shape='prism',ka=args.ka,ri=args.ri,ri_imag=0.,
                prism_aspect=args.prism_aspect,wavelength=args.wavelength,ref=args.ref,geometry_digest=None))
        except (ValueError,AssertionError) as error:parser.error(str(error))
    if args.hybrid_cpu and not args.sector_gmres:parser.error('--hybrid-cpu requires --sector-gmres')
    if args.angular_band and args.angular_band<=args.order:
        parser.error('angular band must exceed the T order, or be zero for the default')
    if not 1<=args.strict_interval<=16:parser.error('strict interval must be 1..16')
    if not 0<=args.opposite_face_weight<=1 or not 1<=args.opposite_face_mask<=3 or args.opposite_face_weight and not args.circulant_blend:
        parser.error('opposite-face coupling needs weight in [0,1], mask 1..3 and a circulant inverse')
    if bool(args.solver_restart_from)!=bool(args.solver_restart_proof):
        parser.error('solver restart needs source and proof')
    if not 1<=args.orbit_nodes<=512 or args.sector_gmres and (args.wave_rank or args.fft_inner or args.mbj_overlap):
        parser.error('sector GMRES needs orbit blocks 1..512 and no wave/inner FFT/overlap')
    if not 0<=args.sector_recycle_rank<=64 or not 0<=args.sector_recycle_mask<=4095 or args.sector_recycle_rank and not args.sector_gmres:
        parser.error('sector recycling needs sector GMRES, rank 0..64 and mask 0..4095')
    if not 0<args.sector_projected_factor<=.5 or (args.vertical_preconditioner or args.sector_projected_factor!=.5) and not args.sector_gmres:
        parser.error('vertical inverse and projected factor (0,.5] require sector GMRES')
    if not 0<=args.sector_first_pass_factor<=64 or args.sector_first_pass_factor and not args.sector_gmres:
        parser.error('first pass factor requires sector GMRES and a value in [0,64]')
    if not 0<=args.wave_rank<=512 or args.wave_rank%4 or args.mbj_overlap<0:
        parser.error('invalid wave rank or MBJ overlap')
    if not 0<=args.fft_inner<=32 or args.fft_inner and (args.wave_rank or args.circulant_blend):
        parser.error('nested FFT needs 1..32 steps and MBJ without wave/circulant correction')
    if not math.isfinite(args.fft_radius) or args.fft_radius<0:parser.error('invalid FFT radius')
    if args.adaptive_tight_tol is not None and (not math.isfinite(args.adaptive_tight_tol)
                                              or not 0<args.adaptive_tight_tol<=args.tol):
        parser.error('adaptive tight tolerance must be finite, positive and <= tol')
    if args.ka <= 0 or args.ref < 0 or not 1 <= args.order <= 256 or args.tol <= 0:
        parser.error('invalid numerical parameters')
    if (not all(math.isfinite(x) and x>0 for x in (args.ka,args.ri,args.prism_aspect,args.tol))
            or args.wavelength is not None and (not math.isfinite(args.wavelength) or args.wavelength<=0)
            or not 2<=args.restart<=256 or args.max_iters<1
            or not math.isfinite(args.circulant_blend) or not 0<=args.circulant_blend<=1
            or args.mbj_nodes is not None and not 1<=args.mbj_nodes<=256):
        parser.error('invalid geometry, wavelength or solver parameters')
    if args.cutoff_orders and any(not 1<=order<=args.order for order in args.cutoff_orders):
        parser.error('cutoff controls must be between one and the full order')
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (out/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runtime = out/'runtime'; frozen_runtime(runtime,worker_binary=args.worker_binary.resolve() if args.worker_binary else None)
    base = out/f'main_ref{args.ref}_L{args.order}'
    env = {k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
    profile=solver_profile(args.ref,args.mbj_nodes)
    if args.hybrid_cpu:
        profile.update(BEM_TMATRIX_HYBRID='1',BEM_TMATRIX_CPU_STRICT='1')
        profile.update(OMP_WAIT_POLICY=os.environ.get('OMP_WAIT_POLICY','PASSIVE'),
                       GOMP_SPINCOUNT=os.environ.get('GOMP_SPINCOUNT','0'))
        profile.update({k:v for k,v in os.environ.items() if k.startswith('BEM_CPU_') or k=='BEM_CIRCULANT_CPU_FFT'})
    if args.gpu_assembly or args.strict_face_fft or args.gpu_host_correction:
        profile.update(BEM_MULLER_GPU_ASSEMBLY='1', BEM_FMM_MULTILEVEL_GPU_ASSEMBLY='1')
    if args.gpu_host_correction:
        profile['BEM_MULLER_GPU_HOST_CORRECTION']='1'
    if args.strict_face_fft:
        profile.update(BEM_TMATRIX_FACE_FFT='1', BEM_FACE_FFT_STRICT_ONLY='1',
                       BEM_FACE_FFT_NEAR_CACHE='1', BEM_FACE_FFT_TARGET_FACES='1',
                       BEM_FACE_FFT_REUSE_CROSS='1', BEM_FMM_PRUNE_EMPTY='1', BEM_MULLER_GPU_STRICT_ONLY='1',
                       BEM_FACE_FFT_SMOOTH_PADDING='1', BEM_FACE_FFT_DEDUP_KERNELS='1')
    profile['BEM_HOST_GMRES_STRICT_INTERVAL']=str(args.strict_interval)
    if args.mbj_overlap>=int(profile['BEM_TMATRIX_MBJ_NODES']):parser.error('overlap must be smaller than MBJ block size')
    if args.mbj_overlap:profile['BEM_TMATRIX_MBJ_OVERLAP']=str(args.mbj_overlap)
    if args.wave_rank:profile['BEM_TMATRIX_WAVE_RANK']=str(args.wave_rank)
    if args.fft_inner:profile.update(BEM_TMATRIX_FFT_INNER=str(args.fft_inner),BEM_TMATRIX_FFT_RADIUS=str(args.fft_radius),BEM_PFFT_CORR_MEMORY=args.fft_memory)
    if args.circulant_blend: profile['BEM_TMATRIX_CIRCULANT_BLEND']=str(args.circulant_blend)
    if args.opposite_face_weight:profile.update(BEM_CIRCULANT_OPPOSITE_WEIGHT=str(args.opposite_face_weight),BEM_CIRCULANT_OPPOSITE_MASK=str(args.opposite_face_mask))
    if args.sector_gmres:profile.update(BEM_TMATRIX_SECTOR_GMRES='1',BEM_TMATRIX_ORBIT_NODES=str(args.orbit_nodes))
    if args.sector_recycle_rank:profile.update(BEM_TMATRIX_SECTOR_RECYCLE_RANK=str(args.sector_recycle_rank),BEM_TMATRIX_SECTOR_RECYCLE_MASK=str(args.sector_recycle_mask))
    if args.vertical_preconditioner:profile['BEM_TMATRIX_VERTICAL_PRECONDITIONER']='1'
    if args.inner_local_coordinates:profile['BEM_FMM_P2P_LOCAL_COORDINATES']='1'
    if args.sector_projected_factor!=.5:profile['BEM_SECTOR_PROJECTED_FACTOR']=str(args.sector_projected_factor)
    if args.sector_first_pass_factor:profile['BEM_SECTOR_FIRST_PASS_FACTOR']=str(args.sector_first_pass_factor)
    env.update(profile, BEM_SO3_TMPDIR=str(out/'scratch'))
    counts = task_counts(args.order)
    t_gib = counts['mode_count']**2*16/2**30
    so3_fraction = .5 if args.order > 32 else 1.
    parameters = {key:getattr(args,key) for key in ('ka','ref','order','ri','tol','prism_aspect','wavelength','restart','max_iters','mbj_nodes','mbj_overlap','wave_rank','fft_inner','fft_radius','fft_memory','circulant_blend','cutoff_orders')}
    parameters.update(strict_interval=args.strict_interval,incident_prefetch=args.incident_prefetch,
                      angular_band=args.angular_band,gpu_assembly=args.gpu_assembly,strict_face_fft=args.strict_face_fft,
                      gpu_host_correction=args.gpu_host_correction,
                      worker_binary=str(args.worker_binary.resolve()) if args.worker_binary else None,
                      solver_restart_from=str(args.solver_restart_from.resolve()) if args.solver_restart_from else None,
                      solver_restart_proof=str(args.solver_restart_proof.resolve()) if args.solver_restart_proof else None)
    if args.sector_gmres:parameters.update(sector_gmres=True,orbit_nodes=args.orbit_nodes)
    if args.hybrid_cpu:parameters['hybrid_cpu']=True
    if args.opposite_face_weight:parameters.update(opposite_face_weight=args.opposite_face_weight,opposite_face_mask=args.opposite_face_mask)
    if args.sector_recycle_rank:parameters.update(sector_recycle_rank=args.sector_recycle_rank,sector_recycle_mask=args.sector_recycle_mask)
    if args.vertical_preconditioner:parameters['vertical_preconditioner']=True
    if args.inner_local_coordinates:parameters['inner_local_coordinates']=True
    if args.sector_projected_factor!=.5:parameters['sector_projected_factor']=args.sector_projected_factor
    if args.sector_first_pass_factor:parameters['sector_first_pass_factor']=args.sector_first_pass_factor
    command = [sys.executable, str(runtime/'scripts/build_fmm_tmatrix.py'),
        '--shape', 'prism', '--ref', str(args.ref), '--ka', str(args.ka), '--ri', str(args.ri),
        '--order', str(args.order), '--tol', str(args.tol), '--restart', str(args.restart), '--max-iters', str(args.max_iters),
        '--prism-aspect',str(args.prism_aspect),
        '--c2-reuse', '--c6-pack', '--mirror-pack', '--fast-mlfma',
        '--c2-audit-tol', '1e-4', '--c6-audit-tol', '1e-4',
        '--average-backend', 'mstm', '--so3-precision', 'auto', '--so3-threads', '4',
        '--ntheta', '7201', '--out', str(base)]
    if args.wavelength is not None: command+=['--wavelength',str(args.wavelength)]
    if args.angular_band: command+=['--angular-band',str(args.angular_band)]
    if args.incident_prefetch:command+=['--incident-prefetch']
    if args.hybrid_cpu:command+=['--hybrid-cpu']
    if args.adaptive_tight_tol is not None:
        command+=['--adaptive-tight-tol',str(args.adaptive_tight_tol)]
        parameters['adaptive_tight_tol']=args.adaptive_tight_tol
    if args.solver_restart_from:
        command+=['--solver-restart-from',parameters['solver_restart_from'],'--solver-restart-proof',parameters['solver_restart_proof']]
    a_um=args.ka*args.wavelength/(2*math.pi) if args.wavelength else None
    width=(8*math.pi/(3*math.sqrt(3)*args.prism_aspect))**(1/3)
    physical={'volume_equivalent_radius_um':a_um,
              'width_across_flats_um':a_um*width if a_um else None,
              'height_um':a_um*width*args.prism_aspect if a_um else None}
    plan = {'created_utc': stamp(), 'parameters': parameters, 'profile': profile,'physical_dimensions':physical,
        'order_evidence': order_evidence,
        **counts, 'one_T_gib': t_gib, 'columns_kernel_T_and_SO3_temp_gib': (3+so3_fraction)*t_gib,
        'extra_operator_cache_estimate_gib': 10, 'disk_reserve_gib': 40,
        'time_limit': None, 'orientation_average': 'full analytic SO3, all 16 Mueller entries, zero Euler nodes',
        'cutoff_and_mesh_1pct_convergence': 'requires convergence controls for this size and aspect; linear residual is a separate gate',
        'command': command,
        'resume_command': [sys.executable, str(runtime/'scripts/run_large_tmatrix.py'), '--out', str(out),
            '--ka',str(args.ka),'--ref',str(args.ref),'--order',str(args.order),'--ri',str(args.ri),'--tol',str(args.tol),
            '--prism-aspect',str(args.prism_aspect),'--restart',str(args.restart),'--max-iters',str(args.max_iters)]
            + (['--wavelength',str(args.wavelength)] if args.wavelength else [])
            + (['--circulant-blend',str(args.circulant_blend)] if args.circulant_blend else [])
            + (['--cutoff-orders']+list(map(str,args.cutoff_orders)) if args.cutoff_orders else [])
            + (['--mbj-nodes',str(args.mbj_nodes)] if args.mbj_nodes else [])}
    plan['resume_command']+=['--wave-rank',str(args.wave_rank),'--mbj-overlap',str(args.mbj_overlap),'--fft-inner',str(args.fft_inner),'--fft-radius',str(args.fft_radius),'--fft-memory',args.fft_memory]
    plan['resume_command']+=['--strict-interval',str(args.strict_interval)]
    if args.opposite_face_weight:plan['resume_command']+=['--opposite-face-weight',str(args.opposite_face_weight),'--opposite-face-mask',str(args.opposite_face_mask)]
    if args.incident_prefetch:plan['resume_command']+=['--incident-prefetch']
    if args.hybrid_cpu:plan['resume_command']+=['--hybrid-cpu']
    if args.adaptive_tight_tol is not None:plan['resume_command']+=['--adaptive-tight-tol',str(args.adaptive_tight_tol)]
    if args.angular_band:plan['resume_command']+=['--angular-band',str(args.angular_band)]
    if args.gpu_assembly:plan['resume_command']+=['--gpu-assembly']
    if args.gpu_host_correction:plan['resume_command']+=['--gpu-host-correction']
    if args.strict_face_fft:plan['resume_command']+=['--strict-face-fft']
    if args.sector_gmres:plan['resume_command']+=['--sector-gmres','--orbit-nodes',str(args.orbit_nodes)]
    if args.sector_recycle_rank:plan['resume_command']+=['--sector-recycle-rank',str(args.sector_recycle_rank),'--sector-recycle-mask',str(args.sector_recycle_mask)]
    if args.vertical_preconditioner:plan['resume_command']+=['--vertical-preconditioner']
    if args.inner_local_coordinates:plan['resume_command']+=['--inner-local-coordinates']
    if args.sector_projected_factor!=.5:plan['resume_command']+=['--sector-projected-factor',str(args.sector_projected_factor)]
    if args.sector_first_pass_factor:plan['resume_command']+=['--sector-first-pass-factor',str(args.sector_first_pass_factor)]
    if args.worker_binary:plan['resume_command']+=['--worker-binary',str(args.worker_binary.resolve())]
    if args.order_study:plan['resume_command']+=['--order-study',str(args.order_study.resolve()),
        '--order-error-budget',str(args.order_error_budget),'--order-guard',str(args.order_guard)]
    if args.solver_restart_from:
        plan['resume_command']+=['--solver-restart-from',parameters['solver_restart_from'],'--solver-restart-proof',parameters['solver_restart_proof']]
    previous = read_json(out/'plan.json')
    if previous and (previous['parameters'] != parameters or previous['profile'] != profile):
        raise RuntimeError('resume parameters changed; use a new run directory')
    if not previous: atomic_json(out/'plan.json', plan)
    if shutil.disk_usage(out).free/2**30 < (3+so3_fraction)*t_gib+50:
        raise RuntimeError('insufficient disk space including caches and reserve')
    if psutil.virtual_memory().available < 32*2**30:
        raise RuntimeError('large host-GMRES profile requires at least 32 GiB available initially')
    if args.prepare_only:
        estimate(out, base, counts, 'prepared')
        print(json.dumps(plan, indent=2)); lock.close(); return 0
    (out/'supervisor.pid').write_text(str(os.getpid())+'\n')
    stop = threading.Event(); active = {'phase':'starting'}
    def monitor():
        while not stop.is_set():
            try: estimate(out, base, counts, active['phase'])
            except (OSError, ValueError, KeyError) as error:
                print(f'ETA monitor: {error}', flush=True)
            stop.wait(15)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    try:
        # A one-pair pilot can end halfway through a coded audit, discarding
        # its reference solution and the worker's recycle space. Continuations
        # already have a checkpoint (or a solver-migration proof checked by
        # the builder); let one persistent worker finish all required audits.
        has_checkpoint=(base/'manifest.json').is_file() and any((base/'columns').glob('*.npy'))
        stages=[] if args.solver_restart_from or has_checkpoint else [('pilot',command+['--max-pairs-this-run','1'])]
        stages.append(('full',command))
        if args.cutoff_orders:
            stages.append(('cutoff_controls',[sys.executable,str(runtime/'scripts/study_tmatrix_cutoff.py'),
                '--run',str(base),'--out',str(out/'cutoff_controls'),'--ntheta','7201','--threads','4',
                '--orders']+list(map(str,args.cutoff_orders))))
        for phase, stage_command in stages:
            active['phase'] = phase
            atomic_json(out/'status.json', {'state':'running', 'phase':phase,
                'updated_utc':stamp(), 'supervisor_pid':os.getpid(), 'main_output':str(base)})
            result = run(stage_command, out/f'supervisor_{phase}', env=env,
                         rss_limit_gib=48, available_gib=6, disk_reserve_gib=40, wall_limit_s=None)
            if result['state'] != 'complete':
                active['phase'] = 'failed'
                atomic_json(out/'status.json', {'state':'failed','phase':phase,'updated_utc':stamp(),'stage':result})
                return 1
        active['phase'] = 'complete'
        atomic_json(out/'status.json', {'state':'complete','updated_utc':stamp(),'main_output':str(base)})
        return 0
    finally:
        stop.set(); thread.join(timeout=10)
        estimate(out, base, counts, active['phase'])
        lock.close()


if __name__ == '__main__': raise SystemExit(main())
