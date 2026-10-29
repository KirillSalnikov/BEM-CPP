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
SCRIPTS = ('run_large_tmatrix.py', 'bounded_bem_run.py', 'build_fmm_tmatrix.py',
           'tmatrix_reference.py', 'tmatrix_incident.py', 'tmatrix_projection.py',
           'tmatrix_so3_mstm.py', 'tmatrix_symmetry.py', 'tmatrix_c6.py', 'study_tmatrix_cutoff.py', 'tmatrix_checkpoint_migration.py','tmatrix_pipeline.py')
BINARIES = ('muller_tmatrix_fmm_server', 'tmatrix_so3_mstm', 'tmatrix_so3_mstm_fp32')
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
        for directory, names in (('scripts', SCRIPTS), ('bin', BINARIES)):
            (temporary/directory).mkdir(exist_ok=True)
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
        durations = [p['total_worker_s']+p.get('incident_wait_s',p.get('incident_field_s', 0.)) for p in pairs]
        result['completed_pilot_pairs'] = len(durations)
        result['pilot_mean_pair_s'] = sum(durations)/len(durations)
        result['pilot_extrapolated_remaining_days'] = remaining*result['pilot_mean_pair_s']/86400
        result['pilot_limitation'] = 'initial independent modes; coded high-degree packets and fallbacks can cost differently'
    if 'last_pair' in progress:
        last = progress['last_pair']
        duration = last['total_worker_s']+last.get('incident_wait_s',last.get('incident_field_s', 0.))
        result['last_completed_pair_s'] = duration
        result['last_pair_extrapolated_remaining_days'] = remaining*duration/86400
    atomic_json(out/'eta.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ka', type=float, default=112.)
    parser.add_argument('--ref', type=int, default=7)
    parser.add_argument('--order', type=int, default=149)
    parser.add_argument('--ri', type=float, default=1.3)
    parser.add_argument('--prism-aspect', type=float, default=1.)
    parser.add_argument('--wavelength', type=float, help='Vacuum wavelength in micrometres')
    parser.add_argument('--restart', type=int, default=80)
    parser.add_argument('--strict-interval', type=int, default=4)
    parser.add_argument('--incident-prefetch', action='store_true')
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
    parser.add_argument('--sector-gmres', action='store_true', help='Independent C6/reflection Krylov sectors with orbit-local blocks')
    parser.add_argument('--orbit-nodes', type=int, default=50)
    parser.add_argument('--cutoff-orders', type=int, nargs='+', help='Analytic truncation controls after completion')
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not 1<=args.strict_interval<=16:parser.error('strict interval must be 1..16')
    if bool(args.solver_restart_from)!=bool(args.solver_restart_proof):
        parser.error('solver restart needs source and proof')
    if not 1<=args.orbit_nodes<=512 or args.sector_gmres and (args.wave_rank or args.fft_inner or args.mbj_overlap):
        parser.error('sector GMRES needs orbit blocks 1..512 and no wave/inner FFT/overlap')
    if not 0<=args.wave_rank<=512 or args.wave_rank%4 or args.mbj_overlap<0:
        parser.error('invalid wave rank or MBJ overlap')
    if not 0<=args.fft_inner<=32 or args.fft_inner and (args.wave_rank or args.circulant_blend):
        parser.error('nested FFT needs 1..32 steps and MBJ without wave/circulant correction')
    if not math.isfinite(args.fft_radius) or args.fft_radius<0:parser.error('invalid FFT radius')
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
    profile['BEM_HOST_GMRES_STRICT_INTERVAL']=str(args.strict_interval)
    if args.mbj_overlap>=int(profile['BEM_TMATRIX_MBJ_NODES']):parser.error('overlap must be smaller than MBJ block size')
    if args.mbj_overlap:profile['BEM_TMATRIX_MBJ_OVERLAP']=str(args.mbj_overlap)
    if args.wave_rank:profile['BEM_TMATRIX_WAVE_RANK']=str(args.wave_rank)
    if args.fft_inner:profile.update(BEM_TMATRIX_FFT_INNER=str(args.fft_inner),BEM_TMATRIX_FFT_RADIUS=str(args.fft_radius),BEM_PFFT_CORR_MEMORY=args.fft_memory)
    if args.circulant_blend: profile['BEM_TMATRIX_CIRCULANT_BLEND']=str(args.circulant_blend)
    if args.sector_gmres:profile.update(BEM_TMATRIX_SECTOR_GMRES='1',BEM_TMATRIX_ORBIT_NODES=str(args.orbit_nodes))
    env.update(profile, BEM_SO3_TMPDIR=str(out/'scratch'))
    counts = task_counts(args.order)
    t_gib = counts['mode_count']**2*16/2**30
    so3_fraction = .5 if args.order > 32 else 1.
    parameters = {key:getattr(args,key) for key in ('ka','ref','order','ri','tol','prism_aspect','wavelength','restart','max_iters','mbj_nodes','mbj_overlap','wave_rank','fft_inner','fft_radius','fft_memory','circulant_blend','cutoff_orders')}
    parameters.update(strict_interval=args.strict_interval,incident_prefetch=args.incident_prefetch,
                      worker_binary=str(args.worker_binary.resolve()) if args.worker_binary else None,
                      solver_restart_from=str(args.solver_restart_from.resolve()) if args.solver_restart_from else None,
                      solver_restart_proof=str(args.solver_restart_proof.resolve()) if args.solver_restart_proof else None)
    if args.sector_gmres:parameters.update(sector_gmres=True,orbit_nodes=args.orbit_nodes)
    command = [sys.executable, str(runtime/'scripts/build_fmm_tmatrix.py'),
        '--shape', 'prism', '--ref', str(args.ref), '--ka', str(args.ka), '--ri', str(args.ri),
        '--order', str(args.order), '--tol', str(args.tol), '--restart', str(args.restart), '--max-iters', str(args.max_iters),
        '--prism-aspect',str(args.prism_aspect),
        '--c2-reuse', '--c6-pack', '--mirror-pack', '--fast-mlfma',
        '--c2-audit-tol', '1e-4', '--c6-audit-tol', '1e-4',
        '--average-backend', 'mstm', '--so3-precision', 'auto', '--so3-threads', '4',
        '--ntheta', '7201', '--out', str(base)]
    if args.wavelength is not None: command+=['--wavelength',str(args.wavelength)]
    if args.incident_prefetch:command+=['--incident-prefetch']
    if args.solver_restart_from:
        command+=['--solver-restart-from',parameters['solver_restart_from'],'--solver-restart-proof',parameters['solver_restart_proof']]
    a_um=args.ka*args.wavelength/(2*math.pi) if args.wavelength else None
    width=(8*math.pi/(3*math.sqrt(3)*args.prism_aspect))**(1/3)
    physical={'volume_equivalent_radius_um':a_um,
              'width_across_flats_um':a_um*width if a_um else None,
              'height_um':a_um*width*args.prism_aspect if a_um else None}
    plan = {'created_utc': stamp(), 'parameters': parameters, 'profile': profile,'physical_dimensions':physical,
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
    if args.incident_prefetch:plan['resume_command']+=['--incident-prefetch']
    if args.sector_gmres:plan['resume_command']+=['--sector-gmres','--orbit-nodes',str(args.orbit_nodes)]
    if args.worker_binary:plan['resume_command']+=['--worker-binary',str(args.worker_binary.resolve())]
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
        stages=[('pilot',command+['--max-pairs-this-run','1']),('full',command)]
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
