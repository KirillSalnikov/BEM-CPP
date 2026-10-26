#!/usr/bin/env python3
"""Checkpointed ka=20 prism build, with sampled convergence gates and a day budget."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import psutil
from bounded_bem_run import run, atomic_json
from tmatrix_reference import modes
from tmatrix_symmetry import c2x_plan
from tmatrix_c6 import reflection_audit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ('run_tmatrix_day.py', 'bounded_bem_run.py', 'build_fmm_tmatrix.py',
           'tmatrix_incident.py', 'tmatrix_projection.py', 'tmatrix_reference.py',
           'tmatrix_so3_mstm.py', 'tmatrix_symmetry.py', 'tmatrix_c6.py')
BINARIES = ('muller_tmatrix_fmm_server', 'tmatrix_so3_mstm', 'tmatrix_so3_mstm_fp32')
PROFILE = {
    'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '1',
    'BEM_FMM_MULTILEVEL': '1', 'BEM_FMM_MULTILEVEL_GPU_ASSEMBLY': '1',
    'BEM_MULLER_GPU_ASSEMBLY': '1', 'BEM_FMM_BANDED_SPLIT_DEPTH': '3',
    'BEM_FMM_BANDED_COARSE_MAX_LEAF': '1024', 'BEM_FMM_BANDED_MIDDLE_MAX_LEAF': '128',
    'BEM_TMATRIX_MAX_LEAF': '64', 'BEM_TMATRIX_MBJ_NODES': '50',
    'BEM_FMM_PAIR_CURRENTS': '0', 'BEM_FMM_PHASE_CACHE': '0',
    'BEM_FMM_FLAT_NEAR_SOURCES': '0', 'BEM_FMM_STRICT_PAIR_WORKSPACE': '0',
    'BEM_FMM_BANDED_MINIMAL_WORKSPACE': '1', 'BEM_FMM_CONCURRENT_MEDIA': '1',
}


def stamp(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def freeze_runtime(destination, source):
    """Pin a runtime once and verify it without modifying existing files."""
    if not destination.exists():
        temporary=destination.with_name(destination.name+'.preparing')
        for directory,names in (('scripts',SCRIPTS),('bin',BINARIES)):
            (temporary/directory).mkdir(parents=True,exist_ok=True)
            for name in names:
                shutil.copy2(source/directory/name,temporary/directory/name)
        atomic_json(temporary/'sha256.json',{
            str(p.relative_to(temporary)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in temporary.rglob('*') if p.is_file()})
        temporary.rename(destination)
    for name,digest in json.loads((destination/'sha256.json').read_text()).items():
        if hashlib.sha256((destination/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Frozen runtime was modified: {name}')


def sampled_comparison(base, other, indices, order_other):
    """Compare identical physical input modes and aligned output modes."""
    modal = modes(44)
    other_modes = modes(order_other)
    lookup = {mode: i for i, mode in enumerate(other_modes)}
    rows = np.array([lookup[mode] for mode in modal])
    extra = np.ones(len(other_modes), dtype=bool)
    extra[rows] = False
    records = []
    for index in indices:
        other_index = lookup[modal[index]]
        x = np.load(base / 'columns' / f'{index:06d}.npy')
        y = np.load(other / 'columns' / f'{other_index:06d}.npy')
        denominator = float(np.linalg.norm(y))
        difference = float(np.linalg.norm(x-y[rows]))
        tail = float(np.linalg.norm(y[extra]))
        records.append({'mode': modal[index], 'column': index,
                        'reference_norm': denominator, 'difference_norm': difference,
                        'tail_norm': tail,
                        'relative_difference': difference/max(denominator, 1e-300),
                        'relative_tail': tail/max(denominator, 1e-300)})
    norm = np.sqrt(sum(r['reference_norm']**2 for r in records))
    floor = 1e-4*max(r['reference_norm'] for r in records)
    active = [r for r in records if r['reference_norm'] >= floor]
    result = {'columns': records, 'sampled_only': True,
              'does_not_bound_unsampled_incoming_modes': True,
              'active_norm_floor_relative': 1e-4,
              'aggregate_relative_difference': float(np.sqrt(sum(r['difference_norm']**2 for r in records))/max(norm, 1e-300)),
              'aggregate_relative_tail': float(np.sqrt(sum(r['tail_norm']**2 for r in records))/max(norm, 1e-300)),
              'maximum_active_relative_difference': max(r['relative_difference'] for r in active),
              'maximum_active_relative_tail': max(r['relative_tail'] for r in active)}
    tolerance = .01 if order_other == 44 else .001
    result['tolerance'] = tolerance
    metrics = [result[key] for key in ('aggregate_relative_difference', 'aggregate_relative_tail',
                                      'maximum_active_relative_difference', 'maximum_active_relative_tail')]
    result['passed'] = bool(all(np.isfinite(v) and v <= tolerance for v in metrics))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=24.)
    parser.add_argument('--no-time-limit', action='store_true',
                        help='Run without a wall-clock deadline; retain numerical and resource gates')
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--c6-pack', action='store_true')
    parser.add_argument('--c6-audit-tol', type=float, default=1e-4)
    parser.add_argument('--mirror-pack', action='store_true')
    parser.add_argument('--correction-factor', type=float, default=.05)
    parser.add_argument('--duffy-order', type=int, default=4,
                        help='Near-singular integration order, shared by main and convergence controls')
    parser.add_argument('--seed-from', type=Path, help='Compatible stopped main T build to seed')
    parser.add_argument('--deadline-utc', help='Preserve an existing absolute deadline (ISO 8601 with timezone)')
    args = parser.parse_args()
    if not np.isfinite(args.hours) or args.hours <= 0:
        parser.error('--hours must be finite and positive')
    if not np.isfinite(args.tol) or args.tol <= 0:
        parser.error('--tol must be finite and positive')
    if not np.isfinite(args.c6_audit_tol) or args.c6_audit_tol <= 0:
        parser.error('--c6-audit-tol must be finite and positive')
    if args.mirror_pack and not args.c6_pack:
        parser.error('--mirror-pack requires --c6-pack')
    if not np.isfinite(args.correction_factor) or not .05 <= args.correction_factor <= .5:
        parser.error('--correction-factor must lie in [0.05, 0.5]')
    if not 2 <= args.duffy_order <= 12:
        parser.error('--duffy-order must lie in [2, 12]')
    if args.no_time_limit and args.deadline_utc:
        parser.error('--no-time-limit cannot be combined with --deadline-utc')
    absolute_deadline = None
    if args.deadline_utc:
        parsed_deadline = datetime.fromisoformat(args.deadline_utc)
        if parsed_deadline.tzinfo is None:
            parser.error('--deadline-utc requires a timezone')
        absolute_deadline = parsed_deadline.timestamp()
        if absolute_deadline <= time.time():
            parser.error('--deadline-utc has already passed')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = out / 'runtime'
        freeze_runtime(runtime,ROOT)
        resume_runner=runtime/'scripts/run_tmatrix_day.py'
        controller_digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if hashlib.sha256(resume_runner.read_bytes()).hexdigest()!=controller_digest:
            # Upgrade orchestration without changing the operator/basis files
            # that identify and validate already computed T columns.
            controller=out/'controllers'/controller_digest
            freeze_runtime(controller,ROOT)
            resume_runner=controller/'scripts/run_tmatrix_day.py'
        start = time.time()
        deadline = None if args.no_time_limit else (
            min(start + args.hours*3600, absolute_deadline) if absolute_deadline else start + args.hours*3600)
        attempt = out / 'supervision' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        attempt.mkdir(parents=True)
        (out / 'scratch').mkdir(exist_ok=True)
        env = {key: value for key, value in os.environ.items() if not key.startswith('BEM_')}
        env.update(PROFILE)
        if args.mirror_pack:
            env['BEM_TMATRIX_D6H_MESH']='1'
        if args.correction_factor!=.05:
            env['BEM_MIXED_CORRECTION_FACTOR']=str(args.correction_factor)
        if args.duffy_order!=4:
            env['BEM_TMATRIX_DUFFY_ORDER']=str(args.duffy_order)
        env['BEM_SO3_TMPDIR'] = str(out / 'scratch')
        base = out / 'main_ref5_L44'
        active_path = out.parent / 'active_bem_job.json'
        state = {'state': 'running', 'launcher_pid': os.getpid(), 'started_utc': stamp(start),
                 'deadline_utc': None if deadline is None else stamp(deadline),
                 'time_limit_enabled':deadline is not None,'resume_controller':str(resume_runner),
                 'out': str(out), 'ka': 20,
                 'ref': 5, 'order': 44, 'tol': args.tol, 'total_columns': 4048, 'independent_columns': 2078,
                 'checkpoint_directory': str(base / 'columns'),
                 'phase': 'preflight', 'supervision_attempt': str(attempt)}
        if args.c6_pack:
            state['individual_rhs_baseline'] = state.pop('independent_columns')
            state['c6_pack'] = True
            state['c6_phase_codes_per_packet'] = 2
            state['mirror_pack'] = args.mirror_pack

        def status(**updates):
            state.update(updates)
            state['updated_utc'] = stamp(time.time())
            atomic_json(out / 'status.json', state)
            atomic_json(active_path, state)
            print(json.dumps(state), flush=True)

        def stage(name, command, environment=env):
            remaining = None if deadline is None else deadline-time.time()
            if remaining is not None and remaining <= 0:
                status(state='time_limit', phase=name, reason='time budget exhausted; checkpoints retained')
                raise SystemExit(2)
            status(phase=name)
            result = run(command, attempt / name, env=environment, wall_limit_s=remaining)
            if result['state'] != 'complete':
                audit_path = base / 'c2_audit.json'
                audit_failure = (name == 'c2_pilot' and audit_path.exists()
                                 and not json.loads(audit_path.read_text()).get('passed'))
                status(state='validation_failed' if audit_failure else result['state'],
                       reason='C2 symmetry audit failed' if audit_failure else result.get('reason') or 'stage process failed',
                       stage_log=str(attempt / name / 'worker.log'))
                raise SystemExit(2)

        def command(destination, ref, order):
            extra = []
            if destination == base:
                if args.c6_pack:
                    extra += ['--c6-pack', '--c6-audit-tol', str(args.c6_audit_tol)]
                if args.mirror_pack:
                    extra += ['--mirror-pack']
                if args.seed_from:
                    extra += ['--seed-from', str(args.seed_from.resolve())]
            elif args.seed_from:
                previous = args.seed_from.resolve().parent / destination.name
                if (previous / 'manifest.json').exists():
                    extra += ['--seed-from', str(previous)]
            if args.seed_from:
                previous = (args.seed_from.resolve() if ref == 5 else
                            args.seed_from.resolve().parent / destination.name)
                if (previous / 'operator.near').exists() and (previous / 'operator.mbj').exists():
                    extra += ['--operator-cache-prefix', str(previous / 'operator')]
            return [sys.executable, str(runtime / 'scripts/build_fmm_tmatrix.py'),
                    '--shape', 'prism', '--ref', str(ref), '--ka', '20', '--ri', '1.3',
                    '--order', str(order), '--tol', str(args.tol), '--restart', '80', '--max-iters', '500',
                    '--fast-mlfma', '--ntheta', '181', '--so3-threads', '4', '--so3-precision', 'fp64',
                    '--out', str(destination)] + extra

        status()
        atomic_json(out / 'plan.json', {
            'geometry': 'regular hexagonal prism, aspect=1, equal-volume radius a=1',
            'ka': 20, 'refractive_index': [1.3, 0.], 'main_ref': 5, 'main_order': 44, 'solver_tolerance': args.tol,
            'c6_pack': args.c6_pack, 'c6_audit_tolerance': args.c6_audit_tol,
            'mirror_pack': args.mirror_pack, 'correction_factor': args.correction_factor,
            'duffy_order': args.duffy_order,
            'environment': {k: v for k, v in env.items() if k.startswith('BEM_') or k in PROFILE},
            'mesh_check': '10 sampled physical input modes at ref=5 and ref=6; 1% gate',
            'order_check': 'same modes at L=44 and L=48; common rows and outgoing tail; 0.1% gate',
            'sampled_checks_do_not_prove_global_convergence': True,
            'orientation_average': 'analytic full SO(3), all 16 Mueller elements, zero Euler quadrature nodes',
            'time_limit_hours_this_attempt': None if deadline is None else (deadline-start)/3600,
            'disk_reserve_gib': 40,
            'resume_command': [sys.executable, str(resume_runner), '--out', str(out),
                               '--tol', str(args.tol)] +
                              (['--c6-pack', '--c6-audit-tol', str(args.c6_audit_tol)] if args.c6_pack else []) +
                              (['--mirror-pack'] if args.mirror_pack else []) +
                              ['--correction-factor',str(args.correction_factor),
                               '--duffy-order',str(args.duffy_order)] +
                              (['--no-time-limit'] if deadline is None else
                               ['--hours',str(args.hours),'--deadline-utc',stamp(deadline)])})
        if shutil.disk_usage(out).free < 50*2**30 or psutil.virtual_memory().available < 8*2**30:
            status(state='failed', reason='insufficient initial RAM/disk reserve')
            return 1
        try:
            stage('c2_pilot', command(base, 5, 44)+['--c2-reuse', '--max-pairs-this-run',
                                                '32' if args.c6_pack else '10'])
            audit = json.loads((base / 'c2_audit.json').read_text())
            if not audit['passed']:
                raise RuntimeError('C2 symmetry audit failed')
            if args.c6_pack:
                pilot=json.loads((base/'summary.json').read_text())
                if not pilot.get('c6_packing',{}).get('independent_reference_audits_complete'):
                    raise RuntimeError('packed RHS reference audits did not complete in the pilot')
            if args.mirror_pack:
                independent={int(p.stem):np.load(p.with_suffix('.npy')) for p in (base/'columns').glob('*.json')
                             if 'derived_by' not in json.loads(p.read_text())}
                mirror_report=reflection_audit(modes(44),independent,args.c6_audit_tol)
                atomic_json(out/'mirror_check.json',mirror_report)
                if not mirror_report['passed']:
                    status(state='validation_failed',reason='horizontal/vertical reflection audit failed',
                           report=str(out/'mirror_check.json'))
                    return 3
            permutation, _, _, audits = c2x_plan(44)
            indices = [int(permutation[index]) for index in audits]
            fine = out / 'mesh_ref6_L44'
            fine_env = dict(env, BEM_FMM_BANDED_MIDDLE_MAX_LEAF='512',
                            BEM_FMM_BANDED_COARSE_MAX_LEAF='4096')
            stage('mesh_check', command(fine, 6, 44)+['--column-indices', *map(str, indices), '--no-average'], fine_env)
            mesh_report = sampled_comparison(base, fine, indices, 44)
            atomic_json(out / 'mesh_check.json', mesh_report)
            if not mesh_report['passed']:
                status(state='validation_failed', reason='sampled ref=5/ref=6 difference exceeds 1%', report=str(out / 'mesh_check.json'))
                return 3
            higher = out / 'order_ref5_L48'
            lookup = {mode: i for i, mode in enumerate(modes(48))}
            higher_indices = [lookup[modes(44)[index]] for index in indices]
            stage('order_check', command(higher, 5, 48)+['--column-indices', *map(str, higher_indices),
                  '--operator-cache-prefix', str((args.seed_from.resolve() if args.seed_from else base) / 'operator'), '--no-average'])
            order_report = sampled_comparison(base, higher, indices, 48)
            atomic_json(out / 'order_check.json', order_report)
            if not order_report['passed']:
                status(state='validation_failed', reason='sampled L=44/L=48 difference or outgoing tail exceeds 0.1%', report=str(out / 'order_check.json'))
                return 3
            stage('full_t_and_analytic_average', command(base, 5, 44)+['--c2-reuse'])
            result = json.loads((base / 'summary.json').read_text())
            average = json.loads((base / 'average.json').read_text())
            field_error = result.get('new_direction_field_error')
            passed = (result['complete'] and field_error is not None and np.isfinite(field_error)
                      and field_error <= .001 and average.get('Euler_quadrature_nodes') == 0
                      and np.isfinite(np.asarray(average['mueller'])).all())
            status(state='complete' if passed else 'validation_failed', phase='finished',
                   new_direction_field_error=field_error, summary=str(base / 'summary.json'),
                   average=str(base / 'average.json'), elapsed_s=time.time()-start)
            return 0 if passed else 3
        except Exception as error:
            status(state='failed', reason=str(error))
            raise


if __name__ == '__main__':
    raise SystemExit(main())
