#!/usr/bin/env python3
"""Queued absorbing-prism BEM/MBS comparison, with explicit numerical gates.

Wait for the existing size series, compare dense/FMM actions through the
same plane-wave solves, refine the mesh, then build T and average analytically.
The four plane-wave mesh controls are sampled evidence, not an SO3 error proof.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
from absorbing_fixed_comparison import BETAS, THETA, difference, far_to_mueller, frames, mbs_fixed, plot
from bounded_bem_run import atomic_json, run
from build_fmm_tmatrix import receive_into, receive_message
from run_tmatrix_neighbors import Pipeline, digest, read
from study_tmatrix_cutoff import analyze

RUNTIME = Path(__file__).resolve().parents[1]


def profile(plan, ref, duffy):
    result = dict(plan['gpu_profile'])
    result.update(BEM_FMM_BANDED_SPLIT_DEPTH=str(max(1, ref-2)),
                  BEM_FMM_ORDER_REFERENCE_DEPTH=str(ref),
                  BEM_TMATRIX_DUFFY_ORDER=str(duffy),
                  BEM_TMATRIX_PRISM_ASPECT=str(plan['prism_aspect']))
    return result


def fixed(out, ref, duffy):
    plan = read(out/'plan.json')
    case = out/'gpu_fixed'/f'ref{ref}_d{duffy}'
    case.mkdir(parents=True, exist_ok=True)
    incoming, directions, _ = frames()
    directions.reshape(-1, 3).tofile(case/'directions.bin')
    env = {k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
    env.update(profile(plan, ref, duffy))
    results, fields = [], []
    with tempfile.TemporaryDirectory(prefix='abs-bem-') as temp, (case/'worker.log').open('a') as log:
        address = str(Path(temp)/'worker.sock')
        command = [str(RUNTIME/'bin/muller_tmatrix_fmm_server'), address, 'prism', str(ref),
                   str(plan['ka']), str(plan['ri']), str(plan['ri_imag']), str(case/'directions.bin'),
                   str(case/'operator'), str(plan['bem_residual_tolerance']), '64', '1000']
        atomic_json(case/'command.json', dict(command=command, environment=profile(plan, ref, duffy)))
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            while True:
                try:
                    connection.connect(address)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if process.poll() is not None:
                        raise RuntimeError('absorbing FMM setup failed; inspect '+str(case/'worker.log'))
                    time.sleep(.1)
            hello = receive_message(connection)
            atomic_json(case/'operator_metadata.json', hello)
            positions = np.empty((hello['rhs_points'], 3))
            receive_into(connection, positions)
            del positions
            if not hello.get('prism_dihedral_mesh'):
                raise RuntimeError('D6h geometry was not enabled')
            for index, (direction, p, s) in enumerate(incoming):
                packet = np.concatenate([p, s, direction]).astype(np.float64)
                connection.sendall(struct.pack('=i', 2))
                connection.sendall(memoryview(packet).cast('B'))
                status = receive_message(connection)
                far = np.empty((2, len(BETAS)*len(THETA), 3), dtype=np.complex128)
                receive_into(connection, far)
                residuals = [status['strict_residual_x'], status['strict_residual_y']]
                if not np.isfinite(residuals).all() or max(residuals)>plan['bem_residual_tolerance']:
                    atomic_json(case/'rejected.json', status)
                    raise RuntimeError('absorbing FMM strict residual failed')
                fields.append(far[:, index*len(THETA):(index+1)*len(THETA)].copy())
                results.append(dict(beta=float(BETAS[index]), **status))
                atomic_json(case/'progress.json', dict(solved=len(results), orientations=results))
            connection.sendall(struct.pack('=i', 0))
            connection.close()
            if process.wait(timeout=60):
                raise RuntimeError('worker failed during shutdown')
        finally:
            connection.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
    far = np.array(fields)
    mueller, jones = far_to_mueller(far, plan)
    if not np.isfinite(mueller).all() or np.any(mueller[...,0,0]<=0):
        raise RuntimeError('invalid absorbing FMM far field')
    np.savez_compressed(case/'fields.npz', far=far, jones_um=jones, mueller_um2_per_sr=mueller)
    atomic_json(case/'complete.json', dict(complete=True, ref=ref, duffy=duffy,
        scope='Four fixed incident directions; strict residual is not a mesh accuracy certificate.'))


class MBSComparison(Pipeline):
    """Reuse the independently validated fullauto stop boundary for each binary."""
    def __init__(self, out, plan, bem, label, env):
        self.out = out/'averaged_mbs'/label
        self.out.mkdir(parents=True, exist_ok=True)
        key = 'mbs_binary' if label=='original' else 'mbs_corrected_binary'
        self.bem, self.env, self.active, self.active_pid = bem, env, {}, None
        self.plan = dict(ri=plan['ri'], ri_imag=plan['ri_imag'], wavelength=plan['wavelength'],
            aspect=plan['prism_aspect'], mbs_tolerance=plan['mbs_fullauto_tolerance'],
            mbs_binary=plan[key], mbs_manifest=dict(
                refractive_index=[plan['ri'], plan['ri_imag']], wavelength=plan['wavelength'],
                source_commit=plan['mbs_source_commit'], binary_sha256=plan[key+'_sha256'],
                version_label=label, external_reflection_corrected=label=='corrected'))

    def sample(self, ka):
        return self.bem


def execute(out):
    plan = read(out/'plan.json')
    identity = read(out/'queue_identity.json')
    if digest(out/'plan.json')!=identity['plan_sha256']:
        raise RuntimeError('absorbing queue plan changed')
    for filename, expected in read(RUNTIME/'sha256.json').items():
        if digest(RUNTIME/filename)!=expected:
            raise RuntimeError('frozen absorbing runtime changed: '+filename)
    for key in ['mbs_binary', 'mbs_corrected_binary']:
        if digest(plan[key])!=plan[key+'_sha256']:
            raise RuntimeError('MBS binary changed: '+key)
    def status(state, **extra):
        atomic_json(out/'gpu_status.json', dict(state=state, pid=os.getpid(),
            updated_unix=time.time(), **extra))
    dependency = Path(plan['after_series'])/'status.json'
    while read(dependency).get('state') not in ('complete', 'direct_complete_with_interpolation_rejections'):
        status('queued', waiting_for=str(dependency), dependency_state=read(dependency).get('state'),
               note='A failed or stopped prior series never authorizes an automatic GPU launch.')
        time.sleep(30)
    while read(out/'cpu_status.json').get('state')!='complete_preview':
        state = read(out/'cpu_status.json')
        if state.get('state')=='failed':
            raise RuntimeError('dense absorbing reference failed: '+str(state))
        status('waiting_cpu_reference'); time.sleep(30)
    env = {k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
    env.update(OMP_NUM_THREADS='16', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               BEM_SO3_TMPDIR=str(out/'scratch'))
    def wait_gpu():
        while True:
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
            if not result.stdout.strip():
                return
            status('waiting_gpu', pids=result.stdout.split()); time.sleep(15)
    def monitored(command, folder, environment=env):
        result = run(command, folder, env=environment, rss_limit_gib=48,
                     available_gib=6, disk_reserve_gib=50, wall_limit_s=None)
        if result['state']!='complete':
            raise RuntimeError(str(result))
    mbs = mbs_fixed(out, plan)
    def evaluate(ref, duffy):
        case = out/'gpu_fixed'/f'ref{ref}_d{duffy}'
        if not read(case/'complete.json').get('complete'):
            wait_gpu(); status('fixed_BEM', ref=ref, duffy=duffy)
            monitored([sys.executable, str(Path(__file__).resolve()), '--out', str(out),
                '--fixed-ref', str(ref), '--duffy', str(duffy)], case/'supervisor')
        with np.load(case/'fields.npz') as values:
            mean = values['mueller_um2_per_sr']
        plot(case/'comparison', mean, mbs, f'BEM ref{ref}, d{duffy}')
        atomic_json(out/'gpu_latest.json', dict(case=str(case), ref=ref, duffy=duffy,
            mesh_accuracy_certified=False))
        return mean
    # Same mesh and quadrature isolate accelerated-operator accuracy from
    # discretization error. No lossless validation is borrowed for Im(m)=10.
    reference = np.load(out/'cpu_preview/ref2_q7_d8/fields.npz')['mueller_um2_per_sr']
    control = evaluate(2, 8)
    error = difference(control, reference)
    atomic_json(out/'dense_fmm_control.json', error)
    if error['max_all16_over_reference_M11']>.005:
        raise RuntimeError('absorbing dense/FMM agreement failed 0.5% gate')
    previous, selected = None, None
    for ref in [3, 4, 5, 6]:
        current = evaluate(ref, 12)
        if previous is not None:
            mesh = difference(previous, current)
            atomic_json(out/f'mesh_control_ref{ref}.json', mesh)
            if mesh['max_all16_over_reference_M11']<=.005:
                quad = difference(evaluate(ref, 8), current)
                atomic_json(out/f'quadrature_control_ref{ref}.json', quad)
                if quad['max_all16_over_reference_M11']<=.005:
                    selected = ref
                    break
        previous = current
    if selected is None:
        raise RuntimeError('No mesh passed sampled fixed-angle controls through ref6; T was not launched')
    atomic_json(out/'sampled_controls.json', dict(ref=selected, duffy=12,
        passed=True, mesh=mesh, quadrature=quad,
        limitation='Four orientations only; this is not a 1% certificate for the analytic SO3 mean.'))
    chosen = None
    for order in [44, 52]:
        bem = out/'tmatrix'/f'ref{selected}_L{order}'
        wait_gpu(); status('T_and_analytic_SO3', ref=selected, order=order, directory=str(bem))
        command = [sys.executable, str(RUNTIME/'scripts/build_fmm_tmatrix.py'), '--out', str(bem),
            '--shape', 'prism', '--ka', str(plan['ka']), '--ri', str(plan['ri']),
            '--ri-imag', str(plan['ri_imag']), '--prism-aspect', str(plan['prism_aspect']),
            '--wavelength', str(plan['wavelength']), '--ref', str(selected), '--order', str(order),
            '--tol', str(plan['bem_residual_tolerance']), '--restart', '64', '--max-iters', '1000',
            '--c2-reuse', '--c6-pack', '--mirror-pack', '--incident-prefetch',
            '--average-backend', 'mstm', '--so3-precision', 'fp64', '--so3-threads', '4', '--ntheta', '1801']
        monitored(command, bem/'supervisor', dict(env, **profile(plan, selected, 12)))
        status('modal_control', order=order)
        report = analyze(bem, bem/'cutoff_controls', ntheta=1801, threads=4, orders=[order-8, order-4, order])
        tail = next(item for item in report['records'] if item['order']==order-4)
        if tail['acceptance_error']<=.005:
            chosen = bem
            break
    if chosen is None:
        raise RuntimeError('Modal tail control did not pass by L52')
    for label in ['original', 'corrected']:
        pipeline = MBSComparison(out, plan, chosen, label, env)
        for mode in ['coherent', 'incoherent']:
            status('MBS_fullauto', version=label, mode=mode, BEM=str(chosen))
            case = pipeline.ensure_mbs(plan['ka'], mode)
            pipeline.comparison(chosen, case, out/'so3_comparison'/label/mode)
    status('complete', BEM=str(chosen),
        comparison=str(out/'so3_comparison'),
        note='Analytic SO3 complete. Mesh evidence is sampled; MBS final physics audit stop is recorded separately.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fixed-ref', type=int)
    parser.add_argument('--duffy', type=int, default=12)
    args = parser.parse_args(); out = args.out.resolve()
    if args.fixed_ref is not None:
        fixed(out, args.fixed_ref, args.duffy)
        return
    with (out/'gpu_controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        def stop(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        try:
            execute(out)
        except BaseException as error:
            atomic_json(out/'gpu_status.json', dict(state='stopped' if isinstance(error,KeyboardInterrupt) else 'failed',
                pid=os.getpid(), error=repr(error), updated_unix=time.time()))
            raise


if __name__=='__main__':
    main()
