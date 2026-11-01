#!/usr/bin/env python3
"""Continue a frozen local T(ka) family, validate it, then compare with MBS.

The existing central T is excluded from fitting and reused as a control.
No new full T matrices are built solely for validation. This single-point
control does not certify an error throughout the size interval. Long jobs
are sequential, checkpointed and have no wall limit.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
from bounded_bem_run import atomic_json, run
from study_tmatrix_cutoff import cutoff_indices, error_metrics
from tmatrix_size_interpolation import RationalMatrix, fit_aaa
from tmatrix_so3_mstm import analytic_mueller_tmatrix

RUNTIME = Path(__file__).resolve().parents[1]
THETA = np.linspace(0, 180, 1801)


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def windows(center):
    """Only fitting anchors need new full solves; the center already exists."""
    if not np.isfinite(center) or center <= .5:
        raise ValueError('invalid central size')
    return [dict(name='narrow', lo=center-.25, hi=center+.25,
                 stages=[dict(training=[center-.25, center+.25], validation=[center])]),
            dict(name='wide', lo=center-.5, hi=center+.5,
                 stages=[dict(training=[center-.5, center-.25, center+.25, center+.5],
                              validation=[center])])]


def mbs_command(plan, ka, manifest, result, mode):
    if mode not in ('coherent', 'incoherent'):
        raise ValueError('unknown MBS coherence mode')
    command = [plan['mbs_binary'], '--method', 'po', '--backend', 'cuda',
        '--particle', '1', repr(manifest['particle_L_over_D_vertex']), '1', '--k-eq', str(ka),
        '--refractive-index', str(plan['ri']), str(plan.get('ri_imag', 0)), '--wavelength-um', str(plan['wavelength']),
        '--fullauto', str(plan['mbs_tolerance']), '--fullauto-theta-range', '0', '180',
        '--output', str(result), '--close']
    if mode == 'incoherent':
        command.insert(1, '--incoherent')
    return command


def acceptance(interpolation, modal_indicator, plan):
    if plan.get('validation_mode') == 'reuse_completed_center':
        return bool(np.isfinite(interpolation) and interpolation <= plan['bem_target'])
    total = interpolation + modal_indicator
    return bool(np.isfinite(total) and interpolation <= plan['interpolation_budget']
                and modal_indicator <= plan['modal_budget'] and total <= plan['tolerance'])


def check_family(parameters, ka, plan):
    expected = {'ka': ka, 'ref': plan['ref'], 'order': plan['order'],
                'ri': plan['ri'], 'ri_imag': 0., 'wavelength': plan['wavelength'],
                'prism_aspect': plan['aspect'], 'shape': 'prism'}
    for key, value in expected.items():
        if parameters.get(key) != value:
            raise ValueError(f'physical/basis mismatch: {key}: {parameters.get(key)} != {value}')


class Pipeline:
    def __init__(self, out):
        self.out = out
        self.plan = read(out / 'plan.json')
        if not self.plan:
            raise ValueError('missing frozen plan')
        identity = read(out/'identity.json')
        if identity.get('plan_sha256') != digest(out/'plan.json'):
            raise ValueError('series plan changed; use an explicit new series identity')
        if identity.get('runtime_sha256') != digest(RUNTIME/'sha256.json'):
            raise ValueError('frozen runtime manifest changed')
        self.active = {}
        self.active_pid = None
        self.done = threading.Event()
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('BEM_')}
        self.env.update(self.plan['profile'], OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                        BEM_SO3_TMPDIR=str(out / 'scratch'))
        os.environ['BEM_SO3_TMPDIR'] = str(out / 'scratch')
        for name, expected in read(RUNTIME / 'sha256.json').items():
            if digest(RUNTIME / name) != expected:
                raise ValueError('frozen runtime changed: ' + name)
        if digest(self.plan['mbs_binary']) != self.plan['mbs_manifest']['binary_sha256']:
            raise ValueError('frozen MBS executable changed')

    def monitor(self):
        while not self.done.wait(15):
            try:
                active = dict(self.active)
                progress = read(Path(active['directory'])/'progress.json') if active.get('directory') else {}
                if active.get('phase') == 'MBS_fullauto':
                    progress = read(Path(active['directory'])/'result/execution_progress.json')
                last = progress.get('last_pair', {})
                seconds = last.get('total_worker_s', 0) + last.get('incident_field_s', 0)
                atomic_json(self.out/'eta.json', dict(updated_unix=time.time(), active=active,
                    progress=progress, current_T_remaining_estimate_s=(
                        seconds*progress.get('remaining_worker_pairs', 0) if seconds else None),
                    scope='Recent-packet estimate; early independent audits are not representative '
                          'of all production packets. Not a completion guarantee.'))
            except (OSError, ValueError, KeyError):
                pass

    def status(self, state='running', **extra):
        atomic_json(self.out / 'status.json', dict(state=state, pid=os.getpid(),
                    updated_unix=time.time(), active=self.active, **extra))

    def case(self, ka):
        return self.out / 'direct' / f'ka{ka:.8f}_ref{self.plan["ref"]}_L{self.plan["order"]}'

    def sample(self, ka):
        return self.case(ka) / f'main_ref{self.plan["ref"]}_L{self.plan["order"]}'

    def wait_gpu(self):
        while True:
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=10)
            if result.returncode:
                raise RuntimeError(result.stderr)
            if not result.stdout.strip():
                return
            self.status(waiting_for_gpu_pids=result.stdout.split())
            time.sleep(10)

    def monitored(self, command, folder, *, env=None):
        folder.mkdir(parents=True, exist_ok=True)
        result = run(command, folder, env=self.env if env is None else env,
                     rss_limit_gib=48, available_gib=6, disk_reserve_gib=50, wall_limit_s=None)
        if result['state'] != 'complete':
            raise RuntimeError(f'{folder}: {result}')

    def average(self, t, ka, order=None, angles=THETA):
        order = self.plan['order'] if order is None else order
        mean, metadata = analytic_mueller_tmatrix(t, order, angles, precision='fp64',
                                                  threads=4, cyclic_order=6)
        if metadata['Euler_quadrature_nodes'] != 0:
            raise RuntimeError('orientation averaging must be analytic')
        metadata.update(Csca=float(2*np.pi/ka**2*np.vdot(t, t).real),
                        Cext=float(-2*np.pi/ka**2*np.trace(t).real))
        return mean, metadata

    def save_average(self, dest, t, ka, parameters, *, source=None, interpolation=None):
        dest.mkdir(parents=True, exist_ok=True)
        mean, meta = self.average(t, ka)
        atomic_json(dest / 'average.json', dict(ka=ka, **meta,
            theta=THETA.tolist(), mueller=np.moveaxis(mean, 0, -1).tolist()))
        summary = dict(complete=True, parameters=parameters,
            scattering_cross_section=meta['Csca'], extinction_cross_section=meta['Cext'],
            analytic_so3=meta, completed_columns=t.shape[1], source=source)
        if interpolation:
            summary['size_interpolation'] = interpolation
        atomic_json(dest / 'summary.json', summary)
        return mean, meta

    def prepare_reference(self):
        ka = self.plan['center']
        dest = self.sample(ka)
        if read(dest / 'summary.json').get('complete'):
            self.load(ka)
            return
        self.active = dict(phase='crop_completed_reference', ka=ka)
        self.status()
        source = Path(self.plan['reference'])
        summary = read(source / 'summary.json')
        if not summary.get('complete') or digest(source / 'tmatrix.npy') != self.plan['reference_T_sha256']:
            raise ValueError('reference T identity or completeness changed')
        parameters = dict(summary['parameters'], order=self.plan['order'])
        check_family(parameters, ka, self.plan)
        original = np.load(source / 'tmatrix.npy', mmap_mode='r')
        indices = cutoff_indices(summary['parameters']['order'], self.plan['order'])
        dest.mkdir(parents=True, exist_ok=True)
        temporary = dest / 'tmatrix.tmp.npy'
        target = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.complex128,
                                           shape=(len(indices), len(indices)), fortran_order=True)
        for start in range(0, len(indices), 32):
            target[:, start:start+32] = original[np.ix_(indices, indices[start:start+32])]
        target.flush()
        del target
        temporary.replace(dest / 'tmatrix.npy')
        t = np.load(dest / 'tmatrix.npy', mmap_mode='r')
        self.save_average(dest, t, ka, parameters, source=dict(
            path=str(source), source_order=summary['parameters']['order'],
            source_T_sha256=self.plan['reference_T_sha256'],
            cutoff_error=self.plan['reference_cutoff_error'], operation='exact modal submatrix'))

    def load(self, ka):
        path = self.sample(ka)
        summary = read(path / 'summary.json')
        if not summary.get('complete'):
            raise ValueError('incomplete direct T: ' + str(path))
        check_family(summary['parameters'], ka, self.plan)
        t = np.load(path / 'tmatrix.npy', mmap_mode='r')
        count = 2*self.plan['order']*(self.plan['order']+2)
        if t.shape != (count, count):
            raise ValueError('basis shape mismatch')
        return t, summary

    def ensure_direct(self, ka):
        if read(self.sample(ka) / 'summary.json').get('complete'):
            self.load(ka)
            return
        self.active = dict(phase='direct_T', ka=ka, directory=str(self.sample(ka)))
        self.status()
        launch = read(self.case(ka) / 'launch.json')
        if launch.get('pid') and psutil.pid_exists(launch['pid']):
            process = psutil.Process(launch['pid'])
            if process.status() != psutil.STATUS_ZOMBIE:
                if str(self.case(ka)) not in process.cmdline():
                    raise RuntimeError('adopted controller identity mismatch')
                self.active_pid = process.pid
                while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    progress = read(self.sample(ka) / 'progress.json')
                    self.status(progress=progress)
                    time.sleep(15)
                self.active_pid = None
                self.load(ka)
                return
        self.wait_gpu()
        command = [sys.executable, str(RUNTIME/'scripts/build_fmm_tmatrix.py'),
            '--shape', 'prism', '--ref', str(self.plan['ref']), '--ka', str(ka),
            '--ri', str(self.plan['ri']), '--prism-aspect', str(self.plan['aspect']),
            '--wavelength', str(self.plan['wavelength']), '--order', str(self.plan['order']),
            '--tol', '2e-6', '--restart', '64', '--max-iters', '1000',
            '--c2-reuse', '--c6-pack', '--mirror-pack', '--fast-mlfma', '--incident-prefetch',
            '--so3-precision', 'fp64', '--so3-threads', '4', '--ntheta', '1801',
            '--out', str(self.sample(ka))]
        self.monitored(command, self.case(ka)/'pipeline_supervisor')
        self.load(ka)

    def direct_control(self, ka):
        dest = self.out/'controls'/f'ka{ka:.8f}'
        if ka == self.plan['center'] and self.plan.get('validation_mode') == 'reuse_completed_center':
            # Subsample the existing analytic output, never interpolate angles.
            source = Path(self.plan['reference'])
            summary = read(source/'summary.json')
            average = read(source/'average.json')
            angles = np.array(average['theta'])
            index = np.rint(THETA/(angles[1]-angles[0])).astype(int)
            np.testing.assert_allclose(angles[index], THETA, rtol=0, atol=1e-12)
            mean = np.moveaxis(np.array(average['mueller']), -1, 0)[index]
            saved = dict(ka=ka, Csca=summary['scattering_cross_section'],
                Cext=summary['extinction_cross_section'], modal_indicator=self.plan['reference_cutoff_error'],
                reference_order=summary['parameters']['order'], source=str(source),
                T_sha256=self.plan['reference_T_sha256'], Euler_quadrature_nodes=0,
                scope='Existing full L64 result at center, excluded from interpolation fitting.')
            return mean, saved
        saved = read(dest/'control.json')
        if saved:
            if saved['T_sha256'] != digest(self.sample(ka)/'tmatrix.npy'):
                raise ValueError('direct control matrix changed')
            with np.load(dest/'mueller.npz') as arrays:
                return arrays['mueller'], saved
        self.active = dict(phase='modal_control', ka=ka)
        self.status()
        t, _ = self.load(ka)
        mean, meta = self.average(t, ka)
        indices = cutoff_indices(self.plan['order'], self.plan['order']-2)
        truncated, trunc_meta = self.average(t[np.ix_(indices, indices)], ka, self.plan['order']-2)
        error = error_metrics(truncated, mean, THETA)
        cross = max(abs(trunc_meta[key]/meta[key]-1) for key in ('Csca', 'Cext'))
        saved = dict(ka=ka, T_sha256=digest(self.sample(ka)/'tmatrix.npy'),
            Csca=meta['Csca'], Cext=meta['Cext'], modal_indicator=max(
                cross, error['maximum_all16_error_over_reference_M11']),
            modal_comparison=error, Euler_quadrature_nodes=0,
            scope='Adjacent modal cutoffs on ref6; not an absolute mesh-error certificate.')
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest/'mueller.npz', theta=THETA, mueller=mean, lower_order=truncated)
        atomic_json(dest/'control.json', saved)
        return mean, saved

    def fit(self, dest, definition):
        dest.mkdir(parents=True, exist_ok=True)
        saved = read(dest/'model.json')
        nodes = definition['training']
        samples = [self.load(ka)[0] for ka in nodes]
        identities = [digest(self.sample(ka)/'tmatrix.npy') for ka in nodes]
        if saved:
            if saved['training'] != nodes or saved['training_sha256'] != identities:
                raise ValueError('frozen fitting inputs changed')
            indices = saved['support_indices']
            model = RationalMatrix(np.array(nodes)[indices], tuple(samples[i] for i in indices),
                np.array(saved['weights_real'])+1j*np.array(saved['weights_imag']), tuple(saved['domain']))
        else:
            self.active = dict(phase='fit_complex_T', training=nodes, output=str(dest))
            self.status()
            if len(nodes) == 2:
                model = RationalMatrix(np.array(nodes), tuple(samples), np.array([-1., 1.]),
                                       (float(min(nodes)), float(max(nodes))))
                diagnostics = dict(algorithm='linear complex T in barycentric form', support_indices=[0, 1])
            else:
                model, diagnostics = fit_aaa(nodes, samples, tolerance=1e-6)
            saved = dict(training=nodes, validation=definition['validation'],
                support_indices=diagnostics['support_indices'], weights_real=model.weights.real.tolist(),
                weights_imag=model.weights.imag.tolist(), domain=list(model.domain),
                diagnostics=diagnostics, training_sha256=identities,
                frozen_before_validation=True, created_unix=time.time())
            atomic_json(dest/'model.json', saved)
        return model, saved

    def validate(self, dest, definition, model):
        records = []
        for ka in definition['training']+definition['validation']:
            self.active = dict(phase='validate_interpolation', ka=ka, output=str(dest))
            self.status()
            direct, control = self.direct_control(ka)
            predicted, meta = self.average(model.evaluate(ka), ka)
            error = error_metrics(predicted, direct, THETA)
            cross = {key: abs(meta[key]/control[key]-1) for key in ('Csca', 'Cext')}
            maximum = max(error['maximum_all16_error_over_reference_M11'], *cross.values())
            records.append(dict(ka=ka, role='holdout' if ka in definition['validation'] else 'training',
                maximum=maximum, cross_section_errors=cross, **error,
                modal_indicator=control['modal_indicator']))
            np.savez_compressed(dest/f'validation_ka{ka:.8f}.npz', theta=THETA,
                                direct=direct, predicted=predicted)
        interpolation = max(row['maximum'] for row in records)
        modal = max(self.plan['reference_cutoff_error'],
                    max(row['modal_indicator'] for row in records))
        reused_center = self.plan.get('validation_mode') == 'reuse_completed_center'
        checked = interpolation if reused_center else interpolation+modal
        report = dict(records=records, maximum_interpolation_error=interpolation,
            maximum_modal_indicator=modal, empirical_budget_sum=interpolation+modal,
            maximum_checked_error=checked, tolerance=self.plan['bem_target'],
            passed=acceptance(interpolation, modal, self.plan),
            metric='max over 1801 angles and all16 |delta Mij|/direct M11, plus relative Csca/Cext',
            Euler_quadrature_nodes=0, mesh_error_certified=False,
            new_full_T_validation_solves=0 if reused_center else len(definition['validation']),
            scope=('Existing center excluded from fitting; other checks use training samples. '
                   'Center comparison includes L58 vs full L64 truncation. No new full validation T solves; '
                   'no uniform interval accuracy certificate.' if reused_center else
                   'Finite independent-size and modal checks; not a uniform or absolute error certificate.'))
        if reused_center:
            report.pop('empirical_budget_sum')
        atomic_json(dest/'validation.json', report)
        return report

    def ensure_mbs(self, ka, mode):
        parent = self.out/'mbs'/f'ka{ka:.8f}'
        case = parent/mode
        result = case/'result'
        result.mkdir(parents=True, exist_ok=True)
        manifest = dict(self.plan['mbs_manifest'])
        a = ka*self.plan['wavelength']/(2*np.pi)
        width = a*(8*np.pi/(3*np.sqrt(3)*self.plan['aspect']))**(1/3)
        manifest.update(ka=ka, equiv_radius=a, height_um=width*self.plan['aspect'],
            width_across_flats_um=width, diameter_vertex_um=width*2/np.sqrt(3),
            prism_height_to_width_across_flats=self.plan['aspect'],
            particle_L_over_D_vertex=self.plan['aspect']*np.sqrt(3)/2,
            binary=self.plan['mbs_binary'], bem=str(self.sample(ka)),
            note='Frozen same executable for both modes. Stop after fresh-seed validation '
                 'and table publication, before the long final physics audit, per user request.')
        manifest.pop('coherence_mode', None)
        old = read(parent/'manifest.json')
        if old and old != manifest:
            raise ValueError('MBS physical manifest changed')
        atomic_json(parent/'manifest.json', manifest)
        audit = read(result/'independent_final_validation.json')
        stopped = read(case/'user_stop_final_checks.json')
        native = read(result/'fullauto_status.json')
        if audit.get('passed') and (native.get('status') == 'converged' or stopped.get('stopped')):
            if stopped and digest(result/'mueller_fullauto.dat') != stopped['table_sha256_after']:
                raise ValueError('saved MBS table changed after stop')
            return case
        self.active = dict(phase='MBS_fullauto', ka=ka, mode=mode, directory=str(case))
        self.status()
        self.wait_gpu()
        command = mbs_command(self.plan, ka, manifest, result, mode)
        launch = read(case/'launch_metadata.json')
        if launch and launch['command'] != command:
            raise ValueError('MBS resume command changed')
        atomic_json(case/'launch_metadata.json', dict(command=command, wall_limit_s=None))
        stopped_after_audit = False
        started = time.monotonic()
        with (case/'worker.log').open('a') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=self.env, start_new_session=True)
            self.active_pid = process.pid
            try:
                while process.poll() is None:
                    audit = read(result/'independent_final_validation.json')
                    table = result/'mueller_fullauto.dat'
                    audit_file = result/'independent_final_validation.json'
                    native = read(result/'fullauto_status.json')
                    if (audit.get('passed') and table.exists() and
                            table.stat().st_mtime_ns >= audit_file.stat().st_mtime_ns):
                        if native.get('status') != 'converged':
                            os.killpg(process.pid, signal.SIGTERM)
                            stopped_after_audit = True
                        break
                    if psutil.virtual_memory().available < 6*2**30:
                        raise RuntimeError('MBS RAM reserve reached')
                    if __import__('shutil').disk_usage(self.out).free < 50*2**30:
                        raise RuntimeError('MBS disk reserve reached')
                    self.status(native_progress=read(result/'execution_progress.json'))
                    time.sleep(2)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                audit = read(result/'independent_final_validation.json')
                if not audit.get('passed') or not (result/'mueller_fullauto.dat').exists():
                    raise RuntimeError('MBS exited before independent validation and table publication')
                native = read(result/'fullauto_status.json')
                if not stopped_after_audit and (process.returncode != 0 or native.get('status') != 'converged'):
                    raise RuntimeError('MBS did not complete or stop at the approved audit boundary')
                receipt = dict(stopped=stopped_after_audit, requested_by='user: stop long final checks',
                    independent_validation=audit, final_physics_checks_complete=not stopped_after_audit,
                    table_sha256_after=digest(result/'mueller_fullauto.dat'),
                    elapsed_s=time.monotonic()-started, returncode=process.returncode)
                atomic_json(case/'user_stop_final_checks.json', receipt)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                self.active_pid = None
        return case

    def comparison(self, bem, case, output):
        report = read(output/'comparison_summary.json')
        if report:
            if report['mbs_table_sha256'] != digest(case/'result/mueller_fullauto.dat'):
                raise ValueError('MBS comparison source changed')
            return report
        command = [sys.executable, str(RUNTIME/'scripts/plot_bem_mbs_all16.py'),
                   '--bem', str(bem), '--mbs', str(case), '--out', str(output), '--allow-intermediate']
        self.monitored(command, output/'supervisor')
        return read(output/'comparison_summary.json')

    def export_grid(self, dest, model, validation):
        saved = read(dest/'grid.json')
        if saved:
            if not (dest/'size_grid.npz').exists():
                raise ValueError('missing completed grid')
            return
        lo, hi = model.domain
        ka_values = np.linspace(lo, hi, int(round((hi-lo)/self.plan['grid_step']))+1)
        means, csca, cext = [], [], []
        for ka in ka_values:
            self.active = dict(phase='analytic_size_grid', ka=float(ka), output=str(dest))
            self.status()
            mean, meta = self.average(model.evaluate(ka), ka)
            means.append(mean*(self.plan['wavelength']/(2*np.pi))**2)
            radius = ka*self.plan['wavelength']/(2*np.pi)
            csca.append(meta['Csca']*radius**2)
            cext.append(meta['Cext']*radius**2)
        temporary = dest/'size_grid.tmp.npz'
        np.savez_compressed(temporary, ka=ka_values, theta_deg=THETA,
            mueller_um2_per_sr=np.array(means), Csca_um2=csca, Cext_um2=cext,
            source=np.array(['INTERPOLATED_T_ANALYTIC_SO3']*len(ka_values)))
        temporary.replace(dest/'size_grid.npz')
        atomic_json(dest/'grid.json', dict(sizes=len(ka_values), interval=[lo, hi],
            ka_step=self.plan['grid_step'], Euler_quadrature_nodes=0,
            additional_BEM_solves=0, validation=validation,
            interpolation_marker='INTERPOLATED_T_ANALYTIC_SO3'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
        color_map = plt.get_cmap('viridis')
        for ka, mean in zip(ka_values, means):
            ax.semilogy(THETA, mean[:, 0, 0], color=color_map((ka-lo)/(hi-lo)), lw=1.)
        color_bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(lo, hi), cmap=color_map), ax=ax)
        color_bar.set_label('ka')
        ax.set(xlim=(0, 180), xlabel='θ, °', ylabel='M₁₁, мкм²/ср',
               title='Интерполированная T(ka), аналитическое усреднение SO(3)\n'
                     f'n={self.plan["ri"]}, λ={self.plan["wavelength"]} мкм, H/ширина=10/7')
        ax.grid(alpha=.2)
        for extension in ('png', 'pdf'):
            fig.savefig(dest/f'm11_size_grid.{extension}', dpi=170)
        plt.close(fig)

    def export_comparisons(self, dest, definition, model, validation):
        # Include direct endpoints, the existing center and all independent
        # interpolation controls. MBS is solved at the same physical sizes.
        targets = sorted(set(definition['training'] + definition['validation']))
        records = []
        for ka in targets:
            prediction = dest/'predicted'/f'ka{ka:.8f}'
            predicted = model.evaluate(ka)
            prediction.mkdir(parents=True, exist_ok=True)
            if not (prediction/'tmatrix.npy').exists():
                temporary = prediction/'tmatrix.tmp.npy'
                with temporary.open('wb') as stream:
                    np.save(stream, predicted)
                temporary.replace(prediction/'tmatrix.npy')
            _, source = self.load(ka)
            if not read(prediction/'summary.json').get('complete'):
                self.save_average(prediction, predicted, ka, source['parameters'],
                    interpolation=dict(model=str(dest/'model.json'), validated=True,
                        checked_discrepancy=validation['maximum_checked_error'],
                        requested_tolerance=self.plan['bem_target'], validation_scope=validation['scope'],
                        source='complex T rational interpolation'))
            del predicted
            for mode in ('coherent', 'incoherent'):
                case = self.ensure_mbs(ka, mode)
                self.active = dict(phase='compare_Mueller', ka=ka, mode=mode)
                self.status()
                direct_source = Path(self.plan['reference']) if ka == self.plan['center'] else self.sample(ka)
                direct_report = self.comparison(direct_source, case, self.case(ka)/'comparison'/mode)
                interpolated_report = self.comparison(prediction, case,
                    dest/'comparisons'/f'ka{ka:.8f}'/mode)
                records.append(dict(ka=ka, mode=mode,
                    direct_BEM_comparison=str(self.case(ka)/'comparison'/mode),
                    interpolated_T_comparison=str(dest/'comparisons'/f'ka{ka:.8f}'/mode),
                    direct_forward_M11_difference_percent=direct_report['forward_M11_difference_percent'],
                    interpolated_forward_M11_difference_percent=interpolated_report['forward_M11_difference_percent'],
                    maximum_difference_percent_of_BEM_M11=interpolated_report['worst_difference_percent_of_BEM_M11'],
                    MBS_independent_validation=interpolated_report['mbs_independent_validation'],
                    MBS_final_checks=interpolated_report.get('mbs_user_stop')))
                atomic_json(dest/'comparison_index.json', records)
            # These are reproducible predictions, not training/checkpoint T's.
            # Retain Mueller curves, weights and identities; avoid one large
            # matrix file for every interpolated point.
            (prediction/'tmatrix.npy').unlink()
        return records

    def direct_comparisons(self, definition):
        records = []
        for ka in sorted(set(definition['training'] + definition['validation'])):
            for mode in ('coherent', 'incoherent'):
                case = self.ensure_mbs(ka, mode)
                source = Path(self.plan['reference']) if ka == self.plan['center'] else self.sample(ka)
                output = self.case(ka)/'comparison'/mode
                report = self.comparison(source, case, output)
                records.append(dict(ka=ka, mode=mode, comparison=str(output),
                    forward_M11_difference_percent=report['forward_M11_difference_percent']))
        return records

    def execute(self):
        self.prepare_reference()
        for window in self.plan['windows']:
            window_out = self.out/window['name']
            if read(window_out/'results.json').get('state') == 'complete':
                continue
            accepted = False
            for stage, definition in enumerate(window['stages']):
                self.active = dict(phase='training', window=window['name'], stage=stage)
                self.status()
                # Finish the already running first upper anchor first; sizes
                # common to earlier stages/windows are loaded, never solved again.
                for ka in sorted(definition['training'], reverse=True):
                    self.ensure_direct(ka)
                dest = window_out/f'model_{stage}'
                model, _ = self.fit(dest, definition)
                for ka in definition['validation']:
                    self.ensure_direct(ka)
                validation = self.validate(dest, definition, model)
                if (self.plan.get('validation_mode') != 'reuse_completed_center' and
                        validation['maximum_modal_indicator'] > self.plan['modal_budget']):
                    self.status('modal_control_failed', validation=validation,
                                reason='Do not expand or publish a 4% family until increasing L is audited.')
                    return
                if not validation['passed']:
                    continue
                self.export_grid(dest, model, validation)
                comparisons = self.export_comparisons(dest, definition, model, validation)
                window_out.mkdir(parents=True, exist_ok=True)
                atomic_json(window_out/'results.json', dict(state='complete', stage=stage,
                    interval=[window['lo'], window['hi']], model=str(dest),
                    validation=validation, comparisons=comparisons,
                    Euler_quadrature_nodes=0, all_16_mueller_entries=True))
                accepted = True
                break
            if not accepted:
                if self.plan.get('validation_mode') != 'reuse_completed_center':
                    self.status('interpolation_control_failed', window=window['name'])
                    return
                comparisons = self.direct_comparisons(definition)
                atomic_json(window_out/'results.json', dict(state='direct_complete_interpolation_rejected',
                    interval=[window['lo'], window['hi']], validation=validation, comparisons=comparisons,
                    reason='Center control failed. Dense interpolated grid withheld; widening adds '
                           'new fitting anchors, not expensive validation-only T matrices.'))
        outputs = [self.out/w['name']/'results.json' for w in self.plan['windows']]
        all_passed = all(read(p).get('state') == 'complete' for p in outputs)
        self.status('complete' if all_passed else 'direct_complete_with_interpolation_rejections',
                    windows=list(map(str, outputs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    lock = (out/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pipeline = Pipeline(out)
    monitor = threading.Thread(target=pipeline.monitor, daemon=True)
    monitor.start()
    (out/'supervisor.pid').write_text(str(os.getpid())+'\n')

    def stop(signum, frame):
        if pipeline.active_pid is not None:
            try:
                os.kill(pipeline.active_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        pipeline.execute()
    except KeyboardInterrupt:
        pipeline.status('stopped_by_user')
    except Exception as error:
        pipeline.status('failed', reason=str(error))
        raise
    finally:
        pipeline.done.set()
        monitor.join(timeout=20)
        lock.close()


if __name__ == '__main__':
    main()
