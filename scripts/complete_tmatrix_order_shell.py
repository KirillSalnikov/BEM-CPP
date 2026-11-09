#!/usr/bin/env python3
"""Combine a direct old T and a newly solved incident-order shell.

This is an experimental reciprocal completion, not a direct complete BEM
solve or an order certificate. It preserves known entries exactly and
reports independent low-column controls, without fitting a normalization.
The original run directories are read only. Use a separate output folder.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from build_fmm_tmatrix import atomic_array, atomic_json
from study_tmatrix_cutoff import digest, error_metrics
from tmatrix_reference import modes
from tmatrix_reciprocal_extension import complete_order_extension, extension_layout, extension_work_counts
from tmatrix_symmetry import c2x_plan, c2x_partner_column
from tmatrix_so3_mstm import analytic_mueller_tmatrix


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matching_parameters(old, new):
    """No implicit migration of geometry, solver, worker, or incident convention."""
    require(isinstance(old.get('order'), int) and isinstance(new.get('order'), int), 'Missing integer orders')
    require(1 <= old['order'] < new['order'] <= 256, 'Shell must extend the old order')
    require(old.get('incident_min_order', 1) == 1, 'Old T is itself a partial shell')
    require(new.get('incident_min_order') == old['order']+1, 'Shell starts at the wrong degree')
    for key in ['shape', 'ref', 'ka', 'ri', 'ri_imag', 'prism_aspect', 'geometry', 'geometry_digest',
                'binary_digest', 'incident_digest', 'projection_digest', 'harmonics_digest', 'BEM_environment', 'tol']:
        require(key in old and key in new, 'Missing source identity: '+key)
    allowed = {'order', 'angular_band', 'incident_min_order', 'order_shell_digest', 'c6_driver_digest'}
    for key in (set(old) | set(new))-allowed:
        a, b = old.get(key), new.get(key)
        if key == 'BEM_environment':
            a = {k:v for k,v in a.items() if k != 'BEM_SO3_TMPDIR'}
            b = {k:v for k,v in b.items() if k != 'BEM_SO3_TMPDIR'}
        require(a == b, 'Source and shell identities differ: '+key)


class AcceptedColumns:
    """Recheck original residual/receipt/C2 gates before reading each column."""
    def __init__(self, folder, manifest):
        self.folder, self.manifest = Path(folder), manifest
        self.parameters = manifest['parameters']
        self.modal = modes(self.parameters['order'])
        require(manifest['modes'] == [list(m) for m in self.modal], 'Wrong saved modal order')
        self.flip, self.signs, _, _ = c2x_plan(self.parameters['order'])
        self.hashes = {}

    def get(self, index, active=()):
        require(isinstance(index, (int, np.integer)) and 0 <= index < len(self.modal) and index not in active,
                'Invalid or cyclic column provenance')
        path = self.folder/'columns'/f'{index:06d}.npy'
        meta_path = path.with_suffix('.json')
        info = read(meta_path)
        require(info['mode'] == list(self.modal[index]), 'Column mode mismatch')
        status = info['status']
        residuals = [status['strict_residual_x'], status['strict_residual_y']]
        require(all(math.isfinite(v) and 0 <= v <= self.parameters['tol'] for v in residuals),
                'Original FP64 residual gate failed')
        if 'adaptive_tight_tol' in self.parameters:
            limit = status.get('requested_tolerance')
            require(limit in (self.parameters['tol'], self.parameters['adaptive_tight_tol']) and max(residuals) <= limit,
                    'Adaptive residual gate failed')
            if 'c6_batch' not in info:
                require(limit == self.parameters['adaptive_tight_tol'], 'Independent control was not solved tightly')
        value = np.load(path, allow_pickle=False)
        require(value.shape == (len(self.modal),) and value.dtype == np.complex128 and np.isfinite(value).all(),
                'Invalid kernel column')
        derived = info.get('derived_by')
        require(derived in (None, 'C2x_half_turn', 'C6_packed_RHS'), 'Unknown or inferred column provenance')
        if derived == 'C2x_half_turn':
            origin = info['source_column']
            require(self.parameters.get('c2_reuse') and isinstance(origin, int) and 0 <= origin < len(self.modal)
                    and self.modal[origin][2] > 0 and int(self.flip[origin]) == index, 'Invalid C2 parent')
            parent, _ = self.get(origin, (*active, index))
            require(np.array_equal(value, c2x_partner_column(parent, origin, self.flip, self.signs)),
                    'Corrupt C2 descendant')
            parent_info = read(self.folder/'columns'/f'{origin:06d}.json')
            require(info.get('c6_batch') == parent_info.get('c6_batch'), 'C2/C6 provenance mismatch')
        if derived == 'C6_packed_RHS' or 'c6_batch' in info:
            batch = info.get('c6_batch')
            require(self.parameters.get('c6_pack') and isinstance(batch, str) and len(batch) == 24 and
                    all(c in '0123456789abcdef' for c in batch), 'Invalid C6 receipt name')
            receipt_path = self.folder/'c6_batches'/f'{batch}.json'
            receipt = read(receipt_path)
            origin = info.get('source_column', index)
            require(receipt['passed'] and receipt['signature'] == self.manifest['signature'] and origin in receipt['columns'],
                    'Unaccepted C6 packet')
            require(digest(self.folder/'columns'/f'{origin:06d}.npy') == receipt['column_sha256'][str(origin)],
                    'C6 column hash mismatch')
            self.hashes[str(receipt_path.resolve())] = digest(receipt_path)
        self.hashes[str(path.resolve())] = digest(path)
        self.hashes[str(meta_path.resolve())] = digest(meta_path)
        return value, info


def kernel_to_t_column(column, index, order, ka):
    phase = np.array([1j**(degree-p) for p, degree, _ in modes(order)])
    return -1j*ka/(4*np.pi)*phase*column*phase[index].conj()


def cross_sections(t, ka):
    power = sum(float(np.vdot(t[:, i], t[:, i]).real) for i in range(len(t)))
    return dict(scattering_cross_section=2*np.pi/ka**2*power,
                extinction_cross_section=float(-2*np.pi/ka**2*np.trace(t).real))


def assemble(source, shell, out, *, control_gate=1e-4, ntheta=1801, threads=2):
    source, shell, out = map(lambda p:Path(p).resolve(), (source, shell, out))
    require(source != shell and all(out != p and p not in out.parents for p in (source, shell)),
            'Output must be outside both source runs')
    require(math.isfinite(control_gate) and 0 < control_gate <= .01 and ntheta >= 181 and threads >= 1,
            'Invalid control or postprocessing settings')
    require(not out.exists() or not any(out.iterdir()), 'Output is not empty; preserve previous evidence')
    begin = time.monotonic()
    a, b = read(source/'summary.json'), read(shell/'summary.json')
    ma, mb = read(source/'manifest.json'), read(shell/'manifest.json')
    require(a['complete'] and not a.get('reciprocity_extension'), 'Old T must be a complete direct build')
    require(b['requested_columns_complete'], 'New incident shell is incomplete')
    require(a['parameters'] == ma['parameters'] and b['parameters'] == mb['parameters'] and
            a['signature'] == ma['signature'] and b['signature'] == mb['signature'], 'Summary/manifest mismatch')
    matching_parameters(a['parameters'], b['parameters'])
    old_order, order, ka = a['parameters']['order'], b['parameters']['order'], b['parameters']['ka']
    low, high, *_ = extension_layout(old_order, order)
    require(b['incident_order_shell']['requested_columns'] == high.tolist(), 'Wrong shell columns')
    for summary in (a, b):
        if summary['parameters'].get('c2_reuse'):
            require(summary['symmetry_reuse']['passed'], 'C2 audit failed')
        if summary['parameters'].get('c6_pack'):
            require(summary['c6_packing']['independent_reference_audits_complete'], 'C6 audit is incomplete')
    before = {str(p):digest(p) for p in [source/'tmatrix.npy', source/'summary.json', source/'manifest.json',
                                       shell/'summary.json', shell/'manifest.json']}
    old_t = np.load(source/'tmatrix.npy', mmap_mode='r', allow_pickle=False)
    require(old_t.shape == (len(low), len(low)) and old_t.dtype == np.complex128 and np.isfinite(old_t).all(),
            'Invalid old T')
    columns = AcceptedColumns(shell, mb)
    new_t = np.empty((len(low)+len(high), len(high)), complex)
    for j, i in enumerate(high):
        values, _ = columns.get(int(i))
        new_t[:, j] = kernel_to_t_column(values, i, order, ka)
    candidate = complete_order_extension(old_t, new_t, old_order, order)
    # Low-degree controls are independently solved by the shell worker, not
    # reconstructed via C2 or unpacked from a phase-coded RHS.
    controls = []
    for path in sorted((shell/'columns').glob('*.json')):
        i = int(path.stem)
        if columns.modal[i][1] > old_order or read(path).get('derived_by'):
            continue
        raw, _ = columns.get(i)
        direct = kernel_to_t_column(raw, i, order, ka)
        denominator = max(float(np.linalg.norm(direct)), 1e-300)
        controls.append(dict(column=i, mode=columns.modal[i],
            full_column_relative_error=float(np.linalg.norm(candidate[:, i]-direct)/denominator),
            old_rows_relative_to_full_column=float(np.linalg.norm(candidate[low, i]-direct[low])/denominator),
            inferred_rows_relative_to_full_column=float(np.linalg.norm(candidate[high, i]-direct[high])/denominator)))
    require(controls and {row['mode'][0] for row in controls} == {0, 1}, 'Need independent low controls of both polarizations')
    maximum = max(c['full_column_relative_error'] for c in controls)
    passed = math.isfinite(maximum) and maximum <= control_gate
    report = dict(method='experimental_reciprocal_order_extension', source=str(source), shell=str(shell),
        old_order=old_order, order=order, ka=ka, parameters=b['parameters'],
        complete_direct_T=False, reciprocity_extension=True, production_enabled=False, completion_proven=False,
        work_counts=extension_work_counts(old_order, order),
        controls=controls, independent_control_gate=control_gate, maximum_control_error=maximum,
        sampled_controls_passed=passed, source_sha256=before, shell_column_sha256=columns.hashes,
        limitations=['Missing high-outgoing/low-incident entries are inferred, not independently solved.',
                     'Sampled column controls do not bound all missing entries or the Mueller error.',
                     'Needs a separate direct full-T control before production use.',
                     'Neither surface mesh accuracy nor modes above the new order are certified.',
                     'The old and new projection grids can differ; low controls include that difference.'])
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'report.json', report)
    require(passed, 'Independent old-column controls reject the completed T')
    theta = np.linspace(0, 180, ntheta)
    mean, metadata = analytic_mueller_tmatrix(candidate, order, theta, precision='fp64', threads=threads,
        cyclic_order=6 if b['parameters']['shape'] == 'prism' and b['parameters']['geometry_digest'] is None else 1)
    old_mean, old_metadata = analytic_mueller_tmatrix(old_t, old_order, theta, precision='fp64', threads=threads,
        cyclic_order=6 if b['parameters']['shape'] == 'prism' and b['parameters']['geometry_digest'] is None else 1)
    require(metadata['Euler_quadrature_nodes'] == old_metadata['Euler_quadrature_nodes'] == 0 and
            not metadata['reciprocity_projection_applied'], 'Unexpected averaging projection')
    old_cross, new_cross = cross_sections(old_t, ka), cross_sections(candidate, ka)
    metrics = error_metrics(old_mean, mean, theta)
    for name in old_cross:
        metrics[name+'_relative_error'] = abs(old_cross[name]/new_cross[name]-1)
    require(all(digest(Path(p)) == h for p,h in {**before, **columns.hashes}.items()), 'Input changed during completion')
    atomic_array(out/'tmatrix.npy', candidate)
    np.savez_compressed(out/'comparison.npz', theta_deg=theta, old=old_mean, extended=mean)
    report.update(analytic_so3=metadata, old_analytic_so3=old_metadata, old_cross_sections=old_cross,
                  extended_cross_sections=new_cross, old_vs_extended=metrics,
                  elapsed_s=time.monotonic()-begin, T_sha256=digest(out/'tmatrix.npy'),
                  status='sampled_controls_passed_independent_full_T_validation_pending')
    atomic_json(out/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--shell', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--control-gate', type=float, default=1e-4)
    parser.add_argument('--ntheta', type=int, default=1801)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    report = assemble(args.source, args.shell, args.out, control_gate=args.control_gate,
                      ntheta=args.ntheta, threads=args.threads)
    print(json.dumps({k:report[k] for k in ('status', 'maximum_control_error', 'old_vs_extended', 'elapsed_s')}))
