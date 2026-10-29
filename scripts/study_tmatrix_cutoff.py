#!/usr/bin/env python3
"""Measure modal truncation against a complete T matrix with analytic SO(3).

Rows AND columns are selected by (polarization, l, m), not by a leading
square slice. The reference keeps the same BEM discretization. This study
does not measure the mesh error or modes beyond the reference cutoff.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from tmatrix_reference import modes
from tmatrix_so3_mstm import analytic_mueller_tmatrix


def cutoff_indices(full_order, order):
    if not 1 <= order <= full_order:
        raise ValueError('cutoff must lie inside the reference basis')
    return np.array([i for i, (_, degree, _) in enumerate(modes(full_order))
                     if degree <= order], dtype=int)


def error_metrics(candidate, reference, theta):
    intensity = reference[:, 0, 0]
    if not np.isfinite(reference).all() or np.any(intensity <= 0):
        raise ValueError('reference M11 must be finite and positive')
    delta = np.abs(candidate-reference)
    scaled = delta/intensity[:, None, None]
    index = np.unravel_index(np.argmax(scaled), scaled.shape)
    # A separate diagnostic for nonzero elements; it is not the primary gate.
    active = np.abs(reference) >= .01*intensity[:, None, None]
    return {
        'maximum_all16_error_over_reference_M11': float(np.max(scaled)),
        'maximum_M11_relative_error': float(np.max(scaled[:, 0, 0])),
        'forward_M11_relative_error': float(scaled[0, 0, 0]),
        'backward_M11_relative_error': float(scaled[-1, 0, 0]),
        'maximum_active_element_relative_error': float(np.max(delta[active]/np.abs(reference[active]))),
        'active_element_floor_over_M11': .01,
        'worst_theta_deg': float(theta[index[0]]),
        'worst_entry': f'M{index[1]+1}{index[2]+1}',
    }


def stable_minimum(records, tolerance, key='acceptance_error'):
    """Require every larger tested cutoff to pass, not an accidental crossing."""
    tail_maximum = 0.
    selected = None
    for record in reversed(sorted(records, key=lambda x: x['order'])):
        tail_maximum = max(tail_maximum, record[key])
        if tail_maximum <= tolerance:
            selected = record['order']
    return selected


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def analyze(run, out, ntheta=1801, threads=4, orders=None):
    run, out = run.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run/'summary.json').read_text())
    parameters = summary['parameters']
    if not summary['complete']:
        raise ValueError('reference T matrix is incomplete')
    full_order, ka = parameters['order'], parameters['ka']
    cyclic = 6 if parameters['shape'] == 'prism' and not parameters.get('obj') else 1
    theta = np.linspace(0, 180, ntheta)
    source = run/'tmatrix.npy'
    root = Path(__file__).resolve().parents[1]
    identity = {'T_sha256': digest(source), 'reference_order': full_order,
                'ntheta': ntheta, 'cyclic_order': cyclic, 'precision': 'fp64',
                'implementation_sha256': {str(p.relative_to(root)): digest(p) for p in (
                    Path(__file__).resolve(), root/'scripts/tmatrix_so3_mstm.py',
                    root/'scripts/tmatrix_reference.py', root/'bin/tmatrix_so3_mstm')}}
    identity_path = out/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('study source/settings changed; choose a new output directory')
    atomic_json(identity_path, identity)
    t = np.load(source, mmap_mode='r')
    expected = 2*full_order*(full_order+2)
    if t.shape != (expected, expected):
        raise ValueError('reference basis dimensions do not match the order')

    def evaluate(order):
        cache = out/f'L{order:03d}.npz'
        meta = out/f'L{order:03d}.json'
        if cache.exists() and meta.exists():
            with np.load(cache) as data:
                return data['mueller'], json.loads(meta.read_text())
        indices = cutoff_indices(full_order, order)
        reduced = np.asarray(t[np.ix_(indices, indices)])
        mean, metadata = analytic_mueller_tmatrix(reduced, order, theta,
            precision='fp64', threads=threads, cyclic_order=cyclic)
        power = float(np.vdot(reduced, reduced).real)
        metadata.update(scattering_cross_section=2*np.pi/ka**2*power,
                        extinction_cross_section=float(-2*np.pi/ka**2*np.trace(reduced).real),
                        T_frobenius_squared=power)
        temporary = cache.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, theta_deg=theta, mueller=mean)
        temporary.replace(cache)
        atomic_json(meta, metadata)
        return mean, metadata

    reference, ref_meta = evaluate(full_order)
    requested = sorted(set(orders or range(1, full_order+1)) | {full_order})
    records = []
    for order in requested:
        begin = time.monotonic()
        mean, metadata = evaluate(order)
        record = {'order': order, 'mode_count': 2*order*(order+2),
                  **error_metrics(mean, reference, theta)}
        for observable in ('scattering_cross_section', 'extinction_cross_section'):
            record[observable] = metadata[observable]
            record[observable+'_relative_error'] = abs(metadata[observable]-ref_meta[observable])/abs(ref_meta[observable])
        missing_power = max(0., 1-metadata['T_frobenius_squared']/ref_meta['T_frobenius_squared'])
        record['T_relative_frobenius_error'] = float(np.sqrt(missing_power))
        record['acceptance_error'] = max(record['maximum_all16_error_over_reference_M11'],
            record['scattering_cross_section_relative_error'], record['extinction_cross_section_relative_error'])
        record['T_complex128_mib'] = record['mode_count']**2*16/2**20
        records.append(record)
        atomic_json(out/'progress.json', {'source_run': str(run), 'ka': ka, 'records': records})
        print(json.dumps({'ka': ka, 'L': order, 'error': record['acceptance_error'],
                          'worst_theta': record['worst_theta_deg'], 'elapsed_s': time.monotonic()-begin}), flush=True)
    minima = {str(tolerance): stable_minimum(records, tolerance) for tolerance in (.05, .01, .001)}
    report = {'source_run': str(run), 'source_T_sha256': identity['T_sha256'],
        'ka': ka, 'ri': parameters['ri'], 'ri_imag': parameters.get('ri_imag', 0.),
        'ref': parameters['ref'], 'reference_order': full_order,
        'reference_mode_count': expected, 'tested_orders': requested,
        'all_integer_orders_tested': requested == list(range(1, full_order+1)),
        'theta_step_deg': 180/(ntheta-1), 'Euler_quadrature_nodes': 0,
        'metric': 'max over output theta and all 16 entries of abs(M_L-M_ref)/M11_ref, plus relative Csca and Cext',
        'minima_relative_to_reference': minima, 'records': records,
        'limitations': ['Same surface mesh for every truncation; mesh error is not measured.',
                        'Modes above the reference L are not assessed by truncation.',
                        'Maxima are measured on the stated output-angle grid, not a continuous-angle proof.',
                        'SO(3) orientation integration is analytic; theta is the output scattering angle.']}
    atomic_json(out/'summary.json', report)
    with (out/'convergence.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    plot_report(report, out)
    return report


def plot_report(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = report['records']
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    for key, label in (('maximum_all16_error_over_reference_M11', 'Все 16 элементов / M₁₁ эталона'),
                       ('scattering_cross_section_relative_error', 'Сечение рассеяния'),
                       ('T_relative_frobenius_error', 'Норма отброшенной части T')):
        ax.semilogy([r['order'] for r in records], [max(r[key], 1e-16) for r in records], '.-', label=label)
    for tolerance in (.05, .01, .001):
        ax.axhline(tolerance, color='gray', lw=.7, ls='--')
        ax.text(records[0]['order'], tolerance*1.15, f'{100*tolerance:g}%', fontsize=9)
    ax.set(xlabel='Максимальный порядок L', ylabel='Относительная ошибка', ylim=(1e-10, 3),
           title=f"ka={report['ka']:g}: сходимость к L={report['reference_order']}\nПолное аналитическое SO(3); одна и та же сетка BEM")
    ax.grid(alpha=.25); ax.legend(fontsize=9)
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'convergence.{suffix}', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ntheta', type=int, default=1801)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--orders', type=int, nargs='+')
    args = parser.parse_args()
    if args.ntheta < 2: parser.error('ntheta must be at least two')
    report = analyze(args.run, args.out, args.ntheta, args.threads, args.orders)
    print(json.dumps({'ka': report['ka'], 'minima': report['minima_relative_to_reference']}), flush=True)
