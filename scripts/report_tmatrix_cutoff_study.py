#!/usr/bin/env python3
"""Summarize measured T cutoffs and explicitly provisional size extrapolations."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from tmatrix_reference import modes
from tmatrix_symmetry import c2x_plan
from tmatrix_c6 import packets


def resources(order):
    count = 2*order*(order+2)
    t_bytes = count*count*16
    groups = len(packets(modes(order), c2x_plan(order)[2], True))
    storage_factor = .5 if order > 32 else 1.
    dm_bytes = (2*order+3)*3*order*2*order*2*16
    return {'order': order, 'mode_count': count, 'T_complex128_gib': t_bytes/2**30,
            'SO3_T_storage_precision': 'fp32' if order > 32 else 'fp64',
            'three_T_files_gib': 3*t_bytes/2**30,
            'three_T_files_plus_SO3_temporary_gib': (3+storage_factor)*t_bytes/2**30,
            'SO3_estimated_RAM_four_threads_gib': (storage_factor*t_bytes+10*dm_bytes)/2**30,
            'D6h_packets_before_audits_and_fallback': groups}


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, action='append', required=True,
                        help='Directory containing a completed all-integer cutoff scan')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--radius-over-a', type=float, default=1.291648635563488)
    parser.add_argument('--mesh-probes', type=Path,
                        help='JSONL from mesh_capacity_probe with --d6h')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mesh_probes = ([json.loads(line) for line in args.mesh_probes.read_text().splitlines() if line.strip()]
                   if args.mesh_probes else [])
    cases = sorted([(p, json.loads((p/'summary.json').read_text())) for p in args.case], key=lambda x:x[1]['ka'])
    if not all(d['all_integer_orders_tested'] for _, d in cases):
        raise ValueError('use complete integer scans, not only selected cutoffs')
    if len({(d['ri'], d['ri_imag']) for _, d in cases}) != 1:
        raise ValueError('a single material is required for this fit')
    measured = []
    for path, d in cases:
        order = d['minima_relative_to_reference']['0.01']
        at = next(r for r in d['records'] if r['order'] == order)
        before = next((r for r in d['records'] if r['order'] == order-1), None)
        measured.append({'ka': d['ka'], 'ref': d['ref'], 'L_reference': d['reference_order'],
            'L_5pct': d['minima_relative_to_reference']['0.05'], 'L_1pct': order,
            'L_0p1pct': d['minima_relative_to_reference']['0.001'],
            'error_at_L_1pct': at['acceptance_error'],
            'error_at_previous_L': before['acceptance_error'] if before else None,
            'last_reference_shell_error': d['records'][-2]['acceptance_error'],
            'N_1pct': at['mode_count'], 'T_1pct_MiB': at['T_complex128_mib'],
            'source_summary': str((path/'summary.json').resolve())})
    ka = np.array([r['ka'] for r in measured])
    orders = np.array([r['L_1pct'] for r in measured])
    slope, intercept = np.polyfit(ka, orders, 1)
    loo = []
    for i in range(len(ka)):
        mask = np.arange(len(ka)) != i
        a, b = np.polyfit(ka[mask], orders[mask], 1)
        loo.append(float(a*ka[i]+b-orders[i]))
    offset = float(np.max(orders-args.radius_over_a*ka))
    forecasts = []
    for size in (20., 28., 40., 56., 80., 112.):
        x = size*args.radius_over_a
        fitted = math.ceil(slope*size+intercept)
        guarded = math.ceil(x+offset+2)
        old = math.ceil(x+4*x**(1/3)+2)
        resource = resources(guarded)
        mesh = next((m for m in sorted(mesh_probes, key=lambda m:m['ref'])
                     if m['ka_at_8_equivalent_P2_points_per_inner_wavelength'] >= size), None)
        forecasts.append({'ka': size, 'L_linear_extrapolation': fitted,
            'L_with_empirical_radius_offset_and_two_extra_orders': guarded,
            'L_old_initial_estimate': old,
            **{k:v for k,v in resource.items() if k!='order'},
            'mesh_ref_from_resolution_heuristic': mesh['ref'] if mesh else None,
            'surface_system_dofs': mesh['system_dofs'] if mesh else None,
            'storage_reduction_vs_old_estimate': resources(old)['T_complex128_gib']/resource['T_complex128_gib'],
            'extrapolation_beyond_measured_sizes': bool(size > ka.max())})
    result = {'geometry': 'regular hexagonal prism, H/D_flats=1, equal-volume radius a',
        'ri': cases[0][1]['ri'], 'ri_imag': cases[0][1]['ri_imag'],
        'primary_tolerance': .01, 'primary_metric': cases[0][1]['metric'],
        'measured': measured,
        'linear_fit': {'L_equals_slope_times_ka_plus_intercept': [float(slope), float(intercept)],
                       'measured_ka_interval': [float(ka.min()), float(ka.max())],
                       'rms_L_residual': float(np.sqrt(np.mean((slope*ka+intercept-orders)**2))),
                       'leave_one_out_L_errors': loo},
        'provisional_planning_rule': {'formula': 'ceil((R/a)*ka + offset + 2)',
                    'R_over_a': args.radius_over_a, 'offset': offset,
                    'extra_orders': 2, 'certified_error_bound': False},
        'forecasts': forecasts,
        'mesh_probes': mesh_probes,
        'limitations': ['The minimum is relative to each finite reference T on the same surface mesh.',
                       'The criterion concerns the orientation-averaged Mueller matrix, not every fixed orientation.',
                       'Every output angle is evaluated analytically on a finite grid; this is not Euler quadrature.',
                       'The fitted law is empirical for this shape and refractive index only.',
                       'Extrapolated L values are starting estimates requiring new convergence checks.',
                       'The 8-point surface-resolution heuristic is not a mesh-convergence certificate.',
                       'Storage estimates exclude near/MBJ caches, surface data and logs.',
                       'Forecast SO(3) uses auto storage precision (fp32 above L=32), with fp64 contractions.',
                       'Packet counts exclude audits/fallback and do not predict total wall time.']}
    (args.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    write_csv(args.out/'measured.csv', measured)
    write_csv(args.out/'forecasts.csv', forecasts)
    make_plots(cases, result, args.out)
    print(json.dumps(result, indent=2))


def make_plots(cases, report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout='constrained')
    fig.suptitle('Минимальный порядок T-матрицы и зависимость от размера\nГексагональная призма, m=1.3; аналитическое SO(3)', fontsize=14)
    ax = axes[0, 0]
    for _, d in cases:
        ax.semilogy([r['order'] for r in d['records']],
                    [max(r['acceptance_error'], 1e-12) for r in d['records']], '.-', label=f"ka={d['ka']:g}")
    ax.axhline(.01, color='black', ls='--', lw=1, label='Допуск 1%')
    ax.set(xlabel='Порядок L', ylabel='Ошибка относительно полного эталона', ylim=(1e-5, 2), title='Измеренная сходимость')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    measured = report['measured']; x = np.array([r['ka'] for r in measured])
    for key, label in (('L_5pct', '5%'), ('L_1pct', '1%'), ('L_0p1pct', '0.1%')):
        ax.plot(x, [r[key] for r in measured], 'o-', label=label)
    ax.set(xlabel='ka', ylabel='Минимальный L', title='Измерения на полных T-матрицах')
    ax.legend()
    ax = axes[1, 0]
    forecasts = report['forecasts']; fx = [r['ka'] for r in forecasts]
    for key, label in (('L_linear_extrapolation', 'Линейная экстраполяция'),
                       ('L_with_empirical_radius_offset_and_two_extra_orders', 'Оценка с запасом +2'),
                       ('L_old_initial_estimate', 'Оценка kR + 4∛(kR) + 2')):
        ax.plot(fx, [r[key] for r in forecasts], '--o', ms=4, label=label)
    ax.plot(x, [r['L_1pct'] for r in measured], 'ko', label='Измеренный минимум 1%')
    ax.axvspan(x.max(), fx[-1], color='gray', alpha=.08)
    ax.set(xlabel='ka', ylabel='L', title='Прогноз: выше ka=20 пока не проверен')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.semilogy(fx, [r['T_complex128_gib'] for r in forecasts], 'o-', label='Одна T, complex128')
    ax.semilogy(fx, [r['three_T_files_plus_SO3_temporary_gib'] for r in forecasts], 'o-', label='T + kernel + столбцы + временный файл')
    ax.set(xlabel='ka', ylabel='ГиБ', title='Объём при оценке L с запасом')
    ax.legend(fontsize=8)
    for ext in ('png', 'pdf'): fig.savefig(out/f'cutoff_and_size_scaling.{ext}', dpi=170)
    plt.close(fig)
    for path, d in cases:
        if d['ka'] != 20: continue
        selected = [26, d['minima_relative_to_reference']['0.01'], 30, d['reference_order']]
        with np.load(path/f"L{d['reference_order']:03d}.npz") as z:
            theta, reference = z['theta_deg'], z['mueller']
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
        entries = [(0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, 3)]
        for order in selected:
            with np.load(path/f'L{order:03d}.npz') as z: mean = z['mueller']
            for ax, (i,j) in zip(axes.flat, entries):
                if (i,j) == (0,0): ax.semilogy(theta, mean[:,i,j], label=f'L={order}', lw=1.2)
                else: ax.plot(theta, mean[:,i,j]/reference[:,0,0], label=f'L={order}', lw=1.2)
                ax.set(xlim=(0,180), xlabel='Угол рассеяния θ, °',
                       ylabel=f'M{i+1}{j+1}' if (i,j)==(0,0) else f'M{i+1}{j+1}/M₁₁ эталона')
        axes[0,0].legend()
        fig.suptitle('ka=20: усечение T по входным и выходным модам\nЕдиный масштаб; эталон L=44, минимальный L=27 при допуске 1%')
        for ext in ('png','pdf'): fig.savefig(out/f'ka20_cutoff_overlay.{ext}', dpi=170)
        plt.close(fig)


if __name__ == '__main__': main()
