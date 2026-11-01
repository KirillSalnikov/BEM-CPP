#!/usr/bin/env python3
"""Summarize completed aspect-study cases without modifying solver checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from study_tmatrix_aspect import average
from study_tmatrix_cutoff import cutoff_indices, error_metrics, atomic_json, stable_minimum
from tmatrix_reference import numerical_so3_mueller
from tmatrix_so3_mstm import analytic_mueller_tmatrix


def plot_all16(case_out, row, theta, coarse, fine):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    intensity = fine[:,0,0]
    error = 100*np.abs(coarse-fine)/intensity[:,None,None]
    np.savez_compressed(case_out/'mesh_comparison_mueller.npz', theta_deg=theta,
                        coarse_mueller=coarse, fine_mueller=fine,
                        coarse_ref=row['coarse_ref'], fine_ref=row['fine_ref'],
                        common_order=row['common_mesh_comparison_L'])
    fig, axes = plt.subplots(4,4,figsize=(13,10),layout='constrained')
    for i in range(4):
        for j in range(4):
            ax = axes[i,j]
            denominator = 1. if (i,j)==(0,0) else intensity
            c, f = coarse[:,i,j]/denominator, fine[:,i,j]/denominator
            ax.plot(theta,c,'--',lw=1.,color='tab:orange',label=f"ref={row['coarse_ref']}")
            ax.plot(theta,f,lw=1.,color='tab:blue',label=f"ref={row['fine_ref']}")
            peak = int(np.argmax(error[:,i,j]))
            ax.plot(theta[peak],f[peak],'o',ms=3,color='purple')
            ax.text(.03,.97,f"макс. Δ/M₁₁: {error[peak,i,j]:.3g}%\nθ={theta[peak]:g}°",
                    transform=ax.transAxes,va='top',fontsize=7,
                    bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':1})
            label = f'M{i+1}{j+1}'
            if (i,j)==(0,0):
                ax.set_yscale('log')
            else:
                label += ' / M₁₁ эталона'
                if max(np.max(np.abs(c)),np.max(np.abs(f)))<1e-8:
                    ax.set_ylim(-1e-6,1e-6)
            ax.set_title(label,fontsize=10)
            ax.set_xlim(0,180); ax.grid(alpha=.2)
            if i==3: ax.set_xlabel('θ, град.')
    axes[0,0].legend(loc='lower left',fontsize=8)
    fig.suptitle(f"ka={row['ka']:g}, H/D={row['aspect']:g}; аналитическое SO(3), все 16 элементов\n"
                 f"n=1.3116, λ=0.532 мкм, L={row['common_mesh_comparison_L']}; M₁₁ в исходных единицах, остальные / M₁₁ более мелкой сетки")
    fig.savefig(case_out/'mesh_all16.png',dpi=170)
    fig.savefig(case_out/'mesh_all16.pdf')
    plt.close(fig)


def independent_average_check(run, out, requested_order, max_order=12):
    path = out/'so3_independent_check.json'
    summary = json.loads((run/'summary.json').read_text())
    order = min(max_order, requested_order)
    identity = {'source_run': str(run), 'source_signature': summary['signature'], 'order': order}
    if path.exists():
        old = json.loads(path.read_text())
        if all(old.get(k)==v for k,v in identity.items()):
            return old
    ids = cutoff_indices(summary['parameters']['order'], order)
    t = np.load(run/'tmatrix.npy', mmap_mode='r')[np.ix_(ids, ids)]
    kernel = np.load(run/'kernel.npy', mmap_mode='r')[np.ix_(ids, ids)]
    theta = np.array([0., 15., 30., 60., 90., 120., 150., 165., 180.])
    start = time.monotonic()
    numerical = summary['parameters']['ka']**2*numerical_so3_mueller(kernel, order, theta)
    analytic, _ = analytic_mueller_tmatrix(t, order, theta, precision='fp64', threads=4, cyclic_order=1)
    cyclic, _ = analytic_mueller_tmatrix(t, order, theta, precision='fp64', threads=4, cyclic_order=6)
    report = {**identity, 'theta': theta.tolist(),
              'orientation_nodes': (2*order+1)*(4*order+1)**2,
              'analytic_vs_cubature': error_metrics(analytic, numerical, theta),
              'cyclic_projection_vs_unprojected': error_metrics(cyclic, analytic, theta),
              'elapsed_s': time.monotonic()-start,
              'scope': 'Independent cubature of the same truncated T; not a test of the BEM mesh or omitted higher modes.'}
    atomic_json(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-so3', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    plan = json.loads((out/'plan.json').read_text())
    records = []
    curves = []
    active_curves = {}
    for case in plan['cases']:
        by_ref = {}
        for p in (out/case['name']).glob('ref*/summary.json'):
            s = json.loads(p.read_text())
            if s.get('complete'):
                ref = s['parameters']['ref']
                old = by_ref.get(ref)
                if old is None or s['parameters']['order']>old[1]['parameters']['order']:
                    by_ref[ref] = (p.parent, s)
        completed = [by_ref[r] for r in sorted(by_ref)]
        if len(completed)<2:
            continue
        fine_run, fine = completed[-1]
        coarse_run, coarse = completed[-2]
        order = min(fine['parameters']['order'], coarse['parameters']['order'])
        theta, f = average(fine_run, order)
        tcoarse, c = average(coarse_run, order)
        assert np.array_equal(theta, tcoarse)
        metrics = error_metrics(c, f, theta)
        active = np.abs(f) >= .01*f[:,0,0,None,None]
        element_relative = np.divide(np.abs(c-f), np.abs(f),
                                     out=np.zeros_like(f), where=active)
        active_worst = np.unravel_index(np.argmax(element_relative), f.shape)
        active_curves[case['name']] = 100*np.max(element_relative, axis=(1,2))
        cutoff_path = out/case['name']/'cutoff/summary.json'
        if not cutoff_path.exists():
            continue
        cutoff = json.loads(cutoff_path.read_text())
        if cutoff['source_run']!=str(fine_run):
            continue
        minimum = cutoff['minima_relative_to_reference']['0.01']
        active_minimum = stable_minimum(cutoff['records'], .01,
                                       key='maximum_active_element_relative_error')
        minimum_record = next(r for r in cutoff['records'] if r['order']==minimum)
        tail = next(r for r in cutoff['records'] if r['order']==fine['parameters']['order']-2)
        row = {'case': case['name'], 'ka': case['ka'], 'aspect': case['height_over_width'],
               'R_over_a': case['circumscribing_radius_over_a'],
               'coarse_ref': coarse['parameters']['ref'], 'fine_ref': fine['parameters']['ref'],
               'common_mesh_comparison_L': order, 'reference_L': fine['parameters']['order'],
               'minimum_L_1pct_over_M11': minimum,
               'minimum_L_1pct_active_elements': active_minimum,
               'active_element_floor_over_M11': .01,
               'cutoff_error_pct': 100*minimum_record['acceptance_error'],
               'last_two_degrees_error_pct': 100*tail['acceptance_error'],
               'mesh_change_pct': 100*metrics['maximum_all16_error_over_reference_M11'],
               'mesh_change_below_1pct_over_M11': metrics['maximum_all16_error_over_reference_M11'] < .01,
               'mesh_active_element_relative_pct': 100*metrics['maximum_active_element_relative_error'],
               'mesh_active_worst_entry': f'M{active_worst[1]+1}{active_worst[2]+1}',
               'mesh_active_worst_theta_deg': float(theta[active_worst[0]]),
               'mesh_worst_theta_deg': metrics['worst_theta_deg'], 'mesh_worst_entry': metrics['worst_entry'],
               'fine_dofs': fine['operator']['system_dofs'],
               'plane_reconstruction_error_pct': 100*fine['new_direction_field_error'],
               'energy_balance_error_pct': 100*abs(fine['extinction_cross_section']/fine['scattering_cross_section']-1),
               'source_run': str(fine_run)}
        if args.check_so3:
            check = independent_average_check(fine_run, out/case['name'], minimum)
            row['analytic_vs_cubature_relative'] = check['analytic_vs_cubature']['maximum_all16_error_over_reference_M11']
            row['independent_cubature_L'] = check['order']
        records.append(row)
        curves.append((row, theta, 100*np.max(np.abs(c-f), axis=(1,2))/f[:,0,0], cutoff))
        plot_all16(out/case['name'], row, theta, c, f)
    report = {'parameters': {'ri': plan['ri'], 'wavelength_um': plan['wavelength_um'],
              'shape': plan['shape'], 'ka_definition': plan['ka_definition']},
              'completed_cases': len(records), 'planned_cases': len(plan['cases']),
              'records': records, 'limitations': plan['limitations'],
              'percentage_definition': '100*max_theta_ij abs(delta_Mij)/M11_reference; not relative error of each small element.'}
    atomic_json(out/'report.json', report)
    if not records:
        return
    with (out/'report.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14,7), layout='constrained', squeeze=False)
    for ax, (row, theta, error, cutoff) in zip(axes.flat, curves):
        ax.semilogy(theta, np.maximum(error,1e-8), lw=1.1, label='Все 16 / M₁₁')
        ax.semilogy(theta, np.maximum(active_curves[row['case']],1e-8), lw=.9,
                    label='От элемента, |Mᵢⱼ| ≥ 0.01 M₁₁')
        ax.axhline(1., color='crimson', ls='--', lw=.8)
        ax.plot(row['mesh_worst_theta_deg'], row['mesh_change_pct'], 'o', color='crimson', ms=4)
        ax.set(title=f"ka={row['ka']:g}, H/D={row['aspect']:g}; ref {row['coarse_ref']}→{row['fine_ref']}\nмакс. / M₁₁: {row['mesh_change_pct']:.3g}%, {row['mesh_worst_entry']}",
               xlabel='Угол рассеяния, град.', ylabel='Изменение, %', xlim=(0,180))
        ax.grid(alpha=.25); ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(curves):]:
        ax.set_visible(False)
    fig.suptitle('BEM → T → аналитическое SO(3): изменение при сгущении сетки\nn=1.3116; λ=0.532 мкм; одинаковый L в каждом сравнении')
    fig.savefig(out/'mesh_convergence.png', dpi=170)
    fig.savefig(out/'mesh_convergence.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1,2, figsize=(11,4.5), layout='constrained')
    for ka, ax in zip(sorted({r['ka'] for r in records}), axes):
        for row, _, _, cutoff in curves:
            if row['ka']!=ka:
                continue
            r = cutoff['records'][:-1]
            ax.semilogy([x['order'] for x in r], [max(100*x['acceptance_error'],1e-12) for x in r], '.-', label=f"H/D={row['aspect']:g}")
        ax.axhline(1., color='crimson', ls='--', lw=.8)
        ax.set(title=f'ka={ka:g}', xlabel='Порядок L', ylabel='Изменение к полному L, %', ylim=(1e-5,150))
        ax.grid(alpha=.25); ax.legend()
    fig.suptitle('Усечение T: все 16 элементов после аналитического SO(3), ошибка / M₁₁')
    fig.savefig(out/'cutoff_convergence.png', dpi=170)
    fig.savefig(out/'cutoff_convergence.pdf')
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
