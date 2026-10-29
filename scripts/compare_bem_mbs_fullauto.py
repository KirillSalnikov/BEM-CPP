#!/usr/bin/env python3
"""Compare BEM analytic SO(3) to a converged native MBS fullauto run.

MBS exports dimensional d_sigma/d_Omega; BEM exports k^2 d_sigma/d_Omega.
Stokes conventions are related by D M D, D=diag(1,1,-1,-1), from the
respective Jones-to-Mueller formulas. No fitted scale or sign is used.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tmatrix_so3_mstm import analytic_mueller_tmatrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bem', type=Path, required=True)
    p.add_argument('--mbs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    status = json.loads((args.mbs / 'result/fullauto_status.json').read_text())
    audit = json.loads((args.mbs / 'result/independent_final_validation.json').read_text())
    if status['status'] != 'converged' or not audit['passed']:
        raise RuntimeError('MBS fullauto must pass production and independent validation')
    manifest = json.loads((args.mbs.parent / 'manifest.json').read_text())
    summary = json.loads((args.bem / 'summary.json').read_text())
    original = json.loads((args.bem / 'average.json').read_text())
    assert summary['complete'] and original['Euler_quadrature_nodes'] == 0
    table = np.loadtxt(args.mbs / 'result/mueller_fullauto.dat', skiprows=1)
    theta, weights = table[:, 0], table[:, 1]
    assert table.shape[1] == 18 and theta[0] == 0 and theta[-1] == 180
    assert np.all(np.diff(theta) > 0) and np.isclose(weights.sum(), 4*np.pi)
    k = 2*np.pi/manifest['wavelength']
    assert np.isclose(k*manifest['equiv_radius'], original['ka'])
    stokes = np.array([1., 1., -1., -1.])
    mbs = table[:, 2:].reshape(-1, 4, 4)*k**2*stokes[None, :, None]*stokes[None, None, :]
    bem, meta = analytic_mueller_tmatrix(
        np.load(args.bem / 'tmatrix.npy', mmap_mode='r'), original['order'], theta,
        precision='fp64', threads=4, cyclic_order=original['cyclic_order'])
    assert np.all(np.isfinite(mbs)) and np.all(np.isfinite(bem))
    assert np.all(mbs[:, 0, 0] > 0) and np.all(bem[:, 0, 0] > 0)
    error = np.abs(mbs-bem)/bem[:, 0, 0, None, None]
    ratio_error = np.abs(mbs/mbs[:, 0, 0, None, None]-bem/bem[:, 0, 0, None, None])
    worst = np.unravel_index(error.argmax(), error.shape)
    with (args.mbs / 'result/mueller_fullauto_integrals.tsv').open() as f:
        integral = next(csv.DictReader(f, delimiter='\t'))
    supervision = json.loads((args.mbs / 'supervision/status.json').read_text())
    command = json.loads((args.mbs / 'launch_metadata.json').read_text())['command']
    tolerance = float(command[command.index('--fullauto')+1])
    tag = f'{100*tolerance:g}pct'
    components = {}
    for i in range(4):
        for j in range(4):
            q = int(error[:, i, j].argmax())
            components[f'M{i+1}{j+1}'] = {
                'max_absolute_difference_over_BEM_M11': float(error[q, i, j]),
                'theta_at_max_deg': float(theta[q]),
                'max_difference_of_Mij_over_own_M11': float(ratio_error[:, i, j].max())}
    report = {
        'mbs_commit': manifest['source_commit'], 'mbs_tolerance': tolerance, 'mbs_status': status,
        'mbs_independent_validation': audit, 'mbs_wall_s': supervision['elapsed_s'],
        'conversion': {'MBS_to_BEM_scale': float(k**2), 'Stokes_D': stokes.tolist(),
                       'fitted_normalization': False, 'units': 'k^2 d_sigma/d_Omega'},
        'bem_reevaluation': meta, 'angles': len(theta),
        'all_angles_all16_within_5pct_BEM_M11': bool(np.all(error <= .05)),
        'worst_difference_over_BEM_M11': float(error[worst]),
        'worst_theta_deg': float(theta[worst[0]]),
        'worst_component': f'M{worst[1]+1}{worst[2]+1}',
        'M11_angles_within_5pct': int(np.sum(error[:, 0, 0] <= .05)),
        'all16_angles_within_5pct': int(np.sum(error.max(axis=(1, 2)) <= .05)),
        'weighted_relative_L2_all16': float(np.sqrt(np.sum(weights[:, None, None]*(mbs-bem)**2)/np.sum(weights[:, None, None]*bem**2))),
        'M11_relative_forward_signed': float(mbs[0, 0, 0]/bem[0, 0, 0]-1),
        'M11_relative_backward_signed': float(mbs[-1, 0, 0]/bem[-1, 0, 0]-1),
        'BEM_Csca': summary['scattering_cross_section'],
        'BEM_Cext': summary['extinction_cross_section'],
        'MBS_integral_output': integral,
        'MBS_Csca_note': 'For nonabsorbing MBS output Csca=Cext by balance; angular integral is reported separately.',
        'components': components,
    }
    (args.out / 'comparison_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    names = [f'M{i+1}{j+1}' for i in range(4) for j in range(4)]
    header = ['theta_deg'] + [prefix+n for prefix in ('BEM_', 'MBS_', 'absdiff_over_BEM_M11_') for n in names]
    np.savetxt(args.out / 'comparison_all16.csv', np.column_stack((theta, bem.reshape(-1, 16), mbs.reshape(-1, 16), error.reshape(-1, 16))), delimiter=',', header=','.join(header), comments='', fmt='%.12e')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .23, 'savefig.facecolor': 'white'})
    entries = [(0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, 3)]
    curves = [(bem, '#1765a3', '-', 'BEM · T-матрица, аналитическое SO(3)'),
              (mbs, '#d75c22', '--', f'MBS-fast · fullauto, допуск {100*tolerance:g}%')]

    def save(fig, name):
        fig.savefig(args.out / (name+'.png'), dpi=170, bbox_inches='tight')
        fig.savefig(args.out / (name+'.pdf'), bbox_inches='tight')
        plt.close(fig)

    for mode in ('absolute', 'common_M11'):
        fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.6))
        fig.subplots_adjust(left=.075, right=.985, bottom=.15, top=.86, hspace=.34, wspace=.30)
        fig.suptitle('BEM и MBS-fast · общая нормировка, без подгонки\n'
                     r'$ka=20$, $m=1.3+0i$ · гексагональная призма $H/D_{\rm flats}=1$', fontsize=15)
        for ax, (i, j) in zip(axes.flat, entries):
            for values, color, style, label in curves:
                y = values[:, i, j]
                if mode == 'common_M11' and (i, j) != (0, 0):
                    y = y/bem[:, 0, 0]
                ax.plot(theta, y, style, color=color, lw=1.6, label=label)
            if (i, j) == (0, 0) or (mode == 'absolute' and (i, j) == (1, 1)):
                ax.set_yscale('log')
            elif mode == 'absolute':
                ax.set_yscale('symlog', linthresh=5.)
                ymin, ymax = ax.get_ylim()
                ticks = [-10.**n for n in range(5, 0, -1)] + [0.] + [10.**n for n in range(1, 6)]
                ax.set_yticks([v for v in ticks if ymin <= v <= ymax])
            ax.axhline(0, color='#444444', lw=.5, alpha=.4)
            label = rf'$M_{{{i+1}{j+1}}}$'
            if mode == 'common_M11' and (i, j) != (0, 0):
                label = rf'$M_{{{i+1}{j+1}}}/M_{{11}}^{{\rm BEM}}$'
            ax.set(xlim=(0, 180), xticks=np.arange(0, 181, 30), xlabel='Угол рассеяния θ, °', ylabel=label)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .045), ncol=2, fontsize=11, frameon=False)
        foot = ('Все элементы в единицах k² dσ/dΩ; для знакопеременных — симметричная логарифмическая шкала'
                if mode == 'absolute' else 'M₁₁ в единицах k² dσ/dΩ; остальные: обе кривые делятся на один M₁₁ BEM')
        fig.text(.5, .018, foot, fontsize=10, ha='center')
        save(fig, 'bem_mbs_'+tag+'_'+mode)

    fig, ax = plt.subplots(figsize=(11.5, 5.2), layout='constrained')
    ax.plot(theta, 100*error[:, 0, 0], color='#d75c22', label='|ΔM₁₁| / M₁₁ BEM')
    ax.plot(theta, 100*error.max(axis=(1, 2)), color='#1765a3', ls='--', label='max по всем 16: |ΔMᵢⱼ| / M₁₁ BEM')
    ax.axhline(5, color='#555555', ls=':', label='5% расхождения с BEM')
    ax.set(xlim=(0, 180), xticks=np.arange(0, 181, 15), xlabel='Угол рассеяния θ, °',
           ylabel='Расхождение, % от M₁₁ BEM', title='Расхождение методов после усреднения · ka = 20')
    ax.legend(fontsize=10)
    save(fig, 'bem_mbs_'+tag+'_error')
    print(json.dumps({k:v for k,v in report.items() if k not in ('mbs_status','mbs_independent_validation','components','bem_reevaluation')}, indent=2))


if __name__ == '__main__':
    main()
