#!/usr/bin/env python3
"""Freeze the current MBS snapshot and overlay it on the saved BEM/5% comparison."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = args.run / 'result'
    status = json.loads((root / 'fullauto_status.json').read_text())
    converged = status['status'] == 'converged'
    if converged:
        snapshot = {'converged': True, 'round': status['round'],
                    'combined_output95': status['combined_error95'], 'snapshot_folder': str(root)}
    else:
        snapshot = json.loads((root / 'intermediate_current.json').read_text())
    source = Path(snapshot['snapshot_folder']) / 'mueller_fullauto_annotated.csv'
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    frozen = args.out / 'progress_snapshots' / (stamp + '_round' + str(snapshot['round']))
    frozen.mkdir(parents=True, exist_ok=False)
    (frozen / source.name).write_bytes(raw)
    (frozen / 'snapshot_metadata.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    (frozen / 'fullauto_status_at_capture.json').write_text(json.dumps(status, indent=2) + '\n')
    rows = list(csv.DictReader(raw.decode().splitlines()))
    names = [f'M{i+1}{j+1}' for i in range(4) for j in range(4)]
    theta = np.array([float(row['theta_deg']) for row in rows])
    current = np.array([[float(row[n]) for n in names] for row in rows]).reshape(-1, 4, 4)
    baseline = np.loadtxt(args.baseline, delimiter=',', skiprows=1)
    # Use the already evaluated analytic BEM values; never interpolate to align grids.
    if baseline.shape[0] != len(theta) or not np.allclose(baseline[:, 0], theta, rtol=0, atol=1e-9):
        raise RuntimeError('Different angular grids: analytic BEM reevaluation is required')
    bem, previous = baseline[:, 1:17].reshape(-1, 4, 4), baseline[:, 17:33].reshape(-1, 4, 4)
    manifest = json.loads((args.run.parent / 'manifest.json').read_text())
    scale = (2*np.pi/manifest['wavelength'])**2
    d = np.array([1., 1., -1., -1.])
    current *= scale*d[None, :, None]*d[None, None, :]
    assert np.all(np.isfinite(current)) and np.all(current[:, 0, 0] > 0)
    output_error = max(float(row['max_combined95']) for row in rows)
    assert np.isclose(output_error, snapshot['combined_output95'], rtol=1e-10)
    difference = np.abs(current-previous)/previous[:, 0, 0, None, None]
    bem_error = np.abs(current-bem)/bem[:, 0, 0, None, None]
    report = {'captured_at_utc': datetime.now(timezone.utc).isoformat(), 'converged': converged,
              'round': snapshot['round'], 'source': str(source), 'source_sha256': digest,
              'snapshot': snapshot, 'current_estimated_combined_error': output_error,
              'comparison_angles': len(theta), 'MBS_to_BEM_scale': scale,
              'normalization': 'k^2 d_sigma/d_Omega, no fitted scale; D=diag(1,1,-1,-1)',
              'max_change_from_5pct_all16_scaled_previous_M11': float(difference.max()),
              'max_change_from_5pct_M11_relative': float(difference[:, 0, 0].max()),
              'max_difference_from_BEM_all16_scaled_BEM_M11': float(bem_error.max()),
              'M11_difference_from_BEM_forward_signed': float(current[0, 0, 0]/bem[0, 0, 0]-1),
              'M11_difference_from_BEM_backward_signed': float(current[-1, 0, 0]/bem[-1, 0, 0]-1)}
    (frozen / 'plot_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savetxt(frozen / 'all_three_all16.csv', np.column_stack((theta, bem.reshape(-1, 16), previous.reshape(-1, 16), current.reshape(-1, 16))), delimiter=',', comments='',
               header=','.join(['theta_deg']+[prefix+n for prefix in ('BEM_', 'MBS_5pct_', 'MBS_target_1pct_current_') for n in names]), fmt='%.12e')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .23})
    entries = [(0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, 3)]
    current_label = 'MBS 1% · готово' if converged else 'MBS цель 1% · промежуточный'
    curves = [(bem, '#1765a3', '-', 'BEM · аналитическое SO(3)'),
              (previous, '#d75c22', '--', 'MBS 5% · готово'),
              (current, '#18844c', (0, (3, 1, 1, 1)), current_label)]

    def save(fig, stem):
        for ext in ('png', 'pdf'):
            path = frozen / (stem+'.'+ext)
            fig.savefig(path, dpi=175, bbox_inches='tight', facecolor='white')
            shutil.copy2(path, args.out / path.name)
        plt.close(fig)

    for mode in ('absolute', 'common_M11'):
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.subplots_adjust(left=.075, right=.985, bottom=.15, top=.82, hspace=.34, wspace=.30)
        note = 'допуск подтверждён' if converged else 'допуск 1% ещё не подтверждён'
        fig.suptitle('BEM + MBS 5% + текущий прогон MBS 1%\n'
                     r'$ka=20$, $m=1.3+0i$ · общая нормировка, без подгонки'
                     f'\nСнимок: раунд {snapshot["round"]}, оценка {100*output_error:.2f}%; {note}', fontsize=14)
        for ax, (i, j) in zip(axes.flat, entries):
            for values, color, style, label in curves:
                y = values[:, i, j]
                if mode == 'common_M11' and (i, j) != (0, 0):
                    y = y/bem[:, 0, 0]
                ax.plot(theta, y, color=color, ls=style, lw=1.7, label=label)
            if (i, j) == (0, 0) or (mode == 'absolute' and (i, j) == (1, 1)):
                ax.set_yscale('log')
            elif mode == 'absolute':
                ax.set_yscale('symlog', linthresh=5.)
                lo, hi = ax.get_ylim()
                ticks = [-10.**n for n in range(5, 0, -1)] + [0.] + [10.**n for n in range(1, 6)]
                ax.set_yticks([v for v in ticks if lo <= v <= hi])
            label = rf'$M_{{{i+1}{j+1}}}$'
            if mode == 'common_M11' and (i, j) != (0, 0):
                label = rf'$M_{{{i+1}{j+1}}}/M_{{11}}^{{\rm BEM}}$'
            ax.set(xlim=(0, 180), xticks=np.arange(0, 181, 30), xlabel='Угол рассеяния θ, °', ylabel=label)
            ax.axhline(0, color='#444444', lw=.5, alpha=.4)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .05), ncol=3, fontsize=10, frameon=False)
        foot = ('Все элементы: k² dσ/dΩ; знакопеременные элементы — симметричная логарифмическая шкала'
                if mode == 'absolute' else 'M₁₁: k² dσ/dΩ; остальные: все три кривые делятся на один M₁₁ BEM')
        fig.text(.5, .025, foot, ha='center', fontsize=10)
        save(fig, 'bem_mbs_5pct_1pct_current_'+mode)
    shutil.copy2(frozen / 'plot_summary.json', args.out / 'bem_mbs_5pct_1pct_current_summary.json')
    shutil.copy2(frozen / 'all_three_all16.csv', args.out / 'bem_mbs_5pct_1pct_current_all16.csv')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
