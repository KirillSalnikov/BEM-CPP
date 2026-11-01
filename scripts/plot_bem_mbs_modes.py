#!/usr/bin/env python3
"""Overlay validated BEM/MBS comparison snapshots without amplitude fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_case(directory, mode):
    report = json.loads((directory / 'comparison_summary.json').read_text())
    if (not report['mbs_converged_and_validated']
            or report['mbs_status']['status'] != 'converged'
            or not report['mbs_independent_validation']['passed']):
        raise ValueError(f'{mode}: completed and independently validated MBS required')
    expected = 'coherent Jones sum' if mode == 'coherent' else 'incoherent Mueller sum'
    if report['mbs_beam_accumulation'] != expected:
        raise ValueError(f'{mode}: wrong beam accumulation mode')
    if report['conversion']['fitted_scale'] or report['conversion']['empirical_correction']:
        raise ValueError('Only physical normalization without fitted corrections is supported')
    snapshot = directory / 'mbs_mueller_snapshot.dat'
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != report['mbs_table_sha256']:
        raise ValueError(f'{mode}: snapshot hash mismatch')
    with np.load(directory / 'comparison_all16.npz') as data:
        arrays = {key: data[key] for key in data.files}
    if not all(np.all(np.isfinite(x)) for x in arrays.values()):
        raise ValueError(f'{mode}: nonfinite comparison data')
    # Confirm that the converted data really match the preserved native table.
    raw = np.loadtxt(snapshot, skiprows=1)
    parity = np.array(report['conversion']['Stokes_D'])
    converted = (raw[:, 2:].reshape(-1, 4, 4)
                 * report['conversion']['scale_k_squared']
                 * parity[None, :, None] * parity[None, None, :])
    np.testing.assert_array_equal(raw[:, 0], arrays['theta'])
    np.testing.assert_allclose(converted, arrays['mbs'], rtol=2e-14, atol=0)
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coherent', type=Path, required=True)
    parser.add_argument('--incoherent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reports, data = {}, {}
    for mode in ('coherent', 'incoherent'):
        reports[mode], data[mode] = load_case(getattr(args, mode), mode)
    for field in ('parameters', 'bem_directory', 'conversion'):
        if reports['coherent'][field] != reports['incoherent'][field]:
            raise ValueError(f'Incompatible comparisons: {field}')
    c, u = data['coherent'], data['incoherent']
    # These completed ka=40 runs use identical requested grids. Never silently
    # interpolate a different grid, in particular the sharp forward peak.
    for field in ('theta', 'bem_drawing_theta'):
        np.testing.assert_array_equal(c[field], u[field])
    bem_snapshot_differences = {}
    # Independent FP64 contractions may differ in roundoff, especially in
    # symmetry-zero elements. Use the physical M11 scale to compare them.
    for field in ('bem', 'bem_drawing'):
        if np.any(u[field][:, 0, 0] <= 0):
            raise ValueError('Nonpositive BEM M11')
        delta = float(np.max(np.abs(c[field] - u[field]) / u[field][:, 0, 0, None, None]))
        if delta > 1e-12:
            raise ValueError(f'BEM snapshots differ: {field}, scaled difference {delta}')
        bem_snapshot_differences[field] = delta
    np.testing.assert_allclose(c['weights'], u['weights'], rtol=1e-12, atol=1e-12)
    theta, bem, denom = u['theta'], u['bem'], u['bem'][:, 0, 0]
    errors = {mode: np.abs(d['mbs'] - bem) / denom[:, None, None]
              for mode, d in data.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        'parameters': reports['coherent']['parameters'],
        'bem_directory': reports['coherent']['bem_directory'],
        'conversion': reports['coherent']['conversion'],
        'identical_angular_grids': True,
        'bem_snapshot_max_difference_over_m11': bem_snapshot_differences,
        'interpolation_applied': False,
        'source_comparisons': {m: str(getattr(args, m).resolve()) for m in data},
        'cases': reports,
        'scope': 'Completed runs; different MBS commits and numerical tolerances. '
                 'Differences cannot be attributed exclusively to coherence mode. '
                 'Fullauto tolerances describe MBS numerical error, not BEM agreement.',
    }
    (args.out / 'comparison_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.out / 'comparison_all16.npz', theta=theta, bem=bem,
                        coherent=c['mbs'], incoherent=u['mbs'],
                        coherent_error_over_bem_m11=errors['coherent'],
                        incoherent_error_over_bem_m11=errors['incoherent'],
                        bem_drawing_theta=u['bem_drawing_theta'], bem_drawing=u['bem_drawing'])
    names = [f'M{i+1}{j+1}' for i in range(4) for j in range(4)]
    headers = ['theta_deg'] + [prefix + n for prefix in
               ('BEM_', 'MBS_coherent_', 'MBS_incoherent_',
                'coherent_absdiff_over_BEM_M11_', 'incoherent_absdiff_over_BEM_M11_')
               for n in names]
    np.savetxt(args.out / 'comparison_all16.csv', np.column_stack(
        [theta] + [x.reshape(-1, 16) for x in
                   (bem, c['mbs'], u['mbs'], errors['coherent'], errors['incoherent'])]),
        delimiter=',', header=','.join(headers), comments='')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .2, 'savefig.facecolor': 'white'})
    colors = {'coherent': '#18814b', 'incoherent': '#d75c22'}
    styles = {'coherent': '-.', 'incoherent': '--'}
    labels = {}
    for mode, name in [('coherent', 'когерентный'), ('incoherent', 'некогерентный')]:
        r = reports[mode]
        labels[mode] = (f'MBS-fast: {name}, fullauto {100*r["mbs_numerical_target"]:g}% '
                        f'({r["mbs_source_commit"][:7]})')
    curves = [(u['bem_drawing_theta'], u['bem_drawing'], u['bem_drawing'][:, 0, 0],
               '#1765a3', '-', 'BEM: T-матрица, аналитическое SO(3)')]
    curves += [(theta, data[m]['mbs'], denom, colors[m], styles[m], labels[m])
               for m in data]
    p = report['parameters']
    title = (f'ka={p["ka"]:g} · n={p["index_real"]:g} · λ={p["wavelength"]:g} мкм · '
             f'H/ширина={p["aspect_flats"]:.6g}')

    def save(fig, name):
        for ext in ('png', 'pdf'):
            fig.savefig(args.out / f'{name}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)

    def panels(entries, shape, name):
        full = len(entries) == 16
        fig, axes = plt.subplots(*shape, figsize=(16, 14) if full else (16, 10))
        fig.subplots_adjust(left=.065, right=.985, bottom=.20 if full else .25,
                            top=.89 if full else .85, hspace=.58, wspace=.30)
        fig.suptitle('Усреднение по всем ориентациям: BEM и два режима MBS-fast\n' + title,
                     fontsize=16)
        for ax, (i, j) in zip(axes.flat, entries):
            for angles, values, common_m11, color, style, label in curves:
                y = values[:, i, j] if (i, j) == (0, 0) else values[:, i, j] / common_m11
                ax.plot(angles, y, style, color=color, lw=1.35, label=label)
            ax.set(xlim=(0, 180), xticks=np.arange(0, 181, 30), xlabel='θ, °')
            ax.set_title(rf'$M_{{{i+1}{j+1}}}$' if (i, j) == (0, 0)
                         else rf'$M_{{{i+1}{j+1}}}/M_{{11}}^{{\rm BEM}}$')
            if (i, j) == (0, 0):
                ax.set_yscale('log')
            else:
                ax.axhline(0, color='#555555', lw=.5)
                if max(np.max(np.abs(v[:, i, j]) / d) for _, v, d, _, _, _ in curves) < .02:
                    ax.set_ylim(-.02, .02)
            annotations = []
            for mode, short in [('coherent', 'К'), ('incoherent', 'НК')]:
                q = int(errors[mode][:, i, j].argmax())
                value = data[mode]['mbs'][q, i, j]
                y = value if (i, j) == (0, 0) else value / denom[q]
                ax.scatter(theta[q], y, color=colors[mode], s=18, zorder=5)
                annotations.append(f'{short}: {100*errors[mode][q,i,j]:.2f}% · {theta[q]:.2f}°')
            ax.text(.98, .97, '\n'.join(annotations), ha='right', va='top',
                    transform=ax.transAxes, fontsize=7.7,
                    bbox={'facecolor': 'white', 'alpha': .88, 'edgecolor': 'none', 'pad': 2})
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.5, .064),
                   ncol=1, frameon=False, fontsize=11)
        fig.text(.5, .031, 'M₁₁: k² dσ/dΩ. Остальные: общий знаменатель M₁₁ BEM. '
                 'Без подгонки масштаба.\nТочки: max |MBS−BEM|/M₁₁ BEM. '
                 'К — когерентный, НК — некогерентный. Оба расчёта MBS завершены.',
                 ha='center', fontsize=10)
        save(fig, name)

    panels([(i, j) for i in range(4) for j in range(4)], (4, 4),
           'all16_bem_coherent_incoherent')
    panels([(0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, 3)], (2, 3),
           'six_bem_coherent_incoherent')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.subplots_adjust(left=.065, right=.985, bottom=.32, top=.80, wspace=.30)
    fig.suptitle('M₁₁: BEM, когерентный и некогерентный MBS-fast\n' + title, fontsize=14)
    for angles, values, _, color, style, label in curves:
        for ax in axes[:2]:
            ax.plot(angles, values[:, 0, 0], style, color=color, label=label)
    axes[0].set(xlim=(0, 180), yscale='log', xlabel='θ, °', ylabel='M₁₁, k² dσ/dΩ', title='Все углы')
    axes[1].set(xlim=(0, 12), xlabel='θ, °', title='Переднее направление')
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    for mode in data:
        axes[2].plot(theta, 100*(data[mode]['mbs'][:, 0, 0]/denom - 1),
                     styles[mode], color=colors[mode])
    axes[2].axhline(0, lw=.6, color='#555555')
    axes[2].set(xlim=(0, 180), xlabel='θ, °', ylabel='(MBS/BEM − 1), %', title='Различие по M₁₁')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.5, .025),
               ncol=1, frameon=False, fontsize=10)
    save(fig, 'm11_bem_coherent_incoherent')
    print(args.out.resolve())


if __name__ == '__main__':
    main()
