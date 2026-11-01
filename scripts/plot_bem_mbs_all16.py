#!/usr/bin/env python3
"""Compare completed analytic BEM and native fullauto MBS with physical units.

No fitted amplitude or wave correction. MBS d_sigma/d_Omega is converted to
BEM k^2 d_sigma/d_Omega and its Stokes convention by D M D, D=(1,1,-1,-1).
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tmatrix_so3_mstm import analytic_mueller_tmatrix


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bem', type=Path, required=True)
    parser.add_argument('--mbs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-intermediate', action='store_true',
                        help='Freeze and clearly mark an unfinished MBS result')
    parser.add_argument('--matrix', type=Path,
                        help='Explicit pilot matrix; requires --allow-intermediate and is never marked validated')
    args = parser.parse_args()
    if args.matrix is not None and not args.allow_intermediate:
        parser.error('--matrix requires --allow-intermediate')
    args.out.mkdir(parents=True, exist_ok=True)
    summary = read_json(args.bem / 'summary.json')
    average = read_json(args.bem / 'average.json')
    manifest = read_json(args.mbs.parent / 'manifest.json')
    status = read_json(args.mbs / 'result/fullauto_status.json')
    audit_path = args.mbs / 'result/independent_final_validation.json'
    audit = read_json(audit_path) if audit_path.exists() else {'passed': False}
    stop_path = args.mbs / 'user_stop_final_checks.json'
    stopped = read_json(stop_path) if stop_path.exists() else {}
    command = read_json(args.mbs / 'launch_metadata.json')['command']
    incoherent = '--incoherent' in command
    tolerance = float(command[command.index('--fullauto')+1])
    if not summary['complete'] or average['Euler_quadrature_nodes'] != 0:
        raise ValueError('A completed analytic BEM average is required')
    validated = status['status'] == 'converged' and audit['passed'] and args.matrix is None
    if not validated and not args.allow_intermediate:
        raise ValueError('MBS must pass convergence and independent validation')
    params = summary['parameters']
    checks = {
        'ka': (params['ka'], manifest['ka']),
        'wavelength': (params['wavelength'], manifest['wavelength']),
        'index_real': (params['ri'], manifest['refractive_index'][0]),
        'index_imag': (params['ri_imag'], manifest['refractive_index'][1]),
        'aspect_flats': (params['prism_aspect'], manifest['prism_height_to_width_across_flats']),
        'aspect_vertex': (params['prism_aspect']*np.sqrt(3)/2, manifest['particle_L_over_D_vertex']),
    }
    for key, (x, y) in checks.items():
        if not np.isclose(x, y, rtol=1e-12, atol=1e-14):
            raise ValueError(f'Geometry/physics mismatch: {key}: {x} != {y}')
    source = args.matrix or args.mbs / 'result/mueller_fullauto.dat'
    snapshot = source.read_bytes()
    if stopped.get('stopped') and hashlib.sha256(snapshot).hexdigest() != stopped['table_sha256_after']:
        raise ValueError('MBS table changed after the recorded user stop')
    if read_json(args.mbs / 'result/fullauto_status.json') != status:
        raise RuntimeError('MBS updated during snapshot: rerun the plot')
    (args.out / 'mbs_mueller_snapshot.dat').write_bytes(snapshot)
    (args.out / 'mbs_status_snapshot.json').write_text(json.dumps(status, indent=2)+'\n')
    table = np.loadtxt(io.BytesIO(snapshot), skiprows=1)
    theta, weights = table[:, 0], table[:, 1]
    if (table.shape[1] != 18 or theta[0] != 0 or theta[-1] != 180
            or not np.all(np.diff(theta) > 0) or not np.isclose(weights.sum(), 4*np.pi)):
        raise ValueError('Invalid angular grid or quadrature weights')
    k = 2*np.pi/manifest['wavelength']
    if not np.isclose(k*manifest['equiv_radius'], params['ka'], rtol=1e-12):
        raise ValueError('Size parameter mismatch')
    parity = np.array([1., 1., -1., -1.])
    mbs = table[:, 2:].reshape(-1, 4, 4)*k**2*parity[None, :, None]*parity[None, None, :]
    # Evaluate both the comparison nodes and a dense drawing grid analytically.
    # A sparse MBS pilot must not alias the oscillations in the BEM curve.
    drawing_theta = np.unique(np.concatenate((np.asarray(average['theta']), theta)))
    drawing_bem, metadata = analytic_mueller_tmatrix(
        np.load(args.bem / 'tmatrix.npy', mmap_mode='r'), average['order'], drawing_theta,
        precision='fp64', threads=4, cyclic_order=average['cyclic_order'])
    bem = drawing_bem[np.searchsorted(drawing_theta, theta)]
    if (not np.all(np.isfinite(bem)) or not np.all(np.isfinite(mbs))
            or np.any(bem[:, 0, 0] <= 0) or np.any(mbs[:, 0, 0] <= 0)):
        raise ValueError('Invalid Mueller values')
    denominator = bem[:, 0, 0, None, None]
    error = np.abs(mbs-bem)/denominator
    worst = np.unravel_index(error.argmax(), error.shape)
    components = {}
    for i in range(4):
        for j in range(4):
            q = int(error[:, i, j].argmax())
            components[f'M{i+1}{j+1}'] = {
                'max_difference_percent_of_BEM_M11': float(100*error[q, i, j]),
                'theta_deg': float(theta[q]),
            }
    report = {
        'bem_directory': str(args.bem.resolve()), 'mbs_directory': str(args.mbs.resolve()),
        'parameters': {key: float(x) for key, (x, _) in checks.items()},
        'mbs_source_commit': manifest['source_commit'],
        'mbs_beam_accumulation': 'incoherent Mueller sum' if incoherent else 'coherent Jones sum',
        'mbs_binary_sha256': manifest['binary_sha256'],
        'mbs_table_sha256': hashlib.sha256(snapshot).hexdigest(),
        'mbs_table_source': str(source.resolve()), 'explicit_pilot_matrix': args.matrix is not None,
        'mbs_converged_and_validated': validated,
        'mbs_numerical_target': tolerance, 'mbs_status': status,
        'mbs_independent_validation': audit, 'bem_reevaluation': metadata,
        'mbs_user_stop': stopped or None,
        'bem_size_interpolation': summary.get('size_interpolation'),
        'conversion': {'scale_k_squared': float(k*k), 'Stokes_D': parity.tolist(),
                       'fitted_scale': False, 'empirical_correction': False},
        'angular_points': len(theta),
        'analytic_BEM_drawing_points': len(drawing_theta),
        'forward_M11_difference_percent': float(100*(mbs[0, 0, 0]/bem[0, 0, 0]-1)),
        'backward_M11_difference_percent': float(100*(mbs[-1, 0, 0]/bem[-1, 0, 0]-1)),
        'worst_difference_percent_of_BEM_M11': float(100*error[worst]),
        'worst_component': f'M{worst[1]+1}{worst[2]+1}',
        'worst_theta_deg': float(theta[worst[0]]),
        'weighted_relative_L2_all16': float(np.sqrt(np.sum(weights[:, None, None]*(mbs-bem)**2)
                                                   / np.sum(weights[:, None, None]*bem**2))),
        'cross_sections_um2': {
            'BEM_from_T': float(summary['scattering_cross_section']*manifest['equiv_radius']**2),
            'BEM_angular_quadrature_on_MBS_grid': float(weights @ bem[:, 0, 0]/k**2),
            'MBS_angular_quadrature': float(weights @ mbs[:, 0, 0]/k**2),
        },
        'components': components,
        'scope': 'Fullauto tolerance is MBS numerical averaging error, not agreement with Maxwell/BEM. BEM mesh error is not certified here.',
    }
    (args.out / 'comparison_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    names = [f'M{i+1}{j+1}' for i in range(4) for j in range(4)]
    header = ['theta_deg']+[prefix+n for prefix in ('BEM_', 'MBS_', 'absdiff_over_BEM_M11_') for n in names]
    np.savetxt(args.out / 'comparison_all16.csv', np.column_stack((theta, bem.reshape(-1, 16),
               mbs.reshape(-1, 16), error.reshape(-1, 16))), delimiter=',', header=','.join(header), comments='')
    np.savez_compressed(args.out / 'comparison_all16.npz', theta=theta, weights=weights, bem=bem, mbs=mbs,
                        error_over_bem_m11=error, bem_drawing_theta=drawing_theta, bem_drawing=drawing_bem)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .22, 'savefig.facecolor': 'white'})
    index_label = f"{params['ri']:g}" + (f"{params['ri_imag']:+g}i" if params.get('ri_imag', 0) else '')
    title = (f"ka={params['ka']:g} · m={index_label} · λ={params['wavelength']:g} мкм · "
             f"H/ширина={params['prism_aspect']:.6g}")
    mode = 'некогерентный' if incoherent else 'когерентный'
    mbs_label = f'MBS-fast: {mode}, fullauto {100*tolerance:g}%'
    if not validated:
        mbs_label += ' — предварительно'
        title += ('\nMBS: независимая проверка пройдена; заключительные проверки остановлены'
                  if stopped.get('stopped') and audit.get('passed') else
                  '\nMBS: финальные проверки ещё не завершены')
    if args.matrix is not None:
        mbs_label += ' (пилот)'
    bem_label = ('BEM: интерполированная T(ka), аналитическое SO(3)'
                 if summary.get('size_interpolation') else 'BEM: T-матрица, аналитическое SO(3)')
    curves = [(drawing_bem, drawing_theta, drawing_bem[:, 0, 0], '#1765a3', '-', bem_label),
              (mbs, theta, bem[:, 0, 0], '#d75c22', '--', mbs_label)]

    def save(fig, name):
        fig.savefig(args.out / (name+'.png'), dpi=170, bbox_inches='tight')
        fig.savefig(args.out / (name+'.pdf'), bbox_inches='tight')
        plt.close(fig)

    def panels(entries, shape, name):
        full = len(entries) == 16
        fig, axes = plt.subplots(*shape, figsize=(15, 12) if full else (15, 9))
        fig.subplots_adjust(left=.075, right=.985, bottom=.15 if full else .20,
                            top=.86 if full else .82, hspace=.52, wspace=.32)
        fig.suptitle('Усреднение по всем ориентациям: BEM и MBS-fast\n'+title, fontsize=15)
        for ax, (i, j) in zip(axes.flat, entries):
            for values, angles, common_M11, color, style, label in curves:
                y = values[:, i, j] if (i, j) == (0, 0) else values[:, i, j]/common_M11
                ax.plot(angles, y, style, color=color, lw=1.4, label=label)
            ax.set(xlim=(0, 180), xticks=np.arange(0, 181, 30), xlabel='θ, °')
            ax.set_title(rf'$M_{{{i+1}{j+1}}}$' if (i, j) == (0, 0)
                         else rf'$M_{{{i+1}{j+1}}}/M_{{11}}^{{\rm BEM}}$', fontsize=11)
            if (i, j) == (0, 0):
                ax.set_yscale('log')
            else:
                ax.axhline(0, color='#555555', lw=.6)
                if max(np.max(np.abs(values[:, i, j])/denom)
                       for values, _, denom, _, _, _ in curves) < .02:
                    ax.set_ylim(-.02, .02)
            q = int(error[:, i, j].argmax())
            y = mbs[q, i, j] if (i, j) == (0, 0) else mbs[q, i, j]/bem[q, 0, 0]
            ax.scatter(theta[q], y, color='#9e2517', s=18, zorder=5)
            ax.text(.98, .96, f'max Δ: {100*error[q,i,j]:.2f}%\nθ={theta[q]:.2f}°',
                    ha='right', va='top', transform=ax.transAxes, fontsize=8,
                    bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none', 'pad': 2})
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .05), ncol=2, frameon=False)
        fig.text(.5, .023, 'M₁₁: k² dσ/dΩ. Остальные: общий знаменатель M₁₁ BEM. Точка: max |MBS−BEM|/M₁₁ BEM.',
                 ha='center', fontsize=10)
        save(fig, name)

    panels([(i, j) for i in range(4) for j in range(4)], (4, 4), 'all16_common_normalization')
    panels([(0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, 3)], (2, 3), 'six_common_normalization')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    fig.suptitle('M₁₁ и расхождение методов · '+title, fontsize=13)
    for values, angles, _, color, style, label in curves:
        axes[0].plot(angles, values[:, 0, 0], style, color=color, label=label)
        axes[1].plot(angles, values[:, 0, 0], style, color=color)
    axes[0].set(xlim=(0, 180), yscale='log', xlabel='θ, °', ylabel='M₁₁, k² dσ/dΩ', title='Все углы')
    axes[1].set(xlim=(0, 12), xlabel='θ, °', title='Переднее направление')
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[2].plot(theta, 100*error[:, 0, 0], color='#d75c22', label='M₁₁')
    axes[2].plot(theta, 100*error.max(axis=(1, 2)), '--', color='#1765a3', label='max по 16 элементам')
    axes[2].set(xlim=(0, 180), xlabel='θ, °', ylabel='|ΔM|/M₁₁ BEM, %', title='Различие моделей')
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    save(fig, 'm11_forward_and_difference')
    print(json.dumps({k: report[k] for k in ['forward_M11_difference_percent',
        'backward_M11_difference_percent', 'worst_difference_percent_of_BEM_M11', 'worst_component',
        'worst_theta_deg', 'weighted_relative_L2_all16']}, indent=2))


if __name__ == '__main__':
    main()
