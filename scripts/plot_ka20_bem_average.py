#!/usr/bin/env python3
"""Plot the completed BEM T-matrix SO(3) average without orientation sampling."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tmatrix_so3_mstm import analytic_mueller_tmatrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run / 'summary.json').read_text())
    original = json.loads((args.run / 'average.json').read_text())
    if not summary['complete'] or original['Euler_quadrature_nodes'] != 0:
        raise RuntimeError('A complete T matrix with analytic SO(3) average is required')
    theta = np.linspace(0, 180, 1801)
    matrix, metadata = analytic_mueller_tmatrix(
        np.load(args.run / 'tmatrix.npy', mmap_mode='r'), original['order'], theta,
        precision='fp64', threads=4, cyclic_order=original['cyclic_order'])
    if not np.all(np.isfinite(matrix)) or not np.all(matrix[:, 0, 0] > 0):
        raise RuntimeError('Invalid Mueller data')
    # Check the denser analytic evaluation at every original output angle.
    coarse = np.asarray(original['mueller']).transpose(2, 0, 1)
    assert np.array_equal(theta[::10], original['theta'])
    dense_check = np.max(np.abs(matrix[::10] - coarse) / coarse[:, 0, 0, None, None])
    if dense_check > 1e-10:
        raise RuntimeError(f'Dense-angle evaluation changed the original result: {dense_check}')
    ka = original['ka']
    normalized = matrix / matrix[:, 0, 0, None, None]
    np.savez_compressed(args.out / 'bem_so3_ka20.npz', theta_deg=theta,
                        mueller_dimensionless=matrix, mueller_over_M11=normalized)
    header = ['theta_deg'] + [f'M{i+1}{j+1}' for i in range(4) for j in range(4)]
    np.savetxt(args.out / 'bem_so3_ka20.csv', np.column_stack((theta, matrix.reshape(-1, 16))),
               delimiter=',', header=','.join(header), comments='', fmt='%.12e')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .22, 'savefig.facecolor': 'white'})
    color = '#1765a3'
    title = 'BEM · аналитическое усреднение по всем ориентациям'
    subtitle = r'$ka=20$, $m=1.3+0i$ · гексагональная призма $H/D_{\rm flats}=1$ · $L=44$'

    def save(fig, stem):
        fig.savefig(args.out / (stem + '.png'), dpi=170, bbox_inches='tight')
        fig.savefig(args.out / (stem + '.pdf'), bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), layout='constrained')
    fig.suptitle(title + '\n' + subtitle, fontsize=15)
    entries = [(0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, 3)]
    for ax, (i, j) in zip(axes.flat, entries):
        if (i, j) == (0, 0):
            ax.semilogy(theta, matrix[:, 0, 0], color=color, lw=1.6)
            ax.set_ylabel(r'$M_{11}=k^2\,d\sigma/d\Omega$')
            ax.set_title('Интенсивность, логарифмическая шкала')
        else:
            ax.plot(theta, normalized[:, i, j], color=color, lw=1.6)
            ax.axhline(0, color='#444444', lw=.65, alpha=.5)
            ax.set_ylabel(rf'$M_{{{i+1}{j+1}}}/M_{{11}}$')
        ax.set(xlim=(0, 180), xticks=np.arange(0, 181, 30), xlabel='Угол рассеяния θ, °')
    fig.supxlabel('Полное SO(3) через T-матрицу · без квадратуры по углам Эйлера · шаг вывода 0.1°', fontsize=11)
    save(fig, 'bem_so3_ka20_overview')

    fig, axes = plt.subplots(4, 4, figsize=(13.2, 10.8), sharex=True, layout='constrained')
    fig.suptitle('Все 16 элементов после аналитического усреднения\n' + subtitle, fontsize=15)
    forbidden = []
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ax.plot(theta, normalized[:, i, j], color=color, lw=1.3)
            ax.set_title(rf'$M_{{{i+1}{j+1}}}/M_{{11}}$', fontsize=12)
            ax.set(xlim=(0, 180), ylim=(-1.05, 1.05), xticks=[0, 60, 120, 180], yticks=[-1, 0, 1])
            ax.axhline(0, color='#444444', lw=.6, alpha=.5)
            if (i < 2) != (j < 2):
                residual = float(np.max(np.abs(normalized[:, i, j])))
                forbidden.append(residual)
                ax.text(.04, .07, f'max |M/M₁₁| = {residual:.1e}', transform=ax.transAxes, fontsize=8)
    fig.supxlabel('Угол рассеяния θ, °; единый масштаб по вертикали, малые остатки не обнулены', fontsize=11)
    save(fig, 'bem_so3_ka20_all16')
    trapezoid = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    csca_grid = 2*np.pi/ka**2 * trapezoid(matrix[:, 0, 0]*np.sin(np.radians(theta)), np.radians(theta))
    report = {
        'source_run': str(args.run.resolve()), 'ka': ka, 'refractive_index': [1.3, 0],
        'theta_step_deg': .1, 'angle_evaluation': 'analytic GSF evaluation, no interpolation',
        'mueller_units': 'dimensionless k^2 d_sigma/d_Omega; a_eq=1',
        'orientation_average': metadata, 'dense_vs_original_max_scaled_M11': float(dense_check),
        'scattering_cross_section_T': summary['scattering_cross_section'],
        'scattering_cross_section_angular_trapezoid': float(csca_grid),
        'extinction_cross_section_T': summary['extinction_cross_section'],
        'max_mirror_forbidden_scaled_M11': max(forbidden),
        'M11_forward': float(matrix[0, 0, 0]), 'M11_backward': float(matrix[-1, 0, 0])}
    (args.out / 'bem_so3_ka20_plot_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
