#!/usr/bin/env python3
"""Assemble the experimental wave-correction audit without fitting references."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/d6h_20261026/mbs_wave_correction'
CHECKOUT = ROOT / 'experiments/MBS-fast-wave'


def read(path):
    return json.loads(path.read_text())


def save(figure, name):
    figure.savefig(OUT / f'{name}.png', dpi=170, bbox_inches='tight')
    figure.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    plt.close(figure)


def main():
    reference = read(ROOT / 'benchmarks/d6h_20261026/plots/axial_size_comparison/size_comparison.json')
    cuda = read(OUT / 'cuda_validation.json')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.grid': True,
                         'grid.alpha': .22, 'axes.spines.top': False, 'axes.spines.right': False})
    figures = [plt.subplots(2, 3, figsize=(13, 7.5), layout='constrained') for _ in range(2)]
    rows = []
    colors = ['#1863a3', '#d76422', '#26935c']
    for column, ref in enumerate(reference['rows']):
        ka = ref['ka']
        data = np.load(OUT / f'ka{ka}_cuda_gamma30/result.npz')
        summary = read(OUT / f'ka{ka}_cuda_gamma30/summary.json')
        bem = np.load(Path(ref['BEM_result']).with_name('fields.npz'))
        theta = data['theta_deg']
        index = np.searchsorted(bem['theta_deg'], theta)
        np.testing.assert_allclose(bem['theta_deg'][index], theta, atol=1e-12)
        values = [bem['mueller'][index, 0, 0], data['baseline_mueller'][:, 0, 0], data['corrected_mueller'][:, 0, 0]]
        error = [(value/values[0]-1)*100 for value in values[1:]]
        row = {'ka': ka, 'BEM_ref': ref['BEM_ref'], 'BEM_M11_forward': float(values[0][0]),
               'native_M11_forward': float(values[1][0]), 'wave_M11_forward': float(values[2][0]),
               'native_error_percent': float(error[0][0]), 'wave_error_percent': float(error[1][0]),
               'angles_deg': theta.tolist(), 'native_errors_percent': error[0].tolist(),
               'wave_errors_percent': error[1].tolist(), 'reference_validation': ref,
               'solver_summary': str(OUT / f'ka{ka}_cuda_gamma30/summary.json')}
        rows.append(row)
        for plot_index, (fig, axes) in enumerate(figures):
            top, bottom = axes[:, column]
            labels = ['BEM', 'MBS исходный', 'MBS + волновая поправка']
            styles = ['-', '--', '-']
            for value, color, label, style in zip(values, colors, labels, styles):
                top.plot(theta, value/values[0][0], style, color=color, label=label, lw=1.9)
            for value, color, style in zip(error, colors[1:], styles[1:]):
                bottom.plot(theta, value, style, color=color, lw=1.7)
            top.set_title(f'ka = {ka}; ошибка вперёд {error[0][0]:+.2f}% → {error[1][0]:+.2f}%')
            bottom.axhline(0, color='#333333', lw=.8)
            bottom.axhspan(-5, 5, color='#26935c', alpha=.09)
            for ax in [top, bottom]:
                ax.set_xlim(0, 2 if plot_index == 0 else 20)
                ax.set_xlabel('Угол рассеяния θ, градусы')
            if plot_index == 0:
                selected = theta <= 2
                v = np.concatenate([value[selected]/values[0][0] for value in values])
                top.set_ylim(0, max(v)*1.08)
                e = np.concatenate([value[selected] for value in error])
                bottom.set_ylim(min(-6, min(e)-4), max(6, max(e)+4))
            else:
                top.set_yscale('log')
                bottom.set_yscale('symlog', linthresh=5)
            if column == 0:
                top.set_ylabel('M₁₁(θ) / M₁₁,BEM(0) — общий масштаб')
                bottom.set_ylabel('Отклонение от BEM, %')
                top.legend(fontsize=9)
    figures[0][0].suptitle('Быстрая GPU-поправка: центральная часть переднего лепестка\n'
                           'Фиксированная осевая ориентация; m = 1.3, H/D = 1; без подгонки масштаба', fontsize=13)
    figures[1][0].suptitle('Предел применимости: за пределами центрального лепестка ошибки остаются большими\n'
                           'Та же общая нормировка; нижняя шкала симметрично-логарифмическая', fontsize=13)
    save(figures[0][0], 'forward_main_lobe')
    save(figures[1][0], 'forward_cone_limits')

    quad = np.load(ROOT / 'benchmarks/d6h_20261026/forward_diagnosis/forward_quadrature_89_30.npz')
    gpu = np.load(OUT / 'cuda_mean_89_30.npz')
    corrected = gpu['jones']+gpu['delta'][:, None, None]*np.eye(2)[None, :, :]
    forward = [200*np.sum(abs(j)**2, axis=(1, 2)) for j in [quad['jones'], gpu['jones'], corrected]]
    weights = quad['weights']
    np.testing.assert_allclose(weights.sum(), 1)
    means = [float(np.dot(weights, value)) for value in forward]
    errors = [value/forward[0]-1 for value in forward[1:]]
    orientation = {}
    for label, error in zip(['native', 'wave'], errors):
        orientation[label] = {'minimum_relative_error': float(min(error)), 'maximum_relative_error': float(max(error)),
                              'weighted_mean_absolute_relative_error': float(np.dot(weights, abs(error))),
                              'solid_angle_fraction_within_5_percent': float(np.dot(weights, abs(error) <= .05))}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout='constrained')
    names = ['BEM, T-матрица', 'MBS', 'MBS + поправка']
    ratios = np.array(means)/means[0]
    axes[0].bar(names, ratios, color=colors, width=.65)
    axes[0].axhline(1, color='#333333', ls=':', lw=1)
    for i, (ratio, mean) in enumerate(zip(ratios, means)):
        axes[0].text(i, ratio+.025, f'{mean:,.1f}\n{(ratio-1)*100:+.2f}%', ha='center', fontsize=10)
    axes[0].set_ylim(0, 1.22)
    axes[0].set_ylabel('Среднее M₁₁(0) / среднее BEM')
    axes[0].set_title('Среднее: ошибка уменьшилась до +1,18%')
    for error, color, label in zip(errors, colors[1:], names[1:]):
        axes[1].hist(error*100, bins=np.arange(-45, 40, 2.5), weights=weights,
                     histtype='step', lw=1.8, color=color, label=label)
    axes[1].axvspan(-5, 5, color=colors[2], alpha=.09)
    axes[1].set_xlabel('Ошибка M₁₁(0) отдельной ориентации, %')
    axes[1].set_ylabel('Доля телесного угла в интервале')
    axes[1].set_title('Отдельные ориентации: вплоть до −39%')
    axes[1].legend()
    fig.suptitle('ka = 20: проверка на 2670 направлениях\n'
                 'BEM усреднён аналитически; для новой поправки проверка численная', fontsize=13)
    save(fig, 'forward_orientation_mean')

    batch_cpu = read(OUT / 'batch_cpu_16/batch_summary.json')
    batch_gpu = read(OUT / 'batch_cuda_16/batch_summary.json')
    warm_cpu = np.array([row['total_seconds'] for row in batch_cpu['rows'][1:]])
    warm_gpu = np.array([row['total_seconds'] for row in batch_gpu['rows'][1:]])
    hashes = {str(path.relative_to(CHECKOUT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [CHECKOUT / name for name in ['waveoptics/prism.py', 'waveoptics/cuda_backend.py',
                           'waveoptics/cuda_wave.cu', 'waveoptics/libmbs_wave_cuda.so', 'tools/mbs_wave.py']]}
    report = {'experimental': True, 'physical_accuracy_certified': False,
              'scope': 'scalar forward correction; no fullauto integration or analytical T matrix for the new model',
              'normalization': 'k^2 d_sigma/d_Omega; plots use one common BEM forward denominator',
              'reference_data_used_by_solver': False, 'cuda_validation': cuda,
              'fixed_axial_cases': rows, 'orientation_error_distribution': orientation,
              'orientation_refinement': read(OUT / 'average_refinement.json'),
              'wave_grid_average_refinement': read(OUT / 'wave_grid_average_check.json'),
              'held_out_prototype_cases': read(OUT / 'holdout.json'),
              'held_out_note': 'ka=111 prototype grid256 fails the production Nyquist check; excluded from validated cases',
              'batch_16': {'cpu_in_process_seconds': batch_cpu['total_seconds'],
                           'cuda_in_process_seconds': batch_gpu['total_seconds'],
                           'whole_process_measured_seconds': {'cpu': 1.57787838299555, 'cuda': .739277356005914},
                           'warm_cpu_median_seconds': float(np.median(warm_cpu)),
                           'warm_cuda_median_seconds': float(np.median(warm_gpu)),
                           'warm_complete_pipeline_speedup': float(np.median(warm_cpu)/np.median(warm_gpu))},
              'tests': {'physics_and_cpu_cuda_checks_passed': 11,
                        'batch_all_mueller_cpu_gpu_max_error_test': 'passed: <0.02% of local M11 for 16 orientations, 41 angles each'},
              'source_sha256': hashes,
              'limitations': ['The correction is scalar and omits diffraction of reflected paths.',
                              'Small average error does not bound errors of individual orientations.',
                              'Some weak side lobes become worse; no all-angle 1% or 5% claim.',
                              'GPU timing for 2670 directions excludes cached native ray calculations.',
                              'CUDA startup can make one isolated orientation slower than CPU.',
                              'BEM ref5/ref6 and ADDA cross-checks are finite-resolution references, not exact solutions.']}
    (OUT / 'validation_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'cases': [{k:r[k] for k in ['ka','native_error_percent','wave_error_percent']} for r in rows],
                      'orientation': orientation, 'batch': report['batch_16']}, indent=2))


if __name__ == '__main__':
    main()
