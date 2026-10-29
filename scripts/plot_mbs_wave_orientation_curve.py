#!/usr/bin/env python3
"""Plot the orientation-averaged experimental M11 and its convergence checks."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'benchmarks/d6h_20261026/mbs_wave_correction'
DATA = OUT/'orientation_curve'


def main():
    coarse = np.load(DATA/'mean_45_16.npz')
    fine = np.load(DATA/'mean_89_30.npz')
    bem_cone = np.load(DATA/'bem_reference.npz')
    old = json.loads((ROOT/'runs/ka20_d6h_q6_20261026_170903/main_ref5_L44/average.json').read_text())
    bem_theta, bem_all = np.asarray(old['theta']), np.asarray(old['mueller'])[0, 0]
    fullauto_path = ROOT/'runs/mbs_fullauto_ka20_20261026/fullauto_1pct/result/mueller_fullauto.dat'
    fullauto = np.loadtxt(fullauto_path, skiprows=1)
    theta, native, wave = fine['theta_deg'], fine['baseline_M11'], fine['corrected_M11']
    np.testing.assert_array_equal(theta, bem_cone['theta_deg'])
    bem = bem_cone['mueller'][:, 0, 0]
    native_error, wave_error = 100*(native/bem-1), 100*(wave/bem-1)
    orientation_change = coarse['corrected_M11']/wave-1
    phi_change = fine['corrected_phi32_M11']/wave-1
    mbs_fullauto = np.interp(theta, fullauto[:, 0], fullauto[:, 2]*20**2)
    report = {
        'ka': 20, 'index': [1.3, 0], 'aspect_H_over_Dflats': 1,
        'observable': 'orientation-averaged unpolarized M11(theta)',
        'units': 'k^2 d_sigma/d_Omega; a_eq=1; no fitted normalization',
        'BEM_average': 'analytic full SO(3), L44, ref5',
        'MBS_wave_average': 'numerical; 89x30 incident quadrature folded by D6h to 720 directions, 64 outgoing azimuths',
        'new_correction_theta_range_deg': [0, 30], 'theta_step_deg': .5,
        'forward': {'BEM': float(bem[0]), 'MBS': float(native[0]), 'MBS_wave': float(wave[0]),
                    'native_error_percent': float(native_error[0]), 'wave_error_percent': float(wave_error[0])},
        'error_samples': [{ 'theta_deg': float(theta[i]), 'native_percent': float(native_error[i]),
                            'wave_percent': float(wave_error[i])} for i in [0, 2, 4, 10, 20, 30, 40, 50, 60]],
        'max_orientation_refinement_relative_change': float(np.max(abs(orientation_change))),
        'max_phi32_to64_relative_change': float(np.max(abs(phi_change))),
        'max_native_quadrature_vs_interpolated_fullauto_relative_difference': float(np.max(abs(native/mbs_fullauto-1))),
        'symmetry_validation': json.loads((DATA/'symmetry_validation.json').read_text()),
        'limitations': ['This is numerical averaging of the new correction, not a T matrix for MBS.',
                        'The correction is experimental and scalar; no all-angle accuracy guarantee.',
                        'No corrected values are computed or extrapolated for theta>30 degrees.',
                        'Orientation and azimuth refinement test quadrature, not physical-model accuracy.'],
    }
    (DATA/'plot_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savetxt(DATA/'averaged_M11.csv', np.column_stack([theta, bem, native, wave, native_error, wave_error,
                coarse['corrected_M11'], fine['corrected_phi32_M11']]), delimiter=',',
               header='theta_deg,BEM_analytic,MBS_quadrature,MBS_wave,native_error_percent,wave_error_percent,wave_coarse_orientation,wave_phi32', comments='')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.grid': True, 'grid.alpha': .22,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#1765a3', '#d75c22', '#26935c']
    labels = ['BEM · аналитическое усреднение T', 'MBS · без поправки', 'MBS + волновая поправка']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    fig.subplots_adjust(left=.055, right=.985, bottom=.22, top=.79, wspace=.28)
    axes[0].semilogy(bem_theta, bem_all, color=colors[0], lw=1.8, label=labels[0])
    axes[0].semilogy(fullauto[:, 0], fullauto[:, 2]*400, '--', color=colors[1], lw=1.6, label=labels[1])
    axes[0].semilogy(theta, wave, color=colors[2], lw=1.9, label=labels[2])
    axes[0].axvline(30, color='#777777', ls=':', lw=1)
    axes[0].set(xlim=(0, 180), xticks=np.arange(0,181,30), title='Весь диапазон; поправка только до 30°',
                ylabel='⟨M₁₁⟩, единицы k² dσ/dΩ', xlabel='Угол рассеяния θ, градусы')
    for value, color, label, style in zip([bem, native, wave], colors, labels, ['-', '--', '-']):
        axes[1].semilogy(theta, value, style, color=color, lw=1.9, label=label)
    axes[1].set(xlim=(0,30), title='После полного усреднения ориентаций',
                ylabel='⟨M₁₁⟩, тот же абсолютный масштаб', xlabel='Угол рассеяния θ, градусы')
    for error, color, style in zip([native_error, wave_error], colors[1:], ['--','-']):
        axes[2].plot(theta, error, style, color=color, lw=1.8)
    axes[2].axhline(0, color='#333333', lw=.8)
    axes[2].axhspan(-5,5,color=colors[2],alpha=.10)
    axes[2].set_yscale('symlog', linthresh=5)
    axes[2].set_yticks([-20, -5, 0, 5, 20, 100, 500], labels=['−20', '−5', '0', '5', '20', '100', '500'])
    axes[2].set(xlim=(0,30), title='Расхождение с усреднённым BEM',
                ylabel='(⟨M₁₁⟩ / ⟨M₁₁,BEM⟩ − 1), %', xlabel='Угол рассеяния θ, градусы')
    fig.suptitle('Усреднение по ориентациям · ka = 20, m = 1.3, H/D = 1\n'
                 f'Вперёд: исходный MBS {native_error[0]:+.2f}%, с поправкой {wave_error[0]:+.2f}%', fontsize=15, y=.98)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,.065), ncol=3, frameon=False)
    fig.text(.5,.025,'Без подгонки. Слева исходный MBS — fullauto 1%; в центре и справа — та же квадратура, что у поправки. Шкала ошибки: symlog.',
             ha='center', fontsize=10)
    for extension in ['png','pdf']:
        fig.savefig(OUT/f'orientation_averaged_M11.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.plot(theta, 100*orientation_change, label='Направления падения: 45×16 → 89×30')
    ax.plot(theta, 100*phi_change, label='Вращение вокруг луча: 32 → 64 узла')
    ax.axhline(0,color='#333333',lw=.8)
    ax.set(xlim=(0,30),xlabel='Угол рассеяния θ, градусы',ylabel='Изменение среднего M₁₁, %',
           title='Численная сходимость усреднения новой поправки')
    ax.legend()
    for extension in ['png','pdf']:
        fig.savefig(DATA/f'quadrature_convergence.{extension}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
