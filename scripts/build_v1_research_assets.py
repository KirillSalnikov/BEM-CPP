#!/usr/bin/env python3
"""Collect audited research evidence once; regenerate release figures from it.

Default: use the portable reference/v1.0.0/research dataset. --collect-from
requires the original research tree, preserves source hashes and publishes
Mueller curves/summary data only, never large T matrices or surface currents.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'reference/v1.0.0/research'
FIG = ROOT/'manual_assets/v1.0.0'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def collect(tree):
    provenance = []

    def record(relative):
        path = tree/relative
        provenance.append(dict(path=relative, sha256=digest(path), bytes=path.stat().st_size))
        return path

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str):
            value = value.replace(str(tree.resolve())+'/', '')
            # External executable locations are machine-local, not release inputs.
            if value.startswith('/home/'):
                return 'external-local/'+Path(value).name
        return value

    def read(relative):
        return clean(json.loads(record(relative).read_text()))

    b = 'benchmarks/acceleration_plan_20261101'
    data = dict(schema_version=1, date='2026-11-01', release='1.0.0',
                metric='max_theta,i,j abs(M_candidate-M_reference)/M_reference_11; relative Csca/Cext separately',
                current_particle=dict(ri=1.3116, wavelength_um=.532, height_over_width_flats=10/7))
    historical = read('benchmarks/cutoff_study_20261027/summary.json')
    data['historical_cutoff'] = {k: historical[k] for k in ['geometry', 'ri', 'measured', 'linear_fit', 'limitations']}
    data['current_cutoff'] = [read(f'{b}/P7/{case}_selection.json') for case in ['ka5', 'ka10', 'ka40']]
    data['shape'] = read('runs/aspect_small_20261030/report.json')
    data['mesh_series'] = [read(f'runs/size_series_n13116_0532_12h_20261028/ka{ka}_assessment.json') for ka in [10, 20]]
    steps = read('benchmarks/interpolation_step_20261031/summary.json')
    data['interpolation_steps'] = dict(
        scope='Same-mesh T interpolation at ka3/6, m=1.3116, H/D_flats=10/7; not a ka40 certificate.',
        cases=[dict(ka=c['ka0'], models=[{k: m[k] for k in ['method', 'step', 'training_count', 'control_count',
                  'maximum_error', 'maximum_mueller_error', 'passes_1pct']} for m in c['T_methods']]) for c in steps['cases']])
    data['interpolation_rejected_models'] = read('benchmarks/interpolation_improvement_20261031/summary.json')
    data['multisize_holdouts'] = read('runs/multisize_n13116_0532_20261028/holdout_audit_20261028/report.json')
    series = read('runs/multisize_n13116_0532_20261028/results.json')
    data['multisize_series'] = {k: series[k] for k in ['state', 'ka_range', 'sizes', 'stage', 'ri', 'wavelength_um',
        'grid_evaluation_s', 'additional_BEM_solves_for_grid', 'Euler_quadrature_nodes', 'checks',
        'requested_range', 'full_requested_range_achieved']}
    data['multisize_series']['height_over_width_flats'] = 1.
    data['taylor'] = read('benchmarks/size_taylor_20261031/comparison_summary.json')
    data['adaptive_recenter'] = read(f'{b}/P11/recentering_assessment.json')
    data['reciprocal_extension'] = read(f'{b}/P7/reciprocal_extension/report.json')
    tolerance = read('benchmarks/solver_tolerance_20261031/summary.json')
    data['solver_tolerance'] = {k: tolerance[k] for k in ['comparison_table', 'limits', 'production_tolerance_changed',
        'full_ka40_analytic_average_at_loose_tolerance_verified']}
    data['adaptive_tolerance'] = read('benchmarks/adaptive_tolerance_20261101/summary.json')
    hybrid = read('benchmarks/production_hybrid_20261031/summary.json')
    data['hybrid'] = {k: hybrid[k] for k in ['measurements', 'scope', 'acceptance_tolerance', 'complete_small_T_and_analytic_SO3']}
    fft = read('benchmarks/fft_aspect_20261030/summary.json')
    data['face_fft'] = dict(large_packet_timing={k: v for k, v in fft['large_packet_timing'].items() if k != 'runs'},
                            limitations=fft['limitations'])
    large = 'runs/ka40_n1p3116_l0p532_aspect10over7_fft_blend_20261030/main_ref6_L64'
    full = read(large+'/summary.json'); migration = read(large+'/solver_migration.json')
    data['ka40_completed'] = {k: full[k] for k in ['mode_count', 'completed_columns', 'complete', 'elapsed_this_run_s',
        'end_to_end_s', 'maximum_strict_residual', 'new_direction_field_error', 'analytic_so3_s',
        'scattering_cross_section', 'extinction_cross_section', 'maximum_strict_residual_applies_to']}
    data['ka40_completed'].update(parameters=full['parameters'], operator=full['operator'],
        columns_reused_before_final_segment=migration['copied_columns'], final_segment_packets=len(full['pairs_this_run']),
        cold_full_build_time_measured=False, timing_scope='Final resumed segment, not a cold complete T construction.',
        T_sha256=digest(tree/large/'tmatrix.npy'), T_bytes=(tree/large/'tmatrix.npy').stat().st_size,
        canonical_T_unmodified=True)
    mbs_path = 'runs/mbs_surface_shadow_ka40_20261101/comparison'
    data['mbs_ka40'] = read(mbs_path+'/comparison_summary.json')
    with np.load(record(mbs_path+'/comparison_all16.npz')) as z:
        data['mbs_curves'] = {k: z[k].tolist() for k in ['theta', 'bem', 'new_coherent', 'new_incoherent']}
    plan = read(f'{b}/status.json')
    data['acceleration_plan'] = [{k: item.get(k) for k in ['id', 'title', 'status', 'completion_proven', 'evidence']}
                                 for item in plan['items']]
    extras = {
        'interpolation_all16.png': 'runs/multisize_n13116_0532_20261028/plots_four_holdouts/all16_normalized.png',
        'interpolation_M11.png': 'runs/multisize_n13116_0532_20261028/plots_four_holdouts/M11_four_holdouts.png',
        'shape_cutoff.png': 'runs/aspect_small_20261030/cutoff_convergence.png',
        'shape_mesh.png': 'runs/aspect_small_20261030/mesh_convergence.png',
        'taylor.png': 'benchmarks/size_taylor_20261031/size_continuation_comparison.png',
        'reciprocal_extension.png': f'{b}/P7/reciprocal_extension/comparison.png',
        'recenter.png': f'{b}/P11/recentering_comparison.png',
    }
    for case in ['ka2_aspect0p2','ka2_aspect1','ka2_aspect5','ka5_aspect0p2','ka5_aspect1','ka5_aspect5']:
        extras[f'{case}_all16.png'] = f'runs/aspect_small_20261030/{case}/mesh_all16.png'
    for destination, source in extras.items():
        shutil.copy2(record(source), FIG/destination)
    dump(DATA/'data.json', data)
    dump(DATA/'source_manifest.json', dict(sources=provenance, published_data_sha256=digest(DATA/'data.json'),
        copied_figures={name: digest(FIG/name) for name in extras},
        scope='Original numerical summaries and Mueller curves only. Large T matrices, currents and private credentials excluded.'))
    return data


def figures(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .22})

    def save(fig, name):
        fig.savefig(FIG/(name+'.png'), dpi=180, bbox_inches='tight')
        fig.savefig(FIG/(name+'.pdf'), bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    old = data['historical_cutoff']['measured']; new = data['current_cutoff']
    axes[0].plot([r['ka'] for r in old], [r['L_1pct'] for r in old], 'o-', label='m=1.3, H/D=1: minimum')
    axes[0].plot([r['physical_key']['ka'] for r in new], [r['minimum_L'] for r in new], 's-', label='m=1.3116, H/D=10/7: minimum')
    axes[0].plot([r['physical_key']['ka'] for r in new], [r['selected_L'] for r in new], 's--', label='H/D=10/7: +2-order guard')
    axes[0].set(xlabel='ka (equal-volume radius)', ylabel='L', title='Measured cutoffs; no ka=100 extrapolation')
    axes[0].legend(fontsize=8)
    for selection in new:
        rows = selection['records']
        axes[1].semilogy([r['order'] for r in rows], [max(100*r['acceptance_error'],1e-9) for r in rows],
                         label=f"ka={selection['physical_key']['ka']:g}")
    axes[1].axhline(1, color='black', ls='--'); axes[1].set(xlabel='L', ylabel='All16 / M11 + cross-section error, %',
        ylim=(1e-4, 200), title='Truncation relative to finite reference L'); axes[1].legend()
    save(fig, 'order_vs_size')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for ka in [2, 5]:
        rows = [r for r in data['shape']['records'] if r['ka']==ka]
        axes[0].plot([r['aspect'] for r in rows], [r['minimum_L_1pct_over_M11'] for r in rows], 'o-', label=f'ka={ka}')
        axes[1].plot([r['aspect'] for r in rows], [r['mesh_change_pct'] for r in rows], 'o-', label=f'ka={ka}')
    for ax in axes:
        ax.set_xscale('log'); ax.set_xlabel('H / width across flats'); ax.legend()
    axes[0].set(ylabel='Minimum L at 1% / M11', title='Oblate and elongated hexagonal prisms')
    axes[1].axhline(1, color='black', ls='--'); axes[1].set(ylabel='Adjacent-mesh all16 change / M11, %', title='Surface error is a separate control')
    save(fig, 'shape_and_mesh')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for ax, case in zip(axes, data['interpolation_steps']['cases']):
        for method, marker in [('linear','o'), ('local_cubic','s'), ('AAA','^')]:
            rows = sorted([r for r in case['models'] if r['method']==method], key=lambda x:x['step'])
            ax.loglog([r['step'] for r in rows], [100*r['maximum_error'] for r in rows], marker+'-', label=method)
        ax.axhline(1,color='black',ls='--');ax.set(xlabel='Anchor spacing in ka',ylabel='Largest tested error, %',title=f"Center ka={case['ka']:g}");ax.legend()
    save(fig, 'interpolation_spacing')
    fig, ax = plt.subplots(figsize=(8.5,4.3), layout='constrained')
    records=data['multisize_holdouts']['records'];x=np.arange(4)
    for offset,key,label in [(-.25,'maximum_all16_error_over_reference_M11','All16 / M11'),(0,'Csca_relative_error','Csca'),(.25,'T_relative_frobenius_error','Complex T Frobenius')]:
        ax.bar(x+offset,[100*r[key] for r in records],width=.24,label=label)
    ax.axhline(1,color='black',ls='--');ax.set(xticks=x,xticklabels=[str(r['ka']) for r in records],xlabel='Independent size',ylabel='Error, %',title='Nine training sizes; four excluded controls, H/D=1');ax.legend()
    save(fig,'holdout_errors')
    fig, ax = plt.subplots(figsize=(9,4.5),layout='constrained')
    rows=data['solver_tolerance']['comparison_table']
    for key,label in [('ka3_averaged_error_percent','ka3, full SO(3), ref2'),('ka20_averaged_error_percent','ka20, full SO(3), ref2'),('ka40_axial_fixed_error_percent','ka40, fixed axial, ref4'),('ka40_tilt_fixed_error_percent','ka40, fixed tilted, ref4')]:
        selected=[r for r in rows if r.get(key) is not None]
        ax.loglog([r['tolerance'] for r in selected],[r[key] for r in selected],'o-',label=label)
    ax.axhline(1,color='black',ls='--');ax.set(xlabel='Original-operator residual target',ylabel='All16 / M11 error, %',title='Residual is not an observable-error guarantee');ax.legend(fontsize=8)
    save(fig,'residual_accuracy')
    curves=data['mbs_curves'];theta=np.asarray(curves['theta']);reference=np.asarray(curves['bem'])
    matrices=[reference,np.asarray(curves['new_coherent']),np.asarray(curves['new_incoherent'])]
    labels=['BEM, analytic SO(3)','MBS coherent, fixed samples','MBS incoherent, fixed samples']
    fig, axes=plt.subplots(4,4,figsize=(12,10),sharex=True,layout='constrained')
    for i in range(4):
        for j in range(4):
            ax=axes[i,j]
            for matrix,label in zip(matrices,labels):
                values=matrix[:,i,j] if (i,j)==(0,0) else matrix[:,i,j]/reference[:,0,0]
                ax.plot(theta,values,lw=1,label=label)
            if (i,j)==(0,0):ax.set_yscale('log')
            ax.set_title(f'M{i+1}{j+1}'+(' (absolute)' if (i,j)==(0,0) else ' / BEM M11'))
            if i==3:ax.set_xlabel('Scattering angle, deg')
    axes[0,0].legend(fontsize=6);fig.suptitle('ka=40, m=1.3116, H/D_flats=10/7; common physical normalization')
    save(fig,'mbs_ka40_all16')
    fig, axes=plt.subplots(2,1,figsize=(9,7),sharex=True,layout='constrained')
    for matrix,label in zip(matrices,labels):axes[0].semilogy(theta,matrix[:,0,0],label=label)
    for matrix,label in zip(matrices[1:],labels[1:]):axes[1].plot(theta,100*(matrix[:,0,0]/reference[:,0,0]-1),label=label)
    axes[0].set(ylabel='M11, common absolute units',title='ka=40: coherent and incoherent MBS versus BEM');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Scattering angle, deg',ylabel='M11 difference / BEM, %');axes[1].axhline(0,color='black',lw=.6)
    save(fig,'mbs_ka40_M11')
    # Compact, portable numerical source for external plotting.
    columns=[theta]+[matrix[:,i,j] for matrix in matrices for i in range(4) for j in range(4)]
    names=['theta_deg']+[f'{label}_M{i+1}{j+1}' for label in ['bem','mbs_coherent','mbs_incoherent'] for i in range(4) for j in range(4)]
    np.savetxt(DATA/'mbs_ka40_all16.csv',np.column_stack(columns),delimiter=',',header=','.join(names),comments='')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect-from',type=Path,help='Original research worktree; default regenerates from portable data')
    args=parser.parse_args();DATA.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    data=collect(args.collect_from.resolve()) if args.collect_from else json.loads((DATA/'data.json').read_text())
    figures(data)
    print('Release research figures generated from recorded evidence.')


if __name__=='__main__':main()
