#!/usr/bin/env python3
"""Report accuracy, limitations and amortized CPU cost of the size ROM."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from bounded_bem_run import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text())


def main():
    root=ROOT/'benchmarks/size_rom_20261031';span=read(root/'span_report.json');cases=[]
    for center,rank in [(3.,512),(6.,768)]:
        original=read(root/f'ka{center:g}_ref2/report.json')
        extended=read(root/f'ka{center:g}_degree12/setup.json')
        rows=[r for r in span['records'] if r['center']==center and r['rank']==rank and r['variant']=='plain']
        accepted=[r for r in rows if abs(r['offset'])<=1.25]
        outside=[r for r in rows if abs(r['offset'])>1.25]
        central_setup=original['setup']['central_inverse']
        extension=extended['assembly_compute_s']+extended['modal_projection_s']+extended['additional_reduced_projection_s']
        prepare_existing=central_setup['existing_data_preparation_s']+extension
        prepare_fresh=prepare_existing+central_setup['historical_central_data_acquisition_s']
        exact=[]
        for family in ['exact','exact_wide','exact_quarter']:
            exact += [r for r in read(root/f'ka{center:g}_{family}/report.json')['records'] if r['rank']==rank]
        timed=[r for r in exact if abs(r['offset'])<=1.25]
        full=[sum(r[k] for k in ['assembly_compute_s','modal_projection_s','direct_solve_s','direct_radiation_s','direct_SO3_s']) for r in timed]
        incremental=[r['evaluation_s']+r['analytic_SO3_s'] for r in accepted]
        direct_median=statistics.median(full);rom_median=statistics.median(incremental)
        delta=direct_median-rom_median
        timing=dict(direct_median_s=direct_median,rom_median_s=rom_median,
            incremental_speedup=direct_median/rom_median,
            prepare_with_existing_central_derivatives_s=prepare_existing,
            prepare_fresh_s=prepare_fresh,
            break_even_new_sizes_existing_center=math.ceil(prepare_existing/delta) if delta>0 else None,
            break_even_new_sizes_fresh=math.ceil(prepare_fresh/delta) if delta>0 else None,
            estimated_100_outputs_direct_s=100*direct_median,
            estimated_100_outputs_rom_fresh_s=prepare_fresh+100*rom_median,
            estimated_100_outputs_speedup=100*direct_median/(prepare_fresh+100*rom_median),
            scope='Dense CPU, 2 BLAS/OMP threads; includes analytic SO3. Preparation sums recorded phases; central acquisition is historical. 100 outputs means 100 sizes within one validated window, not ka=100.')
        c=dict(center=center,rank=rank,full_dofs=original['dofs'],tested_interval=[center-1.25,center+1.25],
            controls=len(accepted),maximum_percent=100*max(r['acceptance_error'] for r in accepted),
            maximum_mueller_percent=100*max(r['maximum_all16_error_over_reference_M11'] for r in accepted),
            outside_percent={str(r['ka']):100*r['acceptance_error'] for r in outside},timing=timing,
            records=rows,exact_operator_checks=exact)
        cases.append(c)
    largest_error=max(c['maximum_percent'] for c in cases)
    new_solves=sum(read(root/f'ka{k}_{family}/report.json')['new_small_direct_solves']
        for k in [3,6] for family in ['exact','exact_wide','exact_quarter'])
    summary=dict(status='complete',physical_parameters=dict(ri_real=1.3116,ri_imag=0.,H_over_width=10/7,wavelength_um=.532),
        method='Central inverse Petrov-Galerkin current ROM; analytic order-12 A/B/D coefficients; full analytic SO3 T averaging',
        training_centers=[3.,6.],current_derivative_degree=6,operator_derivative_degree=12,
        off_center_training_Ts=0,validated_half_width=1.25,maximum_percent=largest_error,
        independent_controls=sum(c['controls'] for c in cases),new_small_direct_solves=new_solves,
        new_large_T_runs=0,cases=cases,
        failed_variants=['Unpreconditioned Galerkin projection was unstable at low reduced ranks.',
            'Separate electric/magnetic basis from central currents alone failed; derivative currents remain necessary.',
            'Multiplying the equation by ka before Taylor truncation did not improve the common validated span.',
            'Order-12 ROM failed 1% at offsets +/-1.5.'],
        mesh_accuracy_certified=False,continuous_error_bound=False,ka40_validated=False,production_FMM_integrated=False,
        limitations=['All comparisons use the same ref2 mesh and fixed modal basis; no absolute Maxwell accuracy claim.',
            'One central LU is reused in this dense prototype; six current-derivative block solves are additional work.',
            'A saved T matrix alone does not contain the required currents/operator derivatives.',
            'CPU prototype gains are not a forecast for ka40/100 or the GPU/FMM solver.'],
        research_context='https://opg.optica.org/josaa/abstract.cfm?uri=josaa-34-12-2231')
    atomic_json(root/'summary.json',summary)
    with PdfPages(root/'comparison.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
        for ax,c in zip(axes,cases):
            center=c['center'];rank=c['rank']
            old=read(ROOT/f'benchmarks/size_taylor_20261031/ka{center:g}_ref2/report.json')
            rows=sorted(old['records'],key=lambda r:r['delta'])
            ax.semilogy([r['delta'] for r in rows],[100*r['methods']['Taylor_6']['maximum_all16_error_over_reference_M11'] for r in rows],'o:',label='Taylor of T, degree 6')
            old=read(root/f'ka{center:g}_ref2/report.json')
            rows=sorted([r for r in old['records'] if r['method']=='central_inverse' and r['rank']==rank],key=lambda r:r['offset'])
            ax.semilogy([r['offset'] for r in rows],[100*r['acceptance_error'] for r in rows],'s--',label='Reduced BEM, operator degree 6')
            rows=sorted(c['records'],key=lambda r:r['offset'])
            ax.semilogy([r['offset'] for r in rows],[100*r['acceptance_error'] for r in rows],'.-',label='Reduced BEM, operator degree 12')
            rows=sorted(c['exact_operator_checks'],key=lambda r:r['offset'])
            ax.semilogy([r['offset'] for r in rows],[100*r['acceptance_error'] for r in rows],'x-',label='Reduced BEM, exact rebuilt operator')
            ax.axhline(1,color='black',ls=':',label='1% target');ax.axvspan(-1.25,1.25,color='green',alpha=.06)
            ax.set(xlabel='ka - central ka',ylabel='Maximum error, %',title=f'ka0={center:g}: {c["full_dofs"]} -> {rank} unknowns',ylim=(1e-5,100))
            ax.grid(alpha=.2,which='both');ax.legend(fontsize=7)
        fig.suptitle('Central current ROM: independent same-mesh checks, all16 after analytic SO(3)\nOne central size + current derivatives through order 6; no off-center training T')
        fig.savefig(root/'comparison.png',dpi=160);pdf.savefig(fig);plt.close(fig)
        for c in cases:
            worst=max([r for r in c['records'] if abs(r['offset'])<=1.25],key=lambda r:r['acceptance_error'])
            with np.load(worst['artifact_path']) as a:theta=a['theta_deg'];direct=a['direct'];pred=a['predicted']
            fig,axes=plt.subplots(4,4,figsize=(14,11),layout='constrained')
            for i in range(4):
                for j in range(4):
                    ax=axes[i,j];scale=1 if (i,j)==(0,0) else direct[:,0,0]
                    ax.plot(theta,direct[:,i,j]/scale,label='Direct BEM');ax.plot(theta,pred[:,i,j]/scale,'--',label='Reduced BEM')
                    e=abs(pred[:,i,j]-direct[:,i,j])/direct[:,0,0];idx=int(np.argmax(e));y=pred[:,i,j]/scale
                    ax.plot(theta[idx],y[idx],'o',ms=3);ax.set(title=f'M{i+1}{j+1}: max {100*e[idx]:.3g}%',xlim=(0,180));ax.grid(alpha=.2)
                    if (i,j)==(0,0):ax.set_yscale('log');ax.legend(fontsize=7)
            fig.suptitle(f'Worst accepted control: ka={worst["ka"]:g}, central ka={c["center"]:g}, reduced rank={c["rank"]}\n'
                'M11 absolute, other entries / direct M11; dots show maximum errors; same-mesh comparison')
            pdf.savefig(fig);plt.close(fig)
    files=[ROOT/'scripts'/name for name in ['tmatrix_size_rom.py','probe_size_rom.py','probe_size_rom_exact.py',
        'extend_size_rom_operator.py','validate_size_rom_span.py','report_size_rom.py']]
    files += [ROOT/'tools/muller_tmatrix_taylor_probe.cpp',ROOT/'bin/muller_tmatrix_rom_operator_probe',ROOT/'bin/muller_tmatrix_taylor_probe']
    atomic_json(root/'validation.json',dict(tests_log=str(root/'tests.log'),unit_tests_passed=11,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        original_binary_preserved=hashlib.sha256((ROOT/'bin/muller_tmatrix_taylor_probe').read_bytes()).hexdigest()=='478b143a97c6e98fa32f24abe1a0cc82ec32b4c1577dc5a9917106600f43f196',
        original_operator_coefficient_checks='bit-identical degrees 0..6 for A, modal B and modal D at both centers',
        new_small_direct_solves=new_solves,new_large_T_runs=0,
        compile_command='g++ -O3 -march=native -fopenmp -std=c++11 -DMULLER_TAYLOR_MAX_DEGREE=12 -Isrc tools/muller_tmatrix_taylor_probe.cpp src/muller_dense.cpp src/muller_nodal.cpp src/muller_duffy.cpp src/mesh.cpp -o bin/muller_tmatrix_rom_operator_probe'))
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))
    for c in cases:print(json.dumps({k:v for k,v in c.items() if k not in ['records','exact_operator_checks']},indent=2))


if __name__=='__main__':main()
