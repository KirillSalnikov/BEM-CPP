#!/usr/bin/env python3
"""Summarize independent size-continuation controls; count only accurate targets."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from bounded_bem_run import atomic_json
from benchmark_size_taylor import plot
from study_tmatrix_cutoff import error_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    root=parser.parse_args().directory.resolve()
    cases=[]
    for ka in [3,6]:
        for ref in [1,2]:
            folder=root/f'ka{ka}_ref{ref}'
            report=json.loads((folder/'report.json').read_text())
            model=json.loads((folder/'model.json').read_text())
            setup=model['timings']['total_setup_s']
            records=report['records']
            for name,timing in report['timings'].items():
                accepted=[r for r in records if r['methods'][name]['maximum_all16_error_over_reference_M11']<=.01]
                direct=sum(r['direct_setup_s']+r['direct_analytic_SO3_s'] for r in accepted)
                predicted=sum(r['methods'][name]['evaluation_s']+r['methods'][name]['analytic_SO3_s'] for r in accepted)
                timing.update(accepted_target_count=len(accepted),accepted_direct_s=direct,
                    accepted_model_with_setup_s=setup+predicted,accepted_speedup_including_setup=direct/(setup+predicted),
                    timing_scope='Compute incl. analytic SO3; excludes independent checks and artifact serialization.')
            atomic_json(folder/'report.json',report)
            curves=[]
            for r in records:
                with np.load(folder/f'control_{r["ka"]:.8f}/comparison.npz') as data:
                    curves.append(dict(data))
            plot(folder,report,curves)
            # Same four independently solved neighboring sizes for every case.
            local=[r for r in records if abs(r['delta']) in [.125,.25]]
            assert len(local)==4
            direct=np.mean([r['direct_setup_s']+r['direct_analytic_SO3_s'] for r in local])
            methods={}
            for name in ['Taylor_6','Pade_4_2']:
                incremental=float(np.mean([r['methods'][name]['evaluation_s']+r['methods'][name]['analytic_SO3_s'] for r in local]))
                methods[name]=dict(errors_pct={str(step):100*max(r['methods'][name]['maximum_all16_error_over_reference_M11']
                    for r in records if abs(r['delta'])==step) for step in [.125,.25,.5]},
                    mean_incremental_target_s=incremental,
                    four_targets_including_setup_s=setup+4*incremental,
                    four_target_speedup=4*direct/(setup+4*incremental),
                    estimated_break_even_targets=math.floor(setup/(direct-incremental))+1,
                    estimated_20_target_speedup=20*direct/(setup+20*incremental))
            cases.append(dict(ka0=ka,ref=ref,dofs=model['timings']['bridge']['dofs'],setup_s=setup,
                direct_per_target_s=float(direct),four_direct_targets_s=float(4*direct),methods=methods,
                reference_max_residual=max(r['direct_residual'] for r in records),
                local_max_modal_tail_pct=100*max(r['modal_tail_error'] for r in local),
                records=records,curves=curves,folder=folder))

    mesh_errors=[]
    for ka in [3,6]:
        coarse=next(c for c in cases if c['ka0']==ka and c['ref']==1)
        fine=next(c for c in cases if c['ka0']==ka and c['ref']==2)
        for record,curve in zip(fine['records'],fine['curves']):
            if abs(record['delta'])>.25:continue
            index=next(i for i,r in enumerate(coarse['records']) if r['ka']==record['ka'])
            metrics=error_metrics(coarse['curves'][index]['direct'],curve['direct'],curve['theta_deg'])
            mesh_errors.append(dict(ka=record['ka'],error_pct=100*metrics['maximum_all16_error_over_reference_M11']))
    summary=dict(method='Analytic BEM Taylor moments (AWE), single center, degree 6; optional common-denominator Pade 4/2',
        off_center_T_training=0,independent_control_solves=sum(len(c['records']) for c in cases),
        Euler_quadrature_nodes=0,ri=1.3116,wavelength=.532,H_over_width=10/7,
        error_definition='100 * max over theta (0:0.25:180 degrees) and i,j of abs(M_model-M_direct)/M_direct_11',
        timing_scope='CPU computation including analytic SO3, excluding validation and artifact serialization; four threads per BEM job, two SO3 threads.',
        range_scope='Accuracy established only at the discrete tested offsets, not every intermediate real size.',
        forecast_scope='Break-even and 20-size gains extrapolate measured per-target cost, not 20 validated sizes or GPU/FMM performance.',
        mesh_scope='Same-mesh continuation verified; absolute Maxwell/mesh accuracy not certified.',
        cases=[{k:v for k,v in c.items() if k not in ['records','curves','folder']} for c in cases],
        coarse_to_fine_direct_differences=mesh_errors)
    atomic_json(root/'comparison_summary.json',summary)

    with PdfPages(root/'size_continuation_comparison.pdf') as pdf:
        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
        for ax,ka in zip(axes[0],[3,6]):
            for case in [c for c in cases if c['ka0']==ka]:
                for method,style in [('Taylor_6','-'),('Pade_4_2','--')]:
                    rows=sorted(case['records'],key=lambda r:r['delta'])
                    ax.semilogy([r['delta'] for r in rows],
                        [100*r['methods'][method]['maximum_all16_error_over_reference_M11'] for r in rows],
                        style,marker='.',color='#1765a3' if case['ref']==1 else '#dc672a',
                        label=f'{method}, ref {case["ref"]}')
            ax.axhline(1,color='red',ls=':',label='1% limit')
            ax.set(xlabel='delta ka',ylabel='max all16 error / direct M11, %',title=f'One center ka={ka}')
            ax.grid(alpha=.25);ax.legend(fontsize=8)
        labels=[f'ka={c["ka0"]}, ref={c["ref"]}' for c in cases];x=np.arange(len(cases))
        axes[1,0].bar(x-.18,[c['four_direct_targets_s'] for c in cases],width=.36,label='4 independent BEM sizes')
        axes[1,0].bar(x+.18,[c['methods']['Taylor_6']['four_targets_including_setup_s'] for c in cases],width=.36,
            label='Taylor: full setup + 4 sizes')
        axes[1,0].set(xticks=x,xticklabels=labels,ylabel='CPU compute seconds including analytic SO(3)',title='Offsets ±0.125, ±0.25: all pass 1%')
        axes[1,0].tick_params(axis='x',labelsize=8);axes[1,0].legend(fontsize=8)
        axes[1,1].axis('off')
        rows=[[label,f'{c["setup_s"]:.2f}',f'{c["direct_per_target_s"]:.2f}',
               f'{c["methods"]["Taylor_6"]["mean_incremental_target_s"]:.2f}',
               str(c['methods']['Taylor_6']['estimated_break_even_targets'])] for label,c in zip(labels,cases)]
        table=axes[1,1].table(cellText=rows,colLabels=['Case','Setup, s','BEM / size, s','Model / size, s','Break-even N*'],
            loc='center',cellLoc='center',colWidths=[.27,.15,.19,.2,.19])
        table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,2)
        axes[1,1].set_title('Start-up cost matters; per-size cost includes SO(3)')
        axes[1,1].text(.02,.05,'* Estimated from measured timings, within a validated neighborhood.\n'
            'Same-mesh continuation error; no absolute mesh-accuracy claim.\n'
            'Artifact writes and independent validation excluded from timings.',fontsize=9,transform=axes[1,1].transAxes)
        fig.suptitle('Single-center size continuation: analytic Taylor / Padé vs independent BEM\n'
            'Hexagonal column; m=1.3116, wavelength=0.532, H/width=10/7; analytic orientation average')
        fig.savefig(root/'size_continuation_comparison.png',dpi=150);pdf.savefig(fig);plt.close(fig)

        for case in [c for c in cases if c['ref']==2]:
            idx=next(i for i,r in enumerate(case['records']) if r['delta']==.25)
            data=case['curves'][idx];theta=data['theta_deg'];ref=data['direct'];norm=ref[:,0,0]
            fig,axes=plt.subplots(4,4,figsize=(15,12),layout='constrained')
            for i in range(4):
                for j in range(4):
                    ax=axes[i,j];scale=1 if (i,j)==(0,0) else norm
                    for name,color,style in [('direct','#1765a3','-'),('Taylor_6','#21864b','--'),('Pade_4_2','#dc672a',':')]:
                        ax.plot(theta,data[name][:,i,j]/scale,style,color=color,label=name,lw=1.3)
                    err=np.abs(data['Taylor_6'][:,i,j]-ref[:,i,j])/norm
                    worst=int(np.argmax(err));ys=ref[:,i,j]/scale
                    ax.plot(theta[worst],ys[worst],'o',color='#21864b',markersize=3)
                    ax.set(title=f'M{i+1}{j+1}: Taylor max {100*err[worst]:.3g}%',xlim=(0,180),xlabel='theta, degrees')
                    if (i,j)==(0,0):ax.set_yscale('log')
                    ax.grid(alpha=.2)
            axes[0,0].legend(fontsize=8)
            fig.suptitle(f'ka={case["ka0"]} → {case["ka0"]+.25:g}, ref=2; analytic orientation average\n'
                'M11 absolute; other entries / direct M11. Dots: maximum Taylor error / M11 for each entry.')
            fig.savefig(root/f'all16_ka{case["ka0"]}_ref2.png',dpi=150);pdf.savefig(fig);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
