#!/usr/bin/env python3
"""Publish existing complete Mueller comparisons without starting new solves."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    sources = {
        'coherent': ROOT/'runs/mbs_fullauto_ka40_n1p3116_20261030/comparison_refresh_20261031/coherent_1pct_snapshot',
        'incoherent': ROOT/'runs/mbs_incoherent_ka40_20261031/fullauto_1pct/comparison',
    }
    data, reports = {}, {}
    for key, source in sources.items():
        report = read(source/'comparison_summary.json')
        current = Path(report['mbs_directory'])/'result/mueller_fullauto.dat'
        assert hashlib.sha256(current.read_bytes()).hexdigest()==report['mbs_table_sha256']
        assert report['mbs_independent_validation']['passed']
        assert not report['conversion']['fitted_scale']
        assert not report['conversion']['empirical_correction']
        assert report['bem_reevaluation']['Euler_quadrature_nodes']==0
        reports[key] = report
        with np.load(source/'comparison_all16.npz') as values:
            data[key] = {k:values[k] for k in values.files}
    assert reports['incoherent']['mbs_converged_and_validated']
    stop = read(ROOT/'runs/mbs_fullauto_ka40_n1p3116_20261030/user_stop_final_checks_20261031.json')
    assert stop['stopped'] and stop['table_sha256_after']==reports['coherent']['mbs_table_sha256']
    for key in ['parameters','bem_directory','conversion']:
        assert reports['coherent'][key]==reports['incoherent'][key]
    ref = data['coherent']
    np.testing.assert_array_equal(ref['theta'], data['incoherent']['theta'])
    np.testing.assert_allclose(ref['bem'],data['incoherent']['bem'],rtol=1e-12,atol=1e-10)
    old = ROOT/'runs/multisize_n13116_0532_20261028'
    audit = read(old/'holdout_audit_20261028/report.json')
    assert audit['model_refitted'] is False and audit['Euler_quadrature_nodes']==0
    meta = read(old/'plots_four_holdouts/plot_manifest.json')
    assert hashlib.sha256((old/'model_1/model.json').read_bytes()).hexdigest()==meta['model_sha256']
    curves = np.load(old/'plots_four_holdouts/curves.npz')
    direct = curves['mueller_um2_per_sr'][curves['kind']=='BEM']
    interpolated = curves['mueller_um2_per_sr'][curves['kind']=='interpolated']
    ka = curves['ka'][curves['kind']=='BEM']; theta_old = curves['theta_deg']
    np.testing.assert_array_equal(ka,curves['ka'][curves['kind']=='interpolated'])
    error_old = 100*abs(interpolated-direct)/direct[:,:,0,0,None,None]
    for index, record in enumerate(audit['records']):
        assert ka[index]==record['ka']
        np.testing.assert_allclose(error_old[index].max(),100*record['maximum_all16_error_over_reference_M11'],rtol=1e-10)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.grid':True,
                         'grid.alpha':.2,'axes.spines.top':False,'axes.spines.right':False})
    colors = {'coherent':'#21814e','incoherent':'#df6828'}
    labels = {'coherent':'MBS когерентный 1%: независимая проверка пройдена; финальные проверки остановлены',
              'incoherent':'MBS некогерентный 1%: расчёт завершён'}
    title40 = 'ka=40 · m=1.3116 · λ=0.532 мкм · H/ширина=10/7\nПолное усреднение по ориентациям: BEM — аналитическое SO(3)'
    foot40 = 'Общая нормировка, без подгонки. Версии MBS различаются: когерентный 1df49a, некогерентный 96ec35a.'
    with PdfPages(out/'comparisons_and_interpolation.pdf') as book:
        def save(fig,name):
            book.savefig(fig,bbox_inches='tight')
            for ext in ['pdf','png']:fig.savefig(out/f'{name}.{ext}',dpi=150,bbox_inches='tight')
            plt.close(fig)
        def draw40(ax,i,j):
            dense=ref['bem_drawing']; denominator=ref['bem'][:,0,0]
            ax.plot(ref['bem_drawing_theta'],dense[:,i,j]/(dense[:,0,0] if (i,j)!=(0,0) else 1),
                color='#1765a3',label='BEM: полная T, ref=6, L=64',lw=1.6)
            for key,style in [('coherent','-.'),('incoherent','--')]:
                ax.plot(ref['theta'],data[key]['mbs'][:,i,j]/(denominator if (i,j)!=(0,0) else 1),
                    style,color=colors[key],label=labels[key],lw=1.3)
            ax.set_xlim(0,180);ax.set_xlabel('θ, °')
        fig,axes=plt.subplots(1,3,figsize=(16,5.8))
        fig.subplots_adjust(bottom=.32,top=.77,wspace=.3)
        fig.suptitle(title40,fontsize=14)
        draw40(axes[0],0,0);axes[0].set(yscale='log',title='M₁₁: все углы',ylabel='k² dσ/dΩ')
        draw40(axes[1],0,0);axes[1].set(xlim=(0,12),title='M₁₁: вперёд')
        for key in data:
            axes[2].plot(ref['theta'],100*(data[key]['mbs'][:,0,0]/ref['bem'][:,0,0]-1),color=colors[key])
        axes[2].axhline(0,color='.5',lw=.6);axes[2].set(title='Расхождение M₁₁',xlabel='θ, °',ylabel='(MBS/BEM − 1), %',xlim=(0,180))
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.06),fontsize=9)
        fig.text(.5,.02,foot40,ha='center',fontsize=9)
        save(fig,'ka40_M11_latest')
        fig,axes=plt.subplots(4,4,figsize=(16,14))
        fig.subplots_adjust(bottom=.17,top=.88,hspace=.5,wspace=.3);fig.suptitle(title40,fontsize=15)
        for i in range(4):
            for j in range(4):
                ax=axes[i,j];draw40(ax,i,j)
                ax.set_title(f'M{i+1}{j+1}'+(' / M₁₁ BEM' if (i,j)!=(0,0) else ' [k² dσ/dΩ]'))
                if (i,j)==(0,0):ax.set_yscale('log')
                annotations=[]
                for key,short in [('coherent','К'),('incoherent','НК')]:
                    e=100*abs(data[key]['mbs'][:,i,j]-ref['bem'][:,i,j])/ref['bem'][:,0,0];q=e.argmax()
                    value=data[key]['mbs'][q,i,j]/(ref['bem'][q,0,0] if (i,j)!=(0,0) else 1)
                    ax.scatter(ref['theta'][q],value,s=17,color=colors[key],zorder=4)
                    annotations.append(f'{short}: {e[q]:.2f}% при {ref["theta"][q]:.1f}°')
                ax.text(.98,.97,'\n'.join(annotations),transform=ax.transAxes,ha='right',va='top',fontsize=7,
                    bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
                if (i,j)!=(0,0) and max(abs(data[k]['mbs'][:,i,j]/ref['bem'][:,0,0]).max() for k in data)<.02:ax.set_ylim(-.02,.02)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.045),fontsize=9)
        fig.text(.5,.015,foot40+'\nТочки: max |MBS−BEM|/M₁₁ BEM.',ha='center',fontsize=9)
        save(fig,'ka40_all16_latest')
        palette=['#1765a3','#e07820','#258454','#a34aa0']
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        fig.subplots_adjust(top=.84,bottom=.1,hspace=.35,wspace=.25)
        fig.suptitle('Проверенная интерполяция комплексной T: серия 28 октября, ka=10–12\n'
            'm=1.3116 · λ=0.532 мкм · H/ширина=1 · затем аналитическое SO(3)\n'
            'Это отдельная геометрия; интерполяция около ka=40 ещё считается.',fontsize=13)
        for k,ax in enumerate(axes.flat):
            ax.semilogy(theta_old,direct[k,:,0,0],color=palette[k],label='Прямой BEM')
            ax.semilogy(theta_old,interpolated[k,:,0,0],'--',color='#1c1c1c',lw=1,label='Интерполированная T')
            q=error_old[k,:,0,0].argmax()
            ax.scatter(theta_old[q],interpolated[k,q,0,0],color='#d73330',s=25,zorder=5)
            ax.set(title=f'ka={ka[k]:g}; max ΔM₁₁/M₁₁ = {error_old[k,:,0,0].max():.3f}%',
                xlabel='θ, °',ylabel='M₁₁, мкм²/ср',xlim=(0,180))
            ax.legend(fontsize=9)
        fig.text(.5,.025,'Четыре независимые проверочные точки не участвовали в подборе интерполятора.',ha='center')
        save(fig,'interpolation_four_sizes_M11')
        fig,axes=plt.subplots(4,4,figsize=(16,14));fig.subplots_adjust(top=.88,bottom=.13,hspace=.5,wspace=.3)
        fig.suptitle('Интерполяция T против прямого BEM: все 16 элементов, четыре проверочных размера\n'
            'Серия 28 октября · ka=10–12 · m=1.3116 · λ=0.532 мкм · H/ширина=1',fontsize=14)
        for i in range(4):
            for j in range(4):
                ax=axes[i,j];notes=[]
                for k,size in enumerate(ka):
                    denominator=direct[k,:,0,0] if (i,j)!=(0,0) else 1
                    ax.plot(theta_old,direct[k,:,i,j]/denominator,color=palette[k],label=f'ka={size:g}')
                    ax.plot(theta_old,interpolated[k,:,i,j]/denominator,'--',color=palette[k],lw=1)
                    q=error_old[k,:,i,j].argmax()
                    y=interpolated[k,q,i,j]/(direct[k,q,0,0] if (i,j)!=(0,0) else 1)
                    ax.scatter(theta_old[q],y,s=15,color=palette[k],zorder=4)
                    notes.append(f'{size:g}: {error_old[k,q,i,j]:.3f}%')
                ax.set(xlim=(0,180),xlabel='θ, °',title=f'M{i+1}{j+1}'+(' / M₁₁ BEM' if (i,j)!=(0,0) else ' [мкм²/ср]'))
                if (i,j)==(0,0):ax.set_yscale('log')
                elif abs(direct[:,:,i,j]/direct[:,:,0,0]).max()<.02:ax.set_ylim(-.02,.02)
                ax.text(.98,.97,'\n'.join(notes),transform=ax.transAxes,ha='right',va='top',fontsize=7,
                    bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.055),ncol=4)
        fig.text(.5,.02,'Сплошная: прямой BEM. Пунктир: интерполированная T. Точки: max |ΔMᵢⱼ|/M₁₁ прямого BEM.\n'
            'Одинаковая нормировка обеих кривых при каждом размере; эти данные не являются сравнением с MBS.',ha='center',fontsize=10)
        save(fig,'interpolation_four_sizes_all16')
    current = read(ROOT/'runs/multisize_ka40_n13116_aspect10over7_4pct_20261031/status.json')
    result = dict(created=datetime.now().isoformat(),MBScases={key:{
        k:r[k] for k in ['mbs_table_sha256','mbs_converged_and_validated','forward_M11_difference_percent',
            'backward_M11_difference_percent','worst_difference_percent_of_BEM_M11','weighted_relative_L2_all16']}
        for key,r in reports.items()},sources={k:str(v) for k,v in sources.items()},
        interpolation_source=str(old),interpolation_geometry_H_over_width=1,
        interpolation_error_metric='100 * max over theta, i, j abs(M_interp-M_direct)/M11_direct',
        interpolation_errors_percent={str(size):float(error_old[i].max()) for i,size in enumerate(ka)},
        ka40_interpolation_available=False,ka40_current_status=current)
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='ka40_current_status'},indent=2))


if __name__=='__main__':
    main()
