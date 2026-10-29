#!/usr/bin/env python3
"""Plot direct training/holdout Mueller curves and frozen T interpolants."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import SymmetricalLogLocator
from matplotlib.transforms import Bbox
import matplotlib.patheffects as effects
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--holdouts-only',action='store_true',help='Show only the independent BEM/interpolation pairs')
    parser.add_argument('--curves-from',type=Path,help='Reuse a previously exported curves.npz (physical units)')
    parser.add_argument('--mark-max-errors',action='store_true',help='Mark each holdout/entry maximum |delta Mij|/direct M11')
    args=parser.parse_args();run=args.run.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(run/'runtime_parity_20261028/scripts'))
    from tmatrix_so3_mstm import analytic_mueller_tmatrix
    import run_tmatrix_multisize as r
    plan=r.read(run/'plan.json');definition=r.read(run/'model_1/model.json')
    training=definition['training'];holdouts=definition['validation'];sizes=sorted(holdouts if args.holdouts_only else training+holdouts)
    assert len(sizes)==len(set(sizes)) and not (set(training)&set(holdouts))
    theta=np.linspace(0,180,1801);factor=(plan['wavelength_um']/(2*np.pi))**2
    direct={};interpolated={};identities={}
    os.environ['BEM_SO3_TMPDIR']=str(out/'scratch')
    if args.curves_from:
        with np.load(args.curves_from,allow_pickle=False) as data:
            theta=data['theta_deg'].copy()
            for ka,kind,value in zip(data['ka'],data['kind'],data['mueller_um2_per_sr']):
                if ka in sizes:
                    (direct if kind=='BEM' else interpolated)[float(ka)]=value.copy()
        assert set(direct)==set(sizes) and set(interpolated)==set(holdouts)
        assert all(np.all(np.isfinite(x)) for src in (direct,interpolated) for x in src.values())
        cached_manifest=json.loads(args.curves_from.with_name('plot_manifest.json').read_text())
        assert cached_manifest['model_sha256']==hashlib.sha256((run/'model_1/model.json').read_bytes()).hexdigest()
        identities={str(ka):cached_manifest['source_T_sha256'][str(ka)] for ka in sizes}
    else:
        model,parameters=r.load_model(run,plan,1)
    for ka in ([] if args.curves_from else sizes):
        t,report=r.load_sample(run,plan,ka)
        r.validate_family([parameters,r.family_parameters(report,plan)])
        value,meta=analytic_mueller_tmatrix(t,plan['order'],theta,precision='fp64',threads=2,cyclic_order=6)
        assert meta['Euler_quadrature_nodes']==0 and np.all(value[:,0,0]>0)
        direct[ka]=value*factor
        if ka in holdouts:
            value,meta=analytic_mueller_tmatrix(model.evaluate(ka),plan['order'],theta,precision='fp64',threads=2,cyclic_order=6)
            assert meta['Euler_quadrature_nodes']==0
            interpolated[ka]=value*factor
        source=r.sample_path(run,ka,plan['ref'],plan['order'])/'tmatrix.npy'
        identities[str(ka)]=hashlib.sha256(source.read_bytes()).hexdigest()
        print(f'Analytic averages ready: ka={ka:g}',flush=True)

    maxima={};maximum_records=[]
    for row in range(4):
        for col in range(4):
            for ka in holdouts:
                error=100*np.abs(interpolated[ka][:,row,col]-direct[ka][:,row,col])/direct[ka][:,0,0]
                index=int(np.argmax(error))
                reference=float(direct[ka][index,row,col]);prediction=float(interpolated[ka][index,row,col])
                record={'ka':ka,'element':f'M{row+1}{col+1}','theta_index':index,'theta_deg':float(theta[index]),
                        'maximum_error_percent_over_BEM_M11':float(error[index]),
                        'BEM_value_um2_per_sr':reference,'interpolated_value_um2_per_sr':prediction,
                        'BEM_M11_um2_per_sr':float(direct[ka][index,0,0]),
                        'own_element_relative_percent_at_this_angle':100*abs(prediction-reference)/abs(reference) if reference else None}
                maxima[row,col,ka]=record;maximum_records.append(record)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
        'axes.axisbelow':True,'savefig.facecolor':'white','pdf.fonttype':42})
    colors={ka:plt.colormaps['Blues'](.36+.60*i/(len(training)-1)) for i,ka in enumerate(training)}
    colors.update(dict(zip(holdouts,['#d96800','#ba302d','#9c3994','#7b6810'])))
    size_handles=[Line2D([],[],color=colors[ka],lw=2,label=f'{ka:g}'+(' *' if ka in holdouts else '')) for ka in sizes]
    style_handles=[Line2D([],[],color='#36649b',lw=1.2,label='BEM: 9 опорных размеров'),
        Line2D([],[],color='#ba302d',lw=.8,marker='o',mfc='white',ms=4,label='BEM: 4 проверочных размера *'),
        Line2D([],[],color='#ba302d',lw=2,ls='--',label='Интерполяция T: пунктир *')]
    if args.holdouts_only:
        size_handles=[Line2D([],[],color=colors[ka],lw=2,label=f'ka = {ka:g}') for ka in sizes]
        style_handles=[Line2D([],[],color='#555555',lw=1,marker='o',mfc='white',ms=4,label='Прямой BEM: линия и кружки'),
                       Line2D([],[],color='#555555',lw=2,ls='--',label='Интерполяция T: пунктир')]
    if args.mark_max_errors:
        style_handles.append(Line2D([],[],color='#555555',ls='',marker='o',mec='white',ms=6,
                                    label='Закрашенная точка: максимум ошибки'))
    material=f"Призма H/D по граням = 1 · n = {plan['ri']} · λ = {plan['wavelength_um']} мкм · ref={plan['ref']}, L={plan['order']}"

    def decorate(fig,title,grid=False):
        fig.suptitle(title,y=.985,fontsize=17 if grid else 16,fontweight='medium')
        fig.text(.5,.952 if grid else .935,material,ha='center',fontsize=10,color='#444444')
        fig.legend(handles=size_handles,loc='upper center',bbox_to_anchor=(.5,.932 if grid else .915),ncol=4 if args.holdouts_only else 7,
                   title='Независимые проверочные размеры' if args.holdouts_only else 'Размер ka; * — независимые проверки интерполяции',frameon=False,fontsize=9,title_fontsize=9)
        fig.legend(handles=style_handles,loc='upper center',bbox_to_anchor=(.5,(.884 if grid else .842) if args.holdouts_only else (.862 if grid else .810)),
                   ncol=3,frameon=False,fontsize=9)
        fig.text(.5,.018,'Полное аналитическое усреднение по ориентациям SO(3). Прямой BEM — сравнение на той же сетке и при том же L.',
                 ha='center',fontsize=9,color='#555555')
        if args.mark_max_errors:
            fig.text(.5,.065 if grid else .048,
                     'Процент у точки: maxθ 100·|Mij(интерп.) − Mij(BEM)| / M11(BEM); поиск по углам с шагом 0.1°.',
                     ha='center',fontsize=9,color='#333333')

    def curves(ax,row,col,normalized=False):
        for ka in sizes:
            denominator=direct[ka][:,0,0] if normalized else 1.
            values=direct[ka][:,row,col]/denominator
            check=ka in holdouts
            ax.plot(theta,values,color=colors[ka],lw=.9 if check else 1.05,alpha=.85 if args.holdouts_only else (.52 if check else .65),
                    marker='o' if check else None,markevery=100,ms=2.8,mfc='white',mew=.65,zorder=2)
        for ka in holdouts:
            denominator=direct[ka][:,0,0] if normalized else 1.
            ax.plot(theta,interpolated[ka][:,row,col]/denominator,color=colors[ka],lw=1.65,
                    ls=(0,(5,3)),zorder=3,path_effects=[effects.Stroke(linewidth=2.8,foreground='white'),effects.Normal()])
        ax.set_xlim(0,180);ax.set_xticks(np.arange(0,181,30));ax.tick_params(labelsize=8)

    def save(fig,name):
        for suffix in ('png','pdf'):fig.savefig(out/f'{name}.{suffix}',dpi=190)
        plt.close(fig)

    def mark_max_errors(fig,panels):
        if not args.mark_max_errors:return
        # Place labels in display coordinates: keep them inside axes, away from
        # other labels/markers and preferentially away from the plotted curves.
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for ax,row,col,normalized in panels:
            occupied=[t.get_window_extent(renderer).expanded(1.1,1.2) for t in ax.texts]
            samples=np.concatenate([ax.transData.transform(np.column_stack((line.get_xdata(),line.get_ydata())))[::8]
                                    for line in ax.lines if len(line.get_xdata())>100])
            positions=[]
            for ka in holdouts:
                rec=maxima[row,col,ka];index=rec['theta_index']
                y=interpolated[ka][index,row,col]/(direct[ka][index,0,0] if normalized else 1.)
                positions.append((ka,float(theta[index]),float(y)))
                ax.plot(theta[index],y,marker='o',ls='',ms=6.5 if len(panels)==1 else 5.5,
                        mfc=colors[ka],mec='white',mew=.75,zorder=6,clip_on=False)
            marker_pixels=ax.transData.transform(np.array([(x,y) for _,x,y in positions]))
            for ka,x,y in positions:
                value=maxima[row,col,ka]['maximum_error_percent_over_BEM_M11']
                label=f'{value:.3f}%' if value>=.001 else f'{value:.1e}%'
                fontsize=10 if len(panels)==1 else 8.5
                probe=ax.text(0,0,label,fontsize=fontsize)
                box=probe.get_window_extent(renderer);probe.remove()
                width,height=box.width+9,box.height+7
                px,py=ax.transData.transform((x,y));bounds=ax.get_window_extent(renderer)
                options=[]
                for dy in (-25,25,-45,45,-65,65,-90,90):
                    for dx in (-width/2-10,width/2+10,-width-25,width+25,0):
                        cx=np.clip(px+dx,bounds.x0+width/2+3,bounds.x1-width/2-3)
                        cy=np.clip(py+dy,bounds.y0+height/2+3,bounds.y1-height/2-3)
                        candidate=Bbox.from_bounds(cx-width/2,cy-height/2,width,height)
                        overlap=sum(candidate.overlaps(b) for b in occupied)
                        inside=lambda points: ((points[:,0]>=candidate.x0)&(points[:,0]<=candidate.x1)&
                                               (points[:,1]>=candidate.y0)&(points[:,1]<=candidate.y1))
                        score=100000*overlap+10000*np.count_nonzero(inside(marker_pixels))+4*np.count_nonzero(inside(samples))
                        score+=np.hypot(cx-px,cy-py)
                        options.append((score,cx,cy,candidate))
                _,cx,cy,box=min(options,key=lambda item:item[0]);occupied.append(box.expanded(1.1,1.2))
                ax.annotate(label,xy=(x,y),xycoords='data',
                            xytext=ax.transAxes.inverted().transform((cx,cy)),textcoords='axes fraction',
                            ha='center',va='center',fontsize=fontsize,color=colors[ka],zorder=8,
                            bbox=dict(boxstyle='round,pad=.17',fc='white',ec=colors[ka],lw=.45,alpha=.94),
                            arrowprops=dict(arrowstyle='-',color=colors[ka],lw=.65,shrinkA=2,shrinkB=3),
                            annotation_clip=False)

    fig,ax=plt.subplots(figsize=(13,8))
    fig.subplots_adjust(left=.085,right=.985,bottom=.13 if args.mark_max_errors else .11,top=.79 if args.holdouts_only else .76)
    curves(ax,0,0);ax.set_yscale('log')
    ax.set_xlabel('Угол рассеяния θ, °');ax.set_ylabel('M₁₁, мкм²/ср')
    decorate(fig,'M₁₁: интерполяция и прямой BEM — только 4 проверки' if args.holdouts_only else 'M₁₁ по углам: все 13 размеров и 4 интерполированные проверки')
    mark_max_errors(fig,[(ax,0,0,False)])
    save(fig,'M11_four_holdouts' if args.holdouts_only else 'M11_all_sizes')

    for normalized in (False,True):
        fig,axs=plt.subplots(4,4,figsize=(17,14),sharex=True)
        fig.subplots_adjust(left=.065,right=.985,bottom=.115 if args.mark_max_errors else .10,top=.84 if args.holdouts_only else .815,hspace=.38,wspace=.33)
        for row in range(4):
            for col in range(4):
                ax=axs[row,col];curves(ax,row,col,normalized)
                label=f'M{row+1}{col+1}'
                ax.set_title(label+(' / M11 (BEM)' if normalized else ''))
                ratio_max=max(np.max(np.abs(x[:,row,col])/direct[ka][:,0,0])
                              for source in (direct,interpolated) for ka,x in source.items())
                if ratio_max<1e-6:
                    if normalized:ax.set_ylim(-1e-6,1e-6)
                    else:
                        floor=max(np.max(x[:,0,0]) for x in direct.values())*1e-6
                        ax.set_ylim(-floor,floor)
                    ax.text(.5,.83,f'≈ 0; max |Mij|/M11 = {ratio_max:.1e}',transform=ax.transAxes,
                            ha='center',fontsize=8,color='#555555')
                    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
                elif not normalized:
                    scale=max(np.max(np.abs(x[:,row,col])) for source in (direct,interpolated) for x in source.values())
                    minimum=min(np.min(x[:,row,col]) for source in (direct,interpolated) for x in source.values())
                    if minimum>0:ax.set_yscale('log')
                    else:
                        ax.set_yscale('symlog',linthresh=scale*1e-4,linscale=.7)
                        locator=SymmetricalLogLocator(linthresh=scale*1e-4,base=10)
                        locator.set_params(numticks=7)
                        ticks=locator.tick_values(*ax.get_ylim())
                        ax.set_yticks([v for v in ticks if v==0 or abs(v)>=scale*1e-3])
                if row==3:ax.set_xlabel('θ, °')
                if col==0:ax.set_ylabel('Безразмерное отношение' if normalized else 'мкм²/ср')
                if row!=col or row>1:ax.axhline(0,color='#777777',lw=.45,zorder=1)
        title='Все 16 элементов: общая нормировка на M₁₁ прямого BEM' if normalized else 'Все 16 элементов Мюллера: абсолютные значения'
        decorate(fig,title,True)
        fig.text(.5,.047,'Для интерполяции и BEM одного размера знаменатель одинаков.' if normalized else
                 'M11 и M22: логарифмическая шкала; остальные ненулевые элементы: знаковая логарифмическая шкала.',
                 ha='center',fontsize=9,color='#444444')
        mark_max_errors(fig,[(axs[row,col],row,col,normalized) for row in range(4) for col in range(4)])
        save(fig,'all16_normalized' if normalized else 'all16_absolute')

    curves_data=[(ka,'BEM',direct[ka]) for ka in sizes]+[(ka,'interpolated',interpolated[ka]) for ka in holdouts]
    np.savez_compressed(out/'curves.npz',theta_deg=theta,ka=[x[0] for x in curves_data],
                        kind=[x[1] for x in curves_data],mueller_um2_per_sr=np.array([x[2] for x in curves_data]))
    if args.mark_max_errors:
        with (out/'maximum_errors.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(maximum_records[0]));writer.writeheader();writer.writerows(maximum_records)
        r.atomic_json(out/'maximum_errors.json',{'metric':'100*abs(interpolated Mij - direct BEM Mij)/direct BEM M11',
            'search':'maximum over 1801 sampled angles, step 0.1 degrees; not a continuous maximum',
            'marker_location':'on the interpolated curve at the maximizing angle',
            'records':maximum_records})
    r.atomic_json(out/'plot_manifest.json',{'training':training,'independent_holdouts':holdouts,
        'direct_curves':len(direct),'interpolated_curves':len(interpolated),'theta_count':len(theta),
        'orientation_average':'analytic full SO(3)','Euler_quadrature_nodes':0,
        'absolute_scale':'dimensionless Mueller times (wavelength_um/(2*pi))^2; no fitted scaling',
        'normalized_scale':'Both curves at a given ka divide by the DIRECT BEM M11 at that ka and angle.',
        'source_T_sha256':identities,'model_sha256':hashlib.sha256((run/'model_1/model.json').read_bytes()).hexdigest(),
        'mesh_control_complete':bool(r.read(run/'model_1/controls.json')),
        'source_curves':str(args.curves_from.resolve()) if args.curves_from else None,
        'maximum_error_markers':args.mark_max_errors,
        'maximum_error_metric':'100*abs(interpolated Mij - direct BEM Mij)/direct BEM M11' if args.mark_max_errors else None,
        'scope':'Only four independent holdouts, direct BEM and interpolated T.' if args.holdouts_only else '13 direct sizes and four interpolation holdouts in ka=10..12.'})
    print('Plots saved to '+str(out),flush=True)


if __name__=='__main__':main()
