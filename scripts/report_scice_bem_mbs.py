#!/usr/bin/env python3
"""Append completed, matched-geometry BEM data locally and rebuild the ScIce PDF.

The published ScIce input is immutable. BEM has its own method label in the
derived table. MBS errors use M11(MBS) as requested for this report.
"""
import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BLUE,ORANGE,GREEN='#1768AC','#D76A24','#24835b'
ENTRIES=[(0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)]
LAMBDA=.532
FACTOR=(LAMBDA/(2*np.pi))**2

def read(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def save_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)

def absolute_integral(theta,values):
    """Integrate |piecewise-linear(values)| sin(theta), including sign crossings."""
    t=np.radians(theta);y=np.asarray(values)
    cross=np.flatnonzero(y[:-1]*y[1:]<0)
    z=t[cross]+np.diff(t)[cross]*abs(y[cross])/(abs(y[cross])+abs(y[cross+1]))
    t=np.r_[t,z];y=np.r_[y,np.zeros(len(z))];order=np.argsort(t);t=t[order];y=y[order]
    nodes,w=np.polynomial.legendre.leggauss(4);u=(nodes+1)/2;h=np.diff(t)
    return float(np.sum(h[:,None]/2*w*np.sin(t[:-1,None]+h[:,None]*u)*
                        abs(y[:-1,None]+np.diff(y)[:,None]*u)))

def load_case(case,comparison,extension,checked):
    ka=int(case['ka']);directory=Path(case['bem'])
    summary=read(directory/'summary.json');average=read(directory/'average.json');p=summary['parameters']
    assert summary['complete'] and average['Euler_quadrature_nodes']==0
    assert p['shape']=='prism' and p['ka']==ka and p['ri']==1.3116 and p['ri_imag']==0
    np.testing.assert_allclose([p['wavelength'],p['prism_aspect']],[LAMBDA,1/(.7*np.cos(np.pi/6))],rtol=1e-12)
    receipt=read(comparison/f'ka{ka}_comparison.json')
    assert receipt['mbs_converged'] and receipt['mbs_status']['coherent']
    assert receipt['mbs_independent_validation']['passed'] and receipt['mbs_fullauto_tolerance']==.01
    assert receipt['theta_range_deg']==[175,180] and not receipt['BEM_angular_interpolation']
    for source,digest in receipt['sources'].items():
        if source not in checked:
            checked[source]=sha(Path(source))
        if checked[source]!=digest:raise ValueError('Comparison source changed: '+source)
    with np.load(comparison/f'ka{ka}_comparison.npz') as values:
        theta=values['theta_deg'].copy();bem=values['bem_mueller_um2_sr'].copy();mbs=values['mbs_mueller_um2_sr'].copy()
    drawing_theta=np.array(average['theta']);drawing=np.moveaxis(np.array(average['mueller']),-1,0)*FACTOR
    assert drawing.shape==(len(drawing_theta),4,4) and drawing_theta[-1]==theta[-1]==180
    assert np.isfinite(bem).all() and np.isfinite(mbs).all() and np.isfinite(drawing).all()
    assert np.all(bem[:,0,0]>0) and np.all(mbs[:,0,0]>0)
    np.testing.assert_allclose(bem[-1],drawing[-1],rtol=1e-8,atol=1e-10)
    a=ka*LAMBDA/(2*np.pi);volume=4*np.pi*a**3/3
    height=(volume/(3*np.sqrt(3)/8*.7**2))**(1/3);diameter=.7*height
    mean_area=(3*np.sqrt(3)/4*diameter**2+3*diameter*height)/4
    matrix=drawing[-1];ldr=(matrix[0,0]-matrix[1,1])/(matrix[0,0]+matrix[1,1])
    # Native cross sections use volume-equivalent radius a=1. Convert to um².
    cext=summary['extinction_cross_section']*a*a
    t=np.load(directory/'tmatrix.npy',mmap_mode='r')
    np.testing.assert_allclose(cext,-2*np.pi*FACTOR*np.trace(t).real,rtol=1e-12)
    del t
    row=dict(xeq=float(ka),L=float(height),D=float(diameter),Dmax=float(np.hypot(height,diameter)),
             V=float(volume),LR=float(cext/matrix[0,0]),DR=float(ldr),**{'S/4':float(mean_area)},
             Cext=float(cext),M11=float(matrix[0,0]),M22=float(matrix[1,1]),
             S11=float(matrix[0,0]/FACTOR),S22=float(matrix[1,1]/FACTOR),Qext=float(cext/mean_area),
             method='BEM_T_analytic_SO3',ref=p['ref'],T_order=p['order'],source=str(directory),interpolated=False)
    np.savez_compressed(extension/f'BEM_ka{ka}_Mueller.npz',ka=ka,theta_deg=drawing_theta,
                        mueller_um2_per_sr=drawing,mueller_dimensionless=drawing/FACTOR,
                        ref=p['ref'],T_order=p['order'],wavelength_um=LAMBDA,ri=1.3116,
                        D_vertices_over_H=.7,interpolated=False,orientation='uniform SO(3), analytical')
    error=100*(bem-mbs)/mbs[:,0,0,None,None]
    bn=bem/bem[:,0,0,None,None];mn=mbs/mbs[:,0,0,None,None]
    metrics=[]
    for i in range(4):
        for j in range(4):
            index=int(np.argmax(abs(error[:,i,j])))
            zero=max(np.max(abs(bn[:,i,j])),np.max(abs(mn[:,i,j])))<1e-6
            numerator=absolute_integral(theta,bem[:,i,j]-mbs[:,i,j])
            denominator=absolute_integral(theta,mbs[:,i,j])
            normalized_denominator=absolute_integral(theta,mn[:,i,j])
            metrics.append(dict(ka=ka,element=f'M{i+1}{j+1}',symmetry_zero=bool(zero),
                maximum_error_over_MBS_M11_percent=float(abs(error[index,i,j])),theta_at_max_deg=float(theta[index]),
                raw_integral_error_over_MBS_element_percent=None if zero else 100*numerator/denominator,
                normalized_integral_error_over_MBS_element_percent=None if zero or (i,j)==(0,0) else
                    100*absolute_integral(theta,bn[:,i,j]-mn[:,i,j])/normalized_denominator,
                integral_residual_over_MBS_M11_percent=100*numerator/absolute_integral(theta,mbs[:,0,0])))
    return dict(ka=ka,row=row,theta=theta,bem=bem,mbs=mbs,bn=bn,mn=mn,error=error,metrics=metrics,
                drawing_theta=drawing_theta,drawing=drawing,summary=summary)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,default=ROOT/'exports/scice_bem_mbs_cone5_1pct_20261109')
    parser.add_argument('--out',type=Path,default=ROOT/'benchmarks/scice_2025_comparison_20261103')
    parser.add_argument('--extension',type=Path,default=ROOT/'data/external/scice_2025/BEM_extension_20261109')
    parser.add_argument('--scice',type=Path,default=ROOT/'data/external/scice_2025/extracted/ScIce-2025 Database/Column/Column_complete_lmd0.532_n1.3116_k0.txt')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);args.extension.mkdir(parents=True,exist_ok=True)
    original_digest=sha(args.scice)
    with args.scice.open() as stream:published=list(csv.DictReader(stream))
    table=np.genfromtxt(args.scice,names=True,delimiter=',');size=2*np.pi/LAMBDA*np.cbrt(3*table['V']/(4*np.pi))
    order=np.argsort(size,kind='stable');size=size[order];table=table[order]
    np.testing.assert_allclose(table['V'],3*np.sqrt(3)/8*table['D']**2*table['L'],rtol=1e-12)
    np.testing.assert_allclose(table['M11'],table['S11']*FACTOR,rtol=1e-6)
    np.testing.assert_allclose(table['DR'],(table['M11']-table['M22'])/(table['M11']+table['M22']),rtol=1e-12)
    np.testing.assert_allclose(table['D'][size<=41]/table['L'][size<=41],.7,rtol=1e-10)
    checked={str(args.scice):original_digest}
    cases=[load_case(case,args.comparison,args.extension,checked) for case in read(args.comparison/'campaign.json')['cases']]
    assert [d['ka'] for d in cases]==[20,25,30,35,40]
    print('Validated five completed BEM solutions and saved MBS comparisons.',flush=True)
    backup=args.out/'archive_before_matched_geometry_20261109'
    if not backup.exists():
        backup.mkdir()
        for path in args.out.iterdir():
            if path.is_file():shutil.copy2(path,backup/path.name)
    bem_rows=[d['row'] for d in cases]
    combined=[dict(row,method='ScIce_2025_published',ref='',T_order='',source=str(args.scice),interpolated=False) for row in published]+bem_rows
    combined.sort(key=lambda r:float(r['xeq']))
    save_csv(args.extension/'Column_BEM_lmd0.532_n1.3116_k0.csv',bem_rows)
    save_csv(args.extension/'Column_ScIce_plus_BEM_lmd0.532_n1.3116_k0.csv',combined)
    save_csv(args.out/'bem_backscatter.csv',bem_rows)
    comparisons=[]
    for d in cases:
        ka=d['ka'];b=d['bem'][-1];m=d['mbs'][-1]
        assert size[0]<=ka<=size[-1]
        s11=float(np.interp(ka,size,table['M11']));s22=float(np.interp(ka,size,table['M22']))
        bd=(b[0,0]-b[1,1])/(b[0,0]+b[1,1]);md=(m[0,0]-m[1,1])/(m[0,0]+m[1,1]);sd=(s11-s22)/(s11+s22)
        comparisons.append(dict(ka=ka,BEM_M11_um2_sr=float(b[0,0]),MBS_M11_um2_sr=float(m[0,0]),
            ScIce_M11_interpolated_um2_sr=s11,BEM_LDR_percent=100*float(bd),MBS_LDR_percent=100*float(md),
            ScIce_LDR_interpolated_percent=100*sd,M11_BEM_minus_MBS_over_MBS_percent=100*float(b[0,0]/m[0,0]-1),
            M11_BEM_minus_ScIce_over_ScIce_percent=100*float(b[0,0]/s11-1),
            LDR_BEM_minus_MBS_percentage_points=100*float(bd-md),
            LDR_BEM_minus_ScIce_percentage_points=100*float(bd-sd),geometry_matches=True))
    save_csv(args.out/'comparison_at_BEM_sizes.csv',comparisons)
    save_csv(args.out/'BEM_MBS_cone_errors.csv',[row for d in cases for row in d['metrics']])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    physics='m = 1,3116; λ = 0,532 мкм; D по вершинам/H = 0,7; случайная ориентация'
    pdf=args.out/'bem_vs_scice_backscatter.pdf';temporary=pdf.with_suffix('.tmp.pdf')
    def finish(book,fig,name=None):
        book.savefig(fig)
        if name:fig.savefig(args.out/(name+'.png'),dpi=150)
        plt.close(fig)
    with PdfPages(temporary) as book:
        fig,axes=plt.subplots(1,2,figsize=(13.8,7.6));fig.subplots_adjust(top=.77,bottom=.19,wspace=.25,left=.08,right=.98)
        fig.suptitle('SciIce + BEM + MBS-fast: обратное рассеяние столбиков',fontsize=17,fontweight='bold',y=.96)
        fig.text(.5,.91,physics,ha='center',fontsize=11)
        mask=(size>=15)&(size<=41);kas=[d['ka'] for d in cases]
        for ax,key,ylabel in zip(axes,('M11','LDR'),('M₁₁(180°), мкм²/ср','LDR(180°), %')):
            sy=table['M11'] if key=='M11' else 100*table['DR']
            ax.plot(size[mask],sy[mask],color=GREEN,lw=1.6,label='SciIce-2025: опубликованные узлы')
            for prefix,color,marker,label in [('BEM',BLUE,'o','BEM: T → аналитическое SO(3)'),('MBS',ORANGE,'s','MBS-fast: coherent fullauto 1%')]:
                column=prefix+('_M11_um2_sr' if key=='M11' else '_LDR_percent')
                ax.plot(kas,[r[column] for r in comparisons],color=color,marker=marker,lw=1.2,ls='--',ms=6,label=label)
            ax.set_xlim(15,41);ax.set_xticks([15,20,25,30,35,40]);ax.set_xlabel('ka по объёму');ax.set_ylabel(ylabel);ax.grid(alpha=.18)
            if key=='M11':ax.set_yscale('log')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.865),ncol=3,fontsize=9,frameon=False)
        fig.text(.5,.09,'Пять BEM-решений рассчитаны напрямую. Масштаб не подгонялся; геометрия совпадает.\n'
                 'Штриховые отрезки соединяют узлы; они не являются проверенной интерполяцией.',ha='center',fontsize=10)
        fig.text(.5,.035,'SciIce: Konoshonkin et al. (2025), doi:10.5281/zenodo.15144868. Опубликованная таблица сохранена.',ha='center',fontsize=9,color='#485563')
        svg=args.out/'bem_vs_scice_backscatter.svg';fig.savefig(svg)
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        finish(book,fig,'bem_vs_scice_backscatter')

        fig,axes=plt.subplots(1,2,figsize=(13.8,7.6));fig.subplots_adjust(top=.79,bottom=.43,wspace=.28,left=.08,right=.98)
        fig.suptitle('Расхождения при 180° на совпадающих размерах',fontsize=17,y=.96)
        fig.text(.5,.91,'SciIce: линейная интерполяция M11 и M22 к ka BEM; BEM и MBS: готовые значения в узлах.',ha='center',fontsize=10)
        for prefix,color in [('MBS',ORANGE),('ScIce',GREEN)]:
            axes[0].plot(kas,[r[f'M11_BEM_minus_{prefix}_over_{prefix}_percent'] for r in comparisons],marker='o',color=color,label='относительно '+prefix)
            axes[1].plot(kas,[r[f'LDR_BEM_minus_{prefix}_percentage_points'] for r in comparisons],marker='o',color=color,label='BEM − '+prefix)
        axes[0].set_ylabel('100 (M11 BEM − M11 X) / M11 X, %');axes[1].set_ylabel('LDR BEM − LDR X, процентные пункты')
        for ax in axes:ax.set_xlabel('ka');ax.set_xticks(kas);ax.grid(alpha=.18);ax.axhline(0,color='#555',lw=.7);ax.legend(fontsize=9)
        table_ax=fig.add_axes([.07,.06,.9,.27]);table_ax.axis('off')
        rows=[[r['ka'],f"{r['BEM_M11_um2_sr']:.5f}",f"{r['MBS_M11_um2_sr']:.5f}",f"{r['ScIce_M11_interpolated_um2_sr']:.5f}",
               f"{r['M11_BEM_minus_MBS_over_MBS_percent']:+.2f}%",f"{r['M11_BEM_minus_ScIce_over_ScIce_percent']:+.2f}%"] for r in comparisons]
        grid=table_ax.table(cellText=rows,colLabels=['ka','M11 BEM','M11 MBS','M11 SciIce*','Δ/MBS','Δ/SciIce*'],loc='center',cellLoc='center')
        grid.auto_set_font_size(False);grid.set_fontsize(10);grid.scale(1,1.6)
        finish(book,fig,'backscatter_errors')

        fig,axes=plt.subplots(2,5,figsize=(17.5,7.6));fig.subplots_adjust(left=.055,right=.99,bottom=.12,top=.80,wspace=.35,hspace=.45)
        fig.suptitle('BEM и MBS-fast: конус 175–180°, допуск fullauto 1%',fontsize=17,y=.96)
        fig.text(.5,.91,physics,ha='center',fontsize=10)
        fig.text(.5,.86,'Ошибка: 100 (M11 BEM − M11 MBS) / M11 MBS. Допуск fullauto относится к сходимости MBS.',ha='center',fontsize=10)
        for col,d in enumerate(cases):
            ax,er=axes[:,col];ax.plot(d['theta'],d['bem'][:,0,0],color=BLUE,label='BEM');ax.plot(d['theta'],d['mbs'][:,0,0],color=ORANGE,ls='--',label='MBS')
            ax.set_title(f"ka={d['ka']}; ref={d['row']['ref']}, L={d['row']['T_order']}");ax.set_ylabel('M11, мкм²/ср');ax.legend(fontsize=8)
            error_plot(er,d['theta'],d['error'][:,0,0]);er.set_ylabel('% от M11 MBS')
            for item in (ax,er):angle_axes(item)
        finish(book,fig,'BEM_MBS_cone_overview')
        for d in cases:
            fig,axes=plt.subplots(4,4,figsize=(16,12));fig.subplots_adjust(top=.88,bottom=.07,left=.06,right=.98,wspace=.43,hspace=.65)
            fig.suptitle(f"ka={d['ka']}: ненулевые элементы и ошибки относительно MBS",fontsize=17,y=.975)
            fig.text(.5,.942,'Кривые: M11 размерный; остальные Mij / собственный M11 каждого метода.',ha='center',fontsize=11)
            fig.text(.5,.915,'Ошибка исходных элементов: 100 (Mij BEM − Mij MBS) / M11 MBS.',ha='center',fontsize=11)
            for n,(i,j) in enumerate(ENTRIES):
                row,col=n//2,2*(n%2);ax,er=axes[row,col:col+2]
                b=d['bem'][:,i,j] if n==0 else d['bn'][:,i,j];m=d['mbs'][:,i,j] if n==0 else d['mn'][:,i,j]
                ax.plot(d['theta'],b,color=BLUE,label='BEM');ax.plot(d['theta'],m,color=ORANGE,ls='--',label='MBS');ax.legend(fontsize=8)
                ax.set_title(f'M{i+1}{j+1}'+('' if n==0 else '/собственный M11'))
                error_plot(er,d['theta'],d['error'][:,i,j]);er.set_ylabel('% от M11 MBS')
                for item in (ax,er):angle_axes(item)
            finish(book,fig,f"ka{d['ka']}_BEM_MBS")
        for key,title in [('raw_integral_error_over_MBS_element_percent','Исходные элементы'),
                          ('normalized_integral_error_over_MBS_element_percent','Элементы / собственный M11')]:
            fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off')
            fig.suptitle('Интегральные ошибки относительно MBS — '+title,fontsize=14)
            fig.text(.5,.90,'100 ∫|Q(BEM) − Q(MBS)| sinθ dθ / ∫|Q(MBS)| sinθ dθ; θ = 175…180°',ha='center',fontsize=11)
            cells=[]
            for index in range(16):
                cells.append([f'M{index//4+1}{index%4+1}']+['—' if d['metrics'][index][key] is None else f"{d['metrics'][index][key]:.3f}" for d in cases])
            grid=ax.table(cellText=cells,colLabels=['Элемент']+[f"ka={d['ka']}" for d in cases],loc='center',cellLoc='center')
            grid.auto_set_font_size(False);grid.set_fontsize(10);grid.scale(1,1.5)
            fig.text(.12,.07,'«—»: нулевой по симметрии элемент либо тождественное M11/M11.\nОшибка здесь — расхождение методов; 1% fullauto не означает 1% согласия с BEM.',fontsize=10)
            finish(book,fig)
    temporary.replace(pdf)
    assert sha(args.scice)==original_digest
    report=dict(generated_utc=datetime.now(timezone.utc).isoformat(),BEM_nodes=5,ScIce_rows=len(published),
        derived_database_rows=len(combined),BEM_sizes=[d['ka'] for d in cases],geometry_matches=True,
        common_physics=dict(n=1.3116,n_imag=0.,wavelength_um=LAMBDA,D_vertices_over_H=.7,orientation='uniform SO(3)'),
        BEM_averaging='analytic SO(3); no orientation quadrature',BEM_size_interpolation=False,
        ScIce_interpolation='Linear M11 and M22 only for the comparison at BEM ka; LDR formed afterwards.',
        MBS_configuration='coherent fullauto 1%, independent final check passed, cone 175–180 degrees',
        angular_error='100*(BEM_ij-MBS_ij)/MBS_11',integral_error='100*integral(abs(BEM_ij-MBS_ij)*sin(theta))/integral(abs(MBS_ij)*sin(theta))',
        cross_section_units='Native Cext with a_eq=1 multiplied by a_eq_um**2; verified against -2*pi/k0**2*Re(trace(T)).',
        source_file=str(args.scice),source_sha256=original_digest,source_table_modified=False,
        source_url='https://zenodo.org/records/15144868',checked_sources=checked,
        database_extension=str(args.extension),pdf=str(pdf),comparison=comparisons,
        saved_matrices='Only Mueller matrices; no T matrices copied.',new_BEM_solves=0,ref7_included=False)
    save_json(args.out/'summary.json',report);save_json(args.extension/'provenance.json',report)
    (args.extension/'README.txt').write_text('Local BEM supplement to the published ScIce-2025 column table.\n'
        'Column_BEM: five direct matched-geometry calculations, ka=20,25,30,35,40.\n'
        'Column_ScIce_plus_BEM: published rows plus BEM rows, distinguished by method.\n'
        'Original published table is unchanged. NPZ files contain full-angle Mueller matrices, not T matrices.\n'
        'M11/M22 and Cext: um^2/sr and um^2, respectively; S11/S22 are dimensionless.\n'
        'DR=(M11-M22)/(M11+M22); L is particle height, T_order is the modal cutoff.\n'
        'See provenance.json for geometry, hashes, tolerances and source directories.\n')
    print(json.dumps(dict(pdf=str(pdf),extension=str(args.extension),BEM_nodes=5,rows=len(combined)),ensure_ascii=False),flush=True)

def angle_axes(ax):
    ax.set_xlim(175,180);ax.set_xticks([175,176,177,178,179,180]);ax.set_xlabel('θ, °');ax.grid(alpha=.16)

def error_plot(ax,theta,error):
    index=int(np.argmax(abs(error)));ax.plot(theta,error,color=ORANGE)
    ax.scatter(theta[index],error[index],color='#B22222',s=23,zorder=5)
    ax.set_title(f'|max| = {abs(error[index]):.2f}% при {theta[index]:.2f}°',fontsize=9)
    limit=max(.01,float(np.max(abs(error)))*1.15);ax.set_ylim(-limit,limit);ax.axhline(0,color='#555',lw=.6)

if __name__=='__main__':main()
