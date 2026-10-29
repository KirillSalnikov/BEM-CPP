#!/usr/bin/env python3
"""Report mesh-checked BEM/PO agreement for axial incidence as ka increases."""
import argparse
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rows=[];fields=[]
    for ka in [20,40,60]:
        selected=args.run/('ka20_T_reference' if ka==20 else f'ka{ka}_ref6')
        bem=json.loads((selected/'result.json').read_text())
        mbsdir=args.run/f'mbs_ka{ka}_depth24'
        mbs=json.loads((mbsdir/'summary.json').read_text())
        row={'ka':ka,'BEM_ref':bem['ref'],'BEM_M11_forward':bem['M11_forward'],'BEM_Cext':bem['Cext'],
             'MBS_M11_forward':mbs['M11_forward'],'MBS_Cext':mbs['Cext'],
             'M11_signed_relative_difference':mbs['M11_forward']/bem['M11_forward']-1,
             'Cext_signed_relative_difference':mbs['Cext']/bem['Cext']-1,
             'BEM_result':str(selected/'result.json'),'MBS_result':str(mbsdir/'summary.json')}
        if ka>20:
            coarse=json.loads((args.run/f'ka{ka}_ref5/result.json').read_text())
            row['mesh_ref5_vs_ref6_M11_relative']=coarse['M11_forward']/bem['M11_forward']-1
            row['mesh_ref5_vs_ref6_Cext_relative']=coarse['Cext']/bem['Cext']-1
            row['BEM_solve']=bem['solve']
        else:
            row['validation']='Previously checked against independent ADDA at dpl12/dpl18; maximum difference of forward M11 0.0373%.'
        adda=(args.run.parent.parent/'benchmarks/d6h_20261026/forward_diagnosis/adda_axial_dpl18'
              if ka==20 else args.run/f'adda_ka{ka}_dpl12')
        if (adda/'complete.json').exists():
            aa=np.loadtxt(adda/'mueller',skiprows=1)
            row['ADDA_M11_forward']=float(aa[0,1])
            row['ADDA_Cext']=float(np.mean([float(re.search(r'Cext\s*=\s*([^\n]+)',(adda/f'CrossSec-{pol}').read_text())[1]) for pol in ['X','Y']]))
            row['ADDA_vs_BEM_M11_relative']=row['ADDA_M11_forward']/bem['M11_forward']-1
            row['ADDA_vs_BEM_Cext_relative']=row['ADDA_Cext']/bem['Cext']-1
        archived=args.run/f'archived_adda_ka{ka}_dpl15.json'
        if archived.exists():
            row['archived_ADDA']=json.loads(archived.read_text())
            row['archived_ADDA']['M11_relative_to_BEM']=row['archived_ADDA']['M11_forward']/bem['M11_forward']-1
            row['archived_ADDA']['Cext_relative_to_BEM']=row['archived_ADDA']['Cext']/bem['Cext']-1
        a=np.load(selected/'fields.npz');table=np.loadtxt(mbsdir/'result/result.dat',skiprows=1)
        assert table.shape[1]==19 and np.allclose(a['theta_deg'],table[:,0])
        fields.append((a['theta_deg'],a['mueller'][:,0,0],ka**2*table[:,3]))
        rows.append(row)
    report={'geometry':'regular hexagonal prism H/D_flats=1, equivalent-volume radius=1',
            'refractive_index':[1.3,0.],'incident_direction':[0,0,1],
            'orientation_average':False,'normalization':'M11 = k^2 d_sigma/d_Omega, common scale, no fit',
            'MBS_reflection_depth':24,'rows':rows}
    (args.out/'size_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.22})
    fig,axs=plt.subplots(1,3,figsize=(14,5.4))
    for ax,row,(theta,bem,mbs) in zip(axs,rows,fields):
        ax.semilogy(theta,bem,color='#1765a3',label='BEM',lw=1.8)
        ax.semilogy(theta,mbs,color='#d75c22',ls='--',label='MBS, глубина 24',lw=1.6)
        if 'ADDA_M11_forward' in row:
            ax.plot(0,row['ADDA_M11_forward'],'*',ms=10,color='#18844c',label='ADDA, вперёд',clip_on=False)
        if 'archived_ADDA' in row:
            ax.plot(0,row['archived_ADDA']['M11_forward'],'^',ms=7,mfc='none',mec='#6f3e96',label='ADDA dpl15, архив',clip_on=False)
        ax.set(xlim=(0,20),xlabel='Угол рассеяния θ, °',ylabel='M₁₁ = k² dσ/dΩ',
               title=f'ka = {row["ka"]}; вперёд: {100*row["M11_signed_relative_difference"]:+.2f}%')
        visible=np.concatenate((bem[theta<=20],mbs[theta<=20]))
        positive=visible[visible>0]
        ax.set_ylim(float(positive.min())*.5,float(positive.max())*1.8)
        ax.legend(fontsize=9)
    fig.suptitle('Одна ориентация: падение вдоль оси призмы\nH/D по граням = 1, m = 1.3; без подгонки нормировки',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.89))
    for ext in ['png','pdf']:fig.savefig(args.out/('forward_lobes_vs_size.'+ext),dpi=175,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,quantity,label in zip(axs,['M11','Cext'],['M₁₁ вперёд','Экстинкция']):
        errors=np.array([r[quantity+'_signed_relative_difference'] for r in rows])*100
        ax.plot([r['ka'] for r in rows],errors,'o-',color='#d75c22',label='MBS относительно BEM')
        for row,e in zip(rows,errors):ax.annotate(f'{e:+.2f}%',(row['ka'],e),xytext=(0,8),textcoords='offset points',ha='center')
        ax.axhline(0,color='#555555',lw=.8)
        ax.set(xlabel='ka',ylabel='(MBS / BEM − 1), %',title=label,xticks=[20,40,60],xlim=(15,65))
        ax.margins(y=.2)
    fig.suptitle('Зависимость расхождения от размера · одна осевая ориентация',fontsize=14)
    for ext in ['png','pdf']:fig.savefig(args.out/('forward_error_vs_size.'+ext),dpi=175,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
