"""Matched-geometry, native-angle BEM/MBS comparison for the five SciIce sizes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
CONFIG=json.loads((OUT/'campaign.json').read_text())
LOW,HIGH=CONFIG.get('theta_range_deg',[0,180])
TOL=CONFIG['fullauto_tolerance']
sys.path.insert(0,str(ROOT/'scripts'))
from tmatrix_so3_mstm import analytic_mueller_tmatrix
ENTRIES=[(0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)]
BLUE,ORANGE='#1768AC','#D76A24'

def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
    temporary=p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');temporary.replace(p)

def integral(theta,y):
    t=np.radians(theta);y=np.asarray(y);cross=np.flatnonzero(y[:-1]*y[1:]<0)
    z=t[cross]+np.diff(t)[cross]*abs(y[cross])/(abs(y[cross])+abs(y[cross+1]))
    t=np.r_[t,z];y=np.r_[y,np.zeros(len(z))];s=np.argsort(t);t=t[s];y=y[s]
    nodes,w=np.polynomial.legendre.leggauss(4);u=(nodes+1)/2;h=np.diff(t)
    return float(np.sum(h[:,None]/2*w*np.sin(t[:-1,None]+h[:,None]*u)*
                        abs(y[:-1,None]+np.diff(y)[:,None]*u)))

def load(case):
    ka=case['ka'];bem_dir=Path(case['bem']);mbs_dir=Path(case['mbs'])
    summary=read(bem_dir/'summary.json');average=read(bem_dir/'average.json');p=summary['parameters']
    assert summary['complete'] and average['Euler_quadrature_nodes']==0
    assert p['ka']==ka and p['ri']==1.3116 and p['ri_imag']==0 and p['wavelength']==.532
    np.testing.assert_allclose(p['prism_aspect'],1/(.7*np.cos(np.pi/6)),rtol=1e-12)
    factor=(.532/(2*np.pi))**2
    drawing_theta=np.asarray(average['theta']);mask=(drawing_theta>=LOW)&(drawing_theta<=HIGH)
    result=dict(ka=ka,ref=p['ref'],order=p['order'],ready=False,mbs_state='running',
        drawing_theta=drawing_theta[mask],drawing_bem=(np.moveaxis(average['mueller'],-1,0)*factor)[mask],
        maximum_strict_residual=summary['maximum_strict_residual'])
    status_path=mbs_dir/'result/fullauto_status.json';finish=mbs_dir/'finished.json'
    if not status_path.exists() or not finish.exists():return result
    status=read(status_path);done=read(finish)
    if status['status']!='converged' or done['returncode']!=0:
        result['mbs_state']='failed: '+status['status'];return result
    audit=read(mbs_dir/'result/independent_final_validation.json')
    assert status['coherent'] and audit['passed'] and status['independent_final_validation']
    launch=read(mbs_dir/'launch.json');cmd=launch['command']
    assert float(cmd[cmd.index('--fullauto')+1])==TOL
    start=cmd.index('--fullauto-theta-range')+1
    assert list(map(float,cmd[start:start+2]))==[LOW,HIGH]
    vertices=np.unique(np.array([[float(x) for x in line.split()] for line in
        (mbs_dir/'particle_scaled.txt').read_text().splitlines() if len(line.split())==3]),axis=0)
    h=np.ptp(vertices[:,2]);diam=np.max(np.linalg.norm(vertices[:,None,:2]-vertices[None,:,:2],axis=-1))
    radius=(3*ConvexHull(vertices).volume/(4*np.pi))**(1/3)
    np.testing.assert_allclose([diam/h,2*np.pi/.532*radius],[.7,ka],rtol=2e-11)
    table=mbs_dir/'result/mueller_fullauto_annotated.csv'
    with table.open() as f:rows=list(csv.DictReader(f))
    theta=np.array([float(r['theta_deg']) for r in rows]);sign=np.array([1,1,-1,-1])
    mbs=np.array([[float(r[f'M{i}{j}']) for i in range(1,5) for j in range(1,5)] for r in rows]).reshape(-1,4,4)
    mbs*=sign[None,:,None]*sign[None,None,:]
    assert theta[0]==LOW and theta[-1]==HIGH and np.all(np.diff(theta)>0)
    cache=OUT/f'ka{ka}_bem_native_angles.npz';receipt=OUT/f'ka{ka}_bem_native_angles.json'
    sources={str(f):sha(f) for f in [bem_dir/'tmatrix.npy',bem_dir/'average.json',ROOT/'scripts/tmatrix_so3_mstm.py']}
    if cache.exists() and receipt.exists() and read(receipt)['sources']==sources:
        with np.load(cache) as z:
            np.testing.assert_array_equal(z['theta'],theta);bem=z['bem'].copy()
    else:
        bem,meta=analytic_mueller_tmatrix(np.load(bem_dir/'tmatrix.npy',mmap_mode='r'),p['order'],theta,
                                        precision='auto',threads=4,cyclic_order=average['cyclic_order'])
        bem*=factor;np.savez_compressed(cache,theta=theta,bem=bem)
        save(receipt,dict(sources=sources,contraction=meta))
    assert np.isfinite(bem).all() and np.isfinite(mbs).all() and np.all(bem[:,0,0]>0) and np.all(mbs[:,0,0]>0)
    error=100*(mbs-bem)/bem[:,0,0,None,None];bn=bem/bem[:,0,0,None,None];mn=mbs/mbs[:,0,0,None,None]
    measures=[]
    for i in range(4):
        for j in range(4):
            zero=np.max(abs(bn[:,i,j]))<1e-6;k=int(np.argmax(abs(error[:,i,j])))
            raw=integral(theta,mbs[:,i,j]-bem[:,i,j]);base=integral(theta,bem[:,i,j])
            ratio=integral(theta,mn[:,i,j]-bn[:,i,j]);ratio_base=integral(theta,bn[:,i,j])
            measures.append(dict(ka=ka,element=f'M{i+1}{j+1}',symmetry_zero=bool(zero),
                max_error_over_BEM_M11_percent=float(abs(error[k,i,j])),theta_at_max_deg=float(theta[k]),
                raw_integral_error_percent=None if zero else 100*raw/base,
                normalized_integral_error_percent=None if zero or (i,j)==(0,0) else 100*ratio/ratio_base,
                integral_residual_over_BEM_M11_percent=100*raw/integral(theta,bem[:,0,0])))
    np.savez_compressed(OUT/f'ka{ka}_comparison.npz',theta_deg=theta,bem_mueller_um2_sr=bem,
                       mbs_mueller_um2_sr=mbs,error_over_BEM_M11_percent=error)
    names=[f'M{i}{j}' for i in range(1,5) for j in range(1,5)]
    np.savetxt(OUT/f'ka{ka}_comparison.csv',np.c_[theta,bem.reshape(-1,16),mbs.reshape(-1,16),error.reshape(-1,16)],
        delimiter=',',header=','.join(['theta_deg']+[prefix+n for prefix in ['BEM_','MBS_','error_percent_'] for n in names]),comments='')
    metadata=dict(ka=ka,ref=p['ref'],L=p['order'],geometry=dict(D_vertices_over_H=float(diam/h),ka_verified=float(2*np.pi/.532*radius)),
        mbs_converged=True,mbs_fullauto_tolerance=TOL,theta_range_deg=[LOW,HIGH],mbs_backend=launch['backend'],mbs_elapsed_s=done['elapsed_s'],
        mbs_status=status,mbs_independent_validation=audit,BEM_angular_interpolation=False,
        BEM_averaging='analytic SO(3), zero Euler quadrature nodes',fitted_scale=False,
        dimensional_scale_BEM=factor,Stokes_basis='D M D, D=diag(1,1,-1,-1)',
        lower_edge_M11_error_percent=float(error[0,0,0]),backward_M11_error_percent=float(error[-1,0,0]),
        metrics=measures,sources={**sources,str(table):sha(table),str(mbs_dir/'particle_scaled.txt'):sha(mbs_dir/'particle_scaled.txt')})
    save(OUT/f'ka{ka}_comparison.json',metadata)
    result.update(ready=True,mbs_state='converged',theta=theta,bem=bem,mbs=mbs,bn=bn,mn=mn,error=error,metadata=metadata)
    return result

def decorate(ax):
    ax.set_xlim(LOW,HIGH);ax.set_xticks(np.linspace(LOW,HIGH,6 if HIGH-LOW<=5 else 5));ax.grid(alpha=.16);ax.set_xlabel('θ, °')

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    manifest=read(OUT/'campaign.json');cases=[load(c) for c in manifest['cases']]
    pdf=OUT/CONFIG.get('report_pdf','BEM_MBS_full_series.pdf');temporary=pdf.with_suffix('.tmp.pdf')
    with PdfPages(temporary) as book:
        fig,axes=plt.subplots(2,5,figsize=(18,7),layout='constrained')
        fig.suptitle(f'BEM и MBS-fast: столбики SciIce, ka=20–40; θ={LOW}–{HIGH}°\nm=1,3116; λ=0,532 мкм; D по вершинам/H=0,7',fontsize=15)
        for col,d in enumerate(cases):
            ax,er=axes[:,col];ax.semilogy(d['drawing_theta'],d['drawing_bem'][:,0,0],color=BLUE,label='BEM: аналитическое SO(3)')
            ax.set_title(f'ka={d["ka"]}; ref={d["ref"]}, L={d["order"]}')
            if d['ready']:
                ax.semilogy(d['theta'],d['mbs'][:,0,0],color=ORANGE,ls='--',label=f'MBS: coherent fullauto {TOL*100:g}%')
                err=d['error'][:,0,0];idx=int(np.argmax(abs(err)));er.plot(d['theta'],err,color=ORANGE)
                er.scatter(d['theta'][idx],err[idx],s=18,color='#B22222')
                er.set_title(f'Макс. |ошибка| {abs(err[idx]):.2f}% при {d["theta"][idx]:.2f}°',fontsize=9)
                limit=max(1,float(np.max(abs(err)))*1.12);er.set_ylim(-limit,limit)
            else:er.text(.5,.5,'MBS ещё считается' if d['mbs_state']=='running' else d['mbs_state'],ha='center',transform=er.transAxes)
            ax.set_ylabel('M₁₁, мкм²/ср');er.set_ylabel('100(MBS−BEM)/BEM, %')
            er.axhline(0,color='#555555',lw=.6)
            for item in (ax,er):decorate(item)
        axes[0,0].legend(fontsize=7);book.savefig(fig);fig.savefig(OUT/'overview.png',dpi=145);plt.close(fig)
        for d in cases:
            if not d['ready']:continue
            fig,axes=plt.subplots(4,4,figsize=(16,12),layout='constrained')
            fig.suptitle(f'ka={d["ka"]}: все ненулевые элементы и ошибки\n'
                'M₁₁ размерный; остальные Mᵢⱼ/M₁₁ каждого метода. Ошибка: 100(Mᵢⱼ MBS − Mᵢⱼ BEM)/M₁₁ BEM',fontsize=13)
            for n,(i,j) in enumerate(ENTRIES):
                row,col=n//2,2*(n%2);ax,er=axes[row,col:col+2];name=f'M{i+1}{j+1}'
                b=d['bem'][:,i,j] if (i,j)==(0,0) else d['bn'][:,i,j]
                m=d['mbs'][:,i,j] if (i,j)==(0,0) else d['mn'][:,i,j]
                ax.plot(d['theta'],b,color=BLUE,label='BEM');ax.plot(d['theta'],m,color=ORANGE,ls='--',label='MBS-fast')
                if (i,j)==(0,0):ax.set_yscale('log')
                ax.set_title(name if (i,j)==(0,0) else name+'/собственный M11');ax.legend(fontsize=7)
                err=d['error'][:,i,j];idx=int(np.argmax(abs(err)));er.plot(d['theta'],err,color=ORANGE)
                er.scatter(d['theta'][idx],err[idx],s=20,color='#B22222')
                er.set_title(f'|max|={abs(err[idx]):.2f}% при θ={d["theta"][idx]:.2f}°')
                limit=max(.01,float(np.max(abs(err)))*1.12);er.set_ylim(-limit,limit);er.set_ylabel('% от M11 BEM')
                for item in (ax,er):decorate(item)
            book.savefig(fig);fig.savefig(OUT/f'ka{d["ka"]}_all_nonzero.png',dpi=130);plt.close(fig)
        ready=[d for d in cases if d['ready']]
        for key,label in [('raw_integral_error_percent','Исходные Mij'),('normalized_integral_error_percent','Mij / собственный M11')]:
            fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off');data=[]
            for idx in range(16):
                cells=[f'M{idx//4+1}{idx%4+1}']
                for d in cases:
                    v=d['metadata']['metrics'][idx][key] if d['ready'] else None
                    cells.append('ожидание' if not d['ready'] else '—' if v is None else f'{v:.3f}')
                data.append(cells)
            table=ax.table(cellText=data,colLabels=['Элемент']+[f'ka={d["ka"]}' for d in cases],loc='center',cellLoc='center')
            table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.55)
            fig.suptitle('Интегральные ошибки, % — '+label,fontsize=15)
            fig.text(.5,.90,f'100 ∫|Q(MBS)−Q(BEM)| sinθ dθ / ∫|Q(BEM)| sinθ dθ, θ={LOW}…{HIGH}°',ha='center',fontsize=12)
            fig.text(.1,.06,'«—»: нулевой по симметрии элемент либо тождественное M11/M11.\n'
                f'Ошибка означает расхождение методов; допуск fullauto {TOL*100:g}% относится к сходимости MBS.',fontsize=10)
            book.savefig(fig);plt.close(fig)
    temporary.replace(pdf)
    records=[r for d in cases if d['ready'] for r in d['metadata']['metrics']]
    if records:
        with (OUT/'integral_and_max_errors.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    save(OUT/'status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),complete=len(ready)==5,
        ready_sizes=[d['ka'] for d in ready],pending_sizes=[d['ka'] for d in cases if not d['ready']],
        mbs_states={d['ka']:d['mbs_state'] for d in cases},
        pdf=str(pdf),theta_range_deg=[LOW,HIGH],fullauto_tolerance=TOL,all_sizes_same_geometry=True,coherent_only=True))
    print(json.dumps(read(OUT/'status.json')),flush=True)

if __name__=='__main__':main()
