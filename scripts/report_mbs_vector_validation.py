#!/usr/bin/env python3
"""Plot vector/scalar/native orientation averages with common BEM normalization."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/d6h_20261026/mbs_vector_correction'
SCALAR=ROOT/'benchmarks/d6h_20261026/mbs_wave_correction/orientation_curve'
CHECKOUT=ROOT/'experiments/MBS-fast-wave'
sys.path.insert(0,str(CHECKOUT))
from waveoptics.prism import mueller_from_jones
from average_mbs_wave_curve import folded_quadrature,case_path,THETA,PHI


def read(path): return json.loads(path.read_text())


def native_scalar_mueller():
    cached=OUT/'native_scalar_mueller.npz'
    if cached.exists():
        d=np.load(cached)
        return d['native'],d['scalar']
    phi=np.deg2rad(PHI)
    r=np.zeros((len(THETA),len(PHI),2,2))
    r[:,:,0,0]=r[:,:,1,1]=np.cos(phi)
    r[:,:,0,1]=-np.sin(phi)
    r[:,:,1,0]=np.sin(phi)
    r[0]=np.eye(2)
    right=r.swapaxes(-2,-1).copy()
    right[0,:,0,0]=right[0,:,1,1]=np.cos(phi)
    right[0,:,0,1]=np.sin(phi)
    right[0,:,1,0]=-np.sin(phi)
    left=right[0].swapaxes(-2,-1)
    native=np.zeros((len(THETA),4,4))
    scalar=np.zeros_like(native)
    for key,weight in folded_quadrature('89_30').items():
        j=np.load(case_path(key)/'native.npz')['jones']
        delta=np.load(case_path(key)/'corrected.npz')['delta_f']
        for source,target in [(j,native),(j+delta[:,:,None,None]*r,scalar)]:
            rotated=source@right
            rotated[0]=left@rotated[0]
            target+=weight*400*mueller_from_jones(rotated).mean(axis=1)
    parity=np.array([1,1,-1,-1])
    mask=(1+parity[:,None]*parity[None,:])/2
    native*=mask
    scalar*=mask
    np.savez_compressed(cached,theta_deg=THETA,native=native,scalar=scalar)
    return native,scalar


def save(fig,name):
    for suffix in ['png','pdf']: fig.savefig(OUT/f'{name}.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def main():
    fine=np.load(OUT/'g256_s128_decay/mean_89_30.npz')
    coarse=np.load(OUT/'g256_s128_decay/mean_45_16.npz')
    spatial=np.load(OUT/'g512_s256_decay/mean_45_16.npz')
    bem=np.load(SCALAR/'bem_reference.npz')['mueller']
    native,scalar=native_scalar_mueller()
    previous=np.load(SCALAR/'mean_89_30.npz')
    np.testing.assert_allclose(native[:,0,0],previous['baseline_M11'],rtol=1e-12)
    np.testing.assert_allclose(scalar[:,0,0],previous['corrected_M11'],rtol=1e-12)
    vector=fine['mueller_native']
    d=np.array([1,1,-1,-1])
    curves=[bem]+[m*d[None,:,None]*d[None,None,:] for m in [native,scalar,vector]]
    colors=['#1765a3','#d75c22','#26935c','#8d4bb0']
    labels=['BEM · аналитическая T','MBS исходный','Скалярная поправка','Векторная поправка']
    styles=['-','--',':','-']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.grid':True,'grid.alpha':.22,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,5.6))
    fig.subplots_adjust(left=.055,right=.985,bottom=.23,top=.81,wspace=.28)
    for m,color,label,style in zip(curves,colors,labels,styles):
        axes[0].plot(THETA,m[:,0,0]/bem[0,0,0],style,color=color,lw=1.9,label=label)
        axes[1].semilogy(THETA,m[:,0,0],style,color=color,lw=1.9,label=label)
        if m is not bem: axes[2].plot(THETA,100*(m[:,0,0]/bem[:,0,0]-1),style,color=color,lw=1.9)
    axes[0].set(xlim=(0,7),xlabel='Угол θ, градусы',ylabel='⟨M₁₁⟩ / ⟨M₁₁,BEM(0)⟩',title='Центральный передний лепесток')
    axes[1].set(xlim=(0,30),xlabel='Угол θ, градусы',ylabel='⟨M₁₁⟩, k² dσ/dΩ',title='Усреднённая угловая зависимость')
    axes[2].set(xlim=(0,30),xlabel='Угол θ, градусы',ylabel='Расхождение с BEM, %',title='Ошибка: векторизация не устранила рост')
    axes[2].set_yscale('symlog',linthresh=5)
    axes[2].set_yticks([-20,-5,0,5,20,100,500],labels=['−20','−5','0','5','20','100','500'])
    axes[2].axhspan(-5,5,color='#888888',alpha=.10)
    axes[2].axhline(0,color='#333333',lw=.7)
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='lower center',bbox_to_anchor=(.5,.07),ncol=4,frameon=False)
    fig.suptitle('Векторное обобщение волновой поправки · ka = 20, m = 1.3, H/D = 1\n'
                 'Связанные Ex, Ey, Ez; две входные поляризации; полная добавка 2×2 к Джонсу',fontsize=14)
    fig.text(.5,.025,'MBS-варианты усреднены численно с D6h-симметрией; BEM — аналитически. Общая нормировка, без подгонки.',ha='center',fontsize=10)
    save(fig,'orientation_averaged_vector_vs_scalar')
    fig,axes=plt.subplots(4,4,figsize=(16,12))
    fig.subplots_adjust(left=.06,right=.985,top=.89,bottom=.09,hspace=.36,wspace=.31)
    for i in range(4):
        for j in range(4):
            ax=axes[i,j]
            for m,color,label,style in zip(curves,colors,labels,styles):
                y=m[:,i,j] if (i,j)==(0,0) else m[:,i,j]/bem[:,0,0]
                ax.plot(THETA,y,style,color=color,lw=1.4,label=label)
            ax.set_xlim(0,30)
            if (i,j)==(0,0): ax.set_yscale('log')
            ax.set_title(f'M{i+1}{j+1}'+('' if (i,j)==(0,0) else ' / M11 BEM'))
            if i==3: ax.set_xlabel('θ, градусы')
    fig.suptitle('Все 16 элементов после усреднения ориентаций · ka = 20\n'
                 'Один знаменатель BEM для всех методов; межблочные элементы обнулены зеркальной симметрией',fontsize=14)
    fig.legend(handles,legend,loc='lower center',bbox_to_anchor=(.5,.025),ncol=4,frameon=False)
    save(fig,'orientation_averaged_all16_vector_vs_scalar')
    orientation_delta=coarse['M11']/fine['M11']-1
    spatial_delta=spatial['M11']/coarse['M11']-1
    azimuth_delta=fine['M11_phi32']/fine['M11']-1
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    for data,label in [(orientation_delta,'Ориентации: 45×16 → 89×30'),(spatial_delta,'Волновая сетка: 256²×128 → 512²×256'),
                       (azimuth_delta,'Азимут: 32 → 64')]: ax.plot(THETA,100*data,label=label)
    ax.axhline(0,color='#333333',lw=.7)
    ax.set(xlim=(0,30),xlabel='θ, градусы',ylabel='Относительное изменение M₁₁, %',title='Сходимость векторного приближения')
    ax.legend()
    save(fig,'vector_convergence')
    a=np.load(OUT/'fixed_b37_g17_cpu64/result.npz')
    b=np.load(OUT/'fixed_b37_g17_cuda32/result.npz')
    backend_error=float(np.max(abs(a['corrected_mueller']-b['corrected_mueller'])/a['corrected_mueller'][:,0,0,None,None]))
    report={'experimental':True,'method':'coupled three-component one-way projected phase screens',
            'ka':20,'physical_accuracy_certified':False,'dielectric_boundary_conditions_solved':False,
            'T_matrix_of_new_model_computed':False,'orientation_average':'numerical D6h plus uniform azimuth',
            'BEM_average':'analytic full SO(3) T matrix',
            'error_samples':[{ 'theta_deg':float(THETA[i]),
                               **{label:float(100*(m[i,0,0]/bem[i,0,0]-1))
                                  for label,m in zip(['native','scalar','vector'],curves[1:])}}
                              for i in [0,2,4,10,20,30,40,50,60]],
            'max_orientation_refinement_relative_change':float(np.max(abs(orientation_delta))),
            'max_spatial_refinement_relative_change':float(np.max(abs(spatial_delta))),
            'forward_spatial_refinement_relative_change':float(spatial_delta[0]),
            'max_phi32_vs64_relative_change':float(np.max(abs(azimuth_delta))),
            'CPU64_vs_CUDA32_max_Mueller_difference_over_local_M11':backend_error,
            'timing':read(OUT/'g256_s128_decay/summary_89_30.json'),
            'sources':[{'url':'https://arxiv.org/abs/2208.10965','used_for':'dyadic background propagation operator, not the hybrid model'},
                       {'url':'https://arxiv.org/abs/2505.13869','used_for':'three-component transverse angular-spectrum projection'}],
            'source_sha256':{name:hashlib.sha256((CHECKOUT/name).read_bytes()).hexdigest()
                             for name in ['waveoptics/vector.py','tools/mbs_wave.py','tests/test_wave_vector.py']},
            'limitations':['Background Q transversality does not impose div(epsilon E)=0 at dielectric boundaries.',
                           'No reflected-wave propagation or complete interface matching.',
                           'Adding the field difference to native MBS remains an unmatched hybrid approximation.',
                           'The vector model retains large off-axis errors after orientation averaging.']}
    (OUT/'validation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(OUT/'all16_averages_BEM_convention.npz',theta_deg=THETA,BEM=bem,
                        MBS=curves[1],scalar=curves[2],vector=curves[3])
    np.savetxt(OUT/'averaged_M11.csv',np.column_stack([THETA]+[m[:,0,0] for m in curves]),delimiter=',',
               header='theta_deg,BEM_analytic,MBS_native,MBS_scalar,MBS_vector',comments='')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
