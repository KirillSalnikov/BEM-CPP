"""Report measured solve costs and independent A7 / far-field acceptance."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
B=Path(__file__).resolve().parent;R=B.parents[1];sys.path.insert(0,str(R/'scripts'))
from tmatrix_reference import coherency_to_mueller

def mueller(folder,q,variant):
    fields=[np.fromfile(folder/f'plane_{p}.q{q}.{variant}.far.bin',np.complex128).reshape(-1,3) for p in range(2)]
    theta=np.repeat(np.linspace(0,np.pi,181),3);phi=np.tile([0.,.37,1.1],181)
    et=np.c_[np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-np.sin(theta)]
    ep=np.c_[-np.sin(phi),np.cos(phi),np.zeros(len(phi))]
    j=np.empty((len(theta),2,2),complex)
    for p,f in enumerate(fields):j[:,0,p]=np.sum(f*et,axis=1);j[:,1,p]=np.sum(f*ep,axis=1)
    return np.array([coherency_to_mueller(np.outer(v.ravel(),v.ravel().conj())) for v in j])

cases=[]
for folder in sorted(B.glob('*_ka*_ref*')):
    if not (folder/'case.json').exists():continue
    case=json.loads((folder/'case.json').read_text());case['directory']=str(folder);case['quadrature']=[]
    for source in sorted(folder.glob('q[137].jsonl')):
        records=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        setup=next((r for r in records if r['type']=='setup'),None)
        if setup is None:continue
        q=setup['q'];solves=[r for r in records if r['type']=='solve'];action=[r for r in records if r['type']=='action'][-1]
        variants=sorted(set(r['variant'] for r in solves));rows=[]
        for variant in variants:
            v=[r for r in solves if r['variant']==variant];names={r['rhs'] for r in v}
            base=[r for r in solves if r['variant']=='baseline' and r['rhs'] in names]
            total=sum(r['solve_s'] for r in v);baseline=sum(r['solve_s'] for r in base)
            accepted=all(r['accepted'] for r in v)
            row=dict(variant=variant,rhs_count=len(v),complete=len(v)==len(case['rhs']),accepted=accepted,
                solve_s=total,baseline_s=baseline,solve_speedup=baseline/total,
                including_extra_operator_setup_speedup=baseline/(total+(0 if variant.startswith('baseline') else setup['cheap_setup_s'])),
                including_common_and_extra_setup_speedup=(baseline+setup['exact_setup_s']+setup['mbj_setup_s'])/
                    (total+setup['exact_setup_s']+setup['mbj_setup_s']+(0 if variant.startswith('baseline') else setup['cheap_setup_s'])),
                maximum_strict_residual=max(r['strict_residual'] for r in v),
                maximum_far_relative_error=max(r['far_relative_error'] for r in v),
                full_actions=sum(r['full_actions'] for r in v),cheap_actions=sum(r['cheap_actions'] for r in v),
                strict_actions=sum(r['strict_actions'] for r in v),
                outer_iterations=sum(r['iterations'] for r in v))
            if len(v)>0 and baseline>total and not variant.startswith('baseline'):
                row['estimated_setup_break_even_rhs']=setup['cheap_setup_s']/((baseline-total)/len(v))
            if {'plane_0','plane_1'}<=names:
                m=mueller(folder,q,variant);ref=mueller(folder,q,'baseline')
                scaled=abs(m-ref)/ref[:,0,0,None,None];index=np.unravel_index(np.argmax(scaled),scaled.shape)
                row['fixed_incidence_maximum_all16_error_over_reference_M11']=float(scaled[index])
                row['worst_observation_theta_deg']=float(index[0]//3)
                row['fixed_incidence_observation_directions']=543
            rows.append(row)
        case['quadrature'].append(dict(setup=setup,action_speedup=action['full_mean_s']/action['cheap_mean_s'],
            operator_relative_difference_on_RHS=action['relative_action_difference_on_RHS'],variants=rows,solves=solves))
    if case['quadrature']:cases.append(case)
report=dict(cases=cases,scope='Reduced Galerkin quadrature as an inner inverse, inspired by the paper; not a full Nystrom EFIE implementation.',
    strict_equation='Same mesh, BDM1 currents, 7-point regular quadrature, Duffy6, mass13, tol2e-6.',
    limitations=['Same-mesh comparison, not a mesh convergence certificate.',
        'Fixed incident directions and selected spherical RHS; no complete new T matrix or orientation average.',
        'GPU timing isolates production by temporary SIGSTOP/SIGCONT; pause receipts saved.',
        'FGMRES does not currently share the baseline Krylov recycling; each variant owns its history.'])
(B/'report.json').write_text(json.dumps(report,indent=2)+'\n')
with PdfPages(B/'reduced_quadrature.pdf') as pdf:
    for c in cases:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        labels=[];costs=[];errors=[]
        for q in c['quadrature']:
            for v in q['variants']:
                if not v['complete']:continue
                labels.append('q'+str(q['setup']['q'])+' '+v['variant']+(' REJECTED' if not v['accepted'] else ''));costs.append(v['solve_speedup']);errors.append(v['maximum_strict_residual'])
        x=np.arange(len(labels));axes[0].barh(x,costs);axes[0].axvline(1,color='black',lw=1);axes[0].set_xlabel('Solve speedup over matched A7 baseline')
        axes[1].barh(x,errors);axes[1].axvline(2e-6,color='red',ls='--');axes[1].set_xscale('log');axes[1].set_xlabel('Maximum residual in original A7 equation')
        for ax in axes:ax.set_yticks(x,labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        fig.suptitle(f"{c['shape_name']} ka={c['ka']:g}, ref={c['ref']}; same-mesh diagnostic")
        fig.tight_layout();pdf.savefig(fig);plt.close(fig)
for c in cases:
    print(c['shape_name'],c['ka'],c['ref'])
    for q in c['quadrature']:
        print(' q',q['setup']['q'],'action speedup',round(q['action_speedup'],2))
        for r in q['variants']:
            print(' ',r['variant'],r['rhs_count'],round(r['solve_speedup'],3),'accepted',r['accepted'],'max residual',r['maximum_strict_residual'])
