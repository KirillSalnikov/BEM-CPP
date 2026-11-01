#!/usr/bin/env python3
"""Confirm the fixed wide-window AAA on every remaining saved direct sample."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from bounded_bem_run import atomic_json
from probe_size_interpolation_transforms import ROOT, load_case
from study_tmatrix_cutoff import cutoff_indices, error_metrics
from tmatrix_size_interpolation import fit_aaa
from tmatrix_so3_mstm import analytic_mueller_tmatrix


def main():
    root=ROOT/'benchmarks/interpolation_improvement_20261031'
    out=root/'wide_AAA_all_controls';out.mkdir(exist_ok=True)
    cases=[load_case(k) for k in (3.,6.)];samples={};means={};theta=cases[0]['theta']
    for c in cases:
        index=cutoff_indices(16,c['order'])
        for k,t in c['samples'].items():
            a=np.zeros((576,576),complex);a[np.ix_(index,index)]=t;samples[k]=a
        means.update(c['means'])
    nodes=[2.,2.5,3.,3.5,4.,5.,6.,6.5,7.]
    controls=[k for k in sorted(samples) if k not in nodes]
    atomic_json(out/'plan.json',dict(nodes=nodes,controls=controls,tolerance=.01,fit_tolerance=1e-6,
        no_controls_used_in_fit=True,note='Same fixed raw-AAA algorithm and training nodes as initial wide comparison; confirms additional saved controls. No controls in (4,5).'))
    model,info=fit_aaa(nodes,[samples[k] for k in nodes],tolerance=1e-6)
    atomic_json(out/'model.json',dict(diagnostics=info,support=model.nodes.tolist(),
        weights_real=model.weights.real.tolist(),weights_imag=model.weights.imag.tolist()))
    checks=[];training_checks=[]
    omitted=[k for k in nodes if k not in model.nodes]
    for k in controls+omitted:
        pred=model.evaluate(k);direct=samples[k]
        mean,meta=analytic_mueller_tmatrix(pred,16,theta,precision='fp64',threads=2,cyclic_order=1)
        assert meta['Euler_quadrature_nodes']==0
        csca=float(abs(np.vdot(pred,pred).real/np.vdot(direct,direct).real-1))
        cext=float(abs(np.trace(pred).real/np.trace(direct).real-1))
        check=dict(ka=k,**error_metrics(mean,means[k],theta),Csca_relative_error=csca,Cext_relative_error=cext)
        check['acceptance_error']=max(check['maximum_all16_error_over_reference_M11'],csca,cext)
        (training_checks if k in omitted else checks).append(check)
        np.savez_compressed(out/f'ka{k:.8f}.npz',theta_deg=theta,predicted=mean,direct=means[k])
    wide=dict(nodes=nodes,checks=checks,training_checks=training_checks,control_count=len(checks),
        maximum_error=max(c['acceptance_error'] for c in checks+training_checks))
    atomic_json(out/'report.json',wide)
    local=json.loads((root/'local/report.json').read_text())['records']
    pooled=[]
    for name in sorted({r['name'] for r in local}):
        subset=[r for r in local if r['name']==name]
        pooled.append(dict(name=name,maximum_percent=100*max(r['maximum_error'] for r in subset),
            per_case_percent={str(r['center']):100*r['maximum_error'] for r in subset}))
    methods={}
    for name in ['loewner_local','loewner_local_regularized','loewner_wide','loewner_wide_regularized']:
        report=json.loads((root/name/'report.json').read_text())
        methods[name]=[dict(center=r['center'],maximum_percent=100*r['maximum_error'],
            rank=r['fit']['rank'],fit_seconds=r['fit_seconds']) for r in report['records']]
    differences=[]
    for c in cases:
        k=c['center'];a=c['means'][k];b=c['means'][k+.25]
        differences.append(dict(ka0=k,ka1=k+.25,
            forward_M11_change_percent=100*float(b[0,0,0]/a[0,0,0]-1),
            maximum_all16_difference_percent=100*error_metrics(a,b,theta)['maximum_all16_error_over_reference_M11']))
    summary=dict(status='complete',new_Maxwell_solves=0,new_large_T_runs=0,mesh_accuracy_certified=False,
        local_nodes=5,local_spacing=.5,local_screening=pooled,
        wide_nodes=9,wide_domain=[2,7],wide_control_count=len(checks),wide_maximum_percent=100*wide['maximum_error'],
        wide_gaps=np.diff(nodes).tolist(),wide_no_controls_between=[4,5],loewner=methods,
        direct_size_changes=differences,
        conclusion='No tested model validated a larger interpolation spacing at 1%. The earlier 0.25 result is method-specific, not a universal spacing limit.',
        verified_scope='Same-mesh saved T samples only, m=1.3116 and H/width=10/7; ka40 not tested.',
        sources={'reactance_matrix':'https://arxiv.org/abs/1210.0936',
                 'loewner':'https://www.sciencedirect.com/science/article/pii/S0024379507001280'})
    atomic_json(root/'summary.json',summary)
    with PdfPages(root/'comparison.pdf') as pdf:
        old=json.loads((ROOT/'benchmarks/interpolation_step_20261031/summary.json').read_text())
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
        for c in old['cases']:
            r=next(r for r in c['T_methods'] if r['method']=='AAA' and r['step']==.25)
            rows=sorted(r['checks'],key=lambda x:x['ka'])
            ax.plot([r['ka'] for r in rows],[100*r['acceptance_error'] for r in rows],'o-',label=f'9 anchors, width 2, ka near {c["ka0"]:g}')
        for a,b in [(2,4),(5,7)]:
            rows=[r for r in checks if a<r['ka']<b]
            ax.plot([r['ka'] for r in rows],[100*r['acceptance_error'] for r in rows],'s--',color='C3',label='9 anchors over ka2-7' if a==2 else None)
        ax.axhline(1,color='black',ls=':',label='1% target')
        ax.set(yscale='log',xlabel='ka (independent direct control)',ylabel='Maximum error, %',
            title='Equal number of actual T samples; raw matrix-valued AAA + analytic SO(3)')
        ax.legend();ax.grid(alpha=.2,which='both');pdf.savefig(fig);fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
        c=cases[1];fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        norm=c['means'][6.][0,0,0]
        for k in [6.,6.25,6.5]:
            axes[0].semilogy(theta,c['means'][k][:,0,0]/norm,label=f'ka={k:g}')
        axes[0].set(ylabel='M11 / M11(ka6, forward)',xlabel='Scattering angle, deg',title='Direct BEM: one shared normalization')
        a=c['means'][6.];b=c['means'][6.25]
        axes[1].plot(theta,100*np.max(abs(a-b),axis=(1,2))/b[:,0,0])
        axes[1].set(ylabel='max over ij |Mij(6)-Mij(6.25)| / M11(6.25), %',xlabel='Scattering angle, deg',title='Actual size change; not interpolation error')
        for ax in axes:ax.grid(alpha=.2);ax.set_xlim(0,180)
        axes[0].legend();pdf.savefig(fig);plt.close(fig)
    files=[ROOT/'scripts'/name for name in ['probe_size_interpolation_transforms.py','probe_size_loewner.py',
        'tmatrix_size_loewner.py','report_interpolation_improvement.py']]
    atomic_json(root/'source_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
