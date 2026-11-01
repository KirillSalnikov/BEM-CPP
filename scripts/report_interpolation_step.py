#!/usr/bin/env python3
"""Extra held-out controls and a size-spacing recommendation from measured data."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from bounded_bem_run import atomic_json
from study_size_interpolation_step import polynomial
from tmatrix_size_interpolation import fit_aaa
from tmatrix_so3_mstm import analytic_mueller_tmatrix
from study_tmatrix_cutoff import error_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    root=parser.parse_args().directory.resolve();repository=Path(__file__).resolve().parents[1]
    cases=[]
    for ka0 in [3.,6.]:
        folder=root/f'ka{ka0:g}';report=json.loads((folder/'report.json').read_text());plan=report['plan']
        refs=json.loads((folder/'references.json').read_text());samples={};means={}
        for r in refs:
            path=Path(r['T'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise RuntimeError('reference T changed')
            samples[r['delta']]=np.load(path,mmap_mode='r')
            with np.load(folder/f'direct_ka{r["ka"]:.8f}/average.npz') as d:theta=d['theta_deg'];means[r['delta']]=d['mueller']
        records=[]
        for index,original in enumerate(report['records']):
            record=dict(original);nodes=np.array(record['nodes']);method=record['method'];step=record['step']
            if method=='AAA' and step<=.25:
                training=[samples[x] for x in nodes];model,diag=fit_aaa(nodes,training,tolerance=1e-6)
                old=json.loads((folder/f'{index:02d}_{method}_h{step:g}/model.json').read_text())
                if diag['support_indices']!=old['diagnostics']['support_indices']:raise RuntimeError('refit changed support')
                checks=list(record['checks']);known={r['delta'] for r in checks}
                extra=[x for x in sorted(samples) if nodes[0]<x<nodes[-1] and x not in nodes and x not in known]
                omitted_training=[float(x) for x in nodes if x not in model.nodes]
                training_checks=[]
                target=folder/f'extra_AAA_h{step:g}';target.mkdir(exist_ok=True)
                atomic_json(target/'model.json',dict(nodes=nodes.tolist(),support=model.nodes.tolist(),
                    weights_real=model.weights.real.tolist(),weights_imag=model.weights.imag.tolist(),
                    additional_controls=extra,omitted_training_nodes_to_check=omitted_training,holdouts_excluded_from_fit=True,
                    note='Same predetermined algorithm and training matrices as the midpoint test.'))
                for delta in extra+omitted_training:
                    pred=model.evaluate(delta);direct=samples[delta]
                    cache=target/f'control_{ka0+delta:.8f}.npz'
                    if cache.exists():
                        with np.load(cache) as data:mean=data['predicted']
                    else:
                        mean,meta=analytic_mueller_tmatrix(pred,plan['order'],theta,precision='fp64',threads=2,cyclic_order=1)
                        assert meta['Euler_quadrature_nodes']==0
                    metrics=error_metrics(mean,means[delta],theta)
                    csca=float(abs(np.vdot(pred,pred).real/np.vdot(direct,direct).real-1))
                    cext=float(abs((np.trace(pred).real-np.trace(direct).real)/np.trace(direct).real))
                    check=dict(ka=ka0+delta,delta=delta,**metrics,Csca_relative_error=csca,Cext_relative_error=cext,
                        acceptance_error=max(metrics['maximum_all16_error_over_reference_M11'],csca,cext))
                    (training_checks if delta in omitted_training else checks).append(check)
                    if not cache.exists():np.savez_compressed(cache,theta_deg=theta,predicted=mean,direct=means[delta])
                record.update(checks=checks,control_count=len(checks),training_checks=training_checks,
                    maximum_error=max(r['acceptance_error'] for r in checks+training_checks),
                    maximum_mueller_error=max(r['maximum_all16_error_over_reference_M11'] for r in checks+training_checks))
                record['passes_1pct']=record['maximum_error']<=.01
                atomic_json(target/'report.json',record)
                print(json.dumps(dict(ka0=ka0,method=method,h=step,controls=len(checks),max_percent=100*record['maximum_error'])),flush=True)
            records.append(record)
        mueller=[]
        for record in report['records']:
            if record['method']=='AAA':continue
            nodes=np.array(record['nodes']);count=2 if record['method']=='linear' else 4;checks=[]
            for delta in record['controls']:
                left=max(0,min(int(np.searchsorted(nodes,delta))-1-(count==4),len(nodes)-count));support=nodes[left:left+count]
                predicted=polynomial(support,[means[x] for x in support],delta).real
                checks.append(dict(ka=ka0+delta,**error_metrics(predicted,means[delta],theta)))
            mueller.append(dict(method='Mueller_'+record['method'],step=record['step'],training_count=len(nodes),
                control_count=len(checks),maximum_mueller_error=max(x['maximum_all16_error_over_reference_M11'] for x in checks),checks=checks,
                note='Interpolates already analytically averaged Mueller arrays. Does not reconstruct a complex T matrix.'))
        atomic_json(folder/'mueller_interpolation.json',mueller)
        cases.append(dict(ka0=ka0,plan=plan,T_methods=records,Mueller_methods=mueller,
            modal_tail_max_percent=100*max(x['error'] for x in report['modal_tail_controls'])))
    # Existing independent results at larger sizes, with a DIFFERENT aspect.
    oldroot=repository/'runs/multisize_n13116_0532_20261028';historical=[]
    for stage,h in [(0,.5),(1,.25)]:
        d=json.loads((oldroot/f'model_{stage}/validation.json').read_text());model=json.loads((oldroot/f'model_{stage}/model.json').read_text())
        historical.append(dict(ka_range=[10,12],ri=1.3116,H_over_width=1,step=h,method='AAA',
            training_count=len(model['training']),control_count=len(d['records']),
            maximum_mueller_error=max(r['maximum_all16_error_over_reference_M11'] for r in d['records']),
            source=str(oldroot/f'model_{stage}/validation.json')))
    recommendation=dict(tolerance=.01,recommended_T_anchor_step=.25,T_model='Matrix-valued AAA, nine actual T samples over a ka interval of width 2',
        reserve_T_anchor_step=.125,reserve_model='Nine T samples over a ka interval of width 1',
        local_four_anchor_step=.125,local_four_anchor_method='cubic interpolation of complex T',
        two_anchor_step=.05,two_anchor_scope='Central midpoint tests near ka3 and ka6 only; less accuracy margin.',
        equal_volume_radius_step_um=.532*.25/(2*np.pi),
        ka40_validated=False,ka40_reason='Near ka40 only the central size has a complete T at production resolution; partial T40.25 cannot validate all16 orientation averages.',
        continuous_error_bound=False,mesh_accuracy_certified=False,
        definition='Maximum over tested sizes, 1801 scattering angles and 16 entries: abs(delta Mij)/M11_reference. Cross-section errors also checked for T models.',
        cases=cases,historical=historical,new_large_T_runs=0)
    atomic_json(root/'summary.json',recommendation)
    with PdfPages(root/'spacing_comparison.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
        for ax,case in zip(axes,cases):
            for method,label in [('linear','T: linear, 2 anchors'),('local_cubic','T: local cubic, 4 anchors'),('AAA','T: rational AAA, 5 / 9 anchors')]:
                rows=sorted([r for r in case['T_methods'] if r['method']==method],key=lambda r:r['step'])
                ax.loglog([r['step'] for r in rows],[100*r['maximum_mueller_error'] for r in rows],'o-',label=label)
            ax.axhline(1,color='red',ls='--',label='1% limit')
            ax.set(xlabel='Anchor spacing delta ka',ylabel='max |delta Mij| / direct M11, %',title=f'Current prism, ka near {case["ka0"]:g}')
            ax.grid(alpha=.25,which='both');ax.legend(fontsize=8)
        fig.suptitle('Spacing of actual BEM T samples; analytic orientation averaging\n'
            'm=1.3116, H/width=10/7; same-mesh comparisons; ka40 not certified')
        fig.savefig(root/'spacing_comparison.png',dpi=160);pdf.savefig(fig);plt.close(fig)
        case=cases[1];r=next(r for r in case['T_methods'] if r['method']=='AAA' and r['step']==.25)
        worst=max(r['checks']+r.get('training_checks',[]),key=lambda x:x['maximum_all16_error_over_reference_M11']);ka=worst['ka'];folder=root/'ka6'
        path=folder/f'05_AAA_h0.25/control_{ka:.8f}.npz'
        if not path.exists():path=folder/f'extra_AAA_h0.25/control_{ka:.8f}.npz'
        with np.load(path) as data:direct=data['direct'];predicted=data['predicted'];theta=data['theta_deg']
        fig,axes=plt.subplots(4,4,figsize=(14,11),layout='constrained')
        for i in range(4):
            for j in range(4):
                ax=axes[i,j];scale=1 if (i,j)==(0,0) else direct[:,0,0]
                ax.plot(theta,direct[:,i,j]/scale,label='Direct BEM');ax.plot(theta,predicted[:,i,j]/scale,'--',label='Interpolated T')
                errors=np.abs(predicted[:,i,j]-direct[:,i,j])/direct[:,0,0];index=int(np.argmax(errors))
                y=predicted[:,i,j]/scale;ax.plot(theta[index],y[index],'o',ms=3)
                ax.set(title=f'M{i+1}{j+1}: max {100*errors[index]:.3g}%',xlim=(0,180));ax.grid(alpha=.2)
                if (i,j)==(0,0):ax.set_yscale('log');ax.legend(fontsize=8)
        fig.suptitle(f'Worst checked size in the ka5–7 window: ka={ka:g}; delta ka=.25, nine AAA anchors\n'
            'All16 after analytic SO(3); M11 absolute, other entries / direct M11; dots mark maximum error')
        pdf.savefig(fig);plt.close(fig)
    print(json.dumps({k:v for k,v in recommendation.items() if k not in ['cases','historical']},indent=2))


if __name__=='__main__':main()
