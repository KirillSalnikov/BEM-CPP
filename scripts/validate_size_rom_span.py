#!/usr/bin/env python3
"""Measure the span of the fixed central ROM on old and new direct controls."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from bounded_bem_run import atomic_json
from tmatrix_size_rom import ReducedBEM,ka_scaled_equation
from tmatrix_so3_mstm import analytic_mueller_tmatrix
from study_tmatrix_cutoff import error_metrics

ROOT=Path(__file__).resolve().parents[1]


def main():
    root=ROOT/'benchmarks/size_rom_20261031';records=[]
    for ka0,ranks in [(3.,[512,768]),(6.,[768,1024])]:
        source=root/f'ka{ka0:g}_degree12';out=root/f'ka{ka0:g}_span';out.mkdir(exist_ok=True)
        with np.load(source/'model.npz') as d:model=ReducedBEM(d['A'],d['B'],d['D'],d['phase'],float(d['ka0']))
        previous=json.loads((source/'report.json').read_text())['records']
        order=12 if ka0==3 else 16;theta=np.linspace(0,180,1801)
        refs=[];base=ROOT/f'benchmarks/interpolation_step_20261031/ka{ka0:g}'
        for r in json.loads((base/'references.json').read_text()):
            if r['ka']==ka0:continue
            refs.append(dict(ka=r['ka'],T=r['T'],average=str(base/f'direct_ka{r["ka"]:.8f}/average.npz'),sha256=r['sha256']))
        for family in ['exact_wide','exact_quarter']:
            folder=root/f'ka{ka0:g}_{family}'
            if not (folder/'report.json').exists():continue
            for child in sorted(folder.glob('ka*')):
                t=child/'T.npy'
                if not t.exists():continue
                refs.append(dict(ka=float(child.name[2:]),T=str(t),average=str(child/'direct_average.npz'),sha256=hashlib.sha256(t.read_bytes()).hexdigest()))
        atomic_json(out/'plan.json',dict(references=refs,rank_candidates=ranks,variants=['plain','ka_scaled'],
            operator_degree=12,training_ka=[ka0],no_control_data_used_to_build_models=True,
            model_sha256=hashlib.sha256((source/'model.npz').read_bytes()).hexdigest()))
        for variant,candidate in [('plain',model),('ka_scaled',ka_scaled_equation(model))]:
            for ref in refs:
                ka=ref['ka'];path=Path(ref['T'])
                if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:raise RuntimeError('reference changed')
                direct=np.load(path)
                with np.load(ref['average']) as data:direct_mean=data['mueller']
                for rank in ranks:
                    cache=out/f'{variant}_r{rank}_ka{ka:.8f}.npz';metric_path=cache.with_suffix('.json')
                    if metric_path.exists():
                        r=json.loads(metric_path.read_text());records.append(r);continue
                    reused=next((r for r in previous if r['ka']==ka and r['rank']==rank),None)
                    if variant=='plain' and reused is not None:
                        r=dict(reused,center=ka0,variant=variant,
                               artifact_path=str(source/f'rank{rank}_ka{ka:.8f}.npz'))
                        atomic_json(metric_path,r);records.append(r);continue
                    start=time.perf_counter();pred,y=candidate.evaluate(ka,rank=rank);evaluation_s=time.perf_counter()-start
                    mean,info=analytic_mueller_tmatrix(pred,order,theta,precision='fp64',threads=2,cyclic_order=1)
                    assert info['Euler_quadrature_nodes']==0
                    metrics=error_metrics(mean,direct_mean,theta)
                    csca=float(abs(np.vdot(pred,pred).real/np.vdot(direct,direct).real-1));cext=float(abs(np.trace(pred).real/np.trace(direct).real-1))
                    r=dict(center=ka0,ka=ka,offset=ka-ka0,rank=rank,variant=variant,**metrics,
                        Csca_relative_error=csca,Cext_relative_error=cext,
                        acceptance_error=max(metrics['maximum_all16_error_over_reference_M11'],csca,cext),
                        evaluation_s=evaluation_s,analytic_SO3_s=info['end_to_end_s'],artifact_path=str(cache))
                    np.savez_compressed(cache,theta_deg=theta,predicted=mean,direct=direct_mean);atomic_json(metric_path,r);records.append(r)
                    if abs(ka-ka0)>=1:print(json.dumps(dict(center=ka0,ka=ka,variant=variant,rank=rank,percent=100*r['acceptance_error'])),flush=True)
    atomic_json(root/'span_report.json',dict(status='complete',records=records,tolerance=.01,new_large_T_runs=0,
        mesh_accuracy_certified=False,continuous_error_bound=False))


if __name__=='__main__':main()
