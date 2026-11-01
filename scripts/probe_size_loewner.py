#!/usr/bin/env python3
"""Evaluate a frozen Loewner model on saved independent BEM size controls."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from bounded_bem_run import atomic_json
from probe_size_interpolation_transforms import load_case
from study_tmatrix_cutoff import cutoff_indices, error_metrics
from tmatrix_size_loewner import fit_loewner
from tmatrix_so3_mstm import analytic_mueller_tmatrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--wide', action='store_true')
    p.add_argument('--rtol', type=float, default=1e-10)
    args=p.parse_args(); out=args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    cases = [load_case(k) for k in (3.,6.)]
    if args.wide:
        samples={}; means={}; refs=[]; order=16
        for c in cases:
            index=cutoff_indices(order,c['order'])
            for ka,t in c['samples'].items():
                a=np.zeros((576,576),complex);a[np.ix_(index,index)]=t;samples[ka]=a
            means.update(c['means']);refs+=c['references']
        cases=[dict(center=4.,samples=samples,means=means,order=16,
                    references=refs,theta=cases[0]['theta'],
                    plan={'padding':'L12 -> L16 by explicit mode mapping in ka2-4; no known controls in (4,5)'})]
    records=[]
    for case in cases:
        center=case['center'];samples=case['samples'];theta=case['theta']
        nodes=np.array([2.,2.5,3.,3.5,4.,5.,6.,6.5,7.] if args.wide
                       else [center+x for x in (-1.,-.5,0.,.5,1.)])
        controls=[k for k in sorted(samples) if nodes[0]<k<nodes[-1] and k not in nodes]
        folder=out/f'ka{center:g}';folder.mkdir(exist_ok=True)
        atomic_json(folder/'plan.json',dict(nodes=nodes.tolist(),controls=controls,rtol=args.rtol,
            source=case['plan'],new_Maxwell_solves=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        atomic_json(folder/'references.json',case['references'])
        start=time.perf_counter();model,info=fit_loewner(nodes,[samples[k] for k in nodes],rtol=args.rtol)
        fit_seconds=time.perf_counter()-start
        atomic_json(folder/'fit.json',dict(info=info,fit_seconds=fit_seconds))
        np.savez_compressed(folder/'model.npz',center=model.center,scale=model.scale,central_sample=model.central_sample,
            loewner=model.loewner,shifted=model.shifted,left=model.left,right=model.right,domain=model.domain)
        print(json.dumps(dict(phase='fit',center=center,rank=info['rank'],fit_seconds=fit_seconds,
            training_error=info['maximum_training_relative_error'])),flush=True)
        checks=[]
        for k in controls:
            start=time.perf_counter();pred=model.evaluate(k);eval_seconds=time.perf_counter()-start
            mean,meta=analytic_mueller_tmatrix(pred,case['order'],theta,precision='fp64',threads=2,cyclic_order=1)
            assert meta['Euler_quadrature_nodes']==0
            direct=samples[k];csca=float(abs(np.vdot(pred,pred).real/np.vdot(direct,direct).real-1))
            cext=float(abs(np.trace(pred).real/np.trace(direct).real-1))
            check=dict(ka=k,**error_metrics(mean,case['means'][k],theta),Csca_relative_error=csca,
                Cext_relative_error=cext,T_relative_error=float(np.linalg.norm(pred-direct)/np.linalg.norm(direct)),
                evaluation_seconds=eval_seconds)
            check['acceptance_error']=max(check['maximum_all16_error_over_reference_M11'],csca,cext)
            checks.append(check)
            np.savez_compressed(folder/f'ka{k:.8f}.npz',theta_deg=theta,predicted=mean,direct=case['means'][k])
            atomic_json(folder/'progress.json',checks)
        record=dict(center=center,nodes=nodes.tolist(),controls=controls,checks=checks,fit=info,fit_seconds=fit_seconds,
            maximum_error=max(c['acceptance_error'] for c in checks),
            maximum_mueller_error=max(c['maximum_all16_error_over_reference_M11'] for c in checks))
        records.append(record);atomic_json(folder/'report.json',record)
        print(json.dumps(dict(phase='result',center=center,maximum_percent=100*record['maximum_error'])),flush=True)
    atomic_json(out/'report.json',dict(records=records,new_Maxwell_solves=0,wide=args.wide,status='complete'))


if __name__=='__main__':main()
