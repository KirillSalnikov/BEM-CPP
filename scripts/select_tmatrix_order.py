#!/usr/bin/env python3
"""Select a guarded order from saved analytic SO(3) truncation controls.

The certificate is restricted to its exact size, material, geometry and mesh.
It bounds truncation relative to the saved reference, not modes beyond it.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from study_tmatrix_cutoff import atomic_json,digest,error_metrics,stable_minimum


def guarded_selection(records,reference_order,tolerance=.01,guard=2):
    if not math.isfinite(tolerance) or not 0<tolerance<1 or guard<1:
        raise ValueError('require finite tolerance in (0,1) and at least one guard order')
    rows={r['order']:r for r in records}
    if len(rows)!=len(records) or sorted(rows)!=list(range(1,reference_order+1)):
        raise ValueError('all integer cutoffs through the reference are required')
    if any(not math.isfinite(r['acceptance_error']) or r['acceptance_error']<0 for r in records):
        raise ValueError('nonfinite or negative cutoff error')
    minimum=stable_minimum(records,tolerance)
    selected=minimum+guard if minimum is not None else None
    supported=selected is not None and selected<reference_order
    return dict(minimum_L=minimum,guard_orders=guard,selected_L=selected if supported else None,
                tolerance=tolerance,status='selected' if supported else 'extend_reference_order',
                selected_error=rows[selected]['acceptance_error'] if supported else None)


def physical_key(parameters):
    return {k:parameters.get(k,default) for k,default in
            (('shape',None),('ka',None),('ri',None),('ri_imag',0.),('prism_aspect',1.),
             ('wavelength',None),('ref',None),('geometry_digest',None))}


def validate_target(selection,parameters):
    if selection['status']!='selected':raise ValueError('reference order must be extended')
    key=physical_key(parameters)
    for name,value in selection['physical_key'].items():
        if value!=key[name]:raise ValueError('order evidence does not match target '+name)
    return selection['selected_L']


def audit_and_select(study,tolerance=.01,guard=2):
    study=Path(study).resolve();saved=json.loads((study/'summary.json').read_text())
    source=Path(saved['source_run']);summary=json.loads((source/'summary.json').read_text())
    identity=json.loads((study/'identity.json').read_text())
    if not summary['complete']:raise ValueError('reference T is incomplete')
    order=summary['parameters']['order']
    if order!=saved['reference_order'] or order!=identity['reference_order']:
        raise ValueError('reference order mismatch')
    sha=digest(source/'tmatrix.npy')
    if sha!=identity['T_sha256'] or sha!=saved['source_T_sha256']:
        raise ValueError('reference T changed')
    if saved['Euler_quadrature_nodes']!=0:raise ValueError('analytic SO3 required')
    with np.load(study/f'L{order:03d}.npz') as data:
        theta=data['theta_deg'];reference=data['mueller']
    metadata=json.loads((study/f'L{order:03d}.json').read_text())
    records=[]
    for old in saved['records']:
        degree=old['order']
        with np.load(study/f'L{degree:03d}.npz') as data:
            np.testing.assert_array_equal(data['theta_deg'],theta)
            metrics=error_metrics(data['mueller'],reference,theta)
        md=json.loads((study/f'L{degree:03d}.json').read_text())
        for observable in ('scattering_cross_section','extinction_cross_section'):
            denominator=metadata[observable]
            if not math.isfinite(denominator) or denominator==0:raise ValueError('invalid reference cross section')
            metrics[observable+'_relative_error']=abs(md[observable]/denominator-1)
        error=max(metrics['maximum_all16_error_over_reference_M11'],
                  metrics['scattering_cross_section_relative_error'],metrics['extinction_cross_section_relative_error'])
        if not math.isfinite(error) or abs(error-old['acceptance_error'])>1e-11:
            raise ValueError('saved cutoff metrics differ from arrays')
        records.append(dict(order=degree,acceptance_error=error,metrics=metrics))
    selection=guarded_selection(records,order,tolerance,guard)
    selected=selection['selected_L'];modes=lambda l:2*l*(l+2)
    selection.update(source_study=str(study),source_run=str(source),source_T_sha256=sha,
        physical_key=physical_key(summary['parameters']),reference_L=order,reference_modes=modes(order),
        selected_modes=modes(selected) if selected else None,
        mode_reduction_percent=100*(1-modes(selected)/modes(order)) if selected else None,
        T_storage_reduction_percent=100*(1-(modes(selected)/modes(order))**2) if selected else None,
        angles=len(theta),Euler_nodes=0,records=records,
        limitations=['Relative to the saved reference order, not modes above it.',
                     'Surface discretization and solver tolerance errors require separate controls.',
                     'Finite output-angle grid; orientation integration is analytic.',
                     'Mode/storage savings are not measured elapsed-time savings.',
                     'No reuse at other sizes, refractive indices, aspects or meshes.'])
    return selection


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tolerance',type=float,default=.01)
    parser.add_argument('--guard',type=int,default=2)
    args=parser.parse_args();result=audit_and_select(args.study,args.tolerance,args.guard)
    args.out.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.out,result)
    print(json.dumps({k:result[k] for k in ('status','minimum_L','selected_L','selected_error','mode_reduction_percent')}))
