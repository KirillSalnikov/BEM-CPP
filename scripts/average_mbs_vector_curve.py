#!/usr/bin/env python3
"""Evaluate vector correction on the same orientation nodes and native MBS data."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/MBS-fast-wave'))
from waveoptics.prism import Parameters, mueller_from_jones
from waveoptics.vector import exit_field_difference_vector, CudaVectorRadiator
from average_mbs_wave_curve import folded_quadrature, case_path, THETA, PHI, atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--grid',type=int,default=256)
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--label',default='89_30',choices=['45_16','89_30'])
    parser.add_argument('--evanescent',default='decay',choices=['decay','discard'])
    args=parser.parse_args()
    out=ROOT/'benchmarks/d6h_20261026/mbs_vector_correction'/f'g{args.grid}_s{args.steps}_{args.evanescent}'
    out.mkdir(parents=True,exist_ok=True)
    coordinates=(np.arange(args.grid)-args.grid/2+.5)*8/args.grid
    radiator=CudaVectorRadiator(coordinates,20,THETA[:,None]+np.zeros_like(PHI),PHI[None,:]+np.zeros_like(THETA[:,None]))
    quad=folded_quadrature(args.label)
    mean=np.zeros((len(THETA),len(PHI)))
    mean_mueller=np.zeros((len(THETA),len(PHI),4,4))
    metadata=[]
    started=time.perf_counter()
    phi=np.deg2rad(PHI)
    # Put the native fixed laboratory incident axes into the scattering plane.
    # M11 is invariant to this right rotation. It matters for all other entries.
    rotation=np.empty((len(THETA),len(PHI),2,2))
    rotation[:,:,0,0]=rotation[:,:,1,1]=np.cos(phi)
    rotation[:,:,0,1]=np.sin(phi)
    rotation[:,:,1,0]=-np.sin(phi)
    # At the pole native axes are fixed: perform both input/output rotations.
    pole_left=np.empty((len(PHI),2,2))
    pole_left[:,0,0]=pole_left[:,1,1]=np.cos(phi)
    pole_left[:,0,1]=-np.sin(phi)
    pole_left[:,1,0]=np.sin(phi)
    for count,(key,weight) in enumerate(quad.items(),1):
        filename=out/f'b{key[0]:.10f}_g{key[1]:.10f}.npz'
        if filename.exists():
            with np.load(filename) as data:
                dj=data['delta_jones']
            meta=json.loads(filename.with_suffix('.json').read_text())
        else:
            parameters=Parameters(20,beta_deg=key[0],gamma_deg=key[1],grid=args.grid,
                                  steps=args.steps,precision='fp32')
            x,delta,meta=exit_field_difference_vector(parameters,backend='cuda',evanescent=args.evanescent)
            dj=radiator(delta,meta['propagation_length'])
            np.savez_compressed(filename,delta_jones=dj)
            atomic_json(filename.with_suffix('.json'),meta)
        metadata.append(meta)
        native=np.load(case_path(key)/'native.npz')['jones']
        corrected=native+dj
        intensity=200*np.sum(abs(corrected)**2,axis=(-2,-1))
        mean+=weight*intensity
        scattering_basis=corrected@rotation
        scattering_basis[0]=pole_left@scattering_basis[0]
        mean_mueller+=weight*400*mueller_from_jones(scattering_basis)
        if count%40==0 or count==len(quad):
            progress={'complete':False,'label':args.label,'completed':count,'total':len(quad),
                      'elapsed_seconds':time.perf_counter()-started}
            atomic_json(out/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    # Mirror-related particle orientations contribute D M D, D=diag(1,1,-1,-1)
    # in the scattering basis. Folding the orientation quadrature is therefore
    # exact for the two diagonal 2x2 blocks; the cross blocks cancel.
    parity=np.array([1,1,-1,-1])
    average=mean_mueller.mean(axis=1)*(1+parity[:,None]*parity[None,:])/2
    np.testing.assert_allclose(average[:,0,0],mean.mean(axis=1),rtol=1e-12)
    np.savez_compressed(out/f'mean_{args.label}.npz',theta_deg=THETA,
                        M11=mean.mean(axis=1),M11_phi32=mean[:,::2].mean(axis=1),
                        M11_phi16=mean[:,::4].mean(axis=1),mueller_native=average)
    report={'complete':True,'label':args.label,'orientations_after_D6h':len(quad),'phi_points':len(PHI),
            'elapsed_seconds':time.perf_counter()-started,'grid':args.grid,'steps':args.steps,
            'evanescent':args.evanescent,'forward_M11':float(mean[0].mean()),
            'max_azimuth32_vs64_relative_change':float(np.max(abs(mean[:,::2].mean(axis=1)/mean.mean(axis=1)-1))),
            'median_propagation_seconds':float(np.median([m['propagation_s'] for m in metadata])),
            'max_background_divergence_relative':max(m['background_spectral_divergence_relative'] for m in metadata),
            'longitudinal_power_fraction_range':[min(m['longitudinal_exit_power_fraction'] for m in metadata),
                                                  max(m['longitudinal_exit_power_fraction'] for m in metadata)],
            'native_amplitudes_reused':True,'physical_accuracy_certified':False}
    atomic_json(out/f'summary_{args.label}.json',report)
    atomic_json(out/'progress.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
