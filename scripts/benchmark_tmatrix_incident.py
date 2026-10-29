#!/usr/bin/env python3
"""CPU comparison of incident-field preparation with identical phase codes."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import scipy
import tmatrix_incident as incident
from tmatrix_c6 import encode_fields
from tmatrix_reference import magnetic_angular_components, sph_harm_y


def legacy_angular(theta, phi, l, m):
    """The previous two-neighbor derivative, retained only for benchmarking."""
    y = sph_harm_y(l, m, theta, phi)
    derivative = np.zeros_like(y)
    if m < l:
        derivative += .5*np.sqrt((l-m)*(l+m+1))*np.exp(-1j*phi)*sph_harm_y(l,m+1,theta,phi)
    if m > -l:
        derivative -= .5*np.sqrt((l+m)*(l-m+1))*np.exp(1j*phi)*sph_harm_y(l,m-1,theta,phi)
    sine = np.sin(theta)
    pole = np.abs(sine)<1e-10
    xt = -m*y/(np.sqrt(l*(l+1))*np.where(pole,1.,sine))
    xp = -1j*derivative/np.sqrt(l*(l+1))
    xt[pole]=0;xp[pole]=0
    if abs(m)==1 and pole.any():
        south=theta[pole]>np.pi/2
        value=np.sqrt((2*l+1)/(16*np.pi))*np.exp(1j*m*phi[pole])*np.where(south,(-1)**(l+m),1)
        xt[pole]=value;xp[pole]=1j*m*value*np.where(south,-1,1)
    return y,xt,xp


def prism_samples(count):
    """Deterministic samples on a hexagonal prism, H/D_flats=1, volume=4pi/3.

    They exercise the incident fields, not mesh quadrature or BEM scattering.
    """
    rng=np.random.default_rng(205)
    width=(8*np.pi/(3*np.sqrt(3)))**(1/3)
    angle=np.arange(6)*np.pi/3
    corners=np.column_stack((np.cos(angle),np.sin(angle)))*width/np.sqrt(3)
    side=rng.integers(0,6,count);u=rng.random(count)
    xy=(1-u[:,None])*corners[side]+u[:,None]*corners[(side+1)%6]
    z=width*(rng.random(count)-.5)
    cap=np.arange(count)%3==0
    xy[cap]*=np.sqrt(rng.random(cap.sum()))[:,None]
    z[cap]=rng.choice([-width/2,width/2],cap.sum())
    return np.column_stack((xy,z))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points',type=int,default=65536)
    parser.add_argument('--degree',type=int,default=149)
    parser.add_argument('--ka',type=float,default=112.)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.points<1 or args.degree<1 or args.repeats<1:parser.error('positive dimensions required')
    positions=prism_samples(args.points)
    starts=sorted({0,args.degree//2,max(0,args.degree-5)})
    packets=[[(p,args.degree,m) for m in range(start,min(start+6,args.degree+1)) for p in (0,1)] for start in starts]
    fields=np.empty((2,len(positions),6),complex)
    references={};rows=[]
    original=incident.magnetic_angular_components
    try:
        for trial in range(args.repeats):
            variants=[('legacy',legacy_angular),('one_neighbor',magnetic_angular_components)]
            if trial%2:variants.reverse()
            for name,function in variants:
                incident.magnetic_angular_components=function
                cache=incident.RegularFieldCache(positions,args.ka)
                for packet,modal in enumerate(packets):
                    start=time.perf_counter()
                    scales,combined=encode_fields(positions,modal,list(range(len(modal))),args.ka,
                        fields,reflection=True,field_cache=cache)
                    elapsed=time.perf_counter()-start
                    if trial==0 and name=='legacy':references[packet]=(fields.copy(),scales.copy(),combined.copy())
                    reference=references[packet]
                    errors=[float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
                        for a,b in zip((fields,scales,combined),reference)]
                    if max(errors)>2e-11:raise RuntimeError(f'encoded incident fields changed: {errors}')
                    row={'trial':trial,'variant':name,'packet_m_start':starts[packet],
                        'seconds':elapsed,'relative_field_error':errors[0],
                        'relative_scale_error':errors[1],'relative_combined_scale_error':errors[2]}
                    rows.append(row);print(json.dumps(row),flush=True)
    finally:
        incident.magnetic_angular_components=original
    summary=[]
    for start in starts:
        times={name:statistics.median(r['seconds'] for r in rows
                    if r['variant']==name and r['packet_m_start']==start)
               for name in ('legacy','one_neighbor')}
        summary.append({'packet_m_start':start,'median_seconds':times,
                        'preparation_speedup':times['legacy']/times['one_neighbor']})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'parameters':{**vars(args),'out':str(args.out)},
        'scipy_version':scipy.__version__,'numpy_version':np.__version__,
        'includes_radial_refresh_in_first_packet':True,'includes_two_phase_codes':True,
        'includes_solver_or_averaging':False,'summaries':summary,'measurements':rows},indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
