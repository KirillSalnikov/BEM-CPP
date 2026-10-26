#!/usr/bin/env python3
"""Resumable maximum-ka search with operator, memory, mesh and angular gates.

The ceiling is a stated resolution convention, not a universal BEM limit.
Surface convergence is checked independently at three incident directions.
Only accepted mesh cases proceed to adaptive orientation averaging.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from bounded_bem_run import run, atomic_json

ROOT=Path(__file__).resolve().parents[1]


def mesh_difference(coarse_directory, fine_directory):
    results=[]
    for part in sorted((fine_directory/'parts').glob('part_*.json')):
        fine=json.loads(part.read_text())
        coarse=json.loads((coarse_directory/'parts'/part.name).read_text())
        theta=np.radians(fine['theta_degrees'])
        a=np.asarray(coarse['mueller']).reshape(16,-1)
        b=np.asarray(fine['mueller']).reshape(16,-1)
        weights=np.sin(theta)
        weights[0]=weights[-1]=0
        numerator=np.sum((a-b)**2*weights[None,:])
        denominator=np.sum(b*b*weights[None,:])
        delta={'beta_degrees':fine['beta_degrees'],'gamma_degrees':fine['gamma_degrees'],
               'full_mueller_weighted_l2':float(np.sqrt(numerator/max(denominator,1e-300))),
               'forward_m11_relative':float(abs(a[0,0]-b[0,0])/max(abs(b[0,0]),1e-300)),
               'maximum_component_error_over_m11_peak':float(np.max(abs(a-b))/max(np.max(abs(b[0])),1e-300))}
        results.append(delta)
    if len(results)!=3:
        raise RuntimeError('mesh convergence requires all three independent incident directions')
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ka',type=float,default=112.)
    parser.add_argument('--ref',type=int,default=7)
    parser.add_argument('--mesh-tol',type=float,default=.01)
    parser.add_argument('--max-angular-level',type=int,default=5)
    parser.add_argument('--validation-only',action='store_true',default=True)
    parser.add_argument('--with-numerical-orientation-average',dest='validation_only',action='store_false',
                        help='Legacy alpha-only method; full analytic SO3 uses build_fmm_tmatrix.py')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'study.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    binary=out/'muller_solver'
    source=ROOT/'bin/muller_nodal_fmm_demo_fp32'
    if not binary.exists(): shutil.copy2(source,binary)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',
               BEM_FMM_PAIR_CURRENTS='0',BEM_MULLER_GPU_ASSEMBLY='0',
               BEM_FMM_PHASE_CACHE='0',BEM_FMM_FLAT_NEAR_SOURCES='0',
               BEM_FMM_STRICT_PAIR_WORKSPACE='0',BEM_FMM_L2P_FP32='0',
               BEM_MIXED_ITERATIVE_REFINEMENT='1',BEM_FMM_BANDED_MINIMAL_WORKSPACE='1',
               BEM_FMM_CONCURRENT_MEDIA='1',BEM_FARFIELD_ANALYTIC_ALPHA='1',
               BEM_FARFIELD_SPECTRAL_ALPHA='auto')
    probe=subprocess_output([str(ROOT/'bin/mesh_capacity_probe'),'1','8','1.3'])
    meshes={d['ref']:d for d in map(json.loads,probe.splitlines())}
    ceiling=meshes[args.ref]['ka_at_8_equivalent_P2_points_per_inner_wavelength']
    if args.ref not in range(2,8) or args.ka>ceiling or args.ka<=0:
        raise RuntimeError('candidate violates the measured 8-point mesh ceiling or supported resource plan')
    disk_free=shutil.disk_usage(out).free/2**30
    radius=meshes[args.ref]['bounding_radius']
    kr=args.ka*radius
    order=math.ceil(kr+4*kr**(1/3)+2)
    modes=2*order*(order+2)
    disk_plan={'free_gib':disk_free,'reserve_gib':40,'native_study_estimate_gib':120,
               'estimated_t_order':order,'estimated_t_modes':modes,
               'one_complex128_t_gib':modes*modes*16/2**30,
               't_checkpoints_plus_kernel_plus_t_gib':3*modes*modes*16/2**30,
               'dense_projection_file_gib':modes*4*(2*order+4)**2*16/2**30,
               'fft_projection_file_gib':0,
               't_build_started':False,
               't_order_rule_is_initial_estimate':True}
    plan={'shape':'regular_hexagonal_prism','aspect':1,'ri':1.3,'target_ka':args.ka,
          'target_ref':args.ref,'mesh_resolution_convention':'8 equivalent P2 points per shortest wavelength',
          'mesh_tolerance':args.mesh_tol,'true_residual_ceiling':1e-5,
          'C6_commutator_ceiling':1e-4,'mesh_probes':list(meshes.values()),'disk':disk_plan,
          'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
          'notes':['A C6 check is necessary here, but is not a general bound on FMM error.',
                   'A ref-to-ref difference estimates self convergence, not exact physical error.',
                   'Full analytic SO3 is implemented by the MSTM backend in build_fmm_tmatrix.py; this script validates mesh resolution.',
                   'Ref 8 is not launched: its fourfold size exceeds the measured memory envelope of this implementation.']}
    atomic_json(out/'plan.json',plan)
    if args.dry_run:
        print(json.dumps(plan,indent=2)); return 0
    failures=[]

    def stage(ka,ref,name,extra,layout='three_band',env_overrides=None):
        directory=out/f'ka{ka:g}_ref{ref}_{layout}'/name
        directory.mkdir(parents=True,exist_ok=True)
        cache=out/f'ka{ka:g}_ref{ref}'
        combined_points=2*meshes[ref]['triangles']*7
        leaf=230 if ref>=7 else 64
        depth=math.ceil(math.log(combined_points/leaf,8))
        split=max(1,depth-2)
        coarse_leaf=2**math.ceil(math.log2(combined_points/8**split))
        middle_leaf=2**math.ceil(math.log2(combined_points/8**(split+1)))
        if layout=='two_band': leaf=middle_leaf
        local=env.copy();local.update(BEM_FMM_BANDED_SPLIT_DEPTH=str(split),
             BEM_FMM_BANDED_COARSE_MAX_LEAF=str(coarse_leaf),
             BEM_FMM_BANDED_MIDDLE_MAX_LEAF=str(middle_leaf))
        kr_local=ka*radius
        alpha_samples=16*math.ceil((2*(kr_local+4*kr_local**(1/3)+2)+8)/16)
        local['BEM_FARFIELD_SPECTRAL_ALPHA']=str(alpha_samples)
        if env_overrides: local.update(env_overrides)
        ntheta=max(181,int(math.ceil(ka*radius*8)))
        if ntheta%2==0: ntheta+=1
        command=[str(binary),'--shape','prism','--sides','6','--aspect','1','--ref',str(ref),
                 '--ka',str(ka),'--ri','1.3','--edge-mode','hdiv','--quad','7','--duffy-order','4',
                 '--digits','5','--max-leaf',str(leaf),'--fmm-near-radius','3','--tol','5e-6',
                 '--max-iters','1200','--gmres-restart','100','--mbj-only','--mbj-nodes','12',
                 '--no-dense-validation','--near-correction-cache',str(cache)+'.near',
                 '--mbj-cache',str(cache)+'.mbj','--out',str(directory/'result.json'),
                 '--ntheta',str(ntheta),'--iteration-log',str(directory/'iterations.csv'),
                 '--checkpoint',str(directory/'checkpoint'),*extra]
        atomic_json(out/'status.json',{'state':'running','supervisor_pid':os.getpid(),'ka':ka,
                    'ref':ref,'stage':name,'layout':layout,'directory':str(directory),
                    'elapsed_s':time.time()-started,'failures':failures})
        print(f'ka={ka:g}, ref={ref}, {layout}, {name}',flush=True)
        status=run(command,directory,local)
        if status.get('reason','') and status['reason'].startswith('signal'):
            raise InterruptedError(status['reason'])
        if status['state']!='complete':
            raise RuntimeError(f'{name} failed: {status.get("reason") or status["returncode"]}')
        return directory,json.loads((directory/'result.json').read_text())

    accepted=None
    # Decrease ka only after an actual resource/operator/mesh failure. Each
    # tested candidate is recorded; this does not assert a mathematical max.
    candidates=list(dict.fromkeys([args.ka,*[float(k) for k in (104,96,88,80,72,64,56,48,40) if k<args.ka]]))
    orientations=out/'mesh_incidents.txt'
    orientations.write_text('0 0 0 0.3333333333333333\n1 57 17 0.3333333333333333\n2 90 30 0.3333333333333333\n')
    for ka in candidates:
        fine_ref=args.ref
        if fine_ref==6 and ka>meshes[6]['ka_at_8_equivalent_P2_points_per_inner_wavelength']: continue
        for layout in ('three_band','two_band'):
            try:
                controls={}
                for ref in (fine_ref,fine_ref-1):
                    stage(ka,ref,'setup',['--setup-only'],layout)
                    _,diagnostic=stage(ka,ref,'symmetry',[
                        '--physical-check','--cyclic-polarization','--symmetry-operator-check-only'],layout)
                    error=diagnostic['fmm_relative_commutator']
                    if not np.isfinite(error) or error>1e-4:
                        raise RuntimeError(f'operator C6 error {error} exceeds 1e-4')
                incident_files=[]
                for index,(beta,gamma) in enumerate(((0,0),(57,17),(90,30))):
                    incident=out/f'incident_{index}.txt'
                    incident.write_text(f'{index} {beta} {gamma} 1\n')
                    incident_files.append(incident)
                for ref in (fine_ref-1,fine_ref):
                    directory=out/f'ka{ka:g}_ref{ref}_{layout}'/'mesh'
                    (directory/'parts').mkdir(parents=True,exist_ok=True)
                    for index,incident in enumerate(incident_files):
                        path,physical=stage(ka,ref,f'mesh_{index}',[
                            '--orient-average','4096','1','1','--orient-file',str(incident),
                            '--orient-parts-dir',str(directory/'parts')],layout)
                        residual=physical['iterations']['maximum_residual']
                        if not np.isfinite(residual) or residual>1e-5:
                            raise RuntimeError(f'true residual {residual} exceeds 1e-5')
                    controls[ref]=directory
                differences=mesh_difference(controls[fine_ref-1],controls[fine_ref])
                passed=all(max(d['full_mueller_weighted_l2'],d['forward_m11_relative'],
                               d['maximum_component_error_over_m11_peak'])<=args.mesh_tol for d in differences)
                atomic_json(out/f'mesh_ka{ka:g}_{layout}.json',{'ka':ka,'fine_ref':fine_ref,
                            'passed':passed,'directions':differences})
                if not passed: raise RuntimeError('surface convergence target not reached')
                # Double the azimuthal spectrum on the same converged
                # currents, independently of the surface-refinement test.
                alpha_directory=out/f'ka{ka:g}_ref{fine_ref}_{layout}'/'alpha_check'
                (alpha_directory/'parts').mkdir(parents=True,exist_ok=True)
                kr_local=ka*radius
                alpha_samples=16*math.ceil((2*(kr_local+4*kr_local**(1/3)+2)+8)/16)
                for index,incident in enumerate(incident_files):
                    checkpoint=out/f'ka{ka:g}_ref{fine_ref}_{layout}'/f'mesh_{index}'/'result.json.orient.checkpoint'
                    stage(ka,fine_ref,f'alpha_{index}',[
                        '--orient-average','4096','1','1','--orient-file',str(incident),
                        '--orient-parts-dir',str(alpha_directory/'parts')],layout,
                        {'BEM_FARFIELD_REPLAY_CHECKPOINT':str(checkpoint),
                         'BEM_FARFIELD_SPECTRAL_ALPHA':str(2*alpha_samples)})
                alpha_differences=mesh_difference(controls[fine_ref],alpha_directory)
                atomic_json(out/f'alpha_ka{ka:g}_{layout}.json',{'directions':alpha_differences})
                if any(max(d['full_mueller_weighted_l2'],d['forward_m11_relative'],
                           d['maximum_component_error_over_m11_peak'])>1e-3 for d in alpha_differences):
                    raise RuntimeError('alpha spectrum failed the 0.1% convergence control')
                accepted={'ka':ka,'ref':fine_ref,'layout':layout,'mesh_differences':differences}
                break
            except RuntimeError as error:
                failures.append({'ka':ka,'ref':fine_ref,'layout':layout,'error':str(error)})
                print(failures[-1],flush=True)
        if accepted: break
    if accepted is None:
        atomic_json(out/'status.json',{'state':'no_validated_candidate','failures':failures})
        return 1
    atomic_json(out/'accepted_mesh.json',accepted)
    if not args.validation_only:
        ka=accepted['ka'];ref=accepted['ref'];layout=accepted['layout']
        directory=out/f'ka{ka:g}_ref{ref}_{layout}'/'average'
        (directory/'parts').mkdir(parents=True,exist_ok=True)
        _,average=stage(ka,ref,'average',[
            '--orient-average','4096','1','1','--orient-adaptive','2',str(args.max_angular_level),
            '--orient-symmetry-order','6','--orient-dihedral-symmetry',
            '--orient-adaptive-m11-tol','0.01','--orient-adaptive-integral-tol','0.01',
            '--orient-adaptive-component-tol','0.01','--orient-parts-dir',str(directory/'parts')],layout)
        accepted['orientation_result']=str(directory/'result.json')
        accepted['orientation_converged']=average.get('adaptive',{}).get('converged',False)
    atomic_json(out/'status.json',{'state':('finished' if args.validation_only or accepted.get('orientation_converged') else 'angular_limit_without_convergence'),'accepted':accepted,'failures':failures,
                                  'elapsed_s':time.time()-started})
    return 0


def subprocess_output(command):
    import subprocess
    return subprocess.check_output(command,text=True,cwd=ROOT)


if __name__=='__main__':
    sys.exit(main())
