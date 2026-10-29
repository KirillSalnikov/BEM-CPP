#!/usr/bin/env python3
"""Adaptive local T(ka) surrogate, independent holdouts and analytic SO(3).

The absolute deadline is inherited from an existing run, never reset here.
All direct training, validation and mesh controls count toward that budget.
The production backend is pinned per run. Audited runtime upgrades preserve
sample identities while allowing an explicitly validated kernel replacement.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import numpy as np
from bounded_bem_run import atomic_json, run
from run_size_series import FILES, BINARIES, read, iso, profile, compare, EndSeries
from study_tmatrix_cutoff import cutoff_indices
from tmatrix_size_interpolation import RationalMatrix, fit_aaa, validate_family
from tmatrix_so3_mstm import analytic_mueller_tmatrix

ROOT=Path(__file__).resolve().parents[1]
THETA=np.linspace(0,180,1801)


def stages(lo,hi):
    if not np.isfinite([lo,hi]).all() or not 0<lo<hi:
        raise ValueError('positive increasing size interval required')
    width=hi-lo
    return [
        {'training':np.linspace(lo,hi,5).tolist(),
         'validation':(lo+width*np.array([1,3,5,7])/8).tolist()},
        {'training':np.linspace(lo,hi,9).tolist(),
         'validation':(lo+width*np.array([1,7,9,15])/16).tolist()}]


def sample_path(out,ka,ref,order):
    return out/f'ka{ka:.8f}_ref{ref}_L{order}'


def check_lower_control(lower,plan):
    if not (lower.get('converged') and lower['ka']==plan['ka_min'] and
            lower['ri']==plan['ri'] and lower['ref']==plan['ref']+1 and
            lower['wavelength_um']==plan['wavelength_um']):
        raise ValueError('missing or incompatible lower endpoint mesh control')


def family_parameters(report,plan):
    """Compare physical definitions across an explicitly audited binary upgrade.

    Original manifests and their signatures remain untouched. Only the two
    approved executable identities and the tested parity switch normalize;
    all other numerical, geometry and environment differences still fail.
    """
    p=dict(report['parameters'])
    upgrade=plan.get('runtime_upgrade')
    if not upgrade:return p
    old=upgrade['canonical_binary_digest'];new=upgrade['optimized_binary_digest']
    digest=p.get('binary_digest');env=dict(p.get('BEM_environment',{}))
    flag=env.pop('BEM_FMM_MULTILEVEL_PARITY',None)
    if digest==old:
        if flag not in (None,'0'):raise ValueError('unexpected parity setting in original sample')
    elif digest==new:
        if flag!='1':raise ValueError('optimized sample must explicitly enable validated parity')
    else:raise ValueError('sample executable is not approved by runtime upgrade')
    p.update(binary_digest=old,BEM_environment=env)
    return p


def load_sample(out,plan,ka,ref=None):
    ref=plan['ref'] if ref is None else ref
    path=sample_path(out,ka,ref,plan['order'])
    report=read(path/'summary.json')
    if not report.get('complete'):raise ValueError(f'{path}: incomplete T matrix')
    p=report['parameters']
    if (p['ka']!=ka or p['ref']!=ref or p['order']!=plan['order'] or
        p['ri']!=plan['ri'] or p['ri_imag']!=0 or p['geometry']!='prism'):
        raise ValueError('sample does not belong to this physical family')
    if report['operator'].get('operator_backend')!='fmm':
        raise ValueError('unvalidated operator backend in size sweep')
    family_parameters(report,plan)
    t=np.load(path/'tmatrix.npy',mmap_mode='r')
    q=2*plan['order']*(plan['order']+2)
    if t.shape!=(q,q):raise ValueError('wrong T basis dimensions')
    return t,report


def average(t,ka,order):
    mean,meta=analytic_mueller_tmatrix(t,order,THETA,precision='fp64',threads=4,cyclic_order=6)
    if meta['Euler_quadrature_nodes']!=0:raise RuntimeError('orientation averaging must be analytic')
    meta.update(Csca=float(2*np.pi/ka**2*np.vdot(t,t).real),
                Cext=float(-2*np.pi/ka**2*np.trace(t).real))
    return mean,meta


def fit_stage(out,plan,stage):
    dest=out/f'model_{stage}';dest.mkdir(exist_ok=True)
    nodes=plan['stages'][stage]['training']
    samples=[load_sample(out,plan,x) for x in nodes]
    validate_family([family_parameters(s,plan) for _,s in samples])
    start=time.monotonic()
    model,diag=fit_aaa(nodes,[t for t,_ in samples],tolerance=1e-6)
    atomic_json(dest/'model.json',{'training':nodes,'validation':plan['stages'][stage]['validation'],
        'support_indices':diag['support_indices'],'weights_real':model.weights.real.tolist(),
        'weights_imag':model.weights.imag.tolist(),'domain':list(model.domain),
        'diagnostics':diag,'fit_s':time.monotonic()-start,
        'training_sha256':[hashlib.sha256((sample_path(out,x,plan['ref'],plan['order'])/'tmatrix.npy').read_bytes()).hexdigest() for x in nodes],
        'frozen_before_validation':True,'created_utc':iso(time.time())})


def load_model(out,plan,stage):
    saved=read(out/f'model_{stage}'/'model.json')
    if not saved:raise RuntimeError('model must be frozen before evaluating holdouts')
    samples=[load_sample(out,plan,x) for x in saved['training']]
    validate_family([family_parameters(s,plan) for _,s in samples])
    for x,expected in zip(saved['training'],saved['training_sha256']):
        path=sample_path(out,x,plan['ref'],plan['order'])/'tmatrix.npy'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('training matrix changed after the fit')
    indices=saved['support_indices']
    model=RationalMatrix(np.array(saved['training'])[indices],tuple(samples[i][0] for i in indices),
        np.array(saved['weights_real'])+1j*np.array(saved['weights_imag']),tuple(saved['domain']))
    return model,family_parameters(samples[0][1],plan)


def validate_stage(out,plan,stage):
    model,parameters=load_model(out,plan,stage)
    records=[];dest=out/f'model_{stage}'
    for ka in plan['stages'][stage]['validation']:
        direct,report=load_sample(out,plan,ka)
        validate_family([parameters,family_parameters(report,plan)])
        pred=model.evaluate(ka)
        a,ac=average(pred,ka,plan['order']);b,bc=average(direct,ka,plan['order'])
        record={'ka':ka,**compare(a,b,ac,bc,THETA),
            'T_relative_error':float(np.linalg.norm(pred-direct)/np.linalg.norm(direct))}
        records.append(record)
        np.savez_compressed(dest/f'validation_ka{ka:.8f}.npz',theta_deg=THETA,predicted=a,direct=b)
    worst=max(r['maximum'] for r in records)
    atomic_json(dest/'validation.json',{'records':records,'maximum':worst,
        'passes_interpolation_budget':bool(worst<=plan['interpolation_budget']),
        'Euler_quadrature_nodes':0,'theta_step_deg':.1})


def controls(out,plan,stage):
    hi=plan['ka_max'];coarse,_=load_sample(out,plan,hi)
    fine,_=load_sample(out,plan,hi,plan['ref']+1)
    a,ac=average(coarse,hi,plan['order']);b,bc=average(fine,hi,plan['order'])
    mesh_high=compare(a,b,ac,bc,THETA)
    # A completed independent adjacent-mesh audit at the lower endpoint is
    # retained as evidence, not mixed into the common-basis training family.
    lower=read(out/'lower_control.json')
    check_lower_control(lower,plan)
    mesh_low=lower['mesh_control']
    tails=[];indices=cutoff_indices(plan['order'],plan['order']-4)
    nodes=sorted(set(plan['stages'][stage]['training']+plan['stages'][stage]['validation']))
    for ka in nodes:
        t,_=load_sample(out,plan,ka)
        a,ac=average(t[np.ix_(indices,indices)],ka,plan['order']-4)
        b,bc=average(t,ka,plan['order'])
        tails.append({'ka':ka,**compare(a,b,ac,bc,THETA)})
    interpolation=read(out/f'model_{stage}'/'validation.json')['maximum']
    mesh=max(mesh_low['maximum'],mesh_high['maximum'])
    tail=max(x['maximum'] for x in tails)
    # This is an empirical error budget. Separate finite-sample controls do
    # not constitute a uniform bound over a continuous size/angle interval.
    budget=interpolation+mesh+tail
    atomic_json(out/f'model_{stage}'/'controls.json',{
        'lower_mesh_control':mesh_low,'upper_mesh_control':mesh_high,
        'modal_tail_controls':tails,'interpolation_maximum':interpolation,
        'mesh_maximum':mesh,'tail_maximum':tail,'empirical_budget_sum':budget,
        'passes_1pct_controls':bool(interpolation<=plan['interpolation_budget'] and budget<=.01),
        'limitation':'Mesh controls at endpoints, modal controls at direct sizes, interpolation at independent holdouts; no uniform or absolute error certificate.'})


def produce_grid(out,plan,stage):
    checks=read(out/f'model_{stage}'/'controls.json')
    if not checks.get('passes_1pct_controls'):raise RuntimeError('size grid blocked by failed accuracy controls')
    model,_=load_model(out,plan,stage)
    sizes=np.linspace(plan['ka_min'],plan['ka_max'],201)
    mean=[];csca=[];cext=[];begin=time.monotonic()
    for ka in sizes:
        a,meta=average(model.evaluate(ka),ka,plan['order'])
        mean.append(a*(plan['wavelength_um']/(2*np.pi))**2)
        radius=ka*plan['wavelength_um']/(2*np.pi)
        csca.append(meta['Csca']*radius**2);cext.append(meta['Cext']*radius**2)
    temporary=out/'size_grid.tmp.npz'
    np.savez_compressed(temporary,ka=sizes,equal_volume_radius_um=sizes*plan['wavelength_um']/(2*np.pi),
        theta_deg=THETA,mueller_um2_per_sr=np.array(mean),Csca_um2=np.array(csca),Cext_um2=np.array(cext))
    temporary.replace(out/'size_grid.npz')
    atomic_json(out/'results.json',{'state':'complete','ka_range':[plan['ka_min'],plan['ka_max']],
        'sizes':len(sizes),'stage':stage,'ri':plan['ri'],'wavelength_um':plan['wavelength_um'],
        'grid_evaluation_s':time.monotonic()-begin,'additional_BEM_solves_for_grid':0,
        'Euler_quadrature_nodes':0,'checks':checks,'operator_backend':'fmm',
        'size_approximation':'shared-denominator rational interpolation of full complex T',
        'requested_range':[10,40],'full_requested_range_achieved':False})


def freeze(destination):
    if not destination.exists():
        temp=destination.with_name(destination.name+'.preparing');temp.mkdir(exist_ok=True)
        files=tuple(FILES)+('run_tmatrix_multisize.py','tmatrix_size_interpolation.py')
        for directory,names in [('scripts',files),('bin',BINARIES)]:
            (temp/directory).mkdir(exist_ok=True)
            for name in names:shutil.copy2(ROOT/directory/name,temp/directory/name)
        atomic_json(temp/'sha256.json',{str(p.relative_to(temp)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in temp.rglob('*') if p.is_file() and p.name!='sha256.json'})
        temp.rename(destination)
    expected=read(destination/'sha256.json')
    if not expected:raise RuntimeError('missing frozen runtime identity')
    for name,digest in expected.items():
        if hashlib.sha256((destination/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('frozen runtime changed: '+name)


class Sweep:
    def __init__(self,out,plan):
        self.out,self.plan=out,plan;self.active={};self.done=threading.Event()
        self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()

    def remaining(self):return max(0.,self.plan['deadline_unix']-time.time())

    def status(self,state,**extra):
        atomic_json(self.out/'status.json',{'state':state,'updated_utc':iso(time.time()),
            'deadline_utc':iso(self.plan['deadline_unix']),'remaining_s':self.remaining(),
            'supervisor_pid':os.getpid(),'active':dict(self.active),**extra})

    def monitor(self):
        while not self.done.wait(10):
            active=dict(self.active)
            progress=read(Path(active.get('run',self.out))/'progress.json')
            last=progress.get('last_pair',{})
            rate=last.get('total_worker_s',0)+last.get('incident_field_s',0)
            atomic_json(self.out/'eta.json',{'updated_utc':iso(time.time()),'remaining_budget_s':self.remaining(),
                'active':active,'progress':progress,
                'current_T_remaining_estimate_s':rate*progress.get('remaining_worker_pairs',0)})

    def bounded(self,command,supervisor,env):
        if self.remaining()<90:raise EndSeries('original deadline reserve reached')
        if self.plan.get('runtime_upgrade'):
            supervisor=self.out/self.plan['runtime_upgrade']['supervisor_directory']/supervisor.name
        state=run(command,supervisor,env=env,rss_limit_gib=44,available_gib=8,
            disk_reserve_gib=80,wall_limit_s=self.remaining()-45)
        if state['state']!='complete':
            if state['state']=='time_limit' or str(state.get('reason')).startswith('signal'):
                raise EndSeries(str(state.get('reason')))
            raise RuntimeError(str(state))

    def environment(self):
        env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='1',
                   BEM_SO3_TMPDIR=str(self.out/'scratch'))
        return env

    def build(self,ka,ref):
        dest=sample_path(self.out,ka,ref,self.plan['order'])
        if read(dest/'summary.json').get('complete'):
            load_sample(self.out,self.plan,ka,ref);return
        self.active={'phase':'direct_T','ka':ka,'ref':ref,'order':self.plan['order'],'run':str(dest)}
        self.status('running')
        env=self.environment();env.update(profile(ref),BEM_TMATRIX_FACE_FFT='0')
        if self.plan.get('runtime_upgrade'):env['BEM_FMM_MULTILEVEL_PARITY']='1'
        command=[sys.executable,str(ROOT/'scripts/build_fmm_tmatrix.py'),'--shape','prism',
            '--ref',str(ref),'--ka',str(ka),'--ri',str(self.plan['ri']),'--ri-imag','0',
            '--order',str(self.plan['order']),'--tol','2e-6','--restart','80','--max-iters','600',
            '--c2-reuse','--c6-pack','--mirror-pack','--fast-mlfma','--no-average','--out',str(dest)]
        self.bounded(command,self.out/(dest.name+'_supervisor'),env)
        load_sample(self.out,self.plan,ka,ref)
        # A measured estimate includes all solves, not merely AAA support nodes.
        times=[]
        for ka0 in sorted(set(sum([s['training']+s['validation'] for s in self.plan['stages']],[]))):
            report=read(sample_path(self.out,ka0,self.plan['ref'],self.plan['order'])/'summary.json')
            if report.get('complete'):times.append(report['end_to_end_s'])
        if times:
            atomic_json(self.out/'forecast.json',{'completed_coarse_samples':len(times),
                'median_completed_T_s':float(np.median(times)),
                'stage0_remaining_coarse_estimate_s':max(0,9-len(times))*float(np.median(times)),
                'stage1_remaining_coarse_estimate_s':max(0,13-len(times))*float(np.median(times)),
                'upper_mesh_control_estimate_s':5*float(np.median(times)),
                'remaining_budget_s':self.remaining(),
                'limitation':'Timing estimate, not a convergence or completion guarantee; resumed summaries omit earlier process lifetimes.'})

    def step(self,name,stage):
        self.active={'phase':name,'stage':stage};self.status('running')
        command=[sys.executable,str(Path(__file__).resolve()),'--out',str(self.out),
                 '--step',name,'--stage',str(stage)]
        self.bounded(command,self.out/f'{name}_{stage}_supervisor',self.environment())

    def execute(self):
        for stage,definition in enumerate(self.plan['stages']):
            # Measure the most expensive endpoint first. Stage 1 reuses the
            # previous holdouts as training; all its new holdouts are disjoint.
            for ka in reversed(definition['training']):self.build(ka,self.plan['ref'])
            self.step('fit',stage)
            for ka in definition['validation']:self.build(ka,self.plan['ref'])
            self.step('validate',stage)
            result=read(self.out/f'model_{stage}'/'validation.json')
            if not result['passes_interpolation_budget']:continue
            self.build(self.plan['ka_max'],self.plan['ref']+1)
            self.step('controls',stage)
            if not read(self.out/f'model_{stage}'/'controls.json')['passes_1pct_controls']:
                self.status('accuracy_controls_failed',stage=stage);return
            self.step('grid',stage)
            self.status('complete',stage=stage,results=str(self.out/'results.json'));return
        self.status('interpolation_controls_failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--deadline-from',type=Path)
    parser.add_argument('--ka-min',type=float,default=10.)
    parser.add_argument('--ka-max',type=float,default=12.)
    parser.add_argument('--order',type=int,default=24)
    parser.add_argument('--ref',type=int,default=4)
    parser.add_argument('--ri',type=float,default=1.3116)
    parser.add_argument('--wavelength-um',type=float,default=.532)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--runtime-name',default='runtime',help='Separate immutable runtime directory for an audited upgrade')
    parser.add_argument('--step',choices=['fit','validate','controls','grid'],help=argparse.SUPPRESS)
    parser.add_argument('--stage',type=int,default=0,help=argparse.SUPPRESS)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.step:
        plan=read(out/'plan.json')
        if not plan or args.stage not in (0,1):raise ValueError('missing plan or invalid stage')
        {'fit':fit_stage,'validate':validate_stage,'controls':controls,'grid':produce_grid}[args.step](out,plan,args.stage)
        return
    if Path(args.runtime_name).name!=args.runtime_name or args.runtime_name in ('.','..'):
        parser.error('runtime-name must be a directory name')
    runtime=out/args.runtime_name;frozen=runtime/'scripts'/Path(__file__).name
    if Path(__file__).resolve()!=frozen:
        freeze(runtime);os.execv(sys.executable,[sys.executable,str(frozen),*sys.argv[1:]])
    freeze(runtime)
    lock=(out/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config={'ka_min':args.ka_min,'ka_max':args.ka_max,'order':args.order,'ref':args.ref,
            'ri':args.ri,'wavelength_um':args.wavelength_um}
    if not 3<=args.ref<=5 or args.order<8:parser.error('use ref=3..5 and order>=8')
    if not np.isfinite([args.ri,args.wavelength_um]).all() or min(args.ri,args.wavelength_um)<=0:
        parser.error('positive finite material and wavelength required')
    plan=read(out/'plan.json')
    if plan and plan['configuration']!=config:raise RuntimeError('resume configuration changed')
    if not plan:
        if args.deadline_from is None:parser.error('--deadline-from must identify the original plan.json')
        original=read(args.deadline_from)
        if not original or original['deadline_unix']<=time.time():raise EndSeries('original deadline has expired')
        lower=read(args.deadline_from.parent/f'ka{args.ka_min:g}_assessment.json')
        check_lower_control(lower,config)
        atomic_json(out/'lower_control.json',lower)
        plan={'configuration':config,**config,'stages':stages(args.ka_min,args.ka_max),
            'deadline_unix':original['deadline_unix'],'deadline_utc':iso(original['deadline_unix']),
            'original_plan':str(args.deadline_from.resolve()),'created_utc':iso(time.time()),
            'interpolation_budget':.004,'total_empirical_budget':.01,
            'operator_backend':'fmm','face_fft_enabled':False,
            'FFT_decision':'Complete hybrid operator validated but slower on ref=1,3,4; not selected.',
            'orientation_average':'analytic full SO(3); zero Euler quadrature nodes',
            'shape':'regular hexagonal prism, h/D_flats=1, equal-volume radius a',
            'training_and_validation_costs_included':True}
        atomic_json(out/'plan.json',plan)
    if args.prepare_only:print(json.dumps(plan,indent=2));return
    (out/'supervisor.pid').write_text(str(os.getpid())+'\n');sweep=Sweep(out,plan)
    try:sweep.execute()
    except EndSeries as exc:sweep.status('stopped',reason=str(exc))
    except Exception as exc:
        sweep.status('failed',reason=str(exc));raise
    finally:sweep.done.set();sweep.thread.join()


if __name__=='__main__':main()
