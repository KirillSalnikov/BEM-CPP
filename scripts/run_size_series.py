#!/usr/bin/env python3
"""Deadline-bounded prism size series with full T and analytic SO(3) controls.

Only completed mesh/cutoff controls can label a result converged. A partial
T is a resumable checkpoint, never an orientation average. The experimental
single-face FFT is not a replacement for the complete Muller operator.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import numpy as np
from bounded_bem_run import atomic_json, run
from run_large_tmatrix import task_counts
from study_tmatrix_cutoff import cutoff_indices, error_metrics
from tmatrix_so3_mstm import analytic_mueller_tmatrix

ROOT=Path(__file__).resolve().parents[1]
FILES=('run_size_series.py','run_large_tmatrix.py','bounded_bem_run.py','build_fmm_tmatrix.py',
       'tmatrix_reference.py','tmatrix_incident.py','tmatrix_projection.py','tmatrix_so3_mstm.py',
       'tmatrix_symmetry.py','tmatrix_c6.py','study_tmatrix_cutoff.py')
BINARIES=('muller_tmatrix_fmm_server','tmatrix_so3_mstm','tmatrix_so3_mstm_fp32')


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def iso(seconds):return datetime.fromtimestamp(seconds,timezone.utc).isoformat()


def selection(ka):
    # A starting estimate, verified by adjacent meshes and a modal tail test.
    fine=max(4,3+math.ceil(math.log2(ka/7)))
    return min(fine,6),math.ceil(1.291648635563488*ka+8)


def profile(ref):
    seg=2**ref; nz=math.ceil(math.sqrt(3)*seg);nz+=nz%2
    points=7*12*seg*(seg+nz)
    depth=max(3,math.ceil(math.log(max(1,2*points/64),8)))
    env={'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'BEM_FMM_MULTILEVEL':'1','BEM_MULLER_GPU_ASSEMBLY':'1','BEM_FMM_MULTILEVEL_GPU_ASSEMBLY':'1',
        'BEM_FMM_BANDED_SPLIT_DEPTH':str(depth-2),
        'BEM_FMM_BANDED_COARSE_MAX_LEAF':str(math.ceil(2*points/8**(depth-2))),
        'BEM_FMM_BANDED_MIDDLE_MAX_LEAF':str(math.ceil(2*points/8**(depth-1))),
        'BEM_TMATRIX_MAX_LEAF':'64','BEM_TMATRIX_MBJ_NODES':'50',
        'BEM_FMM_PAIR_CURRENTS':'0','BEM_FMM_PHASE_CACHE':'0','BEM_FMM_FLAT_NEAR_SOURCES':'0',
        'BEM_FMM_STRICT_PAIR_WORKSPACE':'0','BEM_FMM_BANDED_MINIMAL_WORKSPACE':'1',
        'BEM_FMM_CONCURRENT_MEDIA':'1','BEM_FMM_MULTILEVEL_COMPACT_TEMP':'1','BEM_FMM_P2P_SHARED_SOURCES':'1',
        'BEM_TMATRIX_D6H_MESH':'1','BEM_TMATRIX_DUFFY_ORDER':'6','BEM_MIXED_ITERATIVE_REFINEMENT':'1',
        'BEM_MIXED_ADAPTIVE_INNER':'1','BEM_MIXED_CORRECTION_FACTOR':'0.5'}
    return env


def freeze(destination):
    if not destination.exists():
        temporary=destination.with_name(destination.name+'.preparing')
        for directory,names in [('scripts',FILES),('bin',BINARIES)]:
            (temporary/directory).mkdir(parents=True,exist_ok=True)
            for name in names:shutil.copy2(ROOT/directory/name,temporary/directory/name)
        atomic_json(temporary/'sha256.json',{str(p.relative_to(temporary)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in temporary.rglob('*') if p.is_file()})
        temporary.rename(destination)
    for name,expected in read(destination/'sha256.json').items():
        if hashlib.sha256((destination/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('frozen runtime changed: '+name)


def compare(a,b,ac,bc,theta):
    result=error_metrics(a,b,theta)
    for key in ('Csca','Cext'):result[key+'_relative_error']=abs(ac[key]-bc[key])/abs(bc[key])
    result['maximum']=max(result['maximum_all16_error_over_reference_M11'],result['Csca_relative_error'],result['Cext_relative_error'])
    return result


class EndSeries(Exception):pass
class Deferred(Exception):pass


class Series:
    def __init__(self,out,plan):
        self.out,self.plan=out,plan;self.runtime=out/'runtime'
        self.deadline=plan['deadline_unix'];self.active={};self.done=threading.Event()
        self.results=read(out/'results.json').get('sizes',[])
        self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()

    def remaining(self):return max(0.,self.deadline-time.time())

    def status(self,state,**extra):
        atomic_json(self.out/'status.json',{'state':state,'updated_utc':iso(time.time()),
            'deadline_utc':iso(self.deadline),'remaining_s':self.remaining(),'supervisor_pid':os.getpid(),
            'active':self.active,'converged_ka':[r['ka'] for r in self.results if r.get('converged')],**extra})

    def monitor(self):
        while not self.done.wait(10):
            active=dict(self.active)
            p=read(Path(active.get('run',self.out))/'progress.json')
            last=p.get('last_pair',{})
            rate=last.get('total_worker_s',0)+last.get('incident_field_s',0)
            atomic_json(self.out/'eta.json',{'updated_utc':iso(time.time()),'remaining_budget_s':self.remaining(),
                'active':active,'progress':p,'last_pair_s':rate,
                'remaining_stage_estimate_s':rate*p.get('remaining_worker_pairs',0),
                'estimate_is_not_a_convergence_guarantee':True})

    def build(self,ka,ref,order):
        if self.remaining()<180:raise EndSeries('deadline reserve reached')
        name=f'ka{ka:g}_ref{ref}_L{order}';dest=self.out/name
        old=read(dest/'summary.json')
        if old.get('complete'):return dest
        q=2*order*(order+2)
        if shutil.disk_usage(self.out).free<80*2**30+4*q*q*16+8*2**30:
            raise EndSeries('disk reserve reached')
        env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
        env.update(profile(ref),BEM_SO3_TMPDIR=str(self.out/'scratch'))
        command=[sys.executable,str(self.runtime/'scripts/build_fmm_tmatrix.py'),'--shape','prism',
            '--ref',str(ref),'--ka',str(ka),'--ri',str(self.plan['ri']),'--ri-imag','0',
            '--order',str(order),'--tol','2e-6','--restart','80','--max-iters','600',
            '--c2-reuse','--c6-pack','--mirror-pack','--fast-mlfma','--no-average','--out',str(dest)]
        for phase,extra in [('pilot',['--max-pairs-this-run','4']),('full',[])]:
            self.active={'ka':ka,'ref':ref,'order':order,'phase':phase,'run':str(dest)};self.status('running')
            if phase=='full':
                summary=read(dest/'summary.json');times=[r['total_worker_s']+r.get('incident_field_s',0) for r in summary.get('pairs_this_run',[])]
                if not times:raise RuntimeError('pilot produced no completed pairs')
                estimated=1.8*float(np.median(times))*task_counts(order)['initial_worker_pairs_including_audits']+60
                atomic_json(dest/'forecast.json',{'estimated_full_s':estimated,'pilot_pair_s':times,
                    'remaining_budget_s':self.remaining(),'safety_factor':1.8,'not_a_finish_guarantee':True})
                if estimated>self.remaining()-180:raise Deferred(f'{name}: estimate {estimated:.0f}s exceeds remaining budget')
            state=run(command+extra,self.out/(name+'_'+phase),env=env,rss_limit_gib=48,
                      available_gib=8,disk_reserve_gib=80,wall_limit_s=max(1,self.remaining()-120))
            if state['state']!='complete':
                if state.get('reason','') and str(state['reason']).startswith('signal'):raise EndSeries(state['reason'])
                if state['state']=='time_limit':raise EndSeries('12-hour deadline reached')
                raise RuntimeError(f'{name} {phase}: {state}')
        if not read(dest/'summary.json').get('complete'):raise RuntimeError('incomplete T after full build')
        return dest

    def average(self,run_dir,ka,order,cutoff=None,local=False):
        cutoff=order if cutoff is None else cutoff
        cache=run_dir/f'analytic_L{cutoff}.npz';meta=run_dir/f'analytic_L{cutoff}.json'
        if cache.exists() and meta.exists():
            with np.load(cache) as f:return f['mueller'],read(meta)
        if not local:
            if self.remaining()<30:raise EndSeries('deadline before analytic averaging')
            command=[sys.executable,str(self.runtime/'scripts/run_size_series.py'),'--out',str(self.out),
                '--postprocess',str(run_dir),str(ka),str(order),str(cutoff)]
            env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
            env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',BEM_SO3_TMPDIR=str(self.out/'scratch'))
            state=run(command,run_dir/f'analytic_supervisor_L{cutoff}',env=env,rss_limit_gib=48,
                      available_gib=8,disk_reserve_gib=80,wall_limit_s=max(1,self.remaining()-15))
            if state['state']!='complete':raise EndSeries('analytic averaging stopped: '+str(state.get('reason')))
            with np.load(cache) as f:return f['mueller'],read(meta)
        t=np.load(run_dir/'tmatrix.npy',mmap_mode='r')
        if cutoff!=order:
            idx=cutoff_indices(order,cutoff);t=t[np.ix_(idx,idx)]
        theta=np.linspace(0,180,1801)
        mean,data=analytic_mueller_tmatrix(t,cutoff,theta,precision='fp64',threads=4,cyclic_order=6)
        data.update(Csca=float(2*np.pi/ka**2*np.vdot(t,t).real),Cext=float(-2*np.pi/ka**2*np.trace(t).real))
        radius=ka*self.plan['wavelength_um']/(2*np.pi)
        data.update(Csca_um2=data['Csca']*radius**2,Cext_um2=data['Cext']*radius**2)
        np.savez_compressed(cache,theta_deg=theta,mueller=mean);atomic_json(meta,data)
        if cutoff==order:
            physical=mean*(self.plan['wavelength_um']/(2*np.pi))**2
            header='theta_deg,'+','.join(f'M{i+1}{j+1}_um2_per_sr' for i in range(4) for j in range(4))
            np.savetxt(run_dir/'mueller_analytic_physical.csv',np.column_stack((theta,physical.reshape(len(theta),16))),
                       delimiter=',',header=header,comments='')
        return mean,data

    def calculate(self,ka):
        ref,order=selection(ka);theta=np.linspace(0,180,1801)
        for attempt in range(3):
            coarse=self.build(ka,ref-1,order)
            fine=self.build(ka,ref,order)
            self.active={'ka':ka,'ref':ref,'order':order,'phase':'analytic_average_and_convergence','run':str(fine)};self.status('running')
            a,ac=self.average(coarse,ka,order);b,bc=self.average(fine,ka,order)
            reduced,rc=self.average(fine,ka,order,order-4)
            mesh=compare(a,b,ac,bc,theta);tail=compare(reduced,b,rc,bc,theta)
            record={'ka':ka,'ri':self.plan['ri'],'wavelength_um':self.plan['wavelength_um'],
                'equal_volume_radius_um':ka*self.plan['wavelength_um']/(2*np.pi),
                'height_equals_width_um':ka*self.plan['wavelength_um']/(2*np.pi)*(8*np.pi/(3*np.sqrt(3)))**(1/3),
                'ref':ref,'order':order,'coarse_run':str(coarse),'fine_run':str(fine),
                'mesh_control':mesh,'modal_tail_control':tail,'Euler_quadrature_nodes':0,
                'converged':mesh['maximum']<=.008 and tail['maximum']<=.002,
                'meaning_of_converged':'Adjacent-mesh change <=0.8% and last-four-mode-shell change <=0.2%, all16/M11 plus Csca/Cext; not a rigorous absolute error bound.',
                'theta_step_deg':.1}
            atomic_json(self.out/f'ka{ka:g}_assessment.json',record)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(4,4,figsize=(12,10),sharex=True,constrained_layout=True)
            for i in range(4):
                for j in range(4):
                    axes[i,j].plot(theta,b[:,i,j]/b[:,0,0]);axes[i,j].set_title(f'M{i+1}{j+1}/M11');axes[i,j].grid(alpha=.25)
            fig.suptitle(f'ka={ka:g}, n={self.plan["ri"]}, analytic SO(3); mesh check={100*mesh["maximum"]:.3g}%')
            fig.savefig(self.out/f'ka{ka:g}_all16.png',dpi=140);plt.close(fig)
            if record['converged']:return record
            if tail['maximum']>.002:order+=6
            elif ref<6:ref+=1
            else:return record
        return record

    def save(self):atomic_json(self.out/'results.json',{'sizes':self.results,'updated_utc':iso(time.time()),
        'orientation_average':'analytic full SO(3), all16 Mueller','global_size_interpolation':'not certified across this broad interval'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=12)
    parser.add_argument('--ri',type=float,default=1.3116)
    parser.add_argument('--wavelength-um',type=float,default=.532)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--postprocess',nargs=4,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not all(np.isfinite(x) and x>0 for x in (args.hours,args.ri,args.wavelength_um)):parser.error('positive finite parameters required')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);runtime=out/'runtime';freeze(runtime)
    frozen=runtime/'scripts/run_size_series.py'
    if Path(__file__).resolve()!=frozen:
        os.execv(sys.executable,[sys.executable,str(frozen),*sys.argv[1:]])
    if args.postprocess:
        worker=object.__new__(Series);worker.out=out;worker.runtime=runtime;worker.plan=read(out/'plan.json')
        directory,ka,order,cutoff=args.postprocess
        worker.average(Path(directory),float(ka),int(order),int(cutoff),local=True)
        return
    lock=(out/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config={'hours':args.hours,'ri':args.ri,'wavelength_um':args.wavelength_um,'shape':'regular hexagonal prism','h_over_D_flats':1.}
    plan=read(out/'plan.json')
    if plan and plan['configuration']!=config:raise RuntimeError('resume configuration changed')
    if not plan:
        now=time.time();plan={'configuration':config,**config,'start_utc':iso(now),'deadline_unix':now+args.hours*3600,
            'deadline_utc':iso(now+args.hours*3600),'preferred_ka_range':[10,40],
            'candidate_ka':[10,20,30,40,15,25,35,12.5,17.5,22.5,27.5,32.5,37.5,45,50],
            'disk_reserve_GiB':80,'solver':'validated paired-device MLFMA with C2/C6/mirror packing',
            'experimental_single_face_FFT_enabled':False,
            'FFT_limitation':'Single-face FFT tested separately; complete inter-face Muller integration not yet validated.',
            'resume_command':[sys.executable,str(frozen),'--out',str(out),'--hours',str(args.hours),'--ri',str(args.ri),'--wavelength-um',str(args.wavelength_um)]}
        atomic_json(out/'plan.json',plan)
    if args.prepare_only:print(json.dumps(plan,indent=2));return
    (out/'supervisor.pid').write_text(str(os.getpid())+'\n');series=Series(out,plan)
    try:
        for ka in plan['candidate_ka']:
            if any(r['ka']==ka and r.get('converged') for r in series.results):continue
            try:
                record=series.calculate(ka)
                series.results=[r for r in series.results if r['ka']!=ka]+[record];series.save()
            except Deferred as e:
                atomic_json(out/f'ka{ka:g}_deferred.json',{'ka':ka,'reason':str(e),'updated_utc':iso(time.time())})
                continue
            except RuntimeError as e:
                atomic_json(out/f'ka{ka:g}_failed.json',{'ka':ka,'reason':str(e),'updated_utc':iso(time.time())})
                if 'resource reserve' in str(e):raise EndSeries(str(e))
                continue
        series.status('completed_candidate_list')
    except EndSeries as e:series.status('stopped',reason=str(e))
    finally:
        series.done.set();series.thread.join(timeout=10);series.save();lock.close()


if __name__=='__main__':main()
