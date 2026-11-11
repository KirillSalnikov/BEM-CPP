"""Resume the audited bullet checkpoint when the shared GPU becomes available.

No wall-time deadline. Retains the existing campaign lock and reports waiting
separately from actual computation. Other GPU owners are never interrupted.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

B=Path(__file__).resolve().parent;R=B.parents[1]
BASE=R/'runs/scice_regular_shapes_ka20_20261111'
STOP=False;CHILD=None

def stamp():return datetime.now(timezone.utc).isoformat()
def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def stop(signum,frame):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:
        try:os.killpg(CHILD.pid,signal.SIGTERM)
        except ProcessLookupError:pass

def main():
    global CHILD
    campaign=(BASE/'campaign.lock').open('a');fcntl.flock(campaign,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    plan=json.loads((B/'restart_plan.json').read_text());runtime=Path(plan['runtime']);out=Path(plan['target'])
    for relative,expected in json.loads((runtime/'sha256.json').read_text()).items():
        assert hashlib.sha256((runtime/relative).read_bytes()).hexdigest()==expected,relative
    assert shutil.disk_usage(out).free>20*2**30
    config=json.loads((BASE/'manifest.json').read_text())
    case=next(c for c in config['cases'] if c['name']=='Bullet')
    case['bem']=str(out);case['bem_command']=plan['command']
    save(BASE/'manifest.json',config)
    status=json.loads((BASE/'status.json').read_text());job=status['jobs']['Bullet/bem']
    def publish(state,**details):
        job.clear();job.update(state=state,directory=str(out),saved_column_files=len(list((out/'columns').glob('*.npy'))),**details)
        status.update(pid=os.getpid(),updated_utc=stamp(),stop_requested=STOP)
        save(BASE/'status.json',status);save(B/'continuation_status.json',dict(job,controller_pid=os.getpid(),updated_utc=stamp()))
    save(B/'controller.json',dict(pid=os.getpid(),started_utc=stamp(),command=plan['command'],runtime=str(runtime)))
    save(BASE/'process.json',dict(pid=os.getpid(),started_utc=stamp(),command=[str(Path(__file__).resolve())]))
    locks=[];start=None
    try:
        publish('waiting_for_gpu')
        for suffix in ('','.1','.2'):
            lock=open('/home/kirill/neuro/agent-sandbox/.gpu.lock'+suffix,'a');locks.append(lock)
            while not STOP:
                if (BASE/'STOP').exists():stop(signal.SIGTERM,None);break
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:publish('waiting_for_gpu');time.sleep(5)
            if STOP:publish('stopped');return
        env={k:v for k,v in os.environ.items() if not k.startswith(('BEM_','MBS_'))};env.update(plan['environment'])
        start=time.monotonic()
        with (out/'driver.log').open('a') as log:
            CHILD=subprocess.Popen(plan['command'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(out/'launch.json',dict(pid=CHILD.pid,supervisor_pid=os.getpid(),started_utc=stamp(),command=plan['command'],environment=plan['environment']))
            while CHILD.poll() is None:
                if (BASE/'STOP').exists():stop(signal.SIGTERM,None)
                publish('stopping' if STOP else 'running',pid=CHILD.pid,elapsed_s=time.monotonic()-start)
                time.sleep(5)
        code=CHILD.wait()
        save(out/'finished.json',dict(returncode=code,elapsed_s=time.monotonic()-start,finished_utc=stamp()))
        if code!=0:publish('stopped' if STOP else 'failed',returncode=code);return
        summary=json.loads((out/'summary.json').read_text())
        assert summary['complete'] and summary['symmetry_reuse']['passed']
        assert json.loads((out/'average.json').read_text())['Euler_quadrature_nodes']==0
        publish('complete',returncode=code,elapsed_s=time.monotonic()-start)
        with (BASE/'report.log').open('a') as log:
            result=subprocess.run(['python3',str(BASE/'report.py')],stdout=log,stderr=subprocess.STDOUT,cwd=R)
        save(BASE/'report_status.json',dict(returncode=result.returncode,finished_utc=stamp()))
    except Exception as error:
        stop(signal.SIGTERM,None)
        if CHILD is not None:CHILD.wait()
        publish('failed',error=repr(error));raise
    finally:
        for lock in locks:lock.close()

if __name__=='__main__':main()
