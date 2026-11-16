#!/usr/bin/env python3
"""Run a long BEM process with persistent status and resource limits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import psutil


def atomic_json(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def run(command, out, env=None, rss_limit_gib=48, available_gib=6, disk_reserve_gib=40,
        wall_limit_s=None, progress_callback=None):
    if wall_limit_s is not None and wall_limit_s<=0:
        raise ValueError('wall_limit_s must be positive')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy() if env is None else env
    identity={'command':command,'BEM_environment':{k:v for k,v in env.items() if k.startswith('BEM_')}}
    identity['binary_sha256']=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    status_path=out/'status.json'
    if status_path.exists():
        old=json.loads(status_path.read_text())
        if old.get('signature')!=signature:
            raise RuntimeError(f'{out}: command/binary changed; use a new directory')
        if old.get('state')=='complete':
            return old
    atomic_json(out/'command.json',identity)
    start=time.monotonic(); peak_rss=peak_vram=0; reason=None;last_notice=start
    with (out/'worker.log').open('a',buffering=1) as log, (out/'resources.jsonl').open('a',buffering=1) as samples:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state={'state':'running','signature':signature,'pid':process.pid,'supervisor_pid':os.getpid(),
               'peak_rss_gib':0,'peak_gpu_mib':0}
        def terminate(signum, frame):
            nonlocal reason
            reason=f'signal {signum}'
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
        previous={s:signal.signal(s,terminate) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            while process.poll() is None:
                try:
                    root=psutil.Process(process.pid)
                    children=[root]+root.children(recursive=True)
                    rss=sum(p.memory_info().rss for p in children if p.is_running())
                except psutil.Error:
                    rss=0
                free=shutil.disk_usage(out).free; available=psutil.virtual_memory().available
                used=gpu_free=0
                # CPU-only hosts and multiple GPUs are valid configurations.
                # Device totals are telemetry, not memory owned by this child.
                if shutil.which('nvidia-smi'):
                    try:
                        gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],
                                           text=True,capture_output=True,timeout=2)
                        rows=[list(map(float,line.split(','))) for line in gpu.stdout.splitlines()] if gpu.returncode==0 else []
                        used=sum(row[0] for row in rows);gpu_free=sum(row[1] for row in rows)
                    except (OSError,ValueError,subprocess.TimeoutExpired):pass
                peak_rss=max(peak_rss,rss); peak_vram=max(peak_vram,used)
                sample={'elapsed_s':time.monotonic()-start,'rss_gib':rss/2**30,'available_ram_gib':available/2**30,
                        'disk_free_gib':free/2**30,'gpu_used_mib':used,'gpu_free_mib':gpu_free}
                samples.write(json.dumps(sample)+'\n')
                state={'state':'running','signature':signature,'pid':process.pid,'supervisor_pid':os.getpid(),
                       **sample,'peak_rss_gib':peak_rss/2**30,'peak_gpu_mib':peak_vram}
                atomic_json(status_path,state)
                if progress_callback is not None and time.monotonic()-last_notice>=15:
                    progress_callback(state);last_notice=time.monotonic()
                if rss>rss_limit_gib*2**30 or available<available_gib*2**30 or free<disk_reserve_gib*2**30:
                    reason='resource reserve reached (RAM/RSS/disk)'
                if wall_limit_s is not None and time.monotonic()-start>=wall_limit_s:
                    reason='wall time limit reached'
                if reason:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL)
                    break
                try: process.wait(timeout=2)
                except subprocess.TimeoutExpired: pass
            code=process.wait()
        finally:
            for sig,handler in previous.items(): signal.signal(sig,handler)
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL); process.wait()
    state.update(state=('complete' if code==0 and reason is None else
                        'time_limit' if reason=='wall time limit reached' else 'failed'),returncode=code,
                 reason=reason,elapsed_s=time.monotonic()-start)
    atomic_json(status_path,state)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--wall-hours',type=float)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command: parser.error('provide a command after --')
    result=run(command,args.out,wall_limit_s=None if args.wall_hours is None else args.wall_hours*3600)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['state']=='complete' else 1)
