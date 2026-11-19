"""Build new ka80 factors, one bounded block at a time, then recompress them.

An accepted native action/near export is required before this stage. All GPU
ownership is released before the next experiment. No T solve is launched here.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',OMP_WAIT_POLICY='PASSIVE')
import fcntl
import gc
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import time
import numpy as np

B=Path(__file__).resolve().parent
R=B.parents[1]
CASE=B/'fft_ka80'
sys.path.insert(0,str(B/'runtime_v2/scripts'))
from compressed_fft_service import file_digest,prepare_manifest
from packed_current_csr import prepare
from symmetric_galerkin_fft import ParallelGalerkinFFT
from streamed_galerkin_fft import compress_streamed,parallel_symbol_gpu,sketch_workspace_bytes
from recompress_spectral_factors import recompress_factors

def save(path,data):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(data,indent=2)+'\n');p.replace(path)
def reserve(extra=0):
    if shutil.disk_usage(CASE).free < extra+40*2**30:raise RuntimeError('disk reserve would be exceeded')
def stopped():
    if (B/'STOP').exists():raise RuntimeError('STOP requested')

started=time.monotonic()
progress=CASE/'streamed_preparation.json'
report=json.loads(progress.read_text()) if progress.exists() else dict(status='waiting_for_native',records=[],ref=7,ka=80)
if report['status']=='complete':print('already complete');sys.exit(0)
save(progress,report)
while True:
    stopped()
    state=json.loads((B/'current_native_resume_run/status.json').read_text())
    if state['state']=='failed':raise RuntimeError('current native validation failed')
    if state['state']=='complete':break
    time.sleep(3)
receipt=json.loads((B/'current_native.json').read_text())
if receipt['status']!='complete':raise RuntimeError('missing completed current-native receipt')
native=B/'native_ka80'
for name,identity in receipt['near_export'].items():
    source=native/name
    if source.stat().st_size!=identity['bytes'] or file_digest(source)!=identity['sha256']:
        raise RuntimeError('native export changed: '+name)
    target=CASE/name
    if not target.exists():target.symlink_to(source)
    elif target.resolve()!=source.resolve():raise RuntimeError('unexpected existing export link')
report['native_receipt_sha256']=file_digest(B/'current_native.json')

# This CPU job prepares only wavelength-correct regular parallel-face symbols,
# independently of the native near export. Consume its completed receipt once;
# never let two writers mutate the GPU preparation checkpoint.
cpu_receipt=B/'fft_parallel_cpu/receipt.json'
while True:
    stopped()
    state=json.loads((B/'parallel_cpu_run/status.json').read_text())
    if state['state']=='failed':raise RuntimeError('CPU parallel preparation failed')
    if state['state']=='complete':break
    time.sleep(3)
cpu=json.loads(cpu_receipt.read_text())
if cpu['status']!='complete' or cpu['binding_sha256']!=file_digest(CASE/'bound_operator.pkl'):
    raise RuntimeError('CPU symbols have no matching completed receipt')
for row in cpu['records']:
    source=cpu_receipt.parent/row['file']
    if source.stat().st_size!=row['file_bytes'] or file_digest(source)!=row['sha256']:
        raise RuntimeError('CPU symbol changed: '+str(source))
    target=CASE/row['file']
    if not target.exists():target.symlink_to(source)
    elif target.resolve()!=source.resolve():raise RuntimeError('Unexpected parallel symbol')
    if not any(old['index']==row['index'] for old in report['records']):
        report['records'].append(dict(row,precomputed_receipt=str(cpu_receipt)))
report['parallel_cpu_receipt_sha256']=file_digest(cpu_receipt)
report['status']='preparing_local';save(progress,report)
if not (CASE/'packed_local/metadata.json').exists():
    reserve(4*2**30)
    print('PACKED',prepare(CASE,CASE/'packed_local'),flush=True)

locks=[]
while True:
    stopped()
    try:
        for suffix in ('','.1','.2'):
            f=open('/home/kirill/neuro/agent-sandbox/.gpu.lock'+suffix,'a');locks.append(f)
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import subprocess
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if not busy:break
    except BlockingIOError:pass
    for f in locks:f.close()
    locks=[];time.sleep(3)

import torch
torch.set_num_threads(1)
with (CASE/'bound_operator.pkl').open('rb') as f:operator=pickle.load(f)
if operator.geometry.meta['ka']!=80:raise RuntimeError('incorrect spectral geometry')
report['status']='building';save(progress,report)
try:
    for index,block in enumerate(operator.canonical):
        stopped()
        if any(row['index']==index for row in report['records']):continue
        assert block.ka==80 and block.ri==complex(1.3116)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
        if isinstance(block,ParallelGalerkinFFT):
            reserve(block.nfx*block.nfy*block.tc*block.sc*8)
            symbol=parallel_symbol_gpu(block,precision=64).astype(np.complex64)
            name=f'parallel_{index}.npy';np.save(CASE/name,symbol)
            row=dict(index=index,kind='2D',file=name,bytes=symbol.nbytes);del symbol
        else:
            rank=128 if index in (1,2) else 192
            attempts=[]
            while True:
                stopped();torch.cuda.empty_cache()
                nf,nr,nc=block.nfft,2*block.tc,2*block.sc
                # The three full sketches dominate preparation VRAM. Preserve
                # 3 GiB for tiles, QR/lstsq workspaces and CUDA runtime overhead.
                transposed=nc>nr
                sketch_bytes=sketch_workspace_bytes(nf,nr,nc,rank,transpose_sketch=transposed)
                free,_=torch.cuda.mem_get_info()
                if sketch_bytes+3*2**30>free:
                    raise RuntimeError(f'block {index} rank {rank} requires frequency-batched preparation; sketch={sketch_bytes/2**30:.3f} GiB')
                # Raw factors and recompression temporary/final output coexist.
                reserve(4*nf*(nr+nc)*rank*8)
                with tempfile.TemporaryDirectory(prefix=f'raw_cross_{index}_r{rank}_',dir=CASE) as raw:
                    def tick(done,total,elapsed):
                        if done==total or done%max(1,total//8)==0:print('TILES',index,rank,done,total,round(elapsed,2),flush=True)
                    factors,info=compress_streamed(block,raw,rank,precision=64,tile=2,progress=tick,transpose_sketch=transposed)
                    attempts.append(info)
                    save(CASE/f'cross_{index}_attempts.json',dict(attempts=attempts))
                    if info['status']=='complete':
                        directory=CASE/f'cross_{index}_recompressed'
                        def cpu_tick(row,total):
                            if (row['frequency']+1)%64==0:print('RECOMPRESS',index,row['frequency']+1,total,flush=True)
                        packed,recompression=recompress_factors(*factors,directory,tolerance=1e-5,rank_multiple=8,progress=cpu_tick)
                        row=dict(index=index,kind='1D compressed',directory=directory.name,
                                 bytes=recompression['new_bytes'],rank=recompression['rank'],
                                 sketch_rank=rank,holdout_max=info['holdout_max'],
                                 sketch_transposed=transposed,sketch_workspace_bytes=sketch_bytes,
                                 recompression_error_bound=recompression['maximum_relative_error_bound'],
                                 raw_factor_policy='New temporary sketch factors removed after accepted recompression; diagnostic receipt retained')
                        del packed,factors;gc.collect()
                        break
                    del factors;gc.collect()
                rank+=32
                if rank>320:raise RuntimeError('rank screen exhausted; no automatic tolerance relaxation')
        row.update(seconds=time.monotonic()-begin,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        report['records'].append(row);save(progress,report);print('READY',json.dumps(row),flush=True)
        gc.collect();torch.cuda.empty_cache()
    report.update(status='complete',wait_included_wall_s=time.monotonic()-started,
                  symbol_and_factor_bytes=sum(r['bytes'] for r in report['records']))
    save(progress,report)
    save(CASE/'service_assets.json',prepare_manifest(CASE))
    print('ka80 spectral coefficients complete; full operator accuracy/pilot still required',flush=True)
except BaseException as e:
    report.update(status='failed',error=repr(e));save(progress,report);raise
