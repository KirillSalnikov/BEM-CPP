"""Prepare reusable ka80 parallel-face symbols on CPU while GPU is occupied.

Uses the frozen runtime and unchanged FP64 Green/quadrature contractions.
The receipt is separate from the GPU preparation checkpoint. The full combined
FFT action still requires its native-operator accuracy gate before a solve.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='16',OMP_WAIT_POLICY='PASSIVE')
from dataclasses import replace
import gc
import json
from pathlib import Path
import pickle
import resource
import shutil
import sys
import time

import numpy as np
import torch

B=Path(__file__).resolve().parent
CASE=B/'fft_ka80'
OUT=B/'fft_parallel_cpu'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(B/'runtime_v2/scripts'))
from compressed_fft_service import file_digest
from symmetric_galerkin_fft import ParallelGalerkinFFT
from streamed_galerkin_fft import parallel_symbol_gpu

def save(data):
    path=OUT/'receipt.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)

torch.set_num_threads(16)
binding=CASE/'bound_operator.pkl'
identity=file_digest(binding)
report=dict(status='building',ka=80,ref=7,device='cpu',threads=16,
            kernel_precision=64,stored_dtype='complex64',binding_sha256=identity,
            script_sha256=file_digest(Path(__file__)),records=[],
            scope='Parallel regular blocks only. Full combined native-operator gate remains mandatory.')
receipt=OUT/'receipt.json'
if receipt.exists():
    previous=json.loads(receipt.read_text())
    if previous['binding_sha256']!=identity:raise RuntimeError('Binding changed')
    report=previous
    if report['status']=='complete':raise SystemExit(0)
with binding.open('rb') as f:operator=pickle.load(f)
if operator.geometry.meta['ka']!=80:raise RuntimeError('Wrong geometry')
start=time.monotonic()
save(report)
try:
    for index in (0,3,6,7):
        if (B/'STOP').exists():raise RuntimeError('STOP requested')
        if any(r['index']==index for r in report['records']):continue
        block=operator.canonical[index]
        if not isinstance(block,ParallelGalerkinFFT) or block.ka!=80 or block.ri!=complex(1.3116):
            raise RuntimeError('Wrong physical block')
        # Independent NumPy quadrature and 2D FFT on patches of the actual faces.
        target=replace(block.target,nx=2,ny=3)
        source=replace(block.source,nx=3,ny=2)
        small=ParallelGalerkinFFT(block.fixture,target,source,block.ka,block.ri).build()
        candidate=parallel_symbol_gpu(small,device='cpu',precision=64)
        error=float(np.linalg.norm(candidate-small.symbol)/np.linalg.norm(small.symbol))
        if not np.isfinite(error) or error>1e-11:raise RuntimeError(f'CPU block {index} check {error}')
        del small,candidate
        bytes_out=block.nfx*block.nfy*block.tc*block.sc*8
        if shutil.disk_usage(OUT).free<bytes_out+40*2**30:raise RuntimeError('Disk reserve')
        print('BUILD',index,bytes_out,flush=True)
        begin=time.monotonic()
        symbol=parallel_symbol_gpu(block,device='cpu',precision=64)
        kernel_fft_s=time.monotonic()-begin
        stored=symbol.astype(np.complex64)
        del symbol;gc.collect()
        if not np.isfinite(stored).all():raise RuntimeError('Non-finite symbol')
        name=f'parallel_{index}.npy'
        temporary=OUT/(name+'.tmp')
        with temporary.open('wb') as f:np.save(f,stored)
        temporary.replace(OUT/name)
        shape=list(stored.shape);del stored;gc.collect()
        row=dict(index=index,kind='2D',file=name,shape=shape,bytes=bytes_out,
                 file_bytes=(OUT/name).stat().st_size,sha256=file_digest(OUT/name),
                 device='cpu',kernel_fft_s=kernel_fft_s,seconds=time.monotonic()-begin,
                 small_patch_relative_error=error,GPU_peak_allocated_bytes=0,
                 cumulative_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        report['records'].append(row);save(report)
        print('READY',json.dumps(row),flush=True)
    report.update(status='complete',wall_s=time.monotonic()-start)
    save(report)
except BaseException as e:
    report.update(status='failed',error=repr(e));save(report);raise
