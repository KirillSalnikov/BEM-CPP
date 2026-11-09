"""Compare real ref7 actions with saved original FMM outputs, on a shared GPU.

No native FMM allocation is made here. Timings under another project's GPU
load are recorded only as diagnostics, not an isolated throughput benchmark.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
SOURCE=BASE.parent/'ref7_memory_20261109/geometry_ref7';CASE=BASE/'geometry_ref7'
sys.path.insert(0,str(ROOT/'scripts'))
from compressed_fft_service import load_operator
assert json.loads((BASE/'preparation.json').read_text())['status']=='complete'
torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.35)
report=dict(status='running',GPU_shared_with_another_project=True,
            strict_native_oracles=str(SOURCE),records=[])
def save():
    tmp=BASE/'actions.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(BASE/'actions.json')
save()
settings=[('baseline',SOURCE,{}),
          ('previous_half',SOURCE,dict(symbol_storage='half')),
          ('recompressed_packed_0',CASE,dict(symbol_storage='packed_half',shared_stream_buffers=True,spectral_budget_MiB=0)),
          ('recompressed_packed_1024',CASE,dict(symbol_storage='packed_half',shared_stream_buffers=True,spectral_budget_MiB=1024)),
          ('recompressed_full_0',CASE,dict(symbol_storage='full',shared_stream_buffers=True,spectral_budget_MiB=0))]
for name,source,override in settings:
    variant=BASE/'variants'/name;variant.mkdir(parents=True,exist_ok=True)
    for path in source.iterdir():
        if path.name in ('execution.json','service_assets.json'):continue
        link=variant/path.name
        if not link.exists():link.symlink_to(path.resolve(),target_is_directory=path.is_dir())
    config=json.loads((source/'execution.json').read_text());config.update(override)
    (variant/'execution.json').write_text(json.dumps(config,indent=2)+'\n')
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    gpu=load_operator(variant)
    row=dict(name=name,case=str(variant),configuration=config,load_s=time.perf_counter()-start,
             operator_bytes=gpu.bytes,stream_bytes_per_action=gpu.stream_transfer_bytes,
             pinned_bytes=gpu.pinned_bytes,map_bytes=gpu.map_bytes,local_host_bytes=gpu.local.host_bytes,actions=[])
    report['records'].append(row);save()
    for label,input_file,reference_file in [('random','probe_input.bin','probe_A_0.bin'),
                                            ('smooth','smooth_input.bin','smooth_baseline6.bin')]:
        x=torch.as_tensor(np.fromfile(SOURCE/input_file,np.complex128),device='cuda',dtype=torch.complex64)
        reference=np.fromfile(SOURCE/reference_file,np.complex128)
        for repeat in range(2):
            torch.cuda.synchronize();start=time.perf_counter();value=gpu(x);torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            error=float(np.linalg.norm(value.cpu().numpy()-reference)/np.linalg.norm(reference))
            row['actions'].append(dict(input=label,repeat=repeat,seconds=seconds,relative_native_error=error))
            print(name,label,repeat,seconds,error,flush=True);save()
        del x,value,reference
    row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               peak_reserved_bytes=torch.cuda.max_memory_reserved(),
               passes_original_action_gate=all(np.isfinite(a['relative_native_error']) and a['relative_native_error']<=5e-4 for a in row['actions']))
    gpu.local.pool.shutdown();del gpu;gc.collect();torch.cuda.empty_cache();save()
report['status']='complete';save()
