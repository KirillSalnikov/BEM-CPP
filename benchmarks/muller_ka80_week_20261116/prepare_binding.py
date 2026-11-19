"""Reuse wave-independent BDM maps, generate a ka80 spectral operator shell.

No ka40 Green functions or near-correction coefficients are reused.
"""
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

B=Path(__file__).resolve().parent
R=B.parents[1]
sys.path.insert(0,str(R/'scripts'))
from galerkin_cross_fft import CrossGalerkinFFT
from symmetric_galerkin_fft import ParallelGalerkinFFT

old=R/'benchmarks/ref7_memory_20261109/geometry_ref7'
case=B/'fft_ka80'
case.mkdir(exist_ok=True)
meta=json.loads((old/'geometry.json').read_text())
assert (meta['ref'],meta['system_dofs'],meta['aspect'],meta['ri'])==(7,4552704,1.6495721976846451,[1.3116,0])
meta['ka']=80
manifest=case/'binding_provenance.json'
if manifest.exists():
    if json.loads((case/'geometry.json').read_text())!=meta:raise RuntimeError('existing geometry mismatch')
    print('binding already prepared');sys.exit(0)
with (old/'bound_operator.pkl').open('rb') as f:op=pickle.load(f)
assert op.geometry.N==meta['system_dofs'] and len(op.canonical)==8 and op.local is None
fixture=json.loads((R/'benchmarks/galerkin_cross_fft_20261108/fixture.json').read_text())
blocks=[];shapes=[]
for i,block in enumerate(op.canonical):
    assert block.ka==40 and block.ri==complex(*meta['ri']) and block.symbol is None
    factory=ParallelGalerkinFFT if isinstance(block,ParallelGalerkinFFT) else CrossGalerkinFFT
    new=factory(fixture,block.target,block.source,80,complex(*meta['ri']))
    new.symbol=None;new.factor_bytes=0;blocks.append(new)
    shape=[new.nfx,new.nfy,new.tc,new.sc] if isinstance(new,ParallelGalerkinFFT) else [new.nfft,2*new.tc,2*new.sc]
    shapes.append(dict(index=i,kind=factory.__name__,shape=shape))
op.canonical=blocks
op.geometry=SimpleNamespace(directory=case,n=meta['current_dofs'],N=meta['system_dofs'],meta=meta)
op.packed_local_directory=case/'packed_local'
(case/'geometry.json').write_text(json.dumps(meta,indent=2)+'\n')
with (case/'bound_operator.pkl').open('wb') as f:pickle.dump(op,f)
for name in ('mass_rows.bin','mass_cols.bin','mass_values.bin'):
    p=case/name
    if not p.exists():p.symlink_to(old/name)
execution=dict(local_storage='host_packed',spectral_budget_MiB=1024,frequency_batch=8,
               compact_maps=True,symbol_storage='packed_half',shared_stream_buffers=True)
(case/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
source=old/'bound_operator.pkl'
manifest.write_text(json.dumps(dict(status='complete',source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    binding_sha256=hashlib.sha256((case/'bound_operator.pkl').read_bytes()).hexdigest(),
    geometry=meta,blocks=shapes,reused='BDM incidence/sign/alias maps and mass only',
    pending='new ka80 spectral coefficients and native near correction'),indent=2)+'\n')
print(json.dumps(shapes),flush=True)
