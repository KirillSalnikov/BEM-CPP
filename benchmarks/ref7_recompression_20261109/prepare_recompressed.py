"""Recompress all ref7 cross factors on CPU, retaining the original case."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
import json
from pathlib import Path
import sys
import time
import numpy as np

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
SOURCE=BASE.parent/'ref7_memory_20261109/geometry_ref7';CASE=BASE/'geometry_ref7'
sys.path.insert(0,str(ROOT/'scripts'))
from recompress_spectral_factors import recompress_factors
from bounded_spectral_storage import half_factors

CASE.mkdir(exist_ok=True)
for source in SOURCE.iterdir():
    if source.name in ('execution.json','streamed_preparation.json','service_assets.json','.spectral-cache'):continue
    target=CASE/source.name
    if not target.exists():target.symlink_to(source.resolve(),target_is_directory=source.is_dir())
cache=CASE/'.spectral-cache';cache.mkdir(exist_ok=True)
for index in (0,3,6,7):
    target=cache/str(index)
    if not target.exists():target.symlink_to((SOURCE/'.spectral-cache'/str(index)).resolve(),target_is_directory=True)
execution=json.loads((SOURCE/'execution.json').read_text())
execution.update(symbol_storage='half',shared_stream_buffers=True)
(CASE/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
preparation=json.loads((SOURCE/'streamed_preparation.json').read_text())
started=time.perf_counter();records=[]
def save(status):
    temp=BASE/'preparation.tmp'
    temp.write_text(json.dumps(dict(status=status,records=records,seconds=time.perf_counter()-started),indent=2)+'\n')
    temp.replace(BASE/'preparation.json')
save('running')
for row in preparation['records']:
    if row['kind']=='2D':continue
    index=row['index'];directory=CASE/f'cross_{index}_recompressed'
    left=np.load(SOURCE/row['directory']/'left.npy',mmap_mode='r')
    right=np.load(SOURCE/row['directory']/'right.npy',mmap_mode='r')
    def progress(record,total):
        if (record['frequency']+1)%32==0:
            print(json.dumps(dict(block=index,frequency=record['frequency']+1,total=total,
                                  rank=record['rank'],elapsed_s=time.perf_counter()-started)),flush=True)
    (a,b),report=recompress_factors(left,right,directory,tolerance=1e-5,rank_multiple=8,progress=progress)
    u,us=half_factors(a,cache/str(index),'left');v,vs=half_factors(b,cache/str(index),'right')
    records.append(dict(index=index,old_rank=report['old_rank'],rank=report['rank'],
                        old_bytes=report['old_bytes'],bytes=report['new_bytes'],
                        half_bytes=u.nbytes+v.nbytes+us.nbytes+vs.nbytes,
                        bound=report['maximum_relative_error_bound'],seconds=report['seconds']))
    row.update(directory=directory.name,rank=report['rank'],bytes=report['new_bytes'],
               recompression_error_bound=report['maximum_relative_error_bound'],
               recompression_reference='Existing FP32 factors; subsequent half encoding requires an action check')
    for key in ('holdout_max','seconds','GPU_peak_allocated_bytes'):
        if key in row:row['source_'+key]=row.pop(key)
    row['recompression_seconds']=report['seconds']
    save('running');print('BLOCK',json.dumps(records[-1]),flush=True)
preparation['source_total_s_this_run']=preparation.pop('total_s_this_run',None)
preparation['recompression_and_encoding_s']=time.perf_counter()-started
preparation['symbol_and_factor_bytes']=sum(row['bytes'] for row in preparation['records'])
(CASE/'streamed_preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
save('complete')
