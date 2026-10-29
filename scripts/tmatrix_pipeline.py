"""Prepare one future incident packet while the worker solves the current one.

The socket send must finish before prefetch: the same field buffer is reused.
Only the producer accesses the radial/angular cache. A changed fallback task
discards speculative work, never its key or normalization metadata.
"""
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from tmatrix_incident import regular_fields, RegularFieldCache
from tmatrix_c6 import encode_fields


class IncidentPreparation:
    def __init__(self, positions, modal, ka, fields, cache=True, prefetch=False):
        self.positions,self.modal,self.ka,self.fields=positions,modal,ka,fields
        self.cache=RegularFieldCache(positions,ka) if cache else None
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='incident') if prefetch else None
        self.pending=None

    @staticmethod
    def key(kind,group,reflection):
        return kind,tuple(group),bool(reflection)

    def prepare(self,key):
        kind,group,reflection=key
        start=time.perf_counter()
        combined=None
        if kind=='single':
            pair=group if len(group)==2 else group*2
            scales=[]
            for slot,index in enumerate(pair):
                if self.cache is not None:
                    self.cache.fill(self.modal[index],self.fields[slot])
                else:
                    for point in range(0,len(self.positions),65536):
                        self.fields[slot,point:point+65536]=regular_fields(
                            self.positions[point:point+65536],self.modal[index],self.ka)
                scale=float(np.max(np.abs(self.fields[slot])))
                if not np.isfinite(scale) or scale<1e-280:
                    raise RuntimeError('regular-wave field underflow; reduce order or change normalization')
                self.fields[slot]/=scale;scales.append(scale)
        else:
            scales,combined=encode_fields(self.positions,self.modal,group,self.ka,self.fields,
                reflection=reflection,field_cache=self.cache)
        return scales,combined,time.perf_counter()-start

    def get(self,kind,group,reflection):
        key=self.key(kind,group,reflection);start=time.perf_counter();prefetched=False
        result=None
        if self.pending is not None:
            previous,future=self.pending;self.pending=None
            if previous==key:
                result=future.result();prefetched=True
            else:
                # A rejection can insert fallback work ahead of speculation.
                # Finish the stale writer, but its unused error is irrelevant.
                future.cancel()
                try:future.result()
                except Exception:pass
        if result is None:
            result=self.executor.submit(self.prepare,key).result() if self.executor else self.prepare(key)
        return (*result,time.perf_counter()-start,prefetched)

    def prefetch(self,kind,group,reflection):
        if self.executor is not None:
            if self.pending is not None:raise RuntimeError('incident prefetch already pending')
            key=self.key(kind,group,reflection)
            self.pending=(key,self.executor.submit(self.prepare,key))

    def close(self):
        if self.executor is not None:self.executor.shutdown(wait=True)
