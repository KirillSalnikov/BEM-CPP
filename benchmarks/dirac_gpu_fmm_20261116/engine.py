"""Experimental working-action GPU adapter; full-surface audits stay on CPU."""
import importlib.util
import time
from pathlib import Path
import numpy as np

B=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dirac_gpu_native',B/'native.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class WorkingGPU:
    def __init__(self,cpu,sources,targets,waves,depth,digits=6):
        self.cpu=cpu;self.sources=sources;self.targets_array=targets
        self.plans={};self.setup_s=0.;self.calls=0;self.seconds=0.
        self.signature=lambda a:(a.ctypes.data,a.shape,a.strides,a.dtype.str)
        self.source_signature=self.signature(sources);self.target_signature=self.signature(targets)
        try:
            for k in dict.fromkeys(map(complex,waves)):
                begin=time.perf_counter()
                self.plans[complex(k)]=module.GPUFMM(sources,targets,k,digits=digits,depth=depth)
                self.setup_s+=time.perf_counter()-begin
        except BaseException:
            self.close();raise

    def targets(self,*,eps,zk,sources,charges,targets):
        if self.signature(sources)!=self.source_signature or self.signature(targets)!=self.target_signature:
            raise ValueError('GPU plan geometry changed')
        if not 1e-12<=eps<=1e-7:raise ValueError('experimental GPU working tolerance')
        begin=time.perf_counter();out=self.plans[complex(zk)].apply(charges)
        self.calls+=1;self.seconds+=time.perf_counter()-begin
        return out

    def hfmm3d(self,**kwargs):
        return self.cpu.hfmm3d(**kwargs)

    @property
    def info(self):
        return {str(k):dict(p.info) for k,p in self.plans.items()}

    def close(self):
        for p in self.plans.values():p.close()
        self.plans.clear()
