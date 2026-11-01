"""Bounded CPU FP64 verification of immutable solution snapshots.

The supplied action must be an independent CPU operator, safe to call in the
worker thread. Wrapping a CUDA/FMM action here does not offload its work and
is unsafe if it shares mutable buffers with the GPU solver.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
import numpy as np


@dataclass(frozen=True)
class ResidualTicket:
    token: int
    label: str
    version: int


@dataclass
class ResidualVerdict:
    ticket: ResidualTicket
    candidate: np.ndarray
    residual: np.ndarray
    relative_by_column: np.ndarray
    maximum_relative: float
    seconds: float
    started: float
    finished: float

    def accepts(self, tolerance):
        if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('positive finite tolerance required')
        return self.maximum_relative<=tolerance


class AsyncFP64Residual:
    def __init__(self, cpu_action, *, workers=1, capacity=2):
        if workers<1 or capacity<1:raise ValueError('positive worker and queue counts required')
        self.action=cpu_action;self.capacity=capacity;self.pending={};self.next_token=0;self.closed=False
        self.executor=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='bem-fp64')

    def submit(self, candidate, rhs, *, label, version=0):
        if self.closed:raise RuntimeError('verifier is closed')
        if len(self.pending)>=self.capacity:raise RuntimeError('verification queue is full')
        x=np.array(candidate,dtype=np.complex128,copy=True,order='C')
        b=np.array(rhs,dtype=np.complex128,copy=True,order='C')
        if x.ndim!=2 or x.shape!=b.shape or not np.isfinite(x).all() or not np.isfinite(b).all():
            raise ValueError('finite, matching 2D solution/RHS blocks required')
        x.setflags(write=False);b.setflags(write=False)
        ticket=ResidualTicket(self.next_token,str(label),int(version));self.next_token+=1
        def verify():
            start=time.perf_counter();ax=np.asarray(self.action(x))
            if ax.shape!=b.shape or ax.dtype!=np.complex128 or not np.isfinite(ax).all():
                raise RuntimeError('CPU action must return a finite complex128 block of the same shape')
            r=b-ax;bn=np.linalg.norm(b,axis=0);rn=np.linalg.norm(r,axis=0)
            relative=np.divide(rn,bn,out=np.where(rn==0,0.,np.inf),where=bn>0)
            end=time.perf_counter()
            return ResidualVerdict(ticket,x,r,relative,float(np.max(relative)),end-start,start,end)
        self.pending[ticket.token]=(ticket,self.executor.submit(verify))
        return ticket

    def take(self,ticket):
        if ticket.token not in self.pending or self.pending[ticket.token][0] is not ticket:
            raise ValueError('unknown, substituted or already consumed residual ticket')
        expected,future=self.pending[ticket.token]
        try:
            result=future.result()
            if result.ticket is not expected:raise RuntimeError('verification snapshot mismatch')
            return result
        finally:del self.pending[ticket.token]

    def close(self):
        self.closed=True;self.executor.shutdown(wait=True)

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
