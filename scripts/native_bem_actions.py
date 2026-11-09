"""Column-block client for the experimental original FFT/FMM action worker.

The worker serializes actions through its existing GPU workspace. This is
not GPU batching and not a resident client: measured file transfer and setup
costs must remain part of the P12 assessment. Returned currents are guesses
until corrected and checked against the original strict operator.
"""
import json
from pathlib import Path
import subprocess
import time
import numpy as np


class NativeBEMActions:
    def __init__(self, command, directory, *, env, cwd=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise ValueError('native bridge directory must be empty')
        if any(c.isspace() for c in str(self.directory)):
            raise ValueError('native line protocol needs paths without whitespace')
        self.calls = []
        self.solves = []
        self.worker = None
        self.closed = False
        self.setup = None
        self.log = (self.directory/'worker.log').open('w')
        before = time.perf_counter()
        try:
            self.worker = subprocess.Popen(list(map(str,command)), cwd=cwd, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
            self._record('launch.json', dict(pid=self.worker.pid, command=list(map(str,command)), started_unix=time.time()))
            if self.worker.stdout.readline().strip() != 'READY':
                raise RuntimeError('native BEM startup failed; inspect '+str(self.directory/'worker.log'))
            self.setup = json.loads((self.directory/'setup.json').read_text())
            self.dofs = int(self.setup['dofs']); self.rows = int(self.setup['constraint_rows'])
            if not 0 < self.rows <= self.dofs:
                raise ValueError('native BEM returned invalid dimensions')
            self.startup_wall_s = time.perf_counter()-before
        except BaseException:
            self.close()
            raise

    def _record(self, name, value):
        path = self.directory/name
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)

    def _send(self, request):
        if self.closed or self.worker.poll() is not None:
            raise RuntimeError('native BEM worker is not live')
        self.worker.stdin.write(request+'\n'); self.worker.stdin.flush()
        return self.worker.stdout.readline().strip().split()

    def action(self, name, value, *, scope='candidate'):
        if name not in ('A','AT','AH','C','CH','GUESS'):
            raise ValueError('unknown native BEM action')
        rows = self.rows if name == 'CH' else self.dofs
        output_rows = self.rows if name == 'C' else self.dofs
        value = np.asarray(value, dtype=np.complex128)
        if value.ndim != 2 or value.shape[0] != rows or not 1 <= value.shape[1] <= 64 or not np.isfinite(value).all():
            raise ValueError('invalid native BEM input shape/values')
        inp,out = self.directory/'input.bin',self.directory/'output.bin'
        out.unlink(missing_ok=True)
        before = time.perf_counter(); np.ascontiguousarray(value.T).tofile(inp)
        ident = len(self.calls); batch = value.shape[1]
        answer = self._send(f'{name} {ident} {batch} {inp} {out}')
        if len(answer) != 6 or answer[:2] != ['DONE',str(ident)]:
            raise RuntimeError('invalid native action reply: '+repr(answer))
        elapsed = float(answer[2]); counts = list(map(int,answer[3:]))
        if not np.isfinite(elapsed) or elapsed < 0 or min(counts) < 0:
            raise RuntimeError('invalid native action counters')
        if not out.exists() or out.stat().st_size != output_rows*batch*16:
            raise RuntimeError('incomplete native action result')
        result = np.fromfile(out,np.complex128).reshape(batch,output_rows).T
        if not np.isfinite(result).all():
            raise RuntimeError('nonfinite native action result')
        self.calls.append(dict(kind=name,scope=scope,columns=batch,native_s=elapsed,
            including_IO_s=time.perf_counter()-before,mixed_actions=counts[0],strict_actions=counts[1],adjoint_actions=counts[2]))
        self._record('action_progress.json',dict(completed_actions=len(self.calls),completed_solves=len(self.solves),last_action=self.calls[-1]))
        return result

    def solve(self, name, column, rhs, seed, prefix, tolerance=2e-6):
        if name not in ('P2_zero','P2_zero_repeat','P2_guess','P2_mbs025','P2_mbs1','P2_exact_far'):
            raise ValueError('unknown native solve variant')
        if not isinstance(column,int) or isinstance(column,bool) or column<0:
            raise ValueError('invalid modal column')
        if not np.isfinite(tolerance) or not 0<tolerance<=1e-3:
            raise ValueError('invalid native solve tolerance')
        rhs,seed = np.asarray(rhs,np.complex128),np.asarray(seed,np.complex128)
        if rhs.shape != (self.dofs,) or seed.shape != rhs.shape or not np.isfinite(rhs).all() or not np.isfinite(seed).all():
            raise ValueError('invalid native RHS/current seed')
        prefix=Path(prefix).resolve()
        if any(c.isspace() for c in str(prefix)):
            raise ValueError('native line protocol needs paths without whitespace')
        outputs=[Path(str(prefix)+suffix) for suffix in ('.currents.bin','.json')]
        if any(p.exists() for p in outputs):
            raise ValueError('refusing to replace a saved native solution')
        prefix.parent.mkdir(parents=True,exist_ok=True)
        rhs_file,seed_file=self.directory/'rhs.bin',self.directory/'seed.bin'
        before=time.perf_counter();rhs.tofile(rhs_file);seed.tofile(seed_file)
        answer=self._send(f'SOLVE {name} {column} {rhs_file} {seed_file} {prefix} {tolerance:.17g}')
        if answer!=['SOLVED',name,str(column)]:
            raise RuntimeError('invalid native solve reply: '+repr(answer))
        if not outputs[0].exists() or outputs[0].stat().st_size!=self.dofs*16:
            raise RuntimeError('incomplete native current')
        result=np.fromfile(outputs[0],np.complex128)
        record=json.loads(outputs[1].read_text())
        if not np.isfinite(result).all() or record['variant']!=name or record['column']!=column or \
           not np.isfinite(record['strict_residual']) or record['strict_residual']>tolerance:
            raise RuntimeError('native original-RHS acceptance failed')
        record.update(including_IO_s=time.perf_counter()-before)
        self.solves.append(record)
        self._record('action_progress.json',dict(completed_actions=len(self.calls),completed_solves=len(self.solves),last_solve=record))
        return result,record

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.worker is not None and self.worker.poll() is None:
                try:self.worker.stdin.write('QUIT\n');self.worker.stdin.flush()
                except BrokenPipeError:pass
                self.worker.wait(timeout=60)
        finally:
            if self.worker is not None:
                for pipe in (self.worker.stdin,self.worker.stdout):
                    if pipe is not None:
                        try:pipe.close()
                        except BrokenPipeError:pass
            self.log.close()
            self._record('execution.json',dict(setup=self.setup,calls=self.calls,solves=self.solves,
                startup_wall_s=getattr(self,'startup_wall_s',None),
                returncode=self.worker.poll() if self.worker is not None else None))

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
