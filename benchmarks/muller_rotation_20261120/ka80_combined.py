"""Check the production host CSR on all six characters, then the full target action."""
import ctypes
import json
import os
import subprocess
import tempfile
from time import perf_counter

import numpy as np
import torch

from ka80_action import B, HERE, load_cpu, project
from prism_fft_rotation import local_rotation_rows


class Near:
    def __init__(self, original, folder, permutation, sign):
        lib = os.path.join(folder, "local.so")
        subprocess.run(["g++", "-O3", "-march=native", "-std=c++11", "-fopenmp",
                        "-shared", "-fPIC", str(HERE.parents[1]/"scripts/packed_current_cpu.cpp"), "-o", lib], check=True)
        self.lib = ctypes.CDLL(lib)
        self.full = self.lib.bem_packed_current_cpu
        self.full.argtypes = original.apply.argtypes
        self.full.restype = None
        self.rotated = self.lib.bem_packed_current_rotated
        self.rotated.argtypes = self.full.argtypes + [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p]*3
        self.rotated.restype = None
        self.original = original
        self.tables = local_rotation_rows(permutation, sign)

    def begin(self, x, mode, host_input=None):
        self.y = torch.empty_like(x)
        array = self.original
        args = (x.numel()//2, array.rows.ctypes.data, array.cols.ctypes.data,
                array.values.ctypes.data, x.data_ptr(), self.y.data_ptr(), 16)
        if mode < 0:
            self.full(*args)
        else:
            reps, owner, ids, phases = self.tables
            self.rotated(*args, reps.ctypes.data, len(reps), owner.ctypes.data,
                         ids.ctypes.data, phases[mode].ctypes.data)

    def finish(self):
        return self.y


def main():
    torch.set_num_threads(4)
    op = load_cpu()
    folding = op.rotation_folding
    perm, sign = folding.perm.numpy(), folding.sign.numpy()
    report = dict(near=[], full=[])
    target = HERE/"ka80_combined.json"
    def save(row, section):
        print(row, flush=True)
        report[section].append(row)
        target.write_text(json.dumps(report, indent=2)+"\n")
    def rel(a, b):
        return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))
    def vector(label):
        return np.fromfile(B/"native_ka80"/(label+".bin"), complex)
    def modal(x, m):
        return torch.as_tensor(project(x, perm, sign, m) if m >= 0 else x, dtype=torch.complex64)
    schedules = json.loads((B/"spectral_traffic_screen.json").read_text())
    with tempfile.TemporaryDirectory() as folder:
        near = Near(op.local, folder, perm, sign)
        op.local = near
        op.local_storage = "host_packed"
        for mode in range(6):
            x = modal(vector("random"), mode)
            if folding.detect(x) != mode:
                raise ValueError("rotation detection failed")
            times = [[], []]
            for folded in (False, True, True, False):
                t = perf_counter()
                near.begin(x, mode if folded else -1)
                y = near.finish()
                times[folded].append(perf_counter()-t)
                if folded:
                    err = rel(y, reference)
                else:
                    reference = y
            save(dict(m=mode, seconds_full=times[0], seconds_folded=times[1], relative_error=err), "near")
            if not np.isfinite(err) or err > 5e-4:
                raise ValueError("near rotation mismatch")
        for label, mode in [("random",1), ("smooth",0), ("random",-1), ("smooth",-1)]:
            x = modal(vector(label), mode)
            y = modal(vector("current_depth6_"+label), mode)
            for candidate in ((False,True,True,False) if mode==1 else (True,)):
                op.rotation_folding = folding if candidate else None
                for i, schedule in schedules.items():
                    ranks = next(r["batch_ranks"] for r in schedule["rows"] if r["tol"]==.0005)
                    op.groups[int(i)]["streamed"].ranks = ranks if candidate else None
                t = perf_counter()
                z = op(x)
                elapsed = perf_counter()-t
                err = rel(z, y)
                save(dict(input=label, m=mode, candidate=candidate, seconds=elapsed, relative_native_error=err),"full")
                if not np.isfinite(err) or err > 5e-4:
                    raise ValueError("full rotation mismatch")


if __name__ == "__main__":
    main()
