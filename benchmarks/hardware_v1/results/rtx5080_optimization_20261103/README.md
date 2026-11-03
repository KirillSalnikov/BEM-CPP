# RTX 5080 exact-block assembly optimization — 2026-11-03

Machine: kika, RTX 5080 16 GB (`sm_120`), Ryzen 9 9950X3D,
8 OpenMP threads, one BLAS thread, WSL2, driver 621.01,
CUDA Toolkit 13.4.3 / nvcc 13.4.92, GCC 14.3.0.

The change removes repeated CPU work while the GPU solver and its tolerances
remain the same: material Green functions are shared across three kernels,
regular-quadrature bases are prepared once per supported element, and the test
function is contracted once before the loop over source basis functions.
FP64 arithmetic, the cancellation-safe Taylor branch, quadrature rules,
meshes and stopping conditions are preserved. This is an application-level
optimization measured with RTX 5080, not a new Blackwell-only GPU kernel.

## Paired measurements

The fixed ka=10 column has 16128 unknowns, two solved polarizations and all
181 angles / 16 Mueller elements. Baseline and candidate were remeasured
consecutively, each with one excluded warmup and five fresh-process repeats.
No checkpoint/operator cache reuse. All numerical and residual gates passed.

| Build | Median, s | Min–max, s | MAD, s (relative) |
|---|---:|---:|---:|
| Before | 4.242 | 4.124–4.381 | 0.118 (2.78%) |
| After | 2.954 | 2.840–2.969 | 0.015 (0.50%) |

Total speedup: **1.436×**, or **30.4% less wall time**.
Median preconditioner setup: **2.826 → 1.539 s**.
[Before](paired_baseline.json), [after](paired_optimized.json),
[machine-readable comparison](comparison.json). This compares software on the
same hardware; the source fingerprint correctly changes.

[The complete fixed suite](mixed.json) also passes, including all 48 T-matrix
columns / 24 solve pairs followed by FP64 analytic SO(3). Its median pipeline
time is 3.785 s versus 3.841 s in the initial machine report;
that small difference is not evidence of a meaningful total-pipeline speedup.
Native measured revisions: baseline `ee83342`, optimized `6b3d590`.

## Independent and larger controls

[Independent controls](independent_controls.json) compare 128 combined-kernel
cases with the original scalar formulas, including absorbing materials,
near-equal refractive indices and Taylor/non-Taylor distances. Six additional
controls compare the optimized principal blocks with the independent dense
Galerkin assembly for curved P2 and BDM1 bases, ka=0.001, 1 and 10, including
absorption. The worst block relative L2 difference is below 2e-16.

[Larger native controls](size_validation.json) use all 181 angles / 16 entries
and require both independently checked residuals below 1e-5:

| Case | Unknowns | Before, s | After, s | Max matrix difference on M11 scale |
|---|---:|---:|---:|---:|
| ka20-r3 | 16128 | 5.918 | 3.533 | 0.00236% |
| ka10-r4 | 64512 | 15.480 | 9.622 | 0.01588% |

These larger cases are single-run numerical/scaling controls, not five-repeat
performance scores. Matrix differences include the mixed-precision solver's
run-to-run variation; independent dense controls isolate the assembly change.
None of these checks establishes mesh-converged physical accuracy.

```bash
make tests/muller_kernel_bundle_check tests/muller_local_block_check
tests/muller_kernel_bundle_check
tests/muller_local_block_check
python3 scripts/benchmark_hardware.py run --profile mixed --output runs/optimized
```

Remaining work includes profiling the GPU solve and larger T-matrix workloads.
The existing mixed-precision GPU backend and analytic SO(3) were retained.
