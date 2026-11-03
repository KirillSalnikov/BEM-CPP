# Hardware benchmark v1

This suite measures repeatable application workloads on different CPU/GPU
systems. It is not an equal-accuracy speed comparison between MBS and BEM:
they solve different mathematical models and the workloads differ.

## Run and compare

Install/build the program first. The RTX installer places supported binaries
and provenance in `~/.local/opt/mbs-fast` or `~/.local/opt/bem-cpp`.

```bash
# MBS-fast checkout: mixed FP32 storage, precise FP64 phase math
python3 scripts/benchmark_hardware.py run --profile gpu-fp32 --output runs/hw_mixed
# MBS: FP64 GPU or CPU/OpenMP, same fixed optical workload
python3 scripts/benchmark_hardware.py run --profile gpu-fp64 --output runs/hw_fp64
python3 scripts/benchmark_hardware.py run --profile cpu --output runs/hw_cpu

# BEM-CPP checkout: native mixed-precision GPU and complete T/SO(3) pipeline
python3 scripts/benchmark_hardware.py run --profile mixed --output runs/hw_bem
```

Only the commands for the current repository apply. Use `--prefix` to select
another installer prefix, or `--binary` to select a native build. A matching
source tree must remain available. BEM locates the Python environment from the
installed `bem` wrapper; `--python /path/to/venv/bin/python` overrides it while
preserving the virtual environment. The runner itself uses only the standard
library. `--case CASE_ID` selects one BEM subtest.

Defaults: **8 OpenMP threads, one excluded warmup, five measured repeats**.
At least three measured repeats and one warmup are required. Use the same
thread count, profile and case definition when comparing equipment.

```bash
python3 scripts/benchmark_hardware.py compare baseline/results.json other/results.json
```

A speedup of 2 means that the candidate took half the baseline median wall
time. Different numerical source hashes or recorded toolchains are labeled
software/toolchain-and-hardware comparisons. Incompatible work, precision,
thread count or frozen reference is rejected. Compiler, driver and OS changes
remain visible; this is an application/platform benchmark, not an isolated
measurement of silicon throughput.

## Frozen workloads

All cases use a regular six-sided prism, real refractive index 1.3116 and
height / width across opposite faces = 10/7. The MBS vertex-to-vertex diameter
is converted explicitly; it is not the width across faces.

| Case | Fixed work |
|---|---|
| `mbs-prism-ka10-2048` | ka=10, wavelength 0.532 µm, height 1.8162922498144123 µm, vertex diameter 1.4680915471003348 µm, 2048 seeded Owen-Sobol/Haar orientations, seed 314159, depth 8, all cutoffs off, chunk 64, 64 azimuths, 181 theta points, all 16 Mueller entries |
| `bem-prism-ka10-r3` | ka=10, refinement 3, H(div)-BDM1, 16128 unknowns/2688 triangles, two independently solved polarizations, MBJ 50 nodes, FMM digits 5/leaf 64/near radius 3, quadrature 7/Duffy 4, tolerance 1e-5, restart 100, limit 500, 181 theta points/all 16 entries |
| `bem-tmatrix-ka1p5-r1-L4` | ka=1.5, refinement 1, 1008 unknowns, complete L=4 T matrix (48 columns, 24 solve pairs), tolerance 1e-5, restart 100/limit 500, FP64 analytic SO(3), 181 theta points/all 16 entries, zero Euler quadrature nodes |

Native defaults not exposed by a pipeline's CLI are constrained by its
numerical-source hash and checked operator metadata. `suite.json` and the
frozen reference data define version 1. Changing physical inputs, the mesh,
sampling, tolerances or precision requires a new case/version, not replacing
old results under the same name.

## Timing protocol

Each warmup and repeat starts a **fresh process in a fresh directory**. Solver
checkpoints and completed columns cannot be reused. The OS file cache and
CUDA driver/JIT cache are warmed; the benchmark does not claim a cold-boot or
cold-driver measurement. One case's matrix is never reused as measured work.

Reported metrics include median, min/max, mean and median absolute deviation
(MAD). Relative MAD above 10% is flagged as unstable. Individual repeats and
excluded warmups are retained. End-to-end wall time is measured externally;
native preparation, solving, diffraction and far-field/SO(3) timings are
reported separately. Native phases can overlap or be subsets and must not be
blindly added together. MBS phase profiling and GPU timing are enabled for
every measured run, consistently across systems.

CPU user/system time and peak child RSS come from `wait4`. GPU temperature,
utilization, memory, power and clocks are sampled once per second when the
driver exposes them. Unsupported WSL counters remain null. Sampled VRAM is
not a guaranteed transient peak. No clocks, power limits or display drivers
are modified. A busy GPU is rejected by default; `--allow-busy-gpu` records an
explicit exception, and such runs should not be used as clean baselines.

All inherited MBS/BEM/CUDA/OpenMP tuning variables are cleared. The suite sets
8 OpenMP threads with spread/core binding (unless the count is explicitly
overridden), one BLAS thread, eager CUDA
module loading and a run-local driver cache. Device kernel selection remains
the native architecture-aware implementation, and the program/build versions
are recorded. This suite is not restricted to RTX 50 cards.

## Numerical gates and publication

Every curve must contain exactly 181 angles from 0 to 180 and all 16 finite
matrix entries with positive M11. MBS additionally checks the exact orientation
count, zero trace-limit hits/incomplete orientations, and actual GPU timing.
BEM verifies independently recomputed residuals, the exact mesh size, both
polarizations, all requested T columns and solve pairs, and FP64 analytic SO(3).

Every warmup and measured run is compared with a frozen matrix. Max angular
error uses `abs(delta Mij) / max(M11_reference(theta), global_M11*1e-12)`;
the full matrix also has an L2 gate. Limits are:

| Profile | Maximum scaled error | Relative L2 |
|---|---:|---:|
| MBS GPU FP32 | 2e-4 | 5e-5 |
| MBS GPU FP64 / CPU | 1e-6 | 1e-7 |
| BEM mixed | 5e-3 | 1e-3 |

These gates establish regression equivalence for the fixed discrete problem,
not an independent physical-accuracy or mesh/orientation-convergence certificate.
Failed, partial or unvalidated solves are not ranked. Run/native-binary/source,
reference and case hashes are recorded to prevent silent workload changes.

The committed `reference/` matrices were generated once on the documented
RTX 5080 installation (CPU reference is identified separately). Regenerating
references is a maintainer action, excluded from measured scores:

```bash
python3 scripts/benchmark_hardware.py reference --profile gpu-fp32 --output /new/reference_folder
```

This command refuses to overwrite a reference. It does not produce a validated
timing score. Keep the published references for subsequent hardware tests.
Published machine results are under `results/`. Large operator caches, T-array
checkpoints and intermediate run directories are intentionally not committed.
