# RTX 5080 baseline — 2026-11-03

Measured on **kika**, NVIDIA GeForce RTX 5080 (16 GB, native `sm_120`),
AMD Ryzen 9 9950X3D, **8 OpenMP threads**, one BLAS thread.
Windows/WSL2 Linux 6.6.114.1; driver 621.01; CUDA Toolkit 13.4.3
(nvcc 13.4.92), GCC 14.3.0. GPU clock/power settings were not changed.

One excluded warmup and **five fresh-process repeats** per case. The score is
external wall time including process startup, preparation and result output.
Every warmup/repeat passed completeness, numerical-reference and solver gates.
See the [fixed workload and protocol](../../README.md).

| Profile / case | Median, s | Min–max, s | MAD, s (relative) | Largest scaled matrix difference from reference |
|---|---:|---:|---:|---:|
| bem-prism-ka10-r3 | 4.334 | 4.199–6.352 | 0.081 (1.87%) | 0.0187109% |
| bem-tmatrix-ka1p5-r1-L4 | 3.841 | 3.426–3.911 | 0.069 (1.80%) | 1.48778e-07% |

The error column is the maximum across all 181 angles, all 16 matrix entries
and all six invocations, normalized by reference M11 at each angle. It is a
repeatability/regression measure, not an independent physical-accuracy estimate.
No slow repeats were discarded. Full samples, resource telemetry, commands,
compiler versions, binary/source/reference hashes and validation are in
[mixed.json](mixed.json), [CSV](mixed.csv).

## Native phase medians

These phases may overlap or be subsets of other phases; do not add them to
reconstruct total wall time.

| Profile / case | Native metric | Median, s |
|---|---|---:|
| bem-prism-ka10-r3 | farfield_s | 0.0012 |
| bem-prism-ka10-r3 | fmm_setup_s | 0.4969 |
| bem-prism-ka10-r3 | mbj_setup_s | 2.9226 |
| bem-prism-ka10-r3 | solve_s | 0.6963 |
| bem-tmatrix-ka1p5-r1-L4 | analytic_so3_s | 0.0127 |
| bem-tmatrix-ka1p5-r1-L4 | farfield_s | 0.0049 |
| bem-tmatrix-ka1p5-r1-L4 | mbj_setup_s | 0.3330 |
| bem-tmatrix-ka1p5-r1-L4 | operator_setup_s | 0.1801 |
| bem-tmatrix-ka1p5-r1-L4 | solve_s | 2.7047 |

## Reproduce and compare

```bash
python3 scripts/benchmark_hardware.py run --profile mixed --output runs/new_machine
python3 scripts/benchmark_hardware.py compare benchmarks/hardware_v1/results/rtx5080_20261103/mixed.json runs/new_machine/results.json
```

Keep the committed references, workload, profile and eight-thread setting.
A changed numerical-source or toolchain fingerprint is identified explicitly
as a software/toolchain-and-hardware comparison. MBS and BEM run different
models/workloads and these times must not be ranked against each other.

The measured native revision is `ee83342`.
Result/README publication commits can follow this revision without changing
the measured numerical sources. Each JSON records the complete source hash.

The T-matrix case completes all 48 columns (24 solve pairs) before analytic FP64 SO(3). It performs zero Euler quadrature nodes. NumPy/SciPy and all native pipeline components are fingerprinted in the JSON.
