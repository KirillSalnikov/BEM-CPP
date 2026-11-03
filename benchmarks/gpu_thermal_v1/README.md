# Minute-scale prepared GPU operator benchmark v1

This workload isolates repeated application of the real Muller operator on a
large prism: ka=20, n=1.3116, aspect 10/7, ref=5, 43,008 triangles and **258,048
unknowns**. Each repeat performs exactly **1536 resident device matvecs**;
alternating analytic inputs x and 2x prevent completed-result reuse. Setup and
transfers are excluded; no host input/output transfer occurs inside a measured
repeat. This tests the prepared operator, not a full BEM solve or T-matrix build.

Install using [the RTX 50 installer](../../docs/INSTALL_RTX50.md), which now
includes the `bem-gpu-thermal` component, then run on an idle GPU:

```bash
python3 scripts/benchmark_gpu_thermal.py --output runs/thermal-5080
```

Use `--prefix` to select an installation. A continuous **120-second warmup** is
excluded, followed by at least three fixed-work repeats totaling at least 180
seconds. On faster GPUs increase `--repeats`; do not reduce work per repeat.
`--pilot` performs an unscored 32-action calibration, not a hardware score.
The 1536-action workload was frozen after a pilot and before scored publication.

The native tool checks host/device agreement, agreement with its strict native
application path, linearity, zero input, and every repeat's complete output
vector. These are numerical regression controls, not independent BEM physical
convergence tests. Currents and combination use FP64; this mixed build uses FP32
near interactions and selected far-field storage/caches. The strict native path
control must not be interpreted as an all-FP64 independent implementation.

The wrapper rejects scores unless GPU utilization has median >=95% and tenth
percentile >=90% in **every** repeat. CUDA-event spans include host submission
gaps; they are not presented as pure kernel-active time. Per-second telemetry,
CPU usage/RSS, setup time, native binary and workload source hashes, toolchain,
correctness errors and exact commands are retained in `results.json`.

Compute PIDs are checked continuously. Any foreign/unidentifiable compute job
rejects the score. The optional `--paused-processes LEASE.json` accepts only
explicitly leased PID/start-time identities that remain stopped throughout the
test; the runner itself never suspends jobs. The published run used an operator
controller with automatic resume and a watchdog; all production jobs resumed.

```bash
python3 scripts/benchmark_gpu_thermal.py compare baseline/results.json candidate/results.json
python3 benchmarks/gpu_thermal_v1/plot_results.py runs/thermal-5080/results.json
```

Only equal workload/protocol keys can be compared. Changes to numerical or
workload-tool source and recorded toolchains are distinguished from pure
hardware/platform differences. The short [hardware suite](../hardware_v1/README.md)
continues to measure complete solves and T-matrix pipelines separately.

[RTX 5080 results, checks and limitations](results/rtx5080_20261103/README.md) ·
[Independent Green-function audit](../../docs/GREEN_FUNCTION_AUDIT_20261103.md).
