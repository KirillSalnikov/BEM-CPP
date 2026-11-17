# Optional GPU working FMM with independent CPU checks

Raw filenames below identify local experiment receipts; they are not all
distributed. The portable ka80 CPU/GPU measurements are in
[reference/performance_20261117](../../reference/performance_20261117/README.md).

Ryzen 7950X / RTX 3090 Ti, SciIce column, refractive index 1.3116,
height / width across flats 1.6495721976846451. These are operator and
small complete-T experiments, not a completed ka80 T calculation.

## What changed

- Three scalar potentials and their gradients share a multilevel Helmholtz
  traversal. Eight Dirac components are processed as 3 + 3 + 2.
- Near interactions have an opt-in two-float compensated kernel. Coordinates
  are subtracted in FP64 before conversion; far translations remain FP64.
  This is not a plain FP32 residual check.
- Fine ref7 meshes use adjacent depth bands 7/6/5/4/3; the final band has
  enough expansion terms for the electrical size of the particle.
- The scalar bridge releases unused Hessians, fourth-channel multipoles,
  and generic complex upload buffers. Two medium plans remain resident.
- Very small electrical cells are avoided by reducing the requested depth
  until |k| times cell width is at least 1.4, or depth reaches one. This is
  an empirical accuracy guard. Independent strict CPU audits are mandatory.
- The expert T driver can select CUDA working actions. Strict full-surface
  residual checks still use the original CPU FMM at epsilon=1e-12.
  CPU checks and GPU iterations are currently sequential, not overlapped.

The optional library does not replace the production CPU FMM library or
change the default working device. Existing Muller vector kernels are not
redirected to this path.

## Measurements

`operator_ref7.json`: actual ka80/ref7/P3, 758,784 panels, 9,864,192 sources,
60,702,720 full unknowns, 5,058,560 sector unknowns. Includes exact near
correction and P3 projection for both media, with GMRES(60) memory allocated.

| Working action, 32 CPU threads | Seconds |
|---|---:|
| Optimized CPU, component packet 8 | 65.448 |
| GPU, first action after setup | 29.358 |
| GPU, warm action | 29.154 |

Matched speedup 2.245; relative action difference 4.149e-11. GPU preparation
31.38 s, geometry/near/factor/workspace preparation 42.93 s. Two GPU plans
allocate 19.955 GiB after lazy allocations; nvidia-smi observes 20,719 MiB
total. The comparison process peaks at 37.03 GiB RSS, including the CPU
baseline and retained reference output. This is not the peak of a complete
T driver, and no independent GPU-only host-memory peak is claimed.

`operator_ref5.json`: same ka80 and material, P3/ref5. Current CPU settings
20.58–21.04 s versus GPU 4.13–4.23 s. The CPU baseline here is the driver's
current leaf choice, not a new search for its best leaf size. Random currents
and both plane-wave polarizations were also compared with a full CPU strict
action: differences 3.63e-11, 3.98e-11 and 4.60e-11. This verifies arithmetic,
not spatial convergence of ref5 at ka80.

`scalar3_ref7_fullbands.json`: one three-component packet, potential and
gradient, GPU 4.073 s versus strict CPU 22.991 s. Errors 4.34e-11 and 7.10e-11.
Do not use this packet speedup as the speedup of the complete Dirac operator.

`validation.json` / `T126/report.json`: full ka3/ref1/P2/L7 T, 126 columns,
analytic SO(3), zero Euler quadrature nodes. Relative T difference 2.06e-14;
maximum difference of all 16 averaged elements divided by CPU M11 is
4.52e-14. Same 350 iterations, maximum strict residual 8.205e-9. This tiny
case is slower on GPU: 101.12 s total versus 61.48 s CPU. It is a correctness
test, not a speed claim. GPU launch/transfer overhead matters at small size.

`finalize.py` reruns the independent GPU oracle, complete T and large ref7
action after the low-frequency guard was finalized. Its separate outputs
are `validation_final.json`, `T126_final/` and `operator_ref7_final.json`.
The final report includes the exact GPU build manifest and actual tree depth.
Final verification completed: depth7 in both media, CPU 66.415 s versus GPU
29.185 s (2.276 times), action error 4.149e-11, same 19.955 GiB plan memory.
The repeated full T has error 2.061e-14, all16/M11 error 4.460e-14, and the
same 350 iterations. Its total time is 100.24 s. `measured_report.json`
collects these final values; regenerate it with `python3
benchmarks/dirac_gpu_fmm_20261116/summarize.py`.

## Rejected variants and limits

- Depth7 with only three expansion bands gave relative errors 7.14e-4 in
  potential and 1.28e-4 in gradient (`scalar3_ref7_depth7.json`). Rejected.
- Small electrical cells caused loss of accuracy even in the original FP64
  near kernel. Direct-sum tests retain their 2e-9 bound; the tree-selection
  fix does not relax that bound. Low-frequency screens and first failures
  are retained in the logs.
- The large ref7 GPU action is compared against the independently audited
  CPU working operator. A new full strict ref7 pass was not performed here;
  full strict comparisons were performed at ref5 and in the complete small T.
- The near kernel, tolerances and large-case speedup are not a general
  accuracy certificate for arbitrary geometries or refractive indices.
- No weekly T80 runtime claim: the number of columns, iterations, full
  residual checks, memory of coarse corrections and mesh/L convergence still
  need to be addressed. The RAM planner keeps conservative reserves.

## Build and reproduce

The isolated build currently targets sm_86 and uses nvcc with g++-12:

```bash
python3 benchmarks/dirac_gpu_fmm_20261116/build.py
# Numerical oracle, only on an available compatible GPU:
BEM_TEST_GPU_FMM=1 python3 -m unittest discover -s tests -p test_dirac_scalar_fmm_gpu.py
```

Completed bounded runs are retained rather than rerun. Use fresh output
directories when intentionally measuring a changed implementation.
`build.json` hashes sources, headers and the generated library; the T driver
rejects a stale build and records those hashes in checkpoint configuration.

For an existing expert sector-FMM T command, add:

```text
--fmm-working-device cuda --gpu-fmm-depth 7
--fmm-components 8 --fmm-reference-components 4 --fmm-packet 1
```

Depth7 is the measured ka80/ref7 case, not a universal setting. The driver
defaults to CPU; CUDA requires `--backend fmm-sector`. Never remove the RAM
or strict-residual guards to force a large run.

The direct GPU oracle covers real and absorbing k, shifted coordinates,
coincident source/target exclusion, eight channels, all three near kernels,
both direct-only and genuine far translations. 39 CPU tests and 21 frontend
tests have separate logs here. Small-test telemetry can miss subsecond GPU
allocations; do not interpret its low sampled VRAM as the actual plan size.
