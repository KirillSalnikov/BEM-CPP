Symmetry-reduced FMM and matrix-free coarse W, 2026-11-16
=======================================================

This is an optional memory-constrained backend in the experimental CPU Dirac
T builder. Dense SIMD remains the default and is faster on measured sizes.
The production Muller/FFT/GPU solvers and existing production queues were not
changed or restarted. New library: libdirac_optimized.so in this directory.

Implementation
--------------
* Store exact Galerkin-minus-13x13-point corrections for representative near
  panel pairs only. No dense far cache is constructed before compression.
* FMM3D evaluates eight Clifford charges for each medium at representative
  targets on iterations. Rotations and horizontal/vertical reflections are
  retained. Full residual audits evaluate arbitrary full densities, including
  unretained rows, at the stricter FMM epsilon; sampled rows are independently
  checked by fresh direct panel quadrature. This remains an approximate FMM
  audit, NOT an exact stored-matrix action or a spatial-convergence proof.
* Assemble the coarse Ac in bounded direct-quadrature tiles. Retain only its
  LU. W = B A Z - Z is not stored: apply Mx = B(x - AZ Ac^-1 Rx) + Z Ac^-1 Rx.
  The additional FMM action per inverse is a deliberate memory/time tradeoff.
* Full FP64 column and mirror residual checks, atomic checksummed checkpoints,
  and analytic SO(3) averaging remain active. No Euler orientation grid.
* Preflight accounts for LU, panel factors, T arrays and working reserves;
  near cache has a hard native budget. FMM3D internal workspace is a planning
  reserve, not an allocator-enforced process RSS limit.

Measured comparison: same SciIce column H/W_flats=1.6495721976846451,
n=1.3116, P3, target13/near24, near_subdiv1, C6 with mirrors, 16 CPU threads,
BLAS1. Two modal RHS, both polarizations, same sector, tol=1e-8.
Preparation and strict checks included; THESE ARE NOT COMPLETE T10/T20 RUNS.

                             dense          FMM p2, eps1e-10
ka10/ref2, 768 panels:
  operator cache, MiB        600.000         97.571
  coarse cache, MiB          156.006         36.006
  peak RSS, GiB              0.926           1.302
  entire experiment, s       5.844           24.964
  iterations per RHS         6,6             6,6

ka20/ref3, 3072 panels:
  operator cache, GiB        9.375           0.355
  coarse cache, GiB          2.438           0.563
  peak RSS, GiB              12.379          3.920
  entire experiment, s       55.900          98.845
  iterations per RHS         7,7             7,7

Selected lower-memory FMM profile: packet1, working eps1e-8, strict eps1e-12.
ka20 peak RSS2.946GiB, entire experiment90.404s, solve/check46.633s versus
12.243s dense. Peak reduction4.20x; total time1.62x longer. Do not describe
this backend as a speed win. The packet2 eps1e-10 profile is retained as a
more conservative numerical control, not as the smallest-memory profile.
Working selected-action error2.176e-11; full strict-action error1.377e-14;
solution-density L2 difference1.292e-10 against dense; max strict residual
5.721e-9. Iterations remain7,7. All are same-discretization comparisons.

The epsilon experiment rejects an unjustified relaxation to1e-6: on arbitrary
densities its measured action error5.644e-8 is above the solver target1e-8.
Default CLI working epsilon remains1e-10; use the measured1e-8 explicitly.

Fresh complete T at ka3/ref1/P2/L7 on prototype aspect1.1547005383792517:
126 columns, including56 mirrored. Both FMM profiles are compared with a
fresh dense run and all16 analytically averaged Mueller entries. Exact
numbers, residuals, times and RSS are in report.json. This validates wiring
and numerical agreement; it is not a ka20 mesh/L convergence certificate.

Ref4 capacity check at ka40 is OPERATOR ONLY: setup and two arbitrary full
checks plus selected actions. It excludes coarse LU, GMRES and T arrays.
See report.json / screen_ka40_fmm_p1_eps8.json for measured RSS and timing.
It does not establish that a complete T40 computation has finished.

Remaining limits
----------------
Ac LU is still dense: P1 one-sector storage at SciIce ref4 is about8.45GiB,
at ref5 about135GiB. Its preflight rejects oversized requests before near
cache allocation. Large meshes need a multilevel/compressed coarse solve.
Panel factors and near data still grow with mesh size. Dirac P3 ref7 is not
the former Muller ref7; no claim that ka80/ref7 now fits or meets1% accuracy.
Next speed priorities: reuse FMM tree/setup across calls and compress W to
avoid the extra full action; measured FMM time dominates near correction.

Reproduction from repository root (new output directory required):

  OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
  BEM_DIRAC_LIBRARY=$PWD/benchmarks/dirac_sector_fmm_20261116/libdirac_optimized.so \
  python3 benchmarks/clifford_dielectric_20261115/tmatrix.py \
    --shape prism --ka 3 --ref 1 --degree 2 --L 7 --order 24 \
    --target-order 13 --near-subdiv 1 --backend fmm-sector \
    --block-panels 1 --batch-size 8 --symmetry c6hv \
    --sector-coarse-basis 3 --fmm-packet 1 --fmm-eps 1e-8 \
    --fmm-strict-eps 1e-12 --so3 mstm --out /tmp/dirac_fmm_example

Native dependency: benchmarks/clifford_fmm_20261115/FMM3D/lib/libfmm3d.so.
measure.py, accuracy_screen.py, screen.py and extra.py contain the exact
screen/full-T commands. validate.py runs the CPU regressions, native tests,
checkpoint recovery tests and a byte-identical rebuild. summarize.py writes
report.json, selected.json and memory_vs_time.pdf. Measurements are sequential
on a shared workstation, not a dedicated exclusive benchmark host.
