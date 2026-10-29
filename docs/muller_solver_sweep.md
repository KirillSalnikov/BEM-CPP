# Streaming solver experiments, 2026-10-28/29

The new diagnostic driver `muller_solver_fft_probe` compares matrix-free
solvers on the same regular D6h prism, material, right-hand sides and
original FP64 FMM residual. It does not export a T matrix or certify mesh
or modal-cutoff convergence. Defaults are `n=1.3116`, residual tolerance
`2e-6`, quadrature 7, Duffy order 6, and two oblique plane-wave right-hand
sides. Setup, one-off polynomial training, solve time, actual FMM/FFT
actions, and solution differences are recorded separately.

```bash
make -j4 bin/muller_solver_fft_probe tests/streaming_flexible_check
tests/streaming_flexible_check
python3 benchmarks/week_ka100_20261028/run_solver_probe.py example \
  --ref 5 --ka 40 --restart 80 --methods baseline,nested8,poly4,idr4
python3 benchmarks/week_ka100_20261028/report_solvers.py
```

The bounded launcher writes command/binary identity, environment, resource
samples, and a persistent status. It is a diagnostic runner, not a
resumable production solve; each completed RHS is flushed to JSONL.
Do not run simultaneous GPU benchmarks.

## Implemented candidates

- `baseline`: existing fixed-right-preconditioned host GMRES with local MBJ.
- `nested4/8/12/20`: flexible GMRES with at most that many inner MBJ-pFFT
  GMRES steps, stopping the inner cycle at relative residual 0.1.
- `poly4/8/12`: train a GMRES inverse polynomial once on a seeded random
  vector, then reuse its coefficients on subsequent right-hand sides.
  The number denotes training steps; the inverse polynomial has degree
  at most one less. Evaluation uses the Arnoldi recurrence, not monomials.
- `idr4/8`: complex IDR(s)-biortho with fixed local right preconditioning.
  An iteration in the report is one matrix action, not an entire s+1 cycle.
  Strict residual replacement restarts the recurrence; a recursive residual
  alone can never accept a solution.
- `baseline_fast`: same GMRES and strict operator with the optional real-k
  FP32 P2P fast-trigonometry kernel enabled for inner actions.
- `tfqmr` / `tfqmr_circ`: fixed-right-preconditioned complex TFQMR with MBJ
  or the quarter-weight circulant inverse. It requires only forward actions.
  The quasi-residual bound triggers a full FP64 residual check; it cannot
  accept a solution by itself. Periodic residual replacement restarts the
  recurrence. `BEM_SOLVER_VERIFY_INTERVAL` counts actual mixed FMM actions.
  This implementation is diagnostic-only, with tests in
  `tests/streaming_tfqmr_check.cpp` for nonnormal complex systems, perturbed
  inner actions, warm starts, iteration limits, breakdown and zero RHS.

`BEM_PROBE_MBJ_NODES` and `BEM_PROBE_MBJ_OVERLAP` select the existing local
block inverse and its restricted overlap. The launcher exposes these as
`--nodes` and `--overlap`. `--restart` defaults to 80 to match the large-run
driver. Initial screening also used restart 32; the report preserves this
distinction. A smaller outer iteration count is not sufficient evidence
of acceleration when each inverse application contains multiple FFTs.

Flexible mode is opt-in in `streaming_host_gmres.h`; it retains each actual
preconditioned Arnoldi vector. It rejects recycling, whose current fixed
inverse implementation cannot be reused unchanged. The ordinary fixed
inverse path retains its previous memory layout and behavior. Polynomial
and IDR prototypes are available in the diagnostic driver, not selected
automatically by the production T worker.

## Measured scope and memory limit

For `ka=40, ref=5`, there are 202,752 unknowns. Across two RHS, GMRES(32)
took about 61 seconds per RHS, and nested FFT with eight inner steps took
about 47 seconds. IDR(4) and IDR(8) took about 83 and 70 seconds respectively.
The full data, including the separate restart-80 controls, are in
`benchmarks/week_ka100_20261028/solver_results.csv` and `solver_summary.json`.
These are solve-stage times with cached/lazy operator initialization warmed.
For thousands of T columns, setup is amortized; for one RHS it must be added.

On the full ka=100/ref=7 operator (3,225,600 unknowns), an ABBA control
measured 4.70 seconds with 512-thread P2P blocks versus 4.36 seconds with
128-thread blocks, retaining the same 32 leaf partitions. The large-run
profile now selects 128 threads (about 7.2% less operator time). The optional
real-wave fast-trigonometry variant remains explicit: its full operator
ABBA control showed about 8.5% less time with a relative action difference
of 3.81e-7. On both ka=40 RHS, enabling it preserved iteration counts and
strict acceptance; solution differences were below 1.60e-7. It is not
enabled automatically by this profile change.

This does not demonstrate a 1% Mueller result on these meshes or a
week-long ka=100 calculation. It measures linear-solver accuracy and cost.
The proposed solvers still solve incident columns; the SO(3) averaging
algorithm is not replaced by orientation quadrature.

The current full 3D pFFT implementation does not fit alongside the measured
ka=100/ref=7 FMM configuration on a 24 GiB card. At interpolation order 2,
the estimated padded grid is 315 x 360 x 315. The two media's spectra,
basic FFT buffers, and interpolation stencils alone need about 10.93 GiB,
before near corrections, point buffers and cuFFT workspace. Added to the
measured 20.5 GiB FMM peak this is already about 31.4 GiB. The calculation is
recorded in `pfft_ka100_memory_bound.json`. This bound concerns existing 3D
pFFT, not a prospective 2D face circulant inverse.

## QMR and restart controls on the target aspect 10/7 prism

The later ka=40/ref=5 mesh has 258,048 unknowns (its aspect differs from the
older sweep above). One identical oblique RHS, MBJ50 plus circulant 0.25,
and the same final FP64 tolerance 2e-6 gave:

| method | mixed operator actions | solve seconds | strict residual |
| --- | ---: | ---: | ---: |
| GMRES(32), strict check every 4 cycles | 160 | 62.570 | 1.585e-6 |
| GMRES(64), strict check every 2 cycles | 163 | 64.074 | 9.996e-7 |
| GMRES(128), strict check every cycle | 162 | 64.475 | 9.531e-7 |
| TFQMR, replacement interval 128 actions | 256 | 93.996 | 3.497e-7 |
| TFQMR, replacement interval 500 actions | 257 | 94.552 | 5.135e-7 |

All five solves used two full strict actions. The TFQMR far field in the
first comparison differed from GMRES by relative L2 1.837e-6 over 721 angles.
GMRES(32) spent 50.15 s in mixed actions, 9.54 s in strict checks, and only
0.38 s in orthogonalization. Thus the short recurrence saved little compared
with the cost of its additional FMM actions. This single-RHS result rejects
these settings for this run; it does not establish general superiority of
GMRES over QMR. General QMR needs transpose/adjoint actions, which the
current operator interface does not expose; TFQMR avoids that requirement
([PETSc documentation](https://petsc.org/release/manualpages/KSP/KSPTFQMR/)).

Raw commands, binary hashes, resource samples, fields and results are under
`runs/qmr_hotspot_20261029`; the repeatable launcher is
`benchmarks/qmr_hotspot_20261029/run_probe.py`. On ref=6, two warm mixed actions
took about 0.88/0.89 s and full strict actions about 16.9 s. CUDA stage timing
for the two concurrent media includes contention and must not be summed as
exclusive durations. A forward/reverse sweep over P2P thread/split settings
found only about a 1% action-time difference between the best setting and
the existing 128-thread/32-split setting; no new default follows from that.

The optional mixed-kernel setting `BEM_FMM_P2P_LOCAL_COORDINATES=1` subtracts
the first target in each leaf as a common double-precision origin before
rounding coordinates to float. Pair arithmetic stays FP32 and the strict
kernel is unchanged. It passed the 144 original P2P cases plus 54 small-cloud
cases translated by 0 and +/-10000, compared directly with FP64 (maximum
relative error 5.351e-6). `tests/p2p_shared_check` exercises empty leaves,
coincident points, incomplete tiles, multiple thread/split settings and
absorbing media. On ref=6, the full solve still took 190 actions and 218.567 s
versus the earlier 220.357 s, with strict residual 9.875e-7. This approximately
0.8% timing difference is insufficient to justify a production change and
does not demonstrate fewer iterations. The ref=5 trial combining local
coordinates with check interval 8 took 170 actions/65.691 s, also slower
than the production control. Both settings remain experimental and disabled
in the frozen ka=40 worker.

## Explicit transpose/adjoint and classical QMR

`MullerFmmOperator` now exposes host `matvec_transpose`, `matvec_adjoint`,
`matvec_transpose_strict` and `matvec_adjoint_strict`. Arrays retain the same
`[J,M]` ordering and `system_dofs` length; in-place calls are supported.
The strict variants temporarily select FP64 and restore the previous mode,
including on exceptions. These methods currently require the FMM backend;
pFFT and the optional face-FFT backend are explicitly rejected.

The implementation reverses the actual discretized Galerkin/FMM chain:
derivative L2P, within-tree L2L/M2L/M2M, transfers between angular grids,
P2M, near interactions, projection/assembly and sparse Duffy corrections.
It does not assume that the assembled Muller matrix is symmetric. For the
multilevel transfers `U = S- I` and `D = Wf^-1 I^T Wc S+`, it applies
`U^T = I^T S-` and `D^T = Wc S+ I Wf^-1`. The derivative contractions and
Helmholtz prefactors move to the reversed L2P. M2L uses cached rows grouped
by the original source, avoiding an atomic update for each interaction.
The near list is explicitly reversed, including self-leaf interactions;
different target/source point sets are supported by the lower-level API.
Existing expansion buffers are reused. Additional interaction indices and
work vectors are sparse/linear storage, not a dense system matrix.

Complex adjoints use `conj(A^T conj(x))`, including material coefficients.
FP64 transposes match the implemented traversal to floating-point rounding;
mixed actions have FP32 accumulation/rounding differences and still require
the original FP64 residual gate. They are not substituted for the strict gate.

`MullerMbjPreconditioner` supplies transpose and adjoint solves, including
LU pivots, stored inverses, restricted overlap and its optional coarse
correction. Restriction masks the input of the transposed local solve and
overlapping outputs are summed. `BlockCirculantFFT` supplies both operations;
`PrismCirculantInverse` also reverses the distinct equation/solution signs
and the row-dependent blend. Its adjoint accepts the local inverse's adjoint
callback explicitly.

The diagnostic methods `qmr`, `qmr_circ`, and `qmr_strict_circ` implement
complex QMR without look-ahead, following the mathematical recurrence in
[SIAM Templates, Figure 2.8](https://www.netlib.org/templates/templates.html).
For fixed right inverse P, Lanczos uses `B=A P` and `B^H=P^H A^H`; solution
updates retain physical directions `P p`. Both forward and adjoint A calls
count towards the action limit and are reported separately. Full residual
replacement restarts the recurrence; numerical breakdown never accepts an
unchecked solution. This solver is available in the probe, not automatically
selected by the frozen production T worker.

`make cuda-adjoint-check` runs the QMR, MBJ, circulant and rectangular FMM
transpose checks sequentially (requires a CUDA GPU).
`benchmarks/qmr_adjoint_20261029/run_probe.py` exercises complete BEM identities
and solver comparisons. Controls include an explicitly assembled 648-square
complex FMM matrix, absorbing media, unequal source/target sets, multilevel
transfers, and the composite circulant inverse. Raw commands, binary hashes,
timings and error metrics are retained in `runs/qmr_adjoint_20261029`.

The explicitly assembled absorbing ref=0 control (648 unknowns) agreed with
the transpose action to relative vector error 9.03e-16. On the full ka=40,
ref=6 mesh (1,032,192 unknowns), normalized FP64 transpose/adjoint dot-test
errors were 1.28e-17 / 8.16e-18; mixed errors were 2.17e-9 / 5.83e-10.
These dot tests check discrete duality, not physical discretization accuracy.
The absorbing ref=3 multiband mixed check had larger errors, up to 4.24e-6;
its FP64 checks remained below 8.5e-15. Rectangular point-set, MBJ, circulant
and CPU QMR test executables also passed independently.

The completed ka=40 solver comparison used ref=5 (258,048 unknowns), the
same incident RHS, MBJ50 plus circulant blend 0.25, and FP64 tolerance 2e-6:

| Solver | Forward / adjoint actions | Solve seconds | Strict residual |
| --- | ---: | ---: | ---: |
| GMRES(32) | 160 / 0 | 65.120 | 1.585e-6 |
| QMR, residual replacement every 128 total actions | 220 / 220 | 295.821 | 9.831e-7 |

The far-field relative L2 difference was 1.39e-6 over 721 sampled angles.
QMR spent 197.48 s in adjoint actions: the new mixed reverse traversal is
about 2.8 times as expensive as the forward traversal at this size. It also
needed more forward actions. This is evidence against this QMR configuration,
not a proof that every QMR variant or another solver must lose. Ref=6 was
used for operator and inverse identities only, not a full QMR solve.

The frozen ka=40 production worker was resumed unchanged from 2,278 saved
columns / 109 committed packets. It retains GMRES(32), incident prefetch,
C6/reflection packing, C2 reuse and analytic SO(3) averaging. Measurements,
source hashes, checkpoint handover and resume records are collected in
`benchmarks/qmr_adjoint_20261029`.

These observations motivated independent Krylov iteration in the
C6/horizontal-reflection sectors, implemented and measured below. Its checks
measure `A S_solution - S_rhs A` and `P S_rhs - S_solution P`; the equation and
current reflection signs differ. The cap's rectangular circulant extension
and arbitrary MBJ partition do not preserve these sectors in the measured
controls. Orbit-consistent local blocks avoid averaging twelve complete
local inverse applications.
Current packing already combines up to twelve sectors in one RHS, so a
reduced-domain implementation must be compared by total completed T-packet
time; twelve sectors do not imply an additional twelvefold speedup. Generic
rank-32 GCRO/two-level tests at ka=112 previously reduced the residual only
about 1.3% at equal 32-step budgets while costing 2.4% / 3.2% more time
(`benchmarks/global_solver_20261027/summary.json`); those were truncated
diagnostics, not converged ka=40 or symmetry-sector recycling comparisons.

## Orbit-local inverse and independent symmetry sectors

`PrismOrbitInverse` partitions representative edge orbits in one C6 / horizontal
reflection wedge, assembles and factorizes each local block once, and applies
its signed images throughout the mesh. Both BDM1 edge moments and both currents
are retained together. Edges fixed by the mirror have shorter orbits; duplicate
images are averaged with their orbit multiplicity. Separate seed blocks own
disjoint edge orbits, so parallel applications have no scatter races. The
equation and current signs are distinct, as required by
`P S_rhs = S_solution P`. This construction addresses the commuting subgroup
`C6 x sigma_h`, not every element of the full noncommuting D6h group.

`PrismSymmetrySectors` builds a unitary signed Fourier basis on each discrete
edge-moment/current orbit, including stabilizers. It stores the twelve sectors
contiguously. Equation and solution coordinates have opposite horizontal
parity labels. `streaming_sector_gmres.h` keeps independent Arnoldi/Hessenberg
recurrences in these coordinates while combining their vectors into a single
full-surface FMM action. It does not perform twelve full FMM solves per step.
The solve uses `B = U_rhs^H A P U_rhs`; the physical result is `P U_rhs y`.
Completed sectors stop contributing to batched actions. Restart residuals can
be carried without an FP64 action, with the same bounded check interval as the
global solver. Every accepted solve is checked with the original complete
FP64 operator, including any small symmetry defects of the numerical FMM.

`PrismSymmetricCirculant` applies the orbit-local inverse once and averages
only the FFT contribution over the twelve group elements. The covered-row
blend mask is also averaged; symmetrizing the FFT term alone is insufficient.
The pure FFT action is checked against the original composite inverse's
decomposition. A structural check verifies whether rotation/reflection map
entire side patches to identical channel/grid slots with matching input and
output signs, and whether reflection does the same for the cap patches. If
so, side FFTs are applied once and only the caps are averaged over six
rotations. Otherwise the full twelve-element average is retained. The fast
action is also compared numerically with that complete average. This shortcut
does not rely on a small sampled commutator alone. The ordinary forward
circulant implementation is unchanged.

The diagnostic methods are `orbit`, `sector_orbit`, `orbit_circ` and
`sector_orbit_circ`; `BEM_PROBE_ORBIT_NODES` controls the representative block
size independently of the control MBJ size. `symmetry_circ` measures operator
and inverse intertwining, while `inverse_symmetry_circ` checks only inverses.
`BEM_PROBE_RHS_FILE` accepts both assembled phase-code RHS vectors from a real
twelve-mode T packet. `tools/prism_packet_rhs.cpp` and
`benchmarks/symmetry_solver_20261029/prepare_packet.py` use the worker's mass
quadrature and incident-field encoder. Packet comparisons include the strict
check of the second phase code after symmetry warm start; they do not infer
whole-packet speed from a single plane-wave RHS.

CPU checks (`make host-symmetry-check`): `tests/prism_symmetry_sectors_check` verifies unitary transforms,
rotation and mirror characters, short orbits, local LU pivoting, in-place
inverse application and intertwining, plus an independently invertible
electric/magnetic exchange operator. `tests/streaming_sector_gmres_check`
compares completed solves with an independently evaluated complex block
operator, adds cross-sector inner-action perturbations, and covers zero
sectors/RHS and action limits. Measurement records, per-binary hashes and
checkpoint handovers are in `benchmarks/symmetry_solver_20261029`.

For the complete twelve-mode, degree-36 packet at ka=40, n=1.3116,
height/width=10/7, ref=6 (1,032,192 unknowns), including the second code's
FP64 warm-start check:

| Solver | Mixed FMM actions | FP64 checks | Packet seconds |
| --- | ---: | ---: | ---: |
| Global GMRES32, MBJ50 + original circulant 0.25 | 181 | 3 | 232.669 |
| Independent sectors, orbit50 + symmetric circulant 0.25 | 156 | 3 | 208.917 |

This is 10.2% less solve time for this packet. Both use the original 2e-6
residual gate. On ref=5 the initial orbit50-only/global variant was slower
(80.70 s versus 71.83 s for the control); independent sectors alone took
70.19 s. Increasing representative blocks to 200 was also slower (76.86 s
with sectors). The FFT/sector combination, including its mask, matters.
These measurements do not establish a universal speedup for every T mode.

At ref=6 the local inverse's accounted storage falls from 3157.85 MiB to
268.73 MiB (11.75x). This is not an elevenfold reduction of total memory:
unitary transforms, inverse maps and temporary vectors add storage, while
FMM and Krylov bases remain. The persistent worker's validation peak was
7035.82 MiB RSS, versus 9352.44 MiB in the preceding production checkpoint;
the cases share the mesh/operator/settings but are different mode packets.
GPU storage is essentially unchanged. The actual worker validated both phase
codes against the control far fields, with maximum relative L2 difference
8.35e-6 and maximum FP64 residual 1.064e-6; a separate small check compared
directly with the previous frozen worker. See `worker_validation.json`.

The persistent worker supports `BEM_TMATRIX_SECTOR_GMRES=1` and
`BEM_TMATRIX_ORBIT_NODES=50`. The frozen controller exposes
`--sector-gmres --orbit-nodes 50 --circulant-blend .25`. Existing defaults are
unchanged. A solver-only checkpoint migration copies verified columns and
receipts, preserves the physical operator/projection/analytic average, and
requires fresh coded audits. The ka=40 handover retained 2388 columns and
114 packed receipts, with no wall-time limit; its validation proof is in
`benchmarks/symmetry_solver_20261029/migration_proof.json`.

## Research basis and remaining candidates

[Embree et al. (2025)](https://arxiv.org/abs/2502.18317) motivate reusable
GMRES inverse polynomials for multiple right-hand sides and discuss their
stability. Our prototype is limited to short Arnoldi polynomials; it does
not implement their high-degree root ordering/stabilization. The indefinite
balancing techniques of [Henson and Morgan (2025)](https://arxiv.org/abs/2510.14816)
were reviewed but have not been implemented here.

The IDR recurrence follows the mathematical algorithm in
[van Gijzen and Sonneveld, Algorithm 913 (2011)](https://doi.org/10.1145/2049662.2049667).
The code is an independent implementation with explicit complex arithmetic
and an outer strict-residual acceptance gate.

[Groth et al.](https://arxiv.org/abs/1903.09802) demonstrate FFT circulant
inverses for volume DDA, including hexagonal ice prisms. The surface adaptation
here uses a Galerkin block symbol, mirrored triangle layouts and a local
fallback for edge/cross-face coupling. Its measurements are reported above;
the paper's volume-DDA speedups do not transfer directly to this BEM system.

[Reduced Calderon preconditioning](https://discovery.ucl.ac.uk/id/eprint/10144775/)
targets PMCHWT regularization and cheaper operator products. Its benefit
cannot be assumed for this already second-kind Muller formulation; no
Calderon or OSRC implementation was added by this sweep.
