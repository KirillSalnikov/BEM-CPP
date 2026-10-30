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

## Sector recycling and restart controls (2026-10-30)

`--sector-recycle-rank K` retains a separate GCRO space in each enabled
symmetry sector of the fixed right-preconditioned operator. Harmonic Ritz
directions are extracted from the unrotated Arnoldi relation without extra
FMM calls. `RecycleSpace::learn_strided` reads each sector directly from the
shared basis, avoiding a second restart-sized allocation. The optional
`--sector-recycle-mask` selects sectors; its default 4095 enables all twelve.

Recycled images approximate the mixed operator. A small coarse residual
therefore triggers the original strict action; if that check fails, the
solver disables coarse projection for the remainder of that RHS and corrects
the full residual. Every accepted solution still passes the same FP64
tolerance. Independent columns and phase-code audits remain cold. The space
is retained in worker RAM across production packets and relearned after a
worker restart; the existing T-column checkpoints are unaffected.

On ref5 ka=40, n=1.3116, aspect 10/7, two successive 12-mode packets at degrees
36 and 38, including both phase codes and all training costs:

| restart | recycled directions per sector | packet 36 seconds | packet 38 seconds |
|---|---|---|---|
| 64 | 0 | 59.60 | 60.93 |
| 32 | 8 | 57.49 | 49.26 |
| 64 | 8 | 56.00 | 48.86 |
| 64 | 16 | 54.26 | 43.61 |

The rank-16 screening candidate used 115/85 mixed actions, compared with
126/130 for plain sector GMRES(64). Its maximum far-field relative L2
difference from that control was 4.07e-6 and its maximum strict residual
1.256e-6, below the unchanged 2e-6 gate. These are linear-solver controls,
not a mesh/truncation convergence certificate. Reproduction scripts and
machine-readable screening results are in
`benchmarks/sector_recycle_20261030/`.

The actual persistent worker comparison on the full ref6 mesh (1,032,192
unknowns) retained the same fields, projection directions, operator and
2e-6 strict tolerance:

| configuration | packet 36 seconds | packet 38 seconds | total seconds |
|---|---|---|---|
| sector GMRES(32), no recycling | 205.984 | 209.666 | 415.650 |
| sector GMRES(64), rank 16 per sector | 175.471 | 143.671 | 319.142 |

This is 23.2% less solve time for the two-packet sequence, including cold
training and both phase-code checks. The mixed action counts were 156/160
before and 124/93 after. The maximum far-field relative L2 difference was
4.87e-6. Each packet still used three strict actions: two on the first code
and one to accept the symmetry-derived second code. The recycled vectors
occupy 504 MiB; increasing restart 32 to 64 adds another 504 MiB. Measured
peak RSS of the same validation harness rose from 6.97 to 8.29 GiB, including
temporary extraction buffers. Full T-builder memory includes additional
incident-field/projection storage. These measurements do not extrapolate
the same speedup to every remaining packet or another size.

The diagnostic `sector_orbit_circ_vertical` completes the commuting-group
inverse to D6h by averaging its two cosets under a vertical reflection.
It applies the original inverse twice and is not enabled by default.
`vertical_symmetry_circ` measures the operator and preconditioner defects
for that extra generator. CPU tests verify the involution, the dihedral
generator relation, and preservation of rotation and horizontal reflection.
The production option is `--vertical-preconditioner`. On the ref5 diagnostic
the inverse's vertical equivariance defect fell from 0.19337 to zero at
floating-point precision; the strict physical operator defect was 2.05e-7.
This is a preconditioner symmetry diagnostic, separate from solution error.

The full-ref6 persistent worker with rank 16, restart 64 and this D6h inverse
took 186.142/126.205 seconds on the same two packets: 312.348 seconds total,
24.85% less than the original sector-GMRES(32) control. Mixed action counts
were 123/67, maximum far-field relative L2 difference 4.94e-6 and maximum
strict residual 1.067e-6. The first cold packet costs more than recycling
alone, while the second packet benefits from the fully symmetric inverse.
Peak RSS in the same validation harness was 8.32 GiB versus 6.97 GiB for the
control. `target_validation.json` records the selected executable and results.

FP64-count experiments kept the original acceptance tolerance. Local near
coordinates alone retained six strict actions for two packets. Combining
local coordinates with a tighter projected factor .25 accepted the first
code earlier in the strict correction sequence, but its near-tolerance
residual then required correction of the symmetry-derived second code.
With the same warm-guess correction as the worker, the total remained six
strict actions and took 96.15 seconds on ref5, versus 90.90 seconds with the
standard .5 factor and the D6h inverse. These precision options were left
disabled in the resumed production run. The initial cold-fallback diagnostic
is excluded from comparisons; the `workerlike` rerun matches worker behavior.

The ka=40 handover retained all 2388 verified columns and 114 packet receipts.
It uses restart 64, rank 16 in all twelve sectors, the full D6h inverse,
the original 2e-6 acceptance gate, fresh coded audits and analytic full-SO3
averaging. No wall-clock limit was introduced. Evidence and the reproducible
launcher are in `benchmarks/sector_recycle_20261030/`.

## Native full-D6h FFT inverse (2026-10-30)

The full inverse previously averaged two vertical-reflection cosets at
application time. Together with the cap rotation average, each application
needed two side-bank and twelve cap-bank FFT actions. The new path prepares
the group average of the **inverse convolution kernel** once, then applies
each bank once. It retains the two local-block actions and the correctly
transformed coverage mask. Averaging the forward symbol and then inverting
would be a different preconditioner and is deliberately not used here.

`PrismCirculantSymmetry` verifies every covered DOF, channel sign, affine
grid permutation and zero-extension mask. Side rotation/reflection maps
preserve the shared symbol. Cap rotations and vertical mirrors induce signed
affine channel permutations; both caps must share the same conjugation.
If the structural proof fails, the worker keeps the explicit group average.
Equation and current reflection signs are different and are audited separately.

CPU tests compare kernel averaging with explicit signed group actions and
check the D6 group identities on ref2/3/5 meshes with two aspect ratios.
CUDA tests export/reload inverse kernels, including rectangular grids and
adjoint actions. On actual ref5/ref6 BEM systems, native versus explicit
full-inverse action differs by at most 1.26e-16; the defects under rotation,
horizontal reflection and vertical reflection are at most 1.03e-16.
Alternating ten applications of each path measured:

| Mesh | Unknowns | Explicit D6h inverse | Native D6h inverse |
|---|---:|---:|---:|
| ref5 | 258,048 | 19.42 ms | 8.73 ms |
| ref6 | 1,032,192 | 68.58 ms | 32.66 ms |

These are preconditioner timings, not full-solver speedups. The extra
prepared inverse bank uses 48 MiB of GPU memory on ref6 (12 MiB on ref5).
The persistent full-ref6 worker took 306.00 s for degree36/38 coded packets,
versus 312.35 s for the explicit D6h average (2.03% less solve time). This
is a two-packet comparison, not a long-run throughput guarantee. All six
strict FP64 checks were retained; no residual acceptance gate was relaxed.
Measurements and the full-worker validation are in
`benchmarks/fft_symmetry_20261030/` and `runs/fft_symmetry_20261030/`.

Checkpoint continuations now enter the full persistent worker directly.
The former one-pair pilot could stop halfway through a coded audit and
discard both its reference solution and its recycle basis before launching
the full worker. Fresh runs retain the pilot. Migration proof validation,
coded audits after an executable change, strict FP64 residual gates and
resource monitoring remain in the builder/full stage.

## Opposite-face spectral blocks (2026-10-30)

The optional `--opposite-face-weight W` couples each side patch to the
aligned patch on the opposite face and couples the top/bottom caps.
`--opposite-face-mask 1/2/3` selects sides/caps/both. The default weight is
zero. These are solver-only controls and survive frozen launches/resumes;
checkpoint migration still requires a validated worker and fresh coded audits.

The original kernel sampling already computes each source's field on the
entire boundary. Sampling the opposite patch from the same output adds
the cross-face kernel X without additional FMM actions. Corresponding
tangential axes and zero-extension masks must match. The side pairing
reverses the lateral half index on the opposite face; simply taking a
180-degree rotated patch would not align the translation lattice.

An orthonormal sum/difference transform reduces the periodic block system
`[[D, W X], [W X, D]]` to two 12-channel symbols, `D + W X` and `D - W X`.
Both are inverted once. Forward and adjoint applications use the same real
orthonormal pair transform. The cap inverse kernels are averaged over D6h
after inversion, separately in the two parity sectors. Structural tests
require the symmetry generators to preserve pairing and parity, in addition
to the existing channel/grid/mask checks. The original full physical FMM
operator, incident fields and strict residual tolerance are unchanged.

Tests compare the two parity inverses with an independent full 24-channel
inverse using complex, noncommuting self/cross kernels and multiple pairs.
The largest forward/adjoint difference was below 5e-16. Mesh tests check
actual opposite-reflection DOF maps on ref2/3/5 for two aspect ratios.
Real packet timing and field comparisons are recorded in
`benchmarks/opposite_faces_20261030/` and `runs/opposite_faces_20261030/`.

On ref5, side-only, cap-only and combined coupling did not reduce iterations:
the two packets used 114/63 mixed actions, versus 114/62 without coupling.
The persistent full-ref6 comparison confirmed no useful acceleration:
290.910 s without coupling versus 291.375 s with both banks coupled, with
identical 123/67 action counts and six strict checks. The maximum far-field
relative L2 difference was 2.063e-6. Coupling therefore remains disabled for
production. The side blocks connect aligned opposite patches, not all
cross-quarter interactions of the two entire faces. The interrupted
half-weight run is archived and excluded from complete timing comparisons.

Increasing the recycled space is not automatically beneficial. With restart
64, ref5 rank 32 took 86.590 s (112/60 actions), versus 88.602 s for rank 16
(114/62). Rank 64 took 89.124 s, and rank 32 with restart 128 took 89.726 s.
On the actual full-ref6 worker, rank 32 regressed to 315.448 s (120/92 actions)
versus 290.910 s for rank 16 (123/67). The recycled cache also doubled from
0.492 to 0.984 GiB. Production selection therefore uses the full-mesh result.

The probe now accepts `BEM_PROBE_SYMMETRIC_CIRCULANT_BLEND`, including in its
native-versus-explicit D6h checks. Ref5 weight 0.5 with rank 16 took 84.207 s
(106/60 actions) without opposite coupling and 84.064 s (106/59) with it.
That one-action difference does not justify coupling; full-mesh validation
uses the uncoupled inverse. Every case retained all six strict checks.

## Earlier first strict correction (2026-10-30)

`--sector-first-pass-factor F` optionally stops the first mixed inner pass
at `F * tolerance` and then evaluates the original FP64 residual. After that
first evaluation, the original tight projected target is restored for every
correction. Zero, the default, preserves the previous algorithm. The upper
bound is 64. This control does not change the acceptance tolerance, the
physical operator, or the final check of the symmetry-derived second code.
It adds no vector storage or extra operator sampling.

The intent is to avoid spending many mixed iterations below the useful
accuracy of that operator before the inevitable strict correction. It is
an empirical tuning control: an earlier correction can also cost extra
strict actions, so the complete persistent-worker time must improve.
Unit tests include a two-dimensional diagonal problem whose first coarse
pass must be rejected by the strict gate, mixed nonnormal systems, recycled
spaces and cross-sector perturbations. Frozen-launch/resume and migration
tests retain the new solver-only control. Full-mesh timing and far-field
comparisons are in `benchmarks/opposite_faces_20261030/first_pass_validation.json`.

The full-ref6 comparison isolated the two controls:

| FFT blend | First-pass factor | Mixed actions, packets 36/38 | Solve seconds |
|---|---|---|---:|
| 0.25 | default | 123 / 67 | 290.910 |
| 0.50 | 8 | 112 / 71 | 282.627 |
| 0.50 | default | 115 / 64 | 279.644 |

The selected configuration uses FFT blend 0.5 and the original first-pass
target: 3.87% less time on these two full worker packets. All six strict
checks remain. These measurements include training and symmetry warm starts,
but are not a guarantee for every subsequent packet. The new first-pass
control remains available and disabled by default. The main ka=40 run was
resumed in `runs/ka40_n1p3116_l0p532_aspect10over7_fft_blend_20261030/`, with
all 2,412 column hashes / 115 packet receipts verified and full analytic
SO(3) averaging retained. No wall-time limit was set.

## Two-level face block Jacobi experiment (2026-10-30)

`PrismFaceCoarseBasis` adds 312 constant lattice-channel aggregates on the
26 FFT patches. The aggregates have disjoint support and unit norm. Their
span is closed under all D6h generators, with separate equation/current
reflection parity. Sampling one column per group orbit requires 16 full
mixed FMM actions. Stabilizer projection followed by signed transport
defines an equivariant coarse approximation, including its response on
interface rows outside the aggregate supports. This approximation belongs
only to the preconditioner; the physical action and strict checks are unchanged.

`DisjointCoarseInverse` applies
`P = B + (V - B A_c V) (V* A_c V)^-1 V*`, where B is the existing native
D6h FFT/local inverse and A_c denotes the sampled coarse approximation.
The basis V is implicit, so projection costs O(N) and only the update
`V - B A_c V` is stored densely on the GPU. There are no additional FMM
actions per application. Storage is 1,232.44 MiB on ref5, approximately
4.81 GiB on ref6, instead of storing both dense V and the updates.

CPU tests cover ref2/3/5 and two aspect ratios, including a deliberately
non-equivariant source action and stabilizer reconstruction. CUDA tests
compare the implicit implementation against the independent dense-V path
on a complex nonnormal system, including damping and aliasing. On the real
ref5 problem, the coarse exactness error was 7.82e-18 and the maximum D6h
generator defect was 1.61e-16.

The diagnostic method is `sector_orbit_circ_vertical_native_face_coarse`;
`face_coarse_check` checks its identities and
`BEM_PROBE_FACE_COARSE_WEIGHT` controls the correction weight. For the same
ka=40, n=1.3116, aspect 10/7, degree36/38 packets, rank16 recycling and FFT
blend 0.5:

| Coarse correction | Mixed actions | Solve seconds | Extra setup seconds |
|---|---|---:|---:|
| disabled | 106 / 60 | 84.207 | 0 |
| weight 1 | 106 / 60 | 84.706 | 7.920 |
| weight 0.5 | 105 / 60 | 84.428 | 7.915 |

Both corrected solves passed the unchanged strict gate and matched the
control far fields to at most 1.20e-6 relative L2. Neither configuration
provided a useful speedup, so the coarse correction remains diagnostic;
it was not enabled in the production worker. The full ka=40 run resumed
unchanged from 2,526 columns / 120 committed packets. Source, tests and
measurements are in `benchmarks/face_coarse_20261030/` and
`runs/face_coarse_20261030/`.

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
