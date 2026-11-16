Bounded global correction for the experimental Dirac T builder
=============================================================
2026-11-16. CPU only; production Muller/FFT/GPU binaries and queues unchanged.

Problem and implementation
--------------------------
The earlier sector FMM removes the dense far operator and the full stored W,
but still factors the complete P1 sector matrix. Its LU alone grows from
0.5625GiB at SciIce ref3 to8.446GiB at ref4 and135.14GiB at ref5.

New optional --coarse-factorization hierarchical:
1. Assemble exact local panel-pair P1 blocks without a dense full P1 matrix.
   Apply all rotational/horizontal sector phases; factor the sparse matrix
   with incomplete LU and an explicit fill target/memory preflight.
2. Partition representative panels within each physical face. Never merge
   faces with different normals. Build constant or linear patch polynomials
   and transform local vector/pseudovector components into a common frame.
   QR of exact P1 coefficients gives an orthonormal transfer Q; it does not
   use nearest-neighbor current interpolation.
3. Cap the number of patches (tested profiles:256 at ka20 /512 at ka40, P1, 6144 global
   columns once the cap binds). Assemble A Q by bounded direct-quadrature
   tiles. The small global matrix is G=Q* A Q.
4. In full-space mode retain W=Bs A Q-Q, where Bs is a panel-block inverse
   plus a local P1 ILU correction. Keep the high/low polynomial coupling of
   the panel inverse: Bs=B+Z1(Cnear-R1 B Z1)R1. Apply
       M x = Bs x - W G^-1 Q* x.
   Applying this preconditioner makes ZERO FMM calls. This is verified by a
   unit test. The previous inner-coarse correction remains an explicit control.

Only the preconditioner is approximate. The FMM operator, tol1e-8, strict
epsilon1e-12 full-space residual checks, mirror audits, atomic checksummed
T columns and analytic SO(3) averaging remain active.

The global LU and W may equal the complete P1 space on small meshes before
the cap binds. After that their storage is bounded-size LU plus O(N) W for
a fixed cap. The sparse factor has a fill target, checked payload and working
reserve; there is no hard process-wide allocator cap. Fixed-rank storage does
NOT guarantee size-independent iterations or spatial/angular convergence.

Measurements
------------
Geometry: actual SciIce column H/W_flats=1.6495721976846451, n=1.3116,
P3, target13/near24, near_subdiv1, C6 plus mirrors, OMP16/BLAS1.
Two modal RHS (both polarizations, one sector), working FMM1e-8, packet1.
These are NOT complete T20/T40 calculations.

ka20/ref3, selected full-space correction with256 patches:
  iterations7,7; maximum strict residual5.721e-9;
  density difference from previous FMM solution6.335e-15;
  preconditioner application total1.361s, with zero internal FMM calls;
  solve/check46.633 ->29.097s, observed phase ratio1.603;
  preparation23.601 ->47.799s;
  whole two-RHS experiment90.404 ->89.645s;
  peak process RSS2.946 ->4.920GiB.
Thus this is an option for many columns reusing one sector, not a win for
every isolated solve. Do not report1.603 as a measured complete-T speedup.

ka40/ref4,512 P1 patches (12288 global columns): iterations27,27,
maximum strict residual5.24e-9; coarse preparation358.142s, solve352.713s,
whole two-RHS experiment757.191s, peak RSS23.026GiB.
The256-patch probe did NOT finish within its420s solve budget. Its file
ka40_outer_final.json is a failed diagnostic, not a converged solution.
The selected result is ka40_bounded_g512.json and report.json.

SciPy sparse matrix multiplication flattened the Fortran-order global
column matrix, making a complete extra copy. project_global now uses
32-column tiles; the new ka20 peak falls from5.642 to4.920GiB. A unit test
rejects any unbounded projection RHS. This is distinct from the cost of
keeping W, which still uses more memory than the previous matrix-free W.
An earlier interrupted ka40 attempt produced no solution and is not counted.
The completed ka20 result was retained after the workstation reboot.

Exploration retained in this directory:
- Local-only sparse P1: ka10 iterations22,21 versus6,6; ka20 did not finish
  the bounded180s solve probe. It is not the recommended isolated replacement.
- Global P0 patches restored convergence, but ka20 still needed31,28 steps.
- Inner P1 patch correction needed21,21 steps; each inverse still used FMM.
- The first outer prototype dropped the panel high/low coupling and needed
  more iterations. It was replaced by the additive smoother above. Its log
  is diagnostic only and is not the selected profile.
Unrelated CPU work affected some intermediate trials; see
timing_interference.json. Do not infer precise speedup ratios between those.

The complete small T126, all16 analytic averages, native tests, checkpoint
recovery and binary rebuild evidence are in report.json / validation.json.
The Python suite has27 tests, including sector reuse (sparse offset arrays
must not be mutated by sum_duplicates), polynomial/frame transfer, global
range exactness, coarse budget rejection, both polarizations and the no-FMM
preconditioner check. Checkpoints store T columns; operators/preconditioners
are rebuilt on restart.

Reproduction (new output directory required)
-------------------------------------------
  python3 benchmarks/clifford_dielectric_20261115/build.py --fast-complex \
    --output benchmarks/dirac_sparse_coarse_20261116/libdirac_hierarchy.so

  OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
  BEM_DIRAC_LIBRARY=$PWD/benchmarks/dirac_sparse_coarse_20261116/libdirac_hierarchy.so \
  python3 benchmarks/clifford_dielectric_20261115/tmatrix.py \
    --shape prism --ka 3 --ref 1 --degree 2 --L 7 --order 24 \
    --target-order 13 --near-subdiv 1 --backend fmm-sector \
    --block-panels 1 --batch-size 8 --symmetry c6hv --sector-coarse-basis 3 \
    --coarse-factorization hierarchical --coarse-near-radius 1.5 \
    --coarse-ilu-drop 1e-3 --coarse-ilu-fill 4 \
    --coarse-aggregate-groups 256 --coarse-aggregate-basis 3 \
    --coarse-global-scope full --fmm-eps 1e-8 --fmm-packet 1 \
    --so3 mstm --out /tmp/dirac_bounded_global_example

L is the T-matrix angular cutoff and is independent of the global patch cap.
Changing either numerical setting is rejected when resuming an incompatible
checkpoint. Operational memory budgets/cache policy may change.

Remaining limits
----------------
Global-column preparation still traverses fine panel pairs and is quadratic
in time. Reuse FMM tree/setup and reduce this preparation before claiming a
large-T throughput gain at arbitrary ka. T arrays, near cache, panel factors,
Krylov vectors and FMM workspace also consume memory. This change is not a
ka80/ref7 fit certificate. Dirac P3 ref7 is not the old Muller ref7 mesh.
The dense backend remains the default and is faster when its matrices fit.

measure_final.py skips saved converged large-size probes and continues the
unfinished work. screen.py contains the exact two-mode experiment; auxiliary
logs preserve failed probes. summarize.py writes the aggregate report and
coarse_tradeoff.pdf without extrapolating a complete T20/T40 runtime.
