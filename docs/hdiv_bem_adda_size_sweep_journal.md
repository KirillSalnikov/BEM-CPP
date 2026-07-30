# Strict H(div) Muller BEM versus ADDA size sweep

This journal records the reproducible `ka=10,15,20,25,30` study for a
hexagonal prism with `h/D=1`, refractive index `m=1.3`, and relative solver
tolerance `1e-5`.

## Commands

ADDA results are generated once and reused:

```bash
STAGE=adda KAS='10 15 20 25 30' \
  ./scripts/run_hdiv_bem_adda_size_sweep.sh
```

The strict BEM calculation uses the H(div)-conforming BDM1 basis, the Muller
second-kind equation, radius-3 near correction, double precision, MBJ, and the
exact C6 rotational relation between the two incident polarizations:

```bash
STAGE=bem KAS='10 15 20 25 30' BEM_HYBRID=0 \
  BEM_VARIANT=batch3_fused ./scripts/run_hdiv_bem_adda_size_sweep.sh
```

The sweep is resumable. Existing nonempty result files are reused unless
`FORCE=1` is set.

## Iteration journal

Every BEM run writes `<stem>.iterations.csv` and flushes it after every row.
The columns are:

- `solver` and `event`: solver stage and initial/iteration/cycle/final event;
- `iteration`: Krylov iteration number;
- `projected_residual`: residual estimate inside GMRES;
- `operator_residual`: explicitly recomputed operator residual at cycle
  boundaries and at the end;
- `matvec_s`: time spent applying the full Muller operator;
- `preconditioner_s`: MBJ application time;
- `orthogonalization_s`: Krylov orthogonalization time;
- `elapsed_s`: elapsed time inside this solver stage.

For example:

```bash
tail -f runs/hdiv_bem_vs_adda_sweep_n1p3/ka20/\
bem_ref5_sparse_c6_batch3_fused.iterations.csv
```

The corresponding `.stdout.log` contains mesh/FMM setup diagnostics and the
`.time` file contains process wall time and peak resident memory.

## Performance investigation

The original surface FMM octree created all eight children at every level,
including empty volume cells. At `ref=5` this produced 37,449 nodes,
59,100,608 M2L pairs, and 9,495,360 near-box pairs. The sparse tree now creates
only occupied children: 5,833 nodes, 1,251,816 M2L pairs, and 290,696 near-box
pairs. This reduced setup from about 30.5 s to about 1.3 s and a strict
radius-3 operator application from roughly 39 s to 13.2 s.

A Muller operator application then still performed twelve independent FMM
traversals: two surface currents, three Cartesian components, and two
wavenumbers. The `batch3_fused` implementation shares each traversal between
the three Cartesian components and evaluates the near-field gradient and
Hessian in one pass. The formulas, radius-3 direct near field, and double
precision are unchanged. At `ref=5`, one operator application decreased from
about 13.2 s to 4.7 s.

The batch implementation is checked against direct differentiation:

```bash
tests/fmm_hessian_check
tests/fmm_hessian_check --surface --points 3000 \
  --near-radius 3 --max-leaf 64 --digits 5
```

Observed relative errors were approximately `7.2e-6` for a random volume test
and `9.0e-8` for the surface test. A complete `ka=10, ref=4` result agreed with
the pre-batch result to `1.5e-15` in the forward-normalized full Mueller
matrix, while wall time decreased from 128.79 s to 67.08 s.

## Incremental report

```bash
python3 scripts/report_hdiv_bem_adda_size_sweep.py
```

This creates `size_sweep_batch3_fused.csv`, `.json`, and `.png` in the sweep
directory. Only completed pairs of BEM refinements are included.

## Final results

All times below are process wall times. The BEM iteration columns contain the
first polarization followed by the strict correction of the C6-rotated second
polarization.

| ka | BEM ref=5 iterations | BEM, s | ADDA dpl=20, s | BEM/ADDA time | BEM ref4->ref5, full | BEM vs ADDA dpl=20, full |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 23 + 1 | 47.56 | 1.55 | 30.68 | 0.160% | 0.145% |
| 15 | 30 + 5 | 214.86 | 4.85 | 44.30 | 0.174% | 0.367% |
| 20 | 41 + 7 | 287.63 | 20.70 | 13.90 | 0.046% | 0.056% |
| 25 | 54 + 18 | 416.82 | 69.97 | 5.96 | 0.680% | 0.704% |
| 30 | 70 + 38 | 637.25 | 251.14 | 2.54 | 0.143% | 0.056% |

The `ka=25` result is the least grid-converged point in this refinement pair;
it should not be presented as a 0.1%-accurate reference. The other four points
have a ref4-to-ref5 full-Mueller change below 0.18%. Agreement between two
methods can occasionally be better than either method's own refinement
estimate, so both quantities must be reported.

At `ref=5`, the median fused operator time ranges from 4.73 s at `ka=15` to
5.20 s at `ka=30`. The increase is caused by the higher far-field expansion
order, while the dominant radius-3 near field has the same geometry. The MBJ
application itself remains about 0.011 s per iteration. For `ka=30`, the
axis-aligned approximate FMM has a C6-rotated initial residual of `2.25e-3`;
38 correction iterations are retained to enforce the same explicit `1e-5`
criterion for the second polarization.

## `ka=25` edge-refined follow-up

The original local edge-refinement pass produced 18.5-degree transition
triangles and therefore fell back to a uniform refinement. The mesh closure
now extends the red-refined band only when a green transition would violate
the 25-degree quality threshold. For `ref=4, edge-refine=1`, this gives 22,152
triangles, a 30-degree minimum angle, 132,912 system unknowns, and no uniform
fallback.

With exact C6 geometric reconstruction of the second polarization, the new
calculation converges in `53+0` iterations. The first cold run takes 201.75 s;
loading the reusable near-correction cache reduces this to 186.88 s. The old
uniform `ref=5` calculation took 416.82 s, so the warm speedup is `2.23x`.
The edge-refined result differs from the old `ref=5` result by only `0.0448%`
in the solid-angle-weighted full Mueller matrix.

The residual disagreement with ADDA is dominated by ADDA grid convergence:

| ADDA grid | BEM vs ADDA, full | BEM vs ADDA, M11 | ADDA wall time |
|---:|---:|---:|---:|
| dpl=15 | 1.085% | 0.932% | 28.61 s |
| dpl=20 | 0.692% | 0.610% | 69.97 s |
| dpl=25 | 0.473% | 0.424% | 194.07 s |

ADDA itself changes by `0.510%` from dpl 15 to 20 and by `0.296%` from dpl
20 to 25. Thus a solver residual of `1e-5` does not imply a physical
discretization error of `1e-5`. At dpl=25, the cached edge-refined BEM run is
`1.04x` faster while its BEM mesh check is substantially tighter.

Rebuild the dedicated figure and machine-readable summary with:

```bash
PYTHONPATH=scripts python3 scripts/report_ka25_edge_refinement.py
```

## `ka=30` strict pFFT-FGMRES follow-up

For `ka=30`, the same `ref=4, edge-refine=1` mesh has 132,912 unknowns and
differs from the uniform `ref=5` FMM result by `0.00996%` in the
solid-angle-weighted full Mueller matrix. A direct edge-refined FMM solve
needs 68 iterations and 257.09 s on a cold near-correction cache.

Strict pFFT-FGMRES uses pFFT only for the inner approximate inverse. The
outer six iterations are evaluated with the reference FMM action and finish
at an explicit FMM residual of `3.33e-6`. The complete run takes 76.48 s,
which is `3.28x` faster than the 251.14 s ADDA dpl=20 run and `6.36x`
faster than the 486.05 s ADDA dpl=25 run. The accelerated Mueller matrix
differs from the direct edge-refined FMM result by `0.0000259%`, from
uniform BEM ref=5 by `0.00996%`, from ADDA dpl=20 by `0.05784%`, and from
ADDA dpl=25 by `0.04451%`.

ADDA changes by `0.03269%` from dpl 15 to 20 and by `0.01581%` from dpl 20
to 25. Thus the BEM edge-to-ref5 mesh change (`0.00996%`) is already below
the latest measured ADDA grid change. The pFFT approximation does not set
the reported physical answer: replacing it with the direct FMM solve changes
the full Mueller matrix by only `0.0000259%`.

The dedicated comparison is generated after all requested ADDA grids have
finished:

```bash
PYTHONPATH=scripts python3 scripts/report_ka30_bem_adda.py
```
