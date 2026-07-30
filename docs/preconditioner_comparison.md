# PMCHWT preconditioner comparison

## Controlled setup

All measured rows use the same balanced PMCHWT operator, paired GPU GMRES,
quadrature order 7, FMM accuracy 5 digits, GMRES tolerance `1e-3`, and a
hexagonal prism with `h/D=1`, `ka=4.6086956522`, and `m=2.3+0i`.

| Mesh | Method | FMM actions | Solve, s | Total, s | Max true residual |
|---|---:|---:|---:|---:|---:|
| ref=1, 216 RWG | none | 66 | 6.019 | 6.637 | 9.89e-4 |
| ref=1, 216 RWG | local Schwarz | 21 | 3.162 | 3.830 | 6.18e-4 |
| ref=1, 216 RWG | ILU(0) | 20 | 1.455 | 1.979 | 9.80e-4 |
| ref=2, 792 RWG | none | 273 | 43.480 | 44.404 | 9.92e-4 |
| ref=2, 792 RWG | local Schwarz | 294 | 46.773 | 47.587 | 5.87e-3 (rejected) |
| ref=2, 792 RWG | ILU(0) | 108 | 17.706 | 18.530 | 9.65e-4 |

For `ref=2`, ILU(0) gives `2.53x` fewer FMM actions and `2.46x` lower solve
time. Its Mueller matrix differs from the converged baseline by `5.62e-4` in
relative L2 norm, consistent with the requested solver tolerance.

The `ref=5` ILU run and the queued `m=1.3` comparison were stopped before
completion on 2026-07-22. Their directories are retained for provenance, but
they contain no validated timing result.

## RWG mass-matrix smoke result

The mass-matrix implementation was checked in a separate back-to-back run on
the otherwise idle GPU with the same `ref=1`, `ka=4.6086956522`, and `m=2.3`
case:

| Method | FMM actions | Solve, s | Total, s | Max true residual |
|---|---:|---:|---:|---:|
| none | 66 | 4.355 | 4.863 | 9.89e-4 |
| RWG L2 mass | 65 | 4.368 | 4.727 | 8.84e-4 |

The action-count ratio is only `1.015x`, and the solve-time ratio is `0.997x`.
Thus this quasi-uniform small mesh shows no useful mass-preconditioner speedup.
The Mueller matrices agree to `2.52e-4` in relative L2 norm. The inner mass CG
used 16.86 iterations on average and reached at most `9.94e-11` relative
residual, so the neutral result is not caused by an inaccurate inverse action.

`--prec mass` assembles the exact sparse RWG L2 Gram matrix and applies its
inverse independently to the electric and magnetic current blocks as a left
preconditioner. It is retained as a classical control, especially for future
nonuniform-mesh tests, but it is not enabled by the automatic policy.

## Experimental RWG operator square

`--prec calderon-rwg` implements the strong algebraic product
`G^-1 A G^-1 A` using the existing RWG/RWG L2 Gram matrix. Its extra full FMM
applications are included in the reported `gmres_matvecs` count. It is separate
from all existing preconditioners and is never selected by the automatic
policy.

A bounded `ref=1` probe with one 20-step GMRES cycle produced:

| Method | Outer steps | Total full-operator actions | Max true residual | Solve, s |
|---|---:|---:|---:|---:|
| experimental RWG square | 20 | 44 | 4.75e-1 | 3.021 |

Of the 44 actions, 23 were the additional PMCHWT applications required by the
left preconditioner. The inverse mass actions reached relative residual below
`1e-10`, so the failure is not an inner-CG accuracy problem. The same case
converges without preconditioning in 66 actions to `9.89e-4`. Extrapolating
the interrupted unrestricted run is not valid; only the bounded probe is
reported.

This result is expected to differ from a conforming Calderon method. A strict
discretization needs Buffa-Christiansen functions on a barycentrically refined
mesh and the twisted RWG/BC duality mass matrix. Those spaces are not present
in the current code. See Cools, Andriulli, and Michielssen,
[doi:10.1109/TAP.2011.2165465](https://doi.org/10.1109/TAP.2011.2165465), and
Kleanthous et al., [arXiv:1808.10539](https://arxiv.org/abs/1808.10539).

## ILU(0) implementation

`--prec ilu0` factors the sparse near-field PMCHWT matrix without fill. The
matrix uses the same local Galerkin blocks and singular corrections as the
GraphSAI feature graph. Factorization is performed once on the CPU. The lower
and upper triangular solves remain on the GPU through cuSPARSE SpSV, so every
GMRES step avoids host-device vector transfers. ILU is applied on the right,
matching GraphSAI.

## Calderon status

A strict multiplicative Calderon preconditioner is not present in this code.
For Maxwell PMCHWT it requires:

1. a barycentrically refined surface mesh;
2. Buffa-Christiansen basis functions dual to the coarse RWG basis;
3. mixed RWG/BC identity (mass) matrices and their inverse actions;
4. a second accelerated boundary-operator application in the dual space;
5. validation that iteration counts remain stable under mesh refinement.

Using the current RWG coefficient matrix itself as a right preconditioner would
form an algebraic `A^2`, but it is not a conforming discretization of the
Calderon product. It may be labelled only as an operator-squaring experiment,
not as a Calderon benchmark.

Before implementing full Calderon, the next classical control should be the
strong-form/mass-matrix PMCHWT operator. Published single-particle experiments
report that this simpler method can outperform full Calderon preconditioning.
