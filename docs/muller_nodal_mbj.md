# P2 Nodal Muller Equation and Morton Block-Jacobi

## Scope

This is the physically validated replacement path for the experimental
RWG `muller2` operator. It implements the smooth-surface formulation from:

Y. Luo, "A High-Order Nodal Galerkin Formulation for the Muller Equation:
Bypassing Divergence Conformity via Kernel Cancellation," arXiv:2604.21181,
2026. <https://arxiv.org/abs/2604.21181>

The equation is of the **second kind**. This is a statement about its
identity-plus-compact-operator structure; it does not mean a
second-order differential equation.

## Discrete system

For nonmagnetic media, the unknowns are the two tangential components of
the electric and magnetic surface currents at every quadratic P2 node.
With `Np` geometric nodes, each current has `2*Np` coefficients and the
coupled system has `4*Np` complex unknowns:

```text
[ (i/omega) K1              ((eps_i+eps_a)/2) M + K2_eps ] [ J ] = [ b_E ]
[ ((mu_i+mu_a)/2) M + K2_mu       -(i/omega) K1          ] [ M ]   [ b_H ]
```

The implementation uses:

- quadratic isoparametric triangular geometry and P2 nodal basis functions;
- metric-weighted nodal normals and a continuously interpolated tangent
  frame;
- direct exterior/interior kernel cancellation;
- a Taylor expansion for the cancelled Hessian when `max(|k|)*r < 1e-2`;
- Sauter-Schwab/Duffy quadrature for coincident, edge-adjacent, and
  vertex-adjacent triangle pairs;
- full two-polarization far-field and Mueller-matrix evaluation.

The implementation is in:

- `src/muller_nodal.cpp`: P2 geometry, tangent frame, stable kernels;
- `src/muller_duffy.cpp`: singular quadrature maps;
- `src/muller_dense.cpp`: Galerkin assembly, RHS, and far field;
- `src/muller_mbj.cpp`: right Morton block-Jacobi;
- `src/muller_fmm.cpp`: matrix-free gradient/Hessian action and exact
  near-field replacement;
- `src/muller_mbj_fmm.cpp`: local-block MBJ setup without a global dense
  matrix;
- `tools/muller_nodal_demo.cpp`: reproducible solver and benchmark.
- `tools/muller_nodal_fmm_demo.cpp`: scalable GPU/FMM solve.

## Morton block-Jacobi

P2 nodes are sorted by a three-dimensional Morton code and split into
nonoverlapping geometric blocks. Each block contains both tangential
components of both currents, so 50 scalar nodes produce a local dense
system of dimension 200. Each local system is LU-factorized once and
applied as a right preconditioner:

```text
A P^(-1) y = b,  x = P^(-1) y.
```

The reported residual is recomputed as `||b-Ax||/||b||`, not taken only
from the projected GMRES residual.

## Reproduction

```bash
git clone https://github.com/KirillSalnikov/BEM-CPP.git
cd BEM-CPP
make bin/muller_nodal_demo CXX=g++-12 CUDA_HOME=/usr

bin/muller_nodal_demo \
  --ka 6 --ri 1.5 0 --ref 2 --ntheta 37 \
  --benchmark-gmres --mbj-nodes 50 \
  --out runs/muller_nodal_ref2_mb_jacobi_ka6_n1p5.json

python3 verify_mie.py --skip-run \
  --out runs/muller_nodal_ref2_mb_jacobi_ka6_n1p5.json \
  --ka 6 --ri 1.5 --n-im 0

make bin/muller_nodal_fmm_demo CXX=g++-12 CUDA_HOME=/usr

bin/muller_nodal_fmm_demo \
  --ref 2 --ka 6 --ri 1.5 --tol 1e-6 \
  --digits 7 --max-leaf 128 --mbj-nodes 50 \
  --out runs/muller_nodal_fmm_ref2_ka6_n1p5.json
```

## Current measured results

All rows use `ref=2`, 642 P2 nodes, 2568 complex unknowns, GMRES tolerance
`1e-8`, and 50 scalar nodes per MBJ block.

| `ka` | `n` | no preconditioner | MBJ | baseline time | MBJ setup + solve | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.3 | 116 | 17 | 0.886 s | 0.189 s | 4.69x |
| 3 | 1.5 | 204 | 23 | 1.683 s | 0.245 s | 6.87x |
| 3 | 2.0 | 286 | 44 | 2.520 s | 0.475 s | 5.31x |
| 6 | 1.5 | 376 | 38 | 3.301 s | 0.356 s | 9.28x |

The `ka=6`, `n=1.5` sphere result agrees with Mie theory to about 0.5-8.3%
in sampled normalized `M11` values on this deliberately coarse mesh.
The linear solve itself reaches a true relative residual below `1e-8`.
Mesh-convergence error and algebraic-solver error are therefore reported
separately.

### Matrix-free GPU/FMM

For `ref=2`, `ka=6`, `n=1.5`, and tolerance `1e-6`:

| quantity | no preconditioner | local MBJ |
|---|---:|---:|
| GMRES iterations | 314 | 29 |
| solve time | 37.84 s | 3.61 s |
| residual against dense matrix | 1.89e-6 | 1.84e-6 |

The solve-only speedup is `10.48x`. One-time setup costs are 11.81 s for
the FMM operator and singular corrections and 6.10 s for local MBJ.
Therefore the first solve is `2.31x` faster, while repeated incident
polarizations and orientations retain the `10.48x` solve speedup.

The matrix-free action differed from the dense matrix by `7.30e-7` in a
random-vector check with `max_leaf=128`. Its direct-quadrature reference
agreed with the dense matrix to `1.70e-15`.

The `ref=3`, `ka=12`, `n=1.5` run has 10248 complex unknowns and does not
form a dense matrix. It reached tolerance `1e-5` in 65 MBJ iterations and
30.75 s. Setup took 46.63 s for FMM/near corrections and 24.50 s for MBJ
and is reusable.

## Sharp edges and remaining limitations

Sharp prisms now use an H(div)-conforming BDM1 edge basis. Two oriented
co-normal flux moments are shared on every topological edge, and a surface
Piola transform maps the six local vector functions to each triangle.
This replaces both normal averaging and unrelated face-local nodal traces.
The old smooth and split P2 modes remain available for comparisons.

The BDM1 implementation has dense/FMM agreement, an analytic sphere check,
and three-level prism convergence. It does not yet add explicit singular
edge enrichment, and an independent sharp-body solver comparison remains
required. See [`muller_edges.md`](muller_edges.md).

The dense path remains the reference for small systems. The scalable path
now has analytic GPU/FMM gradients and Hessians, exact Duffy replacement
for singular element pairs, and MBJ blocks assembled without a global
dense matrix.

Current performance limitations are:

1. source interpolation and test projection still pass through host
   buffers between FMM traversals;
2. the three Cartesian current components are evaluated sequentially
   instead of in one batched tree traversal;
3. local MBJ factorization is on the CPU;
4. setup is substantial for a single right-hand side, although it is
   amortized over orientations and polarizations;
5. the high-order plane-wave FMM needs a parameter policy that avoids its
   low-frequency high-order instability.

The neural training path is maintained as a separate project. Existing
GraphSAI weights for the RWG PMCHWT operator are incompatible with the Muller
operator.
The local training target can be exported without a global dense matrix:

```bash
make bin/muller_training_dump CXX=g++-12 CUDA_HOME=/usr
bin/muller_training_dump \
  --out runs/muller_train_ref2.raw \
  --shape sphere --ref 2 --ka 6 --ri-real 1.5 \
  --block-nodes 50 --digits 6
```

The scalar Helmholtz FMM exposes an analytic symmetric-Hessian action. Its
far field differentiates the plane-wave local expansion analytically,
while its near field evaluates the exact Hessian kernel directly. Run the
GPU checks with:

```bash
make cuda-hessian-check CXX=g++-12 CUDA_HOME=/usr
make cuda-muller-fmm-check CXX=g++-12 CUDA_HOME=/usr
```
