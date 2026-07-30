# H(div)-Conforming Edge Basis for the Muller Solver

## Why the nodal basis was insufficient

The original P2 discretization stored two tangent components at every
surface node. On a polyhedron, the tangent plane is discontinuous at an
edge. Sharing one nodal frame averages incompatible face normals, while
duplicating the node leaves two unrelated surface currents. Neither choice
enforces the normal-flux continuity required by the surface H(div) space.

The production sharp-edge mode now uses a BDM1 basis. The old modes remain
available as:

| option | purpose |
|---|---|
| `--edge-mode hdiv` | production BDM1 edge basis; default for prisms |
| `--edge-mode smooth` | legacy shared nodal P2 comparison |
| `--edge-mode split` | earlier piecewise-smooth nodal experiment |

## Discrete space

Every topological triangle edge owns two degrees of freedom: the zeroth
and first moments of the co-normal current component along the edge. A
triangle therefore has six local BDM1 vector functions. A closed mesh with
`E` edges has `2E` current unknowns and `4E` unknowns in the coupled
electric/magnetic Muller system.

The six reference-triangle functions are dual to the oriented edge
moments. They are mapped to the physical curved triangle by the
contravariant surface Piola transform

```text
v(x) = (a_1 v_xi + a_2 v_eta) / J,
```

where `a_1` and `a_2` are the geometric covariant tangents and
`J = |a_1 x a_2|`. Each mesh edge receives one global low-to-high vertex
orientation. Local basis signs are transformed to that orientation, so
the co-normal flux is single-valued across a shared edge.

Dense assembly, FMM source projection, test projection, near Duffy
correction, right-hand side, far field, and Morton block-Jacobi all use
the same generic vector-basis evaluator.

## FMM accuracy for surface derivatives

The Muller operator uses gradients and Hessians of the Helmholtz Green
function. The conventional `3x3x3` near stencil was too small for the
plane-wave expansion: on coincident source and target surfaces it gave
relative errors `2.4e-2` for the Hessian and `1.87e-1` for the gradient.

The octree now classifies target/source box pairs recursively. Muller uses
`--fmm-near-radius 3`, while the generic scalar FMM keeps radius 1. The
surface derivative expansion is capped at the empirically stable
`digits=5` order; requesting a larger value prints a warning and uses 5.
On a depth-three 3000-point sphere this gives:

| quantity | relative L2 error |
|---|---:|
| Hessian | `9.03e-8` |
| gradient | `6.24e-8` |
| repeated combined traversal | `6.4e-12` |

For the complete `ref=2` H(div) Muller operator, the FMM action differed
from the independently assembled dense matrix by `2.60e-9`; the direct
quadrature action differed by `1.39e-15`.

## Production command

```bash
cd /home/kirill/neuro/BEM-CPP-muller-clean

make bin/muller_nodal_fmm_demo \
  CXX=g++-12 CUDA_HOME=/usr -j8

bin/muller_nodal_fmm_demo \
  --shape prism --sides 6 --aspect 1 --ref 2 \
  --edge-mode hdiv \
  --ka 3 --ri 1.3 --tol 1e-5 \
  --digits 5 --fmm-near-radius 3 --max-leaf 64 \
  --mbj-nodes 50 --gmres-restart 100 --max-iters 600 \
  --mbj-only --no-dense-validation \
  --physical-check --ntheta 73 \
  --near-correction-cache runs/muller_hdiv_prism.cache \
  --out runs/muller_hdiv_prism.json
```

The near-correction cache can be reused only for unchanged geometry,
material, wavelength, basis, and quadrature parameters.

## Validation

The checks cover:

1. co-normal flux continuity of every BDM1 edge function;
2. finite dense matrix, right-hand side, and far field;
3. FMM and direct actions against an independent dense matrix;
4. exact agreement of dense and local MBJ blocks;
5. cache acceptance for matching input and rejection after a physical
   parameter change;
6. a sphere against the analytic Mie solution;
7. three-level convergence of a sharp hexagonal prism.

Sphere at `ka=3`, `n=1.3`, `ref=2`, tolerance `1e-5`:

| metric | value |
|---|---:|
| MBJ iterations per polarization | 14 |
| true residual, second polarization | `6.33e-6` |
| absolute M11 solid-angle relative L2 vs Mie | `1.15e-3` |
| forward M11 BEM/Mie | `0.998477` |
| worst normalized Mueller component error | `5.88e-4` |

Hexagonal prism at `ka=3`, `n=1.3`, tolerance `1e-5`:

| refinement | system unknowns | quadrature points | iterations | true residual | solve time |
|---:|---:|---:|---:|---:|---:|
| 1 | 864 | 1,008 | 16 | `8.57e-6` | 2.13 s |
| 2 | 3,168 | 3,696 | 16 | `5.29e-6` | 6.50 s |
| 3 | 12,672 | 14,784 | 17 | `6.61e-6` | 30.65 s |

The solid-angle relative change in `M11` decreases from `9.70e-3` between
refinements 1 and 2 to `1.17e-3` between refinements 2 and 3. The latter
maximum pointwise change is `1.92e-3` of `M11(0)`.

## Independent ADDA comparison

The `ref=3` H(div)-BEM prism was compared with ADDA-OCL for the same
equivalent-volume hexagonal prism, `h/D=1`, `ka=3`, `n=1.3`, orientation
`(0,0,0)`, residual tolerance `1e-5`, and 73 scattering angles. ADDA was
run at `dpl=15,20,30,40,60,80`.

The ADDA `dpl=60 -> 80` change is `4.84e-4` for the full forward-normalized
Mueller matrix and `3.89e-4` for `M11`. Against the selected `dpl=80`
solution, BEM differs by:

| metric | relative difference |
|---|---:|
| raw full Mueller matrix | `8.38e-4` |
| solid-angle weighted normalized full matrix | `4.42e-4` |
| solid-angle weighted normalized M11 | `1.45e-4` |
| forward M11, ADDA/BEM | `1.000672` |

The complete BEM process, including setup and two polarizations, took
`60.99 s`. ADDA `dpl=80` took `0.91 s` as an external process measurement,
so ADDA was `67.0x` faster for this small, regular-grid-friendly case.
This is a comparison of complete methods, not preconditioners alone.

Reproduce the study with:

```bash
scripts/run_hdiv_bem_adda_ka3.sh
```

Results and plots are written to
`runs/hdiv_bem_vs_adda_ka3_n1p3/report/`.

Run the automated checks with:

```bash
make host-checks CXX=g++-12 CUDA_HOME=/usr -j8
make tests/fmm_hessian_check tests/muller_fmm_check \
  CXX=g++-12 CUDA_HOME=/usr -j8

tests/fmm_hessian_check \
  --surface --points 3000 --max-leaf 64 \
  --near-radius 3 --digits 5

tests/muller_fmm_check \
  --shape sphere --ref 1 --edge-mode hdiv \
  --digits 5 --near-radius 3 --max-leaf 128
```

## Remaining limitation

The BDM1 space supplies the correct H(div) conformity, but it does not
contain explicit singular edge-enrichment functions. Publication-grade
polyhedral results should still use at least three mesh levels and be
compared with an independent edge-capable formulation. Rotational and
mirror polarization reuse currently supports only the nodal basis; H(div)
solution transforms have not yet been implemented.
