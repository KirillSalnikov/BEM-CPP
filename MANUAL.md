# BEM-CPP Operational Manual

## 1. Purpose and Solver Selection

BEM-CPP computes electromagnetic scattering by a homogeneous dielectric
particle represented by a closed triangular surface. It contains two solver
families that must not be confused.

| Family | Executable | Unknowns and formulation | Recommended use |
|---|---|---|---|
| Muller | `muller_nodal_fmm_demo[_fp32]` | electric and magnetic tangential currents; second-kind Muller equation | current sharp-particle and large-grid work |
| PMCHWT | `bem_cuda_fmm` | RWG electric and magnetic currents; PMCHWT or balanced PMCHWT | legacy controls, OBJ tools, GraphSAI |

The legacy PMCHWT option named `muller2` is an operator experiment in the
RWG path. It is not equivalent to the separate P2/H(div) Muller solver.

The dense Muller executable, `muller_nodal_demo`, is a small-problem
reference. It assembles a dense matrix and is useful for checking the
matrix-free action and preconditioner.

## 2. Mathematical Object Being Solved

For a fixed particle, material, wavelength, and incidence, discretization
produces a complex linear system

```text
A x = b,
x = [J, M]^T.
```

`J` and `M` are coefficients of equivalent electric and magnetic surface
currents. `b` is generated from the tangential incident electric and magnetic
fields. The matrix `A` contains exterior and interior Green-function
interactions, singular local integrals, and second-kind identity terms.

The code does not normally store the full matrix. It evaluates `A*v` through:

1. projection of current coefficients to quadrature points;
2. exterior and interior FMM or pFFT interactions;
3. sparse correction of singular and adjacent interactions;
4. assembly of the two Muller rows;
5. FP64 Krylov updates and residual reductions.

The final far field is computed from the converged currents. Two orthogonal
incident polarizations produce the four complex amplitude functions and the
complete real `4 x 4` Mueller matrix as a function of scattering angle.

## 3. Surface Discretizations

### 3.1 Smooth P2 mode

`--edge-mode smooth` uses quadratic nodal interpolation with a smooth local
tangent frame. It is appropriate for spheres and smooth convergence controls.
It is not the preferred representation across a sharp polyhedral edge because
a single smooth tangent frame is not physically natural there.

### 3.2 Split nodal mode

`--edge-mode split` duplicates selected feature-edge degrees of freedom. It is
retained for controlled comparisons with earlier work, not as the production
sharp-particle mode.

### 3.3 H(div)-BDM1 mode

`--edge-mode hdiv` uses linearly varying BDM1 edge moments. A global edge
orientation and the surface Piola transform preserve the shared co-normal
flux. This is the default recommendation for prisms, cubes, and sharp OBJ
surfaces.

The option `--feature-angle` controls feature detection where it is relevant.
The surface must still be closed, manifold, non-degenerate, and consistently
oriented.

### 3.4 Refinement

`--ref N` recursively refines the surface. It is not interchangeable with
ADDA's dipoles-per-wavelength parameter. A converged linear residual only
shows that the chosen discrete system was solved. Physical convergence must
be established by comparing at least two refinement levels.

## 4. Acceleration Components

### 4.1 FMM

The fast multipole method evaluates long-range Green-function interactions
without forming a dense matrix. The Muller action requires scalar,
gradient/curl, and Hessian contractions for both exterior and interior
wavenumbers.

`--digits N` controls the requested FMM expansion accuracy.
`--max-leaf N` controls octree depth and work distribution. At high frequency,
an excessively deep tree can reduce the admissible expansion order; the
driver contains a guard for this condition.

### 4.2 Near correction

FMM and pFFT approximations are replaced by accurately integrated entries for
singular and topologically adjacent element pairs. `--fmm-near-radius`
controls the geometric near region. `--near-correction-cache FILE` stores the
validated correction and rejects incompatible geometry or physics.

Equivalent local configurations on regular polyhedra reuse correction
templates. This reduces setup cost without changing the operator.

### 4.3 Morton block-Jacobi

MBJ partitions surface unknowns in Morton order, assembles dense local blocks,
and stores their LU factors:

```text
P approximately equals A,
A P^{-1} y = b,
x = P^{-1} y.
```

The preconditioner is applied on the right. Therefore the reported physical
solution is still `x`, and the true residual must be evaluated with the full
selected operator.

`--mbj-nodes` controls nominal block size. `--mbj-overlap` enables a restricted
additive-Schwarz overlap. Larger blocks or overlap may reduce iterations but
increase setup time and memory. They are not universally faster.

`--mbj-cache FILE` reuses the LU factors when geometry, material, basis,
quadrature, and MBJ parameters are unchanged.

### 4.4 pFFT-FGMRES

`--pfft-fgmres` uses an approximate pFFT solve as a variable right
preconditioner inside an outer FGMRES solve. The outer operator remains FMM:

```text
z_k approximately solves A_pFFT z_k = v_k,
w_k = A_FMM z_k.
```

The inner solve is intentionally loose. `--pfft-inner-tol` and
`--pfft-inner-iters` limit its cost; `auto` selects a size-dependent cap.
Because the inner action varies, flexible GMRES is required rather than
ordinary right-preconditioned GMRES.

The outer residual decides convergence. A fast inner solve that increases the
outer iteration count can be slower overall, so setup, inner, outer, and
far-field times must be reported separately.

## 5. Precision

Build targets:

```bash
make bin/muller_nodal_fmm_demo CXX=g++-12 CUDA_HOME=/usr
make muller-fp32 CXX=g++-12 CUDA_HOME=/usr -j"$(nproc)"
```

`muller_nodal_fmm_demo` uses FP64 operator storage/work where implemented.
`muller_nodal_fmm_demo_fp32` enables FP32 pFFT arrays and selected near-field
work. Krylov vectors, MBJ factors, orthogonalization, and norm reductions
remain FP64.

This is mixed precision, not a pure single-precision calculation. The selected
mixed operator is still slightly different from the FP64 operator. Validate a
new regime by checking:

1. final independently recomputed residual;
2. FP32 versus FP64 Mueller difference;
3. surface-refinement convergence;
4. sensitivity of weak Mueller elements.

Use `--fmm-near-fp64` for an FP64 near-field control with the mixed binary.

## 6. Building

### 6.1 Dependencies

- Linux;
- CUDA toolkit, cuFFT, cuSPARSE;
- a CUDA-compatible C++ compiler;
- OpenMP;
- Python 3 for reports.

### 6.2 RTX 3090 Ti

```bash
git clone https://github.com/KirillSalnikov/BEM-CPP.git
cd BEM-CPP

make muller-fp32 CXX=g++-12 CUDA_HOME=/usr -j"$(nproc)"
make fmm-only CXX=g++-12 CUDA_HOME=/usr \
  ARCH=-arch=sm_86 -j"$(nproc)"
```

The mixed Muller target sets `sm_86`. For another GPU, adjust the Makefile
target or build flags.

## 7. Single-Orientation Muller Calculation

```bash
mkdir -p runs/prism_ka25_ref5

OMP_NUM_THREADS=16 bin/muller_nodal_fmm_demo_fp32 \
  --shape prism --sides 6 --aspect 1 \
  --ref 5 --ka 25 --ri 1.3 \
  --edge-mode hdiv \
  --quad 7 --duffy-order 4 \
  --digits 5 --max-leaf 64 --fmm-near-radius 3 \
  --tol 1e-5 --max-iters 500 --gmres-restart 100 \
  --mbj-only --mbj-nodes 50 --mbj-overlap 0 \
  --near-correction-cache runs/prism_ka25_ref5/operator.near \
  --mbj-cache runs/prism_ka25_ref5/mbj50.cache \
  --pfft-fgmres \
  --pfft-inner-tol 4e-2 --pfft-inner-iters auto \
  --pfft-outer-restart 32 \
  --pfft-order 2 --pfft-correction-radius 0 \
  --pfft-grid-safety 1 \
  --physical-check --ntheta 181 \
  --no-dense-validation \
  --iteration-log runs/prism_ka25_ref5/iterations.csv \
  --out runs/prism_ka25_ref5/result.json
```

Interpretation of the main controls:

| Control | Meaning |
|---|---|
| `--ka 25` | equal-volume size parameter |
| `--ri 1.3` | real refractive index |
| `--ref 5` | fifth recursive surface refinement |
| `--tol 1e-5` | requested relative residual |
| `--physical-check` | solve both incident polarizations |
| `--ntheta 181` | one-degree scattering grid |
| `--mbj-only` | skip an unpreconditioned comparison solve |
| `--no-dense-validation` | do not attempt impossible dense assembly |

For small systems, remove `--mbj-only` and use the FP64 executable to compare
baseline and preconditioned solutions.

## 8. Checkpoints and Caches

Unless `--no-checkpoint` is supplied, the driver stores solver checkpoints
under the output-derived prefix:

```text
result.json.checkpoint.<stage>.bin
```

The checkpoint contains the current outer Krylov state/solution and is saved
after each outer iteration. Compatibility includes geometry, material,
quadrature, precision mode, system dimensions, and the right-hand side.

Use `--checkpoint PREFIX` to choose another location. Repeating an identical
command resumes automatically. A changed signature is rejected.

Three mechanisms have different purposes:

| File | Purpose |
|---|---|
| solver checkpoint | resume interrupted Krylov work |
| near-correction cache | avoid rebuilding local exact corrections |
| MBJ cache | avoid rebuilding local LU factors |

Do not describe cache loading as solver acceleration unless complete wall time
and cold-start time are both reported.

## 9. Orientation Averaging

Recommended wrapper:

```bash
KA=25 RI=1.3 REF=5 \
ALPHA=8 BETA=8 GAMMA=4 NTHETA=181 THREADS=16 \
OUT="$PWD/runs/prism_ka25_avg" \
scripts/run_muller_orientation_average.sh
```

`--orient-average Na Nb Ng` uses quadrature nodes in `cos(beta)` and uniform
azimuthal nodes. Two polarization solves at each `(beta,gamma)` provide the
alpha dependence in the far field. Consequently, `Na` increases far-field
sampling but does not multiply the number of linear systems.

`--orient-symmetry-order 6` restricts a regular hexagonal prism to one gamma
sector. This is exact only when:

- the material is rotationally invariant;
- the generated or imported mesh preserves the declared symmetry;
- no orientation-dependent external feature breaks that symmetry.

Nearby base orientations may use a previous solution as an initial guess.
`--orient-warm-max-angle` limits this transfer and `--orient-zero-start`
disables it. An initial guess changes iteration count, not the converged
equation.

The orientation checkpoint stores accumulated weights, Mueller values, timing,
the next orientation, and previous solutions. It is replaced atomically after
every completed base orientation.

## 10. Output and Interpretation

The JSON output contains:

- geometry, material, mesh, basis, and quadrature metadata;
- number of system unknowns and quadrature points;
- solver and preconditioner settings;
- projected and independently checked residual information;
- iteration counts;
- setup, solve, inner-preconditioner, and far-field timing;
- scattering angles and Mueller elements;
- checkpoint and cache metadata.

The iteration CSV records residual and phase timing for diagnosing whether
time is spent in the FMM action, pFFT inner solves, preconditioning, or
orthogonalization.

Always distinguish:

1. iteration speedup;
2. solve-only wall-time speedup;
3. complete wall-time speedup including setup, cache generation, and far field.

The same iteration count can have different wall time because an MBJ or pFFT
application adds mathematical work to every outer step.

## 11. Accuracy Protocol

### 11.1 Linear convergence

The true residual must meet the requested tolerance:

```text
||b - A x||_2 / ||b||_2 <= tolerance.
```

A projected GMRES residual alone is insufficient, especially for inexact or
mixed-precision operators.

### 11.2 Surface convergence

Repeat at the next refinement level. Compare the complete Mueller matrix with
a solid-angle-weighted norm and inspect important weak components separately.

### 11.3 Operator convergence

Increase FMM digits, quadrature order, Duffy order, and near radius one at a
time. The observable change should be below the target physical error.

### 11.4 Independent physics

- sphere: compare with Mie theory;
- prism or OBJ: compare with a converged ADDA calculation or another
  independently implemented surface/volume method;
- compare matching geometry, orientation, refractive index, wavelength,
  angular grid, polarization convention, and normalization.

Agreement of forward `M11` alone is not enough. Compare all nonzero Mueller
elements and integrated quantities.

## 12. Main Muller Command-Line Controls

### Geometry

| Option | Description |
|---|---|
| `--shape sphere` | built-in sphere |
| `--shape prism --sides N --aspect F` | regular prism |
| `--shape cube` | structured cube |
| `--obj FILE` | imported triangular surface |
| `--ref N` | recursive surface refinement |
| `--edge-refine N` | local edge refinement experiment |
| `--edge-mode smooth|split|hdiv` | surface-current basis |

### Operator

| Option | Description |
|---|---|
| `--digits N` | FMM accuracy target |
| `--quad N` | regular triangle quadrature |
| `--duffy-order N` | singular/adjacent quadrature |
| `--max-leaf N` | FMM leaf occupancy |
| `--fmm-near-radius N` | exact near-correction radius |
| `--operator-backend fmm|pfft` | selected direct action backend |
| `--fmm-near-fp32|--fmm-near-fp64` | near precision |

### Solver and preconditioner

| Option | Description |
|---|---|
| `--tol F` | requested relative residual |
| `--max-iters N` | maximum outer iterations |
| `--gmres-restart N` | restart size; zero requests unrestarted mode |
| `--mbj-nodes N` | nominal MBJ block size |
| `--mbj-overlap N` | Schwarz overlap |
| `--pfft-fgmres` | pFFT-preconditioned FGMRES |
| `--pfft-inner-tol F` | inner tolerance |
| `--pfft-inner-iters auto|N` | inner iteration cap |
| `--setup-only` | construct operator and report resources |

### Persistence and diagnostics

| Option | Description |
|---|---|
| `--checkpoint PREFIX` | explicit solver checkpoint prefix |
| `--no-checkpoint` | disable checkpointing |
| `--near-correction-cache FILE` | reusable local correction |
| `--mbj-cache FILE` | reusable MBJ factors |
| `--iteration-log FILE` | per-iteration CSV |
| `--iteration-log-every N` | logging interval |
| `--out FILE` | result JSON |

## 13. Legacy PMCHWT Solver

Build:

```bash
make fmm-only CXX=g++-12 CUDA_HOME=/usr \
  ARCH=-arch=sm_86 -j"$(nproc)"
```

Example:

```bash
bin/bem_cuda_fmm \
  --shape hex_prism --prism-aspect 1 \
  --ka 10 --ri 1.3 0 --ref 4 \
  --system balanced --solver fmm --single \
  --quad 7 --fmm-digits 5 --gmres-tol 1e-5 \
  --prec auto --ntheta 181 \
  --out runs/pmchwt_prism.json
```

Available PMCHWT preconditioners include `mass`, `local`, `ilu0`, and the
experimental `calderon-rwg` operator square. The latter is not a strict
Calderon RWG/BC discretization and must not be reported as one.

Run:

```bash
bin/bem_cuda_fmm --help
```

for the complete legacy CLI.

## 14. Neural Interfaces

### 14.1 PMCHWT GraphSAI

Export:

```bash
bin/bem_cuda_fmm \
  --shape sphere --ref 2 --ka 4.2 --ri 2.2 0 \
  --system balanced --solver fmm --single \
  --quad 7 --fmm-digits 5 \
  --neural-neighbors 24 \
  --neural-dump runs/case.raw
```

Import an exact-system GraphSAI file with `--neural-prec FILE`.
`--neural-action-dump` records Krylov/operator actions for training or
diagnostics.

### 14.2 Muller training export

```bash
make bin/muller_training_dump CXX=g++-12 CUDA_HOME=/usr
```

The resulting exporter provides local blocks and full-operator actions to the
separate neural-training project. Training artifacts are intentionally not
stored in this repository.

## 15. Verification Commands

```bash
make host-checks CXX=g++-12 CUDA_HOME=/usr -j4
make cuda-hessian-check CXX=g++-12 CUDA_HOME=/usr
make cuda-pfft-hessian-check CXX=g++-12 CUDA_HOME=/usr
make cuda-muller-fmm-check CXX=g++-12 CUDA_HOME=/usr
make cuda-muller-edge-check CXX=g++-12 CUDA_HOME=/usr
```

These checks cover mesh topology, dense Muller assembly, singular quadrature,
FMM derivatives, pFFT derivatives, matrix-free action, MBJ cache reuse, and
H(div) prism handling.

## 16. Troubleshooting

### CUDA out of memory

- reduce `ref`;
- reduce GMRES restart;
- use the mixed binary;
- reduce MBJ overlap/block size;
- run `--setup-only` before a long solve;
- do not run two large jobs on the same GPU.

### Residual stagnates

- verify mesh orientation and quality;
- increase quadrature/Duffy order;
- increase FMM digits;
- compare MBJ block sizes;
- inspect the iteration CSV;
- use pFFT-FGMRES only when the pFFT inner action is accurate enough.

### Checkpoint rejected

The geometry, material, quadrature, precision, operator, right-hand side, or
system size changed. Start a new output prefix. Do not force migration for a
production result.

### Result changes after refinement

This is discretization error, even when both linear residuals are small.
Continue refinement or lower the physical size range claimed for that mesh.

### BEM and ADDA disagree

Check, in order:

1. equal-volume scaling and `ka`;
2. exact geometry and aspect-ratio convention;
3. Euler angles and scattering plane;
4. Mueller normalization and polarization signs;
5. independent BEM mesh convergence;
6. independent ADDA DPL convergence;
7. angular interpolation and integration weights.

## 17. Additional Documentation

- `MANUAL.pdf`: mathematical manual in Russian.
- `docs/muller_nodal_mbj.md`: dense/FMM Muller and MBJ implementation.
- `docs/muller_edges.md`: sharp-edge basis and checks.
- `docs/muller_pfft.md`: pFFT-FGMRES details.
- `docs/preconditioner_comparison.md`: PMCHWT preconditioners.
- `docs/hdiv_bem_adda_size_sweep_journal.md`: detailed validation journal.

Rebuild the Russian PDF with:

```bash
tectonic MANUAL.tex --keep-logs --keep-intermediates
```

Generated binaries, `runs/`, checkpoints, and caches are excluded from Git.
Record the exact commit, command line, GPU, compiler, and CUDA version with
every result intended for publication.
