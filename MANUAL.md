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

The two vector currents of the Muller action are evaluated together by
default. Their six scalar components share the far traversal, L2P geometry,
and direct near-field Green-function values. The mixed build also keeps FP32
coordinate and charge copies for the FP32 near kernel. For a real wavenumber,
that kernel omits attenuation and zero imaginary-wavenumber terms. An FP32
phase cache, direct FP32 multipole and M2L-transfer storage, FP32 local
coefficients through M2L/L2L, and FP32 accumulation are used in the paired
M2L and L2P stages; assembled outputs, Krylov vectors, MBJ factors, and
residual norms remain FP64. On the Shape A benchmark this reduced the steady
`ref=2` action from 1.086 s to 0.203--0.213 s. With the optional strict
depth-6 tree, the `ref=3` action decreased from 10.39 s to 1.713 s.

`BEM_FMM_PAIR_CURRENTS=0` selects the previous separate-current path and avoids
allocating the paired far workspace when set before startup. Stage-isolation
controls are `BEM_FMM_PAIR_FAR=0` and `BEM_FMM_PAIR_L2P=0`.

The RTX 3090 Ti mixed defaults are 512 threads for paired P2P and 256 threads
for paired L2P and far work. Tuning controls are
`BEM_FMM_P2P_PAIR_THREADS=64|128|256|512`,
`BEM_FMM_L2P_PAIR_THREADS=64|128|256`, and
`BEM_FMM_PAIR_FAR_THREADS=64|128|256|512`. The 512-thread far option improved
a one-step microbenchmark but was 7% slower over ten iterations, so 256 remains
the production default.
`BEM_FMM_PROFILE_BATCH3=1` reports the far, L2P, and P2P timings, but
introduces synchronization and is not a production benchmark.
Use `BEM_FMM_PHASE_CACHE=0`, `BEM_FMM_M2L_STORAGE_FP32=0`,
`BEM_FMM_MULTI_STORAGE_FP32=0`, `BEM_FMM_LOCAL_STORAGE_FP32=0`,
`BEM_FMM_M2L_FP32=0`,
`BEM_FMM_L2P_FP32=0`, and
`BEM_FMM_P2P_FAST_TRIG=0` for strict stage-isolation controls.
`BEM_FMM_PHASE_CACHE_RESERVE_MB` controls the default 6144 MiB free-memory
reserve.

When the direction-major L2P phase table does not fit, the mixed solver
automatically uses a warp-per-target kernel with the primary phase table. On
Shape A at `ka=40, ref=3`, this saves about 4.43 GiB and is 1.086x faster than
the old non-transposed fallback. It remains about 13% slower than retaining
both phase layouts. `BEM_FMM_L2P_WARP_PER_TARGET=1|0` forces either fallback
for controlled measurements.

The default high-frequency guard limits the tree to depth 5 when a deeper tree
would lower the FMM expansion order. `BEM_FMM_ALLOW_DEPTH6=1` permits the
validated depth-6 path and automatically retains the depth-5 expansion order.
Use it with `--max-leaf 16` only when the larger workspace fits: the measured
Shape A `ref=3` run used 13134 MiB. A native reduced depth-6 order was rejected
because it changed the tested operator by about 2%.

### 4.2 Near correction

FMM and pFFT approximations are replaced by accurately integrated entries for
singular and topologically adjacent element pairs. `--fmm-near-radius`
controls the geometric near region. `--near-correction-cache FILE` stores the
validated correction and rejects incompatible geometry or physics.

Equivalent local configurations on regular polyhedra reuse correction
templates. This reduces setup cost without changing the operator.

Radius 3 remains the strict Muller default. Radius 2 must be validated for the
actual `ka` and refractive index: it was accurate in low-contrast checks but
failed a `ka=20, m=3` prism operator check by about `8e-3`.
For the tested `ka=20, m=1.3` prism,
`--fmm-near-radius 2 --digits 6` gave `1.23e-6` operator error and reduced a
three-step `ref=2` benchmark from 1.531 s to 1.393 s, at the cost of about
604 MiB more GPU memory. Orders above the conservative Muller cap require an
explicit `BEM_MULLER_FMM_DIGITS_CAP`; this is a validation control, not a
general production default.

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
`muller_nodal_fmm_demo_fp32` enables FP32 pFFT arrays, near-field work, cached
plane-wave phases, and selected M2L/L2P accumulations. Krylov vectors, MBJ
factors, stage outputs, orthogonalization, and norm reductions remain FP64.

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

### 6.3 User-facing launcher

Ordinary calculations should use the root-level `bem` launcher. It applies a
reviewed parameter profile and records the full low-level command, so the
operator remains reproducible without requiring users to choose every tuning
flag.

```bash
./bem run --shape prism --ka 25 --ri 1.3
./bem average --shape prism --ka 25 --ri 1.3 \
  --alpha 256
```

`standard` is the default. It uses direct FMM+MBJ below `ka=10` and the
validated pFFT-FGMRES acceleration from `ka=10` upward. `quick` is a
lower-accuracy exploratory mode, but it never selects less than two uniform
surface refinements because `ref=1` produced an 11% forward-intensity error
in a small-sphere control. `strict` applies the same size policy to an FP64
operator, runs two consecutive surface refinements, and rejects the result
when the normalized Mueller matrix or forward `M11` changes by more than 5%.
The current profile definitions are always available from:

```bash
./bem presets
./bem explain standard
```

For sphere, cube, and regular-prism generators, the launcher selects `ref`
from a points-per-wavelength target. If `h0` is the longest edge scale of the
initial built-in mesh, the selected level is

```text
ref = max(ref_min, ceil(log2(P * ka * h0 / (2*pi)))),
```

where `P` is 4 for `quick` and 8 for `standard`/`strict`. This keeps the
exterior wavelength resolved as particle size grows. It is an initial
discretization rule, not a convergence proof: `strict` always calculates both
that level and `ref+1`. `--points-per-wavelength P` changes the automatic
target and explicit `--ref N` has highest priority. OBJ meshes require
`--ref` explicitly because their initial edge scale is not known to the
launcher. Before execution, the launcher prints a conservative unknown-count
and GPU-memory estimate and blocks a projected allocation above 85% of device
memory.

Every profile parameter is a default. Explicit solver, tolerance, quadrature,
FMM, MBJ, pFFT, angular, and mesh options replace the corresponding profile
value; they are not appended as conflicting duplicate flags. The resolved
command is recorded in `effective_config.json` and `command.sh`.

The output directory is self-describing:

| File | Purpose |
|---|---|
| `effective_config.json` | inputs, profile, estimates, and exact command |
| `command.sh` | directly reproducible low-level invocation |
| `run.log` | complete solver output |
| `result.json` | numerical result and Mueller matrix |
| `validation.json` | residual and finite-value checks |

Continue or inspect a calculation with:

```bash
./bem resume runs/OUTPUT_DIRECTORY
./bem validate runs/OUTPUT_DIRECTORY/result.json
```

## 7. Single-Orientation Muller Calculation

This section documents the expert interface generated by `bem`.

```bash
mkdir -p runs/prism_ka25_ref5

OMP_NUM_THREADS=16 bin/muller_nodal_fmm_demo_fp32 \
  --shape prism --sides 6 --aspect 1 \
  --ref 5 --ka 25 --ri 1.3 \
  --edge-mode hdiv \
  --quad 7 --duffy-order 4 \
  --digits 5 --max-leaf 32 --fmm-near-radius 3 \
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

For the Shape A `ref=2` measurement, the cold FMM and MBJ setup took
`5.550 + 6.570 = 12.120 s`. Loading both persistent caches reduced it to
`1.696 + 0.071 = 1.767 s`, a `6.86x` setup speedup. This does not reduce the
cost of a Krylov iteration; it removes repeated preparation only.

Do not describe cache loading as solver acceleration unless complete wall time
and cold-start time are both reported.

## 9. Orientation Averaging

Recommended adaptive interface:

```bash
./bem average --shape prism --ka 25 --ri 1.3 \
  --quality standard --alpha 256
```

All three profiles use nested adaptive beta/gamma refinement by default:

| Profile | Levels | `M11` curve | `M11` integral | normalized components |
|---|---:|---:|---:|---:|
| `quick` | `J=1..3` | 5% | 5% | 25% |
| `standard` | `J=2..4` | 1% | 1% | 10% |
| `strict` | `J=2..5` | 0.2% | 0.2% | 2% |

Level `J` contains `N_beta=2^J+1` quadrature nodes in `cos(beta)` and
`N_gamma=2^J` uniform azimuthal nodes before particle symmetry is applied.
The nodes are nested, so values already computed at a coarser level are loaded
from `orientation_parts` instead of solved again. After each level, the code
compares the `M11` angular curve, its weighted integral, and every normalized
Mueller component with the preceding level. It accepts the first level that
passes all three tolerances. If the maximum level is reached without passing,
the result is retained for diagnosis but `bem validate` rejects it.

The adaptive controls may be replaced explicitly:

```bash
./bem average --shape prism --ka 25 --ri 1.3 \
  --adaptive-levels 2 5 \
  --adaptive-m11-tol 2e-3 \
  --adaptive-integral-tol 2e-3 \
  --adaptive-component-tol 2e-2
```

Supplying `--beta` or `--gamma`, or selecting `--fixed-grid`, requests the
previous fixed-grid mode. Fixed and adaptive controls cannot be mixed because
their stopping rules are different. The equivalent low-level fixed-grid
wrapper is:

```bash
KA=25 RI=1.3 REF=5 SOLVER=fmm RECYCLE_RANK=8 \
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

For direct FMM+MBJ averaging, `--orient-paired-gpu-gmres` keeps both
polarizations, the Arnoldi bases, and MBJ applications on the GPU. It is the
default compatible orientation path; `--no-orient-paired-gpu-gmres` selects
the previous CPU-managed solver. `--orient-recycle-rank R` keeps an updating
RHS/solution basis across orientations and accepts its projected initial guess
only when it is at least 2% better than the neighboring solution. The tested
rank-8 case reduced 312 to 298 iterations and solve time from 1.653 s to
1.570 s. The paired GPU path reduced a separate large fixed-step solve from
8.780 s to 8.226 s. pFFT-FGMRES and coarse MBJ retain their existing solver
path because they require a flexible or coarse application not implemented by
the paired GPU solver.

The orientation checkpoint stores accumulated weights, Mueller values, timing,
the next orientation, and previous solutions. It is replaced atomically after
every completed base orientation.

Explicit `--orient-warm-max-angle`, `--orient-recycle-rank`,
`--orient-zero-start`, and `--[no-]orient-paired-gpu-gmres` values override the
profile choices in both adaptive and fixed-grid modes. pFFT-FGMRES uses its
flexible CPU-managed outer iteration and therefore disables the specialized
paired-GPU GMRES path. The complete resolved choice is stored in the output
configuration.

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
| `--edge-refine N` | conforming local refinement near prism or OBJ feature edges |
| `--feature-angle F` | OBJ dihedral threshold in degrees |
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

Rebuild the Russian PDF with:

```bash
tectonic MANUAL.tex --keep-logs --keep-intermediates
```

Generated binaries, `runs/`, checkpoints, and caches are excluded from Git.
Record the exact commit, command line, GPU, compiler, and CUDA version with
every result intended for publication.
