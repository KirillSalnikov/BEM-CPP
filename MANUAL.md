# BEM-CPP Operational Manual

This manual describes release `0.1.0-alpha.3`. Automatic hierarchy selection
is available for large built-in meshes, but published speed and physical-error
claims remain limited to the explicitly documented regular-prism controls.
For every new shape, refractive index, or refinement range, compare two meshes
before treating the result as publication quality.

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

When FP32 L2P is active and the direction-major phase table does not fit, the mixed solver
automatically uses a warp-per-target kernel with the primary phase table. On
Shape A at `ka=40, ref=3`, this saves about 4.43 GiB and is 1.086x faster than
the old non-transposed fallback. It remains about 13% slower than retaining
both phase layouts. `BEM_FMM_L2P_WARP_PER_TARGET=1|0` forces either fallback
for controlled measurements.

For larger expansion orders where no complete table fits,
`BEM_FMM_L2P_PHASE_CACHE_FP16=1` enables a target-major partial L2P cache and
`BEM_FMM_L2P_PHASE_CACHE_FP16_MB` limits its memory per medium. Uncached
directions still evaluate their phases on demand; arithmetic and accumulation
are not converted to FP16. A 1024 MiB budget per medium reduced the `ka=111,
ref=6` three-step wall time from 36.87 s to 30.61 s with a `9.21e-4%` change
in the full Mueller matrix. Use `BEM_FMM_STRICT_PAIR_WORKSPACE=0` when the
dedicated paired FP64 restart buffers would leave too little memory for pFFT;
the strict residual then uses the sequential fallback without changing the
operator.

The `standard` profile instead uses FP64 L2P and FP64 restart residuals
automatically for both single calculations and orientation averages. The
unused transposed FP32 L2P table is not allocated in that mode.

The default high-frequency guard limits the tree to depth 5 when a deeper tree
would lower the FMM expansion order. `BEM_FMM_ALLOW_DEPTH6=1` permits the
validated depth-6 path and automatically retains the depth-5 expansion order.
Use it with `--max-leaf 16` only when the larger workspace fits: the measured
Shape A `ref=3` run used 13134 MiB. A native reduced depth-6 order was rejected
because it changed the tested operator by about 2%.

The present diagonal FMM still uses one plane-wave quadrature order on every
tree level. At very large `ka`, `--digits` alone is therefore not a sufficient
accuracy certificate: the order selected from a small leaf can under-resolve
coarser boxes. A `ka=111`, `m=1.3`, `ref=6` hexagonal-prism audit measured the
relative C6 commutator `||AT-TA||/||AT||` as `7.58e-2` for the old depth-5
tree. FP64 gave the same value, so this was FMM truncation rather than roundoff.
Depth 4 reduced it to `8.48e-3`; a depth-3 control with `--max-leaf 4096`
reached `7.12e-6`. The depth-3 setting is a memory-fitting control for this
case, not a universal default, because larger leaves increase direct near
work.

As a causality check, `BEM_FMM_ORDER_REFERENCE_DEPTH=3` retained the depth-5
tree but raised its single global order to the depth-3 requirement. The
commutator fell to `1.83e-5`, but the two media occupied about 21.7 GiB before
GPU operator assembly; enabling that assembly ran out of memory. This setting
is diagnostic only. It proves that coarse-level angular resolution is missing,
but does not provide the memory scaling needed for production runs.

The experimental banded FMM obtains level-dependent angular resolution
without interpolating between unequal spherical grids. It partitions the
original interaction list into disjoint tree-level bands and adds their
actions:

- the fine tree evaluates its finest M2L level and the direct near field;
- an optional middle tree evaluates the next M2L level without P2P;
- the coarse tree evaluates all remaining M2L levels without P2P.

For the validated `ka=111, m=1.3, ref=6` prism, use:

```bash
env BEM_FMM_BANDED_SPLIT_DEPTH=3 \
    BEM_FMM_BANDED_COARSE_MAX_LEAF=4096 \
    BEM_FMM_BANDED_MIDDLE_MAX_LEAF=512 \
    BEM_FMM_PAIR_CURRENTS=0 \
    BEM_MULLER_GPU_ASSEMBLY=0 \
  bin/muller_nodal_fmm_demo_fp32 \
    --shape prism --sides 6 --aspect 1 --ref 6 --ka 111 --ri 1.3 \
    --edge-mode hdiv --quad 7 --duffy-order 4 --digits 5 \
    --max-leaf 256 --fmm-near-radius 3 --fmm-near-fp32 \
    --physical-check --cyclic-polarization \
    --symmetry-operator-check-only --no-dense-validation \
    --out symmetry_operator_check_banded.json
```

This creates depth-5, depth-4, and depth-3 trees. Their exterior/interior
orders were `24/28`, `36/42`, and `55/65`, respectively. Setup took 31.28 s,
occupied 12426 MiB on the tested 24 GiB GPU, and the C6 commutator was
`2.23e-6`, better than both the old depth-5 operator and the monolithic
depth-3 control. Two commutator actions took 70.97 s, versus 233.89 s for the
strict scalar-band control (`3.30x`). A small prism independently agreed with
dense assembly to `2.30e-7` in a deterministic operator probe. Fused bands
without P2P are therefore enabled, while the scalar implementation remains an
independent reference. This is not yet a claimed full-solve speedup. The
program rejects split settings that do not produce the required tree depths,
and the result JSON records all band controls.

For an exactly symmetric particle, audit the implemented discrete operator
without running GMRES:

```bash
bin/muller_nodal_fmm_demo_fp32 \
  --shape prism --sides 6 --aspect 1 --ref 6 --ka 111 --ri 1.3 \
  --edge-mode hdiv --quad 7 --duffy-order 4 --digits 5 \
  --max-leaf 4096 --fmm-near-radius 3 --fmm-near-fp32 \
  --physical-check --cyclic-polarization --cyclic-exact-geometry \
  --symmetry-operator-check-only --no-dense-validation \
  --out symmetry_operator_check.json
```

`--symmetry-checkpoint FILE` additionally evaluates the first-polarization
checkpoint and its reconstructed second polarization. Small meshes may omit
`--no-dense-validation` to record the dense commutator as an independent
reference. Result JSON now records the actual tree depth, expansion order
`p`, and direction count `L` for both media in `fmm_expansion`.

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

The frontend selects an inner tolerance of `0.04` below 500000 system
unknowns and `0.08` for larger systems. An explicit `--pfft-inner-tol`
always overrides this rule.

The inner GMRES accepts its Arnoldi projected residual without a redundant
pFFT action. This is safe for its role as an inexact preconditioner because
only the independently recomputed outer FMM residual can finish the solve.
The default outer restart is 40; together with a 10% projected-residual safety
margin, this avoids an expensive FP64 restart check just above the target on
large meshes.

The outer residual decides convergence. A fast inner solve that increases the
outer iteration count can be slower overall, so setup, inner, outer, and
far-field times must be reported separately.

`--trust-final-projected-residual` is an explicit approximate-mode control. It
can skip the final FMM action only when the projected residual has reached its
target or the fixed outer-iteration limit has been exhausted. The result JSON
then records `fmm_residual_verified=false` and preserves the projected value;
normal profiles never enable this option. The launcher uses it only in the
case-specific `preview` profile described below.

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

`standard` is the default. `preview`, `physical-fast`, `quick`, `standard`, and
`memory` output 181 scattering angles by default. `standard` uses direct FMM+MBJ below `ka=10` and the
validated pFFT-FGMRES acceleration from `ka=10` upward. It recomputes phases
and traverses near-source lists directly, avoiding two caches that did not
improve measured wall time. `quick` is a
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

For a fixed orientation, the `quick`, `standard`, and `memory` run profiles
automatically use a two-stage solve on sufficiently large built-in sphere,
cube, and prism meshes (`ka>=60`, `ref>=4`, and at least 100,000 estimated
unknowns). Up to three inexpensive pFFT-FGMRES steps create the initial field,
after which the checkpoint is migrated to the accurate banded-FMM operator.
The split depth and leaf occupancies are calculated from mesh refinement and
electrical density. The second stage continues to the profile's own residual
target: `1e-3` for `quick` and `1e-5` for `standard` and `memory`.
`--single-stage` disables this automatic path for a control calculation.

Orientation averaging keeps one prepared operator for the entire Euler-angle
schedule and atomically checkpoints every completed base orientation. It uses
paired GPU-GMRES where available. A banded-pFFT averaging variant was measured
but rejected as a default: at `ka=60` it took 252.3 s versus 139.9 s for the
paired path on the same mesh and angular grid.

The end-to-end `quick`, `ka=60` control took 169.58 s from an empty output
directory and reached a verified `6.851e-4` residual. The `standard` control
continued the same strategy to `8.429e-6` without changing the strict BEM
Mueller result within a `3.28e-8` normalized relative L2 difference.

`preview` is restricted to a fixed axial regular hexagonal prism with
`ka=80`, `m=1.3`, and `ref=6`. It uses three outer FGMRES steps and accepts a
projected residual without a final accurate-operator check. Its historical
34.44 s comparison used the withdrawn uniform high-frequency FMM reference;
therefore the former `0.388%` number is not a current accuracy guarantee. Use
`physical-fast` for the checked fast physical result. The launcher rejects
`preview` outside this fixed case.

`physical-fast` is the corresponding physical-result profile for the same
regular prism at `ka=60`, `80`, or `111`, `m=1.3`, and `ref=6`. It
automatically performs the three-step preview, migrates the checkpoint,
makes corrections with the accurate banded-FMM operator, and evaluates the
complete Mueller matrix on 181 angles:

```bash
./bem run --shape prism --sides 6 --aspect 1 --ka 80 --ri 1.3 \
  --ref 6 --quality physical-fast --out runs/prism_ka80_physical_fast
```

The cold measured time is 282.54 s including local-operator and MBJ cache
construction. The exact-operator residual is `3.424e-3`; the weighted relative
L2 difference of all Mueller elements from the strict BEM reference is
`7.780e-5` (0.00778%). This is a physical-observable acceptance profile, not a
replacement for `standard` when a `1e-5` linear residual is mandatory.

The cold measurements at `ka=60/80/111` are 246.22/281.43/455.48 s, giving
`4.13x/11.67x/34.01x` over the saved ADDA-OCL FP32 baselines. The `ka=111`
speed is an operator result, not a 1% discretization claim: `ref=6` provides
only 5.25 nodes per internal wavelength there.

`memory` has the same mesh rule, residual target, quadrature, mixed precision,
pFFT-FGMRES policy, and angular controls as `standard`. In addition to the
compact standard cache policy, it sets `BEM_FMM_PAIR_CURRENTS=0`. Electric and
magnetic FMM currents are evaluated sequentially, avoiding the paired mixed
and strict FP64 workspaces. This preserves the discrete operator and final
accuracy; its tradeoff is an additional FMM traversal.
Use it as:

```bash
./bem run --shape prism --ka 60 --ri 1.3 \
  --quality memory --ref 6
```

For the `ka=60, ref=6` prism, the current conservative launcher estimates are
15.34 GiB for `standard` and 12.02 GiB for `memory`.

For sphere, cube, and regular-prism generators, the launcher selects `ref`
from a points-per-wavelength target. If `h0` is the longest edge scale of the
initial built-in mesh, the selected level is

```text
ref = max(ref_min, ceil(log2(P * ka * h0 * max(1, |m|) / (2*pi)))),
```

where `P` is 4 for `quick` and 8 for `standard`/`memory`/`strict`, and `m` is the
relative refractive index. The multiplier `max(1, |m|)` resolves the shortest
of the exterior and interior wavelengths. It is an initial discretization
rule, not a convergence proof: `strict` always calculates both that level and
`ref+1`. `--points-per-wavelength P` changes the shortest-wavelength target
and explicit `--ref N` has highest priority. OBJ meshes require `--ref`
explicitly because their initial edge scale is not known to the launcher.
Before execution, the launcher prints a conservative unknown-count and
GPU-memory estimate and blocks a projected allocation above 85% of device
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
  --pfft-outer-restart 40 \
  --pfft-order 2 --pfft-correction-radius 0 \
  --pfft-grid-safety 1 \
  --physical-check --ntheta 181 \
  --cyclic-polarization --cyclic-exact-geometry \
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

For a generated regular prism under axial incidence, cyclic symmetry provides
a candidate for the second incident polarization. The two flags in the
command above verify the rotated mesh mapping and transformed right-hand side,
then check the candidate against the actual discrete operator. A Krylov
correction is solved when the residual is too large. The opt-in
`--trust-cyclic-exact-geometry` flag skips this decisive operator check and is
reserved for diagnostics; it is not safe for production physics. At
`ka=111`, the RHS mapping error was `1.39e-14` while the reconstructed-solution
residual was `0.435`, demonstrating why the operator check is mandatory.
With the stricter depth-3 FMM control, the same checkpoint no longer solved the
changed operator (`0.325` residual); after reconvergence the C6 candidate
residual was `3.53e-5` and one correction step reached `9.74e-6`. The corrected
full Mueller matrix differed from the stored ADDA FP32 `dpl=15` result by
`2.74%` in the solid-angle-weighted norm, while it differed from the old
depth-5 BEM result by `54.4%`. This is an operator-accuracy correction, not a
change that can be obtained by continuing GMRES on the old operator.

The previous 255.56 s benchmark and its reported `9.26e-6` residual used the
old uniform high-frequency FMM operator. Its checkpoint has a `1.24e-1`
residual under the corrected two-band operator. Therefore the old `9.90x` BEM
and `12.86x` ADDA speedup claims are withdrawn. The checked replacement is the
`physical-fast` pipeline: 282.54 s cold versus 3285.48 s for ADDA-OCL FP32
`dpl=15`, or `11.63x`, with an exact-operator residual of `3.424e-3` and a
`0.00778%` weighted full-Mueller difference from the strict BEM result.

The `./bem run` frontend selects this exact-prism reconstruction automatically
for `quick`, `standard`, and `memory`. Pass `--independent-polarizations` to
request the two-solve control path. `strict` always keeps independent
polarizations on both meshes.

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
The wrapper writes `execution_state.json` atomically before and after every
pipeline stage. A solver checkpoint is considered usable only when it is
large enough to contain its complete header and solution vector. Before a run,
the wrapper checks current free VRAM and reserves enough disk space for both
the old and temporary checkpoint images used by atomic replacement.

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

All four orientation-capable profiles use nested adaptive beta/gamma
refinement by default; `preview` is restricted to one fixed orientation.
`memory` uses the same angular thresholds as `standard`:

| Profile | Levels | `M11` curve | `M11` integral | normalized components |
|---|---:|---:|---:|---:|
| `quick` | `J=1..3` (`J=1..4` for `m>=2`) | 5% | 5% | 25% |
| `standard` | `J=2..4` (`J=2..5` for `m>=2`) | 1% | 1% | 10% |
| `strict` | `J=2..5` | 0.2% | 0.2% | 2% |

For `standard` orientation averages, `m>=2.5` also selects 100-node MBJ
blocks. An explicit `--mbj-nodes` value overrides this policy.

Level `J` contains `N_beta=2^J+1` quadrature nodes in `cos(beta)` and
`N_gamma=2^J` uniform azimuthal nodes before particle symmetry is applied.
The nodes are nested, so values already computed at a coarser level are loaded
from `orientation_parts` instead of solved again. After each level, the code
compares the `M11` angular curve, its weighted integral, and every normalized
Mueller component with the preceding level. It accepts the first level that
passes all three tolerances. If the maximum level is reached without passing,
the result is retained for diagnosis but `bem validate` rejects it.

Adaptive averages of regular prisms and cubes also use their proper dihedral
rotations. The exact relation
`(beta,gamma) -> (180 degrees-beta, -gamma mod gamma_period)` maps both nodes
to the same alpha-averaged Mueller matrix. Both quadrature weights remain in
the sum, but only one linear-system pair is solved. At `J=5`, this reduces
`33*32=1056` quadrature nodes to 529 unique base-orientation solves. This reuse
requires an even uniform alpha count because the equivalent Euler descriptions
differ by an alpha shift of 180 degrees. It is not inferred for a sphere or an
arbitrary OBJ mesh.

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
