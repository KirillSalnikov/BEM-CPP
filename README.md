# BEM-CPP

Current release: `0.1.0-alpha.5`. The command-line and file formats may still
change before `1.0`; numerical results must include independent convergence
checks described below.

This release includes the automatic hierarchy and restart work from alpha.2,
the CUDA-independent host audit from alpha.3, and a release audit that works
both in a Git checkout and in the published source archive. The primary manual
is checked against the current `./bem` interface. The experimental banded-pFFT
averaging path remains opt-in because validation found it slower than the
default paired GPU solve.

GPU-accelerated boundary-element solvers for electromagnetic scattering by
homogeneous dielectric particles. The repository contains two independent
formulations:

1. a legacy RWG/PMCHWT solver in `bin/bem_cuda_fmm`;
2. a second-kind Muller solver in `bin/muller_nodal_fmm_demo[_fp32]`.

For new sharp-particle calculations, use the Muller solver with the
H(div)-conforming BDM1 edge basis. The PMCHWT path remains available for
regression tests, OBJ workflows, and neural GraphSAI experiments.

## Current Capabilities

- dense reference and matrix-free FMM actions;
- GPU FMM gradient and Hessian kernels;
- pFFT as an inner operator for flexible GMRES;
- Morton block-Jacobi (MBJ) right preconditioning;
- FP32 FFT/near-field work with FP64 Krylov algebra and residual norms;
- P2 nodal currents on smooth surfaces;
- H(div)-BDM1 currents on prisms, cubes, and sharp OBJ meshes;
- GPU far-field and complete Mueller-matrix output;
- adaptive orientation averaging with alpha reconstruction, rotational
  symmetry, and per-orientation checkpoints;
- atomic solver and orientation checkpoints;
- optional GraphSAI import and training-data export.

## Requirements

- Linux;
- CUDA toolkit with cuFFT and cuSPARSE;
- CUDA-capable GPU, compute capability 7.0 or newer;
- C++11 compiler compatible with the installed CUDA toolkit;
- OpenMP;
- Python 3 for build detection and validation;
- optional Python packages for comparison plots and mesh tools.

The core solver has no Python runtime dependency. Install the common report
and mesh-analysis packages with `python3 -m pip install -r
requirements-analysis.txt`. Specialized optional scripts additionally state
their own requirements, such as Gmsh, Bempp-cl, PyTorch, or cuFINUFFT; none is
linked into the production solver.

The mixed-precision target is configured for `sm_86` and was developed on an
RTX 3090 Ti with 24 GiB.

The release test machine uses Ubuntu 24.04, GCC 12.4, CUDA 12.0, and an RTX
3090 Ti. [`environment.cuda.yml`](environment.cuda.yml) provides a separately
pinned CUDA 12.2 build environment. Other CUDA 12.x toolkits should be treated
as unverified until the release audit passes on that system.

The BEM calculation itself runs entirely in C++/CUDA and does not invoke
Python. The `bem` convenience launcher uses only the Python standard library.
The remaining Python files are limited to validation against Mie/ADDA, mesh
conversion, toolchain detection, and automated tests.

## Simple Interface

The recommended entry point is `./bem`. It chooses the surface refinement,
precision, solver, quadrature, cache paths, checkpoint paths, and output names
from one of six reviewed profiles. A normal prism calculation needs only:

```bash
./bem run --shape prism --ka 25 --ri 1.3
```

The default `standard` profile targets a `1e-5` true residual, computes both
incident polarizations, and writes the complete Mueller matrix. In a
fixed-orientation run, its Krylov corrections use sequential current actions
with FP32 L2P, while every restart recomputes the true residual with the FP64
FMM operator. Redundant phase and flattened near-source performance
caches are disabled by default after validation showed a 25.6% memory saving
without a measurable wall-time penalty. It uses
FMM+MBJ for `ka<10` and automatically enables the faster nested pFFT-FGMRES
path for `ka>=10`. The executable is built automatically when absent.
For this nested path, the loose inner pFFT solve trusts its projected residual,
while convergence is accepted only from the independently recomputed outer
FMM residual. The default outer restart is 40.
The inner tolerance is `0.04` below 500000 unknowns and `0.08` for larger
systems; `--pfft-inner-tol` remains an explicit override.

Use the same interface for orientation averaging:

```bash
./bem average --shape prism --ka 25 --ri 1.3 \
  --alpha 256
```

Beta/gamma refinement is adaptive by default in every quality profile. The
solver increases a nested angular level until the Mueller curve, its integral,
and the normalized matrix components meet the profile tolerances. For a
regular prism or cube, both axial rotational symmetry and exact dihedral
beta-mirror reuse are selected automatically when the uniform alpha count is
even. The full quadrature and its weights are preserved, but only one member
of each equivalent orientation pair is solved. Supplying
`--beta` or `--gamma`, or using `--fixed-grid`, deliberately selects the old
fixed-grid mode.

The available quality levels are:

| Profile | Intended use | Numerical and angular control |
|---|---|---|
| `preview` | historical rapid inspection of the `ka=80`, `m=1.3`, `ref=6` regular prism | projected-residual output only; its former reference used the withdrawn uniform high-frequency FMM operator |
| `fast` | adaptive fixed-orientation calculation on any supported shape and material | exact residual ladder `4e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5`; stop only after two independent Mueller-stability checks, otherwise fall back to `standard` |
| `quick` | exploratory runs only | mixed precision, residual `1e-3`, 181 scattering angles; adaptive two-stage fixed-orientation path for sufficiently large built-in meshes |
| `standard` | normal calculations | mixed precision, residual `1e-5`; the automatic two-stage path does not relax this target |
| `memory` | standard-accuracy runs near the GPU-memory limit | same residual as `standard`; sequential electric/magnetic FMM currents reduce peak memory |
| `strict` | publication control | FP64, residual `1e-6`, adaptive `J=2..5`, two successive meshes |

`physical-fast` remains accepted as a deprecated command-line alias for
`fast`; it no longer selects a three-case parameter table. For `fast`,
`quick`, `standard`, and `memory`, rotational symmetry only
supplies a candidate second polarization. The final stage evaluates its full
operator residual and solves a correction whenever the candidate exceeds the
profile tolerance. `./bem validate` rejects output from the former unchecked
symmetry shortcut.

The case-specific preview is intentionally separate from `quick` and
`standard`: it is not a residual-converged result and is rejected for any
unvalidated geometry or parameters. Its historical 34.44 s validation used
the now-withdrawn uniform high-frequency FMM result as the reference. Do not
use the former `0.388%` difference as a current accuracy claim.

```bash
./bem run --shape prism --ka 80 --ri 1.3 --quality preview
```

The universal adaptive mode needs no stored reference result:

```bash
./bem run --shape prism --sides 7 --aspect 1.4 --ka 72 --ri 1.7 \
  --quality fast --out runs/prism_ka72_fast
```

The mesh is selected from `ka`, refractive index, and geometry. Large systems
receive a three-step pFFT warm start; small systems start directly with
FMM+MBJ. At every exact level the full operator residual is recalculated for
both polarizations. The complete solid-angle-weighted Mueller matrix,
normalized `M11`, forward `M11`, and integrated `M11` must change by at most
`1e-3` on two levels that genuinely reduce the residual. The optical-theorem
extinction observable `Re[S1(0)+S2(0)]` is checked at the same tolerance. A
level that performs no new correction is not counted. If the gate is inconclusive, the same
checkpoint continues automatically to the `standard` residual `1e-5`.
The selected result is written to `result.json`; every intermediate level and
`adaptive_fast_summary.json` remain available for audit.

No equal-accuracy speedup over ADDA is currently claimed. A valid claim
requires both programs to use the same residual target, independently
recalculate the final residual, produce the same angular output, and pass
their own discretization-convergence studies.

The launcher does not print a speedup from a stored ADDA timing. Such a number
is valid only when residual target, precision, angular grid, particle
discretization, and wall-time boundary match the BEM run; it therefore belongs
in an explicit comparison report.

For fixed-orientation `./bem run`, `quick`, `standard`, and `memory` select
two-stage checkpoint migration on built-in sphere, cube, and prism meshes
when `ka>=60`, `ref>=4`, the estimated system has at least 100,000 unknowns,
and no expert pFFT hierarchy override was supplied. Split depth and leaf sizes
are derived from refinement and electrical density rather than a table of
three `ka` values. Each profile keeps its own final residual target. Pass
`--single-stage` for a diagnostic control using the previous solver path.

`./bem average` deliberately keeps the paired GPU-GMRES path. The geometry,
operator, MBJ factors, and GPU far-field buffers are constructed once and
reused for every base orientation; completed orientations are checkpointed
atomically. A banded-pFFT alternative was tested at `ka=20` and `ka=60`, but
was respectively `1.05x` and `1.80x` slower in total wall time, so it is not
selected automatically.

An end-to-end cold `quick` check at `ka=60` completed in 169.58 s and reached
a verified `6.851e-4` residual. Its normalized Mueller matrix differed from
the legacy quadrature-7 control result by `0.0131%`. A separate `standard`
continuation reached `8.429e-6`; its normalized Mueller difference from the
strict BEM result was `3.28e-8`. The saved ADDA run used a different residual
criterion, so no cross-program speedup is inferred from these timings.

The first invocation builds the near-correction and MBJ factors under a
content-addressed path in `~/.cache/bem-cpp/operators/v2`; later output
directories with exactly the same geometry, mesh, material, frequency, and
quadrature reuse them. A changed input gets a different path, and both binary
cache formats independently verify their full operator signatures before a
hit is accepted. Set `BEM_CACHE_DIR` to relocate this cache. Final outputs from
`fast`, `quick`, `standard`, and `memory` record verified FMM
residuals. Use `standard` or `strict` when publication accuracy rather than
exploratory or case-specific accuracy is required.

A separate fixed-`ref=6`, `m=1.3` study checked the same strategy at
`ka=20,30,60,80,111`. The first four cases remained below 1% full-Mueller
difference in 15.76--34.44 s. The three-step `ka=111` candidate was rejected
at 45.7% error, so the launcher does not extrapolate `preview` to that size.
See `runs/preview_size_sweep_20260803/preview_size_sweep.pdf` and regenerate it
with `scripts/report_preview_size_sweep.py`.

Use the memory-saving profile without changing the discretization or target
residual:

```bash
./bem run --shape prism --ka 60 --ri 1.3 --quality memory --ref 6
```

On the completed `ka=20, ref=5` control, the new `standard` reduced peak
allocation from the former cached value of 5272 MiB to 3924 MiB (25.6%) with
no measurable wall-time penalty. `memory` reduced it further to 2724 MiB:
30.6% below the new default and 48.3% below the former default. Its full wall
time was only 1.142x the new `standard` time. The normalized full-Mueller
difference was `2.261e-12`. The exact FMM operator, pFFT inner operator, MBJ
factors, quadrature, and FP64 true-residual check remain unchanged.
For `ka=60, ref=6`, the launcher estimates about 15.34 GiB for the new
`standard` policy and 12.02 GiB for `memory`; these estimates are conservative
and the actual allocation depends on the FMM tree.
It cannot make a mesh whose irreducible operator storage exceeds GPU memory
fit; in particular, automatic `ref=7` at `ka=60` remains too large for 24 GiB.
Before starting, `bem` compares the estimate with both total and currently
free VRAM. Sequential-current plans may use up to 92% of total memory while
retaining a 0.75 GiB free-memory reserve; other plans use an 85% ceiling. It
also checks free disk space for two simultaneous checkpoint images, because
atomic replacement briefly retains the old and new files.

Measured cross-profile accuracy and cold-start timings are documented in
[Quality-profile validation](docs/quality_profiles.md).

Reproduce the expanded orientation matrix (multiple shapes, sizes, and
refractive indices) and rebuild its JSON/CSV/Markdown/PNG report with:

```bash
scripts/validate_orientation_profiles.py run --keep-going
scripts/validate_orientation_profiles.py report
```

Completed cases are skipped on restart. Failed stress controls remain marked
as failures and are excluded from cross-profile accuracy comparisons.

For orientation averages with `m>=2.5`, `standard` also selects MBJ100. This
is based on the eight-sided-prism stress control; MBJ50 remains selected below
that threshold because MBJ100 did not improve the `m=2` cube.

Inspect the presets or the exact planned command without running it:

```bash
./bem presets
./bem explain standard
./bem run --shape prism --ka 25 --ri 1.3 --dry-run
```

Each output directory contains `effective_config.json`, executable
`command.sh`, `run.log`, `execution_state.json`, `result.json`, and
`validation.json`. Interrupted
work is resumed with the directory printed at startup:

```bash
./bem resume runs/prism_ka25_m1p3_standard_YYYYMMDD_HHMMSS
./bem validate runs/prism_ka25_m1p3_standard_YYYYMMDD_HHMMSS/result.json
```

For built-in shapes, automatic refinement is a wavelength-resolution rule,
not proof of mesh convergence. Use `--quality strict` for a two-mesh check.
Imported OBJ meshes require an explicit `--ref` because their initial triangle
size is unknown. The launcher refuses an estimated GPU allocation above 85%
of available memory unless the expert override `--allow-memory-risk` is given.

For a built-in shape the initial level is
`ceil(log2(P * ka * L0 * max(1, |m|) / (4*pi)))`, bounded by the profile
minimum. Here `P` is the target points per shortest wavelength, `m` is the
relative refractive index, and `L0` is the longest initial edge scale. The
factor `4*pi` counts the quadratic P2 nodes, including edge midpoints, rather
than only the linear mesh vertices. For `|m|>1`, the shortest wavelength is
the internal wavelength. Thus `ref` grows logarithmically with both particle
size and refractive index, while the number of surface elements grows by
approximately four per added level.
Override the target with `--points-per-wavelength`; an explicit `--ref` always
wins. The `strict` profile then adds a second calculation at `ref+1`.

Profile values are defaults, not immutable settings. Explicit `--solver`,
`--tol`, `--digits`, `--quad`, `--duffy-order`, `--ntheta`, iteration, MBJ,
pFFT, angular-adaptation, and mesh-resolution options replace their profile
values and appear only once in the generated command. The final values and
environment are recorded in `effective_config.json` and `command.sh`.

## Clone and Build

```bash
git clone https://github.com/KirillSalnikov/BEM-CPP.git
cd BEM-CPP

# No separate build command is needed for the simple interface.
./bem run --shape sphere --ka 1 --ri 1.3 --quality quick

# Recommended Muller solver for RTX 3090/3090 Ti
make muller-fp32 CXX=g++-12 CUDA_HOME=/usr -j"$(nproc)"

# FP64 Muller reference and legacy PMCHWT/FMM solver
make bin/muller_nodal_fmm_demo CXX=g++-12 CUDA_HOME=/usr -j"$(nproc)"
make fmm-only CXX=g++-12 CUDA_HOME=/usr ARCH=-arch=sm_86 -j"$(nproc)"
```

Set `CUDA_HOME` to the toolkit prefix when CUDA is not installed under
`/usr`. Change `ARCH` for another GPU architecture.

Inspect the executable before starting a calculation:

```bash
bin/muller_nodal_fmm_demo_fp32 --version
bin/muller_nodal_fmm_demo_fp32 --help
```

Unknown options and options with missing values are rejected. Parent
directories for results, logs, caches, and checkpoints are created
automatically.

## Expert Muller Run

The simple equivalent is `./bem run --shape prism --ka 25 --ri 1.3`. The full
command below exposes every selected control for auditing. It solves a
six-sided prism with `ka=25`, refractive index `1.3`,
H(div)-BDM1 currents, tolerance `1e-5`, MBJ, and pFFT-FGMRES:

```bash
mkdir -p runs/prism_ka25_ref5

OMP_NUM_THREADS=16 bin/muller_nodal_fmm_demo_fp32 \
  --shape prism --sides 6 --aspect 1 \
  --ref 5 --ka 25 --ri 1.3 \
  --edge-mode hdiv --quad 7 --duffy-order 4 \
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

The output path also defines the default checkpoint prefix. Repeating the
same command resumes compatible interrupted solves. Operator, geometry,
material, precision, and right-hand-side signatures are checked before a
checkpoint is accepted.

The public `./bem run` launcher adds the two cyclic-polarization flags shown
automatically for a built-in regular prism in `quick`, `standard`, and
`memory` modes. Use `--independent-polarizations` for an explicit two-solve
control. The `strict` profile always uses independent polarizations.

An earlier benchmark claimed 255.56 s and a `9.26e-6` residual for this
`ka=80`, `m=1.3`, `ref=6` prism. That residual belonged to the old uniform FMM
operator. Rechecking its checkpoint with the corrected high-frequency
two-band operator gave `1.24e-1`, so the old `9.90x` BEM and `12.86x` ADDA
claims are withdrawn. The former three-case `physical-fast` experiment is
retained only as historical regression data: 282.54 s cold, `3.424e-3`
exact-operator residual, and a directly measured `0.00778%` weighted
full-Mueller difference from the strict BEM reference. It is not the stopping
rule of the current universal `fast` profile.

Use `--no-checkpoint` only for disposable benchmarks. Do not use
`--allow-checkpoint-migration` unless an intentionally changed operator is
being audited.

The `--near-correction-cache` and `--mbj-cache` files in the command above are
the useful persistent precomputation. On the Shape A `ref=2` case, a cold
setup took `5.550 + 6.570 = 12.120 s` for the FMM operator and MBJ. Reusing
both caches reduced this to `1.696 + 0.071 = 1.767 s` (`6.86x` faster setup)
without changing the iterative operator or the converged result.

## Orientation Averaging

The recommended adaptive calculation uses the four general-purpose profiles
(`quick`, `standard`, `memory`, and `strict`). The case-specific `preview`
profile does not support orientation averaging:

```bash
./bem average --shape prism --ka 25 --ri 1.3 \
  --quality standard --alpha 256
```

At adaptive level `J`, the nested base grid contains `(2^J+1) * 2^J`
beta/gamma orientations before exact particle symmetry is applied. Increasing
`alpha` changes far-field reconstruction only; it does not add linear solves.
Each completed base orientation is saved under `orientation_parts`, so an
interrupted calculation resumes without discarding completed angles. Reaching
the maximum level without meeting all angular tolerances makes validation
fail rather than silently accepting an under-resolved average.

Use explicit controls only when needed:

```bash
./bem average --shape prism --ka 25 --ri 1.3 \
  --adaptive-levels 2 5 \
  --adaptive-m11-tol 2e-3 \
  --adaptive-integral-tol 2e-3 \
  --adaptive-component-tol 2e-2

# Backward-compatible fixed angular grid.
./bem average --shape prism --ka 25 --ri 1.3 \
  --alpha 256 --beta 8 --gamma 4
```

The low-level shell wrapper also runs the same solver on an explicitly fixed
Euler grid:

```bash
KA=25 RI=1.3 REF=5 SOLVER=fmm RECYCLE_RANK=8 \
ALPHA=8 BETA=8 GAMMA=4 NTHETA=181 THREADS=16 \
OUT="$PWD/runs/prism_ka25_avg" \
scripts/run_muller_orientation_average.sh
```

For `Na x Nb x Ng`, the code solves `Nb x Ng` base orientations, each for two
incident polarizations. The `Na` rotations about the incident beam are
reconstructed in the GPU far-field stage. A regular six-sided prism may use
`--orient-symmetry-order 6`, reducing gamma to a 60-degree fundamental sector.
Only declare a symmetry that is preserved by the material and the actual mesh.

After every base orientation, `OUT.orient.checkpoint` is replaced atomically.
Restarting the same command continues at the next unfinished orientation.

For a large alpha grid, an optional angular Fourier reconstruction reduces
the direct far-field work while preserving complex phase:

```bash
BEM_FARFIELD_SPECTRAL_ALPHA=auto \
  OMP_NUM_THREADS=16 bin/muller_nodal_fmm_demo_fp32 ... \
  --orient-average 256 1 1 --ntheta 73
```

For 256 requested samples, replaying the same converged currents at `ka=25`
and `30` accelerated the far-field stage by `2.49--3.12x` in `auto` mode on a
prism, sphere, and asymmetric polyhedron. The worst Mueller error normalized
by the global result peak was `6.9e-8`. One-step large-`ka` controls selected
112 samples at `ka=40` and 144 at `ka=60`; their errors were respectively
`1.3e-8` and `6.0e-6`. `auto` rounds the conservative bandwidth estimate
`2*(ka+12)` to a multiple of 16 rather than a power of two. Reproduce the
converged-current controls with `scripts/benchmark_farfield_spectral_replay.sh`.
That historical replay requires the original ignored `runs/` checkpoints;
it is not part of the self-contained release smoke test.

With `SOLVER=fmm`, both incident polarizations use GPU-resident FP64 Krylov
vectors and GPU MBJ application; no Krylov vector is copied to the host between
iterations. `RECYCLE_RANK=8` retains an updating basis from completed
orientations and uses it only when its estimated initial residual is at least
2% better than the neighboring-orientation solution. Set `RECYCLE_RANK=0` to
disable it. `SOLVER=pfft` preserves the nested pFFT-FGMRES path and
automatically falls back to its existing CPU-managed Krylov implementation.

## Precision Modes

| Executable | Operator work | Krylov vectors, MBJ, reductions |
|---|---|---|
| `muller_nodal_fmm_demo` | FP64 | FP64 |
| `muller_nodal_fmm_demo_fp32` | FP32 FFT/near work and selected FMM accumulations | FP64 |

The mixed build is not a pure FP32 solver. It reduces GPU memory and FFT cost
while retaining FP64 solution vectors and convergence bookkeeping. Always
check the independently recomputed final residual and compare the Mueller
matrix with an FP64 or mesh-refined control before using a new parameter
range.

`--fmm-near-fp64` switches near interactions back to FP64 in the mixed binary.
`--fmm-near-fp32` enables them explicitly in a compatible build.

`BEM_MIXED_ITERATIVE_REFINEMENT=1` recomputes restart residuals with the full
FP64 FMM operator and uses the mixed operator only for Krylov corrections.
For fixed-orientation `standard`/`memory` runs, sequential current actions make
FP32 L2P accurate enough for Krylov corrections; the FP64 restart residual is
the acceptance criterion. Orientation averaging retains FP64 L2P because its
paired-current path has not passed the same mixed-operator validation. The
dedicated FP64 residual buffers are allocated only in refinement mode.
`BEM_FMM_FOUR_FIELD=1`
enables the experimental joint 12-channel FMM traversal for both
polarizations. Splitting its M2L accumulation into two six-channel launches
reduced register pressure, but it was still 6% slower on the `ka=20` prism, so
both features remain opt-in.

### RTX 3090 Ti FMM tuning

The Muller action contains two vector currents. They are processed together by
default: one six-channel far traversal, one paired L2P kernel, and one paired
near-field kernel reuse geometry and Green-function values for both currents.
Set `BEM_FMM_PAIR_CURRENTS=0` before program startup only for a legacy timing or
when the paired workspace does not fit. `BEM_FMM_PAIR_FAR=0` and
`BEM_FMM_PAIR_L2P=0` isolate individual stages for diagnostics.

The mixed build uses 512 threads for the paired near kernel, 256 for paired
L2P, and 256 for the paired far traversal. The research driver defaults to
`--max-leaf 32`. Hardware-tuning overrides are
`BEM_FMM_P2P_PAIR_THREADS=64|128|256|512`,
`BEM_FMM_L2P_PAIR_THREADS=64|128|256`, and
`BEM_FMM_PAIR_FAR_THREADS=64|128|256|512`. The default 256-thread far block was
fastest over a sustained ten-iteration RTX 3090 Ti run; 512 is retained only
as a diagnostic override. `BEM_FMM_PROFILE_BATCH3=1` prints the far, L2P, and
P2P stage times;
profiling adds CUDA synchronization and must not be used for production timing.

For a real wavenumber, the paired near kernel uses a dedicated formula without
the unused attenuation and imaginary-wavenumber terms. The mixed build also
enables an FP32 phase cache, direct FP32 multipole and M2L-transfer storage,
FP32 local-expansion storage through M2L/L2L, and FP32 accumulation in the
paired P2M, M2L, and L2P stages. It precomputes a balanced near-source index
cache and uses 32 blocks per target leaf; it also keeps a direction-major L2P
phase copy when FP32 L2P is active and memory permits. Assembled operator outputs, Krylov vectors,
MBJ factors, and residual norms remain FP64. The controls
`BEM_FMM_PHASE_CACHE=0`, `BEM_FMM_M2L_STORAGE_FP32=0`,
`BEM_FMM_MULTI_STORAGE_FP32=0`, `BEM_FMM_LOCAL_STORAGE_FP32=0`,
`BEM_FMM_M2L_FP32=0`, `BEM_FMM_L2P_FP32=0`, and
`BEM_FMM_P2P_FAST_TRIG=0` provide stage-by-stage controls. The phase cache
keeps 6144 MiB free by default; change this with
`BEM_FMM_PHASE_CACHE_RESERVE_MB`.

`BEM_FMM_FLAT_NEAR_SOURCES=0` disables the balanced source cache, and
`BEM_FMM_L2P_TRANSPOSED_PHASE_CACHE=0` disables the additional L2P phase
layout. If the direction-major copy does not fit, a warp-per-target L2P kernel
uses the primary target-major table automatically. It saves about 4.43 GiB on
the Shape A `ka=40, ref=3` case and was 1.086x faster than the previous
non-transposed fallback, although it remains about 13% slower than the
memory-rich transposed path. Force it with
`BEM_FMM_L2P_WARP_PER_TARGET=1` or disable it with `=0`.
Both caches have automatic memory fallbacks. On Shape A at `ka=40, ref=3`,
the complete depth-5 action decreased from 2.279 s to 0.398 s (`5.73x`);
the final mixed operator differed from dense assembly by `2.306e-6`.

When no complete phase table fits, set
`BEM_FMM_L2P_PHASE_CACHE_FP16=1` and cap each medium with
`BEM_FMM_L2P_PHASE_CACHE_FP16_MB=1024`. Only that cached part of L2P phases is
stored in FP16; uncached phases, local coefficients, accumulation, and
operator outputs keep their normal precision. At `ka=111, ref=6`, two 919 MiB
partial tables reduced a three-step wall time from 36.87 s to 30.61 s and
changed the full Mueller matrix by only `9.21e-4%`.
`BEM_FMM_STRICT_PAIR_WORKSPACE=0` avoids eagerly reserving the paired FP64
restart workspace on memory-bound refinement runs; strict residuals then
evaluate the currents sequentially.

For orientation averaging on the same `ref=3` geometry, paired GPU GMRES
reduced a fixed ten-step two-polarization solve from 8.780 s to 8.226 s
(`1.067x`) while preserving the residual and all Mueller elements to
`3.8e-9` relative scale. On a separate `ref=2` orientation grid, updating
rank-8 recycling reduced 312 to 298 total iterations and 1.653 s to 1.570 s
(`1.053x`). These gains are incremental to the FMM kernel speedups above, not
additional factors of five.

Keep `--fmm-near-radius 3` for strict calculations. Radius 2 was accurate to
about `2e-6` in low-contrast dense checks, but produced about `8e-3` operator
error for the tested `ka=20, m=3` prism. It is therefore an explicitly
validated low-contrast experiment, not a general speed option.

On the Shape A OBJ microbenchmark, the full optimization sequence reduced the
steady `ref=2` action from 1.086 s to 0.203--0.213 s (`5.1--5.3x`) and the
three-step solve from 6.73 s to 1.328 s (`5.07x`). Measured GPU memory was
2376 MiB. An optional depth-6 tree with the strict depth-5 expansion-order
floor reduced the `ref=3` steady action from 10.39 s to 1.713 s (`6.06x`) and
the one-step solve from 21.60 s to 3.674 s (`5.88x`), while using 13134 MiB.
The depth-6 mode is therefore useful only when the larger workspace fits.
A hard `ka=20, m=3` prism check had relative operator error `2.18e-6`
against dense assembly.

The driver keeps the conservative depth-5 guard by default. To opt into the
validated depth-6 configuration, set `BEM_FMM_ALLOW_DEPTH6=1` and use
`--max-leaf 16`; the driver automatically preserves the depth-5 expansion
order. Native depth-6 orders were rejected because they changed the tested
operator by about 2%.

At very large `ka`, one global plane-wave order can also under-resolve coarse
tree levels. On the `ka=111`, `m=1.3`, `ref=6` prism, the exact-C6 operator
commutator was `7.58e-2` at depth 5 and `7.12e-6` in a depth-3
`--max-leaf 4096` control. FP64 did not improve the depth-5 value. Use
`--symmetry-operator-check-only` before interpreting a high-frequency
symmetric run as strict; `fmm_expansion` in the JSON reports the actual depth,
order, and direction count for both media. The depth-3 control is case-specific;
large leaves also make the direct near field expensive.

For diagnosis, `BEM_FMM_ORDER_REFERENCE_DEPTH=3` can retain a deeper tree
while applying the order required by a depth-3 box. On the same `ka=111` case
this reduced the commutator to `1.83e-5`, but consumed about 21.7 GiB before
GPU operator assembly and therefore is not a production 24 GiB setting.

An experimental memory-efficient alternative partitions the existing FMM
interaction list by tree level. Fine interactions and the direct near field
remain on the depth-5 tree, level 4 uses a depth-4 tree, and levels 1--3 use a
depth-3 tree. Each band therefore stores only the angular order required by
its box size; the three actions are added without interpolating expansion
coefficients. On the same `ka=111, ref=6` prism this reduced the C6 commutator
to `2.23e-6`. The fused three-field implementation used 12426 MiB after setup
and evaluated the two commutator actions in 70.97 s, `3.30x` faster than the
strict scalar-band control at 233.89 s. Enable the validated layout with:

```bash
BEM_FMM_BANDED_SPLIT_DEPTH=3 \
BEM_FMM_BANDED_COARSE_MAX_LEAF=4096 \
BEM_FMM_BANDED_MIDDLE_MAX_LEAF=512 \
BEM_FMM_PAIR_CURRENTS=0 BEM_MULLER_GPU_ASSEMBLY=0 \
bin/muller_nodal_fmm_demo_fp32 [normal solver options]
```

The split depths must match the generated fine, middle, and coarse trees; the
program rejects inconsistent settings. The fused no-P2P bands were checked
against both the scalar-band implementation and a small dense operator. The
launcher now selects the measured layouts automatically only inside the
validated fixed-prism envelope described above.

For a tested low-contrast `ka=20, m=1.3` prism, the explicitly selected
combination `--fmm-near-radius 2 --digits 6` reduced a three-step `ref=2`
microbenchmark from 1.531 s to 1.393 s while a dense small-mesh check gave
operator error `1.23e-6`. It used about 604 MiB more GPU memory. To test
orders above the conservative Muller cap of five, set
`BEM_MULLER_FMM_DIGITS_CAP=6`; never reuse this setting for a new parameter
range without a radius-3 or dense control.
See [`docs/fmm_optimization_3090ti.md`](docs/fmm_optimization_3090ti.md) for
the sweep, accuracy controls, and rejected variants.

## Geometry and Discretization

| Geometry | Typical option | Recommended current basis |
|---|---|---|
| sphere | `--shape sphere` | smooth P2 or H(div) for convergence controls |
| regular prism | `--shape prism --sides N --aspect H/D` | `--edge-mode hdiv` |
| cube | `--shape cube` | `--edge-mode hdiv` |
| imported mesh | `--obj particle.obj` | `--edge-mode hdiv` for sharp edges |

`ref` is a recursive surface-refinement level, not the number of points per
wavelength. Increasing `ka` without increasing `ref` eventually makes the
surface discretization inaccurate. A converged iterative residual only proves
that the discrete system was solved; it does not prove mesh convergence.

For OBJ files, use a closed, consistently oriented, manifold triangular
surface. Inspect the generated mesh and compare at least two refinement levels.
`--edge-refine N --feature-angle F` now also creates a conforming local
refinement band around OBJ edges whose adjacent face normals differ by at
least `F` degrees. It resolves edge singularities; it does not replace the
global `--ref` required to resolve the wavelength.

## Main Muller Options

| Option | Meaning |
|---|---|
| `--ka F` | equal-volume size parameter |
| `--ri F` | real refractive index used by this driver |
| `--ref N` | surface refinement level |
| `--edge-refine N` | local sharp-edge refinement passes |
| `--feature-angle F` | dihedral threshold for local edge refinement |
| `--tol F` | requested relative linear residual |
| `--digits N` | FMM expansion-accuracy target |
| `--quad N` | regular surface quadrature order |
| `--duffy-order N` | singular/adjacent Duffy quadrature order |
| `--max-leaf N` | maximum quadrature sources per FMM leaf |
| `--mbj-nodes N` | target scalar nodes per Morton MBJ block |
| `--mbj-overlap N` | restricted additive-Schwarz overlap |
| `--pfft-fgmres` | use pFFT as a variable right preconditioner |
| `--iteration-log FILE` | write per-iteration timing and residuals |
| `--checkpoint FILE` | override the default checkpoint prefix |
| `--setup-only` | build operators and report memory without solving |
| `--physical-check` | compute both polarizations and Mueller observables |
| `--cyclic-polarization --cyclic-exact-geometry` | build a rotational candidate for the second polarization, verify its operator residual, and correct it when required |
| `--trust-cyclic-exact-geometry` | experimental diagnostic only: skip that residual check; unsafe for production physics because an exact RHS mapping does not guarantee a commuting discrete operator |
| `--symmetry-operator-check-only` | measure `||AT-TA||/||AT||` without a solve for a declared polarization symmetry |
| `--symmetry-checkpoint FILE` | also test a saved solution and its symmetry-reconstructed polarization |
| `--ntheta N` | scattering-angle sample count |

The Muller research executable provides a concise `--help` page. Use this
README and [`MANUAL.md`](MANUAL.md) for the numerical assumptions behind the
options.

## Legacy PMCHWT Path

The legacy solver supports RWG PMCHWT systems, dense/FMM/pFFT backends, OBJ
import, several classical preconditioners, and GraphSAI:

```bash
bin/bem_cuda_fmm \
  --shape hex_prism --prism-aspect 1 \
  --ka 10 --ri 1.3 0 --ref 4 \
  --system balanced --solver fmm --single \
  --quad 7 --fmm-digits 5 --gmres-tol 1e-5 \
  --prec auto --ntheta 181 --out runs/pmchwt_prism.json
```

Run `bin/bem_cuda_fmm --help` for its complete command-line reference.
The legacy `--system muller2` option is not the P2/H(div) second-kind Muller
solver and must not be used as a substitute for it.

## Neural Preconditioner Interface

The PMCHWT executable can export an exact local graph:

```bash
bin/bem_cuda_fmm \
  --shape sphere --ref 2 --ka 4.2 --ri 2.2 0 \
  --system balanced --solver fmm --single \
  --quad 7 --fmm-digits 5 \
  --neural-neighbors 24 \
  --neural-dump runs/training_case.raw
```

A GraphSAI block-CSR file exported by the separate neural project is loaded
with `--neural-prec FILE`. The graph, ordering, material, geometry, and system
must match the exported operator.

The Muller training-data exporter is built with:

```bash
make bin/muller_training_dump CXX=g++-12 CUDA_HOME=/usr
```

## Verification

Run the native checks before a production calculation:

```bash
make host-checks CXX=g++-12 CUDA_HOME=/usr -j4
make cuda-hessian-check CXX=g++-12 CUDA_HOME=/usr
make cuda-pfft-hessian-check CXX=g++-12 CUDA_HOME=/usr
make cuda-muller-fmm-check CXX=g++-12 CUDA_HOME=/usr
make cuda-muller-edge-check CXX=g++-12 CUDA_HOME=/usr
```

Run the complete release audit, including a clean build, CLI checks, all host
tests, all CUDA operator tests, and a small GPU setup calculation:

```bash
scripts/release_audit.sh --gpu
```

The host-only checks used in continuous integration are available as:

```bash
scripts/release_audit.sh --host
```

A small end-to-end physical check solves a `ka=1`, `m=1.3` sphere and enforces
explicit Mueller-matrix error limits against Mie theory:

```bash
examples/run_small_sphere_mie_check.sh
```

The measured alpha-release reference and its acceptance thresholds are stored
in [`reference/v0.1.0-alpha.5/small_sphere.json`](reference/v0.1.0-alpha.5/small_sphere.json).
The same directory contains the raw solver output, validation log, and their
SHA-256 checksums.

A strict physical study additionally requires:

1. a lower-tolerance control;
2. convergence between at least two surface meshes;
3. quadrature and FMM-order controls;
4. all relevant Mueller elements, not only `M11`;
5. comparison with Mie theory for spheres or an independently converged method;
6. complete wall time including setup, solve, and far field.

### Equal-accuracy BEM/ADDA benchmark

The reproducible optimized ten-case benchmark in
[`benchmarks/equal_accuracy_10_optimized_20260804`](benchmarks/equal_accuracy_10_optimized_20260804)
uses the clean official ADDA commit `8f550a7`, three independent complete-wall
repetitions, two independently solved polarizations, a `1e-5` final
recalculated/operator residual, 181 common angles, adjacent discretization
controls, and Mie checks for spheres. The cases were declared before execution:
sphere and regular hexagonal prism at `ka=2/4/6/8/10`, `m=1.3`.
Application caches were new for every repetition; system CUDA/OpenCL compiler
caches were warm and were not flushed.

Relative to the previous BEM implementation under the identical protocol,
mixed iterative refinement gives a median **1.620x** cold full-process
speedup and up to **2.321x**. The complete before/after table and plot are in
[`benchmarks/bem_optimization_10_20260804`](benchmarks/bem_optimization_10_20260804).
On the production `prism_ka6, ref=5` case, a shared validated setup-cache hit
reduces the optimized calculation from 45.73 s to 28.22 s; this is **3.617x**
faster than the former 102.07 s BEM path, with a `3.43e-9` relative L2 change
in the complete Mueller matrix.

This accelerates BEM, but it does not make BEM faster than ADDA in the ten
declared cases. The optimized cold `ADDA wall / BEM wall` ratios are
`0.0070x` to `0.0677x`, so official ADDA remains 14.8 to 142.4 times faster.
These results must not be extrapolated to large particles or orientation
averaging.

## Documentation

- [`MANUAL.md`](MANUAL.md): operational manual.
- [`MANUAL.pdf`](MANUAL.pdf): mathematical and numerical manual in Russian.
- [`docs/muller_nodal_mbj.md`](docs/muller_nodal_mbj.md): Muller and MBJ details.
- [`docs/muller_edges.md`](docs/muller_edges.md): sharp-edge discretization.
- [`docs/muller_pfft.md`](docs/muller_pfft.md): pFFT-FGMRES implementation.
- [`docs/preconditioner_comparison.md`](docs/preconditioner_comparison.md):
  classical PMCHWT preconditioners.

Rebuild the PDF manual with:

```bash
tectonic MANUAL.tex --keep-logs --keep-intermediates
```

Generated binaries, checkpoints, caches, and the complete `runs/` directory
are intentionally excluded from Git.

## License and Citation

The project is distributed under the [MIT License](LICENSE). Third-party
attribution is recorded in [NOTICE](NOTICE). Cite the software metadata from
[CITATION.cff](CITATION.cff) and the numerical methods relevant to the selected
solver.

When publishing numerical results, cite the underlying PMCHWT/RWG, Muller,
FMM/MLFMA, and GMRES/FGMRES methods, and record the exact Git commit and full
command line used for the calculation.

Release history is maintained in [CHANGELOG.md](CHANGELOG.md). Sharp-edge
H(div)-BDM1 results remain alpha quality until mesh convergence and an
independent edge-capable reference agree for the claimed parameter range.
After the audited commit is tagged, `scripts/package_release.sh` creates the
versioned source archive and SHA-256 checksum under `dist/`.
