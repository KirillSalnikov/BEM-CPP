# BEM-CPP

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
- orientation averaging with alpha reconstruction and rotational symmetry;
- atomic solver and orientation checkpoints;
- optional GraphSAI import and training-data export.

## Requirements

- Linux;
- CUDA toolkit with cuFFT and cuSPARSE;
- CUDA-capable GPU, compute capability 7.0 or newer;
- C++11 compiler compatible with the installed CUDA toolkit;
- OpenMP;
- Python 3 for build detection and validation;
- NumPy and Matplotlib only for optional comparison plots.

The mixed-precision target is configured for `sm_86` and was developed on an
RTX 3090 Ti with 24 GiB.

The BEM calculation itself runs entirely in C++/CUDA and does not invoke
Python. The remaining Python files are limited to validation against
Mie/ADDA, mesh conversion, toolchain detection, and automated tests.

## Clone and Build

```bash
git clone https://github.com/KirillSalnikov/BEM-CPP.git
cd BEM-CPP

# Recommended Muller solver for RTX 3090/3090 Ti
make muller-fp32 CXX=g++-12 CUDA_HOME=/usr -j"$(nproc)"

# FP64 Muller reference and legacy PMCHWT/FMM solver
make bin/muller_nodal_fmm_demo CXX=g++-12 CUDA_HOME=/usr -j"$(nproc)"
make fmm-only CXX=g++-12 CUDA_HOME=/usr ARCH=-arch=sm_86 -j"$(nproc)"
```

Set `CUDA_HOME` to the toolkit prefix when CUDA is not installed under
`/usr`. Change `ARCH` for another GPU architecture.

## Recommended Muller Run

This command solves a six-sided prism with `ka=25`, refractive index `1.3`,
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
  --pfft-outer-restart 32 \
  --pfft-order 2 --pfft-correction-radius 0 \
  --pfft-grid-safety 1 \
  --physical-check --ntheta 181 \
  --no-dense-validation \
  --iteration-log runs/prism_ka25_ref5/iterations.csv \
  --out runs/prism_ka25_ref5/result.json
```

The output path also defines the default checkpoint prefix. Repeating the
same command resumes compatible interrupted solves. Operator, geometry,
material, precision, and right-hand-side signatures are checked before a
checkpoint is accepted.

Use `--no-checkpoint` only for disposable benchmarks. Do not use
`--allow-checkpoint-migration` unless an intentionally changed operator is
being audited.

The `--near-correction-cache` and `--mbj-cache` files in the command above are
the useful persistent precomputation. On the Shape A `ref=2` case, a cold
setup took `5.550 + 6.570 = 12.120 s` for the FMM operator and MBJ. Reusing
both caches reduced this to `1.696 + 0.071 = 1.767 s` (`6.86x` faster setup)
without changing the iterative operator or the converged result.

## Orientation Averaging

The maintained wrapper runs the same solver for an Euler grid:

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
FP64 FMM operator and uses the mixed operator only for Krylov corrections. It
is a validation/recovery mode: on the tested `ka=20` prism it increased solve
time from 9.844 s to 17.204 s. Its dedicated FP64 pair buffers reduce strict
residual cost but are allocated only in this mode. `BEM_FMM_FOUR_FIELD=1`
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
phase copy when memory permits. Assembled operator outputs, Krylov vectors,
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
| `--ntheta N` | scattering-angle sample count |

The Muller research executable currently has no generated `--help` page.
Unknown flags are not a substitute for documentation; use this README,
[`MANUAL.md`](MANUAL.md), and the maintained wrapper scripts.

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

A strict physical study additionally requires:

1. a lower-tolerance control;
2. convergence between at least two surface meshes;
3. quadrature and FMM-order controls;
4. all relevant Mueller elements, not only `M11`;
5. comparison with Mie theory for spheres or an independently converged method;
6. complete wall time including setup, solve, and far field.

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

No standalone license file is currently present. Confirm usage and
redistribution terms with the repository owner before redistribution.

When publishing numerical results, cite the underlying PMCHWT/RWG, Muller,
FMM/MLFMA, and GMRES/FGMRES methods, and record the exact Git commit and full
command line used for the calculation.
