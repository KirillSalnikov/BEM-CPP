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

The output path also defines the default checkpoint prefix. Repeating the
same command resumes compatible interrupted solves. Operator, geometry,
material, precision, and right-hand-side signatures are checked before a
checkpoint is accepted.

Use `--no-checkpoint` only for disposable benchmarks. Do not use
`--allow-checkpoint-migration` unless an intentionally changed operator is
being audited.

## Orientation Averaging

The maintained wrapper runs the same solver for an Euler grid:

```bash
KA=25 RI=1.3 REF=5 \
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

## Precision Modes

| Executable | FFT and selected near-field storage/work | Krylov vectors, MBJ, reductions |
|---|---|---|
| `muller_nodal_fmm_demo` | FP64 | FP64 |
| `muller_nodal_fmm_demo_fp32` | FP32 | FP64 |

The mixed build is not a pure FP32 solver. It reduces GPU memory and FFT cost
while retaining FP64 solution vectors and convergence bookkeeping. Always
check the independently recomputed final residual and compare the Mueller
matrix with an FP64 or mesh-refined control before using a new parameter
range.

`--fmm-near-fp64` switches near interactions back to FP64 in the mixed binary.
`--fmm-near-fp32` enables them explicitly in a compatible build.

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

## Main Muller Options

| Option | Meaning |
|---|---|
| `--ka F` | equal-volume size parameter |
| `--ri F` | real refractive index used by this driver |
| `--ref N` | surface refinement level |
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
