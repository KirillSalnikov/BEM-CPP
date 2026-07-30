# BEM-CUDA: GPU-Accelerated Boundary Element Method for Light Scattering

CUDA/C++ implementation of the Boundary Element Method (BEM) with PMCHWT formulation for electromagnetic scattering by dielectric particles.

## Features

- **Dense solver** (LU factorization via cuSOLVER) for small problems (N < 10000)
- **FMM+GMRES** (plane-wave MLFMA) for large problems
  - Multilevel Fast Multipole Algorithm with GPU-accelerated kernels
  - P2P near-field with float32 transcendentals + double accumulation
  - CSR-optimized M2L translations with shared memory transfer reuse
  - Batched evaluation: two charge vectors in a single tree traversal
- **pFFT+GMRES** (precorrected FFT) -- faster than FMM for smooth geometries
- **Surface pFFT** (`--solver spfft`) -- 2D FFT per flat face for hex prisms
  - FP32 C2C FFT (2x less memory than Z2Z)
  - CUDA streams: per-face async execution
  - Mixed-radix grid (7-smooth: 2,3,5,7)
  - Density-based grid spacing (~4 pts/cell)
  - Inter-face P2P for cross-face interactions
- **Preconditioners**:
  - `ilu0` -- ILU(0) on near-field sparse matrix
  - `blockj` -- Block-Jacobi with spatial cell blocking, dense LU per block, RAS overlap (GPU-accelerated)
- **GMRES variants**: standard, paired (two RHS in lockstep)
- **Particle shapes**: icosphere, hexagonal prism (with aspect ratio), OBJ file import
- **Orientation averaging** with Gauss-Legendre quadrature
- **Mueller matrix** computation from far-field amplitudes (GPU-batched)

## Requirements

- CUDA Toolkit 11.0+ (tested with 12.8)
- GPU with compute capability 7.0+ (tested on RTX 3080 Ti, sm_86)
- g++ with C++11 support (g++-13 recommended for CUDA 12.8+)

## Build

```bash
make -j$(nproc)
```

If your default gcc is too new for nvcc, specify an older compiler:
```bash
make -j$(nproc) NVFLAGS="-arch=sm_86 -O3 --use_fast_math -ccbin g++-13 -Xcompiler '-O2 -Wall -std=c++11 -fopenmp' -std=c++11"
```

Set GPU architecture in `Makefile` (default: `sm_86`):
```makefile
ARCH = -arch=sm_86
```

## Quick Start

### Dense solver (small N, exact)
```bash
bin/bem_cuda --ka 5 --ref 3 --ri 1.3116 0 --single --out result.json
```

### FMM+GMRES (large N, iterative)
```bash
bin/bem_cuda --solver fmm --ka 10 --ref 4 --ri 1.3116 0 --prec blockj --single
```

### Surface pFFT for hex prisms
```bash
bin/bem_cuda --solver spfft --ka 10 --ref 3 --shape hex --ar 1.4286 --prec blockj --single
```

### Full orientation averaging
```bash
bin/bem_cuda --solver spfft --ka 5 --ref 3 --ri 1.3116 0 --shape hex --prec blockj --orient 8 8 1 --out result.json
```

### High-ka sweep with RAS preconditioner (ref=4)
```bash
bin/bem_cuda --solver spfft --shape hex --ar 0.7 --ka 20 --ref 4 --ri 1.3116 0 \
  --prec blockj --prec-r 2.0 --prec-bs 1000 --prec-overlap 1 \
  --gmres-restart 200 --gmres-tol 1e-4 --ntheta 181 \
  --orient 45 31 1 --out hex_ka20_r4.json
```
12 blocks with RAS overlap, 14 matvecs/orientation, ~30s/orientation.

## Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--ka F` | Size parameter (required) | -- |
| `--ri RE IM` | Complex refractive index | 1.3116 0 |
| `--ref N` | Mesh refinement level | 3 |
| `--shape TYPE` | Particle: `sphere`, `hex` | sphere |
| `--ar F` | Hex aspect ratio H/D | 1.0 |
| `--obj FILE` | Load mesh from OBJ file | -- |
| `--single` | Single orientation (no averaging) | off |
| `--orient NA NB NG` | Orientation quadrature grid | 8 8 1 |
| `--solver TYPE` | Solver: `dense`, `fmm`, `pfft`, `spfft` | dense |
| `--digits N` | Solver accuracy digits | 3 |
| `--max-leaf N` | Max particles per octree leaf | 64 |
| `--prec TYPE` | FMM preconditioner: `auto`, `mass`, `calderon-rwg`, `local`, `ilu0`, `none` | auto |
| `--gmres-restart N` | GMRES restart parameter | 100 |
| `--gmres-tol F` | GMRES relative tolerance | 1e-4 |
| `--ntheta N` | Number of scattering angles | 181 |
| `--quad N` | Triangle quadrature order: 4, 7, 13 | 7 |
| `--out FILE` | Output JSON file | result.json |

## P2 nodal Muller second-kind solver

The repository also contains a separate implementation of the P2 nodal
Galerkin Muller equation of the second kind described by Luo (2026). It is
kept separate from the legacy RWG `--system muller2` experiment: the two
discretizations are not equivalent, and the legacy mode must not be used as
a substitute for the nodal formulation.

Build and run the physically validated dense path:

```bash
make bin/muller_nodal_demo CXX=g++-12 CUDA_HOME=/usr

bin/muller_nodal_demo \
  --ka 6 --ri 1.5 0 --ref 2 --ntheta 37 \
  --benchmark-gmres --mbj-nodes 50 \
  --out runs/muller_nodal_ref2_mb_jacobi_ka6_n1p5.json
```

`--mbj-nodes 50` forms Morton-ordered blocks with 50 scalar P2 nodes,
equivalent to 100 tangential current unknowns per electric or magnetic
current block. The preconditioner is applied on the right. The JSON records
the true residual, iteration counts, setup time, application time, storage,
and the Mueller matrix.

The FMM tool also accepts `--mbj-overlap N`. It builds a restricted
additive-Schwarz variant: each local solve includes `N` Morton neighbours
on both sides, while only the non-overlapping core is written to the
result. Zero preserves the original MBJ exactly.

Build and run the matrix-free GPU/FMM path. MBJ blocks are independent
and assembled in parallel; on the 16-core Ryzen 9 7950X use
`OMP_NUM_THREADS=16`:

```bash
make bin/muller_nodal_fmm_demo CXX=g++-12 CUDA_HOME=/usr

OMP_NUM_THREADS=16 bin/muller_nodal_fmm_demo \
  --ref 2 --ka 6 --ri 1.5 --tol 1e-6 \
  --digits 7 --max-leaf 128 --mbj-nodes 50 \
  --out runs/muller_nodal_fmm_ref2_ka6_n1p5.json
```

For a system above the dense-validation limit:

```bash
bin/muller_nodal_fmm_demo \
  --ref 3 --ka 12 --ri 1.5 --tol 1e-5 \
  --digits 7 --max-leaf 128 --mbj-nodes 50 \
  --mbj-only --no-dense-validation \
  --out runs/muller_nodal_fmm_ref3_ka12_n1p5.json
```

### Experimental pFFT backend for the nodal Muller operator

The nodal solver can use a regular 3D grid for the nonlocal Green-function
action while retaining the P2 surface mesh and Duffy near correction. The
implementation contracts the gradient and symmetric Hessian kernels with
the three-component current in Fourier space before the inverse transforms.
This reduces one complete Muller matvec from 108 to 24 inverse FFTs. The
exterior and interior kernels use the same auxiliary grid because their
leading singular terms must cancel in the Muller equation.

### Structured cube mesh

The nodal Muller solver has a separate `--shape cube` geometry for testing
structured flat-face discretizations. At refinement `r`, every face contains
`2^r x 2^r` congruent square cells with checkerboard diagonals. Vertices are
shared across cube edges, so the default sharp-edge `H(div)-BDM1` basis remains
conforming. This is different from `--shape prism --sides 4 --aspect 1`, whose
top and bottom faces use radial triangular sectors.

```bash
make muller-fp32 CXX=g++-12 CUDA_HOME=/usr
mkdir -p runs/structured_cube/cache

OMP_NUM_THREADS=16 bin/muller_nodal_fmm_demo_fp32 \
  --shape cube --ref 4 --ka 10 --ri 1.3 \
  --quad 7 --duffy-order 6 --digits 5 --max-leaf 512 \
  --tol 1e-5 --max-iters 4000 --gmres-restart 0 \
  --mbj-nodes 8 --mbj-only --no-dense-validation \
  --physical-check --auto-polarization-symmetry \
  --ntheta 37 --fmm-near-fp32 \
  --pfft-fgmres --pfft-inner-tol 0.1 \
  --pfft-inner-iters auto --pfft-outer-restart 8 \
  --pfft-order 2 --pfft-correction-radius 1 \
  --pfft-grid-safety 0.96 \
  --near-correction-cache runs/structured_cube/cache/ref4.mnc \
  --mbj-cache runs/structured_cube/cache/ref4.mbj \
  --out runs/structured_cube/ref4.json
```

On the RTX 3090 Ti test at `ka=10`, `m=1.3`, and residual `1e-5`, the
structured `ref=4` cube used 18,432 system unknowns instead of 25,344 for the
four-sided prism mesh. Cold strict-pFFT wall time changed from 16.98 s to
13.40 s (`1.27x`). The full Mueller matrices differed by `7.24e-4` in
solid-angle-weighted relative L2 norm. Reusing the exact near-correction and
MBJ caches reduced the structured-cube wall time to 3.35 s.
The checkerboard triangulation is invariant under a 90-degree rotation.
With `--cyclic-polarization --cyclic-exact-geometry`, the second incident
polarization is reconstructed from the first and verified with the full FMM
operator. The warm-cache ref=4 wall time then falls to 2.45 s; its Mueller
matrix differs from an independent two-polarization solve by `2.08e-7`.

Near-correction assembly now identifies adjacent element pairs that are
congruent under a proper rigid rotation. It evaluates the Duffy and regular
quadratures once per unique local template. For H(div)-BDM1 it scatters that
block with the actual global edge indices and orientation signs. For nodal P2
the template key additionally contains every nodal normal and both tangent
directions, so geometrically equal triangles with different local frames are
not mixed. This is enabled by default for regular prisms, spheres, and
subdivided OBJ meshes. `--no-near-template-reuse` retains the previous
pair-by-pair assembly for validation.

For the `ref=4` cube, 45,984 adjacent pairs reduce to 579 templates. The
near-correction build falls from 7.91 s to 0.186 s (`42.6x`). Together with
`--pfft-order 2`, `--mbj-nodes 8`, and exact C4 reconstruction, the complete
cold run takes 3.31 s and a cache-hit run takes 2.15 s. The Mueller matrix
differs from the previous strict order-3 calculation by `1.43e-7`, while the
independently evaluated FMM residual is below `5.6e-6`.

The same automatic template reuse at `ref=4` gives 476 templates for 109,764
pairs on a hexagonal prism. H(div) edge-orientation signs are normalized
outside the template key, so an asymmetric subdivided OBJ now needs 3,922
templates for 79,834 pairs instead of 12,824. Hexagonal-prism near-correction
setup falls from 21.48 s to 0.301 s; the asymmetric OBJ falls from 15.61 s to
0.73 s. The MBJ builder also constructs the element support index once instead
of scanning the complete OBJ mesh separately for every local block.

For strict `--pfft-fgmres` OBJ calculations, omitted tuning options now select
the measured fast defaults `--mbj-nodes 8` and
`--pfft-correction-radius 0`. This is safe in strict mode because pFFT is only
the variable inner preconditioner; every outer action and the final residual
still use FMM. Explicit command-line values always override these defaults.
Use exact near-correction and MBJ caches when the same mesh, material, and
quadrature are solved more than once:

```bash
make muller-fp32 CXX=g++-12 CUDA_HOME=/usr
mkdir -p runs/obj_cache

OMP_NUM_THREADS=16 bin/muller_nodal_fmm_demo_fp32 \
  --shape obj --obj particle.obj --ref 4 --ka 10 --ri 1.3 \
  --quad 7 --duffy-order 6 --digits 5 --max-leaf 512 \
  --tol 1e-5 --max-iters 4000 --gmres-restart 0 \
  --mbj-only --no-dense-validation --physical-check --ntheta 37 \
  --fmm-near-fp32 --pfft-fgmres --pfft-inner-tol 0.1 \
  --pfft-inner-iters auto --pfft-outer-restart 8 \
  --pfft-order 2 --pfft-grid-safety 0.96 \
  --near-correction-cache runs/obj_cache/particle.mnc \
  --mbj-cache runs/obj_cache/particle_mbj8.mbj \
  --out runs/particle_fast.json
```

The first invocation creates both caches; later invocations validate and load
them. On the asymmetric `ref=4`, `ka=10`, `m=1.3` OBJ test, the old strict
two-polarization run took 17.93 s. Sign-normalized templates and the tuned
inner solver reduced a cold run to 13.85 s, while a cache-hit run took
11.62 s (`1.54x` faster). Both FMM residuals were below `8.7e-6`. The complete
Mueller matrix differed from the correction-radius-one strict result by
`2.26e-7` in relative L2 norm.

`--auto-polarization-symmetry` selects exact C4 reconstruction for the cube,
mirror reconstruction for a regular prism, and the C5 mesh symmetry for a
sphere. Every reconstructed solution is evaluated with the full FMM operator.
If its residual exceeds the requested tolerance, the program solves a
correction or falls back to an independent solve. No symmetry is assumed for
an OBJ. On the `ref=4` hexagonal prism this changes the two-polarization wall
time from 11.51 s to 8.62 s cold and 5.72 s when the near and MBJ caches hit.

Template reuse also supports the smooth nodal-P2 sphere: its near correction
falls from 41.00 s to 6.98 s. The explicit `--edge-mode hdiv` sphere is faster
still: the complete C5-checked `ref=4` calculation takes 6.26 s, and comparison
with Mie theory gives a solid-angle-weighted relative M11 error of `6.58e-4`.
The nodal-P2 result takes 15.57 s and gives `8.79e-4` on the same test.

The current pFFT still uses its generic auxiliary 3D interpolation grid.
Therefore this result measures the smaller conforming discretization, not yet
an exact face-lattice convolution. A direct block-Toeplitz face operator must
retain Duffy corrections at coincident, edge-adjacent, and vertex-adjacent
element pairs and must be verified by the outer FMM residual.

FMM remains the default reference backend. The strict accelerated mode uses
pFFT as an inner approximate inverse and evaluates every outer FGMRES action
and accepted residual with FMM:

```bash
mkdir -p runs/cache

OMP_NUM_THREADS=16 bin/muller_nodal_fmm_demo \
  --shape prism --sides 6 --aspect 1 --prism-azimuth-deg 15 \
  --ref 5 --ka 20 --ri 1.5 \
  --quad 4 --duffy-order 6 --tol 1e-5 \
  --mbj-nodes 50 --mbj-only --no-dense-validation \
  --physical-check --mirror-polarization --ntheta 37 \
  --pfft-fgmres --pfft-inner-tol 1e-1 \
  --pfft-inner-iters auto --pfft-outer-restart 8 \
  --pfft-order 3 --pfft-correction-radius 1 \
  --pfft-grid-safety 0.96 \
  --near-correction-cache runs/cache/ref5_ka20_q4_d6.mnc \
  --out runs/ref5_ka20_pfft_fgmres.json
```

`--pfft-correction-radius` is measured in FFT-grid cells. Increasing it
improves the exact-minus-grid near correction but increases setup memory
and the cost of every operator application. `--pfft-grid-safety` multiplies
the common spacing selected from geometry and the larger wave number; a
smaller value creates a finer, more expensive grid.

On the RTX 3090 Ti, paired strict prism runs gave:

| case | FMM / outer iterations | solve speedup | repeated first-system speedup including setup | relative Mueller difference |
|---|---:|---:|---:|---:|
| `ref=3, ka=10` | 47 / 5 | 3.85x | 4.27x | `1.53e-6` |
| `ref=4, ka=15` | 82 / 6 | 5.60x | 5.65x | `7.14e-6` |
| `ref=5, ka=20` | 133 / 10 | 4.96x | 4.75x | `2.08e-6` |

The `ref=5` solve fell from 385.4 s to 77.7 s and the repeated first-system
setup-plus-solve time from 544.2 s to 114.6 s. pFFT, FMM, and MBJ together
used 4.9 GiB of GPU memory. The inner pFFT correction coefficients use
FP32 storage, while FFT arithmetic, outer FMM actions, Krylov algebra, and
the final residual remain FP64. Standalone `--operator-backend pfft`
remains an approximate profiling mode and must not be confused with the
strict `--pfft-fgmres` result. The benchmark report and plot are generated by
[`scripts/report_muller_pfft.py`](scripts/report_muller_pfft.py).

The optional `--near-correction-cache` makes repeated processes reuse the
exact singular/adjacent Galerkin correction. Its fingerprint includes the
complete P2 mesh, material, and quadrature parameters, so incompatible or
partial files are rebuilt. On `ref=5, ka=20`, loading the 710 MiB cache took
0.27 s instead of 140.3 s of integration. Complete physical wall time was
118.3 s, a 4.64x speedup over the 548.6 s FMM reference. The corresponding
plot is generated by
[`scripts/report_muller_cache.py`](scripts/report_muller_cache.py).

### RTX 3090 Ti mixed-precision FMM

The dedicated target compiles for Ampere `sm_86`, uses
`-O3 -march=native` for host code, and selects FP32 for the exact
near-field P2P kernel. In pFFT-FGMRES mode the inner approximate operator
also uses FP32 C2C transforms and Fourier-space kernels. Far-field FMM
translations, Galerkin corrections, MBJ factors, Krylov vectors, accepted
residuals, and Mueller postprocessing remain FP64:

```bash
make muller-fp32 CXX=g++-12 CUDA_HOME=/usr -j8
mkdir -p runs/cache

OMP_NUM_THREADS=16 bin/muller_nodal_fmm_demo_fp32 \
  --shape prism --sides 6 --aspect 1 \
  --ref 5 --ka 20 --ri 1.3 \
  --edge-mode hdiv --quad 7 --duffy-order 4 \
  --digits 5 --max-leaf 64 --fmm-near-radius 3 \
  --operator-backend fmm --tol 1e-5 --max-iters 100 \
  --mbj-only --mbj-nodes 50 --mbj-overlap 0 \
  --near-correction-cache runs/cache/ka20_ref5.near \
  --mbj-cache runs/cache/ka20_ref5_mbj50.cache \
  --physical-check --cyclic-polarization --ntheta 73 \
  --no-dense-validation \
  --iteration-log runs/ka20_ref5_fp32.iterations.csv \
  --out runs/ka20_ref5_fp32.json
```

`--fmm-near-fp64` switches the same binary back to the reference near-field
arithmetic. The standard binary can opt in with `--fmm-near-fp32`.
Checkpoint signatures include this choice, so a checkpoint cannot silently
resume under a different operator.

`--cyclic-polarization` may use a prism rotation only as an initial guess.
Even with `--cyclic-exact-geometry`, the reconstructed polarization is
accepted only when the full discrete FMM operator verifies the requested
residual. Otherwise the solver runs a separately checkpointed correction.

An older checkpoint can be used deliberately with
`--allow-checkpoint-migration`. The file format, vector size, and exact
right-hand-side hash must still match, and the loaded solution is immediately
checked by the selected operator before another Krylov iteration is accepted.

For the command above, FP64 required 41+7 Krylov iterations and 282.25 s
across FMM setup, MBJ setup, both solves, and far-field postprocessing.
Mixed precision required 42+8 iterations and 69.46 s, a `4.06x` complete
speedup. Both final residuals were below `1e-5`. The relative L2 difference
over all 16 Mueller-matrix elements was `2.68e-7` (`M11`: `1.69e-7`).

The exact FMM action used by this mode now contracts the three vector
charges directly during L2P and P2P. It does not materialize three full
gradients and Hessians or evaluate unused scalar potentials. Its
P2M-M2M-M2L-L2L traversal uses three lanes rather than a fourth zero lane,
and M2L interactions are grouped by target expansion.

For the strict `ref=5, ka=25, n=1.3` pFFT-FGMRES case, the measured repeated
complete wall time is 23.70 s with inner tolerance `4e-2`. The exact outer
iteration counts are `13+1`, the verified residuals are `7.70e-6` and
`9.76e-6`, and the inner pFFT counts are `110+7`. This is `23.26x` faster
than the original 551.2 s strict FP64 run. Against an independently solved
two-polarization result on the same mirror-symmetric mesh, the
solid-angle-weighted relative L2 difference is `1.72e-6` over the full
Mueller matrix and `1.04e-6` for M11.

### GPU-resident Muller action and orientation averaging

The mixed-precision binary keeps the exact FMM Galerkin action on the GPU.
The current coefficients are uploaded once per system action. CUDA kernels
project them to quadrature points, evaluate the mass term, combine the
exterior and interior curl/Hessian actions, apply the sparse near correction,
and assemble the two Muller rows. Only the completed `A*x` vector returns to
the host Krylov solver. Set `BEM_MULLER_GPU_ASSEMBLY=0` to select the previous
CPU assembly path for validation.

On the strict `ref=5, ka=25, n=1.3` case this changed the mean exact action
from 0.6282 s to 0.6092 s. The GPU far-field kernel changed the 73-angle
two-polarization postprocessing from 1.421 s to 0.022 s (`63.6x`). Its full
Mueller result differs from the previous FP64 CPU reduction by `2.28e-8` in
relative L2 norm. The complete repeated run changed from 25.30 s to 23.70 s.

The same executable can average orientations without rebuilding the mesh,
near correction, FMM/pFFT engines, or MBJ factors:

```bash
make muller-fp32 CXX=g++-12 CUDA_HOME=/usr -j8
ALPHA=8 BETA=8 GAMMA=4 NTHETA=181 \
  scripts/run_muller_orientation_average.sh
```

`--orient-average Na Nb Ng` uses Gauss-Legendre nodes in `cos(beta)`.
Only `Nb*Ng` systems, each with two incident polarizations, are solved.
The `Na` rotations about the incident beam are reconstructed from that
polarization pair and evaluated in one GPU far-field batch. For a regular
hexagonal prism, `--orient-symmetry-order 6` restricts `gamma` to the
fundamental 60-degree sector. Thus the command above solves 32 base
orientations while representing 1536 samples of the unreduced Euler grid.
This rotational reduction is valid only when the mesh and material preserve
the declared symmetry.

Base orientations are ordered by nearest rotation. A previous solution is
used as an initial guess only when its SO(3) distance is below
`--orient-warm-max-angle` (25 degrees by default); use
`--orient-zero-start` to disable this. The output includes total/mean
iterations, maximum verified residual, solve/far-field timing, and the
quadrature metadata.

After every base orientation, `OUT.orient.checkpoint` atomically stores the
accumulated Mueller matrix and both current solutions. Repeating the same
command resumes at the next orientation. The checkpoint signature covers
the geometry, material, Euler grid, tolerance, and system size. Use
`--no-checkpoint` only for disposable tests.

A direct validation compared 24 unreduced gamma systems with four systems in
the sixfold sector for the same 48-sample grid. The relative full-Mueller
difference was `7.09e-7`, consistent with the `1e-5` linear-solve tolerance,
and the orientation loop changed from 4.74 s to 0.79 s (`6.0x`).

These values verify solver consistency, not superior discretization
accuracy relative to ADDA. Such a claim still requires independent BEM
mesh/quadrature convergence and an ADDA DPL-convergence study with matched
geometry, orientation, normalization, and error targets.

`--mbj-cache` stores the validated FP64 LU blocks. At `ref=5` the 621 MiB
cache loaded in 0.21 s instead of 16.69 s of repeated assembly. Geometry,
material, basis, quadrature, block size, and overlap are fingerprinted;
incompatible and partial files are rejected.

`--pfft-inner-iters auto` uses measured size-dependent limits of 12, 20,
and 24 inner iterations while retaining residual-based early stopping.
The supporting sweep is generated by
[`scripts/report_pfft_inner_sweep.py`](scripts/report_pfft_inner_sweep.py).

For axial incidence on a regular hexagonal prism, the two incident
polarizations can be related by a mirror plane. The following mode rotates
the prism by 15 degrees so that the mirror is the Cartesian `x=y` plane,
uses a mirror-invariant side triangulation, solves only the x-polarized
system, and reconstructs the y-polarized solution. The reconstructed field
is accepted only when its independently evaluated FMM residual satisfies
`--tol`; otherwise GMRES corrects it.

```bash
OMP_NUM_THREADS=16 bin/muller_nodal_fmm_demo \
  --shape prism --sides 6 --aspect 1 \
  --ref 5 --ka 20 --ri 1.5 --tol 1e-5 \
  --digits 5 --quad 4 --duffy-order 6 --max-leaf 64 \
  --mbj-nodes 50 --mbj-overlap 0 --mbj-only \
  --max-iters 400 --gmres-restart 0 \
  --no-dense-validation --physical-check \
  --mirror-polarization --ntheta 73 \
  --out runs/ka20_ref5_mirror_physical.json
```

`--duffy-order 6` is required here: order 4 left a symmetry residual above
the solver tolerance. On the validated `ka=20`, `ref=5`, `n=1.5` case,
mirror reconstruction used 133+0 iterations instead of 133+133. Wall time
was 9:12 instead of 15:33 (`1.70x` overall, `1.99x` for the solve), while
the complete Mueller matrix differed by `4.56e-7` in relative L2 norm from
the independent two-polarization calculation. This optimization is valid
only when the geometry, incidence direction, material, mesh, and requested
polarization basis possess the stated mirror symmetry.

Use `--setup-only` to measure FMM/MBJ construction and memory without
allocating a GMRES basis or starting a solve. The output JSON includes
`system_dofs`, `quadrature_points`, `gpu_memory_delta_mb`, and detailed
FMM/MBJ setup timings:

```bash
OMP_NUM_THREADS=16 bin/muller_nodal_fmm_demo \
  --ref 6 --ka 12 --ri 1.5 --digits 7 --max-leaf 128 \
  --mbj-nodes 50 --setup-only --no-dense-validation \
  --out runs/muller_nodal_fmm_ref6_setup.json
```

On the current CPU validation implementation, at tolerance `1e-8`:

| `ka` | refractive index | baseline GMRES | GMRES + MBJ | solve speedup including MBJ setup |
|---:|---:|---:|---:|---:|
| 1 | 1.3 | 116 | 17 | 4.69x |
| 3 | 1.5 | 204 | 23 | 6.87x |
| 3 | 2.0 | 286 | 44 | 5.31x |
| 6 | 1.5 | 376 | 38 | 9.28x |

The dense table validates the equation and MBJ independently of FMM. The
optimized matrix-free `ref=2`, `ka=6`, `n=1.5` run at tolerance `1e-6`
reduced GMRES from 314 to 29 iterations and solve time from 38.01 s to
3.62 s (`10.49x`). FMM setup took 0.73 s and parallel local-MBJ setup
took 0.23 s. Including both, the first solve was `8.46x` faster;
subsequent polarizations or orientations reuse that setup. The measured
FMM residuals were `8.84e-7` and `8.08e-7`.

The `ref=3`, `ka=12` matrix-free run used 10248 complex unknowns without a
dense matrix and converged to `9.20e-6` in 65 MBJ-GMRES iterations and
30.98 s. FMM setup took 1.94 s and the 52 local MBJ blocks took 0.79 s.
A larger 96-node block did not reduce the iteration count and increased
setup and storage, so 50-node blocks remain the default for this case.

The same `ka=12`, `n=1.5`, `1e-5` case scales as follows on the RTX 3090 Ti.
Setup memory is measured before and after constructing both FMM engines.

| refinement | complex unknowns | quadrature points | FMM setup | MBJ setup | MBJ solve | true residual | GPU setup delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 10,248 | 8,960 | 1.94 s | 0.79 s | 65 it, 30.98 s | 9.20e-6 | not recorded |
| 4 | 40,968 | 35,840 | 5.61 s | 2.84 s | 63 it, 68.97 s | 9.19e-6 | 770 MiB |
| 5 | 163,848 | 143,360 | 20.09 s | 11.06 s | 55 it, 322.53 s | 8.42e-6 | 940 MiB |
| 6 | 655,368 | 573,440 | 82.27 s | 44.77 s | 56 it, 1,069.92 s | 1.08e-5 | 4,268 MiB |

At `ref=6`, the projected GMRES residual crossed `1e-5`; the independently
recomputed true residual was `1.079e-5`. Thus this row is within 8% of the
requested residual rather than strictly below it. The first solve, including
the 127.04 s setup, took 1,196.96 s (19.95 min); subsequent right-hand sides
reuse the setup.

On `ref=4`, the matching unpreconditioned GMRES run reached only
`1.21e-2` after 500 iterations and 557.17 s. MBJ reached `9.18e-6` in
63 iterations and 69.69 s. The observed solve-time ratio is `7.99x`;
because the baseline did not reach the requested tolerance, this is a lower
bound on the same-tolerance acceleration. Including shared FMM setup and the
additional MBJ setup gives a first-solve lower bound of `7.20x`.

For a sharp six-sided prism at `ref=2`, `ka=4`, and `n=2.5` (4976 complex
unknowns), unpreconditioned GMRES only reached `2.60e-3` after 500
iterations. MBJ reached `9.99e-6` in 123 iterations. The reported `4.09x`
time ratio is not a same-tolerance speedup because the baseline did not
converge; this case demonstrates restored convergence.
For this prism, `--mbj-overlap 64` reduced MBJ from 123 to 91 iterations
and solve time from 24.32 s to 18.26 s. Setup increased from 0.40 s to
1.83 s and storage from 15.2 MiB to 181.3 MiB. The same overlap was not
useful on the `ref=3` sphere, so it remains opt-in rather than a default.

The result JSON reports `mbj_setup_breakdown` with Morton ordering,
integral-block assembly, LU factorization, and the effective thread count.
The expensive Green-function and derivative terms are evaluated once per
quadrature-point pair and reused for all four tangential-component pairs.
Near-correction assembly visits only topologically adjacent elements and
uses conflict-free element colors for parallel accumulation.
The LU factors and MBJ setup can be reused for every right-hand side whose
geometry, material, and wavelength are unchanged.

### H(div)-conforming sharp-edge mode

Prisms use a BDM1 surface-current basis by default. Each triangle has two
linearly varying normal-flux moments on each of its three edges. A single
global orientation is assigned to every topological edge, and the
contravariant surface Piola map preserves the shared co-normal flux. The
surface current is therefore H(div)-conforming without averaging normals
or tangent frames through a sharp edge. The original nodal P2 and
piecewise-split modes remain available for controlled comparisons.

The production command is:

```bash
bin/muller_nodal_fmm_demo \
  --shape prism --sides 6 --aspect 1 --ref 2 \
  --edge-mode hdiv \
  --ka 3 --ri 1.3 --tol 1e-5 \
  --digits 5 --fmm-near-radius 3 --max-leaf 64 \
  --mbj-nodes 50 --gmres-restart 100 \
  --mbj-only --no-dense-validation --physical-check --ntheta 73 \
  --near-correction-cache runs/muller_prism_hdiv.cache \
  --out runs/muller_prism_hdiv.json
```

Use `--edge-mode smooth` for the old shared nodal frame and
`--edge-mode split` for the earlier piecewise-smooth nodal experiment.
Neither is the production sharp-particle discretization.

### Strict large-sphere convergence and rotational symmetry

For a sphere, `--sphere-rotational-farfield` solves one incident
polarization and evaluates its far field in two orthogonal scattering
planes. The second polarization follows from the continuous rotational
symmetry of the sphere. The linear operator is not replaced or reduced,
and the final residual is still evaluated by the full FMM action.

The strict `ka=60`, `n=1.3`, tolerance `1e-5` H(div)-BDM1 comparison is:

| mesh | complex unknowns | exact residual | full Mueller error vs Mie | M11 error vs Mie |
|---|---:|---:|---:|---:|
| `ref=6` | 491,520 | `9.19e-6` | `0.769%` | `0.767%` |
| `ref=7` | 1,966,080 | `9.68e-6` | `0.641%` | `0.639%` |

The complete Mueller matrices at `ref=6` and `ref=7` differ by `0.131%`;
their forward-normalized M11 shapes differ by `0.0157%`. The report
generator is
[`scripts/report_sphere_ref6_ref7_convergence.py`](scripts/report_sphere_ref6_ref7_convergence.py).

All analytically nonzero sphere elements were also checked:

| elements | signal L2 norm relative to M11 | ref=6 vs Mie | ref=7 vs Mie | ref=7 vs ref=6 |
|---|---:|---:|---:|---:|
| M11, M22 | `100%` | `0.767%` | `0.639%` | `0.130%` |
| M33, M44 | `99.999%` | `0.767%` | `0.639%` | `0.130%` |
| M12, M21 | `0.444%` | `12.80%` | `9.65%` | `3.72%` |
| M34, M43 | `0.153%` | `38.59%` | `37.28%` | `1.45%` |

The large component-relative percentages for M12 and M34 refer to signals
whose norms are only `0.444%` and `0.153%` of M11. Their ref=7 RMS errors
normalized by forward M11 are `0.00379%` and `0.00504%`, respectively.
Both the component-relative and forward-M11-normalized metrics are kept in
the generated element CSV.

This test also exposed a high-frequency FMM accuracy failure. With
`ref=7 --max-leaf 64`, the tree reached depth 6 and reduced the two
plane-wave orders from `p=18/21` to `p=13/15`. The solver reached a
residual below `1e-5` for that approximate operator, but the full Mueller
error was `17.5%`. The Muller driver now detects this order reduction and
raises the effective leaf limit enough to retain depth 5. For this case it
prints:

```text
[FMM accuracy guard] max-leaf 64 -> 140 to keep depth <= 5 ...
```

The guard preserves the original setting when another tree level does not
reduce the estimated plane-wave order.

Implementation and validation details are in
[`docs/muller_nodal_mbj.md`](docs/muller_nodal_mbj.md).
Sharp-edge implementation details and checks are in
[`docs/muller_edges.md`](docs/muller_edges.md).
The strict `ka=10,15,20,25,30` BEM/ADDA sweep, per-iteration CSV format,
performance investigation, and final timing table are in
[`docs/hdiv_bem_adda_size_sweep_journal.md`](docs/hdiv_bem_adda_size_sweep_journal.md).
Run it with:

```bash
STAGE=bem KAS='10 15 20 25 30' BEM_HYBRID=0 \
  ./scripts/run_hdiv_bem_adda_size_sweep.sh
tail -f runs/hdiv_bem_vs_adda_sweep_n1p3/sweep_journal.log
```

The independent P2-Muller neural training pipeline and RTX 3090 Ti
configuration are in `/home/kirill/neuro/BEM-neural-preconditioner`.

## Preconditioner Guide

| Mode | Best for | Notes |
|------|----------|-------|
| `none` | Baseline | No setup or application cost |
| `mass` | RWG basis normalization | Left preconditioning by the inverse sparse RWG L2 Gram matrix; GPU GMRES only |
| `calderon-rwg` | Operator-squaring control | Experimental strong product with the existing RWG space; not strict RWG/BC Calderon |
| `local` | Small local correction | Overlapping local dense blocks plus near-field Richardson sweep |
| `ilu0` | Larger or more difficult systems | ILU(0) of the sparse near-field PMCHWT matrix; triangular solves use cuSPARSE on GPU |
| `auto` | Default policy | Uses measured shape/accuracy policy |

`ilu0` is right-preconditioned, like GraphSAI. Its sparse pattern is assembled from the same local
PMCHWT blocks used by the neural feature exporter. Set `BEM_PREC_NEAR=N` to change the requested local
degree; large meshes use the topological singular-correction graph by default.

`mass` assembles the real sparse Gram matrix
`G_ij = integral_surface f_i(r) dot f_j(r) dS` for the RWG basis and applies `G^-1`
independently to the electric and magnetic current blocks. The inverse action is computed on the GPU
with a tightly converged inner CG solve. It is a left preconditioner and currently requires
`--krylov gpu-gmres`:

```bash
bin/bem_cuda --solver fmm --shape hex_prism --ref 3 --ka 4.6087 --ri 1.3 0 \
  --system balanced --single --fmm-digits 5 --gmres-tol 1e-3 \
  --krylov gpu-gmres --prec mass --out mass.json
```

`BEM_MASS_TOL` (default `1e-10`) and `BEM_MASS_MAX_ITERS` (default `40`) control the
inner solve. Loosening them makes the preconditioner cheaper but also makes its action less accurate.

`calderon-rwg` applies `G^-1 A G^-1` as a left preconditioner, so GPU GMRES solves the
strong squared system `G^-1 A G^-1 A x = G^-1 A G^-1 b`. The extra application of the
full FMM operator is included in `gmres_matvecs`. This mode uses the available RWG/RWG L2 Gram
matrix and is retained only as a reproducible operator-squaring experiment:

```bash
bin/bem_cuda --solver fmm --shape hex_prism --ref 1 --ka 4.6087 --ri 2.3 0 \
  --system balanced --single --fmm-digits 5 --gmres-tol 1e-3 \
  --krylov gpu-gmres --prec calderon-rwg --out calderon_rwg.json
```

It is not enabled by `auto`. A bounded 20-step control on the documented `ref=1` prism reduced
the true residual only to `4.75e-1` after 44 full-operator evaluations, so this non-dual RWG
square is not currently useful as an accelerator.

A strict Calderon preconditioner is not currently implemented. It requires a conforming RWG/BC dual
discretization, barycentric mesh refinement, and inverse RWG-BC mass matrices. Squaring the existing
RWG/RWG matrix is only a non-conforming operator-squaring experiment and must not be reported as a
Calderon result.

Block-Jacobi details:
- Spatial cell blocks with dense LU per block
- Adaptive splitting: blocks > `--prec-bs` RWG automatically bisected
- GPU apply via CUDA kernel (warp-parallel triangular solve)
- Auto-fallback to CPU if GPU memory insufficient
- **RAS overlap** (`--prec-overlap 1`): extends each block with neighboring RWGs,
  solves on extended system, scatters only own RWGs (Restricted Additive Schwarz).
  Dramatically reduces iteration count at high ka.

Recommended config for ref=4, high ka:
```bash
--prec blockj --prec-r 2.0 --prec-bs 1000 --prec-overlap 1 --gmres-restart 200
```
This creates ~12 blocks with RAS overlap, giving 14 matvecs instead of 700+ without RAS.

### Neural GraphSAI preconditioner

The optional GraphSAI integration loads a sparse complex `2x2` block inverse exported by the sibling
`BEM-neural-preconditioner` project. Existing preconditioners are unchanged. Neural GPU GMRES uses right
preconditioning, matching the network's training objective.

Prepare local features for an exact operator and exit without solving:

```bash
bin/bem_cuda_fmm \
  --solver fmm --shape sphere --ref 2 --ka 4.2 --ri 2.2 0 \
  --system balanced --single --quad 7 --fmm-digits 5 \
  --neural-dump runs/case.raw
```

Checkpoints trained with a wider graph must use the matching dump option, for example
`--neural-neighbors 24 --neural-dump runs/case_k24.raw`. The original model defaults to 10 neighbours.

Run inference in the neural project:

```bash
cd /home/kirill/neuro/BEM-neural-preconditioner
.venv/bin/python -m bem_neural.infer_bem \
  --checkpoint runs/rtx3090ti_graph_sai_v1/best.pt \
  --dump /home/kirill/neuro/BEM-CPP/runs/case.raw \
  --out exports/case.bin
```

Load and apply the result entirely on the GPU:

```bash
cd /home/kirill/neuro/BEM-CPP
bin/bem_cuda_fmm \
  --solver fmm --shape sphere --ref 2 --ka 4.2 --ri 2.2 0 \
  --system balanced --single --quad 7 --fmm-digits 5 \
  --gmres-tol 1e-5 --gmres-restart 100 --gmres-max-cycles 20 \
  --krylov gpu-gmres --neural-prec \
  /home/kirill/neuro/BEM-neural-preconditioner/exports/case.bin \
  --out runs/case_neural.json
```

The dump and solve geometry and physics must be identical. The binary loader checks the RWG count, size
parameter, complex refractive index, PMCHWT scaling, and a rotation-invariant mesh signature before allocating
GPU blocks. One preconditioner can be reused for all incident polarizations and orientations of that operator.

## Output

JSON file containing:
- `mueller`: 4x4 x Ntheta Mueller matrix elements
- `theta`: scattering angles (degrees)
- `ka`, `ri_re`, `ri_im`: input parameters
- Timing breakdown: assembly, solve, far-field, total

## Mesh Sizes

| Refinement | Triangles | RWG (N) | System (2N) | Suitable ka |
|-----------|-----------|---------|-------------|-------------|
| 2 | 320 | 480 | 960 | 1-2 |
| 3 | 1280 | 1920 | 3840 | 2-5 |
| 4 | 5120 | 7680 | 15360 | 5-10 |
| 5 | 20480 | 30720 | 61440 | 10-20 |
| 6 | 81920 | 122880 | 245760 | 20-40 |

Rule of thumb: ~10 elements per wavelength, N ~ 8 ka^2.

## Architecture

```
src/
  main.cpp            CLI entry point
  types.h             Common types (cdouble, Vec3, Timer, CUDA macros)
  mesh.cpp/h          Icosphere + hex prism mesh generation
  rwg.cpp/h           RWG basis functions
  quadrature.h        Dunavant triangle quadrature (orders 4, 7, 13)
  graglia.h           Graglia singular integrals
  rhs.cpp/h           Plane-wave RHS assembly
  assembly.cu/h       Dense Z-matrix assembly (GPU)
  pmchwt.cu/h         PMCHWT system operators
  solver.cu/h         Dense LU solver (cuSOLVER)
  octree.h            Adaptive octree (CPU, header-only)
  sphere_quad.h       Sphere quadrature for FMM
  fmm.cu/h            FMM engine (P2M, M2M, M2L, L2L, L2P)
  p2p.cu/h            P2P near-field CUDA kernels
  pfft.cu/h           3D pFFT acceleration
  surface_pfft.cu/h   2D surface pFFT (per-face, hex prisms)
  bem_fmm.cu/h        BEM-FMM/pFFT coupling (L/K operators, matvec)
  gmres.cu/h          GMRES(m) solver
  block_gmres.cu/h    Paired GMRES (two RHS in lockstep)
  precond.cu/h        Preconditioners (ILU0, Block-Jacobi + RAS, GPU)
  farfield.cu/h       Far-field + Mueller matrix (GPU-batched)
  orient.cpp/h        Orientation averaging (Gauss-Legendre)
  output.cpp/h        JSON output
```

## Performance (RTX 3080 Ti, m=1.3116, hex D/L=0.7)

### Single orientation

| ka | ref | N | Mode | Precond | Matvecs | Assembly | Solve | Total |
|----|-----|------|------|---------|---------|----------|-------|-------|
| 5 | 3 | 1920 | Dense LU | -- | -- | -- | 0.5s | 0.5s |
| 10 | 3 | 2304 | SurfPFFT | none | 451 | -- | 85s | 85s |
| 10 | 3 | 2304 | SurfPFFT | blockj | 392 | 1.4s | 115s | 115s |
| 16 | 4 | 9216 | SurfPFFT | blockj (4 blk) | 717+ | 634s | 1806s+ | 2440s+ |
| **16** | **4** | **9216** | **SurfPFFT** | **blockj+RAS (12 blk)** | **14** | **154s** | **29s** | **184s** |
| 20 | 4 | 9216 | SurfPFFT | blockj (4 blk) | 183 | 600s | 453s | 1054s |
| 20 | 4 | 9216 | SurfPFFT | blockj+RAS (4 blk) | 13 | 2432s | 43s | 2476s |

RAS overlap at ref=4: **51x fewer iterations, 13x faster** vs baseline (ka=16).

### Orientation-averaged (hex D/L=0.7)

| ka | ref | N | Mode | Precond | Total |
|----|-----|------|------|---------|-------|
| 10 | 3 | 2304 | SurfPFFT | ilu0 | 4778s |
| 10 | 3 | 2304 | SurfPFFT | blockj | ~4000s (est) |

## References

- PMCHWT formulation: Rao, Wilton, Glisson (1982)
- Plane-wave MLFMA: Chew, Jin, Michielssen, Song (2001)
- Graglia singular integrals: Graglia (1993)
- pFFT: Phillips, White (1997)
