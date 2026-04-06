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
- **Surface pFFT** (`--spfft`) -- 2D FFT per flat face for hex prisms
  - FP32 C2C FFT (2x less memory than Z2Z)
  - CUDA streams: per-face async execution
  - Mixed-radix grid (7-smooth: 2,3,5,7)
  - Density-based grid spacing (~4 pts/cell)
  - Inter-face P2P for cross-face interactions
- **Preconditioners**:
  - `diag` -- diagonal scaling (not recommended for high ka)
  - `ilu0` -- ILU(0) on near-field sparse matrix
  - `nearlu` -- full LU on near-field sparse matrix (small N only)
  - `blockj` -- Block-Jacobi with spatial cell blocking and dense LU per block (GPU-accelerated apply)
- **GMRES variants**: standard, paired (two RHS), GCRO-DR (deflated restarting)
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
bin/bem_cuda --ka 10 --ref 4 --ri 1.3116 0 --fmm --prec blockj --single
```

### Surface pFFT for hex prisms
```bash
bin/bem_cuda --ka 10 --ref 3 --shape hex --ar 1.4286 --spfft --prec blockj --single
```

### Full orientation averaging
```bash
bin/bem_cuda --ka 5 --ref 3 --ri 1.3116 0 --spfft --shape hex --prec blockj --orient 8 8 1 --out result.json
```

### High-ka sweep with RAS preconditioner (ref=4)
```bash
bin/bem_cuda --spfft --shape hex --ar 0.7 --ka 20 --ref 4 --ri 1.3116 0 \
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
| `--fmm` | FMM+GMRES mode | off |
| `--pfft` | pFFT+GMRES mode | off |
| `--spfft` | Surface pFFT+GMRES (hex only) | off |
| `--fmm-digits N` | FMM/pFFT accuracy digits | 3 |
| `--max-leaf N` | Max particles per octree leaf | 64 |
| `--prec TYPE` | Preconditioner: `diag`, `ilu0`, `nearlu`, `blockj` | none |
| `--prec-r F` | Preconditioner radius multiplier | 2.0 |
| `--prec-bs N` | Max block size for Block-Jacobi (triggers adaptive bisection) | 1500 |
| `--prec-overlap N` | RAS overlap layers (0 = standard Block-Jacobi) | 0 |
| `--gmres-restart N` | GMRES restart parameter | 100 |
| `--gmres-tol F` | GMRES relative tolerance | 1e-4 |
| `--gmres-dr` | Use GCRO-DR (deflated restarting) | off |
| `--gmres-k N` | Deflation subspace size | 20 |
| `--ntheta N` | Number of scattering angles | 181 |
| `--quad N` | Triangle quadrature order: 4, 7, 13 | 7 |
| `--out FILE` | Output JSON file | result.json |
| `--fmm-test` | Standalone FMM accuracy test | off |

## Preconditioner Guide

| Mode | Best for | Notes |
|------|----------|-------|
| none | ref <= 3, low ka | Baseline, no setup cost |
| `nearlu` | ref <= 3, many orientations | 9x speedup for 128 orientations |
| `blockj` | ref >= 4, any ka | GPU-accelerated, adaptive block splitting |
| `ilu0` | ref = 3 | Slower than blockj at ref >= 4 |
| `diag` | -- | **Harmful** for PMCHWT at high ka, avoid |

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
  gmres_dr.cu/h       GCRO-DR (deflated restarting GMRES)
  precond.cu/h        Preconditioners (DIAG, ILU0, NEARLU, Block-Jacobi + GPU)
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
