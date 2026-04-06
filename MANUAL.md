# BEM-CUDA Manual

## Overview

BEM-CUDA solves the electromagnetic scattering problem for dielectric particles
using the Boundary Element Method with PMCHWT formulation.

The 2N x 2N system:

```
[ eta_e*L_ext + eta_i*L_int    -(K_ext + K_int)        ] [J]   [b_J]
[  K_ext + K_int           L_ext/eta_e + L_int/eta_i   ] [M] = [b_M]
```

where L, K are single-layer and double-layer BEM operators,
eta_e = 1, eta_i = 1/|m|, and m is the complex refractive index.

## Solvers

### Dense LU (default)
Direct LU factorization via cuSOLVER. O(N^3) time, O(N^2) memory.
Practical for N <= 8000 (ref <= 4).

### FMM+GMRES (`--fmm`)
Plane-wave MLFMA with GMRES iterative solver.
Matrix-free O(N log N) matvec. GPU-accelerated P2P, P2M, M2L, L2P kernels.

### pFFT+GMRES (`--pfft`)
Precorrected FFT: 3D Cartesian grid with FFT-based far-field.
Faster than FMM for smooth geometries.

### Surface pFFT (`--spfft`)
Specialized for particles with flat faces (hex prisms).
Uses 2D FFT per face instead of 3D FFT.

Features:
- FP32 C2C FFT (2x less memory than Z2Z)
- CUDA streams: per-face async kernel execution
- Mixed-radix grids (7-smooth: factors of 2,3,5,7)
- Automatic density-based grid spacing (~4 points/cell)
- Inter-face P2P kernel for cross-face Green's function

## Preconditioners

Right preconditioning in GMRES: solve Z*M^{-1} * (M*x) = b.

### Diagonal (`--prec diag`)
z_i = r_i / Z_ii.
**Not recommended** for PMCHWT at high ka: diagonal elements are nearly
pure imaginary, dividing rotates the residual spectrum 90 deg, *worsening*
convergence.

### ILU(0) (`--prec ilu0`)
Incomplete LU with zero fill-in on near-field sparse matrix.
Requires >=30% coverage. Build cost scales poorly with coverage.

### Near-field LU (`--prec nearlu`)
Full dense LU on near-field sparse matrix. Best for ref <= 3 with many
orientations (9x speedup for 128 orientations). Limited to N2 <= 8000.

### Block-Jacobi (`--prec blockj`)
Spatial cell blocks with dense LU per block.

- Cell size: `bb_max_dim / (2.5 / radius_mult)`, giving ~8-20 blocks
- **Adaptive splitting**: blocks > `--prec-bs` RWG are automatically bisected
  along the longest axis (recursive, up to 20 rounds)
- **GPU-accelerated apply**: LU factors uploaded to GPU (row-major for
  coalesced access), CUDA kernel with warp-parallel triangular solve
  (32 threads/block, warp shuffle reduction)
- Auto-fallback to CPU (OpenMP) if GPU memory insufficient
- Build: 5x faster than ILU(0) at ref=4
- Convergence: 21% fewer iterations than ILU(0) at ref=4

Parameters:
- `--prec-r F` — radius multiplier for cell size (default 2.0)
- `--prec-bs N` — max block size in RWG; larger blocks are bisected (default 1500)
- `--prec-overlap N` — RAS overlap layers (default 0 = standard Block-Jacobi)

### RAS Overlap (`--prec-overlap N`)

Restricted Additive Schwarz (RAS) extends each Block-Jacobi block with
neighboring RWGs from other blocks. The extended system is factorized,
but only own RWGs are scattered back (restricted).

- `overlap_dist = overlap_layers * avg_extent * 2.0`
- Extended blocks: own RWGs first, then overlap RWGs
- LU factorization on extended 2*B_ext system
- GPU kernel gathers B_ext, solves extended system, writes only B_own

**Optimal config for ref=4, high ka:**
```bash
--prec blockj --prec-r 2.0 --prec-bs 1000 --prec-overlap 1 --gmres-restart 200
```
Creates 12 blocks (avg 768 RWG, extended to ~1400 with 83.6% overlap).

**Benchmark** (hex D/L=0.7, m=1.3116, RTX 3080 Ti):

| ref | ka | Precond | Blocks | Build (s) | Matvecs | Total (s) |
|-----|----|---------|---------|-----------|---------| ----------|
| 3   | 10 | none    | --      | --        | 451     | 85        |
| 3   | 10 | blockj  | 4       | 1.4       | 392     | 115       |
| 3   | 10 | ilu0    | --      | 16.4      | 451     | 103       |
| 3   | 20 | blockj  | 4       | 144       | 33      | --        |
| 3   | 20 | blockj+RAS(1) | 4 | 520       | 8       | --        |
| 4   | 5  | none    | --      | --        | --      | 1145      |
| 4   | 5  | blockj  | 4       | 88        | 333     | 1019      |
| 4   | 16 | blockj  | 4       | 634       | 717+    | 2440+     |
| **4** | **16** | **blockj+RAS(1)** | **12** | **154** | **14** | **184** |
| 4   | 20 | blockj  | 4       | 600       | 183     | 1054      |
| 4   | 20 | blockj+RAS(1) | 4  | 2432      | 13      | 2476      |

Key: more smaller blocks with RAS >> fewer large blocks with RAS.
12-block RAS at ka=16: **51x fewer iterations, 13x faster** vs 4-block baseline.

## GMRES Variants

### Standard GMRES(m)
Restarted GMRES with cuBLAS vector operations and batched Gram-Schmidt
orthogonalization (cublasZgemv).

### Paired GMRES
Two independent GMRES iterations in lockstep for two polarizations,
sharing the batched matvec. Halves GPU kernel launches.

### GCRO-DR (`--gmres-dr`)
Deflated restarting: recycles a subspace of k Ritz vectors across restarts
and orientations. Benefit: 12-18% at small restart (m <= 30), negligible
at m = 50+.

## Particle Shapes

### Sphere (`--shape sphere`)
Icosphere with configurable refinement level.
Refinement r: 20 * 4^r triangles, 30 * 4^r - 60 RWG basis functions.

### Hex Prism (`--shape hex --ar F`)
Hexagonal prism with aspect ratio H/D = F.
For D/L comparison: if D/L = 0.7 then `--ar 1.4286` (= 1/0.7).
Best with `--spfft` (surface pFFT exploits flat faces).

### OBJ Import (`--obj FILE`)
Load triangulated mesh from Wavefront OBJ file.

## Orientation Averaging

`--orient NA NB NG` specifies an NA x NB x NG Gauss-Legendre quadrature
grid over Euler angles. Typical: `--orient 8 8 1` (64 orientations).

Previous-orientation solution reused as initial guess for the next,
reducing iterations for neighboring angles.

Far-field computed in single GPU-batched call for all orientations.

## Output

JSON file with fields:
- `mueller`: 16 x Ntheta array (Mueller matrix M11...M44)
- `theta`: scattering angles in degrees
- `ka`, `ri_re`, `ri_im`: input parameters
- `time_assembly`, `time_solve`, `time_farfield`, `time_total`: timing (s)

## Build

```bash
make -j$(nproc)
```

For CUDA 12.8+ with gcc 15 (too new), use an older compiler:
```bash
make -j$(nproc) NVFLAGS="-arch=sm_86 -O3 --use_fast_math -ccbin g++-13 -Xcompiler '-O2 -Wall -std=c++11 -fopenmp' -std=c++11"
```

## Examples

```bash
# Dense LU, sphere, ka=5, single orientation
bin/bem_cuda --ka 5 --ref 3 --ri 1.3116 0 --single

# Surface pFFT, hex D/L=0.7, Block-Jacobi, ka=10
bin/bem_cuda --ka 10 --ref 3 --shape hex --ar 1.4286 --spfft --prec blockj --single

# Orientation-averaged Mueller, 64 orientations
bin/bem_cuda --ka 5 --ref 3 --shape hex --ar 1.4286 --spfft --prec blockj --orient 8 8 1 --out mueller.json

# High-ka sweep with RAS preconditioner (ref=4, hex D/L=0.7)
# 12 blocks, RAS overlap=1, ~14 matvecs/orient, ~30s/orient
bin/bem_cuda --spfft --shape hex --ar 0.7 --ka 20 --ref 4 --ri 1.3116 0 \
  --prec blockj --prec-r 2.0 --prec-bs 1000 --prec-overlap 1 \
  --gmres-restart 200 --gmres-tol 1e-4 --ntheta 181 \
  --orient 45 31 1 --out hex_ka20_r4.json
```

## References

1. Rao, Wilton, Glisson, "Electromagnetic scattering by surfaces of arbitrary shape," IEEE TAP, 1982.
2. Chew, Jin, Michielssen, Song, *Fast and Efficient Algorithms in CEM*, 2001.
3. Graglia, "On the numerical integration of the linear shape functions times the 3-D Green's function," IEEE TAP, 1993.
4. Phillips, White, "A precorrected-FFT method for electrostatic analysis," IEEE TCAD, 1997.
