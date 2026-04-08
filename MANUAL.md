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

## Command-Line Reference

### Required

| Flag | Type | Description |
|------|------|-------------|
| `--ka F` | float | Size parameter ka = 2*pi*a_eff/lambda |

### Physical Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--ri RE IM` | 2 floats | 1.3116 0 | Complex refractive index m = RE + i*IM |

### Geometry & Mesh

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--shape TYPE` | string | sphere | Particle shape: `sphere`, `hex` |
| `--ref N` | int | 3 | Icosphere refinement level (ref=3: 1920 RWG, ref=4: 7680) |
| `--ar F` | float | 1.0 | Hex prism aspect ratio H/D |
| `--obj FILE` | string | -- | Load mesh from Wavefront OBJ file |
| `--subdiv N` | int | 0 | Subdivide OBJ mesh N times (each iteration 4x triangles) |

**Mesh size by refinement level** (sphere/hex):

| ref | Triangles | RWG (N) | System (2N) | Suitable ka |
|-----|-----------|---------|-------------|-------------|
| 2 | 320 | 480 | 960 | 1-2 |
| 3 | 1280 | 1920 | 3840 | 2-5 |
| 4 | 5120 | 7680 | 15360 | 5-10 |
| 5 | 20480 | 30720 | 61440 | 10-20 |
| 6 | 81920 | 122880 | 245760 | 20-40 |

Rule of thumb: ~10 elements per interior wavelength (lambda_int/h >= 10).
The solver prints mesh resolution diagnostics: `lambda_ext/h` and `lambda_int/h`.

### Solver Selection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--solver TYPE` | string | dense | Solver type (see below) |

**Solver types:**

| Solver | Method | Complexity | Best for |
|--------|--------|------------|----------|
| `dense` | Direct LU (cuSOLVER) | O(N^3) time, O(N^2) mem | N <= 8000 |
| `fmm` | Plane-wave MLFMA + GMRES | O(N log N) matvec | Large N, any shape |
| `pfft` | Precorrected FFT + GMRES | O(N log N) matvec | Smooth geometries |
| `spfft` | Surface pFFT + GMRES | O(N log N) matvec | Hex prisms (flat faces) |
| `auto` | Auto-select + auto precond | -- | Recommended for production |

`auto` selects solver based on N, ka, shape and enables auto-preconditioner.

### GMRES Parameters (for iterative solvers)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--gmres-tol F` | float | 1e-4 | Relative convergence tolerance |
| `--gmres-restart N` | int | 100 | Restart parameter m (Krylov subspace size) |

GMRES memory: restart * 2N * 16 bytes. Example: restart=100, N=9216 -> 60 MB.

**Important:** GMRES tolerance should not be tighter than FMM accuracy:
- `--digits 3` -> FMM error ~1e-3 -> use `--gmres-tol 1e-3` or `1e-4` with preconditioner
- Without preconditioner, GMRES may not converge at ref >= 4

### FMM Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--digits N` | int | 3 | FMM accuracy digits (higher = slower but more accurate) |
| `--max-leaf N` | int | 64 | Max particles per octree leaf |

### Preconditioner

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prec TYPE` | string | none | Preconditioner type |
| `--prec-r F` | float | 2.0 | Block-Jacobi cell radius multiplier |
| `--prec-bs N` | int | 1500 | Max RWG per block (larger blocks bisected) |
| `--prec-overlap N` | int | 0 | RAS overlap layers (0 = standard Block-Jacobi) |

**Preconditioner types:**

| Type | Description | Build time | Iteration reduction |
|------|-------------|------------|---------------------|
| `none` | No preconditioning | 0 | -- |
| `diag` | 2x2 block-diagonal (self-interaction) | ~0s | Moderate |
| `blockj` | Block-Jacobi with spatial cells + dense LU | ~30s (ref=4) | 50-90% |
| `blockj` + `--prec-overlap 1` | RAS overlap (extends blocks with neighbors) | ~150s (ref=4) | 95-99% |
| `ilu0` | ILU(0) on near-field sparse matrix | Slow | ~50% |
| `auto` | Auto-select (with `--solver auto`) | Adaptive | Adaptive |

**Recommended for production (ref >= 4, high ka):**
```bash
--prec blockj --prec-r 2.0 --prec-bs 1000 --prec-overlap 1 --gmres-restart 200
```
Creates ~12 blocks with RAS overlap, ~14-22 matvecs instead of 700+ without.

### Orientation Averaging

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--single` | flag | off | Single orientation (no averaging) |
| `--orient NA NB NG` | 3 ints | 8 8 1 | GL quadrature grid (alpha x beta x gamma) |
| `--orient-auto [R]` | opt float | R=4.0 | Auto-compute orientation count from ka, D_max |
| `--orient-tol F` | float | 0 | Adaptive stop tolerance (0 = disabled) |
| `--orient-sym B G` | 2 ints | 1 1 | Symmetry factors: B=2 (beta mirror), G=6 (C6 hex) |
| `--gamma-mirror` | flag | off | Gamma mirror symmetry (sigma_v) |
| `--orient-range I0 I1` | 2 ints | -- | Compute orientations [I0, I1) only (for cluster) |

**`--orient-auto` formula:** angular step = 0.69 * lambda / D_max / (3*R), giving
NA = ceil(360/step), NB = ceil(180/step). Larger R = more orientations.

### Output

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--out FILE` | string | result.json | Output JSON file |
| `--ntheta N` | int | 181 | Scattering angles (0 to 180 degrees) |
| `--quad N` | int | 7 | Triangle quadrature order: 4, 7, 13 |

**Output JSON structure:**
- `mueller`: 16 x Ntheta array (M11, M12, ..., M44)
- `theta`: scattering angles in degrees
- `ka`, `ri`: input parameters
- `timing`: assembly, solve, farfield, total (seconds)

Additionally creates `<outfile>.per_orient` binary with per-orientation Mueller data.

## Working with OBJ Meshes

### Preparing the Mesh

The OBJ file must contain a closed, manifold triangulated surface.
Non-manifold edges or open boundaries will cause incorrect results.

**Requirements:**
- Triangulated faces only (no quads)
- Closed surface (no holes)
- Manifold edges (each edge shared by exactly 2 triangles)
- Consistent face orientation (normals pointing outward)

Repair tools: MeshLab (Filters -> Cleaning), Blender (3D Print Toolkit).

### Mesh Resolution

The solver automatically reports mesh resolution:
```
Resolution: h_avg=0.1057, lambda_ext/h=11.9, lambda_int/h=9.1 [OK]
```

| lambda_int/h | Status | Action |
|--------------|--------|--------|
| >= 10 | OK | Good resolution |
| 6 - 10 | Marginal | Results may have ~5-10% error |
| < 6 | WARNING | Mesh too coarse, use `--subdiv` or reduce ka |

Formula: lambda_int = 2*pi / (ka * |m|), h_avg = sqrt(2 * total_area / (N_tri * sqrt(3)))

### Scaling

The OBJ mesh is automatically scaled so that D_max (maximum bounding box dimension)
corresponds to the size parameter ka. No manual scaling needed.

### Subdivision

`--subdiv N` performs N rounds of Loop-like flat subdivision:
- Each triangle splits into 4 triangles
- N=1: 4x triangles, N=2: 16x triangles
- Example: 1200-triangle OBJ + `--subdiv 2` = 19200 triangles, ~28800 RWG

### Example: Greek Statue

```bash
# Coarse mesh, quick test (1 orientation)
bin/bem_cuda --ka 5 --ri 1.3116 0 --obj model.obj --subdiv 1 \
  --solver auto --single --out greek_test.json

# Full computation with auto-orientations
bin/bem_cuda --ka 5 --ri 1.3116 0 --obj model.obj --subdiv 1 \
  --solver auto --orient-auto --out greek_ka5.json
```

### Memory Estimates for OBJ Meshes

With leaf-to-leaf P2P optimization, GPU memory usage is modest:

| N (RWG) | FMM arrays | Preconditioner | GMRES (m=100) | Total |
|---------|-----------|----------------|---------------|-------|
| 2,304 | ~100 MB | ~200 MB | 15 MB | ~0.3 GB |
| 9,216 | ~170 MB | ~1.2 GB | 60 MB | ~1.4 GB |
| 30,000 | ~500 MB | ~3 GB | 200 MB | ~3.7 GB |
| 100,000 | ~1.5 GB | ~8 GB | 600 MB | ~10 GB |

RTX 3080 Ti (12 GB): up to ~30,000 RWG comfortably.

## Solvers

### Dense LU (default)
Direct LU factorization via cuSOLVER. O(N^3) time, O(N^2) memory.
Practical for N <= 8000 (ref <= 4). No iterative convergence issues.

### FMM+GMRES (`--solver fmm`)
Plane-wave MLFMA with GMRES iterative solver.
Matrix-free O(N log N) matvec. GPU-accelerated P2P, P2M, M2L, L2P kernels.

**P2P near-field**: Leaf-to-leaf neighbor structure (~1 MB) instead of point-to-point
CSR (~6 GB). Each target looks up its leaf, iterates over neighboring leaves, and
evaluates Green's function for all sources in those leaves.

**Two-stream pipeline**: P2P and FMM tree traversal run concurrently on separate
CUDA streams, with results merged at the end.

**Shared tree**: For PMCHWT (two wavenumbers k_ext, k_int), the octree, P2P structure,
and point positions are shared between the two FMM instances. Only k-dependent arrays
(multipole/local coefficients, M2L transfers) are duplicated.

### pFFT+GMRES (`--solver pfft`)
Precorrected FFT: 3D Cartesian grid with FFT-based far-field.
Faster than FMM for smooth geometries.

### Surface pFFT (`--solver spfft`)
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

### Block-Jacobi (`--prec blockj`)
Spatial cell blocks with dense LU per block.

- Cell size: `bb_max_dim / (2.5 / radius_mult)`, giving ~8-20 blocks
- **Adaptive splitting**: blocks > `--prec-bs` RWG are automatically bisected
  along the longest axis (recursive, up to 20 rounds)
- **GPU-accelerated apply**: LU factors uploaded to GPU (row-major for
  coalesced access), CUDA kernel with warp-parallel triangular solve
- Build: typically 5-30s at ref=4
- With RAS overlap (`--prec-overlap 1`): 51x fewer iterations at high ka

### RAS Overlap (`--prec-overlap N`)

Restricted Additive Schwarz (RAS) extends each Block-Jacobi block with
neighboring RWGs from other blocks. The extended system is factorized,
but only own RWGs are scattered back (restricted).

**Benchmark** (hex D/L=0.7, m=1.3116, RTX 3080 Ti):

| ref | ka | Precond | Blocks | Build (s) | Matvecs | Total (s) |
|-----|----|---------|---------|-----------|---------| ----------|
| 3 | 10 | none | -- | -- | 451 | 85 |
| 3 | 10 | blockj | 4 | 1.4 | 392 | 115 |
| 4 | 5 | blockj+RAS(1) | 73 | 31 | 22 | 50 |
| **4** | **16** | **blockj+RAS(1)** | **12** | **154** | **14** | **184** |
| 4 | 20 | blockj+RAS(1) | 4 | 2432 | 13 | 2476 |

## Orientation Averaging

`--orient NA NB NG` specifies an NA x NB x NG Gauss-Legendre quadrature
grid over Euler angles. Typical: `--orient 8 8 1` (64 orientations).

Previous-orientation solution reused as initial guess for the next,
reducing iterations for neighboring angles.

`--orient-auto` computes the orientation count automatically based on
ka * D_max / lambda, ensuring sufficient angular sampling for the
diffraction pattern.

`--orient-range I0 I1` allows splitting orientations across multiple
compute nodes for cluster parallelism.

## Examples

```bash
# 1. Dense LU, sphere, ka=5, single orientation
bin/bem_cuda --ka 5 --ref 3 --ri 1.3116 0 --single

# 2. FMM, hex D/L=0.7, Block-Jacobi+RAS, 64 orientations
bin/bem_cuda --solver fmm --ka 10 --ref 4 --shape hex --ar 1.4286 \
  --prec blockj --prec-overlap 1 --gmres-restart 200 \
  --orient 8 8 1 --out hex_ka10.json

# 3. Auto solver, auto orientations, hex
bin/bem_cuda --solver auto --ka 15 --ref 4 --shape hex --ar 0.7 \
  --orient-auto --out hex_ka15.json

# 4. OBJ mesh with subdivision
bin/bem_cuda --ka 5 --ri 1.5 0.01 --obj statue.obj --subdiv 1 \
  --solver auto --orient-auto --out statue_ka5.json

# 5. Cluster parallelism: split 805 orientations across 4 nodes
bin/bem_cuda --solver auto --ka 15 --ref 4 --shape hex --ar 0.7 \
  --orient 35 23 1 --orient-range 0 202 --out node0.json     # node 0
bin/bem_cuda ... --orient-range 202 404 --out node1.json       # node 1
bin/bem_cuda ... --orient-range 404 606 --out node2.json       # node 2
bin/bem_cuda ... --orient-range 606 805 --out node3.json       # node 3

# 6. High-ka sweep with RAS preconditioner (ref=4, hex D/L=0.7)
bin/bem_cuda --solver spfft --shape hex --ar 0.7 --ka 20 --ref 4 --ri 1.3116 0 \
  --prec blockj --prec-r 2.0 --prec-bs 1000 --prec-overlap 1 \
  --gmres-restart 200 --gmres-tol 1e-4 --ntheta 181 \
  --orient 45 31 1 --out hex_ka20_r4.json
```

## Build

```bash
make -j$(nproc)
```

For CUDA 12.8+ with gcc 15 (too new for nvcc), use an older compiler:
```bash
make -j$(nproc) NVFLAGS="-arch=sm_86 -O3 --use_fast_math -ccbin g++-13 -Xcompiler '-O2 -Wall -std=c++11 -fopenmp' -std=c++11"
```

Set GPU architecture in `Makefile` (default: `sm_86`).

## References

1. Rao, Wilton, Glisson, "Electromagnetic scattering by surfaces of arbitrary shape," IEEE TAP, 1982.
2. Chew, Jin, Michielssen, Song, *Fast and Efficient Algorithms in CEM*, 2001.
3. Graglia, "On the numerical integration of the linear shape functions times the 3-D Green's function," IEEE TAP, 1993.
4. Phillips, White, "A precorrected-FFT method for electrostatic analysis," IEEE TCAD, 1997.
