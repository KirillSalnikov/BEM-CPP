# Experimental pFFT backend for the P2 Muller solver

## Why FFT is not automatic in a surface BEM

ADDA places all dipoles on a regular volume lattice. Its translation
operator is therefore a discrete convolution and one matrix-vector product
can use FFT directly.

The P2 Muller BEM has unknown tangential currents on an unstructured
triangular surface. The Green-function interaction is translation
invariant, but the source and observation points are not a regular lattice.
The pFFT backend introduces an auxiliary Cartesian grid:

1. interpolate weighted surface currents from quadrature points to the grid;
2. transform the grid charges once;
3. contract the gradient and Hessian kernels with the vector current in
   Fourier space;
4. inverse-transform only the three antisymmetric-gradient and three
   Hessian-action components needed by the Muller operator;
5. add exact-minus-grid corrections for nearby point pairs;
6. add the existing Duffy element correction for singular and adjacent
   surface integrals.

This is a precorrected FFT/AIM-style approximation, not the exact volume
convolution used by DDA.

## Muller-specific requirements

The exterior and interior Green-function derivatives enter as differences.
Their leading singular terms cancel analytically. If the two kernels use
different auxiliary grids, their interpolation errors do not cancel.
`MullerFmmOperator` therefore computes one spacing from the larger wave
number and passes exactly the same spacing to both pFFT engines.

`HelmholtzPFFT::evaluate_vector_actions` computes the three charge spectra
once. It forms the antisymmetric gradient and
`H q - trace(H) q` combinations before inverse FFT. The old scalar path
needed 108 inverse FFTs for two current fields and two media per Muller
matvec. The contracted vector path needs 24. The exact-minus-grid near
corrections use the same contractions, so this optimization does not change
the approximated operator.

The exterior and interior instances also share the prepared charge
spectrum. Their kernel spectra and exact-minus-grid corrections remain
separate. Near-correction coefficients are stored in FP32 to reduce memory
traffic; the FFT, outer FMM action, Krylov algebra, residual, and final
solution remain FP64.

## Strict pFFT-FGMRES mode

Standalone pFFT solves a nearby approximate integral equation. It is useful
for profiling, but its physical error grows on the largest tested mesh.
The strict mode instead uses nested flexible GMRES:

1. an inner MBJ-GMRES approximately solves the pFFT system;
2. flexible right preconditioning stores that inner solution as a Krylov
   vector;
3. the outer Arnoldi action is evaluated with the reference FMM operator;
4. convergence is accepted only from the FMM residual.

Thus pFFT changes convergence speed, not the equation whose residual is
driven to the requested tolerance.

## Reproducible strict command

```bash
make bin/muller_nodal_fmm_demo CXX=g++-12 CUDA_HOME=/usr
mkdir -p runs/cache

OMP_NUM_THREADS=16 bin/muller_nodal_fmm_demo \
  --shape prism --sides 6 --aspect 1 --prism-azimuth-deg 15 \
  --ref 5 --ka 20 --ri 1.5 \
  --quad 4 --duffy-order 6 --digits 5 \
  --tol 1e-5 --max-iters 4000 --gmres-restart 0 \
  --mbj-nodes 50 --mbj-only --no-dense-validation \
  --physical-check --mirror-polarization --ntheta 37 \
  --pfft-fgmres --pfft-inner-tol 1e-1 \
  --pfft-inner-iters auto --pfft-outer-restart 8 \
  --pfft-order 3 --pfft-correction-radius 1 \
  --pfft-grid-safety 0.96 \
  --near-correction-cache runs/cache/ref5_ka20_q4_d6.mnc \
  --out runs/ref5_ka20_pfft_fgmres.json
```

FMM is selected by omitting `--operator-backend pfft` or by passing
`--operator-backend fmm`. Standalone approximate pFFT remains available
with `--operator-backend pfft`.

## Measured strict result

For the regular hexagonal prism with `h/D=1`, `n=1.5`, residual tolerance
`1e-5`, and mirror reconstruction of the second polarization:

| case | correction radius | FMM / outer iterations | inner pFFT iterations | FMM solve | strict solve | solve speedup | repeated setup + first solve speedup | Mueller relative L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ref=3, ka=10` | 2 | 47 / 5 | 50 | 12.33 s | 3.20 s | 3.85x | 4.27x | `1.53e-6` |
| `ref=4, ka=15` | 1 | 82 / 6 | 99 | 61.50 s | 10.99 s | 5.60x | 5.65x | `7.14e-6` |
| `ref=5, ka=20` | 1 | 133 / 10 | 206 | 385.38 s | 77.67 s | 4.96x | 4.75x | `2.08e-6` |

The largest strict run used 4.9 GiB of GPU memory for pFFT, FMM, and MBJ
together. Radius 1 is best for the first `ref=4/5` solve because it reduces
setup and sparse-correction traffic. Radius 2 was faster at `ref=3`; it can
also become preferable when many right-hand sides amortize its setup.

On `ref=5`, vector contraction reduced the inner pFFT time from 75.7 s to
38.3 s and total strict solve time from 124.2 s to 86.6 s at the original
12-iteration inner limit. The automatic limit then reduced exact outer FMM
iterations from 15 to 10 and strict solve time to 77.7 s.

`--pfft-inner-iters auto` selects limits 12, 20, or 24 for at most 16,384,
65,536, or more quadrature points. The inner residual tolerance still stops
each application early. These thresholds come from measured sweeps on
`ref=3/4/5`; an explicit positive integer preserves manual control. The
The development sweep is preserved in the benchmark history rather than in
the production command set.

## Exact near-correction cache

`--near-correction-cache PATH` stores the singular/adjacent Galerkin
correction after its first construction. The cache header contains two
independent 64-bit fingerprints over all P2 nodes, local frames,
connectivity, material parameters, and quadrature orders. A truncated file
or any parameter mismatch is rejected and rebuilt. The file is written to a
temporary path and atomically renamed only after it is complete.

For `ref=5, ka=20`, the cache contains 13,276,876 sparse coefficients and
occupies 710 MiB. Loading took 0.27 s instead of 140.3 s. Including pFFT and
MBJ construction, setup fell from 175.9 s to 34.4 s. Measured complete
physical times were:

| mode | complete time | speedup over FMM |
|---|---:|---:|
| reference FMM | 548.6 s | 1.00x |
| strict pFFT-FGMRES, first run with fixed limit 12 | 267.5 s | 2.05x |
| strict pFFT-FGMRES, cache hit and automatic limit | 118.3 s | 4.64x |

The cache is independent of the incident field. It is therefore reusable
when an orientation sweep changes the propagation direction and
polarization while keeping the particle mesh fixed. Rotating or retriangling
the mesh changes the fingerprint and intentionally causes a cache miss.

Cache benchmarks should report both the cold build and the warm cache-hit run.

## RTX 3090 Ti mixed precision

`make muller-fp32` builds `bin/muller_nodal_fmm_demo_fp32` for `sm_86` with
host `-O3 -march=native`. The exact near-field P2P gradient/Hessian kernel
uses FP32. In pFFT-FGMRES mode the inner approximate operator additionally
uses FP32 C2C transforms and Fourier-space kernels.
The FMM far field, exact sparse Galerkin correction, MBJ, Krylov algebra,
residual verification, and physical output remain FP64. Use
`--fmm-near-fp64` for the reference path or `--fmm-near-fp32` to opt into
the same mode from the standard binary.

Prism symmetry is never a substitute for the residual check.
`--cyclic-polarization --cyclic-exact-geometry` first evaluates the rotated
solution with the full discrete FMM operator. A result above `--tol` is used
only as an initial guess for a separately checkpointed correction solve.

On the hexagonal `ref=5, ka=20, n=1.3` case at residual `1e-5`, the first
polarization changed from 41 iterations and 186.78 s to 42 iterations and
51.20 s. Including the corrected second polarization, FMM and MBJ setup,
and 73-angle far field, measured time changed from 282.25 s to 69.46 s
(`4.06x`). The relative L2 difference over all Mueller elements was
`2.68e-7`.

For the regular hexagonal `ref=5, ka=25, n=1.3` prism at a verified FMM
residual of `1e-5`, the optimized exact action contracts only the six
vector derivative combinations required by the Muller operator. The FMM
traversal uses three charge lanes, skips unused scalar-potential L2P, and
groups M2L interactions by target expansion.

With inner tolerance `4e-2`, GPU-resident Galerkin assembly and GPU far-field
postprocessing reduce the repeated complete physical run to 23.70 s. It uses
`13+1` exact outer iterations and `110+7` inner pFFT
iterations; the two verified residuals are `7.70e-6` and `9.76e-6`.
Relative to the original 551.2 s strict FP64 implementation this is
`23.26x`. The solid-angle-weighted full-Mueller difference from an
independent two-polarization solve on the same mesh is `1.72e-6`.

The exterior and interior contracted derivative buffers now remain on the
GPU. One CUDA kernel forms their material-weighted combinations and tests
the surface basis; GPU kernels also apply the mass and exact sparse near
correction terms. The mean exact `A*x` time changed from 0.6282 s to
0.6092 s. A separate GPU reduction evaluates all far-field directions,
changing the 73-angle postprocessing from 1.421 s to 0.022 s while changing
the full Mueller matrix by only `2.28e-8` relative to the prior CPU
reduction.

For random-orientation work, use
`scripts/run_muller_orientation_average.sh`. The strict Muller operator,
pFFT, and MBJ factors are constructed once. A solved polarization pair is
reused for every alpha rotation in the GPU far-field batch, and a regular
N-sided prism is integrated only over its `2*pi/N` gamma sector. An atomic
checkpoint is written after every base orientation.

This comparison isolates implementation and iterative-solver error. It
does not establish that the BEM mesh is more accurate than a converged ADDA
grid; that requires separate mesh, quadrature, and DPL convergence audits.

`--mbj-cache PATH` separately stores the FP64 MBJ LU factors. The cache
fingerprint covers geometry, basis, material, quadrature, and MBJ block
parameters. For this `ref=5` case, loading 621 MiB took 0.21 s instead of
16.69 s of repeated block assembly.

## Solver checkpoints

Every outer GMRES or FGMRES solve now saves its current solution after each
outer iteration. By default the files are written next to the JSON result as
`OUT.checkpoint.SOLVER.bin`. Repeating the same command verifies the mesh,
operator, material, right-hand side, and vector size, then resumes from the
last complete iteration. Checkpoints are written through a temporary file
and atomically renamed, so interrupting a write does not damage the preceding
checkpoint.

Use `--checkpoint PATH` to select a common checkpoint prefix, or
`--no-checkpoint` to disable this behavior. Iteration CSV files opened with
`--iteration-log PATH` are appended on restart instead of being truncated.
When capturing standard output, use `tee -a` for the same append behavior.

`--allow-checkpoint-migration` explicitly permits an older operator
signature while retaining file-format, vector-size, and right-hand-side
hash checks. The selected operator recomputes the residual immediately after
loading. Without this flag, a precision or operator change remains rejected.
