# Muller FMM Optimization on RTX 3090 Ti

## Scope

These measurements use the H(div)-BDM1 Muller operator, quadrature order 7,
Duffy order 4, five FMM digits, mixed FP32 near work, and an RTX 3090 Ti.
The main benchmark is `model_repaired.obj`, refractive index 1.6. Timings are
wall-clock values for the same discrete operator unless stated otherwise.

## Accepted changes

1. The research driver defaults to `--max-leaf 32`. Its existing high-frequency
   guard still prevents an unsafe sixth octree level.
2. The mixed build defaults to 512 CUDA threads for the contracted Muller P2P
   kernel. `BEM_FMM_P2P_THREADS` permits an explicit hardware-specific override.
3. `BEM_FMM_PROFILE_BATCH3=1` profiles the GPU-resident vector action as copy,
   far field, contracted L2P, and P2P.
4. `--edge-refine` and `--feature-angle` now work for arbitrary OBJ meshes, not
   only generated prisms.
5. The two Muller currents now share a six-channel far traversal, paired L2P,
   and paired P2P. The default RTX 3090 Ti launch sizes are 256, 256, and 512
   threads respectively.
6. Mixed-precision paired P2P reads cached FP32 copies of target/source
   coordinates and projected charges instead of loading FP64 values and
   converting them for every near pair.
7. Real-wavenumber P2P has a compile-time-specialized path that removes the
   unused attenuation and imaginary-wavenumber arithmetic.
8. Reusable FP32 plane-wave phases avoid evaluating the same P2M/L2P
   trigonometric functions on every operator action. The cache is enabled by
   default in the mixed build while reserving 6144 MiB for the solve.
9. The L2P contraction applies the common `ik` and `-k^2` factors after
   accumulation instead of inside every direction term. Mixed L2P and M2L
   accumulation use FP32.
10. The mixed build stores six-channel multipoles and M2L transfers directly
    in FP32. This removes the full FP64 multipole workspace and the per-action
    conversion; the strict build or environment override allocates FP64
    storage instead.
11. FP32 local-expansion storage carries the already-FP32 M2L result through
    L2L and into L2P. This avoids widening and rereading values in FP64; the
    final assembled operator output remains FP64.
12. An opt-in depth-6 tree keeps the full depth-5 expansion order. This
    decreases leaf work without sacrificing the tested `1e-5` operator target.
13. The mixed build expands each target leaf's compact near-leaf list once
    during setup. The resulting source-index cache balances source work
    exactly across 32 P2P blocks instead of assigning whole leaves round-robin.
    At `ref=3` the cache occupies 120.0 MiB per medium.
14. Paired P2M accumulation and its twelve input charge arrays are FP32 in the
    mixed build. The same packed charge cache is reused by P2M and P2P; FP64
    Krylov vectors and final operator outputs are unchanged.
15. A direction-major copy of the FP32 phase table makes L2P reads coalesced.
    It is enabled only when the configured memory reserve still fits, and is
    reused by every action and orientation.
16. `BEM_FMM_PROFILE_FAR_DETAIL=1` separates clear, P2M, M2M, M2L, and L2L
    timing. This diagnostic synchronizes the GPU and is not a production mode.
17. Orientation averaging can use paired GPU GMRES. Both polarization Krylov
    bases and MBJ application remain on the device; only final solutions return
    to the host for checkpointing and far-field output.
18. Orientation recycling is now an updating bounded basis instead of a frozen
    first-sample basis. A recycled projection is used only when its estimated
    initial residual is at least 2% below the exact neighboring-RHS estimate.
19. If the direction-major L2P phase table does not fit, a warp-per-target
    kernel consumes the primary target-major phase table without allocating a
    second copy.
20. Paired far-field evaluation has a real-wavenumber specialization and uses
    `sincosf` for the phase in the mixed build. Inputs and final output remain
    FP64; only per-thread partial sums and their block reduction use FP32.
    On the `ka=25, ref=5` prism with 256 alpha samples, far-field time decreased
    from 3.63 to 0.79 s and complete wall time from 25.32 to 21.20 s. Across
    the prism, sphere, and asymmetric controls, the maximum Mueller change
    normalized by `M11(0)` stayed below `1.6e-7`.

On Shape A at `ka=40, ref=3`, the balanced source cache reduced paired P2P
from about 0.79--0.87 s to 0.11--0.20 s. FP32 P2M reduced the corresponding
P2M stages from 0.19--0.34 s to 0.05--0.09 s. The transposed phase table
reduced representative L2P stages from 0.06--0.12 s to 0.03--0.07 s.
Exterior and interior media run concurrently, so stage times must not be
summed to obtain wall time.

## Strict timing results

| Case | Previous action | Optimized action | Action speedup |
|---|---:|---:|---:|
| Shape A, `ka=20, ref=2`, radius 3 | 3.285 s | 1.109 s | 2.96x |
| Shape A, `ka=40, ref=3`, radius 3 | 18.6 s | 10.64 s | 1.75x |

The table above records an earlier optimization stage. The complete sequence,
using the same discrete operator and cached setup, is:

| Case | Original separate path | Final mixed path | Speedup |
|---|---:|---:|---:|
| Shape A, `ka=20, ref=2`, steady action | 1.086 s | 0.203--0.213 s | 5.1--5.3x |
| Shape A, `ka=20, ref=2`, three-step solve | 6.73 s | 1.328 s | 5.07x |
| Shape A, `ka=40, ref=3`, one-step solve, strict depth 6 | 21.60 s | 3.674 s | 5.88x |
| Shape A, `ka=40, ref=3`, steady action, strict depth 6 | 10.39 s | 1.713 s | 6.06x |
| Shape A, `ka=40, ref=3`, depth 5 action before balanced cache | 2.279 s | 0.400 s | 5.70x |
| Shape A, `ka=40, ref=3`, depth 5 one-step solve | 4.913 s | 0.963 s | 5.10x |

The `ref=3` action above used 13006 MiB, including both phase layouts. Without
the transposed L2P copy it used 8574 MiB. The extra copy is automatically
skipped when it would violate the memory reserve. The warp-per-target fallback
was 1.086x faster than the previous non-transposed kernel and about 13% slower
than the full-memory transposed path. Set
`BEM_FMM_PAIR_CURRENTS=0` before startup to recover the separate-current
memory footprint, or `BEM_FMM_PHASE_CACHE=0` if the primary cache prevents a
larger mesh from fitting.

For one Shape A `ka=40, ref=3` base orientation and ten forced iterations,
paired GPU GMRES reduced the two-polarization solve from 8.780 s to 8.226 s
(`1.067x`). The final residual was unchanged at `1.926e-1`, and the maximum
Mueller-array difference normalized by its largest element was `3.71e-9`.
On a prism `ka=15, ref=2` grid with nine base orientations, updating rank-8
recycling reduced 312 to 298 total iterations and 1.653 s to 1.570 s
(`1.053x`). The latter comparison used tolerance `1e-3`, so it establishes
solver acceleration rather than publication-grade physical convergence.

The hard dense control was a six-sided prism at `ka=20`, `m=3`,
`near-radius=3`: the final mixed operator differed from dense assembly by
`2.306e-6`; paired and separate current paths differed by `1.897e-6`. At
`ka=10`, `m=1.3`, the extra error from FP32 M2L was about
`1e-8`; the total operator error remained below `2.4e-7`.

A 100-step Shape A `ref=2` run reached a true residual of `2.808e-4` in
15.396 s. A ten-step `ref=3` run took 5.190 s and reached `1.437e-1`.

For Shape A at `ka=20, ref=2`, strict pFFT-FGMRES reached a true FMM residual
of `9.38e-6` in 33 outer and 726 inner iterations. Solve time decreased from
196.11 s to 83.41 s, a 2.35x speedup.

At `ka=20, ref=1`, both polarizations took 20.79 s instead of 89.65 s. All 16
Mueller elements on 181 angles had aggregate relative L2 difference
`7.66e-8`; the maximum absolute difference normalized by `M11(0)` was
`6.08e-8`.

## Two-polarization experiments added in July 2026

Three additional paths are available for controlled experiments. None of
them silently changes the default production operator.

### Four-field FMM traversal

`BEM_FMM_FOUR_FIELD=1` allocates a 12-channel FMM workspace and evaluates
the electric and magnetic currents for both incident polarizations in one
upward/M2L/downward traversal. The near field is evaluated by two paired
kernels, and the four contracted results remain separate through Galerkin
assembly.

The implementation is numerically correct but is not a general speedup on
the RTX 3090 Ti. The M2L stage now uses two six-channel launches instead of
one 12-accumulator kernel; this reduced the one-step probe from 0.352 s to
0.334 s. On a six-sided prism with `ka=20`, `m=1.3`, `ref=5`, and residual
tolerance `1e-5`, the sequential paired-current path took 9.844 s for the
solve and the revised four-field path took 10.439 s. Both used 61 iterations.
The remaining P2M/L2L and launch overhead still erase the shared traversal.
A small `ka=6, ref=2` sphere improved from 0.101 s to 0.089 s, so the path
remains opt-in for future architectures and small trees. When it is disabled,
the extra charge and operator buffers are not allocated.

### Mixed-precision iterative refinement

`BEM_MIXED_ITERATIVE_REFINEMENT=1` keeps the Krylov basis and correction
solves on the fast mixed operator but recomputes the residual with a fully
FP64 FMM action after every restart cycle. If that residual exceeds the
requested tolerance, the next cycle solves for the residual correction.

Strict residual actions now evaluate the electric and magnetic currents in a
paired FP64 traversal. A dedicated six-channel FP64 multipole/local workspace
is allocated only when refinement is requested; allocation failure falls back
to the previous sequential strict action. For the same `ka=20` prism, the
strict residual reached `8.19e-6` in 60 iterations. Solve time decreased from
18.842 s for sequential strict actions to 17.204 s (`1.095x`), while the
largest Mueller change relative to the previous strict result was `2.01e-7`
of the result peak. It is still a validation mode rather than the default fast
path: the ordinary mixed solve took 9.844 s.

### Angular spectral far field

The useful regular structure of orientation averaging is exploited directly.
Set
`BEM_FARFIELD_SPECTRAL_ALPHA=N` to evaluate the complex far-field amplitudes
on `N` uniformly spaced alpha angles and reconstruct the requested alpha grid
by periodic Fourier interpolation. Interpolation is applied to the complex
vector field before rotations and Mueller conversion, never to Mueller data.
`BEM_FARFIELD_SPECTRAL_ALPHA=auto` rounds `2*(ka+12)` up to a multiple of 16
and never exceeds the requested alpha count.

For 256 requested alpha samples at `ka=20`:

| Shape | Direct far field | Spectral `N=64` | Far-field speedup | max error / `M11(0)` |
|---|---:|---:|---:|---:|
| Six-sided prism | 0.750 s | 0.197 s | 3.82x | `2.29e-7` |
| Sphere | 0.482 s | 0.119 s | 4.04x | `5.67e-8` |
| Asymmetric polyhedron | 0.574 s | 0.145 s | 3.95x | `5.13e-7` |

Replaying exactly the same converged currents at `ka=25` and `30` showed
`2.49--3.12x` `auto` speedup across the three shapes, with worst normalized
Mueller error `6.9e-8`. Large-`ka` one-step controls selected `N=112` at
`ka=40` (error `1.32e-8`) and `N=144` at `ka=60` (error `5.99e-6`). On the
prism, `N=128` at `ka=60` gave `1.11e-4` and is rejected for a `1e-5` target.
Run `scripts/benchmark_farfield_spectral_replay.sh` to reproduce the six
converged-current comparisons.

An isolated cuFINUFFT 2.5.1 type-3 benchmark used 236544 sources, 18688 target
directions, 12 complex FP64 transforms, and `ka=30`. At requested epsilon
`1e-7`, its reusable execution took 0.370 s and its direct-subset relative
error was `1.61e-6`; plan and point setup took 0.043 s. This is slower than the
validated regular-alpha reconstruction (about 0.20 s on the production-sized
prism), so cuFINUFFT is not a project dependency. The optional benchmark is
`scripts/benchmark_cufinufft_farfield.py`.

## Current 24 GiB resource limit check

The current mixed build completed a setup-only six-sided-prism case with
`ka=60`, `ref=6`, 806400 system unknowns, and 940800 quadrature points. pFFT
order 3 with radius-one correction and MBJ-50 took 136.67 s wall time. Peak
process RSS was 10217 MiB; peak GPU memory reported by `nvidia-smi` was
14395 MiB, while the internal allocation delta was 13464 MiB. The sampled
peak GPU utilization and power were 68% and 128.8 W because setup is dominated
by CPU near-correction and MBJ assembly. The output and resource trace are in
`runs/resource_limit_current/ref6_ka60/`.

## Rejected or limited variants

- Converting far-field positions and all 24 split current arrays to FP32 was
  slower. The conversion and extra buffers increased far-field time from 2.32
  to 2.88 s and added about 48 MiB of GPU allocation.
- A batched phase-matrix plus FP64 `cuBLAS ZGEMM` far-field path was also
  slower. One `ka=25` average contains 73 scattering angles times 256 alpha
  samples, or 18688 directions. Materializing the full phase matrix would need
  about 123 GiB; batching it in groups of 128 raised peak observed GPU memory
  from 7648 to 7914 MiB and took 2.43 s instead of 2.34 s.
- A symmetric cache of FP32 `sin(kr), cos(kr)` values was slower despite
  halving storage with `r_ij = r_ji`. At `ref=2`, the near field contained
  `0.384` billion directed point pairs and `0.216` billion symmetric entries.
  The cache used `1648.9 MiB` per medium (`3.22 GiB` total) and increased P2P
  time from about `118--140 ms` to `155--174 ms`. Irregular cache traffic was
  more expensive than recomputing `__sincosf`.
- Packing the twelve FP32 source-current components into three `float4`
  records increased P2P time to about `136--144 ms`. The scalar structure of
  arrays remains preferable because same-source warp loads are already served
  efficiently by the GPU caches and the packed path raised register pressure.
- Disabling concurrent exterior/interior evaluation was slower. At
  `ka=20, ref=2`, the useful tree took about 1.55 s concurrently and 2.64 s
  sequentially before the 512-thread change.
- FP64 near work took 7.29 s per action versus 1.40 s for FP32 in the radius-2
  microbenchmark. FP64 remains a validation mode.
- The current rank-4 and rank-8 polynomial coarse corrections did not improve
  convergence. After 100 iterations, residuals were `8.69e-4` and `8.71e-4`,
  compared with `8.64e-4` without coarse correction. Setup cost was 5.74 s and
  11.30 s.
- Radius 2 gave operator errors near `1e-6` for low-contrast sphere and prism
  checks, but `7.82e-3` for a `ka=20, m=3` prism. Radius 3 reduced the latter to
  `1.58e-6`, so radius 2 is not a universal setting.
- OBJ `ref=2` plus one feature-edge pass used 191,112 unknowns instead of
  307,968 for uniform `ref=3`, but had only 2.62 minimum P2 nodes per wavelength
  instead of 4.17 at `ka=40`. Local edge refinement cannot replace global wave
  resolution.
- Paired-P2P blocks of 640 and 768 threads did not improve the RTX 3090 Ti
  timing; 800 threads exceeded the kernel resource limit. Far-stage blocks of
  64 and 128 threads were also slower than 256. A 512-thread far block won a
  one-step microbenchmark by 2.6% but lost 7% over ten iterations. These
  variants are not defaults.
- A 384-thread paired-P2P block was slower than 512. Forcing two 512-thread
  blocks per SM reduced registers from 72 to 64 but introduced 36 bytes of
  spill traffic per thread and did not improve wall time.
- A single kernel for both media increased the paired-P2P time from about
  106 ms to 537 ms. It required 127 registers per thread and a 192-byte stack
  frame, so the implementation was removed.
- Native depth-6 orders reduced memory and action time, but changed the tested
  residual by about 2%. Scaling the reference box to 0.875 still changed it by
  about 0.24%. Only the full depth-5 order floor is retained.
- Shared-memory copies of P2M sources and L2P coefficients did not improve
  timing and remain disabled diagnostic variants.
- Raising expansion accuracy can partly replace a large near radius, but is
  not universally cheaper. For a `ka=20, m=3` prism at radius 2, operator
  errors for digits 5, 6, 7, 8, and 9 were respectively `7.82e-3`,
  `1.53e-3`, `1.74e-4`, `1.62e-5`, and `1.43e-6`. The required order and
  far-field memory are too high for a general default.

## Low-contrast radius/order tradeoff

For the tested `ka=20, m=1.3` prism, radius 2 with digits 6 had dense operator
error `1.225e-6`. On the Shape A `ref=2` benchmark it changed near leaf pairs
from 240130 to 118126 and reduced a three-step solve from 1.531 s to 1.393 s.
GPU memory increased from 2388 to 2992 MiB because of the higher expansion
order. Reproduce it only with:

```bash
BEM_MULLER_FMM_DIGITS_CAP=6 \
bin/muller_nodal_fmm_demo_fp32 ... \
  --fmm-near-radius 2 --digits 6
```

This is an explicitly validated low-contrast option. Radius 3 and the default
digits cap remain the strict general settings.

## Reproduce the strict operator benchmark

```bash
OMP_NUM_THREADS=16 BEM_FMM_CONCURRENT_MEDIA=1 \
BEM_FMM_PAIR_CURRENTS=1 \
bin/muller_nodal_fmm_demo_fp32 \
  --obj model_repaired.obj --ref 2 --ka 20 --ri 1.6 \
  --edge-mode hdiv --quad 7 --duffy-order 4 \
  --digits 5 --max-leaf 32 --fmm-near-radius 3 \
  --fmm-near-fp32 --tol 1e-30 --max-iters 3 \
  --gmres-restart 1 --mbj-only --mbj-nodes 50 \
  --no-dense-validation --no-checkpoint \
  --iteration-log runs/fmm_benchmark/iterations.csv \
  --out runs/fmm_benchmark/result.json
```

The deliberately impossible tolerance and one-vector GMRES restart turn this
into a repeated MatVec benchmark. Use a normal tolerance, checkpointing, and
pFFT-FGMRES for production calculations.

The mixed build enables the accepted fast paths automatically. Set any of
the following to zero for an FP64 or no-cache control:

```bash
BEM_FMM_PHASE_CACHE=0
BEM_FMM_M2L_STORAGE_FP32=0
BEM_FMM_MULTI_STORAGE_FP32=0
BEM_FMM_LOCAL_STORAGE_FP32=0
BEM_FMM_M2L_FP32=0
BEM_FMM_L2P_FP32=0
BEM_FMM_P2P_FAST_TRIG=0
BEM_FMM_FLAT_NEAR_SOURCES=0
BEM_FMM_L2P_TRANSPOSED_PHASE_CACHE=0
```

The balanced near-source cache keeps 4096 MiB free by default; change this
with `BEM_FMM_FLAT_NEAR_RESERVE_MB`. The transposed L2P table follows the
6144 MiB phase-cache reserve. `BEM_FMM_P2P_LEAF_SPLIT=1|2|4|8|16|32|64|128`
overrides the partition count; 32 was fastest with the balanced cache on the
RTX 3090 Ti.

For the validated `ref=3` depth-6 benchmark, add:

```bash
BEM_FMM_ALLOW_DEPTH6=1 \
bin/muller_nodal_fmm_demo_fp32 \
  --obj model_repaired.obj --ref 3 --ka 40 --ri 1.6 \
  --edge-mode hdiv --quad 7 --duffy-order 4 \
  --digits 5 --max-leaf 16 --fmm-near-radius 3 \
  --fmm-near-fp32 --tol 1e-30 --max-iters 1 \
  --gmres-restart 1 --mbj-only --mbj-nodes 50 \
  --no-dense-validation --no-checkpoint \
  --iteration-log runs/fmm_ref3_depth6/iterations.csv \
  --out runs/fmm_ref3_depth6/result.json
```

The driver sets the depth-5 expansion-order floor automatically. Do not use a
native reduced depth-6 order for a physical calculation.
