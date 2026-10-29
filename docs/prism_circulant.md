# Experimental prism circulant inverse

`PrismCirculantInverse` is a fixed right preconditioner for host GMRES. The BEM
operator, singular quadrature and final FP64 residual remain unchanged. Enable
it explicitly with `BEM_TMATRIX_CIRCULANT_BLEND` in `(0,1]`, or the large-run
controller's `--circulant-blend`. The default remains MBJ.

Each regular D6h side face is split into four uniformly triangulated rectangular
patches. The two caps use masked triangular lattices. Three edge directions,
two BDM1 edge moments and two currents give twelve channels per cell. A side
kernel and a cap kernel are shared across the 26 patches. Each kernel samples
12 impulses of the actual BEM operator, wraps the centered displacement stencil
and inverts its 12 by 12 Fourier symbols. This is a sampled Strang approximation,
not the Frobenius-optimal Chan circulant and not an exact periodic reformulation.
The inverse uses batched FP64 cuFFT; its additional storage grows linearly with
surface size. It requires neither a volumetric grid nor dense surface blocks.

Edges crossing a patch boundary retain MBJ. Covered edges use the requested
blend between MBJ and the circulant. This local partition omits inter-face
coupling and introduces artificial periodic connections; convergence must be
measured, particularly at increasing ka. It is not guaranteed to improve MBJ.
The cap's rectangular periodic extension need not itself commute with C6.
Symmetry audits and strict residuals apply to the original BEM equation.

Reflections transform tested equations and unknown currents differently:
`A S_solution = S_rhs A`. For determinant d and BDM edge orientation factor E,
rhs signs are `(E, d E)` and solution signs `(d E, E)` for the two currents.
E is the edge direction sign for moment zero and one for moment one.
`PrismCirculantLayout` checks the corresponding physical basis mapping over all
covered supports. Singular Fourier symbols fall back explicitly to the
invertible zero-offset block; their count is reported. A singular zero-offset
block is an error. Optional `BEM_CIRCULANT_SHIFT` adds a multiple of that local
block to each Fourier symbol; it never changes the physical operator.

Validation:

- `make cuda-circulant-check`: direct periodic convolution followed by the FFT
  inverse, with nonsymmetric complex blocks, odd rectangular dimensions and
  multiple batches; geometry and reflection checks at aspects 1 and 10/7.
- `make tests/prism_circulant_blocks_check` and run that executable: independent
  direct BEM blocks under patch rotations/reflections.
- `python3 benchmarks/circulant_20261029/run_probe.py --ref 5 --ka 40
  --rhs 2 --methods baseline,circ,circ0.25 --name my_control`: physical solves
  with identical RHS, mesh and strict tolerance. Reports include setup time,
  complete solve time and differences from baseline solutions.

The new target prism has H/D_flats=10/7. The T builder accepts
`--prism-aspect 1.4285714285714286 --wavelength 0.532`; both are recorded in
checkpoint identities and physical scale metadata. Wavelength does not change
the dimensionless equation at fixed ka and refractive index. At ka=40,
a_volume=3.38681719 um, D_flats=5.08561830 um and H=7.26516900 um. Existing
Mueller normalization is preserved; adding wavelength metadata does not silently
rescale the exported Mueller values. The legacy exact face-FFT backend is
rejected for nonunit prism aspect until its own geometry mapping is generalized.

Measured initial controls (2026-10-29, RTX 3090 Ti, n=1.3116, H/D=10/7):

| ka / refinement | MBJ seconds per RHS | circulant, weight 1 | circulant, weight 0.25 |
| --- | ---: | ---: | ---: |
| 20 / 4, one RHS | 7.974 | 7.714 | 7.799 |
| 40 / 5, mean of two RHS | 76.017 | 71.903 | 66.070 |
| 40 / 6, one RHS | 249.484 | not tested | 220.357 |

All six ka=40 solves passed the same 2e-6 strict residual gate. The quarter-weight
inverse used 160/180 iterations versus 192/212 for MBJ. Its one-time setup took
7.63 s and its GPU banks/workspace occupied 19.04 MiB. Physical block covariance
checks at both aspects gave separated-block errors below 1.7e-14; touching blocks
with the direct assembler's order-four Duffy quadrature agreed within 2.1e-6.
These are solver and symmetry checks, not a one-percent discretization certificate.
Raw results live in `runs/circulant_20261029`; compact reports in
`benchmarks/circulant_20261029`.

On the target ref=6 mesh (1,032,192 DOFs), the quarter-weight circulant saved
11.68% of the solve time, using 190 versus 221 iterations. Setup took 21.73 s
and FFT storage was 76.13 MiB. The relative far-field L2 difference from MBJ was
9.421e-07. This is a single-RHS timing control, not a guarantee of the same
speedup for all packed T columns.

Additional preconditioner trials, 2026-10-29 (same ka=40 prism):

| ref | inverse | iterations | solve seconds | strict relative residual |
| --- | --- | ---: | ---: | ---: |
| 5 | MBJ50 + circulant 0.25, fresh control | 160 | 62.55 | 1.58e-6 |
| 5 | same local inverse + global wave rank 32 | 181 | 69.87 | 1.00e-6 |
| 5 | same local inverse + global wave rank 64 | 184 | 70.86 | 9.71e-7 |
| 5 | same local inverse + global wave rank 128 | 192 | 73.71 | 9.64e-7 |
| 5 | MBJ overlap 8 + circulant 0.25 | 160 | 64.10 | 1.42e-6 |
| 5 | flexible GMRES, inner pFFT/MBJ 8 steps | 32 | 65.68 | 1.75e-7 |

The global inverse is `B + (V - B A V) (V* A V)^-1 V*`, with a fixed
orthonormal basis of plane-wave BDM1 edge moments over the entire surface.
It includes cross-face coupling through full-operator samples during setup.
Rank 128 took 42.68 s to build and about 1012 MiB GPU storage at ref=5.
It passed exactness/linearity tests but did **not** accelerate this control.
The production default therefore remains rank zero, overlap zero. Test with
`make cuda-global-coarse-check`; enable experimental ranks explicitly with
`run_large_tmatrix.py --wave-rank 128 --circulant-blend .25`.

The optional nested pFFT inverse uses flexible outer GMRES, stores actual
preconditioned Arnoldi vectors, and restores the FMM backend after every
inner application. The final residual always uses the full FP64 FMM action.
`--fft-inner 8 --fft-radius .5` selects this alternative with MBJ; it cannot
be combined with wave/circulant correction or Krylov recycling. At ref=6,
radius 0.5 h fit in 21.3 GiB GPU memory, but 31 outer iterations already took
218.85 s before the strict check, which returned 9.13e-6. This trial was
stopped because it had exceeded the completed circulant control's time;
it was not run to final convergence.

`--fft-memory host` (worker setting `BEM_PFFT_CORR_MEMORY=host`) moves only
pFFT correction-value tables into managed memory with CPU preferred location
and GPU access advice. FFT grids, FMM, correction indices and workspaces stay
on the GPU. The default is `device`. RAM-backed ref=4 and GPU-backed actions
agreed to relative L2 8.67e-8 in FP32; times were 0.155 and 0.0292 s.
The full radius-1 ref=6 tables contain 252,988,272 pairs per medium and occupy
37.70 GiB together. RAM mode ran successfully with peak RSS 46.36 GiB and
GPU allocation 13,217 MiB, but a single warm pFFT action took **37.46 s**.
This removes the capacity barrier at the cost of PCIe traffic; it is not
selected for this production calculation.

The ka=40 T build resumed in its original frozen runtime after 1878 accepted
columns and 90 committed packets were checksum-verified. No experimental
inverse was enabled in that run. Analytic SO(3) averaging and both rotational
and mirror symmetry remain enabled. Raw trials and the restart receipt are
in `benchmarks/wave_coarse_20261029` and `runs/wave_coarse_20261029`.

For a future solver change that actually wins validation, the explicit
`--solver-restart-from` / `--solver-restart-proof` path can import packed and
C2-derived checkpoints into a **new** run directory. It checks geometry,
operator/projection identities, strict residual metadata, packet hashes and
C2 reconstruction; records source provenance; and reruns coded symmetry
audits with the new worker. The proof binds old/new binaries, unchanged
physical sources and a physical field comparison. This is separate from
ordinary resume, whose frozen-runtime identity remains strict. Tests are
`tests/test_tmatrix_checkpoint_migration.py` and
`tests/test_large_tmatrix_runner.py`. An iteration residual is not a 1%
all-angle mesh/cutoff convergence certificate.

Packet preparation can be overlapped with GPU work using
`run_large_tmatrix.py --incident-prefetch`. After the socket has consumed the
current incident fields, one producer thread reuses that same buffer for the
next packet. The geometry/radial cache has exactly one owner. A symmetry
rejection, changed task or already-completed column invalidates speculation by
its exact task key; the worker never receives mismatched normalization data.
`incident_field_s` measures total preparation, `incident_wait_s` measures its
exposed delay, and ETA uses only the latter. There is no second full field slab.

On all 2,236,416 actual ref=6 quadrature points, representative packets at
L=30,45,64 (including both incident polarizations and reflection codes) were
bitwise identical to serial preparation, including scales. Their preparation
times were approximately 10,19,31 seconds, with a radial refresh in each case.
CPU protocol tests cover a complete T build, ordinary resume, rejected packets,
stale speculative failures, both cached and uncached evaluation, and genuinely
concurrent execution. See `tests/test_tmatrix_pipeline.py` and
`benchmarks/packet_pipeline_20261029/incident_equivalence.json`.

`--strict-interval N` exposes the existing frequency of intermediate FP64
checks in the host GMRES solver (1..16 restart cycles). Every accepted result,
warm-start acceptance and iteration-limit exit still uses the full strict
operator. It does not change the requested residual tolerance. Synthetic
rounded-action tests now explicitly include N=8. `--worker-binary PATH` can
freeze a previously validated executable while upgrading only the Python
scheduler, keeping the physical worker binary byte-for-byte unchanged.

The ref=6 strict-interval control rejected N=8: it needed 222 iterations and
248.93 s versus 190 iterations and 220.36 s at N=4. Its first strict residual
was 9.59e-6 despite an inner estimate below 1e-6, requiring another correction.
The production run therefore keeps N=4 and enables **only incident prefetch**.

For this scheduling-only restart, the worker executable, every solver setting,
incident/projection/symmetry modules and numerical gates are unchanged. The
migration proof explicitly identifies bitwise equality of the incident inputs;
it does not claim a new scattering measurement. Existing coded audits may be
retained with source provenance under this stricter identity check, and are
marked `remeasured: false`. Changing even the strict-check interval forbids
this shortcut. Solver changes still require new coded audits. This avoids
repeating approximately fourteen minutes of identical control solves merely
to change CPU scheduling. All 2048 saved columns and 98 packets were verified
and carried into `runs/ka40_n1p3116_l0p532_aspect10over7_pipeline_20261029`.

The production pipeline subsequently measured 11.30--12.89 s of incident
preparation hidden behind the GPU solve, with exposed waits below 5 microseconds.
After 2152 saved columns and 103 committed packets, the run was stopped at a
packet boundary for the QMR/restart/hotspot controls documented in
`docs/muller_solver_sweep.md`. None demonstrated a sufficient solve-time gain,
so the same frozen pipeline runtime resumed without changing its worker,
solver settings, physical parameters, symmetry packing or analytic averaging.
Checkpoint and restart receipts are in `benchmarks/qmr_hotspot_20261029`.
