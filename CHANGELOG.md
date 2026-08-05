# Changelog

All notable changes to BEM-CPP are documented in this file. The project uses
Semantic Versioning while its public interfaces are stabilizing.

## Unreleased

### Added

- added a reproducible ten-case BEM/ADDA benchmark with three wall-time
  repetitions, independently recalculated residuals, adjacent-grid controls,
  common angular output, and sphere checks against Mie theory;
- added mixed-operator iterative refinement: FP32 L2P Krylov corrections are
  accepted only after an FP64 FMM restart-residual check;
- added content-addressed near-correction and MBJ caches shared across output
  directories, including OBJ-content hashing and atomic concurrent writes;
- documented all ten before/after BEM timings without selection: median cold
  speedup is `1.620x`, maximum is `2.321x`, while ADDA remains faster in every
  declared case;
- replaced the three-case `physical-fast` gate with a universal fixed-
  orientation `fast` profile: automatic mesh selection, optional pFFT warm
  start, an exact residual ladder, two independent Mueller-stability gates,
  atomic final-result publication, and an automatic `1e-5` standard fallback;
- keep FMM/pFFT operators, MBJ factors, and Krylov solutions resident across
  the adaptive residual ladder, and evaluate both fixed-orientation far fields
  in one OpenMP-parallel element traversal.

### Fixed

- record prism side count and aspect ratio in both setup-only and final Muller
  result files so that non-unit-aspect calculations are self-describing;
- verify reconstructed prism polarizations with the final FMM operator and
  remove stale speedups based on stored, non-comparable ADDA timings;
- count quadratic P2 edge-midpoint nodes when selecting automatic surface
  refinement and compare differing angular grids on the coarser native grid.

## [0.1.0-alpha.5] - 2026-08-03

### Fixed

- removed cross-program acceleration claims that compared BEM and ADDA runs
  with different residual criteria;
- documented that equal-accuracy ADDA acceleration remains unproven until
  both final residuals are independently recalculated and both discretizations
  pass convergence checks;
- added a documentation regression check that rejects the withdrawn headline
  ratios from the primary README and manual.

## [0.1.0-alpha.4] - 2026-08-03

### Fixed

- the release metadata audit now works both in a Git checkout and in the
  published source archive, which intentionally has no `.git` directory.

### Validation

- the complete host audit passes directly from the packaged source archive.

## [0.1.0-alpha.3] - 2026-08-03

### Added

- CUDA-independent compatibility headers for the host-only audit on clean
  GitHub Actions runners;
- an automated freshness check that keeps the release number, primary entry
  point, documented options, and reference-data link synchronized.

### Changed

- the PDF manual now consistently identifies BEM-CPP, uses the current
  `./bem` launcher and quality profiles, and distinguishes historical V100
  measurements from the RTX 3090 Ti release environment;
- current orientation-averaging documentation now describes adaptive nested
  beta/gamma refinement, exact prism symmetry reuse, per-orientation restart,
  and one operator setup per schedule.

### Validation

- the complete host audit passes with no CUDA toolkit installed;
- the full CUDA audit and physical Mie gate pass on the release machine.

## [0.1.0-alpha.2] - 2026-08-03

### Added

- adaptive two-stage planning for large fixed-orientation sphere, cube, and
  prism calculations, with the hierarchy derived from mesh refinement and
  electrical density;
- atomic `execution_state.json` pipeline status and checkpoint disk-space
  estimates;
- GPU far-field projection for the experimental host-assembled banded FMM.

### Changed

- resource admission now checks currently free VRAM, profile-specific safety
  margins, and temporary disk space required by atomic checkpoints;
- orientation plans explicitly record that one prepared operator is reused
  and that restart granularity is one completed base orientation;
- `bem resume` preserves sequential-current memory policies from the original
  two-stage plan.

### Validation

- banded-pFFT orientation averaging passed residual and Mueller checks at
  `ka=20` and `ka=60`, but was slower than paired GPU-GMRES and therefore was
  not enabled by default;
- the complete host and CUDA release audit passes on the release machine;
- the mixed-precision `ka=1`, `m=1.3` sphere reaches a maximum true residual
  of `7.705e-6`, a solid-angle-weighted `M11` relative L2 error of `3.2276%`
  against Mie theory, and a `4.3028%` main normalized Mueller error.

## [0.1.0-alpha.1] - 2026-07-31

### Added

- second-kind Muller solver with smooth P2 and H(div)-BDM1 surface bases;
- GPU FMM, pFFT-FGMRES, and Morton block-Jacobi preconditioning;
- mixed-precision RTX 3090 Ti build with FP64 Krylov convergence checks;
- two-polarization GPU solve and orientation averaging;
- atomic solver and orientation checkpoints plus reusable operator caches;
- angular spectral reconstruction for large alpha grids;
- `bem` launcher with automatic mesh selection, quality profiles, memory
  checks, restart, and result validation;
- adaptive FP64 pFFT acceleration for normal-size strict runs and a safer
  minimum quick mesh selected from cross-profile Mueller comparisons;
- profile-aware adaptive orientation averaging with nested angular levels,
  per-orientation persistence, convergence validation, and explicit fixed-grid
  compatibility;
- command-line overrides for profile solver/numerical controls and documented
  size-dependent automatic surface refinement;
- command-line help, version reporting, and automatic output directories;
- host and CUDA regression tests and release metadata.

### Known limitations

- the optimized mixed-precision target is tuned and validated primarily on
  NVIDIA Ampere `sm_86` hardware;
- publication-grade sharp-edge calculations still require mesh convergence
  and comparison with an independent edge-capable solver;
- optional four-field FMM and strict mixed iterative refinement remain
  experimental and disabled by default;
- historical large benchmarks require separately retained `runs/` artifacts.

[0.1.0-alpha.1]: https://github.com/KirillSalnikov/BEM-CPP/releases/tag/v0.1.0-alpha.1
[0.1.0-alpha.2]: https://github.com/KirillSalnikov/BEM-CPP/releases/tag/v0.1.0-alpha.2
[0.1.0-alpha.3]: https://github.com/KirillSalnikov/BEM-CPP/releases/tag/v0.1.0-alpha.3
[0.1.0-alpha.4]: https://github.com/KirillSalnikov/BEM-CPP/releases/tag/v0.1.0-alpha.4
[0.1.0-alpha.5]: https://github.com/KirillSalnikov/BEM-CPP/releases/tag/v0.1.0-alpha.5
