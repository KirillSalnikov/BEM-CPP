# Changelog

All notable changes to BEM-CPP are documented in this file. The project uses
Semantic Versioning while its public interfaces are stabilizing.

## Unreleased

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
