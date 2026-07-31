# Changelog

All notable changes to BEM-CPP are documented in this file. The project uses
Semantic Versioning while its public interfaces are stabilizing.

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
