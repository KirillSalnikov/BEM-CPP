# Resident adaptive-fast benchmark

This benchmark measures the replacement of the old multi-process residual
ladder with one resident solver process and paired OpenMP far-field evaluation.

## Method

- Hardware: NVIDIA GeForce RTX 3090 Ti 24 GiB.
- Geometry: regular hexagonal prism, `h/D=1`.
- Material: relative refractive index `m=1.3`.
- Output: 181 scattering angles.
- Numerics: automatic refinement, quadrature 7, Duffy order 4, five-digit FMM,
  and the same mixed FP32/FP64 build for both profiles.
- Timing: isolated initially empty operator caches; complete launcher wall time.
- `standard`: independently verified operator residual `1e-5`.
- `fast`: residual ladder with two successive independent `1e-3` stability
  checks of the complete Mueller matrix, normalized `M11`, forward `M11`,
  integrated `M11`, and optical-theorem extinction.

Speedup is `T_standard / T_fast`. This is a profile-level, physical-stability
comparison, not an equal-residual comparison.

## Results

| ka | ref | unknowns | standard, s | old fast, s | resident fast, s | standard / resident fast | selected target | maximum residual | normalized full-Mueller difference |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 5 | 202752 | 47.88 | 93.02 | 48.88 | 0.980x | 3e-4 | 1.84e-4 | 7.21e-5 |
| 60 | 6 | 806400 | 640.13 | 692.17 | 582.24 | 1.099x | 1e-4 | 8.08e-5 | 2.51e-6 |

At `ka=25`, the resident path is effectively tied with `standard` and is
`1.90x` faster than the old implementation. At `ka=60`, it is `1.099x`
faster than `standard` and `1.189x` faster than the old implementation.

The `ka=60` gate rejected the `1e-3` level because forward and integrated
`M11` still changed by about `2.1e-3`. It stopped at `1e-4` only after the
second consecutive stable comparison. The final forward-`M11` differences
from strict `standard` are `5.54e-5` at `ka=25` and `1.13e-5` at `ka=60`.

The paired OpenMP far-field component reduced the measured `ka=60` stage from
8.011 s to 0.579 s (`13.84x`) on an identical checkpoint. Its complete
normalized Mueller difference from the prior serial implementation was
`2.59e-14`.

These two prism cases validate the implementation and demonstrate one
large-case benefit. They do not establish universal speedup across shapes,
materials, orientations, or surface refinements.

## Artifacts

- `adaptive_fast_resident.csv`: machine-readable table.
- `adaptive_fast_resident.json`: complete scalar measurements and provenance.
- `adaptive_fast_vs_standard_wall_time.png`: wall-time comparison.

The full ignored run outputs remain under
`runs/adaptive_fast_benchmark_20260805/` on the measurement machine.
