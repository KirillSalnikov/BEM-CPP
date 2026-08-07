# Harmonic-Ritz deflation during orientation averaging

The controlled case is a regular hexagonal prism with `ka=5`, `m=2`,
`ref=2`, FP64 near field, FMM digits 5, quadrature 13, Duffy order 6, and a
true-residual target of `2e-6`. The fixed angular grid contains 8 alpha, 4
beta, and 4 gamma nodes. Sixfold prism symmetry means that 16 independently
solved beta/gamma orientations represent 768 full orientation samples. Each
base orientation uses two independent incident polarizations and produces 181
Mueller scattering angles.

Warm starts, Krylov recycling, and paired GPU GMRES were disabled in both
runs. Both runs loaded the same prebuilt near-correction and MBJ caches.

| mode | iterations | solve time | time with setup | maximum residual |
|---|---:|---:|---:|---:|
| baseline | 3,161 | 179.106 s | 179.518 s | `1.996e-6` |
| rank-32 harmonic deflation | 2,258 | 128.994 s | 129.404 s | `1.997e-6` |

The measured speedup is `1.400x` by iterations, `1.388x` by solve time, and
`1.387x` by complete time including setup. Every independently solved
orientation used fewer iterations. The first orientation, which also builds
the basis, improved by `1.185x`; later orientations improved by up to
`1.453x`.

The rank-32 basis required 0.022 s to construct, 1.568 s for all projection
applications, and 3.09 MiB of storage. The complete Mueller matrices agree to
`3.63e-7` in relative Frobenius norm. The largest absolute element difference
is `3.76e-7` after normalization by baseline `M11(0)`.

This is a same-discretization solver comparison. It does not establish the
absolute surface-discretization accuracy of `ref=2`.

Artifacts:

- `orientation_deflation_comparison.png`: per-orientation iterations and time.
- `orientation_deflation_comparison.csv`: numerical values for every base orientation.
- `orientation_deflation_summary.json`: aggregate metrics and all-element Mueller comparison.
- `prism_ka5_m2_r2_b4g4_baseline`: raw baseline result, log, and iteration history.
- `prism_ka5_m2_r2_b4g4_rank32`: raw deflated result, basis, log, and iteration history.
