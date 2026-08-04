# Ten predeclared BEM versus ADDA timing ratios

The ratio is `median ADDA complete wall time / median BEM complete wall time` from three independent runs with fresh application-cache directories. CUDA/OpenCL driver and compiler caches were not flushed, so this is a warm-system benchmark. A value below one is a BEM slowdown, not an acceleration. All ten cases were declared before execution; failed or unfavorable rows are retained.

Both programs use a `1e-5` linear residual target, two independently solved polarizations, and the same 181 scattering angles. ADDA reports a final recalculated residual. BEM reports a final exact-FMM-operator residual. The production discretizations are BEM >=15 points per shortest wavelength and ADDA dpl=20; BEM one-level-coarser and ADDA dpl=15 controls must change the normalized complete Mueller matrix by no more than 2%. The two production matrices must agree within 5%, including forward M11.
For spheres, both production matrices must also agree with exact Mie theory within 2% in normalized shape and forward M11.
The ADDA baseline is the clean official adda-team/adda commit 8f550a7, not the locally modified FP32 experimental build.
BEM binary SHA-256: `021b696494719e28022c176c60872e6b907a8fc7be0dd2364a72c9c454467544`. ADDA binary SHA-256: `98c96b7a8c3b00815383da6ef42c4b5e0052f0ee6490ca412fb2773e245d6582`.

| case | BEM wall, s | ADDA wall, s | ADDA/BEM | BEM residual | ADDA residual | BEM grid change | ADDA grid change | BEM-ADDA | valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sphere_ka2 | 3.76 | 0.21 | 0.056x (17.90x BEM slowdown) | 6.50e-06 | 6.81e-06 | 0.62% | 0.35% | 0.81% | yes |
| sphere_ka4 | 12.88 | 0.25 | 0.019x (51.52x BEM slowdown) | 6.32e-06 | 2.52e-07 | 0.24% | 0.24% | 0.39% | yes |
| sphere_ka6 | 54.10 | 0.38 | 0.007x (142.37x BEM slowdown) | 5.45e-06 | 1.97e-06 | 0.12% | 0.09% | 0.22% | yes |
| sphere_ka8 | 52.52 | 0.70 | 0.013x (75.03x BEM slowdown) | 7.37e-06 | 5.62e-06 | 0.17% | 0.19% | 0.20% | yes |
| sphere_ka10 | 49.51 | 1.28 | 0.026x (38.68x BEM slowdown) | 4.06e-06 | 7.77e-06 | 0.17% | 0.24% | 0.32% | yes |
| prism_ka2 | 3.10 | 0.21 | 0.068x (14.76x BEM slowdown) | 6.14e-06 | 7.30e-06 | 0.15% | 0.52% | 0.46% | yes |
| prism_ka4 | 11.69 | 0.25 | 0.021x (46.76x BEM slowdown) | 6.36e-06 | 7.04e-06 | 0.03% | 1.06% | 0.33% | yes |
| prism_ka6 | 45.90 | 0.34 | 0.007x (135.00x BEM slowdown) | 7.43e-06 | 2.15e-06 | 0.00% | 0.11% | 0.34% | yes |
| prism_ka8 | 47.45 | 0.57 | 0.012x (83.25x BEM slowdown) | 6.15e-06 | 9.18e-06 | 0.01% | 0.32% | 0.14% | yes |
| prism_ka10 | 38.61 | 1.04 | 0.027x (37.12x BEM slowdown) | 9.89e-06 | 7.62e-06 | 0.02% | 0.25% | 0.14% | yes |

A row marked `no` has no publishable speedup claim. Inspect the CSV/JSON gate fields instead of quoting its wall-time ratio.
