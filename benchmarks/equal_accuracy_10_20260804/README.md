# Ten predeclared BEM versus ADDA timing ratios

The ratio is `median ADDA complete wall time / median BEM complete wall time` from three independent runs with fresh application-cache directories. CUDA/OpenCL driver and compiler caches were not flushed, so this is a warm-system benchmark. A value below one is a BEM slowdown, not an acceleration. All ten cases were declared before execution; failed or unfavorable rows are retained.

Both programs use a `1e-5` linear residual target, two independently solved polarizations, and the same 181 scattering angles. ADDA reports a final recalculated residual. BEM reports a final exact-FMM-operator residual. The production discretizations are BEM >=15 points per shortest wavelength and ADDA dpl=20; BEM one-level-coarser and ADDA dpl=15 controls must change the normalized complete Mueller matrix by no more than 2%. The two production matrices must agree within 5%, including forward M11.
For spheres, both production matrices must also agree with exact Mie theory within 2% in normalized shape and forward M11.
The ADDA baseline is the clean official adda-team/adda commit 8f550a7, not the locally modified FP32 experimental build.
BEM binary SHA-256: `4adf35cabc82a98ffa28dce8a55535542aba42ddbfe492eef68d7068c56ef49a`. ADDA binary SHA-256: `98c96b7a8c3b00815383da6ef42c4b5e0052f0ee6490ca412fb2773e245d6582`.

| case | BEM wall, s | ADDA wall, s | ADDA/BEM | BEM residual | ADDA residual | BEM grid change | ADDA grid change | BEM-ADDA | valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sphere_ka2 | 5.10 | 0.21 | 0.041x (24.29x BEM slowdown) | 4.39e-06 | 6.81e-06 | 0.62% | 0.35% | 0.81% | yes |
| sphere_ka4 | 19.71 | 0.25 | 0.013x (78.84x BEM slowdown) | 4.04e-06 | 2.52e-07 | 0.24% | 0.24% | 0.39% | yes |
| sphere_ka6 | 81.77 | 0.38 | 0.005x (215.18x BEM slowdown) | 8.14e-06 | 1.97e-06 | 0.12% | 0.09% | 0.22% | yes |
| sphere_ka8 | 89.78 | 0.70 | 0.008x (128.26x BEM slowdown) | 5.56e-06 | 5.62e-06 | 0.17% | 0.19% | 0.20% | yes |
| sphere_ka10 | 51.54 | 1.28 | 0.025x (40.27x BEM slowdown) | 1.47e-06 | 7.77e-06 | 0.17% | 0.24% | 0.32% | yes |
| prism_ka2 | 6.04 | 0.21 | 0.035x (28.76x BEM slowdown) | 7.58e-06 | 7.30e-06 | 0.15% | 0.52% | 0.46% | yes |
| prism_ka4 | 24.78 | 0.25 | 0.010x (99.12x BEM slowdown) | 8.49e-06 | 7.04e-06 | 0.03% | 1.06% | 0.33% | yes |
| prism_ka6 | 102.07 | 0.34 | 0.003x (300.21x BEM slowdown) | 8.94e-06 | 2.15e-06 | 0.00% | 0.11% | 0.34% | yes |
| prism_ka8 | 110.13 | 0.57 | 0.005x (193.21x BEM slowdown) | 8.34e-06 | 9.18e-06 | 0.01% | 0.32% | 0.14% | yes |
| prism_ka10 | 38.48 | 1.04 | 0.027x (37.00x BEM slowdown) | 2.26e-06 | 7.62e-06 | 0.02% | 0.25% | 0.14% | yes |

A row marked `no` has no publishable speedup claim. Inspect the CSV/JSON gate fields instead of quoting its wall-time ratio.
