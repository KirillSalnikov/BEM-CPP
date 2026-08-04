# Strict BEM optimization benchmark

This compares the BEM implementation before and after mixed iterative
refinement. Both source benchmarks use the same predeclared ten cases, three
independent cold complete-process runs per case, a `1e-5` true FMM residual,
two independently solved polarizations, 181 angles, mesh-convergence gates,
ADDA agreement, and Mie checks for spheres. No unfavorable row was removed.

| case | before, s | optimized cold, s | BEM speedup |
|---|---:|---:|---:|
| sphere_ka2 | 5.10 | 3.76 | 1.356x |
| sphere_ka4 | 19.71 | 12.88 | 1.530x |
| sphere_ka6 | 81.77 | 54.10 | 1.511x |
| sphere_ka8 | 89.78 | 52.52 | 1.709x |
| sphere_ka10 | 51.54 | 49.51 | 1.041x |
| prism_ka2 | 6.04 | 3.10 | 1.948x |
| prism_ka4 | 24.78 | 11.69 | 2.120x |
| prism_ka6 | 102.07 | 45.90 | 2.224x |
| prism_ka8 | 110.13 | 47.45 | 2.321x |
| prism_ka10 | 38.48 | 38.61 | 0.997x |

Median cold speedup: **1.620x**. Geometric mean:
**1.612x**. The range is
**0.997x--2.321x**.
The `ka=10` pFFT cases barely change because this optimization targets the
direct FMM+MBJ Krylov path.

The content-addressed setup cache was tested separately on the production
`prism_ka6, ref=5` case. Complete wall time changed from **45.73 s**
with an empty cache to **28.22 s** in a new output directory with a
validated cache hit: another **1.620x**. Relative
L2 change of the full Mueller matrix was **3.431e-09**, and the
maximum true residual was **7.435e-06**.

Relative to the pre-optimization BEM time of
`102.07 s`, the repeated calculation now
takes `28.22 s`, a total BEM speedup of
**3.617x**.

This is an implementation speedup claim for BEM. It is not a claim that BEM
beats ADDA; ADDA remains faster in all ten equal-accuracy cases.
