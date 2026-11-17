# CPU/GPU: measurements through ka=80

These are preserved measurements from 16–17 November 2026, not new runs made
for publication. [Provenance](provenance.json) records original paths and hashes.
Absolute checkout prefixes in the exported JSON were made relative. The large
currents, FFT factors and machine-specific libraries are not distributed.
The separate [publication validation](validation.json) records clean-source
builds and CPU tests performed on November 17. It does not rerun these
performance measurements or any GPU numerical calculation.

Hardware: Ryzen 7950X (16 cores, 32 OpenMP threads), RTX 3090 Ti 24 GiB,
about 62 GiB RAM. Shape: regular SciIce hexagonal column, volume-equivalent ka,
H/width across flats=1.6495721976846451, m=1.3116, wavelength=0.532 micrometres.
Build and run commands are in the [main README](../../README.md).

## Four incident modes, ka=10–40

[Data](cpu_four_modes.json) · [Plots, seven pages](cpu_dirac_comparison.pdf)

Modes `(p,l,m)=(0,2,-1),(1,1,-1),(0,ka,-1),(1,ka-1,-1)` share a symmetry
sector. CPU Dirac uses P3, residual ≤1e-8, 32 OpenMP threads, 16 BLAS threads
only inside coarse LU. At ka10 the field reference is finer Dirac ref3;
ka20/30/40 use saved Muller ref5/ref6/ref6 results. The intensity check is
`max_theta |I_D-I_ref|/I_ref`, `I=sum(abs(E)**2)`, at 181 angles in one
meridian. No fitted amplitude or intensity normalization is used. This does
not establish convergence of every element of an orientation-averaged matrix.

| ka | CPU profile | Four-mode time, s | Peak RSS, GiB | Iterations |
|---:|---|---:|---:|---|
| 10 | dense SIMD, ref2 | 4.984 | 0.845 | 6/6/6/6 |
| 20 | dense SIMD, ref3 | 45.312 | 12.043 | 7/7/7/7 |
| 30 | FMM full P1, ref4, before compact storage | 523.749 | 25.003 | 7/7/7/7 |
| 40 | FMM compact full P1, ref4 | 439.257 | 21.019 | 11/11/11/12 |

ka40 CPU time sums a cold solve of 438.449 s and an independently measured
0.808 s outgoing projection to L65. The cold solve used FMM packet 1.
Packet 2 is selected in the current frontend after a separate working-action
comparison; no new faster cold solve timing is inferred from it.

The same four modes and outgoing L65 took 344.126 s on GPU Muller ref6,
residual ≤2e-6, **reusing prepared FFT factors**. CPU/GPU=1.276. The original
FFT construction took 212.024 s on November 9; adding it gives 556.151 s,
not a new cold end-to-end measurement (older geometry/binding preparation
may also be absent). Full-T packing across sectors was not timed here.
The JSON's legacy `speedup_Muller_over_Dirac` field is GPU/CPU=0.783;
the README's speed ratio is its reciprocal, CPU/GPU.

Maximum sampled intensity errors at ka10/20/40: 0.07871% / 0.60569% /
0.39682%. ka40 T-column relative L2 differences: 0.02432%, 0.02401%,
0.04632%, 0.06158%. At ka30, direct Dirac versus L50-truncated Muller
initially differed by 1.1824%; projecting both to L50 gives 0.4013%.
Dirac's own L50 truncation reaches 1.0704%, L60 only 0.006115%. The saved
currents were reused for this diagnosis; no tighter boundary solve was needed.

The compact P1 correction removed redundant low rows of AZ and low-space
ILU work, preserving the panel inverse's high-to-low coupling. ka40 coarse
preparation fell 421.283→186.570 s, cold solve 674.370→438.449 s, peak
RSS 25.496→21.019 GiB. Small complete T126 regression after compact storage
and the selected packet 2: relative T error 1.676e-15, all-16-element
SO(3) error normalized by reference M11, 1.266e-14.

## ka80

These are **operator measurements**, not completed column or T-matrix timings.
Both formulations here use 758,784 triangles, but different polynomial/vector
bases: Muller BDM1 has 4,552,704 unknowns; Dirac P3 has 60,702,720.

| Measurement | Seconds | Data / qualification |
|---|---:|---|
| Muller GPU compressed FFT action | 1.699–1.748 | [Complete action check](muller_fft_ka80.json); error ≤1.612e-4 versus native, gate 5e-4 |
| Muller full strict action | 51.871 | [Native comparison](muller_strict_ka80.json); previously 87.669 s, change ≤3.28e-13 |
| Dirac CPU full working action, both media | 62.049 | [CPU kernel measurements](dirac_cpu_ka80.json), 32 threads, eight components |
| Dirac GPU full working action, both media | 29.185 | [GPU measurements](dirac_gpu_ka80.json); error 4.15e-11 versus the earlier CPU baseline |

Dirac CPU/GPU≈2.13 uses the latest CPU timing and a separate GPU run. Its
GPU plans occupy 19.955 GiB. The comparison-process peak RSS of 37.027 GiB
includes CPU baseline/working arrays, not just the GPU path's host data.
The CPU report's strict 105.784 s is **one exterior-medium four-component
packet** and must not be compared as a whole residual against Muller's
51.871 s. GPU Dirac retains CPU strict checks.

The [interrupted Muller pilot](muller_interrupted_ka80.json) averaged
1.978748 s/iteration for its first 128 iterations; its first true residual
was 3.613826e-4, above the unchanged 2e-6 target. It saved no columns.
Projected residuals are not accepted as final residuals.

For planning only, let `n` be mean iterations per independent RHS, `c` full
strict checks per RHS, and P the scheduled modal-pair count. An approximate
main-loop time is `P * 2 * (n * 1.978748 + c * 51.870809)` seconds.
The actual [symmetry scheduler budget](ka80_budget.json) gives P=1299/1472/1676
for candidate L=120/128/137, including initial controls.

| Assumption | L120 | L128 | L137 |
|---|---:|---:|---:|
| n=100, c=2 | 9.07 d | 10.28 d | 11.70 d |
| n=150, c=2 | 12.04 d | 13.65 d | 15.54 d |
| n=200, c=3 | 16.58 d | 18.79 d | 21.39 d |

This excludes global preparation, additional audits, radiation, assembly,
analytic averaging and ref/L convergence studies. Neither n nor the L/ref
needed for 1% is established. One complex128 T occupies 12.775/16.504/21.615
GiB at these orders; caches, checkpoints and working arrays are additional.
The seven-day target is not demonstrated.
