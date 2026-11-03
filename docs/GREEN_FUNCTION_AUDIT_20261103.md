# Independent Green-function audit, 2026-11-03

For the repository's exp(-i omega t) convention the outgoing fundamental
solution is **G_k(R)=exp(+i k R)/(4 pi R)**, satisfying
(Laplacian+k²)G=-delta. Positive Im(k) attenuates the outgoing wave. This sign,
normalization and absorption agree with the
[outgoing Helmholtz Green function derivation](https://farside.ph.utexas.edu/teaching/jk1/Electromagnetism/node126.html).
The direct CUDA P2P code uses the same exponential and 1/(4 pi R).

The audit found a real error in the small-argument **Hessian**, not in this scalar
definition. In the difference G_ke-G_ki, the nth exponential-series term is
proportional to R^(n-1). Its Cartesian Hessian is

```text
d_i d_j R^(n-1) = (n-1) R^(n-3) [delta_ij + (n-3) u_i u_j],  u = r/R.
```

The previous implementation multiplied the identity term by (n-1) but omitted
that factor on the radial dyadic term. The error starts at n=4. It existed in
both `src/muller_nodal.cpp` and the frequency-series copy
`src/muller_taylor_kernel.h`; both are corrected. Previous comparisons between
those two implementations could share the same error.

The new [independent test](../tests/test_green_function_independent.py) evaluates
the actual production combined kernel via a C++ probe, while its oracle
differentiates exp(ikR)/(4 pi R) directly in Cartesian coordinates using mpmath
at 80 decimal digits. It does not call production derivative helpers. There are
48 cases covering complex absorption, reversed displacement, very small radius,
and both sides of the |kR|=0.01 switch, plus the Helmholtz trace identity.

```bash
python3 -m pip install -r requirements-validation.txt
make green-audit CXX=g++
python3 benchmarks/gpu_thermal_v1/results/rtx5080_20261103/reproduce_dense_impact.py --output runs/dense-green-control
```

The uncorrected source fails the new test. Corrected worst relative errors:
Hessian **3.396e-11**, gradients **1.740e-15**, scalar **1.905e-15**; trace defect
scaled by Hessian norm **3.027e-11**. Existing nodal, contour/Taylor-moment,
combined-bundle and local-block controls also pass. Detailed measurements:
[independent audit](../benchmarks/gpu_thermal_v1/results/rtx5080_20261103/green_independent.json).

An exact dense-LU before/after control on the same coarse sphere mesh (168
unknowns, n=1.3116, both polarizations, 181 angles) gives:

| ka | max abs(delta Mij)/M11_before | Full-matrix relative L2 |
|---:|---:|---:|
| 0.01 | 5.671e-6 (0.000567%) | 3.946e-6 |
| 1 | 2.073e-8 | 1.844e-8 |
| 10 | 2.093e-10 | 1.014e-10 |

All dense residuals are below 1.4e-15. These controls measure the correction's
impact on fixed discrete systems; they do not establish mesh-converged physical
accuracy or a universal bound for production shapes.
[Raw dense controls](../benchmarks/gpu_thermal_v1/results/rtx5080_20261103/green_dense_solution_impact.json).

Near-correction cache algorithm version is bumped from 2 to 3; MBJ cache format
from 1 to 2, so old local operators cannot silently survive the correction.
An actual old-write/new-reject/new-hit control passed. Corrected full-solver
hardware-suite reference gates also passed for both published cases.

## Remaining numerical limitations

This audit does not certify the entire GPU/FMM implementation. Separate surface
FMM checks fail strict 1e-7 internal path-consistency gates (differences roughly
1.5e-6 and 2.6e-6), while direct-reference Hessian/gradient L2 errors are about
3.8e-5 / 2.4e-4. Explicit near FP64 plus six far-storage/cache overrides did not
remove those discrepancies. FMM and P2P source files are unchanged by this patch.
Raw failures and exact environments are published in the
[thermal result report](../benchmarks/gpu_thermal_v1/results/rtx5080_20261103/README.md).

Exploratory mixed-FMM prism ref=2 solves at ka=0.01 and 1 did not converge on the
old baseline, before this correction (residuals about 1.02 and 0.137 after 500
iterations). Such outputs were rejected and are not presented as valid solutions
or benchmark scores. The independent dense-LU controls above avoid conflating
that pre-existing solver issue with the Green Hessian correction. Further
small-ka FMM/cancellation investigation is required before certifying those
mixed-precision regimes.
