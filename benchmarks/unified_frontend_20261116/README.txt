Unified BEM interface and memory lifecycle validation, 2026-11-16
================================================================
Entry points: ./bem solve, ./bem wizard, ./bem methods, ./bem continue.
The Russian MANUAL.md/MANUAL.pdf documents physical conventions and limits.

report.json records:
- 27 frontend/dispatch tests plus16 subtests; the legacy frontend also passed.
- 28 distinct Dirac CPU Python tests (original27 suite plus the close-lifecycle
  test in the seven-test hierarchy rerun).
- Native Dirac, compact-cache, two-medium CPU controls.
- GPU two-medium P2P:432 cases, and optional FINUFFT radiation:48 cases.
- Real fixed-incidence CPU Dirac and GPU Muller solves, tilted absorbing prism;
  full small T/SO3 for both; T-only and checkpoint continuation; legacy PMCHWT.
- Full T126 after mapped storage and explicit cleanup, compared with the saved
  dense control. These are same-mesh equivalence checks, not a1% mesh certificate.
- A clean checkout exported only from the Git index: rebuilt Dirac and SO3,
  completed a small T and average without workspace binaries/data dependencies.

The initial native_checks.log requested the optional FINUFFT target without
its include/library flags and records that missing-header error. The separate
finufft_check.log records the successful configured rebuild and48-case test.
The standard Muller worker builds and runs without FINUFFT.

Long ka20/ka40 probes and actual peak RSS are recorded separately in
../dirac_sparse_coarse_20261116/report.json. Frontend smoke RSS is sampled
periodically and is not a benchmark peak for short processes. No full T20/T40
speedup is inferred from the small interface tests. Completed continue does not
rerun solves or audits; unfinished runs retain native checkpoint validation.
