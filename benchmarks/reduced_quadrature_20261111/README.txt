Reduced quadrature inner inverse, 2026-11-11

Reduced Galerkin quadrature as an inner inverse, inspired by the paper; not a full Nystrom EFIE implementation.

Same mesh, BDM1 currents, 7-point regular quadrature, Duffy6, mass13, tol2e-6.

Completed screening:
bullet ka=10 ref=3
  q=1: operator action speedup 3.423
  nested4: solve speedup 1.323; accepted=True; max strict residual 1.846e-07
  nested8: solve speedup 1.349; accepted=True; max strict residual 1.299e-07
  refine: solve speedup 0.426; accepted=False; max strict residual 2.398e-06
  q=3: operator action speedup 2.213
  nested4: solve speedup 1.194; accepted=True; max strict residual 1.266e-07
  nested8: solve speedup 1.245; accepted=True; max strict residual 3.063e-08
  refine: solve speedup 0.787; accepted=True; max strict residual 1.649e-06
bullet ka=20 ref=4
  q=1: operator action speedup 2.714
  nested4: solve speedup 1.192; accepted=True; max strict residual 1.749e-07
  nested8: solve speedup 1.094; accepted=True; max strict residual 1.493e-07
  q=3: operator action speedup 1.960
  nested4: solve speedup 1.067; accepted=True; max strict residual 1.306e-07
  nested8: solve speedup 1.177; accepted=True; max strict residual 9.570e-08

This is an isolated experimental binary. No production worker was replaced.
The 1/3-point rules are isolated in source/quadrature.h; original rules remain unchanged.
The initial 3-point launch exposed incompatible FMM band occupancy. It was corrected
by scaling leaf populations with point count, preserving the spatial depth policy.
Moment exactness for degree 1/2 and the original degree-5 rule was checked.
A7 and Aq share BDM1 unknown ordering and exact adjacent-pair Duffy6 corrections.
The mass and RHS/far-field quadratures retain 13 points.
A7 MBJ50 is shared as the local inverse for matched comparisons.
baseline/refine keep separate recycle8 histories; flexible variants retain each actual inner inverse image.
Refine uses Aq in Arnoldi and checks original A7 at correction boundaries.
Nested4/8 use the original A7 outer operator and at most 4/8 Aq inner steps (target .1).
Every final solution is independently checked against A7 at tolerance 2e-6.
Rejected rows are recorded and never accepted as solutions.

Files: build.py, prepare.py, run_one.py, report.py, probe.cpp.
run_one.py temporarily pauses the verified production process group and resumes it
in finally; an independent watchdog also resumes it after a bounded interval.
production_pauses.jsonl records interruptions; each q*.run.json records commands and environment.
Report: report.json, reduced_quadrature.pdf.

Limitations:
Same-mesh comparison, not a mesh convergence certificate.
Fixed incident directions and selected spherical RHS; no complete new T matrix or orientation average.
GPU timing isolates production by temporary SIGSTOP/SIGCONT; pause receipts saved.
FGMRES does not currently share the baseline Krylov recycling; each variant owns its history.

Production-mesh packed test: bullet_ka20_ref5_packed/ (see its live q1.jsonl).

FINAL production-mesh result:
{
  "variant": "nested8",
  "rhs_count": 2,
  "complete": true,
  "accepted": true,
  "solve_s": 27.177356478,
  "baseline_s": 43.521455284,
  "solve_speedup": 1.601386629315125,
  "including_extra_operator_setup_speedup": 1.4086151910203284,
  "including_common_and_extra_setup_speedup": 1.121559935413765,
  "maximum_strict_residual": 1.8546280569495112e-07,
  "maximum_far_relative_error": 2.6045507726589357e-07,
  "full_actions": 24,
  "cheap_actions": 174,
  "strict_actions": 6,
  "outer_iterations": 24,
  "estimated_setup_break_even_rhs": 0.4551206568372725
}
Extra Aq setup: 3.719268492 seconds.
One C6 packet, two phase codes, four CPU threads; not a full-T speedup certificate.
See docs/acceleration_plan.md, 2026-11-11 entry, for the next held-out validation step.
