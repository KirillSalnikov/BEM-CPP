# Contributing

Use a focused branch and keep numerical changes separate from report-only
changes. Before opening a pull request, run:

```bash
scripts/release_audit.sh --host
```

Changes to CUDA kernels, quadrature, discretization, preconditioners, Krylov
logic, symmetry handling, or mixed precision also require:

```bash
scripts/release_audit.sh --gpu
examples/run_small_sphere_mie_check.sh
```

A numerical pull request must state:

- the exact command and Git commit;
- GPU, driver, CUDA, compiler, and precision mode;
- true full-operator residual, not only projected Krylov residual;
- setup, solve, far-field, and complete wall times;
- mesh, quadrature, and FMM convergence controls;
- changes in all relevant Mueller elements.

Do not commit `runs/`, binaries, caches, checkpoints, or machine-specific
paths. Add compact deterministic reference data only when a test consumes it.
