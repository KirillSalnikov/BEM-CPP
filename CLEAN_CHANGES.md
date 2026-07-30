# Clean BEM-CPP development snapshot

This directory is a fresh clone of:

- repository: `https://github.com/KirillSalnikov/BEM-CPP`;
- branch: `main`;
- baseline commit: `0efded5c427d01e86b744532de2398e07a148520`.

The working-tree changes on top of that commit contain the completed parts
of the BEM development work. Generated binaries, caches, incomplete runs,
and the original large `runs/` directory are excluded.

## Included changes

1. GPU FMM gradient and symmetric Hessian, including a combined traversal
   and exact near-field Hessian.
2. GPU right preconditioning and current GraphSAI import/action-dump path.
3. PMCHWT controls: ILU(0), local blocks, RWG mass matrix, and the explicitly
   experimental nonconforming RWG operator square.
4. The separate Muller second-kind equation with both the smooth nodal P2
   reference and an H(div)-conforming BDM1 edge basis for sharp particles.
5. Dense reference assembly, singular Duffy quadrature, matrix-free FMM
   action, Morton block-Jacobi, plane-wave RHS, and far field.
6. A local-block exporter for the independent neural Muller training path.
7. CPU and CUDA source tests plus the minimal JSON results used in the
   report.

## Verification

```bash
make host-checks CXX=g++-12 CUDA_HOME=/usr -j4
make cuda-hessian-check CXX=g++-12 CUDA_HOME=/usr
make cuda-pfft-hessian-check CXX=g++-12 CUDA_HOME=/usr
make cuda-muller-fmm-check CXX=g++-12 CUDA_HOME=/usr
make cuda-muller-edge-check CXX=g++-12 CUDA_HOME=/usr
```

These checks were run successfully in this directory on 2026-07-30.

## Report

The Russian technical report is:

```text
docs/work_report_ru.pdf
```

Its editable source is `docs/work_report_ru.html`; plots are rebuilt from
`results/reference/`:

```bash
docs/build_work_report.sh
```

## Inspecting the development history

```bash
git status --short
git log --stat
git show
```

The first integrated Muller/FMM development commit is based directly on
the published baseline listed above. Generated binaries, local caches, and
the large `runs/` workspace remain excluded from version control.
