# Release Checklist

## Automated gate

1. Run `scripts/release_audit.sh --gpu` from a clean checkout.
2. Confirm that `git status --short` is empty after the audit.
3. Run `examples/run_small_sphere_mie_check.sh` and archive its JSON and log.
4. Record the GPU, driver, CUDA, compiler, command line, and Git commit.

## Numerical gate

For every parameter range claimed by a release:

1. verify the true residual with the full operator;
2. compare at least two surface refinements;
3. vary quadrature and FMM accuracy independently;
4. compare mixed precision with an FP64 control;
5. inspect all relevant Mueller elements;
6. compare spheres with Mie theory;
7. compare sharp and asymmetric particles with an independently converged
   edge-capable method;
8. report setup, solve, far-field, and complete wall times separately.

H(div)-BDM1 sharp-edge results are alpha quality until item 7 is satisfied for
the claimed size, refractive-index, and shape range.

## Publication gate

1. Update `VERSION`, `CHANGELOG.md`, and `CITATION.cff` together.
2. Confirm `LICENSE` and `NOTICE` are present in the source archive.
3. Publish the small reference outputs and SHA-256 checksums.
4. Create an annotated `v<version>` Git tag from the audited commit.
5. Run `scripts/package_release.sh` and verify its SHA-256 checksum.
6. Attach the source archive, checksum, manual, and release audit log.
