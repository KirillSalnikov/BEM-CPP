# Quality-profile validation

## Large-prism preview

The separate `preview` profile is validated only for the axial regular
hexagonal prism at `ka=80`, `m=1.3`, and `ref=6` (806,400 unknowns). It is not
part of the general orientation suite and does not claim a converged FMM
residual. It stops after three outer pFFT-FGMRES steps, accepts the projected
residual, checks the C6 candidate for the second polarization with the
discrete operator, and applies a two-step correction when required.

| Historical result | Withdrawn uniform-FMM reference | Preview |
|---|---:|---:|
| Complete wall time with reusable local caches | 255.56 s | 34.44 s |
| Outer iterations | 34 | 3 primary + 2 correction |
| Reported residual | old-operator `9.26e-6` | projected `2.17e-2` / `1.25e-2` |
| Weighted full-Mueller difference | reference | 0.388% |
| Weighted M11 difference | reference | 0.214% |
| Forward M11 difference | reference | 0.282% |

These are retained historical measurements, not a current speed/accuracy
claim: the reference checkpoint has residual `1.24e-1` under the corrected
two-band FMM operator. The reference/result paths and overlay plot are in
`runs/preview_size_sweep_20260803/ka80_preview_c6verified/` and can be
regenerated with `scripts/report_preview_validation.py`.

### Fixed-refinement size controls

The same fast stopping policy was checked against independently converged
two-polarization results at `m=1.3`, `ref=6`. Rotational reconstruction was
always checked by the discrete operator and corrected when required.

| `ka` | Wall time | Primary / correction steps | Full-Mueller difference | Status |
|---:|---:|---:|---:|---|
| 20 | 15.76 s | 3 / 0 | 0.666% | accepted control |
| 30 | 19.21 s | 4 / 0 | 0.127% | accepted control |
| 60 | 30.02 s | 4 / 2 | 0.229% | accepted control |
| 80 | 34.44 s | 3 / 2 | 0.388% | public `preview` case |
| 111 | 30.61 s | 3 / 0 | 45.7% | rejected |

The failure at `ka=111` is deliberate evidence against extrapolating a fixed
iteration count. A true-residual solve of both polarizations recovered the
independent reference to `0.0002%`. The machine-readable metrics and plots are
`runs/preview_size_sweep_20260803/preview_size_sweep.{json,png,pdf}`; regenerate
them with `scripts/report_preview_size_sweep.py`.

The broader reproducible orientation suite is implemented by
`scripts/validate_orientation_profiles.py`. It covers spheres, a cube,
regular prisms with 5--8 sides and different aspect ratios, and an asymmetric
OBJ mesh over `ka=1..10` and `m=1.1..2.5`. Run `run --keep-going`, then
`report`; the generated report records failed stress controls instead of
silently comparing them with accepted results.

For built-in shapes, the automatic surface refinement is selected from the
shortest exterior/interior wavelength:

```text
ref = max(ref_min, ceil(log2(P * ka * h0 * max(1, |m|) / (4*pi)))).
```

Here `P=4` for `quick` and `P=8` for `standard`/`memory`, while `h0` is the longest
edge scale of the initial shape mesh. The `4*pi` denominator counts quadratic
P2 nodes, including edge midpoints; the previous `2*pi` formula counted
elements while reporting them as nodes. Consequently, increasing either `ka`
or `|m|` can increase `ref`; an explicit `--ref` remains an expert override.

The `quick`, `standard`, and `strict` presets were compared on the same RTX
3090 Ti with 16 host threads. All reported residuals are true residuals of the
full FMM operator. Mueller differences use all 16 elements, normalized by each
result's `M11(0)`. Wall time includes cold operator and preconditioner setup,
both incident polarizations, the far field, and wrapper validation.

## Hexagonal prism

Parameters: equal-volume `ka=10`, real relative refractive index `m=1.5`,
height/diameter ratio 1.

| Profile | Mesh | Unknowns | Iterations (two polarizations) | Maximum residual | Wall time | Normalized Mueller difference from strict fine | Forward `M11` difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| quick | ref3 | 12,672 | 27 / 28 | 9.81e-4 | 2.30 s | 0.249% | 0.255% |
| standard | ref4 | 50,688 | 8 / 8 | 9.39e-6 | 10.07 s | 0.0226% | 0.0101% |
| strict coarse | ref4 | 50,688 | 10 / 10 | 3.07e-7 | 35.91 s | 0.0221% | 0.0111% |
| strict fine | ref5 | 202,752 | 11 / 11 | 3.63e-7 | 249.17 s | reference | reference |

The complete strict suite took 285.09 s and passed its two-mesh gate. On the
same ref4 FP64 problem, pFFT used only as a right preconditioner reduced the
outer iterations from 63/64 to 10/10 and the two-polarization solve time from
99.47 s to 26.62 s (3.74x). Its normalized Mueller difference from direct FMM
was 1.53e-7.

## GPU-memory profiles

The default `standard` profile now disables the FP32 FMM phase cache and the
flattened near-source cache. The `memory` profile additionally evaluates the
electric and magnetic FMM currents sequentially, avoiding paired mixed and
strict FP64 workspaces. The following cold-start runs used the same hexagonal prism, `ka=20`,
`m=1.3`, explicit `ref=5`, 202,752 system unknowns, and 181 scattering angles.

| Profile | Peak combined GPU allocation | Two-polarization solve | Full wall time | Iterations | Maximum true residual |
|---|---:|---:|---:|---:|---:|
| cached control (former default) | 5272 MiB | 51.66 s | 71.89 s | 9 / 9 | 6.654e-6 |
| `standard` | 3924 MiB | 52.01 s | 71.78 s | 9 / 9 | 6.654e-6 |
| `memory` | 2724 MiB | 62.19 s | 81.95 s | 9 / 9 | 6.654e-6 |

The default cache change saves 1348 MiB (25.6%) with a 0.998x total-time ratio,
within timing noise. Relative to the new default, `memory` saves another 1200
MiB (30.6%) and costs 1.142x in total time. Relative to the former default it
saves 2548 MiB (48.3%). Comparing `memory` with `standard`, the normalized L2
difference over all 16 Mueller elements was `2.261e-12`, and the maximum
absolute element difference normalized by `M11(0)` was `1.883e-12`. Thus this
test isolates a storage/performance tradeoff rather than a numerical-quality
tradeoff. The artifacts are under
`runs/memory_profile_validation_ka20_ref5/`.

## Small sphere

Parameters: `ka=3`, real relative refractive index `m=1.3`.

| Profile | Mesh | Unknowns | Iterations (two polarizations) | Maximum residual | Wall time | Normalized Mueller difference from strict fine | Forward `M11` difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| quick | ref2 | 2,568 | 7 / 7 | 8.08e-4 | 0.94 s | 0.657% | 4.42% |
| standard | ref2 | 2,568 | 12 / 12 | 6.52e-6 | 1.16 s | 0.308% | 1.58% |
| strict coarse | ref2 | 2,568 | 14 / 14 | 9.14e-7 | 4.14 s | 0.290% | 1.55% |
| strict fine | ref3 | 10,248 | 14 / 14 | 4.15e-7 | 13.93 s | reference | reference |

An earlier quick minimum of ref1 took 0.87 s but had an 11.26% forward `M11`
error. Raising the minimum to ref2 costs only 0.07 s in this control and makes
the exploratory profile materially safer. pFFT remains disabled below `ka=10`,
where direct FMM+MBJ is faster and avoids unnecessary setup.

## Adaptive orientation averaging

For a non-averaged built-in regular prism, `quick`, `standard`, and `memory`
map the first-polarization solution through cyclic symmetry to obtain a
candidate for the second polarization. The full FMM operator then evaluates
its residual and the solver computes a correction whenever the profile
tolerance is not met. `--independent-polarizations` disables the candidate,
and `strict` always solves both polarizations independently. Orientation
averaging has its own two-polarization reuse path and is unchanged by this
single-orientation optimization.

The adaptive beta/gamma path was exercised through the public `bem average`
interface on a sphere with `ka=1`, `m=1.3`, automatic `ref`, and eight alpha
samples. Each profile started from its own minimum level and stopped only after
all three angular criteria passed. Times include cold setup, all orientation
solves, far-field evaluation, and wrapper validation.

For regular prisms and cubes, adaptive runs now identify the proper dihedral
rotation `(beta,gamma) -> (180 degrees-beta,-gamma mod gamma_period)` and solve
only one member of each equivalent pair. Existing full-grid checkpoints gave
maximum pairwise complete-Mueller relative L2 differences of `4.36e-6`,
`1.96e-6`, and `4.68e-6` for five-, six-, and seven-sided prisms, and at most
`3.02e-5` for the tested cubes. On the high-contrast eight-sided prism,
reconstructing the complete `J=4` weighted average from one member per pair
changed the aggregate matrix by `2.42e-6` in relative L2 and changed forward
`M11` by `1.46e-6`. These differences are at or below the linear-solver error.
The reduction preserves every quadrature weight and lowers a complete `J=5`
level from 1056 to 529 unique base-orientation solves.

| Profile | Mesh | Allowed levels | Accepted level | Solved base orientations | Total iterations | Maximum residual | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| quick | ref2 | 1--3 | 2 | 20 | 184 | 8.47e-4 | 1.29 s |
| standard | ref2 | 2--4 | 3 | 72 | 1,228 | 9.00e-6 | 3.47 s |
| strict coarse | ref2 | 2--5 | 3 | 72 | 1,736 | 8.98e-7 | 64.25 s |
| strict fine | ref3 | 2--5 | 3 | 72 | 1,674 | 8.42e-7 | 245.49 s |

The strict two-mesh suite took 309.74 s and passed. Its normalized Mueller
difference between ref2 and ref3 was 0.116%, and the forward `M11` difference
was 0.782%. Relative to strict fine, standard differed by 0.116% in the
normalized complete Mueller matrix and 0.775% in forward `M11`; quick differed
by 0.439% and 1.72%, respectively. The quick comparison required angular
interpolation because that profile intentionally writes a smaller scattering
grid.

These data validate control flow, checkpointing, automatic stopping, and the
quality ordering on one small case. They do not establish that level 3 is
sufficient for every shape or size; nonconvergence at a profile's maximum
level is reported as a failed validation.

## Expanded orientation matrix

The expanded matrix used 64 alpha samples and adaptive nested beta/gamma
quadrature. It contains ten physical cases: three spheres, two cubes, regular
prisms with 5, 6, 7, and 8 sides and aspect ratios from 0.5 to 1.4, and an
asymmetric OBJ mesh. Sizes span `ka=1..10` and real refractive indices span
`m=1.1..2.5`.

All 20 calculations completed and passed both the true-residual and profile
acceptance gates. All ten cases produced valid `quick`/`standard` pairs. Their normalized
full-Mueller L2 differences ranged from 0.012% to 0.324%. Forward `M11`
differences ranged from 0.079% to 2.10%; forward scattering is the more
sensitive diagnostic in the small-sphere controls.

| Case | quick | standard | quick/standard full-Mueller L2 |
|---|---:|---:|---:|
| sphere, `ka=1`, `m=1.3` | PASS, `ref=2`, `J=2` | PASS, `ref=2`, `J=3` | 0.324% |
| sphere, `ka=3`, `m=1.5` | PASS, `ref=2`, `J=2` | PASS, `ref=3`, `J=3` | 0.0572% |
| cube, `ka=2`, `m=1.3` | PASS, `ref=2`, `J=3` | PASS, `ref=2`, `J=3` | 0.0120% |
| prism 6, `ka=3`, `m=1.5` | PASS, `ref=2`, `J=3` | PASS, `ref=3`, `J=3` | 0.0533% |
| prism 7, `ka=5`, `m=1.3` | PASS, `ref=2`, `J=3` | PASS, `ref=3`, `J=4` | 0.0818% |
| prism 5, `ka=2`, `m=1.8` | PASS, `ref=2`, `J=3` | PASS, `ref=3`, `J=3` | 0.0571% |
| prism 8, `ka=4`, `m=2.5` | PASS, `ref=3`, `J=4` | PASS, `ref=4`, `J=5` | 0.153% |
| cube, `ka=5`, `m=2` | PASS, `ref=3`, `J=4` | PASS, `ref=4`, `J=5` | 0.140% |
| sphere, `ka=10`, `m=1.1` | PASS, `ref=3`, `J=2` | PASS, `ref=4`, `J=3` | 0.102% |
| asymmetric OBJ, `ka=3`, `m=1.5` | PASS, `ref=0`, `J=2` | PASS, `ref=1`, `J=3` | 0.0917% |

The independent Mie controls show the expected quality ordering. The
normalized complete-Mueller errors for `standard` were 0.269% at
`ka=1, m=1.3`, 0.0291% at `ka=3, m=1.5`, and 0.00394% at
`ka=10, m=1.1`. These are independent physical comparisons, not only
comparisons between two BEM profiles.

High contrast exposed two separate limitations. On the `ka=5, m=2` cube,
every linear system reached the requested true residual, but the result still
changed by 13.1% in the most sensitive normalized Mueller component from
`J=3` to `J=4`; this exceeded the 10% `standard` angular criterion. The
completed `J=5` control passed: from `J=4` to `J=5`, the M11 curve L2 change
was `3.66e-6`, the integral change was `6.82e-8`, and the maximum normalized
component change was 0.0575%. A direct complete-Mueller comparison between
the stored `J=4` and `J=5` results gave a normalized L2 difference of
`6.35e-6` and a forward M11 difference of `1.60e-7`. On the `ka=4, m=2.5`
eight-sided prism, the completed `J=5` control also passed. Relative to `J=4`,
the M11 curve L2 change was `3.92e-5`, the integral change was `1.86e-7`, and
the maximum normalized component change was 0.0315%. The direct complete-
Mueller normalized L2 difference was `4.67e-5`, and the forward M11 difference
was `5.17e-5`. Its maximum true residual was `9.97e-6`. Dihedral reuse reduced
the complete level from 1056 to 529 unique base-orientation solves; its
incremental continuation time is excluded from cold-runtime comparisons.

The `quick` high-contrast policy was therefore extended to `Jmax=4` for
`m>=2` and to 400 maximum iterations for `m>=2.5`. Both corresponding quick
stress cases then passed. Increasing the MBJ block size from 50 to 100 on the
`ka=5, m=2` cube reduced one orientation from 344 to only 340 iterations and
changed solve time from 6.88 to 6.93 seconds. The MBJ50 probe reused its
factorization cache whereas MBJ100 built a new one, so their total setup times
are not used for the decision. By contrast, on the `m=2.5` prism MBJ100 reduced
a representative heavy orientation from roughly 42 to 30 seconds. Therefore
MBJ50 remains the default below `m=2.5`, while `standard` selects MBJ100 at and
above that threshold. The remaining bottleneck is global high-contrast
convergence, not local block setup.

The generated detailed artifacts are under
`runs/orientation_profile_validation_v4/`: `report.md`, `summary.csv`,
`summary.json`, and `orientation_profile_validation.png`. The `runs/`
directory is intentionally not versioned; regenerate it with the commands
below.

## Reproduction

Replace `PROFILE` with `quick`, `standard`, `memory`, or `strict`:

```bash
./bem run --shape prism --sides 6 --aspect 1 --ka 10 --ri 1.5 \
  --quality PROFILE --out runs/profile_prism_PROFILE --yes

./bem run --shape sphere --ka 3 --ri 1.3 \
  --quality PROFILE --out runs/profile_sphere_PROFILE --yes
```

Compare a result with the fine strict reference:

```bash
./bem validate runs/profile_prism_standard/result.json \
  --reference runs/profile_prism_strict/fine/result.json
```

Run or resume the expanded quick/standard matrix and regenerate all reports:

```bash
scripts/validate_orientation_profiles.py run --keep-going
scripts/validate_orientation_profiles.py report
```

The report command returns a nonzero status while any case is missing or has
failed a numerical gate. This is intentional and prevents an incomplete
stress matrix from being mistaken for a fully passing release test.

## Validated two-stage high-frequency prism mode

`physical-fast` is a deliberately narrow two-stage profile for the regular
six-sided prism with `ka=60`, `80`, or `111`, `m=1.3`, and `ref=6`. It first
makes three cheap pFFT-FGMRES preview steps, then migrates that checkpoint to
the accurate two- or three-band FMM operator. The final Mueller matrix uses
181 scattering angles. C6 symmetry supplies a candidate second polarization,
but the full banded-FMM operator checks its residual and computes a correction
whenever it exceeds the selected tolerance.

```bash
./bem run --shape prism --sides 6 --aspect 1 --ka 80 --ri 1.3 \
  --ref 6 --quality physical-fast --out runs/prism_ka80_physical_fast
```

The measured cold run, including near-operator and MBJ cache construction,
took 282.54 s. A saved ADDA-OCL FP32 `dpl=15` run took 3285.48 s, but the two
times are not an acceleration benchmark: the BEM exact-operator residual was
`3.424e-3`, whereas ADDA requested `1e-4`. Against the fully converged BEM
reference, the solid-angle weighted relative L2 difference of the complete
Mueller matrix was `7.780e-5` (0.00778%), and the largest absolute element
difference divided by forward `M11` was `8.248e-5` (0.00825%).

No equal-accuracy speedup over ADDA is currently claimed. A valid comparison
must independently recalculate both final residuals, use the same target and
angular output, and establish discretization convergence for both methods.

This profile is not a `1e-5` linear-residual result and is not enabled outside
the validated particle and parameter set. Use `standard` when a `1e-5`
operator residual is the required acceptance criterion.

The fixed-orientation `quick`, `standard`, and `memory` profiles also use an
adaptive form of this pipeline for built-in meshes with `ka>=60`, `ref>=4`,
and at least 100,000 estimated unknowns. Its band split and leaf occupancy are
computed from refinement and electrical density, so operation is no longer
limited to three exact `ka` values. Only the regular-prism controls above carry
the published ADDA baseline and physical-validation numbers. All profiles
retain their normal residual targets (`1e-3`, `1e-5`, and `1e-5`), and
`--single-stage` provides a control run.

`fast` is a short command-line alias for `physical-fast`. Final output from
`quick`, `physical-fast`/`fast`, and `memory` requires verified operator
residuals for both polarizations. The validator rejects files produced with
the former unchecked cyclic-symmetry shortcut.

Orientation averaging already constructs the operator once and reuses it for
all Euler-angle right-hand sides. It additionally keeps lossless per-angle
parts and an atomic orientation checkpoint. A banded-pFFT averaging candidate
passed the residual and Mueller checks, but was slower than paired GPU-GMRES:
22.16 versus 21.09 s at `ka=20`, and 252.33 versus 139.93 s at `ka=60`.
Consequently it remains an experimental backend rather than a default.
