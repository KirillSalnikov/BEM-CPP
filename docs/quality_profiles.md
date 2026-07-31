# Quality-profile validation

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

## Reproduction

Replace `PROFILE` with `quick`, `standard`, or `strict`:

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
