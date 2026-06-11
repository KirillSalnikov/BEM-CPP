# Greek ADDA large-size comparison

Reference data:
`/home/user/cluster/BEM-CPP/greek/ADDA_for_PO_comparison/refr_1_6__0_002/`

Common physics/run settings:
- `m = 1.6 + 0.002i`
- `ntheta = 181`
- orientation grid requested as `--orient 95 65 20`
- alpha is averaged analytically/in far field by the code as `alpha_avg=95`, so only `65*20=1300` BEM solves are done
- valid production solver here is dense PMCHWT, not the experimental `muller2`

Stable large-size mesh:
- `meshes/shapeafine_res_f4200_ag8.obj`
- 4200 triangles, 6297 RWG
- generated from `runs/greek_mesh_candidates/ShapeAfineRes_decim4000.obj` using trimesh quadric decimation, `face_count=4200`, `aggression=8`
- mesh quality check at generation: min angle about `1.7 deg`, 1st percentile angle about `7.85 deg`

Faster large-size mesh:
- `meshes/shapeafine_res_f3400_ag8.obj`
- 3399 triangles, 5097 RWG
- generated from the same ShapeAfineRes source, `face_count=3400`, `aggression=8`
- mesh quality check at generation: min angle about `1.7 deg`, 1st percentile angle about `8.74 deg`

Mesh-quality warning:

```bash
python3 scripts/mesh_quality.py runs/greek_larger_valid/meshes/shapeafine_res_f3400_ag8.obj \
  runs/greek_larger_valid/meshes/shapeafine_res_f4200_ag8.obj \
  runs/greek_larger_valid/meshes/shapeafine_res_f5000_ag8.obj \
  runs/greek_larger_valid/meshes/shapeafine_res_f6000_ag6.obj
```

Current output shows that these meshes are not watertight and still contain boundary
edges. The p1/p5 angle quality also worsens as faces are added by plain quadric
decimation. This matches the BEM/ADDA behavior at `A_x=20.76`: f5000 is slower but
does not recover S12/S34 enough. The next mesh attempt should prioritize watertight
edge-aware remeshing without skinny triangles, not just higher face count.

Gmsh isotropic remeshing of the repaired f4200 mesh gives the first useful
large-size improvement:

```bash
python3 scripts/gmsh_remesh.py \
  --src runs/greek_larger_valid/meshes/shapeafine_res_f4200_ag8_closed.obj \
  --out runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f3400_a35.obj \
  --faces 3400 --angle 35
```

The resulting mesh has 3980 triangles, 5970 RWG in BEM, is watertight, and has
angle p1/p5 around `37.1/43.0 deg`. It is both faster and more accurate than the
old decimated f4200 profile at `A_x=20.76`.

Dense LU speed note:

- the current remote node does not have cuSOLVER headers/libraries installed
  (`cusolverDn.h` and `libcusolver` are missing), so `CUSOLVER=1` cannot be built there
- `LAPACK=1` also cannot be linked on that node because `-llapack -lblas` are missing
- current dense runs therefore use the internal CPU LU factorization plus GPU batched
  RHS solve/far-field; the main speed lever for now is keeping RWG count as low as
  accuracy allows

Results so far:

| A_x | mesh/profile | quad | time, s | score6 vs ADDA | notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 11.89 | mc3200_z0p9955 | 4 | 184.84 | 0.84480 | previous best fast dense baseline |
| 13.06 | shapeafine1400 | 4 | 43.85 | 1.26031 | fast but inaccurate |
| 13.06 | mc3200_z0p9955 | 4 | 266.83 | 0.93681 | fast-ish baseline |
| 13.06 | mc3200 | 7 | 350.64 | 0.94366 | q7 did not improve |
| 13.06 | shapeafine_res_f4200_ag8 | 4 | 556.42 | 0.56032 | much better accuracy, about 2.1x slower than mc3200_z |
| 14.30 | mc3200_z0p9955 | 4 | 269.13 | 1.03618 | fast baseline degrades |
| 14.30 | shapeafine_res_f3400_ag8 | 4 | 339.53 | 0.50369 | faster compromise profile, much better than mc3200_z |
| 14.30 | shapeafine_res_f4200_ag8 | 4 | 430.72 | 0.36288 | best current large-size profile, about 1.6x slower |
| 15.68 | shapeafine_res_f3400_ag8 | 4 | 310.32 | 0.41750 | faster than f4200 and still accurate enough |
| 15.68 | shapeafine_res_f4200_ag8 | 4 | 584.28 | 0.51697 | accuracy remains much better than mc3200_z trend; run during GPU contention |
| 17.19 | shapeafine_res_f3400_ag8 | 4 | 307.01 | 0.96549 | too coarse here; S12/S34 degrade |
| 17.19 | shapeafine_res_f4200_ag8 | 4 | 530.58 | 0.58673 | profile still holds beyond A_x=17 |
| 18.94 | shapeafine_res_f4200_ag8 | 4 | 460.77 | 0.42819 | still stable; f3400 is not trusted past A_x=15.68 |
| 20.76 | shapeafine_res_f4200_ag8 | 4 | 528.71 | 1.04812 | too inaccurate; S12/S34 degrade |
| 20.76 | shapeafine_res_f4200_ag8_closed | 4 | 532.13 | 1.04633 | watertight repair has no meaningful accuracy effect |
| 20.76 | shapeafine_res_f5000_ag8 | 4 | 848.03 | 0.87895 | improves only slightly for much higher LU cost; not a good production tradeoff |
| 20.76 | shapeafine_res_f4200_closed_gmsh_f4200_a35 | 4 | 694.99 | 0.60253 | good quality mesh; better than f5000 but slower than needed |
| 20.76 | shapeafine_res_f4200_closed_gmsh_f3400_a35 | 4 | 475.75 | 0.35066 | new best profile: faster than old f4200 and much more accurate |
| 22.83 | shapeafine_res_f4200_closed_gmsh_f3400_a35 | 4 | 391.97 | 0.89327 | too coarse at this size; S12 degrades |
| 22.83 | shapeafine_res_f4200_closed_gmsh_f4200_a35 | 4 | 773.18 | 0.29538 | new best for this size; LU was slower on this loaded run |
| 25.09 | shapeafine_res_f4200_closed_gmsh_f4200_a35 | 4 | 717.35 | 0.37298 | still holds; S12/S34 acceptable compared with decimated meshes |
| 27.50 | shapeafine_res_f4200_closed_gmsh_f4200_a35 | 4 | 735.90 | 0.93335 | too coarse at this size; S12/S34 degrade again |
| 27.50 | shapeafine_res_f4200_closed_gmsh_f5200_a35 | 4 | 845.39 | 0.76605 | intermediate mesh is still not enough |
| 27.50 | shapeafine_res_f4200_closed_gmsh_f6000_a35 | 4 | 1157.79 | 0.49273 | accuracy recovers, but runtime is high |
| 27.50 | shapeafine_res_f4200_closed_gmsh_f6000_a45 | 4 | 1206.89 | 0.43783 | best current accuracy at this size; slightly slower than a35 |
| 30.25 | shapeafine_res_f4200_closed_gmsh_f6000_a45 | 4 | 1227.51 | 1.35528 | too coarse/insufficient at this size; S12 fails badly |
| 30.25 | shapeafine_res_f4200_closed_gmsh_f7000_a45 | 4 | 1908.06 | 0.95742 | still fails S12 and is too slow; not a useful production tradeoff |
| 30.25 | greek_adda_dpl25_mc_decim6000_ag6_merge6 | 4 | 1345.43 | 3.78269 | ADDA-voxel MC derived mesh is much worse; do not use |

Scoring note:

The original `score6` is intentionally strict: each Mueller element is normalized by
its own reference norm. At large sizes this can make weak polarization elements
(`S12`, `S34`) dominate the score even when the absolute error is small compared with
`S11`. `scripts/score_mbs.py` also reports `score6_s11w`, where every component error
is normalized by the `S11` norm. This is not a replacement for `score6`, but it avoids
mistaking a small absolute `S12` discrepancy for a large total-intensity error.

Examples:

| A_x | profile | score6 | score6_s11w | interpretation |
| ---: | --- | ---: | ---: | --- |
| 25.09 | gmsh_f4200_a35 | 0.37298 | 0.03798 | good strict score and good absolute score |
| 27.50 | gmsh_f6000_a45 | 0.43783 | 0.01400 | best current strict/absolute tradeoff at this size |
| 30.25 | gmsh_f7000_a45 | 0.95742 | 0.02678 | total-intensity-scale error is still small; strict score is dominated by weak S12 |

Alpha-averaging check:

The fast `alpha_avg` path was compared with explicit alpha orientation solves on a
small diagnostic mesh (`shapeafine_f1400_ag8`, `A_x=13.06`, `orient 4 5 3`,
`ntheta=31`). The maximum Mueller-matrix difference was `1.3e-15`
(`rel_fro=1.3e-17`). This confirms that the alpha far-field acceleration is not the
source of the large-size weak-component discrepancy; keep it enabled for production
runs.

Recommended production profile:

- for `A_x <= 15.68`: use `shapeafine_res_f3400_ag8.obj` when speed matters; it is the best current speed/accuracy compromise on the larger Greek particle runs
- for `17.19 <= A_x <= 18.94`: use `shapeafine_res_f4200_ag8.obj`; f3400 already degrades at `A_x=17.19`
- at `A_x=20.76`, switch to `shapeafine_res_f4200_closed_gmsh_f3400_a35.obj`. It cuts score6 from `1.04812` to `0.35066` and reduces runtime from `528.71s` to `475.75s`. This confirms that mesh quality dominates plain face count for the larger Greek particle runs.
- at `A_x=22.83`, use `shapeafine_res_f4200_closed_gmsh_f4200_a35.obj`; the faster gmsh_f3400 profile is already too coarse in S12.
- at `A_x=25.09`, the same gmsh_f4200 profile still holds with score6 `0.37298`; continue testing larger database sizes with this mesh before increasing face count.
- at `A_x=27.5`, gmsh_f4200 is no longer enough, and gmsh_f5200 is still too coarse. gmsh_f6000 is required; `a45` improves score6 to `0.43783` versus `0.49273` for `a35`, at a small extra runtime cost.
- at `A_x=30.25`, even gmsh_f6000_a45 is no longer enough (`score6=1.35528`, dominated by S12). Increasing to gmsh_f7000_a45 is still not enough (`score6=0.95742`) and costs `1908.06s`, so this is the current validated limit of the dense production path before a stronger mesh strategy or a faster validated iterative/FMM path is needed.
- a direct ADDA-voxel marching-cubes decimation path was tested at `A_x=30.25`
  (`greek_adda_dpl25_mc_decim6000_ag6_merge6`, watertight after vertex-merge repair).
  It is much worse (`score6=3.78269`, dominated by S12), so matching the voxelized
  surface directly is not the right fix for the large-size BEM discrepancy.
- for exact ADDA comparison, only use A_x values that exist in the ADDA database. For example, `2*11.89 = 23.78` is not present there; the nearest database sizes are `22.83` and `25.09`, so they are not an exact x2 validation.

Stable large-size command:

```bash
./bin/bem_cuda_fmm --solver dense --system pmchwt \
  --obj runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f3400_a35.obj \
  --ka <A_x> --ri 1.6 0.002 --quad 4 \
  --orient 95 65 20 --ntheta 181 \
  --out runs/greek_larger_valid/bem_shapeafine_res_f4200_closed_gmsh_f3400_a35_Ax<A_x>_a95b65g20_q4_n181.json
```

Faster command for the range where f3400 is still validated:

```bash
./bin/bem_cuda_fmm --solver dense --system pmchwt \
  --obj runs/greek_larger_valid/meshes/shapeafine_res_f3400_ag8.obj \
  --ka <A_x> --ri 1.6 0.002 --quad 4 \
  --orient 95 65 20 --ntheta 181 \
  --out runs/greek_larger_valid/bem_shapeafine_res_f3400_ag8_Ax<A_x>_a95b65g20_q4_n181.json
```

Do not use the previous `muller2` results for ADDA comparison yet. They converge faster in some variants, but the Mueller matrix is not validated against dense PMCHWT/ADDA.
