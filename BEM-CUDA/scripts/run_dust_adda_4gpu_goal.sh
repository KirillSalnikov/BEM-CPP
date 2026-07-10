#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/kirill_epyc/BEM-CUDA}
RUN_ROOT=${RUN_ROOT:-runs/dust_adda_ka30plus_adaptive_20260710}
MANIFEST=${MANIFEST:-$RUN_ROOT/nested_bg_J2_J5_alpha256/nested_bg_manifest.json}
MESH=${MESH:-runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f7000_a45.obj}
RI_IM=${RI_IM:-0}
MAX_LEAF=${MAX_LEAF:-128}
NTHETA=${NTHETA:-1801}
GPU=${GPU:?set GPU to one physical GPU index}
KA=${KA:?set KA to an ADDA database size}
ADDA=${ADDA:?set ADDA to the matching reference table}

tag=${KA//./p}
case_dir="$RUN_ROOT/refr_1_6__${RI_IM//./_}/ka$tag"
mkdir -p "$case_dir"

if [[ -s "$case_dir/adaptive_nested_bg_manifest.json" ]]; then
  status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' \
    "$case_dir/adaptive_nested_bg_manifest.json")
  if [[ "$status" == complete ]]; then
    echo "already converged: $case_dir"
    exit 0
  fi
fi

export BEM_ORIENT_KEEP_CHUNK_SIZE=1
export BEM_ORIENT_PROGRESS=1
export BEM_GMRES_VERBOSE=1
export BEM_ALLOW_LOOSE_OBJ_GMRES=1
export BEM_GMRES_DEVICE=1
export BEM_FAST_REORTH_OFF=1

python3 scripts/adaptive_nested_bg_orient_queue.py \
  --nested-manifest "$MANIFEST" \
  --out-dir "$case_dir" \
  --gpus "$GPU" \
  --chunk-size 64 \
  --chunk-order spread \
  --omp-threads 8 \
  --alpha-avg 256 \
  --orient-warm-start zero \
  --tol 0.01 \
  --max-tol 0.03 \
  --scale-tol 0.01 \
  --component-floor 1e-4 \
  --min-levels 3 \
  -- \
  --ka "$KA" \
  --ri 1.6 "$RI_IM" \
  --shape obj \
  --obj "$MESH" \
  --subdiv 0 \
  --ntheta "$NTHETA" \
  --quad 7 \
  --solver fmm \
  --system balanced \
  --fmm-digits 7 \
  --gmres-tol 1e-3 \
  --gmres-restart 200 \
  --krylov gpu-gmres \
  --max-leaf "$MAX_LEAF" \
  --no-prec \
  --accurate

accepted=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("accepted", ""))' \
  "$case_dir/adaptive_nested_bg_manifest.json")
if [[ -n "$accepted" && -s "$accepted" ]]; then
  python3 scripts/summarize_bem_adda_m11.py --bem "$accepted" --adda "$ADDA" \
    > "$case_dir/m11_vs_adda.txt"
fi
