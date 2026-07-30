#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${OUT:-"$ROOT/runs/goal_ka25_ref5/mirror_hdiv"}
MODE=${MODE:-optimized}
BIN=${BIN:-"$ROOT/bin/muller_nodal_fmm_demo_fp32"}
THREADS=${THREADS:-16}

mkdir -p "$OUT"

case "$MODE" in
  baseline)
    case_dir="$OUT"
    stem=current_fmm_fp64_independent
    mode_args=(
      --fmm-near-fp64
      --prism-azimuth-deg 15 --mirror-symmetric-mesh
    )
    ;;
  optimized)
    case_dir="$OUT"
    stem=optimized_batch3_inner004_mirror
    mode_args=(
      --fmm-near-fp32
      --mirror-polarization
      --pfft-fgmres
      --pfft-inner-tol 4e-2 --pfft-inner-iters auto
      --pfft-outer-restart 32
      --pfft-order 2 --pfft-correction-radius 0
      --pfft-grid-safety 1
    )
    ;;
  optimized-cold)
    case_dir="$OUT/cold_optimized"
    stem=optimized_batch3_inner004_mirror
    mode_args=(
      --fmm-near-fp32
      --mirror-polarization
      --pfft-fgmres
      --pfft-inner-tol 4e-2 --pfft-inner-iters auto
      --pfft-outer-restart 32
      --pfft-order 2 --pfft-correction-radius 0
      --pfft-grid-safety 1
    )
    ;;
  *)
    printf 'MODE must be baseline, optimized, or optimized-cold\n' >&2
    exit 2
    ;;
esac

mkdir -p "$case_dir"

common=(
  --shape prism --sides 6 --aspect 1
  --ref 5 --ka 25 --ri 1.3
  --edge-mode hdiv --quad 7 --duffy-order 4
  --digits 5 --max-leaf 64 --fmm-near-radius 3
  --tol 1e-5 --max-iters 500 --gmres-restart 100
  --mbj-only --mbj-nodes 50 --mbj-overlap 0
  --near-correction-cache "$case_dir/ka25_ref5.near"
  --mbj-cache "$case_dir/ka25_ref5_mbj50.cache"
  --physical-check --ntheta 73 --no-dense-validation
)

OMP_NUM_THREADS="$THREADS" /usr/bin/time -v \
  -o "$case_dir/$stem.time" \
  "$BIN" \
  "${common[@]}" \
  "${mode_args[@]}" \
  --iteration-log "$case_dir/$stem.iterations.csv" \
  --out "$case_dir/$stem.json" \
  >"$case_dir/$stem.stdout.log" 2>&1
