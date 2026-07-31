#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-"$ROOT/bin/muller_nodal_fmm_demo_fp32"}
OUT=${OUT:-"$ROOT/runs/muller_orientation_average_ka25"}
THREADS=${THREADS:-16}
KA=${KA:-25}
RI=${RI:-1.3}
REF=${REF:-5}
ALPHA=${ALPHA:-8}
BETA=${BETA:-8}
GAMMA=${GAMMA:-4}
NTHETA=${NTHETA:-181}
SOLVER=${SOLVER:-pfft}
RECYCLE_RANK=${RECYCLE_RANK:-8}

mkdir -p "$OUT"

solver_args=(--mbj-only)
case "$SOLVER" in
  fmm)
    solver_args+=(--orient-paired-gpu-gmres)
    ;;
  pfft)
    solver_args+=(
      --pfft-fgmres
      --pfft-inner-tol 4e-2 --pfft-inner-iters auto
      --pfft-outer-restart 32
      --pfft-order 2 --pfft-correction-radius 0
      --pfft-grid-safety 1
    )
    ;;
  *)
    echo "SOLVER must be fmm or pfft" >&2
    exit 2
    ;;
esac

OMP_NUM_THREADS="$THREADS" "$BIN" \
  --shape prism --sides 6 --aspect 1 \
  --ref "$REF" --ka "$KA" --ri "$RI" \
  --edge-mode hdiv --quad 7 --duffy-order 4 \
  --digits 5 --max-leaf 64 --fmm-near-radius 3 \
  --tol 1e-5 --max-iters 500 --gmres-restart 100 \
  --mbj-nodes 50 --mbj-overlap 0 \
  --near-correction-cache "$OUT/operator.near" \
  --mbj-cache "$OUT/mbj50.cache" \
  --fmm-near-fp32 \
  "${solver_args[@]}" \
  --orient-average "$ALPHA" "$BETA" "$GAMMA" \
  --orient-symmetry-order 6 \
  --orient-warm-max-angle 25 \
  --orient-recycle-rank "$RECYCLE_RANK" \
  --ntheta "$NTHETA" \
  --no-dense-validation \
  --iteration-log "$OUT/iterations.csv" \
  --out "$OUT/average.json" \
  2>&1 | tee -a "$OUT/run.log"
