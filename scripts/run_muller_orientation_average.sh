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

mkdir -p "$OUT"

OMP_NUM_THREADS="$THREADS" "$BIN" \
  --shape prism --sides 6 --aspect 1 \
  --ref "$REF" --ka "$KA" --ri "$RI" \
  --edge-mode hdiv --quad 7 --duffy-order 4 \
  --digits 5 --max-leaf 64 --fmm-near-radius 3 \
  --tol 1e-5 --max-iters 500 --gmres-restart 100 \
  --mbj-only --mbj-nodes 50 --mbj-overlap 0 \
  --near-correction-cache "$OUT/operator.near" \
  --mbj-cache "$OUT/mbj50.cache" \
  --fmm-near-fp32 \
  --pfft-fgmres \
  --pfft-inner-tol 4e-2 --pfft-inner-iters auto \
  --pfft-outer-restart 32 \
  --pfft-order 2 --pfft-correction-radius 0 \
  --pfft-grid-safety 1 \
  --orient-average "$ALPHA" "$BETA" "$GAMMA" \
  --orient-symmetry-order 6 \
  --orient-warm-max-angle 25 \
  --ntheta "$NTHETA" \
  --no-dense-validation \
  --iteration-log "$OUT/iterations.csv" \
  --out "$OUT/average.json" \
  2>&1 | tee -a "$OUT/run.log"
