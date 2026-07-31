#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-$ROOT/bin/muller_nodal_fmm_demo_fp32}
OUT=${OUT:-$ROOT/runs/release_small_sphere}

if [[ ! -x "$BIN" ]]; then
  echo "Missing $BIN; run 'make muller-fp32' first." >&2
  exit 2
fi

mkdir -p "$OUT"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} "$BIN" \
  --shape sphere --ref 2 --ka 1 --ri 1.3 \
  --edge-mode smooth --quad 7 --duffy-order 4 \
  --digits 5 --max-leaf 128 --fmm-near-radius 3 \
  --tol 1e-5 --max-iters 200 --gmres-restart 100 \
  --mbj-only --mbj-nodes 50 --physical-check --ntheta 73 \
  --no-dense-validation --no-checkpoint \
  --out "$OUT/result.json" | tee "$OUT/run.log"

python3 "$ROOT/verify_mie.py" --skip-run \
  --out "$OUT/result.json" --ka 1 --ri 1.3 \
  --max-m11-l2 0.04 --max-main-floor2 0.05 | tee "$OUT/mie.log"
