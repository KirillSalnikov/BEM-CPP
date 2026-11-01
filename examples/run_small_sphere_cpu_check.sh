#!/usr/bin/env bash
# Independent CPU dense/Mie reference; this does not exercise GPU FMM/FFT.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-$ROOT/bin/muller_nodal_demo}
OUT=${OUT:-$ROOT/runs/release_small_sphere_cpu}
if [[ ! -x "$BIN" ]]; then
  echo "Missing $BIN; run 'make bin/muller_nodal_demo' first." >&2
  exit 2
fi
mkdir -p "$OUT"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} "$BIN" \
  --shape sphere --ref 2 --ka 1 --ri 1.3 0 --edge-mode smooth \
  --regular-quad 7 --duffy-order 4 --ntheta 73 \
  --out "$OUT/result.json" | tee "$OUT/run.log"
python3 "$ROOT/verify_mie.py" --skip-run --out "$OUT/result.json" \
  --ka 1 --ri 1.3 --max-m11-l2 0.04 --max-main-floor2 0.05 \
  | tee "$OUT/mie.log"
