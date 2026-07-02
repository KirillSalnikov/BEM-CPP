#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-kirill_epyc@172.16.0.73}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/kirill_epyc/BEM-CUDA}"
STRICT_DIR="${STRICT_DIR:-runs/sphere_ri_sweep_20260622}"
FALLBACK_DIR="${FALLBACK_DIR:-runs/sphere_ri_sweep_fallback_20260622}"
POLL_S="${POLL_S:-300}"
MAX_POLLS="${MAX_POLLS:-0}"

sync_results() {
  mkdir -p "$STRICT_DIR" "$FALLBACK_DIR"
  rsync -av \
    --include='*.json' --include='summary_mie.csv' --include='driver.log' \
    --include='logs/' --include='logs/*.rc' --include='logs/*.log' \
    --exclude='*' \
    "$REMOTE:$REMOTE_ROOT/$STRICT_DIR/" "$STRICT_DIR/"
  rsync -av \
    --include='*.json' --include='summary_mie.csv' --include='fallback_queue.status' \
    --include='logs/' --include='logs/*.rc' --include='logs/*.log' \
    --exclude='*' \
    "$REMOTE:$REMOTE_ROOT/$FALLBACK_DIR/" "$FALLBACK_DIR/"
}

rebuild_assets() {
  python3 scripts/summarize_sphere_ri_sweep.py "$STRICT_DIR" || true
  python3 scripts/summarize_sphere_ri_sweep.py "$FALLBACK_DIR" || true
  python3 poster_a0/make_assets.py
  (
    cd poster_a0
    pdflatex -interaction=nonstopmode poster_a0.tex >/tmp/poster_pdflatex1.log
    pdflatex -interaction=nonstopmode poster_a0.tex >/tmp/poster_pdflatex2.log
    python3 validate_poster.py
  )
}

coverage_done() {
  python3 - <<'PY'
import csv
from pathlib import Path
table_path = Path("poster_a0/assets/table_index_sweep.csv")
coverage_path = Path("poster_a0/assets/table_index_sweep_coverage.csv")
if not table_path.exists() or not coverage_path.exists():
    raise SystemExit(1)
def truthy(value):
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}
table_rows = list(csv.DictReader(table_path.open()))
coverage_rows = list(csv.DictReader(coverage_path.open()))
required = {(5.0,1.5),(5.0,3.0),(5.0,4.5),(5.0,6.0),
            (10.0,1.5),(10.0,3.0),(10.0,4.5),(10.0,6.0),
            (15.0,1.5),(15.0,3.0),(15.0,4.5),(15.0,6.0)}
full_pass = {
    (float(r["ka"]), float(r["n"]))
    for r in table_rows
    if truthy(r.get("pass10_full_mueller", ""))
}
coverage_pass = {
    (float(r["ka"]), float(r["n"]))
    for r in coverage_rows
    if truthy(r.get("any_pass", "")) or r.get("status") == "PASS"
}
bad_coverage = sorted((coverage_pass & required) - full_pass)
if bad_coverage:
    print("BAD_COVERAGE", ",".join(f"ka={ka:g}:n={n:g}" for ka,n in bad_coverage))
    raise SystemExit(2)
got = full_pass & required
missing = sorted(required - got)
if missing:
    print("PENDING", ",".join(f"ka={ka:g}:n={n:g}" for ka,n in missing))
    raise SystemExit(1)
print("PASS all required sphere RI points with full-Mueller evidence")
PY
}

poll=0
while true; do
  poll=$((poll + 1))
  echo "$(date '+%F %T') sync sphere RI results, poll=$poll"
  sync_results
  rebuild_assets
  if coverage_done; then
    exit 0
  fi
  if (( MAX_POLLS > 0 && poll >= MAX_POLLS )); then
    exit 2
  fi
  sleep "$POLL_S"
done
