#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

poster_dir="${POSTER_REFRESH_DIR:-poster_a0_work_refresh}"
results_dir="${POSTER_REFRESH_RESULTS:-runs/poster_true_residual_refresh_20260630}"

if [[ ! -d "$poster_dir" ]]; then
  echo "missing poster refresh copy: $poster_dir" >&2
  exit 2
fi

echo "== result files =="
find "$results_dir" -maxdepth 1 -type f -name '*.json' -printf '%f\n' 2>/dev/null | sort || true

echo "== metadata check =="
if compgen -G "$results_dir/*.json" >/dev/null; then
  for json in "$results_dir"/*.json; do
    args=(--strict --require-converged --validate-numeric)
    if [[ "$(basename "$json")" == dust_* ]]; then
      args+=(--require-complex-operator)
    fi
    if python3 scripts/check_result_metadata.py "${args[@]}" "$json" >/tmp/poster_refresh_meta.$$ 2>&1; then
      echo "OK $(basename "$json")"
    else
      echo "BAD $(basename "$json")"
      sed 's/^/  /' /tmp/poster_refresh_meta.$$
    fi
  done
  rm -f /tmp/poster_refresh_meta.$$
else
  echo "no JSON results yet in $results_dir"
fi

echo "== rebuild assets =="
echo "== refresh accuracy matrix =="
if python3 scripts/audit_accuracy_matrix_15.py \
    --out "$poster_dir/assets/table_accuracy_matrix_15.csv" \
    --require-current-metadata \
    --require-complex-operator-for-absorbing \
    > "$poster_dir/assets/table_accuracy_matrix_15.audit.log" 2>&1; then
  echo "accuracy matrix audit: all selected rows pass strict metadata gates"
else
  rc=$?
  echo "accuracy matrix audit: incomplete/failed rc=$rc; keeping CSV for plots and gap tables"
  tail -40 "$poster_dir/assets/table_accuracy_matrix_15.audit.log" || true
fi

python3 "$poster_dir/make_assets.py"

echo "== build pdf =="
(
  cd "$poster_dir"
  pdflatex -interaction=nonstopmode -halt-on-error poster_a0.tex >/tmp/poster_refresh_pdflatex1.log
  pdflatex -interaction=nonstopmode -halt-on-error poster_a0.tex >/tmp/poster_refresh_pdflatex2.log
)
tail -8 /tmp/poster_refresh_pdflatex2.log

echo "== validate poster =="
python3 "$poster_dir/validate_poster.py"

echo "== accuracy matrix =="
python3 - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("poster_a0_work_refresh/assets/table_accuracy_matrix_15.csv")
df = pd.read_csv(path)
truth = lambda s: s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "pass"})
pass5 = truth(df["pass5"]) if "pass5" in df else df["status"].eq("PASS") & (pd.to_numeric(df["gate_error"], errors="coerce") <= 0.05)
pass10 = truth(df["pass10"]) if "pass10" in df else df["status"].eq("PASS")
raw5 = truth(df["raw_pass5"]) if "raw_pass5" in df else pd.to_numeric(df["gate_error"], errors="coerce").le(0.05)
raw10 = truth(df["raw_pass10"]) if "raw_pass10" in df else pd.to_numeric(df["gate_error"], errors="coerce").le(0.10)
print(f"PASS5 {int(pass5.sum())}/{len(df)} (raw {int(raw5.sum())}/{len(df)})")
print(f"PASS10 {int(pass10.sum())}/{len(df)} (raw {int(raw10.sum())}/{len(df)})")
print(df[["shape", "ka", "status", "status_5pct", "pass5", "pass10", "metadata_status", "gate_error", "bem_file"]].to_string(index=False))
PY
