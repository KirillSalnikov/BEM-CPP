#!/usr/bin/env bash
set -euo pipefail

HOST="kirill_epyc@172.16.0.117"
REMOTE_DIR="/home/kirill_epyc/BEM-CUDA"
REMOTE_RUNS="runs/dust_ka10_20_bem_vs_adda_db_20260710"
LOCAL_RUNS="runs/dust_ka10_20_bem_vs_adda_db_20260710"

usage() {
  cat <<'EOF'
Usage:
  scripts/fetch_dust_adda_db_adaptive_results.sh [options]

Fetches adaptive dust BEM-vs-ADDA run artifacts and refreshes the local goal
audit CSV.  This fetches only compact JSON/log/report files, not large binaries.

Options:
  --host HOST          SSH host, default kirill_epyc@172.16.0.117
  --remote-dir DIR     Remote BEM-CUDA dir, default /home/kirill_epyc/BEM-CUDA
  --remote-runs DIR    Remote runs dir relative to remote-dir
  --local-runs DIR     Local runs dir
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --remote-runs) REMOTE_RUNS="$2"; shift 2 ;;
    --local-runs) LOCAL_RUNS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$LOCAL_RUNS"

rsync -av --prune-empty-dirs \
  --include='*/' \
  --include='adaptive_nested_bg_manifest.json' \
  --include='m11_vs_adda_db_accepted.txt' \
  --include='adaptive.nohup' \
  --include='watch_compare.nohup' \
  --include='level*/bem.json' \
  --include='level*/_queue_weighted_sum.json' \
  --include='parts/part_*.json' \
  --include='parts/group_*.json' \
  --include='parts/*.log' \
  --exclude='*' \
  "$HOST:$REMOTE_DIR/$REMOTE_RUNS/" "$LOCAL_RUNS/"

python3 scripts/report_dust_adda_db_adaptive_goal.py \
  --runs-dir "$LOCAL_RUNS" \
  --csv "$LOCAL_RUNS/adaptive_goal_audit.csv"
