#!/usr/bin/env bash
set -euo pipefail

HOST="kirill_epyc@172.16.0.117"
REMOTE_DIR="/home/kirill_epyc/BEM-CUDA"
GPUS="0,1,3"
SIZES=(10.86 14.3 18.94 30.25 33.28)
DRY_RUN=0
WAIT=1

usage() {
  cat <<'EOF'
Usage:
  scripts/resume_dust_adda_db_adaptive_batch.sh [options]

Queues several independent dust BEM adaptive orientation averages against the
ready ADDA refr_1_6__0 database.  This does not run ADDA.

Options:
  --host HOST        SSH host, default kirill_epyc@172.16.0.117
  --remote-dir DIR   Remote BEM-CUDA dir, default /home/kirill_epyc/BEM-CUDA
  --gpus LIST        GPU list passed to each run, default 0,1,3
  --sizes "A B C"    Space-separated ka values, default 10.86 14.3 18.94 30.25 33.28
  --no-wait          Start all requested runs without waiting; use only with disjoint GPUs
  --dry-run          Print commands only
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --sizes) read -r -a SIZES <<< "$2"; shift 2 ;;
    --no-wait) WAIT=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

args=(--host "$HOST" --remote-dir "$REMOTE_DIR" --gpus "$GPUS")
if [[ "$DRY_RUN" -eq 1 ]]; then
  args+=(--dry-run)
fi

for ka in "${SIZES[@]}"; do
  echo "=== queue ka=$ka ==="
  scripts/resume_dust_adda_db_adaptive.sh "${args[@]}" --ka "$ka"
  if [[ "$DRY_RUN" -eq 0 && "$WAIT" -eq 1 ]]; then
    run_glob="runs/dust_ka10_20_bem_vs_adda_db_20260710/ka${ka//./p}_adaptive_nested_*"
    while true; do
      status=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "cd '$REMOTE_DIR' && python3 - <<PY
import json
from pathlib import Path
matches=sorted(Path('.').glob('$run_glob'))
if not matches:
    print('starting')
else:
    p=matches[-1] / 'adaptive_nested_bg_manifest.json'
    try:
        print(json.load(p.open()).get('status') or 'running')
    except Exception:
        print('running')
PY
" 2>/dev/null || echo unreachable)
      echo "ka=$ka status=$status"
      [[ "$status" == "complete" ]] && break
      sleep 300
    done
  fi
done
