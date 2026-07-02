#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 USER@HOST [REMOTE_DIR]" >&2
  echo "Example: $0 kirill_epyc@172.16.1.149 /home/kirill_epyc/BEM-CUDA" >&2
  exit 2
fi

remote="$1"
remote_dir="${2:-/home/kirill_epyc/BEM-CUDA}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$root"

ssh -o BatchMode=yes -o ConnectTimeout=10 "$remote" "mkdir -p '$remote_dir'"

rsync -av --relative \
  ./Makefile \
  ./src \
  ./tests \
  ./scripts/queue_poster_true_residual_refresh.sh \
  ./scripts/audit_poster_refresh_plan.py \
  ./scripts/rebuild_poster_refresh_copy.sh \
  ./scripts/run_guarded_bem_case.sh \
  ./scripts/check_result_metadata.py \
  ./scripts/audit_accuracy_matrix_15.py \
  ./scripts/queue_status_json.py \
  ./poster_a0_work_refresh/make_assets.py \
  ./poster_a0_work_refresh/validate_poster.py \
  "$remote:$remote_dir/"

ssh -o BatchMode=yes "$remote" "cd '$remote_dir' && chmod +x scripts/queue_poster_true_residual_refresh.sh scripts/audit_poster_refresh_plan.py scripts/rebuild_poster_refresh_copy.sh scripts/run_guarded_bem_case.sh && make host-checks"

cat <<EOF
Synced poster-refresh code to $remote:$remote_dir

Remote build/run:
  cd $remote_dir
  make fmm-only
  bash scripts/queue_poster_true_residual_refresh.sh --plan
  bash scripts/queue_poster_true_residual_refresh.sh --run --max-power 200
EOF
