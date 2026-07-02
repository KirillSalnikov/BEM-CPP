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
mkdir -p runs/poster_true_residual_refresh_20260630

rsync -av \
  "$remote:$remote_dir/runs/poster_true_residual_refresh_20260630/" \
  runs/poster_true_residual_refresh_20260630/

bash scripts/rebuild_poster_refresh_copy.sh

echo
echo "Fetched remote results and rebuilt poster copy:"
echo "  $root/poster_a0_work_refresh/poster_a0.pdf"
echo "  $root/poster_a0_work_refresh/assets/table_accuracy_matrix_15.csv"
