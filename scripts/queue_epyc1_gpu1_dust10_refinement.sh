#!/usr/bin/env bash
set -euo pipefail

repo="${REPO:-/home/kirill_epyc/BEM-CUDA}"
cd "$repo"

out="runs/production_matrix_refinement"
gpu="${BEM_QUEUE_GPU:-1}"
mkdir -p "$out/logs"

cases=(
  dust_ka10_gmsh6000_balanced_q13_d6_tol5e4
  dust_ka5_gmsh5200_balanced_q13_d6_tol5e4
  dust_ka15_gmsh5200_balanced_q13_d6_tol5e4
)

queue_log="$out/logs/gpu${gpu}.dust_refinement_queue.log"
lock_file="$out/logs/gpu${gpu}.dust_refinement_queue.lock"

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "QUEUE_LOCKED gpu=$gpu $(date -Is)" >> "$queue_log"
  exit 0
fi

{
  echo "QUEUE_START gpu=$gpu cases=${cases[*]} $(date -Is)"
  for case_name in "${cases[@]}"; do
    json="$out/${case_name}.json"
    if [[ -s "$json" ]]; then
      echo "CASE_SKIP_EXISTING case=$case_name json=$json $(date -Is)"
      continue
    fi
    while true; do
      apps="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
      if [[ -z "$apps" ]]; then
        echo "GPU_FREE case=$case_name gpu=$gpu $(date -Is)"
        break
      fi
      echo "QUEUE_WAIT case=$case_name gpu=$gpu $(date -Is) apps=${apps//$'\n'/; }"
      sleep 300
    done

    echo "CASE_START case=$case_name gpu=$gpu $(date -Is)"
    if scripts/run_accuracy_matrix_case.sh \
        --gpu "$gpu" \
        --case "$case_name" \
        --out "$out" \
        --max-power 310 \
        --max-bad-samples 4; then
      echo "CASE_DONE case=$case_name gpu=$gpu $(date -Is)"
    else
      rc=$?
      echo "CASE_FAIL case=$case_name gpu=$gpu rc=$rc $(date -Is)"
    fi
  done
  echo "QUEUE_DONE gpu=$gpu $(date -Is)"
} >> "$queue_log" 2>&1
