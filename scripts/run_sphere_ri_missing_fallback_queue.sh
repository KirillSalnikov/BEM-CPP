#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$HOME/BEM-CUDA}"
BIN="${BIN:-$ROOT/bin/bem_cuda_fmm}"
STRICT_OUT="${STRICT_OUT:-$ROOT/runs/sphere_ri_sweep_20260622}"
OUT="${OUT:-$ROOT/runs/sphere_ri_sweep_fallback_20260622}"
GPUS_CSV="${GPUS_CSV:-0,1,2}"
POLL_S="${POLL_S:-120}"
MEM_MAX_MB="${MEM_MAX_MB:-1400}"
UTIL_MAX="${UTIL_MAX:-20}"
GMRES_MAX_CYCLES="${GMRES_MAX_CYCLES:-80}"
GMRES_STAGNATION_CYCLES="${GMRES_STAGNATION_CYCLES:-4}"
GMRES_STAGNATION_REL="${GMRES_STAGNATION_REL:-0.005}"
NVIDIA_SMI="${BEM_NVIDIA_SMI:-nvidia-smi}"
ALLOW_COMPUTE_SHARE="${BEM_ALLOW_COMPUTE_SHARE:-0}"
source "$SCRIPT_DIR/gpu_guard.sh"
mkdir -p "$OUT/logs"

jobs_file="$OUT/missing_jobs.tsv"
cat > "$jobs_file" <<'JOBS'
ka5_n4p5_ref4_tol5e2	ka5_n4p5_ref4	5	4.5	4	4	7	5e-2	100
ka5_n6_ref4_tol5e2	ka5_n6_ref4	5	6.0	4	4	7	5e-2	100
ka10_n3_ref4_tol5e2	ka10_n3_ref4	10	3.0	4	4	7	5e-2	100
JOBS

free_gpu() {
  local allowed=",$GPUS_CSV,"
  local candidates=() gpu
  mapfile -t candidates < <(
  "$NVIDIA_SMI" --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits |
    awk -F, -v allowed="$allowed" -v mem="$MEM_MAX_MB" -v util="$UTIL_MAX" '
      {
        g=$1; u=$2; m=$3;
        gsub(/^[ \t]+|[ \t]+$/, "", g);
        gsub(/^[ \t]+|[ \t]+$/, "", u);
        gsub(/^[ \t]+|[ \t]+$/, "", m);
        if (index(allowed, "," g ",") && (u + 0) <= util && (m + 0) <= mem) {
          print g;
        }
      }'
  )
  for gpu in "${candidates[@]}"; do
    [[ -n "$gpu" ]] || continue
    if ! bem_require_gpu_free "$gpu" "$ALLOW_COMPUTE_SHARE" 2>> "$OUT/fallback_queue.status"; then
      continue
    fi
    printf '%s\n' "$gpu"
    return 0
  done
}

wait_for_gpu() {
  local gpu
  while true; do
    gpu=$(free_gpu)
    if [[ -n "$gpu" ]]; then
      echo "$gpu"
      return 0
    fi
    {
      date '+%F %T waiting for one free GPU'
      "$NVIDIA_SMI" --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    } >> "$OUT/fallback_queue.status"
    sleep "$POLL_S"
  done
}

json_is_running() {
  local json="$1"
  pgrep -f -- "--out $json" >/dev/null 2>&1
}

while read -r name strict_name ka n ref digits quad tol restart; do
  strict_json="$STRICT_OUT/${strict_name}.json"
  json="$OUT/${name}.json"
  log="$OUT/logs/${name}.log"
  rcfile="$OUT/logs/${name}.rc"
  if [[ -s "$strict_json" ]]; then
    echo "$(date '+%F %T') skip $name: strict result exists" | tee -a "$OUT/fallback_queue.status"
    continue
  fi
  if [[ -s "$json" ]]; then
    echo "$(date '+%F %T') skip $name: fallback result exists" | tee -a "$OUT/fallback_queue.status"
    continue
  fi
  if json_is_running "$json"; then
    echo "$(date '+%F %T') skip $name: matching fallback process already running" | tee -a "$OUT/fallback_queue.status"
    continue
  fi
  gpu=$(wait_for_gpu)
  if [[ -s "$strict_json" ]]; then
    echo "$(date '+%F %T') skip $name after wait: strict result exists" | tee -a "$OUT/fallback_queue.status"
    continue
  fi
  if [[ -s "$json" ]]; then
    echo "$(date '+%F %T') skip $name after wait: fallback result exists" | tee -a "$OUT/fallback_queue.status"
    continue
  fi
  if json_is_running "$json"; then
    echo "$(date '+%F %T') skip $name after wait: matching fallback process already running" | tee -a "$OUT/fallback_queue.status"
    continue
  fi
  echo "$(date '+%F %T') run $name on gpu=$gpu ka=$ka n=$n tol=$tol" | tee -a "$OUT/fallback_queue.status"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}" \
    BEM_FMM_BATCH4=1 BEM_FMM_STORE_Z=1 \
    BEM_GMRES_MAX_CYCLES="$GMRES_MAX_CYCLES" \
    BEM_GMRES_STAGNATION_CYCLES="$GMRES_STAGNATION_CYCLES" \
    BEM_GMRES_STAGNATION_REL="$GMRES_STAGNATION_REL" \
    "$BIN" --shape sphere --ka "$ka" --ref "$ref" --ri "$n" 0 \
    --single --ntheta 181 --solver fmm --fmm-digits "$digits" \
    --gmres-tol "$tol" --gmres-restart "$restart" --max-leaf 128 --quad "$quad" \
    --out "$json" > "$log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$rcfile"
  echo "$(date '+%F %T') done $name rc=$rc" | tee -a "$OUT/fallback_queue.status"
done < "$jobs_file"

echo "$(date '+%F %T') fallback queue done" | tee -a "$OUT/fallback_queue.status"
