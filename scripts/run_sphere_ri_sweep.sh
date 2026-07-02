#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/BEM-CUDA}"
BIN="${BIN:-$ROOT/bin/bem_cuda_fmm}"
OUT="${OUT:-$ROOT/runs/sphere_ri_sweep_20260622}"
TIMEOUT_S="${TIMEOUT_S:-0}"
GPUS_CSV="${GPUS_CSV:-0,1,2}"
GMRES_STAGNATION_CYCLES="${GMRES_STAGNATION_CYCLES:-0}"
GMRES_STAGNATION_REL="${GMRES_STAGNATION_REL:-0.002}"
GMRES_MAX_CYCLES="${GMRES_MAX_CYCLES:-80}"
NVIDIA_SMI="${BEM_NVIDIA_SMI:-nvidia-smi}"
ALLOW_COMPUTE_SHARE="${BEM_ALLOW_COMPUTE_SHARE:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/gpu_guard.sh"
mkdir -p "$OUT/logs"

jobs_file="$OUT/jobs.tsv"
cat > "$jobs_file" <<'JOBS'
ka5_n1p5_ref4	5	1.5	4	4	7	1e-2
ka5_n3_ref4	5	3.0	4	4	7	1e-2
ka5_n4p5_ref4	5	4.5	4	4	7	1e-2
ka5_n6_ref4	5	6.0	4	4	7	1e-2
ka10_n1p5_ref4	10	1.5	4	4	7	1e-2
ka10_n3_ref4	10	3.0	4	4	7	1e-2
ka10_n4p5_ref4	10	4.5	4	4	7	1e-2
ka10_n6_ref4	10	6.0	4	4	7	1e-2
ka15_n1p5_ref4	15	1.5	4	4	7	1e-2
ka15_n3_ref4	15	3.0	4	4	7	1e-2
ka15_n4p5_ref4	15	4.5	4	4	7	1e-2
ka15_n6_ref4	15	6.0	4	4	7	1e-2
JOBS

mapfile -t GPUS < <(bem_filter_free_gpus_csv "$GPUS_CSV" "$ALLOW_COMPUTE_SHARE")
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No free GPUs from GPUS_CSV=$GPUS_CSV" >&2
  exit 3
fi
total_jobs=$(wc -l < "$jobs_file")
if (( TIMEOUT_S > 0 )); then
  printf '%s\n' "START sphere_ri_sweep out=$OUT jobs=$total_jobs gpus=$GPUS_CSV timeout=${TIMEOUT_S}s"
else
  printf '%s\n' "START sphere_ri_sweep out=$OUT jobs=$total_jobs gpus=$GPUS_CSV timeout=none"
fi

worker() {
  local gpu="$1"
  while true; do
    local line idx name ka n ref digits quad tol
    line=""
    (
      flock -x 9
      idx_file="$OUT/.next_index"
      if [[ ! -s "$idx_file" ]]; then
        echo 1 > "$idx_file"
      fi
      idx=$(cat "$idx_file")
      if (( idx > total_jobs )); then
        exit 20
      fi
      line=$(sed -n "${idx}p" "$jobs_file")
      echo $((idx + 1)) > "$idx_file"
      printf '%s' "$line" > "$OUT/.worker_${gpu}_job"
    ) 9>"$OUT/.queue.lock" || {
      rc=$?
      [[ "$rc" == 20 ]] && break
      exit "$rc"
    }
    line=$(cat "$OUT/.worker_${gpu}_job")
    read -r name ka n ref digits quad tol <<<"$line"
    local json="$OUT/${name}.json"
    local log="$OUT/logs/${name}.log"
    local rcfile="$OUT/logs/${name}.rc"
    if [[ -s "$json" ]]; then
      echo "SKIP gpu=$gpu $name"
      continue
    fi
    echo "RUN gpu=$gpu $name ka=$ka n=$n ref=$ref digits=$digits quad=$quad tol=$tol"
    cmd=(
      "$BIN" --shape sphere --ka "$ka" --ref "$ref" --ri "$n" 0
      --single --ntheta 181 --solver fmm --fmm-digits "$digits"
      --gmres-tol "$tol" --gmres-restart 150 --max-leaf 128 --quad "$quad"
      --out "$json"
    )
    set +e
    if (( TIMEOUT_S > 0 )); then
      CUDA_VISIBLE_DEVICES="$gpu" BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}" \
        BEM_FMM_BATCH4=1 BEM_FMM_STORE_Z=1 \
        BEM_GMRES_MAX_CYCLES="$GMRES_MAX_CYCLES" \
        BEM_GMRES_STAGNATION_CYCLES="$GMRES_STAGNATION_CYCLES" \
        BEM_GMRES_STAGNATION_REL="$GMRES_STAGNATION_REL" \
        timeout "$TIMEOUT_S" \
        "${cmd[@]}" > "$log" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$gpu" BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}" \
        BEM_FMM_BATCH4=1 BEM_FMM_STORE_Z=1 \
        BEM_GMRES_MAX_CYCLES="$GMRES_MAX_CYCLES" \
        BEM_GMRES_STAGNATION_CYCLES="$GMRES_STAGNATION_CYCLES" \
        BEM_GMRES_STAGNATION_REL="$GMRES_STAGNATION_REL" \
        "${cmd[@]}" > "$log" 2>&1
    fi
    rc=$?
    set -e
    echo "$rc" > "$rcfile"
    echo "DONE gpu=$gpu $name rc=$rc"
  done
}

pids=()
for gpu in "${GPUS[@]}"; do
  worker "$gpu" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "DONE sphere_ri_sweep out=$OUT"
