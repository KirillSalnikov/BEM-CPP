#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/BEM-CUDA}"
ADDA_OCL="${ADDA_OCL:-$HOME/adda/src/ocl/adda_ocl}"
OUT="${OUT:-$ROOT/runs/adda_ocl_sphere_ri_sweep_20260623}"
GPUS_CSV="${GPUS_CSV:-0,1}"
DPL="${DPL:-20}"
EPS="${EPS:-5}"
NTHETA="${NTHETA:-181}"
TIMEOUT_S="${TIMEOUT_S:-0}"
NVIDIA_SMI="${BEM_NVIDIA_SMI:-nvidia-smi}"
ALLOW_COMPUTE_SHARE="${BEM_ALLOW_COMPUTE_SHARE:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/gpu_guard.sh"

mkdir -p "$OUT/logs"
summary="$OUT/summary.csv"
if [[ ! -s "$summary" ]]; then
  printf 'case,shape,ka,n,dpl,ntheta,status,time_s,backend,dir,log,note\n' > "$summary"
fi

jobs_file="$OUT/jobs.tsv"
cat > "$jobs_file" <<'JOBS'
ka5_n1p5	5	1.5
ka5_n3	5	3.0
ka5_n4p5	5	4.5
ka5_n6	5	6.0
ka10_n1p5	10	1.5
ka10_n3	10	3.0
ka10_n4p5	10	4.5
ka10_n6	10	6.0
ka15_n1p5	15	1.5
ka15_n3	15	3.0
ka15_n4p5	15	4.5
ka15_n6	15	6.0
JOBS

quote_csv() {
  local s=${1//\"/\"\"}
  printf '"%s"' "$s"
}

emit_row() {
  local case_name=$1 ka=$2 n=$3 status=$4 time_s=$5 dir=$6 log=$7 note=$8
  (
    flock -x 9
    {
      quote_csv "$case_name"; printf ','
      quote_csv "sphere"; printf ','
      printf '%s,%s,%s,%s,' "$ka" "$n" "$DPL" "$NTHETA"
      quote_csv "$status"; printf ','
      printf '%s,' "$time_s"
      quote_csv "adda_ocl"; printf ','
      quote_csv "$dir"; printf ','
      quote_csv "$log"; printf ','
      quote_csv "$note"; printf '\n'
    } >> "$summary"
  ) 9>"$OUT/.summary.lock"
}

total_jobs=$(wc -l < "$jobs_file")
mapfile -t GPUS < <(bem_filter_free_gpus_csv "$GPUS_CSV" "$ALLOW_COMPUTE_SHARE")
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No free GPUs from GPUS_CSV=$GPUS_CSV" >&2
  exit 3
fi
echo "START adda_ocl_sphere_ri_sweep out=$OUT jobs=$total_jobs gpus=$GPUS_CSV dpl=$DPL ntheta=$NTHETA timeout=$TIMEOUT_S"

worker() {
  local gpu="$1"
  while true; do
    local idx line name ka n
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
      printf '%s\n' "$line" > "$OUT/.worker_${gpu}_job"
    ) 9>"$OUT/.queue.lock" || {
      rc=$?
      [[ "$rc" == 20 ]] && break
      exit "$rc"
    }
    read -r name ka n < "$OUT/.worker_${gpu}_job"
    local dir="$OUT/$name"
    local log="$dir/run.log"
    local elapsed_file="$dir/elapsed_s"
    local rcfile="$dir/exit_code"
    mkdir -p "$dir"
    if [[ -s "$dir/mueller" && -s "$elapsed_file" ]]; then
      echo "SKIP gpu=$gpu $name"
      continue
    fi
    local cmd=(
      "$ADDA_OCL"
      -gpu "$gpu"
      -dir "$dir"
      -shape sphere
      -m "$n" 0
      -dpl "$DPL"
      -eps "$EPS"
      -orient 0 0 0
      -ntheta "$NTHETA"
      -scat_matr muel
      -sym no
      -eq_rad "$ka"
    )
    printf '+ %q ' "${cmd[@]}" > "$log"
    printf '\n' >> "$log"
    echo "RUN gpu=$gpu $name ka=$ka n=$n"
    local start end rc elapsed status note
    start=$(date +%s.%N)
    set +e
    if (( TIMEOUT_S > 0 )); then
      timeout "$TIMEOUT_S" "${cmd[@]}" >> "$log" 2>&1
      rc=$?
    else
      "${cmd[@]}" >> "$log" 2>&1
      rc=$?
    fi
    set -e
    end=$(date +%s.%N)
    elapsed=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.6f", b-a}')
    echo "$elapsed" > "$elapsed_file"
    echo "$rc" > "$rcfile"
    if [[ "$rc" == 0 ]]; then
      status="measured"
      note="single orientation, Mie check in poster pipeline"
    elif [[ "$rc" == 124 ]]; then
      status="timeout"
      note="timeout ${TIMEOUT_S}s"
    else
      status="failed"
      note="exit code $rc"
    fi
    emit_row "$name" "$ka" "$n" "$status" "$elapsed" "$dir" "$log" "$note"
    echo "DONE gpu=$gpu $name rc=$rc elapsed=$elapsed"
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

echo "DONE adda_ocl_sphere_ri_sweep out=$OUT summary=$summary"
