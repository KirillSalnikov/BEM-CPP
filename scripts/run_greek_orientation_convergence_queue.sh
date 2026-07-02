#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT=${RUN_ROOT:-runs/greek_orient_convergence_Ax58p81_20260622}
OBJ=${OBJ:-runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f8000_a45.obj}
KA=${KA:-58.81}
RI_RE=${RI_RE:-1.6}
RI_IM=${RI_IM:-0.002}
NT_GPU=${NT_GPU:-4}
GPUS_CSV=${GPUS_CSV:-0,1,2}
MEM_FREE_MB=${MEM_FREE_MB:-1200}
UTIL_MAX=${UTIL_MAX:-15}
POLL_S=${POLL_S:-120}
CHUNK_SIZE=${CHUNK_SIZE:-32}
NVIDIA_SMI="${BEM_NVIDIA_SMI:-nvidia-smi}"
ALLOW_COMPUTE_SHARE="${BEM_ALLOW_COMPUTE_SHARE:-0}"
source "$SCRIPT_DIR/gpu_guard.sh"

mkdir -p "$RUN_ROOT"/logs

cat > "$RUN_ROOT/manifest.txt" <<EOF
target=greek dust particle, largest ADDA database point
adda_reference=/home/user/BEM-CPP/greek/ADDA_for_PO_comparison/refr_1_6__0_002/A_x=58.81_refr_1_6__0_002.dat
ka=$KA
ri=$RI_RE+$RI_IM i
obj=$OBJ
system=balanced
solver=fmm
accurate_obj_profile=1
quad=7
fmm_digits=6
gmres_tol=5e-4
gmres_restart=500
max_leaf=128
ntheta=181
chunk_size=$CHUNK_SIZE
levels: name alpha_avg beta gamma
coarse_a60_b15_g8 60 15 8
mid_a120_b31_g12 120 31 12
midfine_a240_b45_g16 240 45 16
fine_a360_b65_g20 360 65 20
full_a600_b90_g45 600 90 45
EOF

free_gpus() {
  local candidates=() gpu
  mapfile -t candidates < <(
  "$NVIDIA_SMI" --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits |
    awk -F, -v allowed=",$GPUS_CSV," -v mem="$MEM_FREE_MB" -v util="$UTIL_MAX" '
      {
        g=$1; u=$2; m=$3;
        gsub(/^[ \t]+|[ \t]+$/, "", g);
        gsub(/^[ \t]+|[ \t]+$/, "", u);
        gsub(/^[ \t]+|[ \t]+$/, "", m);
        if (index(allowed, "," g ",") && (u + 0) <= util && (m + 0) <= mem) print g;
      }'
  )
  for gpu in "${candidates[@]}"; do
    [[ -n "$gpu" ]] || continue
    if ! bem_require_gpu_free "$gpu" "$ALLOW_COMPUTE_SHARE" 2>> "$RUN_ROOT/queue.status"; then
      continue
    fi
    printf '%s\n' "$gpu"
  done | head -n "$NT_GPU" | paste -sd, -
}

wait_for_gpus() {
  local gpus
  while true; do
    gpus=$(free_gpus)
    local count=0
    if [[ -n "$gpus" ]]; then
      count=$(awk -F, '{print NF}' <<< "$gpus")
    fi
    if (( count >= NT_GPU )); then
      echo "$gpus"
      return 0
    fi
    date '+%F %T waiting for free GPUs' | tee -a "$RUN_ROOT/queue.status"
    "$NVIDIA_SMI" --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits |
      tee -a "$RUN_ROOT/queue.status"
    sleep "$POLL_S"
  done
}

run_level() {
  local name=$1
  local alpha=$2
  local beta=$3
  local gamma=$4
  local out="$RUN_ROOT/${name}.json"
  local work="$RUN_ROOT/${name}_parts"
  local log="$RUN_ROOT/logs/${name}.driver.log"

  if [[ -s "$out" ]]; then
    echo "$(date '+%F %T') skip existing $out" | tee -a "$RUN_ROOT/queue.status"
    return 0
  fi

  local gpus
  gpus=$(wait_for_gpus)
  echo "$(date '+%F %T') start $name on GPUs $gpus" | tee -a "$RUN_ROOT/queue.status"
  BEM_ORIENT_PROGRESS=50 python3 run_orient_queue.py \
    --exe ./bin/bem_cuda_fmm \
    --out "$out" \
    --work-dir "$work" \
    --gpus "$gpus" \
    --chunk-size "$CHUNK_SIZE" \
    --omp-threads 8 \
    --shape obj \
    --obj "$OBJ" \
    --subdiv 0 \
    --ka "$KA" \
    --ri "$RI_RE" "$RI_IM" \
    --orient 1 "$beta" "$gamma" \
    --alpha-avg "$alpha" \
    --ntheta 181 \
    --scat-plane yz \
    --solver fmm \
    --accurate \
    --system balanced \
    --quad 7 \
    --fmm-digits 6 \
    --gmres-tol 5e-4 \
    --gmres-restart 500 \
    --max-leaf 128 \
    --no-prec \
    > "$log" 2>&1
  echo "$(date '+%F %T') done $name" | tee -a "$RUN_ROOT/queue.status"
}

run_level coarse_a60_b15_g8 60 15 8
run_level mid_a120_b31_g12 120 31 12
run_level midfine_a240_b45_g16 240 45 16
run_level fine_a360_b65_g20 360 65 20
run_level full_a600_b90_g45 600 90 45

echo "$(date '+%F %T') all levels done" | tee -a "$RUN_ROOT/queue.status"
