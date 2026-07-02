#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/BEM-CUDA}"
NVIDIA_SMI="${BEM_NVIDIA_SMI:-nvidia-smi}"
ALLOW_COMPUTE_SHARE="${BEM_ALLOW_COMPUTE_SHARE:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/gpu_guard.sh"
cd "$ROOT"
mkdir -p runs/sphere30_ref6_rerun/logs

run_case() {
  local gpu="$1"
  local name="$2"
  shift 2
  bem_require_gpu_free "$gpu" "$ALLOW_COMPUTE_SHARE" || return $?
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}"
    export BEM_GMRES_STAGNATION_CYCLES="${BEM_GMRES_STAGNATION_CYCLES:-0}"
    export BEM_GMRES_STAGNATION_REL="${BEM_GMRES_STAGNATION_REL:-0.003}"
    echo "START $name gpu=$gpu $(date -Is)"
    ./bin/bem_cuda_fmm \
      --shape sphere \
      --ka 30 \
      --ri 1.3116 0 \
      --single \
      --solver fmm \
      --ntheta 181 \
      --out "runs/sphere30_ref6_rerun/${name}.json" \
      "$@" \
      > "runs/sphere30_ref6_rerun/logs/${name}.log" 2>&1
    echo "DONE $name $(date -Is)"
  ) > "runs/sphere30_ref6_rerun/logs/${name}.driver.log" 2>&1 &
}

run_case 0 sphere_ka30_ref6_q7_d7_tol3e3 "--ref" "6" "--quad" "7" "--fmm-digits" "7" "--gmres-tol" "3e-3" "--gmres-restart" "220" "--max-leaf" "128"
run_case 1 sphere_ka30_ref6_q7_d7_tol1e3 "--ref" "6" "--quad" "7" "--fmm-digits" "7" "--gmres-tol" "1e-3" "--gmres-restart" "300" "--max-leaf" "128"
run_case 2 sphere_ka30_ref6_q13_d7_tol3e3_leaf256 "--ref" "6" "--quad" "13" "--fmm-digits" "7" "--gmres-tol" "3e-3" "--gmres-restart" "220" "--max-leaf" "256"

wait

run_case 1 sphere_ka30_ref5_q13_d7_tol1e3 "--ref" "5" "--quad" "13" "--fmm-digits" "7" "--gmres-tol" "1e-3" "--gmres-restart" "260" "--max-leaf" "128"

wait
echo "ALL_DONE $(date -Is)"
