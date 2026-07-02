#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO:-$(cd "$script_dir/.." && pwd)}"
out="${OUT:-runs/production_matrix_15_complexop}"
bin="${BIN:-$repo/bin/bem_cuda_fmm.next}"
gpu_list="${BEM_COMPLEXOP_GPUS:-0,1}"
power_limit="${BEM_COMPLEXOP_POWER_LIMIT_W:-200}"
monitor_interval="${BEM_COMPLEXOP_MONITOR_INTERVAL_S:-5}"
nvidia_smi="${BEM_NVIDIA_SMI:-nvidia-smi}"
allow_compute_share="${BEM_ALLOW_COMPUTE_SHARE:-0}"
NVIDIA_SMI="$nvidia_smi"
source "$script_dir/gpu_guard.sh"

mkdir -p "$repo/$out/logs"
cd "$repo"

if [[ "${BEM_ALLOW_LEGACY_DUST:-0}" != "1" ]]; then
  echo "legacy complex-operator dust refresh is disabled by default" >&2
  echo "Use run_accuracy_matrix_15_queue.sh for q7_d6_tol5e4 dust production runs." >&2
  echo "Set BEM_ALLOW_LEGACY_DUST=1 only to reproduce archived q7_d5_tol1e3 refreshes." >&2
  exit 2
fi

trim_number() {
  local value="$1"
  value="${value// /}"
  value="${value%%.*}"
  printf '%s\n' "$value"
}

set_power_limit() {
  local gpu="$1"
  if [[ -n "$power_limit" ]]; then
    "$nvidia_smi" -i "$gpu" -pl "$power_limit" >/dev/null 2>&1 || true
  fi
}

gpu_sample() {
  local gpu="$1" line temp util mem power
  line="$("$nvidia_smi" -i "$gpu" --query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits)"
  IFS=',' read -r temp util mem power <<<"$line"
  temp="$(trim_number "$temp")"
  util="$(trim_number "$util")"
  mem="$(trim_number "$mem")"
  power="$(trim_number "$power")"
  printf '%s,%s,%s,%s\n' "$temp" "$util" "$mem" "$power"
}

metadata_args() {
  printf '%s\n' --strict --require-converged --validate-numeric --require-complex-operator
  if [[ "${BEM_METADATA_SKIP_CLOUDE:-0}" != "1" ]]; then
    printf '%s\n' --require-cloude-physical
  fi
}

result_is_complex_current() {
  local name="$1"
  local -a meta_args
  [[ -s "$out/$name.json" ]] || return 1
  mapfile -t meta_args < <(metadata_args)
  python3 scripts/check_result_metadata.py "${meta_args[@]}" "$out/$name.json" >/dev/null 2>&1
}

run_with_monitor() {
  local gpu="$1" name="$2"
  shift 2
  local monitor="$out/logs/$name.gpu.csv"
  local pid rc ts sample temp util mem power
  printf 'timestamp_s,gpu,temp_c,util_pct,mem_mib,power_w\n' > "$monitor"
  set_power_limit "$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}" "$@" &
  pid="$!"
  while kill -0 "$pid" 2>/dev/null; do
    ts="$(date +%s)"
    if sample="$(gpu_sample "$gpu" 2>> "$out/logs/$name.log")"; then
      IFS=',' read -r temp util mem power <<<"$sample"
      printf '%s,%s,%s,%s,%s,%s\n' "$ts" "$gpu" "$temp" "$util" "$mem" "$power" >> "$monitor"
    fi
    sleep "$monitor_interval"
  done
  set +e
  wait "$pid"
  rc="$?"
  set -e
  return "$rc"
}

run_case() {
  local gpu="$1" name="$2"
  shift 2
  if result_is_complex_current "$name"; then
    echo "SKIP $name complex-operator-current"
    return 0
  fi
  echo "START $name gpu=$gpu bin=$bin"
  local rc
  set +e
  (
    export BEM_GMRES_VERBOSE="${BEM_GMRES_VERBOSE:-1}"
    run_with_monitor "$gpu" "$name" "$bin" "$@" --out "$out/$name.json"
  ) > "$out/logs/$name.log" 2>&1
  rc="$?"
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "FAIL $name rc=$rc"
    return "$rc"
  fi
  local -a meta_args
  mapfile -t meta_args < <(metadata_args)
  python3 scripts/check_result_metadata.py "${meta_args[@]}" "$out/$name.json"
  echo "DONE $name"
}

worker() {
  local gpu="$1"
  shift
  bem_require_gpu_free "$gpu" "$allow_compute_share"
  local mesh4200="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f4200_a35.obj"
  local mesh5200="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f5200_a35.obj"
  local mesh6000="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f6000_a45.obj"
  local mesh7000="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f7000_a45.obj"
  local common=(--ri 1.6 0.002 --single --ntheta 181 --solver fmm --system balanced
                --quad 7 --fmm-digits 5 --gmres-tol 1e-3 --gmres-restart 220
                --max-leaf 96 --no-prec)
  local tag
  for tag in "$@"; do
    case "$tag" in
      ka5)
        run_case "$gpu" dust_ka5_gmsh4200_complexop_balanced_q7_d5_tol1e3 \
          --obj "$mesh4200" --ka 5 "${common[@]}"
        ;;
      ka10)
        run_case "$gpu" dust_ka10_gmsh5200_complexop_balanced_q7_d5_tol1e3 \
          --obj "$mesh5200" --ka 10 "${common[@]}"
        ;;
      ka15)
        run_case "$gpu" dust_ka15_gmsh6000_complexop_balanced_q7_d5_tol1e3 \
          --obj "$mesh6000" --ka 15 "${common[@]}"
        ;;
      ka20)
        run_case "$gpu" dust_ka20_gmsh4200_complexop_balanced_q7_d5_tol1e3 \
          --obj "$mesh4200" --ka 20 "${common[@]}"
        ;;
      ka30)
        run_case "$gpu" dust_ka30_gmsh7000_complexop_balanced_q7_d5_tol1e3 \
          --obj "$mesh7000" --ka 30 "${common[@]}"
        ;;
      *)
        echo "Unknown case tag: $tag" >&2
        return 2
        ;;
    esac
  done
}

[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 2; }

IFS=',' read -r -a gpus <<<"$gpu_list"
if [[ "${#gpus[@]}" -eq 1 ]]; then
  worker "${gpus[0]}" ka5 ka10 ka15 ka20 ka30
else
  pids=()
  worker "${gpus[0]}" ka5 ka15 ka30 &
  pids+=("$!")
  worker "${gpus[1]}" ka10 ka20 &
  pids+=("$!")
  rc=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  exit "$rc"
fi
