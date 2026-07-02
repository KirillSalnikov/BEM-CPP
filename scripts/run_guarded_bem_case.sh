#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_guarded_bem_case.sh --gpu N --name NAME --out-dir DIR [guard options] -- BEM_ARGS...

Runs one BEM-CUDA case on one GPU, writes DIR/logs/NAME.log and
DIR/logs/NAME.gpu.csv, and terminates the run if measured GPU power stays
above the guard threshold for several samples.

Guard options:
  --bin PATH              BEM executable (default: bin/bem_cuda_fmm.next if present)
  --max-power W           terminate above this power (default: 260)
  --max-temp C            terminate above this temperature (default: 78)
  --max-bad-samples N     consecutive bad samples before TERM (default: 2)
  --interval SEC          monitor interval (default: 2)
  --require-complex       require complex-operator metadata after success
  --force                 archive existing output/log before running
  --allow-compute-share   Allow starting while nvidia-smi reports existing CUDA
                          compute processes on the target GPU.
EOF
}

gpu=""
name=""
out_dir=""
bin=""
max_power="${BEM_GUARD_MAX_POWER_W:-260}"
max_temp="${BEM_GUARD_MAX_TEMP_C:-78}"
max_bad="${BEM_GUARD_MAX_BAD_SAMPLES:-2}"
interval="${BEM_GUARD_INTERVAL_S:-2}"
require_complex=0
force=0
nvidia_smi="${BEM_NVIDIA_SMI:-nvidia-smi}"
allow_compute_share="${BEM_GUARD_ALLOW_COMPUTE_SHARE:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --name) name="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --bin) bin="$2"; shift 2 ;;
    --max-power) max_power="$2"; shift 2 ;;
    --max-temp) max_temp="$2"; shift 2 ;;
    --max-bad-samples) max_bad="$2"; shift 2 ;;
    --interval) interval="$2"; shift 2 ;;
    --require-complex) require_complex=1; shift ;;
    --force) force=1; shift ;;
    --allow-compute-share) allow_compute_share=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$gpu" || -z "$name" || -z "$out_dir" || $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

if [[ -z "$bin" ]]; then
  if [[ -x bin/bem_cuda_fmm.next ]]; then
    bin="bin/bem_cuda_fmm.next"
  else
    bin="bin/bem_cuda_fmm"
  fi
fi
if [[ ! -x "$bin" ]]; then
  echo "BEM executable is missing or not executable: $bin" >&2
  exit 6
fi

preflight_compute_apps() {
  local apps
  [[ "$allow_compute_share" == "1" ]] && return 0
  if ! command -v "$nvidia_smi" >/dev/null 2>&1; then
    return 0
  fi
  apps="$("$nvidia_smi" -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  apps="$(printf '%s\n' "$apps" | sed '/^[[:space:]]*$/d' | head -n 3)"
  if [[ -n "$apps" ]]; then
    apps="${apps//$'\n'/; }"
    echo "GPU_BUSY gpu=$gpu compute_apps=$apps" >&2
    return 3
  fi
}

if [[ -z "${CUDA_HOME:-}" && -x scripts/detect_cuda_toolchain.py ]]; then
  if python3 scripts/detect_cuda_toolchain.py --print-env >/tmp/bemcuda_guard_cuda_env.$$ 2>/tmp/bemcuda_guard_cuda_env.err.$$; then
    # shellcheck disable=SC1090
    source /tmp/bemcuda_guard_cuda_env.$$
    rm -f /tmp/bemcuda_guard_cuda_env.$$ /tmp/bemcuda_guard_cuda_env.err.$$
  fi
fi

preflight_compute_apps

mkdir -p "$out_dir/logs"
json="$out_dir/$name.json"
log="$out_dir/logs/$name.log"
gpu_log="$out_dir/logs/$name.gpu.csv"

if [[ -e "$json" && "$force" != "1" ]]; then
  echo "output exists, use --force to replace: $json" >&2
  exit 4
fi

if [[ "$force" == "1" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  [[ -e "$json" ]] && mv "$json" "$out_dir/$name.replaced_$stamp.json"
  [[ -e "$log" ]] && mv "$log" "$out_dir/logs/$name.replaced_$stamp.log"
  [[ -e "$gpu_log" ]] && mv "$gpu_log" "$out_dir/logs/$name.replaced_$stamp.gpu.csv"
fi

printf 'timestamp_s,gpu,temp_c,util_pct,mem_mib,power_w,bad_samples\n' > "$gpu_log"

(
  export CUDA_VISIBLE_DEVICES="$gpu"
  export BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}"
  export BEM_GMRES_VERBOSE="${BEM_GMRES_VERBOSE:-1}"
  export BEM_GMRES_STAGNATION_CYCLES="${BEM_GMRES_STAGNATION_CYCLES:-0}"
  export BEM_GMRES_STAGNATION_REL="${BEM_GMRES_STAGNATION_REL:-0.003}"
  "$bin" "$@" --out "$json" &
  pid="$!"
  bad=0
  term_sent=0
  while kill -0 "$pid" 2>/dev/null; do
    ts="$(date +%s)"
    line="$("$nvidia_smi" -i "$gpu" --query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "$line" ]]; then
      IFS=',' read -r temp util mem power <<<"$line"
      temp="${temp// /}"
      util="${util// /}"
      mem="${mem// /}"
      power="${power// /}"
      power_int="${power%%.*}"
      temp_int="${temp%%.*}"
      if [[ -n "$power_int" && -n "$temp_int" ]] &&
         (( power_int > max_power || temp_int > max_temp )); then
        bad=$((bad + 1))
      else
        bad=0
      fi
      printf '%s,%s,%s,%s,%s,%s,%s\n' "$ts" "$gpu" "$temp" "$util" "$mem" "$power" "$bad" >> "$gpu_log"
      if (( bad >= max_bad && term_sent == 0 )); then
        echo "POWER_GUARD terminate gpu=$gpu temp=${temp}C power=${power}W bad_samples=$bad limit=${max_power}W/${max_temp}C" >&2
        kill -TERM "$pid" 2>/dev/null || true
        term_sent=1
      fi
    fi
    sleep "$interval"
  done
  wait "$pid"
  rc="$?"
  if [[ "$rc" -eq 0 ]]; then
    meta_args=(--strict --require-converged --validate-numeric)
    if [[ "${BEM_METADATA_SKIP_CLOUDE:-0}" != "1" ]]; then
      meta_args+=(--require-cloude-physical)
    fi
    if [[ "$require_complex" == "1" ]]; then
      meta_args+=(--require-complex-operator)
    fi
    python3 scripts/check_result_metadata.py "${meta_args[@]}" "$json"
    echo "DONE $name rc=0"
  else
    echo "FAIL $name rc=$rc"
  fi
  exit "$rc"
) > "$log" 2>&1
