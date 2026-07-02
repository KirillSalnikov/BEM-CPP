#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  resume_accuracy_matrix_cases.sh [--run] [--out DIR] [--gpus LIST] [--max-jobs N] [guard options]

Starts missing/stale production accuracy cases, assigning one different case to
one idle GPU. The default mode is dry-run: pass --run to actually launch.

Options:
  --run                  Launch jobs; default is dry-run
  --out DIR              Output directory (default: runs/production_matrix_15)
  --gpus LIST            Space/comma-separated GPU ids, or auto (default: auto)
  --max-jobs N           Maximum jobs to start in this invocation
  --cases LIST           Only consider these case names (space/comma-separated)
  --max-temp C           Idle GPU temperature limit (default: 78)
  --max-util PCT         Idle GPU utilization limit (default: 20)
  --max-mem MB           Idle GPU memory-used limit (default: 2048)
  --allow-compute-share  Allow scheduling on GPUs with existing CUDA compute
                         processes. Default: skip any GPU with compute apps.
  --case-max-power W     Guard power limit passed to the case runner
  --case-max-temp C      Guard temperature limit passed to the case runner
  --case-max-bad-samples N
                         Guard bad-sample count passed to the case runner
  --no-health-check      Do not query nvidia-smi before selecting GPUs
  --allow-oversubscribe  Allow more than one case per GPU in one invocation
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO:-$(cd "$script_dir/.." && pwd)}"
cd "$repo"

out="runs/production_matrix_15"
gpu_list="${BEM_RESUME_GPUS:-auto}"
exclude_gpus="${BEM_RESUME_EXCLUDE_GPUS:-3}"
nvidia_smi="${BEM_NVIDIA_SMI:-nvidia-smi}"
run=0
max_jobs=0
max_temp="${BEM_RESUME_MAX_TEMP_C:-78}"
max_util="${BEM_RESUME_MAX_UTIL_PCT:-20}"
max_mem="${BEM_RESUME_MAX_MEM_MB:-2048}"
allow_compute_share="${BEM_RESUME_ALLOW_COMPUTE_SHARE:-0}"
case_guard_args=()
health_check=1
allow_oversubscribe=0
case_filter=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) run=1; shift ;;
    --dry-run) run=0; shift ;;
    --out|--out-dir) out="$2"; shift 2 ;;
    --gpus) gpu_list="$2"; shift 2 ;;
    --max-jobs) max_jobs="$2"; shift 2 ;;
    --cases) case_filter="$2"; shift 2 ;;
    --max-temp) max_temp="$2"; shift 2 ;;
    --max-util) max_util="$2"; shift 2 ;;
    --max-mem) max_mem="$2"; shift 2 ;;
    --allow-compute-share) allow_compute_share=1; shift ;;
    --case-max-power) case_guard_args+=(--max-power "$2"); shift 2 ;;
    --case-max-temp) case_guard_args+=(--max-temp "$2"); shift 2 ;;
    --case-max-bad-samples) case_guard_args+=(--max-bad-samples "$2"); shift 2 ;;
    --no-health-check) health_check=0; shift ;;
    --allow-oversubscribe) allow_oversubscribe=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

normalize_gpu_list() {
  local item normalized
  if [[ "$gpu_list" == "auto" ]]; then
    if ! command -v "$nvidia_smi" >/dev/null 2>&1; then
      echo "GPU_AUTO failed: missing $nvidia_smi" >&2
      return 1
    fi
    "$nvidia_smi" --query-gpu=index --format=csv,noheader,nounits 2>/dev/null \
      | awk -v exclude=",$exclude_gpus," 'NF {
          gsub(/^[ \t]+|[ \t]+$/, "", $1);
          if (!index(exclude, "," $1 ",")) print $1;
        }'
    return 0
  fi
  normalized="${gpu_list//,/ }"
  for item in $normalized; do
    printf '%s\n' "$item"
  done
}

case_selected() {
  local case_name="$1"
  local selected
  if [[ -z "$case_filter" ]]; then
    return 0
  fi
  local normalized
  normalized="${case_filter//,/ }"
  for selected in $normalized; do
    [[ -n "$selected" ]] || continue
    if [[ "$selected" == "$case_name" ]]; then
      return 0
    fi
  done
  return 1
}

dedupe_preserve_order() {
  awk 'NF && !seen[$0]++'
}

validate_case_name() {
  local case_name="$1" validation_output
  if ! validation_output="$(scripts/run_accuracy_matrix_case.sh \
      --gpu 0 --case "$case_name" --out "$out" --print 2>&1 >/dev/null)"; then
    echo "CASE_INVALID case=$case_name" >&2
    printf '%s\n' "$validation_output" >&2
    return 1
  fi
}

trim_int() {
  local value="$1"
  value="${value// /}"
  value="${value%%.*}"
  printf '%s\n' "$value"
}

gpu_idle() {
  local gpu="$1" line temp util mem apps
  if ! line="$("$nvidia_smi" -i "$gpu" --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits 2>&1)"; then
    echo "GPU_SKIP gpu=$gpu nvidia-smi_failed $line" >&2
    return 1
  fi
  IFS=',' read -r temp util mem <<<"$line"
  temp="$(trim_int "$temp")"
  util="$(trim_int "$util")"
  mem="$(trim_int "$mem")"
  if [[ -z "$temp" || -z "$util" || -z "$mem" ]]; then
    echo "GPU_SKIP gpu=$gpu unparsable $line" >&2
    return 1
  fi
  if (( temp > max_temp || util > max_util || mem > max_mem )); then
    echo "GPU_BUSY gpu=$gpu temp=${temp}C util=${util}% mem=${mem}MiB" >&2
    return 1
  fi
  if [[ "$allow_compute_share" != "1" ]]; then
    apps="$("$nvidia_smi" -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    apps="$(printf '%s\n' "$apps" | sed '/^[[:space:]]*$/d' | head -n 3)"
    if [[ -n "$apps" ]]; then
      apps="${apps//$'\n'/; }"
      echo "GPU_BUSY gpu=$gpu compute_apps=$apps" >&2
      return 1
    fi
  fi
  echo "$gpu"
}

collect_pending_cases() {
  local status_text state name selected normalized line
  status_text="$(OUT="$out" scripts/run_accuracy_matrix_15_queue.sh --status || true)"
  if [[ -n "$case_filter" ]]; then
    normalized="${case_filter//,/ }"
    for selected in $normalized; do
      found=0
      while IFS= read -r line; do
        read -r state name _ <<<"$line"
        [[ "$name" == "$selected" ]] || continue
        found=1
        case "$state" in
          MISSING|STALE) printf '%s %s\n' "$state" "$name" ;;
        esac
      done <<<"$status_text"
      if [[ "$found" == "0" ]]; then
        printf 'MISSING %s\n' "$selected"
      fi
    done
    return
  fi

  while IFS= read -r line; do
    read -r state name _ <<<"$line"
    case "$state" in
      MISSING|STALE) printf '%s %s\n' "$state" "$name" ;;
    esac
  done <<<"$status_text"
}

mapfile -t pending < <(collect_pending_cases | dedupe_preserve_order)
for pending_item in "${pending[@]}"; do
  case_name="${pending_item#* }"
  validate_case_name "$case_name"
done

if [[ "$health_check" == "1" ]]; then
  mapfile -t idle_gpus < <(for gpu in $(normalize_gpu_list); do gpu_idle "$gpu" || true; done)
else
  mapfile -t idle_gpus < <(normalize_gpu_list)
fi

echo "RESUME pending=${#pending[@]} idle_gpus=${#idle_gpus[@]} mode=$([[ "$run" == "1" ]] && echo run || echo dry-run)"
if [[ "${#pending[@]}" -eq 0 ]]; then
  exit 0
fi
if [[ "${#idle_gpus[@]}" -eq 0 ]]; then
  echo "RESUME no idle GPUs" >&2
  exit 3
fi

mkdir -p "$out/logs"
started=0
selection_limit="${#idle_gpus[@]}"
if (( max_jobs > 0 && max_jobs < selection_limit )); then
  selection_limit="$max_jobs"
fi
if [[ "$allow_oversubscribe" == "1" && "$max_jobs" -gt 0 ]]; then
  selection_limit="$max_jobs"
fi
for pending_item in "${pending[@]}"; do
  if (( started >= selection_limit )); then
    break
  fi
  state="${pending_item%% *}"
  case_name="${pending_item#* }"
  gpu="${idle_gpus[$((started % ${#idle_gpus[@]}))]}"
  cmd=(scripts/run_accuracy_matrix_case.sh --gpu "$gpu" --case "$case_name" --out "$out")
  if [[ "$state" == "STALE" ]]; then
    cmd+=(--force)
  fi
  cmd+=("${case_guard_args[@]}")
  if [[ "$run" == "1" ]]; then
    launcher_log="$out/logs/$case_name.launcher.log"
    printf 'START_CASE gpu=%s case=%s\n' "$gpu" "$case_name" | tee "$launcher_log"
    BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}" nohup "${cmd[@]}" >> "$launcher_log" 2>&1 &
    pid="$!"
    printf '%s\n' "$pid" > "$out/logs/$case_name.pid"
    echo "STARTED gpu=$gpu case=$case_name pid=$pid"
  else
    printf 'DRYRUN gpu=%s case=%s cmd=' "$gpu" "$case_name"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  fi
  started=$((started + 1))
done

echo "RESUME selected=$started"
