#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  remote_accuracy_refinement_wave.sh [--run] --hosts LIST [options]

Plans the next accuracy-refinement wave from the local accuracy CSV and
optionally starts different cases on different reachable remote GPUs. Default
mode is dry-run.

Options:
  --run                  Start remote jobs; default is dry-run
  --status-only          Print plan and remote GPU availability; never start
                         jobs and return success even when GPUs are busy
  --csv FILE             Accuracy CSV (default: poster_a0/assets/table_accuracy_matrix_15.csv)
  --refresh-audit        Rebuild the accuracy CSV before planning
  --hosts LIST           Space/comma-separated remote hosts to probe, or auto
  --user USER            SSH user; default is empty, so SSH config aliases work
  --remote-repo DIR      Remote BEM-CUDA path; default is auto-detect per host
  --gpus LIST            GPU ids per host, or auto (default: 0 1 2)
  --max-cases N          Maximum cases in this wave; default: hosts * listed GPUs
  --all-cases            Plan every pending case
  --only-reason MODE     all, accuracy, or metadata (default: all)
  --out DIR              Remote output directory (default: runs/production_matrix_refinement)
  --plan-csv FILE        Local plan CSV path (default: OUT/remote_plan.csv)
  --status-json FILE     Write machine-readable remote availability/status JSON.
                         Use '-' to print JSON to stdout.
  --max-temp C           Idle remote GPU temperature limit (default: 78)
  --max-util PCT         Idle remote GPU utilization limit (default: 20)
  --max-mem MB           Idle remote GPU memory-used limit (default: 2048)
  --allow-compute-share  Allow scheduling on GPUs with existing CUDA compute apps
  --sync-launchers       Sync local launcher/audit scripts to remote repos
                         before remote jobs. Default: enabled with --run.
  --no-sync-launchers    Disable launcher sync.
  --scan-hosts           With --hosts auto, scan subnets for SSH hosts
  --no-scan-hosts        With --hosts auto, use only known host candidates
  --scan-subnets LIST    Space/comma-separated CIDR subnets to scan
  --case-max-power W     Guard power limit for each remote case (default: 290)
  --case-max-temp C      Guard temperature limit for each remote case
  --case-max-bad-samples N
                         Guard bad-sample limit for each remote case (default: 4)
  --ssh-connect-timeout S
                         SSH connect timeout (default: 3)
  --wait-free            Retry remote probing until at least one GPU is usable
  --min-free-gpus N      With --wait-free, require at least N usable remote GPUs
                         before starting real jobs (default: 1)
  --wait-interval SEC    Poll interval for --wait-free (default: 60)
  --wait-timeout SEC     Stop waiting after this many seconds; 0 means forever
                         (default: 0)
  --continuous           With --run, keep polling and launching more waves
                         until no selected case starts. This is a remote queue:
                         one case per usable GPU per wave, never one case on
                         multiple GPUs.
  --queue-interval SEC   Poll interval for --continuous (default: 60)
  --queue-timeout SEC    Stop continuous queue after this many seconds; 0 means
                         forever (default: 0)
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO:-$(cd "$script_dir/.." && pwd)}"
cd "$repo"

run=0
status_only=0
csv="poster_a0/assets/table_accuracy_matrix_15.csv"
refresh_audit=0
hosts="${BEM_REMOTE_REFINEMENT_HOSTS:-}"
user="${USER_REMOTE:-}"
remote_repo=""
gpus="${BEM_REMOTE_REFINEMENT_GPUS:-0 1 2}"
max_cases=""
all_cases=0
only_reason="all"
out="runs/production_matrix_refinement"
plan_csv=""
max_temp="${BEM_REMOTE_REFINEMENT_MAX_TEMP_C:-78}"
max_util="${BEM_REMOTE_REFINEMENT_MAX_UTIL_PCT:-20}"
max_mem="${BEM_REMOTE_REFINEMENT_MAX_MEM_MB:-2048}"
allow_compute_share=0
sync_launchers="${BEM_REMOTE_REFINEMENT_SYNC_LAUNCHERS:-auto}"
case_max_power="${BEM_REMOTE_REFINEMENT_MAX_POWER:-290}"
case_max_temp=""
case_max_bad_samples="${BEM_REMOTE_REFINEMENT_MAX_BAD_SAMPLES:-4}"
connect_timeout="${BEM_REMOTE_REFINEMENT_CONNECT_TIMEOUT:-3}"
scan_hosts=""
scan_subnets=""
wait_free=0
min_free_gpus="${BEM_REMOTE_REFINEMENT_MIN_FREE_GPUS:-1}"
wait_interval="${BEM_REMOTE_REFINEMENT_WAIT_INTERVAL:-60}"
wait_timeout="${BEM_REMOTE_REFINEMENT_WAIT_TIMEOUT:-0}"
status_json=""
continuous=0
queue_interval="${BEM_REMOTE_REFINEMENT_QUEUE_INTERVAL:-60}"
queue_timeout="${BEM_REMOTE_REFINEMENT_QUEUE_TIMEOUT:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) run=1; shift ;;
    --status-only) status_only=1; shift ;;
    --dry-run) run=0; shift ;;
    --csv) csv="$2"; shift 2 ;;
    --refresh-audit) refresh_audit=1; shift ;;
    --hosts) hosts="$2"; shift 2 ;;
    --user) user="$2"; shift 2 ;;
    --remote-repo) remote_repo="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --max-cases) max_cases="$2"; shift 2 ;;
    --all-cases) all_cases=1; shift ;;
    --only-reason) only_reason="$2"; shift 2 ;;
    --out|--out-dir) out="$2"; shift 2 ;;
    --plan-csv) plan_csv="$2"; shift 2 ;;
    --status-json) status_json="$2"; shift 2 ;;
    --max-temp) max_temp="$2"; shift 2 ;;
    --max-util) max_util="$2"; shift 2 ;;
    --max-mem) max_mem="$2"; shift 2 ;;
    --allow-compute-share) allow_compute_share=1; shift ;;
    --sync-launchers) sync_launchers=1; shift ;;
    --no-sync-launchers) sync_launchers=0; shift ;;
    --scan-hosts) scan_hosts=1; shift ;;
    --no-scan-hosts) scan_hosts=0; shift ;;
    --scan-subnets) scan_subnets="$2"; shift 2 ;;
    --case-max-power) case_max_power="$2"; shift 2 ;;
    --case-max-temp) case_max_temp="$2"; shift 2 ;;
    --case-max-bad-samples) case_max_bad_samples="$2"; shift 2 ;;
    --ssh-connect-timeout) connect_timeout="$2"; shift 2 ;;
    --wait-free) wait_free=1; shift ;;
    --min-free-gpus) min_free_gpus="$2"; shift 2 ;;
    --wait-interval) wait_interval="$2"; shift 2 ;;
    --wait-timeout) wait_timeout="$2"; shift 2 ;;
    --continuous) continuous=1; shift ;;
    --queue-interval) queue_interval="$2"; shift 2 ;;
    --queue-timeout) queue_timeout="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$hosts" ]]; then
  echo "no hosts supplied; use --hosts or BEM_REMOTE_REFINEMENT_HOSTS" >&2
  usage >&2
  exit 2
fi

if [[ "$status_only" == "1" ]]; then
  run=0
  wait_free=0
  continuous=0
  sync_launchers=0
fi

if [[ -z "$plan_csv" ]]; then
  plan_csv="$out/remote_plan.csv"
fi

count_list() {
  local input="$1" count=0 item
  input="${input//,/ }"
  for item in $input; do
    [[ -n "$item" ]] && count=$((count + 1))
  done
  printf '%s\n' "$count"
}

effective_max_cases="$max_cases"
if [[ -z "$effective_max_cases" && "$all_cases" != "1" && "${gpus,,}" != "auto" && "${hosts,,}" != "auto" ]]; then
  host_count="$(count_list "$hosts")"
  gpu_count="$(count_list "$gpus")"
  effective_max_cases=$((host_count * gpu_count))
fi

plan_args=(
  --csv "$csv"
  --gpus "$gpus"
  --only-reason "$only_reason"
  --out "$out"
  --plan-csv "$plan_csv"
  --case-max-power "$case_max_power"
  --case-max-bad-samples "$case_max_bad_samples"
  --no-health-check
)
if [[ "$refresh_audit" == "1" ]]; then
  plan_args+=(--refresh-audit)
fi
if [[ -n "$effective_max_cases" ]]; then
  plan_args+=(--max-cases "$effective_max_cases")
fi
if [[ "$all_cases" == "1" || "${gpus,,}" == "auto" || "${hosts,,}" == "auto" ]]; then
  plan_args+=(--all-cases)
fi
if [[ "$allow_compute_share" == "1" ]]; then
  plan_args+=(--allow-compute-share)
fi

cases=""
planned_count=0
mode="dry-run"
if [[ "$run" == "1" ]]; then
  mode="run"
fi
if [[ "$status_only" == "1" ]]; then
  mode="status"
fi
echo "REMOTE_REFINEMENT_WAVE mode=$mode hosts=$hosts gpus=$gpus out=$out"

write_status_json() {
  local json_path="$1" remote_output="$2" remote_rc="$3" usable="$4" selected="$5" plan_failed="${6:-0}" payload tmp
  [[ -z "$json_path" ]] && return 0
  payload="$(
    STATUS_MODE="$mode" \
    STATUS_HOSTS="$hosts" \
    STATUS_GPUS="$gpus" \
    STATUS_OUT="$out" \
    STATUS_PLAN_CSV="$plan_csv" \
    STATUS_CASES="$cases" \
    STATUS_PLANNED="$planned_count" \
    STATUS_REMOTE_RC="$remote_rc" \
    STATUS_USABLE="${usable:-0}" \
    STATUS_SELECTED="${selected:-0}" \
    STATUS_MIN_FREE_GPUS="$min_free_gpus" \
    STATUS_REMOTE_OUTPUT="$remote_output" \
    STATUS_PLAN_FAILED="$plan_failed" \
    python3 - <<'PY'
import json
import os
import re

remote_output = os.environ.get("STATUS_REMOTE_OUTPUT", "")
lines = [line for line in remote_output.splitlines() if line.strip()]

def to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

status = {
    "mode": os.environ.get("STATUS_MODE", ""),
    "hosts": os.environ.get("STATUS_HOSTS", ""),
    "gpus": os.environ.get("STATUS_GPUS", ""),
    "out": os.environ.get("STATUS_OUT", ""),
    "plan_csv": os.environ.get("STATUS_PLAN_CSV", ""),
    "planned_cases": to_int(os.environ.get("STATUS_PLANNED")),
    "cases": [case for case in os.environ.get("STATUS_CASES", "").split(",") if case],
    "remote_rc": to_int(os.environ.get("STATUS_REMOTE_RC")),
    "plan_failed": os.environ.get("STATUS_PLAN_FAILED", "0") == "1",
    "usable_remote_gpus": to_int(os.environ.get("STATUS_USABLE")),
    "min_free_gpus": to_int(os.environ.get("STATUS_MIN_FREE_GPUS"), 1),
    "selected": to_int(os.environ.get("STATUS_SELECTED")),
    "remote_status_lines": [line for line in lines if line.startswith("REMOTE_")],
    "busy_gpus": [line for line in lines if line.startswith("REMOTE_GPU_BUSY ")],
    "skipped_gpus": [line for line in lines if line.startswith("REMOTE_GPU_SKIP ") or line.startswith("REMOTE_GPU_LIST_SKIP ")],
}
status["enough_free_gpus"] = status["usable_remote_gpus"] >= status["min_free_gpus"]

for line in lines:
    if line.startswith("REMOTE_HOST_AUTO hosts="):
        status["auto_hosts"] = line.split("hosts=", 1)[1].split()
    if line.startswith("REMOTE_RESUME cases="):
        pairs = dict(re.findall(r"([A-Za-z_]+)=([^ ]+)", line))
        if pairs:
            status["remote_resume"] = pairs

print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
PY
  )"
  if [[ "$json_path" == "-" ]]; then
    printf '%s\n' "$payload"
  else
    tmp="${json_path}.tmp"
    mkdir -p "$(dirname "$json_path")"
    printf '%s\n' "$payload" > "$tmp"
    mv "$tmp" "$json_path"
  fi
}

set +e
plan_output="$(python3 scripts/plan_accuracy_refinement_cases.py "${plan_args[@]}" 2>&1)"
plan_rc="$?"
set -e
printf '%s\n' "$plan_output"
if [[ "$plan_rc" -ne 0 ]]; then
  write_status_json "$status_json" "$plan_output" "$plan_rc" 0 0 1
  exit "$plan_rc"
fi

cases="$(printf '%s\n' "$plan_output" | awk '/^[A-Za-z0-9_]+$/ {printf "%s%s", sep, $0; sep=","}')"
if [[ -n "$cases" ]]; then
  planned_count="$(printf '%s\n' "$cases" | awk -F, '{print NF}')"
else
  planned_count=0
fi

if [[ -z "$cases" ]]; then
  write_status_json "$status_json" "$plan_output" 0 0 0
  exit 0
fi

remote_args=(
  --hosts "$hosts"
  --gpus "$gpus"
  --cases "$cases"
  --out "$out"
  --max-temp "$max_temp"
  --max-util "$max_util"
  --max-mem "$max_mem"
  --case-max-power "$case_max_power"
  --case-max-bad-samples "$case_max_bad_samples"
  --ssh-connect-timeout "$connect_timeout"
)
if [[ "$run" == "1" ]]; then
  remote_args=(--run "${remote_args[@]}")
fi
if [[ -n "$user" ]]; then
  remote_args+=(--user "$user")
fi
if [[ -n "$remote_repo" ]]; then
  remote_args+=(--remote-repo "$remote_repo")
fi
if [[ -n "$effective_max_cases" ]]; then
  remote_args+=(--max-jobs "$effective_max_cases")
fi
if [[ "$allow_compute_share" == "1" ]]; then
  remote_args+=(--allow-compute-share)
fi
if [[ "$sync_launchers" == "1" || ( "$sync_launchers" == "auto" && "$run" == "1" ) ]]; then
  remote_args+=(--sync-launchers)
fi
if [[ "$scan_hosts" == "1" ]]; then
  remote_args+=(--scan-hosts)
elif [[ "$scan_hosts" == "0" ]]; then
  remote_args+=(--no-scan-hosts)
fi
if [[ -n "$scan_subnets" ]]; then
  remote_args+=(--scan-subnets "$scan_subnets")
fi
if [[ -n "$case_max_temp" ]]; then
  remote_args+=(--case-max-temp "$case_max_temp")
fi

echo
echo "remote_command:"
printf '%q ' scripts/remote_resume_accuracy_matrix_cases.sh "${remote_args[@]}"
printf '\n'

run_remote_resume() {
  local attempt=1 start now elapsed rc output usable selected
  local probe_args=()
  local arg
  start="$(date +%s)"
  for arg in "${remote_args[@]}"; do
    case "$arg" in
      --run|--sync-launchers) ;;
      *) probe_args+=("$arg") ;;
    esac
  done
  while true; do
    if [[ "$wait_free" == "1" ]]; then
      set +e
      output="$(scripts/remote_resume_accuracy_matrix_cases.sh "${probe_args[@]}" 2>&1)"
      rc="$?"
      set -e
    else
      set +e
      output="$(scripts/remote_resume_accuracy_matrix_cases.sh "${remote_args[@]}" 2>&1)"
      rc="$?"
      set -e
    fi
    printf '%s\n' "$output"
    usable="$(printf '%s\n' "$output" | sed -n 's/.*usable_remote_gpus=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    selected="$(printf '%s\n' "$output" | sed -n 's/REMOTE_RESUME selected=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    if [[ "$status_only" == "1" ]]; then
      echo "REMOTE_REFINEMENT_STATUS planned_cases=$planned_count usable_remote_gpus=${usable:-0} selected=${selected:-0} remote_rc=$rc"
      write_status_json "$status_json" "$output" "$rc" "${usable:-0}" "${selected:-0}"
      return 0
    fi
    if [[ "$rc" == "0" ]]; then
      usable="${usable:-0}"
      if [[ "$wait_free" == "1" && "$usable" -lt "$min_free_gpus" ]]; then
        rc=3
        :
      elif [[ "$wait_free" == "1" && "$run" == "1" ]]; then
        set +e
        output="$(scripts/remote_resume_accuracy_matrix_cases.sh "${remote_args[@]}" 2>&1)"
        rc="$?"
        set -e
        printf '%s\n' "$output"
        usable="$(printf '%s\n' "$output" | sed -n 's/.*usable_remote_gpus=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
        selected="$(printf '%s\n' "$output" | sed -n 's/REMOTE_RESUME selected=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
        write_status_json "$status_json" "$output" "$rc" "${usable:-0}" "${selected:-0}"
        return "$rc"
      else
        write_status_json "$status_json" "$output" "$rc" "${usable:-0}" "${selected:-0}"
        return 0
      fi
    elif [[ "$wait_free" != "1" || "$rc" != "3" ]]; then
      write_status_json "$status_json" "$output" "$rc" "${usable:-0}" "${selected:-0}"
      return "$rc"
    fi
    if [[ "$wait_free" == "1" && "$rc" == "3" && "${usable:-0}" -lt "$min_free_gpus" ]]; then
      echo "REMOTE_WAIT usable_remote_gpus=${usable:-0} min_free_gpus=$min_free_gpus attempt=$attempt" >&2
    fi
    if [[ "$wait_free" == "1" && ( "$rc" == "3" || "${usable:-0}" -lt "$min_free_gpus" ) ]]; then
      :
    else
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( wait_timeout > 0 && elapsed >= wait_timeout )); then
      echo "REMOTE_WAIT timeout=${wait_timeout}s attempts=$attempt last_rc=$rc" >&2
      write_status_json "$status_json" "$output" "$rc" "${usable:-0}" "${selected:-0}"
      return "$rc"
    fi
    echo "REMOTE_WAIT no usable GPUs attempt=$attempt elapsed=${elapsed}s sleep=${wait_interval}s" >&2
    attempt=$((attempt + 1))
    sleep "$wait_interval"
  done
}

run_continuous_queue() {
  local attempt=1 start now elapsed rc output usable selected enough_free_gpus
  local loop_args=("${remote_args[@]}")
  start="$(date +%s)"
  if [[ "$run" != "1" ]]; then
    echo "--continuous requires --run" >&2
    return 2
  fi
  while true; do
    set +e
    output="$(scripts/remote_resume_accuracy_matrix_cases.sh "${loop_args[@]}" 2>&1)"
    rc="$?"
    set -e
    printf '%s\n' "$output"
    usable="$(printf '%s\n' "$output" | sed -n 's/.*usable_remote_gpus=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    selected="$(printf '%s\n' "$output" | sed -n 's/REMOTE_RESUME selected=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    if [[ "${usable:-0}" -ge "$min_free_gpus" ]]; then
      enough_free_gpus=1
    else
      enough_free_gpus=0
    fi
    echo "REMOTE_QUEUE_STATUS attempt=$attempt usable_remote_gpus=${usable:-0} min_free_gpus=$min_free_gpus enough_free_gpus=$enough_free_gpus selected=${selected:-0} remote_rc=$rc"
    write_status_json "$status_json" "$output" "$rc" "${usable:-0}" "${selected:-0}"

    if [[ "$rc" != "0" && "$rc" != "3" ]]; then
      return "$rc"
    fi
    if [[ "$rc" == "0" && "${selected:-0}" == "0" ]]; then
      echo "REMOTE_QUEUE_DONE no cases started in last wave"
      return 0
    fi

    now="$(date +%s)"
    elapsed=$((now - start))
    if (( queue_timeout > 0 && elapsed >= queue_timeout )); then
      echo "REMOTE_QUEUE_TIMEOUT timeout=${queue_timeout}s attempts=$attempt last_rc=$rc" >&2
      return "$rc"
    fi

    # Sync launchers only on the first wave; repeated rsyncs are wasted while
    # the queue is just waiting for GPUs to free up.
    loop_args=()
    for arg in "${remote_args[@]}"; do
      [[ "$arg" == "--sync-launchers" ]] && continue
      loop_args+=("$arg")
    done
    echo "REMOTE_QUEUE_WAIT attempt=$attempt elapsed=${elapsed}s sleep=${queue_interval}s"
    attempt=$((attempt + 1))
    sleep "$queue_interval"
  done
}

if [[ "$continuous" == "1" ]]; then
  run_continuous_queue
else
  run_remote_resume
fi
