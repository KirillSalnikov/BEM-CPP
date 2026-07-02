#!/usr/bin/env bash
set -euo pipefail

out="${OUT:-runs/production_matrix_15}"
nvidia_smi="${BEM_NVIDIA_SMI:-nvidia-smi}"
snapshot_file="${QUEUE_STATUS_SNAPSHOT:-$out/logs/.queue_live_status.snapshot}"
snapshot_now="$(date +%s)"
active_age_s="${QUEUE_STATUS_ACTIVE_AGE_S:-30}"

declare -A prev_proc_cpu=()
declare -A prev_gpu_samples=()
declare -A curr_proc_cpu=()
declare -A curr_gpu_samples=()
declare -a curr_proc_pids=()
declare -a curr_gpu_cases=()
declare -a active_monitor_lines=()
prev_snapshot_time=""

load_snapshot() {
  local kind key value
  [[ -s "$snapshot_file" ]] || return 0
  while read -r kind key value; do
    case "$kind" in
      time) prev_snapshot_time="$key" ;;
      proc) prev_proc_cpu["$key"]="$value" ;;
      gpu) prev_gpu_samples["$key"]="$value" ;;
    esac
  done < "$snapshot_file"
}

time_to_seconds() {
  local value="$1" days=0 rest h=0 m=0 s=0
  if [[ "$value" == *-* ]]; then
    days="${value%%-*}"
    rest="${value#*-}"
  else
    rest="$value"
  fi
  IFS=: read -r -a parts <<< "$rest"
  if [[ "${#parts[@]}" -eq 3 ]]; then
    h="${parts[0]}"; m="${parts[1]}"; s="${parts[2]}"
  elif [[ "${#parts[@]}" -eq 2 ]]; then
    m="${parts[0]}"; s="${parts[1]}"
  else
    s="${parts[0]:-0}"
  fi
  echo $((10#$days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s))
}

csv_sample_count() {
  local path="$1" lines
  if [[ ! -s "$path" ]]; then
    echo 0
    return
  fi
  lines="$(wc -l < "$path")"
  if (( lines > 0 )); then
    echo $((lines - 1))
  else
    echo 0
  fi
}

gpu_csv_summary() {
  local path="$1"
  awk -F, '
    NR == 1 { next }
    NF >= 6 {
      n += 1
      ts = $1 + 0
      gpu = $2 + 0
      temp = $3 + 0
      util = $4 + 0
      mem = $5 + 0
      power = $6 + 0
      if (n == 1) {
        first_ts = ts
        first_gpu = gpu
        max_temp = temp
        max_mem = mem
        max_power = power
      }
      last_ts = ts
      sum_power += power
      sum_util += util
      if (temp > max_temp) max_temp = temp
      if (mem > max_mem) max_mem = mem
      if (power > max_power) max_power = power
    }
    END {
      if (n == 0) {
        print "gpu_summary=missing"
      } else {
        printf "gpu_summary=gpu=%d duration_s=%d power_mean_w=%.1f power_max_w=%.0f util_mean_pct=%.1f temp_max_c=%.0f mem_max_mib=%.0f\n",
          first_gpu, (last_ts - first_ts), (sum_power / n), max_power, (sum_util / n), max_temp, max_mem
      }
    }
  ' "$path"
}

save_snapshot() {
  local dir tmp pid case_name
  dir="$(dirname "$snapshot_file")"
  mkdir -p "$dir"
  tmp="${snapshot_file}.tmp.$$"
  {
    echo "time $snapshot_now 0"
    for pid in "${curr_proc_pids[@]}"; do
      echo "proc $pid ${curr_proc_cpu[$pid]}"
    done
    for case_name in "${curr_gpu_cases[@]}"; do
      echo "gpu $case_name ${curr_gpu_samples[$case_name]}"
    done
  } > "$tmp"
  mv "$tmp" "$snapshot_file"
}

load_snapshot

file_age_s() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo missing
    return
  fi
  echo $(( $(date +%s) - $(stat -c %Y "$path") ))
}

gmres_log_summary() {
  local log="$1"
  local last_iter last_restart done_line fail_line
  if [[ ! -s "$log" ]]; then
    echo "gmres_last=missing"
    return
  fi
  done_line="$(grep -E '^DONE ' "$log" | tail -n 1 || true)"
  if [[ -n "$done_line" ]]; then
    echo "gmres_done=${done_line}"
    return
  fi
  fail_line="$(grep -E '^FAIL ' "$log" | tail -n 1 || true)"
  if [[ -n "$fail_line" ]]; then
    echo "case_failed=${fail_line}"
    return
  fi
  last_iter="$(grep -E 'GMRES iter [0-9]+: rel' "$log" | tail -n 1 || true)"
  if [[ -n "$last_iter" ]]; then
    echo "gmres_last=${last_iter#"${last_iter%%[![:space:]]*}"}"
    return
  fi
  last_restart="$(grep -E '\[GMRES[^]]*\] restart [0-9]+:' "$log" | tail -n 1 || true)"
  if [[ -n "$last_restart" ]]; then
    echo "gmres_last=${last_restart#"${last_restart%%[![:space:]]*}"}"
    return
  fi
  if grep -q '\[GMRES\] verbose residual logging enabled' "$log"; then
    echo "gmres_last=verbose-enabled-no-iteration-yet"
  else
    echo "gmres_last=no-iteration-lines"
  fi
}

monitor_state_value() {
  local csv_age="$1" log="$2"
  local terminal_line
  terminal_line="$(grep -E '^(DONE|FAIL) ' "$log" 2>/dev/null | tail -n 1 || true)"
  if [[ -n "$terminal_line" ]]; then
    echo "finished"
  elif [[ "$csv_age" =~ ^[0-9]+$ ]] && (( csv_age <= active_age_s )); then
    echo "active"
  else
    echo "stale"
  fi
}

collect_descendants() {
  local parent="$1"
  local child
  pgrep -P "$parent" 2>/dev/null | while read -r child; do
    [[ -n "$child" ]] || continue
    echo "$child"
    collect_descendants "$child"
  done
  return 0
}

print_descendant_processes() {
  local parent="$1"
  local pids pid cputime cpu_s
  pids="$(collect_descendants "$parent" | paste -sd, -)"
  if [[ -z "$pids" ]]; then
    echo "no descendants"
    return
  fi
  ps -p "$pids" -o pid,ppid,etime,time,pcpu,rss,cmd --sort=pid 2>/dev/null || true
  IFS=, read -r -a pid_array <<< "$pids"
  for pid in "${pid_array[@]}"; do
    [[ -n "$pid" ]] || continue
    cputime="$(ps -p "$pid" -o time= 2>/dev/null | tr -d ' ' || true)"
    [[ -n "$cputime" ]] || continue
    cpu_s="$(time_to_seconds "$cputime")"
    curr_proc_cpu["$pid"]="$cpu_s"
    curr_proc_pids+=("$pid")
  done
}

print_progress_delta() {
  local wall_delta pid prev_cpu curr_cpu cpu_delta case_name prev_samples curr_samples sample_delta
  if [[ -z "$prev_snapshot_time" ]]; then
    echo "delta=first-sample"
    return
  fi
  wall_delta=$((snapshot_now - prev_snapshot_time))
  echo "wall_s_delta=$wall_delta"
  for pid in "${curr_proc_pids[@]}"; do
    curr_cpu="${curr_proc_cpu[$pid]}"
    prev_cpu="${prev_proc_cpu[$pid]:-}"
    if [[ -n "$prev_cpu" ]]; then
      cpu_delta=$((curr_cpu - prev_cpu))
      echo "proc_delta pid=$pid cpu_s_delta=$cpu_delta"
    else
      echo "proc_delta pid=$pid cpu_s_delta=new"
    fi
  done
  for case_name in "${curr_gpu_cases[@]}"; do
    curr_samples="${curr_gpu_samples[$case_name]}"
    prev_samples="${prev_gpu_samples[$case_name]:-}"
    if [[ -n "$prev_samples" ]]; then
      sample_delta=$((curr_samples - prev_samples))
      echo "gpu_samples_delta case=$case_name samples_delta=$sample_delta"
    else
      echo "gpu_samples_delta case=$case_name samples_delta=new"
    fi
  done
}

echo "=== queue ==="
if [[ -f "$out/queue.pid" ]]; then
  queue_pid="$(cat "$out/queue.pid")"
  if kill -0 "$queue_pid" 2>/dev/null; then
    ps -p "$queue_pid" -o pid,etime,cmd
    echo "--- children ---"
    pgrep -P "$queue_pid" -a || true
    echo "--- descendants ---"
    print_descendant_processes "$queue_pid"
  else
    echo "queue pid $queue_pid is not running"
  fi
else
  echo "no $out/queue.pid"
fi

if [[ -f "$out/resume_after_current.pid" ]]; then
  resume_pid="$(cat "$out/resume_after_current.pid")"
  if kill -0 "$resume_pid" 2>/dev/null; then
    echo "--- resume watcher ---"
    ps -p "$resume_pid" -o pid,etime,cmd
    tail -n 5 "$out/resume_after_current.log" 2>/dev/null || true
  fi
fi

echo "=== result status ==="
if [[ -x scripts/run_accuracy_matrix_15_queue.sh ]]; then
  scripts/run_accuracy_matrix_15_queue.sh --status 2>/dev/null || true
else
  echo "missing scripts/run_accuracy_matrix_15_queue.sh"
fi

echo "=== gpu ==="
if command -v "$nvidia_smi" >/dev/null 2>&1; then
  "$nvidia_smi" --query-gpu=index,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used \
    --format=csv,noheader,nounits 2>&1 || true
else
  echo "$nvidia_smi missing"
fi

echo "=== monitor files ==="
shopt -s nullglob
for csv in "$out"/logs/*.gpu.csv; do
  log="${csv%.gpu.csv}.log"
  case_name="$(basename "$csv" .gpu.csv)"
  sample_count="$(csv_sample_count "$csv")"
  csv_age="$(file_age_s "$csv")"
  log_age="$(file_age_s "$log")"
  state="$(monitor_state_value "$csv_age" "$log")"
  summary="$(gpu_csv_summary "$csv")"
  curr_gpu_samples["$case_name"]="$sample_count"
  curr_gpu_cases+=("$case_name")
  if [[ "$state" == "active" ]]; then
    active_monitor_lines+=("$case_name age_s=$csv_age samples=$sample_count ${summary#gpu_summary=}")
  fi
  echo "--- $case_name ---"
  echo "gpu_csv_age_s=$csv_age log_age_s=$log_age"
  echo "monitor_state=$state"
  echo "gpu_samples=$sample_count"
  echo "$summary"
  gmres_log_summary "$log"
  tail -n 3 "$csv" || true
done

echo "=== active monitors ==="
if [[ "${#active_monitor_lines[@]}" -eq 0 ]]; then
  echo "none"
else
  printf '%s\n' "${active_monitor_lines[@]}"
fi

echo "=== progress delta ==="
print_progress_delta
save_snapshot
