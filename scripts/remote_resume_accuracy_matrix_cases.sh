#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  remote_resume_accuracy_matrix_cases.sh [--run] --hosts LIST [--cases LIST] [options]

Distributes different accuracy-matrix cases over reachable remote GPUs. Each
selected host/GPU receives at most one case in one invocation. Default mode is
dry-run; pass --run to start remote jobs.

Options:
  --run                  Launch jobs; default is dry-run
  --hosts LIST           Space/comma-separated hosts to probe, or auto
  --user USER            SSH user; default is empty, so SSH config aliases work
  --remote-repo DIR      Remote BEM-CUDA path. If omitted, auto-detects
                         $HOME/BEM-CUDA, /home/kirill_epyc/BEM-CUDA,
                         /home/sasha_tvo_gpu1/BEM-CUDA and
                         /home/sasha_tvo_gpu2/BEM-CUDA per host.
  --out DIR              Remote output directory (default: runs/production_matrix_15)
  --gpus LIST            Space/comma-separated GPU ids per host, or auto (default: 0)
  --cases LIST           Space/comma-separated case names. If omitted, local
                         resume_accuracy_matrix_cases.sh supplies pending cases.
  --max-jobs N           Maximum remote jobs to start; default: number of idle GPUs
  --max-temp C           Idle GPU temperature limit (default: 78)
  --max-util PCT         Idle GPU utilization limit (default: 20)
  --max-mem MB           Idle GPU memory-used limit (default: 2048)
  --allow-compute-share  Allow scheduling on GPUs with existing CUDA compute
                         processes. Default: skip any GPU with compute apps.
  --sync-launchers       Rsync local launcher/audit scripts to the remote repo
                         before starting jobs. Does not rebuild the solver.
  --scan-hosts           With --hosts auto, scan subnets for SSH hosts
                         (default: enabled)
  --no-scan-hosts        With --hosts auto, use only BEM_REMOTE_RESUME_AUTO_HOSTS
  --scan-subnets LIST    Space/comma-separated CIDR subnets to scan
                         (default: 172.16.0.0/22)
  --case-max-power W     Guard power limit passed to the remote case runner
  --case-max-temp C      Guard temperature limit passed to the remote case runner
  --case-max-bad-samples N
                         Guard bad-sample count passed to the remote case runner
  --ssh-connect-timeout S
                         SSH connect timeout (default: 3)
  --ssh-command-timeout S
                         Wall timeout for each remote SSH command; 0 disables
                         (default: 20)
  --case-lease-ttl SEC   Keep a local in-flight lease for started remote cases
                         for this many seconds; 0 disables leases
                         (default: 172800)

Environment:
  BEM_REMOTE_RESUME_SSH  Test hook: command used instead of ssh.
  BEM_REMOTE_RESUME_RSYNC
                         Test hook: command used instead of rsync sync.
  BEM_REMOTE_RESUME_AUTO_HOSTS
                         Space/comma-separated host list used by --hosts auto.
  BEM_REMOTE_RESUME_SCAN_HOSTS
                         1/0: scan subnets for --hosts auto (default: 1).
  BEM_REMOTE_RESUME_SCAN_SUBNETS
                         Space/comma-separated CIDR list (default: 172.16.0.0/22).
  BEM_REMOTE_RESUME_SCAN_OUTPUT
                         Test hook: host list returned by the subnet scanner.
  BEM_REMOTE_RESUME_STATUS_TEXT
                         Test hook: queue status text used instead of
                         run_accuracy_matrix_15_queue.sh --status.
  BEM_NVIDIA_SMI         Remote nvidia-smi command/path (default: nvidia-smi).
  BEM_REMOTE_RESUME_COMMAND_TIMEOUT
                         Default for --ssh-command-timeout.
  BEM_REMOTE_RESUME_CASE_LEASE_TTL
                         Default for --case-lease-ttl.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO:-$(cd "$script_dir/.." && pwd)}"
cd "$repo"

run=0
host_list="${BEM_REMOTE_RESUME_HOSTS:-}"
user="${USER_REMOTE:-}"
remote_repo=""
out="runs/production_matrix_15"
gpu_list="${BEM_REMOTE_RESUME_GPUS:-0}"
case_filter=""
max_jobs=0
max_temp="${BEM_REMOTE_RESUME_MAX_TEMP_C:-78}"
max_util="${BEM_REMOTE_RESUME_MAX_UTIL_PCT:-20}"
max_mem="${BEM_REMOTE_RESUME_MAX_MEM_MB:-2048}"
allow_compute_share="${BEM_REMOTE_RESUME_ALLOW_COMPUTE_SHARE:-0}"
sync_launchers="${BEM_REMOTE_RESUME_SYNC_LAUNCHERS:-0}"
connect_timeout="${BEM_REMOTE_RESUME_CONNECT_TIMEOUT:-3}"
command_timeout="${BEM_REMOTE_RESUME_COMMAND_TIMEOUT:-20}"
nvidia_smi="${BEM_NVIDIA_SMI:-nvidia-smi}"
auto_hosts="${BEM_REMOTE_RESUME_AUTO_HOSTS:-gpu1 gpu2 gpu3 172.16.1.222 172.16.0.73 172.16.0.212 172.16.1.168 172.16.1.149 epyc1}"
scan_hosts="${BEM_REMOTE_RESUME_SCAN_HOSTS:-1}"
scan_subnets="${BEM_REMOTE_RESUME_SCAN_SUBNETS:-172.16.0.0/22}"
scan_timeout="${BEM_REMOTE_RESUME_SCAN_TIMEOUT:-0.18}"
case_lease_ttl="${BEM_REMOTE_RESUME_CASE_LEASE_TTL:-172800}"
case_guard_args=()
leased_cases=0
queue_status_text=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) run=1; shift ;;
    --dry-run) run=0; shift ;;
    --hosts) host_list="$2"; shift 2 ;;
    --user) user="$2"; shift 2 ;;
    --remote-repo) remote_repo="$2"; shift 2 ;;
    --out|--out-dir) out="$2"; shift 2 ;;
    --gpus) gpu_list="$2"; shift 2 ;;
    --cases) case_filter="$2"; shift 2 ;;
    --max-jobs) max_jobs="$2"; shift 2 ;;
    --max-temp) max_temp="$2"; shift 2 ;;
    --max-util) max_util="$2"; shift 2 ;;
    --max-mem) max_mem="$2"; shift 2 ;;
    --allow-compute-share) allow_compute_share=1; shift ;;
    --sync-launchers) sync_launchers=1; shift ;;
    --no-sync-launchers) sync_launchers=0; shift ;;
    --scan-hosts) scan_hosts=1; shift ;;
    --no-scan-hosts) scan_hosts=0; shift ;;
    --scan-subnets) scan_subnets="$2"; shift 2 ;;
    --scan-timeout) scan_timeout="$2"; shift 2 ;;
    --case-max-power) case_guard_args+=(--case-max-power "$2"); shift 2 ;;
    --case-max-temp) case_guard_args+=(--case-max-temp "$2"); shift 2 ;;
    --case-max-bad-samples) case_guard_args+=(--case-max-bad-samples "$2"); shift 2 ;;
    --ssh-connect-timeout) connect_timeout="$2"; shift 2 ;;
    --ssh-command-timeout) command_timeout="$2"; shift 2 ;;
    --case-lease-ttl) case_lease_ttl="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$host_list" ]]; then
  echo "no hosts supplied; use --hosts or BEM_REMOTE_RESUME_HOSTS" >&2
  usage >&2
  exit 2
fi
normalize_list() {
  local input="$1" item
  input="${input//,/ }"
  for item in $input; do
    [[ -n "$item" ]] && printf '%s\n' "$item"
  done
}

remote_gpu_list() {
  local host="$1" smi_q line
  if [[ "${gpu_list,,}" != "auto" ]]; then
    normalize_list "$gpu_list"
    return
  fi
  smi_q="$(shell_quote "$nvidia_smi")"
  if ! remote_ssh "$host" "$smi_q --query-gpu=index --format=csv,noheader,nounits" 2>/dev/null \
      | awk 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}'; then
    echo "REMOTE_GPU_LIST_SKIP host=$host nvidia-smi_failed" >&2
    return 1
  fi
}

dedupe_preserve_order() {
  awk 'NF && !seen[$0]++'
}

case_lease_dir() {
  printf '%s/remote_case_leases/%s.lock\n' "$out" "$1"
}

case_lease_stale() {
  local lock_dir="$1" mtime now age
  (( case_lease_ttl > 0 )) || return 1
  [[ -d "$lock_dir" ]] || return 1
  mtime="$(stat -c %Y "$lock_dir" 2>/dev/null || printf '0')"
  now="$(date +%s)"
  age=$((now - mtime))
  (( age >= case_lease_ttl ))
}

active_case_lease() {
  local case_name="$1" lock_dir
  (( case_lease_ttl > 0 )) || return 1
  lock_dir="$(case_lease_dir "$case_name")"
  [[ -d "$lock_dir" ]] || return 1
  if case_result_current "$case_name"; then
    rm -rf "$lock_dir"
    echo "REMOTE_CASE_LEASE_DONE case=$case_name removed=$lock_dir"
    return 1
  fi
  if case_lease_stale "$lock_dir"; then
    rm -rf "$lock_dir"
    echo "REMOTE_CASE_LEASE_STALE case=$case_name removed=$lock_dir" >&2
    return 1
  fi
  return 0
}

note_case_lease_skip() {
  local case_name="$1"
  leased_cases=$((leased_cases + 1))
  echo "REMOTE_CASE_LEASE_SKIP case=$case_name active=$(case_lease_dir "$case_name")" >&2
}

acquire_case_lease() {
  local case_name="$1" host="$2" gpu="$3" lock_dir
  (( case_lease_ttl > 0 )) || return 0
  mkdir -p "$out/remote_case_leases"
  lock_dir="$(case_lease_dir "$case_name")"
  if mkdir "$lock_dir" 2>/dev/null; then
    printf 'case=%s host=%s gpu=%s pid=%s started=%s ttl_s=%s\n' \
      "$case_name" "$host" "$gpu" "$$" "$(date -Is)" "$case_lease_ttl" > "$lock_dir/owner"
    echo "REMOTE_CASE_LEASE case=$case_name host=$host gpu=$gpu path=$lock_dir"
    return 0
  fi
  if case_lease_stale "$lock_dir"; then
    rm -rf "$lock_dir"
    if mkdir "$lock_dir" 2>/dev/null; then
      printf 'case=%s host=%s gpu=%s pid=%s started=%s ttl_s=%s\n' \
        "$case_name" "$host" "$gpu" "$$" "$(date -Is)" "$case_lease_ttl" > "$lock_dir/owner"
      echo "REMOTE_CASE_LEASE case=$case_name host=$host gpu=$gpu path=$lock_dir stale_replaced=1"
      return 0
    fi
  fi
  echo "REMOTE_CASE_LEASE_SKIP case=$case_name active=$lock_dir" >&2
  return 1
}

release_case_lease() {
  local case_name="$1" lock_dir
  (( case_lease_ttl > 0 )) || return 0
  lock_dir="$(case_lease_dir "$case_name")"
  rm -rf "$lock_dir"
}

case_queue_state() {
  local case_name="$1" line state name
  while IFS= read -r line; do
    read -r state name _ <<<"$line"
    if [[ "$name" == "$case_name" ]]; then
      printf '%s\n' "$state"
      return 0
    fi
  done <<<"$queue_status_text"
  return 1
}

case_result_current() {
  local state
  state="$(case_queue_state "$1" || true)"
  [[ "$state" == "CURRENT" ]]
}

trim_int() {
  local value="$1"
  value="${value// /}"
  value="${value%%.*}"
  printf '%s\n' "$value"
}

shell_quote() {
  printf '%q' "$1"
}

ssh_target() {
  local host="$1"
  if [[ -n "$user" ]]; then
    printf '%s@%s\n' "$user" "$host"
  else
    printf '%s\n' "$host"
  fi
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

remote_ssh() {
  local host="$1"
  shift
  if [[ -n "${BEM_REMOTE_RESUME_SSH:-}" ]]; then
    "$BEM_REMOTE_RESUME_SSH" "$host" "$@"
  else
    local target
    target="$(ssh_target "$host")"
    local ssh_cmd=(ssh -o BatchMode=yes -o ConnectTimeout="$connect_timeout" \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR "$target" "$@")
    if (( command_timeout > 0 )); then
      timeout "$command_timeout" "${ssh_cmd[@]}"
    else
      "${ssh_cmd[@]}"
    fi
  fi
}

scan_host_candidates() {
  if [[ "${scan_hosts:-0}" != "1" ]]; then
    return 0
  fi
  if [[ -n "${BEM_REMOTE_RESUME_SCAN_OUTPUT:-}" ]]; then
    normalize_list "$BEM_REMOTE_RESUME_SCAN_OUTPUT"
    return 0
  fi
  python3 - "$scan_subnets" "$scan_timeout" <<'PY'
import concurrent.futures
import ipaddress
import socket
import sys

subnets_arg = sys.argv[1].replace(",", " ")
timeout = float(sys.argv[2])
hosts = []
for token in subnets_arg.split():
    try:
        network = ipaddress.ip_network(token, strict=False)
    except ValueError:
        continue
    hosts.extend(str(ip) for ip in network.hosts())

def open_ssh(host: str):
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((host, 22))
        return host
    except OSError:
        return None
    finally:
        try:
            sock.close()
        except OSError:
            pass

with concurrent.futures.ThreadPoolExecutor(max_workers=192) as executor:
    for result in executor.map(open_ssh, hosts):
        if result:
            print(result)
PY
}

discover_hosts() {
  local candidate probe reachable=0
  for candidate in $({ normalize_list "$auto_hosts"; scan_host_candidates; } | dedupe_preserve_order); do
    if probe="$(remote_ssh "$candidate" "printf BEM_REMOTE_OK" 2>/dev/null)" \
        && [[ "$probe" == *BEM_REMOTE_OK* ]]; then
      printf '%s\n' "$candidate"
      reachable=$((reachable + 1))
    fi
  done
  if (( reachable == 0 )); then
    return 1
  fi
}

sync_remote_launchers() {
  local host="$1" repo_path="$2" repo_q target rsync_ssh sync_key
  if [[ "$sync_launchers" != "1" ]]; then
    return 0
  fi
  sync_key="$host|$repo_path"
  if [[ -n "${synced_remote_repos[$sync_key]:-}" ]]; then
    return 0
  fi
  repo_q="$(shell_quote "$repo_path")"
  remote_ssh "$host" "mkdir -p $repo_q/scripts $repo_q/bemcuda" >/dev/null
  if [[ -n "${BEM_REMOTE_RESUME_RSYNC:-}" ]]; then
    if ! "$BEM_REMOTE_RESUME_RSYNC" "$host" "$repo_path"; then
      echo "REMOTE_SYNC_SKIP host=$host repo=$repo_path test_rsync_failed" >&2
      return 1
    fi
    synced_remote_repos[$sync_key]=1
    return
  fi
  target="$(ssh_target "$host")"
  rsync_ssh="ssh -o BatchMode=yes -o ConnectTimeout=$connect_timeout -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
  if ! rsync -av -e "$rsync_ssh" \
    scripts/audit_accuracy_matrix_15.py \
    scripts/check_result_metadata.py \
    scripts/detect_cuda_toolchain.py \
    scripts/gpu_guard.sh \
    scripts/plan_accuracy_refinement_cases.py \
    scripts/remote_refinement_queue_status.py \
    scripts/queue_status_json.py \
    scripts/remote_accuracy_refinement_wave.sh \
    scripts/remote_resume_accuracy_matrix_cases.sh \
    scripts/resume_accuracy_matrix_cases.sh \
    scripts/run_remote_refinement_queue_supervisor.sh \
    scripts/run_accuracy_matrix_15_queue.sh \
    scripts/run_accuracy_matrix_case.sh \
    scripts/run_accuracy_refinement_wave.sh \
    scripts/run_guarded_bem_case.sh \
    scripts/start_remote_refinement_queue_supervisor.sh \
    "$target:$repo_path/scripts/" >/dev/null; then
    echo "REMOTE_SYNC_SKIP host=$host repo=$repo_path launcher_rsync_failed" >&2
    return 1
  fi
  if compgen -G "bemcuda/*.py" >/dev/null; then
    if ! rsync -av -e "$rsync_ssh" bemcuda/*.py "$target:$repo_path/bemcuda/" >/dev/null; then
      echo "REMOTE_SYNC_SKIP host=$host repo=$repo_path python_rsync_failed" >&2
      return 1
    fi
  fi
  remote_ssh "$host" "chmod +x $repo_q/scripts/"'*.sh '"$repo_q/scripts/"'*.py 2>/dev/null || true' >/dev/null
  echo "REMOTE_SYNC host=$host repo=$repo_path launchers=ok"
  synced_remote_repos[$sync_key]=1
}

detect_remote_repo() {
  local host="$1" quoted_repo cmd repo
  if [[ -n "$remote_repo" ]]; then
    quoted_repo="$(shell_quote "$remote_repo")"
    cmd="test -f $quoted_repo/scripts/resume_accuracy_matrix_cases.sh && printf '%s\n' $quoted_repo"
  else
    cmd='for d in "$HOME/BEM-CUDA" /home/kirill_epyc/BEM-CUDA /home/sasha_tvo_gpu1/BEM-CUDA /home/sasha_tvo_gpu2/BEM-CUDA; do test -f "$d/scripts/resume_accuracy_matrix_cases.sh" && { printf "%s\n" "$d"; exit 0; }; done; exit 44'
  fi
  if ! repo="$(remote_ssh "$host" "$cmd" 2>&1 | tail -n 1)"; then
    echo "REMOTE_REPO_SKIP host=$host repo_not_found $repo" >&2
    return 1
  fi
  [[ -n "$repo" ]] || {
    echo "REMOTE_REPO_SKIP host=$host repo_empty" >&2
    return 1
  }
  printf '%s\n' "$repo"
}

remote_binary_ready() {
  local host="$1" repo="$2" repo_q
  repo_q="$(shell_quote "$repo")"
  if remote_ssh "$host" "test -x $repo_q/bin/bem_cuda_fmm.next || test -x $repo_q/bin/bem_cuda_fmm" >/dev/null 2>&1; then
    return 0
  fi
  echo "REMOTE_BINARY_SKIP host=$host repo=$repo missing bin/bem_cuda_fmm.next and bin/bem_cuda_fmm" >&2
  return 1
}

gpu_idle_remote() {
  local host="$1" gpu="$2" line temp util mem power name smi_q apps apps_cmd
  smi_q="$(shell_quote "$nvidia_smi")"
  if ! line="$(remote_ssh "$host" \
      "$smi_q -i '$gpu' --query-gpu=index,name,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader,nounits" 2>&1)"; then
    echo "REMOTE_GPU_SKIP host=$host gpu=$gpu nvidia-smi_failed $line" >&2
    return 1
  fi
  line="$(printf '%s\n' "$line" | tail -n 1)"
  IFS=',' read -r _ name temp power mem util <<<"$line"
  temp="$(trim_int "$temp")"
  util="$(trim_int "$util")"
  mem="$(trim_int "$mem")"
  power="$(trim_int "$power")"
  if [[ -z "$temp" || -z "$util" || -z "$mem" ]]; then
    echo "REMOTE_GPU_SKIP host=$host gpu=$gpu unparsable $line" >&2
    return 1
  fi
  if [[ "$allow_compute_share" != "1" ]]; then
    apps_cmd="$smi_q -i '$gpu' --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits"
    apps="$(remote_ssh "$host" "$apps_cmd" 2>/dev/null || true)"
    apps="$(printf '%s\n' "$apps" | sed '/^[[:space:]]*$/d' | head -n 3)"
    apps="${apps//$'\n'/; }"
  fi
  if (( temp > max_temp || util > max_util || mem > max_mem )); then
    if [[ -n "$apps" ]]; then
      echo "REMOTE_GPU_BUSY host=$host gpu=$gpu temp=${temp}C util=${util}% mem=${mem}MiB power=${power}W compute_apps=$apps" >&2
    else
      echo "REMOTE_GPU_BUSY host=$host gpu=$gpu temp=${temp}C util=${util}% mem=${mem}MiB power=${power}W" >&2
    fi
    return 1
  fi
  if [[ "$allow_compute_share" != "1" ]]; then
    if [[ -n "$apps" ]]; then
      echo "REMOTE_GPU_BUSY host=$host gpu=$gpu compute_apps=$apps" >&2
      return 1
    fi
  fi
  printf '%s %s %s %s %s %s\n' "$host" "$gpu" "$temp" "$util" "$mem" "${name#"${name%%[![:space:]]*}"}"
}

collect_cases() {
  local normalized selected line case_name
  if [[ -n "$case_filter" ]]; then
    normalize_list "$case_filter"
    return
  fi
  while IFS= read -r line; do
    case "$line" in
      DRYRUN\ gpu=*)
        case_name="${line#* case=}"
        case_name="${case_name%% cmd=*}"
        [[ -n "$case_name" ]] && printf '%s\n' "$case_name"
        ;;
    esac
  done < <(scripts/resume_accuracy_matrix_cases.sh --dry-run --no-health-check \
      --gpus 0 --allow-oversubscribe --max-jobs "${max_jobs:-0}" --out "$out" || true)
}

if [[ "${host_list,,}" == "auto" ]]; then
  if ! resolved_hosts="$(discover_hosts | dedupe_preserve_order | paste -sd' ' -)"; then
    echo "REMOTE_HOST_AUTO no reachable hosts from BEM_REMOTE_RESUME_AUTO_HOSTS" >&2
    exit 3
  fi
  host_list="$resolved_hosts"
  echo "REMOTE_HOST_AUTO hosts=$host_list"
fi

if [[ -n "${BEM_REMOTE_RESUME_STATUS_TEXT:-}" ]]; then
  queue_status_text="$BEM_REMOTE_RESUME_STATUS_TEXT"
else
  queue_status_text="$(OUT="$out" scripts/run_accuracy_matrix_15_queue.sh --status || true)"
fi

if [[ -n "$case_filter" ]]; then
  while IFS= read -r selected; do
    validate_case_name "$selected"
  done < <(normalize_list "$case_filter")
fi

mapfile -t candidate_cases < <(collect_cases | dedupe_preserve_order)
cases=()
for candidate_case in "${candidate_cases[@]}"; do
  if active_case_lease "$candidate_case"; then
    note_case_lease_skip "$candidate_case"
  else
    cases+=("$candidate_case")
  fi
done
mapfile -t remote_gpus < <(
  for host in $(normalize_list "$host_list"); do
    for gpu in $(remote_gpu_list "$host"); do
      gpu_idle_remote "$host" "$gpu" || true
    done
  done
)
remote_targets=()
declare -A synced_remote_repos=()
for remote_gpu in "${remote_gpus[@]}"; do
  read -r host gpu _ <<<"$remote_gpu"
  if repo_path="$(detect_remote_repo "$host")" \
      && sync_remote_launchers "$host" "$repo_path" \
      && remote_binary_ready "$host" "$repo_path"; then
    remote_targets+=("$host $gpu $repo_path")
  fi
done

echo "REMOTE_RESUME cases=${#cases[@]} leased_cases=$leased_cases idle_remote_gpus=${#remote_gpus[@]} usable_remote_gpus=${#remote_targets[@]} mode=$([[ "$run" == "1" ]] && echo run || echo dry-run)"
if [[ "${#cases[@]}" -eq 0 ]]; then
  echo "REMOTE_RESUME selected=0"
  exit 0
fi
if [[ "${#remote_targets[@]}" -eq 0 ]]; then
  echo "REMOTE_RESUME no usable remote GPUs" >&2
  exit 3
fi

limit="${#remote_targets[@]}"
if (( max_jobs > 0 && max_jobs < limit )); then
  limit="$max_jobs"
fi
if (( limit > ${#cases[@]} )); then
  limit="${#cases[@]}"
fi

started=0
for case_name in "${cases[@]}"; do
  if (( started >= limit )); then
    break
  fi
  read -r host gpu repo_path <<<"${remote_targets[$started]}"
  remote_cmd=(cd "$repo_path" "&&" env BEM_NO_AUTO_MGPU=1 bash scripts/resume_accuracy_matrix_cases.sh --run
    --out "$out" --gpus "$gpu" --max-jobs 1 --cases "$case_name"
    --no-health-check "${case_guard_args[@]}")
  if [[ "$run" == "1" ]]; then
    if ! acquire_case_lease "$case_name" "$host" "$gpu"; then
      continue
    fi
    set +e
    remote_output="$(remote_ssh "$host" "$(printf '%q ' "${remote_cmd[@]}")" 2>&1)"
    remote_rc="$?"
    set -e
    printf '%s\n' "$remote_output"
    if [[ "$remote_rc" != "0" ]]; then
      echo "REMOTE_START_FAILED host=$host gpu=$gpu case=$case_name rc=$remote_rc" >&2
      release_case_lease "$case_name"
      exit "$remote_rc"
    fi
    if printf '%s\n' "$remote_output" | grep -q '^STARTED '; then
      printf 'REMOTE_START host=%s gpu=%s case=%s\n' "$host" "$gpu" "$case_name"
    else
      printf 'REMOTE_CASE_SKIP host=%s gpu=%s case=%s no_remote_start\n' "$host" "$gpu" "$case_name"
      release_case_lease "$case_name"
      continue
    fi
  else
    target="$(ssh_target "$host")"
    printf 'REMOTE_DRYRUN host=%s gpu=%s case=%s cmd=ssh %q %q\n' \
      "$host" "$gpu" "$case_name" "$target" "$(printf '%q ' "${remote_cmd[@]}")"
  fi
  started=$((started + 1))
done

echo "REMOTE_RESUME selected=$started"
