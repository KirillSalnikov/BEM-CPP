#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  fetch_accuracy_matrix_15_results.sh [options]

Fetches production/refinement accuracy-matrix results from one or more remote
BEM-CUDA hosts, then runs the local accuracy audit.

Options:
  --hosts LIST           Space/comma-separated hosts, or auto (default: auto)
  --user USER            SSH user. Empty default lets SSH aliases work.
  --remote-repo DIR      Remote BEM-CUDA path; default auto-detects per host
  --out-dirs LIST        Space/comma-separated remote result dirs
                         (default: production_matrix_refinement and matrix_15)
  --local-repo DIR       Local repo path (default: this repo)
  --scan-hosts           With --hosts auto, scan 172.16.0.0/22 for SSH hosts
  --no-scan-hosts        With --hosts auto, use known candidates only (default)
  --ssh-connect-timeout S
                         SSH connect timeout (default: 5)
  --no-audit             Fetch only; do not run local audit
  --strict-audit         Return the local audit exit code after fetching (default)
  --audit-best-effort    Print the local audit exit code but return success
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
local_repo="${LOCAL_REPO:-$(cd "$script_dir/.." && pwd)}"
cd "$local_repo"

hosts="${BEM_FETCH_HOSTS:-auto}"
user="${USER_REMOTE:-}"
remote_repo="${REMOTE_REPO:-}"
out_dirs="${BEM_FETCH_OUT_DIRS:-runs/production_matrix_refinement runs/production_matrix_15}"
nvidia_smi="${BEM_NVIDIA_SMI:-nvidia-smi}"
scan_hosts="${BEM_FETCH_SCAN_HOSTS:-0}"
connect_timeout="${BEM_FETCH_CONNECT_TIMEOUT:-5}"
run_audit=1
strict_audit="${BEM_FETCH_STRICT_AUDIT:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hosts) hosts="$2"; shift 2 ;;
    --user) user="$2"; shift 2 ;;
    --remote-repo) remote_repo="$2"; shift 2 ;;
    --out-dirs|--out-dir) out_dirs="$2"; shift 2 ;;
    --local-repo) local_repo="$2"; shift 2 ;;
    --scan-hosts) scan_hosts=1; shift ;;
    --no-scan-hosts) scan_hosts=0; shift ;;
    --ssh-connect-timeout) connect_timeout="$2"; shift 2 ;;
    --no-audit) run_audit=0; shift ;;
    --strict-audit) strict_audit=1; shift ;;
    --audit-best-effort) strict_audit=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$local_repo"

ssh_opts=(
  -o BatchMode=yes
  -o ConnectTimeout="$connect_timeout"
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
)
rsync_ssh="ssh -o BatchMode=yes -o ConnectTimeout=$connect_timeout -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

known_hosts="${BEM_REMOTE_RESUME_AUTO_HOSTS:-gpu1 gpu2 gpu3 172.16.1.222 172.16.0.73 172.16.0.212 172.16.1.168 172.16.1.149 epyc1}"

normalize_list() {
  local input="$1" item
  input="${input//,/ }"
  for item in $input; do
    [[ -n "$item" ]] && printf '%s\n' "$item"
  done
}

dedupe_preserve_order() {
  awk 'NF && !seen[$0]++'
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

remote_ssh() {
  local host="$1"
  shift
  ssh "${ssh_opts[@]}" "$(ssh_target "$host")" "$@"
}

scan_host_candidates() {
  if [[ "$scan_hosts" != "1" ]]; then
    return 0
  fi
  python3 - <<'PY'
import concurrent.futures
import socket

hosts = [f"172.16.{a}.{b}" for a in range(0, 4) for b in range(1, 255)]

def check(host):
    sock = socket.socket()
    sock.settimeout(0.15)
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
    for result in executor.map(check, hosts):
        if result:
            print(result)
PY
}

discover_hosts() {
  local candidate probe
  if [[ "${hosts,,}" != "auto" ]]; then
    normalize_list "$hosts" | dedupe_preserve_order
    return
  fi
  for candidate in $({ normalize_list "$known_hosts"; scan_host_candidates; } | dedupe_preserve_order); do
    if probe="$(remote_ssh "$candidate" "printf BEM_REMOTE_OK" 2>/dev/null)" \
        && [[ "$probe" == *BEM_REMOTE_OK* ]]; then
      printf '%s\n' "$candidate"
    fi
  done
}

detect_remote_repo() {
  local host="$1" repo_q cmd repo
  if [[ -n "$remote_repo" ]]; then
    repo_q="$(shell_quote "$remote_repo")"
    cmd="test -d $repo_q && printf '%s\n' $repo_q"
  else
    cmd='for d in "$HOME/BEM-CUDA" /home/kirill_epyc/BEM-CUDA /home/sasha_tvo_gpu1/BEM-CUDA /home/sasha_tvo_gpu2/BEM-CUDA; do test -d "$d" && { printf "%s\n" "$d"; exit 0; }; done; exit 44'
  fi
  if ! repo="$(remote_ssh "$host" "$cmd" 2>/dev/null | tail -n 1)"; then
    echo "FETCH_REPO_SKIP host=$host repo_not_found" >&2
    return 1
  fi
  [[ -n "$repo" ]] || return 1
  printf '%s\n' "$repo"
}

mapfile -t resolved_hosts < <(discover_hosts)
if [[ "${#resolved_hosts[@]}" -eq 0 ]]; then
  echo "FETCH no reachable hosts" >&2
  exit 3
fi

echo "FETCH_HOSTS ${resolved_hosts[*]}"
fetched_any=0
for host in "${resolved_hosts[@]}"; do
  if ! repo="$(detect_remote_repo "$host")"; then
    continue
  fi
  echo "FETCH_REMOTE host=$host repo=$repo"
  repo_q="$(shell_quote "$repo")"
  smi_q="$(shell_quote "$nvidia_smi")"
  remote_ssh "$host" "cd $repo_q && \
    echo ===PROCS=== && ps -eo pid,etime,args | egrep 'run_accuracy_matrix|remote_accuracy_refinement|bem_cuda_fmm|adda_ocl' | grep -v egrep || true && \
    echo ===GPU=== && if command -v $smi_q >/dev/null 2>&1; then $smi_q --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits || true; else echo nvidia-smi-missing; fi && \
    echo ===FILES=== && for d in ${out_dirs}; do ls -lh \"\$d\"/*.json 2>/dev/null || true; done && \
    echo ===AUDIT=== && { python3 scripts/audit_accuracy_matrix_15.py 2>&1; rc=\$?; echo REMOTE_AUDIT_RC=\$rc; } || true"

  for out_dir in $(normalize_list "$out_dirs"); do
    mkdir -p "$local_repo/$out_dir"
    out_q="$(shell_quote "$repo/$out_dir")"
    if ! remote_ssh "$host" "test -d $out_q" >/dev/null 2>&1; then
      echo "FETCH_DIR_MISSING host=$host dir=$out_dir"
      continue
    fi
    if rsync -av --ignore-missing-args -e "$rsync_ssh" \
        "$(ssh_target "$host"):$repo/$out_dir/" \
        "$local_repo/$out_dir/"; then
      fetched_any=1
    else
      echo "FETCH_RSYNC_SKIP host=$host dir=$out_dir" >&2
    fi
  done
done

mkdir -p "$local_repo/runs/adda_ocl_benchmark_ext/dust_ka15_m1p6_dpl20_scaled"
for host in "${resolved_hosts[@]}"; do
  repo="$(detect_remote_repo "$host" 2>/dev/null || true)"
  [[ -n "$repo" ]] || continue
  adda_dir="$repo/runs/adda_ocl_benchmark_ext/dust_ka15_m1p6_dpl20_scaled"
  adda_q="$(shell_quote "$adda_dir")"
  remote_ssh "$host" "test -d $adda_q" >/dev/null 2>&1 || continue
  rsync -av --ignore-missing-args -e "$rsync_ssh" \
    "$(ssh_target "$host"):$adda_dir/" \
    "$local_repo/runs/adda_ocl_benchmark_ext/dust_ka15_m1p6_dpl20_scaled/" || true
done

if [[ "$fetched_any" != "1" ]]; then
  echo "FETCH no result directories copied" >&2
fi

if [[ "$run_audit" == "1" ]]; then
  set +e
  python3 scripts/audit_accuracy_matrix_15.py
  audit_rc=$?
  set -e
  echo "FETCH_AUDIT_RC $audit_rc"
  if [[ "$strict_audit" == "1" && "$audit_rc" != "0" ]]; then
    exit "$audit_rc"
  fi
fi
