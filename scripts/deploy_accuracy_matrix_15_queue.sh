#!/usr/bin/env bash
set -euo pipefail

user="${USER_REMOTE:-}"
remote_repo="${REMOTE_REPO:-}"
local_repo="${LOCAL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
start_queue="${START_QUEUE:-1}"
start_power_watch="${START_POWER_WATCH:-1}"
run_preflight="${RUN_PREFLIGHT:-1}"
sync_code="${SYNC_CODE:-1}"
build_remote="${BUILD_REMOTE:-1}"
nvidia_smi="${BEM_NVIDIA_SMI:-nvidia-smi}"
ssh_opts=(
  -o BatchMode=yes
  -o ConnectTimeout=5
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
)
rsync_ssh="ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

known_hosts=(
  gpu1
  gpu2
  172.16.1.222
  172.16.0.73
  172.16.0.212
  172.16.1.168
  172.16.1.149
  epyc1
)

ssh_target() {
  local host="$1"
  if [[ -n "$user" ]]; then
    printf '%s@%s\n' "$user" "$host"
  else
    printf '%s\n' "$host"
  fi
}

quote_remote() {
  printf '%q' "$1"
}

remote_repo_probe_cmd() {
  if [[ -n "$remote_repo" ]]; then
    printf 'test -d %s && printf "%%s\\n" %s' "$(quote_remote "$remote_repo")" "$(quote_remote "$remote_repo")"
  else
    printf 'test -d "$HOME/BEM-CUDA" && printf "%%s\\n" "$HOME/BEM-CUDA"'
  fi
}

scan_hosts() {
  python3 - <<'PY'
import socket, concurrent.futures
hosts=[f"172.16.{a}.{b}" for a in range(0,4) for b in range(1,255)]
def check(h):
    s=socket.socket(); s.settimeout(0.15)
    try:
        s.connect((h,22)); s.close(); return h
    except Exception:
        return None
with concurrent.futures.ThreadPoolExecutor(max_workers=192) as ex:
    for r in ex.map(check, hosts):
        if r:
            print(r)
PY
}

try_host() {
  local host="$1" target smi_q
  target="$(ssh_target "$host")"
  smi_q="$(quote_remote "$nvidia_smi")"
  ssh "${ssh_opts[@]}" "$target" \
    "$(remote_repo_probe_cmd) >/dev/null && command -v $smi_q >/dev/null && hostname" 2>/dev/null
}

remote="${REMOTE_HOST:-}"
if [[ -n "$remote" ]]; then
  if ! try_host "$remote" >/tmp/bem_cuda_deploy_host.$$; then
    echo "Requested REMOTE_HOST=$remote is not a reachable BEM-CUDA GPU host." >&2
    rm -f /tmp/bem_cuda_deploy_host.$$
    exit 2
  fi
fi

for host in "${known_hosts[@]}"; do
  [[ -z "$remote" ]] || break
  if try_host "$host" >/tmp/bem_cuda_deploy_host.$$; then
    remote="$host"
    break
  fi
done

if [[ -z "$remote" ]]; then
  while read -r host; do
    [[ -z "$host" ]] && continue
    if try_host "$host" >/tmp/bem_cuda_deploy_host.$$; then
      remote="$host"
      break
    fi
  done < <(scan_hosts)
fi
rm -f /tmp/bem_cuda_deploy_host.$$

if [[ -z "$remote" ]]; then
  echo "No reachable BEM-CUDA GPU host found on known addresses or 172.16.0.0/22." >&2
  exit 2
fi

target="$(ssh_target "$remote")"
if [[ -z "$remote_repo" ]]; then
  remote_repo="$(ssh "${ssh_opts[@]}" "$target" "$(remote_repo_probe_cmd)" | tail -n 1)"
fi
if [[ -z "$remote_repo" ]]; then
  echo "Could not determine remote repo on $remote." >&2
  exit 2
fi

echo "REMOTE=$remote"
echo "REMOTE_TARGET=$target"
echo "REMOTE_REPO=$remote_repo"
remote_repo_q="$(quote_remote "$remote_repo")"
ssh "${ssh_opts[@]}" "$target" "mkdir -p $remote_repo_q/scripts $remote_repo_q/src $remote_repo_q/bin \
  '$remote_repo/runs/production_matrix_15/meshes/dust5_adda_shape' \
  '$remote_repo/runs/adda_ocl_benchmark_ext/shapes'"

if [[ "$sync_code" == "1" ]]; then
  rsync -av -e "$rsync_ssh" \
    "$local_repo/Makefile" \
    "$local_repo/environment.cuda.yml" \
    "$target:$remote_repo/"
  rsync -av --delete -e "$rsync_ssh" \
    --include='*/' \
    --include='*.cu' \
    --include='*.cpp' \
    --include='*.h' \
    --exclude='*' \
    "$local_repo/src/" \
    "$target:$remote_repo/src/"
fi

rsync -av -e "$rsync_ssh" \
  "$local_repo/scripts/build_cuda_fmm.sh" \
  "$local_repo/scripts/gpu_guard.sh" \
  "$local_repo/scripts/audit_accuracy_matrix_15.py" \
  "$local_repo/scripts/check_result_metadata.py" \
  "$local_repo/scripts/detect_cuda_toolchain.py" \
  "$local_repo/scripts/ipmi_power_control.sh" \
  "$local_repo/scripts/plan_accuracy_refinement_cases.py" \
  "$local_repo/scripts/queue_live_status.sh" \
  "$local_repo/scripts/remote_refinement_queue_status.py" \
  "$local_repo/scripts/queue_status_json.py" \
  "$local_repo/scripts/queue_watch_once.sh" \
  "$local_repo/scripts/resume_accuracy_matrix_cases.sh" \
  "$local_repo/scripts/run_accuracy_matrix_15_queue.sh" \
  "$local_repo/scripts/run_accuracy_matrix_case.sh" \
  "$local_repo/scripts/run_guarded_bem_case.sh" \
  "$local_repo/scripts/remote_power_watch.sh" \
  "$local_repo/scripts/remote_accuracy_refinement_wave.sh" \
  "$local_repo/scripts/remote_resume_accuracy_matrix_cases.sh" \
  "$local_repo/scripts/run_accuracy_refinement_wave.sh" \
  "$local_repo/scripts/run_remote_refinement_queue_supervisor.sh" \
  "$local_repo/scripts/start_remote_refinement_queue_supervisor.sh" \
  "$local_repo/scripts/summarize_gpu_power_monitor.py" \
  "$target:$remote_repo/scripts/"
rsync -av -e "$rsync_ssh" \
  "$local_repo/runs/production_matrix_15/meshes/dust5_adda_shape/" \
  "$target:$remote_repo/runs/production_matrix_15/meshes/dust5_adda_shape/"
rsync -av -e "$rsync_ssh" \
  "$local_repo/runs/adda_ocl_benchmark_ext/shapes/greek_scaled_ka15_dpl20.shape" \
  "$target:$remote_repo/runs/adda_ocl_benchmark_ext/shapes/"

ssh "${ssh_opts[@]}" "$target" "cd $remote_repo_q && \
  chmod +x scripts/build_cuda_fmm.sh scripts/gpu_guard.sh scripts/audit_accuracy_matrix_15.py scripts/check_result_metadata.py scripts/detect_cuda_toolchain.py scripts/plan_accuracy_refinement_cases.py scripts/queue_status_json.py scripts/queue_watch_once.sh scripts/resume_accuracy_matrix_cases.sh scripts/run_accuracy_matrix_15_queue.sh scripts/run_accuracy_matrix_case.sh scripts/run_guarded_bem_case.sh scripts/remote_power_watch.sh && \
  chmod +x scripts/remote_resume_accuracy_matrix_cases.sh && \
  chmod +x scripts/remote_accuracy_refinement_wave.sh scripts/run_accuracy_refinement_wave.sh scripts/run_remote_refinement_queue_supervisor.sh scripts/start_remote_refinement_queue_supervisor.sh scripts/remote_refinement_queue_status.py && \
  mkdir -p runs/production_matrix_15 runs/monitoring && \
  if [ '$build_remote' = '1' ]; then \
    TARGET_FMM=bin/bem_cuda_fmm.next scripts/build_cuda_fmm.sh > runs/production_matrix_15/build.last.log 2>&1; \
    sha256sum bin/bem_cuda_fmm.next > runs/production_matrix_15/build.last.sha256; \
  else \
    echo BUILD_REMOTE_SKIPPED; \
  fi && \
  bash scripts/run_accuracy_matrix_15_queue.sh --plan > runs/production_matrix_15/queue.plan && \
  if [ '$run_preflight' = '1' ]; then \
    bash scripts/run_accuracy_matrix_15_queue.sh --preflight > runs/production_matrix_15/preflight.last.log 2>&1; \
  else \
    echo PREFLIGHT_SKIPPED; \
  fi && \
  if [ '$start_queue' != '1' ]; then \
    echo QUEUE_START_SKIPPED; \
  elif [ -f runs/production_matrix_15/queue.pid ] && kill -0 \$(cat runs/production_matrix_15/queue.pid) 2>/dev/null; then \
    echo QUEUE_ALREADY_RUNNING=\$(cat runs/production_matrix_15/queue.pid); \
  else \
    (nohup bash scripts/run_accuracy_matrix_15_queue.sh > runs/production_matrix_15/queue.nohup.log 2>&1 & echo \$! > runs/production_matrix_15/queue.pid); \
    echo QUEUE_STARTED=\$(cat runs/production_matrix_15/queue.pid); \
  fi && \
  if [ '$start_power_watch' != '1' ]; then \
    echo POWER_WATCH_START_SKIPPED; \
  elif [ -f runs/monitoring/power_watch.pid ] && kill -0 \$(cat runs/monitoring/power_watch.pid) 2>/dev/null; then \
    echo POWER_WATCH_ALREADY_RUNNING=\$(cat runs/monitoring/power_watch.pid); \
  else \
    (nohup scripts/remote_power_watch.sh 10 > runs/monitoring/power_watch.log 2>&1 & echo \$! > runs/monitoring/power_watch.pid); \
    echo POWER_WATCH_STARTED=\$(cat runs/monitoring/power_watch.pid); \
  fi"
