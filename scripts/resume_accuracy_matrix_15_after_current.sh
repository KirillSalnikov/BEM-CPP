#!/usr/bin/env bash
set -euo pipefail

user="${USER_REMOTE:-kirill_epyc}"
remote="${REMOTE_HOST:-172.16.1.222}"
remote_repo="${REMOTE_REPO:-/home/kirill_epyc/BEM-CUDA}"
local_repo="${LOCAL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
wait_interval="${WAIT_INTERVAL_S:-60}"
max_wait="${MAX_WAIT_S:-0}"
deploy_script="$local_repo/scripts/deploy_accuracy_matrix_15_queue.sh"
rsync_ssh="ssh -o BatchMode=yes -o ConnectTimeout=6 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

ssh_opts=(
  -o BatchMode=yes
  -o ConnectTimeout=6
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
)

usage() {
  cat <<EOF
Usage: $0 [--status-only] [--wait-and-resume] [--install-remote-watcher]

Waits for a currently running remote production_matrix_15 queue to finish, then
runs deploy_accuracy_matrix_15_queue.sh so missing/stale cases resume with the
current local queue script.

Environment:
  REMOTE_HOST=${remote}
  USER_REMOTE=${user}
  REMOTE_REPO=${remote_repo}
  WAIT_INTERVAL_S=${wait_interval}
  MAX_WAIT_S=${max_wait}   # 0 means no timeout
EOF
}

remote_queue_state() {
  ssh "${ssh_opts[@]}" "$user@$remote" "cd '$remote_repo' 2>/dev/null || exit 3; \
    if [ -f runs/production_matrix_15/queue.pid ]; then \
      pid=\$(cat runs/production_matrix_15/queue.pid); \
      if kill -0 \$pid 2>/dev/null; then \
        echo RUNNING:\$pid; \
      else \
        echo DEAD_PID:\$pid; \
      fi; \
    else \
      echo NO_PID; \
    fi; \
    bash scripts/run_accuracy_matrix_15_queue.sh --status 2>/dev/null || true"
}

queue_is_running() {
  local state
  state="$(remote_queue_state | sed -n '1p')"
  [[ "$state" == RUNNING:* ]]
}

status_only() {
  remote_queue_state
}

wait_and_resume() {
  local start now elapsed state
  start="$(date +%s)"
  while true; do
    state="$(remote_queue_state | sed -n '1p')"
    echo "QUEUE_STATE $state"
    if [[ "$state" != RUNNING:* ]]; then
      break
    fi
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( max_wait > 0 && elapsed >= max_wait )); then
      echo "Timed out after ${elapsed}s while waiting for current queue." >&2
      return 124
    fi
    sleep "$wait_interval"
  done

  REMOTE_HOST="$remote" USER_REMOTE="$user" REMOTE_REPO="$remote_repo" \
    "$deploy_script"
}

mode="${1:---status-only}"
case "$mode" in
  --status-only)
    status_only
    ;;
  --wait-and-resume)
    wait_and_resume
    ;;
  --install-remote-watcher)
    rsync -av -e "$rsync_ssh" \
      "$local_repo/scripts/run_accuracy_matrix_15_queue.sh" \
      "$user@$remote:$remote_repo/scripts/run_accuracy_matrix_15_queue.next.sh"
    rsync -av -e "$rsync_ssh" \
      "$local_repo/scripts/audit_accuracy_matrix_15.py" \
      "$local_repo/scripts/gpu_guard.sh" \
      "$local_repo/scripts/check_result_metadata.py" \
      "$local_repo/scripts/queue_status_json.py" \
      "$local_repo/scripts/queue_watch_once.sh" \
      "$local_repo/scripts/resume_accuracy_matrix_cases.sh" \
      "$local_repo/scripts/run_accuracy_matrix_case.sh" \
      "$local_repo/scripts/run_guarded_bem_case.sh" \
      "$local_repo/scripts/remote_power_watch.sh" \
      "$local_repo/scripts/remote_resume_accuracy_matrix_cases.sh" \
      "$local_repo/scripts/summarize_gpu_power_monitor.py" \
      "$user@$remote:$remote_repo/scripts/"
    ssh "${ssh_opts[@]}" "$user@$remote" \
      "REMOTE_REPO=$(printf '%q' "$remote_repo") WAIT_INTERVAL_S=$(printf '%q' "$wait_interval") bash -s" <<'REMOTE'
set -euo pipefail
cd "$REMOTE_REPO"
chmod +x scripts/run_accuracy_matrix_15_queue.next.sh \
  scripts/audit_accuracy_matrix_15.py scripts/gpu_guard.sh scripts/check_result_metadata.py \
  scripts/queue_status_json.py scripts/queue_watch_once.sh \
  scripts/resume_accuracy_matrix_cases.sh scripts/run_accuracy_matrix_case.sh \
  scripts/run_guarded_bem_case.sh scripts/remote_power_watch.sh \
  scripts/remote_resume_accuracy_matrix_cases.sh scripts/summarize_gpu_power_monitor.py
export REMOTE_REPO WAIT_INTERVAL_S
if [ -f runs/production_matrix_15/resume_after_current.pid ] \
    && kill -0 "$(cat runs/production_matrix_15/resume_after_current.pid)" 2>/dev/null; then
  echo "REMOTE_RESUME_ALREADY_RUNNING=$(cat runs/production_matrix_15/resume_after_current.pid)"
  exit 0
fi
nohup bash -s > runs/production_matrix_15/resume_after_current.log 2>&1 <<'WATCHER' &
set -euo pipefail
cd "$REMOTE_REPO"
while [ -f runs/production_matrix_15/queue.pid ] \
    && kill -0 "$(cat runs/production_matrix_15/queue.pid)" 2>/dev/null; do
  echo "WAIT_QUEUE $(date -Is) pid=$(cat runs/production_matrix_15/queue.pid)"
  sleep "$WAIT_INTERVAL_S"
done
mv scripts/run_accuracy_matrix_15_queue.next.sh scripts/run_accuracy_matrix_15_queue.sh
chmod +x scripts/run_accuracy_matrix_15_queue.sh
bash scripts/run_accuracy_matrix_15_queue.sh --preflight \
  > runs/production_matrix_15/preflight.resume.log 2>&1
(
  nohup bash scripts/run_accuracy_matrix_15_queue.sh \
    > runs/production_matrix_15/queue.resume.nohup.log 2>&1 &
  echo $! > runs/production_matrix_15/queue.pid
)
echo "RESUME_STARTED $(cat runs/production_matrix_15/queue.pid) $(date -Is)"
WATCHER
echo $! > runs/production_matrix_15/resume_after_current.pid
echo "REMOTE_RESUME_STARTED=$(cat runs/production_matrix_15/resume_after_current.pid)"
REMOTE
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "unknown argument: $mode" >&2
    usage >&2
    exit 2
    ;;
esac
