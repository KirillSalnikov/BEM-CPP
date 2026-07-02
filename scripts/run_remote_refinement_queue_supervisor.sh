#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_remote_refinement_queue_supervisor.sh [options]

Keeps the remote accuracy-refinement queue alive. Each queue wave still starts
at most one different case per usable remote GPU. GPU exhaustion is treated as
normal queue waiting; other nonzero exits are treated as wrapper restarts.

Options:
  --queue-dir DIR        Local supervisor/log directory
                         (default: runs/remote_refinement_queue)
  --hosts LIST           Remote hosts for the queue, or auto (default: auto)
  --gpus LIST            Remote GPU ids, or auto (default: auto)
  --out DIR              Remote results directory
                         (default: runs/production_matrix_refinement)
  --case-max-power W     Guard power limit passed to each case (default: 200)
  --min-free-gpus N      Minimum usable remote GPUs considered enough
                         by the queue status (default: 1)
  --queue-interval SEC   Poll interval inside the queue wrapper (default: 60)
  --restart-interval SEC Sleep before restarting after non-GPU nonzero exit
                         (default: 10)
  --no-sync-launchers    Do not sync local launcher scripts before starts
  --scan-hosts           Let remote host auto-discovery scan the subnet
  --no-scan-hosts        Use only known/explicit host candidates (default)
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO:-$(cd "$script_dir/.." && pwd)}"
cd "$repo"

queue_dir="runs/remote_refinement_queue"
hosts="auto"
gpus="auto"
out="runs/production_matrix_refinement"
case_max_power=200
min_free_gpus=1
queue_interval=60
restart_interval=10
sync_launchers=1
scan_hosts=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --queue-dir) queue_dir="$2"; shift 2 ;;
    --hosts) hosts="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --out|--out-dir) out="$2"; shift 2 ;;
    --case-max-power) case_max_power="$2"; shift 2 ;;
    --min-free-gpus) min_free_gpus="$2"; shift 2 ;;
    --queue-interval) queue_interval="$2"; shift 2 ;;
    --restart-interval) restart_interval="$2"; shift 2 ;;
    --sync-launchers) sync_launchers=1; shift ;;
    --no-sync-launchers) sync_launchers=0; shift ;;
    --scan-hosts) scan_hosts=1; shift ;;
    --no-scan-hosts) scan_hosts=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$queue_dir"
printf '%s\n' "$$" > "$queue_dir/supervisor.pid"

echo "REMOTE_QUEUE_SUPERVISOR start pid=$$ queue_dir=$queue_dir hosts=$hosts gpus=$gpus out=$out min_free_gpus=$min_free_gpus"
trap 'echo "REMOTE_QUEUE_SUPERVISOR signal=HUP"; exit 129' HUP
trap 'echo "REMOTE_QUEUE_SUPERVISOR signal=INT"; exit 130' INT
trap 'echo "REMOTE_QUEUE_SUPERVISOR signal=TERM"; exit 143' TERM
attempt=1
while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "REMOTE_QUEUE_SUPERVISOR attempt=$attempt start_time=\"$ts\""
  args=(
    --run
    --continuous
    --hosts "$hosts"
    --gpus "$gpus"
    --queue-interval "$queue_interval"
    --queue-timeout 0
    --min-free-gpus "$min_free_gpus"
    --case-max-power "$case_max_power"
    --out "$out"
    --status-json "$queue_dir/status.json"
    --plan-csv "$queue_dir/plan.csv"
  )
  if [[ "$sync_launchers" == "1" ]]; then
    args+=(--sync-launchers)
  else
    args+=(--no-sync-launchers)
  fi
  if [[ "$scan_hosts" == "1" ]]; then
    args+=(--scan-hosts)
  else
    args+=(--no-scan-hosts)
  fi

  set +e
  bash scripts/remote_accuracy_refinement_wave.sh "${args[@]}"
  rc="$?"
  set -e
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "REMOTE_QUEUE_SUPERVISOR attempt=$attempt rc=$rc end_time=\"$ts\""
  if [[ "$rc" == "0" ]]; then
    echo "REMOTE_QUEUE_SUPERVISOR done"
    exit 0
  fi
  if [[ "$rc" == "3" ]]; then
    echo "REMOTE_QUEUE_SUPERVISOR gpu_wait sleep=${queue_interval}s"
    attempt=$((attempt + 1))
    sleep "$queue_interval"
    continue
  fi
  echo "REMOTE_QUEUE_SUPERVISOR restart_after=${restart_interval}s"
  attempt=$((attempt + 1))
  sleep "$restart_interval"
done
