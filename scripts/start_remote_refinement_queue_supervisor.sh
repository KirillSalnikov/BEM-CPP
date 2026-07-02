#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  start_remote_refinement_queue_supervisor.sh [supervisor options]

Starts run_remote_refinement_queue_supervisor.sh in a detached session. The
default options are conservative for the flaky multi-GPU host: known hosts only,
one case per free GPU, and 200 W per-case guard power.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO:-$(cd "$script_dir/.." && pwd)}"
cd "$repo"

queue_dir="runs/remote_refinement_queue"
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --queue-dir)
      if (( i + 1 >= ${#args[@]} )); then
        echo "--queue-dir requires a value" >&2
        exit 2
      fi
      queue_dir="${args[$((i + 1))]}"
      ;;
    --help|-h)
      usage
      echo
      bash scripts/run_remote_refinement_queue_supervisor.sh --help
      exit 0
      ;;
  esac
done

mkdir -p "$queue_dir"
pid_file="$queue_dir/supervisor.pid"
log_file="$queue_dir/supervisor.log"
launcher_log="$queue_dir/supervisor.launcher.log"

if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "SUPERVISOR_ALREADY_RUNNING pid=$(cat "$pid_file")"
  exit 0
fi

default_args=(
  --hosts auto
  --gpus auto
  --no-scan-hosts
  --queue-interval 60
  --restart-interval 10
  --min-free-gpus 1
  --case-max-power 200
  --queue-dir "$queue_dir"
)

cmd=(bash scripts/run_remote_refinement_queue_supervisor.sh)
if [[ "${#args[@]}" -gt 0 ]]; then
  cmd+=("${args[@]}")
else
  cmd+=("${default_args[@]}")
fi

{
  printf 'START %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf 'CMD '
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >> "$launcher_log"

setsid "${cmd[@]}" </dev/null >> "$log_file" 2>&1 &
launcher_pid="$!"

sleep 2
if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "SUPERVISOR_STARTED pid=$(cat "$pid_file") launcher_pid=$launcher_pid"
else
  echo "SUPERVISOR_START_FAILED launcher_pid=$launcher_pid" >&2
  tail -80 "$log_file" >&2 || true
  exit 1
fi
