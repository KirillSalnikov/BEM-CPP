#!/usr/bin/env bash
set -euo pipefail

user="${USER_REMOTE:-kirill_epyc}"
remote="${REMOTE_HOST:-172.16.1.222}"
limit_w="${GPU_POWER_LIMIT_W:-200}"
gpus="${GPU_LIST:-0 1 2}"
nvidia_smi="${BEM_NVIDIA_SMI:-nvidia-smi}"

ssh_opts=(
  -o BatchMode=yes
  -o ConnectTimeout=6
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
)

usage() {
  cat <<EOF
Usage: $0 [--show] [--set]

Environment:
  REMOTE_HOST=${remote}
  USER_REMOTE=${user}
  GPU_POWER_LIMIT_W=${limit_w}
  GPU_LIST="${gpus}"

--show only reads current limits. --set requires passwordless sudo on the
remote host; otherwise run the printed sudo commands manually on the server.
EOF
}

show_remote() {
  ssh "${ssh_opts[@]}" "$user@$remote" \
    "NVIDIA_SMI=$(printf '%q' "$nvidia_smi") bash -s" <<'REMOTE'
set -euo pipefail
"$NVIDIA_SMI" --query-gpu=index,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits
REMOTE
}

set_remote() {
  local quoted_gpus
  printf -v quoted_gpus '%q' "$gpus"
  ssh "${ssh_opts[@]}" "$user@$remote" \
    "GPU_POWER_LIMIT_W=$(printf '%q' "$limit_w") GPU_LIST=$quoted_gpus NVIDIA_SMI=$(printf '%q' "$nvidia_smi") bash -s" <<'REMOTE'
set -euo pipefail
echo "before:"
"$NVIDIA_SMI" --query-gpu=index,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits
echo "setting:"
for gpu in $GPU_LIST; do
  if ! sudo -n "$NVIDIA_SMI" -i "$gpu" -pl "$GPU_POWER_LIMIT_W"; then
    echo "sudo is required. Run manually on the server:" >&2
    for manual_gpu in $GPU_LIST; do
      echo "  sudo $NVIDIA_SMI -i $manual_gpu -pl $GPU_POWER_LIMIT_W" >&2
    done
    exit 3
  fi
done
echo "after:"
"$NVIDIA_SMI" --query-gpu=index,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits
REMOTE
}

mode="${1:---show}"
case "$mode" in
  --show)
    show_remote
    ;;
  --set)
    set_remote
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
