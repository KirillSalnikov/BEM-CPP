#!/usr/bin/env bash
set -u

interval="${1:-10}"
once="${POWER_WATCH_ONCE:-0}"
nvidia_smi="${BEM_NVIDIA_SMI:-nvidia-smi}"
if [[ "${1:-}" == "--once" ]]; then
  once=1
  interval="${2:-10}"
fi

have_passwordless_sudo() {
  sudo -n true >/dev/null 2>&1
}

print_ipmi_unavailable() {
  echo "ipmi unavailable without passwordless sudo"
}

while true; do
  sudo_ready=0
  if have_passwordless_sudo; then
    sudo_ready=1
  fi
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %z') ==="
  hostname
  uptime
  who -b || true
  echo "--- last reboot/shutdown ---"
  last -x shutdown reboot crash 2>/dev/null | head -8 || true
  echo "--- gpu ---"
  if command -v "$nvidia_smi" >/dev/null 2>&1; then
    "$nvidia_smi" --query-gpu=index,name,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true
  else
    echo "$nvidia_smi not found"
  fi
  echo "--- ipmi sel tail ---"
  if [[ "$sudo_ready" -eq 1 ]]; then
    sudo -n ipmitool sel elist 2>&1 | tail -20 || true
  else
    print_ipmi_unavailable
  fi
  echo "--- ipmi sensors ---"
  if [[ "$sudo_ready" -eq 1 ]]; then
    sudo -n ipmitool sdr elist full 2>&1 | egrep -i "12V|5VCC|3.3VCC|CPU Temp|System Temp|Peripheral Temp|VRM|FAN1|power|psu|volt" || true
  else
    print_ipmi_unavailable
  fi
  echo "--- chassis ---"
  if [[ "$sudo_ready" -eq 1 ]]; then
    sudo -n ipmitool chassis status 2>&1 || true
  else
    print_ipmi_unavailable
  fi
  echo
  if [[ "$once" == "1" ]]; then
    exit 0
  fi
  sleep "$interval"
done
