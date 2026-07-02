#!/usr/bin/env bash
set -euo pipefail

BMC_PROFILE_FILE="${BMC_PROFILE_FILE:-$HOME/.config/bemcuda/bmc.env}"
if [[ -r "$BMC_PROFILE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$BMC_PROFILE_FILE"
fi

BMC_HOST="${BMC_HOST:-192.168.0.103}"
BMC_USER="${BMC_USER:-ADMIN}"
BMC_PASS_FILE="${BMC_PASS_FILE:-}"
if [[ "$BMC_PASS_FILE" == "~/"* ]]; then
  BMC_PASS_FILE="${HOME}/${BMC_PASS_FILE#~/}"
fi
BMC_INTERFACE="${BMC_INTERFACE:-lanplus}"
BMC_VIA_SSH_HOST="${BMC_VIA_SSH_HOST:-}"
BMC_VIA_SSH_USER="${BMC_VIA_SSH_USER:-kirill_epyc}"
BMC_WAIT_TIMEOUT_S="${BMC_WAIT_TIMEOUT_S:-180}"
BMC_WAIT_INTERVAL_S="${BMC_WAIT_INTERVAL_S:-5}"

usage() {
  cat <<EOF
Usage: $0 [--setup-profile|--check|--status|--on|--wait-on|--off|--cycle|--soft|--reset-bmc|--lan-print|--print]

Environment:
  BMC_PROFILE_FILE  Shell profile with BMC settings, default ${BMC_PROFILE_FILE}
  BMC_HOST       BMC/IPMI host, default ${BMC_HOST}
  BMC_USER       BMC user, default ${BMC_USER}
  BMC_PASS_FILE  File with BMC password, safer than command-line password
  BMC_PASSWORD   Password via environment; used only if BMC_PASS_FILE is unset
  BMC_INTERFACE  ipmitool interface, default ${BMC_INTERFACE}
  BMC_WAIT_TIMEOUT_S  --wait-on timeout, default ${BMC_WAIT_TIMEOUT_S}
  BMC_WAIT_INTERVAL_S --wait-on poll interval, default ${BMC_WAIT_INTERVAL_S}
  BMC_VIA_SSH_HOST  Optional host that already has ipmitool; not useful if
                    that host is the powered-off server itself
  BMC_VIA_SSH_USER  SSH user for BMC_VIA_SSH_HOST, default ${BMC_VIA_SSH_USER}

Examples:
  $0 --setup-profile
  $0 --check
  $0 --status
  $0 --wait-on
  BMC_HOST=192.168.0.103 BMC_USER=ADMIN $0 --status  # prompts for password
EOF
}

need_ipmitool() {
  if command -v ipmitool >/dev/null 2>&1; then
    return 0
  fi
  cat >&2 <<'EOF'
ipmitool is not installed on this machine.
Install it locally, then rerun this script:
  sudo apt install ipmitool
or on openSUSE:
  sudo zypper install ipmitool
EOF
  return 127
}

ipmi_base_args() {
  printf '%s\0' -I "$BMC_INTERFACE" -H "$BMC_HOST" -U "$BMC_USER"
  if [[ -n "$BMC_PASS_FILE" ]]; then
    printf '%s\0' -f "$BMC_PASS_FILE"
  elif [[ -n "${BMC_PASSWORD:-}" ]]; then
    printf '%s\0' -E
  else
    printf '%s\0' -a
  fi
}

run_ipmi() {
  if [[ -n "$BMC_VIA_SSH_HOST" ]]; then
    run_ipmi_via_ssh "$@"
    return
  fi
  need_ipmitool
  local -a args
  mapfile -d '' -t args < <(ipmi_base_args)
  if [[ -n "${BMC_PASSWORD:-}" && -z "$BMC_PASS_FILE" ]]; then
    IPMI_PASSWORD="$BMC_PASSWORD" ipmitool "${args[@]}" "$@"
  else
    ipmitool "${args[@]}" "$@"
  fi
}

run_ipmi_via_ssh() {
  local -a ssh_opts
  local quoted_args remote_cmd
  ssh_opts=(
    -o BatchMode=yes
    -o ConnectTimeout=6
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o LogLevel=ERROR
  )
  printf -v quoted_args '%q ' "$@"
  if [[ -n "$BMC_PASS_FILE" ]]; then
    remote_cmd="set -euo pipefail; pass_file=\$(mktemp); trap 'rm -f \"\$pass_file\"' EXIT; cat > \"\$pass_file\"; ipmitool -I $(printf '%q' "$BMC_INTERFACE") -H $(printf '%q' "$BMC_HOST") -U $(printf '%q' "$BMC_USER") -f \"\$pass_file\" ${quoted_args}"
    ssh "${ssh_opts[@]}" "${BMC_VIA_SSH_USER}@${BMC_VIA_SSH_HOST}" "$remote_cmd" < "$BMC_PASS_FILE"
  elif [[ -n "${BMC_PASSWORD:-}" ]]; then
    BMC_PASSWORD="$BMC_PASSWORD" ssh "${ssh_opts[@]}" "${BMC_VIA_SSH_USER}@${BMC_VIA_SSH_HOST}" \
      "IPMI_PASSWORD=\"\$BMC_PASSWORD\" ipmitool -I $(printf '%q' "$BMC_INTERFACE") -H $(printf '%q' "$BMC_HOST") -U $(printf '%q' "$BMC_USER") -E ${quoted_args}"
  else
    echo "BMC_VIA_SSH_HOST requires BMC_PASS_FILE or BMC_PASSWORD; interactive -a cannot pass through BatchMode SSH." >&2
    return 2
  fi
}

write_profile() {
  local config_dir pass_file password
  config_dir="$(dirname "$BMC_PROFILE_FILE")"
  pass_file="${BMC_PASS_FILE:-$config_dir/bmc.pass}"
  mkdir -p "$config_dir"
  chmod 700 "$config_dir"

  if [[ -n "${BMC_PASSWORD:-}" ]]; then
    password="$BMC_PASSWORD"
  elif [[ -r "$pass_file" ]]; then
    password=""
  else
    printf 'BMC password for %s@%s: ' "$BMC_USER" "$BMC_HOST" >&2
    IFS= read -r -s password
    printf '\n' >&2
  fi

  if [[ -n "$password" ]]; then
    printf '%s\n' "$password" > "$pass_file"
    chmod 600 "$pass_file"
  fi

  cat > "$BMC_PROFILE_FILE" <<EOF
BMC_HOST=${BMC_HOST@Q}
BMC_USER=${BMC_USER@Q}
BMC_INTERFACE=${BMC_INTERFACE@Q}
BMC_PASS_FILE=${pass_file@Q}
EOF
  chmod 600 "$BMC_PROFILE_FILE"
  echo "profile=${BMC_PROFILE_FILE}"
  echo "password_file=${pass_file}"
  echo "next_check=scripts/ipmi_power_control.sh --check"
}

check_setup() {
  local ready=1
  echo "BMC_PROFILE_FILE=${BMC_PROFILE_FILE}"
  if [[ -r "$BMC_PROFILE_FILE" ]]; then
    echo "profile=ok"
  else
    ready=0
    echo "profile=missing"
  fi
  echo "BMC_HOST=${BMC_HOST}"
  echo "BMC_USER=${BMC_USER}"
  echo "BMC_INTERFACE=${BMC_INTERFACE}"

  if ip route get "$BMC_HOST" >/tmp/bemcuda_bmc_route.$$ 2>&1; then
    echo "route=ok $(tr '\n' ' ' </tmp/bemcuda_bmc_route.$$)"
  else
    ready=0
    echo "route=fail"
  fi
  rm -f /tmp/bemcuda_bmc_route.$$

  if ping -c 1 -W 1 "$BMC_HOST" >/dev/null 2>&1; then
    echo "bmc_ping=ok"
  else
    ready=0
    echo "bmc_ping=fail"
  fi

  if command -v ipmitool >/dev/null 2>&1; then
    echo "local_ipmitool=ok $(command -v ipmitool)"
  else
    ready=0
    echo "local_ipmitool=missing"
    echo "install_hint=install ipmitool locally for power-on after the server OS is off"
  fi

  if command -v nc >/dev/null 2>&1; then
    if nc -zu -w 2 "$BMC_HOST" 623 >/dev/null 2>&1; then
      echo "ipmi_udp_623_probe=ok"
    else
      echo "ipmi_udp_623_probe=unknown_or_blocked"
    fi
  fi

  if [[ -n "$BMC_PASS_FILE" ]]; then
    if [[ -r "$BMC_PASS_FILE" ]]; then
      echo "password_file=ok ${BMC_PASS_FILE}"
      local mode
      mode="$(stat -c '%a' "$BMC_PASS_FILE" 2>/dev/null || true)"
      if [[ "$mode" != "600" ]]; then
        echo "password_file_mode=warn ${mode:-unknown}; recommended chmod 600"
      fi
    else
      ready=0
      echo "password_file=missing ${BMC_PASS_FILE}"
    fi
  elif [[ -n "${BMC_PASSWORD:-}" ]]; then
    echo "password=ok env"
  else
    ready=0
    echo "password=missing"
  fi

  if [[ -n "$BMC_VIA_SSH_HOST" ]]; then
    echo "via_ssh=${BMC_VIA_SSH_USER}@${BMC_VIA_SSH_HOST}"
    echo "via_ssh_note=this works only while that SSH host is already powered on"
  fi

  if [[ "$ready" -eq 1 ]]; then
    echo "remote_power_on=ready"
  else
    echo "remote_power_on=not_ready"
    echo
    print_commands
    return 1
  fi
}

print_commands() {
  cat <<EOF
# Recommended: keep password outside shell history.
mkdir -p ~/.config/bemcuda
chmod 700 ~/.config/bemcuda
printf '%s\n' 'YOUR_BMC_PASSWORD' > ~/.config/bemcuda/bmc.pass
chmod 600 ~/.config/bemcuda/bmc.pass
cat > ~/.config/bemcuda/bmc.env <<'PROFILE'
BMC_HOST='${BMC_HOST}'
BMC_USER='${BMC_USER}'
BMC_INTERFACE='${BMC_INTERFACE}'
BMC_PASS_FILE="\$HOME/.config/bemcuda/bmc.pass"
PROFILE
chmod 600 ~/.config/bemcuda/bmc.env

# Or let this script create the same files:
scripts/ipmi_power_control.sh --setup-profile

# Check state:
scripts/ipmi_power_control.sh --status

# Power on after AC/BMC is reachable:
scripts/ipmi_power_control.sh --wait-on

# Raw ipmitool equivalent:
ipmitool -I ${BMC_INTERFACE} -H ${BMC_HOST} -U ${BMC_USER} -f ~/.config/bemcuda/bmc.pass chassis power status
ipmitool -I ${BMC_INTERFACE} -H ${BMC_HOST} -U ${BMC_USER} -f ~/.config/bemcuda/bmc.pass chassis power on

# Temporary workaround while local ipmitool is missing and the SSH host is on:
BMC_VIA_SSH_HOST=172.16.1.222 BMC_PASS_FILE=~/.config/bemcuda/bmc.pass \\
  scripts/ipmi_power_control.sh --status
EOF
}

wait_on() {
  local start now status
  run_ipmi chassis power on >/dev/null || true
  start="$(date +%s)"
  while true; do
    status="$(run_ipmi chassis power status 2>&1 || true)"
    echo "$status"
    if grep -qi 'on' <<<"$status"; then
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= BMC_WAIT_TIMEOUT_S )); then
      echo "timeout waiting for chassis power on after ${BMC_WAIT_TIMEOUT_S}s" >&2
      return 124
    fi
    sleep "$BMC_WAIT_INTERVAL_S"
  done
}

mode="${1:---status}"
case "$mode" in
  --setup-profile)
    write_profile
    ;;
  --check)
    check_setup
    ;;
  --status)
    run_ipmi chassis power status
    ;;
  --on)
    run_ipmi chassis power on
    ;;
  --wait-on)
    wait_on
    ;;
  --off)
    run_ipmi chassis power off
    ;;
  --cycle)
    run_ipmi chassis power cycle
    ;;
  --soft)
    run_ipmi chassis power soft
    ;;
  --reset-bmc)
    run_ipmi mc reset cold
    ;;
  --lan-print)
    run_ipmi lan print 1
    ;;
  --print)
    print_commands
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
