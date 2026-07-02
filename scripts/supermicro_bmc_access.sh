#!/usr/bin/env bash
set -euo pipefail

BMC_IP="${BMC_IP:-192.168.0.103}"
BMC_LOCAL_IP="${BMC_LOCAL_IP:-192.168.0.10}"
BMC_IFACE="${BMC_IFACE:-eth0}"
BMC_PORT="${BMC_PORT:-8443}"
REMOTE_HOST="${REMOTE_HOST:-172.16.1.222}"
REMOTE_USER="${REMOTE_USER:-kirill_epyc}"

usage() {
  cat <<EOF
Usage: $0 [--diagnose-local] [--remote-diagnose] [--configure-route] [--tunnel] [--print]

Environment:
  BMC_IP        BMC address, default ${BMC_IP}
  BMC_LOCAL_IP  Temporary host address in BMC subnet, default ${BMC_LOCAL_IP}
  BMC_IFACE     Interface connected to BMC subnet, default ${BMC_IFACE}
  BMC_PORT      Local forwarded HTTPS port, default ${BMC_PORT}
  REMOTE_HOST   SSH host that can reach BMC, default ${REMOTE_HOST}
  REMOTE_USER   SSH user, default ${REMOTE_USER}

Modes:
  --print            Print route and tunnel commands.
  --diagnose-local   Check BMC from the current host.
  --remote-diagnose  Check BMC through SSH on REMOTE_USER@REMOTE_HOST.
  --configure-route  Add local ${BMC_LOCAL_IP}/24 and route to ${BMC_IP} subnet. Uses sudo.
  --tunnel           Run SSH tunnel; open https://localhost:${BMC_PORT} in browser.
EOF
}

subnet_cidr() {
  local ip="$1"
  IFS=. read -r a b c _ <<<"$ip"
  printf "%s.%s.%s.0/24" "$a" "$b" "$c"
}

print_route_commands() {
  local subnet
  subnet="$(subnet_cidr "$BMC_IP")"
  cat <<EOF
# Run on the server if ${BMC_IP} is not reachable:
sudo ip addr add ${BMC_LOCAL_IP}/24 dev ${BMC_IFACE} 2>/dev/null || true
sudo ip route replace ${subnet} dev ${BMC_IFACE} src ${BMC_LOCAL_IP}
ping -c 3 ${BMC_IP}
curl -kI --connect-timeout 5 https://${BMC_IP}

# Run on your workstation, then open https://localhost:${BMC_PORT}
ssh -N -L ${BMC_PORT}:${BMC_IP}:443 ${REMOTE_USER}@${REMOTE_HOST}
EOF
}

configure_route() {
  local subnet
  subnet="$(subnet_cidr "$BMC_IP")"
  if ! ip addr show dev "$BMC_IFACE" | grep -q " ${BMC_LOCAL_IP}/24"; then
    sudo ip addr add "${BMC_LOCAL_IP}/24" dev "$BMC_IFACE"
  fi
  sudo ip route replace "$subnet" dev "$BMC_IFACE" src "$BMC_LOCAL_IP"
}

diagnose_local() {
  echo "BMC_IP=${BMC_IP}"
  echo "BMC_IFACE=${BMC_IFACE}"
  ip -brief addr show dev "$BMC_IFACE" || true
  ip route get "$BMC_IP" || true
  if ping -c 2 -W 1 "$BMC_IP"; then
    echo "bmc_ping=ok"
  else
    echo "bmc_ping=fail"
  fi
  if command -v curl >/dev/null 2>&1; then
    if curl -kI --connect-timeout 5 "https://${BMC_IP}" >/tmp/bemcuda_bmc_https.$$ 2>&1; then
      echo "bmc_https=ok"
    else
      echo "bmc_https=fail"
    fi
    sed -n '1,12p' /tmp/bemcuda_bmc_https.$$ 2>/dev/null || true
    rm -f /tmp/bemcuda_bmc_https.$$
  else
    echo "bmc_https=skip curl_missing"
  fi
}

remote_diagnose() {
  local remote="${REMOTE_USER}@${REMOTE_HOST}"
  local q_bmc_ip q_iface
  printf -v q_bmc_ip '%q' "$BMC_IP"
  printf -v q_iface '%q' "$BMC_IFACE"
  ssh -o BatchMode=yes -o ConnectTimeout=6 -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$remote" \
    "BMC_IP=${q_bmc_ip} BMC_IFACE=${q_iface} bash -s" <<'REMOTE'
set -euo pipefail
echo "remote_host=$(hostname)"
echo "BMC_IP=${BMC_IP}"
ip -brief addr show dev "$BMC_IFACE" || true
ip route get "$BMC_IP" || true
if ping -c 2 -W 1 "$BMC_IP"; then
  echo "bmc_ping=ok"
else
  echo "bmc_ping=fail"
fi
if command -v curl >/dev/null 2>&1; then
  if curl -kI --connect-timeout 5 "https://${BMC_IP}" >/tmp/bemcuda_bmc_https.$$ 2>&1; then
    echo "bmc_https=ok"
  else
    echo "bmc_https=fail"
  fi
  sed -n '1,12p' /tmp/bemcuda_bmc_https.$$ 2>/dev/null || true
  rm -f /tmp/bemcuda_bmc_https.$$
else
  echo "bmc_https=skip curl_missing"
fi
REMOTE
}

run_tunnel() {
  exec ssh -N -L "${BMC_PORT}:${BMC_IP}:443" "${REMOTE_USER}@${REMOTE_HOST}"
}

if [[ $# -eq 0 ]]; then
  usage
  echo
  print_route_commands
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --print)
      print_route_commands
      ;;
    --diagnose-local)
      diagnose_local
      ;;
    --remote-diagnose)
      remote_diagnose
      ;;
    --configure-route)
      configure_route
      ;;
    --tunnel)
      run_tunnel
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done
