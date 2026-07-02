#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-172.16.1.222}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/kirill_epyc/BEM-CUDA}"
RUN_DIR="runs/hex_euler_scaling_20260623"
LEVELS="1x1x1,2x2x2,4x2x2,4x4x2,4x4x4,8x4x4,8x8x4"
GPUS="${GPUS:-auto}"

ssh "kirill_epyc@${HOST}" "mkdir -p '${REMOTE_ROOT}/scripts' '${REMOTE_ROOT}/bemcuda' '${REMOTE_ROOT}/${RUN_DIR}'"
scp run_orient_queue.py "kirill_epyc@${HOST}:${REMOTE_ROOT}/"
scp scripts/run_hex_euler_scaling_benchmark.py "kirill_epyc@${HOST}:${REMOTE_ROOT}/scripts/"
scp bemcuda/*.py "kirill_epyc@${HOST}:${REMOTE_ROOT}/bemcuda/"
ssh "kirill_epyc@${HOST}" "cd '${REMOTE_ROOT}' && \
  chmod +x run_orient_queue.py scripts/run_hex_euler_scaling_benchmark.py && \
  (nohup python3 scripts/run_hex_euler_scaling_benchmark.py \
    --out '${RUN_DIR}' \
    --levels '${LEVELS}' \
    --gpus '${GPUS}' \
    --bem-chunk-size 2 \
    --adda-dpl 20 \
    --adda-eps 5 \
    > '${RUN_DIR}/queue_256.log' 2>&1 & \
    echo \$! > '${RUN_DIR}/queue_256.pid') && \
  cat '${RUN_DIR}/queue_256.pid'"
