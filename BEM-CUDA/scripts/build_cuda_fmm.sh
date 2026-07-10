#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${CUDA_HOME:-}" ]]; then
  if python3 scripts/detect_cuda_toolchain.py --print-env >/tmp/bemcuda_cuda_env.sh 2>/tmp/bemcuda_cuda_detect.err; then
    # shellcheck disable=SC1091
    source /tmp/bemcuda_cuda_env.sh
  else
    echo "no usable CUDA toolkit found; details:" >&2
    python3 scripts/detect_cuda_toolchain.py --json-out runs/cuda_toolchain_detect.json || true
    exit 2
  fi
else
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi

nvcc_extra="${NVCC_EXTRA_FLAGS:-}"
for ccbin in \
  "$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++" \
  "$CUDA_HOME/bin/x86_64-conda_cos6-linux-gnu-g++" \
  /usr/bin/g++-12 \
  /usr/local/bin/g++-12; do
  if [[ -x "$ccbin" ]]; then
    nvcc_extra="${nvcc_extra:+$nvcc_extra }-ccbin $ccbin"
    break
  fi
done

target_fmm="${TARGET_FMM:-bin/bem_cuda_fmm}"

if [[ -n "$nvcc_extra" ]]; then
  make fmm-only TARGET_FMM="$target_fmm" CUDA_HOME="$CUDA_HOME" NVCC_EXTRA_FLAGS="$nvcc_extra"
else
  make fmm-only TARGET_FMM="$target_fmm" CUDA_HOME="$CUDA_HOME"
fi
