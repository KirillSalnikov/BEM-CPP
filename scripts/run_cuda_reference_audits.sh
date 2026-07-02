#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p runs/audit_1_6_cuda

if [[ -z "${CUDA_HOME:-}" ]]; then
  if python3 scripts/detect_cuda_toolchain.py --print-env >/tmp/bemcuda_cuda_env.sh 2>/tmp/bemcuda_cuda_detect.err; then
    # shellcheck disable=SC1091
    source /tmp/bemcuda_cuda_env.sh
  else
    echo "no usable CUDA toolkit found; details:" >&2
    python3 scripts/detect_cuda_toolchain.py --json-out runs/audit_1_6_cuda/cuda_detect.json || true
    exit 2
  fi
else
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ ! -x ./bin/bem_cuda_fmm ]]; then
  scripts/build_cuda_fmm.sh
fi

if [[ ! -x ./bin/bem_cuda_fmm ]]; then
  echo "missing executable: ./bin/bem_cuda_fmm" >&2
  echo "build it first, for example: make fmm-only" >&2
  exit 2
fi

if ! python3 scripts/detect_cuda_toolchain.py \
    --json-out runs/audit_1_6_cuda/cuda_runtime_detect.json \
    --require-runtime >/tmp/bemcuda_cuda_runtime.json; then
  python3 scripts/audit_1_6.py \
    --run-cuda \
    --binary ./bin/bem_cuda_fmm \
    --out runs/audit_1_6_cuda/report.json
  python3 scripts/check_audit_1_6_report.py runs/audit_1_6_cuda/report.json
  cat /tmp/bemcuda_cuda_runtime.json >&2 || true
  echo "CUDA runtime is not ready on this host; see runs/audit_1_6_cuda/cuda_runtime_detect.json" >&2
  exit 3
fi

python3 scripts/audit_1_6.py \
  --run-cuda \
  --require-cuda-reference \
  --binary ./bin/bem_cuda_fmm \
  --out runs/audit_1_6_cuda/report.json
python3 scripts/check_audit_1_6_report.py runs/audit_1_6_cuda/report.json

echo "cuda reference audits: ok"
