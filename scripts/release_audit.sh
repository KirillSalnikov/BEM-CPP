#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:---host}
CXX=${CXX:-g++-12}
CUDA_HOME=${CUDA_HOME:-/usr}
JOBS=${JOBS:-4}

if [[ "$MODE" != --host && "$MODE" != --gpu ]]; then
  echo "usage: $0 [--host|--gpu]" >&2
  exit 2
fi

cd "$ROOT"

make host-checks CXX="$CXX" CUDA_HOME="$CUDA_HOME" -j"$JOBS"
python3 scripts/mueller_audit.py --self-test
python3 scripts/operator_block_audit.py --self-test >/dev/null

for test in tests/test_*.py; do
  case "$test" in
    tests/test_main_syntax_smoke.py|tests/test_rhs_header_compile.py)
      [[ "$MODE" == --gpu ]] || continue
      ;;
  esac
  echo "==> $test"
  python3 "$test"
done

for script in scripts/*.sh examples/*.sh; do
  bash -n "$script"
done
python3 -m compileall -q scripts tests verify_mie.py
python3 -m py_compile bem

if [[ "$MODE" == --host ]]; then
  echo "Host release audit: ok"
  exit 0
fi

make clean
make muller-fp32 CXX="$CXX" CUDA_HOME="$CUDA_HOME" -j"$JOBS"

BIN="$ROOT/bin/muller_nodal_fmm_demo_fp32"
VERSION=$(<VERSION)
"$BIN" --help | grep -q '^Usage:'
"$BIN" --version | grep -q "$VERSION"

set +e
"$BIN" --definitely-unknown > /tmp/bem_release_unknown.log 2>&1
unknown_rc=$?
set -e
if [[ $unknown_rc -ne 2 ]]; then
  cat /tmp/bem_release_unknown.log >&2
  echo "unknown option returned $unknown_rc instead of 2" >&2
  exit 1
fi

smoke=$(mktemp -d /tmp/bem-release-smoke.XXXXXX)
trap 'rm -rf "$smoke" /tmp/bem_release_unknown.log /tmp/bem_release_unwritable.log' EXIT
(
  cd "$smoke"
  OMP_NUM_THREADS=2 "$BIN" \
    --shape prism --sides 6 --aspect 1 --ref 0 --ka 1 --ri 1.3 \
    --edge-mode hdiv --quad 7 --duffy-order 4 \
    --digits 5 --max-leaf 512 --fmm-near-radius 3 \
    --setup-only --no-checkpoint --no-dense-validation
)
python3 - "$smoke/runs/muller_nodal_fmm_benchmark.json" "$VERSION" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["software_version"] == sys.argv[2], data
assert data["setup_only"] is True, data
assert data["solver"].startswith("muller_hdiv_bdm1_"), data
print("GPU setup smoke: ok")
PY

set +e
"$BIN" --setup-only --out /proc/bem-release/result.json \
  > /tmp/bem_release_unwritable.log 2>&1
unwritable_rc=$?
set -e
if [[ $unwritable_rc -ne 1 ]] || \
   ! grep -q '^fatal: cannot create output directory' \
     /tmp/bem_release_unwritable.log; then
  cat /tmp/bem_release_unwritable.log >&2
  echo "output error was not handled cleanly" >&2
  exit 1
fi

make cuda-hessian-check cuda-pfft-hessian-check \
  cuda-muller-fmm-check cuda-muller-edge-check \
  CXX="$CXX" CUDA_HOME="$CUDA_HOME" ARCH=-arch=sm_86 -j"$JOBS"

echo "GPU release audit: ok"
