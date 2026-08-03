#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

ROOT="${ROOT:-${REPO_ROOT}/runs/ref6_vs_adda_fp32_ka_gt60_20260802}"
KA_VALUES="${KA_VALUES:-60 80 100 111}"
ADDA_BIN="${ADDA_BIN:-${REPO_ROOT}/../adda_fp32/src/ocl/adda_ocl_fp32_reliable_v3}"
ADDA_RESTART_PERIOD="${ADDA_RESTART_PERIOD:-32}"
ADDA_RESTART_DROP="${ADDA_RESTART_DROP:-0.8}"
CLFFT_LIB="${CLFFT_LIB:-${REPO_ROOT}/../adda_neuro_prepare-main/third_party/clfft/usr/lib/x86_64-linux-gnu}"
FFTW_LIB="${FFTW_LIB:-${HOME}/.local/lib}"

export LD_LIBRARY_PATH="${FFTW_LIB}:${CLFFT_LIB}:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${ROOT}"

run_adda() {
  local ka=$1
  local output="${ROOT}/adda_fp32_ka${ka}_dpl15_e4"
  local timing="${ROOT}/adda_fp32_ka${ka}_dpl15_e4.time"
  local stdout="${ROOT}/adda_fp32_ka${ka}_dpl15_e4.stdout"
  local equivalent_radius

  if [[ -s "${output}/mueller" ]] && rg -q "Total wall time:" "${output}/log"; then
    echo "ADDA ka=${ka}: complete, skipping"
    return
  fi

  equivalent_radius=$(awk -v value="${ka}" \
    'BEGIN { pi=atan2(0,-1); printf "%.15g", value/(2*pi) }')
  echo "ADDA ka=${ka}: mixed FP32 operator to 1e-4"
  ADDA_FP32_RELIABLE_RESTART="${ADDA_RESTART_PERIOD}" \
    ADDA_FP32_RELIABLE_DROP="${ADDA_RESTART_DROP}" \
    /usr/bin/time -f 'ACTUAL_WALL_S=%e\nMAXRSS_KB=%M' -o "${timing}" \
    stdbuf -oL -eL "${ADDA_BIN}" \
      -dir "${output}" \
      -shape prism 6 1 \
      -eq_rad "${equivalent_radius}" \
      -lambda 1 \
      -m 1.3 0 \
      -dpl 15 \
      -eps 4 \
      -recalc_resid \
      -ntheta 360 \
      -scat_matr muel \
      -orient 0 0 0 \
      -gpu 0 \
      -iter bcgs2 \
      >"${stdout}" 2>&1
}

run_bem() {
  local ka=$1
  local output="${ROOT}/bem_ka${ka}_ref6_pfft"
  local timing="${ROOT}/bem_ka${ka}_ref6_pfft.time"
  local stdout="${ROOT}/bem_ka${ka}_ref6_pfft.stdout"

  if [[ -s "${output}/result.json" ]]; then
    echo "BEM ka=${ka}: complete, skipping"
    return
  fi

  echo "BEM ka=${ka}: starting"
  /usr/bin/time -f 'ACTUAL_WALL_S=%e\nMAXRSS_KB=%M' -o "${timing}" \
    "${REPO_ROOT}/bem" run \
      --shape prism \
      --sides 6 \
      --aspect 1 \
      --ka "${ka}" \
      --ri 1.3 \
      --quality memory \
      --ref 6 \
      --solver pfft \
      --ntheta 361 \
      --max-iters 2000 \
      --out "${output}" \
      --yes \
      >"${stdout}" 2>&1
}

for ka in ${KA_VALUES}; do
  run_adda "${ka}"
  run_bem "${ka}"
done

python3 "${SCRIPT_DIR}/report_ref6_adda_fp32_boundary.py" --root "${ROOT}"
