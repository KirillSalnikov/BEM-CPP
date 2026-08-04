#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/runs/equal_accuracy_10_20260804}"
PUBLISH_DIR="${PUBLISH_DIR:-${REPO_ROOT}/benchmarks/equal_accuracy_10_20260804}"
ADDA_BIN="${ADDA_BIN:-${REPO_ROOT}/../adda_clean_check_20260629/adda/src/ocl/adda_ocl}"
CLFFT_LIB="${CLFFT_LIB:-${REPO_ROOT}/../adda_neuro_prepare-main/third_party/clfft/usr/lib/x86_64-linux-gnu}"
FFTW_LIB="${FFTW_LIB:-${HOME}/.local/lib}"
GPU="${GPU:-0}"
THREADS="${THREADS:-16}"

BEM_POINTS_PER_WAVELENGTH="${BEM_POINTS_PER_WAVELENGTH:-15}"
ADDA_DPL="${ADDA_DPL:-20}"
ADDA_CONTROL_DPL="${ADDA_CONTROL_DPL:-15}"
TOLERANCE="${TOLERANCE:-1e-5}"
REPLICATES="${REPLICATES:-3}"
KA_VALUES=(2 4 6 8 10)
SHAPES=(sphere prism)

export LD_LIBRARY_PATH="${FFTW_LIB}:${CLFFT_LIB}:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${OUT_ROOT}"
SCAT_GRID="${OUT_ROOT}/adda_scat_grid_181.dat"
printf '%s\n' \
  'global_type=grid' \
  'theta:' \
  'type=range' \
  'N=181' \
  'min=0' \
  'max=180' \
  'phi:' \
  'type=range' \
  'N=1' \
  'min=90' \
  'max=90' \
  >"${SCAT_GRID}"

if [[ ! -x "${ADDA_BIN}" ]]; then
  echo "Missing ADDA executable: ${ADDA_BIN}" >&2
  exit 2
fi
if [[ ! -x "${REPO_ROOT}/bin/muller_nodal_fmm_demo_fp32" ]]; then
  echo "Missing BEM executable; run 'make muller-fp32' first." >&2
  exit 2
fi

equivalent_radius() {
  awk -v ka="$1" 'BEGIN { pi=atan2(0,-1); printf "%.15g", ka/(2*pi) }'
}

bem_result_valid() {
  python3 - "$1" "${TOLERANCE}" <<'PY'
import json
import math
import sys

try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    tolerance = float(sys.argv[2])
    residuals = []
    section = data.get("pfft_fgmres") or data.get("mbj") or {}
    value = section.get("fmm_residual")
    if value is not None:
        residuals.append(float(value))
    value = (data.get("physical") or {}).get("parallel_fmm_residual")
    if value is not None:
        residuals.append(float(value))
    theta = (data.get("physical") or {}).get("theta_degrees", [])
    mueller = (data.get("physical") or {}).get("mueller", [])
    valid = len(residuals) == 2 and max(residuals) <= 1.05 * tolerance
    valid = valid and len(theta) == 181 and len(mueller) == 4
    valid = valid and all(len(row) == 4 for row in mueller)
    valid = valid and all(
        len(values) == 181 and all(math.isfinite(float(v)) for v in values)
        for row in mueller for values in row
    )
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

adda_result_valid() {
  python3 - "$1" "${TOLERANCE}" <<'PY'
import math
import re
import sys
from pathlib import Path

directory = Path(sys.argv[1])
tolerance = float(sys.argv[2])
try:
    text = (directory / "log").read_text(encoding="utf-8", errors="replace")
    residuals = [
        float(value)
        for value in re.findall(
            r"Final \(recalculated\) residual norm:\s*([0-9.eE+-]+)", text
        )
    ]
    rows = (directory / "mueller_scatgrid").read_text(encoding="utf-8").splitlines()[1:]
    valid = len(residuals) == 2 and max(residuals) <= 1.05 * tolerance
    valid = valid and len(rows) == 181
    valid = valid and all(
        len(line.split()) == 18 and all(math.isfinite(float(v)) for v in line.split())
        for line in rows
    )
except (OSError, ValueError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

bem_refinement() {
  local shape=$1
  local ka=$2
  local out=$3
  local -a shape_args=(--shape "${shape}")
  if [[ "${shape}" == "prism" ]]; then
    shape_args+=(--sides 6 --aspect 1)
  fi
  "${REPO_ROOT}/bem" run \
    "${shape_args[@]}" \
    --ka "${ka}" --ri 1.3 \
    --quality standard \
    --points-per-wavelength "${BEM_POINTS_PER_WAVELENGTH}" \
    --single-stage --independent-polarizations \
    --ntheta 181 --max-iters 1000 \
    --out "${out}" --yes --dry-run --json |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["inputs"]["refinement"])'
}

run_bem() {
  local shape=$1
  local ka=$2
  local refinement=$3
  local label=$4
  local case_root="${OUT_ROOT}/${shape}_ka${ka}"
  local output="${case_root}/bem_${label}"
  local timing="${case_root}/bem_${label}.time.txt"
  local stdout="${case_root}/bem_${label}.stdout.log"
  local -a shape_args=(--shape "${shape}")
  if [[ "${shape}" == "prism" ]]; then
    shape_args+=(--sides 6 --aspect 1)
  fi

  if bem_result_valid "${output}/result.json"; then
    echo "[skip] BEM ${shape} ka=${ka} ${label}"
    return
  fi

  echo "[run] BEM ${shape} ka=${ka} ref=${refinement} ${label}"
  env OMP_NUM_THREADS="${THREADS}" \
    /usr/bin/time -f 'wall_s=%e\nmax_rss_kb=%M\nexit_code=%x' \
      -o "${timing}" \
    "${REPO_ROOT}/bem" run \
      "${shape_args[@]}" \
      --ka "${ka}" --ri 1.3 \
      --quality standard --ref "${refinement}" \
      --single-stage --independent-polarizations \
      --tol "${TOLERANCE}" --ntheta 181 --max-iters 1000 \
      --threads "${THREADS}" --gpu "${GPU}" --no-build \
      --out "${output}" --yes \
      >"${stdout}" 2>&1

  if ! bem_result_valid "${output}/result.json"; then
    echo "BEM validation failed: ${output}" >&2
    exit 1
  fi
}

run_adda() {
  local shape=$1
  local ka=$2
  local dpl=$3
  local label=$4
  local output="${OUT_ROOT}/${shape}_ka${ka}/adda_${label}"
  local radius
  local -a shape_args=(-shape sphere)
  if [[ "${shape}" == "prism" ]]; then
    shape_args=(-shape prism 6 1)
  fi

  if adda_result_valid "${output}"; then
    echo "[skip] ADDA ${shape} ka=${ka} ${label}"
    return
  fi

  mkdir -p "${output}"
  radius=$(equivalent_radius "${ka}")
  echo "[run] ADDA ${shape} ka=${ka} dpl=${dpl} ${label}"
  ADDA_FP32_RELIABLE_RESTART=32 \
  ADDA_FP32_RELIABLE_DROP=0.8 \
    /usr/bin/time -f 'wall_s=%e\nmax_rss_kb=%M\nexit_code=%x' \
      -o "${output}/time.txt" \
    "${ADDA_BIN}" \
      -dir "${output}" \
      "${shape_args[@]}" \
      -eq_rad "${radius}" -lambda 1 \
      -m 1.3 0 -dpl "${dpl}" \
      -eps 5 -recalc_resid -maxiter 1000000 \
      -scat_grid_inp "${SCAT_GRID}" -store_scat_grid -scat_matr muel \
      -orient 0 0 0 -sym no -gpu "${GPU}" -iter bcgs2 \
      >"${output}/stdout.log" 2>&1

  if ! adda_result_valid "${output}"; then
    echo "ADDA validation failed: ${output}" >&2
    exit 1
  fi
}

for shape in "${SHAPES[@]}"; do
  for ka in "${KA_VALUES[@]}"; do
    case_root="${OUT_ROOT}/${shape}_ka${ka}"
    mkdir -p "${case_root}"
    ref=$(bem_refinement "${shape}" "${ka}" "${case_root}/plan")
    if (( ref < 1 )); then
      echo "Invalid BEM refinement ${ref} for ${shape} ka=${ka}" >&2
      exit 1
    fi
    printf '%s\n' "${ref}" >"${case_root}/bem_production_ref.txt"

    run_bem "${shape}" "${ka}" "$((ref - 1))" control
    run_adda "${shape}" "${ka}" "${ADDA_CONTROL_DPL}" official_control_dpl15
    for replicate in $(seq 1 "${REPLICATES}"); do
      suffix=""
      if (( replicate > 1 )); then
        suffix="_r${replicate}"
      fi
      run_bem "${shape}" "${ka}" "${ref}" "production${suffix}"
      run_adda "${shape}" "${ka}" "${ADDA_DPL}" "official_production_dpl20${suffix}"
    done
  done
done

python3 "${SCRIPT_DIR}/report_equal_accuracy_10.py" \
  --root "${OUT_ROOT}" \
  --bem-bin "${REPO_ROOT}/bin/muller_nodal_fmm_demo_fp32" \
  --adda-bin "${ADDA_BIN}"

mkdir -p "${PUBLISH_DIR}"
cp "${OUT_ROOT}/REPORT.md" "${PUBLISH_DIR}/README.md"
cp "${OUT_ROOT}/equal_accuracy_10.csv" "${PUBLISH_DIR}/equal_accuracy_10.csv"
cp "${OUT_ROOT}/equal_accuracy_10.json" "${PUBLISH_DIR}/equal_accuracy_10.json"
cp "${OUT_ROOT}/equal_accuracy_10.png" "${PUBLISH_DIR}/equal_accuracy_10.png"
cp "${OUT_ROOT}/provenance.json" "${PUBLISH_DIR}/provenance.json"
