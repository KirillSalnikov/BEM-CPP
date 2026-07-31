#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-${ROOT}/runs/orientation_bem_adda_recheck_20260731}"
BIN="${ROOT}/bin/muller_nodal_fmm_demo_fp32"
OBJ="${ROOT}/runs/orientation_bem_adda_shapes/asymmetric_oblique_heptagon.obj"

if [[ ! -x "${BIN}" ]]; then
  echo "Missing executable: ${BIN}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/results"

case_is_complete() {
  local result="$1"
  python3 - "${result}" <<'PY'
import json
import math
import sys

try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    residual = float(data["iterations"]["maximum_residual"])
    mueller = data["mueller"]
    valid = residual <= 1.0e-5 and len(mueller) == 4
    valid = valid and all(len(row) == 4 for row in mueller)
    valid = valid and all(
        len(values) == 73 and all(math.isfinite(float(v)) for v in values)
        for row in mueller for values in row
    )
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

run_case() {
  local shape="$1"
  local ka="$2"
  local old_root symmetry
  local -a shape_args

  case "${shape}" in
    prism)
      old_root="${ROOT}/runs/orientation_bem_adda_crossover/ka${ka}"
      symmetry=6
      shape_args=(--shape prism --sides 6 --aspect 1)
      ;;
    sphere)
      old_root="${ROOT}/runs/orientation_bem_adda_shapes/sphere/ka${ka}"
      symmetry=1
      shape_args=(--shape sphere)
      ;;
    asymmetric)
      old_root="${ROOT}/runs/orientation_bem_adda_shapes/asymmetric/ka${ka}"
      symmetry=1
      shape_args=(--obj "${OBJ}")
      ;;
    *)
      echo "Unknown shape: ${shape}" >&2
      exit 1
      ;;
  esac

  local out="${OUT_ROOT}/results/${shape}/ka${ka}"
  local result="${out}/average.json"
  mkdir -p "${out}"

  if case_is_complete "${result}"; then
    echo "[skip] ${shape} ka=${ka}: valid result already exists"
    return
  fi

  if case_is_complete "${out}/average.direct_optimized.json"; then
    cp "${out}/average.direct_optimized.json" "${result}"
    cp "${out}/time.direct_optimized.txt" "${out}/time.txt"
    printf 'paired_gpu_gmres\n' > "${out}/selected_solver.txt"
    echo "[recover/direct] ${shape} ka=${ka}"
    return
  fi
  if case_is_complete "${out}/average.pfft.selected.json"; then
    cp "${out}/average.pfft.selected.json" "${result}"
    cp "${out}/time.pfft.selected.txt" "${out}/time.txt"
    cp "${out}/selected_solver.pfft.txt" "${out}/selected_solver.txt"
    echo "[recover/pFFT-FGMRES] ${shape} ka=${ka}"
    return
  fi

  local -a common=(
    "${BIN}"
    "${shape_args[@]}"
    --ref 5
    --ka "${ka}"
    --ri 1.3
    --edge-mode hdiv
    --quad 7
    --duffy-order 4
    --digits 5
    --max-leaf 64
    --fmm-near-radius 3
    --tol 1e-5
    --gmres-restart 100
    --mbj-only
    --mbj-nodes 50
    --mbj-overlap 0
    --near-correction-cache "${old_root}/cache/operator.near"
    --mbj-cache "${old_root}/cache/mbj50.cache"
    --fmm-near-fp32
    --orient-average 256 1 1
    --orient-symmetry-order "${symmetry}"
    --orient-zero-start
    --ntheta 73
    --no-dense-validation
    --no-checkpoint
  )

  local direct_result="${out}/average.direct.json"
  local -a direct_cmd=(
    "${common[@]}"
    --max-iters 100
    --orient-paired-gpu-gmres
    --out "${direct_result}"
  )
  printf '%q ' "${direct_cmd[@]}" > "${out}/command.direct.sh"
  printf '\n' >> "${out}/command.direct.sh"

  echo "[run/direct] ${shape} ka=${ka}"
  set +e
  env OMP_NUM_THREADS=16 /usr/bin/time \
    -f 'wall_s=%e\nmax_rss_kb=%M\nexit_code=%x' \
    -o "${out}/time.direct.txt" \
    "${direct_cmd[@]}" > "${out}/run.direct.log" 2>&1
  local direct_rc=$?
  set -e
  printf '%s\n' "${direct_rc}" > "${out}/rc.direct"

  if [[ "${direct_rc}" -eq 0 ]] && case_is_complete "${direct_result}"; then
    cp "${direct_result}" "${result}"
    cp "${out}/time.direct.txt" "${out}/time.txt"
    printf 'paired_gpu_gmres\n' > "${out}/selected_solver.txt"
    return
  fi

  local pfft_result="${out}/average.pfft.selected.json"
  local pfft_solver="pfft_fgmres"
  local -a pfft_environment=(BEM_FMM_PAIR_CURRENTS=0)
  case "${shape}/ka${ka}" in
    prism/ka17|prism/ka30|sphere/ka17|asymmetric/ka30)
      pfft_solver="pfft_fgmres_paired_strict"
      pfft_environment=(
        BEM_FMM_PAIR_CURRENTS=1
        BEM_FMM_PHASE_CACHE=0
        BEM_FMM_M2L_STORAGE_FP32=0
        BEM_FMM_MULTI_STORAGE_FP32=0
        BEM_FMM_LOCAL_STORAGE_FP32=0
        BEM_FMM_M2L_FP32=0
        BEM_FMM_L2P_FP32=0
      )
      ;;
  esac
  local -a pfft_cmd=(
    "${common[@]}"
    --max-iters 500
    --pfft-fgmres
    --pfft-inner-tol 1e-1
    --pfft-inner-iters auto
    --pfft-outer-restart 12
    --pfft-order 2
    --pfft-correction-radius 0
    --pfft-grid-safety 1
    --out "${pfft_result}"
  )
  printf '%q ' "${pfft_cmd[@]}" > "${out}/command.pfft.sh"
  printf '\n' >> "${out}/command.pfft.sh"

  echo "[fallback/pFFT-FGMRES] ${shape} ka=${ka}"
  set +e
  env OMP_NUM_THREADS=16 "${pfft_environment[@]}" /usr/bin/time \
    -f 'wall_s=%e\nmax_rss_kb=%M\nexit_code=%x' \
    -o "${out}/time.pfft.selected.txt" \
    "${pfft_cmd[@]}" > "${out}/run.pfft.selected.log" 2>&1
  local pfft_rc=$?
  set -e
  printf '%s\n' "${pfft_rc}" > "${out}/rc.pfft"

  if [[ "${pfft_rc}" -ne 0 ]] || ! case_is_complete "${pfft_result}"; then
    echo "[failed] ${shape} ka=${ka}; inspect ${out}/run.pfft.log" >&2
    exit 1
  fi
  cp "${pfft_result}" "${result}"
  cp "${out}/time.pfft.selected.txt" "${out}/time.txt"
  printf '%s\n' "${pfft_solver}" > "${out}/selected_solver.pfft.txt"
  cp "${out}/selected_solver.pfft.txt" "${out}/selected_solver.txt"
}

for shape in prism sphere asymmetric; do
  for ka in 17 18 20 25 30; do
    run_case "${shape}" "${ka}"
  done
done

echo "All 15 BEM reruns completed: ${OUT_ROOT}/results"
