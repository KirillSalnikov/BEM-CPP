#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-${ROOT}/runs/farfield_spectral_replay}"
BIN="${ROOT}/bin/muller_nodal_fmm_demo_fp32"
OBJ="${ROOT}/runs/orientation_bem_adda_shapes/asymmetric_oblique_heptagon.obj"

run_case() {
  local shape="$1"
  local ka="$2"
  local mode="$3"
  local source_root symmetry checkpoint
  local -a shape_args spectral_env

  case "${shape}" in
    prism)
      source_root="${ROOT}/runs/orientation_bem_adda_crossover/ka${ka}"
      checkpoint="${source_root}/bem_ref5_alpha256/average.json.orient.checkpoint"
      symmetry=6
      shape_args=(--shape prism --sides 6 --aspect 1)
      ;;
    sphere)
      source_root="${ROOT}/runs/orientation_bem_adda_shapes/sphere/ka${ka}"
      checkpoint="${source_root}/bem_ref5_alpha256/average.json.orient.checkpoint"
      symmetry=1
      shape_args=(--shape sphere)
      ;;
    asymmetric)
      source_root="${ROOT}/runs/orientation_bem_adda_shapes/asymmetric/ka${ka}"
      checkpoint="${source_root}/bem_ref5_alpha256/average.json.orient.checkpoint"
      symmetry=1
      shape_args=(--obj "${OBJ}")
      ;;
    *)
      echo "unknown shape: ${shape}" >&2
      exit 2
      ;;
  esac

  spectral_env=()
  if [[ "${mode}" == auto_v2 ]]; then
    spectral_env=(BEM_FARFIELD_SPECTRAL_ALPHA=auto)
  elif [[ "${mode}" != full ]]; then
    spectral_env=(BEM_FARFIELD_SPECTRAL_ALPHA="${mode}")
  fi

  local out="${OUT_ROOT}/${shape}/ka${ka}/${mode}"
  mkdir -p "${out}"
  if [[ -s "${out}/average.json" ]] &&
      grep -q '\[farfield replay\]' "${out}/run.log" 2>/dev/null; then
    echo "[skip] ${shape} ka=${ka} ${mode}"
    return
  fi

  echo "[run] ${shape} ka=${ka} ${mode}"
  env OMP_NUM_THREADS=16 \
    BEM_FARFIELD_REPLAY_CHECKPOINT="${checkpoint}" \
    "${spectral_env[@]}" \
    /usr/bin/time -f 'wall_s=%e\nmax_rss_kb=%M' -o "${out}/time.txt" \
    "${BIN}" "${shape_args[@]}" \
      --ref 5 --ka "${ka}" --ri 1.3 \
      --edge-mode hdiv --quad 7 --duffy-order 4 --digits 5 \
      --max-leaf 64 --fmm-near-radius 3 --tol 1e-5 \
      --gmres-restart 100 --max-iters 500 \
      --mbj-only --mbj-nodes 50 --mbj-overlap 0 \
      --near-correction-cache "${source_root}/cache/operator.near" \
      --mbj-cache "${source_root}/cache/mbj50.cache" \
      --fmm-near-fp32 \
      --pfft-fgmres --pfft-inner-tol 1e-1 --pfft-inner-iters auto \
      --pfft-outer-restart 12 --pfft-order 2 \
      --pfft-correction-radius 0 --pfft-grid-safety 1 \
      --orient-average 256 1 1 --orient-symmetry-order "${symmetry}" \
      --orient-zero-start --ntheta 73 --no-dense-validation \
      --no-checkpoint --out "${out}/average.json" \
      > "${out}/run.log" 2>&1
}

for shape in prism sphere asymmetric; do
  for ka in 25 30; do
    run_case "${shape}" "${ka}" full
    run_case "${shape}" "${ka}" 64
    run_case "${shape}" "${ka}" auto_v2
  done
done

python3 "${ROOT}/scripts/report_farfield_spectral_replay.py" "${OUT_ROOT}"
