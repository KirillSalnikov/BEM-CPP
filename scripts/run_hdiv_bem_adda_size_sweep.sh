#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${OUT:-"$ROOT/runs/hdiv_bem_vs_adda_sweep_n1p3"}
ADDA_EXE=${ADDA_EXE:-/home/kirill/adda-orient-warmstart/src/ocl/adda_ocl}
STAGE=${STAGE:-all}
FORCE=${FORCE:-0}
BEM_HYBRID=${BEM_HYBRID:-1}
BEM_VARIANT=${BEM_VARIANT:-batch3_fused}
KAS=(${KAS:-10 15 20 25 30})
DPLS=(${DPLS:-15})
CLFFT_LIB=${CLFFT_LIB:-/home/kirill/neuro/adda_neuro_prepare-main/third_party/clfft/usr/lib/x86_64-linux-gnu}
FFTW_LIB=${FFTW_LIB:-/home/kirill/.local/lib}
JOURNAL=${JOURNAL:-"$OUT/sweep_journal.log"}

mkdir -p "$OUT"
cd "$ROOT"

log() {
  printf '%s\n' "$*" | tee -a "$JOURNAL"
}

bem_refs() {
  case "$1" in
    10) echo "3 4" ;;
    15) echo "4 5" ;;
    20) echo "4 5" ;;
    25) echo "4 5" ;;
    30) echo "4 5" ;;
    *)
      echo "No BEM refinement pair configured for ka=$1" >&2
      return 2
      ;;
  esac
}

run_adda() {
  local ka=$1
  local dpl=$2
  local dir="$OUT/ka${ka}/adda_dpl${dpl}"
  local eq_rad
  eq_rad=$(awk -v ka="$ka" 'BEGIN { pi=atan2(0,-1); printf "%.15g", ka/(2*pi) }')

  if [[ -s "$dir/mueller" && -s "$dir/log" && "$FORCE" != 1 ]]; then
    log "REUSE ADDA ka=$ka dpl=$dpl $(date --iso-8601=seconds)"
    return
  fi

  mkdir -p "$dir"
  log "START ADDA ka=$ka dpl=$dpl $(date --iso-8601=seconds)"
  LD_LIBRARY_PATH="$FFTW_LIB:$CLFFT_LIB:${LD_LIBRARY_PATH:-}" \
    /usr/bin/time \
      -f 'ACTUAL_WALL_S=%e\nMAXRSS_KB=%M' \
      -o "$dir/time.txt" \
      "$ADDA_EXE" \
        -dir "$dir" -shape prism 6 1 \
        -eq_rad "$eq_rad" -lambda 1 \
        -m 1.3 0 -dpl "$dpl" -eps 5 -maxiter 30000 \
        -ntheta 72 -scat_matr muel -orient 0 0 0 \
        -gpu "${GPU:-0}" -iter qmr2 \
        >"$dir/stdout.log" 2>&1
  log "DONE  ADDA ka=$ka dpl=$dpl $(date --iso-8601=seconds)"
}

run_bem() {
  local ka=$1
  local ref=$2
  local dir="$OUT/ka${ka}"
  local stem="bem_ref${ref}_sparse_c6_${BEM_VARIANT}"
  local hybrid_args=()
  if [[ "$BEM_HYBRID" == 1 ]]; then
    stem="${stem}_hybrid"
    hybrid_args=(--hybrid-pfft-fmm --hybrid-pfft-tol 1e-2)
  fi
  local json="$dir/${stem}.json"

  if [[ -s "$json" && "$FORCE" != 1 ]]; then
    log "REUSE BEM ka=$ka ref=$ref variant=$BEM_VARIANT $(date --iso-8601=seconds)"
    return
  fi

  mkdir -p "$dir"
  log "START BEM ka=$ka ref=$ref variant=$BEM_VARIANT $(date --iso-8601=seconds)"
  /usr/bin/time -v -o "$dir/${stem}.time" \
    bin/muller_nodal_fmm_demo \
      --shape prism --sides 6 --aspect 1 --edge-mode hdiv \
      --ref "$ref" --ka "$ka" --ri 1.3 --tol 1e-5 \
      --digits 5 --fmm-near-radius 3 --max-leaf 64 \
      --mbj-nodes 50 --gmres-restart 100 --max-iters 600 \
      --mbj-only --no-dense-validation --physical-check \
      --cyclic-polarization --ntheta 73 \
      "${hybrid_args[@]}" \
      --iteration-log "$dir/${stem}.iterations.csv" \
      --near-correction-cache "$dir/bem_ref${ref}.cache" \
      --out "$json" >"$dir/${stem}.stdout.log" 2>&1
  log "DONE  BEM ka=$ka ref=$ref variant=$BEM_VARIANT $(date --iso-8601=seconds)"
}

case "$STAGE" in
  adda)
    for ka in "${KAS[@]}"; do
      for dpl in "${DPLS[@]}"; do
        run_adda "$ka" "$dpl"
      done
    done
    ;;
  bem)
    make -j"${JOBS:-8}" CXX="${CXX:-g++-12}" \
      CUDA_HOME="${CUDA_HOME:-/usr}" bin/muller_nodal_fmm_demo
    for ka in "${KAS[@]}"; do
      if [[ -n "${BEM_REFS:-}" ]]; then
        read -r -a refs <<<"$BEM_REFS"
      else
        read -r -a refs <<<"$(bem_refs "$ka")"
      fi
      for ref in "${refs[@]}"; do
        run_bem "$ka" "$ref"
      done
    done
    ;;
  all)
    STAGE=adda "$0"
    STAGE=bem "$0"
    ;;
  *)
    echo "STAGE must be one of: adda, bem, all" >&2
    exit 2
    ;;
esac
