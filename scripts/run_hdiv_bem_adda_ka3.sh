#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${OUT:-"$ROOT/runs/hdiv_bem_vs_adda_ka3_n1p3"}
ADDA_EXE=${ADDA_EXE:-/home/kirill/adda-orient-warmstart/src/ocl/adda_ocl}
PYTHON=${PYTHON:-python3}
FORCE=${FORCE:-0}
DPLS=(${DPLS:-15 20 30 40 60 80})
CLFFT_LIB=${CLFFT_LIB:-/home/kirill/neuro/adda_neuro_prepare-main/third_party/clfft/usr/lib/x86_64-linux-gnu}
FFTW_LIB=${FFTW_LIB:-/home/kirill/.local/lib}

mkdir -p "$OUT"
cd "$ROOT"

make -j"${JOBS:-8}" CXX="${CXX:-g++-12}" \
  CUDA_HOME="${CUDA_HOME:-/usr}" bin/muller_nodal_fmm_demo

run_bem() {
  local ref=$1
  local json="$OUT/bem_ref${ref}.json"
  if [[ -s "$json" && "$FORCE" != 1 ]]; then
    echo "Reuse $json"
    return
  fi
  /usr/bin/time -v -o "$OUT/bem_ref${ref}.time" \
    bin/muller_nodal_fmm_demo \
      --shape prism --sides 6 --aspect 1 --edge-mode hdiv \
      --ref "$ref" --ka 3 --ri 1.3 --tol 1e-5 \
      --digits 5 --fmm-near-radius 3 --max-leaf 64 \
      --mbj-nodes 50 --gmres-restart 100 --max-iters 600 \
      --mbj-only --no-dense-validation --physical-check --ntheta 73 \
      --near-correction-cache "$OUT/bem_ref${ref}.cache" \
      --out "$json" >"$OUT/bem_ref${ref}.stdout.log" 2>&1
}

run_adda() {
  local dpl=$1
  local dir="$OUT/adda_dpl${dpl}"
  if [[ -s "$dir/mueller" && -s "$dir/log" && "$FORCE" != 1 ]]; then
    echo "Reuse $dir"
    return
  fi
  mkdir -p "$dir"
  LD_LIBRARY_PATH="$FFTW_LIB:$CLFFT_LIB:${LD_LIBRARY_PATH:-}" \
    /usr/bin/time \
      -f 'ACTUAL_WALL_S=%e\nMAXRSS_KB=%M' \
      -o "$dir/time.txt" \
      "$ADDA_EXE" \
        -dir "$dir" -shape prism 6 1 \
        -eq_rad 0.477464829275686 -lambda 1 \
        -m 1.3 0 -dpl "$dpl" -eps 5 -maxiter 30000 \
        -ntheta 72 -scat_matr muel -orient 0 0 0 \
        -gpu "${GPU:-0}" -iter qmr2 \
        >"$dir/stdout.log" 2>&1
}

run_bem 2
run_bem 3
for dpl in "${DPLS[@]}"; do
  run_adda "$dpl"
done

report_args=(
  --bem "$OUT/bem_ref3.json"
  --bem-log "$OUT/bem_ref3.time"
  --bem-coarse "ref2=$OUT/bem_ref2.json"
  --out-dir "$OUT/report"
)
for dpl in "${DPLS[@]}"; do
  report_args+=(--adda "dpl${dpl}=$OUT/adda_dpl${dpl}")
done
"$PYTHON" scripts/compare_nodal_bem_adda.py "${report_args[@]}"

echo "Report: $OUT/report/comparison_report.md"
echo "Plot:   $OUT/report/bem_vs_adda_selected.png"
