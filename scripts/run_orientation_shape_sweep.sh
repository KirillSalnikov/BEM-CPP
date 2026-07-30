#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SHAPE=${SHAPE:-sphere}
KA_LIST=${KA_LIST:-"17 18 20 25 30"}
BEM_REF=${BEM_REF:-5}
ADDA_DPL=${ADDA_DPL:-15}
THREADS=${THREADS:-16}
ALPHA=${ALPHA:-256}
RI=${RI:-1.3}
ADDA=${ADDA:-/home/kirill/neuro/adda_clean_check_20260629/adda/src/ocl/adda_ocl}
CLFFT_LIB=${CLFFT_LIB:-/home/kirill/neuro/adda_neuro_prepare-main/third_party/clfft/usr/lib/x86_64-linux-gnu}
FFTW_LIB=${FFTW_LIB:-/home/kirill/.local/lib}

case "$SHAPE" in
  prism)
    OUT_ROOT="$ROOT/runs/orientation_bem_adda_crossover"
    BEM_SHAPE=(--shape prism --sides 6 --aspect 1)
    BEM_SYMMETRY_ORDER=6
    AVG_PARAMS="$ROOT/runs/orientation_bem_adda_crossover/avg_params_alpha256_beta1_gamma1.dat"
    ;;
  sphere)
    OUT_ROOT="$ROOT/runs/orientation_bem_adda_shapes/sphere"
    BEM_SHAPE=(--shape sphere)
    BEM_SYMMETRY_ORDER=1
    AVG_PARAMS="$ROOT/runs/orientation_bem_adda_shapes/avg_params_alpha256_beta90_gamma180.dat"
    ;;
  asymmetric)
    OUT_ROOT="$ROOT/runs/orientation_bem_adda_shapes/asymmetric"
    OBJ="$ROOT/runs/orientation_bem_adda_shapes/asymmetric_oblique_heptagon.obj"
    python3 "$ROOT/scripts/generate_asymmetric_benchmark_shape.py" \
      --obj "$OBJ" \
      --metadata "$ROOT/runs/orientation_bem_adda_shapes/asymmetric_oblique_heptagon.json"
    BEM_SHAPE=(--obj "$OBJ")
    BEM_SYMMETRY_ORDER=1
    AVG_PARAMS="$ROOT/runs/orientation_bem_adda_shapes/avg_params_alpha256_beta90_gamma180.dat"
    ;;
  *)
    printf 'SHAPE must be prism, sphere, or asymmetric\n' >&2
    exit 2
    ;;
esac

for ka in $KA_LIST; do
  CASE="$OUT_ROOT/ka$ka"
  CACHE="$CASE/cache"
  BEM_OUT="$CASE/bem_ref${BEM_REF}_alpha${ALPHA}"
  ADDA_OUT="$CASE/adda_dpl${ADDA_DPL}_alpha${ALPHA}"
  mkdir -p "$CACHE" "$BEM_OUT" "$ADDA_OUT"

  if [[ ! -s "$BEM_OUT/average.json" || ! -s "$BEM_OUT/time.txt" ]]; then
    env \
      OMP_NUM_THREADS="$THREADS" \
      BEM_MULLER_GPU_ASSEMBLY=1 \
      BEM_FMM_CONCURRENT_MEDIA=1 \
      BEM_FMM_GPU_FARFIELD=1 \
      /usr/bin/time \
      -f 'ACTUAL_WALL_S=%e\nMAXRSS_KB=%M' \
      -o "$BEM_OUT/time.txt" \
      "$ROOT/bin/muller_nodal_fmm_demo_fp32" \
      "${BEM_SHAPE[@]}" \
      --ref "$BEM_REF" --ka "$ka" --ri "$RI" \
      --edge-mode hdiv --quad 7 --duffy-order 4 \
      --digits 5 --max-leaf 64 --fmm-near-radius 3 \
      --tol 1e-5 --max-iters 500 --gmres-restart 100 \
      --mbj-only --mbj-nodes 50 --mbj-overlap 0 \
      --near-correction-cache "$CACHE/operator.near" \
      --mbj-cache "$CACHE/mbj50.cache" \
      --fmm-near-fp32 \
      --pfft-fgmres \
      --pfft-inner-tol 1e-1 --pfft-inner-iters auto \
      --pfft-outer-restart 32 \
      --pfft-order 2 --pfft-correction-radius 0 \
      --pfft-grid-safety 1 \
      --orient-average "$ALPHA" 1 1 \
      --orient-symmetry-order "$BEM_SYMMETRY_ORDER" \
      --orient-zero-start \
      --ntheta 73 --no-dense-validation \
      --iteration-log "$BEM_OUT/iterations.csv" \
      --out "$BEM_OUT/average.json" \
      > "$BEM_OUT/run.log" 2>&1
  fi

  if [[ "$SHAPE" == asymmetric ]]; then
    GEOM="$CASE/asymmetric_dpl${ADDA_DPL}.geom"
    python3 "$ROOT/scripts/generate_asymmetric_benchmark_shape.py" \
      --obj "$OBJ" --geom "$GEOM" --ka "$ka" --dpl "$ADDA_DPL" \
      --metadata "$CASE/asymmetric_geometry.json"
    ADDA_SHAPE=(-shape read "$GEOM")
    EQ_RAD=$(python3 -c "import math; print($ka/(2*math.pi))")
    ADDA_SIZE=(-eq_rad "$EQ_RAD")
  elif [[ "$SHAPE" == prism ]]; then
    ADDA_SHAPE=(-shape prism 6 1)
    EQ_RAD=$(python3 -c "import math; print($ka/(2*math.pi))")
    ADDA_SIZE=(-eq_rad "$EQ_RAD" -dpl "$ADDA_DPL")
  else
    ADDA_SHAPE=(-shape sphere)
    EQ_RAD=$(python3 -c "import math; print($ka/(2*math.pi))")
    ADDA_SIZE=(-eq_rad "$EQ_RAD" -dpl "$ADDA_DPL")
  fi

  if [[ ! -s "$ADDA_OUT/mueller" || ! -s "$ADDA_OUT/time.txt" ]]; then
    env \
      OMP_NUM_THREADS=1 \
      LD_LIBRARY_PATH="$FFTW_LIB:$CLFFT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      /usr/bin/time \
      -f 'ACTUAL_WALL_S=%e\nMAXRSS_KB=%M' \
      -o "$ADDA_OUT/time.txt" \
      "$ADDA" \
      -dir "$ADDA_OUT" \
      "${ADDA_SHAPE[@]}" \
      "${ADDA_SIZE[@]}" -lambda 1 \
      -m "$RI" 0 \
      -eps 5 -maxiter 30000 \
      -ntheta 72 -scat_matr muel \
      -orient avg "$AVG_PARAMS" \
      -gpu 0 -iter qmr2 \
      > "$ADDA_OUT/stdout.log" 2>&1
  fi
done
