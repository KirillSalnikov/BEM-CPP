#!/usr/bin/env bash
set -uo pipefail

gpu="${1:-2}"
repo="${2:-/home/kirill_epyc/BEM-CUDA}"
cd "$repo" || exit 1
NVIDIA_SMI="${BEM_NVIDIA_SMI:-nvidia-smi}"
allow_compute_share="${BEM_ALLOW_COMPUTE_SHARE:-0}"
source "$repo/scripts/gpu_guard.sh"

export CUDA_HOME="$HOME/cuda-12.2/usr/local/cuda-12.2"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$gpu"
export BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}"

outdir="runs/poster_fig7_memory"
mkdir -p "$outdir/logs"
csv="$outdir/fig7_memory_queue.csv"
if [ ! -s "$csv" ]; then
  printf "shape,size,mesh_level,ref_or_mesh,vram_gb,status,seconds,source\n" > "$csv"
fi

bem_require_gpu_free "$gpu" "$allow_compute_share" || exit $?

gpu_mem_mib() {
  "$NVIDIA_SMI" --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, -v g="$gpu" '$1+0==g+0 {gsub(/ /,"",$2); print $2; exit}'
}

run_case() {
  local shape="$1" size="$2" level="$3" refmesh="$4" tag="$5"
  shift 5
  local json="$outdir/${tag}.json"
  local log="$outdir/logs/${tag}.log"
  local mon="$outdir/logs/${tag}.mem"
  if [ -s "$json" ]; then
    echo "exists $json"
    return 0
  fi

  echo "RUN $tag on physical GPU $gpu"
  : > "$mon"
  (
    while true; do
      m="$(gpu_mem_mib || true)"
      [ -n "$m" ] && printf "%s\n" "$m" >> "$mon"
      sleep 1
    done
  ) &
  local mon_pid=$!

  local t0 t1 rc status peak
  t0="$(date +%s)"
  timeout 21600 "$@" --out "$json" > "$log" 2>&1
  rc=$?
  t1="$(date +%s)"
  kill "$mon_pid" 2>/dev/null || true
  wait "$mon_pid" 2>/dev/null || true

  peak="$(awk 'BEGIN{m=0} $1>m{m=$1} END{printf "%.3f", m/1024.0}' "$mon" 2>/dev/null)"
  [ -n "$peak" ] || peak="nan"
  if [ "$rc" -eq 0 ] && [ -s "$json" ]; then
    status="ok"
  elif [ "$rc" -eq 124 ]; then
    status="timeout"
  else
    status="failed_${rc}"
  fi
  printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$shape" "$size" "$level" "$refmesh" "$peak" "$status" "$((t1-t0))" "$tag" >> "$csv"
}

common=(--ri 1.3116 0 --accurate --quad 7 --ntheta 19 --single --solver fmm --fmm-digits 6 --gmres-tol 3e-3 --gmres-restart 500 --max-leaf 128)
run_case "сфера" 10 4 "ref4" "sphere_ka10_ref4" \
  ./bin/bem_cuda_fmm --shape sphere --ka 10 --ref 4 "${common[@]}"
run_case "столбик" 20 4 "ref4" "hex_ka20_ref4" \
  ./bin/bem_cuda_fmm --shape hex_prism --prism-aspect 1.5 --ka 20 --ref 4 "${common[@]}"
run_case "столбик" 30 5 "ref5" "hex_ka30_ref5" \
  ./bin/bem_cuda_fmm --shape hex_prism --prism-aspect 1.5 --ka 30 --ref 5 "${common[@]}"

dust_common=(--ri 1.6 0.002 --accurate --system balanced --quad 7 --ntheta 19 --single --solver fmm --fmm-digits 6 --gmres-tol 5e-4 --gmres-restart 500 --max-leaf 128 --no-prec)
run_case "пыль" 10 3 "gmsh3400/q7d6" "dust_ka10_gmsh3400_balanced_q7_d6_tol5e4" \
  ./bin/bem_cuda_fmm --obj runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f3400_a35.obj --ka 10 "${dust_common[@]}"
run_case "пыль" 20 4 "gmsh4200/q7d6" "dust_ka20_gmsh4200_balanced_q7_d6_tol5e4" \
  ./bin/bem_cuda_fmm --obj runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f4200_a35.obj --ka 20 "${dust_common[@]}"
run_case "пыль" 30 5 "gmsh7000/q7d6" "dust_ka30_gmsh7000_balanced_q7_d6_tol5e4" \
  ./bin/bem_cuda_fmm --obj runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f7000_a45.obj --ka 30 "${dust_common[@]}"
