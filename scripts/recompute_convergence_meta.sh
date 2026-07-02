#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/home/kirill_epyc/cuda-12.2/usr/local/cuda-12.2}"
BIN="${BIN:-./bin/bem_cuda_fmm}"
OUT="${OUT:-runs/recompute_convergence_meta_20260619}"
NVIDIA_SMI="${BEM_NVIDIA_SMI:-nvidia-smi}"
ALLOW_COMPUTE_SHARE="${BEM_ALLOW_COMPUTE_SHARE:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/gpu_guard.sh"
mkdir -p "$OUT" "$OUT/logs"

run_case() {
  local gpu="$1" name="$2"
  shift 2
  if [[ -s "$OUT/$name.json" ]]; then
    echo "SKIP $name"
    return 0
  fi
  bem_require_gpu_free "$gpu" "$ALLOW_COMPUTE_SHARE" || return $?
  echo "START $name gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}" \
    "$BIN" "$@" --out "$OUT/$name.json" > "$OUT/logs/$name.log" 2>&1
  python3 - "$OUT/$name.json" "$name" <<'PY'
import json
import sys
p, name = sys.argv[1], sys.argv[2]
d = json.load(open(p))
print("DONE", name,
      "total", d.get("timing", {}).get("total_s"),
      "mv", d.get("gmres_matvecs"),
      "nonconv", d.get("gmres_nonconverged_systems"),
      "rel", d.get("gmres_max_final_relres"))
PY
}

(
run_case 0 sphere_ka2_ref3 --shape sphere --ka 2 --ref 3 --ri 1.3116 0 --single --ntheta 181 --solver fmm --fmm-digits 3 --gmres-tol 1e-2 --gmres-restart 120 --max-leaf 128
run_case 0 sphere_ka5_ref4 --shape sphere --ka 5 --ref 4 --ri 1.3116 0 --single --ntheta 181 --solver fmm --fmm-digits 3 --gmres-tol 1e-2 --gmres-restart 120 --max-leaf 128
run_case 0 sphere_ka10_ref4 --shape sphere --ka 10 --ref 4 --ri 1.3116 0 --single --ntheta 181 --solver fmm --fmm-digits 3 --gmres-tol 1e-2 --gmres-restart 120 --max-leaf 128
run_case 0 sphere_ka15_ref4 --shape sphere --ka 15 --ref 4 --ri 1.3116 0 --single --ntheta 181 --solver fmm --fmm-digits 4 --gmres-tol 1e-2 --gmres-restart 120 --max-leaf 128
) &

(
run_case 1 sphere_ka20_ref4 --shape sphere --ka 20 --ref 4 --ri 1.3116 0 --single --ntheta 181 --solver fmm --fmm-digits 4 --gmres-tol 1e-2 --gmres-restart 120 --max-leaf 128
run_case 1 sphere_ka25_ref4 --shape sphere --ka 25 --ref 4 --ri 1.3116 0 --single --ntheta 181 --solver fmm --fmm-digits 4 --gmres-tol 1e-2 --gmres-restart 120 --max-leaf 128
run_case 1 hex_ka2_ref2 --shape hex_prism --prism-aspect 1.5 --ka 2 --ref 2 --ri 1.3116 0 --single --ntheta 181 --solver fmm --fmm-digits 3 --gmres-tol 1e-2 --gmres-restart 120 --max-leaf 128 --quad 4
run_case 1 hex_ka5_ref2 --shape hex_prism --prism-aspect 1.5 --ka 5 --ref 2 --ri 1.3116 0 --single --ntheta 181 --solver fmm --fmm-digits 3 --gmres-tol 1e-2 --gmres-restart 120 --max-leaf 128 --quad 4
) &

(
run_case 2 hex_ka10_ref3 --shape hex_prism --prism-aspect 1.5 --ka 10 --ref 3 --ri 1.3116 0 --single --ntheta 181 --solver fmm --fmm-digits 3 --gmres-tol 1e-2 --gmres-restart 120 --max-leaf 128 --quad 4
run_case 2 hex_ka15_ref4 --shape hex_prism --prism-aspect 1.5 --ka 15 --ref 4 --ri 1.3116 0 --single --ntheta 181 --solver fmm --fmm-digits 3 --gmres-tol 1e-2 --gmres-restart 120 --max-leaf 128 --quad 4
run_case 2 hex_ka20_ref4 --shape hex_prism --prism-aspect 1.5 --ka 20 --ref 4 --ri 1.3116 0 --single --ntheta 181 --solver fmm --fmm-digits 3 --gmres-tol 1e-2 --gmres-restart 120 --max-leaf 128 --quad 4
run_case 2 hex_ka30_ref5 --shape hex_prism --prism-aspect 1.5 --ka 30 --ref 5 --ri 1.3116 0 --single --ntheta 181 --solver fmm --fmm-digits 3 --gmres-tol 2e-2 --gmres-restart 120 --max-leaf 128 --quad 4
) &

(
run_case 2 dust_ka10_gmsh3400_balanced_q7_d6_tol5e4 --obj runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f3400_a35.obj --ka 10 --ri 1.6 0.002 --single --ntheta 181 --solver fmm --accurate --system balanced --fmm-digits 6 --gmres-tol 5e-4 --gmres-restart 500 --max-leaf 128 --quad 7 --no-prec
run_case 2 dust_ka20_gmsh4200_balanced_q7_d6_tol5e4 --obj runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f4200_a35.obj --ka 20 --ri 1.6 0.002 --single --ntheta 181 --solver fmm --accurate --system balanced --fmm-digits 6 --gmres-tol 5e-4 --gmres-restart 500 --max-leaf 128 --quad 7 --no-prec
run_case 2 dust_ka30_gmsh7000_balanced_q7_d6_tol5e4 --obj runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f7000_a45.obj --ka 30 --ri 1.6 0.002 --single --ntheta 181 --solver fmm --accurate --system balanced --fmm-digits 6 --gmres-tol 5e-4 --gmres-restart 500 --max-leaf 128 --quad 7 --no-prec
) &

wait
