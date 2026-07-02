#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  queue_poster_true_residual_refresh.sh --plan
  queue_poster_true_residual_refresh.sh --run [--out DIR] [--bin PATH] [--max-power W]

Runs the poster refresh cases needed to replace stale accuracy/time/memory rows
with current JSON metadata and true GMRES residual checks. Each case is a
single BEM calculation on one GPU; the script does not split one case across
several GPUs.
EOF
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

mode=""
out="runs/poster_true_residual_refresh_20260630"
bin=""
max_power="${BEM_GUARD_MAX_POWER_W:-200}"
max_temp="${BEM_GUARD_MAX_TEMP_C:-78}"
force=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan) mode="plan"; shift ;;
    --run) mode="run"; shift ;;
    --out|--out-dir) out="$2"; shift 2 ;;
    --bin) bin="$2"; shift 2 ;;
    --max-power) max_power="$2"; shift 2 ;;
    --max-temp) max_temp="$2"; shift 2 ;;
    --force) force=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$mode" ]]; then
  usage >&2
  exit 2
fi

if [[ -z "$bin" ]]; then
  if [[ -x bin/bem_cuda_fmm.next ]]; then
    bin="bin/bem_cuda_fmm.next"
  else
    bin="bin/bem_cuda_fmm"
  fi
fi

case_lines=(
  "0 sphere_ka5_ref4_q7_d7_tol1e5 --shape sphere --ka 5 --ref 4 --ri 1.3116 0 --single --ntheta 181 --solver fmm --quad 7 --fmm-digits 7 --gmres-tol 1e-5 --gmres-restart 600 --gmres-max-cycles 80 --max-leaf 96"
  "1 sphere_ka10_ref4_q7_d7_tol1e5 --shape sphere --ka 10 --ref 4 --ri 1.3116 0 --single --ntheta 181 --solver fmm --quad 7 --fmm-digits 7 --gmres-tol 1e-5 --gmres-restart 600 --gmres-max-cycles 80 --max-leaf 96"
  "2 sphere_ka15_ref4_q7_d7_tol1e5 --shape sphere --ka 15 --ref 4 --ri 1.3116 0 --single --ntheta 181 --solver fmm --quad 7 --fmm-digits 7 --gmres-tol 1e-5 --gmres-restart 600 --gmres-max-cycles 80 --max-leaf 96"
  "3 sphere_ka20_ref4_q7_d7_tol1e5 --shape sphere --ka 20 --ref 4 --ri 1.3116 0 --single --ntheta 181 --solver fmm --quad 7 --fmm-digits 7 --gmres-tol 1e-5 --gmres-restart 600 --gmres-max-cycles 80 --max-leaf 96"
  "WAIT"
  "0 hex_ka5_ref2_aspect15_q7_d7_tol1e5 --shape hex_prism --prism-aspect 1.5 --ka 5 --ref 2 --ri 1.3116 0 --single --ntheta 181 --solver fmm --system balanced --quad 7 --fmm-digits 7 --gmres-tol 1e-5 --gmres-restart 600 --gmres-max-cycles 80 --max-leaf 128"
  "1 hex_ka10_ref3_aspect15_q7_d7_tol1e5 --shape hex_prism --prism-aspect 1.5 --ka 10 --ref 3 --ri 1.3116 0 --single --ntheta 181 --solver fmm --system balanced --quad 7 --fmm-digits 7 --gmres-tol 1e-5 --gmres-restart 600 --gmres-max-cycles 80 --max-leaf 128"
  "2 hex_ka15_ref4_aspect15_q7_d7_tol1e5 --shape hex_prism --prism-aspect 1.5 --ka 15 --ref 4 --ri 1.3116 0 --single --ntheta 181 --solver fmm --system balanced --quad 7 --fmm-digits 7 --gmres-tol 1e-5 --gmres-restart 600 --gmres-max-cycles 80 --max-leaf 128"
  "3 hex_ka30_ref5_aspect15_q7_d7_tol1e5 --shape hex_prism --prism-aspect 1.5 --ka 30 --ref 5 --ri 1.3116 0 --single --ntheta 181 --solver fmm --system balanced --quad 7 --fmm-digits 7 --gmres-tol 1e-5 --gmres-restart 600 --gmres-max-cycles 80 --max-leaf 128"
  "WAIT"
  "0 hex_ka30_ref6_aspect15_q13_d7_tol1e5 --shape hex_prism --prism-aspect 1.5 --ka 30 --ref 6 --ri 1.3116 0 --single --ntheta 181 --solver fmm --system balanced --quad 13 --fmm-digits 7 --gmres-tol 1e-5 --gmres-restart 800 --gmres-max-cycles 80 --max-leaf 128"
  "1 dust_ka5_adda_mc_f6000_q13_d7_tol1e5 --obj runs/poster_goal_refresh_20260629/meshes/dust_ka5_shape_meshes/adda_mc_s0p5_l0p5_f6000.obj --ka 5 --ri 1.6 0.002 --single --ntheta 181 --solver fmm --accurate --system balanced --quad 13 --fmm-digits 7 --gmres-tol 1e-5 --gmres-restart 1000 --gmres-max-cycles 80 --max-leaf 128"
  "2 dust_ka10_gmsh3900_a35_q7_d7_tol1e5 --obj runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f3900_a35.obj --ka 10 --ri 1.6 0.002 --single --ntheta 181 --solver fmm --accurate --system balanced --quad 7 --fmm-digits 7 --gmres-tol 1e-5 --gmres-restart 1000 --gmres-max-cycles 80 --max-leaf 128"
  "3 dust_ka15_qdec_f5000_t15_q13_d8_tol1e5 --obj runs/poster_goal_refresh_20260630/meshes/dust_ka15_local_repair/gmsh4200a60_qdec_f5000_t15.obj --ka 15 --ri 1.6 0.002 --single --ntheta 181 --solver fmm --accurate --system balanced --quad 13 --fmm-digits 8 --gmres-tol 1e-5 --gmres-restart 1400 --gmres-max-cycles 80 --max-leaf 128"
  "WAIT"
  "1 dust_ka20_gmsh4200_a35_q7_d7_tol1e5 --obj runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f4200_a35.obj --ka 20 --ri 1.6 0.002 --single --ntheta 181 --solver fmm --accurate --system muller2-balanced --quad 7 --fmm-digits 7 --gmres-tol 1e-5 --gmres-restart 1000 --gmres-max-cycles 80 --max-leaf 128"
  "2 dust_ka30_gmsh7000_a45_q7_d7_tol1e5 --obj runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f7000_a45.obj --ka 30 --ri 1.6 0.002 --single --ntheta 181 --solver fmm --accurate --system muller2-balanced --quad 7 --fmm-digits 7 --gmres-tol 1e-5 --gmres-restart 1000 --gmres-max-cycles 80 --max-leaf 128"
)

wait_wave() {
  local rc=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  pids=()
  return "$rc"
}

if [[ "$mode" == "plan" ]]; then
  printf 'gpu,name,args\n'
  for line in "${case_lines[@]}"; do
    [[ "$line" == "WAIT" ]] && continue
    gpu="${line%% *}"
    rest="${line#* }"
    name="${rest%% *}"
    args="${rest#* }"
    printf '%s,%s,%q\n' "$gpu" "$name" "$args"
  done
  exit 0
fi

mkdir -p "$out/logs"
printf 'started=%s out=%s bin=%s max_power=%s max_temp=%s\n' "$(date -Is)" "$out" "$bin" "$max_power" "$max_temp" \
  > "$out/queue.launch.log"

# Accuracy refresh runs are allowed to spend iterations.  Do not inherit a
# relaxed interactive/debug environment that would accept non-converged output
# or stop after a short apparent plateau.
unset BEM_ALLOW_NONCONVERGED
export BEM_GMRES_STAGNATION_CYCLES="${BEM_GMRES_STAGNATION_CYCLES:-0}"
export BEM_GMRES_STAGNATION_REL="${BEM_GMRES_STAGNATION_REL:-0.003}"

pids=()
rc=0
for line in "${case_lines[@]}"; do
  if [[ "$line" == "WAIT" ]]; then
    if ! wait_wave; then
      rc=1
    fi
    continue
  fi
  gpu="${line%% *}"
  rest="${line#* }"
  name="${rest%% *}"
  args="${rest#* }"
  if [[ -s "$out/$name.json" && "$force" != "1" ]]; then
    echo "SKIP existing $out/$name.json" | tee -a "$out/queue.launch.log"
    continue
  fi
  # shellcheck disable=SC2206
  bem_args=($args)
  cmd=(scripts/run_guarded_bem_case.sh --gpu "$gpu" --name "$name" --out-dir "$out"
       --bin "$bin" --max-power "$max_power" --max-temp "$max_temp")
  if [[ "$force" == "1" ]]; then
    cmd+=(--force)
  fi
  if [[ "$name" == dust_* ]]; then
    cmd+=(--require-complex)
  fi
  cmd+=(-- "${bem_args[@]}")
  printf 'LAUNCH gpu=%s name=%s\n' "$gpu" "$name" | tee -a "$out/queue.launch.log"
  BEM_NO_AUTO_MGPU=1 BEM_GMRES_VERBOSE=1 "${cmd[@]}" &
  pids+=("$!")
  sleep 3
done

if ! wait_wave; then
  rc=1
fi
printf 'finished=%s rc=%s\n' "$(date -Is)" "$rc" >> "$out/queue.launch.log"
exit "$rc"
