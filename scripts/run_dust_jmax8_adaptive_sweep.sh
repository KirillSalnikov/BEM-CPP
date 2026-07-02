#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT=${RUN_ROOT:-runs/dust_jmax8_adaptive_sweep_ri1p6_ka30_20260701}
GPUS=${GPUS:-0,1}
CHUNK_SIZE=${CHUNK_SIZE:-13}
OMP_THREADS=${OMP_THREADS:-8}
POLL_S=${POLL_S:-120}
NVIDIA_SMI="${BEM_NVIDIA_SMI:-nvidia-smi}"
ORIENT_WARM_START=${ORIENT_WARM_START:-previous}
ORIENT_PROGRESS=${ORIENT_PROGRESS:-1}

RI_RE=${RI_RE:-1.6}
RI_IM=${RI_IM:-0}
NTTHETA=${NTTHETA:-181}
SCAT_PLANE=${SCAT_PLANE:-yz}

JMIN_ALPHA=${JMIN_ALPHA:-2}
JMIN_BETA=${JMIN_BETA:-2}
JMIN_GAMMA=${JMIN_GAMMA:-2}
JMAX_ALPHA=${JMAX_ALPHA:-8}
JMAX_BETA=${JMAX_BETA:-8}
JMAX_GAMMA=${JMAX_GAMMA:-8}
ALPHA_AVG_FIXED=${ALPHA_AVG_FIXED:-256}

ORIENT_TOL=${ORIENT_TOL:-0.03}
ORIENT_MAX_TOL=${ORIENT_MAX_TOL:-0.08}
ORIENT_SCALE_TOL=${ORIENT_SCALE_TOL:-0.03}
ORIENT_COMPONENT_FLOOR=${ORIENT_COMPONENT_FLOOR:-1e-4}
ADDA_COMPARE_COMPONENT_FLOOR=${ADDA_COMPARE_COMPONENT_FLOOR:-1e-3}
BEM_STOKES_OUT=${BEM_STOKES_OUT:-1,-1,-1}
BEM_STOKES_IN=${BEM_STOKES_IN:--1,-1,1}
PILOT_SYSTEM=${PILOT_SYSTEM:-muller2-balanced}
PILOT_QUAD=${PILOT_QUAD:-7}
PILOT_DIGITS=${PILOT_DIGITS:-3}
PILOT_GMRES_TOL=${PILOT_GMRES_TOL:-2e-2}
PILOT_GMRES_RESTART=${PILOT_GMRES_RESTART:-120}
PILOT_MAX_LEAF=${PILOT_MAX_LEAF:-128}
FINAL_SYSTEM=${FINAL_SYSTEM:-balanced}
FINAL_QUAD=${FINAL_QUAD:-7}
FINAL_DIGITS=${FINAL_DIGITS:-4}
FINAL_GMRES_TOL=${FINAL_GMRES_TOL:-5e-3}
FINAL_GMRES_RESTART=${FINAL_GMRES_RESTART:-160}
FINAL_MAX_LEAF=${FINAL_MAX_LEAF:-128}
FINAL_FROM_PILOT=${FINAL_FROM_PILOT:-0}
DISABLE_PREC=${DISABLE_PREC:-1}
prec_args=()
preconditioner_mode=auto
if [[ "$DISABLE_PREC" == "1" ]]; then
  prec_args=(--no-prec)
  preconditioner_mode=off
fi

mkdir -p "$RUN_ROOT/logs"

refr_token() {
  python3 - "$1" <<'PY'
import sys
s = sys.argv[1].strip()
if s in {"", "0", "0.0", "0.00", "+0", "+0.0"}:
    print("0")
else:
    print(s.replace("+", "").replace("-", "m").replace(".", "_"))
PY
}

ADDA_BASE_DIR=${ADDA_BASE_DIR:-/home/user/BEM-CPP/greek/ADDA_for_PO_comparison}
ADDA_REF_DIR=${ADDA_REF_DIR:-$ADDA_BASE_DIR/refr_$(refr_token "$RI_RE")__$(refr_token "$RI_IM")}
KA_MIN=${KA_MIN:-30}
KA_MODE=${KA_MODE:-adda}
KA_POINTS=${KA_POINTS:-10}
KA_MAX=${KA_MAX:-}
if [[ -z "${KA_LIST:-}" ]]; then
  if [[ ! -d "$ADDA_REF_DIR" ]]; then
    echo "ADDA reference directory not found: $ADDA_REF_DIR" >&2
    exit 2
  fi
  KA_LIST=$(python3 - "$ADDA_REF_DIR" "$KA_MIN" "$KA_MODE" "$KA_POINTS" "$KA_MAX" <<'PY'
import re
import sys
from pathlib import Path
root = Path(sys.argv[1])
ka_min = float(sys.argv[2])
mode = sys.argv[3].strip().lower()
ka_points = int(sys.argv[4])
ka_max_arg = sys.argv[5].strip()
vals = []
for path in root.glob("A_x=*_refr_*.dat"):
    m = re.search(r"A_x=([0-9.]+)_refr_", path.name)
    if not m:
        continue
    ka = float(m.group(1))
    if ka >= ka_min:
        vals.append(ka)
vals = sorted(set(vals))
if not vals:
    raise SystemExit(f"no ADDA A_x files >= {ka_min:g} under {root}")
ka_max = float(ka_max_arg) if ka_max_arg else vals[-1]
if ka_max < ka_min:
    raise SystemExit(f"KA_MAX={ka_max:g} is below KA_MIN={ka_min:g}")
if mode in {"adda", "reference", "adda-backed"}:
    chosen = [v for v in vals if ka_min <= v <= ka_max]
elif mode in {"uniform", "uniform10"}:
    if ka_points < 2:
        chosen = [ka_min]
    else:
        step = (ka_max - ka_min) / (ka_points - 1)
        chosen = [ka_min + i * step for i in range(ka_points)]
else:
    raise SystemExit(f"unknown KA_MODE={mode!r}; use 'adda' or 'uniform'")
if not chosen:
    raise SystemExit(f"no ka values selected for mode={mode}, range=[{ka_min:g},{ka_max:g}]")
print(" ".join(f"{v:.10g}" for v in chosen))
PY
  )
fi
PILOT_MESH=${PILOT_MESH:-runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f7000_a45.obj}
FINAL_MESH=${FINAL_MESH:-runs/greek_larger_valid/meshes/greek_adda_dpl25_mc_gmsh_nogeom_test.obj}

cat > "$RUN_ROOT/manifest.txt" <<EOF
target=dust particle adaptive BEM orientation averaging
ka_list=$KA_LIST
ka_mode=$KA_MODE
ka_points=$KA_POINTS
ka_min=$KA_MIN
ka_max=${KA_MAX:-auto_from_adda}
ri=$RI_RE+$RI_IM i
adda_reference_dir=$ADDA_REF_DIR
gpus=$GPUS
jmax_alpha=$JMAX_ALPHA
jmax_beta=$JMAX_BETA
jmax_gamma=$JMAX_GAMMA
jmin_alpha=$JMIN_ALPHA
jmin_beta=$JMIN_BETA
jmin_gamma=$JMIN_GAMMA
Nmax=2^Jmax+1
pilot_mesh=$PILOT_MESH
final_mesh=$FINAL_MESH
solver=fmm
pilot_system=$PILOT_SYSTEM
final_system=$FINAL_SYSTEM
pilot_quad=$PILOT_QUAD
final_quad=$FINAL_QUAD
pilot_fmm_digits=$PILOT_DIGITS
pilot_gmres_tol=$PILOT_GMRES_TOL
pilot_gmres_restart=$PILOT_GMRES_RESTART
pilot_max_leaf=$PILOT_MAX_LEAF
final_fmm_digits=$FINAL_DIGITS
final_gmres_tol=$FINAL_GMRES_TOL
final_gmres_restart=$FINAL_GMRES_RESTART
final_max_leaf=$FINAL_MAX_LEAF
preconditioner=$preconditioner_mode
ntheta=$NTTHETA
chunk_size=$CHUNK_SIZE
orient_warm_start=$ORIENT_WARM_START
mode=pilot_mesh_adaptive_then_final_quality
final_from_pilot=$FINAL_FROM_PILOT
adaptive_tol=$ORIENT_TOL
adaptive_max_tol=$ORIENT_MAX_TOL
adaptive_scale_tol=$ORIENT_SCALE_TOL
adaptive_component_floor=$ORIENT_COMPONENT_FLOOR
alpha_avg_fixed=$ALPHA_AVG_FIXED
adda_compare_component_floor=$ADDA_COMPARE_COMPONENT_FLOOR
bem_stokes_out=$BEM_STOKES_OUT
bem_stokes_in=$BEM_STOKES_IN
EOF

gpu_count() {
  awk -F, '{print NF}' <<< "$GPUS"
}

wait_for_gpus() {
  local need
  need=$(gpu_count)
  while true; do
    local busy=0
    IFS=',' read -r -a arr <<< "$GPUS"
    for gpu in "${arr[@]}"; do
      if "$NVIDIA_SMI" -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF{found=1} END{exit found?0:1}'; then
        busy=1
      fi
    done
    if [[ "$busy" == "0" ]]; then
      return 0
    fi
    date '+%F %T waiting for selected GPUs' | tee -a "$RUN_ROOT/queue.status"
    "$NVIDIA_SMI" --query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits |
      tee -a "$RUN_ROOT/queue.status"
    sleep "$POLL_S"
  done
}

label_for_ka() {
  printf 'ka%s' "$(sed 's/\./p/g' <<< "$1")"
}

write_accuracy_summary() {
  python3 scripts/summarize_dust_adda_table_accuracy.py "$RUN_ROOT" \
    --adda-dir "$ADDA_REF_DIR" \
    --bem-stokes-out="$BEM_STOKES_OUT" \
    --bem-stokes-in="$BEM_STOKES_IN" \
    --component-floor "$ADDA_COMPARE_COMPONENT_FLOOR" \
    > "$RUN_ROOT/accuracy_vs_adda_floor${ADDA_COMPARE_COMPONENT_FLOOR}.csv"
}

for ka in $KA_LIST; do
  label=$(label_for_ka "$ka")
  out_dir="$RUN_ROOT/$label"
  pilot_dir="$out_dir/pilot"
  final_dir="$out_dir/final_quality"
  pilot_log="$RUN_ROOT/logs/$label.pilot.log"
  final_log="$RUN_ROOT/logs/$label.final.log"
  if [[ -s "$final_dir/bem.json" ]]; then
    echo "$(date '+%F %T') skip existing $label" | tee -a "$RUN_ROOT/queue.status"
    continue
  fi
  wait_for_gpus
  echo "$(date '+%F %T') start pilot $label mesh=$PILOT_MESH gpus=$GPUS" | tee -a "$RUN_ROOT/queue.status"
  BEM_ORIENT_PROGRESS="$ORIENT_PROGRESS" BEM_FAST_REORTH_OFF=1 python3 scripts/adaptive_jmax_orient_queue.py \
    --queue ./run_orient_queue.py \
    --exe ./bin/bem_cuda_fmm \
    --out-dir "$pilot_dir" \
    --gpus "$GPUS" \
    --chunk-size "$CHUNK_SIZE" \
    --omp-threads "$OMP_THREADS" \
    --jmin-alpha "$JMIN_ALPHA" \
    --jmin-beta "$JMIN_BETA" \
    --jmin-gamma "$JMIN_GAMMA" \
    --jmax-alpha "$JMAX_ALPHA" \
    --jmax-beta "$JMAX_BETA" \
    --jmax-gamma "$JMAX_GAMMA" \
    --fixed-alpha-avg "$ALPHA_AVG_FIXED" \
    --tol "$ORIENT_TOL" \
    --max-tol "$ORIENT_MAX_TOL" \
    --scale-tol "$ORIENT_SCALE_TOL" \
    --component-floor "$ORIENT_COMPONENT_FLOOR" \
    --orient-warm-start "$ORIENT_WARM_START" \
    -- \
    --shape obj \
    --obj "$PILOT_MESH" \
    --subdiv 0 \
    --ka "$ka" \
    --ri "$RI_RE" "$RI_IM" \
    --ntheta "$NTTHETA" \
    --scat-plane "$SCAT_PLANE" \
    --solver fmm \
    --accurate \
    --system "$PILOT_SYSTEM" \
    --quad "$PILOT_QUAD" \
    --fmm-digits "$PILOT_DIGITS" \
    --gmres-tol "$PILOT_GMRES_TOL" \
    --gmres-restart "$PILOT_GMRES_RESTART" \
    --max-leaf "$PILOT_MAX_LEAF" \
    "${prec_args[@]}" \
    > "$pilot_log" 2>&1

  read -r na nb ng ja jb jg accepted_bem < <(python3 - "$pilot_dir/adaptive_jmax_manifest.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    m = json.load(f)
levels = m.get("levels") or []
if not levels:
    raise SystemExit("no adaptive levels")
accepted = None
for rec in levels:
    if rec.get("accepted"):
        accepted = rec
        break
if accepted is None:
    accepted = levels[-1]
N = accepted["N"]
J = accepted["J"]
print(
    N["alpha"], N["beta"], N["gamma"],
    J["alpha"], J["beta"], J["gamma"],
    accepted.get("bem") or accepted["out"],
)
PY
  )

  if [[ "$FINAL_FROM_PILOT" == "1" ]]; then
    mkdir -p "$final_dir"
    cp "$accepted_bem" "$final_dir/bem.json"
    python3 - "$out_dir" "$pilot_dir" "$final_dir" "$ka" "$ja" "$jb" "$jg" "$na" "$nb" "$ng" "$accepted_bem" <<'PY'
import json, sys
out_dir, pilot_dir, final_dir, ka, ja, jb, jg, na, nb, ng, accepted_bem = sys.argv[1:]
record = {
    "mode": "pilot_mesh_adaptive_as_final",
    "ka": float(ka),
    "pilot_manifest": pilot_dir + "/adaptive_jmax_manifest.json",
    "final_bem": final_dir + "/bem.json",
    "accepted_bem": accepted_bem,
    "accepted_J": {"alpha": int(ja), "beta": int(jb), "gamma": int(jg)},
    "accepted_N": {"alpha": int(na), "beta": int(nb), "gamma": int(ng)},
}
with open(out_dir + "/adaptive_final_manifest.json", "w") as f:
    json.dump(record, f, indent=2)
    f.write("\n")
PY
    echo "$(date '+%F %T') accepted pilot as final $label J=${ja}/${jb}/${jg} N=${na}/${nb}/${ng}" | tee -a "$RUN_ROOT/queue.status"
    write_accuracy_summary || true
    continue
  fi

  wait_for_gpus
  mkdir -p "$final_dir/parts"
  echo "$(date '+%F %T') start final $label mesh=$FINAL_MESH J=${ja}/${jb}/${jg} N=${na}/${nb}/${ng} gpus=$GPUS" | tee -a "$RUN_ROOT/queue.status"
  BEM_ORIENT_PROGRESS="$ORIENT_PROGRESS" BEM_FAST_REORTH_OFF=1 python3 ./run_orient_queue.py \
    --exe ./bin/bem_cuda_fmm \
    --out "$final_dir/bem.json" \
    --work-dir "$final_dir/parts" \
    --gpus "$GPUS" \
    --chunk-size "$CHUNK_SIZE" \
    --omp-threads "$OMP_THREADS" \
    --shape obj \
    --obj "$FINAL_MESH" \
    --subdiv 0 \
    --ka "$ka" \
    --ri "$RI_RE" "$RI_IM" \
    --ntheta "$NTTHETA" \
    --scat-plane "$SCAT_PLANE" \
    --solver fmm \
    --accurate \
    --system "$FINAL_SYSTEM" \
    --quad "$FINAL_QUAD" \
    --fmm-digits "$FINAL_DIGITS" \
    --gmres-tol "$FINAL_GMRES_TOL" \
    --gmres-restart "$FINAL_GMRES_RESTART" \
    --max-leaf "$FINAL_MAX_LEAF" \
    "${prec_args[@]}" \
    --orient 1 "$nb" "$ng" \
    --alpha-avg "$na" \
    --orient-warm-start "$ORIENT_WARM_START" \
    > "$final_log" 2>&1
  python3 - "$out_dir" "$pilot_dir" "$final_dir" "$ka" "$ja" "$jb" "$jg" "$na" "$nb" "$ng" <<'PY'
import json, sys
out_dir, pilot_dir, final_dir, ka, ja, jb, jg, na, nb, ng = sys.argv[1:]
record = {
    "mode": "pilot_mesh_adaptive_then_final_quality",
    "ka": float(ka),
    "pilot_manifest": pilot_dir + "/adaptive_jmax_manifest.json",
    "final_bem": final_dir + "/bem.json",
    "accepted_J": {"alpha": int(ja), "beta": int(jb), "gamma": int(jg)},
    "accepted_N": {"alpha": int(na), "beta": int(nb), "gamma": int(ng)},
}
with open(out_dir + "/adaptive_final_manifest.json", "w") as f:
    json.dump(record, f, indent=2)
    f.write("\n")
PY
  echo "$(date '+%F %T') done $label" | tee -a "$RUN_ROOT/queue.status"
  write_accuracy_summary || true
done

echo "$(date '+%F %T') all adaptive dust runs done" | tee -a "$RUN_ROOT/queue.status"
