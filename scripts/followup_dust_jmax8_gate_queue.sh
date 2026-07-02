#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-runs/dust_jmax8_adaptive_sweep_ri1p6_ka30_mc6000_alpha256_20260702}
ADDA_DIR=${ADDA_DIR:-/home/kirill_epyc/BEM-CUDA/reference/ADDA_for_PO_comparison/refr_1_6__0}
DUST_MESH=${DUST_MESH:-runs/greek_larger_valid/meshes/greek_adda_dpl25_mc_decim6000_ag6_merge6.obj}
GPUS=${GPUS:-0,1}
CHUNK_SIZE=${CHUNK_SIZE:-13}
OMP_THREADS=${OMP_THREADS:-8}
POLL_S=${POLL_S:-180}
KA_REMAINING=${KA_REMAINING:-"33.28 36.58 40.22 44.21 48.61 53.47 58.81"}

LEVEL03=${LEVEL03:-$RUN_ROOT/ka30p25/pilot/level03_Ja8_Jb4_Jg4_a256_b17_g17}
LEVEL04=${LEVEL04:-$RUN_ROOT/ka30p25/pilot/level04_Ja8_Jb5_Jg5_a256_b33_g33}
STATUS=$RUN_ROOT/followup.status

mkdir -p "$RUN_ROOT" "$LEVEL04/parts"

log() {
  echo "$(date '+%F %T') $*" | tee -a "$STATUS"
}

wait_for_file() {
  local path=$1
  while [[ ! -s "$path" ]]; do
    log "waiting for $path"
    sleep "$POLL_S"
  done
}

run_gate() {
  local level=$1
  local tag=$2
  python3 scripts/check_dust_adda_gate.py \
    --bem "$level/bem.json" \
    --adda-dir "$ADDA_DIR" \
    --ka 30.25 \
    --component-floor 1e-3 \
    --bem-stokes-out 1,-1,-1 \
    --bem-stokes-in -1,-1,1 \
    --json-out "$level/adda_gate_${tag}.json"
}

start_remaining_sweep() {
  local jmin_beta=$1
  local jmin_gamma=$2
  log "starting remaining ADDA-backed sizes with Jmin beta=$jmin_beta gamma=$jmin_gamma"
  env \
    RUN_ROOT="$RUN_ROOT" \
    ADDA_BASE_DIR="/home/kirill_epyc/BEM-CUDA/reference/ADDA_for_PO_comparison" \
    KA_LIST="$KA_REMAINING" \
    KA_MODE=adda \
    KA_MIN=30 \
    RI_RE=1.6 \
    RI_IM=0 \
    GPUS="$GPUS" \
    CHUNK_SIZE="$CHUNK_SIZE" \
    OMP_THREADS="$OMP_THREADS" \
    JMIN_ALPHA=8 \
    JMAX_ALPHA=8 \
    JMIN_BETA="$jmin_beta" \
    JMIN_GAMMA="$jmin_gamma" \
    JMAX_BETA=8 \
    JMAX_GAMMA=8 \
    ALPHA_AVG_FIXED=256 \
    PILOT_MESH="$DUST_MESH" \
    FINAL_MESH="$DUST_MESH" \
    FINAL_FROM_PILOT=1 \
    PILOT_SYSTEM=balanced \
    PILOT_DIGITS=3 \
    PILOT_GMRES_TOL=2e-2 \
    PILOT_GMRES_RESTART=120 \
    PILOT_MAX_LEAF=128 \
    DISABLE_PREC=1 \
    ORIENT_WARM_START=previous \
    scripts/run_dust_jmax8_adaptive_sweep.sh
}

log "follow-up started"
wait_for_file "$LEVEL03/bem.json"
log "level03 is complete; running ADDA gate"
if run_gate "$LEVEL03" "level03"; then
  log "level03 passed ADDA gate"
  start_remaining_sweep 4 4
  exit 0
fi

log "level03 failed ADDA gate; starting level04 33x33 alpha256 for ka=30.25"
if [[ ! -s "$LEVEL04/bem.json" ]]; then
  BEM_ORIENT_PROGRESS=1 BEM_FAST_REORTH_OFF=1 python3 ./run_orient_queue.py \
    --exe ./bin/bem_cuda_fmm \
    --out "$LEVEL04/bem.json" \
    --work-dir "$LEVEL04/parts" \
    --gpus "$GPUS" \
    --chunk-size "$CHUNK_SIZE" \
    --omp-threads "$OMP_THREADS" \
    --shape obj \
    --obj "$DUST_MESH" \
    --subdiv 0 \
    --ka 30.25 \
    --ri 1.6 0 \
    --ntheta 181 \
    --scat-plane yz \
    --solver fmm \
    --accurate \
    --system balanced \
    --quad 7 \
    --fmm-digits 3 \
    --gmres-tol 2e-2 \
    --gmres-restart 120 \
    --max-leaf 128 \
    --no-prec \
    --orient 1 33 33 \
    --alpha-avg 256 \
    --orient-warm-start previous
fi

log "level04 is complete; running ADDA gate"
if run_gate "$LEVEL04" "level04"; then
  log "level04 passed ADDA gate"
  start_remaining_sweep 5 5
else
  log "level04 failed ADDA gate; stop for parameter/code review before larger sizes"
  exit 1
fi
