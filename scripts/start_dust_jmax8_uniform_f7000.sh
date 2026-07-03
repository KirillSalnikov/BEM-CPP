#!/usr/bin/env bash
set -euo pipefail

# Production launcher for the high-size dust-particle ADDA comparison.
# Uses the edge-aware quality mesh: the previous f7000_a45 surface had
# skinny triangles at A_x≈30 and did not improve under stricter GMRES/FMM.

RUN_ROOT=${RUN_ROOT:-runs/dust_jmax8_uniform10_quality_conn_ri1p6_20260703}
GPUS=${GPUS:-0,1}

KA_LIST=${KA_LIST:-"30.25 33.42333333 36.59666667 39.77 42.94333333 46.11666667 49.29 52.46333333 55.63666667 58.81"}
DUST_MESH=${DUST_MESH:-runs/pass5_followup_20260701/meshes_dpl20/projected_dpl20_quality_conn_a0p75.obj}

mkdir -p "$RUN_ROOT"

env \
  RUN_ROOT="$RUN_ROOT" \
  ADDA_BASE_DIR="/home/kirill_epyc/BEM-CUDA/reference/ADDA_for_PO_comparison" \
  KA_LIST="$KA_LIST" \
  KA_MODE=uniform \
  KA_MIN=30.25 \
  KA_POINTS=10 \
  RI_RE=1.6 \
  RI_IM=0 \
  GPUS="$GPUS" \
  CHUNK_SIZE="${CHUNK_SIZE:-13}" \
  CHUNK_ORDER="${CHUNK_ORDER:-spread}" \
  OMP_THREADS="${OMP_THREADS:-8}" \
  POLL_S="${POLL_S:-120}" \
  JMIN_ALPHA=8 \
  JMAX_ALPHA=8 \
  JMIN_BETA="${JMIN_BETA:-4}" \
  JMIN_GAMMA="${JMIN_GAMMA:-4}" \
  JMAX_BETA=8 \
  JMAX_GAMMA=8 \
  ALPHA_AVG_FIXED=256 \
  ORIENT_TOL="${ORIENT_TOL:-0.025}" \
  ORIENT_MAX_TOL="${ORIENT_MAX_TOL:-0.07}" \
  ORIENT_SCALE_TOL="${ORIENT_SCALE_TOL:-0.025}" \
  ORIENT_COMPONENT_FLOOR="${ORIENT_COMPONENT_FLOOR:-1e-4}" \
  ADDA_COMPARE_COMPONENT_FLOOR="${ADDA_COMPARE_COMPONENT_FLOOR:-1e-3}" \
  PILOT_MESH="$DUST_MESH" \
  FINAL_MESH="$DUST_MESH" \
  FINAL_FROM_PILOT=1 \
  PILOT_SYSTEM="${PILOT_SYSTEM:-muller2-balanced}" \
  PILOT_QUAD="${PILOT_QUAD:-7}" \
  PILOT_DIGITS="${PILOT_DIGITS:-4}" \
  PILOT_GMRES_TOL="${PILOT_GMRES_TOL:-1e-2}" \
  PILOT_GMRES_RESTART="${PILOT_GMRES_RESTART:-160}" \
  PILOT_MAX_LEAF="${PILOT_MAX_LEAF:-128}" \
  DISABLE_PREC=1 \
  ORIENT_WARM_START=previous \
  scripts/run_dust_jmax8_adaptive_sweep.sh
