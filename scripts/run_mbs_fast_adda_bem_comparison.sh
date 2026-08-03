#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

ROOT="${ROOT:-${REPO_ROOT}/runs/ref6_vs_adda_fp32_ka_gt60_20260802}"
MBS_ROOT="${MBS_ROOT:-${HOME}/MBS-fast}"
MBS_BIN="${MBS_BIN:-${MBS_ROOT}/bin/mbs_po}"

if [[ ! -x "${MBS_BIN}" ]]; then
    printf 'MBS-fast binary is not executable: %s\n' "${MBS_BIN}" >&2
    exit 1
fi

for ka in 60 80 111; do
    case_dir="${ROOT}/mbs_fast_po_ka${ka}"
    result_stem="${case_dir}/mbs"
    mkdir -p "${case_dir}"

    command=(
        "${MBS_BIN}"
        --po
        --fixed 0 30
        -p 1 0.8660254037844386 1
        --k_eq "${ka}"
        --ri 1.3 0
        -w 1
        -n 12
        --grid 0 180 1 360
        --threads 16
        --close
        --no_cbs
        --no_finite_edge
        --no_beam_edge_fringe
        -o "${result_stem}"
    )

    printf '%q ' "${command[@]}" > "${case_dir}/command.sh"
    printf '\n' >> "${case_dir}/command.sh"
    start_ns=$(date +%s%N)
    "${command[@]}" > "${case_dir}/run.log" 2>&1
    end_ns=$(date +%s%N)
    wall_s=$(awk -v start="${start_ns}" -v end="${end_ns}" \
        'BEGIN { printf "%.9f", (end - start) / 1.0e9 }')
    printf 'ACTUAL_WALL_S=%s\n' "${wall_s}" > "${case_dir}.time"
    printf 'ka=%s wall=%s s result=%s\n' \
        "${ka}" "${wall_s}" "${result_stem}/mbs.dat"
done
