#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_accuracy_refinement_wave.sh [--run] [--gpus LIST] [--max-cases N] [options]

Plans and optionally starts one refinement wave for the production accuracy
matrix. The default mode is dry-run. A wave contains different case names only;
resume_accuracy_matrix_cases.sh assigns at most one case to each GPU unless
--allow-oversubscribe is passed explicitly.

Options:
  --run                  Execute the planned wave; default is dry-run
  --csv FILE             Accuracy CSV (default: poster_a0/assets/table_accuracy_matrix_15.csv)
  --refresh-audit        Rebuild the accuracy CSV before planning
  --gpus LIST            Space/comma-separated GPU ids, or auto (default: 0 1 2)
  --max-cases N          Maximum cases in this wave; default: number of usable GPUs
  --all-cases            Plan every pending case
  --only-reason MODE     all, accuracy, or metadata (default: all)
  --out DIR              Output directory for refinement runs (default: runs/production_matrix_refinement)
  --plan-csv FILE        Plan CSV path (default: OUT/plan.csv)
  --case-max-power W     Guard power limit for each launched case (default: 290)
  --case-max-bad-samples N
                         Guard bad-sample limit for each launched case (default: 4)
  --no-health-check      Do not query nvidia-smi while planning/selecting GPUs
  --allow-compute-share  Allow scheduling on GPUs with existing CUDA compute apps
  --allow-oversubscribe  Allow resume script to put multiple cases on one GPU
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO:-$(cd "$script_dir/.." && pwd)}"
cd "$repo"

run=0
csv="poster_a0/assets/table_accuracy_matrix_15.csv"
refresh_audit=0
gpus="${BEM_REFINEMENT_WAVE_GPUS:-0 1 2}"
max_cases=""
all_cases=0
only_reason="all"
out="runs/production_matrix_refinement"
plan_csv=""
case_max_power="${BEM_REFINEMENT_WAVE_MAX_POWER:-290}"
case_max_bad_samples="${BEM_REFINEMENT_WAVE_MAX_BAD_SAMPLES:-4}"
no_health_check=0
allow_compute_share=0
allow_oversubscribe=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) run=1; shift ;;
    --dry-run) run=0; shift ;;
    --csv) csv="$2"; shift 2 ;;
    --refresh-audit) refresh_audit=1; shift ;;
    --gpus) gpus="$2"; shift 2 ;;
    --max-cases) max_cases="$2"; shift 2 ;;
    --all-cases) all_cases=1; shift ;;
    --only-reason) only_reason="$2"; shift 2 ;;
    --out|--out-dir) out="$2"; shift 2 ;;
    --plan-csv) plan_csv="$2"; shift 2 ;;
    --case-max-power) case_max_power="$2"; shift 2 ;;
    --case-max-bad-samples) case_max_bad_samples="$2"; shift 2 ;;
    --no-health-check) no_health_check=1; shift ;;
    --allow-compute-share) allow_compute_share=1; shift ;;
    --allow-oversubscribe) allow_oversubscribe=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

plan_args=(
  --csv "$csv"
  --gpus "$gpus"
  --only-reason "$only_reason"
  --out "$out"
  --case-max-power "$case_max_power"
  --case-max-bad-samples "$case_max_bad_samples"
)

if [[ "$refresh_audit" == "1" ]]; then
  plan_args+=(--refresh-audit)
fi
if [[ -n "$max_cases" ]]; then
  plan_args+=(--max-cases "$max_cases")
fi
if [[ "$all_cases" == "1" ]]; then
  plan_args+=(--all-cases)
fi
if [[ -n "$plan_csv" ]]; then
  plan_args+=(--plan-csv "$plan_csv")
fi
if [[ "$no_health_check" == "1" ]]; then
  plan_args+=(--no-health-check)
fi
if [[ "$allow_compute_share" == "1" ]]; then
  plan_args+=(--allow-compute-share)
fi
execute_plan_args=("${plan_args[@]}")
if [[ "$run" == "1" ]]; then
  execute_plan_args+=(--run --execute)
fi

mode="dry-run"
if [[ "$run" == "1" ]]; then
  mode="run"
fi
echo "REFINEMENT_WAVE mode=$mode gpus=$gpus out=$out"

if [[ "$allow_oversubscribe" == "1" ]]; then
  tmp_plan="$(mktemp)"
  python3 scripts/plan_accuracy_refinement_cases.py "${plan_args[@]}" --no-plan-csv > "$tmp_plan"
  cat "$tmp_plan"
  cases="$(awk '/^[A-Za-z0-9_]+$/ {printf "%s%s", sep, $0; sep=","}' "$tmp_plan")"
  rm -f "$tmp_plan"
  if [[ -z "$cases" ]]; then
    exit 0
  fi
  cmd=(scripts/resume_accuracy_matrix_cases.sh --gpus "$gpus" --cases "$cases"
       --case-max-power "$case_max_power" --case-max-bad-samples "$case_max_bad_samples"
       --out "$out" --allow-oversubscribe)
  if [[ -n "$max_cases" ]]; then
    cmd+=(--max-jobs "$max_cases")
  fi
  if [[ "$no_health_check" == "1" ]]; then
    cmd+=(--no-health-check)
  fi
  if [[ "$allow_compute_share" == "1" ]]; then
    cmd+=(--allow-compute-share)
  fi
  if [[ "$run" == "1" ]]; then
    cmd+=(--run)
  fi
  echo
  echo "oversubscribe_command:"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "$run" == "1" ]]; then
    "${cmd[@]}"
  fi
else
  python3 scripts/plan_accuracy_refinement_cases.py "${execute_plan_args[@]}"
fi
