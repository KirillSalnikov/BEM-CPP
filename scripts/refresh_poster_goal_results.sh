#!/usr/bin/env bash
set -euo pipefail

remote="${BEM_REMOTE:-kirill_epyc@172.16.0.117}"
remote_repo="${BEM_REMOTE_REPO:-/home/kirill_epyc/BEM-CUDA}"
ssh_opts=(-o IdentitiesOnly=yes -o ConnectTimeout=10)

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

poster="${BEM_POSTER_DIR:-poster_a0_work}"
refresh_root="runs/poster_goal_refresh_20260629"
refine_root="runs/production_matrix_refinement"
mkdir -p "$refresh_root/logs" "$refine_root/logs"

fetch_if_exists() {
  local remote_path="$1"
  local local_path="$2"
  if ssh "${ssh_opts[@]}" "$remote" "test -s '$remote_repo/$remote_path'"; then
    mkdir -p "$(dirname "$local_path")"
    rsync -a -e "ssh ${ssh_opts[*]}" "$remote:$remote_repo/$remote_path" "$local_path"
    echo "FETCHED $remote_path"
    return 0
  fi
  echo "MISSING $remote_path"
  return 1
}

fetch_optional() {
  local remote_path="$1"
  local local_path="$2"
  if ssh "${ssh_opts[@]}" "$remote" "test -e '$remote_repo/$remote_path'"; then
    mkdir -p "$(dirname "$local_path")"
    rsync -a -e "ssh ${ssh_opts[*]}" "$remote:$remote_repo/$remote_path" "$local_path"
    echo "FETCHED $remote_path"
  else
    echo "MISSING $remote_path"
  fi
}

fetch_optional "$refresh_root/logs/gpu_monitor_20260629_124840.csv" \
  "$refresh_root/logs/gpu_monitor_20260629_124840.csv"
fetch_optional "$refine_root/logs/gpu1.dust_refinement_queue.log" \
  "$refine_root/logs/gpu1.dust_refinement_queue.log"

declare -a fetched=()
for path in \
  "$refresh_root/hex_ka30_ref6_balanced_q7_d5_tol1e3_leaf256.json" \
  "$refresh_root/hex_ka30_ref6_balanced_q7_d6_tol5e4.json" \
  "$refresh_root/hex_ka30_ref6_balanced_q13_d6_tol5e4.json" \
  "$refine_root/dust_ka10_gmsh6000_balanced_q13_d6_tol5e4.json" \
  "$refine_root/dust_ka5_gmsh5200_balanced_q13_d6_tol5e4.json" \
  "$refine_root/dust_ka15_gmsh5200_balanced_q13_d6_tol5e4.json"; do
  if fetch_if_exists "$path" "$path"; then
    fetched+=("$path")
  fi
done

for path in "${fetched[@]}"; do
  case "$path" in
    *hex_ka30_ref6_balanced_q7_d5_tol1e3_leaf256.json)
      python3 scripts/update_poster_candidate_row.py --poster "$poster" \
        --shape 'гексагональная призма' --ka 30 \
        --mesh-label 'ref6/q7d5-leaf256' --mesh-ref 6 \
        --bem "$path" --reference adda --adda runs/adda_ocl_benchmark_ext/hex_ka30/mueller \
        --backend 'BEM-CUDA FMM ref6 q7 d5 leaf256'
      ;;
    *hex_ka30_ref6_balanced_q13_d6_tol5e4.json)
      python3 scripts/update_poster_candidate_row.py --poster "$poster" \
        --shape 'гексагональная призма' --ka 30 \
        --mesh-label 'ref6/q13' --mesh-ref 6 \
        --bem "$path" --reference adda --adda runs/adda_ocl_benchmark_ext/hex_ka30/mueller \
        --backend 'BEM-CUDA FMM ref6 q13 d6'
      ;;
    *hex_ka30_ref6_balanced_q7_d6_tol5e4.json)
      python3 scripts/update_poster_candidate_row.py --poster "$poster" \
        --shape 'гексагональная призма' --ka 30 \
        --mesh-label 'ref6/q7d6' --mesh-ref 6 \
        --bem "$path" --reference adda --adda runs/adda_ocl_benchmark_ext/hex_ka30/mueller \
        --backend 'BEM-CUDA FMM ref6 q7 d6'
      ;;
    *dust_ka10_gmsh6000_balanced_q13_d6_tol5e4.json)
      python3 scripts/update_poster_candidate_row.py --poster "$poster" \
        --shape 'пылевая частица' --ka 10 \
        --mesh-label 'gmsh6000/q13d6' --mesh-ref 6000 \
        --bem "$path" --reference adda --adda runs/adda_ocl_benchmark_ext/dust_ka10_m1p6_dpl20_scaled/mueller \
        --backend 'BEM-CUDA dust q13 d6'
      ;;
    *dust_ka5_gmsh5200_balanced_q13_d6_tol5e4.json)
      python3 scripts/update_poster_candidate_row.py --poster "$poster" \
        --shape 'пылевая частица' --ka 5 \
        --mesh-label 'gmsh5200/q13d6' --mesh-ref 5200 \
        --bem "$path" --reference adda --adda runs/adda_ocl_benchmark_ext/dust_ka5_m1p6_dpl35_scaled/mueller \
        --backend 'BEM-CUDA dust q13 d6'
      ;;
    *dust_ka15_gmsh5200_balanced_q13_d6_tol5e4.json)
      python3 scripts/update_poster_candidate_row.py --poster "$poster" \
        --shape 'пылевая частица' --ka 15 \
        --mesh-label 'gmsh5200/q13d6' --mesh-ref 5200 \
        --bem "$path" --reference adda --adda runs/adda_ocl_benchmark_ext/dust_ka15_m1p6_dpl20_scaled/mueller \
        --backend 'BEM-CUDA dust q13 d6'
      ;;
  esac
done

python3 "$poster/make_assets.py"
(
  cd "$poster"
  pdflatex -interaction=nonstopmode poster_a0.tex >/tmp/poster_refresh_pdflatex1.log
  pdflatex -interaction=nonstopmode poster_a0.tex >/tmp/poster_refresh_pdflatex2.log
)
python3 "$poster/validate_poster.py"

python3 - <<'PY'
import pandas as pd
acc = pd.read_csv('poster_a0_work/assets/table_accuracy_matrix_15.csv')
gate = pd.to_numeric(acc['gate_error'], errors='coerce')
print(f"SUMMARY pass5={(gate <= 0.05).sum()}/{len(acc)} pass10={(gate <= 0.10).sum()}/{len(acc)}")
print(pd.read_csv('poster_a0_work/assets/table_current_accuracy_gaps.csv').to_string(index=False))
PY
