#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO:-$(cd "$script_dir/.." && pwd)}"
out="${OUT:-runs/production_matrix_15}"
default_bin() {
  local root="${1:-$repo}"
  if [[ -x "$root/bin/bem_cuda_fmm.next" ]]; then
    printf '%s\n' "$root/bin/bem_cuda_fmm.next"
  else
    printf '%s\n' "$root/bin/bem_cuda_fmm"
  fi
}
bin="${BIN:-$(default_bin "$repo")}"
adda_ocl="${ADDA_OCL:-/home/kirill_epyc/adda/src/ocl/adda_ocl}"
nvidia_smi="${BEM_NVIDIA_SMI:-nvidia-smi}"
mkdir -p "$out/logs"

archive_bad_result() {
  local name="$1"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  if [[ -e "$out/$name.json" ]]; then
    mv "$out/$name.json" "$out/$name.bad_$stamp.json"
  fi
  if [[ -e "$out/logs/$name.log" ]]; then
    mv "$out/logs/$name.log" "$out/logs/$name.bad_$stamp.log"
  fi
}

result_is_current() {
  local name="$1"
  [[ -s "$out/$name.json" ]] || return 1
  local args=(--strict --require-converged --validate-numeric)
  if [[ "${BEM_METADATA_SKIP_CLOUDE:-0}" != "1" ]]; then
    args+=(--require-cloude-physical)
  fi
  if result_requires_complex_operator "$name"; then
    args+=(--require-complex-operator)
  fi
  python3 scripts/check_result_metadata.py "${args[@]}" "$out/$name.json" >/dev/null 2>&1
}

result_requires_complex_operator() {
  local name="$1"
  [[ "$name" == dust_* ]]
}

queue_lock_fd=
acquire_queue_lock() {
  local lock="$out/.queue.lock"
  mkdir -p "$out"
  if command -v flock >/dev/null 2>&1; then
    exec {queue_lock_fd}>"$lock"
    if ! flock -n "$queue_lock_fd"; then
      echo "QUEUE_LOCK active: $lock" >&2
      return 1
    fi
    printf '%s\n' "$$" 1>&"$queue_lock_fd"
    return 0
  fi

  if mkdir "$lock" 2>/dev/null; then
    printf '%s\n' "$$" > "$lock/pid"
    trap 'rm -rf "$out/.queue.lock"' EXIT
    return 0
  fi
  if [[ -s "$lock/pid" ]] && kill -0 "$(cat "$lock/pid")" 2>/dev/null; then
    echo "QUEUE_LOCK active: $lock pid=$(cat "$lock/pid")" >&2
    return 1
  fi
  rm -rf "$lock"
  mkdir "$lock"
  printf '%s\n' "$$" > "$lock/pid"
  trap 'rm -rf "$out/.queue.lock"' EXIT
}

expected_result_names() {
  cat <<'EOF'
sphere_ka5_ref4_current_q7_d6_tol3e3
sphere_ka10_ref4_current_q7_d6_tol3e3
sphere_ka15_ref4_current_q7_d6_tol3e3
sphere_ka20_ref4_current_q7_d6_tol3e3
sphere_ka30_ref6_current_q7_d6_tol3e3
hex_ka5_ref2_balanced_q7_d5_tol1e3
hex_ka10_ref3_balanced_q7_d5_tol1e3
hex_ka15_ref4_balanced_q7_d5_tol1e3
hex_ka20_ref4_balanced_q7_d5_tol1e3
hex_ka30_ref5_balanced_q7_d5_tol1e3
dust_ka5_gmsh3400_balanced_q7_d6_tol5e4
dust_ka10_gmsh5200_balanced_q7_d6_tol5e4
dust_ka15_gmsh6000_balanced_q7_d6_tol5e4
dust_ka20_gmsh4200_balanced_q7_d6_tol5e4
dust_ka30_gmsh7000_balanced_q7_d6_tol5e4
EOF
}

planned_run_names() {
  expected_result_names
  if [[ "${BEM_QUEUE_EXTRA_DUST_VARIANTS:-0}" == "1" ]]; then
    cat <<'EOF'
dust_ka5_gmsh4200_balanced_q7_d6_tol5e4
dust_ka5_adda_cubical_raw_balanced_q7_d6_tol5e4
dust_ka5_adda_cubical_f6000_balanced_q7_d6_tol5e4
dust_ka5_adda_mc_s0p35_l0p42_f6000_balanced_q7_d6_tol5e4
dust_ka5_adda_mc_s0p5_l0p42_f6000_balanced_q7_d6_tol5e4
dust_ka10_gmsh6000_balanced_q7_d6_tol5e4
EOF
  fi
}

queue_status() {
  local name current=0 stale=0 missing=0
  while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    if result_is_current "$name"; then
      echo "CURRENT $name"
      current=$((current + 1))
    elif [[ -e "$out/$name.json" ]]; then
      echo "STALE   $name"
      stale=$((stale + 1))
    else
      echo "MISSING $name"
      missing=$((missing + 1))
    fi
  done < <(expected_result_names)
  echo "SUMMARY current=$current stale=$stale missing=$missing total=$((current + stale + missing))"
  [[ "$current" -eq 15 && "$stale" -eq 0 && "$missing" -eq 0 ]]
}

accuracy_status() {
  local csv="${BEM_QUEUE_ACCURACY_CSV:-}"
  local audit_rc=0
  local names_file="$out/logs/.accuracy_expected_names.txt"
  mkdir -p "$out/logs"
  expected_result_names > "$names_file"
  if [[ -z "$csv" ]]; then
    csv="$out/logs/.accuracy_status.csv"
    python3 scripts/audit_accuracy_matrix_15.py --out "$csv" \
      > "$out/logs/.accuracy_status.log" 2>&1 || audit_rc="$?"
  fi
  python3 - "$csv" "$names_file" <<'PY'
import csv
import math
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
names_path = Path(sys.argv[2])
names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
rows = []
if csv_path.exists():
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

by_name = {}
by_key = {}
for row in rows:
    bem_file = row.get("bem_file", "")
    base = Path(bem_file).name
    if base:
        by_name[base] = row
    shape = row.get("shape", "")
    try:
        ka = float(row.get("ka", "nan"))
    except ValueError:
        ka = math.nan
    if shape and math.isfinite(ka):
        by_key[(shape, round(ka, 8))] = row

def case_key(name):
    parts = name.split("_")
    if len(parts) < 2:
        return None
    if parts[0] == "sphere" and parts[1].startswith("ka"):
        return ("сфера", round(float(parts[1][2:]), 8))
    if parts[0] == "hex" and parts[1].startswith("ka"):
        return ("гексагональная призма", round(float(parts[1][2:]), 8))
    if parts[0] == "dust" and parts[1].startswith("ka"):
        return ("пылевая частица", round(float(parts[1][2:]), 8))
    return None

def pass10(row):
    value = str(row.get("pass10", "")).strip().lower()
    if value in {"true", "1", "yes", "pass"}:
        return True
    return False

counts = {
    "accurate": 0,
    "accurate_legacy": 0,
    "inaccurate": 0,
    "missing": 0,
}
for name in names:
    row = by_name.get(f"{name}.json")
    key = case_key(name)
    if row is None and key is not None:
        row = by_key.get(key)
    if row is None:
        print(f"MISSING_ACCURACY {name}")
        counts["missing"] += 1
        continue
    ok = pass10(row)
    metadata_ok = row.get("metadata_status") == "ok"
    operator_ok = row.get("operator_status") in {"complex_operator", "not_required"}
    gate = row.get("gate_error", "")
    if ok and metadata_ok and operator_ok:
        print(f"ACCURATE {name} gate={gate}")
        counts["accurate"] += 1
    elif ok:
        print(
            f"ACCURATE_LEGACY {name} gate={gate} "
            f"metadata={row.get('metadata_status', '')} operator={row.get('operator_status', '')}"
        )
        counts["accurate_legacy"] += 1
    else:
        print(
            f"INACCURATE {name} gate={gate} "
            f"metadata={row.get('metadata_status', '')} operator={row.get('operator_status', '')}"
        )
        counts["inaccurate"] += 1

total = sum(counts.values())
print(
    "SUMMARY_ACCURACY "
    f"accurate={counts['accurate']} "
    f"accurate_legacy={counts['accurate_legacy']} "
    f"inaccurate={counts['inaccurate']} "
    f"missing={counts['missing']} total={total}"
)
sys.exit(0 if counts["accurate"] == total else 2)
PY
  local status_rc="$?"
  if [[ "$audit_rc" -ne 0 && "$status_rc" -eq 0 ]]; then
    return "$audit_rc"
  fi
  return "$status_rc"
}

require_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "PREFLIGHT missing file: $path" >&2
    return 1
  fi
}

gpu_count() {
  "$nvidia_smi" --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l
}

trim_number() {
  local value="$1"
  value="${value// /}"
  value="${value%%.*}"
  printf '%s\n' "$value"
}

gpu_health_check() {
  local gpu="$1"
  local max_temp="${BEM_QUEUE_MAX_TEMP_C:-80}"
  local max_util="${BEM_QUEUE_MAX_UTIL_PCT:-20}"
  local max_mem="${BEM_QUEUE_MAX_MEM_MB:-2048}"
  local allow_compute_share="${BEM_QUEUE_ALLOW_COMPUTE_SHARE:-0}"
  local line temp util mem apps
  if ! line="$("$nvidia_smi" -i "$gpu" --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits 2>&1)"; then
    echo "GPU_HEALTH fail gpu=$gpu $nvidia_smi: $line" >&2
    return 1
  fi
  IFS=',' read -r temp util mem <<<"$line"
  temp="$(trim_number "$temp")"
  util="$(trim_number "$util")"
  mem="$(trim_number "$mem")"
  if [[ -z "$temp" || -z "$util" || -z "$mem" ]]; then
    echo "GPU_HEALTH fail gpu=$gpu unparsable: $line" >&2
    return 1
  fi
  if (( temp > max_temp )); then
    echo "GPU_HEALTH fail gpu=$gpu temp=${temp}C max=${max_temp}C" >&2
    return 1
  fi
  if (( util > max_util )); then
    echo "GPU_HEALTH fail gpu=$gpu util=${util}% max=${max_util}%" >&2
    return 1
  fi
  if (( mem > max_mem )); then
    echo "GPU_HEALTH fail gpu=$gpu mem=${mem}MiB max=${max_mem}MiB" >&2
    return 1
  fi
  if [[ "$allow_compute_share" != "1" ]]; then
    apps="$("$nvidia_smi" -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    apps="$(printf '%s\n' "$apps" | sed '/^[[:space:]]*$/d' | head -n 3)"
    if [[ -n "$apps" ]]; then
      apps="${apps//$'\n'/; }"
      echo "GPU_HEALTH fail gpu=$gpu compute_apps=$apps" >&2
      return 1
    fi
  fi
  echo "GPU_HEALTH ok gpu=$gpu temp=${temp}C util=${util}% mem=${mem}MiB"
}

gpu_runtime_sample() {
  local gpu="$1"
  local line temp util mem power
  if ! line="$("$nvidia_smi" -i "$gpu" --query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>&1)"; then
    echo "GPU_SAMPLE fail gpu=$gpu $nvidia_smi: $line" >&2
    return 1
  fi
  IFS=',' read -r temp util mem power <<<"$line"
  temp="$(trim_number "$temp")"
  util="$(trim_number "$util")"
  mem="$(trim_number "$mem")"
  power="$(trim_number "$power")"
  if [[ -z "$temp" || -z "$util" || -z "$mem" || -z "$power" ]]; then
    echo "GPU_SAMPLE fail gpu=$gpu unparsable: $line" >&2
    return 1
  fi
  printf '%s,%s,%s,%s\n' "$temp" "$util" "$mem" "$power"
}

run_with_gpu_monitor() {
  local gpu="$1" name="$2"
  shift 2
  local monitor_log="$out/logs/$name.gpu.csv"
  local interval="${BEM_QUEUE_MONITOR_INTERVAL_S:-5}"
  local pid rc sample temp util mem power ts

  printf 'timestamp_s,gpu,temp_c,util_pct,mem_mib,power_w\n' > "$monitor_log"
  if [[ "${BEM_QUEUE_STDBUF:-1}" != "0" ]] && command -v stdbuf >/dev/null 2>&1; then
    echo "QUEUE_STDOUT line_buffer=stdbuf"
    CUDA_VISIBLE_DEVICES="$gpu" \
      BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}" \
      BEM_GMRES_VERBOSE="${BEM_GMRES_VERBOSE:-1}" \
      BEM_GMRES_STAGNATION_CYCLES="${BEM_GMRES_STAGNATION_CYCLES:-0}" \
      BEM_GMRES_STAGNATION_REL="${BEM_GMRES_STAGNATION_REL:-0.003}" \
      stdbuf -oL -eL "$@" &
  else
    echo "QUEUE_STDOUT line_buffer=default"
    CUDA_VISIBLE_DEVICES="$gpu" \
      BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}" \
      BEM_GMRES_VERBOSE="${BEM_GMRES_VERBOSE:-1}" \
      BEM_GMRES_STAGNATION_CYCLES="${BEM_GMRES_STAGNATION_CYCLES:-0}" \
      BEM_GMRES_STAGNATION_REL="${BEM_GMRES_STAGNATION_REL:-0.003}" \
      "$@" &
  fi
  pid="$!"

  while kill -0 "$pid" 2>/dev/null; do
    ts="$(date +%s)"
    if sample="$(gpu_runtime_sample "$gpu" 2>> "$out/logs/$name.log")"; then
      IFS=',' read -r temp util mem power <<<"$sample"
      printf '%s,%s,%s,%s,%s,%s\n' "$ts" "$gpu" "$temp" "$util" "$mem" "$power" >> "$monitor_log"
    fi
    sleep "$interval"
  done

  set +e
  wait "$pid"
  rc="$?"
  set -e
  return "$rc"
}

preflight() {
  local failures=0 count
  local run_extra_dust_variants="${BEM_QUEUE_EXTRA_DUST_VARIANTS:-0}"
  cd "$repo"
  command -v python3 >/dev/null || { echo "PREFLIGHT missing command: python3" >&2; failures=$((failures + 1)); }
  command -v "$nvidia_smi" >/dev/null || { echo "PREFLIGHT missing command: $nvidia_smi" >&2; failures=$((failures + 1)); }
  [[ -x "$bin" ]] || { echo "PREFLIGHT missing executable: $bin" >&2; failures=$((failures + 1)); }
  [[ -s scripts/check_result_metadata.py ]] || { echo "PREFLIGHT missing script: scripts/check_result_metadata.py" >&2; failures=$((failures + 1)); }
  [[ -s scripts/audit_accuracy_matrix_15.py ]] || { echo "PREFLIGHT missing script: scripts/audit_accuracy_matrix_15.py" >&2; failures=$((failures + 1)); }
  if command -v "$nvidia_smi" >/dev/null; then
    if ! "$nvidia_smi" >/dev/null 2>&1; then
      echo "PREFLIGHT $nvidia_smi failed" >&2
      failures=$((failures + 1))
    else
      count="$(gpu_count)"
      echo "PREFLIGHT gpu_count=$count"
      if [[ "$count" -lt 3 ]]; then
        echo "PREFLIGHT need at least 3 visible GPUs for this queue" >&2
        failures=$((failures + 1))
      fi
      for gpu in 0 1 2; do
        gpu_health_check "$gpu" || failures=$((failures + 1))
      done
    fi
  fi
  require_file runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f3400_a35.obj || failures=$((failures + 1))
  require_file runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f4200_a35.obj || failures=$((failures + 1))
  require_file runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f5200_a35.obj || failures=$((failures + 1))
  require_file runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f6000_a45.obj || failures=$((failures + 1))
  require_file runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f7000_a45.obj || failures=$((failures + 1))
  if [[ "$run_extra_dust_variants" == "1" ]]; then
    require_file runs/production_matrix_15/meshes/dust5_adda_shape/adda_cubical_raw.obj || failures=$((failures + 1))
    require_file runs/production_matrix_15/meshes/dust5_adda_shape/adda_cubical_f6000_ag6.obj || failures=$((failures + 1))
    require_file runs/production_matrix_15/meshes/dust5_adda_shape/adda_mc_s0p35_l0p42_f6000.obj || failures=$((failures + 1))
    require_file runs/production_matrix_15/meshes/dust5_adda_shape/adda_mc_s0p5_l0p42_f6000.obj || failures=$((failures + 1))
  fi
  require_file runs/adda_ocl_benchmark_ext/shapes/greek_scaled_ka15_dpl20.shape || true
  queue_status || true
  if [[ "$failures" -ne 0 ]]; then
    echo "PREFLIGHT failed=$failures" >&2
    return 2
  fi
  echo "PREFLIGHT ok"
}

run_case() {
  local gpu="$1" name="$2"
  local gmres_verbose="${BEM_QUEUE_GMRES_VERBOSE:-1}"
  shift 2
  if result_is_current "$name"; then
    echo "SKIP $name current-metadata-ok"
    return 0
  fi
  if [[ -e "$out/$name.json" ]]; then
    echo "STALE $name: archiving existing result without current metadata"
    archive_bad_result "$name"
  fi
  gpu_health_check "$gpu" | tee "$out/logs/$name.prestart.log"
  echo "START $name gpu=$gpu"
  local rc
  set +e
  (
    export BEM_GMRES_VERBOSE="$gmres_verbose"
    run_with_gpu_monitor "$gpu" "$name" "$bin" "$@" --out "$out/$name.json"
  ) > "$out/logs/$name.log" 2>&1
  rc="$?"
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "FAIL $name rc=$rc" >> "$out/logs/$name.log"
    return "$rc"
  fi
  local meta_args=(--strict --require-converged --validate-numeric)
  if [[ "${BEM_METADATA_SKIP_CLOUDE:-0}" != "1" ]]; then
    meta_args+=(--require-cloude-physical)
  fi
  if result_requires_complex_operator "$name"; then
    meta_args+=(--require-complex-operator)
  fi
  python3 scripts/check_result_metadata.py "${meta_args[@]}" "$out/$name.json" >> "$out/logs/$name.log"
  python3 - "$out/$name.json" "$name" <<'PY'
import json, sys
p, name = sys.argv[1:3]
d = json.load(open(p))
method = d.get("method", {})
print("DONE", name,
      "total", d.get("timing", {}).get("total_s"),
      "system", method.get("system"),
      "profile", method.get("solver_profile"),
      "prec", method.get("preconditioner_reason"),
      "farfield", method.get("farfield_mode"),
      "matvecs", d.get("gmres_matvecs"),
      "nonconv", d.get("gmres_nonconverged_systems"),
      "max_relres", d.get("gmres_max_final_relres"))
PY
}

run_adda_dust15() {
  local gpu="$1"
  local dir="runs/adda_ocl_benchmark_ext/dust_ka15_m1p6_dpl20_scaled"
  local shape="runs/adda_ocl_benchmark_ext/shapes/greek_scaled_ka15_dpl20.shape"
  if [[ -s "$dir/mueller" ]]; then
    echo "SKIP adda dust_ka15"
    return 0
  fi
  if [[ ! -s "$shape" ]]; then
    python3 scripts/scale_adda_shape.py \
      runs/adda_greek_dpl25/greek_ka5p71_dpl25.shape \
      "$shape" --target-ka 15 --dpl 20
  fi
  mkdir -p "$dir"
  echo "START adda dust_ka15 gpu=$gpu"
  "$adda_ocl" -gpu "$gpu" -dir "$dir" -shape read "$shape" \
    -m 1.6 0.002 -dpl 20 -eps 5 -orient 0 0 0 -ntheta 181 \
    -scat_matr muel -sym no > "$dir/run.log" 2>&1
}

main() {
  local mode="${1:-run}"
  cd "$repo"

  if [[ "$mode" == "--status" || "$mode" == "status" ]]; then
    queue_status
    return $?
  fi
  if [[ "$mode" == "--status-accuracy" || "$mode" == "status-accuracy" ]]; then
    accuracy_status
    return $?
  fi
  if [[ "$mode" == "--preflight" || "$mode" == "preflight" ]]; then
    preflight
    return $?
  fi
  if [[ "$mode" == "--plan" || "$mode" == "plan" ]]; then
    planned_run_names
    return 0
  fi

  acquire_queue_lock
  preflight

  common_sphere=(--shape sphere --ri 1.3116 0 --single --ntheta 181 --solver fmm --quad 7 --fmm-digits 6 --gmres-tol 3e-3 --gmres-restart 220 --max-leaf 96)
  common_hex=(--shape hex_prism --prism-aspect 1.5 --ri 1.3116 0 --single --ntheta 181 --solver fmm --system balanced --quad 7 --fmm-digits 5 --gmres-tol 1e-3 --gmres-restart 200 --max-leaf 128 --no-prec)
  common_dust=(--ri 1.6 0.002 --single --ntheta 181 --solver fmm --accurate --system balanced --quad 7 --fmm-digits 6 --gmres-tol 5e-4 --gmres-restart 500 --max-leaf 128 --no-prec)

  mesh3400="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f3400_a35.obj"
  mesh4200="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f4200_a35.obj"
  mesh5200="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f5200_a35.obj"
  mesh6000="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f6000_a45.obj"
  mesh7000="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f7000_a45.obj"
  dust5_vox_dir="runs/production_matrix_15/meshes/dust5_adda_shape"
  run_extra_dust_variants="${BEM_QUEUE_EXTRA_DUST_VARIANTS:-0}"

  (
    run_case 0 sphere_ka5_ref4_current_q7_d6_tol3e3 --ka 5 --ref 4 "${common_sphere[@]}"
    run_case 0 sphere_ka10_ref4_current_q7_d6_tol3e3 --ka 10 --ref 4 "${common_sphere[@]}"
    run_case 0 sphere_ka15_ref4_current_q7_d6_tol3e3 --ka 15 --ref 4 "${common_sphere[@]}"
    run_case 0 sphere_ka20_ref4_current_q7_d6_tol3e3 --ka 20 --ref 4 "${common_sphere[@]}"
    run_case 0 sphere_ka30_ref6_current_q7_d6_tol3e3 --ka 30 --ref 6 "${common_sphere[@]}"
  ) &

  (
    run_case 1 hex_ka5_ref2_balanced_q7_d5_tol1e3 --ka 5 --ref 2 "${common_hex[@]}"
    run_case 1 hex_ka10_ref3_balanced_q7_d5_tol1e3 --ka 10 --ref 3 "${common_hex[@]}"
    run_case 1 hex_ka15_ref4_balanced_q7_d5_tol1e3 --ka 15 --ref 4 "${common_hex[@]}"
    run_case 1 hex_ka20_ref4_balanced_q7_d5_tol1e3 --ka 20 --ref 4 "${common_hex[@]}"
    run_case 1 hex_ka30_ref5_balanced_q7_d5_tol1e3 --ka 30 --ref 5 "${common_hex[@]}"
  ) &

  (
    run_case 2 dust_ka5_gmsh3400_balanced_q7_d6_tol5e4 --obj "$mesh3400" --ka 5 "${common_dust[@]}"
    if [[ "$run_extra_dust_variants" == "1" ]]; then
      run_case 2 dust_ka5_gmsh4200_balanced_q7_d6_tol5e4 --obj "$mesh4200" --ka 5 "${common_dust[@]}"
      run_case 2 dust_ka5_adda_cubical_raw_balanced_q7_d6_tol5e4 --obj "$dust5_vox_dir/adda_cubical_raw.obj" --ka 5 "${common_dust[@]}"
      run_case 2 dust_ka5_adda_cubical_f6000_balanced_q7_d6_tol5e4 --obj "$dust5_vox_dir/adda_cubical_f6000_ag6.obj" --ka 5 "${common_dust[@]}"
      run_case 2 dust_ka5_adda_mc_s0p35_l0p42_f6000_balanced_q7_d6_tol5e4 --obj "$dust5_vox_dir/adda_mc_s0p35_l0p42_f6000.obj" --ka 5 "${common_dust[@]}"
      run_case 2 dust_ka5_adda_mc_s0p5_l0p42_f6000_balanced_q7_d6_tol5e4 --obj "$dust5_vox_dir/adda_mc_s0p5_l0p42_f6000.obj" --ka 5 "${common_dust[@]}"
    fi
    run_case 2 dust_ka20_gmsh4200_balanced_q7_d6_tol5e4 --obj "$mesh4200" --ka 20 "${common_dust[@]}"
    run_case 2 dust_ka30_gmsh7000_balanced_q7_d6_tol5e4 --obj "$mesh7000" --ka 30 "${common_dust[@]}"
    run_case 2 dust_ka10_gmsh5200_balanced_q7_d6_tol5e4 --obj "$mesh5200" --ka 10 "${common_dust[@]}"
    run_case 2 dust_ka15_gmsh6000_balanced_q7_d6_tol5e4 --obj "$mesh6000" --ka 15 "${common_dust[@]}"
    run_adda_dust15 2
  ) &

  wait

  python3 scripts/audit_accuracy_matrix_15.py
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
