#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_accuracy_matrix_case.sh --gpu N --case NAME [--out DIR] [--bin PATH] [--force] [guard options]

Runs one named case from the production accuracy matrix on one GPU.
This is intended for reliable manual recovery: different case names can be
started on different GPUs without splitting one case across several GPUs.

In addition to the fixed production names, NAME may be a parameterized case:
  sphere_ka30_ref7_current_q13_d7_tol1e3
  hex_ka30_ref6_balanced_q13_d6_tol5e4
  dust_ka20_gmsh7000_balanced_q13_d6_tol5e4
  dust_ka20_gmsh4200_balanced_q7_d6_tol5e4

Options:
  --gpu N                CUDA device index visible to nvidia-smi
  --case NAME            Case name from run_accuracy_matrix_15_queue.sh --plan
  --out DIR              Output directory (default: runs/production_matrix_15)
  --bin PATH             BEM executable (default: bin/bem_cuda_fmm.next if present)
  --force                Archive and replace existing JSON/log for this case
  --max-power W          Passed to run_guarded_bem_case.sh
  --max-temp C           Passed to run_guarded_bem_case.sh
  --max-bad-samples N    Passed to run_guarded_bem_case.sh
  --interval SEC         Passed to run_guarded_bem_case.sh
  --print                Print the resolved command instead of running it
  --allow-gpu-share      Do not take the per-GPU run lock. Default: one case
                         per GPU for this output directory.
  --allow-compute-share  Allow starting while nvidia-smi reports existing CUDA
                         compute processes on the target GPU.

Environment:
  BEM_ALLOW_LEGACY_DUST=1
                         Allow archived dust q7/d5/tol1e-3 cases. New dust
                         production runs should use q7/d6/tol5e-4 names.
  BEM_NVIDIA_SMI         nvidia-smi command/path used for the final preflight.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${REPO:-$(cd "$script_dir/.." && pwd)}"
cd "$repo"

gpu=""
case_name=""
out="runs/production_matrix_15"
bin=""
force=0
print_only=0
guard_args=()
allow_gpu_share="${BEM_ALLOW_GPU_SHARE:-0}"
allow_compute_share="${BEM_ALLOW_COMPUTE_SHARE:-0}"
nvidia_smi="${BEM_NVIDIA_SMI:-nvidia-smi}"
gpu_lock_fd=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --case|--name) case_name="$2"; shift 2 ;;
    --out|--out-dir) out="$2"; shift 2 ;;
    --bin) bin="$2"; shift 2 ;;
    --force) force=1; shift ;;
    --max-power|--max-temp|--max-bad-samples|--interval)
      guard_args+=("$1" "$2"); shift 2 ;;
    --print) print_only=1; shift ;;
    --allow-gpu-share) allow_gpu_share=1; shift ;;
    --allow-compute-share) allow_compute_share=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$gpu" || -z "$case_name" ]]; then
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

common_sphere=(--shape sphere --ri 1.3116 0 --single --ntheta 181 --solver fmm --quad 7 --fmm-digits 6 --gmres-tol 3e-3 --gmres-restart 220 --max-leaf 96)
common_hex=(--shape hex_prism --prism-aspect 1.5 --ri 1.3116 0 --single --ntheta 181 --solver fmm --system balanced --quad 7 --fmm-digits 5 --gmres-tol 1e-3 --gmres-restart 200 --max-leaf 128 --no-prec)
# Legacy dust names are kept only to reproduce archived pre-accurate runs.
# New dust reruns use parameterized q*_d6_tol5e4 names below, which add
# --accurate, restart=500, and max_leaf=128.
legacy_dust=(--ri 1.6 0.002 --single --ntheta 181 --solver fmm --system balanced --quad 7 --fmm-digits 5 --gmres-tol 1e-3 --gmres-restart 220 --max-leaf 96 --no-prec)
allow_legacy_dust="${BEM_ALLOW_LEGACY_DUST:-0}"

mesh3400="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f3400_a35.obj"
mesh4200="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f4200_a35.obj"
mesh5200="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f5200_a35.obj"
mesh6000="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f6000_a45.obj"
mesh7000="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f7000_a45.obj"

tol_from_token() {
  local token="$1"
  if [[ "$token" =~ ^([0-9]+)e([0-9]+)$ ]]; then
    printf '%se-%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    return 0
  fi
  echo "bad tolerance token in case name: $token" >&2
  return 2
}

dust_mesh_from_label() {
  local label="$1"
  case "$label" in
    gmsh3400) printf '%s\n' "$mesh3400" ;;
    gmsh4200) printf '%s\n' "$mesh4200" ;;
    gmsh5200) printf '%s\n' "$mesh5200" ;;
    gmsh6000) printf '%s\n' "$mesh6000" ;;
    gmsh7000) printf '%s\n' "$mesh7000" ;;
    *) echo "unknown dust mesh label in case name: $label" >&2; return 2 ;;
  esac
}

validate_quad() {
  local quad="$1"
  case "$quad" in
    4|7|13) return 0 ;;
    *)
      echo "unsupported quadrature in case name: q$quad" >&2
      echo "Supported values are q4, q7, q13." >&2
      return 2
      ;;
  esac
}

legacy_dust_case() {
  if [[ "$allow_legacy_dust" != "1" ]]; then
    echo "legacy dust case disabled: $case_name" >&2
    echo "Use the q7_d6_tol5e4 case name, or set BEM_ALLOW_LEGACY_DUST=1 only to reproduce archived runs." >&2
    return 2
  fi
  return 0
}

dust_accurate_case_contract() {
  local digits="$1"
  local tol="$2"
  if (( digits < 6 )); then
    echo "dust case disabled: $case_name" >&2
    echo "Parameterized dust cases require d6 or better; use q7_d6_tol5e4 or q13_d6_tol5e4." >&2
    return 2
  fi
  if ! python3 - "$tol" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except (IndexError, ValueError):
    sys.exit(1)
sys.exit(0 if math.isfinite(value) and value <= 5e-4 else 1)
PY
  then
    echo "dust case disabled: $case_name" >&2
    echo "Parameterized dust cases require gmres_tol <= 5e-4; use q7_d6_tol5e4 or q13_d6_tol5e4." >&2
    return 2
  fi
}

bem_args=()
require_complex=0
case "$case_name" in
  sphere_ka5_ref4_current_q7_d6_tol3e3) bem_args=(--ka 5 --ref 4 "${common_sphere[@]}") ;;
  sphere_ka10_ref4_current_q7_d6_tol3e3) bem_args=(--ka 10 --ref 4 "${common_sphere[@]}") ;;
  sphere_ka15_ref4_current_q7_d6_tol3e3) bem_args=(--ka 15 --ref 4 "${common_sphere[@]}") ;;
  sphere_ka20_ref4_current_q7_d6_tol3e3) bem_args=(--ka 20 --ref 4 "${common_sphere[@]}") ;;
  sphere_ka30_ref6_current_q7_d6_tol3e3) bem_args=(--ka 30 --ref 6 "${common_sphere[@]}") ;;
  hex_ka5_ref2_balanced_q7_d5_tol1e3) bem_args=(--ka 5 --ref 2 "${common_hex[@]}") ;;
  hex_ka10_ref3_balanced_q7_d5_tol1e3) bem_args=(--ka 10 --ref 3 "${common_hex[@]}") ;;
  hex_ka15_ref4_balanced_q7_d5_tol1e3) bem_args=(--ka 15 --ref 4 "${common_hex[@]}") ;;
  hex_ka20_ref4_balanced_q7_d5_tol1e3) bem_args=(--ka 20 --ref 4 "${common_hex[@]}") ;;
  hex_ka30_ref5_balanced_q7_d5_tol1e3) bem_args=(--ka 30 --ref 5 "${common_hex[@]}") ;;
  dust_ka5_gmsh3400_balanced_q7_d5_tol1e3) legacy_dust_case; require_complex=1; bem_args=(--obj "$mesh3400" --ka 5 "${legacy_dust[@]}") ;;
  dust_ka10_gmsh5200_balanced_q7_d5_tol1e3) legacy_dust_case; require_complex=1; bem_args=(--obj "$mesh5200" --ka 10 "${legacy_dust[@]}") ;;
  dust_ka15_gmsh6000_balanced_q7_d5_tol1e3) legacy_dust_case; require_complex=1; bem_args=(--obj "$mesh6000" --ka 15 "${legacy_dust[@]}") ;;
  dust_ka20_gmsh4200_balanced_q7_d5_tol1e3) legacy_dust_case; require_complex=1; bem_args=(--obj "$mesh4200" --ka 20 "${legacy_dust[@]}") ;;
  dust_ka30_gmsh7000_balanced_q7_d5_tol1e3) legacy_dust_case; require_complex=1; bem_args=(--obj "$mesh7000" --ka 30 "${legacy_dust[@]}") ;;
  *)
    if [[ "$case_name" =~ ^sphere_ka([0-9]+)_ref([0-9]+)_current_q([0-9]+)_d([0-9]+)_tol([0-9]+e[0-9]+)$ ]]; then
      ka="${BASH_REMATCH[1]}"
      ref="${BASH_REMATCH[2]}"
      quad="${BASH_REMATCH[3]}"
      digits="${BASH_REMATCH[4]}"
      tol="$(tol_from_token "${BASH_REMATCH[5]}")"
      validate_quad "$quad"
      bem_args=(--ka "$ka" --ref "$ref" --shape sphere --ri 1.3116 0 --single --ntheta 181 --solver fmm --quad "$quad" --fmm-digits "$digits" --gmres-tol "$tol" --gmres-restart 220 --max-leaf 96)
    elif [[ "$case_name" =~ ^hex_ka([0-9]+)_ref([0-9]+)_balanced_q([0-9]+)_d([0-9]+)_tol([0-9]+e[0-9]+)$ ]]; then
      ka="${BASH_REMATCH[1]}"
      ref="${BASH_REMATCH[2]}"
      quad="${BASH_REMATCH[3]}"
      digits="${BASH_REMATCH[4]}"
      tol="$(tol_from_token "${BASH_REMATCH[5]}")"
      validate_quad "$quad"
      bem_args=(--ka "$ka" --ref "$ref" --shape hex_prism --prism-aspect 1.5 --ri 1.3116 0 --single --ntheta 181 --solver fmm --system balanced --quad "$quad" --fmm-digits "$digits" --gmres-tol "$tol" --gmres-restart 220 --max-leaf 128 --no-prec)
    elif [[ "$case_name" =~ ^dust_ka([0-9]+)_([a-z0-9]+)_balanced_q([0-9]+)_d([0-9]+)_tol([0-9]+e[0-9]+)$ ]]; then
      ka="${BASH_REMATCH[1]}"
      mesh_label="${BASH_REMATCH[2]}"
      quad="${BASH_REMATCH[3]}"
      digits="${BASH_REMATCH[4]}"
      tol="$(tol_from_token "${BASH_REMATCH[5]}")"
      mesh_path="$(dust_mesh_from_label "$mesh_label")"
      require_complex=1
      validate_quad "$quad"
      dust_accurate_case_contract "$digits" "$tol"
      bem_args=(--obj "$mesh_path" --ka "$ka" --ri 1.6 0.002 --single --ntheta 181 --solver fmm --accurate --system balanced --quad "$quad" --fmm-digits "$digits" --gmres-tol "$tol" --gmres-restart 500 --max-leaf 128 --no-prec)
    else
      echo "unknown case: $case_name" >&2
      echo "Known fixed cases:" >&2
      scripts/run_accuracy_matrix_15_queue.sh --plan >&2
      echo "Parameterized examples:" >&2
      echo "  sphere_ka30_ref7_current_q13_d7_tol1e3" >&2
      echo "  hex_ka30_ref6_balanced_q13_d6_tol5e4" >&2
      echo "  dust_ka20_gmsh7000_balanced_q13_d6_tol5e4" >&2
      echo "  dust_ka20_gmsh4200_balanced_q7_d6_tol5e4" >&2
      exit 2
    fi
    ;;
esac

cmd=(scripts/run_guarded_bem_case.sh --gpu "$gpu" --name "$case_name" --out-dir "$out" --bin "$bin")
cmd+=("${guard_args[@]}")
if [[ "$allow_compute_share" == "1" ]]; then
  cmd+=(--allow-compute-share)
fi
if [[ "$force" == "1" ]]; then
  cmd+=(--force)
fi
if [[ "$require_complex" == "1" ]]; then
  cmd+=(--require-complex)
fi
cmd+=(-- "${bem_args[@]}")

if [[ "$print_only" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

acquire_gpu_lock() {
  local lock_dir lock_file
  [[ "$allow_gpu_share" == "1" ]] && return 0
  mkdir -p "$out/locks"
  lock_file="$out/locks/gpu_${gpu}.lock"
  if command -v flock >/dev/null 2>&1; then
    exec {gpu_lock_fd}>"$lock_file"
    if ! flock -n "$gpu_lock_fd"; then
      echo "GPU_LOCK active: $lock_file" >&2
      return 3
    fi
    printf 'pid=%s case=%s gpu=%s started=%s\n' "$$" "$case_name" "$gpu" "$(date -Is)" 1>&"$gpu_lock_fd"
    return 0
  fi

  lock_dir="$out/locks/gpu_${gpu}.lockdir"
  if mkdir "$lock_dir" 2>/dev/null; then
    printf 'pid=%s case=%s gpu=%s started=%s\n' "$$" "$case_name" "$gpu" "$(date -Is)" > "$lock_dir/owner"
    trap 'rm -rf "$lock_dir"' EXIT
    return 0
  fi
  echo "GPU_LOCK active: $lock_dir" >&2
  return 3
}

acquire_gpu_lock

preflight_compute_apps() {
  local apps
  [[ "$allow_compute_share" == "1" ]] && return 0
  if ! command -v "$nvidia_smi" >/dev/null 2>&1; then
    return 0
  fi
  apps="$("$nvidia_smi" -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  apps="$(printf '%s\n' "$apps" | sed '/^[[:space:]]*$/d' | head -n 3)"
  if [[ -n "$apps" ]]; then
    apps="${apps//$'\n'/; }"
    echo "GPU_BUSY gpu=$gpu compute_apps=$apps" >&2
    return 3
  fi
}

preflight_compute_apps
exec "${cmd[@]}"
