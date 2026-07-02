#!/usr/bin/env bash
# Shared shell helpers for avoiding accidental CUDA oversubscription.

: "${NVIDIA_SMI:=${BEM_NVIDIA_SMI:-nvidia-smi}}"

bem_gpu_compute_apps() {
  local gpu="$1"
  "$NVIDIA_SMI" -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null |
    sed '/^[[:space:]]*$/d' || true
}

bem_gpu_busy_text() {
  local gpu="$1"
  bem_gpu_compute_apps "$gpu" | tr '\n' ';'
}

bem_require_gpu_free() {
  local gpu="$1" allow_compute_share="${2:-0}" apps
  if [[ "$allow_compute_share" == "1" ]]; then
    return 0
  fi
  apps="$(bem_gpu_busy_text "$gpu")"
  if [[ -n "$apps" ]]; then
    echo "GPU_BUSY gpu=$gpu compute_apps=$apps" >&2
    return 3
  fi
}

bem_filter_free_gpus_csv() {
  local gpus_csv="$1" allow_compute_share="${2:-0}" gpu apps
  tr ',' '\n' <<<"$gpus_csv" | while read -r gpu; do
    [[ -n "$gpu" ]] || continue
    if [[ "$allow_compute_share" != "1" ]]; then
      apps="$(bem_gpu_busy_text "$gpu")"
      if [[ -n "$apps" ]]; then
        echo "GPU_BUSY gpu=$gpu compute_apps=$apps" >&2
        continue
      fi
    fi
    printf '%s\n' "$gpu"
  done
}
