#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/BEM-CUDA}
ADDA_ROOT=${ADDA_ROOT:-/home/user/adda}
OUT=${OUT:-"$ROOT/runs/adda_ocl_benchmark"}
ADDA_OCL=${ADDA_OCL:-"$ADDA_ROOT/src/ocl/adda_ocl"}
NTHETA=${NTHETA:-181}
DPL=${DPL:-20}
EPS=${EPS:-5}
GPU=${GPU:-0}
TIMEOUT_S=${TIMEOUT_S:-900}
OUT_SUFFIX=${OUT_SUFFIX:-}
CASES=${CASES:-default}
NVIDIA_SMI="${BEM_NVIDIA_SMI:-nvidia-smi}"
ALLOW_COMPUTE_SHARE="${BEM_ALLOW_COMPUTE_SHARE:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/gpu_guard.sh"

mkdir -p "$OUT"
CSV="$OUT/summary.csv"
printf 'case,shape,ka,dpl,ntheta,status,time_s,backend,dir,log,note\n' > "$CSV"

quote_csv() {
    local s=${1//\"/\"\"}
    printf '"%s"' "$s"
}

emit_row() {
    local case_name=$1 shape=$2 ka=$3 dpl=$4 ntheta=$5 status=$6 time_s=$7 backend=$8 dir=$9 log=${10} note=${11}
    {
        quote_csv "$case_name"; printf ','
        quote_csv "$shape"; printf ','
        printf '%s,%s,%s,' "$ka" "$dpl" "$ntheta"
        quote_csv "$status"; printf ','
        printf '%s,' "$time_s"
        quote_csv "$backend"; printf ','
        quote_csv "$dir"; printf ','
        quote_csv "$log"; printf ','
        quote_csv "$note"; printf '\n'
    } >> "$CSV"
}

detect_unavailable() {
    if [[ ! -x "$ADDA_OCL" ]]; then
        local reasons=()
        if [[ ! -f "$ADDA_ROOT/src/ocl/Makefile" ]]; then
            reasons+=("missing ADDA OpenCL source Makefile under $ADDA_ROOT/src/ocl")
        fi
        if ! cpp -x c -include CL/cl.h /dev/null >/dev/null 2>&1; then
            reasons+=("missing OpenCL headers: CL/cl.h")
        fi
        if ! cpp -x c -include clFFT.h /dev/null >/dev/null 2>&1; then
            reasons+=("missing clFFT headers: clFFT.h")
        fi
        if ! { ldconfig -p 2>/dev/null | grep -q 'libclFFT\.so' || find /usr/lib /usr/lib64 /usr/local/lib "$HOME" -name 'libclFFT.so*' -print -quit 2>/dev/null | grep -q .; }; then
            reasons+=("missing clFFT library: libclFFT.so")
        fi
        if ! { ldconfig -p 2>/dev/null | grep -q 'libOpenCL\.so' || find /usr/lib /usr/lib64 /usr/local/lib "$HOME" -name 'libOpenCL.so*' -print -quit 2>/dev/null | grep -q .; }; then
            reasons+=("missing OpenCL library: libOpenCL.so")
        fi
        if [[ ${#reasons[@]} -eq 0 ]]; then
            reasons+=("missing native adda_ocl at $ADDA_OCL; run: cd $ADDA_ROOT/src && make ocl")
        else
            reasons+=("native adda_ocl is not built at $ADDA_OCL")
        fi
        local IFS='; '
        echo "${reasons[*]}"
        return 0
    fi
    if ! ldd "$ADDA_OCL" >/tmp/adda_ocl_ldd.$$ 2>&1; then
        tr '\n' ' ' </tmp/adda_ocl_ldd.$$
        rm -f /tmp/adda_ocl_ldd.$$
        return 0
    fi
    if grep -q 'not found' /tmp/adda_ocl_ldd.$$; then
        tr '\n' ' ' </tmp/adda_ocl_ldd.$$
        rm -f /tmp/adda_ocl_ldd.$$
        return 0
    fi
    rm -f /tmp/adda_ocl_ldd.$$
    if command -v clinfo >/dev/null 2>&1 && clinfo 2>/dev/null | grep -q 'Number of platforms[[:space:]]*0'; then
        echo "OpenCL ICD loader is present but clinfo reports zero platforms"
        return 0
    fi
    return 1
}

require_gpu_free() {
    bem_require_gpu_free "$GPU" "$ALLOW_COMPUTE_SHARE"
}

run_case() {
    local case_name=$1 shape_label=$2 ka=$3
    shift 3
    local dir="$OUT/${case_name}${OUT_SUFFIX}"
    local log="$dir/run.log"
    mkdir -p "$dir"
    local cmd=(
        "$ADDA_OCL"
        -gpu "$GPU"
        -dir "$dir"
        "$@"
        -m 1.3116 0
        -dpl "$DPL"
        -eps "$EPS"
        -orient 0 0 0
        -ntheta "$NTHETA"
        -scat_matr muel
        -sym no
    )
    if [[ " $* " != *" -shape read "* ]]; then
        cmd+=(-eq_rad "$ka")
    fi
    printf '+ %q ' "${cmd[@]}" > "$log"
    printf '\n' >> "$log"
    local start end rc
    start=$(date +%s.%N)
    set +e
    timeout "$TIMEOUT_S" "${cmd[@]}" >>"$log" 2>&1
    rc=$?
    set -e
    end=$(date +%s.%N)
    local elapsed
    elapsed=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.6f", b-a}')
    if [[ $rc -eq 0 ]]; then
        emit_row "$case_name" "$shape_label" "$ka" "$DPL" "$NTHETA" measured "$elapsed" adda_ocl "$dir" "$log" "single orientation"
    elif [[ $rc -eq 124 ]]; then
        emit_row "$case_name" "$shape_label" "$ka" "$DPL" "$NTHETA" timeout "$elapsed" adda_ocl "$dir" "$log" "timeout ${TIMEOUT_S}s"
    else
        emit_row "$case_name" "$shape_label" "$ka" "$DPL" "$NTHETA" failed "$elapsed" adda_ocl "$dir" "$log" "exit code $rc"
    fi
}

main() {
    local unavailable
    if unavailable=$(detect_unavailable); then
        emit_row "sphere_ka5" "sphere" 5 "$DPL" "$NTHETA" unavailable "" adda_ocl "$OUT/sphere_ka5" "" "$unavailable"
        emit_row "hex_ka5" "hex_prism" 5 "$DPL" "$NTHETA" unavailable "" adda_ocl "$OUT/hex_ka5" "" "$unavailable"
        emit_row "dust_ka5p709" "dust" 5.709 "$DPL" "$NTHETA" unavailable "" adda_ocl "$OUT/dust_ka5p709" "" "$unavailable"
        echo "ADDA_OCL unavailable: $unavailable"
        echo "Wrote $CSV"
        return 0
    fi
    require_gpu_free

    local sphere_sizes hex_sizes
    if [[ "$CASES" == "sweep" ]]; then
        sphere_sizes=(${SPHERE_KA:-2 5 10})
        hex_sizes=(${HEX_KA:-2 5 10})
    else
        sphere_sizes=(${SPHERE_KA:-5})
        hex_sizes=(${HEX_KA:-5})
    fi
    for ka in "${sphere_sizes[@]}"; do
        run_case "sphere_ka${ka}" "sphere" "$ka" -shape sphere
    done
    for ka in "${hex_sizes[@]}"; do
        run_case "hex_ka${ka}" "hex_prism" "$ka" -shape prism 6 1.5
    done

    local dust_shape="$ROOT/runs/adda_greek_dpl25/greek_ka5p71_dpl25.shape"
    if [[ -f "$dust_shape" ]]; then
        run_case "dust_ka5p709" "dust" 5.709 -shape read "$dust_shape"
    else
        emit_row "dust_ka5p709" "dust" 5.709 "$DPL" "$NTHETA" unavailable "" adda_ocl "$OUT/dust_ka5p709" "" "missing $dust_shape"
    fi
    echo "Wrote $CSV"
}

main "$@"
