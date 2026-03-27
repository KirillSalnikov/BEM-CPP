#!/bin/bash
# BEM-CUDA 2-day benchmark suite
# Hex prism D/L=0.7, m=1.3116, RTX 3080 Ti
# Total estimated time: ~48 hours

set -e
BIN=bin/bem_cuda
AR=1.4286  # H/D = 1/0.7

echo "=== BEM-CUDA 2-Day Benchmark Suite ==="
echo "Start: $(date)"
echo ""

# -----------------------------------------------------------
# Test 1: ka=10, ref=3, 256 orientations (~5h)
# Baseline: validates orient-averaged Mueller at moderate size
# -----------------------------------------------------------
echo ">>> Test 1: ka=10 ref=3 N~2304 orient=16x16x1 blockj"
$BIN --ka 10 --ref 3 --shape hex --ar $AR \
     --spfft --prec blockj --gmres-restart 200 \
     --orient 16 16 1 --out bench_ka10_ref3_o256.json
echo "Test 1 done: $(date)"
echo ""

# -----------------------------------------------------------
# Test 2: ka=10, ref=4, 64 orientations (~8h)
# Large N: tests Block-Jacobi scaling at ref=4
# -----------------------------------------------------------
echo ">>> Test 2: ka=10 ref=4 N~9216 orient=8x8x1 blockj"
$BIN --ka 10 --ref 4 --shape hex --ar $AR \
     --spfft --prec blockj --gmres-restart 200 \
     --orient 8 8 1 --out bench_ka10_ref4_o64.json
echo "Test 2 done: $(date)"
echo ""

# -----------------------------------------------------------
# Test 3: ka=20, ref=3, 256 orientations (~12h)
# High ka: tests convergence at large size parameter
# -----------------------------------------------------------
echo ">>> Test 3: ka=20 ref=3 N~2304 orient=16x16x1 blockj"
$BIN --ka 20 --ref 3 --shape hex --ar $AR \
     --spfft --prec blockj --gmres-restart 200 \
     --orient 16 16 1 --out bench_ka20_ref3_o256.json
echo "Test 3 done: $(date)"
echo ""

# -----------------------------------------------------------
# Test 4: ka=20, ref=4, 64 orientations (~24h)
# Full-scale: large ka + large N, real production case
# -----------------------------------------------------------
echo ">>> Test 4: ka=20 ref=4 N~9216 orient=8x8x1 blockj"
$BIN --ka 20 --ref 4 --shape hex --ar $AR \
     --spfft --prec blockj --gmres-restart 200 \
     --orient 8 8 1 --out bench_ka20_ref4_o64.json
echo "Test 4 done: $(date)"
echo ""

echo "=== All benchmarks complete: $(date) ==="
