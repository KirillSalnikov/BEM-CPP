#!/bin/bash
# BEM-CUDA Extended 2-Day Benchmark Suite
# Hex prism D/L=0.7, m=1.3116, RTX 3080 Ti (12 GB)
# Total estimated time: ~46 hours
#
# Measured base timings (single orientation, restart=200):
#   ka=5  ref=3  N=2304:  218s, 748 iters  (0.29 s/iter)
#   ka=10 ref=3  N=2304:  115s, 392 iters  (0.29 s/iter)
#   ka=10 ref=4  N=9216:  ~16000s, ~400 iters (~40 s/iter)
#
# Structure:
#   Part A: Preconditioner comparison (ka=10, ref=3)         ~0.5h
#   Part B: GMRES restart sensitivity                        ~2h
#   Part C: High ka stress tests (ref=3)                     ~2h
#   Part D: Orientation averaging 256-orient (ref=3)         ~18h
#   Part E: Large-scale ref=4 single-orient                  ~14h
#   Part F: ref=4 + 4 orientations                           ~9h
#   Part G: Dense vs iterative cross-validation              ~0.5h

set -e
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

BIN=bin/bem_cuda
AR=1.4286  # H/D = 1/0.7
OUTDIR=bench_results
mkdir -p $OUTDIR

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== BEM-CUDA Extended 2-Day Benchmark Suite ==="
log "GPU: $(/usr/lib/wsl/lib/nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
log ""

# ============================================================
# Part A: Preconditioner comparison (ka=10, ref=3, single)
# Compare all 4 preconditioners at moderate problem size
# Expected: ~0.5h
# ============================================================
log "=== PART A: Preconditioner comparison (ka=10, ref=3, single) ==="

for PREC in none blockj ilu0 diag; do
    log ">>> A: prec=$PREC"
    PREC_FLAG=""
    [ "$PREC" != "none" ] && PREC_FLAG="--prec $PREC"
    $BIN --ka 10 --ref 3 --shape hex --ar $AR \
         --spfft $PREC_FLAG --gmres-restart 200 \
         --single --out $OUTDIR/A_ka10_r3_${PREC}.json 2>&1 | tee $OUTDIR/A_${PREC}.log || log "A $PREC FAILED"
    log "A $PREC done"
done

# ============================================================
# Part B: GMRES restart sensitivity (ka=10, ref=3, 4 orient)
# How restart parameter affects iteration count and wall time
# Expected: ~2h
# ============================================================
log "=== PART B: GMRES restart sensitivity ==="

for RESTART in 30 50 100 150 200 300; do
    log ">>> B: restart=$RESTART blockj"
    $BIN --ka 10 --ref 3 --shape hex --ar $AR \
         --spfft --prec blockj --gmres-restart $RESTART \
         --orient 2 2 1 --out $OUTDIR/B_r${RESTART}_blockj_o4.json 2>&1 | tee $OUTDIR/B_r${RESTART}.log
    log "B restart=$RESTART done"
done

log ">>> B: restart=100 no prec (baseline)"
$BIN --ka 10 --ref 3 --shape hex --ar $AR \
     --spfft --gmres-restart 100 \
     --orient 2 2 1 --out $OUTDIR/B_r100_noprec_o4.json 2>&1 | tee $OUTDIR/B_r100_noprec.log
log "B baseline done"

# ============================================================
# Part C: High ka stress tests (ref=3, single)
# Convergence at high size parameter, blockj vs no prec
# Expected: ~2h
# ============================================================
log "=== PART C: High ka stress tests (ref=3, single) ==="

for KA in 15 20 25 30; do
    log ">>> C: ka=$KA blockj"
    $BIN --ka $KA --ref 3 --shape hex --ar $AR \
         --spfft --prec blockj --gmres-restart 200 \
         --single --out $OUTDIR/C_ka${KA}_blockj.json 2>&1 | tee $OUTDIR/C_ka${KA}_blockj.log
    log "C ka=$KA blockj done"

    log ">>> C: ka=$KA no prec"
    $BIN --ka $KA --ref 3 --shape hex --ar $AR \
         --spfft --gmres-restart 200 \
         --single --out $OUTDIR/C_ka${KA}_noprec.json 2>&1 | tee $OUTDIR/C_ka${KA}_noprec.log
    log "C ka=$KA noprec done"
done

# ============================================================
# Part D: Production orientation averaging (ref=3, 256 orient)
# Full orientation-averaged Mueller matrix for comparison
# Expected: ka=10 ~8h, ka=5 ~16h → ~18h total
#   (ka=5: ~220s/orient * 256 * (reuse factor ~0.3) ~ 16h)
# ============================================================
log "=== PART D: Orientation averaging (ref=3, 256 orientations) ==="

log ">>> D1: ka=10 ref=3 blockj orient=16x16x1"
$BIN --ka 10 --ref 3 --shape hex --ar $AR \
     --spfft --prec blockj --gmres-restart 200 \
     --orient 16 16 1 --out $OUTDIR/D1_ka10_r3_o256.json 2>&1 | tee $OUTDIR/D1.log
log "D1 done"

log ">>> D2: ka=5 ref=3 blockj orient=16x16x1"
$BIN --ka 5 --ref 3 --shape hex --ar $AR \
     --spfft --prec blockj --gmres-restart 200 \
     --orient 16 16 1 --out $OUTDIR/D2_ka5_r3_o256.json 2>&1 | tee $OUTDIR/D2.log
log "D2 done"

log ">>> D3: ka=15 ref=3 blockj orient=8x8x1 (64 orientations)"
$BIN --ka 15 --ref 3 --shape hex --ar $AR \
     --spfft --prec blockj --gmres-restart 200 \
     --orient 8 8 1 --out $OUTDIR/D3_ka15_r3_o64.json 2>&1 | tee $OUTDIR/D3.log
log "D3 done"

# ============================================================
# Part E: Large-scale ref=4 single orientation
# Heaviest single-orientation tests (~4.5h each)
# Expected: ~14h total (3 tests)
# ============================================================
log "=== PART E: Large-scale ref=4 single orientation ==="

log ">>> E1: ka=5 ref=4 blockj (~4h)"
$BIN --ka 5 --ref 4 --shape hex --ar $AR \
     --spfft --prec blockj --gmres-restart 200 \
     --single --out $OUTDIR/E1_ka5_r4_blockj.json 2>&1 | tee $OUTDIR/E1.log
log "E1 done"

log ">>> E2: ka=10 ref=4 blockj (~4.5h)"
$BIN --ka 10 --ref 4 --shape hex --ar $AR \
     --spfft --prec blockj --gmres-restart 200 \
     --single --out $OUTDIR/E2_ka10_r4_blockj.json 2>&1 | tee $OUTDIR/E2.log
log "E2 done"

log ">>> E3: ka=10 ref=4 no prec (~5h)"
$BIN --ka 10 --ref 4 --shape hex --ar $AR \
     --spfft --gmres-restart 200 \
     --single --out $OUTDIR/E3_ka10_r4_noprec.json 2>&1 | tee $OUTDIR/E3.log
log "E3 done"

# ============================================================
# Part F: ref=4 orientation averaging (2 orientations only)
# Verifies orient-reuse initial guess benefit at ref=4
# Expected: ~9h
# ============================================================
log "=== PART F: ref=4 with 2 orientations ==="

log ">>> F1: ka=10 ref=4 blockj orient=2x1x1 (2 orientations, ~9h)"
$BIN --ka 10 --ref 4 --shape hex --ar $AR \
     --spfft --prec blockj --gmres-restart 200 \
     --orient 2 1 1 --out $OUTDIR/F1_ka10_r4_blockj_o2.json 2>&1 | tee $OUTDIR/F1.log
log "F1 done"

# ============================================================
# Part G: Dense vs iterative cross-validation (ref=3, single)
# Verify Mueller matrix agreement
# Expected: ~30 min
# ============================================================
log "=== PART G: Cross-validation ==="

for KA in 3 5 10 15; do
    log ">>> G: ka=$KA Dense LU"
    $BIN --ka $KA --ref 3 --shape hex --ar $AR \
         --single --out $OUTDIR/G_ka${KA}_dense.json 2>&1 | tee $OUTDIR/G_ka${KA}_dense.log

    log ">>> G: ka=$KA SurfPFFT+blockj"
    $BIN --ka $KA --ref 3 --shape hex --ar $AR \
         --spfft --prec blockj --gmres-restart 200 \
         --single --out $OUTDIR/G_ka${KA}_spfft.json 2>&1 | tee $OUTDIR/G_ka${KA}_spfft.log
    log "G ka=$KA done"
done

log "=== All benchmarks complete ==="
log "Results in $OUTDIR/"
log ""

# Post-processing
python3 << 'PYEOF'
import json, os, glob
outdir = "bench_results"

print("\n=== TIMING SUMMARY ===")
for f in sorted(glob.glob(os.path.join(outdir, "*.json"))):
    try:
        d = json.load(open(f))
        name = os.path.basename(f).replace(".json","")
        ts = d.get("time_solve", 0)
        tt = d.get("time_total", 0)
        print(f"  {name:45s} solve={ts:8.1f}s  total={tt:8.1f}s")
    except: pass

print("\n=== CROSS-VALIDATION (Part G) ===")
for ka in [3, 5, 10, 15]:
    df = os.path.join(outdir, f"G_ka{ka}_dense.json")
    sf = os.path.join(outdir, f"G_ka{ka}_spfft.json")
    if os.path.exists(df) and os.path.exists(sf):
        d = json.load(open(df))
        s = json.load(open(sf))
        m11d = d["mueller"][0][0]
        m11s = s["mueller"][0][0]
        m11_max = max(abs(v) for v in m11d)
        max_e = max(abs(m11s[k]-m11d[k]) for k in range(len(m11d)))
        print(f"  ka={ka:2d}: M11 abs_err={max_e:.3e}  rel={max_e/m11_max:.3e}")
PYEOF
