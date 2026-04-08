#!/bin/bash
# Run BEM-CUDA on Greek statue for m=1.6+0.002i
# Usage: ./run_greek.sh SERVER_INDEX   (0, 1, or 2)
# Distributes 27 ka values across 3 servers (interleaved)

BEM=~/BEM-CUDA-merged/bin/bem_cuda
OBJ=~/BEM-CUDA-merged/model_repaired.obj
OUTDIR=~/BEM-CUDA-merged/greek/results_m1.6_0.002
RI_RE=1.6
RI_IM=0.002
NTHETA=1801

SERVER=${1:?Usage: $0 SERVER_INDEX}

mkdir -p "$OUTDIR"

# ka values and their subdiv levels (pre-computed)
# subdiv=1 for ka<=7, subdiv=2 for ka<=15, subdiv=3 for ka>15
KA_LIST="5.23:1 5.71:1 6.26:1 6.88:1 7.56:2 8.25:2 9.01:2 9.90:2 10.86:2 11.89:2 13.06:2 14.30:2 15.68:3 17.19:3 18.94:3 20.76:3 22.83:3 25.09:3 27.50:3 30.25:3 33.28:3 36.58:3 40.22:3 44.21:3 48.61:3 53.47:3 58.81:3"

IDX=0
for ENTRY in $KA_LIST; do
    ka=${ENTRY%%:*}
    SUBDIV=${ENTRY##*:}

    if [ $((IDX % 3)) -ne "$SERVER" ]; then
        IDX=$((IDX + 1))
        continue
    fi
    IDX=$((IDX + 1))

    OUT="$OUTDIR/ka_${ka}.json"
    LOG="$OUTDIR/ka_${ka}.log"

    echo "================================================================"
    echo "[$(date '+%H:%M:%S')] ka=$ka  subdiv=$SUBDIV  server=$SERVER"
    echo "================================================================"

    # Skip if result already exists
    if [ -f "$OUT" ]; then
        echo "[$(date '+%H:%M:%S')] ka=$ka SKIPPED (result exists)"
        continue
    fi

    # Use pfft for subdiv>=2 (FMM runs out of memory on deep trees)
    if [ "$SUBDIV" -ge 2 ]; then
        SOLVER="pfft"
    else
        SOLVER="auto"
    fi

    $BEM --ka $ka --ri $RI_RE $RI_IM --obj "$OBJ" --subdiv $SUBDIV \
        --solver $SOLVER --orient-auto --ntheta $NTHETA \
        --out "$OUT" 2>&1 | tee "$LOG" || true

    echo "[$(date '+%H:%M:%S')] ka=$ka DONE (exit=$?)"
    echo ""
done

echo "=== SERVER $SERVER: ALL DONE ==="
