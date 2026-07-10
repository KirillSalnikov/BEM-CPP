#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/resume_dust_adda_db_adaptive.sh [options]

Starts or resumes an adaptive BEM orientation average for the Greek dust mesh
and compares the accepted BEM result against an existing ADDA database file.

Options:
  --host HOST             SSH host, default kirill_epyc@172.16.0.117
  --remote-dir DIR        Remote BEM-CUDA dir, default /home/kirill_epyc/BEM-CUDA
  --ka VALUE              ADDA/BEM size parameter from the database, default 10.86
  --gpus LIST             CUDA_VISIBLE_DEVICES workers for separate chunks, default 0,1,3
  --run-name NAME         Output directory name under runs/dust_ka10_20_bem_vs_adda_db_20260710
  --mesh FILE             OBJ mesh relative to repo, default is the best ka=30 legacy branch
  --quad N                BEM quadrature order, default 7
  --fmm-digits N          FMM digits, default 5
  --gmres-tol TOL         GMRES tolerance request, default 1e-3
  --gmres-restart N       GMRES restart, default 200
  --max-leaf N            FMM max leaf, default 128
  --system NAME           Integral system, default muller2-balanced
  --dry-run               Print commands, do not execute remote launch
  -h, --help              Show this help
EOF
}

HOST="kirill_epyc@172.16.0.117"
REMOTE_DIR="/home/kirill_epyc/BEM-CUDA"
KA="10.86"
GPUS="0,1,3"
BASE_RUN="runs/dust_ka10_20_bem_vs_adda_db_20260710"
RUN_NAME=""
MESH="runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f7000_a45.obj"
QUAD="7"
FMM_DIGITS="5"
GMRES_TOL="1e-3"
GMRES_RESTART="200"
MAX_LEAF="128"
SYSTEM_NAME="muller2-balanced"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --ka) KA="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --mesh) MESH="$2"; shift 2 ;;
    --quad) QUAD="$2"; shift 2 ;;
    --fmm-digits) FMM_DIGITS="$2"; shift 2 ;;
    --gmres-tol) GMRES_TOL="$2"; shift 2 ;;
    --gmres-restart) GMRES_RESTART="$2"; shift 2 ;;
    --max-leaf) MAX_LEAF="$2"; shift 2 ;;
    --system) SYSTEM_NAME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_NAME" ]]; then
  mesh_tag="$(basename "$MESH" .obj | tr -c '[:alnum:]' '_' | sed 's/_*$//')"
  RUN_NAME="ka${KA//./p}_adaptive_nested_J2_J4_alpha256_${mesh_tag}_q${QUAD}_d${FMM_DIGITS}"
fi

ADDA_LOCAL="/home/user/cluster/BEM-CPP/greek/ADDA_for_PO_comparison/refr_1_6__0/A_x=${KA}_refr_1_6__0.dat"
if [[ ! -s "$ADDA_LOCAL" ]]; then
  echo "missing local ADDA reference: $ADDA_LOCAL" >&2
  exit 1
fi
LOCAL_TEMPLATE="$BASE_RUN/avg_params_alpha8_beta3_gamma3.dat"
LOCAL_NESTED="$BASE_RUN/nested_bg_J2_J4_alpha8"
if [[ ! -s "$LOCAL_TEMPLATE" ]]; then
  echo "missing local orientation template: $LOCAL_TEMPLATE" >&2
  exit 1
fi
if [[ ! -s "$LOCAL_NESTED/nested_bg_manifest.json" ]]; then
  echo "missing local nested manifest: $LOCAL_NESTED/nested_bg_manifest.json" >&2
  exit 1
fi
if [[ ! -s "$MESH" ]]; then
  echo "missing local BEM mesh: $MESH" >&2
  exit 1
fi

REMOTE_BASE="$BASE_RUN"
REMOTE_RUN="$REMOTE_BASE/$RUN_NAME"
REMOTE_ADDA_DIR="$REMOTE_BASE/adda_ref_refr_1_6__0"
REMOTE_ADDA="$REMOTE_ADDA_DIR/$(basename "$ADDA_LOCAL")"
REMOTE_MANIFEST="$REMOTE_BASE/nested_bg_J2_J4_alpha8/nested_bg_manifest.json"

run() {
  echo "+ $*" >&2
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

run ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "cd '$REMOTE_DIR' && mkdir -p '$REMOTE_ADDA_DIR' '$REMOTE_BASE'"
run scp scripts/adaptive_nested_bg_orient_queue.py scripts/generate_nested_bg_manifest.py scripts/recombine_orient_parts.py scripts/summarize_bem_adda_m11.py "$HOST:$REMOTE_DIR/scripts/"
run scp run_orient_queue.py "$HOST:$REMOTE_DIR/run_orient_queue.py"
run scp "$LOCAL_TEMPLATE" "$HOST:$REMOTE_DIR/$REMOTE_BASE/avg_params_alpha8_beta3_gamma3.dat"
run scp -r "$LOCAL_NESTED" "$HOST:$REMOTE_DIR/$REMOTE_BASE/"
run ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "cd '$REMOTE_DIR' && mkdir -p '$(dirname "$MESH")'"
run scp "$MESH" "$HOST:$REMOTE_DIR/$MESH"
run scp "$ADDA_LOCAL" "$HOST:$REMOTE_DIR/$REMOTE_ADDA"

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
cd '$REMOTE_DIR'
if [[ ! -s '$REMOTE_MANIFEST' ]]; then
  python3 scripts/generate_nested_bg_manifest.py \\
    --template '$REMOTE_BASE/avg_params_alpha8_beta3_gamma3.dat' \\
    --out-dir '$REMOTE_BASE/nested_bg_J2_J4_alpha8' \\
    --j-alpha 8 --jmin-beta 2 --jmax-beta 4 --jmin-gamma 2 --jmax-gamma 4
fi
R='$REMOTE_RUN'
mkdir -p "\$R"
if [[ -f "\$R/adaptive.pid" ]] && kill -0 "\$(cat "\$R/adaptive.pid")" 2>/dev/null; then
  echo "adaptive run already active: pid=\$(cat "\$R/adaptive.pid")"
else
  nohup env BEM_ORIENT_KEEP_CHUNK_SIZE=1 \\
    BEM_ORIENT_PROGRESS=1 \\
    BEM_GMRES_VERBOSE=1 \\
    BEM_ALLOW_LOOSE_OBJ_GMRES=1 \\
    python3 scripts/adaptive_nested_bg_orient_queue.py \\
      --nested-manifest '$REMOTE_MANIFEST' \\
      --out-dir "\$R" \\
      --gpus '$GPUS' \\
      --chunk-size 8 \\
      --tail-chunk-size 4 \\
      --tail-threshold-chunks 2 \\
      --chunk-order spread \\
      --alpha-avg 256 \\
      --orient-warm-start recycle \\
      --orient-warm-history 4 \\
      --tol 0.025 \\
      --max-tol 0.07 \\
      --scale-tol 0.025 \\
      --component-floor 1e-4 \\
      --min-levels 2 \\
      -- \\
      --ka '$KA' \\
      --ri 1.6 0 \\
      --shape obj \\
      --obj '$MESH' \\
      --ntheta 181 \\
      --quad '$QUAD' \\
      --solver fmm \\
      --system '$SYSTEM_NAME' \\
      --fmm-digits '$FMM_DIGITS' \\
      --gmres-tol '$GMRES_TOL' \\
      --gmres-restart '$GMRES_RESTART' \\
      --max-leaf '$MAX_LEAF' \\
      --no-prec \\
      --accurate > "\$R/adaptive.nohup" 2>&1 &
  echo \$! > "\$R/adaptive.pid"
  echo "started adaptive pid=\$(cat "\$R/adaptive.pid")"
fi

cat > "\$R/watch_compare.sh" <<'WATCH'
#!/usr/bin/env bash
set -euo pipefail
cd __REMOTE_DIR__
R='__REMOTE_RUN__'
ADDA='__REMOTE_ADDA__'
while true; do
  if [[ -s "\$R/adaptive_nested_bg_manifest.json" ]]; then
    accepted=\$(python3 - <<PY
import json
from pathlib import Path
p=Path("\$R/adaptive_nested_bg_manifest.json")
try:
    data=json.load(p.open())
except Exception:
    raise SystemExit(1)
print(data.get("accepted") or "")
PY
)
    status=\$(python3 - <<PY
import json
from pathlib import Path
p=Path("\$R/adaptive_nested_bg_manifest.json")
try:
    data=json.load(p.open())
except Exception:
    raise SystemExit(1)
print(data.get("status") or "")
PY
)
    if [[ -n "\$accepted" && "\$status" == "complete" && -s "\$accepted" ]]; then
      python3 scripts/summarize_bem_adda_m11.py --bem "\$accepted" --adda "\$ADDA" > "\$R/m11_vs_adda_db_accepted.txt"
      exit 0
    fi
  fi
  sleep 60
done
WATCH
sed -i "s#__REMOTE_DIR__#$REMOTE_DIR#g; s#__REMOTE_RUN__#$REMOTE_RUN#g; s#__REMOTE_ADDA__#$REMOTE_ADDA#g" "\$R/watch_compare.sh"
chmod +x "\$R/watch_compare.sh"
if [[ -f "\$R/compare_accepted_watch.pid" ]] && kill -0 "\$(cat "\$R/compare_accepted_watch.pid")" 2>/dev/null; then
  echo "compare watcher already active: pid=\$(cat "\$R/compare_accepted_watch.pid")"
else
  nohup "\$R/watch_compare.sh" > "\$R/watch_compare.nohup" 2>&1 &
  echo \$! > "\$R/compare_accepted_watch.pid"
  echo "started compare watcher pid=\$(cat "\$R/compare_accepted_watch.pid")"
fi
echo "run dir: \$R"
EOF
)

run ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "$REMOTE_CMD"
