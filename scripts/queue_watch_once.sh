#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

out="${OUT:-runs/production_matrix_15}"
status_json="${QUEUE_WATCH_STATUS_JSON:-$out/status.json}"
active_age_s="${QUEUE_STATUS_ACTIVE_AGE_S:-30}"
stall_wall_s="${QUEUE_STATUS_STALL_WALL_S:-120}"
strict_incomplete="${QUEUE_WATCH_STRICT_INCOMPLETE:-0}"
strict_accuracy="${QUEUE_WATCH_STRICT_ACCURACY:-0}"
strict_current_accuracy="${QUEUE_WATCH_STRICT_CURRENT_ACCURACY:-0}"
accuracy_csv="${QUEUE_WATCH_ACCURACY_CSV:-}"
min_usable_gpus="${QUEUE_WATCH_MIN_USABLE_GPUS:-}"

usage() {
  cat <<EOF
Usage: $0 [--strict-incomplete] [--strict-accuracy] [--strict-current-accuracy]

Writes a machine-readable queue status JSON and exits nonzero only for real
watch conditions by default:
  20  solver log contains FAIL
  21  active monitor has no new GPU samples after QUEUE_STATUS_STALL_WALL_S
  25  queue is not running while planned results are still missing/stale
  26  --strict-accuracy: planned point is inaccurate or lacks accuracy evidence
  27  --strict-current-accuracy: accuracy evidence is missing or uses legacy metadata
  29  fewer than QUEUE_WATCH_MIN_USABLE_GPUS local GPUs are currently usable
  30  accuracy summary line disagrees with parsed per-case status rows

Environment:
  OUT                         queue directory, default ${out}
  QUEUE_WATCH_STATUS_JSON     output JSON path, default ${status_json}
  QUEUE_STATUS_ACTIVE_AGE_S   active monitor age, default ${active_age_s}
  QUEUE_STATUS_STALL_WALL_S   stall interval, default ${stall_wall_s}
  QUEUE_WATCH_STRICT_INCOMPLETE=1 also fail on missing/stale result JSONs
  QUEUE_WATCH_STRICT_ACCURACY=1 also fail on inaccurate/missing accuracy
  QUEUE_WATCH_STRICT_CURRENT_ACCURACY=1 also fail on missing/legacy accuracy evidence
  QUEUE_WATCH_ACCURACY_CSV    optional precomputed audit CSV
  QUEUE_WATCH_MIN_USABLE_GPUS require at least this many usable local GPUs
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict-incomplete)
      strict_incomplete=1
      ;;
    --strict-accuracy)
      strict_accuracy=1
      ;;
    --strict-current-accuracy)
      strict_current_accuracy=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

mkdir -p "$(dirname "$status_json")"

args=(
  python3 scripts/queue_status_json.py
  --out "$out"
  --active-age-s "$active_age_s"
  --stall-wall-s "$stall_wall_s"
  --fail-on-failed
  --fail-on-stalled
  --fail-on-stopped-incomplete
  --fail-on-accuracy-summary-mismatch
)
if [[ "$strict_incomplete" == "1" ]]; then
  args+=(--fail-on-incomplete)
fi
if [[ "$strict_accuracy" == "1" ]]; then
  args+=(--fail-on-inaccurate)
fi
if [[ "$strict_current_accuracy" == "1" ]]; then
  args+=(--fail-on-accuracy-legacy)
fi
if [[ -n "$accuracy_csv" ]]; then
  args+=(--accuracy-csv "$accuracy_csv")
fi
if [[ -n "$min_usable_gpus" ]]; then
  args+=(--min-usable-gpus "$min_usable_gpus")
fi

set +e
"${args[@]}" > "$status_json"
rc="$?"
set -e

python3 - "$status_json" "$rc" <<'PY'
import json
import sys

path, rc = sys.argv[1], int(sys.argv[2])
data = json.load(open(path))
counts = data.get("counts", {})
queue = data.get("queue", {})
active = data.get("active_monitors", [])
monitors = data.get("monitors", [])
stalled = data.get("stalled_monitors", [])
failed = data.get("failed_monitors", [])
accuracy = data.get("accuracy", {})
accuracy_counts = accuracy.get("counts", {}) if isinstance(accuracy, dict) else {}
gpu_inventory = data.get("gpu_inventory", {})
gpu_counts = gpu_inventory.get("counts", {}) if isinstance(gpu_inventory, dict) else {}
missing_cases = data.get("missing_cases", [])
stale_cases = data.get("stale_cases", [])
stopped_incomplete = data.get("queue_stopped_incomplete")
total = int(data.get("total") or 0)
current = int(counts.get("current", 0) or 0)
stale = int(counts.get("stale", 0) or 0)
missing = int(counts.get("missing", 0) or 0)
remaining = max(0, total - current)
percent = (100.0 * current / total) if total else 0.0
result_avg_s = data.get("completed_result_duration_mean_s")
result_avg_count = int(data.get("completed_result_duration_count") or 0)

def fmt(value, suffix="", digits=1):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return ("%%.%df%%s" % digits) % (value, suffix)
    return "%s%s" % (value, suffix)

def fmt_duration(seconds):
    if seconds is None:
        return "NA"
    try:
        seconds = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "NA"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return "%dh%02dm" % (hours, minutes)
    if minutes:
        return "%dm%02ds" % (minutes, secs)
    return "%ds" % secs

def active_eta(item):
    duration = item.get("duration_s")
    if duration is None:
        return None
    # We only know case-level progress from the queue plan, not internal solver progress.
    # Use the current case runtime as a conservative unit-time estimate.
    return float(duration)

def completed_case_durations():
    values = []
    for item in monitors:
        if item.get("state") != "finished":
            continue
        if str(item.get("terminal", "")).startswith("FAIL "):
            continue
        duration = item.get("duration_s")
        try:
            values.append(float(duration))
        except (TypeError, ValueError):
            pass
    return values

print("status_json=%s" % path)
print("watch_rc=%d" % rc)
print("queue_pid=%s running=%s" % (queue.get("pid"), queue.get("running")))
print("queue_stopped_incomplete=%s" % stopped_incomplete)
print("counts current={current} stale={stale} missing={missing}".format(
    current=current,
    stale=stale,
    missing=missing,
))
print("matrix_progress done={done} total={total} remaining={remaining} percent={percent:.1f}".format(
    done=current,
    total=total,
    remaining=remaining,
    percent=percent,
))
if accuracy_counts:
    print("accuracy_gate accurate={accurate} accurate_legacy={legacy} inaccurate={inaccurate} missing={missing} total={total} rc={rc}".format(
        accurate=int(accuracy_counts.get("accurate", 0) or 0),
        legacy=int(accuracy_counts.get("accurate_legacy", 0) or 0),
        inaccurate=int(accuracy_counts.get("inaccurate", 0) or 0),
        missing=int(accuracy_counts.get("missing", 0) or 0),
        total=int(accuracy_counts.get("total", 0) or 0),
        rc=accuracy.get("returncode"),
    ))
else:
    print("accuracy_gate unavailable")
if gpu_inventory:
    print("gpu_gate available={available} usable={usable} busy={busy} unhealthy={unhealthy} unparseable={unparseable} total={total}".format(
        available=bool(gpu_inventory.get("available")),
        usable=int(gpu_counts.get("usable", 0) or 0),
        busy=int(gpu_counts.get("busy", 0) or 0),
        unhealthy=int(gpu_counts.get("unhealthy", 0) or 0),
        unparseable=int(gpu_counts.get("unparseable", 0) or 0),
        total=int(gpu_counts.get("total", 0) or 0),
    ))
else:
    print("gpu_gate unavailable")
if missing_cases:
    preview = ",".join(str(name) for name in missing_cases[:5])
    suffix = "" if len(missing_cases) <= 5 else ",..."
    print("next_missing count=%d cases=%s%s" % (len(missing_cases), preview, suffix))
else:
    print("next_missing count=0 cases=none")
if stale_cases:
    preview = ",".join(str(name) for name in stale_cases[:5])
    suffix = "" if len(stale_cases) <= 5 else ",..."
    print("stale_cases count=%d cases=%s%s" % (len(stale_cases), preview, suffix))
if active:
    completed_durations = completed_case_durations()
    completed_avg_s = (sum(completed_durations) / len(completed_durations)) if completed_durations else None
    for item in active:
        case_eta_s = active_eta(item)
        if result_avg_s is not None:
            unit_eta_s = max(float(result_avg_s), case_eta_s or 0.0)
            queue_eta_s = unit_eta_s * remaining
            eta_source = "result_timing_avg_clamped"
        elif completed_avg_s is not None:
            unit_eta_s = max(completed_avg_s, case_eta_s or 0.0)
            queue_eta_s = unit_eta_s * remaining
            eta_source = "completed_avg_clamped"
        else:
            queue_eta_s = case_eta_s * remaining if case_eta_s is not None else None
            eta_source = "active_duration"
        print("active case={case} gpu={gpu} samples={samples} delta={delta} progress={progress} age_s={age} duration_s={duration} case_eta~={case_eta} queue_eta~={queue_eta} eta_source={eta_source} result_avg_s={result_avg} result_avg_n={result_avg_n} completed_avg_s={completed_avg} power_mean_w={pmean} power_max_w={pmax} temp_max_c={temp} mem_max_mib={mem}".format(
            case=item.get("case"),
            gpu=fmt(item.get("gpu")),
            samples=item.get("samples"),
            delta=item.get("sample_delta"),
            progress=item.get("progress_state"),
            age=item.get("age_s"),
            duration=fmt(item.get("duration_s"), digits=0),
            case_eta=fmt_duration(case_eta_s),
            queue_eta=fmt_duration(queue_eta_s),
            eta_source=eta_source,
            result_avg=fmt(result_avg_s, digits=0),
            result_avg_n=result_avg_count,
            completed_avg=fmt(completed_avg_s, digits=0),
            pmean=fmt(item.get("power_mean_w")),
            pmax=fmt(item.get("power_max_w")),
            temp=fmt(item.get("temp_max_c")),
            mem=fmt(item.get("mem_max_mib"), digits=0),
        ))
else:
    print("active none")
if stalled:
    print("stalled " + ",".join(str(item.get("case")) for item in stalled))
if failed:
    print("failed " + ",".join(str(item.get("case")) for item in failed))
PY

exit "$rc"
