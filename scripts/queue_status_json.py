#!/usr/bin/env python3
"""Machine-readable status for the production accuracy queue."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]


def run_text(cmd: List[str], env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=merged_env,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def expected_cases(queue_script: Path) -> List[str]:
    proc = run_text(["bash", str(queue_script), "--plan"])
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def result_state(out_dir: Path, name: str) -> str:
    result = out_dir / f"{name}.json"
    if not result.exists():
        return "missing"
    if result.stat().st_size == 0:
        return "stale"
    checker = ROOT / "scripts" / "check_result_metadata.py"
    args = ["python3", str(checker), "--strict", "--require-converged", "--validate-numeric"]
    if os.environ.get("BEM_METADATA_SKIP_CLOUDE") != "1":
        args.append("--require-cloude-physical")
    if name.startswith("dust_"):
        args.append("--require-complex-operator")
    args.append(str(result))
    proc = run_text(args)
    return "current" if proc.returncode == 0 else "stale"


def result_summary(out_dir: Path, name: str) -> Dict[str, object]:
    result = out_dir / f"{name}.json"
    if not result.exists() or result.stat().st_size == 0:
        return {}
    try:
        data = json.load(result.open())
    except (OSError, ValueError):
        return {}
    item: Dict[str, object] = {}
    for key in ("ka", "prism_aspect", "gmres_tol", "gmres_max_final_relres", "orientation_weight_sum"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            item[key] = value
    for key in ("refinements", "edge_refine", "fmm_digits", "gmres_restart", "gmres_max_cycles",
                "gmres_matvecs", "gmres_converged_systems", "gmres_nonconverged_systems",
                "gmres_stagnation_stops", "gmres_numerical_breakdowns",
                "gmres_restored_best_iterates", "gmres_max_cycle_exhaustions",
                "orient_start", "orient_count", "orient_total"):
        value = data.get(key)
        if isinstance(value, int):
            item[key] = value
    shape = data.get("shape")
    if isinstance(shape, str) and shape:
        item["shape"] = shape
    obj_file = data.get("obj_file")
    if isinstance(obj_file, str) and obj_file:
        item["obj_file"] = obj_file
    method = data.get("method")
    if isinstance(method, dict):
        farfield_mode = method.get("farfield_mode")
        if isinstance(farfield_mode, str) and farfield_mode:
            item["farfield_mode"] = farfield_mode
    mesh = data.get("mesh")
    if isinstance(mesh, dict):
        for key in ("vertices", "triangles", "quality_gate_pass",
                    "near_touch_checked", "near_touch_ratio"):
            if key in mesh:
                item[f"mesh_{key}"] = mesh[key]
    timing = data.get("timing")
    if not isinstance(timing, dict):
        return item
    value = timing.get("total_s")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return item
    if value >= 0.0:
        item["duration_s"] = value
    return item


def parse_accuracy_status(text: str) -> Dict[str, object]:
    states = []
    counts = {"accurate": 0, "accurate_legacy": 0, "inaccurate": 0, "missing": 0}
    summary_counts: Dict[str, int] = {}
    state_map = {
        "ACCURATE": "accurate",
        "ACCURATE_LEGACY": "accurate_legacy",
        "INACCURATE": "inaccurate",
        "MISSING_ACCURACY": "missing",
    }
    for line in text.splitlines():
        if not line.strip():
            continue
        head, *rest = line.split()
        if head == "SUMMARY_ACCURACY":
            for token in rest:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                try:
                    summary_counts[key] = int(value)
                except ValueError:
                    continue
            continue
        state = state_map.get(head)
        if state is None or not rest:
            continue
        item: Dict[str, object] = {"case": rest[0], "accuracy_state": state}
        for token in rest[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key == "gate":
                try:
                    item["gate_error"] = float(value)
                except ValueError:
                    item["gate_error"] = value
            else:
                item[key] = value
        states.append(item)
        counts[state] += 1
    counts["total"] = sum(counts.values())
    result: Dict[str, object] = {
        "counts": counts,
        "cases": states,
        "accurate_cases": [str(item["case"]) for item in states if item["accuracy_state"] == "accurate"],
        "accurate_legacy_cases": [
            str(item["case"]) for item in states if item["accuracy_state"] == "accurate_legacy"
        ],
        "inaccurate_cases": [str(item["case"]) for item in states if item["accuracy_state"] == "inaccurate"],
        "missing_accuracy_cases": [str(item["case"]) for item in states if item["accuracy_state"] == "missing"],
    }
    if summary_counts:
        result["summary_counts"] = summary_counts
        result["summary_mismatch"] = any(
            int(summary_counts.get(key, counts[key])) != int(counts[key])
            for key in ("accurate", "accurate_legacy", "inaccurate", "missing", "total")
        )
    return result


def _number_or_none(value: str) -> Optional[float]:
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "NOT SUPPORTED"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int_or_none(value: str) -> Optional[int]:
    number = _number_or_none(value)
    if number is None:
        return None
    return int(number)


def parse_gpu_inventory(
    gpu_csv: str,
    apps_by_gpu: Dict[str, str],
    *,
    max_temp_c: int,
    max_util_pct: int,
    max_mem_mib: int,
    allow_compute_share: bool,
) -> Dict[str, object]:
    gpus: List[Dict[str, object]] = []
    counts = {
        "total": 0,
        "usable": 0,
        "busy": 0,
        "unhealthy": 0,
        "unparseable": 0,
    }
    for line in gpu_csv.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 7:
            counts["unparseable"] += 1
            gpus.append({"raw": line, "status": "unparseable", "reasons": ["short_row"]})
            continue
        index, name, temp_s, util_s, mem_s, power_s, power_limit_s = fields[:7]
        temp = _int_or_none(temp_s)
        util = _int_or_none(util_s)
        mem = _int_or_none(mem_s)
        power = _number_or_none(power_s)
        power_limit = _number_or_none(power_limit_s)
        reasons: List[str] = []
        apps_text = apps_by_gpu.get(index, "").strip()
        compute_apps = [item.strip() for item in apps_text.splitlines() if item.strip()]
        if temp is None:
            reasons.append("temp_unparseable")
        elif temp > max_temp_c:
            reasons.append(f"temp>{max_temp_c}")
        if util is None:
            reasons.append("util_unparseable")
        elif util > max_util_pct:
            reasons.append(f"util>{max_util_pct}")
        if mem is None:
            reasons.append("mem_unparseable")
        elif mem > max_mem_mib:
            reasons.append(f"mem>{max_mem_mib}")
        if compute_apps and not allow_compute_share:
            reasons.append("compute_apps")

        if any(reason.endswith("_unparseable") for reason in reasons):
            status = "unhealthy"
        elif compute_apps and not allow_compute_share:
            status = "busy"
        elif reasons:
            status = "unhealthy"
        else:
            status = "usable"
        counts[status] += 1
        counts["total"] += 1
        item: Dict[str, object] = {
            "index": int(index) if index.isdigit() else index,
            "name": name,
            "status": status,
            "reasons": reasons,
            "temperature_c": temp,
            "utilization_pct": util,
            "memory_used_mib": mem,
            "power_w": power,
            "power_limit_w": power_limit,
            "compute_apps": compute_apps,
        }
        gpus.append(item)
    return {
        "available": True,
        "allow_compute_share": allow_compute_share,
        "thresholds": {
            "max_temp_c": max_temp_c,
            "max_util_pct": max_util_pct,
            "max_mem_mib": max_mem_mib,
        },
        "counts": counts,
        "usable_gpu_indices": [item["index"] for item in gpus if item.get("status") == "usable"],
        "busy_gpu_indices": [item["index"] for item in gpus if item.get("status") == "busy"],
        "unhealthy_gpu_indices": [item["index"] for item in gpus if item.get("status") == "unhealthy"],
        "gpus": gpus,
    }


def gpu_inventory() -> Dict[str, object]:
    nvidia_smi = os.environ.get("BEM_NVIDIA_SMI", "nvidia-smi")
    max_temp_c = int(os.environ.get("BEM_QUEUE_MAX_TEMP_C", "80"))
    max_util_pct = int(os.environ.get("BEM_QUEUE_MAX_UTIL_PCT", "20"))
    max_mem_mib = int(os.environ.get("BEM_QUEUE_MAX_MEM_MB", "2048"))
    allow_compute_share = os.environ.get("BEM_QUEUE_ALLOW_COMPUTE_SHARE", "0") == "1"
    resolved = shutil.which(nvidia_smi) if os.path.basename(nvidia_smi) == nvidia_smi else nvidia_smi
    if not resolved or not Path(resolved).exists():
        return {
            "available": False,
            "nvidia_smi": nvidia_smi,
            "error": "nvidia-smi missing",
            "counts": {"total": 0, "usable": 0, "busy": 0, "unhealthy": 0, "unparseable": 0},
            "usable_gpu_indices": [],
            "busy_gpu_indices": [],
            "unhealthy_gpu_indices": [],
            "gpus": [],
        }
    query = [
        nvidia_smi,
        "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    proc = run_text(query)
    if proc.returncode != 0:
        return {
            "available": False,
            "nvidia_smi": nvidia_smi,
            "error": proc.stderr.strip() or proc.stdout.strip() or f"{nvidia_smi} failed",
            "counts": {"total": 0, "usable": 0, "busy": 0, "unhealthy": 0, "unparseable": 0},
            "usable_gpu_indices": [],
            "busy_gpu_indices": [],
            "unhealthy_gpu_indices": [],
            "gpus": [],
        }
    apps_by_gpu: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        index = line.split(",", 1)[0].strip()
        apps = run_text([
            nvidia_smi,
            "-i",
            index,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        apps_by_gpu[index] = apps.stdout if apps.returncode == 0 else ""
    result = parse_gpu_inventory(
        proc.stdout,
        apps_by_gpu,
        max_temp_c=max_temp_c,
        max_util_pct=max_util_pct,
        max_mem_mib=max_mem_mib,
        allow_compute_share=allow_compute_share,
    )
    result["nvidia_smi"] = nvidia_smi
    return result


def accuracy_status(queue_script: Path, out_dir: Path, csv_path: Optional[Path]) -> Dict[str, object]:
    env = {"OUT": str(out_dir)}
    if csv_path is not None:
        env["BEM_QUEUE_ACCURACY_CSV"] = str(csv_path)
    proc = run_text(["bash", str(queue_script), "--status-accuracy"], env=env)
    parsed = parse_accuracy_status(proc.stdout)
    parsed["returncode"] = proc.returncode
    if proc.stderr.strip():
        parsed["stderr"] = proc.stderr.strip()
    return parsed


def read_gpu_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({
                    "timestamp_s": float(row["timestamp_s"]),
                    "gpu": float(row["gpu"]),
                    "temp_c": float(row["temp_c"]),
                    "util_pct": float(row["util_pct"]),
                    "mem_mib": float(row["mem_mib"]),
                    "power_w": float(row["power_w"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def terminal_line(log: Path) -> Optional[str]:
    if not log.exists():
        return None
    last: Optional[str] = None
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("DONE ") or line.startswith("FAIL "):
            last = line
    return last


def monitor_state(csv_path: Path, log_path: Path, now_s: int, active_age_s: int) -> str:
    if terminal_line(log_path):
        return "finished"
    if not csv_path.exists():
        return "missing"
    age = max(0, now_s - int(csv_path.stat().st_mtime))
    return "active" if age <= active_age_s else "stale"


def summarize_monitor(csv_path: Path, log_path: Path, now_s: int, active_age_s: int) -> Dict[str, object]:
    rows = read_gpu_rows(csv_path)
    case = csv_path.name[:-len(".gpu.csv")] if csv_path.name.endswith(".gpu.csv") else csv_path.name
    age_s = max(0, now_s - int(csv_path.stat().st_mtime)) if csv_path.exists() else None
    item: Dict[str, object] = {
        "case": case,
        "path": str(csv_path),
        "log": str(log_path),
        "state": monitor_state(csv_path, log_path, now_s, active_age_s),
        "age_s": age_s,
        "samples": len(rows),
        "terminal": terminal_line(log_path),
    }
    if not rows:
        return item
    power = [r["power_w"] for r in rows]
    temp = [r["temp_c"] for r in rows]
    util = [r["util_pct"] for r in rows]
    mem = [r["mem_mib"] for r in rows]
    item.update({
        "gpu": int(rows[0]["gpu"]),
        "duration_s": max(0.0, rows[-1]["timestamp_s"] - rows[0]["timestamp_s"]),
        "power_mean_w": mean(power),
        "power_max_w": max(power),
        "util_mean_pct": mean(util),
        "temp_max_c": max(temp),
        "mem_max_mib": max(mem),
    })
    return item


def queue_process(out_dir: Path) -> Dict[str, object]:
    pid_file = out_dir / "queue.pid"
    if not pid_file.exists():
        return {"pid": None, "running": False}
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        return {"pid": None, "running": False, "error": "invalid-pid"}
    proc = run_text(["bash", "-lc", f"kill -0 {pid} 2>/dev/null"])
    return {"pid": pid, "running": proc.returncode == 0}


def load_snapshot(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"now_s": None, "samples": {}}
    try:
        data = json.load(path.open())
    except (OSError, ValueError):
        return {"now_s": None, "samples": {}}
    if not isinstance(data, dict):
        return {"now_s": None, "samples": {}}
    samples = data.get("samples")
    return {
        "now_s": data.get("now_s"),
        "samples": samples if isinstance(samples, dict) else {},
    }


def write_snapshot(path: Path, now_s: int, monitors: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "now_s": now_s,
        "samples": {str(item["case"]): int(item.get("samples", 0)) for item in monitors},
    }
    tmp = path.with_name(path.name + ".tmp")
    json.dump(payload, tmp.open("w"), ensure_ascii=False, sort_keys=True)
    tmp.replace(path)


def attach_deltas(monitors: List[Dict[str, object]], snapshot: Dict[str, object],
                  now_s: int, stall_wall_s: int) -> Dict[str, object]:
    prev_now = snapshot.get("now_s")
    prev_samples = snapshot.get("samples", {})
    wall_s = now_s - int(prev_now) if isinstance(prev_now, int) else None
    sample_delta: Dict[str, object] = {}
    for item in monitors:
        case = str(item["case"])
        current = int(item.get("samples", 0))
        previous = prev_samples.get(case) if isinstance(prev_samples, dict) else None
        if isinstance(previous, int):
            delta = current - previous
        else:
            delta = None
        item["sample_delta"] = delta
        if item.get("state") == "active":
            if delta is None:
                item["progress_state"] = "unknown"
            elif delta > 0:
                item["progress_state"] = "progressing"
            elif wall_s is None or wall_s < stall_wall_s:
                item["progress_state"] = "unknown"
            else:
                item["progress_state"] = "stalled"
        else:
            item["progress_state"] = item.get("state")
        sample_delta[case] = delta
    return {"wall_s": wall_s, "monitor_sample_delta": sample_delta}


def requested_exit_code(payload: Dict[str, object], args: argparse.Namespace) -> int:
    if args.fail_on_failed and payload.get("failed_monitors"):
        return 20
    if args.fail_on_stalled and payload.get("stalled_monitors"):
        return 21
    accuracy = payload.get("accuracy", {})
    accuracy_counts = accuracy.get("counts", {}) if isinstance(accuracy, dict) else {}
    if args.fail_on_inaccurate and isinstance(accuracy_counts, dict):
        if int(accuracy_counts.get("inaccurate", 0)) > 0 or int(accuracy_counts.get("missing", 0)) > 0:
            return 26
    if args.fail_on_accuracy_legacy and isinstance(accuracy_counts, dict):
        if (
            int(accuracy_counts.get("accurate_legacy", 0)) > 0 or
            int(accuracy_counts.get("missing", 0)) > 0
        ):
            return 27
    if getattr(args, "fail_on_accuracy_summary_mismatch", False) and isinstance(accuracy, dict):
        if bool(accuracy.get("summary_mismatch")):
            return 30
    gpu = payload.get("gpu_inventory", {})
    if isinstance(gpu, dict):
        if getattr(args, "fail_on_gpu_inventory_unavailable", False) and not bool(gpu.get("available")):
            return 28
        min_usable_gpus = getattr(args, "min_usable_gpus", None)
        if min_usable_gpus is not None:
            gpu_counts = gpu.get("counts", {})
            usable = int(gpu_counts.get("usable", 0)) if isinstance(gpu_counts, dict) else 0
            if not bool(gpu.get("available")) or usable < int(min_usable_gpus):
                return 29
    if args.fail_on_stopped_incomplete and payload.get("queue_stopped_incomplete"):
        return 25
    counts = payload.get("counts", {})
    if args.fail_on_missing and isinstance(counts, dict) and int(counts.get("missing", 0)) > 0:
        return 22
    if args.fail_on_stale and isinstance(counts, dict) and int(counts.get("stale", 0)) > 0:
        return 23
    if args.fail_on_incomplete and isinstance(counts, dict):
        if int(counts.get("missing", 0)) > 0 or int(counts.get("stale", 0)) > 0:
            return 24
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/production_matrix_15"))
    parser.add_argument("--active-age-s", type=int, default=30)
    parser.add_argument("--snapshot", type=Path, default=None,
                        help="Snapshot path for sample deltas; default: OUT/logs/.queue_status_json.snapshot")
    parser.add_argument("--stall-wall-s", type=int, default=60,
                        help="Minimum snapshot interval before zero sample_delta means stalled")
    parser.add_argument("--no-snapshot-write", action="store_true",
                        help="Compute deltas from snapshot but do not update it")
    parser.add_argument("--queue-script", type=Path, default=ROOT / "scripts" / "run_accuracy_matrix_15_queue.sh")
    parser.add_argument("--accuracy-csv", type=Path, default=None,
                        help="Use an existing audit CSV for accuracy status instead of recomputing it")
    parser.add_argument("--no-accuracy-status", action="store_true",
                        help="Do not include accuracy-gate status in the JSON payload")
    parser.add_argument("--no-gpu-inventory", action="store_true",
                        help="Do not query nvidia-smi for usable/busy/unhealthy GPU status")
    parser.add_argument("--fail-on-failed", action="store_true",
                        help="Exit 20 when any monitor log contains FAIL")
    parser.add_argument("--fail-on-stalled", action="store_true",
                        help="Exit 21 when any active monitor has zero sample_delta after --stall-wall-s")
    parser.add_argument("--fail-on-stopped-incomplete", action="store_true",
                        help="Exit 25 when queue is not running and planned results are still missing/stale")
    parser.add_argument("--fail-on-missing", action="store_true",
                        help="Exit 22 when any planned result JSON is missing")
    parser.add_argument("--fail-on-stale", action="store_true",
                        help="Exit 23 when any planned result JSON is stale or metadata-invalid")
    parser.add_argument("--fail-on-incomplete", action="store_true",
                        help="Exit 24 when any planned result JSON is missing or stale")
    parser.add_argument("--fail-on-inaccurate", action="store_true",
                        help="Exit 26 when any planned point is inaccurate or lacks accuracy evidence")
    parser.add_argument("--fail-on-accuracy-legacy", action="store_true",
                        help="Exit 27 when accuracy evidence is missing or uses legacy metadata")
    parser.add_argument("--fail-on-accuracy-summary-mismatch", action="store_true",
                        help="Exit 30 when SUMMARY_ACCURACY disagrees with parsed case rows")
    parser.add_argument("--fail-on-gpu-inventory-unavailable", action="store_true",
                        help="Exit 28 when nvidia-smi is missing or cannot query GPUs")
    parser.add_argument("--min-usable-gpus", type=int, default=None,
                        help="Exit 29 unless at least this many GPUs pass the status thresholds")
    args = parser.parse_args()

    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    now_s = int(time.time())
    snapshot_path = args.snapshot or (out_dir / "logs" / ".queue_status_json.snapshot")
    snapshot_path = snapshot_path if snapshot_path.is_absolute() else ROOT / snapshot_path
    cases = expected_cases(args.queue_script)
    case_states = []
    for name in cases:
        state = result_state(out_dir, name)
        item: Dict[str, object] = {"case": name, "state": state}
        if state == "current":
            item.update(result_summary(out_dir, name))
        case_states.append(item)
    counts = {"current": 0, "stale": 0, "missing": 0}
    for item in case_states:
        counts[str(item["state"])] += 1

    logs_dir = out_dir / "logs"
    monitors = [
        summarize_monitor(path, path.with_suffix("").with_suffix(".log"), now_s, args.active_age_s)
        for path in sorted(logs_dir.glob("*.gpu.csv"))
    ] if logs_dir.exists() else []
    delta = attach_deltas(monitors, load_snapshot(snapshot_path), now_s, args.stall_wall_s)

    queue = queue_process(out_dir)
    queue_stopped_incomplete = (
        not bool(queue.get("running"))
        and (counts.get("missing", 0) > 0 or counts.get("stale", 0) > 0)
    )
    result_durations = [
        float(item["duration_s"]) for item in case_states
        if item.get("state") == "current" and "duration_s" in item
    ]
    completed_result_duration_mean_s = mean(result_durations) if result_durations else None

    payload = {
        "out": str(out_dir),
        "now_s": now_s,
        "delta": delta,
        "queue": queue,
        "queue_stopped_incomplete": queue_stopped_incomplete,
        "counts": counts,
        "total": len(case_states),
        "completed_result_duration_count": len(result_durations),
        "completed_result_duration_mean_s": completed_result_duration_mean_s,
        "cases": case_states,
        "current_cases": [str(item["case"]) for item in case_states if item.get("state") == "current"],
        "stale_cases": [str(item["case"]) for item in case_states if item.get("state") == "stale"],
        "missing_cases": [str(item["case"]) for item in case_states if item.get("state") == "missing"],
        "monitors": monitors,
        "active_monitors": [item for item in monitors if item.get("state") == "active"],
        "stalled_monitors": [item for item in monitors if item.get("progress_state") == "stalled"],
        "failed_monitors": [item for item in monitors if str(item.get("terminal", "")).startswith("FAIL ")],
    }
    if not args.no_gpu_inventory:
        payload["gpu_inventory"] = gpu_inventory()
    if not args.no_accuracy_status:
        accuracy_csv = args.accuracy_csv
        if accuracy_csv is not None and not accuracy_csv.is_absolute():
            accuracy_csv = ROOT / accuracy_csv
        payload["accuracy"] = accuracy_status(args.queue_script, out_dir, accuracy_csv)
    if not args.no_snapshot_write:
        write_snapshot(snapshot_path, now_s, monitors)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return requested_exit_code(payload, args)


if __name__ == "__main__":
    raise SystemExit(main())
