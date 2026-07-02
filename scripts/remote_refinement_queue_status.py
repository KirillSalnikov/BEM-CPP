#!/usr/bin/env python3
"""Summarize the detached remote refinement queue state."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def ps_line(pid: int) -> str:
    proc = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,ppid=,sid=,etime=,cmd="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.stdout.strip()


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"json_error": "invalid"}


def file_age_s(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def tail_lines(path: Path, max_lines: int = 4000) -> list[str]:
    if not path.is_file():
        return []
    lines = path.read_text(errors="replace").splitlines()
    return lines[-max_lines:]


def parse_log(lines: list[str]) -> dict:
    attempts = []
    starts = []
    waits = []
    sync_skips = []
    last_status = None
    for line in lines:
        if line.startswith("REMOTE_QUEUE_SUPERVISOR attempt="):
            attempts.append(line)
        elif line.startswith("REMOTE_START "):
            starts.append(line)
        elif line.startswith("REMOTE_QUEUE_WAIT "):
            waits.append(line)
        elif line.startswith("REMOTE_SYNC_SKIP "):
            sync_skips.append(line)
        elif line.startswith("REMOTE_QUEUE_STATUS "):
            last_status = line
    parsed_status = {}
    if last_status:
        parsed_status = dict(re.findall(r"([A-Za-z_]+)=([^ ]+)", last_status))
    return {
        "supervisor_attempts": len([x for x in attempts if " start_time=" in x]),
        "remote_starts": starts,
        "queue_waits": waits,
        "sync_skips": sync_skips,
        "last_queue_status": last_status,
        "last_queue_status_fields": parsed_status,
    }


def queue_health(*, alive: bool, planned_cases: int, usable_remote_gpus: int,
                 selected_last_wave: int, remote_rc, plan_failed: bool = False,
                 status_stale: bool = False, min_usable_gpus: int | None = None) -> str:
    if plan_failed:
        return "plan_failed"
    if status_stale:
        return "status_stale"
    if not alive:
        return "stopped"
    if planned_cases <= 0:
        return "idle"
    if usable_remote_gpus <= 0:
        return "waiting_for_gpus"
    if min_usable_gpus is not None and usable_remote_gpus < min_usable_gpus:
        return "insufficient_gpus"
    if selected_last_wave <= 0 and str(remote_rc) == "0":
        return "no_cases_started"
    return "running_or_ready"


def remote_gpu_summary(*, auto_hosts: list, usable_remote_gpus: int,
                       busy_gpus: list, skipped_gpus: list) -> dict:
    busy_count = len(busy_gpus)
    skipped_count = len(skipped_gpus)
    reachable_count = usable_remote_gpus + busy_count
    return {
        "hosts": len(auto_hosts),
        "usable": usable_remote_gpus,
        "busy": busy_count,
        "skipped": skipped_count,
        "reachable": reachable_count,
        "blocked": busy_count + skipped_count,
    }


def exit_code(*, alive: bool, planned_cases: int, usable_remote_gpus: int,
              min_usable_gpus: int | None) -> int:
    if not alive:
        return 3
    if planned_cases > 0 and min_usable_gpus is not None and usable_remote_gpus < min_usable_gpus:
        return 4
    return 0


def int_or_none(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def bool_or_none(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def status_int(status: dict, log_fields: dict, key: str, default: int = 0) -> int:
    if key in status:
        parsed = int_or_none(status.get(key))
        if parsed is not None:
            return parsed
    parsed = int_or_none(log_fields.get(key))
    return parsed if parsed is not None else default


def nested_status_int(status: dict, outer_key: str, key: str, default: int = 0) -> int:
    nested = status.get(outer_key, {})
    if isinstance(nested, dict):
        parsed = int_or_none(nested.get(key))
        if parsed is not None:
            return parsed
    return default


def scan_case_leases(out_dir: str | None) -> list[dict]:
    if not out_dir:
        return []
    lease_root = Path(out_dir) / "remote_case_leases"
    if not lease_root.is_dir():
        return []
    leases = []
    for lock_dir in sorted(lease_root.glob("*.lock")):
        if not lock_dir.is_dir():
            continue
        owner_path = lock_dir / "owner"
        owner = owner_path.read_text(errors="replace").strip() if owner_path.is_file() else ""
        leases.append({
            "case": lock_dir.name.removesuffix(".lock"),
            "path": str(lock_dir),
            "owner": owner,
        })
    return leases


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Exit codes:
  0  supervisor is alive and enough GPUs are usable, or no cases are planned
  3  supervisor is not alive
  4  supervisor is alive, cases are planned, but usable GPUs are below the effective minimum

The effective GPU minimum is --min-usable-gpus when provided, otherwise the
min_free_gpus stored in status.json, otherwise 1 when planned_cases > 0.
""",
    )
    parser.add_argument("--queue-dir", default="runs/remote_refinement_queue")
    parser.add_argument("--json", action="store_true", help="print JSON only")
    parser.add_argument(
        "--status-max-age-s",
        type=float,
        default=0.0,
        help="mark status_stale when status.json is older than this many seconds; 0 disables the gate",
    )
    parser.add_argument(
        "--min-usable-gpus",
        type=int,
        default=None,
        help="return exit code 4 when the queue is alive but fewer than this many GPUs are usable",
    )
    args = parser.parse_args()

    queue_dir = Path(args.queue_dir)
    pid_path = queue_dir / "supervisor.pid"
    status_path = queue_dir / "status.json"
    log_path = queue_dir / "supervisor.log"

    pid = None
    if pid_path.is_file():
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            pid = None
    alive = bool(pid and pid_alive(pid))
    status = read_json(status_path)
    status_age_s = file_age_s(status_path)
    status_stale = bool(
        args.status_max_age_s > 0
        and (status_age_s is None or status_age_s > args.status_max_age_s)
    )
    log_summary = parse_log(tail_lines(log_path))

    skipped_gpus = status.get("skipped_gpus", [])
    if not skipped_gpus:
        skipped_gpus = [
            line
            for line in status.get("remote_status_lines", [])
            if line.startswith("REMOTE_GPU_SKIP ") or line.startswith("REMOTE_GPU_LIST_SKIP ")
        ]

    log_status_fields = log_summary["last_queue_status_fields"]

    planned_cases = int(status.get("planned_cases", 0) or 0)
    usable_remote_gpus = status_int(status, log_status_fields, "usable_remote_gpus")
    selected_last_wave = status_int(status, log_status_fields, "selected")
    lease_files = scan_case_leases(status.get("out"))
    status_leased_cases = nested_status_int(status, "remote_resume", "leased_cases")
    leased_cases = max(status_leased_cases, len(lease_files))
    remote_resume_cases = nested_status_int(status, "remote_resume", "cases", planned_cases)
    remote_rc = status.get("remote_rc")
    if remote_rc is None:
        remote_rc = log_status_fields.get("remote_rc")
    plan_failed = bool(status.get("plan_failed"))
    status_min_free_gpus = int_or_none(status.get("min_free_gpus"))
    if status_min_free_gpus is None:
        status_min_free_gpus = int_or_none(log_status_fields.get("min_free_gpus"))
    status_enough_free_gpus = bool_or_none(status.get("enough_free_gpus"))
    if status_enough_free_gpus is None:
        status_enough_free_gpus = bool_or_none(log_status_fields.get("enough_free_gpus"))
    effective_min_usable_gpus = (
        args.min_usable_gpus
        if args.min_usable_gpus is not None
        else (status_min_free_gpus if status_min_free_gpus is not None else (1 if planned_cases > 0 else None))
    )
    auto_hosts = status.get("auto_hosts", [])
    busy_gpus = status.get("busy_gpus", [])
    gpu_summary = remote_gpu_summary(
        auto_hosts=auto_hosts,
        usable_remote_gpus=usable_remote_gpus,
        busy_gpus=busy_gpus,
        skipped_gpus=skipped_gpus,
    )

    report = {
        "queue_dir": str(queue_dir),
        "supervisor_pid": pid,
        "supervisor_alive": alive,
        "supervisor_ps": ps_line(pid) if pid and alive else "",
        "planned_cases": planned_cases,
        "remote_resume_cases": remote_resume_cases,
        "leased_cases": leased_cases,
        "status_leased_cases": status_leased_cases,
        "lease_files": lease_files,
        "usable_remote_gpus": usable_remote_gpus,
        "selected_last_wave": selected_last_wave,
        "plan_failed": plan_failed,
        "status_age_s": status_age_s,
        "status_stale": status_stale,
        "status_min_free_gpus": status_min_free_gpus,
        "min_usable_gpus": args.min_usable_gpus,
        "effective_min_usable_gpus": effective_min_usable_gpus,
        "enough_usable_gpus": (
            None if effective_min_usable_gpus is None
            else usable_remote_gpus >= effective_min_usable_gpus
        ),
        "status_enough_free_gpus": status_enough_free_gpus,
        "queue_health": queue_health(
            alive=alive,
            planned_cases=planned_cases,
            usable_remote_gpus=usable_remote_gpus,
            selected_last_wave=selected_last_wave,
            remote_rc=remote_rc,
            plan_failed=plan_failed,
            status_stale=status_stale,
            min_usable_gpus=effective_min_usable_gpus,
        ),
        "auto_hosts": auto_hosts,
        "remote_gpu_summary": gpu_summary,
        "busy_gpus": busy_gpus,
        "skipped_gpus": skipped_gpus,
        "remote_rc": remote_rc,
        "remote_starts": log_summary["remote_starts"],
        "remote_start_count": len(log_summary["remote_starts"]),
        "supervisor_attempts": log_summary["supervisor_attempts"],
        "last_queue_status": log_summary["last_queue_status"],
        "last_queue_status_fields": log_status_fields,
        "sync_skips": log_summary["sync_skips"],
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return exit_code(
            alive=alive,
            planned_cases=planned_cases,
            usable_remote_gpus=usable_remote_gpus,
            min_usable_gpus=effective_min_usable_gpus,
        )

    print(f"QUEUE dir={report['queue_dir']}")
    print(f"SUPERVISOR alive={int(alive)} pid={pid or ''}")
    if report["supervisor_ps"]:
        print(f"PS {report['supervisor_ps']}")
    age_text = "-" if report["status_age_s"] is None else f"{report['status_age_s']:.1f}"
    print(
        "REMOTE "
        f"hosts={','.join(report['auto_hosts']) or '-'} "
        f"usable_gpus={report['usable_remote_gpus']} "
        f"busy_gpus={report['remote_gpu_summary']['busy']} "
        f"skipped_gpus={report['remote_gpu_summary']['skipped']} "
        f"reachable_gpus={report['remote_gpu_summary']['reachable']} "
        f"min_usable_gpus={report['min_usable_gpus'] if report['min_usable_gpus'] is not None else '-'} "
        f"status_min_free_gpus={report['status_min_free_gpus'] if report['status_min_free_gpus'] is not None else '-'} "
        f"effective_min_usable_gpus={report['effective_min_usable_gpus'] if report['effective_min_usable_gpus'] is not None else '-'} "
        f"enough_usable_gpus={report['enough_usable_gpus'] if report['enough_usable_gpus'] is not None else '-'} "
        f"last_selected={report['selected_last_wave']} "
        f"planned_cases={report['planned_cases']} "
        f"remote_resume_cases={report['remote_resume_cases']} "
        f"leased_cases={report['leased_cases']} "
        f"remote_rc={report['remote_rc']} "
        f"status_age_s={age_text} "
        f"stale={int(report['status_stale'])} "
        f"health={report['queue_health']}"
    )
    for busy in report["busy_gpus"]:
        print(f"BUSY {busy}")
    for skipped in report["skipped_gpus"]:
        print(f"SKIP {skipped}")
    for lease in report["lease_files"][:10]:
        print(f"LEASE case={lease['case']} path={lease['path']}")
    print(f"STARTS count={report['remote_start_count']}")
    for start in report["remote_starts"][-10:]:
        print(f"START {start}")
    if report["last_queue_status"]:
        print(f"LAST {report['last_queue_status']}")
    for skip in report["sync_skips"][-5:]:
        print(f"SYNC_SKIP {skip}")
    return exit_code(
        alive=alive,
        planned_cases=planned_cases,
        usable_remote_gpus=usable_remote_gpus,
        min_usable_gpus=effective_min_usable_gpus,
    )


if __name__ == "__main__":
    raise SystemExit(main())
