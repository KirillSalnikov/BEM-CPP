#!/usr/bin/env python3
"""Promote 14-node dust profile checks to final dense-angle adaptive runs."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path


def metric_value(summary_script, bem, adda):
    output = subprocess.check_output(
        ["python3", summary_script, "--bem", str(bem), "--adda", str(adda)],
        universal_newlines=True,
    )
    for line in output.splitlines():
        if line.startswith("m11_integral_rel_l2: "):
            return float(line.split(": ", 1)[1])
    raise RuntimeError("M11 L2 metric is missing from {}".format(bem))


def matching_processes(fragment, required_token=None):
    output = subprocess.check_output(
        ["pgrep", "-af", "--", fragment], universal_newlines=True
    )
    own_pid = os.getpid()
    result = []
    for line in output.splitlines():
        pid_text, command = line.split(" ", 1)
        pid = int(pid_text)
        if (
            pid != own_pid
            and fragment in command
            and (required_token is None or required_token in command)
        ):
            result.append((pid, command))
    return result


def process_references_directory(pid, command, directory):
    try:
        cwd = Path("/proc/{}/cwd".format(pid)).resolve()
        tokens = shlex.split(command)
    except (OSError, ValueError):
        return False
    expected = directory.resolve()
    path_flags = {"--out", "--out-dir", "--work-dir"}
    for index, token in enumerate(tokens[:-1]):
        if token not in path_flags:
            continue
        candidate = Path(tokens[index + 1])
        if not candidate.is_absolute():
            candidate = cwd / candidate
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate == expected or expected in candidate.parents:
            return True
    return False


def stop_profile_processes(case_dir):
    process_names = ("bem_cuda_fmm", "run_orient_queue.py",
                     "adaptive_nested_bg_orient_queue.py")
    try:
        output = subprocess.check_output(
            ["pgrep", "-af", "--", "bem_cuda_fmm|run_orient_queue.py|adaptive_nested_bg_orient_queue.py"],
            universal_newlines=True,
        )
    except subprocess.CalledProcessError:
        output = ""
    stopped = []
    own_pid = os.getpid()
    candidates = []
    for line in output.splitlines():
        try:
            pid_text, command = line.split(" ", 1)
            pid = int(pid_text)
        except ValueError:
            continue
        if (pid != own_pid and any(name in command for name in process_names)
                and process_references_directory(pid, command, case_dir)):
            candidates.append((pid, command))
    # Stop leaf solvers before their queue and adaptive parents.
    candidates.sort(key=lambda item: ("bem_cuda_fmm" not in item[1],
                                      "run_orient_queue.py" not in item[1]))
    for pid, command in candidates:
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append({"pid": pid, "command": command})
        except ProcessLookupError:
            pass
    return stopped


def wait_for_exit(processes, timeout=30.0):
    def is_alive(pid):
        try:
            stat = Path("/proc/{}/stat".format(pid)).read_text()
        except OSError:
            return False
        fields = stat.split()
        return len(fields) > 2 and fields[2] != "Z"

    pending = {item["pid"] for item in processes}
    deadline = time.time() + timeout
    while pending and time.time() < deadline:
        for pid in list(pending):
            if not is_alive(pid):
                pending.remove(pid)
        if pending:
            time.sleep(0.2)
    forced = []
    for pid in pending:
        try:
            os.kill(pid, signal.SIGKILL)
            forced.append(pid)
        except ProcessLookupError:
            pass
    if forced:
        time.sleep(1.0)
    return forced


def launch_final(root, manifest, case, final_root):
    case_id = case.get("id", case["tag"])
    log = final_root / "launcher_logs" / (case_id + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "ROOT": str(root),
        "RUN_ROOT": str(final_root),
        "MANIFEST": str(manifest),
        "GPU": str(case["gpu"]),
        "KA": str(case["ka"]),
        "RI_IM": str(case.get("ri_im", 0)),
        "MESH": case["mesh"],
        "MAX_LEAF": str(case["max_leaf"]),
        "NTHETA": "1801",
        "ADDA": case["adda"],
    })
    stream = log.open("ab", buffering=0)
    process = subprocess.Popen(
        ["bash", "scripts/run_dust_adda_4gpu_goal.sh"],
        cwd=str(root), env=env, stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process.pid, str(log)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--max-l2", type=float, default=0.10)
    ap.add_argument("--summary-script", default="scripts/summarize_bem_adda_m11.py")
    args = ap.parse_args()
    config = json.load(args.config.open())
    root = Path(config["root"])
    final_root = root / config["final_root"]
    manifest = root / config["orientation_manifest"]

    pending = {case.get("id", case["tag"]): case for case in config["cases"]}
    while pending:
        for case_id, case in list(pending.items()):
            result_path = final_root / "promotion" / (case_id + ".json")
            if result_path.is_file():
                pending.pop(case_id)
                continue
            case_dir = root / case["profile_dir"]
            bem_files = sorted(case_dir.glob("level01_*/bem.json"))
            if not bem_files:
                continue
            l2 = metric_value(args.summary_script, bem_files[0], root / case["adda"])
            stopped = stop_profile_processes(case_dir)
            forced = wait_for_exit(stopped)
            record = {
                "ka": case["ka"],
                "profile_bem": str(bem_files[0]),
                "profile_l2": l2,
                "profile_pass": l2 <= args.max_l2,
                "stopped": stopped,
                "forced_stop_pids": forced,
                "time": time.time(),
            }
            if l2 <= args.max_l2:
                pid, log = launch_final(root, manifest, case, final_root)
                record.update({"final_pid": pid, "final_log": log, "ntheta": 1801})
            result_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = result_path.with_suffix(".tmp")
            with tmp.open("w") as stream:
                json.dump(record, stream, indent=2, sort_keys=True)
                stream.write("\n")
            tmp.replace(result_path)
            pending.pop(case_id)
        if pending:
            time.sleep(max(1, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
