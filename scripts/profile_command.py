#!/usr/bin/env python3
"""Run one command with reproducible CPU, RAM, GPU, and I/O measurements."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

try:
    import psutil
except ImportError as error:  # pragma: no cover - exercised on user machines
    raise SystemExit(
        "profile_command.py requires psutil; install requirements-analysis.txt"
    ) from error


TIME_KEYS = {
    "User time (seconds)": "user_time_s",
    "System time (seconds)": "system_time_s",
    "Percent of CPU this job got": "time_cpu_percent",
    "Elapsed (wall clock) time (h:mm:ss or m:ss)": "time_elapsed_text",
    "Maximum resident set size (kbytes)": "time_max_rss_kib",
    "Major (requiring I/O) page faults": "major_page_faults",
    "Minor (reclaiming a frame) page faults": "minor_page_faults",
    "Voluntary context switches": "voluntary_context_switches",
    "Involuntary context switches": "involuntary_context_switches",
    "File system inputs": "filesystem_inputs",
    "File system outputs": "filesystem_outputs",
}

GPU_QUERY = (
    "utilization.gpu,utilization.memory,memory.used,power.draw,"
    "temperature.gpu,clocks.sm"
)

SAMPLE_FIELDS = (
    "elapsed_s",
    "process_cpu_percent",
    "process_rss_bytes",
    "process_threads",
    "system_cpu_percent",
    "system_memory_available_bytes",
    "system_swap_used_bytes",
    "disk_free_bytes",
    "cpu_frequency_mhz",
    "gpu_util_percent",
    "gpu_memory_util_percent",
    "gpu_memory_used_mib",
    "gpu_power_w",
    "gpu_temperature_c",
    "gpu_sm_clock_mhz",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def command_output(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def hardware_snapshot(cwd: Path, gpu_index: int) -> dict[str, Any]:
    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    frequency = psutil.cpu_freq(percpu=False)
    disk = shutil.disk_usage(cwd)
    snapshot: dict[str, Any] = {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "ram_total_bytes": virtual_memory.total,
        "swap_total_bytes": swap_memory.total,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "cpu_frequency_max_mhz": frequency.max if frequency else None,
        "git_commit": command_output(
            ["git", "rev-parse", "HEAD"], cwd
        ),
        "git_dirty": bool(command_output(
            ["git", "status", "--porcelain"], cwd
        )),
    }
    lscpu = command_output(["lscpu", "-J"], cwd)
    if lscpu:
        try:
            snapshot["lscpu"] = json.loads(lscpu)
        except json.JSONDecodeError:
            snapshot["lscpu_raw"] = lscpu
    if gpu_index >= 0 and shutil.which("nvidia-smi"):
        gpu = command_output(
            [
                "nvidia-smi", f"--id={gpu_index}",
                "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
                "--format=csv,noheader,nounits",
            ],
            cwd,
        )
        snapshot["gpu_raw"] = gpu
    else:
        snapshot["gpu_raw"] = None
    return snapshot


def parse_time_report(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        for source, target in TIME_KEYS.items():
            prefix = source + ":"
            if not line.startswith(prefix):
                continue
            raw_value = line[len(prefix):].strip()
            if target == "time_cpu_percent":
                values[target] = float(raw_value.rstrip("%"))
            elif target == "time_elapsed_text":
                values[target] = raw_value
            elif target.endswith("_s"):
                values[target] = float(raw_value)
            else:
                values[target] = int(raw_value)
            break
    return values


def gpu_sample(index: int, cwd: Path) -> dict[str, float] | None:
    if index < 0 or not shutil.which("nvidia-smi"):
        return None
    text = command_output(
        [
            "nvidia-smi", f"--id={index}",
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ],
        cwd,
    )
    if not text:
        return None
    try:
        values = [float(value.strip()) for value in text.splitlines()[0].split(",")]
        return dict(zip(
            (
                "gpu_util_percent", "gpu_memory_util_percent",
                "gpu_memory_used_mib", "gpu_power_w", "gpu_temperature_c",
                "gpu_sm_clock_mhz",
            ),
            values,
        ))
    except (ValueError, IndexError):
        return None


def process_tree_sample(
    process: psutil.Process,
    seen_cpu: dict[int, tuple[float, float, float]],
    seen_io: dict[int, tuple[int, int]],
    sample_time: float,
) -> dict[str, float]:
    processes: list[psutil.Process] = []
    try:
        processes = [process, *process.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    rss = 0
    threads = 0
    cpu_percent = 0.0
    for child in processes:
        try:
            memory = child.memory_info()
            cpu = child.cpu_times()
            io = child.io_counters()
            rss += memory.rss
            threads += child.num_threads()
            previous = seen_cpu.get(child.pid)
            if previous is not None:
                elapsed = sample_time - previous[2]
                used = (cpu.user - previous[0]) + (cpu.system - previous[1])
                if elapsed > 0.0 and used >= 0.0:
                    cpu_percent += 100.0 * used / elapsed
            seen_cpu[child.pid] = (cpu.user, cpu.system, sample_time)
            seen_io[child.pid] = (io.read_bytes, io.write_bytes)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {
        "process_rss_bytes": float(rss),
        "process_threads": float(threads),
        "process_cpu_percent": cpu_percent,
    }


def trapezoid_energy(samples: list[dict[str, Any]], key: str) -> float | None:
    valid = [sample for sample in samples if sample.get(key) is not None]
    if len(valid) < 2:
        return None
    energy = 0.0
    for previous, current in zip(valid, valid[1:]):
        dt = current["elapsed_s"] - previous["elapsed_s"]
        if dt > 0.0:
            energy += 0.5 * dt * (previous[key] + current[key])
    return energy


def finite_statistics(samples: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [
        float(sample[key]) for sample in samples
        if sample.get(key) is not None and math.isfinite(float(sample[key]))
    ]
    if not values:
        return {"mean": None, "maximum": None, "minimum": None}
    return {
        "mean": sum(values) / len(values),
        "maximum": max(values),
        "minimum": min(values),
    }


def parse_environment(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--env requires KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit("--env key must not be empty")
        result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")
    if args.interval < 0.1:
        parser.error("--interval must be at least 0.1 s")
    cwd = args.cwd.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stdout_path = output / "stdout.log"
    time_path = output / "time_verbose.txt"
    samples_path = output / "resource_samples.csv"
    result_path = output / "resources.json"
    environment_overrides = parse_environment(args.env)
    environment = os.environ.copy()
    environment.update(environment_overrides)
    environment["LC_ALL"] = "C"
    baseline_gpu = gpu_sample(args.gpu, cwd)
    metadata = {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "cwd": str(cwd),
        "command": command,
        "command_shell": shlex.join(command),
        "command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "environment_overrides": environment_overrides,
        "sample_interval_s": args.interval,
        "gpu_index": args.gpu,
        "hardware": hardware_snapshot(cwd, args.gpu),
        "measurement_scope": {
            "gnu_time": "launcher and waited-for child processes",
            "process_samples": "live launcher process tree",
            "process_cpu_percent": (
                "sum of child CPU-time deltas divided by sample wall time; "
                "100 percent equals one fully occupied logical CPU"
            ),
            "gpu_samples": "whole selected GPU; exclusive GPU use is required",
            "gpu_energy": "trapezoidal integral of whole-GPU board power",
        },
    }
    atomic_json(output / "profile_plan.json", metadata)
    time_executable = shutil.which("time") or "/usr/bin/time"
    timed_command = [time_executable, "-v", "-o", str(time_path), "--", *command]
    samples: list[dict[str, Any]] = []
    seen_cpu: dict[int, tuple[float, float, float]] = {}
    seen_io: dict[int, tuple[int, int]] = {}
    started = time.monotonic()
    interrupted_signal: int | None = None
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        samples_path.open("w", encoding="utf-8", newline="") as sample_stream,
    ):
        sample_writer = csv.DictWriter(sample_stream, fieldnames=SAMPLE_FIELDS)
        sample_writer.writeheader()
        sample_stream.flush()
        process = subprocess.Popen(
            timed_command,
            cwd=cwd,
            env=environment,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        previous_handlers: dict[int, Any] = {}

        def forward_signal(signum: int, _frame: Any) -> None:
            nonlocal interrupted_signal
            interrupted_signal = signum
            try:
                os.killpg(os.getpgid(process.pid), signum)
            except (ProcessLookupError, PermissionError):
                pass

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, forward_signal)
        tracked = psutil.Process(process.pid)
        tracked.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        try:
            while process.poll() is None:
                sample_started = time.monotonic()
                sample: dict[str, Any] = {
                    "elapsed_s": sample_started - started,
                    "system_cpu_percent": psutil.cpu_percent(interval=None),
                    "system_memory_available_bytes": psutil.virtual_memory().available,
                    "system_swap_used_bytes": psutil.swap_memory().used,
                    "disk_free_bytes": shutil.disk_usage(cwd).free,
                }
                frequency = psutil.cpu_freq(percpu=False)
                sample["cpu_frequency_mhz"] = frequency.current if frequency else None
                sample.update(process_tree_sample(
                    tracked, seen_cpu, seen_io, sample_started
                ))
                gpu = gpu_sample(args.gpu, cwd)
                if gpu:
                    sample.update(gpu)
                samples.append(sample)
                sample_writer.writerow(sample)
                sample_stream.flush()
                remaining_interval = max(
                    0.0, args.interval - (time.monotonic() - sample_started)
                )
                try:
                    process.wait(timeout=remaining_interval)
                except subprocess.TimeoutExpired:
                    pass
            return_code = process.wait()
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    wall_time = time.monotonic() - started

    time_report = parse_time_report(time_path)
    aggregates = {
        key: finite_statistics(samples, key)
        for key in (
            "process_rss_bytes", "process_threads", "process_cpu_percent",
            "system_cpu_percent", "system_memory_available_bytes",
            "system_swap_used_bytes", "disk_free_bytes",
            "cpu_frequency_mhz", "gpu_util_percent",
            "gpu_memory_util_percent", "gpu_memory_used_mib", "gpu_power_w",
            "gpu_temperature_c", "gpu_sm_clock_mhz",
        )
    }
    gpu_energy = trapezoid_energy(samples, "gpu_power_w")
    baseline_power = baseline_gpu.get("gpu_power_w") if baseline_gpu else None
    gpu_incremental_energy = None
    if baseline_power is not None and len(samples) >= 2:
        adjusted = [
            {
                "elapsed_s": sample["elapsed_s"],
                "incremental_gpu_power_w": max(
                    0.0, float(sample.get("gpu_power_w", baseline_power)) - baseline_power
                ),
            }
            for sample in samples
            if sample.get("gpu_power_w") is not None
        ]
        gpu_incremental_energy = trapezoid_energy(
            adjusted, "incremental_gpu_power_w"
        )
    result = {
        **metadata,
        "finished_at_utc": utc_now(),
        "return_code": return_code,
        "interrupted_signal": interrupted_signal,
        "wall_time_s": wall_time,
        "gnu_time": time_report,
        "sample_count": len(samples),
        "baseline_gpu": baseline_gpu,
        "aggregates": aggregates,
        "gpu_board_energy_j": gpu_energy,
        "gpu_incremental_energy_j": gpu_incremental_energy,
        "observed_process_cpu_user_s": sum(value[0] for value in seen_cpu.values()),
        "observed_process_cpu_system_s": sum(value[1] for value in seen_cpu.values()),
        "observed_process_read_bytes": sum(value[0] for value in seen_io.values()),
        "observed_process_write_bytes": sum(value[1] for value in seen_io.values()),
        "disk_free_bytes_at_end": shutil.disk_usage(cwd).free,
        "artifacts": {
            "stdout": stdout_path.name,
            "gnu_time": time_path.name,
            "samples": samples_path.name,
        },
    }
    if baseline_gpu and aggregates["gpu_memory_used_mib"]["maximum"] is not None:
        result["gpu_memory_peak_delta_mib"] = max(
            0.0,
            aggregates["gpu_memory_used_mib"]["maximum"]
            - baseline_gpu["gpu_memory_used_mib"],
        )
    else:
        result["gpu_memory_peak_delta_mib"] = None
    atomic_json(result_path, result)
    print(f"profiled command exit={return_code}, wall={wall_time:.3f}s")
    print(f"resources: {result_path}")
    return 128 + interrupted_signal if interrupted_signal is not None else return_code


if __name__ == "__main__":
    raise SystemExit(main())
