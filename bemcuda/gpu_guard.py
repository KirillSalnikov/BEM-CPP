#!/usr/bin/env python3
"""Shared helpers for avoiding accidental CUDA oversubscription."""

import os
import shutil
import subprocess
import sys


def nvidia_smi_command(nvidia_smi=None):
    return nvidia_smi or os.environ.get("BEM_NVIDIA_SMI", "nvidia-smi")


def command_exists(command):
    return shutil.which(command) is not None or os.path.sep in command


def compute_apps(gpu, nvidia_smi=None):
    """Return non-empty nvidia-smi compute-process rows for one GPU."""

    smi = nvidia_smi_command(nvidia_smi)
    if not command_exists(smi):
        return []
    try:
        proc = subprocess.run(
            [
                smi,
                "-i",
                str(gpu),
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def detect_gpu_indices(nvidia_smi=None):
    """Return GPU indices reported by nvidia-smi, or an empty list."""

    smi = nvidia_smi_command(nvidia_smi)
    if not command_exists(smi):
        return []
    try:
        proc = subprocess.run(
            [smi, "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def parse_gpu_csv(spec):
    return [item.strip() for item in spec.split(",") if item.strip()]


def excluded_gpus():
    """GPU ids excluded from automatic scheduling.

    By default GPU3 is kept out of automatic BEM launchers on epyc1. Set
    BEM_EXCLUDE_GPUS="" to allow every detected GPU, or provide a CSV list.
    """

    return set(parse_gpu_csv(os.environ.get("BEM_EXCLUDE_GPUS", "3")))


def filter_excluded_gpus(gpus):
    excluded = excluded_gpus()
    return [gpu for gpu in gpus if str(gpu) not in excluded]


def busy_gpu_map(gpus, nvidia_smi=None):
    busy = {}
    for gpu in gpus:
        apps = compute_apps(gpu, nvidia_smi=nvidia_smi)
        if apps:
            busy[gpu] = apps
    return busy


def assert_gpus_free(gpus, allow_compute_share=False, nvidia_smi=None):
    if allow_compute_share:
        return
    busy = busy_gpu_map(gpus, nvidia_smi=nvidia_smi)
    if busy:
        details = "; ".join("gpu=%s: %s" % (gpu, " | ".join(apps)) for gpu, apps in busy.items())
        raise SystemExit("GPU_BUSY " + details)


def select_free_gpus(spec, allow_compute_share=False, nvidia_smi=None, *, stderr=None):
    """Select GPUs from a CSV spec, or all detected GPUs for ``auto``."""

    if spec.strip().lower() == "auto":
        detected = detect_gpu_indices(nvidia_smi=nvidia_smi)
        candidates = filter_excluded_gpus(detected)
        if detected and not candidates:
            raise SystemExit(
                "no selectable GPUs after BEM_EXCLUDE_GPUS="
                + os.environ.get("BEM_EXCLUDE_GPUS", "3")
            )
        if not candidates:
            raise SystemExit("no GPUs detected by nvidia-smi")
    else:
        candidates = filter_excluded_gpus(parse_gpu_csv(spec))
        if not candidates:
            raise SystemExit("no selectable GPUs from --gpus " + spec)
    free = []
    busy = busy_gpu_map(candidates, nvidia_smi=nvidia_smi)
    output = stderr if stderr is not None else sys.stderr
    for gpu in candidates:
        apps = busy.get(gpu)
        if apps and not allow_compute_share:
            print("GPU_BUSY gpu=%s compute_apps=%s" % (gpu, " | ".join(apps)), file=output, flush=True)
            continue
        free.append(gpu)
    if not free:
        raise SystemExit("no free GPUs from --gpus " + spec)
    return free
