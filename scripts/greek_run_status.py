#!/usr/bin/env python3
"""Compact status for the Greek ADDA BEM benchmark watcher/run."""

import argparse
import os
import shlex
import subprocess
from pathlib import Path


DEFAULT_REMOTE = "kirill_epyc@172.16.1.168"
DEFAULT_REMOTE_DIR = "/home/kirill_epyc/BEM-CUDA"


def run(cmd, check=True):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        if check:
            raise
        return exc.output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ax", default="33.28")
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--out", default="runs/greek_larger_valid/bem_auto_Ax33.28_a95b65g20_q4_n181.json")
    parser.add_argument("--log", default="runs/greek_larger_valid/bem_auto_Ax33.28_gpu0123_wait.local.log")
    args = parser.parse_args()
    nvidia_smi = shlex.quote(os.environ.get("BEM_NVIDIA_SMI", "nvidia-smi"))

    print(f"A_x={args.ax}")
    print("Local watcher:")
    local = run([
        "bash",
        "-lc",
        (
            f"ps -eo pid,stat,etime,args | "
            f"awk '/run_bem_candidate.py.*{args.ax}/ && !/awk/ {{print}}'"
        ),
    ]).strip()
    print(local if local else "  none")

    print("\nLocal result:")
    local_out = Path(args.out)
    has_local_result = local_out.is_file()
    print(f"  {local_out} {'exists' if has_local_result else 'missing'}")

    print("\nWait log tail:")
    log = Path(args.log)
    if log.is_file():
        tail = run([
            "bash",
            "-lc",
            (
                f"grep -E '^(Waiting for GPUs|\\[.*(Waiting for GPUs|GPU wait satisfied)|"
                f"=== BEM-CUDA|Results written|score6)' {str(log)!r} | tail -8"
            ),
        ]).rstrip()
        print(tail if tail else "  empty")
    else:
        print("  missing")

    remote_cmd = (
        f"ps -eo pid,stat,etime,args | awk '/bem_cuda_fmm.*{args.ax}/ && !/awk/ {{print}}'; "
        "echo GPU_STATUS; "
        f"{nvidia_smi} --query-gpu=index,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader 2>/dev/null || true; "
        "echo REMOTE_RESULT; "
        f"ls -lh {args.remote_dir}/{args.out} 2>/dev/null || true"
    )
    print("\nRemote:")
    remote = run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.remote, remote_cmd],
        check=False,
    ).rstrip()
    remote_unreachable = "No route to host" in remote or "Connection timed out" in remote
    print(remote if remote else "  no BEM process/result output")

    has_bem = "bem_cuda_fmm" in remote
    remote_result = remote.split("REMOTE_RESULT", 1)[1] if "REMOTE_RESULT" in remote else ""
    has_remote_result = args.out in remote_result
    print("\nSummary:")
    if has_local_result:
        print("  result is local; score/update summary next")
    elif remote_unreachable:
        print("  remote is unreachable; reconnect/resubmit watcher when network returns")
    elif has_remote_result:
        print("  result is remote; rsync and score next")
    elif has_bem:
        print("  BEM is running; monitor solve/far-field progress")
    elif local:
        print("  waiting for GPUs; do not start a duplicate watcher")
    else:
        print("  no active watcher/result; submit the benchmark run")


if __name__ == "__main__":
    main()
