#!/usr/bin/env python3
"""Compact status for the Greek ADDA BEM benchmark watcher/run."""

import argparse
import subprocess
from pathlib import Path


DEFAULT_REMOTE = "kirill_epyc@172.16.0.212"
DEFAULT_REMOTE_DIR = "/home/kirill_epyc/BEM-CUDA"


def run(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ax", default="33.28")
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--out", default="runs/greek_larger_valid/bem_auto_Ax33.28_a95b65g20_q4_n181.json")
    parser.add_argument("--log", default="runs/greek_larger_valid/bem_auto_Ax33.28_gpu0123_wait.local.log")
    args = parser.parse_args()

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
    print(f"  {local_out} {'exists' if local_out.is_file() else 'missing'}")

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
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader 2>/dev/null || true; "
        "echo REMOTE_RESULT; "
        f"ls -lh {args.remote_dir}/{args.out} 2>/dev/null || true"
    )
    print("\nRemote:")
    remote = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.remote, remote_cmd]).rstrip()
    print(remote if remote else "  no BEM process/result output")


if __name__ == "__main__":
    main()
