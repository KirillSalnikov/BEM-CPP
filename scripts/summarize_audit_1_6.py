#!/usr/bin/env python3
"""Print a compact human-readable summary for an audit_1_6 JSON report."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_1_6 import determine_exit_code  # noqa: E402


def status(value: object) -> str:
    if value is True:
        return "ok"
    if value is False:
        return "fail"
    if value is None:
        return "skip"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, nargs="?",
                        default=ROOT / "runs" / "audit_1_6_report.json")
    parser.add_argument("--require-cuda-reference", action="store_true",
                        help="Use the same strict exit-code policy as audit_1_6")
    args = parser.parse_args()

    data = json.loads(args.report.read_text())
    print(f"report: {args.report}")
    metadata = data.get("metadata", {})
    if metadata:
        print("metadata:")
        print(f"  created_utc: {metadata.get('created_utc', '')}")
        print(f"  git_commit: {metadata.get('git_commit', '')}")
        print(f"  git_dirty: {'yes' if metadata.get('git_dirty') else 'no'}")
        if metadata.get("git_dirty_count") is not None:
            print(f"  git_dirty_count: {metadata.get('git_dirty_count')}")
    print("requirements:")
    for name, value in data.get("requirements", {}).items():
        print(f"  {name}: {status(value)}")

    print("cuda:")
    print(f"  toolchain_available: {status(data.get('cuda_toolchain_available'))}")
    print(f"  runtime_ready: {status(data.get('cuda_runtime_ready'))}")
    print(f"  reference_verified: {status(data.get('cuda_reference_verified'))}")

    dense = data.get("checks", {}).get("dense_fmm_reference", {})
    if dense.get("skipped"):
        print(f"  dense_fmm_reference: skip ({dense.get('reason', 'no reason')})")
        missing = dense.get("runtime_missing") or []
        if missing:
            print(f"  runtime_missing: {', '.join(missing)}")
    elif dense.get("pass") is False:
        print("  dense_fmm_reference: fail")
    elif dense.get("pass") is True:
        print("  dense_fmm_reference: ok")

    dense_abs = data.get("checks", {}).get("dense_fmm_absorbing_reference", {})
    if dense_abs.get("skipped"):
        print(f"  dense_fmm_absorbing_reference: skip ({dense_abs.get('reason', 'no reason')})")
    elif dense_abs.get("pass") is False:
        print("  dense_fmm_absorbing_reference: fail")
    elif dense_abs.get("pass") is True:
        print("  dense_fmm_absorbing_reference: ok")

    if not data.get("all_local_requirements_pass"):
        print("next: fix failed local requirements before running CUDA reference")
    elif not data.get("cuda_runtime_ready"):
        print("next: run scripts/run_cuda_reference_audits.sh on a host with NVIDIA runtime")
    elif not data.get("cuda_reference_verified"):
        print("next: inspect dense-vs-FMM reference failure under runs/audit_1_6_cuda")
    else:
        print("next: 1-6 local and CUDA reference gates are verified")

    return determine_exit_code(data, require_cuda_reference=args.require_cuda_reference)


if __name__ == "__main__":
    raise SystemExit(main())
