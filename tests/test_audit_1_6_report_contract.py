#!/usr/bin/env python3
"""Unit tests for audit_1_6 report contract validation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_audit_1_6_report import ARTIFACTS, CHECKS, METADATA, REQUIREMENTS, validate_report  # noqa: E402


def valid_report() -> dict:
    return {
        "root": str(ROOT),
        "metadata": {
            key: [] if key == "git_dirty_sample" else 0 if key == "git_dirty_count" else False if key == "git_dirty" else "x"
            for key in METADATA
        },
        "checks": {key: {"pass": True} for key in CHECKS},
        "artifacts": {key: {"exists": True} for key in ARTIFACTS},
        "requirements": {key: True for key in REQUIREMENTS},
        "binary": {"path": "bin/bem_cuda_fmm", "exists": True, "executable": True},
        "all_local_requirements_pass": True,
        "cuda_toolchain_available": True,
        "cuda_runtime_ready": False,
        "cuda_runtime": {"missing": ["/dev/nvidia*"]},
        "cuda_reference_verified": False,
    }


def main() -> int:
    report = valid_report()
    assert validate_report(report) == []

    broken = valid_report()
    del broken["requirements"]["4_mesh_quality_gate"]
    broken["cuda_runtime_ready"] = "no"
    errors = validate_report(broken)
    assert "requirements.4_mesh_quality_gate" in errors
    assert "report.cuda_runtime_ready must be boolean" in errors

    print("audit_1_6 report contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
