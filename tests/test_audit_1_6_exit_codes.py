#!/usr/bin/env python3
"""Unit tests for audit_1_6 exit-code policy."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_1_6 import determine_exit_code  # noqa: E402


def main() -> int:
    local_fail = {
        "all_local_requirements_pass": False,
        "cuda_reference_verified": True,
    }
    assert determine_exit_code(local_fail, require_cuda_reference=False) == 2
    assert determine_exit_code(local_fail, require_cuda_reference=True) == 2

    local_pass_cuda_skipped = {
        "all_local_requirements_pass": True,
        "cuda_reference_verified": False,
    }
    assert determine_exit_code(local_pass_cuda_skipped, require_cuda_reference=False) == 0
    assert determine_exit_code(local_pass_cuda_skipped, require_cuda_reference=True) == 4

    full_pass = {
        "all_local_requirements_pass": True,
        "cuda_reference_verified": True,
    }
    assert determine_exit_code(full_pass, require_cuda_reference=False) == 0
    assert determine_exit_code(full_pass, require_cuda_reference=True) == 0

    print("audit_1_6 exit codes: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
