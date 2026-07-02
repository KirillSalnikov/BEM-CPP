#!/usr/bin/env python3
"""Smoke tests for summarize_audit_1_6 output and exit-code policy."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_audit_1_6.py"


def write_report(path: Path, *, local_ok: bool, runtime_ok: bool,
                 reference_ok: bool, dense: dict) -> None:
    report = {
        "metadata": {
            "created_utc": "2026-06-24T00:00:00+00:00",
            "git_commit": "test-commit",
            "git_dirty": True,
            "git_dirty_count": 3,
        },
        "requirements": {
            "1_cpu_reference_pmchwt": local_ok,
            "2_mueller_sign_audit": local_ok,
            "3_singular_near_singular_hooks": local_ok,
            "4_mesh_quality_gate": local_ok,
            "5_operator_architecture": local_ok,
            "6_python_job_api": local_ok,
            "cuda_build_runtime_gate": local_ok,
        },
        "all_local_requirements_pass": local_ok,
        "cuda_toolchain_available": True,
        "cuda_runtime_ready": runtime_ok,
        "cuda_reference_verified": reference_ok,
        "checks": {
            "dense_fmm_reference": dense,
            "dense_fmm_absorbing_reference": dense,
        },
    }
    path.write_text(json.dumps(report) + "\n")


def run_summary(path: Path, *, strict: bool) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT), str(path)]
    if strict:
        cmd.append("--require-cuda-reference")
    return subprocess.run(cmd, cwd=str(ROOT), universal_newlines=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)

        skipped = tmp / "skipped.json"
        write_report(
            skipped,
            local_ok=True,
            runtime_ok=False,
            reference_ok=False,
            dense={
                "skipped": True,
                "reason": "CUDA toolkit is available, but NVIDIA driver/runtime is not ready on this host",
                "runtime_missing": ["/dev/nvidia*", "libcuda.so"],
            },
        )
        proc = run_summary(skipped, strict=False)
        assert proc.returncode == 0, proc
        assert "git_commit: test-commit" in proc.stdout
        assert "git_dirty: yes" in proc.stdout
        assert "git_dirty_count: 3" in proc.stdout
        assert "runtime_missing: /dev/nvidia*, libcuda.so" in proc.stdout
        assert "next: run scripts/run_cuda_reference_audits.sh" in proc.stdout

        proc = run_summary(skipped, strict=True)
        assert proc.returncode == 4, proc

        local_fail = tmp / "local_fail.json"
        write_report(local_fail, local_ok=False, runtime_ok=False,
                     reference_ok=False, dense={"skipped": True})
        proc = run_summary(local_fail, strict=True)
        assert proc.returncode == 2, proc
        assert "next: fix failed local requirements" in proc.stdout

        full_pass = tmp / "full_pass.json"
        write_report(full_pass, local_ok=True, runtime_ok=True,
                     reference_ok=True, dense={"pass": True})
        proc = run_summary(full_pass, strict=True)
        assert proc.returncode == 0, proc
        assert "dense_fmm_reference: ok" in proc.stdout
        assert "dense_fmm_absorbing_reference: ok" in proc.stdout
        assert "next: 1-6 local and CUDA reference gates are verified" in proc.stdout

    print("audit_1_6 summary: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
