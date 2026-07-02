#!/usr/bin/env python3
"""Smoke-tests for missing-case resume launcher."""

import os
import stat
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "resume_accuracy_matrix_cases.sh"


def run(args, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        [str(SCRIPT), *args],
        cwd=str(ROOT),
        env=merged_env,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    base = [
        "--no-health-check",
        "--gpus", "0 1",
        "--max-jobs", "3",
        "--case-max-power", "290",
        "--case-max-bad-samples", "4",
    ]
    proc = run(base)
    assert proc.returncode == 0, proc.stdout
    assert "RESUME selected=2" in proc.stdout, proc.stdout
    assert proc.stdout.count("DRYRUN gpu=") == 2, proc.stdout

    proc = run([*base, "--allow-oversubscribe"])
    assert proc.returncode == 0, proc.stdout
    assert "RESUME selected=3" in proc.stdout, proc.stdout
    assert proc.stdout.count("DRYRUN gpu=") == 3, proc.stdout

    proc = run([
        "--no-health-check",
        "--gpus", "0 1",
        "--cases", "hex_ka30_ref5_balanced_q7_d5_tol1e3,sphere_ka30_ref6_current_q7_d6_tol3e3",
    ])
    assert proc.returncode == 0, proc.stdout
    assert "RESUME pending=2" in proc.stdout, proc.stdout
    assert "case=hex_ka30_ref5_balanced_q7_d5_tol1e3" in proc.stdout, proc.stdout
    assert "case=sphere_ka30_ref6_current_q7_d6_tol3e3" in proc.stdout, proc.stdout
    assert "case=sphere_ka5_ref4_current_q7_d6_tol3e3" not in proc.stdout, proc.stdout
    assert "DRYRUN gpu=0 case=hex_ka30_ref5_balanced_q7_d5_tol1e3" in proc.stdout, proc.stdout
    assert "DRYRUN gpu=1 case=sphere_ka30_ref6_current_q7_d6_tol3e3" in proc.stdout, proc.stdout

    proc = run([
        "--no-health-check",
        "--gpus", "0 1",
        "--cases", "hex_ka30_ref5_balanced_q7_d5_tol1e3,hex_ka30_ref5_balanced_q7_d5_tol1e3",
        "--max-jobs", "2",
    ])
    assert proc.returncode == 0, proc.stdout
    assert "RESUME pending=1" in proc.stdout, proc.stdout
    assert proc.stdout.count("case=hex_ka30_ref5_balanced_q7_d5_tol1e3") == 1, proc.stdout
    assert "DRYRUN gpu=1" not in proc.stdout, proc.stdout

    proc = run([
        "--no-health-check",
        "--gpus", "0",
        "--cases", "sphere_ka30_ref7_current_q9_d7_tol1e3",
    ])
    assert proc.returncode == 0, proc.stdout
    assert "RESUME pending=1" in proc.stdout, proc.stdout
    assert "DRYRUN gpu=0 case=sphere_ka30_ref7_current_q9_d7_tol1e3" in proc.stdout, proc.stdout

    proc = run([
        "--no-health-check",
        "--gpus", "0",
        "--cases", "dust_ka20_gmsh4200_balanced_q7_d5_tol1e3",
    ])
    assert proc.returncode == 1, proc.stdout
    assert "CASE_INVALID case=dust_ka20_gmsh4200_balanced_q7_d5_tol1e3" in proc.stdout, proc.stdout
    assert "legacy dust case disabled" in proc.stdout, proc.stdout

    proc = run([
        "--no-health-check",
        "--gpus", "0",
        "--cases", "dust_ka20_gmsh4200_balanced_q7_d6_tol5e4",
    ])
    assert proc.returncode == 0, proc.stdout
    assert "DRYRUN gpu=0 case=dust_ka20_gmsh4200_balanced_q7_d6_tol5e4" in proc.stdout, proc.stdout

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        out.mkdir()
        stale = out / "sphere_ka5_ref4_current_q7_d6_tol3e3.json"
        stale.write_text("{}\n")
        proc = run([
            "--no-health-check",
            "--gpus", "0",
            "--max-jobs", "1",
            "--out", str(out),
        ])
        assert proc.returncode == 0, proc.stdout
        assert "case=sphere_ka5_ref4_current_q7_d6_tol3e3" in proc.stdout, proc.stdout
        assert " --force " in proc.stdout, proc.stdout

    with tempfile.TemporaryDirectory() as tmp:
        fake_nvidia_smi = Path(tmp) / "nvidia-smi"
        fake_nvidia_smi.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == "--query-gpu=index --format=csv,noheader,nounits" ]]; then
  printf '0\\n2\\n'
  exit 0
fi
echo "unexpected fake nvidia-smi args: $*" >&2
exit 1
""")
        fake_nvidia_smi.chmod(fake_nvidia_smi.stat().st_mode | stat.S_IXUSR)
        proc = run([
            "--no-health-check",
            "--gpus", "auto",
            "--max-jobs", "3",
        ], env={"BEM_NVIDIA_SMI": str(fake_nvidia_smi)})
        assert proc.returncode == 0, proc.stdout
        assert "RESUME selected=2" in proc.stdout, proc.stdout
        assert "DRYRUN gpu=0 " in proc.stdout, proc.stdout
        assert "DRYRUN gpu=2 " in proc.stdout, proc.stdout

    with tempfile.TemporaryDirectory() as tmp:
        fake_nvidia_smi = Path(tmp) / "nvidia-smi"
        fake_nvidia_smi.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == "-i 0 --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits" ]]; then
  echo "42, 0, 100"
  exit 0
fi
if [[ "$*" == "-i 1 --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits" ]]; then
  echo "43, 0, 120"
  exit 0
fi
if [[ "$*" == "-i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits" ]]; then
  exit 0
fi
if [[ "$*" == "-i 1 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits" ]]; then
  echo "4242, ./mbs_po_gpu_float_fast, 304"
  exit 0
fi
echo "unexpected fake nvidia-smi args: $*" >&2
exit 1
""")
        fake_nvidia_smi.chmod(fake_nvidia_smi.stat().st_mode | stat.S_IXUSR)
        proc = run([
            "--gpus", "0 1",
            "--cases", "hex_ka30_ref5_balanced_q7_d5_tol1e3,sphere_ka30_ref6_current_q7_d6_tol3e3",
            "--max-jobs", "2",
        ], env={"BEM_NVIDIA_SMI": str(fake_nvidia_smi)})
        assert proc.returncode == 0, proc.stdout
        assert "GPU_BUSY gpu=1 compute_apps=4242, ./mbs_po_gpu_float_fast, 304" in proc.stdout, proc.stdout
        assert "RESUME pending=2 idle_gpus=1" in proc.stdout, proc.stdout
        assert "DRYRUN gpu=0 case=hex_ka30_ref5_balanced_q7_d5_tol1e3" in proc.stdout, proc.stdout
        assert "case=sphere_ka30_ref6_current_q7_d6_tol3e3" not in proc.stdout, proc.stdout

        proc = run([
            "--gpus", "0 1",
            "--cases", "hex_ka30_ref5_balanced_q7_d5_tol1e3,sphere_ka30_ref6_current_q7_d6_tol3e3",
            "--max-jobs", "2",
            "--allow-compute-share",
        ], env={"BEM_NVIDIA_SMI": str(fake_nvidia_smi)})
        assert proc.returncode == 0, proc.stdout
        assert "RESUME pending=2 idle_gpus=2" in proc.stdout, proc.stdout
        assert "DRYRUN gpu=1 case=sphere_ka30_ref6_current_q7_d6_tol3e3" in proc.stdout, proc.stdout

    print("resume accuracy matrix cases: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
