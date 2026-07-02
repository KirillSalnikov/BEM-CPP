#!/usr/bin/env python3
"""Smoke-tests for remote accuracy-matrix scheduler."""

import os
import stat
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "remote_resume_accuracy_matrix_cases.sh"
CASE_A = "hex_ka30_ref5_balanced_q7_d5_tol1e3"
CASE_B = "sphere_ka30_ref6_current_q7_d6_tol3e3"
CASE_C = "dust_ka20_gmsh4200_balanced_q7_d6_tol5e4"
LEGACY_DUST = "dust_ka20_gmsh4200_balanced_q7_d5_tol1e3"


FAKE_SSH = """#!/usr/bin/env bash
set -euo pipefail
host="$1"
cmd="$2"
if [[ "$cmd" == *"--query-compute-apps="* ]]; then
  case "$host" in
    host-compute)
      echo "4242, ./mbs_po_gpu_float_fast, 304"
      ;;
    *)
      ;;
  esac
  exit 0
fi
if [[ "$cmd" == "printf BEM_REMOTE_OK" ]]; then
  case "$host" in
    host-a|host-c)
      printf 'BEM_REMOTE_OK'
      ;;
    *)
      exit 255
      ;;
  esac
  exit 0
fi
if [[ "$cmd" == *"--query-gpu=index --format=csv,noheader,nounits"* ]]; then
  case "$host" in
    host-a|host-c)
      printf '0\n1\n'
      ;;
    *)
      exit 255
      ;;
  esac
  exit 0
fi
if [[ "$cmd" == nvidia-smi* || "$cmd" == custom-smi* ]]; then
  case "$host" in
    host-a)
      echo "0, NVIDIA Test GPU, 42, 55.0, 100, 0"
      ;;
    host-b)
      echo "0, NVIDIA Test GPU, 66, 210.0, 512, 100"
      ;;
    host-c)
      echo "0, NVIDIA Test GPU, 45, 60.0, 200, 0"
      ;;
    host-missing-bin)
      echo "0, NVIDIA Test GPU, 45, 60.0, 200, 0"
      ;;
    host-compute)
      echo "0, NVIDIA Test GPU, 43, 60.0, 200, 0"
      ;;
    *)
      exit 255
      ;;
  esac
  exit 0
fi
if [[ "$cmd" == for\\ d* || "$cmd" == test\\ -f* ]]; then
  echo "/home/fake/BEM-CUDA"
  exit 0
fi
if [[ "$cmd" == *"resume_accuracy_matrix_cases.sh --run"* ]]; then
  case_name="${cmd##*--cases }"
  case_name="${case_name%% *}"
  echo "REMOTE_CMD $cmd"
  echo "STARTED gpu=0 case=$case_name pid=4242"
  exit 0
fi
if [[ "$cmd" == *bin/bem_cuda_fmm* ]]; then
  if [[ "$host" == "host-missing-bin" ]]; then
    exit 1
  fi
  exit 0
fi
echo "REMOTE_EXEC host=$host cmd=$cmd"
"""


def run(args, fake_ssh: Path, extra_env=None):
    env = os.environ.copy()
    env["BEM_REMOTE_RESUME_SSH"] = str(fake_ssh)
    env["BEM_NVIDIA_SMI"] = "custom-smi"
    env["BEM_REMOTE_RESUME_SCAN_HOSTS"] = "0"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=str(ROOT),
        env=env,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        fake_ssh = Path(tmp) / "fake_ssh.sh"
        fake_ssh.write_text(FAKE_SSH)
        fake_ssh.chmod(fake_ssh.stat().st_mode | stat.S_IXUSR)
        fake_rsync = Path(tmp) / "fake_rsync.sh"
        fake_rsync.write_text("""#!/usr/bin/env bash
set -euo pipefail
echo "FAKE_RSYNC host=$1 repo=$2"
""")
        fake_rsync.chmod(fake_rsync.stat().st_mode | stat.S_IXUSR)

        proc = run([
            "--hosts", "host-a,host-b,host-c",
            "--gpus", "0",
            "--cases", f"{CASE_A},{CASE_B},{CASE_C}",
            "--max-jobs", "3",
        ], fake_ssh)
        assert proc.returncode == 0, proc.stdout
        assert "REMOTE_GPU_BUSY host=host-b" in proc.stdout, proc.stdout
        assert "REMOTE_RESUME cases=3 leased_cases=0 idle_remote_gpus=2 usable_remote_gpus=2" in proc.stdout, proc.stdout
        assert f"REMOTE_DRYRUN host=host-a gpu=0 case={CASE_A}" in proc.stdout, proc.stdout
        assert f"REMOTE_DRYRUN host=host-c gpu=0 case={CASE_B}" in proc.stdout, proc.stdout
        assert f"case={CASE_C}" not in proc.stdout, proc.stdout
        assert "REMOTE_RESUME selected=2" in proc.stdout, proc.stdout

        proc = run([
            "--hosts", "host-a,host-c",
            "--gpus", "0",
            "--cases", f"{CASE_A},{CASE_A}",
            "--max-jobs", "2",
        ], fake_ssh)
        assert proc.returncode == 0, proc.stdout
        assert "REMOTE_RESUME cases=1" in proc.stdout, proc.stdout
        assert proc.stdout.count(f"case={CASE_A}") == 1, proc.stdout
        assert "REMOTE_DRYRUN host=host-c" not in proc.stdout, proc.stdout

        proc = run([
            "--hosts", "host-a,host-compute",
            "--gpus", "0",
            "--cases", f"{CASE_A},{CASE_B}",
            "--max-jobs", "2",
        ], fake_ssh)
        assert proc.returncode == 0, proc.stdout
        assert "REMOTE_GPU_BUSY host=host-compute gpu=0 compute_apps=4242, ./mbs_po_gpu_float_fast, 304" in proc.stdout, proc.stdout
        assert "REMOTE_RESUME cases=2 leased_cases=0 idle_remote_gpus=1 usable_remote_gpus=1" in proc.stdout, proc.stdout
        assert f"REMOTE_DRYRUN host=host-a gpu=0 case={CASE_A}" in proc.stdout, proc.stdout
        assert f"case={CASE_B}" not in proc.stdout, proc.stdout

        proc = run([
            "--hosts", "host-a,host-c",
            "--gpus", "auto",
            "--cases", f"{CASE_A},{CASE_B},{CASE_C}",
            "--max-jobs", "3",
        ], fake_ssh)
        assert proc.returncode == 0, proc.stdout
        assert "REMOTE_RESUME cases=3 leased_cases=0 idle_remote_gpus=4 usable_remote_gpus=4" in proc.stdout, proc.stdout
        assert f"REMOTE_DRYRUN host=host-a gpu=0 case={CASE_A}" in proc.stdout, proc.stdout
        assert f"REMOTE_DRYRUN host=host-a gpu=1 case={CASE_B}" in proc.stdout, proc.stdout
        assert f"REMOTE_DRYRUN host=host-c gpu=0 case={CASE_C}" in proc.stdout, proc.stdout

        proc = run([
            "--hosts", "host-a",
            "--gpus", "0 1",
            "--cases", f"{CASE_A},{CASE_B}",
            "--max-jobs", "2",
            "--sync-launchers",
        ], fake_ssh, {"BEM_REMOTE_RESUME_RSYNC": str(fake_rsync)})
        assert proc.returncode == 0, proc.stdout
        assert proc.stdout.count("FAKE_RSYNC host=host-a repo=/home/fake/BEM-CUDA") == 1, proc.stdout
        assert f"REMOTE_DRYRUN host=host-a gpu=0 case={CASE_A}" in proc.stdout, proc.stdout
        assert f"REMOTE_DRYRUN host=host-a gpu=1 case={CASE_B}" in proc.stdout, proc.stdout

        proc = run([
            "--hosts", "auto",
            "--gpus", "0",
            "--cases", f"{CASE_A},{CASE_B}",
            "--max-jobs", "2",
        ], fake_ssh, {"BEM_REMOTE_RESUME_AUTO_HOSTS": "host-a host-b host-c"})
        assert proc.returncode == 0, proc.stdout
        assert "REMOTE_HOST_AUTO hosts=host-a host-c" in proc.stdout, proc.stdout
        assert "REMOTE_RESUME cases=2 leased_cases=0 idle_remote_gpus=2 usable_remote_gpus=2" in proc.stdout, proc.stdout
        assert f"REMOTE_DRYRUN host=host-a gpu=0 case={CASE_A}" in proc.stdout, proc.stdout
        assert f"REMOTE_DRYRUN host=host-c gpu=0 case={CASE_B}" in proc.stdout, proc.stdout

        proc = run([
            "--hosts", "auto",
            "--gpus", "0",
            "--cases", f"{CASE_A},{CASE_B}",
            "--max-jobs", "2",
        ], fake_ssh, {
            "BEM_REMOTE_RESUME_AUTO_HOSTS": "host-b",
            "BEM_REMOTE_RESUME_SCAN_HOSTS": "1",
            "BEM_REMOTE_RESUME_SCAN_OUTPUT": "host-a host-c",
        })
        assert proc.returncode == 0, proc.stdout
        assert "REMOTE_HOST_AUTO hosts=host-a host-c" in proc.stdout, proc.stdout
        assert "REMOTE_RESUME cases=2 leased_cases=0 idle_remote_gpus=2 usable_remote_gpus=2" in proc.stdout, proc.stdout

        proc = run([
            "--run",
            "--hosts", "host-a,host-c",
            "--gpus", "0",
            "--cases", f"{CASE_A},{CASE_B}",
            "--max-jobs", "2",
            "--out", str(Path(tmp) / "remote_run_out"),
            "--case-max-power", "200",
        ], fake_ssh)
        assert proc.returncode == 0, proc.stdout
        assert f"REMOTE_START host=host-a gpu=0 case={CASE_A}" in proc.stdout, proc.stdout
        assert f"REMOTE_START host=host-c gpu=0 case={CASE_B}" in proc.stdout, proc.stdout
        assert "--case-max-power 200" in proc.stdout.replace("\\ ", " "), proc.stdout
        normalized = proc.stdout.replace("\\ ", " ")
        assert "env BEM_NO_AUTO_MGPU=1 bash scripts/resume_accuracy_matrix_cases.sh --run" in normalized, proc.stdout
        assert f"--gpus 0 --max-jobs 1 --cases {CASE_A}" in normalized, proc.stdout
        assert f"--gpus 0 --max-jobs 1 --cases {CASE_B}" in normalized, proc.stdout

        lease_out = Path(tmp) / "remote_lease_out"
        proc = run([
            "--run",
            "--hosts", "host-a,host-c",
            "--gpus", "0",
            "--cases", f"{CASE_A},{CASE_B}",
            "--max-jobs", "2",
            "--out", str(lease_out),
        ], fake_ssh)
        assert proc.returncode == 0, proc.stdout
        assert f"REMOTE_CASE_LEASE case={CASE_A} host=host-a gpu=0" in proc.stdout, proc.stdout
        assert f"REMOTE_CASE_LEASE case={CASE_B} host=host-c gpu=0" in proc.stdout, proc.stdout
        assert (lease_out / "remote_case_leases" / f"{CASE_A}.lock" / "owner").is_file(), proc.stdout
        proc = run([
            "--run",
            "--hosts", "host-a,host-c",
            "--gpus", "0",
            "--cases", f"{CASE_A},{CASE_B}",
            "--max-jobs", "2",
            "--out", str(lease_out),
        ], fake_ssh)
        assert proc.returncode == 0, proc.stdout
        assert f"REMOTE_CASE_LEASE_SKIP case={CASE_A}" in proc.stdout, proc.stdout
        assert f"REMOTE_CASE_LEASE_SKIP case={CASE_B}" in proc.stdout, proc.stdout
        assert "REMOTE_RESUME cases=0 leased_cases=2" in proc.stdout, proc.stdout
        assert "REMOTE_START host=" not in proc.stdout, proc.stdout
        assert "REMOTE_RESUME selected=0" in proc.stdout, proc.stdout

        proc = run([
            "--run",
            "--hosts", "host-a",
            "--gpus", "0",
            "--cases", CASE_A,
            "--max-jobs", "1",
            "--out", str(lease_out),
        ], fake_ssh, {
            "BEM_REMOTE_RESUME_STATUS_TEXT": f"CURRENT {CASE_A}\nSUMMARY current=1 stale=0 missing=0 total=1\n",
        })
        assert proc.returncode == 0, proc.stdout
        assert f"REMOTE_CASE_LEASE_DONE case={CASE_A}" in proc.stdout, proc.stdout
        assert "REMOTE_RESUME cases=1 leased_cases=0" in proc.stdout, proc.stdout
        assert f"REMOTE_START host=host-a gpu=0 case={CASE_A}" in proc.stdout, proc.stdout

        proc = run([
            "--hosts", "host-missing-bin",
            "--gpus", "0",
            "--cases", CASE_A,
            "--max-jobs", "1",
        ], fake_ssh)
        assert proc.returncode == 3, proc.stdout
        assert "REMOTE_BINARY_SKIP host=host-missing-bin" in proc.stdout, proc.stdout
        assert "usable_remote_gpus=0" in proc.stdout, proc.stdout

        proc = run([
            "--hosts", "host-a",
            "--gpus", "0",
            "--cases", LEGACY_DUST,
            "--max-jobs", "1",
        ], fake_ssh)
        assert proc.returncode == 1, proc.stdout
        assert f"CASE_INVALID case={LEGACY_DUST}" in proc.stdout, proc.stdout
        assert "legacy dust case disabled" in proc.stdout, proc.stdout

    print("remote resume accuracy matrix cases: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
