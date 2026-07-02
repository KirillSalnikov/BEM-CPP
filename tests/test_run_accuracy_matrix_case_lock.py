#!/usr/bin/env python3
"""Locking tests for one-case production launcher."""

import fcntl
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_accuracy_matrix_case.sh"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        lock_dir = out / "locks"
        lock_dir.mkdir(parents=True)
        lock_file = lock_dir / "gpu_0.lock"
        with lock_file.open("w") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            proc = subprocess.run(
                [
                    str(SCRIPT),
                    "--gpu", "0",
                    "--case", "hex_ka30_ref5_balanced_q7_d5_tol1e3",
                    "--out", str(out),
                    "--bin", "/bin/true",
                ],
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        assert proc.returncode == 3, proc.stdout
        assert f"GPU_LOCK active: {lock_file}" in proc.stdout, proc.stdout
        assert not (out / "logs" / "hex_ka30_ref5_balanced_q7_d5_tol1e3.log").exists()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        fake_nvidia_smi = Path(tmp) / "nvidia-smi"
        fake_nvidia_smi.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == "-i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits" ]]; then
  echo "4242, ./mbs_po_gpu_float_fast, 304"
  exit 0
fi
echo "unexpected fake nvidia-smi args: $*" >&2
exit 1
""")
        fake_nvidia_smi.chmod(0o755)
        env = {**os.environ, "BEM_NVIDIA_SMI": str(fake_nvidia_smi)}
        proc = subprocess.run(
            [
                str(SCRIPT),
                "--gpu", "0",
                "--case", "hex_ka30_ref5_balanced_q7_d5_tol1e3",
                "--out", str(out),
                "--bin", "/bin/true",
            ],
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert proc.returncode == 3, proc.stdout
        assert "GPU_BUSY gpu=0 compute_apps=4242, ./mbs_po_gpu_float_fast, 304" in proc.stdout, proc.stdout
        assert not (out / "logs" / "hex_ka30_ref5_balanced_q7_d5_tol1e3.log").exists()

        proc = subprocess.run(
            [
                str(SCRIPT),
                "--gpu", "0",
                "--case", "hex_ka30_ref5_balanced_q7_d5_tol1e3",
                "--out", str(out),
                "--bin", "/bin/true",
                "--allow-compute-share",
            ],
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert proc.returncode != 3, proc.stdout
        assert "GPU_BUSY gpu=0 compute_apps=" not in proc.stdout, proc.stdout

    print("run accuracy matrix case lock: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
