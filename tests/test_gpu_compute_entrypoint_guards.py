#!/usr/bin/env python3
"""Smoke-tests for direct GPU entrypoint compute-process guards."""

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fake_smi(path: Path, *, busy_gpus=("0",), gpu_indices=("0", "1")) -> Path:
    busy_case = "\n".join(
        f"""if [[ "$*" == "-i {gpu} --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits" ]]; then
  printf '4242, ./mbs_po_gpu_float_fast, 304\\n'
  exit 0
fi"""
        for gpu in busy_gpus
    )
    indices = "\\n".join(gpu_indices)
    path.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == "--query-gpu=index --format=csv,noheader,nounits" ]]; then
  printf '{indices}\\n'
  exit 0
fi
{busy_case}
exit 0
""")
    path.chmod(0o755)
    return path


def run(cmd, tmp: Path, env=None) -> subprocess.CompletedProcess:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        smi = fake_smi(tmp / "nvidia-smi")
        env = {"BEM_NVIDIA_SMI": str(smi)}

        proc = run(
            [
                "python3",
                str(ROOT / "scripts" / "run_obj_adda_compare.py"),
                "--ka", "20",
                "--adda", str(tmp / "adda"),
                "--obj", "dust.obj",
                "--out", str(tmp / "obj_out"),
                "--gpus", "0",
                "--dry-run",
            ],
            tmp,
            env=env,
        )
        assert proc.returncode != 0, proc.stdout
        assert "GPU_BUSY gpu=0: 4242, ./mbs_po_gpu_float_fast, 304" in proc.stdout, proc.stdout

        proc = run(
            [
                "python3",
                str(ROOT / "scripts" / "run_hex_adda_compare.py"),
                "--ka", "20",
                "--adda", str(tmp / "adda"),
                "--out", str(tmp / "hex_out"),
                "--gpus", "0",
                "--dry-run",
            ],
            tmp,
            env=env,
        )
        assert proc.returncode != 0, proc.stdout
        assert "GPU_BUSY gpu=0: 4242, ./mbs_po_gpu_float_fast, 304" in proc.stdout, proc.stdout

        proc = run(
            [
                "python3",
                str(ROOT / "scripts" / "run_hex_euler_scaling_benchmark.py"),
                "--gpus", "auto",
                "--levels", "bad",
                "--bem-only",
                "--out", str(tmp / "euler_out"),
            ],
            tmp,
            env={"BEM_NVIDIA_SMI": str(fake_smi(tmp / "all_busy-smi", busy_gpus=("0", "1")))},
        )
        assert proc.returncode != 0, proc.stdout
        assert "GPU_BUSY gpu=0 compute_apps=4242, ./mbs_po_gpu_float_fast, 304" in proc.stdout, proc.stdout
        assert "GPU_BUSY gpu=1 compute_apps=4242, ./mbs_po_gpu_float_fast, 304" in proc.stdout, proc.stdout
        assert "no free GPUs from --gpus auto" in proc.stdout, proc.stdout

        proc = run(
            [
                "python3",
                str(ROOT / "scripts" / "run_hex_euler_scaling_benchmark.py"),
                "--gpus", "0",
                "--allow-compute-share",
                "--levels", "bad",
                "--bem-only",
                "--out", str(tmp / "euler_allowed"),
            ],
            tmp,
            env=env,
        )
        assert proc.returncode != 0, proc.stdout
        assert "bad level 'bad'" in proc.stdout, proc.stdout
        assert "GPU_BUSY" not in proc.stdout, proc.stdout

        fake_adda = tmp / "adda_ocl"
        fake_adda.write_text("#!/usr/bin/env bash\nexit 0\n")
        fake_adda.chmod(0o755)
        fake_clinfo = tmp / "clinfo"
        fake_clinfo.write_text("#!/usr/bin/env bash\nprintf 'Number of platforms 1\\n'\n")
        fake_clinfo.chmod(0o755)
        proc = run(
            [
                "bash",
                str(ROOT / "scripts" / "run_adda_ocl_sphere_ri_sweep.sh"),
            ],
            tmp,
            env={
                "BEM_NVIDIA_SMI": str(fake_smi(tmp / "busy_pair-smi", busy_gpus=("0", "1"))),
                "ADDA_OCL": str(fake_adda),
                "OUT": str(tmp / "adda_sweep"),
                "GPUS_CSV": "0,1",
            },
        )
        assert proc.returncode == 3, proc.stdout
        assert "GPU_BUSY gpu=0 compute_apps=4242, ./mbs_po_gpu_float_fast, 304" in proc.stdout, proc.stdout
        assert "No free GPUs from GPUS_CSV=0,1" in proc.stdout, proc.stdout

        proc = run(
            [
                "bash",
                str(ROOT / "scripts" / "run_adda_ocl_benchmark.sh"),
            ],
            tmp,
            env={
                "BEM_NVIDIA_SMI": str(smi),
                "ADDA_OCL": "/bin/true",
                "OUT": str(tmp / "adda_bench"),
                "GPU": "0",
                "PATH": f"{tmp}:{os.environ['PATH']}",
            },
        )
        assert proc.returncode == 3, proc.stdout
        assert "GPU_BUSY gpu=0 compute_apps=4242, ./mbs_po_gpu_float_fast, 304" in proc.stdout, proc.stdout

    print("gpu compute entrypoint guards: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
