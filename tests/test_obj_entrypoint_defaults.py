#!/usr/bin/env python3
"""Smoke-tests for OBJ comparison entrypoint defaults."""

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_obj_adda_compare.py"


def run_obj_compare(tmp: Path, *extra: str) -> subprocess.CompletedProcess:
    adda = tmp / "adda" / "orient_000"
    adda.mkdir(parents=True, exist_ok=True)
    (adda / "log").write_text("dummy\n")
    return subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--ka", "20",
            "--adda", str(tmp / "adda"),
            "--obj", "dust.obj",
            "--out", str(tmp / "out"),
            "--gpus", "0",
            "--dry-run",
            *extra,
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def write_fake_bem(path: Path) -> None:
    path.write_text("""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*"
out=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[[ -n "$out" ]] || exit 3
cat > "$out" <<'JSON'
{
  "theta": [0.0],
  "mueller": [[[[1.0],[0.0],[0.0],[0.0]],[[0.0],[0.0],[0.0],[0.0]],[[0.0],[0.0],[0.0],[0.0]],[[0.0],[0.0],[0.0],[0.0]]]],
  "timing": {"assembly_s": 0.0, "solve_s": 0.0, "farfield_s": 0.0, "total_s": 0.0}
}
JSON
""")
    path.chmod(0o755)


def run_wrapper(script: str, tmp: Path, *extra: str, env=None) -> subprocess.CompletedProcess:
    fake_bem = tmp / "fake_bem.sh"
    write_fake_bem(fake_bem)
    run_env = None
    if env is not None:
        import os
        run_env = {**os.environ, **env}
    return subprocess.run(
        [
            "python3",
            str(ROOT / script),
            "--exe", str(fake_bem),
            "--out", str(tmp / f"{Path(script).stem}.json"),
            "--work-dir", str(tmp / f"{Path(script).stem}_parts"),
            "--gpus", "0",
            "--ka", "20",
            "--shape", "obj",
            "--obj", "dust.obj",
            "--orient", "1", "1", "1",
            "--ntheta", "1",
            *extra,
        ],
        cwd=str(ROOT),
        env=run_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def wrapper_log(tmp: Path, script: str) -> str:
    part_dir = tmp / f"{Path(script).stem}_parts"
    logs = sorted(part_dir.glob("part_*.log"))
    assert logs, f"no logs in {part_dir}"
    return "\n".join(path.read_text() for path in logs)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        proc = run_obj_compare(tmp)
        assert proc.returncode == 0, proc.stdout
        assert "--accurate" in proc.stdout, proc.stdout
        assert "--fmm-digits 3" not in proc.stdout, proc.stdout
        assert "--gmres-tol 2e-2" not in proc.stdout, proc.stdout
        assert "--system pmchwt" not in proc.stdout, proc.stdout

        proc = run_obj_compare(tmp, "--fast-obj")
        assert proc.returncode == 0, proc.stdout
        assert "--accurate" not in proc.stdout, proc.stdout
        assert "--fmm-digits 3" in proc.stdout, proc.stdout
        assert "--gmres-tol 2e-2" in proc.stdout, proc.stdout
        assert "--system pmchwt" in proc.stdout, proc.stdout

        proc = run_wrapper("run_orient_mgpu.py", tmp)
        assert proc.returncode == 0, proc.stdout
        log = wrapper_log(tmp, "run_orient_mgpu.py")
        assert "--accurate" in log, log
        assert "--quad 4" not in log, log

        proc = run_wrapper("run_orient_mgpu.py", tmp, "--fast-obj")
        assert proc.returncode == 0, proc.stdout
        log = wrapper_log(tmp, "run_orient_mgpu.py")
        assert "--accurate" not in log, log

        proc = run_wrapper("run_orient_queue.py", tmp, "--chunk-size", "1")
        assert proc.returncode == 0, proc.stdout
        log = wrapper_log(tmp, "run_orient_queue.py")
        assert "--accurate" in log, log
        assert "--quad 4" not in log, log

        proc = run_wrapper("run_orient_queue.py", tmp, "--chunk-size", "1", "--fast-obj")
        assert proc.returncode == 0, proc.stdout
        log = wrapper_log(tmp, "run_orient_queue.py")
        assert "--accurate" not in log, log

        fake_smi = tmp / "nvidia-smi"
        fake_smi.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == "-i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits" ]]; then
  printf '4242, ./mbs_po_gpu_float_fast, 304\\n'
  exit 0
fi
exit 0
""")
        fake_smi.chmod(0o755)
        env = {"BEM_NVIDIA_SMI": str(fake_smi)}
        proc = run_wrapper("run_orient_mgpu.py", tmp, env=env)
        assert proc.returncode != 0, proc.stdout
        assert "GPU_BUSY gpu=0 compute_apps=4242, ./mbs_po_gpu_float_fast, 304" in proc.stdout, proc.stdout
        proc = run_wrapper("run_orient_queue.py", tmp, "--chunk-size", "1", env=env)
        assert proc.returncode != 0, proc.stdout
        assert "GPU_BUSY gpu=0 compute_apps=4242, ./mbs_po_gpu_float_fast, 304" in proc.stdout, proc.stdout
        proc = run_wrapper("run_orient_mgpu.py", tmp, "--allow-compute-share", env=env)
        assert proc.returncode == 0, proc.stdout

    orient_queue = (ROOT / "scripts" / "run_greek_orientation_convergence_queue.sh").read_text()
    assert "--accurate" in orient_queue
    assert "--fmm-digits 6" in orient_queue
    assert "--gmres-tol 5e-4" in orient_queue
    assert "--gmres-restart 500" in orient_queue

    memory_queue = (ROOT / "scripts" / "run_fig7_memory_queue.sh").read_text()
    assert "dust_common=(--ri 1.6 0.002 --accurate" in memory_queue
    assert "--fmm-digits 6" in memory_queue
    assert "--gmres-tol 5e-4" in memory_queue

    print("obj entrypoint defaults: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
