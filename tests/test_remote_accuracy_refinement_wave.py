#!/usr/bin/env python3
"""Smoke-tests for the remote accuracy refinement wave wrapper."""

import os
import json
import stat
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "remote_accuracy_refinement_wave.sh"
SUPERVISOR = ROOT / "scripts" / "run_remote_refinement_queue_supervisor.sh"
START_SUPERVISOR = ROOT / "scripts" / "start_remote_refinement_queue_supervisor.sh"
QUEUE_STATUS = ROOT / "scripts" / "remote_refinement_queue_status.py"
PRODUCTION_CSV = ROOT / "poster_a0" / "assets" / "table_accuracy_matrix_15.csv"


FAKE_SSH = """#!/usr/bin/env bash
set -euo pipefail
host="$1"
cmd="$2"
if [[ "$cmd" == *"--query-compute-apps="* ]]; then
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
  printf '0\n1\n'
  exit 0
fi
if [[ "$cmd" == custom-smi* ]]; then
  case "$host" in
    host-a|host-b|host-c|host-d)
      echo "0, NVIDIA Test GPU, 42, 55.0, 100, 0"
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
  exit 0
fi
echo "REMOTE_EXEC host=$host cmd=$cmd"
"""


FAKE_SSH_BUSY_THEN_FREE = """#!/usr/bin/env bash
set -euo pipefail
host="$1"
cmd="$2"
state_dir="${BEM_FAKE_SSH_STATE_DIR:?}"
counter="$state_dir/gpu_query_count"
if [[ "$cmd" == *"--query-compute-apps="* ]]; then
  exit 0
fi
if [[ "$cmd" == *"--query-gpu=index --format=csv,noheader,nounits"* ]]; then
  printf '0\n'
  exit 0
fi
if [[ "$cmd" == custom-smi* ]]; then
  count=0
  if [[ -f "$counter" ]]; then
    count="$(cat "$counter")"
  fi
  count=$((count + 1))
  printf '%s\n' "$count" > "$counter"
  if (( count == 1 )); then
    echo "0, NVIDIA Test GPU, 60, 210.0, 300, 100"
  else
    echo "0, NVIDIA Test GPU, 42, 55.0, 100, 0"
  fi
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
  exit 0
fi
echo "REMOTE_EXEC host=$host cmd=$cmd"
"""


FAKE_SSH_ALWAYS_BUSY = """#!/usr/bin/env bash
set -euo pipefail
host="$1"
cmd="$2"
if [[ "$cmd" == *"--query-compute-apps="* ]]; then
  exit 0
fi
if [[ "$cmd" == *"--query-gpu=index --format=csv,noheader,nounits"* ]]; then
  printf '0\n'
  exit 0
fi
if [[ "$cmd" == custom-smi* ]]; then
  echo "0, NVIDIA Test GPU, 61, 211.0, 349, 100"
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
  exit 0
fi
echo "REMOTE_EXEC host=$host cmd=$cmd"
"""


FAKE_SSH_START_ONCE = """#!/usr/bin/env bash
set -euo pipefail
host="$1"
cmd="$2"
state_dir="${BEM_FAKE_SSH_STATE_DIR:?}"
started_file="$state_dir/started_count"
if [[ "$cmd" == *"--query-compute-apps="* ]]; then
  exit 0
fi
if [[ "$cmd" == *"--query-gpu=index --format=csv,noheader,nounits"* ]]; then
  printf '0\n'
  exit 0
fi
if [[ "$cmd" == custom-smi* ]]; then
  echo "0, NVIDIA Test GPU, 42, 55.0, 100, 0"
  exit 0
fi
if [[ "$cmd" == for\\ d* || "$cmd" == test\\ -f* ]]; then
  echo "/home/fake/BEM-CUDA"
  exit 0
fi
if [[ "$cmd" == *"resume_accuracy_matrix_cases.sh --run"* ]]; then
  count=0
  if [[ -f "$started_file" ]]; then
    count="$(cat "$started_file")"
  fi
  count=$((count + 1))
  printf '%s\n' "$count" > "$started_file"
  if (( count == 1 )); then
    case_name="${cmd##*--cases }"
    case_name="${case_name%% *}"
    echo "REMOTE_CMD $cmd"
    echo "STARTED gpu=0 case=$case_name pid=4242"
  else
    echo "RESUME pending=0 idle_gpus=1 mode=run"
    echo "RESUME selected=0"
  fi
  exit 0
fi
if [[ "$cmd" == *bin/bem_cuda_fmm* ]]; then
  exit 0
fi
echo "REMOTE_EXEC host=$host cmd=$cmd"
"""


def run(args, fake_ssh: Path, extra_env=None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["BEM_REMOTE_RESUME_SSH"] = str(fake_ssh)
    env["BEM_NVIDIA_SMI"] = "custom-smi"
    env["BEM_REMOTE_RESUME_AUTO_HOSTS"] = "host-a host-b host-c"
    env["BEM_REMOTE_RESUME_SCAN_HOSTS"] = "0"
    if extra_env:
        env.update(extra_env)
    fake_rsync = fake_ssh.parent / "fake_rsync.sh"
    if fake_rsync.is_file():
        env["BEM_REMOTE_RESUME_RSYNC"] = str(fake_rsync)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=str(ROOT),
        env=env,
        text=True,
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
        plan_csv = Path(tmp) / "remote_plan.csv"
        complete_csv = Path(tmp) / "complete_accuracy.csv"
        complete_csv.write_text(
            "shape,ka,mesh_label,status,metadata_status,operator_status,gate_error,raw_pass10,pass10\n"
            "сфера,5,ref4,PASS,ok,not_required,0.01,True,True\n"
        )

        empty_status_json = Path(tmp) / "remote_empty_status.json"
        empty_proc = run([
            "--csv", str(complete_csv),
            "--hosts", "host-a",
            "--gpus", "0",
            "--status-json", str(empty_status_json),
            "--plan-csv", str(Path(tmp) / "remote_empty_plan.csv"),
        ], fake_ssh)
        assert empty_proc.returncode == 0, empty_proc.stdout
        assert "REFINE threshold=0.1 reason=all planned=0" in empty_proc.stdout, empty_proc.stdout
        assert empty_status_json.is_file(), empty_proc.stdout
        empty_status = json.loads(empty_status_json.read_text())
        assert empty_status["planned_cases"] == 0, empty_status
        assert empty_status["cases"] == [], empty_status
        assert empty_status["remote_rc"] == 0, empty_status
        assert empty_status["plan_failed"] is False, empty_status
        assert empty_status["selected"] == 0, empty_status

        plan_fail_status_json = Path(tmp) / "remote_plan_fail_status.json"
        plan_fail_proc = run([
            "--csv", str(Path(tmp) / "missing_accuracy.csv"),
            "--hosts", "host-a",
            "--gpus", "0",
            "--status-json", str(plan_fail_status_json),
            "--plan-csv", str(Path(tmp) / "remote_plan_fail.csv"),
        ], fake_ssh)
        assert plan_fail_proc.returncode != 0, plan_fail_proc.stdout
        assert plan_fail_status_json.is_file(), plan_fail_proc.stdout
        plan_fail_status = json.loads(plan_fail_status_json.read_text())
        assert plan_fail_status["plan_failed"] is True, plan_fail_status
        assert plan_fail_status["planned_cases"] == 0, plan_fail_status
        assert plan_fail_status["cases"] == [], plan_fail_status
        assert plan_fail_status["remote_rc"] == plan_fail_proc.returncode, plan_fail_status

        proc = run([
            "--csv", str(PRODUCTION_CSV),
            "--hosts", "host-a,host-b,host-c,host-d",
            "--gpus", "0",
            "--max-cases", "4",
            "--plan-csv", str(plan_csv),
        ], fake_ssh)
        assert proc.returncode == 0, proc.stdout
        assert "REMOTE_REFINEMENT_WAVE mode=dry-run" in proc.stdout, proc.stdout
        assert "REFINE threshold=0.1 reason=all planned=4 limit=4" in proc.stdout, proc.stdout
        assert "remote_command:" in proc.stdout, proc.stdout
        assert "REMOTE_RESUME cases=4 leased_cases=0 idle_remote_gpus=4 usable_remote_gpus=4 mode=dry-run" in proc.stdout, proc.stdout
        assert "REMOTE_DRYRUN host=host-a gpu=0 case=dust_ka30_gmsh7000_balanced_q9_d6_tol5e4" in proc.stdout, proc.stdout
        assert "REMOTE_DRYRUN host=host-b gpu=0 case=dust_ka20_gmsh5200_balanced_q9_d6_tol5e4" in proc.stdout, proc.stdout
        assert "REMOTE_DRYRUN host=host-c gpu=0 case=dust_ka15_gmsh7000_balanced_q9_d6_tol5e4" in proc.stdout, proc.stdout
        assert "REMOTE_DRYRUN host=host-d gpu=0 case=dust_ka10_gmsh6000_balanced_q9_d6_tol5e4" in proc.stdout, proc.stdout
        assert "STARTED" not in proc.stdout, proc.stdout
        assert plan_csv.is_file(), proc.stdout

        dryrun_status_json = Path(tmp) / "remote_dryrun_status.json"
        dryrun_status_proc = run([
            "--csv", str(PRODUCTION_CSV),
            "--hosts", "host-a",
            "--gpus", "0",
            "--max-cases", "1",
            "--status-json", str(dryrun_status_json),
            "--plan-csv", str(Path(tmp) / "remote_dryrun_status_plan.csv"),
        ], fake_ssh)
        assert dryrun_status_proc.returncode == 0, dryrun_status_proc.stdout
        assert dryrun_status_json.is_file(), dryrun_status_proc.stdout
        dryrun_status = json.loads(dryrun_status_json.read_text())
        assert dryrun_status["mode"] == "dry-run", dryrun_status
        assert dryrun_status["plan_failed"] is False, dryrun_status
        assert dryrun_status["remote_rc"] == 0, dryrun_status
        assert dryrun_status["usable_remote_gpus"] == 1, dryrun_status
        assert dryrun_status["min_free_gpus"] == 1, dryrun_status
        assert dryrun_status["enough_free_gpus"] is True, dryrun_status
        assert dryrun_status["selected"] == 1, dryrun_status

        run_proc = run([
            "--run",
            "--csv", str(PRODUCTION_CSV),
            "--hosts", "host-a,host-b",
            "--gpus", "0",
            "--max-cases", "2",
            "--out", str(Path(tmp) / "remote_run_out"),
            "--plan-csv", str(Path(tmp) / "remote_run_plan.csv"),
            "--case-max-power", "200",
        ], fake_ssh)
        assert run_proc.returncode == 0, run_proc.stdout
        assert "REMOTE_REFINEMENT_WAVE mode=run" in run_proc.stdout, run_proc.stdout
        assert "FAKE_RSYNC host=host-a repo=/home/fake/BEM-CUDA" in run_proc.stdout, run_proc.stdout
        assert "FAKE_RSYNC host=host-b repo=/home/fake/BEM-CUDA" in run_proc.stdout, run_proc.stdout
        assert "REMOTE_START host=host-a gpu=0 case=dust_ka30_gmsh7000_balanced_q9_d6_tol5e4" in run_proc.stdout
        assert "REMOTE_START host=host-b gpu=0 case=dust_ka20_gmsh5200_balanced_q9_d6_tol5e4" in run_proc.stdout
        assert "--case-max-power 200" in run_proc.stdout.replace("\\ ", " "), run_proc.stdout

        run_status_json = Path(tmp) / "remote_run_status.json"
        run_status_proc = run([
            "--run",
            "--csv", str(PRODUCTION_CSV),
            "--hosts", "host-a",
            "--gpus", "0",
            "--max-cases", "1",
            "--out", str(Path(tmp) / "remote_run_status_out"),
            "--status-json", str(run_status_json),
            "--plan-csv", str(Path(tmp) / "remote_run_status_plan.csv"),
            "--no-sync-launchers",
        ], fake_ssh)
        assert run_status_proc.returncode == 0, run_status_proc.stdout
        assert run_status_json.is_file(), run_status_proc.stdout
        run_status = json.loads(run_status_json.read_text())
        assert run_status["mode"] == "run", run_status
        assert run_status["remote_rc"] == 0, run_status
        assert run_status["usable_remote_gpus"] == 1, run_status
        assert run_status["min_free_gpus"] == 1, run_status
        assert run_status["enough_free_gpus"] is True, run_status
        assert run_status["selected"] == 1, run_status

        multi_slot = run([
            "--csv", str(PRODUCTION_CSV),
            "--hosts", "host-a,host-b",
            "--gpus", "0 1",
            "--plan-csv", str(Path(tmp) / "remote_multi_slot_plan.csv"),
        ], fake_ssh)
        assert multi_slot.returncode == 0, multi_slot.stdout
        assert "REFINE threshold=0.1 reason=all planned=4 limit=4" in multi_slot.stdout, multi_slot.stdout
        assert "REMOTE_RESUME cases=4 leased_cases=0 idle_remote_gpus=4 usable_remote_gpus=4 mode=dry-run" in multi_slot.stdout
        assert "REMOTE_DRYRUN host=host-a gpu=0 case=dust_ka30_gmsh7000_balanced_q9_d6_tol5e4" in multi_slot.stdout
        assert "REMOTE_DRYRUN host=host-a gpu=1 case=dust_ka20_gmsh5200_balanced_q9_d6_tol5e4" in multi_slot.stdout
        assert "REMOTE_DRYRUN host=host-b gpu=0 case=dust_ka15_gmsh7000_balanced_q9_d6_tol5e4" in multi_slot.stdout
        assert "REMOTE_DRYRUN host=host-b gpu=1 case=dust_ka10_gmsh6000_balanced_q9_d6_tol5e4" in multi_slot.stdout
        assert "FAKE_RSYNC" not in multi_slot.stdout, multi_slot.stdout

        auto_gpu = run([
            "--csv", str(PRODUCTION_CSV),
            "--hosts", "host-a,host-b",
            "--gpus", "auto",
            "--plan-csv", str(Path(tmp) / "remote_auto_plan.csv"),
        ], fake_ssh)
        assert auto_gpu.returncode == 0, auto_gpu.stdout
        assert "REFINE threshold=0.1 reason=all planned=15 limit=all" in auto_gpu.stdout, auto_gpu.stdout
        assert "REMOTE_RESUME cases=15 leased_cases=0 idle_remote_gpus=4 usable_remote_gpus=4 mode=dry-run" in auto_gpu.stdout
        assert "REMOTE_RESUME selected=4" in auto_gpu.stdout, auto_gpu.stdout
        assert "FAKE_RSYNC" not in auto_gpu.stdout, auto_gpu.stdout

        auto_host = run([
            "--csv", str(PRODUCTION_CSV),
            "--hosts", "auto",
            "--gpus", "0",
            "--plan-csv", str(Path(tmp) / "remote_auto_host_plan.csv"),
        ], fake_ssh)
        assert auto_host.returncode == 0, auto_host.stdout
        assert "REFINE threshold=0.1 reason=all planned=15 limit=all" in auto_host.stdout, auto_host.stdout
        assert "REMOTE_HOST_AUTO hosts=host-a host-c" in auto_host.stdout, auto_host.stdout
        assert "REMOTE_RESUME cases=15 leased_cases=0 idle_remote_gpus=2 usable_remote_gpus=2 mode=dry-run" in auto_host.stdout

        wait_state = Path(tmp) / "wait_state"
        wait_state.mkdir()
        fake_wait_ssh = Path(tmp) / "fake_wait_ssh.sh"
        fake_wait_ssh.write_text(FAKE_SSH_BUSY_THEN_FREE)
        fake_wait_ssh.chmod(fake_wait_ssh.stat().st_mode | stat.S_IXUSR)
        wait_proc = run([
            "--csv", str(PRODUCTION_CSV),
            "--hosts", "host-a",
            "--gpus", "0",
            "--max-cases", "1",
            "--wait-free",
            "--wait-interval", "0",
            "--wait-timeout", "5",
            "--plan-csv", str(Path(tmp) / "remote_wait_plan.csv"),
        ], fake_wait_ssh, {"BEM_FAKE_SSH_STATE_DIR": str(wait_state)})
        assert wait_proc.returncode == 0, wait_proc.stdout
        assert "REMOTE_GPU_BUSY host=host-a gpu=0" in wait_proc.stdout, wait_proc.stdout
        assert "REMOTE_WAIT no usable GPUs attempt=1" in wait_proc.stdout, wait_proc.stdout
        assert "REMOTE_RESUME cases=1 leased_cases=0 idle_remote_gpus=1 usable_remote_gpus=1 mode=dry-run" in wait_proc.stdout
        assert "REMOTE_DRYRUN host=host-a gpu=0 case=dust_ka30_gmsh7000_balanced_q9_d6_tol5e4" in wait_proc.stdout

        fake_busy_ssh = Path(tmp) / "fake_busy_ssh.sh"
        fake_busy_ssh.write_text(FAKE_SSH_ALWAYS_BUSY)
        fake_busy_ssh.chmod(fake_busy_ssh.stat().st_mode | stat.S_IXUSR)
        status_json = Path(tmp) / "remote_status.json"
        status_proc = run([
            "--status-only",
            "--csv", str(PRODUCTION_CSV),
            "--hosts", "host-a",
            "--gpus", "0",
            "--max-cases", "1",
            "--status-json", str(status_json),
            "--plan-csv", str(Path(tmp) / "remote_status_plan.csv"),
        ], fake_busy_ssh)
        assert status_proc.returncode == 0, status_proc.stdout
        assert "REMOTE_REFINEMENT_WAVE mode=status" in status_proc.stdout, status_proc.stdout
        assert "REMOTE_GPU_BUSY host=host-a gpu=0" in status_proc.stdout, status_proc.stdout
        assert "REMOTE_REFINEMENT_STATUS planned_cases=1 usable_remote_gpus=0 selected=0 remote_rc=3" in status_proc.stdout
        assert "REMOTE_START" not in status_proc.stdout, status_proc.stdout
        status = json.loads(status_json.read_text())
        assert status["mode"] == "status", status
        assert status["planned_cases"] == 1, status
        assert status["usable_remote_gpus"] == 0, status
        assert status["selected"] == 0, status
        assert status["remote_rc"] == 3, status
        assert status["cases"] == ["dust_ka30_gmsh7000_balanced_q9_d6_tol5e4"], status
        assert status["busy_gpus"] == [
            "REMOTE_GPU_BUSY host=host-a gpu=0 temp=61C util=100% mem=349MiB power=211W"
        ], status

        min_wait_state = Path(tmp) / "min_wait_state"
        min_wait_state.mkdir()
        min_wait_proc = run([
            "--run",
            "--csv", str(PRODUCTION_CSV),
            "--hosts", "host-a,host-b",
            "--gpus", "0",
            "--max-cases", "2",
            "--out", str(Path(tmp) / "remote_min_wait_out"),
            "--wait-free",
            "--min-free-gpus", "2",
            "--wait-interval", "0",
            "--wait-timeout", "5",
            "--plan-csv", str(Path(tmp) / "remote_min_wait_plan.csv"),
            "--status-json", str(Path(tmp) / "remote_min_wait_status.json"),
        ], fake_wait_ssh, {"BEM_FAKE_SSH_STATE_DIR": str(min_wait_state)})
        assert min_wait_proc.returncode == 0, min_wait_proc.stdout
        assert "REMOTE_WAIT usable_remote_gpus=1 min_free_gpus=2 attempt=1" in min_wait_proc.stdout
        assert "REMOTE_RESUME cases=2 leased_cases=0 idle_remote_gpus=2 usable_remote_gpus=2 mode=dry-run" in min_wait_proc.stdout
        assert "REMOTE_START host=host-a gpu=0 case=dust_ka30_gmsh7000_balanced_q9_d6_tol5e4" in min_wait_proc.stdout
        assert "REMOTE_START host=host-b gpu=0 case=dust_ka20_gmsh5200_balanced_q9_d6_tol5e4" in min_wait_proc.stdout
        min_wait_status = json.loads((Path(tmp) / "remote_min_wait_status.json").read_text())
        assert min_wait_status["min_free_gpus"] == 2, min_wait_status
        assert min_wait_status["usable_remote_gpus"] == 2, min_wait_status
        assert min_wait_status["enough_free_gpus"] is True, min_wait_status

        start_once_state = Path(tmp) / "start_once_state"
        start_once_state.mkdir()
        fake_start_once_ssh = Path(tmp) / "fake_start_once_ssh.sh"
        fake_start_once_ssh.write_text(FAKE_SSH_START_ONCE)
        fake_start_once_ssh.chmod(fake_start_once_ssh.stat().st_mode | stat.S_IXUSR)
        continuous_proc = run([
            "--run",
            "--continuous",
            "--csv", str(PRODUCTION_CSV),
            "--hosts", "host-a",
            "--gpus", "0",
            "--max-cases", "1",
            "--out", str(Path(tmp) / "remote_continuous_out"),
            "--queue-interval", "0",
            "--queue-timeout", "5",
            "--plan-csv", str(Path(tmp) / "remote_continuous_plan.csv"),
        ], fake_start_once_ssh, {"BEM_FAKE_SSH_STATE_DIR": str(start_once_state)})
        assert continuous_proc.returncode == 0, continuous_proc.stdout
        assert "REMOTE_QUEUE_STATUS attempt=1 usable_remote_gpus=1 min_free_gpus=1 enough_free_gpus=1 selected=1 remote_rc=0" in continuous_proc.stdout
        assert "REMOTE_QUEUE_WAIT attempt=1" in continuous_proc.stdout
        assert "REMOTE_CASE_LEASE_SKIP case=dust_ka30_gmsh7000_balanced_q9_d6_tol5e4" in continuous_proc.stdout
        assert "REMOTE_QUEUE_STATUS attempt=2 usable_remote_gpus=1 min_free_gpus=1 enough_free_gpus=1 selected=0 remote_rc=0" in continuous_proc.stdout
        assert "REMOTE_QUEUE_DONE no cases started in last wave" in continuous_proc.stdout

        supervisor_help = subprocess.run(
            ["bash", str(SUPERVISOR), "--help"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert supervisor_help.returncode == 0, supervisor_help.stdout
        assert "Keeps the remote accuracy-refinement queue alive" in supervisor_help.stdout
        assert "GPU exhaustion is treated as" in supervisor_help.stdout, supervisor_help.stdout
        assert "--min-free-gpus" in supervisor_help.stdout, supervisor_help.stdout
        supervisor_text = SUPERVISOR.read_text()
        assert '[[ "$rc" == "3" ]]' in supervisor_text, supervisor_text
        assert "REMOTE_QUEUE_SUPERVISOR gpu_wait" in supervisor_text, supervisor_text
        queue_status_help = subprocess.run(
            ["python3", str(QUEUE_STATUS), "--help"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert queue_status_help.returncode == 0, queue_status_help.stdout
        assert "Exit codes:" in queue_status_help.stdout, queue_status_help.stdout
        assert "4  supervisor is alive, cases are planned, but usable GPUs are below the effective minimum" in queue_status_help.stdout
        start_help = subprocess.run(
            ["bash", str(START_SUPERVISOR), "--help"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert start_help.returncode == 0, start_help.stdout
        assert "Starts run_remote_refinement_queue_supervisor.sh in a detached session" in start_help.stdout

        status_dir = Path(tmp) / "queue_status"
        status_dir.mkdir()
        leased_out = status_dir / "out"
        for case_name in ("case_a", "case_b"):
            lease_dir = leased_out / "remote_case_leases" / f"{case_name}.lock"
            lease_dir.mkdir(parents=True)
            (lease_dir / "owner").write_text(f"case={case_name} host=gpu1 gpu=0\n")
        (status_dir / "supervisor.pid").write_text("99999999\n")
        (status_dir / "status.json").write_text(json.dumps({
            "planned_cases": 2,
            "usable_remote_gpus": 0,
            "selected": 0,
            "out": str(leased_out),
            "auto_hosts": ["gpu1", "gpu2"],
            "busy_gpus": ["REMOTE_GPU_BUSY host=gpu1 gpu=0 temp=62C util=100% mem=349MiB power=214W"],
            "remote_resume": {"cases": "2", "leased_cases": "1"},
            "remote_rc": 3,
        }))
        (status_dir / "supervisor.log").write_text("\n".join([
            "REMOTE_QUEUE_SUPERVISOR attempt=1 start_time=\"2026-06-25 04:17:35\"",
            "REMOTE_QUEUE_STATUS attempt=1 usable_remote_gpus=0 min_free_gpus=1 enough_free_gpus=0 selected=0 remote_rc=3",
            "REMOTE_QUEUE_WAIT attempt=1 elapsed=19s sleep=60s",
        ]))
        status_proc = subprocess.run(
            ["python3", str(QUEUE_STATUS), "--queue-dir", str(status_dir), "--json"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert status_proc.returncode == 3, status_proc.stdout
        status_payload = json.loads(status_proc.stdout)
        assert status_payload["supervisor_alive"] is False, status_payload
        assert status_payload["queue_health"] == "stopped", status_payload
        assert status_payload["planned_cases"] == 2, status_payload
        assert status_payload["usable_remote_gpus"] == 0, status_payload
        assert status_payload["remote_gpu_summary"] == {
            "hosts": 2,
            "usable": 0,
            "busy": 1,
            "skipped": 0,
            "reachable": 1,
            "blocked": 1,
        }, status_payload
        assert status_payload["busy_gpus"][0].startswith("REMOTE_GPU_BUSY host=gpu1"), status_payload

        (status_dir / "status.json").write_text(json.dumps({
            "planned_cases": 0,
            "usable_remote_gpus": 0,
            "selected": 0,
            "remote_rc": 1,
            "plan_failed": True,
        }))
        plan_failed_proc = subprocess.run(
            ["python3", str(QUEUE_STATUS), "--queue-dir", str(status_dir), "--json"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert plan_failed_proc.returncode == 3, plan_failed_proc.stdout
        plan_failed_payload = json.loads(plan_failed_proc.stdout)
        assert plan_failed_payload["supervisor_alive"] is False, plan_failed_payload
        assert plan_failed_payload["plan_failed"] is True, plan_failed_payload
        assert plan_failed_payload["queue_health"] == "plan_failed", plan_failed_payload

        (status_dir / "status.json").write_text(json.dumps({
            "planned_cases": 2,
            "usable_remote_gpus": 0,
            "selected": 0,
            "out": str(leased_out),
            "auto_hosts": ["gpu1", "gpu2"],
            "busy_gpus": ["REMOTE_GPU_BUSY host=gpu1 gpu=0 temp=62C util=100% mem=349MiB power=214W"],
            "remote_resume": {"cases": "2", "leased_cases": "1"},
            "remote_rc": 3,
        }))
        (status_dir / "supervisor.pid").write_text(f"{os.getpid()}\n")
        alive_status_proc = subprocess.run(
            ["python3", str(QUEUE_STATUS), "--queue-dir", str(status_dir), "--json"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert alive_status_proc.returncode == 4, alive_status_proc.stdout
        alive_payload = json.loads(alive_status_proc.stdout)
        assert alive_payload["supervisor_alive"] is True, alive_payload
        assert alive_payload["queue_health"] == "waiting_for_gpus", alive_payload
        assert alive_payload["status_stale"] is False, alive_payload
        assert alive_payload["status_age_s"] is not None, alive_payload
        assert alive_payload["status_min_free_gpus"] == 1, alive_payload
        assert alive_payload["min_usable_gpus"] is None, alive_payload
        assert alive_payload["effective_min_usable_gpus"] == 1, alive_payload
        assert alive_payload["enough_usable_gpus"] is False, alive_payload
        assert alive_payload["status_enough_free_gpus"] is False, alive_payload
        assert alive_payload["last_queue_status_fields"]["min_free_gpus"] == "1", alive_payload
        assert alive_payload["remote_resume_cases"] == 2, alive_payload
        assert alive_payload["leased_cases"] == 2, alive_payload
        assert alive_payload["status_leased_cases"] == 1, alive_payload
        assert [lease["case"] for lease in alive_payload["lease_files"]] == ["case_a", "case_b"], alive_payload

        not_enough_status_proc = subprocess.run(
            [
                "python3",
                str(QUEUE_STATUS),
                "--queue-dir",
                str(status_dir),
                "--min-usable-gpus",
                "1",
                "--json",
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert not_enough_status_proc.returncode == 4, not_enough_status_proc.stdout
        not_enough_payload = json.loads(not_enough_status_proc.stdout)
        assert not_enough_payload["supervisor_alive"] is True, not_enough_payload
        assert not_enough_payload["status_min_free_gpus"] == 1, not_enough_payload
        assert not_enough_payload["min_usable_gpus"] == 1, not_enough_payload
        assert not_enough_payload["effective_min_usable_gpus"] == 1, not_enough_payload
        assert not_enough_payload["enough_usable_gpus"] is False, not_enough_payload
        assert not_enough_payload["queue_health"] == "waiting_for_gpus", not_enough_payload

        (status_dir / "status.json").write_text(json.dumps({
            "planned_cases": 2,
            "usable_remote_gpus": 1,
            "min_free_gpus": 2,
            "enough_free_gpus": False,
            "selected": 0,
            "out": str(leased_out),
            "auto_hosts": ["gpu1", "gpu2"],
            "busy_gpus": ["REMOTE_GPU_BUSY host=gpu1 gpu=0 temp=62C util=100% mem=349MiB power=214W"],
            "remote_resume": {"cases": "2", "leased_cases": "1"},
            "remote_rc": 3,
        }))
        saved_min_status_proc = subprocess.run(
            ["python3", str(QUEUE_STATUS), "--queue-dir", str(status_dir), "--json"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert saved_min_status_proc.returncode == 4, saved_min_status_proc.stdout
        saved_min_payload = json.loads(saved_min_status_proc.stdout)
        assert saved_min_payload["status_min_free_gpus"] == 2, saved_min_payload
        assert saved_min_payload["min_usable_gpus"] is None, saved_min_payload
        assert saved_min_payload["effective_min_usable_gpus"] == 2, saved_min_payload
        assert saved_min_payload["enough_usable_gpus"] is False, saved_min_payload
        assert saved_min_payload["status_enough_free_gpus"] is False, saved_min_payload
        assert saved_min_payload["queue_health"] == "insufficient_gpus", saved_min_payload

        old_time = 1_600_000_000
        os.utime(status_dir / "status.json", (old_time, old_time))
        stale_status_proc = subprocess.run(
            [
                "python3",
                str(QUEUE_STATUS),
                "--queue-dir",
                str(status_dir),
                "--status-max-age-s",
                "60",
                "--json",
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert stale_status_proc.returncode == 4, stale_status_proc.stdout
        stale_payload = json.loads(stale_status_proc.stdout)
        assert stale_payload["supervisor_alive"] is True, stale_payload
        assert stale_payload["status_stale"] is True, stale_payload
        assert stale_payload["queue_health"] == "status_stale", stale_payload
        assert stale_payload["status_age_s"] > 60, stale_payload
        assert stale_payload["effective_min_usable_gpus"] == 2, stale_payload
        assert stale_payload["enough_usable_gpus"] is False, stale_payload

        human_status_proc = subprocess.run(
            [
                "python3",
                str(QUEUE_STATUS),
                "--queue-dir",
                str(status_dir),
                "--status-max-age-s",
                "60",
                "--min-usable-gpus",
                "2",
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert human_status_proc.returncode == 4, human_status_proc.stdout
        assert "usable_gpus=1 busy_gpus=1 skipped_gpus=0 reachable_gpus=2" in human_status_proc.stdout
        assert "min_usable_gpus=2 status_min_free_gpus=2 effective_min_usable_gpus=2 enough_usable_gpus=False" in human_status_proc.stdout
        assert "remote_resume_cases=2 leased_cases=2" in human_status_proc.stdout
        assert "LEASE case=case_a" in human_status_proc.stdout
        assert "stale=1" in human_status_proc.stdout, human_status_proc.stdout
        assert "health=status_stale" in human_status_proc.stdout, human_status_proc.stdout

    print("remote accuracy refinement wave: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
