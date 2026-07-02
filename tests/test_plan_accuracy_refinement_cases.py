#!/usr/bin/env python3
"""Smoke-tests for accuracy refinement planning."""

import subprocess
import tempfile
import csv
import os
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plan_accuracy_refinement_cases.py"


def run(csv_path: Path, *args: str, env=None) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        [str(SCRIPT), "--csv", str(csv_path), *args],
        cwd=str(ROOT),
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "audit.csv"
        csv_path.write_text(
            "shape,ka,mesh_label,status,metadata_status,operator_status,gate_error,"
            "raw_pass10,pass10,worst_component,worst_component_error,failed_main_10pct,failed_all_20pct\n"
            "сфера,30,ref6,PASS,legacy,not_required,0.02,true,false,M11,0.02,,\n"
            "гексагональная призма,20,ref4,FAIL,ok,not_required,0.2,false,false,M34,0.2,M34,\n"
            "сфера,5,ref4,PASS,ok,not_required,0.01,true,true,M12,0.01,,\n"
        )

        out_dir = Path(tmp) / "out"
        plan_csv = out_dir / "plan.csv"
        proc = run(csv_path, "--no-health-check", "--gpus", "0 1", "--out", str(out_dir))
        assert proc.returncode == 0, proc.stdout
        assert "REFINE threshold=0.1 reason=all planned=2 limit=2" in proc.stdout, proc.stdout
        assert f"plan_csv={plan_csv}" in proc.stdout, proc.stdout
        assert "hex_ka20_ref5_balanced_q9_d6_tol5e4" in proc.stdout, proc.stdout
        assert "sphere_ka30_ref6_current_q7_d6_tol3e3" in proc.stdout, proc.stdout
        assert "sphere_ka5_ref4_current_q7_d6_tol3e3" not in proc.stdout, proc.stdout
        assert "--max-jobs 2" in proc.stdout, proc.stdout
        rows = list(csv.DictReader(plan_csv.open()))
        assert [row["case_name"] for row in rows] == [
            "hex_ka20_ref5_balanced_q9_d6_tol5e4",
            "sphere_ka30_ref6_current_q7_d6_tol3e3",
        ], rows
        assert rows[0]["reason"] == "accuracy", rows
        assert rows[1]["reason"] == "metadata", rows
        assert rows[0]["worst_component"] == "M34", rows
        assert rows[0]["failed_main_10pct"] == "M34", rows
        assert rows[1]["worst_component"] == "M11", rows

        explicit_plan = Path(tmp) / "explicit.csv"
        proc = run(
            csv_path,
            "--no-health-check",
            "--gpus", "0 1",
            "--plan-csv", str(explicit_plan),
        )
        assert proc.returncode == 0, proc.stdout
        assert f"plan_csv={explicit_plan}" in proc.stdout, proc.stdout
        assert explicit_plan.is_file(), explicit_plan

        no_plan_dir = Path(tmp) / "no_plan"
        proc = run(csv_path, "--no-health-check", "--gpus", "0 1", "--out", str(no_plan_dir), "--no-plan-csv")
        assert proc.returncode == 0, proc.stdout
        assert "plan_csv=" not in proc.stdout, proc.stdout
        assert not (no_plan_dir / "plan.csv").exists(), proc.stdout

        proc = run(csv_path, "--all-cases", "--no-health-check", "--gpus", "0 1")
        assert proc.returncode == 0, proc.stdout
        assert "REFINE threshold=0.1 reason=all planned=2 limit=all" in proc.stdout, proc.stdout
        assert "sphere_ka5_ref4_current_q7_d6_tol3e3" not in proc.stdout, proc.stdout
        assert "--max-jobs 2" in proc.stdout, proc.stdout

        proc = run(csv_path, "--max-cases", "1", "--no-health-check", "--gpus", "0 1")
        assert proc.returncode == 0, proc.stdout
        assert "REFINE threshold=0.1 reason=all planned=1 limit=1" in proc.stdout, proc.stdout
        assert "hex_ka20_ref5_balanced_q9_d6_tol5e4" in proc.stdout, proc.stdout
        assert "sphere_ka30_ref6_current_q7_d6_tol3e3" not in proc.stdout, proc.stdout
        assert "--max-jobs 1" in proc.stdout, proc.stdout

        proc = run(csv_path, "--only-reason", "metadata", "--all-cases", "--no-health-check", "--gpus", "0 1")
        assert proc.returncode == 0, proc.stdout
        assert "REFINE threshold=0.1 reason=metadata planned=1 limit=all" in proc.stdout, proc.stdout
        assert "sphere_ka30_ref6_current_q7_d6_tol3e3" in proc.stdout, proc.stdout
        assert "sphere_ka5_ref4_current_q7_d6_tol3e3" not in proc.stdout, proc.stdout
        assert "hex_ka20_ref5_balanced_q9_d6_tol5e4" not in proc.stdout, proc.stdout

        proc = run(csv_path, "--only-reason", "accuracy", "--all-cases", "--no-health-check", "--gpus", "0 1")
        assert proc.returncode == 0, proc.stdout
        assert "REFINE threshold=0.1 reason=accuracy planned=1 limit=all" in proc.stdout, proc.stdout
        assert "hex_ka20_ref5_balanced_q9_d6_tol5e4" in proc.stdout, proc.stdout
        assert "sphere_ka30_ref6_current_q7_d6_tol3e3" not in proc.stdout, proc.stdout

        all_component_csv = Path(tmp) / "all_component_fail.csv"
        all_component_csv.write_text(
            "shape,ka,mesh_label,status,metadata_status,operator_status,gate_error,"
            "failed_all_20pct\n"
            "гексагональная призма,10,ref3,PASS,ok,not_required,0.04,M23\n"
        )
        proc = run(all_component_csv, "--only-reason", "accuracy", "--all-cases",
                   "--no-health-check", "--gpus", "0 1")
        assert proc.returncode == 0, proc.stdout
        assert "REFINE threshold=0.1 reason=accuracy planned=1 limit=all" in proc.stdout, proc.stdout
        assert "hex_ka10_ref4_balanced_q9_d6_tol5e4" in proc.stdout, proc.stdout

        legacy_pass_csv = Path(tmp) / "legacy_pass_without_flags.csv"
        legacy_pass_csv.write_text(
            "shape,ka,mesh_label,status,metadata_status,operator_status,gate_error,"
            "failed_all_20pct\n"
            "сфера,5,ref4,PASS,ok,not_required,0.01,\n"
        )
        proc = run(legacy_pass_csv, "--only-reason", "accuracy", "--all-cases",
                   "--no-health-check", "--gpus", "0 1")
        assert proc.returncode == 0, proc.stdout
        assert "REFINE threshold=0.1 reason=accuracy planned=1 limit=all" in proc.stdout, proc.stdout
        assert "sphere_ka5_ref5_current_q9_d7_tol1e3" in proc.stdout, proc.stdout

        stale_csv = Path(tmp) / "stale.csv"
        stale_csv.write_text(
            "shape,ka,mesh_label,status,metadata_status,operator_status,gate_error,raw_pass10,pass10\n"
            "гексагональная призма,30,ref5,STALE,invalid,not_required,0.084,true,false\n"
            "сфера,20,ref4,STALE,invalid,not_required,0.012,true,false\n"
            "пылевая частица,20,gmsh4200,MISSING,missing,missing,,false,false\n"
        )
        stale_plan = Path(tmp) / "stale_plan.csv"
        proc = run(
            stale_csv,
            "--all-cases",
            "--no-health-check",
            "--gpus", "0 1",
            "--plan-csv", str(stale_plan),
        )
        assert proc.returncode == 0, proc.stdout
        assert "hex_ka30_ref5_balanced_q7_d5_tol1e3" in proc.stdout, proc.stdout
        assert "sphere_ka20_ref4_current_q7_d6_tol3e3" in proc.stdout, proc.stdout
        assert "dust_ka20_gmsh5200_balanced_q9_d6_tol5e4" in proc.stdout, proc.stdout
        assert "hex_ka30_ref6_balanced_q9_d6_tol5e4" not in proc.stdout, proc.stdout
        assert "sphere_ka20_ref5_current_q9_d7_tol1e3" not in proc.stdout, proc.stdout
        rows = list(csv.DictReader(stale_plan.open()))
        reasons = {row["case_name"]: row["reason"] for row in rows}
        assert reasons["hex_ka30_ref5_balanced_q7_d5_tol1e3"] == "metadata", rows
        assert reasons["sphere_ka20_ref4_current_q7_d6_tol3e3"] == "metadata", rows
        assert reasons["dust_ka20_gmsh5200_balanced_q9_d6_tol5e4"] == "accuracy+metadata", rows

        bad_csv = Path(tmp) / "bad_case.csv"
        bad_csv.write_text(
            "shape,ka,mesh_label,status,metadata_status,operator_status,gate_error,raw_pass10,pass10\n"
            "пылевая частица,5,gmsh9999,PASS,legacy,complex_operator,0.01,true,false\n"
        )
        bad_plan = Path(tmp) / "bad_plan.csv"
        proc = run(bad_csv, "--all-cases", "--no-health-check", "--gpus", "0 1", "--plan-csv", str(bad_plan))
        assert proc.returncode == 4, proc.stdout
        assert "case validation failed:" in proc.stdout, proc.stdout
        assert "dust_ka5_gmsh9999_balanced_q7_d6_tol5e4" in proc.stdout, proc.stdout
        assert not bad_plan.exists(), proc.stdout
        proc = run(bad_csv, "--all-cases", "--no-health-check", "--gpus", "0 1", "--no-validate-cases")
        assert proc.returncode == 0, proc.stdout
        assert "dust_ka5_gmsh9999_balanced_q7_d6_tol5e4" in proc.stdout, proc.stdout

        dust_cmd = subprocess.run(
            [
                str(ROOT / "scripts" / "run_accuracy_matrix_case.sh"),
                "--gpu", "0",
                "--case", "dust_ka20_gmsh4200_balanced_q7_d6_tol5e4",
                "--out", str(Path(tmp) / "dust_out"),
                "--print",
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert dust_cmd.returncode == 0, dust_cmd.stdout
        assert "--accurate" in dust_cmd.stdout, dust_cmd.stdout
        assert "--fmm-digits 6" in dust_cmd.stdout, dust_cmd.stdout
        assert "--gmres-tol 5e-4" in dust_cmd.stdout, dust_cmd.stdout
        assert "--gmres-restart 500" in dust_cmd.stdout, dust_cmd.stdout
        assert "--max-leaf 128" in dust_cmd.stdout, dust_cmd.stdout

        legacy_dust_cmd = subprocess.run(
            [
                str(ROOT / "scripts" / "run_accuracy_matrix_case.sh"),
                "--gpu", "0",
                "--case", "dust_ka20_gmsh4200_balanced_q7_d5_tol1e3",
                "--out", str(Path(tmp) / "legacy_dust_out"),
                "--print",
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert legacy_dust_cmd.returncode == 2, legacy_dust_cmd.stdout
        assert "legacy dust case disabled" in legacy_dust_cmd.stdout, legacy_dust_cmd.stdout
        legacy_dust_allowed = subprocess.run(
            [
                str(ROOT / "scripts" / "run_accuracy_matrix_case.sh"),
                "--gpu", "0",
                "--case", "dust_ka20_gmsh4200_balanced_q7_d5_tol1e3",
                "--out", str(Path(tmp) / "legacy_dust_out"),
                "--print",
            ],
            cwd=str(ROOT),
            env={**os.environ, "BEM_ALLOW_LEGACY_DUST": "1"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert legacy_dust_allowed.returncode == 0, legacy_dust_allowed.stdout
        assert "--fmm-digits 5" in legacy_dust_allowed.stdout, legacy_dust_allowed.stdout

        bad_parameterized_dust = subprocess.run(
            [
                str(ROOT / "scripts" / "run_accuracy_matrix_case.sh"),
                "--gpu", "0",
                "--case", "dust_ka20_gmsh4200_balanced_q7_d5_tol2e2",
                "--out", str(Path(tmp) / "bad_parameterized_dust_out"),
                "--print",
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert bad_parameterized_dust.returncode == 2, bad_parameterized_dust.stdout
        assert "Parameterized dust cases require d6 or better" in bad_parameterized_dust.stdout

        loose_parameterized_dust = subprocess.run(
            [
                str(ROOT / "scripts" / "run_accuracy_matrix_case.sh"),
                "--gpu", "0",
                "--case", "dust_ka20_gmsh4200_balanced_q7_d6_tol1e3",
                "--out", str(Path(tmp) / "loose_parameterized_dust_out"),
                "--print",
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert loose_parameterized_dust.returncode == 2, loose_parameterized_dust.stdout
        assert "Parameterized dust cases require gmres_tol <= 5e-4" in loose_parameterized_dust.stdout

        legacy_refresh = subprocess.run(
            [str(ROOT / "scripts" / "run_complex_operator_dust_refresh.sh")],
            cwd=str(ROOT),
            env={**os.environ, "OUT": str(Path(tmp) / "legacy_refresh_out")},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert legacy_refresh.returncode == 2, legacy_refresh.stdout
        assert "legacy complex-operator dust refresh is disabled" in legacy_refresh.stdout, legacy_refresh.stdout
        assert "q7_d6_tol5e4 dust production runs" in legacy_refresh.stdout, legacy_refresh.stdout

        runner_text = (ROOT / "scripts" / "run_accuracy_matrix_case.sh").read_text()
        assert "legacy_dust=(" in runner_text
        assert "common_dust=(" not in runner_text
        assert "New dust reruns use parameterized q*_d6_tol5e4 names" in runner_text

        production_plan = subprocess.run(
            [str(ROOT / "scripts" / "run_accuracy_matrix_15_queue.sh"), "--plan"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert production_plan.returncode == 0, production_plan.stdout
        production_cases = [line.strip() for line in production_plan.stdout.splitlines() if line.strip()]
        assert len(production_cases) == 15, production_cases
        for case_name in production_cases:
            planned_case = subprocess.run(
                [
                    str(ROOT / "scripts" / "run_accuracy_matrix_case.sh"),
                    "--gpu", "0",
                    "--case", case_name,
                    "--out", str(Path(tmp) / "production_plan_out"),
                    "--print",
                ],
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            assert planned_case.returncode == 0, (case_name, planned_case.stdout)
        assert any(case.endswith("_q7_d6_tol5e4") for case in production_cases if case.startswith("dust_"))
        assert not any(
            case.startswith("dust_") and case.endswith("_q7_d5_tol1e3")
            for case in production_cases
        ), production_cases

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
        proc = run(
            csv_path,
            "--gpus", "auto",
            "--no-health-check",
            env={"BEM_NVIDIA_SMI": str(fake_nvidia_smi)},
        )
        assert proc.returncode == 0, proc.stdout
        assert "REFINE threshold=0.1 reason=all planned=2 limit=2" in proc.stdout, proc.stdout
        assert "--gpus auto" in proc.stdout, proc.stdout
        assert "--max-jobs 2" in proc.stdout, proc.stdout

        fake_nvidia_smi.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == "--query-gpu=index --format=csv,noheader,nounits" ]]; then
  printf '0\\n2\\n'
  exit 0
fi
if [[ "$*" == "-i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits" ]]; then
  exit 0
fi
if [[ "$*" == "-i 2 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits" ]]; then
  printf '4242, ./mbs_po_gpu_float_fast, 304\\n'
  exit 0
fi
echo "unexpected fake nvidia-smi args: $*" >&2
exit 1
""")
        fake_nvidia_smi.chmod(fake_nvidia_smi.stat().st_mode | stat.S_IXUSR)
        proc = run(
            csv_path,
            "--gpus", "auto",
            env={"BEM_NVIDIA_SMI": str(fake_nvidia_smi)},
        )
        assert proc.returncode == 0, proc.stdout
        assert "REFINE threshold=0.1 reason=all planned=1 limit=1" in proc.stdout, proc.stdout
        assert "--max-jobs 1" in proc.stdout, proc.stdout
        assert "--allow-compute-share" not in proc.stdout, proc.stdout

        proc = run(
            csv_path,
            "--gpus", "auto",
            "--allow-compute-share",
            env={"BEM_NVIDIA_SMI": str(fake_nvidia_smi)},
        )
        assert proc.returncode == 0, proc.stdout
        assert "REFINE threshold=0.1 reason=all planned=2 limit=2" in proc.stdout, proc.stdout
        assert "--allow-compute-share" in proc.stdout, proc.stdout

    print("plan accuracy refinement cases: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
