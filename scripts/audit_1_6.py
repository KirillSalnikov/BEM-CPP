#!/usr/bin/env python3
"""Collect the current 1-6 control evidence into one JSON report."""

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from time import time
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: List[str], *, timeout: Optional[int] = None) -> dict:
    t0 = time()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "seconds": time() - t0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "pass": proc.returncode == 0,
    }


def file_exists(path: str) -> dict:
    p = ROOT / path
    return {"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() else 0}


def source_contains(path: str, needles: List[str]) -> dict:
    p = ROOT / path
    if not p.exists():
        return {"path": str(p), "exists": False, "contains_all": False, "missing": needles}
    text = p.read_text(errors="replace")
    missing = [needle for needle in needles if needle not in text]
    return {
        "path": str(p),
        "exists": True,
        "contains_all": not missing,
        "missing": missing,
    }


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def command_text(cmd: List[str]) -> Optional[str]:
    proc = subprocess.run(cmd, cwd=str(ROOT), universal_newlines=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def audit_metadata(argv: List[str]) -> dict:
    status_text = command_text(["git", "status", "--short", "--", "."])
    status_lines = status_text.splitlines() if status_text else []
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [Path(sys.executable).name, *argv],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": command_text(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(status_lines),
        "git_dirty_count": len(status_lines),
        "git_dirty_sample": status_lines[:40],
    }


def determine_exit_code(report: dict, *, require_cuda_reference: bool) -> int:
    if not report.get("all_local_requirements_pass"):
        return 2
    if require_cuda_reference and not report.get("cuda_reference_verified"):
        return 4
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "audit_1_6_report.json")
    parser.add_argument("--run-cuda", action="store_true",
                        help="Also run dense/FMM executable checks; requires built binary and CUDA")
    parser.add_argument("--require-cuda-reference", action="store_true",
                        help="Return non-zero unless dense-vs-FMM CUDA reference is verified")
    parser.add_argument("--binary", type=Path, default=ROOT / "bin" / "bem_cuda_fmm")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cuda_detect_path = args.out.parent / "cuda_toolchain_detect.json"
    report = {
        "root": str(ROOT),
        "metadata": audit_metadata(sys.argv[1:]),
        "checks": {},
        "binary": {
            "path": str(args.binary),
            "exists": args.binary.exists(),
            "executable": bool(args.binary.exists() and args.binary.is_file() and (args.binary.stat().st_mode & 0o111)),
        },
    }

    report["checks"]["python_compile"] = run([
        "python3", "-m", "py_compile",
        "bemcuda/__init__.py",
        "bemcuda/job.py",
        "scripts/mueller_audit.py",
        "scripts/operator_block_audit.py",
        "scripts/cpu_pmchwt_centroid_reference.py",
        "scripts/check_result_metadata.py",
        "scripts/audit_1_6.py",
        "scripts/plan_accuracy_refinement_cases.py",
        "scripts/check_audit_1_6_report.py",
        "scripts/summarize_audit_1_6.py",
        "scripts/reference_dense_check.py",
        "scripts/near_singular_audit.py",
        "scripts/hmatrix_memory_audit.py",
        "scripts/summarize_gpu_power_monitor.py",
        "scripts/detect_cuda_toolchain.py",
        "tests/test_bem_job_api.py",
        "tests/test_audit_1_6_exit_codes.py",
        "tests/test_audit_1_6_report_contract.py",
        "tests/test_audit_accuracy_matrix_selection.py",
        "tests/test_result_metadata_check.py",
        "tests/test_mie_mueller_symmetry.py",
        "tests/test_mueller_audit_physical.py",
        "tests/test_plan_accuracy_refinement_cases.py",
        "tests/test_remote_resume_accuracy_matrix_cases.py",
        "tests/test_summarize_audit_1_6.py",
        "tests/test_cuda_toolchain_detection.py",
        "tests/test_near_singular_audit.py",
        "tests/test_cpu_pmchwt_centroid_reference.py",
        "tests/test_hmatrix_memory_audit.py",
    ])
    report["checks"]["shell_syntax"] = run([
        "bash", "-n",
        "scripts/build_cuda_fmm.sh",
        "scripts/deploy_accuracy_matrix_15_queue.sh",
        "scripts/fetch_accuracy_matrix_15_results.sh",
        "scripts/ipmi_power_control.sh",
        "scripts/queue_live_status.sh",
        "scripts/queue_watch_once.sh",
        "scripts/remote_power_watch.sh",
        "scripts/remote_resume_accuracy_matrix_cases.sh",
        "scripts/resume_accuracy_matrix_cases.sh",
        "scripts/resume_accuracy_matrix_15_after_current.sh",
        "scripts/run_accuracy_matrix_case.sh",
        "scripts/run_accuracy_matrix_15_queue.sh",
        "scripts/run_guarded_bem_case.sh",
        "scripts/run_cuda_reference_audits.sh",
        "scripts/run_local_audits.sh",
        "scripts/set_remote_gpu_power_limit.sh",
        "scripts/supermicro_bmc_access.sh",
    ])
    report["checks"]["cuda_toolchain_detect"] = run([
        "python3", "scripts/detect_cuda_toolchain.py",
        "--json-out", str(cuda_detect_path),
    ])
    cuda_detect = read_json(cuda_detect_path)
    report["cuda_toolchain_available"] = bool(cuda_detect.get("usable"))
    report["cuda_runtime_ready"] = bool(cuda_detect.get("runtime_ready"))
    report["cuda_runtime"] = cuda_detect.get("runtime", {})
    report["checks"]["python_job_api"] = run(["python3", "tests/test_bem_job_api.py"])
    report["checks"]["audit_exit_code_test"] = run(["python3", "tests/test_audit_1_6_exit_codes.py"])
    report["checks"]["audit_report_contract_test"] = run(["python3", "tests/test_audit_1_6_report_contract.py"])
    report["checks"]["accuracy_matrix_selection_test"] = run(["python3", "tests/test_audit_accuracy_matrix_selection.py"])
    report["checks"]["result_metadata_numeric_test"] = run(["python3", "tests/test_result_metadata_check.py"])
    report["checks"]["accuracy_refinement_planner_test"] = run(["python3", "tests/test_plan_accuracy_refinement_cases.py"])
    report["checks"]["remote_accuracy_resume_test"] = run(["python3", "tests/test_remote_resume_accuracy_matrix_cases.py"])
    report["checks"]["audit_summary_test"] = run(["python3", "tests/test_summarize_audit_1_6.py"])
    report["checks"]["cuda_toolchain_detection_test"] = run(["python3", "tests/test_cuda_toolchain_detection.py"])
    monitor_fixture = args.out.parent / "gpu_monitor_fixture"
    monitor_fixture.mkdir(parents=True, exist_ok=True)
    (monitor_fixture / "case.gpu.csv").write_text(
        "timestamp_s,gpu,temp_c,util_pct,mem_mib,power_w\n"
        "10,0,40,50,1000,120\n"
        "20,0,50,100,2000,220\n"
    )
    report["checks"]["gpu_power_monitor_summary"] = run([
        "python3", "scripts/summarize_gpu_power_monitor.py",
        str(monitor_fixture),
    ])
    report["checks"]["mueller_self_test"] = run(["python3", "scripts/mueller_audit.py", "--self-test"])
    report["checks"]["mie_mueller_symmetry_test"] = run(["python3", "tests/test_mie_mueller_symmetry.py"])
    report["checks"]["mueller_physical_gate_test"] = run(["python3", "tests/test_mueller_audit_physical.py"])
    report["checks"]["operator_block_self_test"] = run(["python3", "scripts/operator_block_audit.py", "--self-test"])
    report["checks"]["cpu_pmchwt_centroid"] = run([
        "python3", "scripts/cpu_pmchwt_centroid_reference.py",
        "--json-out", str(args.out.parent / "cpu_pmchwt_centroid.json"),
    ])
    report["checks"]["cpu_pmchwt_system_contract_test"] = run(["python3", "tests/test_cpu_pmchwt_centroid_reference.py"])
    report["checks"]["near_singular_routing_test"] = run(["python3", "tests/test_near_singular_audit.py"])
    report["checks"]["hmatrix_memory_audit_test"] = run(["python3", "tests/test_hmatrix_memory_audit.py"])
    if shutil.which("g++"):
        report["checks"]["operator_config_cpp"] = run([
            "g++", "-O2", "-Wall", "-std=c++11", "-Isrc",
            "-o", "/tmp/bem_operator_config_check",
            "tests/operator_config_check.cpp",
        ])
        if report["checks"]["operator_config_cpp"]["pass"]:
            report["checks"]["operator_config_cpp_run"] = run(["/tmp/bem_operator_config_check"])
        else:
            report["checks"]["operator_config_cpp_run"] = {
                "pass": False,
                "skipped": True,
                "reason": "operator_config_check.cpp did not compile",
            }
    else:
        report["checks"]["operator_config_cpp"] = {
            "pass": None,
            "skipped": True,
            "reason": "g++ is not available",
        }
        report["checks"]["operator_config_cpp_run"] = {
            "pass": None,
            "skipped": True,
            "reason": "g++ is not available",
        }

    report["artifacts"] = {
        "mesh_quality_gate": file_exists("src/mesh.cpp"),
        "operator_config": file_exists("src/operator_config.h"),
        "python_api": file_exists("bemcuda/job.py"),
        "local_audits": file_exists("scripts/run_local_audits.sh"),
        "audit_report_checker": file_exists("scripts/check_audit_1_6_report.py"),
        "audit_summary": file_exists("scripts/summarize_audit_1_6.py"),
        "dense_fmm_reference": file_exists("scripts/reference_dense_check.py"),
        "cuda_reference_runner": file_exists("scripts/run_cuda_reference_audits.sh"),
        "cuda_fmm_builder": file_exists("scripts/build_cuda_fmm.sh"),
        "cuda_toolchain_detector": file_exists("scripts/detect_cuda_toolchain.py"),
        "production_queue": file_exists("scripts/run_accuracy_matrix_15_queue.sh"),
        "result_metadata_checker": file_exists("scripts/check_result_metadata.py"),
        "production_deploy": file_exists("scripts/deploy_accuracy_matrix_15_queue.sh"),
        "production_fetch": file_exists("scripts/fetch_accuracy_matrix_15_results.sh"),
        "ipmi_power_control": file_exists("scripts/ipmi_power_control.sh"),
        "queue_live_status": file_exists("scripts/queue_live_status.sh"),
        "queue_watch_once": file_exists("scripts/queue_watch_once.sh"),
        "power_watch": file_exists("scripts/remote_power_watch.sh"),
        "gpu_power_monitor_summary": file_exists("scripts/summarize_gpu_power_monitor.py"),
        "production_resume": file_exists("scripts/resume_accuracy_matrix_15_after_current.sh"),
        "production_resume_cases": file_exists("scripts/resume_accuracy_matrix_cases.sh"),
        "remote_production_resume_cases": file_exists("scripts/remote_resume_accuracy_matrix_cases.sh"),
        "production_case_runner": file_exists("scripts/run_accuracy_matrix_case.sh"),
        "production_guarded_case_runner": file_exists("scripts/run_guarded_bem_case.sh"),
        "production_refinement_planner": file_exists("scripts/plan_accuracy_refinement_cases.py"),
        "gpu_power_limit": file_exists("scripts/set_remote_gpu_power_limit.sh"),
        "bmc_access": file_exists("scripts/supermicro_bmc_access.sh"),
        "near_singular_audit": file_exists("scripts/near_singular_audit.py"),
        "hmatrix_memory_audit": file_exists("scripts/hmatrix_memory_audit.py"),
        "control_docs": file_exists("docs/control_layers_1_6.md"),
        "cuda_env_file": file_exists("environment.cuda.yml"),
        "cuda_runtime_docs": source_contains("docs/control_layers_1_6.md", [
            "scripts/build_cuda_fmm.sh",
            "lib/x86_64-linux-gnu",
            "cuda_toolchain_available",
            "cuda_runtime_ready",
            "python3 scripts/detect_cuda_toolchain.py --require-runtime",
            "python3 scripts/audit_1_6.py --run-cuda --require-cuda-reference --binary ./bin/bem_cuda_fmm",
            "3 - toolkit найден, но NVIDIA runtime/driver не готов",
            "режим возвращает код `4`",
            "scripts/supermicro_bmc_access.sh --remote-diagnose",
            "scripts/resume_accuracy_matrix_15_after_current.sh --wait-and-resume",
            "scripts/resume_accuracy_matrix_cases.sh --run",
            "scripts/remote_resume_accuracy_matrix_cases.sh",
            "--validate-numeric",
            "python3 scripts/plan_accuracy_refinement_cases.py",
            "sphere_ka30_ref7_current_q13_d7_tol1e3",
            "hex_ka30_ref5_balanced_q7_d5_tol1e3,sphere_ka30_ref6_current_q7_d6_tol3e3",
            "scripts/set_remote_gpu_power_limit.sh --set",
            "scripts/ipmi_power_control.sh --on",
            "ACPowerOn",
        ]),
        "cuda_env_pins": source_contains("environment.cuda.yml", [
            "cuda-version=12.2",
            "cuda-nvcc=12.2.140",
            "gcc_linux-64=12",
            "gxx_linux-64=12",
        ]),
        "make_audit_targets": source_contains("Makefile", [
            "host-checks:",
            "$(HOST_TEST_DIR)/operator_config_check:",
            "$(HOST_TEST_DIR)/precond_policy_check:",
            "$(HOST_TEST_DIR)/solver_policy_check:",
            "$(HOST_TEST_DIR)/output_json_mesh_check:",
            "host-audits: host-checks",
            "audit-1-6:",
            "audit-1-6-summary:",
            "cuda-runtime-check:",
            "cuda-audits:",
            "cuda-audits-summary:",
        ]),
        "singular_corrections": source_contains("src/assembly.cu", [
            "apply_singular_corrections",
            "potential_integral_triangle",
            "vector_potential_integral_triangle",
        ]),
        "host_operator_cpp_test": file_exists("tests/operator_config_check.cpp"),
        "python_job_api_test": file_exists("tests/test_bem_job_api.py"),
        "audit_exit_code_test": file_exists("tests/test_audit_1_6_exit_codes.py"),
        "audit_report_contract_test": file_exists("tests/test_audit_1_6_report_contract.py"),
        "audit_summary_test": file_exists("tests/test_summarize_audit_1_6.py"),
    }

    if args.run_cuda and report["cuda_runtime_ready"]:
        report["checks"]["dense_fmm_reference"] = run([
            "python3", "scripts/reference_dense_check.py",
            "--binary", str(args.binary),
            "--out-dir", str(args.out.parent / "dense_fmm_reference"),
        ], timeout=None)
        report["checks"]["dense_fmm_absorbing_reference"] = run([
            "python3", "scripts/reference_dense_check.py",
            "--binary", str(args.binary),
            "--out-dir", str(args.out.parent / "dense_fmm_absorbing_reference"),
            "--ri", "1.6", "0.002",
            "--system", "balanced",
            "--require-complex-operator",
            "--max-l2", "1e-2",
        ], timeout=None)
    elif args.run_cuda:
        report["checks"]["dense_fmm_reference"] = {
            "pass": None,
            "skipped": True,
            "reason": "CUDA toolkit is available, but NVIDIA driver/runtime is not ready on this host",
            "runtime_missing": report["cuda_runtime"].get("missing", []),
        }
        report["checks"]["dense_fmm_absorbing_reference"] = dict(report["checks"]["dense_fmm_reference"])
    else:
        report["checks"]["dense_fmm_reference"] = {
            "pass": None,
            "skipped": True,
            "reason": "use --run-cuda on a host with built bem_cuda_fmm and CUDA",
        }
        report["checks"]["dense_fmm_absorbing_reference"] = dict(report["checks"]["dense_fmm_reference"])

    requirements = {
        "1_cpu_reference_pmchwt": (
            report["checks"]["cpu_pmchwt_centroid"]["pass"]
            and report["checks"]["cpu_pmchwt_system_contract_test"]["pass"]
        ),
        "2_mueller_sign_audit": (
            report["checks"]["mueller_self_test"]["pass"]
            and report["checks"]["mie_mueller_symmetry_test"]["pass"]
            and report["checks"]["mueller_physical_gate_test"]["pass"]
        ),
        "3_singular_near_singular_hooks": report["artifacts"]["near_singular_audit"]["exists"] and report["artifacts"]["singular_corrections"]["contains_all"],
        "4_mesh_quality_gate": report["artifacts"]["mesh_quality_gate"]["exists"],
        "5_operator_architecture": report["artifacts"]["operator_config"]["exists"] and report["checks"]["operator_block_self_test"]["pass"] and report["checks"]["operator_config_cpp_run"]["pass"],
        "6_python_job_api": report["artifacts"]["python_api"]["exists"] and report["checks"]["python_compile"]["pass"] and report["checks"]["python_job_api"]["pass"] and report["checks"]["cuda_toolchain_detection_test"]["pass"],
        "cuda_build_runtime_gate": (
            report["checks"]["shell_syntax"]["pass"]
            and report["artifacts"]["cuda_reference_runner"]["exists"]
            and report["artifacts"]["cuda_fmm_builder"]["exists"]
            and report["artifacts"]["cuda_toolchain_detector"]["exists"]
            and report["artifacts"]["production_queue"]["exists"]
            and report["artifacts"]["result_metadata_checker"]["exists"]
            and report["artifacts"]["production_deploy"]["exists"]
            and report["artifacts"]["production_fetch"]["exists"]
            and report["artifacts"]["ipmi_power_control"]["exists"]
            and report["artifacts"]["queue_live_status"]["exists"]
            and report["artifacts"]["power_watch"]["exists"]
            and report["artifacts"]["gpu_power_monitor_summary"]["exists"]
            and report["checks"]["gpu_power_monitor_summary"]["pass"]
            and report["artifacts"]["production_resume"]["exists"]
            and report["artifacts"]["production_resume_cases"]["exists"]
            and report["artifacts"]["remote_production_resume_cases"]["exists"]
            and report["artifacts"]["production_case_runner"]["exists"]
            and report["artifacts"]["production_guarded_case_runner"]["exists"]
            and report["artifacts"]["gpu_power_limit"]["exists"]
            and report["artifacts"]["bmc_access"]["exists"]
            and report["artifacts"]["cuda_env_file"]["exists"]
            and report["artifacts"]["cuda_runtime_docs"]["contains_all"]
            and report["artifacts"]["cuda_env_pins"]["contains_all"]
            and report["artifacts"]["make_audit_targets"]["contains_all"]
            and report["artifacts"]["audit_exit_code_test"]["exists"]
            and report["artifacts"]["audit_report_checker"]["exists"]
            and report["artifacts"]["audit_report_contract_test"]["exists"]
            and report["artifacts"]["audit_summary"]["exists"]
            and report["checks"]["audit_exit_code_test"]["pass"]
            and report["checks"]["audit_report_contract_test"]["pass"]
            and report["checks"]["result_metadata_numeric_test"]["pass"]
            and report["artifacts"]["audit_summary_test"]["exists"]
            and report["checks"]["audit_summary_test"]["pass"]
            and report["checks"]["remote_accuracy_resume_test"]["pass"]
        ),
    }
    report["requirements"] = requirements
    report["all_local_requirements_pass"] = all(bool(v) for v in requirements.values())
    report["cuda_reference_verified"] = (
        report["checks"]["dense_fmm_reference"].get("pass") is True and
        report["checks"]["dense_fmm_absorbing_reference"].get("pass") is True
    )

    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "all_local_requirements_pass": report["all_local_requirements_pass"],
        "cuda_reference_verified": report["cuda_reference_verified"],
        "cuda_toolchain_available": report["cuda_toolchain_available"],
        "cuda_runtime_ready": report["cuda_runtime_ready"],
    }, indent=2, ensure_ascii=False))
    return determine_exit_code(report, require_cuda_reference=args.require_cuda_reference)


if __name__ == "__main__":
    raise SystemExit(main())
