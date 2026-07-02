#!/usr/bin/env python3
"""Validate the structural contract of an audit_1_6 JSON report."""

import argparse
import json
from pathlib import Path
from typing import List, Set


REQUIREMENTS = {
    "1_cpu_reference_pmchwt",
    "2_mueller_sign_audit",
    "3_singular_near_singular_hooks",
    "4_mesh_quality_gate",
    "5_operator_architecture",
    "6_python_job_api",
    "cuda_build_runtime_gate",
}

CHECKS = {
    "python_compile",
    "shell_syntax",
    "cuda_toolchain_detect",
    "python_job_api",
    "audit_exit_code_test",
    "audit_report_contract_test",
    "accuracy_matrix_selection_test",
    "result_metadata_numeric_test",
    "accuracy_refinement_planner_test",
    "remote_accuracy_resume_test",
    "audit_summary_test",
    "cuda_toolchain_detection_test",
    "gpu_power_monitor_summary",
    "mueller_self_test",
    "mie_mueller_symmetry_test",
    "mueller_physical_gate_test",
    "operator_block_self_test",
    "cpu_pmchwt_centroid",
    "cpu_pmchwt_system_contract_test",
    "operator_config_cpp",
    "operator_config_cpp_run",
    "near_singular_routing_test",
    "hmatrix_memory_audit_test",
    "dense_fmm_reference",
    "dense_fmm_absorbing_reference",
}

ARTIFACTS = {
    "mesh_quality_gate",
    "operator_config",
    "python_api",
    "local_audits",
    "audit_summary",
    "dense_fmm_reference",
    "cuda_reference_runner",
    "cuda_fmm_builder",
    "cuda_toolchain_detector",
    "production_queue",
    "result_metadata_checker",
    "production_deploy",
    "production_fetch",
    "ipmi_power_control",
    "queue_live_status",
    "queue_watch_once",
    "power_watch",
    "gpu_power_monitor_summary",
    "production_resume",
    "production_resume_cases",
    "remote_production_resume_cases",
    "production_case_runner",
    "production_guarded_case_runner",
    "production_refinement_planner",
    "gpu_power_limit",
    "bmc_access",
    "near_singular_audit",
    "hmatrix_memory_audit",
    "control_docs",
    "cuda_env_file",
    "cuda_runtime_docs",
    "cuda_env_pins",
    "make_audit_targets",
    "singular_corrections",
    "host_operator_cpp_test",
    "python_job_api_test",
    "audit_exit_code_test",
    "audit_report_checker",
    "audit_report_contract_test",
    "audit_summary_test",
}

METADATA = {
    "created_utc",
    "command",
    "python",
    "platform",
    "git_commit",
    "git_dirty",
    "git_dirty_count",
    "git_dirty_sample",
}


def missing_keys(container: dict, required: Set[str], prefix: str) -> List[str]:
    return [f"{prefix}.{key}" for key in sorted(required - set(container))]


def validate_report(data: dict) -> List[str]:
    errors: List[str] = []
    top = {
        "root",
        "metadata",
        "checks",
        "artifacts",
        "requirements",
        "binary",
        "all_local_requirements_pass",
        "cuda_toolchain_available",
        "cuda_runtime_ready",
        "cuda_runtime",
        "cuda_reference_verified",
    }
    errors.extend(missing_keys(data, top, "report"))

    metadata = data.get("metadata", {})
    checks = data.get("checks", {})
    artifacts = data.get("artifacts", {})
    requirements = data.get("requirements", {})

    errors.extend(missing_keys(metadata, METADATA, "metadata"))
    errors.extend(missing_keys(checks, CHECKS, "checks"))
    errors.extend(missing_keys(artifacts, ARTIFACTS, "artifacts"))
    errors.extend(missing_keys(requirements, REQUIREMENTS, "requirements"))

    for key in REQUIREMENTS:
        if key in requirements and not isinstance(requirements[key], bool):
            errors.append(f"requirements.{key} must be boolean")

    for key in ("all_local_requirements_pass", "cuda_toolchain_available",
                "cuda_runtime_ready", "cuda_reference_verified"):
        if key in data and not isinstance(data[key], bool):
            errors.append(f"report.{key} must be boolean")

    if metadata and not isinstance(metadata.get("git_dirty_count"), int):
        errors.append("metadata.git_dirty_count must be integer")
    if metadata and not isinstance(metadata.get("git_dirty_sample"), list):
        errors.append("metadata.git_dirty_sample must be list")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()

    data = json.loads(args.report.read_text())
    errors = validate_report(data)
    if errors:
        print("audit_1_6 report contract: fail")
        for error in errors:
            print(f"  {error}")
        return 2
    print("audit_1_6 report contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
