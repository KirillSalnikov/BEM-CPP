#!/usr/bin/env python3
"""Regression tests for the user-facing bem launcher."""

from pathlib import Path
import json
import runpy
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BEM = ROOT / "bem"


def invoke(*arguments: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(BEM), *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == expected, completed.stdout + completed.stderr
    return completed


def plan(*arguments: str) -> dict:
    options = list(arguments)
    if "--allow-memory-risk" not in options:
        options.append("--allow-memory-risk")
    completed = invoke(*options, "--dry-run", "--json")
    return json.loads(completed.stdout)


def command_value(data: dict, option: str) -> str:
    index = data["command"].index(option)
    return data["command"][index + 1]


def synthetic_result(
    path: Path,
    scale: float = 1.0,
    residual: float = 5e-6,
    theta: list[float] | None = None,
) -> None:
    theta = theta or [0.0, 180.0]
    mueller = [
        [[scale * (1.0 if i == j else 0.01) for _ in theta] for j in range(4)]
        for i in range(4)
    ]
    path.write_text(
        json.dumps(
            {
                "software_version": "test",
                "solver": "test_muller",
                "tolerance": 1e-5,
                "mbj": {"fmm_residual": residual},
                "physical": {
                    "parallel_fmm_residual": residual,
                    "trusted_cyclic_exact_geometry_used": False,
                    "theta_degrees": theta,
                    "mueller": mueller,
                    "amplitudes": {
                        "S1": [[-scale, 0.0] for _ in theta],
                        "S2": [[-scale, 0.0] for _ in theta],
                        "S3": [[0.0, 0.0] for _ in theta],
                        "S4": [[0.0, 0.0] for _ in theta],
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    profiles = json.loads(invoke("presets", "--json").stdout)
    assert set(profiles) == {
        "preview", "fast", "quick", "standard", "memory", "strict"
    }
    assert profiles["standard"]["tolerance"] == 1e-5
    assert profiles["strict"]["mixed_precision"] is False
    fast_explanation = json.loads(invoke("explain", "fast", "--json").stdout)
    assert fast_explanation["name"] == "fast"
    legacy_fast_explanation = json.loads(
        invoke("explain", "physical-fast", "--json").stdout
    )
    assert legacy_fast_explanation == fast_explanation

    preview = plan(
        "run", "--shape", "prism", "--ka", "80", "--ri", "1.3",
        "--quality", "preview", "--out", "/tmp/bem-frontend-preview-plan",
    )
    assert preview["inputs"]["refinement"] == 6
    assert preview["effective_parameters"]["maximum_iterations"] == 3
    assert preview["effective_parameters"]["pfft_inner_tolerance"] == 0.12
    assert preview["effective_parameters"]["pfft_outer_restart"] == 3
    assert preview["effective_parameters"]["final_residual_verification"] == (
        "projected_residual_only"
    )
    assert "--trust-final-projected-residual" in preview["command"]
    assert "--trust-cyclic-exact-geometry" not in preview["command"]
    assert "--no-checkpoint" in preview["command"]
    assert preview["runtime"]["environment"]["BEM_FMM_INTERIOR_FIRST"] == "1"
    assert preview["runtime"]["environment"][
        "BEM_FMM_L2P_PHASE_CACHE_FP16_MB"
    ] == "1024"
    invalid_preview = invoke(
        "run", "--shape", "prism", "--ka", "60", "--ri", "1.3",
        "--quality", "preview", "--dry-run", expected=2,
    )
    assert "validated only" in invalid_preview.stderr
    invalid_preview_average = invoke(
        "average", "--shape", "prism", "--ka", "80", "--ri", "1.3",
        "--quality", "preview", "--alpha", "16", "--dry-run", expected=2,
    )
    assert "validated only" in invalid_preview_average.stderr

    adaptive_fast = plan(
        "run", "--shape", "prism", "--sides", "7", "--aspect", "1.4",
        "--ka", "72", "--ri", "1.7", "--ref", "6",
        "--quality", "fast", "--out", "/tmp/bem-adaptive-fast-plan",
    )
    assert adaptive_fast["kind"] == "adaptive_fast_run"
    assert adaptive_fast["quality"] == "fast"
    assert adaptive_fast["validation_envelope"]["shape"] == "prism"
    assert adaptive_fast["validation_envelope"]["sides"] == 7
    assert adaptive_fast["validation_envelope"]["aspect"] == 1.4
    assert adaptive_fast["validation_envelope"]["adaptive_residual_levels"] == [
        4e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5,
    ]
    assert adaptive_fast["validation_envelope"]["fallback"] == "standard_1e-5"
    assert adaptive_fast["effective_parameters"]["adaptive_residual_levels"] == [
        4e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5,
    ]
    assert adaptive_fast["effective_parameters"]["resident_operator_reuse"] is True
    assert adaptive_fast["validation_envelope"]["operator_setup_count"] == 1
    assert "--adaptive-fast" in adaptive_fast["command"]
    assert "--trust-final-projected-residual" not in adaptive_fast["command"]
    assert "--trust-cyclic-exact-geometry" not in adaptive_fast["command"]

    small_fast = plan(
        "run", "--shape", "sphere", "--ka", "1", "--ri", "1.3",
        "--quality", "physical-fast", "--out", "/tmp/bem-small-fast-plan",
    )
    assert small_fast["quality"] == "fast"
    assert small_fast["kind"] == "adaptive_fast_run"
    assert small_fast["effective_parameters"]["solver"] == "fmm_mbj"
    assert small_fast["validation_envelope"]["operator_setup_count"] == 1
    invalid_fast_tolerance = invoke(
        "run", "--shape", "sphere", "--ka", "1", "--ri", "1.3",
        "--quality", "fast", "--tol", "1e-3", "--dry-run", expected=2,
    )
    assert "residual ladder" in invalid_fast_tolerance.stderr
    invalid_fast_average = invoke(
        "average", "--shape", "sphere", "--ka", "1", "--ri", "1.3",
        "--quality", "fast", "--alpha", "8", "--dry-run", expected=2,
    )
    assert "fixed-orientation" in invalid_fast_average.stderr
    invalid_fast_single = invoke(
        "run", "--shape", "sphere", "--ka", "1", "--ri", "1.3",
        "--quality", "fast", "--single-stage", "--dry-run", expected=2,
    )
    assert "multiple exact residual levels" in invalid_fast_single.stderr
    invalid_fast_angles = invoke(
        "run", "--shape", "sphere", "--ka", "1", "--ri", "1.3",
        "--quality", "fast", "--ntheta", "2", "--dry-run", expected=2,
    )
    assert "angular integrals" in invalid_fast_angles.stderr

    quick_two_stage = plan(
        "run", "--shape", "prism", "--ka", "60", "--ri", "1.3",
        "--quality", "quick", "--out", "/tmp/bem-quick-two-stage-plan",
    )
    assert quick_two_stage["kind"] == "profile_two_stage_suite"
    assert quick_two_stage["children"][1]["effective_parameters"][
        "tolerance"
    ] == 1e-3
    assert quick_two_stage["children"][1]["effective_parameters"][
        "solver"
    ] == "fmm_pfft_fgmres"
    quick_exact = quick_two_stage["children"][1]
    assert quick_exact["effective_parameters"]["ntheta"] == 181
    assert quick_exact["effective_parameters"]["final_residual_verification"] == (
        "exact_banded_fmm_operator_residual"
    )
    assert quick_exact["effective_parameters"]["polarization_mode"] == (
        "verified_regular_prism_symmetry_with_correction"
    )
    assert "--trust-cyclic-exact-geometry" not in quick_exact["command"]

    standard_two_stage = plan(
        "run", "--shape", "prism", "--ka", "60", "--ri", "1.3",
        "--quality", "standard", "--ref", "6",
        "--out", "/tmp/bem-standard-two-stage-plan",
    )
    assert standard_two_stage["kind"] == "profile_two_stage_suite"
    assert standard_two_stage["children"][1]["effective_parameters"][
        "tolerance"
    ] == 1e-5
    assert standard_two_stage["children"][1]["runtime"]["environment"][
        "BEM_FMM_L2P_FP32"
    ] == "1"
    assert standard_two_stage["children"][1]["runtime"]["environment"][
        "BEM_FMM_PAIR_CURRENTS"
    ] == "0"

    memory_two_stage = plan(
        "run", "--shape", "prism", "--ka", "80", "--ri", "1.3",
        "--quality", "memory", "--ref", "6",
        "--out", "/tmp/bem-memory-two-stage-plan",
    )
    assert memory_two_stage["kind"] == "profile_two_stage_suite"
    assert memory_two_stage["children"][1]["effective_parameters"][
        "tolerance"
    ] == 1e-5
    assert memory_two_stage["children"][1]["estimate"][
        "gpu_memory_gib"
    ] == 12.02
    memory_exact = memory_two_stage["children"][1]
    assert memory_exact["effective_parameters"]["ntheta"] == 181
    assert memory_exact["effective_parameters"]["final_residual_verification"] == (
        "exact_banded_fmm_operator_residual"
    )
    assert memory_exact["effective_parameters"]["polarization_mode"] == (
        "verified_regular_prism_symmetry_with_correction"
    )
    assert memory_exact["runtime"]["environment"]["BEM_FMM_PAIR_CURRENTS"] == "0"
    assert "--trust-cyclic-exact-geometry" not in memory_exact["command"]

    standard_single_stage = plan(
        "run", "--shape", "prism", "--ka", "60", "--ri", "1.3",
        "--quality", "standard", "--ref", "6", "--single-stage",
        "--out", "/tmp/bem-standard-single-stage-plan",
    )
    assert standard_single_stage["kind"] == "run"

    generalized_two_stage = plan(
        "run", "--shape", "prism", "--sides", "7", "--aspect", "1.4",
        "--ka", "72", "--ri", "1.7", "--ref", "6",
        "--quality", "standard", "--allow-memory-risk",
        "--out", "/tmp/bem-generalized-two-stage-plan",
    )
    assert generalized_two_stage["kind"] == "profile_two_stage_suite"
    assert generalized_two_stage["validation_envelope"]["selection"] == (
        "automatic_by_mesh_and_electrical_density"
    )
    assert generalized_two_stage["validation_envelope"]["shape"] == "prism"
    assert "saved_adda_ocl_fp32_dpl15_wall_time_s" not in (
        generalized_two_stage["validation_envelope"]
    )

    generalized_sphere = plan(
        "run", "--shape", "sphere", "--ka", "60", "--ri", "1.5",
        "--ref", "6", "--quality", "quick",
        "--out", "/tmp/bem-generalized-sphere-plan",
    )
    assert generalized_sphere["kind"] == "profile_two_stage_suite"
    sphere_exact = generalized_sphere["children"][1]
    assert "--trust-cyclic-exact-geometry" not in sphere_exact["command"]
    assert sphere_exact["effective_parameters"]["polarization_mode"] == (
        "independent"
    )

    moderate_frequency = plan(
        "run", "--shape", "prism", "--ka", "20", "--ri", "1.3",
        "--quality", "quick", "--out", "/tmp/bem-moderate-frequency-plan",
    )
    assert moderate_frequency["kind"] == "run"

    standard = plan(
        "run", "--shape", "prism", "--ka", "10", "--ri", "1.3",
        "--out", "/tmp/bem-frontend-standard-plan",
    )
    assert standard["inputs"]["refinement"] == 4
    assert standard["inputs"]["sides"] == 6
    assert standard["estimate"]["system_dofs"] == 50688
    assert standard["quality"] == "standard"
    assert "--pfft-fgmres" in standard["command"]
    assert "--mbj-only" in standard["command"]
    assert "hdiv" in standard["command"]
    assert standard["effective_parameters"]["max_leaf"] == 32
    assert standard["effective_parameters"]["solver"] == "fmm_pfft_fgmres"
    assert standard["effective_parameters"]["pfft_outer_restart"] == 40
    assert standard["effective_parameters"]["pfft_inner_tolerance"] == 0.04
    assert command_value(standard, "--pfft-outer-restart") == "40"
    assert standard["runtime"]["environment"] == {
        "BEM_FMM_FLAT_NEAR_SOURCES": "0",
        "BEM_FMM_L2P_FP32": "1",
        "BEM_FMM_PAIR_CURRENTS": "0",
        "BEM_FMM_PHASE_CACHE": "0",
        "BEM_MIXED_ITERATIVE_REFINEMENT": "1",
    }
    assert standard["effective_parameters"]["strict_residual_refinement"] is True
    assert standard["effective_parameters"]["l2p_precision"] == (
        "fp32_krylov_fp64_restart_residual"
    )
    assert standard["effective_parameters"]["krylov_operator_policy"] == (
        "mixed_fmm_with_fp64_restart_residual"
    )
    same_operator = plan(
        "run", "--shape", "prism", "--ka", "10", "--ri", "1.3",
        "--out", "/tmp/bem-frontend-same-operator-plan",
    )
    assert same_operator["cache_directory"] == standard["cache_directory"]
    assert command_value(
        same_operator, "--near-correction-cache"
    ) == command_value(standard, "--near-correction-cache")
    different_operator = plan(
        "run", "--shape", "prism", "--ka", "10.25", "--ri", "1.3",
        "--ref", "5", "--out", "/tmp/bem-frontend-different-operator-plan",
    )
    assert different_operator["cache_directory"] != standard["cache_directory"]
    different_mbj = plan(
        "run", "--shape", "prism", "--ka", "10", "--ri", "1.3",
        "--mbj-nodes", "72", "--mbj-overlap", "4",
        "--out", "/tmp/bem-frontend-different-mbj-plan",
    )
    assert different_mbj["cache_directory"] == standard["cache_directory"]
    assert command_value(
        different_mbj, "--near-correction-cache"
    ) == command_value(standard, "--near-correction-cache")
    assert command_value(
        different_mbj, "--mbj-cache"
    ) != command_value(standard, "--mbj-cache")

    memory = plan(
        "run", "--shape", "prism", "--ka", "60", "--ri", "1.3",
        "--quality", "memory", "--ref", "6", "--single-stage",
        "--out", "/tmp/bem-frontend-memory-plan",
    )
    assert memory["quality"] == "memory"
    assert memory["effective_parameters"]["tolerance"] == 1e-5
    assert memory["effective_parameters"]["ntheta"] == 181
    assert memory["inputs"]["refinement"] == 6
    assert memory["effective_parameters"]["solver"] == "fmm_pfft_fgmres"
    assert memory["effective_parameters"]["pfft_inner_tolerance"] == 0.08
    assert memory["effective_parameters"]["memory_policy"] == (
        "sequential_currents_and_recompute_caches"
    )
    assert memory["runtime"]["environment"] == {
        "BEM_FMM_FLAT_NEAR_SOURCES": "0",
        "BEM_FMM_L2P_FP32": "1",
        "BEM_FMM_PAIR_CURRENTS": "0",
        "BEM_FMM_PHASE_CACHE": "0",
        "BEM_MIXED_ITERATIVE_REFINEMENT": "1",
    }
    standard_same_mesh = plan(
        "run", "--shape", "prism", "--ka", "60", "--ri", "1.3",
        "--quality", "standard", "--ref", "6", "--single-stage",
        "--allow-memory-risk",
        "--out", "/tmp/bem-frontend-standard-same-mesh-plan",
    )
    assert (
        memory["estimate"]["gpu_memory_gib"] <
        standard_same_mesh["estimate"]["gpu_memory_gib"]
    )

    small = plan(
        "run", "--shape", "sphere", "--ka", "1", "--ri", "1.3",
        "--out", "/tmp/bem-frontend-small-plan",
    )
    assert small["effective_parameters"]["max_leaf"] == 128
    assert small["inputs"]["refinement"] == 2
    assert small["estimate"]["system_dofs"] == 2568
    assert small["effective_parameters"]["solver"] == "fmm_mbj"
    assert "--pfft-fgmres" not in small["command"]

    larger = plan(
        "run", "--shape", "sphere", "--ka", "10", "--ri", "1.3",
        "--out", "/tmp/bem-frontend-large-mesh-plan",
    )
    finer = plan(
        "run", "--shape", "sphere", "--ka", "10", "--ri", "1.3",
        "--points-per-wavelength", "16",
        "--out", "/tmp/bem-frontend-finer-mesh-plan",
    )
    assert larger["inputs"]["refinement"] == 4
    assert finer["inputs"]["refinement"] == 5
    assert larger["inputs"]["refinement_selection"] == "automatic"
    assert larger["inputs"]["estimated_points_per_shortest_wavelength"] >= 8
    assert larger["inputs"]["estimated_points_per_interior_wavelength"] == (
        larger["inputs"]["estimated_points_per_shortest_wavelength"]
    )

    ka30_prism = plan(
        "run", "--shape", "prism", "--sides", "6", "--aspect", "1",
        "--ka", "30", "--ri", "1.3", "--quality", "standard",
        "--single-stage", "--out", "/tmp/bem-frontend-ka30-prism-plan",
    )
    assert ka30_prism["inputs"]["refinement"] == 5
    assert ka30_prism["estimate"]["system_dofs"] == 202752
    assert ka30_prism["inputs"]["estimated_points_per_shortest_wavelength"] >= 8
    assert ka30_prism["effective_parameters"]["polarization_mode"] == (
        "verified_prism_symmetry_with_correction"
    )
    assert "--trust-cyclic-exact-geometry" not in ka30_prism["command"]

    low_index = plan(
        "run", "--shape", "sphere", "--ka", "10", "--ri", "0.8",
        "--out", "/tmp/bem-frontend-low-index-plan",
    )
    high_index = plan(
        "run", "--shape", "sphere", "--ka", "10", "--ri", "2",
        "--out", "/tmp/bem-frontend-high-index-plan",
    )
    assert low_index["inputs"]["refinement"] == 3
    assert high_index["inputs"]["refinement"] == 4
    assert low_index["inputs"]["shortest_wavelength_refractive_factor"] == 1
    assert high_index["inputs"]["shortest_wavelength_refractive_factor"] == 2

    cube_estimate = plan(
        "run", "--shape", "cube", "--ka", "2", "--ri", "1.3",
        "--quality", "quick", "--out", "/tmp/bem-frontend-cube-estimate",
    )
    prism_estimate = plan(
        "run", "--shape", "prism", "--sides", "6", "--aspect", "1",
        "--ka", "3", "--ri", "1.5", "--quality", "quick",
        "--out", "/tmp/bem-frontend-prism-estimate",
    )
    assert cube_estimate["estimate"]["system_dofs"] == 1152
    assert prism_estimate["estimate"]["system_dofs"] == 3168

    overrides = plan(
        "run", "--shape", "prism", "--ka", "25", "--ri", "1.3",
        "--ref", "3", "--solver", "fmm", "--tol", "2e-6",
        "--quad", "13", "--duffy-order", "7", "--digits", "6",
        "--ntheta", "91", "--max-iters", "777", "--gmres-restart", "64",
        "--max-leaf", "48", "--mbj-nodes", "72", "--mbj-overlap", "4",
        "--out", "/tmp/bem-frontend-overrides-plan",
    )
    assert overrides["inputs"]["refinement"] == 3
    assert overrides["inputs"]["refinement_selection"] == "explicit"
    assert overrides["effective_parameters"]["solver"] == "fmm_mbj"
    assert "--pfft-fgmres" not in overrides["command"]
    assert command_value(overrides, "--tol") == "2.0e-06"
    assert command_value(overrides, "--quad") == "13"
    assert command_value(overrides, "--digits") == "6"
    assert command_value(overrides, "--ntheta") == "91"
    assert command_value(overrides, "--mbj-nodes") == "72"
    assert command_value(overrides, "--mbj-cache").endswith(
        "/mbj_n72_o4.cache"
    )
    for option in ("--tol", "--quad", "--digits", "--ntheta", "--mbj-nodes"):
        assert overrides["command"].count(option) == 1

    pfft_overrides = plan(
        "run", "--shape", "prism", "--ka", "10", "--ri", "1.3",
        "--solver", "pfft", "--pfft-inner-tol", "0.08",
        "--pfft-inner-iters", "12", "--pfft-outer-restart", "40",
        "--pfft-order", "3", "--pfft-correction-radius", "1.5",
        "--pfft-grid-safety", "0.9",
        "--out", "/tmp/bem-frontend-pfft-overrides-plan",
    )
    assert command_value(pfft_overrides, "--pfft-inner-tol") == "0.08"
    assert command_value(pfft_overrides, "--pfft-inner-iters") == "12"
    assert command_value(pfft_overrides, "--pfft-outer-restart") == "40"
    assert command_value(pfft_overrides, "--pfft-order") == "3"
    assert command_value(pfft_overrides, "--pfft-correction-radius") == "1.5"
    assert command_value(pfft_overrides, "--pfft-grid-safety") == "0.9"
    assert pfft_overrides["effective_parameters"]["pfft_order"] == 3

    below_threshold = plan(
        "run", "--shape", "prism", "--ka", "9.99", "--ri", "1.3",
        "--out", "/tmp/bem-frontend-threshold-plan",
    )
    assert below_threshold["effective_parameters"]["solver"] == "fmm_mbj"

    average = plan(
        "average", "--shape", "prism", "--ka", "25", "--ri", "1.3",
        "--quality", "quick", "--alpha", "256", "--beta", "4", "--gamma", "4",
        "--out", "/tmp/bem-frontend-average-plan",
    )
    orient = average["command"].index("--orient-average")
    assert average["command"][orient + 1:orient + 4] == ["256", "4", "4"]
    symmetry = average["command"].index("--orient-symmetry-order")
    assert average["command"][symmetry + 1] == "6"
    assert "--orient-adaptive" not in average["command"]
    assert average["effective_parameters"]["orientation"]["mode"] == "fixed"
    assert average["effective_parameters"]["orientation"][
        "operator_setup_reused"
    ] is True
    assert average["effective_parameters"]["orientation"][
        "resume_granularity"
    ] == "completed_base_orientation"
    assert "BEM_FMM_BANDED_SPLIT_DEPTH" not in average["runtime"]["environment"]
    assert average["estimate"]["checkpoint_disk_gib"] > 0.0

    large_average = plan(
        "average", "--shape", "prism", "--ka", "60", "--ri", "1.3",
        "--ref", "6", "--quality", "quick", "--alpha", "4",
        "--beta", "1", "--gamma", "1", "--fixed-grid",
        "--out", "/tmp/bem-large-average-plan",
    )
    assert large_average["kind"] == "average"
    assert "BEM_FMM_BANDED_SPLIT_DEPTH" not in (
        large_average["runtime"]["environment"]
    )
    assert "--orient-paired-gpu-gmres" in large_average["command"]

    adaptive_average = plan(
        "average", "--shape", "sphere", "--ka", "3", "--ri", "1.3",
        "--out", "/tmp/bem-frontend-adaptive-average-plan",
    )
    adaptive_index = adaptive_average["command"].index("--orient-adaptive")
    assert adaptive_average["command"][adaptive_index + 1:adaptive_index + 3] == [
        "2", "4"
    ]
    assert "--orient-parts-dir" in adaptive_average["command"]
    assert adaptive_average["effective_parameters"]["orientation"]["mode"] == "adaptive"
    assert "--orient-dihedral-symmetry" not in adaptive_average["command"]
    assert adaptive_average["effective_parameters"]["orientation"][
        "dihedral_symmetry_reuse"
    ] is False
    assert adaptive_average["runtime"]["environment"] == {
        "BEM_FMM_FLAT_NEAR_SOURCES": "0",
        "BEM_FMM_L2P_FP32": "0",
        "BEM_FMM_PAIR_CURRENTS": "1",
        "BEM_FMM_PHASE_CACHE": "0",
        "BEM_MULLER_GPU_ASSEMBLY": "1",
        "BEM_MIXED_ITERATIVE_REFINEMENT": "1",
    }
    assert adaptive_average["effective_parameters"]["strict_residual_refinement"] is True
    assert adaptive_average["effective_parameters"]["l2p_precision"] == "fp64"

    adaptive_overrides = plan(
        "average", "--shape", "sphere", "--ka", "3", "--ri", "1.3",
        "--quality", "quick", "--adaptive-levels", "2", "5",
        "--adaptive-m11-tol", "0.004", "--adaptive-integral-tol", "0.005",
        "--adaptive-component-tol", "0.03", "--orient-warm-max-angle", "12",
        "--orient-recycle-rank", "3", "--orient-zero-start",
        "--out", "/tmp/bem-frontend-adaptive-overrides-plan",
    )
    adaptive = adaptive_overrides["effective_parameters"]["orientation"]
    assert adaptive["minimum_level"] == 2 and adaptive["maximum_level"] == 5
    assert adaptive["m11_tolerance"] == 0.004
    assert adaptive["integral_tolerance"] == 0.005
    assert adaptive["component_tolerance"] == 0.03
    assert adaptive["warm_start"] is False
    assert adaptive["warm_start_max_angle_degrees"] == 12
    assert adaptive["recycle_rank"] == 3

    quick_adaptive = plan(
        "average", "--shape", "sphere", "--ka", "3", "--ri", "1.3",
        "--quality", "quick", "--out", "/tmp/bem-frontend-quick-adaptive-plan",
    )
    assert "BEM_MIXED_ITERATIVE_REFINEMENT" not in quick_adaptive["runtime"]["environment"]
    assert quick_adaptive["effective_parameters"]["l2p_precision"] == "fp32"

    prism_adaptive = plan(
        "average", "--shape", "prism", "--sides", "8",
        "--ka", "2", "--ri", "1.3", "--quality", "quick",
        "--out", "/tmp/bem-frontend-prism-adaptive-plan",
    )
    assert "--orient-dihedral-symmetry" in prism_adaptive["command"]
    assert prism_adaptive["effective_parameters"]["orientation"][
        "dihedral_symmetry_reuse"
    ] is True
    assert prism_adaptive["effective_parameters"]["orientation"][
        "maximum_solved_base_orientations"
    ] == 37

    odd_alpha_prism = plan(
        "average", "--shape", "prism", "--sides", "8",
        "--ka", "2", "--ri", "1.3", "--quality", "quick", "--alpha", "7",
        "--out", "/tmp/bem-frontend-odd-alpha-prism-plan",
    )
    assert "--orient-dihedral-symmetry" not in odd_alpha_prism["command"]
    assert odd_alpha_prism["effective_parameters"]["orientation"][
        "maximum_solved_base_orientations"
    ] == 72

    high_contrast_quick = plan(
        "average", "--shape", "cube", "--ka", "5", "--ri", "2.5",
        "--quality", "quick", "--out", "/tmp/bem-frontend-high-contrast-quick",
    )
    high_orientation = high_contrast_quick["effective_parameters"]["orientation"]
    assert high_orientation["minimum_level"] == 1
    assert high_orientation["maximum_level"] == 4
    assert high_contrast_quick["effective_parameters"]["maximum_iterations"] == 400
    assert high_contrast_quick["effective_parameters"][
        "automatic_high_contrast_adjustments"
    ] == {
        "extended_iterations": True,
        "extended_angular_levels": True,
        "larger_mbj_blocks": False,
    }

    high_contrast_standard = plan(
        "average", "--shape", "cube", "--ka", "5", "--ri", "2",
        "--quality", "standard", "--out", "/tmp/bem-frontend-high-contrast-standard",
    )
    standard_orientation = high_contrast_standard["effective_parameters"]["orientation"]
    assert standard_orientation["minimum_level"] == 2
    assert standard_orientation["maximum_level"] == 5
    assert high_contrast_standard["effective_parameters"][
        "automatic_high_contrast_adjustments"
    ]["extended_angular_levels"] is True
    assert high_contrast_standard["effective_parameters"]["mbj_nodes"] == 50

    very_high_contrast_standard = plan(
        "average", "--shape", "prism", "--sides", "8", "--aspect", "0.5",
        "--ka", "4", "--ri", "2.5", "--quality", "standard",
        "--out", "/tmp/bem-frontend-very-high-contrast-standard",
    )
    assert very_high_contrast_standard["effective_parameters"]["mbj_nodes"] == 100
    assert very_high_contrast_standard["effective_parameters"][
        "automatic_high_contrast_adjustments"
    ]["larger_mbj_blocks"] is True

    explicit_standard_levels = plan(
        "average", "--shape", "cube", "--ka", "5", "--ri", "2",
        "--quality", "standard", "--adaptive-levels", "2", "4",
        "--out", "/tmp/bem-frontend-explicit-high-contrast-standard",
    )
    assert explicit_standard_levels["effective_parameters"]["orientation"][
        "maximum_level"
    ] == 4
    assert explicit_standard_levels["effective_parameters"][
        "automatic_high_contrast_adjustments"
    ]["extended_angular_levels"] is False

    conflict = invoke(
        "average", "--shape", "sphere", "--ka", "3", "--ri", "1.3",
        "--beta", "8", "--adaptive-levels", "1", "3", "--dry-run",
        expected=2,
    )
    assert "cannot be combined" in conflict.stderr

    standard_average = plan(
        "average", "--shape", "prism", "--ka", "10", "--ri", "1.3",
        "--alpha", "8", "--beta", "4", "--gamma", "4",
        "--out", "/tmp/bem-frontend-standard-average-plan",
    )
    assert "--pfft-fgmres" in standard_average["command"]
    assert "--orient-paired-gpu-gmres" not in standard_average["command"]
    assert "--trust-cyclic-exact-geometry" not in standard_average["command"]

    standard_prism = plan(
        "run", "--shape", "prism", "--ka", "10", "--ri", "1.3",
        "--out", "/tmp/bem-frontend-standard-prism-plan",
    )
    assert "--cyclic-polarization" in standard_prism["command"]
    assert "--cyclic-exact-geometry" in standard_prism["command"]
    assert "--trust-cyclic-exact-geometry" not in standard_prism["command"]
    assert standard_prism["effective_parameters"]["polarization_mode"] == (
        "verified_prism_symmetry_with_correction"
    )

    large_standard_prism = plan(
        "run", "--shape", "prism", "--ka", "111", "--ri", "1.3",
        "--ref", "6", "--single-stage",
        "--out", "/tmp/bem-frontend-standard-ka111-plan",
    )
    assert large_standard_prism["runtime"]["environment"][
        "BEM_FMM_STRICT_PAIR_WORKSPACE"
    ] == "0"
    assert "--trust-cyclic-exact-geometry" not in (
        large_standard_prism["command"]
    )

    independent_prism = plan(
        "run", "--shape", "prism", "--ka", "10", "--ri", "1.3",
        "--independent-polarizations",
        "--out", "/tmp/bem-frontend-independent-prism-plan",
    )
    assert "--cyclic-polarization" not in independent_prism["command"]
    assert independent_prism["effective_parameters"]["polarization_mode"] == (
        "independent"
    )

    strict = plan(
        "run", "--shape", "sphere", "--ka", "1", "--ri", "1.3",
        "--quality", "strict", "--out", "/tmp/bem-frontend-strict-plan",
    )
    assert strict["kind"] == "strict_suite"
    assert [child["inputs"]["refinement"] for child in strict["children"]] == [2, 3]
    assert [child["inputs"]["refinement_selection"] for child in strict["children"]] == [
        "automatic", "strict_fine"
    ]
    assert all("--fmm-near-fp64" in child["command"] for child in strict["children"])
    assert all("--pfft-fgmres" not in child["command"] for child in strict["children"])

    strict_normal = plan(
        "run", "--shape", "prism", "--ka", "10", "--ri", "1.5",
        "--quality", "strict", "--allow-memory-risk",
        "--out", "/tmp/bem-frontend-strict-normal-plan",
    )
    assert all("--pfft-fgmres" in child["command"] for child in strict_normal["children"])
    assert all(
        child["effective_parameters"]["solver"] == "fmm_pfft_fgmres"
        for child in strict_normal["children"]
    )
    assert all(
        "--trust-cyclic-exact-geometry" not in child["command"]
        for child in strict_normal["children"]
    )

    strict_average = plan(
        "average", "--shape", "prism", "--ka", "10", "--ri", "1.5",
        "--quality", "strict", "--allow-memory-risk",
        "--out", "/tmp/bem-frontend-strict-average-plan",
    )
    for child in strict_average["children"]:
        adaptive_index = child["command"].index("--orient-adaptive")
        assert child["command"][adaptive_index + 1:adaptive_index + 3] == ["2", "5"]
        assert "--no-orient-paired-gpu-gmres" in child["command"]

    missing_obj = invoke(
        "run", "--shape", "obj", "--obj", "/dev/null", "--ka", "1", "--ri", "1.3",
        "--dry-run", expected=2,
    )
    assert "specify --ref" in missing_obj.stderr

    with tempfile.TemporaryDirectory(prefix="bem-frontend-test.") as directory:
        root = Path(directory)
        launcher = runpy.run_path(str(BEM))
        stability = launcher["observable_stability_metrics"]
        stable_previous = root / "stable-previous.json"
        stable_current = root / "stable-current.json"
        unstable_current = root / "unstable-current.json"
        synthetic_result(
            stable_previous, scale=1.0, theta=[0.0, 90.0, 180.0]
        )
        synthetic_result(
            stable_current, scale=1.0005, theta=[0.0, 90.0, 180.0]
        )
        synthetic_result(
            unstable_current, scale=1.01, theta=[0.0, 90.0, 180.0]
        )
        stable_metrics = stability(stable_previous, stable_current)
        unstable_metrics = stability(stable_previous, unstable_current)
        assert stable_metrics["passes"] is True
        assert stable_metrics["forward_m11_relative_difference"] < 1e-3
        assert stable_metrics["integrated_m11_relative_difference"] < 1e-3
        assert unstable_metrics["passes"] is False

        fake_solver = root / "fake_muller_solver.py"
        fake_solver.write_text(
            """#!/usr/bin/env python3
import json
from pathlib import Path
import sys

def option(name, default=None):
    if name not in sys.argv:
        return default
    return sys.argv[sys.argv.index(name) + 1]

tolerance = float(option("--tol", "1e-5"))
ri = float(option("--ri", "1.3"))
theta = [float(value) for value in range(181)]
levels = [4e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
level = min(range(len(levels)), key=lambda i: abs(levels[i] - tolerance))
scale = 1.0 + ((0.0004 * level) if ri < 1.305 else (0.01 * level))
mueller = [
    [[scale * (1.0 if row == column else 0.01) for _ in theta]
     for column in range(4)]
    for row in range(4)
]
residual = 0.8 * tolerance
result = {
    "software_version": "test",
    "solver": "fake_muller",
    "tolerance": tolerance,
    "mbj": {"fmm_residual": residual},
    "physical": {
        "parallel_fmm_residual": residual,
        "trusted_cyclic_exact_geometry_used": False,
        "theta_degrees": theta,
        "mueller": mueller,
        "amplitudes": {
            "S1": [[-scale, 0.0] for _ in theta],
            "S2": [[-scale, 0.0] for _ in theta],
            "S3": [[0.0, 0.0] for _ in theta],
            "S4": [[0.0, 0.0] for _ in theta],
        },
    },
}
output = Path(option("--out"))
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result), encoding="utf-8")
""",
            encoding="utf-8",
        )
        fake_solver.chmod(0o755)

        fast_early = root / "fast-early"
        invoke(
            "run", "--shape", "sphere", "--ka", "1", "--ri", "1.3",
            "--ref", "2", "--quality", "fast", "--binary", str(fake_solver),
            "--out", str(fast_early), "--no-build", "--allow-memory-risk",
        )
        early_summary = json.loads(
            (fast_early / "adaptive_fast_summary.json").read_text()
        )
        assert early_summary["selection"] == "observable_stability_early_stop"
        assert early_summary["selected_residual_target"] == 3e-4
        assert early_summary["completed_exact_levels"] == 3
        assert (fast_early / "result.json").is_file()

        fast_fallback = root / "fast-fallback"
        invoke(
            "run", "--shape", "sphere", "--ka", "1", "--ri", "1.31",
            "--ref", "2", "--quality", "fast", "--binary", str(fake_solver),
            "--out", str(fast_fallback), "--no-build", "--allow-memory-risk",
        )
        fallback_summary = json.loads(
            (fast_fallback / "adaptive_fast_summary.json").read_text()
        )
        assert fallback_summary["selection"] == "standard_residual_fallback"
        assert fallback_summary["selected_residual_target"] == 1e-5
        assert fallback_summary["completed_exact_levels"] == 6

        obj = root / "shape.obj"
        obj.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
            encoding="ascii",
        )
        obj_first = plan(
            "run", "--shape", "obj", "--obj", str(obj), "--ref", "1",
            "--ka", "1", "--ri", "1.3", "--out", str(root / "obj-first"),
        )
        obj.write_text(
            "v 0 0 0\nv 2 0 0\nv 0 1 0\nf 1 2 3\n",
            encoding="ascii",
        )
        obj_second = plan(
            "run", "--shape", "obj", "--obj", str(obj), "--ref", "1",
            "--ka", "1", "--ri", "1.3", "--out", str(root / "obj-second"),
        )
        assert obj_first["cache_directory"] != obj_second["cache_directory"]
        result = root / "result.json"
        reference = root / "reference.json"
        synthetic_result(result)
        synthetic_result(reference)
        report = json.loads(
            invoke("validate", str(result), "--reference", str(reference), "--json").stdout
        )
        assert report["comparison"]["passes"] is True
        assert report["comparison"]["reference_interpolated"] is False
        preview_data = json.loads(result.read_text(encoding="utf-8"))
        preview_data["pfft_fgmres"] = {"fmm_residual_verified": False}
        result.write_text(json.dumps(preview_data), encoding="utf-8")
        projected = json.loads(invoke("validate", str(result), "--json").stdout)
        assert projected["residual_verified"] is False
        assert any("Krylov projection" in warning for warning in projected["warnings"])
        synthetic_result(result, theta=[0.0, 90.0, 180.0])
        interpolated = json.loads(
            invoke("validate", str(result), "--reference", str(reference), "--json").stdout
        )
        assert interpolated["comparison"]["passes"] is True
        assert interpolated["comparison"]["reference_interpolated"] is True
        assert interpolated["comparison"]["comparison_angles"] == 2
        assert interpolated["comparison"]["interpolation_target"] == (
            "candidate_to_reference_grid"
        )
        synthetic_result(result, residual=3e-5)
        failed = invoke("validate", str(result), "--json", expected=1)
        assert "exceeds 2*tolerance" in failed.stdout
        synthetic_result(result)
        document = json.loads(result.read_text(encoding="utf-8"))
        document["adaptive"] = {"enabled": True, "converged": False}
        result.write_text(json.dumps(document), encoding="utf-8")
        adaptive_failed = invoke("validate", str(result), "--json", expected=1)
        assert "without satisfying convergence" in adaptive_failed.stdout
        synthetic_result(result)
        document = json.loads(result.read_text(encoding="utf-8"))
        document["adaptive_fast_enabled"] = True
        document["adaptive_fast"] = {
            "selected_residual_target": 1e-4,
            "completed_levels": 2,
            "consecutive_stable_levels": 1,
            "comparisons": [],
        }
        document["tolerance"] = 1e-4
        result.write_text(json.dumps(document), encoding="utf-8")
        fast_failed = invoke("validate", str(result), "--json", expected=1)
        assert "two independent physical stability checks" in fast_failed.stdout
        synthetic_result(result)
        document = json.loads(result.read_text(encoding="utf-8"))
        document["physical"]["trusted_cyclic_exact_geometry_used"] = True
        result.write_text(json.dumps(document), encoding="utf-8")
        trusted_failed = invoke("validate", str(result), "--json", expected=1)
        assert "without a direct operator residual check" in trusted_failed.stdout

    print("bem frontend: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
