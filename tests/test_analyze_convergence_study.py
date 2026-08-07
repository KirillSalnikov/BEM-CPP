#!/usr/bin/env python3
"""Numerical checks for convergence-study error metrics."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_convergence_study",
    ROOT / "scripts" / "analyze_convergence_study.py",
)
assert SPEC and SPEC.loader
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def main() -> int:
    theta = np.linspace(0.0, 180.0, 19)
    reference = np.zeros((4, 4, theta.size))
    reference[0, 0] = 1.0 + 0.2 * np.cos(np.deg2rad(theta))
    reference[1, 1] = 0.5 * reference[0, 0]
    identical = ANALYSIS.compare_mueller(theta, reference, reference)
    assert max(abs(value) for key, value in identical.items() if "ratio" not in key) < 1.0e-14
    assert identical["m11_forward_ratio"] == 1.0

    scaled = ANALYSIS.compare_mueller(theta, 2.0 * reference, reference)
    assert abs(scaled["m11_raw_solid_angle_relative_l2"] - 1.0) < 1.0e-12
    assert scaled["full_normalized_solid_angle_relative_l2"] < 1.0e-14
    assert scaled["m11_forward_ratio"] == 2.0

    with tempfile.TemporaryDirectory() as temporary:
        strict_rows = []
        for ref, scale in ((2, 1.02), (3, 1.0)):
            case_directory = Path(temporary) / str(ref)
            case_directory.mkdir()
            (case_directory / "result.json").write_text(json.dumps({
                "physical": {
                    "theta_degrees": theta.tolist(),
                    "mueller": (scale * reference).tolist(),
                },
            }), encoding="utf-8")
            strict_rows.append({
                "path": str(case_directory),
                "phase": "mesh_polyhedra_hdiv_strict",
                "analysis_series": "mesh_polyhedra_hdiv_strict",
                "repeat": 0,
                "in_current_plan": True,
                "shape": "prism",
                "ka": 10.0,
                "ri": 1.3,
                "ref": ref,
                "edge_mode": "hdiv-bdm1",
                "quad": 13,
                "duffy_order": 6,
                "digits_effective": 7,
                "near_radius": 3,
                "tolerance": 2.0e-6,
            })
        ANALYSIS.add_self_convergence(strict_rows)
        assert strict_rows[0]["next_ref"] == 3
        assert abs(
            strict_rows[0]["next_ref_m11_raw_solid_angle_relative_l2"]
            - 0.02
        ) < 1.0e-12

    def row(ref: int, error: float, next_error: float | None = None) -> dict:
        value = {
            "phase": "mesh_sphere_m13",
            "repeat": 0,
            "in_current_plan": True,
            "ka": 10.0,
            "ref": ref,
            "points_per_shortest_wavelength": float(4 * ref),
            "system_dofs": 100 * 4 ** ref,
            "mesh_triangles": 20 * 4 ** ref,
        }
        for name in (
            "m11_raw_solid_angle_relative_l2",
            "m11_forward_relative_error",
            "m11_integral_relative_error",
            "full_normalized_solid_angle_relative_l2",
            "maximum_normalized_absolute_error",
        ):
            value["mie_" + name] = error
            if next_error is not None:
                value["next_ref_" + name] = next_error
        if next_error is not None:
            value["next_ref"] = ref + 1
        return value

    selection = ANALYSIS.mesh_selection_rows([
        row(2, 2.0e-2), row(3, 8.0e-3, 5.0e-3), row(4, 4.0e-3)
    ])
    assert len(selection) == 1
    assert selection[0]["minimum_ref"] == 3
    assert selection[0]["status"] == "confirmed_by_exact_reference_and_next_ref"

    contrast_rows = []
    for ri, ref, error in (
        (1.5, 3, 2.0e-2), (1.5, 4, 8.0e-3),
        (2.0, 4, 2.0e-2), (2.0, 5, 7.0e-3),
    ):
        value = row(ref, error)
        value["phase"] = "mesh_sphere_contrast"
        value["analysis_series"] = "mesh_sphere_contrast"
        value["ri"] = ri
        contrast_rows.append(value)
    contrast_selection = ANALYSIS.contrast_selection_rows(contrast_rows)
    assert [(item["ri"], item["minimum_ref"]) for item in contrast_selection] == [
        (1.5, 4), (2.0, 5),
    ]

    with tempfile.TemporaryDirectory() as temporary:
        table = Path(temporary) / "selection.tex"
        ANALYSIS.write_mesh_selection_tex(table, selection)
        text = table.read_text(encoding="utf-8")
        assert "Ми + соседняя сетка" in text
        assert r"0.800\%" in text

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first.bin"
        first.write_bytes(b"12345")
        os.link(first, root / "second.bin")
        storage = ANALYSIS.directory_storage(root)
        assert storage["files"] == 1
        assert storage["logical_bytes"] == 5
        assert storage["allocated_bytes"] >= 5

    with tempfile.TemporaryDirectory() as temporary:
        case_directory = Path(temporary)
        attempt = case_directory / "attempts" / "attempt_000"
        attempt.mkdir(parents=True)
        (attempt / "summary.json").write_text(json.dumps({
            "wall_time_s": 3.0,
        }))
        aggregate = ANALYSIS.aggregate_attempt_resources(
            case_directory,
            {
                "wall_time_s": 7.0,
                "gpu_board_energy_j": 11.0,
                "gpu_memory_peak_delta_mib": 5.0,
                "gnu_time": {"user_time_s": 2.0, "time_max_rss_kib": 2048},
            },
        )
        assert aggregate["attempt_count"] == 2
        assert aggregate["cumulative_wall_time_s"] == 10.0
        assert aggregate["cumulative_cpu_user_s"] == 2.0
        assert aggregate["cumulative_gpu_energy_j"] == 11.0
        assert aggregate["attempt_max_rss_mib"] == 2.0

    gate = {
        "setup_only": True, "operator_cache_id": "operator-a",
        "wall_time_s": 100.0, "max_rss_mib": 2000.0,
        "gpu_memory_peak_delta_mib": 3000.0, "gpu_energy_j": 400.0,
        "fmm_setup_s": 60.0, "mbj_setup_s": 20.0,
    }
    warm_solve = {
        "setup_only": False, "operator_cache_id": "operator-a",
        "cache_state_actual": "warm", "wall_time_s": 25.0,
        "max_rss_mib": 500.0, "gpu_memory_peak_delta_mib": 600.0,
        "gpu_energy_j": 50.0, "path": "warm-solve",
        "fmm_setup_s": 5.0, "mbj_setup_s": 3.0,
    }
    cold_solve = {
        "setup_only": False, "operator_cache_id": "operator-b",
        "cache_state_actual": "cold", "wall_time_s": 30.0,
        "max_rss_mib": 700.0, "gpu_memory_peak_delta_mib": 800.0,
        "gpu_energy_j": 60.0,
    }
    ANALYSIS.add_effective_cold_resources([gate, warm_solve, cold_solve])
    assert warm_solve["effective_cold_wall_time_s"] == 97.0
    assert warm_solve["gated_workflow_wall_time_s"] == 125.0
    assert warm_solve["cold_time_method"] == "gate_setup_delta_plus_warm_process"
    assert warm_solve["effective_cold_max_rss_mib"] == 2000.0
    assert warm_solve["effective_cold_gpu_memory_peak_delta_mib"] == 3000.0
    assert warm_solve["effective_cold_gpu_energy_j"] == 450.0
    assert cold_solve["effective_cold_wall_time_s"] == 30.0

    aliased = [{
        "phase": "fmm_radius_scale", "shape": "sphere", "ka": 40,
        "ri": 1.3, "ref": 6, "near_radius": 3,
    }]
    ANALYSIS.apply_analysis_reuse(aliased, {"analysis_reuse": [{
        "series": "mesh_sphere_m13", "phase": "fmm_radius_scale",
        "shape": "sphere", "ka": 40, "ri": 1.3, "ref": 6,
        "near_radius": 3,
    }]})
    assert aliased[0]["analysis_series"] == "mesh_sphere_m13"

    negative_control = [{
        "phase": "mesh_sphere_m13", "shape": "sphere", "ka": 40,
        "ri": 1.3, "ref": 6, "near_radius": 1,
        "in_current_plan": False,
    }]
    ANALYSIS.apply_analysis_reuse(negative_control, {"analysis_reuse": [{
        "series": "fmm_radius_scale", "include_in_current_plan": True,
        "phase": "mesh_sphere_m13", "shape": "sphere", "ka": 40,
        "ri": 1.3, "ref": 6, "near_radius": 1,
    }]})
    assert negative_control[0]["analysis_series"] == "fmm_radius_scale"
    assert negative_control[0]["in_current_plan"] is True
    assert negative_control[0]["analysis_included_by_reuse"] is True

    fit_rows = []
    for index in range(8):
        dofs = float(100 * 2 ** index)
        ka = float(1 + index % 3)
        fit_rows.append({
            "system_dofs": dofs,
            "ka": ka,
            "wall_time_s": 0.25 * dofs ** 1.2 * ka ** 0.7,
            "iterations_x": 10 + index,
            "resumed": False,
        })
    model = ANALYSIS.fit_log_power_model(fit_rows, "wall_time_s")
    assert model is not None
    assert abs(model["coefficients"]["log_system_dofs"]["estimate"] - 1.2) < 1e-10
    assert abs(model["coefficients"]["log_ka"]["estimate"] - 0.7) < 1e-10
    assert abs(model["r_squared_log_space"] - 1.0) < 1e-12

    production_rows = []
    for repeat, wall in enumerate((100.0, 60.0, 64.0)):
        production_rows.append({
            "in_current_plan": True,
            "phase": "resource_scaling",
            "resumed": False,
            "ka": 40.0,
            "ref": 6,
            "system_dofs": 800000,
            "mesh_triangles": 100000,
            "repeat": repeat,
            "wall_time_s": wall,
            "max_rss_mib": 9000.0 if repeat == 0 else 7500.0,
            "gpu_memory_peak_delta_mib": 19000.0,
            "gpu_incremental_energy_j": 1000.0 - 100.0 * repeat,
            "gpu_total_memory_mib": 24564.0,
        })
    production = ANALYSIS.production_resource_scaling_rows(production_rows)
    assert len(production) == 1
    assert production[0]["warm_wall_time_mean_s"] == 62.0
    assert abs(production[0]["cold_to_warm_speedup"] - 100.0 / 62.0) < 1e-12
    limit = ANALYSIS.production_resource_limit(production)
    assert limit is not None
    assert limit["projected_next_ref"] == 7
    assert limit["projected_system_dofs"] == 3200000
    assert limit["projected_gpu_memory_peak_delta_mib"] == 76000.0
    assert limit["fits_measured_gpu"] is False
    assert ANALYSIS.gpu_total_memory_mib({
        "hardware": {"gpu_raw": "GPU, UUID, driver, 24564, 450"}
    }) == 24564.0

    with tempfile.TemporaryDirectory() as temporary:
        angular_rows = []
        for count in (5, 17):
            case_directory = Path(temporary) / str(count)
            case_directory.mkdir()
            angles = np.linspace(0.0, 180.0, count)
            values = np.zeros((4, 4, count))
            values[0, 0] = 1.0 + 0.2 * np.cos(3.0 * np.deg2rad(angles))
            values[1, 1] = 0.5 * values[0, 0]
            (case_directory / "result.json").write_text(json.dumps({
                "physical": {
                    "theta_degrees": angles.tolist(),
                    "mueller": values.tolist(),
                },
            }), encoding="utf-8")
            angular_rows.append({
                "path": str(case_directory), "phase": "farfield_grid",
                "repeat": 0, "in_current_plan": True, "ntheta": count,
                "shape": "sphere", "ka": 2.0, "ri": 1.3, "ref": 2,
                "edge_mode": "smooth", "quad": 13, "duffy_order": 6,
                "digits_effective": 7, "near_radius": 5,
                "max_leaf_effective": 64, "tolerance": 2.0e-6,
            })
        ANALYSIS.add_farfield_grid_comparisons(angular_rows)
        assert angular_rows[0]["angular_full_interpolation_relative_l2"] > 0.0
        assert angular_rows[0]["angular_m11_integral_relative_error"] > 0.0
        assert angular_rows[1]["angular_full_interpolation_relative_l2"] < 1e-14
        assert angular_rows[1]["angular_m11_integral_relative_error"] < 1e-14

    with tempfile.TemporaryDirectory() as temporary:
        solver_rows = []
        for name, tolerance, scale in (
            ("baseline", 1.0e-4, 1.001),
            ("solver_tolerance", 2.0e-6, 1.0),
        ):
            case_directory = Path(temporary) / name
            case_directory.mkdir()
            values = reference.copy()
            values[1, 1] *= scale
            (case_directory / "result.json").write_text(json.dumps({
                "physical": {
                    "theta_degrees": theta.tolist(),
                    "mueller": values.tolist(),
                },
            }), encoding="utf-8")
            solver_rows.append({
                "path": str(case_directory), "phase": "solver_controls",
                "repeat": 0, "in_current_plan": True, "name": name,
                "tolerance": tolerance,
            })
        ANALYSIS.add_solver_reference_comparisons(solver_rows)
        assert solver_rows[0][
            "solver_reference_full_normalized_solid_angle_relative_l2"
        ] > 0.0
        assert solver_rows[1][
            "solver_reference_full_normalized_solid_angle_relative_l2"
        ] < 1.0e-14

    planned = [
        {"phase": "mesh", "base_id": "a", "repeat": 0},
        {"phase": "mesh", "base_id": "b", "repeat": 0},
        {"phase": "fmm", "base_id": "c", "repeat": 0},
    ]
    progress = ANALYSIS.progress_rows(
        planned,
        [{"phase": "mesh", "base_id": "a", "repeat": 0}],
        [{
            "case": {"phase": "fmm", "base_id": "c", "repeat": 0},
            "status": {"state": "failed"},
        }],
    )
    assert progress == [
        {"phase": "mesh", "planned": 2, "completed": 1, "failed": 0,
         "pending": 1},
        {"phase": "fmm", "planned": 1, "completed": 0, "failed": 1,
         "pending": 0},
    ]

    warm = {
        "near_radius": 5, "cache_state_actual": "warm",
        "wall_time_s": 10.0, "repeat": 0, "path": "warm",
    }
    cold = {
        "near_radius": 5, "cache_state_actual": "cold",
        "wall_time_s": 20.0, "repeat": 0, "path": "cold",
    }
    preferred = ANALYSIS.preferred_unique_rows(
        [warm, cold], ("near_radius",)
    )
    assert preferred == [cold]
    assert ANALYSIS.prefer_cold_measurements([warm, cold]) == [cold]
    print("convergence analysis metrics: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
