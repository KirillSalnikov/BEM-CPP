#!/usr/bin/env python3
"""Aggregate convergence runs, compute physical errors, and make study plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from collections import Counter
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from verify_mie import mie_mueller  # noqa: E402
from run_convergence_study import expand_config  # noqa: E402


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def timing_preference(row: dict[str, Any]) -> tuple[int, int, int, str]:
    """Rank measurements for total-time comparisons without warm-cache bias."""
    cache_rank = {"cold": 0, None: 1, "warm": 2}.get(
        row.get("cache_state_actual"), 1
    )
    return (
        cache_rank,
        1 if row.get("resumed") else 0,
        int(row.get("repeat") or 0),
        str(row.get("path") or ""),
    )


def prefer_cold_measurements(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cold = [row for row in rows if row.get("cache_state_actual") == "cold"]
    return cold or rows


def preferred_unique_rows(
    rows: Iterable[dict[str, Any]], key_fields: Iterable[str]
) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    fields = tuple(key_fields)
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        previous = unique.get(key)
        if previous is None or timing_preference(row) < timing_preference(previous):
            unique[key] = row
    return list(unique.values())


def physical_arrays(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    physical = result.get("physical")
    if not isinstance(physical, dict):
        raise ValueError("result has no physical block")
    theta = np.asarray(physical["theta_degrees"], dtype=float)
    mueller = np.asarray(physical["mueller"], dtype=float)
    if mueller.shape != (4, 4, theta.size):
        raise ValueError(f"unexpected Mueller shape {mueller.shape}")
    return theta, mueller


def solid_angle_integral(theta: np.ndarray, values: np.ndarray) -> float:
    radians = np.deg2rad(theta)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy < 2.0
        trapezoid = np.trapz
    return float(trapezoid(values * np.sin(radians), radians))


def compare_mueller(
    theta: np.ndarray,
    computed: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    if computed.shape != reference.shape:
        raise ValueError("Mueller arrays must have identical shapes")
    computed_scale = max(abs(float(computed[0, 0, 0])), 1.0e-300)
    reference_scale = max(abs(float(reference[0, 0, 0])), 1.0e-300)
    computed_normalized = computed / computed_scale
    reference_normalized = reference / reference_scale
    raw_m11_denominator = max(
        solid_angle_integral(theta, reference[0, 0] ** 2), 1.0e-300
    )
    raw_m11_l2 = math.sqrt(
        solid_angle_integral(theta, (computed[0, 0] - reference[0, 0]) ** 2)
        / raw_m11_denominator
    )
    normalized_numerator = 0.0
    normalized_denominator = 0.0
    component_l2: list[float] = []
    for row in range(4):
        for column in range(4):
            difference = (
                computed_normalized[row, column]
                - reference_normalized[row, column]
            )
            numerator = solid_angle_integral(theta, difference ** 2)
            denominator = solid_angle_integral(
                theta, reference_normalized[row, column] ** 2
            )
            normalized_numerator += numerator
            normalized_denominator += denominator
            component_l2.append(
                math.sqrt(max(numerator, 0.0) / max(denominator, 1.0e-12))
            )
    computed_integral = solid_angle_integral(theta, computed[0, 0])
    reference_integral = solid_angle_integral(theta, reference[0, 0])
    return {
        "m11_raw_solid_angle_relative_l2": raw_m11_l2,
        "m11_forward_relative_error": abs(
            computed[0, 0, 0] - reference[0, 0, 0]
        ) / reference_scale,
        "m11_forward_ratio": float(computed[0, 0, 0] / reference[0, 0, 0]),
        "m11_integral_relative_error": abs(computed_integral - reference_integral)
        / max(abs(reference_integral), 1.0e-300),
        "full_normalized_solid_angle_relative_l2": math.sqrt(
            normalized_numerator / max(normalized_denominator, 1.0e-300)
        ),
        "maximum_normalized_absolute_error": float(
            np.max(np.abs(computed_normalized - reference_normalized))
        ),
        "maximum_component_relative_l2": max(component_l2),
    }


def parse_elapsed(value: str | None) -> float | None:
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 2:
            return 60.0 * float(parts[0]) + float(parts[1])
        if len(parts) == 3:
            return 3600.0 * float(parts[0]) + 60.0 * float(parts[1]) + float(parts[2])
    except ValueError:
        return None
    return None


def directory_storage(path: Path) -> dict[str, int]:
    summary = {"files": 0, "logical_bytes": 0, "allocated_bytes": 0}
    if not path.is_dir():
        return summary
    seen: set[tuple[int, int]] = set()
    for item in path.rglob("*"):
        try:
            stat = item.stat()
        except OSError:
            continue
        if not item.is_file():
            continue
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        summary["files"] += 1
        summary["logical_bytes"] += stat.st_size
        summary["allocated_bytes"] += stat.st_blocks * 512
    return summary


def file_size_mib(path: Path) -> float:
    try:
        return path.stat().st_size / 2.0**20
    except OSError:
        return 0.0


def gpu_total_memory_mib(resources: dict[str, Any]) -> float | None:
    raw = (resources.get("hardware") or {}).get("gpu_raw")
    if not isinstance(raw, str):
        return None
    fields = [field.strip() for field in raw.split(",")]
    if len(fields) < 4:
        return None
    try:
        return float(fields[3])
    except ValueError:
        return None


def aggregate_attempt_resources(
    case_directory: Path, current: dict[str, Any]
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for path in sorted(
        (case_directory / "attempts").glob("attempt_*/profile/resources.json")
    ):
        try:
            attempts.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    for path in sorted(
        (case_directory / "attempts").glob("attempt_*/summary.json")
    ):
        try:
            attempts.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    if current:
        attempts.append(current)

    def values(field: str) -> list[float]:
        return [
            float(attempt[field]) for attempt in attempts
            if attempt.get(field) is not None
        ]

    def nested_values(*fields: str) -> list[float]:
        found: list[float] = []
        for attempt in attempts:
            value: Any = attempt
            for field in fields:
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(field)
            if value is not None:
                found.append(float(value))
        return found

    wall = values("wall_time_s")
    cpu_user = nested_values("gnu_time", "user_time_s")
    cpu_system = nested_values("gnu_time", "system_time_s")
    board_energy = values("gpu_board_energy_j")
    incremental_energy = values("gpu_incremental_energy_j")
    max_rss = [value / 1024.0 for value in nested_values(
        "gnu_time", "time_max_rss_kib"
    )]
    peak_vram = values("gpu_memory_peak_delta_mib")
    return {
        "attempt_count": len(attempts),
        "cumulative_wall_time_s": sum(wall) if wall else None,
        "cumulative_cpu_user_s": sum(cpu_user) if cpu_user else None,
        "cumulative_cpu_system_s": sum(cpu_system) if cpu_system else None,
        "cumulative_gpu_energy_j": sum(board_energy) if board_energy else None,
        "cumulative_gpu_incremental_energy_j": (
            sum(incremental_energy) if incremental_energy else None
        ),
        "attempt_max_rss_mib": max(max_rss) if max_rss else None,
        "attempt_max_gpu_memory_peak_delta_mib": (
            max(peak_vram) if peak_vram else None
        ),
    }


def discover(runs_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for status_path in sorted(runs_root.glob("*/*/repeat_*/status.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        case_directory = status_path.parent
        case_path = case_directory / "case.json"
        manifest = (
            json.loads(case_path.read_text(encoding="utf-8"))
            if case_path.exists() else {}
        )
        case = manifest.get("case", {})
        if status.get("state") != "complete":
            failures.append({
                "path": str(case_directory),
                "phase": case.get("phase"),
                "case": case,
                "status": status,
            })
            continue
        result = json.loads((case_directory / "result.json").read_text(encoding="utf-8"))
        resources_path = case_directory / "profile" / "resources.json"
        resources = (
            json.loads(resources_path.read_text(encoding="utf-8"))
            if resources_path.exists() else {}
        )
        attempt_resources = aggregate_attempt_resources(
            case_directory, resources
        )
        gnu_time = resources.get("gnu_time", {})
        aggregates = resources.get("aggregates", {})
        physical = result.get("physical") or {}
        mbj = result.get("mbj") or {}
        wall_time_s = parse_elapsed(gnu_time.get("time_elapsed_text"))
        sampled_process_cpu = (
            aggregates.get("process_cpu_percent", {}).get("mean")
        )
        if (not sampled_process_cpu) and wall_time_s:
            cpu_times = (
                gnu_time.get("user_time_s"),
                gnu_time.get("system_time_s"),
            )
            if all(value is not None for value in cpu_times):
                sampled_process_cpu = (
                    100.0 * sum(map(float, cpu_times)) / wall_time_s
                )
        cache_directory = manifest.get("cache_directory")
        cache_storage = (
            directory_storage(Path(cache_directory))
            if cache_directory else
            {"files": 0, "logical_bytes": 0, "allocated_bytes": 0}
        )
        output_storage = directory_storage(case_directory)
        cache_path = Path(cache_directory) if cache_directory else None
        near_cache_mib = (
            file_size_mib(cache_path / "operator.near") if cache_path else 0.0
        )
        mbj_cache_mib = (
            file_size_mib(cache_path / "mbj.cache") if cache_path else 0.0
        )
        system_dofs = result.get("system_dofs")
        row: dict[str, Any] = {
            "path": str(case_directory),
            "manifest_schema_version": manifest.get("schema_version"),
            "binary_sha256": (
                (manifest.get("operator_cache_identity") or {}).get(
                    "binary_sha256"
                )
            ),
            "provenance_binary_hash_present": bool(
                (manifest.get("operator_cache_identity") or {}).get(
                    "binary_sha256"
                )
            ),
            "phase": case.get("phase"),
            "name": case.get("name"),
            "base_id": case.get("base_id"),
            "repeat": case.get("repeat"),
            "cache_state_expected": case.get("cache_state_expected"),
            "setup_only": bool(case.get("setup_only", False)),
            "operator_cache_id": (
                Path(cache_directory).name if cache_directory else None
            ),
            "shape": result.get("shape"),
            "ka": result.get("ka"),
            "ri": result.get("ri"),
            "ref": result.get("refinements"),
            "edge_mode": result.get("edge_mode"),
            "precision": case.get("precision"),
            "near_precision": result.get("fmm_near_precision"),
            "mesh_vertices": result.get("mesh_vertices"),
            "mesh_triangles": result.get("mesh_triangles"),
            "surface_scalar_nodes": result.get("surface_scalar_nodes"),
            "surface_current_dofs": result.get("surface_current_dofs"),
            "system_dofs": system_dofs,
            "quadrature_points": result.get("quadrature_points"),
            "max_element_edge": result.get("max_element_edge"),
            "ka_h_element": result.get("ka_h_element"),
            "points_per_exterior_wavelength": result.get(
                "p2_nodes_per_exterior_wavelength"
            ),
            "points_per_interior_wavelength": result.get(
                "p2_nodes_per_interior_wavelength"
            ),
            "points_per_shortest_wavelength": result.get(
                "p2_nodes_per_shortest_wavelength"
            ),
            "quad": result.get("regular_quadrature"),
            "duffy_order": result.get("duffy_order"),
            "digits_requested": result.get("fmm_digits_requested"),
            "digits_effective": result.get("fmm_digits"),
            "near_radius": result.get("fmm_near_radius"),
            "near_radius_requested": result.get(
                "fmm_near_radius_requested", result.get("fmm_near_radius")
            ),
            "max_leaf_requested": result.get("fmm_max_leaf_points_requested"),
            "max_leaf_effective": result.get("fmm_max_leaf_points"),
            "fmm_exterior_depth": (result.get("fmm_expansion") or {}).get(
                "exterior", {}
            ).get("tree_depth"),
            "fmm_exterior_order": (result.get("fmm_expansion") or {}).get(
                "exterior", {}
            ).get("order"),
            "tolerance": result.get("tolerance"),
            "gmres_restart": result.get("gmres_restart"),
            "mbj_nodes": mbj.get("nodes_per_block"),
            "mbj_overlap": mbj.get("overlap_nodes"),
            "mbj_coarse_rank": mbj.get("coarse_rank"),
            "mbj_coarse_setup_s": mbj.get("coarse_setup_s"),
            "ntheta": len(physical.get("theta_degrees", [])),
            "iterations_x": mbj.get("iterations"),
            "resumed_iterations_x": mbj.get("resumed_iterations"),
            "iterations_y": physical.get("parallel_iterations"),
            "resumed_iterations_y": physical.get("parallel_resumed_iterations"),
            "residual_x": mbj.get("fmm_residual"),
            "residual_y": physical.get("parallel_fmm_residual"),
            "fmm_setup_s": result.get("fmm_setup_s"),
            "mbj_setup_s": result.get("mbj_local_setup_s"),
            "solve_x_s": mbj.get("solve_s"),
            "solve_y_s": physical.get("parallel_s"),
            "farfield_s": physical.get("farfield_s"),
            "wall_time_s": wall_time_s,
            "outer_profile_wall_time_s": resources.get("wall_time_s"),
            "cpu_user_s": gnu_time.get("user_time_s"),
            "cpu_system_s": gnu_time.get("system_time_s"),
            "process_read_mib": (
                float(resources["observed_process_read_bytes"]) / 2.0**20
                if resources.get("observed_process_read_bytes") is not None
                else None
            ),
            "process_write_mib": (
                float(resources["observed_process_write_bytes"]) / 2.0**20
                if resources.get("observed_process_write_bytes") is not None
                else None
            ),
            "cache_files": cache_storage["files"],
            "near_cache_logical_mib": near_cache_mib,
            "mbj_cache_logical_mib": mbj_cache_mib,
            "cache_logical_mib": cache_storage["logical_bytes"] / 2.0**20,
            "cache_allocated_mib": cache_storage["allocated_bytes"] / 2.0**20,
            "cache_bytes_per_unknown": (
                cache_storage["logical_bytes"] / float(system_dofs)
                if system_dofs else None
            ),
            "output_files": output_storage["files"],
            "output_logical_mib": output_storage["logical_bytes"] / 2.0**20,
            "output_allocated_mib": output_storage["allocated_bytes"] / 2.0**20,
            "max_rss_mib": (
                gnu_time.get("time_max_rss_kib", 0) / 1024.0
                if gnu_time.get("time_max_rss_kib") is not None else None
            ),
            "gpu_memory_peak_delta_mib": resources.get("gpu_memory_peak_delta_mib"),
            "gpu_total_memory_mib": gpu_total_memory_mib(resources),
            "gpu_energy_j": resources.get("gpu_board_energy_j"),
            "gpu_incremental_energy_j": resources.get("gpu_incremental_energy_j"),
            **attempt_resources,
            "gpu_utilization_mean_percent": (
                aggregates.get("gpu_util_percent", {}).get("mean")
            ),
            "gpu_utilization_max_percent": (
                aggregates.get("gpu_util_percent", {}).get("maximum")
            ),
            "gpu_power_mean_w": (
                aggregates.get("gpu_power_w", {}).get("mean")
            ),
            "gpu_power_max_w": aggregates.get("gpu_power_w", {}).get("maximum"),
            "gpu_temperature_max_c": (
                aggregates.get("gpu_temperature_c", {}).get("maximum")
            ),
            "gpu_sm_clock_mean_mhz": (
                aggregates.get("gpu_sm_clock_mhz", {}).get("mean")
            ),
            "gpu_sm_clock_max_mhz": (
                aggregates.get("gpu_sm_clock_mhz", {}).get("maximum")
            ),
            "process_cpu_mean_percent": sampled_process_cpu,
            "process_cpu_max_percent": (
                aggregates.get("process_cpu_percent", {}).get("maximum")
            ),
            "process_threads_max": (
                aggregates.get("process_threads", {}).get("maximum")
            ),
            "system_cpu_mean_percent": (
                aggregates.get("system_cpu_percent", {}).get("mean")
            ),
            "system_memory_available_min_mib": (
                float(aggregates["system_memory_available_bytes"]["minimum"])
                / 2.0**20
                if aggregates.get("system_memory_available_bytes", {}).get(
                    "minimum"
                ) is not None else None
            ),
            "system_swap_used_max_mib": (
                float(aggregates["system_swap_used_bytes"]["maximum"])
                / 2.0**20
                if aggregates.get("system_swap_used_bytes", {}).get(
                    "maximum"
                ) is not None else None
            ),
            "disk_free_min_gib": (
                float(aggregates["disk_free_bytes"]["minimum"]) / 2.0**30
                if aggregates.get("disk_free_bytes", {}).get("minimum")
                is not None else None
            ),
            "cpu_frequency_mean_mhz": (
                aggregates.get("cpu_frequency_mhz", {}).get("mean")
            ),
            "cpu_frequency_max_mhz": (
                aggregates.get("cpu_frequency_mhz", {}).get("maximum")
            ),
            "near_cache_hit": (result.get("near_correction_cache") or {}).get("hit"),
            "mbj_cache_hit": (result.get("mbj_setup_breakdown") or {}).get("cache_hit"),
        }
        near_hit = row["near_cache_hit"]
        mbj_hit = row["mbj_cache_hit"]
        if near_hit is True and mbj_hit is True:
            row["cache_state_actual"] = "warm"
        elif near_hit is False and mbj_hit is False:
            row["cache_state_actual"] = "cold"
        else:
            row["cache_state_actual"] = "partial"
        row["resumed"] = bool(
            (row["resumed_iterations_x"] or 0)
            + (row["resumed_iterations_y"] or 0)
        )
        iteration_values = [
            value for value in (row["iterations_x"], row["iterations_y"])
            if value is not None
        ]
        solve_values = [
            value for value in (row["solve_x_s"], row["solve_y_s"])
            if value is not None
        ]
        row["iterations_total"] = (
            sum(map(float, iteration_values)) if iteration_values else None
        )
        row["solve_total_s"] = (
            sum(map(float, solve_values)) if solve_values else None
        )
        if result.get("shape") == "sphere" and result.get("physical"):
            theta, computed = physical_arrays(result)
            reference = np.asarray(
                mie_mueller(theta.tolist(), complex(float(result["ri"]), 0.0), float(result["ka"])),
                dtype=float,
            )
            row.update({f"mie_{key}": value for key, value in compare_mueller(
                theta, computed, reference
            ).items()})
        rows.append(row)
    return rows, failures


def add_effective_cold_resources(rows: list[dict[str, Any]]) -> None:
    """Recover a one-process cold estimate for cache-warmed physical runs."""
    gates: dict[str, dict[str, Any]] = {}
    cold_peers: dict[str, dict[str, Any]] = {}
    for row in rows:
        cache_id = row.get("operator_cache_id")
        if row.get("setup_only") and cache_id:
            previous = gates.get(str(cache_id))
            if previous is None or float(
                row.get("cumulative_wall_time_s") or row.get("wall_time_s") or 0.0
            ) > float(
                previous.get("cumulative_wall_time_s")
                or previous.get("wall_time_s") or 0.0
            ):
                gates[str(cache_id)] = row
        elif row.get("cache_state_actual") == "cold" and cache_id:
            cold_peers.setdefault(str(cache_id), row)

    def wall_time(row: dict[str, Any]) -> Any:
        return row.get("cumulative_wall_time_s") or row.get("wall_time_s")

    def setup_time(row: dict[str, Any]) -> float | None:
        values = [
            value for value in (row.get("fmm_setup_s"), row.get("mbj_setup_s"))
            if value is not None
        ]
        return sum(map(float, values)) if values else None

    for row in rows:
        wall = wall_time(row)
        row["effective_cold_wall_time_s"] = None
        row["cold_time_method"] = None
        row["gated_workflow_wall_time_s"] = None
        row["effective_cold_max_rss_mib"] = None
        row["effective_cold_gpu_memory_peak_delta_mib"] = None
        row["effective_cold_gpu_energy_j"] = None
        row["setup_gate_wall_time_s"] = None
        row["setup_gate_path"] = None
        if row.get("setup_only"):
            continue

        if row.get("cache_state_actual") == "cold":
            row["effective_cold_wall_time_s"] = wall
            row["cold_time_method"] = "measured_cold_process"
            row["gated_workflow_wall_time_s"] = wall
            row["effective_cold_max_rss_mib"] = (
                row.get("attempt_max_rss_mib") or row.get("max_rss_mib")
            )
            row["effective_cold_gpu_memory_peak_delta_mib"] = (
                row.get("attempt_max_gpu_memory_peak_delta_mib")
                or row.get("gpu_memory_peak_delta_mib")
            )
            row["effective_cold_gpu_energy_j"] = (
                row.get("cumulative_gpu_energy_j") or row.get("gpu_energy_j")
            )
            continue

        cache_id = row.get("operator_cache_id")
        gate = gates.get(str(cache_id)) if cache_id else None
        peer = cold_peers.get(str(cache_id)) if cache_id else None
        source = gate or peer
        if row.get("cache_state_actual") != "warm" or source is None or wall is None:
            continue
        source_wall = wall_time(source)
        if source_wall is None:
            continue
        if gate is not None:
            row["setup_gate_wall_time_s"] = source_wall
            row["setup_gate_path"] = gate.get("path")
            row["gated_workflow_wall_time_s"] = (
                float(source_wall) + float(wall)
            )
        source_setup = setup_time(source)
        warm_setup = setup_time(row)
        if source_setup is not None and warm_setup is not None:
            row["effective_cold_wall_time_s"] = float(wall) + max(
                0.0, source_setup - warm_setup
            )
            row["cold_time_method"] = (
                "gate_setup_delta_plus_warm_process"
                if gate is not None else
                "cold_peer_setup_delta_plus_warm_process"
            )
        elif gate is not None:
            row["effective_cold_wall_time_s"] = (
                float(source_wall) + float(wall)
            )
            row["cold_time_method"] = "gate_plus_warm_process_upper_bound"
        rss = [
            value for value in (
                source.get("attempt_max_rss_mib") or source.get("max_rss_mib"),
                row.get("attempt_max_rss_mib") or row.get("max_rss_mib"),
            ) if value is not None
        ]
        gpu_memory = [
            value for value in (
                source.get("attempt_max_gpu_memory_peak_delta_mib")
                or source.get("gpu_memory_peak_delta_mib"),
                row.get("attempt_max_gpu_memory_peak_delta_mib")
                or row.get("gpu_memory_peak_delta_mib"),
            ) if value is not None
        ]
        row["effective_cold_max_rss_mib"] = max(rss) if rss else None
        row["effective_cold_gpu_memory_peak_delta_mib"] = (
            max(gpu_memory) if gpu_memory else None
        )
        energy = [
            value for value in (
                source.get("cumulative_gpu_energy_j") or source.get("gpu_energy_j"),
                row.get("cumulative_gpu_energy_j") or row.get("gpu_energy_j"),
            ) if value is not None
        ]
        row["effective_cold_gpu_energy_j"] = (
            sum(map(float, energy)) if gate is not None and energy else None
        )


def apply_analysis_reuse(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    """Assign completed cases to an additional configured analysis series."""
    mappings = config.get("analysis_reuse", [])
    for row in rows:
        row["analysis_series"] = row.get("phase")
        for mapping in mappings:
            selector = {
                key: value for key, value in mapping.items()
                if key not in {"series", "include_in_current_plan"}
            }
            if all(row.get(key) == value for key, value in selector.items()):
                row["analysis_series"] = mapping["series"]
                if mapping.get("include_in_current_plan"):
                    row["in_current_plan"] = True
                    row["analysis_included_by_reuse"] = True
                break


def add_self_convergence(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        series = row.get("analysis_series", row.get("phase"))
        if not row.get("in_current_plan") or row.get("repeat") != 0 or series not in {
            "mesh_sphere_m13", "mesh_sphere_contrast", "mesh_prism",
            "basis_edge_control", "mesh_polyhedra_hdiv_strict",
        }:
            continue
        key = (
            series, row.get("shape"), row.get("ka"), row.get("ri"),
            row.get("edge_mode"), row.get("quad"), row.get("duffy_order"),
            row.get("digits_effective"), row.get("near_radius"), row.get("tolerance"),
        )
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: row["ref"])
        for coarse, fine in zip(group, group[1:]):
            coarse_result = json.loads(
                (Path(coarse["path"]) / "result.json").read_text(encoding="utf-8")
            )
            fine_result = json.loads(
                (Path(fine["path"]) / "result.json").read_text(encoding="utf-8")
            )
            theta_coarse, mueller_coarse = physical_arrays(coarse_result)
            theta_fine, mueller_fine = physical_arrays(fine_result)
            if not np.array_equal(theta_coarse, theta_fine):
                continue
            metrics = compare_mueller(theta_coarse, mueller_coarse, mueller_fine)
            coarse["next_ref"] = fine["ref"]
            for key, value in metrics.items():
                coarse[f"next_ref_{key}"] = value


def add_fmm_reference_comparisons(rows: list[dict[str, Any]]) -> None:
    radius_phases = {
        "fmm_near_radius", "fmm_radius_scale", "fmm_radius_cold_audit",
        "fmm_radius_digits_grid", "fmm_radius_dependency",
        "fmm_shared_cold_audit", "mesh_sphere_m13",
    }
    radius_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    digits_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    leaf_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("in_current_plan") or row.get("repeat") != 0:
            continue
        common = (
            row.get("shape"), row.get("ka"), row.get("ri"), row.get("ref"),
            row.get("edge_mode"), row.get("quad"), row.get("duffy_order"),
        )
        if row.get("phase") in radius_phases:
            radius_groups.setdefault(common + (
                row.get("digits_effective"), row.get("max_leaf_effective"),
                row.get("tolerance"),
            ), []).append(row)
        if row.get("phase") in {"fmm_digits", "fmm_radius_digits_grid"}:
            digits_groups.setdefault(common + (
                row.get("near_radius"), row.get("max_leaf_effective"),
                row.get("tolerance"),
            ), []).append(row)
        if row.get("phase") in {"fmm_leaf", "fmm_shared_cold_audit"}:
            leaf_groups.setdefault(common + (
                row.get("digits_effective"), row.get("near_radius"),
                row.get("tolerance"),
            ), []).append(row)

    def compare_groups(
        groups: Iterable[list[dict[str, Any]]], parameter: str
    ) -> None:
        for group in groups:
            if len(group) < 2:
                continue
            anchor = max(group, key=lambda row: float(row[parameter]))
            anchor_result = json.loads(
                (Path(anchor["path"]) / "result.json").read_text(encoding="utf-8")
            )
            theta_reference, mueller_reference = physical_arrays(anchor_result)
            for row in group:
                result = json.loads(
                    (Path(row["path"]) / "result.json").read_text(encoding="utf-8")
                )
                theta, mueller = physical_arrays(result)
                if not np.array_equal(theta, theta_reference):
                    continue
                row["fmm_reference_parameter"] = parameter
                row["fmm_reference_value"] = anchor[parameter]
                for key, value in compare_mueller(
                    theta, mueller, mueller_reference
                ).items():
                    row[f"fmm_reference_{key}"] = value

    compare_groups(radius_groups.values(), "near_radius")
    compare_groups(digits_groups.values(), "digits_effective")
    compare_groups(leaf_groups.values(), "max_leaf_effective")


def add_fmm_grid_reference_comparisons(rows: list[dict[str, Any]]) -> None:
    selected = [
        row for row in rows
        if row.get("in_current_plan")
        and row.get("repeat") == 0
        and row.get("phase") == "fmm_radius_digits_grid"
    ]
    if len(selected) < 2:
        return
    anchor = max(
        selected,
        key=lambda row: (
            int(row["near_radius"]), int(row["digits_effective"])
        ),
    )
    anchor_result = json.loads(
        (Path(anchor["path"]) / "result.json").read_text(encoding="utf-8")
    )
    theta_reference, mueller_reference = physical_arrays(anchor_result)
    for row in selected:
        result = json.loads(
            (Path(row["path"]) / "result.json").read_text(encoding="utf-8")
        )
        theta, mueller = physical_arrays(result)
        if not np.array_equal(theta, theta_reference):
            continue
        row["fmm_grid_reference_radius"] = anchor["near_radius"]
        row["fmm_grid_reference_digits"] = anchor["digits_effective"]
        for key, value in compare_mueller(
            theta, mueller, mueller_reference
        ).items():
            row[f"fmm_grid_reference_{key}"] = value


def add_quadrature_reference_comparisons(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if (
            not row.get("in_current_plan")
            or row.get("repeat") != 0
            or row.get("phase") != "quadrature"
        ):
            continue
        key = (
            row.get("shape"), row.get("ka"), row.get("ri"), row.get("ref"),
            row.get("edge_mode"), row.get("digits_effective"),
            row.get("near_radius"), row.get("max_leaf_effective"),
            row.get("tolerance"),
        )
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        if len(group) < 2:
            continue
        anchor = max(
            group,
            key=lambda row: (int(row["quad"]), int(row["duffy_order"])),
        )
        anchor_result = json.loads(
            (Path(anchor["path"]) / "result.json").read_text(encoding="utf-8")
        )
        theta_reference, mueller_reference = physical_arrays(anchor_result)
        for row in group:
            result = json.loads(
                (Path(row["path"]) / "result.json").read_text(encoding="utf-8")
            )
            theta, mueller = physical_arrays(result)
            if not np.array_equal(theta, theta_reference):
                continue
            row["quadrature_reference_quad"] = anchor["quad"]
            row["quadrature_reference_duffy"] = anchor["duffy_order"]
            for key, value in compare_mueller(
                theta, mueller, mueller_reference
            ).items():
                row[f"quadrature_reference_{key}"] = value


def add_farfield_grid_comparisons(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if (
            not row.get("in_current_plan")
            or row.get("repeat") != 0
            or row.get("phase") != "farfield_grid"
            or not row.get("ntheta")
        ):
            continue
        key = (
            row.get("shape"), row.get("ka"), row.get("ri"), row.get("ref"),
            row.get("edge_mode"), row.get("quad"), row.get("duffy_order"),
            row.get("digits_effective"), row.get("near_radius"),
            row.get("max_leaf_effective"), row.get("tolerance"),
        )
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        if len(group) < 2:
            continue
        anchor = max(group, key=lambda row: int(row["ntheta"]))
        anchor_result = json.loads(
            (Path(anchor["path"]) / "result.json").read_text(encoding="utf-8")
        )
        theta_reference, mueller_reference = physical_arrays(anchor_result)
        reference_scale = max(abs(float(mueller_reference[0, 0, 0])), 1.0e-300)
        reference_normalized = mueller_reference / reference_scale
        reference_integral = solid_angle_integral(
            theta_reference, mueller_reference[0, 0]
        )
        reference_norm = sum(
            solid_angle_integral(theta_reference, component ** 2)
            for component in reference_normalized.reshape(16, -1)
        )
        for row in group:
            result = json.loads(
                (Path(row["path"]) / "result.json").read_text(encoding="utf-8")
            )
            theta, mueller = physical_arrays(result)
            scale = max(abs(float(mueller[0, 0, 0])), 1.0e-300)
            normalized = mueller / scale
            interpolated = np.empty_like(reference_normalized)
            for component in range(16):
                i, j = divmod(component, 4)
                interpolated[i, j] = np.interp(
                    theta_reference, theta, normalized[i, j]
                )
            numerator = sum(
                solid_angle_integral(theta_reference, component ** 2)
                for component in (
                    interpolated - reference_normalized
                ).reshape(16, -1)
            )
            row["angular_reference_ntheta"] = anchor["ntheta"]
            row["angular_full_interpolation_relative_l2"] = math.sqrt(
                max(numerator, 0.0) / max(reference_norm, 1.0e-300)
            )
            integral = solid_angle_integral(theta, mueller[0, 0])
            row["angular_m11_integral_relative_error"] = abs(
                integral - reference_integral
            ) / max(abs(reference_integral), 1.0e-300)


def add_solver_reference_comparisons(rows: list[dict[str, Any]]) -> None:
    selected = [
        row for row in rows
        if row.get("in_current_plan")
        and row.get("phase") == "solver_controls"
        and row.get("repeat") == 0
    ]
    strict = [
        row for row in selected if row.get("name") == "solver_tolerance"
    ]
    if not strict:
        return
    reference = min(strict, key=lambda row: float(row["tolerance"]))
    reference_result = json.loads(
        (Path(reference["path"]) / "result.json").read_text(encoding="utf-8")
    )
    theta_reference, mueller_reference = physical_arrays(reference_result)
    for row in [
        value for value in rows
        if value.get("in_current_plan")
        and value.get("phase") == "solver_controls"
    ]:
        result = json.loads(
            (Path(row["path"]) / "result.json").read_text(encoding="utf-8")
        )
        theta, mueller = physical_arrays(result)
        if not np.array_equal(theta, theta_reference):
            continue
        row["solver_reference_tolerance"] = reference["tolerance"]
        for key, value in compare_mueller(
            theta, mueller, mueller_reference
        ).items():
            row[f"solver_reference_{key}"] = value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)


def progress_rows(
    planned_cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def identity(case: dict[str, Any]) -> tuple[Any, Any, Any]:
        return case.get("phase"), case.get("base_id"), case.get("repeat")

    planned_identities = {identity(case) for case in planned_cases}
    completed = Counter(
        row.get("phase") for row in rows
        if identity(row) in planned_identities
    )
    failed = Counter(
        failure.get("case", {}).get("phase") for failure in failures
        if identity(failure.get("case", {})) in planned_identities
        and failure.get("status", {}).get("state") == "failed"
    )
    planned = Counter(case.get("phase") for case in planned_cases)
    result: list[dict[str, Any]] = []
    for phase in dict.fromkeys(case.get("phase") for case in planned_cases):
        accepted = completed[phase]
        rejected = failed[phase]
        total = planned[phase]
        result.append({
            "phase": phase,
            "planned": total,
            "completed": accepted,
            "failed": rejected,
            "pending": max(0, total - accepted - rejected),
        })
    return result


def write_progress_tex(path: Path, progress: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Серия & План & Принято & Ошибка & Осталось\\",
        r"\midrule",
    ]
    for row in progress:
        phase = str(row["phase"]).replace("_", r"\_")
        lines.append(
            f"\\texttt{{{phase}}} & {row['planned']} & {row['completed']} & "
            f"{row['failed']} & {row['pending']}\\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    atomic_text(path, "\n".join(lines) + "\n")


def mesh_sphere_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("analysis_series", row.get("phase")) == "mesh_sphere_m13"
        and row.get("repeat") == 0
        and row.get("in_current_plan")
        and row.get("mie_m11_raw_solid_angle_relative_l2") is not None
    ]


def mesh_contrast_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("analysis_series", row.get("phase"))
        == "mesh_sphere_contrast"
        and row.get("repeat") == 0
        and row.get("in_current_plan")
        and row.get("mie_m11_raw_solid_angle_relative_l2") is not None
    ]


def mesh_target_passed(row: dict[str, Any], prefix: str = "mie_") -> bool:
    limits = {
        "m11_raw_solid_angle_relative_l2": 1.0e-2,
        "m11_forward_relative_error": 1.0e-2,
        "m11_integral_relative_error": 1.0e-2,
        "full_normalized_solid_angle_relative_l2": 1.0e-2,
        "maximum_normalized_absolute_error": 1.0e-2,
    }
    return all(
        row.get(prefix + key) is not None
        and float(row[prefix + key]) <= limit
        for key, limit in limits.items()
    )


def mesh_selection_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    source = mesh_sphere_rows(rows)
    for ka in sorted({row["ka"] for row in source}):
        values = sorted(
            (row for row in source if row["ka"] == ka),
            key=lambda row: row["ref"],
        )
        candidate = next((row for row in values if mesh_target_passed(row)), None)
        if candidate is None:
            selected.append({
                "ka": ka,
                "status": "target_not_reached",
                "largest_completed_ref": values[-1]["ref"],
                "largest_completed_ppw_shortest": values[-1][
                    "points_per_shortest_wavelength"
                ],
                "largest_completed_m11_error": values[-1][
                    "mie_m11_raw_solid_angle_relative_l2"
                ],
            })
            continue
        confirmation = mesh_target_passed(candidate, "next_ref_")
        selected.append({
            "ka": ka,
            "status": (
                "confirmed_by_exact_reference_and_next_ref"
                if confirmation else "confirmed_by_exact_reference"
            ),
            "minimum_ref": candidate["ref"],
            "minimum_ppw_shortest": candidate["points_per_shortest_wavelength"],
            "system_dofs": candidate["system_dofs"],
            "mesh_triangles": candidate["mesh_triangles"],
            "m11_error": candidate["mie_m11_raw_solid_angle_relative_l2"],
            "full_normalized_error": candidate[
                "mie_full_normalized_solid_angle_relative_l2"
            ],
            "forward_error": candidate["mie_m11_forward_relative_error"],
            "integral_error": candidate["mie_m11_integral_relative_error"],
            "next_ref": candidate.get("next_ref"),
            "next_ref_m11_change": candidate.get(
                "next_ref_m11_raw_solid_angle_relative_l2"
            ),
        })
    return selected


def contrast_selection_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    source = mesh_contrast_rows(rows)
    groups = sorted({(row["ka"], row["ri"]) for row in source})
    for ka, ri in groups:
        values = sorted(
            (
                row for row in source
                if (row["ka"], row["ri"]) == (ka, ri)
            ),
            key=lambda row: row["ref"],
        )
        candidate = next((row for row in values if mesh_target_passed(row)), None)
        if candidate is None:
            selected.append({
                "ka": ka,
                "ri": ri,
                "status": "target_not_reached",
                "largest_completed_ref": values[-1]["ref"],
                "largest_completed_ppw_shortest": values[-1][
                    "points_per_shortest_wavelength"
                ],
                "largest_completed_m11_error": values[-1][
                    "mie_m11_raw_solid_angle_relative_l2"
                ],
            })
            continue
        confirmation = mesh_target_passed(candidate, "next_ref_")
        selected.append({
            "ka": ka,
            "ri": ri,
            "status": (
                "confirmed_by_exact_reference_and_next_ref"
                if confirmation else "confirmed_by_exact_reference"
            ),
            "minimum_ref": candidate["ref"],
            "minimum_ppw_shortest": candidate[
                "points_per_shortest_wavelength"
            ],
            "system_dofs": candidate["system_dofs"],
            "mesh_triangles": candidate["mesh_triangles"],
            "m11_error": candidate["mie_m11_raw_solid_angle_relative_l2"],
            "full_normalized_error": candidate[
                "mie_full_normalized_solid_angle_relative_l2"
            ],
            "forward_error": candidate["mie_m11_forward_relative_error"],
            "integral_error": candidate["mie_m11_integral_relative_error"],
            "next_ref": candidate.get("next_ref"),
            "next_ref_m11_change": candidate.get(
                "next_ref_m11_raw_solid_angle_relative_l2"
            ),
        })
    return selected


def write_mesh_selection_tex(
    path: Path, selection: list[dict[str, Any]]
) -> None:
    status_labels = {
        "confirmed_by_exact_reference_and_next_ref": "Ми + соседняя сетка",
        "confirmed_by_exact_reference": "точный эталон Ми",
        "target_not_reached": "порог не достигнут",
    }
    lines = [
        r"\begin{tabular}{rrrrrrl}",
        r"\toprule",
        r"$ka$ & \texttt{ref} & $P_{\min}$ & $N$ & "
        r"$E_{11}^{\rm raw}$ & $E_{\rm full}$ & статус\\",
        r"\midrule",
    ]
    for row in selection:
        if row["status"] == "target_not_reached":
            lines.append(
                f"{float(row['ka']):g} & --- & --- & --- & "
                f"{100.0 * float(row['largest_completed_m11_error']):.3f}\\% "
                f"(ref={int(row['largest_completed_ref'])}) & --- & "
                f"{status_labels[row['status']]}\\\\"
            )
            continue
        lines.append(
            f"{float(row['ka']):g} & {int(row['minimum_ref'])} & "
            f"{float(row['minimum_ppw_shortest']):.2f} & "
            f"{int(row['system_dofs'])} & "
            f"{100.0 * float(row['m11_error']):.3f}\\% & "
            f"{100.0 * float(row['full_normalized_error']):.3f}\\% & "
            f"{status_labels[row['status']]}\\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    atomic_text(path, "\n".join(lines) + "\n")


def local_scaling_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = [row for row in mesh_sphere_rows(rows) if not row.get("resumed")]
    result: list[dict[str, Any]] = []
    fields = ("wall_time_s", "max_rss_mib", "gpu_memory_peak_delta_mib")
    for ka in sorted({row["ka"] for row in source}):
        values = sorted(
            (row for row in source if row["ka"] == ka),
            key=lambda row: row["ref"],
        )
        for coarse, fine in zip(values, values[1:]):
            n0 = float(coarse["system_dofs"])
            n1 = float(fine["system_dofs"])
            row: dict[str, Any] = {
                "ka": ka,
                "ref_from": coarse["ref"],
                "ref_to": fine["ref"],
                "dofs_from": int(n0),
                "dofs_to": int(n1),
                "dof_ratio": n1 / n0,
            }
            for field in fields:
                y0 = coarse.get(field)
                y1 = fine.get(field)
                if y0 and y1 and y0 > 0 and y1 > 0:
                    row[field + "_ratio"] = float(y1) / float(y0)
                    row[field + "_local_exponent_vs_dofs"] = math.log(
                        float(y1) / float(y0)
                    ) / math.log(n1 / n0)
            result.append(row)
    return result


def fit_log_power_model(
    rows: list[dict[str, Any]], response: str, include_iterations: bool = False
) -> dict[str, Any] | None:
    usable = [
        row for row in rows
        if not row.get("resumed")
        and row.get(response) is not None and float(row[response]) > 0
        and row.get("system_dofs") is not None
        and float(row["system_dofs"]) > 0
        and row.get("ka") is not None and float(row["ka"]) > 0
        and (not include_iterations or (
            row.get("iterations_x") is not None
            and float(row["iterations_x"]) > 0
        ))
    ]
    parameter_count = 4 if include_iterations else 3
    if len(usable) <= parameter_count:
        return None
    names = ["intercept", "log_system_dofs", "log_ka"]
    columns = [
        np.ones(len(usable)),
        np.log([float(row["system_dofs"]) for row in usable]),
        np.log([float(row["ka"]) for row in usable]),
    ]
    if include_iterations:
        names.append("log_iterations_x")
        columns.append(np.log([float(row["iterations_x"]) for row in usable]))
    design = np.column_stack(columns)
    target = np.log([float(row[response]) for row in usable])
    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ beta
    residual = target - fitted
    degrees = len(usable) - design.shape[1]
    residual_sum = float(residual @ residual)
    variance = residual_sum / degrees
    covariance = variance * np.linalg.pinv(design.T @ design)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    total_sum = float(((target - target.mean()) ** 2).sum())
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0
    return {
        "response": response,
        "sample_count": len(usable),
        "formula": "log(y) = " + " + ".join(names),
        "coefficients": {
            name: {
                "estimate": float(value),
                "standard_error": float(error),
                "approximate_95_percent_interval": [
                    float(value - 1.96 * error),
                    float(value + 1.96 * error),
                ],
            }
            for name, value, error in zip(names, beta, standard_error)
        },
        "r_squared_log_space": r_squared,
        "residual_degrees_of_freedom": degrees,
        "scope": "cold non-resumed FP64 sphere m=1.3 points only",
        "warning": (
            "Descriptive OLS fit, not an a-priori complexity theorem; "
            "tree depth/order changes and heteroscedastic timing are not modeled."
        ),
    }


def resource_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = mesh_sphere_rows(rows)
    models = [
        fit_log_power_model(source, "wall_time_s"),
        fit_log_power_model(source, "wall_time_s", include_iterations=True),
        fit_log_power_model(source, "max_rss_mib"),
        fit_log_power_model(source, "gpu_memory_peak_delta_mib"),
    ]
    return [model for model in models if model is not None]


def production_resource_scaling_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row.get("in_current_plan")
        and row.get("phase") == "resource_scaling"
        and not row.get("resumed")
        and row.get("wall_time_s") is not None
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in selected:
        key = (
            row.get("ka"), row.get("ref"), row.get("system_dofs"),
            row.get("mesh_triangles"),
        )
        groups.setdefault(key, []).append(row)

    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    def sample_std(values: list[float]) -> float | None:
        return float(np.std(values, ddof=1)) if len(values) > 1 else None

    aggregated: list[dict[str, Any]] = []
    for (ka, ref, dofs, triangles), group in groups.items():
        group.sort(key=lambda row: int(row.get("repeat") or 0))
        cold_candidates = [row for row in group if int(row.get("repeat") or 0) == 0]
        warm = [row for row in group if int(row.get("repeat") or 0) > 0]
        if not cold_candidates:
            continue
        cold = cold_candidates[0]
        warm_wall = [float(row["wall_time_s"]) for row in warm]
        warm_rss = [
            float(row["max_rss_mib"]) for row in warm
            if row.get("max_rss_mib") is not None
        ]
        warm_vram = [
            float(row["gpu_memory_peak_delta_mib"]) for row in warm
            if row.get("gpu_memory_peak_delta_mib") is not None
        ]
        warm_energy = [
            float(row["gpu_incremental_energy_j"]) for row in warm
            if row.get("gpu_incremental_energy_j") is not None
        ]
        warm_mean = mean(warm_wall)
        cold_wall = float(cold["wall_time_s"])
        aggregated.append({
            "ka": ka,
            "ref": ref,
            "system_dofs": dofs,
            "mesh_triangles": triangles,
            "repeat_count": len(group),
            "cold_wall_time_s": cold_wall,
            "warm_wall_time_mean_s": warm_mean,
            "warm_wall_time_sample_std_s": sample_std(warm_wall),
            "cold_to_warm_speedup": (
                cold_wall / warm_mean if warm_mean and warm_mean > 0 else None
            ),
            "cold_max_rss_mib": cold.get("max_rss_mib"),
            "warm_max_rss_mean_mib": mean(warm_rss),
            "cold_gpu_memory_peak_delta_mib": cold.get(
                "gpu_memory_peak_delta_mib"
            ),
            "warm_gpu_memory_peak_delta_mean_mib": mean(warm_vram),
            "cold_gpu_incremental_energy_j": cold.get(
                "gpu_incremental_energy_j"
            ),
            "warm_gpu_incremental_energy_mean_j": mean(warm_energy),
            "gpu_total_memory_mib": cold.get("gpu_total_memory_mib"),
        })
    return sorted(aggregated, key=lambda row: (float(row["ka"]), int(row["ref"])))


def production_resource_limit(
    scaling: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not scaling:
        return None
    largest = max(scaling, key=lambda row: int(row["system_dofs"]))
    refinement_growth = 4.0
    capacity = largest.get("gpu_total_memory_mib")
    predicted_vram = (
        refinement_growth * float(largest["cold_gpu_memory_peak_delta_mib"])
        if largest.get("cold_gpu_memory_peak_delta_mib") is not None else None
    )
    return {
        "status": "linear_storage_projection_not_a_measurement",
        "measured_maximum": largest,
        "projected_next_ref": int(largest["ref"]) + 1,
        "projected_system_dofs": int(round(
            refinement_growth * int(largest["system_dofs"])
        )),
        "assumed_refinement_storage_growth": refinement_growth,
        "projected_gpu_memory_peak_delta_mib": predicted_vram,
        "gpu_total_memory_mib": capacity,
        "projected_to_capacity_ratio": (
            predicted_vram / float(capacity)
            if predicted_vram is not None and capacity else None
        ),
        "fits_measured_gpu": (
            predicted_vram <= float(capacity)
            if predicted_vram is not None and capacity else None
        ),
    }


def automatic_recommendations(
    rows: list[dict[str, Any]], mesh_selection: list[dict[str, Any]],
    contrast_selection: list[dict[str, Any]],
) -> dict[str, Any]:
    fmm_mueller_difference_budget = 1.0e-3
    source = [
        row for row in rows
        if row.get("in_current_plan")
        and row.get("repeat") == 0
        and not row.get("resumed")
    ]

    def converged(row: dict[str, Any]) -> bool:
        tolerance = float(row.get("tolerance") or 0.0)
        residuals = [
            float(value) for value in (row.get("residual_x"), row.get("residual_y"))
            if value is not None
        ]
        return bool(residuals) and all(
            value <= 5.0 * tolerance for value in residuals
        )

    def summarize_choice(
        values: list[dict[str, Any]], parameter: str, error_field: str
    ) -> dict[str, Any] | None:
        candidates = [
            row for row in values
            if row.get(error_field) is not None
            and float(row[error_field]) <= fmm_mueller_difference_budget
            and row.get("effective_cold_wall_time_s") is not None
            and converged(row)
            and (
                row.get("shape") != "sphere"
                or row.get("mie_m11_raw_solid_angle_relative_l2") is None
                or mesh_target_passed(row)
            )
        ]
        if not candidates:
            return None
        candidates = prefer_cold_measurements(candidates)
        best = min(
            candidates,
            key=lambda row: float(row["effective_cold_wall_time_s"]),
        )
        return {
            "parameter": parameter,
            "selected_value": best.get(parameter),
            "wall_time_s": best.get("wall_time_s"),
            "effective_cold_wall_time_s": best.get(
                "effective_cold_wall_time_s"
            ),
            "solve_total_s": best.get("solve_total_s"),
            "fmm_induced_mueller_relative_difference": best.get(error_field),
            "mie_full_normalized_relative_error": best.get(
                "mie_full_normalized_solid_angle_relative_l2"
            ),
            "residual_x": best.get("residual_x"),
            "residual_y": best.get("residual_y"),
            "max_rss_mib": best.get("max_rss_mib"),
            "gpu_memory_peak_delta_mib": best.get("gpu_memory_peak_delta_mib"),
            "cache_state_actual": best.get("cache_state_actual"),
            "tested_values": sorted({row.get(parameter) for row in values}),
        }

    one_factor: list[dict[str, Any]] = []
    factor_definitions = (
        (
            "digits_effective", {"fmm_digits"},
            (
                "shape", "ka", "ri", "ref", "edge_mode", "quad",
                "duffy_order", "near_radius", "max_leaf_effective",
            ),
        ),
        (
            "near_radius",
            {
                "fmm_near_radius", "fmm_radius_scale",
                "fmm_radius_cold_audit", "fmm_radius_dependency",
                "fmm_shared_cold_audit", "mesh_sphere_m13",
            },
            (
                "shape", "ka", "ri", "ref", "edge_mode", "quad",
                "duffy_order", "digits_effective", "max_leaf_effective",
            ),
        ),
        (
            "max_leaf_effective", {"fmm_leaf", "fmm_shared_cold_audit"},
            (
                "shape", "ka", "ri", "ref", "edge_mode", "quad",
                "duffy_order", "digits_effective", "near_radius",
            ),
        ),
    )
    for parameter, phases, context_fields in factor_definitions:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in source:
            if row.get("phase") in phases:
                groups.setdefault(
                    tuple(row.get(field) for field in context_fields), []
                ).append(row)
        for context, values in groups.items():
            if len({row.get(parameter) for row in values}) < 2:
                continue
            choice = summarize_choice(
                values, parameter,
                "fmm_reference_full_normalized_solid_angle_relative_l2",
            )
            one_factor.append({
                "context": dict(zip(context_fields, context)),
                "selection": choice,
                "completed_distinct_values": len({
                    row.get(parameter) for row in values
                }),
            })

    grid_rows = [
        row for row in source if row.get("phase") == "fmm_radius_digits_grid"
    ]
    grid_candidates = [
        row for row in grid_rows
        if row.get(
            "fmm_grid_reference_full_normalized_solid_angle_relative_l2"
        ) is not None
        and float(row[
            "fmm_grid_reference_full_normalized_solid_angle_relative_l2"
        ]) <= fmm_mueller_difference_budget
        and row.get("effective_cold_wall_time_s") is not None
        and converged(row)
        and (
            row.get("shape") != "sphere"
            or row.get("mie_m11_raw_solid_angle_relative_l2") is None
            or mesh_target_passed(row)
        )
    ]
    grid_choice = None
    if grid_candidates:
        grid_candidates = prefer_cold_measurements(grid_candidates)
        best = min(
            grid_candidates,
            key=lambda row: float(row["effective_cold_wall_time_s"]),
        )
        grid_choice = {
            "near_radius": best.get("near_radius"),
            "digits_effective": best.get("digits_effective"),
            "wall_time_s": best.get("wall_time_s"),
            "effective_cold_wall_time_s": best.get(
                "effective_cold_wall_time_s"
            ),
            "fmm_induced_mueller_relative_difference": best.get(
                "fmm_grid_reference_full_normalized_solid_angle_relative_l2"
            ),
            "gpu_memory_peak_delta_mib": best.get(
                "gpu_memory_peak_delta_mib"
            ),
        }
    return {
        "status": "interim_until_all_planned_series_complete",
        "physical_mesh_relative_error_budget": 1.0e-2,
        "fmm_induced_mueller_relative_difference_budget": (
            fmm_mueller_difference_budget
        ),
        "mesh": mesh_selection,
        "mesh_contrast": contrast_selection,
        "fmm_one_factor": one_factor,
        "fmm_radius_digits_grid": {
            "selection": grid_choice,
            "completed_points": len(grid_rows),
            "planned_points": 9,
        },
        "warning": (
            "Agreement with the largest tested FMM radius is not an absolute "
            "accuracy proof. Sphere candidates also have to pass the exact "
            "Mie test; every non-sphere selection remains local to its "
            "recorded context until ka/contrast/shape transfer and next-ref "
            "confirmation are complete."
        ),
    }


def write_resource_models_tex(
    path: Path, models: list[dict[str, Any]]
) -> None:
    labels = {
        "wall_time_s": "полное время, с",
        "max_rss_mib": "RSS, МиБ",
        "gpu_memory_peak_delta_mib": "прирост VRAM, МиБ",
    }
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Отклик $y$ & $n$ & $C$ & $\alpha_N$ & $\beta_{ka}$ & $R^2$\\",
        r"\midrule",
    ]
    for model in models:
        coefficients = model["coefficients"]
        if "log_iterations_x" in coefficients:
            continue
        intercept = coefficients["intercept"]["estimate"]
        alpha = coefficients["log_system_dofs"]["estimate"]
        beta = coefficients["log_ka"]["estimate"]
        lines.append(
            f"{labels.get(model['response'], model['response'])} & "
            f"{model['sample_count']} & {math.exp(intercept):.3g} & "
            f"{alpha:.3f} & {beta:.3f} & "
            f"{model['r_squared_log_space']:.3f}\\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    atomic_text(path, "\n".join(lines) + "\n")


def plot_mie_convergence(rows: list[dict[str, Any]], output: Path) -> None:
    selected = mesh_sphere_rows(rows)
    if not selected:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ka in sorted({row["ka"] for row in selected}):
        values = sorted(
            (row for row in selected if row["ka"] == ka),
            key=lambda row: row["points_per_shortest_wavelength"],
        )
        x = [row["points_per_shortest_wavelength"] for row in values]
        axes[0].loglog(
            x, [row["mie_m11_raw_solid_angle_relative_l2"] for row in values],
            marker="o", label=f"ka={ka:g}",
        )
        axes[1].loglog(
            x, [row["mie_full_normalized_solid_angle_relative_l2"] for row in values],
            marker="o", label=f"ka={ka:g}",
        )
    for axis, title in zip(axes, (
        "Абсолютная ошибка $M_{11}$",
        "Ошибка формы полной матрицы Мюллера",
    )):
        axis.axhline(1.0e-2, color="black", linestyle="--", linewidth=1, label="1 %")
        axis.set_xlabel("Узлов P2 на кратчайшую длину волны")
        axis.set_ylabel("Относительная среднеквадратичная ошибка")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    figure.suptitle("Сходимость сферы к точному решению Ми, $m=1{,}3$")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_mesh_ref_ka_scaling(rows: list[dict[str, Any]], output: Path) -> None:
    selected = mesh_sphere_rows(rows)
    if not selected:
        return
    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    specifications = (
        ("mie_m11_raw_solid_angle_relative_l2", "Ошибка $M_{11}$", True),
        ("wall_time_s", "Полное время, с", True),
        ("system_dofs", "Число неизвестных", True),
        ("max_rss_mib", "Максимальный RSS, МиБ", True),
        ("gpu_memory_peak_delta_mib", "Прирост VRAM, МиБ", True),
        ("iterations_x", "Итерации первой поляризации", False),
    )
    performance_fields = {
        "wall_time_s", "max_rss_mib", "gpu_memory_peak_delta_mib",
        "iterations_x",
    }
    for ka in sorted({row["ka"] for row in selected}):
        values = sorted(
            (row for row in selected if row["ka"] == ka),
            key=lambda row: row["ref"],
        )
        for axis, (field, _, _) in zip(axes.flat, specifications):
            points = [
                row for row in values
                if row.get(field) is not None
                and not (field in performance_fields and row.get("resumed"))
            ]
            axis.plot(
                [row["ref"] for row in points],
                [row[field] for row in points],
                marker="o", label=f"ka={ka:g}",
            )
    for axis, (field, title, logarithmic) in zip(axes.flat, specifications):
        if logarithmic:
            axis.set_yscale("log")
        if field == "mie_m11_raw_solid_angle_relative_l2":
            axis.axhline(1.0e-2, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Уровень равномерного сгущения ref")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(ncol=2, fontsize=8)
    figure.suptitle(
        "Сходимость и стоимость по уровню сетки и размерному параметру, "
        "$m=1{,}3$"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_mesh_ka_ref_heatmaps(rows: list[dict[str, Any]], output: Path) -> None:
    selected = mesh_sphere_rows(rows)
    if not selected:
        return
    kas = sorted({float(row["ka"]) for row in selected})
    refs = sorted({int(row["ref"]) for row in selected})
    specifications = (
        ("mie_m11_raw_solid_angle_relative_l2", "Ошибка $M_{11}$", True),
        ("points_per_shortest_wavelength", "Узлов на кратчайшую волну", False),
        ("wall_time_s", "Полное время, с", True),
        ("max_rss_mib", "Максимальный RSS, МиБ", True),
        ("gpu_memory_peak_delta_mib", "Прирост VRAM, МиБ", True),
        ("system_dofs", "Число неизвестных", True),
    )
    lookup = {(float(row["ka"]), int(row["ref"])): row for row in selected}
    performance_fields = {
        "wall_time_s", "max_rss_mib", "gpu_memory_peak_delta_mib",
    }
    figure, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for axis, (field, title, logarithmic) in zip(axes.flat, specifications):
        matrix = np.full((len(kas), len(refs)), np.nan)
        for i, ka in enumerate(kas):
            for j, ref in enumerate(refs):
                value = lookup.get((ka, ref), {}).get(field)
                if field in performance_fields and lookup.get((ka, ref), {}).get("resumed"):
                    value = None
                if value is not None and (not logarithmic or float(value) > 0):
                    matrix[i, j] = math.log10(float(value)) if logarithmic else float(value)
        image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(refs)), refs)
        axis.set_yticks(range(len(kas)), [f"{value:g}" for value in kas])
        axis.set_xlabel("ref")
        axis.set_ylabel("ka")
        axis.set_title(title + (" (цвет: $\\log_{10}$)" if logarithmic else ""))
        figure.colorbar(image, ax=axis, shrink=0.82)
        for i in range(len(kas)):
            for j in range(len(refs)):
                value = matrix[i, j]
                if np.isfinite(value):
                    display = 10.0 ** value if logarithmic else value
                    label = f"{display:.2g}"
                    axis.text(j, i, label, ha="center", va="center", fontsize=7,
                              color="white" if value < np.nanmedian(matrix) else "black")
    figure.suptitle("Карта рассчитанных сеток сферы, $m=1{,}3$")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_extended_resources(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [row for row in mesh_sphere_rows(rows) if not row.get("resumed")]
    if not selected:
        return
    specifications = (
        ("cpu_user_s", "Пользовательское CPU-время, с", True),
        ("cpu_system_s", "Системное CPU-время, с", True),
        ("gpu_utilization_mean_percent", "Средняя загрузка GPU, %", False),
        ("gpu_power_mean_w", "Средняя мощность GPU, Вт", False),
        ("gpu_energy_j", "Энергия всей GPU, Дж", True),
        ("gpu_incremental_energy_j", "Добавочная энергия GPU, Дж", True),
        ("fmm_setup_s", "Построение FMM, с", True),
        ("mbj_setup_s", "Построение MBJ, с", True),
        ("solve_total_s", "Два решения, с", True),
    )
    for row in selected:
        values = [row.get("solve_x_s"), row.get("solve_y_s")]
        if all(value is not None for value in values):
            row["solve_total_s"] = float(values[0]) + float(values[1])
    figure, axes = plt.subplots(3, 3, figsize=(17, 13.5))
    for ka in sorted({row["ka"] for row in selected}):
        values = sorted(
            (row for row in selected if row["ka"] == ka),
            key=lambda row: row["ref"],
        )
        for axis, (field, _, _) in zip(axes.flat, specifications):
            points = [
                row for row in values
                if row.get(field) is not None and float(row[field]) > 0
            ]
            axis.plot(
                [row["ref"] for row in points],
                [row[field] for row in points],
                marker="o", label=f"ka={ka:g}",
            )
    for axis, (_, title, logarithmic) in zip(axes.flat, specifications):
        if logarithmic:
            axis.set_yscale("log")
        axis.set_xlabel("Уровень равномерного сгущения ref")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(ncol=2, fontsize=8)
    figure.suptitle("Расширенные ресурсы строгих расчётов сферы, $m=1{,}3$")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_hardware_utilization(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [row for row in mesh_sphere_rows(rows) if not row.get("resumed")]
    if not selected:
        return
    specifications = (
        ("process_cpu_mean_percent", "CPU процесса, % одного ядра", False),
        ("system_cpu_mean_percent", "Загрузка всей системы CPU, %", False),
        ("process_threads_max", "Максимум потоков процесса", False),
        ("cpu_frequency_mean_mhz", "Средняя частота CPU, МГц", False),
        ("system_memory_available_min_mib", "Минимум свободной RAM, МиБ", True),
        ("system_swap_used_max_mib", "Максимум занятого swap, МиБ", True),
        ("gpu_utilization_mean_percent", "Средняя загрузка GPU, %", False),
        ("gpu_memory_peak_delta_mib", "Прирост VRAM, МиБ", True),
        ("gpu_power_mean_w", "Средняя мощность GPU, Вт", False),
        ("gpu_temperature_max_c", "Максимальная температура GPU, °C", False),
        ("gpu_sm_clock_mean_mhz", "Средняя частота SM, МГц", False),
        ("gpu_incremental_energy_j", "Добавочная энергия GPU, Дж", True),
    )
    figure, axes = plt.subplots(4, 3, figsize=(17, 18.0))
    for ka in sorted({row["ka"] for row in selected}):
        values = sorted(
            (row for row in selected if row["ka"] == ka),
            key=lambda row: row["ref"],
        )
        for axis, (field, _, _) in zip(axes.flat, specifications):
            points = [
                row for row in values
                if row.get(field) is not None and float(row[field]) > 0
            ]
            axis.plot(
                [row["ref"] for row in points],
                [row[field] for row in points],
                marker="o", label=f"ka={ka:g}",
            )
    for axis, (_, title, logarithmic) in zip(axes.flat, specifications):
        if logarithmic:
            axis.set_yscale("log")
        axis.set_xlabel("Уровень равномерного сгущения ref")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
    axes[0, 1].legend(ncol=2, fontsize=8)
    figure.suptitle("Загрузка и режим оборудования в строгих расчётах сферы")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_storage_resources(rows: list[dict[str, Any]], output: Path) -> None:
    candidates = [
        row for row in rows
        if row.get("in_current_plan")
        and row.get("repeat") == 0
        and row.get("shape") == "sphere"
        and float(row.get("ri") or 0.0) == 1.3
        and row.get("phase") in {"mesh_sphere_m13", "memory_gates"}
        and not row.get("resumed")
    ]
    by_grid: dict[tuple[float, int], dict[str, Any]] = {}
    for row in candidates:
        key = (float(row["ka"]), int(row["ref"]))
        if key not in by_grid or row.get("phase") == "mesh_sphere_m13":
            by_grid[key] = row
    selected = list(by_grid.values())
    if not selected:
        return
    specifications = (
        ("near_cache_logical_mib", "Ближняя коррекция, МиБ"),
        ("mbj_cache_logical_mib", "Факторы MBJ, МиБ"),
        ("cache_allocated_mib", "Выделено на диске под кэш, МиБ"),
        ("output_logical_mib", "Размер результатов случая, МиБ"),
        ("process_read_mib", "Прочитано процессом, МиБ"),
        ("process_write_mib", "Записано процессом, МиБ"),
        ("disk_free_min_gib", "Минимум свободного диска, ГиБ"),
        ("cache_bytes_per_unknown", "Кэш на одну неизвестную, байт"),
    )
    figure, axes = plt.subplots(2, 4, figsize=(21, 9.5))
    for ka in sorted({row["ka"] for row in selected}):
        values = sorted(
            (row for row in selected if row["ka"] == ka),
            key=lambda row: row["ref"],
        )
        for axis, (field, _) in zip(axes.flat, specifications):
            points = [
                row for row in values
                if row.get(field) is not None and float(row[field]) > 0
                and not (
                    field in {"output_logical_mib", "process_write_mib"}
                    and row.get("phase") == "memory_gates"
                )
            ]
            axis.plot(
                [row["ref"] for row in points],
                [row[field] for row in points],
                marker="o", label=f"ka={ka:g}",
            )
    for axis, (field, title) in zip(axes.flat, specifications):
        axis.set_yscale("log")
        axis.set_xlabel("Уровень равномерного сгущения ref")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(ncol=2, fontsize=8)
    figure.suptitle("Дисковое хранение и ввод-вывод строгих расчётов сферы")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_resource_scaling(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [
        row for row in rows
        if row.get("in_current_plan") and row.get("system_dofs")
        and row.get("wall_time_s") and not row.get("resumed")
    ]
    if not selected:
        return
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    phases = sorted({row["phase"] for row in selected})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(phases), 1)))
    for phase, color in zip(phases, colors):
        values = [row for row in selected if row["phase"] == phase]
        axes[0].scatter(
            [row["system_dofs"] for row in values],
            [row["wall_time_s"] for row in values],
            label=phase, color=color, alpha=0.8,
        )
        rss_values = [row for row in values if row.get("max_rss_mib") is not None]
        axes[1].scatter(
            [row["system_dofs"] for row in rss_values],
            [row["max_rss_mib"] for row in rss_values],
            label=phase, color=color, alpha=0.8,
        )
        gpu_values = [row for row in values if row.get("gpu_memory_peak_delta_mib") is not None]
        axes[2].scatter(
            [row["system_dofs"] for row in gpu_values],
            [row["gpu_memory_peak_delta_mib"] for row in gpu_values],
            label=phase, color=color, alpha=0.8,
        )
    labels = (
        ("Полное время процесса, с", "Время"),
        ("Максимальный RSS, МиБ", "Оперативная память"),
        ("Прирост занятой VRAM, МиБ", "Видеопамять"),
    )
    for axis, (ylabel, title) in zip(axes, labels):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Полное число неизвестных")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    figure.suptitle("Накопленные измерения вычислительных ресурсов")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_production_resource_scaling(
    scaling: list[dict[str, Any]], output: Path
) -> None:
    if not scaling:
        return
    ka = [float(row["ka"]) for row in scaling]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(
        ka, [row["cold_wall_time_s"] for row in scaling],
        marker="o", color="#2f6f4e", label="холодный запуск",
    )
    axes[0, 0].plot(
        ka, [row["warm_wall_time_mean_s"] for row in scaling],
        marker="s", color="#2878b5", label="готовые кэши",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Полное время, с")
    axes[0, 0].legend()

    axes[0, 1].plot(
        ka, [row["cold_max_rss_mib"] for row in scaling],
        marker="o", color="#d17a22", label="RAM",
    )
    axes[0, 1].plot(
        ka, [row["cold_gpu_memory_peak_delta_mib"] for row in scaling],
        marker="s", color="#7b4fa3", label="VRAM",
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Пиковая память, МиБ")
    axes[0, 1].legend()

    axes[1, 0].plot(
        ka, [row["cold_to_warm_speedup"] for row in scaling],
        marker="o", color="#2f6f4e",
    )
    axes[1, 0].axhline(1.0, color="black", linewidth=1.0)
    axes[1, 0].set_ylabel("Ускорение от готовых кэшей, раз")

    energy_rows = [
        row for row in scaling
        if row.get("cold_gpu_incremental_energy_j") is not None
    ]
    axes[1, 1].plot(
        [row["ka"] for row in energy_rows],
        [row["cold_gpu_incremental_energy_j"] for row in energy_rows],
        marker="o", color="#b33c3c", label="холодный запуск",
    )
    warm_energy = [
        row for row in scaling
        if row.get("warm_gpu_incremental_energy_mean_j") is not None
    ]
    axes[1, 1].plot(
        [row["ka"] for row in warm_energy],
        [row["warm_gpu_incremental_energy_mean_j"] for row in warm_energy],
        marker="s", color="#2878b5", label="готовые кэши",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("Добавочная энергия GPU, Дж")
    axes[1, 1].legend()

    titles = (
        "Полное стеночное время", "Оперативная и видеопамять",
        "Польза повторного использования кэшей", "Энергия GPU",
    )
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("Электрический размер ka")
        axis.grid(True, which="both", alpha=0.3)
    for axis in axes.flat:
        for x, row in zip(ka, scaling):
            axis.annotate(
                f"ref={int(row['ref'])}", (x, axis.get_ylim()[0]),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )
    figure.suptitle(
        "Шестигранная призма: масштабирование смешанного H(div)-расчёта"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_fmm_radius(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [
        row for row in rows
        if row.get("in_current_plan")
        and row.get("phase") in {
            "fmm_near_radius", "fmm_radius_scale", "fmm_radius_cold_audit",
            "fmm_radius_dependency", "fmm_shared_cold_audit",
            "mesh_sphere_m13",
        }
        and row.get("fmm_reference_full_normalized_solid_angle_relative_l2") is not None
    ]
    if not selected:
        return
    unique: dict[tuple[Any, Any, Any, Any, Any], dict[str, Any]] = {}
    for row in selected:
        key = (
            row["shape"], row["ri"], row["ka"], row["ref"],
            row["near_radius"],
        )
        previous = unique.get(key)
        if previous is None or timing_preference(row) < timing_preference(previous):
            unique[key] = row
    figure, axes_grid = plt.subplots(2, 2, figsize=(15.5, 10.0))
    axes = axes_grid.ravel()
    groups = sorted({key[:4] for key in unique})
    for shape, ri, ka, ref in groups:
        values = sorted(
            (
                row
                for (row_shape, row_ri, row_ka, row_ref, _), row
                in unique.items()
                if (row_shape, row_ri, row_ka, row_ref)
                == (shape, ri, ka, ref)
            ),
            key=lambda row: row["near_radius"],
        )
        label = f"{shape}, m={ri:g}, ka={ka:g}, ref={ref}"
        radius = [row["near_radius"] for row in values]
        error = [
            max(row["fmm_reference_full_normalized_solid_angle_relative_l2"], 1e-12)
            for row in values
        ]
        exact_fields = (
            "mie_m11_raw_solid_angle_relative_l2",
            "mie_m11_forward_relative_error",
            "mie_m11_integral_relative_error",
            "mie_full_normalized_solid_angle_relative_l2",
            "mie_maximum_normalized_absolute_error",
        )
        exact_values = [
            max(
                float(row[field]) for field in exact_fields
                if row.get(field) is not None
            ) if any(row.get(field) is not None for field in exact_fields)
            else None
            for row in values
        ]
        exact_pairs = [
            (value[0], value[1]) for value in zip(radius, exact_values)
            if value[1] is not None
        ]
        if exact_pairs:
            axes[0].plot(
                [value[0] for value in exact_pairs],
                [100.0 * float(value[1]) for value in exact_pairs],
                marker="o", label=label,
            )
        axes[1].plot(
            radius, [100.0 * value for value in error],
            marker="o", label=label,
        )
        axes[2].plot(
            radius, [row.get("effective_cold_wall_time_s") for row in values],
            marker="o", label=label,
        )
        axes[3].plot(
            radius, [
                row.get("effective_cold_gpu_memory_peak_delta_mib")
                for row in values
            ],
            marker="o", label=label,
        )
    axes[0].set_yscale("log")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Максимум из пяти ошибок относительно Ми, %")
    axes[0].set_title("Абсолютная проверка сферы")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Отличие от максимального радиуса, %")
    axes[1].set_title("Внутренняя FMM-сходимость")
    axes[2].set_ylabel("Полное холодное время, с")
    axes[2].set_title("Сборка оператора и решение")
    axes[3].set_ylabel("Прирост VRAM, МиБ")
    axes[3].set_title("Пиковая видеопамять")
    for axis in axes:
        axis.set_xlabel("Радиус прямой ближней зоны, ячеек")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    figure.suptitle("Влияние радиуса ближней зоны FMM")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_fmm_parameter_tradeoffs(
    rows: list[dict[str, Any]], output: Path
) -> None:
    factors = (
        (
            "digits_effective", {"fmm_digits"},
            "Фактический индекс точности digits",
        ),
        (
            "near_radius",
            {
                "fmm_near_radius", "fmm_radius_scale",
                "fmm_radius_cold_audit", "fmm_shared_cold_audit",
                "mesh_sphere_m13",
            },
            "Радиус прямой зоны R",
        ),
        (
            "max_leaf_effective", {"fmm_leaf", "fmm_shared_cold_audit"},
            "Фактический размер листа",
        ),
    )
    selected = [
        row for row in rows
        if row.get("in_current_plan")
        and row.get("repeat") == 0
        and row.get("fmm_reference_full_normalized_solid_angle_relative_l2")
        is not None
    ]
    if not selected:
        return
    figure, axes = plt.subplots(3, 3, figsize=(17, 14))
    for row_index, (field, phases, x_label) in enumerate(factors):
        factor_rows = [row for row in selected if row.get("phase") in phases]
        factor_rows = preferred_unique_rows(
            factor_rows,
            (
                "shape", "ri", "ka", "ref", "edge_mode", "quad",
                "duffy_order", "digits_effective", "near_radius",
                "max_leaf_effective", "tolerance", field,
            ),
        )
        for ka in sorted({row["ka"] for row in factor_rows}):
            values = sorted(
                (row for row in factor_rows if row["ka"] == ka),
                key=lambda row: float(row[field]),
            )
            label = f"ka={ka:g}"
            x = [row[field] for row in values]
            axes[row_index, 0].plot(
                x,
                [
                    max(
                        float(row[
                            "fmm_reference_full_normalized_solid_angle_relative_l2"
                        ]),
                        1.0e-16,
                    )
                    for row in values
                ],
                marker="o", label=label,
            )
            time_rows = [
                row for row in values
                if not row.get("resumed")
                and row.get("effective_cold_wall_time_s") is not None
            ]
            axes[row_index, 1].plot(
                [row[field] for row in time_rows],
                [row["effective_cold_wall_time_s"] for row in time_rows],
                marker="o", label=label,
            )
            memory_rows = [
                row for row in values
                if not row.get("resumed")
                and row.get("effective_cold_gpu_memory_peak_delta_mib")
            ]
            axes[row_index, 2].plot(
                [row[field] for row in memory_rows],
                [
                    row["effective_cold_gpu_memory_peak_delta_mib"]
                    for row in memory_rows
                ],
                marker="o", label=label,
            )
        axes[row_index, 0].axhline(
            1.0e-3, color="black", linestyle="--", linewidth=1,
            label="0,1 %" if row_index == 0 else None,
        )
        axes[row_index, 0].set_yscale("log")
        axes[row_index, 1].set_yscale("log")
        axes[row_index, 2].set_yscale("log")
        for column, title in enumerate((
            "Отличие от строгого FMM-эталона",
            "Полное холодное время, с",
            "Прирост VRAM, МиБ",
        )):
            axes[row_index, column].set_xlabel(x_label)
            axes[row_index, column].set_ylabel(title)
            axes[row_index, column].grid(True, which="both", alpha=0.3)
        if factor_rows:
            axes[row_index, 0].legend(fontsize=8)
    figure.suptitle(
        "Связанный компромисс точности, времени и памяти параметров FMM"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_fmm_radius_digits_grid(
    rows: list[dict[str, Any]], output: Path
) -> None:
    selected = [
        row for row in rows
        if row.get("in_current_plan")
        and row.get("repeat") == 0
        and row.get("phase") == "fmm_radius_digits_grid"
    ]
    if not selected:
        return
    radii = sorted({int(row["near_radius"]) for row in selected})
    digits = sorted({int(row["digits_effective"]) for row in selected})
    specifications = (
        (
            "fmm_grid_reference_full_normalized_solid_angle_relative_l2",
            "Отличие от строгой пары $(R,d)=(5,7)$", True,
        ),
        ("wall_time_s", "Полное время, с", True),
        ("gpu_memory_peak_delta_mib", "Прирост VRAM, МиБ", True),
    )
    lookup = {
        (int(row["near_radius"]), int(row["digits_effective"])): row
        for row in selected
    }
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    for axis, (field, title, logarithmic) in zip(axes, specifications):
        matrix = np.full((len(radii), len(digits)), np.nan)
        raw = np.full_like(matrix, np.nan)
        for i, radius in enumerate(radii):
            for j, digit in enumerate(digits):
                value = lookup.get((radius, digit), {}).get(field)
                if value is None:
                    continue
                display_value = float(value)
                plot_value = display_value
                if logarithmic and plot_value <= 0.0:
                    if field.startswith("fmm_grid_reference_"):
                        plot_value = 1.0e-16
                    else:
                        continue
                raw[i, j] = display_value
                matrix[i, j] = (
                    math.log10(plot_value) if logarithmic else plot_value
                )
        image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(digits)), digits)
        axis.set_yticks(range(len(radii)), radii)
        axis.set_xlabel("Фактический индекс digits")
        axis.set_ylabel("Радиус прямой зоны R")
        axis.set_title(title + (" (цвет: $\\log_{10}$)" if logarithmic else ""))
        figure.colorbar(image, ax=axis, shrink=0.82)
        finite = matrix[np.isfinite(matrix)]
        midpoint = float(np.median(finite)) if finite.size else 0.0
        for i in range(len(radii)):
            for j in range(len(digits)):
                if np.isfinite(raw[i, j]):
                    axis.text(
                        j, i, f"{raw[i, j]:.2g}", ha="center", va="center",
                        fontsize=8,
                        color="white" if matrix[i, j] < midpoint else "black",
                    )
    figure.suptitle("Совместный выбор радиуса и порядка FMM")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_quadrature_tradeoffs(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [
        row for row in rows
        if row.get("in_current_plan")
        and row.get("repeat") == 0
        and row.get("phase") == "quadrature"
        and row.get(
            "quadrature_reference_full_normalized_solid_angle_relative_l2"
        ) is not None
    ]
    if not selected:
        return
    maximum_quad = max(int(row["quad"]) for row in selected)
    maximum_duffy = max(int(row["duffy_order"]) for row in selected)
    factor_rows = (
        (
            "quad", "Порядок обычной квадратуры",
            [row for row in selected if int(row["duffy_order"]) == maximum_duffy],
        ),
        (
            "duffy_order", "Порядок квадратуры Даффи",
            [row for row in selected if int(row["quad"]) == maximum_quad],
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for row_index, (field, x_label, source) in enumerate(factor_rows):
        labels = sorted({
            (row["shape"], row["ka"], row["ref"]) for row in source
        })
        for shape, ka, ref in labels:
            values = sorted(
                (
                    row for row in source
                    if (row["shape"], row["ka"], row["ref"])
                    == (shape, ka, ref)
                ),
                key=lambda row: float(row[field]),
            )
            label = f"{shape}, ka={ka:g}, ref={ref}"
            axes[row_index, 0].plot(
                [row[field] for row in values],
                [
                    max(float(row[
                        "quadrature_reference_"
                        "full_normalized_solid_angle_relative_l2"
                    ]), 1.0e-16)
                    for row in values
                ],
                marker="o", label=label,
            )
            time_rows = [
                row for row in values
                if not row.get("resumed") and row.get("wall_time_s")
            ]
            axes[row_index, 1].plot(
                [row[field] for row in time_rows],
                [row["wall_time_s"] for row in time_rows],
                marker="o", label=label,
            )
            memory_rows = [
                row for row in values
                if not row.get("resumed")
                and row.get("gpu_memory_peak_delta_mib")
            ]
            axes[row_index, 2].plot(
                [row[field] for row in memory_rows],
                [row["gpu_memory_peak_delta_mib"] for row in memory_rows],
                marker="o", label=label,
            )
        axes[row_index, 0].axhline(
            1.0e-3, color="black", linestyle="--", linewidth=1,
        )
        for column, title in enumerate((
            "Отличие от пары (13, 8)",
            "Полное холодное время, с",
            "Прирост VRAM, МиБ",
        )):
            axes[row_index, column].set_yscale("log")
            axes[row_index, column].set_xlabel(x_label)
            axes[row_index, column].set_ylabel(title)
            axes[row_index, column].grid(True, which="both", alpha=0.3)
        axes[row_index, 0].legend(fontsize=8)
    figure.suptitle(
        "Раздельная сходимость обычной и сингулярной квадратуры"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_farfield_grid(rows: list[dict[str, Any]], output: Path) -> None:
    selected = sorted(
        (
            row for row in rows
            if row.get("in_current_plan")
            and row.get("repeat") == 0
            and row.get("phase") == "farfield_grid"
            and row.get("angular_full_interpolation_relative_l2") is not None
        ),
        key=lambda row: int(row["ntheta"]),
    )
    if not selected:
        return
    ntheta = [row["ntheta"] for row in selected]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    axes[0, 0].plot(
        ntheta,
        [
            max(float(row["angular_full_interpolation_relative_l2"]), 1.0e-16)
            for row in selected
        ],
        marker="o",
    )
    axes[0, 1].plot(
        ntheta,
        [
            max(float(row["angular_m11_integral_relative_error"]), 1.0e-16)
            for row in selected
        ],
        marker="o", color="tab:orange",
    )
    farfield_rows = [row for row in selected if row.get("farfield_s") is not None]
    axes[1, 0].plot(
        [row["ntheta"] for row in farfield_rows],
        [row["farfield_s"] for row in farfield_rows],
        marker="o", color="tab:green",
    )
    output_rows = [row for row in selected if row.get("output_logical_mib") is not None]
    axes[1, 1].plot(
        [row["ntheta"] for row in output_rows],
        [row["output_logical_mib"] for row in output_rows],
        marker="o", color="tab:red",
    )
    specifications = (
        (axes[0, 0], "Ошибка восстановления всей матрицы", True),
        (axes[0, 1], r"Ошибка интеграла $M_{11}\sin\theta$", True),
        (axes[1, 0], "Время дальнего поля, с", False),
        (axes[1, 1], "Размер результатов случая, МиБ", False),
    )
    for axis, ylabel, logarithmic in specifications:
        if logarithmic:
            axis.set_yscale("log")
        axis.set_xlabel(r"Число выходных углов $N_\theta$")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.3)
    figure.suptitle("Сходимость и стоимость угловой сетки дальнего поля")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_solver_controls(rows: list[dict[str, Any]], output: Path) -> None:
    controls = (
        ("gmres_restart", "gmres_restart", "Рестарт GMRES", False),
        ("mbj_nodes", "mbj_nodes", "Узлов в ядре блока MBJ", False),
        ("mbj_overlap", "mbj_overlap", "Узлов перекрытия MBJ", False),
        ("solver_tolerance", "tolerance", "Алгебраический допуск", True),
    )
    metrics = (
        ("solve_total_s", "Решение двух поляризаций, с", False),
        ("iterations_total", "Итерации двух поляризаций", False),
        ("mbj_setup_s", "Построение/загрузка MBJ, с", True),
        ("max_rss_mib", "Максимальная RAM, МиБ", True),
        ("gpu_memory_peak_delta_mib", "Прирост VRAM, МиБ", True),
        (
            "solver_reference_full_normalized_solid_angle_relative_l2",
            "Отличие полной матрицы от строгого допуска", True,
        ),
    )
    selected = [
        row for row in rows
        if row.get("in_current_plan")
        and row.get("phase") == "solver_controls"
        and not row.get("resumed")
    ]
    if not selected:
        return
    figure, axes = plt.subplots(4, 6, figsize=(25.5, 14.5), squeeze=False)
    baseline = [row for row in selected if row.get("name") == "baseline"]
    for row_index, (name, x_field, x_label, x_log) in enumerate(controls):
        source = [
            row for row in selected
            if row.get("name") == name and row.get(x_field) is not None
        ]
        source.extend(
            row for row in baseline if row.get(x_field) is not None
        )
        x_values = sorted({float(row[x_field]) for row in source})
        for column, (field, ylabel, y_log) in enumerate(metrics):
            centers: list[float] = []
            lower: list[float] = []
            upper: list[float] = []
            retained_x: list[float] = []
            for x_value in x_values:
                values = [
                    (
                        max(float(row[field]), 1.0e-16)
                        if field.startswith("solver_reference_")
                        else float(row[field])
                    )
                    for row in source
                    if float(row[x_field]) == x_value
                    and row.get(field) is not None
                    and (
                        not y_log or float(row[field]) > 0.0
                        or field.startswith("solver_reference_")
                    )
                ]
                if not values:
                    continue
                center = float(np.median(values))
                retained_x.append(x_value)
                centers.append(center)
                lower.append(center - min(values))
                upper.append(max(values) - center)
            axis = axes[row_index, column]
            if retained_x:
                axis.errorbar(
                    retained_x, centers, yerr=np.asarray([lower, upper]),
                    marker="o", capsize=4,
                )
            if x_log:
                axis.set_xscale("log")
                axis.invert_xaxis()
            if y_log:
                axis.set_yscale("log")
            axis.set_xlabel(x_label)
            axis.set_ylabel(ylabel)
            axis.grid(True, which="both", alpha=0.3)
    figure.suptitle(
        "Однофакторный выбор параметров GMRES и MBJ: медиана и разброс повторов"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_basis_edge_control(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [
        row for row in rows
        if row.get("in_current_plan")
        and row.get("repeat") == 0
        and row.get("phase") == "basis_edge_control"
    ]
    if not selected:
        return
    specifications = (
        (
            "next_ref_full_normalized_solid_angle_relative_l2",
            "Изменение полной нормированной матрицы",
        ),
        (
            "next_ref_maximum_normalized_absolute_error",
            "Максимальное нормированное отличие",
        ),
        ("system_dofs", "Число неизвестных"),
        ("wall_time_s", "Полное холодное время, с"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    for edge_mode in sorted({row["edge_mode"] for row in selected}):
        values = sorted(
            (row for row in selected if row["edge_mode"] == edge_mode),
            key=lambda row: row["ref"],
        )
        for axis, (field, _) in zip(axes.flat, specifications):
            points = [
                row for row in values
                if row.get(field) is not None
                and not (field == "wall_time_s" and row.get("resumed"))
            ]
            axis.plot(
                [row["ref"] for row in points],
                [row[field] for row in points],
                marker="o", label=edge_mode,
            )
    for axis, (field, title) in zip(axes.flat, specifications):
        axis.set_yscale("log")
        if field.startswith("next_ref_"):
            axis.axhline(1.0e-2, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Уровень равномерного сгущения ref")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend()
    figure.suptitle(
        "Самосходимость базисов на шестигранной призме, $ka=10$, $m=1{,}3$"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_contrast_convergence(rows: list[dict[str, Any]], output: Path) -> None:
    selected = mesh_contrast_rows(rows)
    if not selected:
        return
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    labels = sorted({(row["ka"], row["ri"]) for row in selected})
    for ka, ri in labels:
        values = sorted(
            (
                row for row in selected
                if (row["ka"], row["ri"]) == (ka, ri)
            ),
            key=lambda row: row["ref"],
        )
        label = f"ka={ka:g}, m={ri:g}"
        axes[0, 0].loglog(
            [row["points_per_shortest_wavelength"] for row in values],
            [row["mie_m11_raw_solid_angle_relative_l2"] for row in values],
            marker="o", label=label,
        )
        axes[0, 1].semilogy(
            [row["ref"] for row in values],
            [
                row["mie_full_normalized_solid_angle_relative_l2"]
                for row in values
            ],
            marker="o", label=label,
        )
        clean = [row for row in values if not row.get("resumed")]
        axes[1, 0].loglog(
            [row["system_dofs"] for row in clean],
            [row["wall_time_s"] for row in clean],
            marker="o", label=label,
        )
        axes[1, 1].semilogx(
            [row["system_dofs"] for row in clean],
            [row["iterations_x"] for row in clean],
            marker="o", label=label,
        )
    axes[0, 0].axhline(1.0e-2, color="black", linestyle="--", linewidth=1)
    axes[0, 1].axhline(1.0e-2, color="black", linestyle="--", linewidth=1)
    titles = (
        "Ошибка масштаба $M_{11}$ по узлам на кратчайшую волну",
        "Ошибка формы полной матрицы по ref",
        "Полное холодное время по числу неизвестных",
        "Итерации первой поляризации по числу неизвестных",
    )
    xlabels = (
        "Узлов P2 на кратчайшую длину волны", "ref",
        "Число неизвестных", "Число неизвестных",
    )
    for axis, title, xlabel in zip(axes.flat, titles, xlabels):
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(True, which="both", alpha=0.3)
    axes[0, 0].set_ylabel("Относительная ошибка")
    axes[0, 1].set_ylabel("Относительная ошибка")
    axes[1, 0].set_ylabel("Время, с")
    axes[1, 1].set_ylabel("Итерации")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Влияние показателя преломления на сходимость сферы")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_prism_convergence(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [
        row for row in rows
        if row.get("in_current_plan")
        and row.get("repeat") == 0
        and row.get("phase") == "mesh_prism"
    ]
    if not selected:
        return
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    labels = sorted({(row["ka"], row["ri"]) for row in selected})
    for ka, ri in labels:
        values = sorted(
            (
                row for row in selected
                if (row["ka"], row["ri"]) == (ka, ri)
            ),
            key=lambda row: row["ref"],
        )
        label = f"ka={ka:g}, m={ri:g}"
        convergence = [
            row for row in values
            if row.get("next_ref_full_normalized_solid_angle_relative_l2")
            is not None
        ]
        axes[0, 0].loglog(
            [row["points_per_shortest_wavelength"] for row in convergence],
            [
                row["next_ref_full_normalized_solid_angle_relative_l2"]
                for row in convergence
            ],
            marker="o", label=label,
        )
        clean = [row for row in values if not row.get("resumed")]
        for axis, field in zip(
            (axes[0, 1], axes[1, 0], axes[1, 1]),
            ("wall_time_s", "max_rss_mib", "gpu_memory_peak_delta_mib"),
        ):
            points = [row for row in clean if row.get(field)]
            axis.loglog(
                [row["system_dofs"] for row in points],
                [row[field] for row in points],
                marker="o", label=label,
            )
    axes[0, 0].axhline(1.0e-2, color="black", linestyle="--", linewidth=1)
    specifications = (
        ("Изменение матрицы при следующем ref", "Узлов на кратчайшую волну",
         "Относительное изменение"),
        ("Полное холодное время", "Число неизвестных", "Время, с"),
        ("Максимальный RSS", "Число неизвестных", "RSS, МиБ"),
        ("Прирост VRAM", "Число неизвестных", "VRAM, МиБ"),
    )
    for axis, (title, xlabel, ylabel) in zip(axes.flat, specifications):
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Самосходимость и ресурсы шестигранной призмы")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_strict_hdiv_polyhedra(
    rows: list[dict[str, Any]], output: Path,
) -> None:
    selected = [
        row for row in rows
        if row.get("in_current_plan")
        and row.get("repeat") == 0
        and row.get("phase") == "mesh_polyhedra_hdiv_strict"
    ]
    if not selected:
        return
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    for shape in sorted({row["shape"] for row in selected}):
        values = sorted(
            (row for row in selected if row["shape"] == shape),
            key=lambda row: row["ref"],
        )
        label = "куб" if shape == "cube" else "шестигранная призма"
        convergence = [
            row for row in values
            if row.get("next_ref_full_normalized_solid_angle_relative_l2")
            is not None
        ]
        axes[0, 0].loglog(
            [row["points_per_shortest_wavelength"] for row in convergence],
            [
                row["next_ref_full_normalized_solid_angle_relative_l2"]
                for row in convergence
            ],
            marker="o", label=label,
        )
        m11 = [
            row for row in values
            if row.get("next_ref_m11_raw_solid_angle_relative_l2") is not None
        ]
        axes[0, 1].loglog(
            [row["points_per_shortest_wavelength"] for row in m11],
            [row["next_ref_m11_raw_solid_angle_relative_l2"] for row in m11],
            marker="o", label=label,
        )
        clean = [row for row in values if not row.get("resumed")]
        for axis, field in (
            (axes[1, 0], "wall_time_s"),
            (axes[1, 1], "gpu_memory_peak_delta_mib"),
        ):
            points = [row for row in clean if row.get(field) is not None]
            axis.loglog(
                [row["system_dofs"] for row in points],
                [row[field] for row in points],
                marker="o", label=label,
            )
    for axis in axes[0]:
        axis.axhline(1.0e-2, color="black", linestyle="--", linewidth=1)
    specifications = (
        (
            "Изменение полной нормированной матрицы при следующем ref",
            "Узлов на кратчайшую длину волны", "Относительное изменение",
        ),
        (
            "Изменение ненормированного $M_{11}$ при следующем ref",
            "Узлов на кратчайшую длину волны", "Относительное изменение",
        ),
        ("Полное холодное время", "Число неизвестных", "Время, с"),
        ("Пиковый прирост VRAM", "Число неизвестных", "VRAM, МиБ"),
    )
    for axis, (title, xlabel, ylabel) in zip(axes.flat, specifications):
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend()
    figure.suptitle(
        "Строгая H(div)-BDM1-самосходимость многогранников: "
        "$ka=10$, $m=1{,}3$, $R=3$"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", type=Path,
        default=ROOT / "runs" / "convergence_study_20260805",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "studies" / "bem_convergence_20260805" / "generated",
    )
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "studies" / "bem_convergence_20260805" / "study_config.json",
    )
    args = parser.parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, failures = discover(runs_root)
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    planned_cases = expand_config(config)
    current_ids = {
        (case["phase"], case["base_id"], case["repeat"])
        for case in planned_cases
    }
    for row in rows:
        row["in_current_plan"] = (
            row.get("phase"), row.get("base_id"), row.get("repeat")
        ) in current_ids
    apply_analysis_reuse(rows, config)
    add_effective_cold_resources(rows)
    add_self_convergence(rows)
    add_fmm_reference_comparisons(rows)
    add_fmm_grid_reference_comparisons(rows)
    add_quadrature_reference_comparisons(rows)
    add_farfield_grid_comparisons(rows)
    add_solver_reference_comparisons(rows)
    mesh_selection = mesh_selection_rows(rows)
    contrast_selection = contrast_selection_rows(rows)
    scaling = local_scaling_rows(rows)
    production_scaling = production_resource_scaling_rows(rows)
    production_limit = production_resource_limit(production_scaling)
    models = resource_models(rows)
    recommendations = automatic_recommendations(
        rows, mesh_selection, contrast_selection
    )
    progress = progress_rows(planned_cases, rows, failures)
    write_csv(output / "cases.csv", rows)
    write_csv(output / "mesh_selection.csv", mesh_selection)
    write_csv(output / "mesh_contrast_selection.csv", contrast_selection)
    write_csv(output / "local_scaling.csv", scaling)
    write_csv(output / "production_resource_scaling.csv", production_scaling)
    write_csv(output / "progress.csv", progress)
    write_mesh_selection_tex(output / "mesh_selection_table.tex", mesh_selection)
    write_resource_models_tex(output / "resource_models_table.tex", models)
    write_progress_tex(output / "progress_table.tex", progress)
    atomic_json(output / "cases.json", rows)
    atomic_json(output / "mesh_selection.json", mesh_selection)
    atomic_json(output / "mesh_contrast_selection.json", contrast_selection)
    atomic_json(output / "resource_models.json", models)
    atomic_json(output / "production_resource_scaling.json", production_scaling)
    atomic_json(output / "production_resource_limit.json", production_limit)
    atomic_json(output / "automatic_recommendations.json", recommendations)
    atomic_json(output / "progress.json", progress)
    atomic_json(output / "failures.json", failures)
    summary = {
        "completed_cases": len(rows),
        "completed_current_plan_cases": sum(
            bool(row.get("in_current_plan")) for row in rows
        ),
        "failed_cases": sum(
            failure.get("status", {}).get("state") == "failed"
            for failure in failures
        ),
        "running_or_incomplete_cases": sum(
            failure.get("status", {}).get("state") != "failed"
            for failure in failures
        ),
        "provenance": {
            "binary_hash_present": sum(
                bool(row.get("provenance_binary_hash_present")) for row in rows
            ),
            "legacy_without_binary_hash": sum(
                not bool(row.get("provenance_binary_hash_present")) for row in rows
            ),
        },
        "phases": {
            phase: sum(row.get("phase") == phase for row in rows)
            for phase in sorted({row.get("phase") for row in rows})
        },
        "study_storage": directory_storage(runs_root),
    }
    atomic_json(output / "summary.json", summary)
    plot_mie_convergence(rows, output / "mesh_mie_error_vs_ppw.png")
    plot_mesh_ref_ka_scaling(rows, output / "mesh_ref_ka_scaling.png")
    plot_mesh_ka_ref_heatmaps(rows, output / "mesh_ka_ref_heatmaps.png")
    plot_extended_resources(rows, output / "mesh_ref_ka_extended_resources.png")
    plot_hardware_utilization(rows, output / "mesh_ref_ka_hardware_utilization.png")
    plot_storage_resources(rows, output / "mesh_ref_ka_storage_resources.png")
    plot_resource_scaling(rows, output / "resources_vs_dofs.png")
    plot_production_resource_scaling(
        production_scaling, output / "production_resource_scaling.png"
    )
    plot_fmm_radius(rows, output / "fmm_near_radius_tradeoff.png")
    plot_fmm_parameter_tradeoffs(rows, output / "fmm_parameter_tradeoffs.png")
    plot_fmm_radius_digits_grid(rows, output / "fmm_radius_digits_grid.png")
    plot_quadrature_tradeoffs(rows, output / "quadrature_tradeoffs.png")
    plot_farfield_grid(rows, output / "farfield_grid_convergence.png")
    plot_solver_controls(rows, output / "solver_controls.png")
    plot_basis_edge_control(rows, output / "basis_edge_control.png")
    plot_contrast_convergence(rows, output / "mesh_sphere_contrast.png")
    plot_prism_convergence(rows, output / "mesh_prism_convergence.png")
    plot_strict_hdiv_polyhedra(
        rows, output / "mesh_polyhedra_hdiv_strict.png"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"generated: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
