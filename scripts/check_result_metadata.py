#!/usr/bin/env python3
"""Validate provenance metadata in BEM result JSON files."""

import argparse
import json
from pathlib import Path
import math
import re
from typing import Iterable, List, Optional, Tuple


REQUIRED_METHOD_KEYS = {
    "solver_backend",
    "solver_profile",
    "requested_system",
    "system",
    "system_canonicalized",
    "preconditioner_enabled",
    "schwarz_preconditioner",
    "preconditioner_reason",
    "farfield_mode",
}

PRECONDITIONER_ENABLED_REASONS = {
    "auto",
    "forced",
    "explicit_neural_graph_sai",
    "explicit_mass",
    "explicit_calderon_rwg",
    "explicit_local",
    "explicit_ilu0",
    "obj_strict_block_jacobi_measured",
    "obj_strict_block_jacobi",
    "obj_quality_strict_block_jacobi",
}
PRECONDITIONER_DISABLED_REASONS = {
    "dense_solver",
    "user_disabled",
    "pfft_backend",
    "small_system",
    "small_nonsphere",
    "n_form",
    "sphere_unpreconditioned_measured",
    "hex_unpreconditioned_faster",
    "hex_strict_unpreconditioned_measured",
    "obj_ka_ge_4_unpreconditioned_measured",
    "obj_strict_unpreconditioned_measured",
    "obj_quality_remesh_unpreconditioned",
    "obj_quality_loose_unpreconditioned_measured",
    "obj_quality_strict_unpreconditioned_measured",
}

COMPLEX_OPERATOR_KEYS = {
    "row_h_scale",
    "row_h_scale_imag",
    "row_h_scale_complex",
}

REQUIRED_MESH_KEYS = {
    "vertices",
    "triangles",
    "skinny_triangles",
    "min_angle_deg",
    "max_aspect_ratio",
    "feature_edges_30deg",
    "max_dihedral_deg",
    "mean_feature_dihedral_deg",
    "max_adjacent_area_ratio",
    "near_touch_checked",
    "near_touch_ratio",
    "near_touch_pairs",
    "self_panel_count",
    "edge_adjacent_pair_count",
    "vertex_adjacent_pair_count",
    "near_disjoint_pair_count",
    "taylor_duffy_candidate_count",
    "recommended_min_quad_order",
    "recommended_mesh_strategy",
    "recommended_mesh_action",
    "requires_remesh",
    "edge_refine_requested",
    "edge_refine_applied",
    "edge_refine_uniform_fallback",
    "quality_gate_pass",
}

REQUIRED_TOP_LEVEL_KEYS = {
    "shape",
    "obj_file",
    "prism_aspect",
    "edge_refine",
}


def _missing(container: dict, keys: Iterable[str], prefix: str) -> List[str]:
    return [f"{prefix}.{key}" for key in sorted(set(keys) - set(container))]


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mueller_rows(mueller, theta_len: Optional[int]) -> Tuple[Optional[List[List[float]]], Optional[str]]:
    if not isinstance(mueller, list) or not mueller:
        return None, "mueller must be a non-empty list"

    def numeric_row(flat, idx: int) -> Tuple[Optional[List[float]], Optional[str]]:
        numeric = []
        for jdx, value in enumerate(flat):
            if not _finite_number(value):
                return None, f"mueller[{idx}][{jdx}] must be finite"
            numeric.append(float(value))
        return numeric, None

    if theta_len is not None and len(mueller) == 4:
        is_4x4xn = True
        for i in range(4):
            if not isinstance(mueller[i], list) or len(mueller[i]) != 4:
                is_4x4xn = False
                break
            for j in range(4):
                if not isinstance(mueller[i][j], list) or len(mueller[i][j]) != theta_len:
                    is_4x4xn = False
                    break
            if not is_4x4xn:
                break
        if is_4x4xn:
            rows: List[List[float]] = []
            for idx in range(theta_len):
                flat = [mueller[i][j][idx] for i in range(4) for j in range(4)]
                numeric, error = numeric_row(flat, idx)
                if error:
                    return None, error
                rows.append(numeric or [])
            return rows, None

    if theta_len is not None and len(mueller) == 16:
        is_16xn = all(isinstance(row, list) and len(row) == theta_len for row in mueller)
        if is_16xn:
            rows = []
            for idx in range(theta_len):
                flat = [mueller[k][idx] for k in range(16)]
                numeric, error = numeric_row(flat, idx)
                if error:
                    return None, error
                rows.append(numeric or [])
            return rows, None

    rows: List[List[float]] = []
    for idx, row in enumerate(mueller):
        if not isinstance(row, list):
            return None, f"mueller[{idx}] must be a list"
        if len(row) == 16:
            flat = row
        elif len(row) == 4 and all(isinstance(sub, list) and len(sub) == 4 for sub in row):
            flat = [value for sub in row for value in sub]
        else:
            return None, f"mueller[{idx}] must have 16 values or 4x4 values"
        numeric, error = numeric_row(flat, idx)
        if error:
            return None, error
        rows.append(numeric)
    return rows, None


def _cloude_min_eigenvalue(row: List[float], scale_floor: float) -> float:
    """Return normalized min eigenvalue of the Cloude coherency matrix."""
    try:
        import numpy as np  # Lazy import keeps metadata checks lightweight unless requested.
    except Exception as exc:  # pragma: no cover - depends on host package set
        raise RuntimeError(f"numpy is required for Cloude Mueller check: {exc}") from exc

    pauli = (
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    )
    scale = max(abs(row[0]), scale_floor)
    h = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            h += row[i * 4 + j] * np.kron(pauli[i], pauli[j].conjugate())
    h *= 0.25 / scale
    h = 0.5 * (h + h.conjugate().T)
    return float(np.linalg.eigvalsh(h)[0])


def validate_numeric_payload(data: dict, *, negative_m11_tol: float,
                             max_abs_over_m11: float,
                             physical_tol: float,
                             require_cloude_physical: bool) -> List[str]:
    errors: List[str] = []
    theta = data.get("theta")
    if not isinstance(theta, list) or not theta:
        errors.append("theta must be a non-empty list")
    elif not all(_finite_number(value) for value in theta):
        errors.append("theta values must be finite")

    theta_len = len(theta) if isinstance(theta, list) and theta else None
    rows, mueller_error = _mueller_rows(data.get("mueller"), theta_len)
    if mueller_error:
        errors.append(mueller_error)
    elif isinstance(theta, list) and len(rows) != len(theta):
        errors.append(f"mueller length must match theta length, got {len(rows)} and {len(theta)}")

    if rows:
        m11 = [row[0] for row in rows]
        scale = max(max(abs(value) for value in m11), 1e-300)
        min_allowed = -abs(float(negative_m11_tol)) * scale
        bad = [value for value in m11 if value < min_allowed]
        if bad:
            errors.append(
                f"mueller M11 must be non-negative within relative tolerance {negative_m11_tol:g}; "
                f"min={min(m11):.6g}"
            )
        floor = 1e-12 * scale
        max_ratio = float(max_abs_over_m11)
        worst_ratio = 0.0
        worst_label = ""
        for idx, row in enumerate(rows):
            denom = max(abs(row[0]), floor)
            for comp, value in enumerate(row[1:], start=1):
                ratio = abs(value) / denom
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    worst_label = f"theta_index={idx},component={comp}"
        if worst_ratio > max_ratio:
            errors.append(
                f"mueller |Mij|/M11 must be <= {max_ratio:.6g}; "
                f"worst={worst_ratio:.6g} at {worst_label}"
            )
        max_physical = 1.0 + abs(float(physical_tol))
        worst_polarizance = 0.0
        worst_polarizance_idx = 0
        worst_diattenuation = 0.0
        worst_diattenuation_idx = 0
        for idx, row in enumerate(rows):
            denom = max(abs(row[0]), floor)
            polarizance = math.sqrt(row[4] * row[4] + row[8] * row[8] + row[12] * row[12]) / denom
            diattenuation = math.sqrt(row[1] * row[1] + row[2] * row[2] + row[3] * row[3]) / denom
            if polarizance > worst_polarizance:
                worst_polarizance = polarizance
                worst_polarizance_idx = idx
            if diattenuation > worst_diattenuation:
                worst_diattenuation = diattenuation
                worst_diattenuation_idx = idx
        if worst_polarizance > max_physical:
            errors.append(
                f"mueller polarizance sqrt(M21^2+M31^2+M41^2)/M11 must be <= {max_physical:.6g}; "
                f"worst={worst_polarizance:.6g} at theta_index={worst_polarizance_idx}"
            )
        if worst_diattenuation > max_physical:
            errors.append(
                f"mueller diattenuation sqrt(M12^2+M13^2+M14^2)/M11 must be <= {max_physical:.6g}; "
                f"worst={worst_diattenuation:.6g} at theta_index={worst_diattenuation_idx}"
            )
        if require_cloude_physical:
            try:
                cloude_mins = [_cloude_min_eigenvalue(row, floor) for row in rows]
            except RuntimeError as exc:
                errors.append(str(exc))
            else:
                worst_cloude = min(cloude_mins)
                if worst_cloude < -abs(float(physical_tol)):
                    worst_idx = cloude_mins.index(worst_cloude)
                    errors.append(
                        "mueller Cloude coherency matrix must be positive semidefinite; "
                        f"min_eigenvalue={worst_cloude:.6g} at theta_index={worst_idx}"
                    )

    return errors


def validate_run_parameters(data: dict) -> List[str]:
    """Validate scalar run parameters that make a result admissible for audits."""
    errors: List[str] = []

    positive_finite = (
        ("ka", "ka must be finite and positive"),
        ("gmres_tol", "gmres_tol must be finite and positive"),
        ("orientation_weight_sum", "orientation_weight_sum must be finite and positive"),
    )
    for key, message in positive_finite:
        if key in data and (not _finite_number(data[key]) or float(data[key]) <= 0.0):
            errors.append(message)

    nonnegative_finite = (
        ("n_im", "n_im must be finite and non-negative"),
        ("time_total", "time_total must be finite and non-negative"),
        ("time_solve", "time_solve must be finite and non-negative"),
        ("gmres_max_final_relres", "gmres_max_final_relres must be finite and non-negative"),
    )
    for key, message in nonnegative_finite:
        if key in data and (not _finite_number(data[key]) or float(data[key]) < 0.0):
            errors.append(message)

    finite_only = (
        ("n_re", "n_re must be finite"),
        ("prism_aspect", "prism_aspect must be finite"),
    )
    for key, message in finite_only:
        if key in data and not _finite_number(data[key]):
            errors.append(message)

    positive_int = (
        ("fmm_digits", "fmm_digits must be integer >= 1"),
        ("gmres_restart", "gmres_restart must be integer >= 1"),
        ("gmres_max_cycles", "gmres_max_cycles must be integer >= 1"),
        ("ntheta", "ntheta must be integer >= 1"),
        ("quad_order", "quad_order must be integer >= 1"),
        ("max_leaf", "max_leaf must be integer >= 1"),
        ("gmres_converged_systems", "gmres_converged_systems must be integer >= 0"),
        ("gmres_nonconverged_systems", "gmres_nonconverged_systems must be integer >= 0"),
        ("gmres_stagnation_stops", "gmres_stagnation_stops must be integer >= 0"),
        ("gmres_numerical_breakdowns", "gmres_numerical_breakdowns must be integer >= 0"),
        ("gmres_restored_best_iterates", "gmres_restored_best_iterates must be integer >= 0"),
        ("gmres_max_cycle_exhaustions", "gmres_max_cycle_exhaustions must be integer >= 0"),
    )
    for key, message in positive_int:
        if key not in data:
            continue
        value = data[key]
        lower = 0 if key.startswith("gmres_") and key not in {"gmres_restart", "gmres_max_cycles"} else 1
        if not isinstance(value, int) or value < lower:
            errors.append(message)

    nonnegative_int = (
        ("refinements", "refinements must be integer >= 0"),
        ("edge_refine", "edge_refine must be integer >= 0"),
        ("orient_start", "orient_start must be integer >= 0"),
        ("orient_count", "orient_count must be integer >= 0"),
        ("orient_total", "orient_total must be integer >= 0"),
        ("gmres_matvecs", "gmres_matvecs must be integer >= 0"),
    )
    for key, message in nonnegative_int:
        if key in data and (not isinstance(data[key], int) or data[key] < 0):
            errors.append(message)

    orient_start = data.get("orient_start")
    orient_count = data.get("orient_count")
    orient_total = data.get("orient_total")
    if all(isinstance(value, int) for value in (orient_start, orient_count, orient_total)):
        if orient_start > orient_total:
            errors.append("orient_start must be <= orient_total")
        if orient_count > orient_total - orient_start:
            errors.append("orient_count must fit within orient_total from orient_start")

    return errors


def case_contract_errors(data: dict, result_path: Optional[Path] = None) -> List[str]:
    """Return errors when encoded case-name parameters contradict JSON metadata."""
    if result_path is None:
        return []
    stem = Path(result_path).stem
    errors: List[str] = []
    shape = data.get("shape")
    obj_file = data.get("obj_file")
    if stem.startswith("sphere_") and shape != "sphere":
        errors.append("sphere case name requires shape=sphere")
    if stem.startswith("hex_") and shape not in {"hex_prism", "prism6"}:
        errors.append("hex case name requires shape=hex_prism or prism6")
    if stem.startswith("dust_"):
        if shape != "obj":
            errors.append("dust case name requires shape=obj")
        if not isinstance(obj_file, str) or not obj_file:
            errors.append("dust case name requires non-empty obj_file")
    ka_match = re.search(r"(?:^|_)ka([0-9]+(?:p[0-9]+)?)", stem)
    if ka_match:
        expected_ka = float(ka_match.group(1).replace("p", "."))
        actual_ka = data.get("ka")
        if not _finite_number(actual_ka) or abs(float(actual_ka) - expected_ka) > 1e-9 * max(1.0, abs(expected_ka)):
            errors.append(f"case name requires ka={expected_ka:g}")
    ref_match = re.search(r"_ref([0-9]+)(?:_|$)", stem)
    if ref_match:
        expected_ref = int(ref_match.group(1))
        actual_ref = data.get("refinements")
        if not isinstance(actual_ref, int) or actual_ref != expected_ref:
            errors.append(f"case name requires refinements={expected_ref}")
    match = re.search(r"_q([0-9]+)_d([0-9]+)_tol([0-9]+)e([0-9]+)$", stem)
    if match:
        digits_required = int(match.group(2))
        tol_required = float(match.group(3)) * (10.0 ** -int(match.group(4)))
        digits = data.get("fmm_digits")
        tol = data.get("gmres_tol")
        if not isinstance(digits, int) or digits < digits_required:
            errors.append(f"case name requires fmm_digits >= {digits_required}")
        if not _finite_number(tol) or float(tol) > tol_required:
            errors.append(f"case name requires gmres_tol <= {tol_required:g}")

    if stem.startswith("dust_") and "_d6_" in stem and "_tol5e4" in stem:
        method = data.get("method", {}) if isinstance(data, dict) else {}
        if not isinstance(method, dict):
            method = {}
        if method.get("solver_profile") not in {"obj_accurate", "obj_strict", "obj_mesh_guard"}:
            errors.append("dust d6/tol5e4 results require solver_profile=obj_accurate/obj_strict/obj_mesh_guard")
        restart = data.get("gmres_restart")
        if not isinstance(restart, int) or restart < 500:
            errors.append("dust d6/tol5e4 results require gmres_restart >= 500")
    return errors


def validate_result(data: dict, *, require_complex_operator: bool = False,
                    require_converged: bool = False,
                    max_final_relres: Optional[float] = None,
                    validate_numeric: bool = False,
                    negative_m11_tol: float = 1e-10,
                    max_abs_over_m11: float = 1.000001,
                    physical_tol: float = 1e-8,
                    require_cloude_physical: bool = False,
                    result_path: Optional[Path] = None) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings) for one result JSON."""
    errors: List[str] = []
    warnings: List[str] = []

    method = data.get("method")
    mesh = data.get("mesh")
    if method is None:
        method = {}
        warnings.append("method")
    elif not isinstance(method, dict):
        errors.append("method must be an object")
        method = {}
    if mesh is None:
        mesh = {}
        warnings.append("mesh")
    elif not isinstance(mesh, dict):
        errors.append("mesh must be an object")
        mesh = {}

    missing_method = _missing(method, REQUIRED_METHOD_KEYS, "method")
    missing_mesh = _missing(mesh, REQUIRED_MESH_KEYS, "mesh")
    missing_top_level = _missing(data, REQUIRED_TOP_LEVEL_KEYS, "")
    warnings.extend(missing_method)
    warnings.extend(missing_mesh)
    warnings.extend(key.lstrip(".") for key in missing_top_level)
    if require_complex_operator:
        warnings.extend(_missing(method, COMPLEX_OPERATOR_KEYS, "method"))

    if validate_numeric:
        errors.extend(validate_numeric_payload(
            data,
            negative_m11_tol=negative_m11_tol,
            max_abs_over_m11=max_abs_over_m11,
            physical_tol=physical_tol,
            require_cloude_physical=require_cloude_physical,
        ))
    errors.extend(validate_run_parameters(data))
    errors.extend(case_contract_errors(data, result_path))

    if "system_canonicalized" in method and not isinstance(method["system_canonicalized"], bool):
        errors.append("method.system_canonicalized must be boolean")
    if "preconditioner_enabled" in method and not isinstance(method["preconditioner_enabled"], bool):
        errors.append("method.preconditioner_enabled must be boolean")
    if "schwarz_preconditioner" in method and not isinstance(method["schwarz_preconditioner"], bool):
        errors.append("method.schwarz_preconditioner must be boolean")
    prec_enabled = method.get("preconditioner_enabled")
    schwarz_preconditioner = method.get("schwarz_preconditioner")
    prec_reason = method.get("preconditioner_reason")
    if "preconditioner_reason" in method and not isinstance(prec_reason, str):
        errors.append("method.preconditioner_reason must be string")
    elif isinstance(prec_enabled, bool) and isinstance(prec_reason, str):
        if prec_enabled and prec_reason not in PRECONDITIONER_ENABLED_REASONS:
            errors.append(
                "method.preconditioner_reason is not a known enabled-preconditioner reason"
            )
        if not prec_enabled and prec_reason not in PRECONDITIONER_DISABLED_REASONS:
            errors.append(
                "method.preconditioner_reason is not a known disabled-preconditioner reason"
            )
    if isinstance(prec_enabled, bool) and isinstance(schwarz_preconditioner, bool):
        if not prec_enabled and schwarz_preconditioner:
            errors.append(
                "method.schwarz_preconditioner must be false when preconditioner_enabled is false"
            )
    if "row_h_scale_complex" in method:
        row_h_scale_complex = method["row_h_scale_complex"]
        if (not isinstance(row_h_scale_complex, list) or
                len(row_h_scale_complex) != 2 or
                not all(isinstance(x, (int, float)) for x in row_h_scale_complex)):
            errors.append("method.row_h_scale_complex must be [real, imag]")

    solver_profile = method.get("solver_profile")
    if solver_profile in {"obj_accurate", "obj_strict", "obj_mesh_guard"}:
        digits = data.get("fmm_digits")
        tol = data.get("gmres_tol")
        restart = data.get("gmres_restart")
        max_cycles = data.get("gmres_max_cycles")
        true_residual_checked = method.get("gmres_true_residual_checked")
        min_digits = 8 if solver_profile == "obj_mesh_guard" else 7
        max_tol = 1e-5
        min_restart = 1400 if solver_profile == "obj_mesh_guard" else 1000
        min_max_cycles = 80
        if not isinstance(digits, int) or digits < min_digits:
            errors.append(f"{solver_profile} results require fmm_digits >= {min_digits}")
        if not _finite_number(tol) or float(tol) > max_tol:
            errors.append(f"{solver_profile} results require gmres_tol <= {max_tol:g}")
        if not isinstance(restart, int) or restart < min_restart:
            errors.append(f"{solver_profile} results require gmres_restart >= {min_restart}")
        if not isinstance(max_cycles, int) or max_cycles < min_max_cycles:
            errors.append(f"{solver_profile} results require gmres_max_cycles >= {min_max_cycles}")
        if true_residual_checked is not True:
            errors.append(f"{solver_profile} results require method.gmres_true_residual_checked=true")

    if require_converged:
        nonconv = data.get("gmres_nonconverged_systems")
        if not isinstance(nonconv, int):
            errors.append("gmres_nonconverged_systems must be integer")
        elif nonconv != 0:
            errors.append(f"gmres_nonconverged_systems must be 0, got {nonconv}")

        stagnation = data.get("gmres_stagnation_stops")
        if not isinstance(stagnation, int):
            errors.append("gmres_stagnation_stops must be integer")
        elif stagnation != 0:
            errors.append(f"gmres_stagnation_stops must be 0, got {stagnation}")

        breakdowns = data.get("gmres_numerical_breakdowns")
        if breakdowns is not None:
            if not isinstance(breakdowns, int):
                errors.append("gmres_numerical_breakdowns must be integer")
            elif breakdowns != 0:
                errors.append(f"gmres_numerical_breakdowns must be 0, got {breakdowns}")

        exhausted = data.get("gmres_max_cycle_exhaustions")
        if exhausted is not None:
            if not isinstance(exhausted, int):
                errors.append("gmres_max_cycle_exhaustions must be integer")
            elif exhausted != 0:
                errors.append(f"gmres_max_cycle_exhaustions must be 0, got {exhausted}")

        relres = data.get("gmres_max_final_relres")
        if not isinstance(relres, (int, float)) or not math.isfinite(float(relres)):
            errors.append("gmres_max_final_relres must be finite")
        else:
            limit = max_final_relres
            if limit is None:
                tol = data.get("gmres_tol")
                if isinstance(tol, (int, float)) and math.isfinite(float(tol)) and float(tol) > 0.0:
                    limit = 10.0 * float(tol)
            if limit is not None and float(relres) > float(limit):
                errors.append(
                    f"gmres_max_final_relres must be <= {float(limit):.6g}, got {float(relres):.6g}"
                )

    requested = method.get("requested_system")
    actual = method.get("system")
    canonicalized = method.get("system_canonicalized")
    if isinstance(canonicalized, bool) and requested is not None and actual is not None:
        if canonicalized and requested == actual:
            errors.append("method.system_canonicalized is true but requested_system equals system")
        if not canonicalized and requested != actual:
            errors.append("method.system_canonicalized is false but requested_system differs from system")

    for key in ("vertices", "triangles", "skinny_triangles", "feature_edges_30deg", "near_touch_pairs",
                "self_panel_count", "edge_adjacent_pair_count", "vertex_adjacent_pair_count",
                "near_disjoint_pair_count", "taylor_duffy_candidate_count",
                "recommended_min_quad_order",
                "edge_refine_requested", "edge_refine_applied"):
        if key in mesh and not isinstance(mesh[key], int):
            errors.append(f"mesh.{key} must be integer")
    for key in ("min_angle_deg", "max_aspect_ratio", "max_dihedral_deg",
                "mean_feature_dihedral_deg", "max_adjacent_area_ratio"):
        if key in mesh and not _finite_number(mesh[key]):
            errors.append(f"mesh.{key} must be finite")
        elif key in mesh and float(mesh[key]) < 0.0:
            errors.append(f"mesh.{key} must be non-negative")
    if "edge_refine_uniform_fallback" in mesh and not isinstance(mesh["edge_refine_uniform_fallback"], bool):
        errors.append("mesh.edge_refine_uniform_fallback must be boolean")
    if "requires_remesh" in mesh and not isinstance(mesh["requires_remesh"], bool):
        errors.append("mesh.requires_remesh must be boolean")
    elif mesh.get("requires_remesh") is True:
        errors.append("mesh.requires_remesh must be false for accepted results")
    for key in ("recommended_mesh_strategy", "recommended_mesh_action"):
        if key in mesh and not isinstance(mesh[key], str):
            errors.append(f"mesh.{key} must be string")
    near_touch_skip_allowed = (
        mesh.get("quality_gate_pass") is True
        and mesh.get("requires_remesh") is False
        and isinstance(mesh.get("near_touch_pairs"), int)
        and mesh.get("near_touch_pairs") == 0
        and isinstance(mesh.get("near_disjoint_pair_count"), int)
        and mesh.get("near_disjoint_pair_count") == 0
    )
    if "near_touch_checked" in mesh and not isinstance(mesh["near_touch_checked"], bool):
        errors.append("mesh.near_touch_checked must be boolean")
    elif mesh.get("near_touch_checked") is False and not near_touch_skip_allowed:
        errors.append("mesh.near_touch_checked must be true for accepted results")
    if "near_touch_ratio" in mesh and not _finite_number(mesh["near_touch_ratio"]):
        errors.append("mesh.near_touch_ratio must be finite")
    if isinstance(mesh.get("near_touch_pairs"), int) and mesh.get("near_touch_pairs", 0) > 0:
        errors.append(f"mesh.near_touch_pairs must be 0, got {mesh['near_touch_pairs']}")
    if isinstance(mesh.get("near_disjoint_pair_count"), int) and mesh.get("near_disjoint_pair_count", 0) > 0:
        errors.append(f"mesh.near_disjoint_pair_count must be 0, got {mesh['near_disjoint_pair_count']}")
    if (
        isinstance(mesh.get("self_panel_count"), int)
        and isinstance(mesh.get("triangles"), int)
        and mesh["self_panel_count"] != mesh["triangles"]
    ):
        errors.append(
            f"mesh.self_panel_count must equal mesh.triangles, got {mesh['self_panel_count']} and {mesh['triangles']}"
        )
    if (
        isinstance(mesh.get("taylor_duffy_candidate_count"), int)
        and all(isinstance(mesh.get(k), int) for k in (
            "self_panel_count", "edge_adjacent_pair_count",
            "vertex_adjacent_pair_count", "near_disjoint_pair_count",
        ))
    ):
        expected = (
            mesh["self_panel_count"]
            + mesh["edge_adjacent_pair_count"]
            + mesh["vertex_adjacent_pair_count"]
            + mesh["near_disjoint_pair_count"]
        )
        if mesh["taylor_duffy_candidate_count"] != expected:
            errors.append(
                "mesh.taylor_duffy_candidate_count must equal self+edge+vertex+near-disjoint "
                f"({expected}), got {mesh['taylor_duffy_candidate_count']}"
            )
    if isinstance(mesh.get("recommended_min_quad_order"), int):
        if mesh["recommended_min_quad_order"] not in (4, 7, 13):
            errors.append(
                "mesh.recommended_min_quad_order must be one of 4, 7, 13, "
                f"got {mesh['recommended_min_quad_order']}"
            )
        quad_order = method.get("quad_order")
        if isinstance(quad_order, int) and quad_order < mesh["recommended_min_quad_order"]:
            errors.append(
                f"method.quad_order must be >= mesh.recommended_min_quad_order "
                f"({mesh['recommended_min_quad_order']}), got {quad_order}"
            )
    if "quality_gate_pass" in mesh and not isinstance(mesh["quality_gate_pass"], bool):
        errors.append("mesh.quality_gate_pass must be boolean")
    elif mesh.get("quality_gate_pass") is False:
        errors.append("mesh.quality_gate_pass must be true for accepted results")
    if "shape" in data and not isinstance(data["shape"], str):
        errors.append("shape must be string")
    if "obj_file" in data and data["obj_file"] is not None and not isinstance(data["obj_file"], str):
        errors.append("obj_file must be string or null")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat missing provenance metadata as an error",
    )
    parser.add_argument(
        "--require-complex-operator",
        action="store_true",
        help="require complex operator provenance such as row_h_scale_complex",
    )
    parser.add_argument(
        "--require-converged",
        action="store_true",
        help="require GMRES convergence fields to report no nonconverged systems",
    )
    parser.add_argument(
        "--max-final-relres",
        type=float,
        default=None,
        help="optional absolute upper bound for gmres_max_final_relres",
    )
    parser.add_argument(
        "--validate-numeric",
        action="store_true",
        help="validate theta/mueller shape, finite values and nonnegative M11",
    )
    parser.add_argument(
        "--negative-m11-tol",
        type=float,
        default=1e-10,
        help="relative tolerance for small negative M11 during --validate-numeric",
    )
    parser.add_argument(
        "--max-abs-over-m11",
        type=float,
        default=1.000001,
        help="maximum allowed |Mij|/M11 ratio during --validate-numeric",
    )
    parser.add_argument(
        "--physical-tol",
        type=float,
        default=1e-8,
        help="tolerance for Mueller polarizance/diattenuation bounds during --validate-numeric",
    )
    parser.add_argument(
        "--require-cloude-physical",
        action="store_true",
        help="require positive semidefinite Cloude coherency matrices during --validate-numeric",
    )
    args = parser.parse_args()

    failed = False
    for path in args.json_files:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # pragma: no cover - exercised through CLI use
            print(f"{path}: invalid json: {exc}")
            failed = True
            continue

        errors, warnings = validate_result(
            data,
            require_complex_operator=args.require_complex_operator,
            require_converged=args.require_converged,
            max_final_relres=args.max_final_relres,
            validate_numeric=args.validate_numeric,
            negative_m11_tol=args.negative_m11_tol,
            max_abs_over_m11=args.max_abs_over_m11,
            physical_tol=args.physical_tol,
            require_cloude_physical=args.require_cloude_physical,
            result_path=path,
        )
        if warnings and args.strict:
            errors.extend(warnings)
            warnings = []

        if errors:
            print(f"{path}: metadata fail")
            for error in errors:
                print(f"  error: {error}")
            failed = True
        elif warnings:
            print(f"{path}: metadata legacy")
            for warning in warnings:
                print(f"  warning: missing {warning}")
        else:
            print(f"{path}: metadata ok")

    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
