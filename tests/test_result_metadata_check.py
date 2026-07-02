#!/usr/bin/env python3
"""Unit tests for BEM result JSON provenance validation."""

from pathlib import Path
import sys
import json
import tempfile
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_result_metadata import case_contract_errors, validate_result  # noqa: E402


def valid_result() -> dict:
    return {
        "gmres_nonconverged_systems": 0,
        "gmres_stagnation_stops": 0,
        "gmres_restored_best_iterates": 0,
        "gmres_max_cycle_exhaustions": 0,
        "gmres_max_final_relres": 9e-4,
        "gmres_tol": 1e-3,
        "gmres_max_cycles": 80,
        "ka": 10.0,
        "refinements": 3,
        "shape": "hex_prism",
        "obj_file": None,
        "prism_aspect": 1.5,
        "edge_refine": 1,
        "method": {
            "solver_backend": "FMM",
            "solver_profile": "hex_guarded",
            "requested_system": "muller2-balanced",
            "system": "balanced",
            "system_canonicalized": True,
            "quad_order": 7,
            "row_h_scale": 0.7624,
            "row_h_scale_imag": -0.001,
            "row_h_scale_complex": [0.7624, -0.001],
            "preconditioner_enabled": False,
            "schwarz_preconditioner": False,
            "preconditioner_reason": "small_nonsphere",
            "farfield_mode": "gpu_geometry_direct",
        },
        "mesh": {
            "vertices": 1058,
            "triangles": 2112,
            "skinny_triangles": 0,
            "min_angle_deg": 44.7,
            "max_aspect_ratio": 1.39,
            "feature_edges_30deg": 72,
            "max_dihedral_deg": 90.0,
            "mean_feature_dihedral_deg": 90.0,
            "max_adjacent_area_ratio": 1.25,
            "near_touch_checked": True,
            "near_touch_ratio": 1.1,
            "near_touch_pairs": 0,
            "self_panel_count": 2112,
            "edge_adjacent_pair_count": 3168,
            "vertex_adjacent_pair_count": 0,
            "near_disjoint_pair_count": 0,
            "taylor_duffy_candidate_count": 5280,
            "recommended_min_quad_order": 7,
            "recommended_mesh_strategy": "edge_aware_refinement",
            "recommended_mesh_action": "keep conforming edge-aware refinement near sharp dihedral edges",
            "requires_remesh": False,
            "edge_refine_requested": 1,
            "edge_refine_applied": 0,
            "edge_refine_uniform_fallback": True,
            "quality_gate_pass": True,
        },
    }


def main() -> int:
    errors, warnings = validate_result(valid_result())
    assert errors == [], errors
    assert warnings == [], warnings

    numeric = valid_result()
    numeric["theta"] = [0.0, 1.0]
    numeric["mueller"] = [
        [1.0] + [0.0] * 15,
        [0.9] + [0.0] * 15,
    ]
    errors, warnings = validate_result(numeric, validate_numeric=True)
    assert errors == [], errors
    assert warnings == [], warnings

    numeric_4x4 = valid_result()
    numeric_4x4["theta"] = [0.0]
    numeric_4x4["mueller"] = [[[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]]]
    errors, warnings = validate_result(numeric_4x4, validate_numeric=True)
    assert errors == [], errors

    numeric_4x4xn = valid_result()
    numeric_4x4xn["theta"] = [0.0, 1.0]
    numeric_4x4xn["mueller"] = [
        [[1.0, 0.9], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.9], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.9], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.9]],
    ]
    errors, warnings = validate_result(numeric_4x4xn, validate_numeric=True)
    assert errors == [], errors

    numeric_16xn = valid_result()
    numeric_16xn["theta"] = [0.0, 1.0]
    numeric_16xn["mueller"] = [[1.0, 0.9]] + [[0.0, 0.0] for _ in range(15)]
    errors, warnings = validate_result(numeric_16xn, validate_numeric=True)
    assert errors == [], errors

    bad_numeric = valid_result()
    bad_numeric["theta"] = [0.0, 1.0]
    bad_numeric["mueller"] = [[1.0] + [0.0] * 15]
    errors, warnings = validate_result(bad_numeric, validate_numeric=True)
    assert "mueller length must match theta length, got 1 and 2" in errors, errors

    bad_nan = valid_result()
    bad_nan["theta"] = [0.0]
    bad_nan["mueller"] = [[float("nan")] + [0.0] * 15]
    errors, warnings = validate_result(bad_nan, validate_numeric=True)
    assert "mueller[0][0] must be finite" in errors, errors

    negative_m11 = valid_result()
    negative_m11["theta"] = [0.0, 1.0]
    negative_m11["mueller"] = [
        [1.0] + [0.0] * 15,
        [-0.1] + [0.0] * 15,
    ]
    errors, warnings = validate_result(negative_m11, validate_numeric=True)
    assert any("mueller M11 must be non-negative" in error for error in errors), errors

    impossible_polarization = valid_result()
    impossible_polarization["theta"] = [0.0]
    impossible_polarization["mueller"] = [[1.0, 2.0] + [0.0] * 14]
    errors, warnings = validate_result(impossible_polarization, validate_numeric=True)
    assert any("mueller |Mij|/M11 must be <=" in error for error in errors), errors

    impossible_polarizance_vector = valid_result()
    impossible_polarizance_vector["theta"] = [0.0]
    impossible_polarizance_vector["mueller"] = [[1.0] + [0.0] * 15]
    impossible_polarizance_vector["mueller"][0][4] = 0.8
    impossible_polarizance_vector["mueller"][0][8] = 0.8
    errors, warnings = validate_result(impossible_polarizance_vector, validate_numeric=True)
    assert not any("mueller |Mij|/M11 must be <=" in error for error in errors), errors
    assert any("mueller polarizance" in error for error in errors), errors

    impossible_diattenuation_vector = valid_result()
    impossible_diattenuation_vector["theta"] = [0.0]
    impossible_diattenuation_vector["mueller"] = [[1.0] + [0.0] * 15]
    impossible_diattenuation_vector["mueller"][0][1] = 0.8
    impossible_diattenuation_vector["mueller"][0][2] = 0.8
    errors, warnings = validate_result(impossible_diattenuation_vector, validate_numeric=True)
    assert not any("mueller |Mij|/M11 must be <=" in error for error in errors), errors
    assert any("mueller diattenuation" in error for error in errors), errors

    impossible_cloude = valid_result()
    impossible_cloude["theta"] = [0.0]
    impossible_cloude["mueller"] = [[1.0] + [0.0] * 15]
    impossible_cloude["mueller"][0][5] = 0.8
    impossible_cloude["mueller"][0][10] = 0.8
    impossible_cloude["mueller"][0][15] = -0.8
    errors, warnings = validate_result(impossible_cloude, validate_numeric=True)
    assert errors == [], errors
    errors, warnings = validate_result(
        impossible_cloude,
        validate_numeric=True,
        require_cloude_physical=True,
    )
    assert any("mueller Cloude coherency matrix" in error for error in errors), errors

    no_complex_operator = valid_result()
    del no_complex_operator["method"]["row_h_scale"]
    del no_complex_operator["method"]["row_h_scale_imag"]
    del no_complex_operator["method"]["row_h_scale_complex"]
    errors, warnings = validate_result(no_complex_operator)
    assert errors == [], errors
    assert warnings == [], warnings
    errors, warnings = validate_result(no_complex_operator, require_complex_operator=True)
    assert errors == [], errors
    assert "method.row_h_scale_complex" in warnings, warnings

    legacy = {"method": {"solver_backend": "FMM"}, "mesh": {"triangles": 1}}
    errors, warnings = validate_result(legacy)
    assert errors == [], errors
    assert "method.requested_system" in warnings, warnings
    assert "method.farfield_mode" in warnings, warnings
    assert "mesh.edge_refine_applied" in warnings, warnings

    old_result = {"theta": [0.0], "mueller": [[1.0]]}
    errors, warnings = validate_result(old_result)
    assert errors == [], errors
    assert "method" in warnings, warnings
    assert "mesh" in warnings, warnings

    inconsistent = valid_result()
    inconsistent["method"]["system_canonicalized"] = False
    errors, warnings = validate_result(inconsistent)
    assert warnings == [], warnings
    assert "method.system_canonicalized is false but requested_system differs from system" in errors, errors

    bad_type = valid_result()
    bad_type["mesh"]["triangles"] = "2112"
    errors, warnings = validate_result(bad_type)
    assert "mesh.triangles must be integer" in errors, errors
    bad_prec = valid_result()
    bad_prec["method"]["schwarz_preconditioner"] = "no"
    errors, warnings = validate_result(bad_prec)
    assert "method.schwarz_preconditioner must be boolean" in errors, errors
    bad_prec = valid_result()
    bad_prec["method"]["schwarz_preconditioner"] = True
    errors, warnings = validate_result(bad_prec)
    assert "method.schwarz_preconditioner must be false when preconditioner_enabled is false" in errors, errors
    bad_prec = valid_result()
    bad_prec["method"]["preconditioner_enabled"] = True
    bad_prec["method"]["preconditioner_reason"] = "small_nonsphere"
    errors, warnings = validate_result(bad_prec)
    assert "method.preconditioner_reason is not a known enabled-preconditioner reason" in errors, errors
    bad_prec = valid_result()
    bad_prec["method"]["preconditioner_reason"] = "unknown_experiment"
    errors, warnings = validate_result(bad_prec)
    assert "method.preconditioner_reason is not a known disabled-preconditioner reason" in errors, errors
    hex_strict = valid_result()
    hex_strict["method"]["preconditioner_reason"] = "hex_strict_unpreconditioned_measured"
    errors, warnings = validate_result(hex_strict)
    assert errors == [], errors
    obj_strict_skip = valid_result()
    obj_strict_skip["method"]["preconditioner_reason"] = "obj_strict_unpreconditioned_measured"
    errors, warnings = validate_result(obj_strict_skip)
    assert errors == [], errors
    obj_quality_strict_skip = valid_result()
    obj_quality_strict_skip["method"]["preconditioner_reason"] = "obj_quality_strict_unpreconditioned_measured"
    errors, warnings = validate_result(obj_quality_strict_skip)
    assert errors == [], errors
    obj_quality_prec = valid_result()
    obj_quality_prec["method"]["preconditioner_enabled"] = True
    obj_quality_prec["method"]["preconditioner_reason"] = "obj_quality_strict_block_jacobi"
    errors, warnings = validate_result(obj_quality_prec)
    assert errors == [], errors
    obj_quality_skip_near = valid_result()
    obj_quality_skip_near["shape"] = "obj"
    obj_quality_skip_near["mesh"]["near_touch_checked"] = False
    obj_quality_skip_near["mesh"]["near_touch_ratio"] = 1e300
    obj_quality_skip_near["mesh"]["near_touch_pairs"] = 0
    obj_quality_skip_near["mesh"]["near_disjoint_pair_count"] = 0
    obj_quality_skip_near["mesh"]["quality_gate_pass"] = True
    obj_quality_skip_near["mesh"]["requires_remesh"] = False
    errors, warnings = validate_result(obj_quality_skip_near)
    assert errors == [], errors
    bad_type = valid_result()
    bad_type["mesh"]["feature_edges_30deg"] = 1.5
    errors, warnings = validate_result(bad_type)
    assert "mesh.feature_edges_30deg must be integer" in errors, errors
    bad_edge_metric = valid_result()
    bad_edge_metric["mesh"]["max_dihedral_deg"] = float("nan")
    bad_edge_metric["mesh"]["max_adjacent_area_ratio"] = -1.0
    errors, warnings = validate_result(bad_edge_metric)
    assert "mesh.max_dihedral_deg must be finite" in errors, errors
    assert "mesh.max_adjacent_area_ratio must be non-negative" in errors, errors
    bad_type = valid_result()
    bad_type["shape"] = 7
    bad_type["mesh"]["quality_gate_pass"] = "yes"
    errors, warnings = validate_result(bad_type)
    assert "shape must be string" in errors, errors
    assert "mesh.quality_gate_pass must be boolean" in errors, errors
    bad_mesh = valid_result()
    bad_mesh["mesh"]["quality_gate_pass"] = False
    errors, warnings = validate_result(bad_mesh)
    assert "mesh.quality_gate_pass must be true for accepted results" in errors, errors

    bad_near_touch = valid_result()
    bad_near_touch["shape"] = "obj"
    bad_near_touch["obj_file"] = "dust.obj"
    bad_near_touch["mesh"]["near_touch_checked"] = False
    bad_near_touch["mesh"]["near_touch_pairs"] = 1
    errors, warnings = validate_result(bad_near_touch)
    assert "mesh.near_touch_checked must be true for accepted results" in errors, errors
    generated_hex = valid_result()
    generated_hex["shape"] = "hex_prism"
    generated_hex["mesh"]["near_touch_checked"] = False
    errors, warnings = validate_result(generated_hex)
    assert "mesh.near_touch_checked must be true for accepted results" not in errors, errors
    assert warnings == [], warnings
    bad_near_touch = valid_result()
    bad_near_touch["mesh"]["near_touch_pairs"] = 2
    errors, warnings = validate_result(bad_near_touch)
    assert "mesh.near_touch_pairs must be 0, got 2" in errors, errors
    bad_near_touch = valid_result()
    bad_near_touch["mesh"]["near_disjoint_pair_count"] = 1
    bad_near_touch["mesh"]["taylor_duffy_candidate_count"] = 5281
    errors, warnings = validate_result(bad_near_touch)
    assert "mesh.near_disjoint_pair_count must be 0, got 1" in errors, errors
    bad_near_touch = valid_result()
    bad_near_touch["mesh"]["self_panel_count"] = 2111
    bad_near_touch["mesh"]["taylor_duffy_candidate_count"] = 5279
    errors, warnings = validate_result(bad_near_touch)
    assert "mesh.self_panel_count must equal mesh.triangles, got 2111 and 2112" in errors, errors
    bad_near_touch = valid_result()
    bad_near_touch["mesh"]["taylor_duffy_candidate_count"] = 7
    errors, warnings = validate_result(bad_near_touch)
    assert "mesh.taylor_duffy_candidate_count must equal self+edge+vertex+near-disjoint (5280), got 7" in errors, errors
    bad_near_touch = valid_result()
    bad_near_touch["mesh"]["requires_remesh"] = True
    errors, warnings = validate_result(bad_near_touch)
    assert "mesh.requires_remesh must be false for accepted results" in errors, errors
    bad_near_touch = valid_result()
    bad_near_touch["mesh"]["recommended_min_quad_order"] = 13
    bad_near_touch["method"]["quad_order"] = 7
    errors, warnings = validate_result(bad_near_touch)
    assert "method.quad_order must be >= mesh.recommended_min_quad_order (13), got 7" in errors, errors
    bad_near_touch = valid_result()
    bad_near_touch["mesh"]["recommended_mesh_strategy"] = 123
    bad_near_touch["mesh"]["recommended_mesh_action"] = []
    errors, warnings = validate_result(bad_near_touch)
    assert "mesh.recommended_mesh_strategy must be string" in errors, errors
    assert "mesh.recommended_mesh_action must be string" in errors, errors
    bad_near_touch = valid_result()
    bad_near_touch["mesh"]["near_touch_ratio"] = float("nan")
    errors, warnings = validate_result(bad_near_touch)
    assert "mesh.near_touch_ratio must be finite" in errors, errors

    bad_run = valid_result()
    bad_run.update({
        "ka": 0.0,
        "refinements": -1,
        "fmm_digits": 0,
        "gmres_tol": -1e-3,
        "gmres_restart": 0,
        "gmres_max_final_relres": -0.1,
        "gmres_numerical_breakdowns": -1,
        "gmres_restored_best_iterates": -1,
        "gmres_max_cycle_exhaustions": -1,
        "orientation_weight_sum": 0.0,
        "orient_start": 3,
        "orient_count": 2,
        "orient_total": 4,
        "edge_refine": -1,
    })
    errors, warnings = validate_result(bad_run)
    assert "ka must be finite and positive" in errors, errors
    assert "refinements must be integer >= 0" in errors, errors
    assert "fmm_digits must be integer >= 1" in errors, errors
    assert "gmres_tol must be finite and positive" in errors, errors
    assert "gmres_restart must be integer >= 1" in errors, errors
    assert "gmres_max_final_relres must be finite and non-negative" in errors, errors
    assert "gmres_numerical_breakdowns must be integer >= 0" in errors, errors
    assert "gmres_restored_best_iterates must be integer >= 0" in errors, errors
    assert "gmres_max_cycle_exhaustions must be integer >= 0" in errors, errors
    assert "orientation_weight_sum must be finite and positive" in errors, errors
    assert "orient_count must fit within orient_total from orient_start" in errors, errors
    assert "edge_refine must be integer >= 0" in errors, errors

    bad_nan_run = valid_result()
    bad_nan_run["n_re"] = float("nan")
    bad_nan_run["n_im"] = -0.1
    bad_nan_run["time_total"] = float("inf")
    errors, warnings = validate_result(bad_nan_run)
    assert "n_re must be finite" in errors, errors
    assert "n_im must be finite and non-negative" in errors, errors
    assert "time_total must be finite and non-negative" in errors, errors

    obj_accurate = valid_result()
    obj_accurate["method"]["solver_profile"] = "obj_accurate"
    obj_accurate["fmm_digits"] = 7
    obj_accurate["gmres_tol"] = 1e-5
    obj_accurate["gmres_restart"] = 1000
    obj_accurate["gmres_max_cycles"] = 80
    obj_accurate["method"]["gmres_true_residual_checked"] = True
    obj_accurate["ka"] = 20.0
    obj_accurate["shape"] = "obj"
    obj_accurate["obj_file"] = "dust.obj"
    errors, warnings = validate_result(obj_accurate)
    assert errors == [], errors
    assert warnings == [], warnings

    obj_mesh_guard = valid_result()
    obj_mesh_guard["method"]["solver_profile"] = "obj_mesh_guard"
    obj_mesh_guard["fmm_digits"] = 8
    obj_mesh_guard["gmres_tol"] = 1e-5
    obj_mesh_guard["gmres_restart"] = 1400
    obj_mesh_guard["gmres_max_cycles"] = 80
    obj_mesh_guard["method"]["gmres_true_residual_checked"] = True
    obj_mesh_guard["ka"] = 20.0
    obj_mesh_guard["shape"] = "obj"
    obj_mesh_guard["obj_file"] = "dust.obj"
    errors, warnings = validate_result(obj_mesh_guard)
    assert errors == [], errors
    assert warnings == [], warnings

    stale_obj_mesh_guard = valid_result()
    stale_obj_mesh_guard["method"]["solver_profile"] = "obj_mesh_guard"
    stale_obj_mesh_guard["fmm_digits"] = 6
    stale_obj_mesh_guard["gmres_tol"] = 5e-4
    stale_obj_mesh_guard["gmres_restart"] = 1000
    stale_obj_mesh_guard["gmres_max_cycles"] = 40
    stale_obj_mesh_guard["ka"] = 20.0
    stale_obj_mesh_guard["shape"] = "obj"
    stale_obj_mesh_guard["obj_file"] = "dust.obj"
    errors, warnings = validate_result(stale_obj_mesh_guard)
    assert "obj_mesh_guard results require fmm_digits >= 8" in errors, errors
    assert "obj_mesh_guard results require gmres_tol <= 1e-05" in errors, errors
    assert "obj_mesh_guard results require gmres_restart >= 1400" in errors, errors
    assert "obj_mesh_guard results require gmres_max_cycles >= 80" in errors, errors
    assert "obj_mesh_guard results require method.gmres_true_residual_checked=true" in errors, errors

    stale_obj_no_true_residual = valid_result()
    stale_obj_no_true_residual["method"]["solver_profile"] = "obj_accurate"
    stale_obj_no_true_residual["fmm_digits"] = 7
    stale_obj_no_true_residual["gmres_tol"] = 1e-5
    stale_obj_no_true_residual["gmres_restart"] = 1000
    stale_obj_no_true_residual["gmres_max_cycles"] = 80
    stale_obj_no_true_residual["ka"] = 20.0
    stale_obj_no_true_residual["shape"] = "obj"
    stale_obj_no_true_residual["obj_file"] = "dust.obj"
    errors, warnings = validate_result(stale_obj_no_true_residual)
    assert "obj_accurate results require method.gmres_true_residual_checked=true" in errors, errors

    stale_obj_accurate = valid_result()
    stale_obj_accurate["method"]["solver_profile"] = "obj_accurate"
    stale_obj_accurate["fmm_digits"] = 6
    stale_obj_accurate["gmres_tol"] = 5e-4
    stale_obj_accurate["gmres_restart"] = 500
    stale_obj_accurate["gmres_max_cycles"] = 30
    stale_obj_accurate["ka"] = 20.0
    stale_obj_accurate["shape"] = "obj"
    stale_obj_accurate["obj_file"] = "dust.obj"
    errors, warnings = validate_result(stale_obj_accurate)
    assert "obj_accurate results require fmm_digits >= 7" in errors, errors
    assert "obj_accurate results require gmres_tol <= 1e-05" in errors, errors
    assert "obj_accurate results require gmres_restart >= 1000" in errors, errors
    assert "obj_accurate results require gmres_max_cycles >= 80" in errors, errors

    errors = case_contract_errors(
        obj_accurate,
        Path("dust_ka20_gmsh4200_balanced_q7_d6_tol5e4.json"),
    )
    assert errors == [], errors
    errors = case_contract_errors(
        stale_obj_accurate,
        Path("dust_ka20_gmsh4200_balanced_q7_d6_tol5e4.json"),
    )
    assert errors == [], errors

    mislabeled_dust = valid_result()
    mislabeled_dust["fmm_digits"] = 5
    mislabeled_dust["gmres_tol"] = 1e-3
    mislabeled_dust["gmres_restart"] = 220
    mislabeled_dust["ka"] = 20.0
    errors, warnings = validate_result(
        mislabeled_dust,
        result_path=Path("dust_ka20_gmsh4200_balanced_q7_d6_tol5e4.json"),
    )
    assert "case name requires fmm_digits >= 6" in errors, errors
    assert "case name requires gmres_tol <= 0.0005" in errors, errors
    assert "dust d6/tol5e4 results require solver_profile=obj_accurate/obj_strict/obj_mesh_guard" in errors, errors
    assert "dust d6/tol5e4 results require gmres_restart >= 500" in errors, errors
    assert "dust case name requires shape=obj" in errors, errors
    assert "dust case name requires non-empty obj_file" in errors, errors

    wrong_ka = valid_result()
    wrong_ka["ka"] = 9.0
    errors, warnings = validate_result(
        wrong_ka,
        result_path=Path("hex_ka10_ref3_balanced_q7_d5_tol1e3.json"),
    )
    assert "case name requires ka=10" in errors, errors

    wrong_ref = valid_result()
    wrong_ref["refinements"] = 4
    errors, warnings = validate_result(
        wrong_ref,
        result_path=Path("hex_ka10_ref3_balanced_q7_d5_tol1e3.json"),
    )
    assert "case name requires refinements=3" in errors, errors

    hex_named_sphere = valid_result()
    hex_named_sphere["shape"] = "sphere"
    errors, warnings = validate_result(
        hex_named_sphere,
        result_path=Path("hex_ka10_ref3_balanced_q7_d5_tol1e3.json"),
    )
    assert "hex case name requires shape=hex_prism or prism6" in errors, errors

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dust_ka20_gmsh4200_balanced_q7_d6_tol5e4.json"
        payload = valid_result()
        payload.update({
            "theta": [0.0],
            "mueller": [[1.0] + [0.0] * 15],
            "fmm_digits": 5,
            "gmres_tol": 1e-3,
            "gmres_restart": 220,
            "ka": 20.0,
            "shape": "obj",
            "obj_file": "dust.obj",
        })
        path.write_text(json.dumps(payload))
        proc = subprocess.run(
            [
                "python3",
                str(ROOT / "scripts" / "check_result_metadata.py"),
                "--strict",
                "--validate-numeric",
                str(path),
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert proc.returncode == 2, proc.stdout
        assert "case name requires fmm_digits >= 6" in proc.stdout, proc.stdout
        assert "dust d6/tol5e4 results require solver_profile=obj_accurate" in proc.stdout, proc.stdout

        cloude_path = Path(tmp) / "cloude_bad.json"
        cloude_payload = valid_result()
        cloude_payload.update({
            "theta": [0.0],
            "mueller": [[1.0] + [0.0] * 15],
        })
        cloude_payload["mueller"][0][5] = 0.8
        cloude_payload["mueller"][0][10] = 0.8
        cloude_payload["mueller"][0][15] = -0.8
        cloude_path.write_text(json.dumps(cloude_payload))
        proc = subprocess.run(
            [
                "python3",
                str(ROOT / "scripts" / "check_result_metadata.py"),
                "--validate-numeric",
                "--require-cloude-physical",
                str(cloude_path),
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert proc.returncode == 2, proc.stdout
        assert "mueller Cloude coherency matrix" in proc.stdout, proc.stdout

    not_converged = valid_result()
    not_converged["gmres_nonconverged_systems"] = 1
    errors, warnings = validate_result(not_converged, require_converged=True)
    assert "gmres_nonconverged_systems must be 0, got 1" in errors, errors

    broken_gmres = valid_result()
    broken_gmres["gmres_numerical_breakdowns"] = 1
    errors, warnings = validate_result(broken_gmres, require_converged=True)
    assert "gmres_numerical_breakdowns must be 0, got 1" in errors, errors

    exhausted_gmres = valid_result()
    exhausted_gmres["gmres_max_cycle_exhaustions"] = 1
    errors, warnings = validate_result(exhausted_gmres, require_converged=True)
    assert "gmres_max_cycle_exhaustions must be 0, got 1" in errors, errors

    loose_residual = valid_result()
    loose_residual["gmres_max_final_relres"] = 2e-2
    errors, warnings = validate_result(loose_residual, require_converged=True)
    assert "gmres_max_final_relres must be <= 0.01, got 0.02" in errors, errors

    print("result metadata check: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
