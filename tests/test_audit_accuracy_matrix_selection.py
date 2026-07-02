#!/usr/bin/env python3
"""Tests for selecting the best accuracy-matrix candidate."""

from pathlib import Path
import sys
import subprocess


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_accuracy_matrix_15 import (  # noqa: E402
    audit_exit_code,
    bad_operator_rows,
    best,
    build_cases,
    case_contract_errors,
    component_failure_summary,
    result_metadata,
    select_best_existing,
)
import audit_accuracy_matrix_15  # noqa: E402


def main() -> int:
    legacy_accurate = {
        "pass10": False,
        "raw_pass10": True,
        "metadata_rank": 1,
        "operator_rank": 0,
        "gate_error": 0.04,
        "time_s": 20.0,
        "bem_file": "legacy-but-accurate.json",
    }
    current_inaccurate = {
        "pass10": False,
        "raw_pass10": False,
        "metadata_rank": 0,
        "operator_rank": 0,
        "gate_error": 0.20,
        "time_s": 10.0,
        "bem_file": "current-but-inaccurate.json",
    }
    chosen = select_best_existing([current_inaccurate, legacy_accurate])
    assert chosen["bem_file"] == "legacy-but-accurate.json", chosen

    current_accurate = {
        "pass10": True,
        "raw_pass10": True,
        "metadata_rank": 0,
        "operator_rank": 0,
        "gate_error": 0.06,
        "time_s": 30.0,
        "bem_file": "current-accurate.json",
    }
    chosen = select_best_existing([legacy_accurate, current_accurate])
    assert chosen["bem_file"] == "current-accurate.json", chosen

    string_false = {
        "pass10": "False",
        "raw_pass10": "False",
        "metadata_rank": 0,
        "operator_rank": 0,
        "gate_error": 0.20,
        "time_s": 1.0,
        "bem_file": "string-false.json",
    }
    string_true = {
        "pass10": "False",
        "raw_pass10": "True",
        "metadata_rank": 1,
        "operator_rank": 0,
        "gate_error": 0.04,
        "time_s": 2.0,
        "bem_file": "string-true.json",
    }
    chosen = select_best_existing([string_false, string_true])
    assert chosen["bem_file"] == "string-true.json", chosen

    strict5_candidate = {
        "pass5": True,
        "raw_pass5": True,
        "pass10": True,
        "raw_pass10": True,
        "metadata_rank": 0,
        "operator_rank": 0,
        "gate_error": 0.045,
        "time_s": 20.0,
        "bem_file": "strict-5pct.json",
    }
    practical10_candidate = {
        "pass5": False,
        "raw_pass5": False,
        "pass10": True,
        "raw_pass10": True,
        "metadata_rank": 0,
        "operator_rank": 0,
        "gate_error": 0.020,
        "time_s": 1.0,
        "bem_file": "practical-10pct.json",
    }
    chosen = select_best_existing([practical10_candidate, strict5_candidate])
    assert chosen["bem_file"] == "strict-5pct.json", chosen

    invalid_numeric = {
        "gmres_nonconverged_systems": 0,
        "gmres_stagnation_stops": 0,
        "gmres_max_final_relres": 9e-4,
        "gmres_tol": 1e-3,
        "theta": [0.0, 1.0],
        "mueller": [
            [1.0] + [0.0] * 15,
            [-0.1] + [0.0] * 15,
        ],
        "method": {
            "solver_backend": "FMM",
            "solver_profile": "default",
            "requested_system": "balanced",
            "system": "balanced",
            "system_canonicalized": False,
            "quad_order": 4,
            "preconditioner_enabled": False,
            "schwarz_preconditioner": False,
            "preconditioner_reason": "user_disabled",
            "farfield_mode": "gpu_geometry_direct",
        },
        "mesh": {
            "vertices": 4,
            "triangles": 4,
            "skinny_triangles": 0,
            "min_angle_deg": 50.0,
            "max_aspect_ratio": 1.2,
            "feature_edges_30deg": 0,
            "max_dihedral_deg": 20.0,
            "mean_feature_dihedral_deg": 0.0,
            "max_adjacent_area_ratio": 1.1,
            "near_touch_checked": True,
            "near_touch_ratio": 1.0,
            "near_touch_pairs": 0,
            "self_panel_count": 4,
            "edge_adjacent_pair_count": 6,
            "vertex_adjacent_pair_count": 0,
            "near_disjoint_pair_count": 0,
            "taylor_duffy_candidate_count": 10,
            "recommended_min_quad_order": 4,
            "recommended_mesh_strategy": "uniform_curvature_refinement",
            "recommended_mesh_action": "uniform smooth-surface refinement is acceptable",
            "requires_remesh": False,
            "edge_refine_requested": 0,
            "edge_refine_applied": 0,
            "edge_refine_uniform_fallback": False,
        },
    }
    meta = result_metadata(invalid_numeric)
    assert meta["metadata_status"] == "invalid", meta
    assert meta["farfield_mode"] == "gpu_geometry_direct", meta
    assert "mueller M11 must be non-negative" in meta["metadata_errors"], meta

    nonconverged = dict(invalid_numeric)
    nonconverged["theta"] = [0.0]
    nonconverged["mueller"] = [[1.0] + [0.0] * 15]
    nonconverged["gmres_nonconverged_systems"] = 1
    meta = result_metadata(nonconverged)
    assert meta["metadata_status"] == "invalid", meta
    assert "gmres_nonconverged_systems must be 0, got 1" in meta["metadata_errors"], meta

    cloude_invalid = dict(invalid_numeric)
    cloude_invalid["theta"] = [0.0]
    cloude_invalid["mueller"] = [[1.0] + [0.0] * 15]
    cloude_invalid["mueller"][0][5] = 0.8
    cloude_invalid["mueller"][0][10] = 0.8
    cloude_invalid["mueller"][0][15] = -0.8
    meta = result_metadata(cloude_invalid)
    assert meta["metadata_status"] == "invalid", meta
    assert "mueller Cloude coherency matrix" in meta["metadata_errors"], meta

    failures = component_failure_summary({
        "M11": 0.04,
        "M12": 0.11,
        "M21": 0.03,
        "M34": 0.24,
        "M43": 0.09,
        "M44": 0.01,
    })
    assert failures["worst_component"] == "M34", failures
    assert failures["failed_main_10pct"] == "M12,M34", failures
    assert failures["failed_main_5pct"] == "M12,M34,M43", failures
    assert failures["failed_all_20pct"] == "M34", failures

    accurate_obj_meta = {
        "ka": 20.0,
        "fmm_digits": 6,
        "gmres_tol": 5e-4,
        "gmres_restart": 500,
        "shape": "obj",
        "obj_file": "dust.obj",
        "method": {"solver_profile": "obj_accurate", "farfield_mode": "gpu_geometry_direct"},
    }
    assert not case_contract_errors(
        accurate_obj_meta,
        "dust_ka20_gmsh4200_balanced_q7_d6_tol5e4.json",
    )
    mislabeled_obj_meta = {
        "ka": 20.0,
        "fmm_digits": 5,
        "gmres_tol": 1e-3,
        "gmres_restart": 220,
        "shape": "obj",
        "obj_file": "dust.obj",
        "method": {"solver_profile": "default", "farfield_mode": "gpu_host_geometry_mueller_accum"},
    }
    errors = case_contract_errors(
        mislabeled_obj_meta,
        "dust_ka20_gmsh4200_balanced_q7_d6_tol5e4.json",
    )
    assert "case name requires fmm_digits >= 6" in errors, errors
    assert "case name requires gmres_tol <= 0.0005" in errors, errors
    assert "dust d6/tol5e4 results require solver_profile=obj_accurate/obj_strict/obj_mesh_guard" in errors, errors
    assert "dust d6/tol5e4 results require gmres_restart >= 500" in errors, errors

    original_score_adda = audit_accuracy_matrix_15.score_adda
    try:
        audit_accuracy_matrix_15.score_adda = lambda _bem_path, _ref_path: {
            "gate_error": 0.01,
            "metadata_status": "ok",
            "metadata_rank": 0,
            "operator_status": "not_required",
            "operator_rank": 0,
            "max_main_floor2": 0.01,
            "mie_mean_floor2": float("nan"),
            "mie_max_floor2": float("nan"),
            "worst_component": "M11",
            "worst_component_error": 0.01,
            "failed_main_10pct": "",
            "failed_main_5pct": "",
            "failed_all_20pct": "",
            "bem_file": "candidate-without-full16-flag.json",
            "reference": "ADDA-OCL",
            "reference_file": "adda/mueller",
            "m11": 0.01,
            "m12": 0.01,
            "m34": 0.01,
            "time_s": 1.0,
        }
        row = best(
            "сфера",
            5.0,
            "ref4",
            [(Path("candidate.json"), "ADDA-OCL", Path("mueller"))],
        )
        assert row["status"] == "FAIL", row
        assert row["status_5pct"] == "FAIL", row
        assert row["raw_pass5"] is False, row
        assert row["pass5"] is False, row
        assert row["raw_pass10"] is False, row
        assert row["pass10"] is False, row
    finally:
        audit_accuracy_matrix_15.score_adda = original_score_adda

    assert audit_exit_code(
        [{"pass10": True, "metadata_status": "invalid"}],
        require_current_metadata=False,
        require_complex_operator_for_absorbing=False,
    ) == 5
    assert audit_exit_code(
        [{"pass10": True, "metadata_status": "legacy", "operator_status": "not_required"}],
        require_current_metadata=True,
        require_complex_operator_for_absorbing=False,
    ) == 3
    assert audit_exit_code(
        [{"pass10": True, "metadata_status": "legacy", "operator_status": "not_required"}],
        require_current_metadata=False,
        require_complex_operator_for_absorbing=False,
    ) == 0
    assert audit_exit_code(
        [{
            "pass10": True,
            "metadata_status": "ok",
            "operator_status": "old_absorbing_operator_unverified",
        }],
        require_current_metadata=False,
        require_complex_operator_for_absorbing=True,
    ) == 4
    assert audit_exit_code(
        [{
            "pass10": True,
            "metadata_status": "ok",
            "operator_status": "old_absorbing_operator_unverified",
        }],
        require_current_metadata=False,
        require_complex_operator_for_absorbing=False,
    ) == 0
    assert audit_exit_code(
        [{"pass10": True, "metadata_status": "ok", "operator_status": "missing"}],
        require_current_metadata=False,
        require_complex_operator_for_absorbing=True,
    ) == 4
    bad_operator = bad_operator_rows([
        {"shape": "dust", "operator_status": "complex_operator"},
        {"shape": "sphere", "operator_status": "not_required"},
        {"shape": "dust", "operator_status": "old_absorbing_operator_unverified"},
        {"shape": "hex", "operator_status": "missing"},
        {"shape": "unknown"},
    ])
    assert [row["shape"] for row in bad_operator] == ["dust", "hex", "unknown"], bad_operator
    assert audit_exit_code(
        [{"pass10": False, "metadata_status": "ok", "operator_status": "not_required"}],
        require_current_metadata=False,
        require_complex_operator_for_absorbing=False,
    ) == 2

    help_proc = subprocess.run(
        ["python3", str(ROOT / "scripts" / "audit_accuracy_matrix_15.py"), "--help"],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert help_proc.returncode == 0, help_proc.stdout
    assert "--allow-missing-complex-operator-for-absorbing" in help_proc.stdout, help_proc.stdout

    cases = build_cases()
    candidate_names_all = {
        Path(candidate[0]).name
        for _shape, _ka, _mesh_label, candidates in cases
        for candidate in candidates
    }
    required_refresh_dust = {
        "dust_ka5_adda_mc_f6000_q13_d7_tol1e5.json",
        "dust_ka10_gmsh3900_a35_q7_d7_tol1e5.json",
        "dust_ka15_qdec_f5000_t15_q13_d8_tol1e5.json",
        "dust_ka20_gmsh4200_a35_q7_d7_tol1e5.json",
        "dust_ka30_gmsh7000_a45_q7_d7_tol1e5.json",
    }
    assert required_refresh_dust <= candidate_names_all, (
        sorted(required_refresh_dust - candidate_names_all)
    )

    queue_plan = subprocess.run(
        ["bash", str(ROOT / "scripts" / "queue_poster_true_residual_refresh.sh"), "--plan"],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout.splitlines()
    queue_names = {
        line.split(",", 2)[1] + ".json"
        for line in queue_plan[1:]
        if line.strip()
    }
    assert queue_names <= candidate_names_all, sorted(queue_names - candidate_names_all)
    queue_script = (ROOT / "scripts" / "queue_poster_true_residual_refresh.sh").read_text()
    assert queue_script.count("--gmres-max-cycles 80") >= 5, queue_script

    dust_cases = [case for case in cases if case[0] == "пылевая частица"]
    assert len(dust_cases) == 5, dust_cases
    for shape, ka, _mesh_label, candidates in dust_cases:
        candidate_names = [Path(candidate[0]).name for candidate in candidates]
        q7d6 = [name for name in candidate_names if "balanced_q7_d6_tol5e4" in name]
        assert q7d6, (shape, ka, candidate_names[:8])
        forbidden = ("q7_d5", "tol1e3", "pmchwt", "muller2b", "complexop")
        stale = [name for name in candidate_names
                 if any(token in name for token in forbidden)]
        assert not stale, (shape, ka, stale[:8])

    print("audit accuracy matrix selection: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
