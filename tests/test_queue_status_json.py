#!/usr/bin/env python3
"""Unit tests for machine-readable production queue status."""

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from queue_status_json import (  # noqa: E402
    gpu_inventory,
    parse_accuracy_status,
    parse_gpu_inventory,
    requested_exit_code,
    result_state,
    result_summary,
)


def valid_result(*, complex_operator: bool) -> dict:
    method = {
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
    }
    if complex_operator:
        method.update({
            "row_h_scale": 0.625,
            "row_h_scale_imag": -0.00078125,
            "row_h_scale_complex": [0.625, -0.00078125],
        })
    return {
        "theta": [0.0],
        "mueller": [[1.0] + [0.0] * 15],
        "ka": 5.0,
        "refinements": 4,
        "shape": "sphere",
        "obj_file": None,
        "prism_aspect": 1.0,
        "edge_refine": 0,
        "gmres_nonconverged_systems": 0,
        "gmres_stagnation_stops": 0,
        "gmres_numerical_breakdowns": 0,
        "gmres_restored_best_iterates": 0,
        "gmres_max_cycle_exhaustions": 0,
        "gmres_max_final_relres": 0.0009,
        "gmres_tol": 0.001,
        "gmres_max_cycles": 80,
        "fmm_digits": 6,
        "method": method,
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
            "quality_gate_pass": True,
        },
    }


def accurate_dust_result() -> dict:
    result = valid_result(complex_operator=True)
    result["gmres_tol"] = 1e-5
    result["gmres_max_final_relres"] = 9e-6
    result["gmres_restart"] = 1000
    result["fmm_digits"] = 7
    result["method"]["solver_profile"] = "obj_accurate"
    result["method"]["gmres_true_residual_checked"] = True
    result["shape"] = "obj"
    result["obj_file"] = "dust.obj"
    return result


def current_dust_result() -> dict:
    result = valid_result(complex_operator=True)
    result["shape"] = "obj"
    result["obj_file"] = "dust.obj"
    return result


def main() -> int:
    gpu = parse_gpu_inventory(
        """\
0, Tesla V100, 45, 0, 900, 50.0, 250.0
1, Tesla V100, 52, 5, 1100, 70.0, 250.0
2, Tesla V100, 84, 99, 22000, 220.0, 250.0
""",
        {"1": "1234, ./mbs_po_gpu_float_fast, 304"},
        max_temp_c=80,
        max_util_pct=20,
        max_mem_mib=2048,
        allow_compute_share=False,
    )
    assert gpu["counts"] == {
        "total": 3,
        "usable": 1,
        "busy": 1,
        "unhealthy": 1,
        "unparseable": 0,
    }, gpu
    assert gpu["usable_gpu_indices"] == [0], gpu
    assert gpu["busy_gpu_indices"] == [1], gpu
    assert gpu["unhealthy_gpu_indices"] == [2], gpu
    assert gpu["gpus"][1]["reasons"] == ["compute_apps"], gpu
    assert "temp>80" in gpu["gpus"][2]["reasons"], gpu
    assert "util>20" in gpu["gpus"][2]["reasons"], gpu
    assert "mem>2048" in gpu["gpus"][2]["reasons"], gpu

    gpu = parse_gpu_inventory(
        "1, Tesla V100, 52, 5, 1100, 70.0, 250.0\n",
        {"1": "1234, ./mbs_po_gpu_float_fast, 304"},
        max_temp_c=80,
        max_util_pct=20,
        max_mem_mib=2048,
        allow_compute_share=True,
    )
    assert gpu["counts"]["usable"] == 1, gpu
    assert gpu["gpus"][0]["compute_apps"] == ["1234, ./mbs_po_gpu_float_fast, 304"], gpu

    parsed = parse_accuracy_status("""\
ACCURATE sphere_ka5_ref4_current_q7_d6_tol3e3 gate=0.03
ACCURATE_LEGACY hex_ka10_ref3_balanced_q7_d5_tol1e3 gate=0.04 metadata=legacy operator=not_required
INACCURATE dust_ka15_gmsh6000_balanced_q7_d5_tol1e3 gate=0.22 metadata=ok operator=complex_operator
MISSING_ACCURACY dust_ka20_gmsh4200_balanced_q7_d5_tol1e3
SUMMARY_ACCURACY accurate=1 accurate_legacy=1 inaccurate=1 missing=1 total=4
""")
    assert parsed["counts"] == {
        "accurate": 1,
        "accurate_legacy": 1,
        "inaccurate": 1,
        "missing": 1,
        "total": 4,
    }, parsed
    assert parsed["cases"][0]["gate_error"] == 0.03, parsed
    assert parsed["accurate_cases"] == ["sphere_ka5_ref4_current_q7_d6_tol3e3"], parsed
    assert parsed["accurate_legacy_cases"] == ["hex_ka10_ref3_balanced_q7_d5_tol1e3"], parsed
    assert parsed["inaccurate_cases"] == ["dust_ka15_gmsh6000_balanced_q7_d5_tol1e3"], parsed
    assert parsed["missing_accuracy_cases"] == ["dust_ka20_gmsh4200_balanced_q7_d5_tol1e3"], parsed
    assert parsed["summary_counts"] == {
        "accurate": 1,
        "accurate_legacy": 1,
        "inaccurate": 1,
        "missing": 1,
        "total": 4,
    }, parsed
    assert parsed["summary_mismatch"] is False, parsed

    parsed = parse_accuracy_status("""\
ACCURATE sphere_ka5_ref4_current_q7_d6_tol3e3 gate=0.03
SUMMARY_ACCURACY accurate=2 accurate_legacy=0 inaccurate=0 missing=0 total=2
""")
    assert parsed["counts"] == {
        "accurate": 1,
        "accurate_legacy": 0,
        "inaccurate": 0,
        "missing": 0,
        "total": 1,
    }, parsed
    assert parsed["summary_counts"] == {
        "accurate": 2,
        "accurate_legacy": 0,
        "inaccurate": 0,
        "missing": 0,
        "total": 2,
    }, parsed
    assert parsed["summary_mismatch"] is True, parsed

    class Args:
        fail_on_failed = False
        fail_on_stalled = False
        fail_on_stopped_incomplete = False
        fail_on_missing = False
        fail_on_stale = False
        fail_on_incomplete = False
        fail_on_inaccurate = True
        fail_on_accuracy_legacy = False
        fail_on_accuracy_summary_mismatch = False
        fail_on_gpu_inventory_unavailable = False
        min_usable_gpus = None

    payload = {"accuracy": {"counts": {"inaccurate": 1, "missing": 0, "accurate_legacy": 0}}}
    assert requested_exit_code(payload, Args) == 26
    Args.fail_on_inaccurate = False
    Args.fail_on_accuracy_legacy = True
    payload = {"accuracy": {"counts": {"inaccurate": 0, "missing": 0, "accurate_legacy": 1}}}
    assert requested_exit_code(payload, Args) == 27
    payload = {"accuracy": {"counts": {"inaccurate": 0, "missing": 1, "accurate_legacy": 0}}}
    assert requested_exit_code(payload, Args) == 27
    Args.fail_on_accuracy_legacy = False
    Args.fail_on_accuracy_summary_mismatch = True
    payload = {"accuracy": {"counts": {"inaccurate": 0, "missing": 0}, "summary_mismatch": True}}
    assert requested_exit_code(payload, Args) == 30
    Args.fail_on_accuracy_summary_mismatch = False
    Args.fail_on_gpu_inventory_unavailable = True
    payload = {"gpu_inventory": {"available": False, "counts": {"usable": 0}}}
    assert requested_exit_code(payload, Args) == 28
    Args.fail_on_gpu_inventory_unavailable = False
    Args.min_usable_gpus = 2
    payload = {"gpu_inventory": {"available": True, "counts": {"usable": 1}}}
    assert requested_exit_code(payload, Args) == 29
    payload = {"gpu_inventory": {"available": True, "counts": {"usable": 2}}}
    assert requested_exit_code(payload, Args) == 0
    Args.min_usable_gpus = None

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        sphere = out / "sphere_ka5_ref4_current_q7_d6_tol3e3.json"
        dust_old = out / "dust_ka5_gmsh3400_balanced_q7_d5_tol1e3.json"
        dust_new = out / "dust_ka10_gmsh5200_balanced_q7_d5_tol1e3.json"
        dust_accurate = out / "dust_ka20_gmsh4200_balanced_q7_d6_tol5e4.json"
        dust_mislabeled = out / "dust_ka30_gmsh7000_balanced_q7_d6_tol5e4.json"

        sphere.write_text(json.dumps(valid_result(complex_operator=False)))
        dust_old.write_text(json.dumps(valid_result(complex_operator=False)))
        dust_new_payload = current_dust_result()
        dust_new_payload["ka"] = 10.0
        dust_new.write_text(json.dumps(dust_new_payload))
        dust_accurate_payload = accurate_dust_result()
        dust_accurate_payload["ka"] = 20.0
        dust_accurate.write_text(json.dumps(dust_accurate_payload))
        dust_mislabeled_payload = current_dust_result()
        dust_mislabeled_payload["ka"] = 30.0
        dust_mislabeled.write_text(json.dumps(dust_mislabeled_payload))

        assert result_state(out, sphere.stem) == "current"
        summary = result_summary(out, sphere.stem)
        assert summary["ka"] == 5.0
        assert summary["refinements"] == 4
        assert summary["gmres_tol"] == 0.001
        assert summary["gmres_numerical_breakdowns"] == 0
        assert summary["gmres_restored_best_iterates"] == 0
        assert summary["gmres_max_cycle_exhaustions"] == 0
        assert summary["fmm_digits"] == 6
        assert summary["farfield_mode"] == "gpu_geometry_direct"
        assert summary["shape"] == "sphere"
        assert summary["mesh_quality_gate_pass"] is True
        assert summary["mesh_triangles"] == 4
        assert summary["mesh_near_touch_checked"] is True
        assert summary["mesh_near_touch_ratio"] == 1.0
        assert result_state(out, dust_old.stem) == "stale"
        assert result_state(out, dust_new.stem) == "current"
        assert result_state(out, dust_accurate.stem) == "current"
        assert result_state(out, dust_mislabeled.stem) == "stale"
        assert result_state(out, "missing_case") == "missing"

        not_converged = out / "sphere_ka10_ref4_current_q7_d6_tol3e3.json"
        bad = valid_result(complex_operator=False)
        bad["gmres_nonconverged_systems"] = 1
        bad["ka"] = 10.0
        not_converged.write_text(json.dumps(bad))
        assert result_state(out, not_converged.stem) == "stale"

        breakdown = out / "sphere_ka11_ref4_current_q7_d6_tol3e3.json"
        bad = valid_result(complex_operator=False)
        bad["gmres_numerical_breakdowns"] = 1
        bad["ka"] = 11.0
        breakdown.write_text(json.dumps(bad))
        assert result_state(out, breakdown.stem) == "stale"

        cloude_bad = out / "sphere_ka12_ref4_current_q7_d6_tol3e3.json"
        bad = valid_result(complex_operator=False)
        bad["ka"] = 12.0
        bad["mueller"] = [[1.0] + [0.0] * 15]
        bad["mueller"][0][5] = 0.8
        bad["mueller"][0][10] = 0.8
        bad["mueller"][0][15] = -0.8
        cloude_bad.write_text(json.dumps(bad))
        assert result_state(out, cloude_bad.stem) == "stale"

        bad_mesh_file = out / "sphere_ka15_ref4_current_q7_d6_tol3e3.json"
        bad_mesh = valid_result(complex_operator=False)
        bad_mesh["ka"] = 15.0
        bad_mesh["mesh"]["quality_gate_pass"] = False
        bad_mesh_file.write_text(json.dumps(bad_mesh))
        assert result_state(out, bad_mesh_file.stem) == "stale"

        fake_smi = out / "nvidia-smi"
        fake_smi.write_text("""#!/usr/bin/env bash
if [[ "$*" == "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,power.draw,power.limit --format=csv,noheader,nounits" ]]; then
  printf '0, Tesla V100, 41, 0, 700, 45.0, 250.0\\n'
  printf '1, Tesla V100, 50, 1, 900, 60.0, 250.0\\n'
  exit 0
fi
if [[ "$*" == "-i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits" ]]; then
  exit 0
fi
if [[ "$*" == "-i 1 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits" ]]; then
  printf '4242, ./mbs_po_gpu_float_fast, 304\\n'
  exit 0
fi
echo "unexpected fake nvidia-smi args: $*" >&2
exit 2
""")
        fake_smi.chmod(0o755)
        import os
        old_smi = os.environ.get("BEM_NVIDIA_SMI")
        try:
            os.environ["BEM_NVIDIA_SMI"] = str(fake_smi)
            inv = gpu_inventory()
        finally:
            if old_smi is None:
                os.environ.pop("BEM_NVIDIA_SMI", None)
            else:
                os.environ["BEM_NVIDIA_SMI"] = old_smi
        assert inv["available"] is True, inv
        assert inv["counts"]["usable"] == 1, inv
        assert inv["counts"]["busy"] == 1, inv
        assert inv["usable_gpu_indices"] == [0], inv
        assert inv["busy_gpu_indices"] == [1], inv

    print("queue status json: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
