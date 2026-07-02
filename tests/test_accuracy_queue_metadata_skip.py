#!/usr/bin/env python3
"""Smoke-test metadata-aware skipping in the production queue shell script."""

import subprocess
import tempfile
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = Path(os.environ.get(
    "BEM_QUEUE_SCRIPT",
    ROOT / "scripts" / "run_accuracy_matrix_15_queue.sh",
)).resolve()


def run_bash(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", script], cwd=str(ROOT), universal_newlines=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = tmp_path / "out"
        logs = out / "logs"
        logs.mkdir(parents=True)
        legacy = out / "legacy.json"
        legacy.write_text('{"theta":[0],"mueller":[[1]]}\n')
        (logs / "legacy.log").write_text("old\n")

        current = out / "current.json"
        current.write_text(
            """{
  "theta": [0],
  "mueller": [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
  "ka": 10,
  "refinements": 3,
  "shape": "hex_prism",
  "obj_file": null,
  "prism_aspect": 1.5,
  "edge_refine": 0,
  "gmres_nonconverged_systems": 0,
  "gmres_stagnation_stops": 0,
  "gmres_numerical_breakdowns": 0,
  "gmres_restored_best_iterates": 0,
  "gmres_max_cycle_exhaustions": 0,
  "gmres_max_final_relres": 0.0009,
  "gmres_tol": 0.001,
  "gmres_max_cycles": 80,
  "method": {
    "solver_backend": "FMM",
    "solver_profile": "hex_guarded",
    "requested_system": "balanced",
    "system": "balanced",
    "system_canonicalized": false,
    "quad_order": 4,
    "row_h_scale": 0.625,
    "row_h_scale_imag": -0.00078125,
    "row_h_scale_complex": [0.625, -0.00078125],
    "preconditioner_enabled": false,
    "schwarz_preconditioner": false,
    "preconditioner_reason": "small_nonsphere",
    "farfield_mode": "gpu_geometry_direct"
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
    "near_touch_checked": true,
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
    "requires_remesh": false,
    "edge_refine_requested": 0,
    "edge_refine_applied": 0,
    "edge_refine_uniform_fallback": false,
    "quality_gate_pass": true
  }
}
"""
        )

        script = f"""
set -euo pipefail
source "{QUEUE}"
out="{out}"
result_is_current current
! result_is_current legacy
archive_bad_result legacy
test ! -e "{legacy}"
ls "{out}"/legacy.bad_*.json >/dev/null
ls "{logs}"/legacy.bad_*.log >/dev/null
expected_result_names | wc -l | grep -q '^15$'
"{QUEUE}" --plan > "{tmp_path}/plan.txt"
test "$(wc -l < "{tmp_path}/plan.txt")" = "15"
BEM_QUEUE_EXTRA_DUST_VARIANTS=1 "{QUEUE}" --plan > "{tmp_path}/plan_extra.txt"
test "$(wc -l < "{tmp_path}/plan_extra.txt")" = "21"
grep -q '^dust_ka10_gmsh6000_balanced_q7_d6_tol5e4$' "{tmp_path}/plan_extra.txt"
! queue_status > "{tmp_path}/status.txt"
grep -q '^MISSING sphere_ka5_ref4_current_q7_d6_tol3e3$' "{tmp_path}/status.txt"
grep -q '^SUMMARY current=0 stale=0 missing=15 total=15$' "{tmp_path}/status.txt"
"""
        proc = run_bash(script)
        assert proc.returncode == 0, proc.stdout

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = tmp_path / "out"
        (out / "logs").mkdir(parents=True)
        csv_path = tmp_path / "accuracy_no_pass10.csv"
        csv_path.write_text(
            "bem_file,status,gate_error,metadata_status,operator_status\n"
            "runs/production_matrix_15/sphere_ka5_ref4_current_q7_d6_tol3e3.json,PASS,0.03,ok,not_required\n"
            "runs/production_matrix_15/sphere_ka10_ref4_current_q7_d6_tol3e3.json,,0.04,ok,not_required\n"
        )
        script = f"""
set -euo pipefail
source "{QUEUE}"
out="{out}"
BEM_QUEUE_ACCURACY_CSV="{csv_path}" accuracy_status > "{tmp_path}/accuracy_no_pass10_status.txt" && exit 1 || rc=$?
test "$rc" = "2"
grep -q '^INACCURATE sphere_ka5_ref4_current_q7_d6_tol3e3 gate=0.03 metadata=ok operator=not_required$' "{tmp_path}/accuracy_no_pass10_status.txt"
grep -q '^INACCURATE sphere_ka10_ref4_current_q7_d6_tol3e3 gate=0.04 metadata=ok operator=not_required$' "{tmp_path}/accuracy_no_pass10_status.txt"
grep -q '^SUMMARY_ACCURACY accurate=0 accurate_legacy=0 inaccurate=2 missing=13 total=15$' "{tmp_path}/accuracy_no_pass10_status.txt"
"""
        proc = run_bash(script)
        assert proc.returncode == 0, proc.stdout

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "scripts").mkdir()
        (repo / "bin").mkdir()
        (repo / "bin" / "bem_cuda_fmm").write_text("#!/usr/bin/env bash\n")
        (repo / "bin" / "bem_cuda_fmm").chmod(0o755)
        (repo / "scripts" / "check_result_metadata.py").write_text("#!/usr/bin/env python3\n")
        (repo / "scripts" / "audit_accuracy_matrix_15.py").write_text("#!/usr/bin/env python3\n")
        for rel in [
            "runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f3400_a35.obj",
            "runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f4200_a35.obj",
            "runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f5200_a35.obj",
            "runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f6000_a45.obj",
            "runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f7000_a45.obj",
            "runs/adda_ocl_benchmark_ext/shapes/greek_scaled_ka15_dpl20.shape",
        ]:
            path = repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x\n")
        fake_smi = tmp_path / "custom-smi"
        fake_smi.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' \"$*\" >> '" + str(tmp_path / "custom_smi_calls.txt") + "'\n"
            "if [[ \"$*\" == *\"--query-gpu=index\"* ]]; then printf '0\\n1\\n2\\n3\\n'; exit 0; fi\n"
            "if [[ \"$*\" == *\"temperature.gpu\"* ]]; then printf '45, 0, 10\\n'; exit 0; fi\n"
            "if [[ \"$*\" == *\"--query-compute-apps\"* ]]; then exit 0; fi\n"
            "exit 0\n"
        )
        fake_smi.chmod(0o755)
        script = f"""
set -euo pipefail
BEM_NVIDIA_SMI="{fake_smi}" BEM_QUEUE_MAX_TEMP_C=95 \\
  REPO="{repo}" OUT="runs/production_matrix_15" BIN="{repo}/bin/bem_cuda_fmm" \\
  "{QUEUE}" --preflight > "{tmp_path}/preflight_custom_smi.txt"
grep -q '^PREFLIGHT gpu_count=4$' "{tmp_path}/preflight_custom_smi.txt"
grep -q '^PREFLIGHT ok$' "{tmp_path}/preflight_custom_smi.txt"
grep -q -- '--query-gpu=index --format=csv,noheader' "{tmp_path}/custom_smi_calls.txt"
grep -q -- '-i 2 --query-gpu=temperature.gpu,utilization.gpu,memory.used' "{tmp_path}/custom_smi_calls.txt"
"""
        proc = run_bash(script)
        assert proc.returncode == 0, proc.stdout

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = tmp_path / "out"
        (out / "logs").mkdir(parents=True)
        csv_path = tmp_path / "accuracy.csv"
        csv_path.write_text(
            "bem_file,pass10,gate_error,metadata_status,operator_status\n"
            "runs/production_matrix_15/sphere_ka5_ref4_current_q7_d6_tol3e3.json,True,0.03,ok,not_required\n"
            "runs/production_matrix_15/sphere_ka10_ref4_current_q7_d6_tol3e3.json,True,0.04,legacy,not_required\n"
            "runs/production_matrix_15/sphere_ka15_ref4_current_q7_d6_tol3e3.json,False,0.25,ok,not_required\n"
            "runs/production_matrix_15/sphere_ka20_ref4_current_q7_d6_tol3e3.json,True,0.05,ok,missing\n"
        )
        script = f"""
set -euo pipefail
source "{QUEUE}"
out="{out}"
BEM_QUEUE_ACCURACY_CSV="{csv_path}" accuracy_status > "{tmp_path}/accuracy_status.txt" && exit 1 || rc=$?
test "$rc" = "2"
grep -q '^ACCURATE sphere_ka5_ref4_current_q7_d6_tol3e3 gate=0.03$' "{tmp_path}/accuracy_status.txt"
grep -q '^ACCURATE_LEGACY sphere_ka10_ref4_current_q7_d6_tol3e3 gate=0.04 metadata=legacy operator=not_required$' "{tmp_path}/accuracy_status.txt"
grep -q '^INACCURATE sphere_ka15_ref4_current_q7_d6_tol3e3 gate=0.25 metadata=ok operator=not_required$' "{tmp_path}/accuracy_status.txt"
grep -q '^ACCURATE_LEGACY sphere_ka20_ref4_current_q7_d6_tol3e3 gate=0.05 metadata=ok operator=missing$' "{tmp_path}/accuracy_status.txt"
grep -q '^MISSING_ACCURACY sphere_ka30_ref6_current_q7_d6_tol3e3$' "{tmp_path}/accuracy_status.txt"
grep -q '^SUMMARY_ACCURACY accurate=1 accurate_legacy=2 inaccurate=1 missing=11 total=15$' "{tmp_path}/accuracy_status.txt"
"""
        proc = run_bash(script)
        assert proc.returncode == 0, proc.stdout

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo = tmp_path / "repo"
        (repo / "bin").mkdir(parents=True)
        queue_copy = repo / "scripts" / "run_accuracy_matrix_15_queue.sh"
        queue_copy.parent.mkdir(parents=True)
        queue_copy.write_text(QUEUE.read_text())
        (repo / "bin" / "bem_cuda_fmm").write_text("#!/usr/bin/env bash\n")
        (repo / "bin" / "bem_cuda_fmm").chmod(0o755)
        script = f"""
set -euo pipefail
cd "{repo}"
source "{queue_copy}"
test "$bin" = "{repo}/bin/bem_cuda_fmm"
touch bin/bem_cuda_fmm.next
chmod +x bin/bem_cuda_fmm.next
source "{queue_copy}"
test "$bin" = "{repo}/bin/bem_cuda_fmm.next"
BIN=/custom/bem source "{queue_copy}"
test "$bin" = "/custom/bem"
"""
        proc = run_bash(script)
        assert proc.returncode == 0, proc.stdout

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "scripts").mkdir()
        (repo / "bin").mkdir()
        (repo / "bin" / "bem_cuda_fmm").write_text("#!/usr/bin/env bash\n")
        (repo / "bin" / "bem_cuda_fmm").chmod(0o755)
        (repo / "scripts" / "check_result_metadata.py").write_text("#!/usr/bin/env python3\n")
        (repo / "scripts" / "audit_accuracy_matrix_15.py").write_text("#!/usr/bin/env python3\n")
        needed = [
            "runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f3400_a35.obj",
            "runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f4200_a35.obj",
            "runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f5200_a35.obj",
            "runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f6000_a45.obj",
            "runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f7000_a45.obj",
            "runs/adda_ocl_benchmark_ext/shapes/greek_scaled_ka15_dpl20.shape",
        ]
        extra_needed = [
            "runs/production_matrix_15/meshes/dust5_adda_shape/adda_cubical_raw.obj",
            "runs/production_matrix_15/meshes/dust5_adda_shape/adda_cubical_f6000_ag6.obj",
            "runs/production_matrix_15/meshes/dust5_adda_shape/adda_mc_s0p35_l0p42_f6000.obj",
            "runs/production_matrix_15/meshes/dust5_adda_shape/adda_mc_s0p5_l0p42_f6000.obj",
        ]
        for rel in needed:
            path = repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x\n")
        fakebin = tmp_path / "fakebin"
        fakebin.mkdir()
        (fakebin / "nvidia-smi").write_text(
            "#!/usr/bin/env bash\n"
            "if [[ \"$*\" == *\"--query-gpu=index\"* ]]; then printf '0\\n1\\n2\\n3\\n'; exit 0; fi\n"
            "if [[ \"$*\" == *\"-i 0\"* && \"$*\" == *\"temperature.gpu\"* ]]; then printf '45, 0, 10\\n'; exit 0; fi\n"
            "if [[ \"$*\" == *\"-i 1\"* && \"$*\" == *\"temperature.gpu\"* ]]; then printf '90, 0, 10\\n'; exit 0; fi\n"
            "if [[ \"$*\" == *\"temperature.gpu\"* ]]; then printf '45, 0, 10\\n'; exit 0; fi\n"
            "if [[ \"$*\" == *\"--query-compute-apps\"* ]]; then exit 0; fi\n"
            "exit 0\n"
        )
        (fakebin / "nvidia-smi").chmod(0o755)
        script = f"""
set -euo pipefail
source "{QUEUE}"
PATH="{fakebin}:$PATH"
gpu_health_check 0 > "{tmp_path}/gpu0.txt"
! gpu_health_check 1 > "{tmp_path}/gpu1.txt" 2>&1
grep -q '^GPU_HEALTH ok gpu=0 temp=45C util=0% mem=10MiB$' "{tmp_path}/gpu0.txt"
grep -q 'GPU_HEALTH fail gpu=1 temp=90C' "{tmp_path}/gpu1.txt"
        BEM_QUEUE_MAX_TEMP_C=95 PATH="{fakebin}:$PATH" REPO="{repo}" OUT="runs/production_matrix_15" BIN="{repo}/bin/bem_cuda_fmm" "{QUEUE}" --preflight > "{tmp_path}/preflight.txt"
grep -q '^PREFLIGHT gpu_count=4$' "{tmp_path}/preflight.txt"
        grep -q '^PREFLIGHT ok$' "{tmp_path}/preflight.txt"
grep -q '^SUMMARY current=0 stale=0 missing=15 total=15$' "{tmp_path}/preflight.txt"
BEM_QUEUE_EXTRA_DUST_VARIANTS=1 BEM_QUEUE_MAX_TEMP_C=95 PATH="{fakebin}:$PATH" REPO="{repo}" OUT="runs/production_matrix_15" BIN="{repo}/bin/bem_cuda_fmm" "{QUEUE}" --preflight > "{tmp_path}/preflight_extra_missing.txt" 2>&1 && exit 1 || rc=$?
test "$rc" = "2"
grep -q 'PREFLIGHT missing file: runs/production_matrix_15/meshes/dust5_adda_shape/adda_cubical_raw.obj' "{tmp_path}/preflight_extra_missing.txt"
"""
        proc = run_bash(script)
        assert proc.returncode == 0, proc.stdout

        for rel in extra_needed:
            path = repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x\n")
        script = f"""
set -euo pipefail
BEM_QUEUE_EXTRA_DUST_VARIANTS=1 BEM_QUEUE_MAX_TEMP_C=95 PATH="{fakebin}:$PATH" REPO="{repo}" OUT="runs/production_matrix_15" BIN="{repo}/bin/bem_cuda_fmm" "{QUEUE}" --preflight > "{tmp_path}/preflight_extra.txt"
grep -q '^PREFLIGHT ok$' "{tmp_path}/preflight_extra.txt"
"""
        proc = run_bash(script)
        assert proc.returncode == 0, proc.stdout

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fakebin = tmp_path / "fakebin"
        fakebin.mkdir()
        (fakebin / "nvidia-smi").write_text(
            "#!/usr/bin/env bash\n"
            "if [[ \"$*\" == *\"--query-compute-apps\"* ]]; then printf '4242, ./mbs_po_gpu_float_fast, 304\\n'; exit 0; fi\n"
            "if [[ \"$*\" == *\"temperature.gpu,utilization.gpu,memory.used\"* ]]; then printf '45, 0, 10\\n'; exit 0; fi\n"
            "exit 0\n"
        )
        (fakebin / "nvidia-smi").chmod(0o755)
        script = f"""
set -euo pipefail
source "{QUEUE}"
PATH="{fakebin}:$PATH"
! gpu_health_check 0 > "{tmp_path}/busy.txt" 2>&1
grep -q '^GPU_HEALTH fail gpu=0 compute_apps=4242, ./mbs_po_gpu_float_fast, 304$' "{tmp_path}/busy.txt"
BEM_QUEUE_ALLOW_COMPUTE_SHARE=1 gpu_health_check 0 > "{tmp_path}/shared.txt"
grep -q '^GPU_HEALTH ok gpu=0 temp=45C util=0% mem=10MiB$' "{tmp_path}/shared.txt"
"""
        proc = run_bash(script)
        assert proc.returncode == 0, proc.stdout

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = tmp_path / "out"
        (out / "logs").mkdir(parents=True)
        fakebin = tmp_path / "fakebin"
        fakebin.mkdir()
        (fakebin / "nvidia-smi").write_text(
            "#!/usr/bin/env bash\n"
            "if [[ \"$*\" == *\"-i 0\"* && \"$*\" == *\"power.draw\"* ]]; then printf '83, 99, 123, 210.42\\n'; exit 0; fi\n"
            "exit 0\n"
        )
        (fakebin / "nvidia-smi").chmod(0o755)
        (fakebin / "stdbuf").write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' \"$*\" >> \"{}/stdbuf_args.txt\"\n"
            "while [[ \"$1\" == '-oL' || \"$1\" == '-eL' ]]; do shift; done\n"
            "exec \"$@\"\n".format(tmp_path)
        )
        (fakebin / "stdbuf").chmod(0o755)
        script = f"""
set -euo pipefail
source "{QUEUE}"
PATH="{fakebin}:$PATH"
out="{out}"
sample="$(gpu_runtime_sample 0)"
test "$sample" = "83,99,123,210"
BEM_QUEUE_MONITOR_INTERVAL_S=1 \\
  run_with_gpu_monitor 0 hotcase bash -c 'sleep 2' > "{out}/logs/hotcase.log" 2>&1
grep -q '^timestamp_s,gpu,temp_c,util_pct,mem_mib,power_w$' "{out}/logs/hotcase.gpu.csv"
grep -q -- '^-oL -eL bash -c sleep 2$' "{tmp_path}/stdbuf_args.txt"
grep -q '^QUEUE_STDOUT line_buffer=stdbuf$' "{out}/logs/hotcase.log"
! grep -q 'stop name=hotcase gpu=0' "{out}/logs/hotcase.log"
BEM_QUEUE_STDBUF=0 BEM_QUEUE_MONITOR_INTERVAL_S=1 \\
  run_with_gpu_monitor 0 hotcase_unbuffered bash -c 'sleep 1' > "{out}/logs/hotcase_unbuffered.log" 2>&1
grep -q '^QUEUE_STDOUT line_buffer=default$' "{out}/logs/hotcase_unbuffered.log"
test "$(wc -l < "{tmp_path}/stdbuf_args.txt")" = "1"
BEM_QUEUE_STDBUF=0 BEM_QUEUE_MONITOR_INTERVAL_S=1 \\
  run_with_gpu_monitor 0 hotcase_fail bash -c 'exit 7' > "{out}/logs/hotcase_fail.log" 2>&1 && exit 1 || rc=$?
test "$rc" = "7"
grep -q '^QUEUE_STDOUT line_buffer=default$' "{out}/logs/hotcase_fail.log"
"""
        proc = run_bash(script)
        assert proc.returncode == 0, proc.stdout

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = tmp_path / "out"
        (out / "logs").mkdir(parents=True)
        fakebin = tmp_path / "fakebin"
        fakebin.mkdir()
        (fakebin / "nvidia-smi").write_text(
            "#!/usr/bin/env bash\n"
            "if [[ \"$*\" == *\"-i 0\"* && \"$*\" == *\"power.draw\"* ]]; then printf '45, 99, 123, 260.50\\n'; exit 0; fi\n"
            "exit 0\n"
        )
        (fakebin / "nvidia-smi").chmod(0o755)
        script = f"""
set -euo pipefail
source "{QUEUE}"
PATH="{fakebin}:$PATH"
out="{out}"
BEM_QUEUE_MONITOR_INTERVAL_S=1 \\
  run_with_gpu_monitor 0 power_ok bash -c 'sleep 2'
! grep -q 'stop name=power_ok' "{out}/logs/power_ok.log"
BEM_QUEUE_MONITOR_INTERVAL_S=1 \\
  run_with_gpu_monitor 0 power_high bash -c 'sleep 2'
! grep -q 'stop name=power_high' "{out}/logs/power_high.log"
"""
        proc = run_bash(script)
        assert proc.returncode == 0, proc.stdout

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "scripts").mkdir()
        (repo / "scripts" / "check_result_metadata.py").write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "sys.exit(0)\n"
        )
        bin_path = repo / "solver_stub.sh"
        bin_path.write_text(
            "#!/usr/bin/env bash\n"
            "out=''\n"
            "while [[ $# -gt 0 ]]; do\n"
            "  if [[ \"$1\" == '--out' ]]; then out=\"$2\"; shift 2; else shift; fi\n"
            "done\n"
            "printf '%s\\n' \"$BEM_GMRES_VERBOSE\" >> verbose_seen.txt\n"
            "printf '{\"theta\":[0],\"mueller\":[[1]],\"method\":{},\"mesh\":{}}\\n' > \"$out\"\n"
        )
        bin_path.chmod(0o755)
        fail_bin_path = repo / "solver_fail_stub.sh"
        fail_bin_path.write_text(
            "#!/usr/bin/env bash\n"
            "exit 9\n"
        )
        fail_bin_path.chmod(0o755)
        fakebin = tmp_path / "fakebin"
        fakebin.mkdir()
        (fakebin / "nvidia-smi").write_text(
            "#!/usr/bin/env bash\n"
            "if [[ \"$*\" == *\"temperature.gpu,utilization.gpu,memory.used\"* ]]; then printf '45, 0, 10\\n'; exit 0; fi\n"
            "if [[ \"$*\" == *\"power.draw\"* ]]; then printf '45, 0, 10, 100\\n'; exit 0; fi\n"
            "exit 0\n"
        )
        (fakebin / "nvidia-smi").chmod(0o755)
        script = f"""
set -euo pipefail
source "{QUEUE}"
cd "{repo}"
PATH="{fakebin}:$PATH"
out="{tmp_path}/out"
bin="{bin_path}"
mkdir -p "$out/logs"
BEM_QUEUE_MONITOR_INTERVAL_S=1 run_case 0 verbose_default --ka 1
BEM_QUEUE_GMRES_VERBOSE=0 BEM_QUEUE_MONITOR_INTERVAL_S=1 run_case 0 verbose_off --ka 1
test "$(sed -n '1p' verbose_seen.txt)" = "1"
test "$(sed -n '2p' verbose_seen.txt)" = "0"
bin="{fail_bin_path}"
BEM_QUEUE_MONITOR_INTERVAL_S=1 run_case 0 failing_case --ka 1 && exit 1 || rc=$?
test "$rc" = "9"
grep -q '^FAIL failing_case rc=9$' "$out/logs/failing_case.log"
"""
        proc = run_bash(script)
        assert proc.returncode == 0, proc.stdout

    text = QUEUE.read_text()
    assert "BEM_QUEUE_EXTRA_DUST_VARIANTS" in text
    assert "BEM_QUEUE_GMRES_VERBOSE" in text
    assert "--accurate --system balanced --quad 7 --fmm-digits 6 --gmres-tol 5e-4 --gmres-restart 500 --max-leaf 128" in text
    assert "common_dust=(--ri 1.6 0.002 --single --ntheta 181 --solver fmm --system balanced --quad 7 --fmm-digits 5" not in text
    expected_names = [
        "sphere_ka5_ref4_current",
        "sphere_ka10_ref4_current",
        "sphere_ka15_ref4_current",
        "sphere_ka20_ref4_current",
        "sphere_ka30_ref6_current",
        "hex_ka5_ref2_balanced",
        "hex_ka10_ref3_balanced",
        "hex_ka15_ref4_balanced",
        "hex_ka20_ref4_balanced",
        "hex_ka30_ref5_balanced",
        "dust_ka5_gmsh3400_balanced_q7_d6_tol5e4",
        "dust_ka10_gmsh5200_balanced_q7_d6_tol5e4",
        "dust_ka15_gmsh6000_balanced_q7_d6_tol5e4",
        "dust_ka20_gmsh4200_balanced_q7_d6_tol5e4",
        "dust_ka30_gmsh7000_balanced_q7_d6_tol5e4",
    ]
    missing = [name for name in expected_names if name not in text]
    assert not missing, missing

    print("accuracy queue metadata skip: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
