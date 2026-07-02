#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m py_compile \
  bemcuda/__init__.py \
  bemcuda/gpu_guard.py \
  bemcuda/job.py \
  verify_mie.py \
  scripts/mueller_audit.py \
  scripts/operator_block_audit.py \
  scripts/cpu_pmchwt_centroid_reference.py \
  scripts/queue_status_json.py \
  scripts/remote_refinement_queue_status.py \
  scripts/audit_1_6.py \
  scripts/audit_accuracy_matrix_15.py \
  scripts/plan_accuracy_refinement_cases.py \
  scripts/check_audit_1_6_report.py \
  scripts/check_result_metadata.py \
  scripts/run_hex_adda_compare.py \
  scripts/run_hex_euler_scaling_benchmark.py \
  scripts/run_obj_adda_compare.py \
  scripts/summarize_audit_1_6.py \
  scripts/summarize_sphere_ri_sweep.py \
  scripts/reference_dense_check.py \
  scripts/near_singular_audit.py \
  scripts/hmatrix_memory_audit.py \
  scripts/detect_cuda_toolchain.py \
  scripts/summarize_gpu_power_monitor.py \
  tests/test_bem_job_api.py \
  tests/test_audit_1_6_exit_codes.py \
  tests/test_audit_1_6_report_contract.py \
  tests/test_audit_accuracy_matrix_selection.py \
  tests/test_verify_mie_mueller.py \
  tests/test_mie_mueller_symmetry.py \
  tests/test_mueller_audit_physical.py \
  tests/test_summarize_sphere_ri_sweep.py \
  tests/test_poster_index_sweep_full_mueller.py \
  tests/test_validate_poster_index_sweep.py \
  tests/test_validate_poster_vram_accuracy.py \
  tests/test_validate_poster_truth_flags.py \
  tests/test_watch_sphere_ri_results.py \
  tests/test_time_accuracy_joint_truthy.py \
  tests/test_result_metadata_check.py \
  tests/test_queue_status_json.py \
  tests/test_accuracy_queue_metadata_skip.py \
  tests/test_plan_accuracy_refinement_cases.py \
  tests/test_run_accuracy_refinement_wave.py \
  tests/test_remote_accuracy_refinement_wave.py \
  tests/test_gpu_compute_entrypoint_guards.py \
  tests/test_rhs_workspace_policy.py \
  tests/test_farfield_workspace_policy.py \
  tests/test_single_gpu_queue_policy.py \
  tests/test_rhs_header_compile.py \
  tests/test_main_syntax_smoke.py \
  tests/test_run_accuracy_matrix_case_lock.py \
  tests/test_resume_accuracy_matrix_cases.py \
  tests/test_remote_resume_accuracy_matrix_cases.py \
  tests/test_summarize_audit_1_6.py \
  tests/test_cuda_toolchain_detection.py \
  tests/test_near_singular_audit.py \
  tests/test_cpu_pmchwt_centroid_reference.py \
  tests/test_hmatrix_memory_audit.py

bash -n \
  scripts/deploy_accuracy_matrix_15_queue.sh \
  scripts/deploy_hex_euler_scaling_256.sh \
  scripts/fetch_accuracy_matrix_15_results.sh \
  scripts/ipmi_power_control.sh \
  scripts/queue_live_status.sh \
  scripts/queue_watch_once.sh \
  scripts/remote_power_watch.sh \
  scripts/run_remote_refinement_queue_supervisor.sh \
  scripts/start_remote_refinement_queue_supervisor.sh \
  scripts/remote_resume_accuracy_matrix_cases.sh \
  scripts/remote_accuracy_refinement_wave.sh \
  scripts/resume_accuracy_matrix_cases.sh \
  scripts/run_accuracy_refinement_wave.sh \
  scripts/run_accuracy_matrix_case.sh \
  scripts/gpu_guard.sh \
  scripts/run_guarded_bem_case.sh \
  scripts/run_complex_operator_dust_refresh.sh \
  scripts/run_fig7_memory_queue.sh \
  scripts/run_adda_ocl_benchmark.sh \
  scripts/run_adda_ocl_sphere_ri_sweep.sh \
  scripts/run_greek_orientation_convergence_queue.sh \
  scripts/run_sphere_ri_missing_fallback_queue.sh \
  scripts/run_sphere30_ref6_candidates.sh \
  scripts/recompute_convergence_meta.sh \
  scripts/resume_accuracy_matrix_15_after_current.sh \
  scripts/run_accuracy_matrix_15_queue.sh \
  scripts/run_local_audits.sh \
  scripts/set_remote_gpu_power_limit.sh \
  scripts/supermicro_bmc_access.sh \
  scripts/watch_sphere_ri_results.sh

POWER_WATCH_ONCE=1 scripts/remote_power_watch.sh --once > /tmp/bemcuda_power_watch_once.txt
grep -q -- '--- gpu ---' /tmp/bemcuda_power_watch_once.txt
grep -q -- '--- ipmi sel tail ---' /tmp/bemcuda_power_watch_once.txt

python3 tests/test_bem_job_api.py
python3 tests/test_audit_1_6_exit_codes.py
python3 tests/test_audit_1_6_report_contract.py
python3 tests/test_audit_accuracy_matrix_selection.py
python3 tests/test_verify_mie_mueller.py
python3 tests/test_mie_mueller_symmetry.py
python3 tests/test_mueller_audit_physical.py
python3 tests/test_summarize_sphere_ri_sweep.py
python3 tests/test_poster_index_sweep_full_mueller.py
python3 tests/test_validate_poster_index_sweep.py
python3 tests/test_validate_poster_vram_accuracy.py
python3 tests/test_validate_poster_truth_flags.py
python3 tests/test_watch_sphere_ri_results.py
python3 tests/test_time_accuracy_joint_truthy.py
python3 tests/test_poster_current_result_flags.py
python3 tests/test_result_metadata_check.py
python3 tests/test_queue_status_json.py
python3 tests/test_accuracy_queue_metadata_skip.py
python3 tests/test_plan_accuracy_refinement_cases.py
python3 tests/test_run_accuracy_refinement_wave.py
python3 tests/test_remote_accuracy_refinement_wave.py
python3 tests/test_fetch_accuracy_matrix_results.py
python3 tests/test_gpu_compute_entrypoint_guards.py
python3 tests/test_rhs_workspace_policy.py
python3 tests/test_farfield_workspace_policy.py
python3 tests/test_single_gpu_queue_policy.py
python3 tests/test_rhs_header_compile.py
python3 tests/test_main_syntax_smoke.py
python3 tests/test_run_accuracy_matrix_case_lock.py
python3 tests/test_poster_dust_source_policy.py
python3 tests/test_obj_entrypoint_defaults.py
python3 tests/test_resume_accuracy_matrix_cases.py
python3 tests/test_remote_resume_accuracy_matrix_cases.py
python3 tests/test_summarize_audit_1_6.py
python3 tests/test_cuda_toolchain_detection.py
grep -q 'run_accuracy_matrix_15_queue.next.sh' scripts/resume_accuracy_matrix_15_after_current.sh
grep -q 'run_accuracy_matrix_case.sh' scripts/resume_accuracy_matrix_15_after_current.sh
grep -q 'remote_resume_accuracy_matrix_cases.sh' scripts/resume_accuracy_matrix_15_after_current.sh
grep -q 'check_result_metadata.py' scripts/resume_accuracy_matrix_15_after_current.sh
grep -q -- 'source .*gpu_guard.sh' scripts/run_greek_orientation_convergence_queue.sh
grep -q -- 'source .*gpu_guard.sh' scripts/run_sphere_ri_missing_fallback_queue.sh
grep -q -- '--query-compute-apps=pid,process_name,used_memory' scripts/gpu_guard.sh
grep -q -- 'source .*gpu_guard.sh' scripts/run_fig7_memory_queue.sh
grep -q -- 'source .*gpu_guard.sh' scripts/run_complex_operator_dust_refresh.sh
grep -q -- 'source .*gpu_guard.sh' scripts/run_sphere_ri_sweep.sh
grep -q -- '--query-compute-apps=pid,process_name,used_memory' bemcuda/gpu_guard.py
grep -q -- 'from bemcuda.gpu_guard import assert_gpus_free' scripts/run_hex_adda_compare.py
grep -q -- 'from bemcuda.gpu_guard import assert_gpus_free' scripts/run_obj_adda_compare.py
grep -q -- 'source .*gpu_guard.sh' scripts/run_sphere30_ref6_candidates.sh
grep -q -- 'source .*gpu_guard.sh' scripts/recompute_convergence_meta.sh
grep -q -- 'from bemcuda.gpu_guard import parse_gpu_csv, select_free_gpus' scripts/run_hex_euler_scaling_benchmark.py
grep -q -- 'source .*gpu_guard.sh' scripts/run_adda_ocl_benchmark.sh
grep -q -- 'source .*gpu_guard.sh' scripts/run_adda_ocl_sphere_ri_sweep.sh
grep -q -- '--gpus '\''${GPUS}'\''' scripts/deploy_hex_euler_scaling_256.sh
grep -q -- 'scp bemcuda/\*.py' scripts/deploy_hex_euler_scaling_256.sh
grep -q -- 'python3 scripts/run_hex_euler_scaling_benchmark.py' scripts/deploy_hex_euler_scaling_256.sh
grep -q -- 'scripts/gpu_guard.sh' scripts/deploy_accuracy_matrix_15_queue.sh
grep -q -- 'scripts/detect_cuda_toolchain.py' scripts/deploy_accuracy_matrix_15_queue.sh
grep -q -- 'scripts/detect_cuda_toolchain.py' scripts/remote_resume_accuracy_matrix_cases.sh
grep -q -- 'scripts/remote_refinement_queue_status.py' scripts/deploy_accuracy_matrix_15_queue.sh
grep -q -- 'scripts/run_remote_refinement_queue_supervisor.sh' scripts/deploy_accuracy_matrix_15_queue.sh
grep -q -- 'scripts/start_remote_refinement_queue_supervisor.sh' scripts/deploy_accuracy_matrix_15_queue.sh
grep -q -- 'scripts/remote_refinement_queue_status.py' scripts/remote_resume_accuracy_matrix_cases.sh
grep -q -- 'scripts/run_remote_refinement_queue_supervisor.sh' scripts/remote_resume_accuracy_matrix_cases.sh
grep -q -- 'scripts/start_remote_refinement_queue_supervisor.sh' scripts/remote_resume_accuracy_matrix_cases.sh
grep -q -- 'scripts/gpu_guard.sh' scripts/resume_accuracy_matrix_15_after_current.sh
printf '# smoke reference\n' >/tmp/bemcuda_mbs_smoke.dat
BEM_NVIDIA_SMI=custom-smi python3 scripts/run_bem_candidate.py \
  --dry-run --wait-gpu-free --cuda-devices 0 \
  --mbs /tmp/bemcuda_mbs_smoke.dat --obj dummy.obj --out /tmp/bemcuda_candidate_smoke.json \
  >/tmp/bemcuda_candidate_dryrun.txt 2>/tmp/bemcuda_candidate_dryrun.err || {
    cat /tmp/bemcuda_candidate_dryrun.err >&2
    exit 1
  }
grep -q 'custom-smi --query-gpu=index,memory.used,utilization.gpu' /tmp/bemcuda_candidate_dryrun.txt
grep -q 'custom-smi -i "\$gpu" --query-compute-apps=pid,process_name,used_memory' /tmp/bemcuda_candidate_dryrun.txt
grep -q -- '--accurate' /tmp/bemcuda_candidate_dryrun.txt
grep -q -- '--quad 7' /tmp/bemcuda_candidate_dryrun.txt
grep -q -- '--fmm-digits 6' /tmp/bemcuda_candidate_dryrun.txt
grep -q -- '--gmres-tol 5e-4' /tmp/bemcuda_candidate_dryrun.txt
! grep -q -- '--system pmchwt' /tmp/bemcuda_candidate_dryrun.txt
scripts/run_guarded_bem_case.sh --gpu 0 --name missing_bin --out-dir /tmp/bemcuda_missing_bin --bin /tmp/bemcuda_no_such_binary -- --ka 1 >/tmp/bemcuda_missing_bin.out 2>/tmp/bemcuda_missing_bin.err && {
  echo "run_guarded_bem_case unexpectedly accepted missing binary" >&2
  exit 1
}
case "$?" in
  6) ;;
  *) echo "run_guarded_bem_case missing binary returned unexpected code" >&2; exit 1 ;;
esac
grep -q 'BEM executable is missing or not executable' /tmp/bemcuda_missing_bin.err
mkdir -p /tmp/bemcuda_guard_smi
cat > /tmp/bemcuda_guard_smi/fake_solver.sh <<'EOF'
#!/usr/bin/env bash
out=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--out" ]]; then
    out="$2"
    shift 2
  else
    shift
  fi
done
sleep 1
cat > "$out" <<'JSON'
{
  "theta": [0],
  "mueller": [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
  "ka": 1,
  "refinements": 0,
  "shape": "sphere",
  "obj_file": null,
  "prism_aspect": 1.0,
  "edge_refine": 0,
  "gmres_nonconverged_systems": 0,
  "gmres_stagnation_stops": 0,
  "gmres_max_final_relres": 0.0009,
  "gmres_tol": 0.001,
  "method": {
    "solver_backend": "FMM",
    "solver_profile": "default",
    "requested_system": "balanced",
    "system": "balanced",
    "system_canonicalized": false,
    "quad_order": 4,
    "preconditioner_enabled": false,
    "schwarz_preconditioner": false,
    "preconditioner_reason": "user_disabled",
    "farfield_mode": "test"
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
JSON
EOF
chmod +x /tmp/bemcuda_guard_smi/fake_solver.sh
cat > /tmp/bemcuda_guard_smi/custom-smi <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> /tmp/bemcuda_guard_smi/custom_smi_calls.txt
if [[ "$*" == *"--query-compute-apps"* ]]; then
  exit 0
fi
printf '45, 10, 123, 180.5\n'
EOF
chmod +x /tmp/bemcuda_guard_smi/custom-smi
cat > /tmp/bemcuda_guard_smi/busy-smi <<'EOF'
#!/usr/bin/env bash
if [[ "$*" == *"--query-compute-apps"* ]]; then
  printf '4242, ./mbs_po_gpu_float_fast, 304\n'
  exit 0
fi
printf '45, 10, 123, 180.5\n'
EOF
chmod +x /tmp/bemcuda_guard_smi/busy-smi
rm -rf /tmp/bemcuda_guard_smi/busy_out
BEM_NVIDIA_SMI=/tmp/bemcuda_guard_smi/busy-smi \
  scripts/run_guarded_bem_case.sh --gpu 0 --name guarded_busy \
  --out-dir /tmp/bemcuda_guard_smi/busy_out --bin /tmp/bemcuda_guard_smi/fake_solver.sh \
  --interval 1 --max-power 260 --max-temp 78 -- --ka 1 \
  >/tmp/bemcuda_guard_smi/busy.out 2>/tmp/bemcuda_guard_smi/busy.err && {
    echo "run_guarded_bem_case unexpectedly accepted busy compute GPU" >&2
    exit 1
  }
case "$?" in
  3) ;;
  *) echo "run_guarded_bem_case busy preflight returned unexpected code" >&2; exit 1 ;;
esac
grep -q 'GPU_BUSY gpu=0 compute_apps=4242, ./mbs_po_gpu_float_fast, 304' /tmp/bemcuda_guard_smi/busy.err
test ! -e /tmp/bemcuda_guard_smi/busy_out/logs/guarded_busy.log
rm -rf /tmp/bemcuda_guard_smi/out
BEM_NVIDIA_SMI=/tmp/bemcuda_guard_smi/custom-smi \
  scripts/run_guarded_bem_case.sh --gpu 0 --name guarded_custom_smi \
  --out-dir /tmp/bemcuda_guard_smi/out --bin /tmp/bemcuda_guard_smi/fake_solver.sh \
  --interval 1 --max-power 260 --max-temp 78 -- --ka 1
grep -q -- '-i 0 --query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw' /tmp/bemcuda_guard_smi/custom_smi_calls.txt
grep -q '^.*,0,45,10,123,180.5,0$' /tmp/bemcuda_guard_smi/out/logs/guarded_custom_smi.gpu.csv
grep -q '^DONE guarded_custom_smi rc=0$' /tmp/bemcuda_guard_smi/out/logs/guarded_custom_smi.log
python3 scripts/mueller_audit.py --self-test
python3 scripts/operator_block_audit.py --self-test
python3 scripts/cpu_pmchwt_centroid_reference.py --json-out /tmp/bemcuda_cpu_pmchwt_centroid.json
python3 tests/test_cpu_pmchwt_centroid_reference.py
python3 tests/test_near_singular_audit.py
python3 tests/test_hmatrix_memory_audit.py
python3 scripts/audit_1_6.py --out /tmp/bemcuda_audit_1_6.json
python3 scripts/check_audit_1_6_report.py /tmp/bemcuda_audit_1_6.json
python3 - <<'PY'
from pathlib import Path
Path("/tmp/bemcuda_refine.csv").write_text(
    "shape,ka,mesh_label,status,metadata_status,operator_status,gate_error,raw_pass10,pass10\n"
    "сфера,30,ref6,FAIL,ok,not_required,0.2,false,false\n"
    "гексагональная призма,20,ref4,PASS,ok,not_required,0.03,true,true\n"
)
PY
python3 scripts/plan_accuracy_refinement_cases.py --csv /tmp/bemcuda_refine.csv --plan-csv /tmp/bemcuda_refine_plan.csv --no-health-check --gpus "0 1" > /tmp/bemcuda_refine_plan.txt
grep -q '^sphere_ka30_ref7_current_q13_d7_tol1e3$' /tmp/bemcuda_refine_plan.txt
grep -q '^REFINE threshold=0.1 reason=all planned=1 limit=2$' /tmp/bemcuda_refine_plan.txt
grep -q '^sphere_ka30_ref7_current_q13_d7_tol1e3,accuracy,' /tmp/bemcuda_refine_plan.csv
python3 - <<'PY'
from pathlib import Path
Path("/tmp/bemcuda_refine_legacy.csv").write_text(
    "shape,ka,mesh_label,status,metadata_status,operator_status,gate_error,raw_pass10,pass10\n"
    "сфера,30,ref6,PASS,legacy,not_required,0.02,true,false\n"
)
PY
python3 scripts/plan_accuracy_refinement_cases.py --csv /tmp/bemcuda_refine_legacy.csv --plan-csv /tmp/bemcuda_refine_legacy_plan.csv --no-health-check --gpus "0 1" > /tmp/bemcuda_refine_legacy_plan.txt
grep -q '^sphere_ka30_ref6_current_q7_d6_tol3e3$' /tmp/bemcuda_refine_legacy_plan.txt
! grep -q '^sphere_ka30_ref7_current_q13_d7_tol1e3$' /tmp/bemcuda_refine_legacy_plan.txt
grep -q '^sphere_ka30_ref6_current_q7_d6_tol3e3,metadata,' /tmp/bemcuda_refine_legacy_plan.csv
python3 - <<'PY'
from pathlib import Path
Path("/tmp/bemcuda_refine_limit.csv").write_text(
    "shape,ka,mesh_label,status,metadata_status,operator_status,gate_error,raw_pass10,pass10\n"
    "сфера,5,ref4,PASS,legacy,not_required,0.01,true,false\n"
    "сфера,10,ref4,PASS,legacy,not_required,0.02,true,false\n"
    "сфера,15,ref4,PASS,legacy,not_required,0.03,true,false\n"
)
PY
python3 scripts/plan_accuracy_refinement_cases.py --csv /tmp/bemcuda_refine_limit.csv --no-health-check --gpus "0 1" > /tmp/bemcuda_refine_limit_plan.txt
grep -q '^REFINE threshold=0.1 reason=all planned=2 limit=2$' /tmp/bemcuda_refine_limit_plan.txt
! grep -q '^sphere_ka5_ref4_current_q7_d6_tol3e3$' /tmp/bemcuda_refine_limit_plan.txt
grep -q '^sphere_ka15_ref4_current_q7_d6_tol3e3$' /tmp/bemcuda_refine_limit_plan.txt
grep -q '^sphere_ka10_ref4_current_q7_d6_tol3e3$' /tmp/bemcuda_refine_limit_plan.txt
python3 scripts/plan_accuracy_refinement_cases.py --csv /tmp/bemcuda_refine_limit.csv --all-cases --no-health-check --gpus "0 1" > /tmp/bemcuda_refine_all_plan.txt
grep -q '^REFINE threshold=0.1 reason=all planned=3 limit=all$' /tmp/bemcuda_refine_all_plan.txt
grep -q '^sphere_ka5_ref4_current_q7_d6_tol3e3$' /tmp/bemcuda_refine_all_plan.txt
python3 scripts/plan_accuracy_refinement_cases.py --csv /tmp/bemcuda_refine_limit.csv --only-reason metadata --all-cases --no-health-check --gpus "0 1" > /tmp/bemcuda_refine_metadata_plan.txt
grep -q '^REFINE threshold=0.1 reason=metadata planned=3 limit=all$' /tmp/bemcuda_refine_metadata_plan.txt
grep -q '^sphere_ka5_ref4_current_q7_d6_tol3e3$' /tmp/bemcuda_refine_metadata_plan.txt
python3 scripts/detect_cuda_toolchain.py --json-out /tmp/bemcuda_cuda_toolchain.json || true
python3 - <<'PY'
from pathlib import Path
root = Path("/tmp/bemcuda_gpu_monitor")
root.mkdir(exist_ok=True)
(root / "case.gpu.csv").write_text(
    "timestamp_s,gpu,temp_c,util_pct,mem_mib,power_w\n"
    "10,0,40,50,1000,120\n"
    "20,0,50,100,2000,220\n"
)
PY
python3 scripts/summarize_gpu_power_monitor.py /tmp/bemcuda_gpu_monitor > /tmp/bemcuda_gpu_monitor_summary.csv
grep -q 'case,2,0,10,170.0' /tmp/bemcuda_gpu_monitor_summary.csv
rm -rf /tmp/bemcuda_queue_status
mkdir -p /tmp/bemcuda_queue_status/logs
python3 - <<'PY'
import json
from pathlib import Path
Path("/tmp/bemcuda_queue_status").mkdir(parents=True, exist_ok=True)
payload = {
    "theta": [0, 1],
    "mueller": [
        [1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0],
    ],
    "ka": 5,
    "refinements": 4,
    "shape": "sphere",
    "obj_file": None,
    "prism_aspect": 1.0,
    "edge_refine": 0,
    "timing": {"total_s": 100.0},
    "gmres_nonconverged_systems": 0,
    "gmres_stagnation_stops": 0,
    "gmres_max_final_relres": 0.0009,
    "gmres_tol": 0.001,
    "fmm_digits": 6,
    "mesh": {
        "vertices": 1,
        "triangles": 1,
        "rwg_edges": 1,
        "skinny_triangles": 0,
        "min_angle_deg": 60.0,
        "max_aspect_ratio": 1.0,
        "feature_edges_30deg": 0,
        "max_dihedral_deg": 20.0,
        "mean_feature_dihedral_deg": 0.0,
        "max_adjacent_area_ratio": 1.0,
        "near_touch_checked": True,
        "near_touch_ratio": 1.0,
        "near_touch_pairs": 0,
        "self_panel_count": 1,
        "edge_adjacent_pair_count": 0,
        "vertex_adjacent_pair_count": 0,
        "near_disjoint_pair_count": 0,
        "taylor_duffy_candidate_count": 1,
        "recommended_min_quad_order": 4,
        "recommended_mesh_strategy": "uniform_curvature_refinement",
        "recommended_mesh_action": "uniform smooth-surface refinement is acceptable",
        "requires_remesh": False,
        "edge_refine_requested": 0,
        "edge_refine_applied": 0,
        "edge_refine_uniform_fallback": False,
        "quality_gate_pass": True,
    },
    "method": {
        "solver": "test",
        "solver_backend": "test",
        "solver_profile": "test",
        "requested_system": "pmchwt",
        "system": "pmchwt",
        "system_canonicalized": False,
        "quad_order": 4,
        "preconditioner_enabled": False,
        "schwarz_preconditioner": False,
        "preconditioner_reason": "user_disabled",
        "farfield_mode": "test",
    },
}
Path("/tmp/bemcuda_queue_status/sphere_ka5_ref4_current_q7_d6_tol3e3.json").write_text(json.dumps(payload))
PY
cat > /tmp/bemcuda_queue_status/logs/case.gpu.csv <<'EOF'
timestamp_s,gpu,temp_c,util_pct,mem_mib,power_w
10,0,40,50,1000,120
20,0,50,100,2000,220
EOF
cat > /tmp/bemcuda_queue_status/logs/case.log <<'EOF'
  [GMRES-paired] start: res1=1.00e+00 res2=1.00e+00
    GMRES iter 17: rel1=4.00e-02 rel2=6.00e-02
    GMRES iter 18: rel1=3.00e-02 rel2=5.00e-02
EOF
OUT=/tmp/bemcuda_queue_status scripts/queue_live_status.sh > /tmp/bemcuda_queue_status.txt
grep -q 'gmres_last=GMRES iter 18: rel1=3.00e-02 rel2=5.00e-02' /tmp/bemcuda_queue_status.txt
grep -q 'gpu_summary=gpu=0 duration_s=10 power_mean_w=170.0 power_max_w=220 util_mean_pct=75.0 temp_max_c=50 mem_max_mib=2000' /tmp/bemcuda_queue_status.txt
grep -q 'monitor_state=active' /tmp/bemcuda_queue_status.txt
grep -q '^case age_s=.* samples=2 gpu=0 duration_s=10 power_mean_w=170.0' /tmp/bemcuda_queue_status.txt
cat > /tmp/bemcuda_queue_status/logs/case.log <<'EOF'
  [GMRES] verbose residual logging enabled
EOF
OUT=/tmp/bemcuda_queue_status scripts/queue_live_status.sh > /tmp/bemcuda_queue_status_verbose.txt
grep -q 'gmres_last=verbose-enabled-no-iteration-yet' /tmp/bemcuda_queue_status_verbose.txt
cat > /tmp/bemcuda_queue_status/logs/case.log <<'EOF'
FAIL case rc=9
EOF
OUT=/tmp/bemcuda_queue_status scripts/queue_live_status.sh > /tmp/bemcuda_queue_status_fail.txt
grep -q 'case_failed=FAIL case rc=9' /tmp/bemcuda_queue_status_fail.txt
grep -q 'monitor_state=finished' /tmp/bemcuda_queue_status_fail.txt
cat > /tmp/bemcuda_queue_status/logs/case.log <<'EOF'
  [GMRES] verbose residual logging enabled
EOF
touch -d '2 minutes ago' /tmp/bemcuda_queue_status/logs/case.gpu.csv
OUT=/tmp/bemcuda_queue_status scripts/queue_live_status.sh > /tmp/bemcuda_queue_status_stale.txt
grep -q 'monitor_state=stale' /tmp/bemcuda_queue_status_stale.txt
grep -q '^none$' /tmp/bemcuda_queue_status_stale.txt
(sleep 3) &
queue_status_test_pid=$!
printf '%s\n' "$queue_status_test_pid" > /tmp/bemcuda_queue_status/queue.pid
OUT=/tmp/bemcuda_queue_status scripts/queue_live_status.sh > /tmp/bemcuda_queue_status_proc.txt
grep -q -- '--- descendants ---' /tmp/bemcuda_queue_status_proc.txt
printf '30,0,55,100,3000,230\n' >> /tmp/bemcuda_queue_status/logs/case.gpu.csv
touch /tmp/bemcuda_queue_status/logs/case.gpu.csv
OUT=/tmp/bemcuda_queue_status scripts/queue_live_status.sh > /tmp/bemcuda_queue_status_delta.txt
grep -q 'wall_s_delta=' /tmp/bemcuda_queue_status_delta.txt
grep -q 'gpu_samples_delta case=case samples_delta=1' /tmp/bemcuda_queue_status_delta.txt
mkdir -p /tmp/bemcuda_queue_smi
cat > /tmp/bemcuda_queue_smi/nvidia-smi <<'EOF'
#!/usr/bin/env bash
if [[ "$*" == *"--query-compute-apps"* ]]; then
  exit 0
fi
printf '0, Test GPU, 42, 0, 100, 80.0, 200.0\n'
EOF
chmod +x /tmp/bemcuda_queue_smi/nvidia-smi
BEM_NVIDIA_SMI=/tmp/bemcuda_queue_smi/nvidia-smi \
  python3 scripts/queue_status_json.py --out /tmp/bemcuda_queue_status --active-age-s 30 > /tmp/bemcuda_queue_status.json
python3 - <<'PY'
import json
data = json.load(open("/tmp/bemcuda_queue_status.json"))
assert data["counts"] == {"current": 1, "stale": 0, "missing": 14}, data["counts"]
assert data["total"] == 15, data["total"]
assert data["completed_result_duration_count"] == 1, data
assert abs(data["completed_result_duration_mean_s"] - 100.0) < 1e-9, data
assert len(data["current_cases"]) == 1, data
assert len(data["missing_cases"]) == 14, data
assert data["stale_cases"] == [], data
assert data["monitors"], data
case = data["monitors"][0]
assert case["case"] == "case", case
assert case["samples"] == 3, case
assert case["state"] == "active", case
assert case["age_s"] >= 0, case
assert data["active_monitors"], data
assert data["gpu_inventory"]["counts"]["usable"] == 1, data["gpu_inventory"]
PY
OUT=/tmp/bemcuda_queue_status \
  BEM_NVIDIA_SMI=/tmp/bemcuda_queue_smi/nvidia-smi \
  QUEUE_WATCH_STATUS_JSON=/tmp/bemcuda_queue_status/watch_status.json \
  scripts/queue_watch_once.sh > /tmp/bemcuda_queue_watch_once.txt
grep -q 'watch_rc=0' /tmp/bemcuda_queue_watch_once.txt
grep -q 'queue_stopped_incomplete=False' /tmp/bemcuda_queue_watch_once.txt
grep -q '^accuracy_gate ' /tmp/bemcuda_queue_watch_once.txt
grep -q 'gpu_gate available=True usable=1 busy=0 unhealthy=0 unparseable=0 total=1' /tmp/bemcuda_queue_watch_once.txt
grep -q 'matrix_progress done=1 total=15 remaining=14 percent=6.7' /tmp/bemcuda_queue_watch_once.txt
grep -q 'next_missing count=14 cases=' /tmp/bemcuda_queue_watch_once.txt
grep -q 'active case=case' /tmp/bemcuda_queue_watch_once.txt
grep -q 'duration_s=' /tmp/bemcuda_queue_watch_once.txt
grep -q 'case_eta~=' /tmp/bemcuda_queue_watch_once.txt
grep -q 'queue_eta~=' /tmp/bemcuda_queue_watch_once.txt
grep -q 'eta_source=result_timing_avg_clamped' /tmp/bemcuda_queue_watch_once.txt
grep -q 'result_avg_s=100' /tmp/bemcuda_queue_watch_once.txt
grep -q 'result_avg_n=1' /tmp/bemcuda_queue_watch_once.txt
grep -q 'power_mean_w=' /tmp/bemcuda_queue_watch_once.txt
grep -q 'temp_max_c=' /tmp/bemcuda_queue_watch_once.txt
grep -q 'mem_max_mib=' /tmp/bemcuda_queue_watch_once.txt
OUT=/tmp/bemcuda_queue_status \
  BEM_NVIDIA_SMI=/tmp/bemcuda_queue_smi/nvidia-smi \
  QUEUE_WATCH_STATUS_JSON=/tmp/bemcuda_queue_status/watch_status_gpu_gate.json \
  QUEUE_WATCH_MIN_USABLE_GPUS=2 \
  scripts/queue_watch_once.sh > /tmp/bemcuda_queue_watch_gpu_gate.txt && {
  echo "queue_watch_once unexpectedly accepted too few usable GPUs" >&2
  exit 1
}
case "$?" in
  29) ;;
  *) echo "queue_watch_once GPU gate returned unexpected code" >&2; exit 1 ;;
esac
grep -q 'watch_rc=29' /tmp/bemcuda_queue_watch_gpu_gate.txt
grep -q 'gpu_gate available=True usable=1 busy=0 unhealthy=0 unparseable=0 total=1' /tmp/bemcuda_queue_watch_gpu_gate.txt
cat >/tmp/bemcuda_queue_status/accuracy_legacy.csv <<'CSV'
bem_file,pass10,gate_error,metadata_status,operator_status
runs/production_matrix_15/sphere_ka5_ref4_current_q7_d6_tol3e3.json,True,0.03,ok,missing
CSV
OUT=/tmp/bemcuda_queue_status \
  BEM_NVIDIA_SMI=/tmp/bemcuda_queue_smi/nvidia-smi \
  QUEUE_WATCH_STATUS_JSON=/tmp/bemcuda_queue_status/watch_status_accuracy_legacy.json \
  QUEUE_WATCH_ACCURACY_CSV=/tmp/bemcuda_queue_status/accuracy_legacy.csv \
  scripts/queue_watch_once.sh --strict-current-accuracy > /tmp/bemcuda_queue_watch_accuracy_legacy.txt && {
  echo "queue_watch_once --strict-current-accuracy unexpectedly returned success" >&2
  exit 1
}
case "$?" in
  27) ;;
  *) echo "queue_watch_once --strict-current-accuracy returned unexpected code" >&2; exit 1 ;;
esac
grep -q 'watch_rc=27' /tmp/bemcuda_queue_watch_accuracy_legacy.txt
grep -q 'accuracy_gate accurate=0 accurate_legacy=1 inaccurate=0 missing=14 total=15 rc=2' /tmp/bemcuda_queue_watch_accuracy_legacy.txt
rm -f /tmp/bemcuda_queue_status/queue.pid
OUT=/tmp/bemcuda_queue_status QUEUE_WATCH_STATUS_JSON=/tmp/bemcuda_queue_status/watch_status_stopped.json scripts/queue_watch_once.sh > /tmp/bemcuda_queue_watch_stopped.txt && {
  echo "queue_watch_once unexpectedly returned success for stopped incomplete queue" >&2
  exit 1
}
case "$?" in
  25) ;;
  *) echo "queue_watch_once stopped incomplete returned unexpected code" >&2; exit 1 ;;
esac
grep -q 'watch_rc=25' /tmp/bemcuda_queue_watch_stopped.txt
grep -q 'queue_stopped_incomplete=True' /tmp/bemcuda_queue_watch_stopped.txt
printf '%s\n' "$queue_status_test_pid" > /tmp/bemcuda_queue_status/queue.pid
python3 scripts/queue_status_json.py --out /tmp/bemcuda_queue_status --active-age-s 30 --fail-on-missing >/tmp/bemcuda_queue_status_missing_exit.json && {
  echo "queue_status_json --fail-on-missing unexpectedly returned success" >&2
  exit 1
}
case "$?" in
  22) ;;
  *) echo "queue_status_json --fail-on-missing returned unexpected code" >&2; exit 1 ;;
esac
touch -d '2 seconds' /tmp/bemcuda_queue_status/logs/case.gpu.csv
python3 scripts/queue_status_json.py --out /tmp/bemcuda_queue_status --active-age-s 30 > /tmp/bemcuda_queue_status_future_age.json
python3 - <<'PY'
import json
data = json.load(open("/tmp/bemcuda_queue_status_future_age.json"))
case = data["monitors"][0]
assert case["age_s"] == 0, case
PY
touch /tmp/bemcuda_queue_status/logs/case.gpu.csv
printf '40,0,56,100,3000,231\n' >> /tmp/bemcuda_queue_status/logs/case.gpu.csv
python3 scripts/queue_status_json.py --out /tmp/bemcuda_queue_status --active-age-s 30 > /tmp/bemcuda_queue_status_delta_json.json
python3 - <<'PY'
import json
data = json.load(open("/tmp/bemcuda_queue_status_delta_json.json"))
assert data["delta"]["wall_s"] is not None, data["delta"]
assert data["delta"]["monitor_sample_delta"]["case"] == 1, data["delta"]
case = data["monitors"][0]
assert case["sample_delta"] == 1, case
assert case["progress_state"] == "progressing", case
assert data["stalled_monitors"] == [], data["stalled_monitors"]
PY
python3 scripts/queue_status_json.py --out /tmp/bemcuda_queue_status --active-age-s 30 > /tmp/bemcuda_queue_status_stalled_json.json
python3 - <<'PY'
import json
data = json.load(open("/tmp/bemcuda_queue_status_stalled_json.json"))
case = data["monitors"][0]
assert case["sample_delta"] == 0, case
assert case["progress_state"] == "unknown", case
assert data["stalled_monitors"] == [], data["stalled_monitors"]
PY
python3 - <<'PY'
import json, time
json.dump({"now_s": int(time.time()) - 120, "samples": {"case": 4}},
          open("/tmp/bemcuda_queue_status/logs/.queue_status_json.snapshot", "w"))
PY
python3 scripts/queue_status_json.py --out /tmp/bemcuda_queue_status --active-age-s 30 --stall-wall-s 60 --no-snapshot-write > /tmp/bemcuda_queue_status_stalled_json.json
python3 - <<'PY'
import json
data = json.load(open("/tmp/bemcuda_queue_status_stalled_json.json"))
case = data["monitors"][0]
assert case["sample_delta"] == 0, case
assert case["progress_state"] == "stalled", case
assert data["stalled_monitors"], data
PY
python3 scripts/queue_status_json.py --out /tmp/bemcuda_queue_status --active-age-s 30 --stall-wall-s 60 --fail-on-stalled >/tmp/bemcuda_queue_status_stalled_exit.json && {
  echo "queue_status_json --fail-on-stalled unexpectedly returned success" >&2
  exit 1
}
case "$?" in
  21) ;;
  *) echo "queue_status_json --fail-on-stalled returned unexpected code" >&2; exit 1 ;;
esac
cat > /tmp/bemcuda_queue_status/logs/case.log <<'EOF'
FAIL case rc=9
EOF
python3 scripts/queue_status_json.py --out /tmp/bemcuda_queue_status --active-age-s 30 --fail-on-failed >/tmp/bemcuda_queue_status_failed_exit.json && {
  echo "queue_status_json --fail-on-failed unexpectedly returned success" >&2
  exit 1
}
case "$?" in
  20) ;;
  *) echo "queue_status_json --fail-on-failed returned unexpected code" >&2; exit 1 ;;
esac
wait "$queue_status_test_pid" || true
if python3 scripts/detect_cuda_toolchain.py --print-env >/tmp/bemcuda_cuda_env.sh 2>/dev/null; then
  # shellcheck disable=SC1091
  source /tmp/bemcuda_cuda_env.sh
fi

if command -v g++ >/dev/null 2>&1; then
  cuda_host_include="${CUDA_HOME:-$ROOT/.cuda-local}/targets/x86_64-linux/include"
  g++ -O2 -Wall -std=c++11 -Isrc -I"$cuda_host_include" \
    -o /tmp/bem_operator_config_check \
    tests/operator_config_check.cpp
  /tmp/bem_operator_config_check
  g++ -O2 -Wall -std=c++11 -Isrc -I"$cuda_host_include" \
    -o /tmp/bem_precond_policy_check \
    tests/precond_policy_check.cpp
  /tmp/bem_precond_policy_check
  g++ -O2 -Wall -std=c++11 -Isrc -I"$cuda_host_include" \
    -o /tmp/bem_solver_policy_check \
    tests/solver_policy_check.cpp
  /tmp/bem_solver_policy_check
  g++ -O2 -Wall -std=c++11 -Isrc -I"$cuda_host_include" \
    -o /tmp/bem_mesh_quality_check \
    tests/mesh_quality_check.cpp src/mesh.cpp
  /tmp/bem_mesh_quality_check
  g++ -O2 -Wall -std=c++11 -Isrc -I"$cuda_host_include" \
    -o /tmp/bem_output_json_mesh_check \
    tests/output_json_mesh_check.cpp src/output.cpp
  /tmp/bem_output_json_mesh_check
  python3 scripts/check_result_metadata.py --strict /tmp/bem_output_json_mesh_check.json
  python3 - <<'PY'
import json
data = json.load(open("/tmp/bem_output_json_mesh_check.json"))
assert data["method"]["solver_backend"] == "FMM", data["method"]
assert data["method"]["solver_profile"] == "hex_guarded", data["method"]
assert data["method"]["requested_system"] == "muller2-balanced", data["method"]
assert data["method"]["system"] == "balanced", data["method"]
assert data["method"]["system_canonicalized"] is True, data["method"]
assert data["method"]["preconditioner_enabled"] is False, data["method"]
assert data["method"]["preconditioner_reason"] == "small_nonsphere", data["method"]
assert data["method"]["farfield_mode"] == "gpu_geometry_direct", data["method"]
assert data["mesh"]["triangles"] == 2112, data["mesh"]
assert data["mesh"]["skinny_triangles"] == 0, data["mesh"]
assert data["mesh"]["edge_refine_requested"] == 1, data["mesh"]
assert data["mesh"]["edge_refine_applied"] == 0, data["mesh"]
assert data["mesh"]["edge_refine_uniform_fallback"] is True, data["mesh"]
PY
else
  echo "skip C++ host check: g++ is missing"
fi

if [[ -x ./bin/bem_cuda_fmm ]] && ./bin/bem_cuda_fmm --help 2>&1 | grep -q -- "--mesh-quality-only"; then
  if ./bin/bem_cuda_fmm --ka not-a-number >/tmp/bemcuda_bad_cli.out 2>/tmp/bemcuda_bad_cli.err; then
    echo "expected invalid numeric CLI argument to fail" >&2
    exit 1
  fi
  grep -q -- "Error: --ka expects a finite number" /tmp/bemcuda_bad_cli.err
  if ./bin/bem_cuda_fmm --ka 1 --ntheta 1 >/tmp/bemcuda_bad_cli.out 2>/tmp/bemcuda_bad_cli.err; then
    echo "expected invalid --ntheta range to fail" >&2
    exit 1
  fi
  grep -q -- "Error: --ntheta must be at least 2" /tmp/bemcuda_bad_cli.err
  if ./bin/bem_cuda_fmm --ka 1 --orient 0 1 1 >/tmp/bemcuda_bad_cli.out 2>/tmp/bemcuda_bad_cli.err; then
    echo "expected invalid --orient range to fail" >&2
    exit 1
  fi
  grep -q -- "Error: --orient counts must be positive" /tmp/bemcuda_bad_cli.err
  if ./bin/bem_cuda_fmm --ka 1 --out --single >/tmp/bemcuda_bad_cli.out 2>/tmp/bemcuda_bad_cli.err; then
    echo "expected missing --out value to fail" >&2
    exit 1
  fi
  grep -q -- "Error: --out expects a value" /tmp/bemcuda_bad_cli.err
  if ./bin/bem_cuda_fmm --ka 1 --solver --single >/tmp/bemcuda_bad_cli.out 2>/tmp/bemcuda_bad_cli.err; then
    echo "expected missing --solver value to fail" >&2
    exit 1
  fi
  grep -q -- "Error: --solver expects a value" /tmp/bemcuda_bad_cli.err
  cat >/tmp/bemcuda_obj_negative_indices.obj <<'OBJ'
v 1 0 0
v 0 1 0
v 0 0 1
v 0 0 0
f -4 -3 -2
f -4 -2 -1
f -4 -1 -3
f -3 -1 -2
OBJ
  ./bin/bem_cuda_fmm \
    --obj /tmp/bemcuda_obj_negative_indices.obj --ka 1 --ref 0 --ri 1.3116 0 \
    --single --ntheta 9 --solver fmm --mesh-quality-only \
    >/tmp/bemcuda_obj_negative_indices.out 2>/tmp/bemcuda_obj_negative_indices.err
  cat >/tmp/bemcuda_obj_bad_index.obj <<'OBJ'
v 1 0 0
v 0 1 0
v 0 0 1
f one 2 3
OBJ
  if ./bin/bem_cuda_fmm \
    --obj /tmp/bemcuda_obj_bad_index.obj --ka 1 --ref 0 --ri 1.3116 0 \
    --single --ntheta 9 --solver fmm --mesh-quality-only \
    >/tmp/bemcuda_obj_bad_index.out 2>/tmp/bemcuda_obj_bad_index.err; then
    echo "expected invalid OBJ index to fail" >&2
    exit 1
  fi
  grep -q -- "Error: invalid OBJ face index 'one'" /tmp/bemcuda_obj_bad_index.err
  ./bin/bem_cuda_fmm \
    --shape sphere --ka 1 --ref 1 --ri 1.3116 0 \
    --single --ntheta 9 --solver fmm \
    --mesh-quality-only \
    --mesh-quality-report /tmp/bemcuda_mesh_quality_smoke.json
  ./bin/bem_cuda_fmm \
    --shape hex_prism --ka 5 --ref 2 --edge-refine 1 --ri 1.3116 0 \
    --single --ntheta 9 --solver fmm \
    --mesh-quality-only \
    --mesh-quality-report /tmp/bemcuda_hex_edge_refine_quality.json
  python3 - <<'PY'
import json
path = "/tmp/bemcuda_hex_edge_refine_quality.json"
data = json.load(open(path))
assert data["verdict"] == "pass", data
assert data["skinny_triangles"] == 0, data
assert data["edge_refine_requested"] == 1, data
assert data["edge_refine_applied"] in (0, 1), data
assert isinstance(data["edge_refine_uniform_fallback"], bool), data
if data["edge_refine_applied"] == 0:
    assert data["edge_refine_uniform_fallback"] is True, data
PY
else
  echo "skip executable mesh-quality smoke: ./bin/bem_cuda_fmm is missing or lacks --mesh-quality-only"
fi

if [[ -x ./bin/bem_cuda_fmm.next ]]; then
  env -u LD_LIBRARY_PATH ./bin/bem_cuda_fmm.next --help >/tmp/bemcuda_next_help.txt
  grep -q -- "--mesh-quality-only" /tmp/bemcuda_next_help.txt
  if command -v ldd >/dev/null 2>&1; then
    env -u LD_LIBRARY_PATH ldd ./bin/bem_cuda_fmm.next > /tmp/bemcuda_next_ldd.txt
    ! grep -q 'not found' /tmp/bemcuda_next_ldd.txt
  fi
else
  echo "skip next-binary runtime smoke: ./bin/bem_cuda_fmm.next is missing"
fi

echo "local audits: ok"
