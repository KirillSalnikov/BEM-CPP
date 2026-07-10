#!/usr/bin/env python3
"""Queue launchers should keep one BEM process on the assigned GPU by default."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    guarded = (ROOT / "scripts" / "run_guarded_bem_case.sh").read_text()
    matrix_queue = (ROOT / "scripts" / "run_accuracy_matrix_15_queue.sh").read_text()
    resume_cases = (ROOT / "scripts" / "resume_accuracy_matrix_cases.sh").read_text()
    remote_resume_cases = (ROOT / "scripts" / "remote_resume_accuracy_matrix_cases.sh").read_text()
    orient_queue = (ROOT / "run_orient_queue.py").read_text()
    orient_mgpu = (ROOT / "run_orient_mgpu.py").read_text()
    obj_compare = (ROOT / "scripts" / "run_obj_adda_compare.py").read_text()
    hex_compare = (ROOT / "scripts" / "run_hex_adda_compare.py").read_text()
    complex_refresh = (ROOT / "scripts" / "run_complex_operator_dust_refresh.sh").read_text()
    fig7_memory = (ROOT / "scripts" / "run_fig7_memory_queue.sh").read_text()
    recompute_meta = (ROOT / "scripts" / "recompute_convergence_meta.sh").read_text()
    ri_sweep = (ROOT / "scripts" / "run_sphere_ri_sweep.sh").read_text()
    ri_fallback = (ROOT / "scripts" / "run_sphere_ri_missing_fallback_queue.sh").read_text()
    sphere30 = (ROOT / "scripts" / "run_sphere30_ref6_candidates.sh").read_text()
    bem_candidate = (ROOT / "scripts" / "run_bem_candidate.py").read_text()
    main_cpp = (ROOT / "src" / "main.cpp").read_text()
    pmchwt_cu = (ROOT / "src" / "pmchwt.cu").read_text()
    solver_cu = (ROOT / "src" / "solver.cu").read_text()
    bem_fmm_cu = (ROOT / "src" / "bem_fmm.cu").read_text()
    fmm_cu = (ROOT / "src" / "fmm.cu").read_text()
    gmres_cu = (ROOT / "src" / "gmres.cu").read_text()
    block_gmres_cu = (ROOT / "src" / "block_gmres.cu").read_text()
    precond_cu = (ROOT / "src" / "precond.cu").read_text()
    assembly_cu = (ROOT / "src" / "assembly.cu").read_text()
    surface_pfft_cu = (ROOT / "src" / "surface_pfft.cu").read_text()
    gpu_select = (ROOT / "src" / "gpu_select.h").read_text()

    assert 'export BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}"' in guarded
    assert 'export BEM_GMRES_VERBOSE="${BEM_GMRES_VERBOSE:-1}"' in guarded
    assert 'BEM_GMRES_MAX_CYCLES="${BEM_GMRES_MAX_CYCLES:-30}"' not in guarded
    assert 'export BEM_GMRES_STAGNATION_CYCLES="${BEM_GMRES_STAGNATION_CYCLES:-0}"' in guarded
    assert "--gmres-max-cycles N" in main_cpp
    assert "gmres_max_cycles_set ? gmres_max_cycles_cli : acc_policy.gmres_max_cycles" in main_cpp
    assert matrix_queue.count('BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}"') >= 2
    assert matrix_queue.count('BEM_GMRES_VERBOSE="${BEM_GMRES_VERBOSE:-1}"') >= 2
    assert 'BEM_GMRES_MAX_CYCLES="${BEM_GMRES_MAX_CYCLES:-8}"' not in matrix_queue
    assert matrix_queue.count('BEM_GMRES_STAGNATION_CYCLES="${BEM_GMRES_STAGNATION_CYCLES:-0}"') >= 2
    assert 'BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}" nohup' in resume_cases
    assert 'env BEM_NO_AUTO_MGPU=1 bash scripts/resume_accuracy_matrix_cases.sh --run' in remote_resume_cases
    assert '--gpus "$gpu" --max-jobs 1 --cases "$case_name"' in remote_resume_cases
    assert "--require-cloude-physical" in guarded
    assert matrix_queue.count("--require-cloude-physical") >= 2
    assert "BEM_METADATA_SKIP_CLOUDE" in guarded
    assert "BEM_METADATA_SKIP_CLOUDE" in matrix_queue
    assert 'env.setdefault("BEM_NO_AUTO_MGPU", "1")' in orient_queue
    assert 'env.setdefault("BEM_GMRES_VERBOSE", "1")' in orient_queue
    assert 'env.setdefault("BEM_NO_AUTO_MGPU", "1")' in orient_mgpu
    assert 'env.setdefault("BEM_NO_AUTO_MGPU", "1")' in obj_compare
    assert 'env.setdefault("BEM_NO_AUTO_MGPU", "1")' in hex_compare
    assert 'BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}"' in complex_refresh
    assert "--require-converged" in complex_refresh
    assert "--validate-numeric" in complex_refresh
    assert "--require-complex-operator" in complex_refresh
    assert "--require-cloude-physical" in complex_refresh
    assert "BEM_METADATA_SKIP_CLOUDE" in complex_refresh
    assert 'export BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}"' in fig7_memory
    assert 'BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}"' in recompute_meta
    assert ri_sweep.count('BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}"') >= 2
    assert 'BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}"' in ri_fallback
    assert 'export BEM_NO_AUTO_MGPU="${BEM_NO_AUTO_MGPU:-1}"' in sphere30
    assert 'GMRES_MAX_CYCLES="${GMRES_MAX_CYCLES:-80}"' in ri_sweep
    assert 'GMRES_STAGNATION_CYCLES="${GMRES_STAGNATION_CYCLES:-0}"' in ri_sweep
    assert 'export BEM_GMRES_MAX_CYCLES="${BEM_GMRES_MAX_CYCLES:-8}"' not in sphere30
    assert 'export BEM_GMRES_STAGNATION_CYCLES="${BEM_GMRES_STAGNATION_CYCLES:-0}"' in sphere30
    assert "export BEM_NO_AUTO_MGPU=${BEM_NO_AUTO_MGPU:-1};" in bem_candidate
    assert "export BEM_GMRES_VERBOSE=${BEM_GMRES_VERBOSE:-1};" in bem_candidate
    assert "bem_env_value_enabled" in gpu_select
    assert "bem_env_flag_present" in gpu_select
    assert "bem_env_has_value" in gpu_select
    assert "bem_env_int" in gpu_select
    assert "bem_env_double" in gpu_select
    assert "bem_env_flag_enabled" in main_cpp
    assert "bem_env_flag_enabled" in pmchwt_cu
    assert "bem_env_flag_enabled" in solver_cu
    assert "parse_cli_int_arg" in main_cpp
    assert "parse_cli_double_arg" in main_cpp
    assert "parse_cli_string_arg" in main_cpp
    assert "atoi(argv" not in main_cpp
    assert "atof(argv" not in main_cpp
    for opt in [
        "--shape",
        "--obj",
        "--orient-file",
        "--scat-plane",
        "--out",
        "--solver",
        "--system",
        "--export-currents",
        "--mesh-quality-report",
    ]:
        assert f'parse_cli_string_arg(argc, argv, i, "{opt}"' in main_cpp
    assert 'std::getenv("BEM_NO_AUTO_MGPU")' not in main_cpp
    assert 'std::getenv("BEM_NO_AUTO_MGPU")' not in pmchwt_cu
    assert 'std::getenv("BEM_NO_AUTO_MGPU")' not in solver_cu
    assert 'std::getenv("BEM_FMM_BATCH4")' not in main_cpp
    assert 'std::getenv("BEM_FMM_BATCH4")' not in bem_fmm_cu
    assert 'std::getenv("BEM_FMM_BATCH4")' not in fmm_cu
    assert 'std::getenv("BEM_FMM_NO_BATCH4")' not in main_cpp
    assert 'std::getenv("BEM_FMM_ALLOC_BATCH4")' not in fmm_cu
    assert 'std::getenv("BEM_PINNED_MATVEC_STAGE")' not in bem_fmm_cu
    assert 'std::getenv("BEM_FMM_MV_MEMSET")' not in bem_fmm_cu
    assert 'std::getenv("BEM_SPFFT_FORCE")' not in main_cpp
    assert 'std::getenv("BEM_SPFFT_FORCE")' not in bem_fmm_cu
    assert 'std::getenv("BEM_HEX_UNSAFE_FAST")' not in main_cpp
    assert 'std::getenv("BEM_NO_AUTO_ALPHA_AVG")' not in main_cpp
    assert 'std::getenv("BEM_NO_AUTO_BALANCED")' not in main_cpp
    assert 'std::getenv("BEM_EXPERIMENTAL_NFORM")' not in main_cpp
    assert 'bem_env_flag_present("BEM_FMM_BATCH4")' in main_cpp
    assert 'bem_env_flag_present("BEM_PREC_BLOCK")' in main_cpp
    assert 'bem_env_flag_present("BEM_GMRES_STORE_Z")' in main_cpp
    assert 'bem_env_flag_present("BEM_GMRES_REORTH")' in main_cpp
    for flag in [
        "BEM_PREC_BLOCK_SIZE",
        "BEM_PREC_SWEEPS",
        "BEM_PREC_NEAR",
        "BEM_PREC_OMEGA",
        "BEM_FF_BATCH",
    ]:
        assert f'bem_env_has_value("{flag}")' in main_cpp

    for flag in [
        "BEM_FF_BATCH",
        "BEM_FF_TARGET_MB",
        "BEM_FF_MAX_BATCH",
        "BEM_FF_MAX_BASE_BATCH",
        "BEM_RHS_BATCH",
        "BEM_RHS_TARGET_MB",
        "BEM_RHS_MAX_BATCH",
    ]:
        assert f'atoi(std::getenv("{flag}"))' not in main_cpp
        assert f'atof(std::getenv("{flag}"))' not in main_cpp

    for flag in [
        "BEM_SYSTEM_INT_SIGN",
        "BEM_SYSTEM_K_IDENTITY",
        "BEM_SYSTEM_M_SCALE",
        "BEM_SYSTEM_H_ROW_SCALE",
        "BEM_SYSTEM_H_ROW_SCALE_IMAG",
        "BEM_SYSTEM_NFORM_EPS_INT",
        "BEM_SYSTEM_NFORM_M_IDENTITY",
        "BEM_GMRES_STORE_Z_MAX_MB",
    ]:
        assert f'bem_env_double("{flag}"' in main_cpp
        assert f'atof(std::getenv("{flag}"))' not in main_cpp

    for flag in [
        "BEM_FMM_BATCH4_MAX_N",
        "BEM_GMRES_MAX_CYCLES",
        "BEM_NFORM_FF_MODE",
        "BEM_NFORM_RHS_MODE",
        "BEM_ORIENT_PROGRESS",
        "BEM_ORIENT_RECYCLE",
        "BEM_ASM_MGPU",
        "BEM_ASM_BLOCK_X",
        "BEM_ASM_BLOCK_Y",
    ]:
        combined = main_cpp + pmchwt_cu + assembly_cu
        assert f'bem_env_int("{flag}"' in combined
        assert f'atoi(std::getenv("{flag}"))' not in combined
    assert 'bem_env_double("BEM_SPFFT_CORR_RADIUS_H"' in surface_pfft_cu
    assert 'atof(std::getenv("BEM_SPFFT_CORR_RADIUS_H"))' not in surface_pfft_cu
    assert "Recompute the true" in block_gmres_cu
    assert 'use_fmm && !use_prec && !krylov_kind_set' in main_cpp
    assert 'setenv("BEM_GMRES_DEVICE", "1", 0)' in main_cpp
    assert 'use_gpu_gmres ? "gmres_gpu" : "gmres_cpu"' in main_cpp
    assert "op.matvec_batch2(x1, x2, r1.data(), r2.data())" in block_gmres_cu
    assert "true ||r||/||b||" in gmres_cu
    assert "numerical breakdown while solving Hessenberg" in gmres_cu
    assert "numerical breakdown while solving Hessenberg" in block_gmres_cu
    assert "gmres_numerical_breakdowns" in (ROOT / "src" / "output.cpp").read_text()
    for flag in [
        "BEM_GMRES_REORTH",
        "BEM_GMRES_STORE_Z",
        "BEM_GMRES_FUSED_UPDATE",
        "BEM_GMRES_PAIR_ARNOLDI",
        "BEM_GMRES_NO_STORE_Z",
        "BEM_GMRES_VERBOSE",
        "BEM_PREC_FORCE",
        "BEM_PREC_BLOCK",
        "BEM_PREC_GPU",
        "BEM_PREC_REUSE_WORKSPACE",
        "BEM_PREC_KEEP_HOST",
        "BEM_NO_ORIENT_PROJECT",
    ]:
        needle = f'std::getenv("{flag}")'
        assert needle not in main_cpp
        assert needle not in gmres_cu
        assert needle not in block_gmres_cu
        assert needle not in precond_cu

    for flag in [
        "BEM_NO_LAPACK",
        "BEM_NO_CUSOLVER_LU",
        "BEM_NO_GPU_LU_SOLVE",
        "BEM_GPU_LU_SOLVE",
    ]:
        needle = f'std::getenv("{flag}")'
        assert needle not in solver_cu

    print("single-GPU queue policy: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
