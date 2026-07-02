#!/usr/bin/env python3
"""Static checks for reusable GPU far-field orientation workspaces."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    main_cpp = (ROOT / "src" / "main.cpp").read_text()
    farfield_h = (ROOT / "src" / "farfield.h").read_text()
    farfield_cu = (ROOT / "src" / "farfield.cu").read_text()

    assert "struct FFBatchWorkspace" in farfield_h
    assert "#include <cuda_runtime.h>" in farfield_h
    assert "cudaStream_t stream" in farfield_h
    assert "h_cJ_re" in farfield_h and "h_fv_re" in farfield_h
    assert "h_cJ_re_pinned" in farfield_h
    assert "h_fv_re_pinned" in farfield_h
    assert "reserve_host_coeffs" in farfield_h
    assert "reserve_host_fv" in farfield_h
    assert "reserve_host_mueller" in farfield_h
    assert "reserve_alpha" in farfield_h
    assert "reserve_mueller_accum" in farfield_h

    assert "cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)" in farfield_cu
    assert "cudaHostAlloc(reinterpret_cast<void**>(&ptr)" in farfield_cu
    assert "std::malloc(count * sizeof(double))" in farfield_cu
    assert "upload_complex_coeffs" in farfield_cu
    assert "workspace.reserve_host_coeffs(total_coeffs)" in farfield_cu
    assert "workspace.reserve_host_fv(total_fv)" in farfield_cu
    assert "reserve_host_mueller(total_mueller)" in farfield_cu
    assert "void FFBatchWorkspace::reserve_alpha(int n_alpha)" in farfield_cu
    assert "workspace.reserve_alpha(alpha_avg)" in farfield_cu
    assert "bool sync_after" in farfield_h
    assert "if (sync_after)\n        CUDA_CHECK(cudaStreamSynchronize(workspace.stream));" in farfield_cu
    assert "if (n_orient > cap_alpha)" not in farfield_cu
    assert "void FFBatchWorkspace::reserve_mueller_accum(int ndir)" in farfield_cu
    assert "void FFBatchWorkspace::zero_mueller(int ndir)\n{\n    reserve_mueller_accum(ndir);" in farfield_cu
    assert "reserve_mueller(1, ndir)" not in farfield_cu
    assert "workspace.reserve(total_coeffs, total_rhat, 1)" not in farfield_cu
    assert "workspace.reserve(total_coeffs, 0, 1)" not in farfield_cu
    assert "cudaMemcpyAsync" in farfield_cu
    assert "workspace.stream" in farfield_cu
    assert "cudaMemsetAsync(d_M_accum" in farfield_cu
    assert "cudaStreamSynchronize(workspace.stream)" in farfield_cu
    assert "cudaStreamSynchronize(stream)" in farfield_cu
    assert "std::vector<double> cJ_re" not in farfield_h
    assert "workspace.fv_re" not in farfield_cu
    assert "farfield_batch_kernel<<<grid, block, smem_size, workspace.stream>>>" in farfield_cu
    assert "farfield_mueller_direct_kernel<<<grid, block, 0, workspace.stream>>>" in farfield_cu
    assert "farfield_mueller_alpha_kernel<<<grid, block, 0, workspace.stream>>>" in farfield_cu
    assert "reduce_mueller_partials_kernel<<<grid_reduce, block_reduce, 0, workspace.stream>>>" in farfield_cu

    assert "class PinnedHostBuffer" in main_cpp
    assert "cudaHostAlloc(reinterpret_cast<void**>(&ptr_)" in main_cpp
    assert "size_t grown = cap_ + cap_ / 2;" in main_cpp
    assert "cap_ = new_cap;\n        size_ = n;" in main_cpp
    assert "PinnedHostBuffer<cdouble> batch_coeffs_J" in main_cpp
    assert "PinnedHostBuffer<cdouble> batch_coeffs_M" in main_cpp
    assert "PinnedHostBuffer<double> batch_r_hats" in main_cpp
    assert "PinnedHostBuffer<Vec3> batch_e_par" in main_cpp
    assert "PinnedHostBuffer<Vec3> batch_e_perp" in main_cpp
    assert "PinnedHostBuffer<double> batch_weights" in main_cpp
    assert "PinnedHostBuffer<double> batch_RT" in main_cpp
    assert "PinnedHostBuffer<cdouble> batch_Fv" in main_cpp
    assert "std::vector<cdouble> batch_coeffs_J" not in main_cpp
    assert "std::vector<double> batch_r_hats" not in main_cpp
    assert main_cpp.count("bool orient_pack_omp = !bem_env_flag_enabled(\"BEM_ORIENT_PACK_SERIAL\")") >= 2
    assert "std::getenv(\"BEM_FF_CPU_ACCUM\")" not in main_cpp
    assert "std::getenv(\"BEM_FF_SEPARATE\")" not in main_cpp
    assert "std::getenv(\"BEM_FF_NO_ALPHA_DIRECT\")" not in main_cpp
    assert "std::getenv(\"BEM_FF_NO_ALPHA_GEOM\")" not in main_cpp
    assert "std::getenv(\"BEM_ORIENT_PACK_SERIAL\")" not in main_cpp
    assert "std::getenv(\"BEM_NO_GPU_RHS\")" not in main_cpp
    assert "std::getenv(\"BEM_FF_HOST_PACK\")" not in farfield_cu
    assert "std::getenv(\"BEM_FF_VERBOSE\")" not in farfield_cu
    assert main_cpp.count("#pragma omp parallel for schedule(static) if(orient_pack_omp && N > 2048)") >= 4
    assert "ff_gpu_accum && alpha_avg > 1" not in main_cpp
    assert "FMM geometry-direct GPU far-field enabled" in main_cpp
    assert "Geometry-direct GPU coefficient mixing enabled" in main_cpp
    assert "const char* output_farfield_mode" in main_cpp
    assert "farfield_mode ? farfield_mode : \"unknown\"" in (ROOT / "src" / "output.cpp").read_text()
    assert "gpu_geometry_direct_multi_gpu" in main_cpp
    assert "count, alpha_avg, ntheta, false" in main_cpp
    assert "mgpu_ws[gd]->zero_mueller(ntheta);" in main_cpp
    assert "local_ws.zero_mueller(ntheta)" not in main_cpp
    assert main_cpp.count("download_mueller(mgpu_partial") == 1
    assert "cudaStreamSynchronize(mgpu_ws[(size_t)gd]->stream)" in main_cpp
    assert "if (!ff_alpha_geom)\n\t                    ff_workspace.reserve_mueller(ff_batch_orient, ntheta);" in main_cpp
    assert "local_ws.reserve_mueller(count * alpha_avg, ntheta)" not in main_cpp
    assert "batch_orient_idx" not in main_cpp

    print("farfield workspace policy: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
