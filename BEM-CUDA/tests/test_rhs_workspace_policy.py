#!/usr/bin/env python3
"""Static checks for reusable GPU RHS orientation workspaces."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    main_cpp = (ROOT / "src" / "main.cpp").read_text()
    rhs_cpp = (ROOT / "src" / "rhs.cpp").read_text()
    rhs_h = (ROOT / "src" / "rhs.h").read_text()

    assert "struct RHSBatchWorkspace" in rhs_h
    assert "cudaHostAlloc(&ptr" in rhs_cpp
    assert "h_B_pinned" in rhs_h
    assert "cudaStream_t stream" in rhs_h
    assert "need_host_rhs = true" in rhs_h
    assert "#include <cuda_runtime.h>" in rhs_h
    assert "workspace.host_B()" in main_cpp
    assert "rhs_can_use_workspace_direct" in main_cpp
    assert "solve_b_par = rhs_par" in main_cpp
    assert "B = rhs_workspace.host_B()" in main_cpp
    assert "B_storage.assign" in main_cpp
    assert "lu_solve_full(Z.data(), N2, B, n_total * 2)" in main_cpp
    assert "std::malloc(bytes)" in rhs_cpp
    assert "cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)" in rhs_cpp
    assert "workspace.reserve(n_orient, rhs_elems, B == nullptr)" in rhs_cpp
    assert "if (need_host_rhs && rhs_elems > cap_host_rhs_elems)" in rhs_cpp
    assert "cudaMemcpyAsync" in rhs_cpp
    assert "workspace.stream" in rhs_cpp
    assert "compute_rhs_planewave_pairs_cached_cuda_ws" in rhs_h
    assert "compute_rhs_planewave_pairs_cached_cuda_ws_scaled" in rhs_h
    assert "row_h_scale.real()" in rhs_cpp
    assert "row_h_scale.imag()" in rhs_cpp
    assert "compute_rhs_planewave_pairs_cached_cuda_ws_scaled(" in main_cpp
    assert "n_form ? std::complex<double>(1.0, 0.0) : row_h_scale" in main_cpp
    assert "bp[N + i] *= row_h_scale" in main_cpp
    assert "rhs_batch_B.resize" not in main_cpp

    raw_calls_in_main = main_cpp.count("compute_rhs_planewave_pairs_cached_cuda(")
    ws_calls_in_main = (
        main_cpp.count("compute_rhs_planewave_pairs_cached_cuda_ws(")
        + main_cpp.count("compute_rhs_planewave_pairs_cached_cuda_ws_scaled(")
    )
    assert raw_calls_in_main == 0, "main.cpp must use reusable RHS workspace calls"
    assert ws_calls_in_main >= 2, "dense and GMRES orientation paths should use RHS workspace"
    assert "RHSBatchWorkspace rhs_workspace;" in main_cpp

    print("rhs workspace policy: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
