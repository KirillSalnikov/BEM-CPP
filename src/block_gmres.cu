#include "block_gmres.h"
#include "bem_fmm.h"
#include "precond.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>

// Paired GMRES(m) with restart for two complex systems sharing the same operator.
// Runs two independent GMRES iterations in lockstep, sharing the batched matvec.
// Right-preconditioned: solves A * M^{-1} * u = b, then x = M^{-1} * u.

int gmres_solve_paired(BemFmmOperator& op,
                       const cdouble* b1, const cdouble* b2,
                       cdouble* x1, cdouble* x2,
                       int restart, double tol, int maxiter,
                       bool verbose, NearFieldPrecond* precond)
{
    int n = op.system_size;

    // cuBLAS handle (shared for both systems)
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // ---- GPU memory: two sets of Krylov bases and workspace ----
    cuDoubleComplex *d_V1, *d_V2;      // Krylov bases: (n, restart+1) each
    cuDoubleComplex *d_w1, *d_w2;      // matvec output workspace
    cuDoubleComplex *d_h1, *d_h2;      // Gram-Schmidt coefficient vectors

    CUDA_CHECK(cudaMalloc(&d_V1, (size_t)n * (restart + 1) * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_V2, (size_t)n * (restart + 1) * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_w1, n * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_w2, n * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_h1, (restart + 1) * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_h2, (restart + 1) * sizeof(cuDoubleComplex)));

    // ---- Host memory: two sets of everything ----
    std::vector<cdouble> h_r1(n), h_r2(n);
    std::vector<cdouble> h_w1(n), h_w2(n);
    std::vector<cdouble> h_v1(n), h_v2(n);
    std::vector<cdouble> h_z1(n), h_z2(n);
    std::vector<cdouble> h_x1(n), h_x2(n);
    std::vector<cdouble> h_hvec1(restart + 1), h_hvec2(restart + 1);

    // Right preconditioning: store Z_j = M^{-1} * v_j for each system
    std::vector<cdouble> h_Z1, h_Z2;
    if (precond) {
        h_Z1.resize((size_t)n * restart);
        h_Z2.resize((size_t)n * restart);
    }

    // Hessenberg matrices: (restart+1, restart) stored column-major
    std::vector<cdouble> H1((restart + 1) * restart, cdouble(0));
    std::vector<cdouble> H2((restart + 1) * restart, cdouble(0));

    // Givens rotation data
    std::vector<cdouble> cs1(restart), sn1(restart), s1(restart + 1), y1(restart);
    std::vector<cdouble> cs2(restart), sn2(restart), s2(restart + 1), y2(restart);

    // Use x1/x2 as initial guess (they may be non-zero from previous solve)
    memcpy(h_x1.data(), x1, n * sizeof(cdouble));
    memcpy(h_x2.data(), x2, n * sizeof(cdouble));

    // Check if initial guess is non-zero
    bool has_x0 = false;
    for (int i = 0; i < n && !has_x0; i++)
        if (std::abs(x1[i]) > 1e-30 || std::abs(x2[i]) > 1e-30) has_x0 = true;

    // Compute initial residuals: r = b - A*x0
    int init_matvecs = 0;
    if (has_x0) {
        op.matvec_batch2(h_x1.data(), h_x2.data(), h_r1.data(), h_r2.data());
        init_matvecs = 1;
        for (int i = 0; i < n; i++) {
            h_r1[i] = b1[i] - h_r1[i];
            h_r2[i] = b2[i] - h_r2[i];
        }
    } else {
        memcpy(h_r1.data(), b1, n * sizeof(cdouble));
        memcpy(h_r2.data(), b2, n * sizeof(cdouble));
    }

    // Norms of RHS vectors
    double bnorm1 = 0, bnorm2 = 0;
    for (int i = 0; i < n; i++) {
        bnorm1 += std::norm(b1[i]);
        bnorm2 += std::norm(b2[i]);
    }
    bnorm1 = std::sqrt(bnorm1);
    bnorm2 = std::sqrt(bnorm2);
    if (bnorm1 < 1e-30) bnorm1 = 1.0;
    if (bnorm2 < 1e-30) bnorm2 = 1.0;

    double rnorm1 = 0, rnorm2 = 0;
    for (int i = 0; i < n; i++) {
        rnorm1 += std::norm(h_r1[i]);
        rnorm2 += std::norm(h_r2[i]);
    }
    rnorm1 = std::sqrt(rnorm1);
    rnorm2 = std::sqrt(rnorm2);

    if (verbose) {
        if (has_x0)
            printf("  [GMRES-paired] x0 recycled, init residual: rel1=%.2e rel2=%.2e\n",
                   rnorm1 / bnorm1, rnorm2 / bnorm2);
        else
            printf("  [GMRES-paired] start from zero: rel1=%.2e rel2=%.2e\n",
                   rnorm1 / bnorm1, rnorm2 / bnorm2);
    }

    int total_matvecs = init_matvecs;
    bool converged1 = false, converged2 = false;

    // Track how many Arnoldi steps each system completed in current cycle
    int m1 = 0, m2 = 0;

    for (int cycle = 0; cycle < maxiter; cycle++) {
        // Initialize Krylov bases: v0 = r / ||r||
        if (!converged1) {
            double inv_rnorm1 = 1.0 / rnorm1;
            for (int i = 0; i < n; i++)
                h_v1[i] = h_r1[i] * inv_rnorm1;
            CUDA_CHECK(cudaMemcpy(d_V1, h_v1.data(), n * sizeof(cuDoubleComplex),
                                  cudaMemcpyHostToDevice));
            memset(s1.data(), 0, (restart + 1) * sizeof(cdouble));
            s1[0] = cdouble(rnorm1);
            memset(H1.data(), 0, (restart + 1) * restart * sizeof(cdouble));
        }

        if (!converged2) {
            double inv_rnorm2 = 1.0 / rnorm2;
            for (int i = 0; i < n; i++)
                h_v2[i] = h_r2[i] * inv_rnorm2;
            CUDA_CHECK(cudaMemcpy(d_V2, h_v2.data(), n * sizeof(cuDoubleComplex),
                                  cudaMemcpyHostToDevice));
            memset(s2.data(), 0, (restart + 1) * sizeof(cdouble));
            s2[0] = cdouble(rnorm2);
            memset(H2.data(), 0, (restart + 1) * restart * sizeof(cdouble));
        }

        m1 = 0;
        m2 = 0;
        int j;
        for (j = 0; j < restart; j++) {
            // Apply preconditioner and matvec
            if (precond && precond->gpu_ready) {
                // GPU preconditioner: apply directly on GPU, no D→H for v_j
                if (!converged1) {
                    precond->apply_gpu((const cuDoubleComplex*)d_V1 + (size_t)j * n,
                                       precond->d_buf_z);
                    CUDA_CHECK(cudaMemcpy(h_z1.data(), precond->d_buf_z,
                                          n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    memcpy(&h_Z1[(size_t)j * n], h_z1.data(), n * sizeof(cdouble));
                }
                if (!converged2) {
                    precond->apply_gpu((const cuDoubleComplex*)d_V2 + (size_t)j * n,
                                       precond->d_buf_z);
                    CUDA_CHECK(cudaMemcpy(h_z2.data(), precond->d_buf_z,
                                          n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    memcpy(&h_Z2[(size_t)j * n], h_z2.data(), n * sizeof(cdouble));
                }

                if (!converged1 && !converged2) {
                    op.matvec_batch2(h_z1.data(), h_z2.data(), h_w1.data(), h_w2.data());
                    total_matvecs++;
                } else if (!converged1) {
                    op.matvec(h_z1.data(), h_w1.data());
                    total_matvecs++;
                } else {
                    op.matvec(h_z2.data(), h_w2.data());
                    total_matvecs++;
                }
            } else if (precond) {
                // CPU preconditioner: need v_j on host
                if (!converged1) {
                    CUDA_CHECK(cudaMemcpy(h_v1.data(), (cuDoubleComplex*)d_V1 + (size_t)j * n,
                                          n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    precond->apply(h_v1.data(), h_z1.data());
                    memcpy(&h_Z1[(size_t)j * n], h_z1.data(), n * sizeof(cdouble));
                }
                if (!converged2) {
                    CUDA_CHECK(cudaMemcpy(h_v2.data(), (cuDoubleComplex*)d_V2 + (size_t)j * n,
                                          n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    precond->apply(h_v2.data(), h_z2.data());
                    memcpy(&h_Z2[(size_t)j * n], h_z2.data(), n * sizeof(cdouble));
                }

                if (!converged1 && !converged2) {
                    op.matvec_batch2(h_z1.data(), h_z2.data(), h_w1.data(), h_w2.data());
                    total_matvecs++;
                } else if (!converged1) {
                    op.matvec(h_z1.data(), h_w1.data());
                    total_matvecs++;
                } else {
                    op.matvec(h_z2.data(), h_w2.data());
                    total_matvecs++;
                }
            } else {
                // No preconditioner: download v_j, matvec directly
                if (!converged1)
                    CUDA_CHECK(cudaMemcpy(h_v1.data(), (cuDoubleComplex*)d_V1 + (size_t)j * n,
                                          n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                if (!converged2)
                    CUDA_CHECK(cudaMemcpy(h_v2.data(), (cuDoubleComplex*)d_V2 + (size_t)j * n,
                                          n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

                if (!converged1 && !converged2) {
                    op.matvec_batch2(h_v1.data(), h_v2.data(), h_w1.data(), h_w2.data());
                    total_matvecs++;
                } else if (!converged1) {
                    op.matvec(h_v1.data(), h_w1.data());
                    total_matvecs++;
                } else {
                    op.matvec(h_v2.data(), h_w2.data());
                    total_matvecs++;
                }
            }

            // ---- Independent Arnoldi for system 1 ----
            if (!converged1) {
                CUDA_CHECK(cudaMemcpy(d_w1, h_w1.data(), n * sizeof(cuDoubleComplex),
                                      cudaMemcpyHostToDevice));

                // Batched Gram-Schmidt: h = V1(:,0:j)^H * w1
                {
                    int ncols = j + 1;
                    cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
                    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
                    cuDoubleComplex neg_one = make_cuDoubleComplex(-1.0, 0.0);

                    CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_C,
                        n, ncols,
                        &one,
                        d_V1, n,
                        d_w1, 1,
                        &zero,
                        d_h1, 1));

                    CUDA_CHECK(cudaMemcpy(h_hvec1.data(), d_h1,
                        ncols * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    for (int i = 0; i < ncols; i++)
                        H1[i * restart + j] = h_hvec1[i];

                    // w1 = w1 - V1(:,0:j) * h
                    CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_N,
                        n, ncols,
                        &neg_one,
                        d_V1, n,
                        d_h1, 1,
                        &one,
                        d_w1, 1));
                }

                // h[j+1,j] = ||w1||
                double w_norm1;
                CUBLAS_CHECK(cublasDznrm2(handle, n, d_w1, 1, &w_norm1));
                H1[(j + 1) * restart + j] = cdouble(w_norm1);

                // v_{j+1} = w1 / ||w1||
                if (w_norm1 > 1e-30) {
                    cuDoubleComplex scale = make_cuDoubleComplex(1.0 / w_norm1, 0.0);
                    CUBLAS_CHECK(cublasZscal(handle, n, &scale, d_w1, 1));
                }
                CUDA_CHECK(cudaMemcpy((cuDoubleComplex*)d_V1 + (size_t)(j + 1) * n,
                                      d_w1, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

                // Apply previous Givens rotations to column j of H1
                for (int i = 0; i < j; i++) {
                    cdouble h0 = H1[i * restart + j];
                    cdouble h1 = H1[(i + 1) * restart + j];
                    H1[i * restart + j]       = std::conj(cs1[i]) * h0 + std::conj(sn1[i]) * h1;
                    H1[(i + 1) * restart + j] = -sn1[i] * h0 + cs1[i] * h1;
                }

                // Compute new Givens rotation
                {
                    cdouble h0 = H1[j * restart + j];
                    cdouble h1 = H1[(j + 1) * restart + j];
                    double denom = std::sqrt(std::norm(h0) + std::norm(h1));
                    if (denom > 1e-30) {
                        cs1[j] = h0 / denom;
                        sn1[j] = h1 / denom;
                    } else {
                        cs1[j] = cdouble(1);
                        sn1[j] = cdouble(0);
                    }
                    H1[j * restart + j] = std::conj(cs1[j]) * h0 + std::conj(sn1[j]) * h1;
                    H1[(j + 1) * restart + j] = cdouble(0);

                    cdouble s0 = s1[j];
                    s1[j]     = std::conj(cs1[j]) * s0;
                    s1[j + 1] = -sn1[j] * s0;
                }

                double res_norm1 = std::abs(s1[j + 1]);
                double rel_res1 = res_norm1 / bnorm1;

                m1 = j + 1;

                if (rel_res1 < tol)
                    converged1 = true;
            }

            // ---- Independent Arnoldi for system 2 ----
            if (!converged2) {
                CUDA_CHECK(cudaMemcpy(d_w2, h_w2.data(), n * sizeof(cuDoubleComplex),
                                      cudaMemcpyHostToDevice));

                // Batched Gram-Schmidt: h = V2(:,0:j)^H * w2
                {
                    int ncols = j + 1;
                    cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
                    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
                    cuDoubleComplex neg_one = make_cuDoubleComplex(-1.0, 0.0);

                    CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_C,
                        n, ncols,
                        &one,
                        d_V2, n,
                        d_w2, 1,
                        &zero,
                        d_h2, 1));

                    CUDA_CHECK(cudaMemcpy(h_hvec2.data(), d_h2,
                        ncols * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    for (int i = 0; i < ncols; i++)
                        H2[i * restart + j] = h_hvec2[i];

                    CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_N,
                        n, ncols,
                        &neg_one,
                        d_V2, n,
                        d_h2, 1,
                        &one,
                        d_w2, 1));
                }

                double w_norm2;
                CUBLAS_CHECK(cublasDznrm2(handle, n, d_w2, 1, &w_norm2));
                H2[(j + 1) * restart + j] = cdouble(w_norm2);

                if (w_norm2 > 1e-30) {
                    cuDoubleComplex scale = make_cuDoubleComplex(1.0 / w_norm2, 0.0);
                    CUBLAS_CHECK(cublasZscal(handle, n, &scale, d_w2, 1));
                }
                CUDA_CHECK(cudaMemcpy((cuDoubleComplex*)d_V2 + (size_t)(j + 1) * n,
                                      d_w2, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

                // Apply previous Givens rotations to column j of H2
                for (int i = 0; i < j; i++) {
                    cdouble h0 = H2[i * restart + j];
                    cdouble h1 = H2[(i + 1) * restart + j];
                    H2[i * restart + j]       = std::conj(cs2[i]) * h0 + std::conj(sn2[i]) * h1;
                    H2[(i + 1) * restart + j] = -sn2[i] * h0 + cs2[i] * h1;
                }

                // Compute new Givens rotation
                {
                    cdouble h0 = H2[j * restart + j];
                    cdouble h1 = H2[(j + 1) * restart + j];
                    double denom = std::sqrt(std::norm(h0) + std::norm(h1));
                    if (denom > 1e-30) {
                        cs2[j] = h0 / denom;
                        sn2[j] = h1 / denom;
                    } else {
                        cs2[j] = cdouble(1);
                        sn2[j] = cdouble(0);
                    }
                    H2[j * restart + j] = std::conj(cs2[j]) * h0 + std::conj(sn2[j]) * h1;
                    H2[(j + 1) * restart + j] = cdouble(0);

                    cdouble s0 = s2[j];
                    s2[j]     = std::conj(cs2[j]) * s0;
                    s2[j + 1] = -sn2[j] * s0;
                }

                double res_norm2 = std::abs(s2[j + 1]);
                double rel_res2 = res_norm2 / bnorm2;

                m2 = j + 1;

                if (rel_res2 < tol)
                    converged2 = true;
            }

            // Verbose output: print both residuals on the same line
            if (verbose && (total_matvecs <= 3 || total_matvecs % 10 == 0)) {
                double r1 = converged1 ? 0.0 : std::abs(s1[j + 1]);
                double r2 = converged2 ? 0.0 : std::abs(s2[j + 1]);
                printf("    GMRES iter %d: res1=%.2e (%.2e) res2=%.2e (%.2e)%s%s\n",
                       total_matvecs,
                       r1, r1 / bnorm1,
                       r2, r2 / bnorm2,
                       converged1 ? " [1:done]" : "",
                       converged2 ? " [2:done]" : "");
            }

            if (converged1 && converged2)
                break;
        }

        // ---- Update solution for system 1 ----
        if (m1 > 0) {
            // Solve upper triangular H1*y1 = s1
            for (int i = m1 - 1; i >= 0; i--) {
                y1[i] = s1[i];
                for (int k = i + 1; k < m1; k++)
                    y1[i] -= H1[i * restart + k] * y1[k];
                y1[i] /= H1[i * restart + i];
            }

            if (precond) {
                for (int i = 0; i < m1; i++) {
                    const cdouble* z_i = &h_Z1[(size_t)i * n];
                    for (int k = 0; k < n; k++)
                        h_x1[k] += y1[i] * z_i[k];
                }
            } else {
                for (int i = 0; i < m1; i++) {
                    CUDA_CHECK(cudaMemcpy(h_v1.data(), (cuDoubleComplex*)d_V1 + (size_t)i * n,
                                          n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    for (int k = 0; k < n; k++)
                        h_x1[k] += y1[i] * h_v1[k];
                }
            }
        }

        // ---- Update solution for system 2 ----
        if (m2 > 0) {
            for (int i = m2 - 1; i >= 0; i--) {
                y2[i] = s2[i];
                for (int k = i + 1; k < m2; k++)
                    y2[i] -= H2[i * restart + k] * y2[k];
                y2[i] /= H2[i * restart + i];
            }

            if (precond) {
                for (int i = 0; i < m2; i++) {
                    const cdouble* z_i = &h_Z2[(size_t)i * n];
                    for (int k = 0; k < n; k++)
                        h_x2[k] += y2[i] * z_i[k];
                }
            } else {
                for (int i = 0; i < m2; i++) {
                    CUDA_CHECK(cudaMemcpy(h_v2.data(), (cuDoubleComplex*)d_V2 + (size_t)i * n,
                                          n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    for (int k = 0; k < n; k++)
                        h_x2[k] += y2[i] * h_v2[k];
                }
            }
        }

        if (converged1 && converged2)
            break;

        // Recompute residuals for unconverged systems
        if (!converged1 && !converged2) {
            // Both need residual recomputation — use batched matvec
            op.matvec_batch2(h_x1.data(), h_x2.data(), h_r1.data(), h_r2.data());
            total_matvecs++;
            for (int i = 0; i < n; i++) {
                h_r1[i] = b1[i] - h_r1[i];
                h_r2[i] = b2[i] - h_r2[i];
            }
        } else if (!converged1) {
            op.matvec(h_x1.data(), h_r1.data());
            total_matvecs++;
            for (int i = 0; i < n; i++)
                h_r1[i] = b1[i] - h_r1[i];
        } else if (!converged2) {
            op.matvec(h_x2.data(), h_r2.data());
            total_matvecs++;
            for (int i = 0; i < n; i++)
                h_r2[i] = b2[i] - h_r2[i];
        }

        if (!converged1) {
            rnorm1 = 0;
            for (int i = 0; i < n; i++)
                rnorm1 += std::norm(h_r1[i]);
            rnorm1 = std::sqrt(rnorm1);
        }
        if (!converged2) {
            rnorm2 = 0;
            for (int i = 0; i < n; i++)
                rnorm2 += std::norm(h_r2[i]);
            rnorm2 = std::sqrt(rnorm2);
        }

        if (verbose)
            printf("  [GMRES-paired] restart %d: rel1=%.2e rel2=%.2e\n",
                   cycle + 1,
                   converged1 ? 0.0 : rnorm1 / bnorm1,
                   converged2 ? 0.0 : rnorm2 / bnorm2);

        if (!converged1 && rnorm1 / bnorm1 < tol) converged1 = true;
        if (!converged2 && rnorm2 / bnorm2 < tol) converged2 = true;

        if (converged1 && converged2)
            break;
    }

    // Copy solutions to output
    memcpy(x1, h_x1.data(), n * sizeof(cdouble));
    memcpy(x2, h_x2.data(), n * sizeof(cdouble));

    if (verbose) {
        if (converged1 && converged2)
            printf("  [GMRES-paired] Both converged, %d matvec evaluations\n", total_matvecs);
        else
            printf("  [GMRES-paired] NOT fully converged (%s%s), %d matvecs, res1=%.2e res2=%.2e\n",
                   converged1 ? "" : "sys1 ",
                   converged2 ? "" : "sys2 ",
                   total_matvecs,
                   converged1 ? 0.0 : rnorm1 / bnorm1,
                   converged2 ? 0.0 : rnorm2 / bnorm2);
    }

    // Cleanup
    cudaFree(d_V1);
    cudaFree(d_V2);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_h1);
    cudaFree(d_h2);
    cublasDestroy(handle);

    return total_matvecs;
}
