#include "gmres.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>

// Right-preconditioned GMRES(m) with restart for complex systems
// Solves A * M^{-1} * u = b, then x = M^{-1} * u
// Tracks the TRUE residual ||b - Ax||
// Uses cuBLAS for GPU vector operations

int gmres_solve(BemFmmOperator& op, const cdouble* b, cdouble* x,
                int restart, double tol, int maxiter, bool verbose,
                NearFieldPrecond* precond)
{
    int n = op.system_size;

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Allocate GPU vectors
    cuDoubleComplex *d_x, *d_r, *d_v, *d_w;
    cuDoubleComplex *d_V;  // Krylov basis: (n, restart+1)
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_w, n * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_V, (size_t)n * (restart + 1) * sizeof(cuDoubleComplex)));

    // GPU buffer for batched Gram-Schmidt h vector (size restart+1)
    cuDoubleComplex *d_h;
    CUDA_CHECK(cudaMalloc(&d_h, (restart + 1) * sizeof(cuDoubleComplex)));
    std::vector<cdouble> h_hvec(restart + 1);  // host copy of h vector

    // Host workspace
    std::vector<cdouble> h_r(n), h_w(n), h_v(n), h_z(n);
    std::vector<cdouble> h_x(n);

    // For right preconditioning: store Z_j = M^{-1} * v_j
    std::vector<cdouble> h_Z;
    if (precond)
        h_Z.resize((size_t)n * restart);

    // Hessenberg matrix H: (restart+1, restart)
    std::vector<cdouble> H((restart + 1) * restart, cdouble(0));
    std::vector<cdouble> s(restart + 1);  // RHS of least-squares
    std::vector<cdouble> cs(restart);     // Givens cos
    std::vector<cdouble> sn(restart);     // Givens sin
    std::vector<cdouble> y(restart);      // solution of Hessenberg system

    // Use x as initial guess
    memcpy(h_x.data(), x, n * sizeof(cdouble));

    // Check if initial guess is non-zero
    bool has_x0 = false;
    for (int i = 0; i < n && !has_x0; i++)
        if (std::abs(x[i]) > 1e-30) has_x0 = true;

    // Compute initial residual: r = b - A*x0
    if (has_x0) {
        op.matvec(h_x.data(), h_r.data());
        for (int i = 0; i < n; i++)
            h_r[i] = b[i] - h_r[i];
    } else {
        memcpy(h_r.data(), b, n * sizeof(cdouble));
    }

    // Compute norm of b
    double bnorm = 0;
    for (int i = 0; i < n; i++)
        bnorm += std::norm(b[i]);
    bnorm = std::sqrt(bnorm);
    if (bnorm < 1e-30) bnorm = 1.0;

    double rnorm = 0;
    for (int i = 0; i < n; i++)
        rnorm += std::norm(h_r[i]);
    rnorm = std::sqrt(rnorm);

    if (verbose)
        printf("  [GMRES] start: ||r||/||b|| = %.2e\n", rnorm / bnorm);

    int total_iters = 0;
    bool converged = false;

    for (int cycle = 0; cycle < maxiter; cycle++) {
        // v0 = r / ||r||
        double inv_rnorm = 1.0 / rnorm;
        for (int i = 0; i < n; i++)
            h_v[i] = h_r[i] * inv_rnorm;

        // Upload v0 as first column of V
        CUDA_CHECK(cudaMemcpy(d_V, h_v.data(), n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        // Initialize s = rnorm * e1
        memset(s.data(), 0, (restart + 1) * sizeof(cdouble));
        s[0] = cdouble(rnorm);

        // Clear Hessenberg matrix
        memset(H.data(), 0, (restart + 1) * restart * sizeof(cdouble));

        int j;
        for (j = 0; j < restart; j++) {
            total_iters++;

            if (precond && precond->gpu_ready) {
                // GPU preconditioner: apply directly on GPU, no D→H for v_j
                precond->apply_gpu((const cuDoubleComplex*)(d_V) + (size_t)j * n,
                                   precond->d_buf_z);
                CUDA_CHECK(cudaMemcpy(h_z.data(), precond->d_buf_z,
                                      n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                memcpy(&h_Z[(size_t)j * n], h_z.data(), n * sizeof(cdouble));
                op.matvec(h_z.data(), h_w.data());
            } else {
                // Download v_j from GPU (needed for CPU precond or no-precond matvec)
                CUDA_CHECK(cudaMemcpy(h_v.data(), (cuDoubleComplex*)d_V + (size_t)j * n,
                                      n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                if (precond) {
                    precond->apply(h_v.data(), h_z.data());
                    memcpy(&h_Z[(size_t)j * n], h_z.data(), n * sizeof(cdouble));
                    op.matvec(h_z.data(), h_w.data());
                } else {
                    op.matvec(h_v.data(), h_w.data());
                }
            }

            // Upload w to GPU
            CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

            // Batched Gram-Schmidt orthogonalization using cublasZgemv
            // Step 1: h = V(:,0:j)^H * w  — all dot products in one call
            {
                int ncols = j + 1;
                cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
                cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
                cuDoubleComplex neg_one = make_cuDoubleComplex(-1.0, 0.0);

                // h = V^H * w, where V is (n, ncols), h is (ncols, 1)
                CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_C,
                    n, ncols,
                    &one,
                    d_V, n,       // V stored column-major, leading dim = n
                    d_w, 1,       // input vector w
                    &zero,
                    d_h, 1));     // output vector h

                // Download h to host and store in Hessenberg matrix
                CUDA_CHECK(cudaMemcpy(h_hvec.data(), d_h,
                    ncols * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                for (int i = 0; i < ncols; i++)
                    H[i * restart + j] = h_hvec[i];

                // Step 2: w = w - V(:,0:j) * h  — subtract all projections in one call
                CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_N,
                    n, ncols,
                    &neg_one,
                    d_V, n,       // V stored column-major, leading dim = n
                    d_h, 1,       // h vector on GPU
                    &one,
                    d_w, 1));     // w = w - V*h
            }

            // h[j+1,j] = ||w||
            double w_norm;
            CUBLAS_CHECK(cublasDznrm2(handle, n, d_w, 1, &w_norm));
            H[(j + 1) * restart + j] = cdouble(w_norm);

            // v_{j+1} = w / ||w||
            if (w_norm > 1e-30) {
                cuDoubleComplex scale = make_cuDoubleComplex(1.0 / w_norm, 0.0);
                CUBLAS_CHECK(cublasZscal(handle, n, &scale, d_w, 1));
            }
            CUDA_CHECK(cudaMemcpy((cuDoubleComplex*)d_V + (size_t)(j + 1) * n,
                                  d_w, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

            // Apply previous Givens rotations to column j of H
            for (int i = 0; i < j; i++) {
                cdouble h0 = H[i * restart + j];
                cdouble h1 = H[(i + 1) * restart + j];
                H[i * restart + j]       = std::conj(cs[i]) * h0 + std::conj(sn[i]) * h1;
                H[(i + 1) * restart + j] = -sn[i] * h0 + cs[i] * h1;
            }

            // Compute new Givens rotation
            {
                cdouble h0 = H[j * restart + j];
                cdouble h1 = H[(j + 1) * restart + j];
                double denom = std::sqrt(std::norm(h0) + std::norm(h1));
                if (denom > 1e-30) {
                    cs[j] = h0 / denom;
                    sn[j] = h1 / denom;
                } else {
                    cs[j] = cdouble(1);
                    sn[j] = cdouble(0);
                }

                H[j * restart + j] = std::conj(cs[j]) * h0 + std::conj(sn[j]) * h1;
                H[(j + 1) * restart + j] = cdouble(0);

                // Apply to s
                cdouble s0 = s[j];
                s[j]     = std::conj(cs[j]) * s0;
                s[j + 1] = -sn[j] * s0;
            }

            double res_norm = std::abs(s[j + 1]);
            double rel_res = res_norm / bnorm;

            if (verbose && (total_iters <= 3 || total_iters % 10 == 0))
                printf("    GMRES iter %d: res=%.2e (rel=%.2e)\n",
                       total_iters, res_norm, rel_res);

            if (rel_res < tol) {
                j++;
                converged = true;
                break;
            }
        }

        // Solve upper triangular H*y = s
        int m = j;
        for (int i = m - 1; i >= 0; i--) {
            y[i] = s[i];
            for (int k = i + 1; k < m; k++)
                y[i] -= H[i * restart + k] * y[k];
            y[i] /= H[i * restart + i];
        }

        if (precond) {
            // Right preconditioning: x = x + Z * y  (Z_j = M^{-1} * v_j)
            for (int i = 0; i < m; i++) {
                const cdouble* z_i = &h_Z[(size_t)i * n];
                for (int k = 0; k < n; k++)
                    h_x[k] += y[i] * z_i[k];
            }
        } else {
            // Standard: x = x + V * y
            for (int i = 0; i < m; i++) {
                CUDA_CHECK(cudaMemcpy(h_v.data(), (cuDoubleComplex*)d_V + (size_t)i * n,
                                      n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                for (int k = 0; k < n; k++)
                    h_x[k] += y[i] * h_v[k];
            }
        }

        if (converged) break;

        // Compute new residual: r = b - A*x
        op.matvec(h_x.data(), h_r.data());
        for (int i = 0; i < n; i++)
            h_r[i] = b[i] - h_r[i];

        rnorm = 0;
        for (int i = 0; i < n; i++)
            rnorm += std::norm(h_r[i]);
        rnorm = std::sqrt(rnorm);

        if (verbose)
            printf("  [GMRES] restart %d: ||r||/||b|| = %.2e\n", cycle + 1, rnorm / bnorm);

        if (rnorm / bnorm < tol) {
            converged = true;
            break;
        }
    }

    // Copy solution to output
    memcpy(x, h_x.data(), n * sizeof(cdouble));

    if (verbose) {
        if (converged)
            printf("  [GMRES] Converged in %d iterations\n", total_iters);
        else
            printf("  [GMRES] NOT converged after %d iterations, res=%.2e\n",
                   total_iters, rnorm / bnorm);
    }

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_V);
    cudaFree(d_h);
    cublasDestroy(handle);

    return converged ? 0 : 1;
}
