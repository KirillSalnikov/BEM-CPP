#include "pmchwt.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void assemble_pmchwt(const RWG& rwg, const Mesh& mesh,
                     std::complex<double> k_ext, std::complex<double> k_int,
                     std::complex<double> eta_ext, std::complex<double> eta_int,
                     int quad_order,
                     std::complex<double>* Z,
                     std::complex<double>* L_ext_out,
                     std::complex<double>* K_ext_out)
{
    int N = rwg.N;
    int N2 = 2 * N;
    printf("  Assembling %dx%d PMCHWT matrix (%d RWG functions)...\n", N2, N2, N);

    std::vector<std::complex<double>> L_ext(N*N), K_ext(N*N);
    std::vector<std::complex<double>> L_int(N*N), K_int(N*N);

    Timer t0;
    int asm_mgpu = 1;
    bool asm_mgpu_requested = false;
    if (const char* env = std::getenv("BEM_ASM_MGPU")) {
        asm_mgpu = std::max(1, atoi(env));
        asm_mgpu_requested = true;
    } else if (!std::getenv("BEM_NO_AUTO_MGPU")) {
        int device_count = 0;
        cudaError_t dev_err = cudaGetDeviceCount(&device_count);
        if (dev_err == cudaSuccess && device_count >= 2)
            asm_mgpu = 2;
    }
    if (asm_mgpu > 1) {
        int device_count = 0;
        cudaError_t dev_err = cudaGetDeviceCount(&device_count);
        if (dev_err != cudaSuccess || device_count < 2) {
            if (asm_mgpu_requested)
                fprintf(stderr, "Warning: BEM_ASM_MGPU requested but fewer than 2 CUDA devices are available\n");
            asm_mgpu = 1;
        }
    }

    if (asm_mgpu > 1) {
        printf("  Parallel exterior/interior assembly enabled: 2 GPUs\n");
        #ifdef _OPENMP
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                CUDA_CHECK(cudaSetDevice(0));
                printf("  Exterior operators (k=%.4f+%.4fi)...\n", k_ext.real(), k_ext.imag());
                assemble_L_K_cuda(rwg, mesh, k_ext, quad_order, L_ext.data(), K_ext.data());
            }
            #pragma omp section
            {
                CUDA_CHECK(cudaSetDevice(1));
                printf("  Interior operators (k=%.4f+%.4fi)...\n", k_int.real(), k_int.imag());
                assemble_L_K_cuda(rwg, mesh, k_int, quad_order, L_int.data(), K_int.data());
            }
        }
        CUDA_CHECK(cudaSetDevice(0));
        #else
        asm_mgpu = 1;
        #endif
    }
    if (asm_mgpu == 1) {
        printf("  Exterior operators (k=%.4f+%.4fi)...\n", k_ext.real(), k_ext.imag());
        assemble_L_K_cuda(rwg, mesh, k_ext, quad_order, L_ext.data(), K_ext.data());

        printf("  Interior operators (k=%.4f+%.4fi)...\n", k_int.real(), k_int.imag());
        assemble_L_K_cuda(rwg, mesh, k_int, quad_order, L_int.data(), K_int.data());
    }
    double operator_time = t0.elapsed_s();

    // Form Z matrix
    Timer form_timer;
    std::complex<double> ce = eta_ext, ci = eta_int;
    std::complex<double> inv_ce = 1.0 / eta_ext, inv_ci = 1.0 / eta_int;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::complex<double> le = L_ext[i*N+j], li = L_int[i*N+j];
            std::complex<double> ke = K_ext[i*N+j], ki = K_int[i*N+j];
            std::complex<double> K_sum = ke + ki;

            // Top-left: eta_ext*L_ext + eta_int*L_int
            Z[i * N2 + j] = ce * le + ci * li;
            // Top-right: -(K_ext + K_int)
            Z[i * N2 + (N + j)] = -K_sum;
            // Bottom-left: K_ext + K_int
            Z[(N + i) * N2 + j] = K_sum;
            // Bottom-right: L_ext/eta_ext + L_int/eta_int
            Z[(N + i) * N2 + (N + j)] = inv_ce * le + inv_ci * li;
        }
    }
    printf("  Total PMCHWT assembly: %.1fs (operators %.1fs, form %.1fs)\n",
           t0.elapsed_s(), operator_time, form_timer.elapsed_s());

    // Copy L_ext, K_ext to output if requested
    if (L_ext_out) memcpy(L_ext_out, L_ext.data(), N*N*sizeof(std::complex<double>));
    if (K_ext_out) memcpy(K_ext_out, K_ext.data(), N*N*sizeof(std::complex<double>));
}
