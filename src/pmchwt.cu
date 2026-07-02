#include "pmchwt.h"
#include "gpu_select.h"
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
                     double unknown_m_scale,
                     std::complex<double> row_h_scale,
                     double int_op_sign,
                     double k_identity,
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
    std::vector<int> asm_devices;
    int asm_mgpu = 1;
    bool asm_mgpu_requested = false;
    int device_count = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&device_count);
    if (dev_err != cudaSuccess)
        device_count = 0;
    if (const char* env = std::getenv("BEM_ASM_GPU_LIST")) {
        asm_devices = bem_parse_gpu_list_env(env);
        asm_mgpu_requested = true;
    }
    if (bem_env_has_value("BEM_ASM_MGPU")) {
        asm_mgpu = std::max(1, bem_env_int("BEM_ASM_MGPU", asm_mgpu));
        asm_mgpu_requested = true;
    } else if (!bem_env_flag_enabled("BEM_NO_AUTO_MGPU")) {
        if (device_count >= 2)
            asm_mgpu = 2;
    }
    if (!asm_devices.empty()) {
        if (!bem_validate_gpu_list(asm_devices, device_count)) {
            fprintf(stderr, "Warning: invalid BEM_ASM_GPU_LIST for %d CUDA devices; falling back to one GPU\n",
                    device_count);
            asm_devices.clear();
            asm_mgpu = 1;
        } else {
            if (asm_devices.size() > 2) {
                fprintf(stderr, "Warning: PMCHWT assembly has two independent operators; "
                                "only the first two BEM_ASM_GPU_LIST entries are used\n");
                asm_devices.resize(2);
            }
            asm_mgpu = (int)asm_devices.size();
        }
    } else if (asm_mgpu > 1 && device_count >= 2) {
        asm_devices.push_back(0);
        asm_devices.push_back(1);
        asm_mgpu = 2;
    } else if (asm_mgpu > 1) {
        if (asm_mgpu_requested)
            fprintf(stderr, "Warning: multi-GPU assembly requested but fewer than 2 CUDA devices are available\n");
        asm_mgpu = 1;
    }

    if (asm_mgpu > 1) {
        printf("  Parallel exterior/interior assembly enabled: GPU %d and GPU %d\n",
               asm_devices[0], asm_devices[1]);
        #ifdef _OPENMP
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                CUDA_CHECK(cudaSetDevice(asm_devices[0]));
                printf("  Exterior operators (k=%.4f+%.4fi)...\n", k_ext.real(), k_ext.imag());
                assemble_L_K_cuda(rwg, mesh, k_ext, quad_order, L_ext.data(), K_ext.data());
            }
            #pragma omp section
            {
                CUDA_CHECK(cudaSetDevice(asm_devices[1]));
                printf("  Interior operators (k=%.4f+%.4fi)...\n", k_int.real(), k_int.imag());
                assemble_L_K_cuda(rwg, mesh, k_int, quad_order, L_int.data(), K_int.data());
            }
        }
        CUDA_CHECK(cudaSetDevice(asm_devices[0]));
        #else
        fprintf(stderr, "Warning: binary was built without OpenMP; multi-GPU assembly disabled\n");
        asm_mgpu = 1;
        #endif
    }
    if (asm_mgpu == 1) {
        if (!asm_devices.empty()) {
            CUDA_CHECK(cudaSetDevice(asm_devices[0]));
            printf("  Single-GPU assembly pinned to GPU %d\n", asm_devices[0]);
        }
        printf("  Exterior operators (k=%.4f+%.4fi)...\n", k_ext.real(), k_ext.imag());
        assemble_L_K_cuda(rwg, mesh, k_ext, quad_order, L_ext.data(), K_ext.data());

        printf("  Interior operators (k=%.4f+%.4fi)...\n", k_int.real(), k_int.imag());
        assemble_L_K_cuda(rwg, mesh, k_int, quad_order, L_int.data(), K_int.data());
    }
    double operator_time = t0.elapsed_s();

    // Form Z matrix
    Timer form_timer;
    form_bem_system_matrix(N, L_ext.data(), K_ext.data(), L_int.data(), K_int.data(),
                           eta_ext, eta_int, unknown_m_scale, row_h_scale,
                           int_op_sign, k_identity, Z);
    printf("  Total PMCHWT assembly: %.1fs (operators %.1fs, form %.1fs)\n",
           t0.elapsed_s(), operator_time, form_timer.elapsed_s());

    // Copy L_ext, K_ext to output if requested
    if (L_ext_out) memcpy(L_ext_out, L_ext.data(), N*N*sizeof(std::complex<double>));
    if (K_ext_out) memcpy(K_ext_out, K_ext.data(), N*N*sizeof(std::complex<double>));
}
