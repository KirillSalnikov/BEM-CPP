#include "solver.h"
#include "gpu_select.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#ifdef BEM_USE_CUSOLVER
#include <cusolverDn.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef BEM_USE_CUSOLVER
#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t _status = (call); \
    if (_status != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "CUSOLVER error %d at %s:%d\n", (int)_status, __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)
#endif

static __device__ __host__ inline double2 csub2(double2 a, double2 b)
{
    return make_double2(a.x - b.x, a.y - b.y);
}

static __device__ __host__ inline double2 cmul2(double2 a, double2 b)
{
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

static __device__ __host__ inline double2 cdiv2(double2 a, double2 b)
{
    double den = b.x * b.x + b.y * b.y;
    return make_double2((a.x * b.x + a.y * b.y) / den,
                        (a.y * b.x - a.x * b.y) / den);
}

__global__ void rhs_to_rowmajor_kernel(const double2* __restrict__ B,
                                       double2* __restrict__ X,
                                       int n, int nrhs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * nrhs;
    if (idx >= total)
        return;
    int rhs = idx % nrhs;
    int i = idx / nrhs;
    X[(size_t)i * nrhs + rhs] = B[(size_t)rhs * n + i];
}

__global__ void rowmajor_to_rhs_kernel(const double2* __restrict__ X,
                                       double2* __restrict__ B,
                                       int n, int nrhs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * nrhs;
    if (idx >= total)
        return;
    int rhs = idx % nrhs;
    int i = idx / nrhs;
    B[(size_t)rhs * n + i] = X[(size_t)i * nrhs + rhs];
}

__global__ void pivot_rows_kernel(double2* __restrict__ X, int row_a, int row_b, int nrhs)
{
    int rhs = blockIdx.x * blockDim.x + threadIdx.x;
    if (rhs >= nrhs)
        return;
    double2* a = X + (size_t)row_a * nrhs;
    double2* b = X + (size_t)row_b * nrhs;
    double2 tmp = a[rhs];
    a[rhs] = b[rhs];
    b[rhs] = tmp;
}

__global__ void forward_update_kernel(const double2* __restrict__ Z,
                                      double2* __restrict__ X,
                                      int n, int nrhs, int j)
{
    int rhs = blockIdx.x * blockDim.x + threadIdx.x;
    int row_off = blockIdx.y;
    int i = j + 1 + row_off;
    if (rhs >= nrhs || i >= n)
        return;
    double2 lij = Z[(size_t)i * n + j];
    double2 xj = X[(size_t)j * nrhs + rhs];
    double2 xi = X[(size_t)i * nrhs + rhs];
    X[(size_t)i * nrhs + rhs] = csub2(xi, cmul2(lij, xj));
}

__global__ void divide_row_kernel(const double2* __restrict__ Z,
                                  double2* __restrict__ X,
                                  int n, int nrhs, int j)
{
    int rhs = blockIdx.x * blockDim.x + threadIdx.x;
    if (rhs >= nrhs)
        return;
    double2 diag = Z[(size_t)j * n + j];
    double2 x = X[(size_t)j * nrhs + rhs];
    X[(size_t)j * nrhs + rhs] = cdiv2(x, diag);
}

__global__ void backward_update_kernel(const double2* __restrict__ Z,
                                       double2* __restrict__ X,
                                       int n, int nrhs, int j)
{
    int rhs = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (rhs >= nrhs || i >= j)
        return;
    double2 uij = Z[(size_t)i * n + j];
    double2 xj = X[(size_t)j * nrhs + rhs];
    double2 xi = X[(size_t)i * nrhs + rhs];
    X[(size_t)i * nrhs + rhs] = csub2(xi, cmul2(uij, xj));
}

#ifdef BEM_USE_LAPACK
extern "C" {
void zgetrf_(const int* m, const int* n, std::complex<double>* a,
             const int* lda, int* ipiv, int* info);
void zgetrs_(const char* trans, const int* n, const int* nrhs,
             const std::complex<double>* a, const int* lda, const int* ipiv,
             std::complex<double>* b, const int* ldb, int* info);
}
#endif

static bool use_lapack_lu()
{
#ifdef BEM_USE_LAPACK
    return !bem_env_flag_enabled("BEM_NO_LAPACK");
#else
    return false;
#endif
}

int lu_factorize_cuda(std::complex<double>* Z, int n, int* ipiv) {
    Timer timer;
    double2* A = reinterpret_cast<double2*>(Z);

    for (int k = 0; k < n; k++) {
        int piv = k;
        double2 akk = A[(size_t)k * n + k];
        double best = akk.x * akk.x + akk.y * akk.y;
        for (int i = k + 1; i < n; i++) {
            double2 aik = A[(size_t)i * n + k];
            double v = aik.x * aik.x + aik.y * aik.y;
            if (v > best) {
                best = v;
                piv = i;
            }
        }

        ipiv[k] = piv;
        if (best == 0.0) {
            fprintf(stderr, "  LU factorization failed: singular pivot at %d\n", k);
            return k + 1;
        }

        if (piv != k) {
            double2* row_k = A + (size_t)k * n;
            double2* row_p = A + (size_t)piv * n;
            for (int j = 0; j < n; j++) {
                double2 tmp = row_k[j];
                row_k[j] = row_p[j];
                row_p[j] = tmp;
            }
        }

        const double2* row_k = A + (size_t)k * n;
        double pr = row_k[k].x;
        double pi = row_k[k].y;
        double inv_den = 1.0 / (pr * pr + pi * pi);
        #pragma omp parallel for schedule(static)
        for (int i = k + 1; i < n; i++) {
            double2* row_i = A + (size_t)i * n;
            double zr = row_i[k].x;
            double zi = row_i[k].y;
            double lr = (zr * pr + zi * pi) * inv_den;
            double li = (zi * pr - zr * pi) * inv_den;
            row_i[k] = make_double2(lr, li);
            for (int j = k + 1; j < n; j++)
            {
                double ar = row_k[j].x;
                double ai = row_k[j].y;
                double vr = row_i[j].x - (lr * ar - li * ai);
                double vi = row_i[j].y - (lr * ai + li * ar);
                row_i[j] = make_double2(vr, vi);
            }
        }
    }

    printf("  LU factorization CPU fallback (%dx%d): %.1fs\n", n, n, timer.elapsed_s());
    return 0;
}

int lu_solve_cuda(const std::complex<double>* Z, const int* ipiv,
                  int n, std::complex<double>* B, int nrhs) {
    Timer timer;

    #pragma omp parallel for schedule(static)
    for (int rhs = 0; rhs < nrhs; rhs++) {
        std::complex<double>* b = B + (size_t)rhs * n;

        for (int k = 0; k < n; k++) {
            if (ipiv[k] != k)
                std::swap(b[k], b[ipiv[k]]);
        }

        for (int i = 1; i < n; i++) {
            std::complex<double> sum = b[i];
            for (int j = 0; j < i; j++)
                sum -= Z[i * n + j] * b[j];
            b[i] = sum;
        }

        for (int i = n - 1; i >= 0; i--) {
            std::complex<double> sum = b[i];
            for (int j = i + 1; j < n; j++)
                sum -= Z[i * n + j] * b[j];
            b[i] = sum / Z[i * n + i];
        }
    }

    printf("  LU solve CPU fallback (%dx%d, %d RHS): %.2fs\n", n, n, nrhs, timer.elapsed_s());
    return 0;
}

static bool use_rowmajor_multi_rhs(int nrhs)
{
    const char* layout = std::getenv("BEM_SOLVE_LAYOUT");
    if (layout && layout[0]) {
        if (strcmp(layout, "rhs") == 0 || strcmp(layout, "RHS") == 0)
            return false;
        if (strcmp(layout, "row") == 0 || strcmp(layout, "ROW") == 0)
            return true;
    }
    int threshold = 1000000000;
    threshold = std::max(1, bem_env_int("BEM_ROW_SOLVE_MIN_RHS", threshold));
    return nrhs >= threshold;
}

int lu_solve_many_rhs_rowmajor(const std::complex<double>* Z, const int* ipiv,
                               int n, std::complex<double>* B, int nrhs) {
    Timer timer;
    std::vector<std::complex<double>> X((size_t)n * nrhs);

    #pragma omp parallel for schedule(static)
    for (int rhs = 0; rhs < nrhs; rhs++) {
        const std::complex<double>* b = B + (size_t)rhs * n;
        for (int i = 0; i < n; i++)
            X[(size_t)i * nrhs + rhs] = b[i];
    }

    for (int k = 0; k < n; k++) {
        int piv = ipiv[k];
        if (piv != k) {
            std::complex<double>* row_k = X.data() + (size_t)k * nrhs;
            std::complex<double>* row_p = X.data() + (size_t)piv * nrhs;
            #pragma omp parallel for schedule(static)
            for (int rhs = 0; rhs < nrhs; rhs++)
                std::swap(row_k[rhs], row_p[rhs]);
        }
    }

    for (int j = 0; j < n; j++) {
        const std::complex<double>* xj = X.data() + (size_t)j * nrhs;
        #pragma omp parallel for schedule(static)
        for (int i = j + 1; i < n; i++) {
            std::complex<double> lij = Z[(size_t)i * n + j];
            if (lij == std::complex<double>(0.0, 0.0))
                continue;
            std::complex<double>* xi = X.data() + (size_t)i * nrhs;
            for (int rhs = 0; rhs < nrhs; rhs++)
                xi[rhs] -= lij * xj[rhs];
        }
    }

    for (int j = n - 1; j >= 0; j--) {
        std::complex<double>* xj = X.data() + (size_t)j * nrhs;
        std::complex<double> diag = Z[(size_t)j * n + j];
        #pragma omp parallel for schedule(static)
        for (int rhs = 0; rhs < nrhs; rhs++)
            xj[rhs] /= diag;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < j; i++) {
            std::complex<double> uij = Z[(size_t)i * n + j];
            if (uij == std::complex<double>(0.0, 0.0))
                continue;
            std::complex<double>* xi = X.data() + (size_t)i * nrhs;
            for (int rhs = 0; rhs < nrhs; rhs++)
                xi[rhs] -= uij * xj[rhs];
        }
    }

    #pragma omp parallel for schedule(static)
    for (int rhs = 0; rhs < nrhs; rhs++) {
        std::complex<double>* b = B + (size_t)rhs * n;
        for (int i = 0; i < n; i++)
            b[i] = X[(size_t)i * nrhs + rhs];
    }

    printf("  LU solve row-major multi-RHS CPU fallback (%dx%d, %d RHS): %.2fs\n",
           n, n, nrhs, timer.elapsed_s());
    return 0;
}

int lu_solve_many_rhs_gpu(const std::complex<double>* Z, const int* ipiv,
                          int n, std::complex<double>* B, int nrhs) {
    Timer timer;
    double2* d_Z = nullptr;
    double2* d_B = nullptr;
    double2* d_X = nullptr;
    size_t matrix_elems = (size_t)n * n;
    size_t rhs_elems = (size_t)n * nrhs;
    CUDA_CHECK(cudaMalloc(&d_Z, matrix_elems * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_B, rhs_elems * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_X, rhs_elems * sizeof(double2)));
    CUDA_CHECK(cudaMemcpy(d_Z, reinterpret_cast<const double2*>(Z),
                          matrix_elems * sizeof(double2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, reinterpret_cast<const double2*>(B),
                          rhs_elems * sizeof(double2), cudaMemcpyHostToDevice));

    const int block = 256;
    int total = (int)rhs_elems;
    int grid_1d = (total + block - 1) / block;
    rhs_to_rowmajor_kernel<<<grid_1d, block>>>(d_B, d_X, n, nrhs);
    CUDA_CHECK(cudaGetLastError());

    int rhs_grid = (nrhs + block - 1) / block;
    for (int k = 0; k < n; k++) {
        int piv = ipiv[k];
        if (piv != k) {
            pivot_rows_kernel<<<rhs_grid, block>>>(d_X, k, piv, nrhs);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (int j = 0; j < n; j++) {
        dim3 grid(rhs_grid, n - j - 1);
        if (grid.y > 0) {
            forward_update_kernel<<<grid, block>>>(d_Z, d_X, n, nrhs, j);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (int j = n - 1; j >= 0; j--) {
        divide_row_kernel<<<rhs_grid, block>>>(d_Z, d_X, n, nrhs, j);
        CUDA_CHECK(cudaGetLastError());
        dim3 grid(rhs_grid, j);
        if (grid.y > 0) {
            backward_update_kernel<<<grid, block>>>(d_Z, d_X, n, nrhs, j);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    rowmajor_to_rhs_kernel<<<grid_1d, block>>>(d_X, d_B, n, nrhs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(B), d_B,
                          rhs_elems * sizeof(double2), cudaMemcpyDeviceToHost));

    cudaFree(d_Z);
    cudaFree(d_B);
    cudaFree(d_X);
    printf("  LU solve GPU multi-RHS fallback (%dx%d, %d RHS): %.2fs\n",
           n, n, nrhs, timer.elapsed_s());
    return 0;
}

int lu_solve_many_rhs_mgpu(const std::complex<double>* Z, const int* ipiv,
                           int n, std::complex<double>* B, int nrhs, int ngpu)
{
    Timer timer;
    int original_device = 0;
    cudaGetDevice(&original_device);
    int device_count = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&device_count);
    if (dev_err != cudaSuccess || device_count <= 1)
        return lu_solve_many_rhs_gpu(Z, ipiv, n, B, nrhs);
    std::vector<int> devices;
    if (const char* env = std::getenv("BEM_LU_GPU_LIST")) {
        devices = bem_parse_gpu_list_env(env);
        if (!bem_validate_gpu_list(devices, device_count)) {
            fprintf(stderr, "Warning: invalid BEM_LU_GPU_LIST for %d CUDA devices; disabling LU multi-GPU split\n",
                    device_count);
            devices.clear();
        }
    }
    if (devices.empty()) {
        ngpu = std::max(1, std::min(ngpu, device_count));
        for (int gd = 0; gd < ngpu; gd++)
            devices.push_back(gd);
    } else {
        ngpu = (int)devices.size();
    }
    if (ngpu <= 1) {
        if (!devices.empty())
            CUDA_CHECK(cudaSetDevice(devices[0]));
        int ret = lu_solve_many_rhs_gpu(Z, ipiv, n, B, nrhs);
        cudaSetDevice(original_device);
        return ret;
    }

    std::vector<int> info(ngpu, 0);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int gd = 0; gd < ngpu; gd++) {
        int start = (nrhs * gd) / ngpu;
        int end = (nrhs * (gd + 1)) / ngpu;
        int count = end - start;
        if (count <= 0)
            continue;
        cudaSetDevice(devices[(size_t)gd]);
        info[gd] = lu_solve_many_rhs_gpu(Z, ipiv, n, B + (size_t)start * n, count);
    }
    cudaSetDevice(original_device);
    for (int gd = 0; gd < ngpu; gd++) {
        if (info[gd] != 0)
            return info[gd];
    }
    printf("  LU solve multi-GPU split (%dx%d, %d RHS, %d GPUs): %.2fs\n",
           n, n, nrhs, ngpu, timer.elapsed_s());
    return 0;
}

int lu_solve_full_cusolver(std::complex<double>* Z, int n,
                           std::complex<double>* B, int nrhs)
{
#ifndef BEM_USE_CUSOLVER
    (void)Z;
    (void)n;
    (void)B;
    (void)nrhs;
    return -1;
#else
    if (bem_env_flag_enabled("BEM_NO_CUSOLVER_LU"))
        return -1;

    Timer timer;
    std::vector<std::complex<double>> A_col((size_t)n * n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            A_col[(size_t)i + (size_t)j * n] = Z[(size_t)i * n + j];
    }

    cusolverDnHandle_t handle = nullptr;
    cuDoubleComplex* d_A = nullptr;
    cuDoubleComplex* d_B = nullptr;
    cuDoubleComplex* d_work = nullptr;
    int* d_ipiv = nullptr;
    int* d_info = nullptr;
    int lwork = 0;
    int info_host = 0;
    int ret = -1;

    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)n * n * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)n * nrhs * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_ipiv, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_A, reinterpret_cast<const cuDoubleComplex*>(A_col.data()),
                          (size_t)n * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, reinterpret_cast<const cuDoubleComplex*>(B),
                          (size_t)n * nrhs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    CUSOLVER_CHECK(cusolverDnZgetrf_bufferSize(handle, n, n, d_A, n, &lwork));
    CUDA_CHECK(cudaMalloc(&d_work, (size_t)lwork * sizeof(cuDoubleComplex)));

    Timer fact_timer;
    CUSOLVER_CHECK(cusolverDnZgetrf(handle, n, n, d_A, n, d_work, d_ipiv, d_info));
    CUDA_CHECK(cudaMemcpy(&info_host, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  LU factorization cuSOLVER zgetrf (%dx%d): %.2fs\n", n, n, fact_timer.elapsed_s());
    if (info_host != 0) {
        fprintf(stderr, "  cuSOLVER zgetrf failed with info=%d; falling back to internal LU\n", info_host);
        goto cleanup;
    }

    {
        Timer solve_timer;
        CUSOLVER_CHECK(cusolverDnZgetrs(handle, CUBLAS_OP_N, n, nrhs, d_A, n, d_ipiv, d_B, n, d_info));
        CUDA_CHECK(cudaMemcpy(&info_host, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        printf("  LU solve cuSOLVER zgetrs (%dx%d, %d RHS): %.2fs\n", n, n, nrhs, solve_timer.elapsed_s());
    }
    if (info_host != 0) {
        fprintf(stderr, "  cuSOLVER zgetrs failed with info=%d; falling back to internal LU\n", info_host);
        goto cleanup;
    }

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<cuDoubleComplex*>(B), d_B,
                          (size_t)n * nrhs * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    printf("  Total factorize+solve cuSOLVER: %.2fs\n", timer.elapsed_s());
    ret = 0;

cleanup:
    cudaFree(d_work);
    cudaFree(d_info);
    cudaFree(d_ipiv);
    cudaFree(d_B);
    cudaFree(d_A);
    if (handle)
        cusolverDnDestroy(handle);
    return ret;
#endif
}

int lu_solve_full(std::complex<double>* Z, int n,
                  std::complex<double>* B, int nrhs) {
    Timer timer;
#ifdef _OPENMP
    omp_set_dynamic(0);
    int old_omp_threads = omp_get_max_threads();
    bool restore_omp_threads = false;
    const char* lu_threads_env = std::getenv("BEM_LU_THREADS");
    int lu_threads = 0;
    if (lu_threads_env && lu_threads_env[0]) {
        lu_threads = bem_env_int("BEM_LU_THREADS", lu_threads);
    } else if (!std::getenv("OMP_NUM_THREADS")) {
        lu_threads = std::min(48, omp_get_num_procs());
    }
    if (lu_threads > 0 && lu_threads != old_omp_threads) {
        omp_set_num_threads(lu_threads);
        restore_omp_threads = true;
    }
    auto restore_threads = [&]() {
        if (restore_omp_threads)
            omp_set_num_threads(old_omp_threads);
    };
#else
    auto restore_threads = []() {};
#endif
#ifdef BEM_USE_LAPACK
    if (use_lapack_lu()) {
        std::vector<std::complex<double>> A_col((size_t)n * n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                A_col[(size_t)i + (size_t)j * n] = Z[(size_t)i * n + j];
        }
        std::vector<int> ipiv(n);
        int info = 0;
        Timer fact_timer;
        zgetrf_(&n, &n, A_col.data(), &n, ipiv.data(), &info);
        printf("  LU factorization LAPACK zgetrf (%dx%d): %.1fs\n", n, n, fact_timer.elapsed_s());
        if (info != 0) {
            fprintf(stderr, "  LAPACK zgetrf failed with info=%d, falling back to internal LU\n", info);
        } else {
            Timer solve_timer;
            char trans = 'N';
            zgetrs_(&trans, &n, &nrhs, A_col.data(), &n, ipiv.data(), B, &n, &info);
            printf("  LU solve LAPACK zgetrs (%dx%d, %d RHS): %.2fs\n", n, n, nrhs, solve_timer.elapsed_s());
            printf("  Total factorize+solve LAPACK: %.1fs\n", timer.elapsed_s());
            if (info == 0) {
                restore_threads();
                return 0;
            }
            fprintf(stderr, "  LAPACK zgetrs failed with info=%d, falling back to internal LU\n", info);
        }
    }
#else
    (void)use_lapack_lu;
#endif

    if (n >= 512) {
        int gpu_info = lu_solve_full_cusolver(Z, n, B, nrhs);
        if (gpu_info == 0) {
            restore_threads();
            return 0;
        }
    }

    std::vector<int> ipiv(n);
    int info = lu_factorize_cuda(Z, n, ipiv.data());
    if (info != 0) {
        restore_threads();
        return info;
    }
    bool gpu_lu_solve = (nrhs >= 128) && !bem_env_flag_enabled("BEM_NO_GPU_LU_SOLVE");
    if (bem_env_flag_present("BEM_GPU_LU_SOLVE"))
        gpu_lu_solve = bem_env_flag_enabled("BEM_GPU_LU_SOLVE");
    int lu_mgpu = 1;
    int mgpu_min_rhs = 512;
    mgpu_min_rhs = std::max(1, bem_env_int("BEM_LU_MGPU_MIN_RHS", mgpu_min_rhs));
    if (bem_env_has_value("BEM_LU_MGPU"))
        lu_mgpu = std::max(1, bem_env_int("BEM_LU_MGPU", lu_mgpu));
    else if (bem_env_has_value("BEM_FF_MGPU")) {
        if (nrhs >= mgpu_min_rhs)
            lu_mgpu = std::max(1, bem_env_int("BEM_FF_MGPU", lu_mgpu));
    } else if (nrhs >= mgpu_min_rhs && !bem_env_flag_enabled("BEM_NO_AUTO_MGPU")) {
        int device_count = 0;
        cudaError_t dev_err = cudaGetDeviceCount(&device_count);
        if (dev_err == cudaSuccess)
            lu_mgpu = std::max(1, device_count);
    }

    if (gpu_lu_solve && lu_mgpu > 1)
        info = lu_solve_many_rhs_mgpu(Z, ipiv.data(), n, B, nrhs, lu_mgpu);
    else if (gpu_lu_solve)
        info = lu_solve_many_rhs_gpu(Z, ipiv.data(), n, B, nrhs);
    else if (use_rowmajor_multi_rhs(nrhs))
        info = lu_solve_many_rhs_rowmajor(Z, ipiv.data(), n, B, nrhs);
    else
        info = lu_solve_cuda(Z, ipiv.data(), n, B, nrhs);
    printf("  Total factorize+solve CPU fallback: %.1fs\n", timer.elapsed_s());
    restore_threads();
    return info;
}
