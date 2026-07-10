#include "device_linalg.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace {

__global__ void zero_kernel(double2* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = make_double2(0.0, 0.0);
}

__global__ void sub_kernel(double2* out, const double2* a, const double2* b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double2 av = a[i];
        double2 bv = b[i];
        out[i] = make_double2(av.x - bv.x, av.y - bv.y);
    }
}

__global__ void axpy_kernel(double2* y, double2 alpha, const double2* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double2 xv = x[i];
        double2 yv = y[i];
        y[i] = make_double2(yv.x + alpha.x * xv.x - alpha.y * xv.y,
                            yv.y + alpha.x * xv.y + alpha.y * xv.x);
    }
}

__global__ void scale_kernel(double2* y, const double2* x, double alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double2 xv = x[i];
        y[i] = make_double2(alpha * xv.x, alpha * xv.y);
    }
}

__global__ void norm_kernel(const double2* x, int n, double* block_sums)
{
    extern __shared__ double sh[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    double s = 0.0;
    if (i < n) {
        double2 v = x[i];
        s = v.x * v.x + v.y * v.y;
    }
    sh[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        block_sums[blockIdx.x] = sh[0];
}

__global__ void norm_pair_kernel(const double2* x1, const double2* x2, int n,
                                 double* block_sums1, double* block_sums2)
{
    extern __shared__ double sh[];
    double* sh1 = sh;
    double* sh2 = sh + blockDim.x;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    double s1 = 0.0;
    double s2 = 0.0;
    if (i < n) {
        double2 v1 = x1[i];
        double2 v2 = x2[i];
        s1 = v1.x * v1.x + v1.y * v1.y;
        s2 = v2.x * v2.x + v2.y * v2.y;
    }
    sh1[tid] = s1;
    sh2[tid] = s2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh1[tid] += sh1[tid + stride];
            sh2[tid] += sh2[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_sums1[blockIdx.x] = sh1[0];
        block_sums2[blockIdx.x] = sh2[0];
    }
}

__global__ void dot_kernel(const double2* a, const double2* b, int n, double2* block_sums)
{
    extern __shared__ double sh[];
    double* shr = sh;
    double* shi = sh + blockDim.x;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    double sr = 0.0;
    double si = 0.0;
    if (i < n) {
        double2 av = a[i];
        double2 bv = b[i];
        sr = av.x * bv.x + av.y * bv.y;
        si = av.x * bv.y - av.y * bv.x;
    }
    shr[tid] = sr;
    shi[tid] = si;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shr[tid] += shr[tid + stride];
            shi[tid] += shi[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
        block_sums[blockIdx.x] = make_double2(shr[0], shi[0]);
}

__global__ void dot_pair_kernel(const double2* a1, const double2* b1,
                                const double2* a2, const double2* b2,
                                int n, double2* block_sums1, double2* block_sums2)
{
    extern __shared__ double sh[];
    double* s1r = sh;
    double* s1i = sh + blockDim.x;
    double* s2r = sh + 2 * blockDim.x;
    double* s2i = sh + 3 * blockDim.x;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    double r1 = 0.0, i1 = 0.0, r2 = 0.0, i2 = 0.0;
    if (i < n) {
        double2 av1 = a1[i];
        double2 bv1 = b1[i];
        double2 av2 = a2[i];
        double2 bv2 = b2[i];
        r1 = av1.x * bv1.x + av1.y * bv1.y;
        i1 = av1.x * bv1.y - av1.y * bv1.x;
        r2 = av2.x * bv2.x + av2.y * bv2.y;
        i2 = av2.x * bv2.y - av2.y * bv2.x;
    }
    s1r[tid] = r1;
    s1i[tid] = i1;
    s2r[tid] = r2;
    s2i[tid] = i2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s1r[tid] += s1r[tid + stride];
            s1i[tid] += s1i[tid + stride];
            s2r[tid] += s2r[tid + stride];
            s2i[tid] += s2i[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_sums1[blockIdx.x] = make_double2(s1r[0], s1i[0]);
        block_sums2[blockIdx.x] = make_double2(s2r[0], s2i[0]);
    }
}

int grid_for(int n)
{
    return (n + 255) / 256;
}

double sum_device_blocks(double* d_sums, int blocks)
{
    std::vector<double> h((size_t)blocks);
    CUDA_CHECK(cudaMemcpy(h.data(), d_sums, (size_t)blocks * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (double v : h)
        s += v;
    return s;
}

double2 sum_device_complex_blocks(double2* d_sums, int blocks)
{
    std::vector<double2> h((size_t)blocks);
    CUDA_CHECK(cudaMemcpy(h.data(), d_sums, (size_t)blocks * sizeof(double2),
                          cudaMemcpyDeviceToHost));
    double2 s = make_double2(0.0, 0.0);
    for (double2 v : h) {
        s.x += v.x;
        s.y += v.y;
    }
    return s;
}

} // namespace

void device_complex_zero(double2* x, int n)
{
    if (n <= 0) return;
    zero_kernel<<<grid_for(n), 256>>>(x, n);
    CUDA_CHECK(cudaGetLastError());
}

void device_complex_copy(double2* dst, const double2* src, int n)
{
    if (n <= 0 || dst == src) return;
    CUDA_CHECK(cudaMemcpy(dst, src, (size_t)n * sizeof(double2), cudaMemcpyDeviceToDevice));
}

void device_complex_sub(double2* out, const double2* a, const double2* b, int n)
{
    if (n <= 0) return;
    sub_kernel<<<grid_for(n), 256>>>(out, a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

void device_complex_axpy(double2* y, double2 alpha, const double2* x, int n)
{
    if (n <= 0) return;
    axpy_kernel<<<grid_for(n), 256>>>(y, alpha, x, n);
    CUDA_CHECK(cudaGetLastError());
}

void device_complex_scale(double2* y, const double2* x, double alpha, int n)
{
    if (n <= 0) return;
    scale_kernel<<<grid_for(n), 256>>>(y, x, alpha, n);
    CUDA_CHECK(cudaGetLastError());
}

double device_complex_norm(const double2* x, int n)
{
    if (n <= 0) return 0.0;
    int blocks = grid_for(n);
    double* d_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sums, (size_t)blocks * sizeof(double)));
    norm_kernel<<<blocks, 256, 256 * sizeof(double)>>>(x, n, d_sums);
    CUDA_CHECK(cudaGetLastError());
    double s = sum_device_blocks(d_sums, blocks);
    CUDA_CHECK(cudaFree(d_sums));
    return std::sqrt(s);
}

void device_complex_norm_pair(const double2* x1, const double2* x2, int n,
                              double* norm1, double* norm2)
{
    if (n <= 0) {
        *norm1 = 0.0;
        *norm2 = 0.0;
        return;
    }
    int blocks = grid_for(n);
    double* d_sums1 = nullptr;
    double* d_sums2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sums1, (size_t)blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sums2, (size_t)blocks * sizeof(double)));
    norm_pair_kernel<<<blocks, 256, 2 * 256 * sizeof(double)>>>(x1, x2, n, d_sums1, d_sums2);
    CUDA_CHECK(cudaGetLastError());
    double s1 = sum_device_blocks(d_sums1, blocks);
    double s2 = sum_device_blocks(d_sums2, blocks);
    CUDA_CHECK(cudaFree(d_sums1));
    CUDA_CHECK(cudaFree(d_sums2));
    *norm1 = std::sqrt(s1);
    *norm2 = std::sqrt(s2);
}

double2 device_complex_dot(const double2* a, const double2* b, int n)
{
    if (n <= 0)
        return make_double2(0.0, 0.0);
    int blocks = grid_for(n);
    double2* d_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sums, (size_t)blocks * sizeof(double2)));
    dot_kernel<<<blocks, 256, 2 * 256 * sizeof(double)>>>(a, b, n, d_sums);
    CUDA_CHECK(cudaGetLastError());
    double2 out = sum_device_complex_blocks(d_sums, blocks);
    CUDA_CHECK(cudaFree(d_sums));
    return out;
}

void device_complex_dot_pair(const double2* a1, const double2* b1,
                             const double2* a2, const double2* b2,
                             int n, double2* dot1, double2* dot2)
{
    if (n <= 0) {
        *dot1 = make_double2(0.0, 0.0);
        *dot2 = make_double2(0.0, 0.0);
        return;
    }
    int blocks = grid_for(n);
    double2* d_sums1 = nullptr;
    double2* d_sums2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sums1, (size_t)blocks * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_sums2, (size_t)blocks * sizeof(double2)));
    dot_pair_kernel<<<blocks, 256, 4 * 256 * sizeof(double)>>>(
        a1, b1, a2, b2, n, d_sums1, d_sums2);
    CUDA_CHECK(cudaGetLastError());
    *dot1 = sum_device_complex_blocks(d_sums1, blocks);
    *dot2 = sum_device_complex_blocks(d_sums2, blocks);
    CUDA_CHECK(cudaFree(d_sums1));
    CUDA_CHECK(cudaFree(d_sums2));
}
