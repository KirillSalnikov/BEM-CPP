#include "precond.h"
#include "bem_fmm.h"
#include "gpu_select.h"
#include <cstdio>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <tuple>
#include <unordered_map>

namespace {
constexpr int kMaxDeviceBlockDim = 256;

__global__ void mass_jacobi_init_kernel(
    int n, const double* inv_diag, const double2* rhs, double2* x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double scale = inv_diag[i];
    x[i] = make_double2(scale * rhs[i].x, scale * rhs[i].y);
    x[n + i] = make_double2(scale * rhs[n + i].x, scale * rhs[n + i].y);
}

__global__ void mass_spmv_pair_kernel(
    int n, const int* row_ptr, const int* col_idx, const double* val,
    const double2* x, double2* y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    double2 sum0 = make_double2(0.0, 0.0);
    double2 sum1 = make_double2(0.0, 0.0);
    for (int pos = row_ptr[row]; pos < row_ptr[row + 1]; pos++) {
        const int col = col_idx[pos];
        const double a = val[pos];
        sum0.x += a * x[col].x;
        sum0.y += a * x[col].y;
        sum1.x += a * x[n + col].x;
        sum1.y += a * x[n + col].y;
    }
    y[row] = sum0;
    y[n + row] = sum1;
}

__global__ void mass_residual_init_kernel(
    int n, const double2* rhs, const double2* ax, double2* r, double2* p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * n) return;
    const double2 value = make_double2(rhs[i].x - ax[i].x,
                                       rhs[i].y - ax[i].y);
    r[i] = value;
    p[i] = value;
}

__global__ void mass_cg_update_kernel(
    int n, double alpha0, double alpha1,
    const double2* p, const double2* ap, double2* x, double2* r)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double2 pi = p[i], api = ap[i];
    x[i].x += alpha0 * pi.x;
    x[i].y += alpha0 * pi.y;
    r[i].x -= alpha0 * api.x;
    r[i].y -= alpha0 * api.y;
    pi = p[n + i];
    api = ap[n + i];
    x[n + i].x += alpha1 * pi.x;
    x[n + i].y += alpha1 * pi.y;
    r[n + i].x -= alpha1 * api.x;
    r[n + i].y -= alpha1 * api.y;
}

__global__ void mass_cg_direction_kernel(
    int n, double beta0, double beta1, const double2* r, double2* p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i] = make_double2(r[i].x + beta0 * p[i].x,
                        r[i].y + beta0 * p[i].y);
    p[n + i] = make_double2(r[n + i].x + beta1 * p[n + i].x,
                            r[n + i].y + beta1 * p[n + i].y);
}

__global__ void mass_norm_pair_kernel(
    const double2* x0, const double2* x1, int n,
    double* block_sums0, double* block_sums1)
{
    extern __shared__ double shared[];
    double* sum0 = shared;
    double* sum1 = shared + blockDim.x;
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    double value0 = 0.0;
    double value1 = 0.0;
    if (i < n) {
        const double2 v0 = x0[i];
        const double2 v1 = x1[i];
        value0 = v0.x * v0.x + v0.y * v0.y;
        value1 = v1.x * v1.x + v1.y * v1.y;
    }
    sum0[tid] = value0;
    sum1[tid] = value1;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum0[tid] += sum0[tid + stride];
            sum1[tid] += sum1[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_sums0[blockIdx.x] = sum0[0];
        block_sums1[blockIdx.x] = sum1[0];
    }
}

__global__ void mass_dot_pair_kernel(
    const double2* a0, const double2* b0,
    const double2* a1, const double2* b1, int n,
    double2* block_sums0, double2* block_sums1)
{
    extern __shared__ double shared[];
    double* sum0_re = shared;
    double* sum0_im = shared + blockDim.x;
    double* sum1_re = shared + 2 * blockDim.x;
    double* sum1_im = shared + 3 * blockDim.x;
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    double value0_re = 0.0, value0_im = 0.0;
    double value1_re = 0.0, value1_im = 0.0;
    if (i < n) {
        const double2 av0 = a0[i];
        const double2 bv0 = b0[i];
        const double2 av1 = a1[i];
        const double2 bv1 = b1[i];
        value0_re = av0.x * bv0.x + av0.y * bv0.y;
        value0_im = av0.x * bv0.y - av0.y * bv0.x;
        value1_re = av1.x * bv1.x + av1.y * bv1.y;
        value1_im = av1.x * bv1.y - av1.y * bv1.x;
    }
    sum0_re[tid] = value0_re;
    sum0_im[tid] = value0_im;
    sum1_re[tid] = value1_re;
    sum1_im[tid] = value1_im;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum0_re[tid] += sum0_re[tid + stride];
            sum0_im[tid] += sum0_im[tid + stride];
            sum1_re[tid] += sum1_re[tid + stride];
            sum1_im[tid] += sum1_im[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_sums0[blockIdx.x] = make_double2(sum0_re[0], sum0_im[0]);
        block_sums1[blockIdx.x] = make_double2(sum1_re[0], sum1_im[0]);
    }
}

static void mass_norm_pair(const NearFieldPrecond& precond,
                           const double2* x0, const double2* x1,
                           double* norm0, double* norm1)
{
    constexpr int block = 256;
    mass_norm_pair_kernel<<<precond.mass_reduction_blocks, block,
                            2 * block * sizeof(double)>>>(
        x0, x1, precond.N, precond.d_mass_norm_sum0,
        precond.d_mass_norm_sum1);
    CUDA_CHECK(cudaGetLastError());
    const size_t bytes = (size_t)precond.mass_reduction_blocks * sizeof(double);
    CUDA_CHECK(cudaMemcpy(precond.mass_host_norm0.data(),
                          precond.d_mass_norm_sum0, bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(precond.mass_host_norm1.data(),
                          precond.d_mass_norm_sum1, bytes,
                          cudaMemcpyDeviceToHost));
    double sum0 = 0.0;
    double sum1 = 0.0;
    for (int i = 0; i < precond.mass_reduction_blocks; i++) {
        sum0 += precond.mass_host_norm0[i];
        sum1 += precond.mass_host_norm1[i];
    }
    *norm0 = std::sqrt(std::max(0.0, sum0));
    *norm1 = std::sqrt(std::max(0.0, sum1));
}

static void mass_dot_pair(const NearFieldPrecond& precond,
                          const double2* a0, const double2* b0,
                          const double2* a1, const double2* b1,
                          double2* dot0, double2* dot1)
{
    constexpr int block = 256;
    mass_dot_pair_kernel<<<precond.mass_reduction_blocks, block,
                           4 * block * sizeof(double)>>>(
        a0, b0, a1, b1, precond.N, precond.d_mass_dot_sum0,
        precond.d_mass_dot_sum1);
    CUDA_CHECK(cudaGetLastError());
    const size_t bytes = (size_t)precond.mass_reduction_blocks * sizeof(double2);
    CUDA_CHECK(cudaMemcpy(precond.mass_host_dot0.data(),
                          precond.d_mass_dot_sum0, bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(precond.mass_host_dot1.data(),
                          precond.d_mass_dot_sum1, bytes,
                          cudaMemcpyDeviceToHost));
    double2 sum0 = make_double2(0.0, 0.0);
    double2 sum1 = make_double2(0.0, 0.0);
    for (int i = 0; i < precond.mass_reduction_blocks; i++) {
        sum0.x += precond.mass_host_dot0[i].x;
        sum0.y += precond.mass_host_dot0[i].y;
        sum1.x += precond.mass_host_dot1[i].x;
        sum1.y += precond.mass_host_dot1[i].y;
    }
    *dot0 = sum0;
    *dot1 = sum1;
}

__global__ void precond_split_complex_kernel(const double2* in, double* out_re, double* out_im, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double2 v = in[i];
    out_re[i] = v.x;
    out_im[i] = v.y;
}

__global__ void precond_pack_complex_kernel(const double* in_re, const double* in_im, double2* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = make_double2(in_re[i], in_im[i]);
}

__device__ inline void c_mul(double ar, double ai, double br, double bi, double& cr, double& ci)
{
    cr = ar * br - ai * bi;
    ci = ar * bi + ai * br;
}

__device__ inline void c_div(double ar, double ai, double br, double bi, double& cr, double& ci)
{
    double den = br * br + bi * bi;
    cr = (ar * br + ai * bi) / den;
    ci = (ai * br - ar * bi) / den;
}

__global__ void precond_schwarz_kernel(
    int n_blocks, int N,
    int block_dim,
    const int* offsets, const int* ids, const int* piv,
    const double* lu_re, const double* lu_im,
    const double* weight,
    const double* r_re, const double* r_im,
    double* z_re, double* z_im)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    int off = offsets[b];
    int next = offsets[b + 1];
    int nb = next - off;
    int nd = 2 * nb;
    double xr[kMaxDeviceBlockDim], xi[kMaxDeviceBlockDim];

    for (int i = 0; i < nb; i++) {
        int id = ids[off + i];
        xr[2*i] = r_re[id];
        xi[2*i] = r_im[id];
        xr[2*i + 1] = r_re[N + id];
        xi[2*i + 1] = r_im[N + id];
    }

    int lu_base = b * block_dim * block_dim;
    int piv_base = b * block_dim;
    for (int k = 0; k < nd; k++) {
        int p = piv[piv_base + k];
        if (p != k) {
            double tr = xr[k], ti = xi[k];
            xr[k] = xr[p]; xi[k] = xi[p];
            xr[p] = tr; xi[p] = ti;
        }
    }
    for (int k = 0; k < nd; k++) {
        for (int i = k + 1; i < nd; i++) {
            int a = lu_base + i * block_dim + k;
            double mr, mi;
            c_mul(lu_re[a], lu_im[a], xr[k], xi[k], mr, mi);
            xr[i] -= mr;
            xi[i] -= mi;
        }
    }
    for (int i = nd - 1; i >= 0; i--) {
        double sr = xr[i], si = xi[i];
        for (int j = i + 1; j < nd; j++) {
            int a = lu_base + i * block_dim + j;
            double mr, mi;
            c_mul(lu_re[a], lu_im[a], xr[j], xi[j], mr, mi);
            sr -= mr;
            si -= mi;
        }
        int diag = lu_base + i * block_dim + i;
        c_div(sr, si, lu_re[diag], lu_im[diag], xr[i], xi[i]);
    }

    for (int i = 0; i < nb; i++) {
        int id = ids[off + i];
        double w = weight[id];
        atomicAdd(&z_re[id], xr[2*i] / w);
        atomicAdd(&z_im[id], xi[2*i] / w);
        atomicAdd(&z_re[N + id], xr[2*i + 1] / w);
        atomicAdd(&z_im[N + id], xi[2*i + 1] / w);
    }
}

__global__ void precond_mbj_kernel(
    int n_blocks, int N, int block_dim,
    const int* offsets, const int* ids, const int* piv,
    const double* lu_re, const double* lu_im,
    const double* r_re, const double* r_im,
    double* z_re, double* z_im)
{
    const int b = blockIdx.x;
    if (b >= n_blocks) return;
    const int tid = threadIdx.x;
    const int off = offsets[b];
    const int nb = offsets[b + 1] - off;
    const int nd = 2 * nb;
    const int lu_base = b * block_dim * block_dim;
    const int piv_base = b * block_dim;

    extern __shared__ double shared[];
    double* xr = shared;
    double* xi = xr + block_dim;
    double* sum_re = xi + block_dim;
    double* sum_im = sum_re + blockDim.x;

    for (int i = tid; i < nb; i += blockDim.x) {
        const int id = ids[off + i];
        xr[2 * i] = r_re[id];
        xi[2 * i] = r_im[id];
        xr[2 * i + 1] = r_re[N + id];
        xi[2 * i + 1] = r_im[N + id];
    }
    __syncthreads();

    for (int k = 0; k < nd; k++) {
        if (tid == 0) {
            const int p = piv[piv_base + k];
            if (p != k) {
                const double tr = xr[k], ti = xi[k];
                xr[k] = xr[p];
                xi[k] = xi[p];
                xr[p] = tr;
                xi[p] = ti;
            }
        }
        __syncthreads();
    }
    for (int k = 0; k < nd; k++) {
        const double xkr = xr[k], xki = xi[k];
        for (int i = k + 1 + tid; i < nd; i += blockDim.x) {
            const int a = lu_base + i * block_dim + k;
            double mr, mi;
            c_mul(lu_re[a], lu_im[a], xkr, xki, mr, mi);
            xr[i] -= mr;
            xi[i] -= mi;
        }
        __syncthreads();
    }

    for (int k = nd - 1; k >= 0; k--) {
        double local_re = 0.0;
        double local_im = 0.0;
        for (int j = k + 1 + tid; j < nd; j += blockDim.x) {
            const int a = lu_base + k * block_dim + j;
            double mr, mi;
            c_mul(lu_re[a], lu_im[a], xr[j], xi[j], mr, mi);
            local_re += mr;
            local_im += mi;
        }
        sum_re[tid] = local_re;
        sum_im[tid] = local_im;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sum_re[tid] += sum_re[tid + stride];
                sum_im[tid] += sum_im[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int diag = lu_base + k * block_dim + k;
            c_div(
                xr[k] - sum_re[0], xi[k] - sum_im[0],
                lu_re[diag], lu_im[diag], xr[k], xi[k]);
        }
        __syncthreads();
    }

    for (int i = tid; i < nb; i += blockDim.x) {
        const int id = ids[off + i];
        z_re[id] = xr[2 * i];
        z_im[id] = xi[2 * i];
        z_re[N + id] = xr[2 * i + 1];
        z_im[N + id] = xi[2 * i + 1];
    }
}

__global__ void precond_near_matvec_kernel(
    int N,
    const int* row_ptr, const int* col_idx,
    const double* diag_re, const double* diag_im,
    const double* near_re, const double* near_im,
    const double* x_re, const double* x_im,
    double* y_re, double* y_im)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;

    double xm_r = x_re[m], xm_i = x_im[m];
    double xN_r = x_re[N + m], xN_i = x_im[N + m];
    double yr, yi, tr, ti;

    int db = 4 * m;
    c_mul(diag_re[db], diag_im[db], xm_r, xm_i, yr, yi);
    c_mul(diag_re[db + 1], diag_im[db + 1], xN_r, xN_i, tr, ti);
    yr += tr; yi += ti;

    double yNr, yNi;
    c_mul(diag_re[db + 2], diag_im[db + 2], xm_r, xm_i, yNr, yNi);
    c_mul(diag_re[db + 3], diag_im[db + 3], xN_r, xN_i, tr, ti);
    yNr += tr; yNi += ti;

    for (int jc = row_ptr[m]; jc < row_ptr[m + 1]; jc++) {
        int n = col_idx[jc];
        double xn_r = x_re[n], xn_i = x_im[n];
        double xNn_r = x_re[N + n], xNn_i = x_im[N + n];
        int nb = 4 * jc;
        c_mul(near_re[nb], near_im[nb], xn_r, xn_i, tr, ti);
        yr += tr; yi += ti;
        c_mul(near_re[nb + 1], near_im[nb + 1], xNn_r, xNn_i, tr, ti);
        yr += tr; yi += ti;
        c_mul(near_re[nb + 2], near_im[nb + 2], xn_r, xn_i, tr, ti);
        yNr += tr; yNi += ti;
        c_mul(near_re[nb + 3], near_im[nb + 3], xNn_r, xNn_i, tr, ti);
        yNr += tr; yNi += ti;
    }
    y_re[m] = yr; y_im[m] = yi;
    y_re[N + m] = yNr; y_im[N + m] = yNi;
}

__global__ void neural_sparse_inverse_kernel(
    int N, const int* row_ptr, const int* col_idx,
    const double2* blocks, const double2* r, double2* z)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;

    double2 out_j = make_double2(0.0, 0.0);
    double2 out_m = make_double2(0.0, 0.0);
    for (int jc = row_ptr[m]; jc < row_ptr[m + 1]; jc++) {
        int n = col_idx[jc];
        double2 rj = r[n];
        double2 rm = r[N + n];
        double tr, ti;
        int base = 4 * jc;
        c_mul(blocks[base].x, blocks[base].y, rj.x, rj.y, tr, ti);
        out_j.x += tr; out_j.y += ti;
        c_mul(blocks[base + 1].x, blocks[base + 1].y, rm.x, rm.y, tr, ti);
        out_j.x += tr; out_j.y += ti;
        c_mul(blocks[base + 2].x, blocks[base + 2].y, rj.x, rj.y, tr, ti);
        out_m.x += tr; out_m.y += ti;
        c_mul(blocks[base + 3].x, blocks[base + 3].y, rm.x, rm.y, tr, ti);
        out_m.x += tr; out_m.y += ti;
    }
    z[m] = out_j;
    z[N + m] = out_m;
}

__global__ void neural_coarse_project_kernel(
    int n, const float2* q, const double2* r, double2* coeff)
{
    extern __shared__ double2 partial[];
    int k = blockIdx.x;
    double sum_re = 0.0, sum_im = 0.0;
    const float2* qk = q + (size_t)k * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float2 qi_float = qk[i];
        double2 qi = make_double2((double)qi_float.x, (double)qi_float.y);
        double2 ri = r[i];
        // conj(q_i) * r_i
        sum_re += qi.x * ri.x + qi.y * ri.y;
        sum_im += qi.x * ri.y - qi.y * ri.x;
    }
    partial[threadIdx.x] = make_double2(sum_re, sum_im);
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x].x += partial[threadIdx.x + stride].x;
            partial[threadIdx.x].y += partial[threadIdx.x + stride].y;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        coeff[k] = partial[0];
}

__global__ void neural_coarse_update_kernel(
    int n, int rank, const float2* update, const double2* coeff, double2* z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double out_re = z[i].x, out_im = z[i].y;
    for (int k = 0; k < rank; k++) {
        float2 value_float = update[(size_t)k * n + i];
        double2 value = make_double2((double)value_float.x, (double)value_float.y);
        double2 alpha = coeff[k];
        out_re += value.x * alpha.x - value.y * alpha.y;
        out_im += value.x * alpha.y + value.y * alpha.x;
    }
    z[i] = make_double2(out_re, out_im);
}

__global__ void precond_residual_kernel(
    const double* r_re, const double* r_im,
    const double* Az_re, const double* Az_im,
    double* err_re, double* err_im, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    err_re[i] = r_re[i] - Az_re[i];
    err_im[i] = r_im[i] - Az_im[i];
}

__global__ void precond_axpy_kernel(double* z_re, double* z_im,
                                    const double* corr_re, const double* corr_im,
                                    double omega, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    z_re[i] += omega * corr_re[i];
    z_im[i] += omega * corr_im[i];
}

} // namespace

static bool lu_factor_small(std::vector<cdouble>& A, std::vector<int>& piv, int n)
{
    piv.resize(n);
    for (int k = 0; k < n; k++) {
        int p = k;
        double best = std::abs(A[k*n + k]);
        for (int i = k + 1; i < n; i++) {
            double v = std::abs(A[i*n + k]);
            if (v > best) {
                best = v;
                p = i;
            }
        }
        if (best < 1e-24)
            return false;
        piv[k] = p;
        if (p != k) {
            for (int j = 0; j < n; j++)
                std::swap(A[k*n + j], A[p*n + j]);
        }
        cdouble diag = A[k*n + k];
        for (int i = k + 1; i < n; i++) {
            cdouble f = A[i*n + k] / diag;
            A[i*n + k] = f;
            for (int j = k + 1; j < n; j++)
                A[i*n + j] -= f * A[k*n + j];
        }
    }
    return true;
}

static void lu_solve_small(const std::vector<cdouble>& LU, const std::vector<int>& piv,
                           const cdouble* b, cdouble* x, int n)
{
    for (int i = 0; i < n; i++)
        x[i] = b[i];
    for (int k = 0; k < n; k++) {
        int p = piv[k];
        if (p != k)
            std::swap(x[k], x[p]);
    }
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++)
            x[i] -= LU[i*n + k] * x[k];
    }
    for (int i = n - 1; i >= 0; i--) {
        cdouble s = x[i];
        for (int j = i + 1; j < n; j++)
            s -= LU[i*n + j] * x[j];
        x[i] = s / LU[i*n + i];
    }
}

static int find_csr_col(const std::vector<int>& row_ptr, const std::vector<int>& col_idx,
                        int row, int col)
{
    for (int jc = row_ptr[row]; jc < row_ptr[row + 1]; jc++) {
        if (col_idx[jc] == col)
            return jc;
    }
    return -1;
}

static void push_unique(std::vector<int>& ids, int id)
{
    if (std::find(ids.begin(), ids.end(), id) == ids.end())
        ids.push_back(id);
}

static bool shared_edge_ids(const RWG& rwg, const Mesh& mesh, int edge,
                            int& first, int& second)
{
    int plus[3] = {
        mesh.tris[3 * rwg.tri_p[edge]],
        mesh.tris[3 * rwg.tri_p[edge] + 1],
        mesh.tris[3 * rwg.tri_p[edge] + 2]
    };
    int minus[3] = {
        mesh.tris[3 * rwg.tri_m[edge]],
        mesh.tris[3 * rwg.tri_m[edge] + 1],
        mesh.tris[3 * rwg.tri_m[edge] + 2]
    };
    int found[2] = {-1, -1};
    int count = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (plus[i] == minus[j] && count < 2) {
                found[count++] = plus[i];
                break;
            }
        }
    }
    if (count != 2)
        return false;
    first = std::min(found[0], found[1]);
    second = std::max(found[0], found[1]);
    return true;
}

static uint64_t morton_code_3d(uint32_t x, uint32_t y, uint32_t z)
{
    uint64_t code = 0;
    for (int bit = 0; bit < 21; bit++) {
        code |= (uint64_t)((x >> bit) & 1u) << (3 * bit);
        code |= (uint64_t)((y >> bit) & 1u) << (3 * bit + 1);
        code |= (uint64_t)((z >> bit) & 1u) << (3 * bit + 2);
    }
    return code;
}

uint64_t bem_neural_geometry_signature(const RWG& rwg)
{
    uint64_t hash = 1469598103934665603ULL;
    const uint64_t prime = 1099511628211ULL;
    for (int i = 0; i < rwg.N; i++) {
        long long values[3] = {
            (long long)std::llround(rwg.length[i] * 1e12),
            (long long)std::llround(rwg.area_p[i] * 1e12),
            (long long)std::llround(rwg.area_m[i] * 1e12)
        };
        for (int k = 0; k < 3; k++) {
            hash ^= (uint64_t)values[k];
            hash *= prime;
        }
    }
    return hash;
}

static void bem_cusparse_check(cusparseStatus_t status, const char* call,
                               const char* file, int line)
{
    if (status == CUSPARSE_STATUS_SUCCESS)
        return;
    fprintf(stderr, "cuSPARSE error %d in %s at %s:%d\n",
            (int)status, call, file, line);
    std::abort();
}

#define BEM_CUSPARSE_CHECK(call) \
    bem_cusparse_check((call), #call, __FILE__, __LINE__)

static void build_ilu0_factors(NearFieldPrecond& precond)
{
    Timer timer;
    const int N = precond.N;
    const int N2 = precond.N2;
    std::vector<std::vector<int>> block_cols(N);
    size_t block_nnz = 0;
    for (int row = 0; row < N; row++) {
        std::vector<int>& cols = block_cols[row];
        cols.assign(precond.near_col_idx.begin() + precond.near_row_ptr[row],
                    precond.near_col_idx.begin() + precond.near_row_ptr[row + 1]);
        cols.push_back(row);
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        block_nnz += cols.size();
    }

    const size_t scalar_nnz = 4 * block_nnz;
    if (scalar_nnz > (size_t)std::numeric_limits<int>::max()) {
        fprintf(stderr, "Error: ILU(0) sparse graph is too large\n");
        std::abort();
    }
    precond.ilu_row_ptr.assign((size_t)N2 + 1, 0);
    for (int row = 0; row < N; row++) {
        const int width = 2 * (int)block_cols[row].size();
        precond.ilu_row_ptr[row + 1] = precond.ilu_row_ptr[row] + width;
    }
    for (int row = 0; row < N; row++) {
        const int scalar_row = N + row;
        precond.ilu_row_ptr[scalar_row + 1] =
            precond.ilu_row_ptr[scalar_row] + 2 * (int)block_cols[row].size();
    }
    precond.ilu_col_idx.resize(scalar_nnz);
    precond.ilu_val.assign(scalar_nnz, cdouble(0.0));
    precond.ilu_diag_ptr.assign(N2, -1);

    for (int row = 0; row < N; row++) {
        const std::vector<int>& cols = block_cols[row];
        const int width = (int)cols.size();
        const int base_j = precond.ilu_row_ptr[row];
        const int base_m = precond.ilu_row_ptr[N + row];
        for (int j = 0; j < width; j++) {
            const int col = cols[j];
            cdouble A(0.0), B(0.0), C(0.0), D(0.0);
            if (col == row) {
                A = precond.diag_blk[4 * row];
                B = precond.diag_blk[4 * row + 1];
                C = precond.diag_blk[4 * row + 2];
                D = precond.diag_blk[4 * row + 3];
            } else {
                const int pos = find_csr_col(precond.near_row_ptr,
                                             precond.near_col_idx, row, col);
                if (pos >= 0) {
                    A = precond.near_blk[4 * pos];
                    B = precond.near_blk[4 * pos + 1];
                    C = precond.near_blk[4 * pos + 2];
                    D = precond.near_blk[4 * pos + 3];
                }
            }
            precond.ilu_col_idx[base_j + j] = col;
            precond.ilu_val[base_j + j] = A;
            precond.ilu_col_idx[base_j + width + j] = N + col;
            precond.ilu_val[base_j + width + j] = B;
            precond.ilu_col_idx[base_m + j] = col;
            precond.ilu_val[base_m + j] = C;
            precond.ilu_col_idx[base_m + width + j] = N + col;
            precond.ilu_val[base_m + width + j] = D;
            if (col == row) {
                precond.ilu_diag_ptr[row] = base_j + j;
                precond.ilu_diag_ptr[N + row] = base_m + width + j;
            }
        }
    }

    std::vector<std::unordered_map<int, int>> lookup(N2);
    for (int row = 0; row < N2; row++) {
        const int begin = precond.ilu_row_ptr[row];
        const int end = precond.ilu_row_ptr[row + 1];
        lookup[row].reserve((size_t)(end - begin));
        for (int pos = begin; pos < end; pos++)
            lookup[row][precond.ilu_col_idx[pos]] = pos;
    }

    int shifted_diagonals = 0;
    const double pivot_floor = 1e-14;
    for (int row = 0; row < N2; row++) {
        const int begin = precond.ilu_row_ptr[row];
        const int end = precond.ilu_row_ptr[row + 1];
        for (int pos = begin; pos < end; pos++) {
            const int pivot_row = precond.ilu_col_idx[pos];
            if (pivot_row >= row)
                break;
            cdouble pivot = precond.ilu_val[precond.ilu_diag_ptr[pivot_row]];
            if (std::abs(pivot) < pivot_floor) {
                pivot += cdouble(pivot_floor, 0.0);
                precond.ilu_val[precond.ilu_diag_ptr[pivot_row]] = pivot;
                shifted_diagonals++;
            }
            const cdouble multiplier = precond.ilu_val[pos] / pivot;
            precond.ilu_val[pos] = multiplier;
            for (int target = pos + 1; target < end; target++) {
                const int col = precond.ilu_col_idx[target];
                const auto found = lookup[pivot_row].find(col);
                if (found != lookup[pivot_row].end())
                    precond.ilu_val[target] -= multiplier * precond.ilu_val[found->second];
            }
        }
        const int diag = precond.ilu_diag_ptr[row];
        if (diag < 0) {
            fprintf(stderr, "Error: ILU(0) diagonal is missing in row %d\n", row);
            std::abort();
        }
        if (std::abs(precond.ilu_val[diag]) < pivot_floor) {
            precond.ilu_val[diag] += cdouble(pivot_floor, 0.0);
            shifted_diagonals++;
        }
    }
    printf("  [Precond] ILU(0) factorized: rows=%d nnz=%zu avg=%.1f shifts=%d time=%.2fs\n",
           N2, scalar_nnz, (double)scalar_nnz / (double)N2,
           shifted_diagonals, timer.elapsed_s());
}

static void upload_ilu0_factors(NearFieldPrecond& precond)
{
    const int n = precond.N2;
    const int nnz = (int)precond.ilu_val.size();
    std::vector<double2> values((size_t)nnz);
    for (int i = 0; i < nnz; i++)
        values[i] = make_double2(precond.ilu_val[i].real(), precond.ilu_val[i].imag());

    CUDA_CHECK(cudaMalloc(&precond.d_ilu_row_ptr, ((size_t)n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&precond.d_ilu_col_idx, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&precond.d_ilu_val, (size_t)nnz * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&precond.d_ilu_rhs, (size_t)n * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&precond.d_ilu_tmp, (size_t)n * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&precond.d_ilu_out, (size_t)n * sizeof(double2)));
    CUDA_CHECK(cudaMemcpy(precond.d_ilu_row_ptr, precond.ilu_row_ptr.data(),
                          ((size_t)n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(precond.d_ilu_col_idx, precond.ilu_col_idx.data(),
                          (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(precond.d_ilu_val, values.data(),
                          (size_t)nnz * sizeof(double2), cudaMemcpyHostToDevice));

    BEM_CUSPARSE_CHECK(cusparseCreate(&precond.ilu_handle));
    BEM_CUSPARSE_CHECK(cusparseCreateCsr(
        &precond.ilu_mat_l, n, n, nnz,
        precond.d_ilu_row_ptr, precond.d_ilu_col_idx, precond.d_ilu_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_C_64F));
    BEM_CUSPARSE_CHECK(cusparseCreateCsr(
        &precond.ilu_mat_u, n, n, nnz,
        precond.d_ilu_row_ptr, precond.d_ilu_col_idx, precond.d_ilu_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_C_64F));
    cusparseFillMode_t lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseFillMode_t upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseDiagType_t non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    BEM_CUSPARSE_CHECK(cusparseSpMatSetAttribute(
        precond.ilu_mat_l, CUSPARSE_SPMAT_FILL_MODE, &lower, sizeof(lower)));
    BEM_CUSPARSE_CHECK(cusparseSpMatSetAttribute(
        precond.ilu_mat_l, CUSPARSE_SPMAT_DIAG_TYPE, &unit, sizeof(unit)));
    BEM_CUSPARSE_CHECK(cusparseSpMatSetAttribute(
        precond.ilu_mat_u, CUSPARSE_SPMAT_FILL_MODE, &upper, sizeof(upper)));
    BEM_CUSPARSE_CHECK(cusparseSpMatSetAttribute(
        precond.ilu_mat_u, CUSPARSE_SPMAT_DIAG_TYPE, &non_unit, sizeof(non_unit)));

    BEM_CUSPARSE_CHECK(cusparseCreateDnVec(
        &precond.ilu_vec_in, n, precond.d_ilu_rhs, CUDA_C_64F));
    BEM_CUSPARSE_CHECK(cusparseCreateDnVec(
        &precond.ilu_vec_tmp, n, precond.d_ilu_tmp, CUDA_C_64F));
    BEM_CUSPARSE_CHECK(cusparseCreateDnVec(
        &precond.ilu_vec_out, n, precond.d_ilu_out, CUDA_C_64F));
    BEM_CUSPARSE_CHECK(cusparseSpSV_createDescr(&precond.ilu_spsv_l));
    BEM_CUSPARSE_CHECK(cusparseSpSV_createDescr(&precond.ilu_spsv_u));

    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    size_t bytes_l = 0, bytes_u = 0;
    BEM_CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        precond.ilu_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
        precond.ilu_mat_l, precond.ilu_vec_in, precond.ilu_vec_tmp,
        CUDA_C_64F, CUSPARSE_SPSV_ALG_DEFAULT, precond.ilu_spsv_l, &bytes_l));
    BEM_CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        precond.ilu_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
        precond.ilu_mat_u, precond.ilu_vec_tmp, precond.ilu_vec_out,
        CUDA_C_64F, CUSPARSE_SPSV_ALG_DEFAULT, precond.ilu_spsv_u, &bytes_u));
    if (bytes_l > 0)
        CUDA_CHECK(cudaMalloc(&precond.d_ilu_buffer_l, bytes_l));
    if (bytes_u > 0)
        CUDA_CHECK(cudaMalloc(&precond.d_ilu_buffer_u, bytes_u));
    BEM_CUSPARSE_CHECK(cusparseSpSV_analysis(
        precond.ilu_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
        precond.ilu_mat_l, precond.ilu_vec_in, precond.ilu_vec_tmp,
        CUDA_C_64F, CUSPARSE_SPSV_ALG_DEFAULT, precond.ilu_spsv_l,
        precond.d_ilu_buffer_l));
    BEM_CUSPARSE_CHECK(cusparseSpSV_analysis(
        precond.ilu_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
        precond.ilu_mat_u, precond.ilu_vec_tmp, precond.ilu_vec_out,
        CUDA_C_64F, CUSPARSE_SPSV_ALG_DEFAULT, precond.ilu_spsv_u,
        precond.d_ilu_buffer_u));
    precond.device_ready = true;
    printf("  [Precond] ILU(0) uploaded to GPU: %.1f MiB CSR\n",
           ((double)nnz * (sizeof(int) + sizeof(double2)) +
            (double)(n + 1) * sizeof(int)) / (1024.0 * 1024.0));
}

static bool build_mass_matrix(NearFieldPrecond& precond, const BemFmmOperator& op,
                              const RWG& rwg, const Mesh& mesh)
{
    struct HalfBasis {
        int edge;
        bool plus;
    };
    Timer timer;
    const int N = rwg.N;
    const int Nq = op.Nq;
    std::vector<std::vector<HalfBasis>> triangle_basis(mesh.nt());
    for (int edge = 0; edge < N; edge++) {
        triangle_basis[rwg.tri_p[edge]].push_back({edge, true});
        triangle_basis[rwg.tri_m[edge]].push_back({edge, false});
    }

    std::vector<std::map<int, double>> rows(N);
    for (int triangle = 0; triangle < mesh.nt(); triangle++) {
        const std::vector<HalfBasis>& local = triangle_basis[triangle];
        if (local.size() != 3) {
            fprintf(stderr,
                    "Error: mass matrix requires three RWG functions per closed triangle; "
                    "triangle %d has %zu\n", triangle, local.size());
            return false;
        }
        for (const HalfBasis& test : local) {
            const double* ft = test.plus ?
                &op.f_p[(size_t)test.edge * Nq * 3] :
                &op.f_m[(size_t)test.edge * Nq * 3];
            const double* weights = test.plus ?
                &op.jw_p[(size_t)test.edge * Nq] :
                &op.jw_m[(size_t)test.edge * Nq];
            for (const HalfBasis& trial : local) {
                const double* fs = trial.plus ?
                    &op.f_p[(size_t)trial.edge * Nq * 3] :
                    &op.f_m[(size_t)trial.edge * Nq * 3];
                double value = 0.0;
                for (int q = 0; q < Nq; q++) {
                    value += weights[q] *
                        (ft[3*q] * fs[3*q] +
                         ft[3*q + 1] * fs[3*q + 1] +
                         ft[3*q + 2] * fs[3*q + 2]);
                }
                rows[test.edge][trial.edge] += value;
            }
        }
    }

    precond.mass_row_ptr.assign((size_t)N + 1, 0);
    for (int row = 0; row < N; row++)
        precond.mass_row_ptr[row + 1] =
            precond.mass_row_ptr[row] + (int)rows[row].size();
    const int nnz = precond.mass_row_ptr[N];
    precond.mass_col_idx.resize(nnz);
    precond.mass_val.resize(nnz);
    precond.mass_inv_diag.assign(N, 0.0);
    double max_asymmetry = 0.0;
    double min_diag = std::numeric_limits<double>::infinity();
    double max_diag = 0.0;
    for (int row = 0; row < N; row++) {
        int pos = precond.mass_row_ptr[row];
        for (const auto& entry : rows[row]) {
            precond.mass_col_idx[pos] = entry.first;
            precond.mass_val[pos] = entry.second;
            const auto transpose = rows[entry.first].find(row);
            if (transpose != rows[entry.first].end())
                max_asymmetry = std::max(max_asymmetry,
                                         std::abs(entry.second - transpose->second));
            if (entry.first == row) {
                if (!(entry.second > 0.0) || !std::isfinite(entry.second)) {
                    fprintf(stderr, "Error: invalid mass diagonal in row %d: %.6e\n",
                            row, entry.second);
                    return false;
                }
                precond.mass_inv_diag[row] = 1.0 / entry.second;
                min_diag = std::min(min_diag, entry.second);
                max_diag = std::max(max_diag, entry.second);
            }
            pos++;
        }
        if (precond.mass_inv_diag[row] == 0.0) {
            fprintf(stderr, "Error: missing mass diagonal in row %d\n", row);
            return false;
        }
    }

    precond.mass_cg_tolerance = std::max(
        1e-14, bem_env_double("BEM_MASS_TOL", precond.mass_cg_tolerance));
    precond.mass_cg_max_iterations = std::max(
        1, bem_env_int("BEM_MASS_MAX_ITERS", precond.mass_cg_max_iterations));

    CUDA_CHECK(cudaMalloc(&precond.d_mass_row_ptr, ((size_t)N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&precond.d_mass_col_idx, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&precond.d_mass_val, (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&precond.d_mass_inv_diag, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&precond.d_mass_x, (size_t)2 * N * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&precond.d_mass_r, (size_t)2 * N * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&precond.d_mass_p, (size_t)2 * N * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&precond.d_mass_ap, (size_t)2 * N * sizeof(double2)));
    precond.mass_reduction_blocks = (N + 255) / 256;
    const size_t reduction_count = (size_t)precond.mass_reduction_blocks;
    CUDA_CHECK(cudaMalloc(&precond.d_mass_norm_sum0,
                          reduction_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&precond.d_mass_norm_sum1,
                          reduction_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&precond.d_mass_dot_sum0,
                          reduction_count * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&precond.d_mass_dot_sum1,
                          reduction_count * sizeof(double2)));
    precond.mass_host_norm0.resize(reduction_count);
    precond.mass_host_norm1.resize(reduction_count);
    precond.mass_host_dot0.resize(reduction_count);
    precond.mass_host_dot1.resize(reduction_count);
    CUDA_CHECK(cudaMemcpy(precond.d_mass_row_ptr, precond.mass_row_ptr.data(),
                          ((size_t)N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(precond.d_mass_col_idx, precond.mass_col_idx.data(),
                          (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(precond.d_mass_val, precond.mass_val.data(),
                          (size_t)nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(precond.d_mass_inv_diag, precond.mass_inv_diag.data(),
                          (size_t)N * sizeof(double), cudaMemcpyHostToDevice));
    precond.device_ready = true;
    printf("  [Precond] RWG mass matrix built: N=%d nnz=%d (%.1f/row), "
           "diag=[%.3e, %.3e], asym=%.3e, CG tol=%.1e max=%d, %.2fs\n",
           N, nnz, (double)nnz / (double)N, min_diag, max_diag,
           max_asymmetry, precond.mass_cg_tolerance,
           precond.mass_cg_max_iterations, timer.elapsed_s());
    return true;
}

void NearFieldPrecond::build(BemFmmOperator& op, const RWG* rwg_geometry,
                             const Mesh* mesh_geometry)
{
    Timer timer;
    N = op.N;
    N2 = 2 * N;
    neural_sparse = false;
    ilu0 = bem_env_flag_enabled("BEM_PREC_ILU0");
    morton_block_jacobi = bem_env_flag_enabled("BEM_PREC_MBJ");
    calderon_rwg = bem_env_flag_enabled("BEM_PREC_CALDERON_RWG");
    mass_matrix = bem_env_flag_enabled("BEM_PREC_MASS") || calderon_rwg;
    int Nq = op.Nq;

    if (mass_matrix) {
        ilu0 = false;
        block_schwarz = false;
        richardson_sweeps = 0;
        if (!rwg_geometry || !mesh_geometry) {
            fprintf(stderr, "Error: mass preconditioner requires RWG and mesh geometry\n");
            std::abort();
        }
        if (!build_mass_matrix(*this, op, *rwg_geometry, *mesh_geometry))
            std::abort();
        if (calderon_rwg) {
            calderon_operator = &op;
            const size_t bytes = (size_t)N2 * sizeof(double2);
            CUDA_CHECK(cudaMalloc(&d_calderon_mass0, bytes));
            CUDA_CHECK(cudaMalloc(&d_calderon_mass1, bytes));
            CUDA_CHECK(cudaMalloc(&d_calderon_op0, bytes));
            CUDA_CHECK(cudaMalloc(&d_calderon_op1, bytes));
            printf("  [Precond] Experimental RWG Calderon product ready: "
                   "G^-1 A G^-1 A (not RWG/BC Calderon)\n");
        }
        return;
    }

    if (morton_block_jacobi && op.n_form) {
        fprintf(stderr,
                "Error: RWG MBJ cannot be applied to the experimental N-form; "
                "use the dedicated nodal Muller operator\n");
        std::abort();
    }

    printf("  [Precond] Building %s preconditioner...\n",
           ilu0 ? "near-field ILU(0)" :
           (morton_block_jacobi ? "Morton block Jacobi" :
                                  "2x2 block Jacobi"));

    double inv4pi = 1.0 / (4.0 * M_PI);
    cdouble k_vals[2] = {op.k_ext, op.k_int};
    cdouble eta_e = op.eta_ext, eta_i = op.eta_int;

    richardson_sweeps = std::max(0, bem_env_int("BEM_PREC_SWEEPS", richardson_sweeps));
    richardson_omega = bem_env_double("BEM_PREC_OMEGA", richardson_omega);

    block_schwarz =
        !ilu0 && (morton_block_jacobi ||
                  bem_env_flag_enabled("BEM_PREC_BLOCK"));
    int default_block_basis = morton_block_jacobi ? 100 : max_block_basis;
    max_block_basis = std::max(
        2, bem_env_int("BEM_PREC_BLOCK_SIZE", default_block_basis));
    max_block_basis = std::min(
        max_block_basis, morton_block_jacobi ? 128 : 16);

    int near_degree =
        (block_schwarz || ilu0) ? max_block_basis : 0;
    near_degree = std::max(0, bem_env_int("BEM_PREC_NEAR", near_degree));
    int geom_near_max_n = std::max(0, bem_env_int("BEM_PREC_GEOM_NEAR_MAX_N", 12000));
    bool use_geometric_near =
        !morton_block_jacobi &&
        (block_schwarz || ilu0) && near_degree > 0 &&
        !bem_env_flag_enabled("BEM_PREC_TOPO_NEAR") &&
        (geom_near_max_n == 0 || N <= geom_near_max_n);

    std::vector<std::vector<int>> morton_blocks;
    blk_inv.resize(4 * N);
    diag_blk.resize(4 * N);
    if (morton_block_jacobi) {
        std::vector<double> centers((size_t)N * 3, 0.0);
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < N; m++) {
            if (rwg_geometry && mesh_geometry) {
                int a = -1, b = -1;
                if (shared_edge_ids(
                        *rwg_geometry, *mesh_geometry, m, a, b)) {
                    Vec3 center =
                        (mesh_geometry->verts[a] +
                         mesh_geometry->verts[b]) * 0.5;
                    centers[(size_t)m * 3] = center.x;
                    centers[(size_t)m * 3 + 1] = center.y;
                    centers[(size_t)m * 3 + 2] = center.z;
                    continue;
                }
            }
            double cx = 0.0, cy = 0.0, cz = 0.0;
            for (int q = 0; q < Nq; q++) {
                const double* qp =
                    &op.qpts_p[((size_t)m * Nq + q) * 3];
                const double* qm =
                    &op.qpts_m[((size_t)m * Nq + q) * 3];
                cx += qp[0] + qm[0];
                cy += qp[1] + qm[1];
                cz += qp[2] + qm[2];
            }
            const double inv = 0.5 / (double)Nq;
            centers[(size_t)m * 3] = cx * inv;
            centers[(size_t)m * 3 + 1] = cy * inv;
            centers[(size_t)m * 3 + 2] = cz * inv;
        }

        double lo[3] = {
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()
        };
        double hi[3] = {
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity()
        };
        for (int m = 0; m < N; m++) {
            for (int axis = 0; axis < 3; axis++) {
                const double value = centers[(size_t)m * 3 + axis];
                lo[axis] = std::min(lo[axis], value);
                hi[axis] = std::max(hi[axis], value);
            }
        }
        constexpr double morton_max = (double)((1u << 21) - 1u);
        std::vector<std::pair<uint64_t, int>> order;
        order.reserve(N);
        for (int m = 0; m < N; m++) {
            uint32_t coord[3] = {0, 0, 0};
            for (int axis = 0; axis < 3; axis++) {
                const double span = hi[axis] - lo[axis];
                const double scaled =
                    span > 0.0
                        ? (centers[(size_t)m * 3 + axis] - lo[axis]) /
                              span
                        : 0.0;
                coord[axis] = (uint32_t)std::llround(
                    std::max(0.0, std::min(1.0, scaled)) * morton_max);
            }
            order.emplace_back(
                morton_code_3d(coord[0], coord[1], coord[2]), m);
        }
        std::sort(order.begin(), order.end());
        for (int begin = 0; begin < N; begin += max_block_basis) {
            const int end = std::min(N, begin + max_block_basis);
            std::vector<int> ids;
            ids.reserve(end - begin);
            for (int i = begin; i < end; i++)
                ids.push_back(order[i].second);
            morton_blocks.push_back(std::move(ids));
        }

        std::vector<int> block_of(N, -1);
        for (int block = 0; block < (int)morton_blocks.size(); block++)
            for (int id : morton_blocks[block])
                block_of[id] = block;
        near_row_ptr.assign(N + 1, 0);
        for (int row = 0; row < N; row++) {
            const int block = block_of[row];
            near_row_ptr[row + 1] =
                near_row_ptr[row] + (int)morton_blocks[block].size();
        }
        near_col_idx.resize(near_row_ptr[N]);
        for (int row = 0; row < N; row++) {
            const std::vector<int>& ids = morton_blocks[block_of[row]];
            std::copy(
                ids.begin(), ids.end(),
                near_col_idx.begin() + near_row_ptr[row]);
        }
        printf(
            "  [Precond] Morton partition: blocks=%zu basis/block<=%d "
            "nnz=%zu (%.1f per row)\n",
            morton_blocks.size(), max_block_basis, near_col_idx.size(),
            near_col_idx.empty()
                ? 0.0
                : (double)near_col_idx.size() / (double)N);
    } else if (use_geometric_near) {
        std::vector<double> centers((size_t)N * 3, 0.0);
        std::vector<std::vector<int>> triangle_owners;
        std::vector<int> python_rwg_rank;
        if (rwg_geometry && mesh_geometry) {
            triangle_owners.resize(mesh_geometry->nt());
            python_rwg_rank.assign(N, -1);
            std::map<std::pair<int, int>, int> insertion_rank;
            int next_rank = 0;
            for (int triangle = 0; triangle < mesh_geometry->nt(); triangle++) {
                for (int local = 0; local < 3; local++) {
                    int a = mesh_geometry->tris[3 * triangle + local];
                    int b = mesh_geometry->tris[3 * triangle + (local + 1) % 3];
                    std::pair<int, int> edge(std::min(a, b), std::max(a, b));
                    if (insertion_rank.find(edge) == insertion_rank.end())
                        insertion_rank[edge] = next_rank++;
                }
            }
            for (int i = 0; i < N; i++) {
                triangle_owners[rwg_geometry->tri_p[i]].push_back(i);
                triangle_owners[rwg_geometry->tri_m[i]].push_back(i);
                int a = -1, b = -1;
                if (shared_edge_ids(*rwg_geometry, *mesh_geometry, i, a, b))
                    python_rwg_rank[i] = insertion_rank[std::make_pair(a, b)];
            }
        }
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < N; m++) {
            if (rwg_geometry && mesh_geometry) {
                int a = -1, b = -1;
                if (shared_edge_ids(*rwg_geometry, *mesh_geometry, m, a, b)) {
                    Vec3 center = (mesh_geometry->verts[a] + mesh_geometry->verts[b]) * 0.5;
                    centers[(size_t)m * 3 + 0] = center.x;
                    centers[(size_t)m * 3 + 1] = center.y;
                    centers[(size_t)m * 3 + 2] = center.z;
                    continue;
                }
            }
            double cx = 0.0, cy = 0.0, cz = 0.0;
            for (int q = 0; q < Nq; q++) {
                const double* qp = &op.qpts_p[(m * Nq + q) * 3];
                const double* qm = &op.qpts_m[(m * Nq + q) * 3];
                cx += qp[0] + qm[0];
                cy += qp[1] + qm[1];
                cz += qp[2] + qm[2];
            }
            double inv = 0.5 / (double)Nq;
            centers[(size_t)m * 3 + 0] = cx * inv;
            centers[(size_t)m * 3 + 1] = cy * inv;
            centers[(size_t)m * 3 + 2] = cz * inv;
        }

        double center_mean[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < N; i++) {
            center_mean[0] += centers[(size_t)i * 3];
            center_mean[1] += centers[(size_t)i * 3 + 1];
            center_mean[2] += centers[(size_t)i * 3 + 2];
        }
        if (N > 0) {
            center_mean[0] /= N; center_mean[1] /= N; center_mean[2] /= N;
        }
        double center_scale_sq = 0.0;
        for (int i = 0; i < N; i++) {
            double dx = centers[(size_t)i * 3] - center_mean[0];
            double dy = centers[(size_t)i * 3 + 1] - center_mean[1];
            double dz = centers[(size_t)i * 3 + 2] - center_mean[2];
            center_scale_sq += dx*dx + dy*dy + dz*dz;
        }
        double center_scale = std::sqrt(center_scale_sq / std::max(1, N));
        center_scale = std::max(center_scale, 1e-12);

        std::vector<std::vector<int>> rows(N);
        #pragma omp parallel for schedule(dynamic, 16)
        for (int m = 0; m < N; m++) {
            if (!triangle_owners.empty()) {
                int count = std::min(near_degree + 1, N);
                std::vector<std::tuple<long long, int, int>> nearest;
                nearest.reserve(N);
                double mx = centers[(size_t)m * 3];
                double my = centers[(size_t)m * 3 + 1];
                double mz = centers[(size_t)m * 3 + 2];
                for (int n = 0; n < N; n++) {
                    double dx = mx - centers[(size_t)n * 3];
                    double dy = my - centers[(size_t)n * 3 + 1];
                    double dz = mz - centers[(size_t)n * 3 + 2];
                    double normalized_distance = std::sqrt(dx*dx + dy*dy + dz*dz) / center_scale;
                    nearest.push_back(std::make_tuple(
                        (long long)std::llround(normalized_distance * 1e10),
                        python_rwg_rank[n], n));
                }
                std::sort(nearest.begin(), nearest.end());
                std::vector<int> ids;
                ids.reserve((size_t)count + 5);
                for (int i = 0; i < count; i++)
                    push_unique(ids, std::get<2>(nearest[i]));
                for (int id : triangle_owners[rwg_geometry->tri_p[m]])
                    push_unique(ids, id);
                for (int id : triangle_owners[rwg_geometry->tri_m[m]])
                    push_unique(ids, id);
                std::sort(ids.begin(), ids.end());
                rows[m].swap(ids);
                continue;
            }
            int keep = std::min(std::max(1, near_degree), N);
            std::vector<double> best_d2(keep, std::numeric_limits<double>::infinity());
            std::vector<int> best_id(keep, -1);
            double mx = centers[(size_t)m * 3 + 0];
            double my = centers[(size_t)m * 3 + 1];
            double mz = centers[(size_t)m * 3 + 2];
            for (int n = 0; n < N; n++) {
                if (n == m)
                    continue;
                double dx = mx - centers[(size_t)n * 3 + 0];
                double dy = my - centers[(size_t)n * 3 + 1];
                double dz = mz - centers[(size_t)n * 3 + 2];
                double d2 = dx*dx + dy*dy + dz*dz;
                int worst = 0;
                for (int i = 1; i < keep; i++)
                    if (best_d2[i] > best_d2[worst])
                        worst = i;
                if (d2 < best_d2[worst]) {
                    best_d2[worst] = d2;
                    best_id[worst] = n;
                }
            }
            std::vector<int> order(keep);
            for (int i = 0; i < keep; i++)
                order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return best_d2[a] < best_d2[b];
            });

            std::vector<int> ids;
            ids.reserve((size_t)keep + (size_t)(op.corr_row_ptr[m + 1] - op.corr_row_ptr[m]) + 1);
            push_unique(ids, m);
            for (int idx : order)
                if (best_id[idx] >= 0)
                    push_unique(ids, best_id[idx]);
            if (!triangle_owners.empty()) {
                for (int id : triangle_owners[rwg_geometry->tri_p[m]])
                    push_unique(ids, id);
                for (int id : triangle_owners[rwg_geometry->tri_m[m]])
                    push_unique(ids, id);
            } else {
                for (int jc = op.corr_row_ptr[m]; jc < op.corr_row_ptr[m + 1]; jc++)
                    push_unique(ids, op.corr_col_idx[jc]);
            }
            if (!triangle_owners.empty())
                std::sort(ids.begin(), ids.end());
            rows[m].swap(ids);
        }

        near_row_ptr.assign(N + 1, 0);
        for (int m = 0; m < N; m++)
            near_row_ptr[m + 1] = near_row_ptr[m] + (int)rows[m].size();
        near_col_idx.resize(near_row_ptr[N]);
        for (int m = 0; m < N; m++)
            std::copy(rows[m].begin(), rows[m].end(), near_col_idx.begin() + near_row_ptr[m]);
        printf("  [Precond] Expanded near graph: degree=%d nnz=%zu (%.1f per row)\n",
               near_degree, near_col_idx.size(), near_col_idx.empty() ? 0.0 : (double)near_col_idx.size() / (double)N);
    } else {
        if ((block_schwarz || ilu0) && near_degree > 0) {
            printf("  [Precond] Using topological near graph: N=%d exceeds BEM_PREC_GEOM_NEAR_MAX_N=%d; "
                   "set BEM_PREC_GEOM_NEAR_MAX_N=0 to force geometric nearest neighbors\n",
                   N, geom_near_max_n);
        }
        near_row_ptr = op.corr_row_ptr;
        near_col_idx = op.corr_col_idx;
    }
    near_blk.assign(4 * near_col_idx.size(), cdouble(0));

    // For each RWG m, compute diagonal L(m,m) and K(m,m) entries
    #pragma omp parallel for schedule(dynamic, 16)
    for (int m = 0; m < N; m++) {
        cdouble L_vals_k[2] = {0, 0};
        cdouble K_vals_k[2] = {0, 0};

        // Sum over 4 half-pair combos: (p,p), (p,m), (m,p), (m,m), source=target=m
        for (int hm = 0; hm < 2; hm++) {
            const double* qm = (hm == 0) ? &op.qpts_p[m * Nq * 3] : &op.qpts_m[m * Nq * 3];
            const double* fm = (hm == 0) ? &op.f_p[m * Nq * 3] : &op.f_m[m * Nq * 3];
            double dm = (hm == 0) ? op.div_p[m] : op.div_m[m];
            const double* jwm = (hm == 0) ? &op.jw_p[m * Nq] : &op.jw_m[m * Nq];

            for (int hn = 0; hn < 2; hn++) {
                const double* qn = (hn == 0) ? &op.qpts_p[m * Nq * 3] : &op.qpts_m[m * Nq * 3];
                const double* fn = (hn == 0) ? &op.f_p[m * Nq * 3] : &op.f_m[m * Nq * 3];
                double dn = (hn == 0) ? op.div_p[m] : op.div_m[m];
                const double* jwn = (hn == 0) ? &op.jw_p[m * Nq] : &op.jw_m[m * Nq];

                for (int qi = 0; qi < Nq; qi++) {
                    double rx = qm[qi*3], ry = qm[qi*3+1], rz = qm[qi*3+2];
                    double fxm = fm[qi*3], fym = fm[qi*3+1], fzm = fm[qi*3+2];
                    double wm_val = jwm[qi];

                    for (int qj = 0; qj < Nq; qj++) {
                        double dx = rx - qn[qj*3];
                        double dy = ry - qn[qj*3+1];
                        double dz = rz - qn[qj*3+2];
                        double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                        double wn_val = jwn[qj];
                        double ww = wm_val * wn_val;

                        double fxn = fn[qj*3], fyn = fn[qj*3+1], fzn = fn[qj*3+2];
                        double f_dot = fxm*fxn + fym*fyn + fzm*fzn;

                        for (int ki = 0; ki < 2; ki++) {
                            cdouble kv = k_vals[ki];
                            cdouble ik = cdouble(0, 1) * kv;
                            cdouble iok = cdouble(0, 1) / kv;

                            if (R > 1e-12) {
                                cdouble G = std::exp(ik * R) * inv4pi / R;
                                L_vals_k[ki] += (ik * f_dot - iok * dm * dn) * G * ww;

                                cdouble gG = G * (ik - 1.0/R) / R;
                                double cx = dy*fzn - dz*fyn;
                                double cy = dz*fxn - dx*fzn;
                                double cz = dx*fyn - dy*fxn;
                                K_vals_k[ki] += gG * (fxm*cx + fym*cy + fzm*cz) * ww;
                            } else {
                                cdouble G0 = ik * inv4pi;
                                L_vals_k[ki] += (ik * f_dot - iok * dm * dn) * G0 * ww;
                            }
                        }
                    }
                }
            }
        }

        // Add singular corrections for m=m
        for (int jc = op.corr_row_ptr[m]; jc < op.corr_row_ptr[m + 1]; jc++) {
            if (op.corr_col_idx[jc] == m) {
                L_vals_k[0] += op.corr_L_ext_val[jc];
                K_vals_k[0] += op.corr_K_ext_val[jc];
                L_vals_k[1] += op.corr_L_int_val[jc];
                K_vals_k[1] += op.corr_K_int_val[jc];
            }
        }

        // Assemble 2x2 PMCHWT block
        cdouble Ksum_mm = K_vals_k[0] + op.int_op_sign * K_vals_k[1] + op.k_identity;
        cdouble A_mm = eta_e * L_vals_k[0] + op.int_op_sign * eta_i * L_vals_k[1]; // eta*L
        cdouble B_mm = -Ksum_mm / op.unknown_m_scale;                 // -K/sM
        cdouble C_mm = op.row_h_scale * Ksum_mm;                      // rH*K
        cdouble D_mm = op.row_h_scale *
                       (L_vals_k[0] / eta_e + op.int_op_sign * L_vals_k[1] / eta_i) /
                       op.unknown_m_scale;                            // rH*L/(eta*sM)

        diag_blk[4*m + 0] = A_mm;
        diag_blk[4*m + 1] = B_mm;
        diag_blk[4*m + 2] = C_mm;
        diag_blk[4*m + 3] = D_mm;

        // Invert 2x2 block
        cdouble det = A_mm * D_mm - B_mm * C_mm;
        if (std::abs(det) < 1e-30) det = cdouble(1e-30);
        cdouble inv_det = cdouble(1.0) / det;

        blk_inv[4*m + 0] =  D_mm * inv_det;
        blk_inv[4*m + 1] = -B_mm * inv_det;
        blk_inv[4*m + 2] = -C_mm * inv_det;
        blk_inv[4*m + 3] =  A_mm * inv_det;
    }

    #pragma omp parallel for schedule(dynamic, 8)
    for (int m = 0; m < N; m++) {
        for (int jc = near_row_ptr[m]; jc < near_row_ptr[m + 1]; jc++) {
            int n = near_col_idx[jc];
            if (n == m)
                continue;

            cdouble L_vals_k[2] = {0, 0};
            cdouble K_vals_k[2] = {0, 0};
            int corr_pos = find_csr_col(op.corr_row_ptr, op.corr_col_idx, m, n);
            if (corr_pos >= 0) {
                L_vals_k[0] = op.corr_L_ext_val[corr_pos];
                K_vals_k[0] = op.corr_K_ext_val[corr_pos];
                L_vals_k[1] = op.corr_L_int_val[corr_pos];
                K_vals_k[1] = op.corr_K_int_val[corr_pos];
            }

            for (int hm = 0; hm < 2; hm++) {
                const double* qm = (hm == 0) ? &op.qpts_p[m * Nq * 3] : &op.qpts_m[m * Nq * 3];
                const double* fm = (hm == 0) ? &op.f_p[m * Nq * 3] : &op.f_m[m * Nq * 3];
                double dm = (hm == 0) ? op.div_p[m] : op.div_m[m];
                const double* jwm = (hm == 0) ? &op.jw_p[m * Nq] : &op.jw_m[m * Nq];

                for (int hn = 0; hn < 2; hn++) {
                    const double* qn = (hn == 0) ? &op.qpts_p[n * Nq * 3] : &op.qpts_m[n * Nq * 3];
                    const double* fn = (hn == 0) ? &op.f_p[n * Nq * 3] : &op.f_m[n * Nq * 3];
                    double dn = (hn == 0) ? op.div_p[n] : op.div_m[n];
                    const double* jwn = (hn == 0) ? &op.jw_p[n * Nq] : &op.jw_m[n * Nq];

                    for (int qi = 0; qi < Nq; qi++) {
                        double rx = qm[qi*3], ry = qm[qi*3+1], rz = qm[qi*3+2];
                        double fxm = fm[qi*3], fym = fm[qi*3+1], fzm = fm[qi*3+2];
                        double wm_val = jwm[qi];

                        for (int qj = 0; qj < Nq; qj++) {
                            double dx = rx - qn[qj*3];
                            double dy = ry - qn[qj*3+1];
                            double dz = rz - qn[qj*3+2];
                            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                            double ww = wm_val * jwn[qj];
                            double fxn = fn[qj*3], fyn = fn[qj*3+1], fzn = fn[qj*3+2];
                            double f_dot = fxm*fxn + fym*fyn + fzm*fzn;

                            for (int ki = 0; ki < 2; ki++) {
                                cdouble kv = k_vals[ki];
                                cdouble ik = cdouble(0, 1) * kv;
                                cdouble iok = cdouble(0, 1) / kv;

                                if (R > 1e-12) {
                                    cdouble G = std::exp(ik * R) * inv4pi / R;
                                    L_vals_k[ki] += (ik * f_dot - iok * dm * dn) * G * ww;

                                    cdouble gG = G * (ik - 1.0/R) / R;
                                    double cx = dy*fzn - dz*fyn;
                                    double cy = dz*fxn - dx*fzn;
                                    double cz = dx*fyn - dy*fxn;
                                    K_vals_k[ki] += gG * (fxm*cx + fym*cy + fzm*cz) * ww;
                                } else {
                                    cdouble G0 = ik * inv4pi;
                                    L_vals_k[ki] += (ik * f_dot - iok * dm * dn) * G0 * ww;
                                }
                            }
                        }
                    }
                }
            }

            cdouble Lext = L_vals_k[0];
            cdouble Lint = L_vals_k[1];
            cdouble Ksum = K_vals_k[0] + op.int_op_sign * K_vals_k[1];

            near_blk[4*jc + 0] = eta_e * Lext + op.int_op_sign * eta_i * Lint;
            near_blk[4*jc + 1] = -Ksum / op.unknown_m_scale;
            near_blk[4*jc + 2] = op.row_h_scale * Ksum;
            near_blk[4*jc + 3] = op.row_h_scale *
                                 (Lext / eta_e + op.int_op_sign * Lint / eta_i) /
                                 op.unknown_m_scale;
        }
    }

    if (ilu0) {
        richardson_sweeps = 0;
        build_ilu0_factors(*this);
        if (bem_env_flag_enabled("BEM_PREC_GPU", true))
            upload_ilu0_factors(*this);
        printf("  [Precond] ILU(0) ready: total build %.2fs%s\n",
               timer.elapsed_s(), device_ready ? " + GPU apply" : " + CPU apply");
        return;
    }

    size_t report_block_count = 0;
    blocks.clear();
    max_block_dim = 0;
    block_weight.assign(N, 0.0);
    if (block_schwarz) {
        const int requested_blocks = morton_block_jacobi
            ? (int)morton_blocks.size()
            : N;
        blocks.reserve(requested_blocks);
        for (int m = 0; m < requested_blocks; m++) {
            LocalBlock blk;
            if (morton_block_jacobi) {
                blk.ids = morton_blocks[m];
            } else {
                for (int jc = near_row_ptr[m];
                     jc < near_row_ptr[m + 1]; jc++) {
                    if ((int)blk.ids.size() >= max_block_basis)
                        break;
                    blk.ids.push_back(near_col_idx[jc]);
                }
                if (std::find(blk.ids.begin(), blk.ids.end(), m) ==
                    blk.ids.end()) {
                    if ((int)blk.ids.size() >= max_block_basis)
                        blk.ids.back() = m;
                    else
                        blk.ids.push_back(m);
                }
                std::sort(blk.ids.begin(), blk.ids.end());
                blk.ids.erase(
                    std::unique(blk.ids.begin(), blk.ids.end()),
                    blk.ids.end());
            }

            int nb = (int)blk.ids.size();
            int nd = 2 * nb;
            blk.lu.assign(nd * nd, cdouble(0));
            for (int a = 0; a < nb; a++) {
                int row = blk.ids[a];
                for (int b = 0; b < nb; b++) {
                    int col = blk.ids[b];
                    cdouble A(0), B(0), C(0), D(0);
                    if (row == col) {
                        A = diag_blk[4*row + 0];
                        B = diag_blk[4*row + 1];
                        C = diag_blk[4*row + 2];
                        D = diag_blk[4*row + 3];
                    } else {
                        int pos = find_csr_col(near_row_ptr, near_col_idx, row, col);
                        if (pos >= 0) {
                            A = near_blk[4*pos + 0];
                            B = near_blk[4*pos + 1];
                            C = near_blk[4*pos + 2];
                            D = near_blk[4*pos + 3];
                        }
                    }
                    blk.lu[(2*a)   * nd + (2*b)]   = A;
                    blk.lu[(2*a)   * nd + (2*b+1)] = B;
                    blk.lu[(2*a+1) * nd + (2*b)]   = C;
                    blk.lu[(2*a+1) * nd + (2*b+1)] = D;
                }
            }

            if (lu_factor_small(blk.lu, blk.piv, nd)) {
                for (int id : blk.ids)
                    block_weight[id] += 1.0;
                max_block_dim = std::max(max_block_dim, nd);
                blocks.push_back(std::move(blk));
            } else if (morton_block_jacobi) {
                fprintf(
                    stderr,
                    "Error: singular Morton block %d (dimension %d)\n",
                    m, nd);
                std::abort();
            }
        }
        for (double& w : block_weight)
            if (w == 0.0) w = 1.0;
        report_block_count = blocks.size();
        if (bem_env_flag_enabled("BEM_PREC_GPU", true))
            upload_device();
    }

    printf("  [Precond] Block Jacobi built: %.2fs", timer.elapsed_s());
    if (richardson_sweeps > 0)
        printf(" + %d near sweeps (omega=%.2f)", richardson_sweeps, richardson_omega);
    if (block_schwarz)
        printf(" + %s blocks=%zu max_basis=%d",
               morton_block_jacobi ? "Morton-Jacobi" : "Schwarz",
               report_block_count, max_block_basis);
    if (device_ready)
        printf(" + GPU apply");
    printf("\n");
}

bool NearFieldPrecond::dump_neural_features(const char* path, const RWG& rwg,
                                            const Mesh& mesh, BemFmmOperator& op,
                                            double ka, double n_re, double n_im,
                                            bool balanced_system,
                                            int coarse_rank) const
{
    if (neural_sparse || N != rwg.N || near_row_ptr.size() != (size_t)N + 1 ||
        diag_blk.size() != (size_t)4 * N ||
        near_blk.size() != (size_t)4 * near_col_idx.size()) {
        fprintf(stderr, "Error: local PMCHWT graph is unavailable for neural export\n");
        return false;
    }
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        fprintf(stderr, "Error: cannot create neural feature dump: %s\n", path);
        return false;
    }

    if (coarse_rank < 0 || coarse_rank > 18) {
        fprintf(stderr, "Error: neural coarse rank must be in 0..18\n");
        return false;
    }

    const char magic[8] = {'B','E','M','N','R','A','W','1'};
    uint32_t version = coarse_rank > 0 ? 3u : 2u;
    uint32_t system_code = balanced_system ? 1u : 0u;
    uint64_t file_n = (uint64_t)N;
    uint64_t file_nnz = (uint64_t)near_col_idx.size();
    uint64_t geometry_signature = bem_neural_geometry_signature(rwg);
    stream.write(magic, sizeof(magic));
    stream.write(reinterpret_cast<const char*>(&version), sizeof(version));
    stream.write(reinterpret_cast<const char*>(&system_code), sizeof(system_code));
    stream.write(reinterpret_cast<const char*>(&file_n), sizeof(file_n));
    stream.write(reinterpret_cast<const char*>(&file_nnz), sizeof(file_nnz));
    stream.write(reinterpret_cast<const char*>(&geometry_signature), sizeof(geometry_signature));
    stream.write(reinterpret_cast<const char*>(&ka), sizeof(ka));
    stream.write(reinterpret_cast<const char*>(&n_re), sizeof(n_re));
    stream.write(reinterpret_cast<const char*>(&n_im), sizeof(n_im));

    Vec3 vertex_center;
    for (const Vec3& vertex : mesh.verts)
        vertex_center = vertex_center + vertex;
    if (!mesh.verts.empty())
        vertex_center = vertex_center * (1.0 / (double)mesh.verts.size());
    double center_values[3] = {vertex_center.x, vertex_center.y, vertex_center.z};
    stream.write(reinterpret_cast<const char*>(center_values), sizeof(center_values));

    std::vector<double> nodes((size_t)N * 15, 0.0);
    for (int i = 0; i < N; i++) {
        int a = -1, b = -1;
        if (!shared_edge_ids(rwg, mesh, i, a, b)) {
            fprintf(stderr, "Error: cannot recover vertices for RWG edge %d\n", i);
            return false;
        }
        Vec3 va = mesh.verts[a], vb = mesh.verts[b];
        Vec3 center = (va + vb) * 0.5;
        Vec3 tangent = (vb - va).normalized();
        Vec3 p0, p1, p2, m0, m1, m2;
        mesh.tri_verts(rwg.tri_p[i], p0, p1, p2);
        mesh.tri_verts(rwg.tri_m[i], m0, m1, m2);
        Vec3 normal_plus = (p1 - p0).cross(p2 - p0).normalized();
        Vec3 normal_minus = (m1 - m0).cross(m2 - m0).normalized();
        double* dst = &nodes[(size_t)i * 15];
        dst[0] = center.x; dst[1] = center.y; dst[2] = center.z;
        dst[3] = tangent.x; dst[4] = tangent.y; dst[5] = tangent.z;
        dst[6] = normal_plus.x; dst[7] = normal_plus.y; dst[8] = normal_plus.z;
        dst[9] = normal_minus.x; dst[10] = normal_minus.y; dst[11] = normal_minus.z;
        dst[12] = rwg.length[i]; dst[13] = rwg.area_p[i]; dst[14] = rwg.area_m[i];
    }
    stream.write(reinterpret_cast<const char*>(nodes.data()), nodes.size() * sizeof(double));

    std::vector<int32_t> row32(near_row_ptr.begin(), near_row_ptr.end());
    std::vector<int32_t> col32(near_col_idx.begin(), near_col_idx.end());
    stream.write(reinterpret_cast<const char*>(row32.data()), row32.size() * sizeof(int32_t));
    stream.write(reinterpret_cast<const char*>(col32.data()), col32.size() * sizeof(int32_t));
    std::vector<double> packed((size_t)file_nnz * 8, 0.0);
    for (int row = 0; row < N; row++) {
        for (int jc = near_row_ptr[row]; jc < near_row_ptr[row + 1]; jc++) {
            const cdouble* src = near_col_idx[jc] == row ? &diag_blk[4 * row] : &near_blk[4 * jc];
            for (int k = 0; k < 4; k++) {
                packed[(size_t)jc * 8 + 2 * k] = src[k].real();
                packed[(size_t)jc * 8 + 2 * k + 1] = src[k].imag();
            }
        }
    }
    stream.write(reinterpret_cast<const char*>(packed.data()), packed.size() * sizeof(double));

    if (coarse_rank > 0) {
        Timer coarse_timer;
        const int n2 = 2 * N;
        double radius_sq_sum = 0.0;
        for (int i = 0; i < N; i++) {
            double x = nodes[(size_t)i * 15] - vertex_center.x;
            double y = nodes[(size_t)i * 15 + 1] - vertex_center.y;
            double z = nodes[(size_t)i * 15 + 2] - vertex_center.z;
            radius_sq_sum += x*x + y*y + z*z;
        }
        const double shape_scale = std::max(std::sqrt(radius_sq_sum / std::max(1, N)), 1e-12);
        std::vector<cdouble> coarse_p((size_t)coarse_rank * n2, cdouble(0.0));
        std::vector<cdouble> candidate(n2, cdouble(0.0));
        for (int k = 0; k < coarse_rank; k++) {
            std::fill(candidate.begin(), candidate.end(), cdouble(0.0));
            const int mode = k / 2;
            const int channel_offset = (k % 2) * N;
            for (int i = 0; i < N; i++) {
                const double x = (nodes[(size_t)i * 15] - vertex_center.x) / shape_scale;
                const double y = (nodes[(size_t)i * 15 + 1] - vertex_center.y) / shape_scale;
                const double z = (nodes[(size_t)i * 15 + 2] - vertex_center.z) / shape_scale;
                double value = 0.0;
                switch (mode) {
                    case 0: value = 1.0; break;
                    case 1: value = x; break;
                    case 2: value = y; break;
                    case 3: value = z; break;
                    case 4: value = x * y; break;
                    case 5: value = x * z; break;
                    case 6: value = y * z; break;
                    case 7: value = x * x - y * y; break;
                    case 8: value = 3.0 * z * z - (x * x + y * y + z * z); break;
                    default: break;
                }
                candidate[channel_offset + i] = cdouble(value, 0.0);
            }
            // Two modified Gram-Schmidt passes keep the exported modes stable
            // even on symmetric meshes where some raw polynomial moments correlate.
            for (int pass = 0; pass < 2; pass++) {
                for (int j = 0; j < k; j++) {
                    const cdouble* pj = coarse_p.data() + (size_t)j * n2;
                    cdouble dot(0.0);
                    for (int i = 0; i < n2; i++)
                        dot += std::conj(pj[i]) * candidate[i];
                    for (int i = 0; i < n2; i++)
                        candidate[i] -= pj[i] * dot;
                }
            }
            double norm_sq = 0.0;
            for (const cdouble& value : candidate)
                norm_sq += std::norm(value);
            const double norm = std::sqrt(norm_sq);
            if (!(norm > 1e-10) || !std::isfinite(norm)) {
                fprintf(stderr, "Error: coarse polynomial mode %d is linearly dependent\n", mode);
                return false;
            }
            cdouble* pk = coarse_p.data() + (size_t)k * n2;
            for (int i = 0; i < n2; i++)
                pk[i] = candidate[i] / norm;
        }

        std::vector<cdouble> coarse_ap((size_t)coarse_rank * n2);
        for (int k = 0; k + 1 < coarse_rank; k += 2) {
            op.matvec_batch2(coarse_p.data() + (size_t)k * n2,
                             coarse_p.data() + (size_t)(k + 1) * n2,
                             coarse_ap.data() + (size_t)k * n2,
                             coarse_ap.data() + (size_t)(k + 1) * n2);
        }
        if (coarse_rank % 2 != 0) {
            const int k = coarse_rank - 1;
            op.matvec(coarse_p.data() + (size_t)k * n2,
                      coarse_ap.data() + (size_t)k * n2);
        }

        const uint32_t file_coarse_rank = (uint32_t)coarse_rank;
        stream.write(reinterpret_cast<const char*>(&file_coarse_rank), sizeof(file_coarse_rank));
        auto write_complex_columns = [&](const std::vector<cdouble>& values) {
            std::vector<double> buffer(16384);
            size_t offset = 0;
            while (offset < values.size()) {
                const size_t count = std::min(values.size() - offset, buffer.size() / 2);
                for (size_t i = 0; i < count; i++) {
                    buffer[2 * i] = values[offset + i].real();
                    buffer[2 * i + 1] = values[offset + i].imag();
                }
                stream.write(reinterpret_cast<const char*>(buffer.data()), 2 * count * sizeof(double));
                offset += count;
            }
        };
        write_complex_columns(coarse_p);
        write_complex_columns(coarse_ap);
        printf("  [Precond] Full-FMM coarse probes: rank=%d actions=%d time=%.2fs\n",
               coarse_rank, coarse_rank, coarse_timer.elapsed_s());
    }
    if (!stream) {
        fprintf(stderr, "Error: failed while writing neural feature dump: %s\n", path);
        return false;
    }
    printf("  [Precond] Neural feature graph written: %s (N=%d blocks=%llu coarse_rank=%d)\n",
           path, N, (unsigned long long)file_nnz, coarse_rank);
    return true;
}

namespace {

template <typename T>
bool read_binary(std::ifstream& stream, T& value)
{
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    return (bool)stream;
}

bool close_parameter(double actual, double expected)
{
    double scale = std::max(1.0, std::max(std::abs(actual), std::abs(expected)));
    return std::abs(actual - expected) <= 1e-10 * scale;
}

} // namespace

bool NearFieldPrecond::load_neural(const char* path, int expected_n,
                                   double expected_ka, double expected_n_re,
                                   double expected_n_im, bool expected_balanced,
                                   uint64_t expected_geometry_signature)
{
    cleanup_device();
    neural_coarse_rank = 0;
    neural_coarse_q.clear();
    neural_coarse_update.clear();
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        fprintf(stderr, "Error: cannot open neural preconditioner: %s\n", path);
        return false;
    }

    char magic[8];
    uint32_t version = 0, system_code = 0;
    uint64_t file_n = 0, file_nnz = 0, file_geometry_signature = 0;
    double file_ka = 0.0, file_n_re = 0.0, file_n_im = 0.0;
    stream.read(magic, sizeof(magic));
    if (!stream || std::memcmp(magic, "BEMNSAI1", 8) != 0 ||
        !read_binary(stream, version) || !read_binary(stream, system_code) ||
        !read_binary(stream, file_n) || !read_binary(stream, file_nnz) ||
        !read_binary(stream, file_geometry_signature) ||
        !read_binary(stream, file_ka) || !read_binary(stream, file_n_re) ||
        !read_binary(stream, file_n_im)) {
        fprintf(stderr, "Error: invalid neural preconditioner header: %s\n", path);
        return false;
    }
    if (version != 2 && version != 3) {
        fprintf(stderr, "Error: unsupported neural preconditioner version %u\n", version);
        return false;
    }
    const uint32_t expected_system = expected_balanced ? 1u : 0u;
    if (file_n != (uint64_t)expected_n || system_code != expected_system ||
        file_geometry_signature != expected_geometry_signature ||
        !close_parameter(file_ka, expected_ka) ||
        !close_parameter(file_n_re, expected_n_re) ||
        !close_parameter(file_n_im, expected_n_im)) {
        fprintf(stderr,
                "Error: neural preconditioner does not match this system "
                "(file N=%llu ka=%.17g n=%.17g%+.17gi system=%u geometry=%016llx; "
                "run N=%d ka=%.17g n=%.17g%+.17gi system=%u geometry=%016llx)\n",
                (unsigned long long)file_n, file_ka, file_n_re, file_n_im, system_code,
                (unsigned long long)file_geometry_signature,
                expected_n, expected_ka, expected_n_re, expected_n_im, expected_system,
                (unsigned long long)expected_geometry_signature);
        return false;
    }
    if (file_nnz == 0 || file_nnz > (uint64_t)expected_n * (uint64_t)expected_n) {
        fprintf(stderr, "Error: invalid neural block count: %llu\n",
                (unsigned long long)file_nnz);
        return false;
    }

    N = expected_n;
    N2 = 2 * N;
    near_row_ptr.resize((size_t)N + 1);
    near_col_idx.resize((size_t)file_nnz);
    std::vector<int32_t> row32((size_t)N + 1), col32((size_t)file_nnz);
    stream.read(reinterpret_cast<char*>(row32.data()), row32.size() * sizeof(int32_t));
    stream.read(reinterpret_cast<char*>(col32.data()), col32.size() * sizeof(int32_t));
    std::vector<float> packed((size_t)file_nnz * 8);
    stream.read(reinterpret_cast<char*>(packed.data()), packed.size() * sizeof(float));
    if (!stream) {
        fprintf(stderr, "Error: truncated neural preconditioner: %s\n", path);
        return false;
    }
    if (row32[0] != 0 || row32[N] != (int32_t)file_nnz) {
        fprintf(stderr, "Error: invalid neural CSR row bounds\n");
        return false;
    }
    for (int i = 0; i <= N; i++) {
        if (row32[i] < 0 || (uint64_t)row32[i] > file_nnz ||
            (i > 0 && row32[i] < row32[i - 1])) {
            fprintf(stderr, "Error: invalid neural CSR row pointer at %d\n", i);
            return false;
        }
        near_row_ptr[i] = row32[i];
    }
    for (size_t i = 0; i < col32.size(); i++) {
        if (col32[i] < 0 || col32[i] >= N) {
            fprintf(stderr, "Error: invalid neural CSR column at %zu\n", i);
            return false;
        }
        near_col_idx[i] = col32[i];
    }
    near_blk.resize((size_t)file_nnz * 4);
    for (size_t i = 0; i < near_blk.size(); i++) {
        double re = packed[2 * i];
        double im = packed[2 * i + 1];
        if (!std::isfinite(re) || !std::isfinite(im)) {
            fprintf(stderr, "Error: non-finite neural block value at %zu\n", i);
            return false;
        }
        near_blk[i] = cdouble(re, im);
    }

    if (version == 3) {
        uint32_t file_coarse_rank = 0;
        if (!read_binary(stream, file_coarse_rank) || file_coarse_rank == 0 ||
            file_coarse_rank > 4096) {
            fprintf(stderr, "Error: invalid neural coarse rank in %s\n", path);
            return false;
        }
        neural_coarse_rank = (int)file_coarse_rank;
        const size_t factor_size = (size_t)neural_coarse_rank * N2;
        std::vector<float> q_packed(2 * factor_size), update_packed(2 * factor_size);
        stream.read(reinterpret_cast<char*>(q_packed.data()), q_packed.size() * sizeof(float));
        stream.read(reinterpret_cast<char*>(update_packed.data()), update_packed.size() * sizeof(float));
        if (!stream) {
            fprintf(stderr, "Error: truncated neural coarse factors: %s\n", path);
            return false;
        }
        neural_coarse_q.resize(factor_size);
        neural_coarse_update.resize(factor_size);
        for (size_t i = 0; i < factor_size; i++) {
            const cdouble q(q_packed[2 * i], q_packed[2 * i + 1]);
            const cdouble update(update_packed[2 * i], update_packed[2 * i + 1]);
            if (!std::isfinite(q.real()) || !std::isfinite(q.imag()) ||
                !std::isfinite(update.real()) || !std::isfinite(update.imag())) {
                fprintf(stderr, "Error: non-finite neural coarse value at %zu\n", i);
                return false;
            }
            neural_coarse_q[i] = q;
            neural_coarse_update[i] = update;
        }
    }
    if (stream.peek() != std::ifstream::traits_type::eof()) {
        fprintf(stderr, "Error: unexpected trailing bytes in neural preconditioner: %s\n", path);
        return false;
    }

    neural_sparse = true;
    block_schwarz = false;
    richardson_sweeps = 0;

    if (bem_env_flag_enabled("BEM_PREC_GPU", true)) {
        device_near_nnz = (int)file_nnz;
        std::vector<double2> device_blocks(near_blk.size());
        for (size_t i = 0; i < near_blk.size(); i++)
            device_blocks[i] = make_double2(near_blk[i].real(), near_blk[i].imag());
        CUDA_CHECK(cudaMalloc(&d_near_row_ptr, ((size_t)N + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_near_col_idx, (size_t)device_near_nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_neural_blk, device_blocks.size() * sizeof(double2)));
        CUDA_CHECK(cudaMemcpy(d_near_row_ptr, near_row_ptr.data(), ((size_t)N + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_near_col_idx, near_col_idx.data(), (size_t)device_near_nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neural_blk, device_blocks.data(), device_blocks.size() * sizeof(double2), cudaMemcpyHostToDevice));
        if (neural_coarse_rank > 0) {
            std::vector<float2> device_q(neural_coarse_q.size());
            std::vector<float2> device_update(neural_coarse_update.size());
            for (size_t i = 0; i < neural_coarse_q.size(); i++) {
                device_q[i] = make_float2((float)neural_coarse_q[i].real(),
                                          (float)neural_coarse_q[i].imag());
                device_update[i] = make_float2((float)neural_coarse_update[i].real(),
                                               (float)neural_coarse_update[i].imag());
            }
            CUDA_CHECK(cudaMalloc(&d_neural_coarse_q, device_q.size() * sizeof(float2)));
            CUDA_CHECK(cudaMalloc(&d_neural_coarse_update, device_update.size() * sizeof(float2)));
            CUDA_CHECK(cudaMalloc(&d_neural_coarse_coeff,
                                  (size_t)neural_coarse_rank * sizeof(double2)));
            CUDA_CHECK(cudaMemcpy(d_neural_coarse_q, device_q.data(),
                                  device_q.size() * sizeof(float2), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_neural_coarse_update, device_update.data(),
                                  device_update.size() * sizeof(float2), cudaMemcpyHostToDevice));
        }
        device_ready = true;
    }
    printf("  [Precond] Neural GraphSAI loaded: N=%d blocks=%llu (%.1f per row) coarse_rank=%d%s\n",
           N, (unsigned long long)file_nnz, (double)file_nnz / (double)N,
           neural_coarse_rank,
           device_ready ? " + GPU apply" : " + CPU apply");
    return true;
}

void NearFieldPrecond::apply_block_inv(const cdouble* r, cdouble* z) const
{
    // z[m] = inv_A*r[m] + inv_B*r[N+m]
    // z[N+m] = inv_C*r[m] + inv_D*r[N+m]
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < N; m++) {
        cdouble rm = r[m], rNm = r[N + m];
        z[m]     = blk_inv[4*m+0] * rm + blk_inv[4*m+1] * rNm;
        z[N + m] = blk_inv[4*m+2] * rm + blk_inv[4*m+3] * rNm;
    }
}

void NearFieldPrecond::apply_near(const cdouble* x, cdouble* y) const
{
    std::fill(y, y + N2, cdouble(0));
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < N; m++) {
        cdouble xm = x[m], xNm = x[N + m];

        cdouble ym = diag_blk[4*m+0] * xm + diag_blk[4*m+1] * xNm;
        cdouble yNm = diag_blk[4*m+2] * xm + diag_blk[4*m+3] * xNm;

        for (int jc = near_row_ptr[m]; jc < near_row_ptr[m + 1]; jc++) {
            int n = near_col_idx[jc];
            cdouble xn = x[n], xNn = x[N + n];
            ym  += near_blk[4*jc+0] * xn + near_blk[4*jc+1] * xNn;
            yNm += near_blk[4*jc+2] * xn + near_blk[4*jc+3] * xNn;
        }

        y[m] = ym;
        y[N + m] = yNm;
    }
}

void NearFieldPrecond::apply(const cdouble* r, cdouble* z) const
{
    if (mass_matrix) {
        fprintf(stderr, "Error: mass preconditioner currently requires device GMRES\n");
        std::abort();
    }
    if (ilu0) {
        std::copy(r, r + N2, z);
        for (int row = 0; row < N2; row++) {
            for (int pos = ilu_row_ptr[row]; pos < ilu_diag_ptr[row]; pos++)
                z[row] -= ilu_val[pos] * z[ilu_col_idx[pos]];
        }
        for (int row = N2 - 1; row >= 0; row--) {
            for (int pos = ilu_diag_ptr[row] + 1; pos < ilu_row_ptr[row + 1]; pos++)
                z[row] -= ilu_val[pos] * z[ilu_col_idx[pos]];
            z[row] /= ilu_val[ilu_diag_ptr[row]];
        }
        return;
    }
    if (neural_sparse) {
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < N; m++) {
            cdouble out_j(0), out_m(0);
            for (int jc = near_row_ptr[m]; jc < near_row_ptr[m + 1]; jc++) {
                int n = near_col_idx[jc];
                const cdouble* b = &near_blk[4 * jc];
                out_j += b[0] * r[n] + b[1] * r[N + n];
                out_m += b[2] * r[n] + b[3] * r[N + n];
            }
            z[m] = out_j;
            z[N + m] = out_m;
        }
        if (neural_coarse_rank > 0) {
            std::vector<cdouble> coeff(neural_coarse_rank, cdouble(0.0));
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < neural_coarse_rank; k++) {
                const cdouble* qk = neural_coarse_q.data() + (size_t)k * N2;
                cdouble value(0.0);
                for (int i = 0; i < N2; i++)
                    value += std::conj(qk[i]) * r[i];
                coeff[k] = value;
            }
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N2; i++) {
                cdouble correction(0.0);
                for (int k = 0; k < neural_coarse_rank; k++)
                    correction += neural_coarse_update[(size_t)k * N2 + i] * coeff[k];
                z[i] += correction;
            }
        }
        return;
    }
    if (block_schwarz) {
        if (device_ready)
            apply_block_schwarz_cuda(r, z);
        else
            apply_block_schwarz(r, z);
        return;
    }

    apply_block_inv(r, z);
    if (richardson_sweeps <= 0)
        return;

    bool reuse_workspace = bem_env_flag_enabled("BEM_PREC_REUSE_WORKSPACE", true);
    std::vector<cdouble> local_Az, local_err, local_corr;
    cdouble* Az;
    cdouble* err;
    cdouble* corr;
    if (reuse_workspace) {
        tmp_Az.resize(N2);
        tmp_err.resize(N2);
        tmp_corr.resize(N2);
        Az = tmp_Az.data();
        err = tmp_err.data();
        corr = tmp_corr.data();
    } else {
        local_Az.resize(N2);
        local_err.resize(N2);
        local_corr.resize(N2);
        Az = local_Az.data();
        err = local_err.data();
        corr = local_corr.data();
    }
    for (int sweep = 0; sweep < richardson_sweeps; sweep++) {
        apply_near(z, Az);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N2; i++)
            err[i] = r[i] - Az[i];
        apply_block_inv(err, corr);
        cdouble omega(richardson_omega, 0.0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N2; i++)
            z[i] += omega * corr[i];
    }
}

void NearFieldPrecond::apply_pair(const cdouble* r1, const cdouble* r2,
                                  cdouble* z1, cdouble* z2) const
{
    apply(r1, z1);
    apply(r2, z2);
}

void NearFieldPrecond::apply_block_schwarz(const cdouble* r, cdouble* z) const
{
    std::fill(z, z + N2, cdouble(0));
    std::vector<cdouble> rhs(max_block_dim), sol(max_block_dim);

    for (const LocalBlock& blk : blocks) {
        int nb = (int)blk.ids.size();
        int nd = 2 * nb;
        for (int i = 0; i < nb; i++) {
            int id = blk.ids[i];
            rhs[2*i] = r[id];
            rhs[2*i + 1] = r[N + id];
        }
        lu_solve_small(blk.lu, blk.piv, rhs.data(), sol.data(), nd);
        for (int i = 0; i < nb; i++) {
            int id = blk.ids[i];
            z[id] += sol[2*i] / block_weight[id];
            z[N + id] += sol[2*i + 1] / block_weight[id];
        }
    }
}

void NearFieldPrecond::upload_device()
{
    cleanup_device();
    if (!block_schwarz || blocks.empty() ||
        max_block_dim > kMaxDeviceBlockDim)
        return;

    device_block_count = (int)blocks.size();
    device_block_dim = std::max(1, max_block_dim);
    std::vector<int> offsets(device_block_count + 1, 0);
    for (int b = 0; b < device_block_count; b++)
        offsets[b + 1] = offsets[b] + (int)blocks[b].ids.size();
    device_ids_count = offsets[device_block_count];
    device_lu_count =
        device_block_count * device_block_dim * device_block_dim;

    std::vector<int> flat_ids(device_ids_count);
    std::vector<int> flat_piv(device_block_count * device_block_dim, 0);
    std::vector<double> flat_lu_re(device_lu_count, 0.0), flat_lu_im(device_lu_count, 0.0);
    for (int b = 0; b < device_block_count; b++) {
        const LocalBlock& blk = blocks[b];
        int nb = (int)blk.ids.size();
        int nd = 2 * nb;
        for (int i = 0; i < nb; i++)
            flat_ids[offsets[b] + i] = blk.ids[i];
        for (int i = 0; i < nd; i++)
            flat_piv[b * device_block_dim + i] = blk.piv[i];
        for (int i = 0; i < nd; i++) {
            for (int j = 0; j < nd; j++) {
                cdouble v = blk.lu[i * nd + j];
                int dst = b * device_block_dim * device_block_dim +
                          i * device_block_dim + j;
                flat_lu_re[dst] = v.real();
                flat_lu_im[dst] = v.imag();
            }
        }
    }

    CUDA_CHECK(cudaMalloc(&d_block_offsets, (device_block_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_ids, device_ids_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        &d_block_piv,
        (size_t)device_block_count * device_block_dim * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_lu_re, device_lu_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_lu_im, device_lu_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_weight, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_complex, N2 * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_z_complex, N2 * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_r_re, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_im, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z_re, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z_im, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Az_re, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Az_im, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_err_re, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_err_im, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_corr_re, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_corr_im, N2 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_block_offsets, offsets.data(), (device_block_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_ids, flat_ids.data(), device_ids_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_block_piv, flat_piv.data(),
        (size_t)device_block_count * device_block_dim * sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_lu_re, flat_lu_re.data(), device_lu_count * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_lu_im, flat_lu_im.data(), device_lu_count * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_weight, block_weight.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    device_near_nnz = (int)near_col_idx.size();
    std::vector<double> diag_re(4 * N), diag_im(4 * N);
    for (int i = 0; i < 4 * N; i++) {
        diag_re[i] = diag_blk[i].real();
        diag_im[i] = diag_blk[i].imag();
    }
    std::vector<double> near_re(4 * device_near_nnz), near_im(4 * device_near_nnz);
    for (int i = 0; i < 4 * device_near_nnz; i++) {
        near_re[i] = near_blk[i].real();
        near_im[i] = near_blk[i].imag();
    }
    CUDA_CHECK(cudaMalloc(&d_diag_re, 4 * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_diag_im, 4 * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_near_re, 4 * device_near_nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_near_im, 4 * device_near_nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_near_row_ptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_near_col_idx, device_near_nnz * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_diag_re, diag_re.data(), 4 * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_diag_im, diag_im.data(), 4 * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_near_re, near_re.data(), 4 * device_near_nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_near_im, near_im.data(), 4 * device_near_nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_near_row_ptr, near_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_near_col_idx, near_col_idx.data(), device_near_nnz * sizeof(int), cudaMemcpyHostToDevice));

    device_ready = true;

    // After a successful GPU upload the Schwarz preconditioner is applied
    // entirely from device memory. Drop host mirrors so long orientation runs
    // do not carry duplicate LU blocks and near-field CSR data.
    if (!bem_env_flag_enabled("BEM_PREC_KEEP_HOST")) {
        std::vector<cdouble>().swap(blk_inv);
        std::vector<cdouble>().swap(diag_blk);
        std::vector<int>().swap(near_row_ptr);
        std::vector<int>().swap(near_col_idx);
        std::vector<cdouble>().swap(near_blk);
        std::vector<LocalBlock>().swap(blocks);
        std::vector<double>().swap(block_weight);
    }
}

void NearFieldPrecond::cleanup_device()
{
    if (calderon_operator_actions > 0) {
        printf("  [Precond] RWG Calderon summary: full_operator_actions=%lld\n",
               calderon_operator_actions);
    }
    calderon_operator_actions = 0;
    calderon_operator = nullptr;
    if (mass_apply_count > 0) {
        printf("  [Precond] Mass CG summary: applies=%lld avg_iterations=%.2f "
               "max_iterations=%d max_relative_residual=%.2e\n",
               mass_apply_count,
               (double)mass_iteration_count / (double)mass_apply_count,
               mass_max_iterations_used, mass_max_relative_residual);
    }
    mass_apply_count = 0;
    mass_iteration_count = 0;
    mass_max_iterations_used = 0;
    mass_max_relative_residual = 0.0;
    cudaFree(d_mass_row_ptr); d_mass_row_ptr = nullptr;
    cudaFree(d_mass_col_idx); d_mass_col_idx = nullptr;
    cudaFree(d_mass_val); d_mass_val = nullptr;
    cudaFree(d_mass_inv_diag); d_mass_inv_diag = nullptr;
    cudaFree(d_mass_x); d_mass_x = nullptr;
    cudaFree(d_mass_r); d_mass_r = nullptr;
    cudaFree(d_mass_p); d_mass_p = nullptr;
    cudaFree(d_mass_ap); d_mass_ap = nullptr;
    cudaFree(d_calderon_mass0); d_calderon_mass0 = nullptr;
    cudaFree(d_calderon_mass1); d_calderon_mass1 = nullptr;
    cudaFree(d_calderon_op0); d_calderon_op0 = nullptr;
    cudaFree(d_calderon_op1); d_calderon_op1 = nullptr;
    cudaFree(d_mass_norm_sum0); d_mass_norm_sum0 = nullptr;
    cudaFree(d_mass_norm_sum1); d_mass_norm_sum1 = nullptr;
    cudaFree(d_mass_dot_sum0); d_mass_dot_sum0 = nullptr;
    cudaFree(d_mass_dot_sum1); d_mass_dot_sum1 = nullptr;
    mass_reduction_blocks = 0;
    std::vector<double>().swap(mass_host_norm0);
    std::vector<double>().swap(mass_host_norm1);
    std::vector<double2>().swap(mass_host_dot0);
    std::vector<double2>().swap(mass_host_dot1);
    if (ilu_spsv_l) cusparseSpSV_destroyDescr(ilu_spsv_l);
    if (ilu_spsv_u) cusparseSpSV_destroyDescr(ilu_spsv_u);
    if (ilu_vec_in) cusparseDestroyDnVec(ilu_vec_in);
    if (ilu_vec_tmp) cusparseDestroyDnVec(ilu_vec_tmp);
    if (ilu_vec_out) cusparseDestroyDnVec(ilu_vec_out);
    if (ilu_mat_l) cusparseDestroySpMat(ilu_mat_l);
    if (ilu_mat_u) cusparseDestroySpMat(ilu_mat_u);
    if (ilu_handle) cusparseDestroy(ilu_handle);
    ilu_spsv_l = nullptr; ilu_spsv_u = nullptr;
    ilu_vec_in = nullptr; ilu_vec_tmp = nullptr; ilu_vec_out = nullptr;
    ilu_mat_l = nullptr; ilu_mat_u = nullptr; ilu_handle = nullptr;
    cudaFree(d_ilu_buffer_l); d_ilu_buffer_l = nullptr;
    cudaFree(d_ilu_buffer_u); d_ilu_buffer_u = nullptr;
    cudaFree(d_ilu_row_ptr); d_ilu_row_ptr = nullptr;
    cudaFree(d_ilu_col_idx); d_ilu_col_idx = nullptr;
    cudaFree(d_ilu_val); d_ilu_val = nullptr;
    cudaFree(d_ilu_rhs); d_ilu_rhs = nullptr;
    cudaFree(d_ilu_tmp); d_ilu_tmp = nullptr;
    cudaFree(d_ilu_out); d_ilu_out = nullptr;
    cudaFree(d_block_offsets); d_block_offsets = nullptr;
    cudaFree(d_block_ids); d_block_ids = nullptr;
    cudaFree(d_block_piv); d_block_piv = nullptr;
    cudaFree(d_block_lu_re); d_block_lu_re = nullptr;
    cudaFree(d_block_lu_im); d_block_lu_im = nullptr;
    cudaFree(d_block_weight); d_block_weight = nullptr;
    cudaFree(d_r_complex); d_r_complex = nullptr;
    cudaFree(d_z_complex); d_z_complex = nullptr;
    cudaFree(d_r_re); d_r_re = nullptr;
    cudaFree(d_r_im); d_r_im = nullptr;
    cudaFree(d_z_re); d_z_re = nullptr;
    cudaFree(d_z_im); d_z_im = nullptr;
    cudaFree(d_Az_re); d_Az_re = nullptr;
    cudaFree(d_Az_im); d_Az_im = nullptr;
    cudaFree(d_err_re); d_err_re = nullptr;
    cudaFree(d_err_im); d_err_im = nullptr;
    cudaFree(d_corr_re); d_corr_re = nullptr;
    cudaFree(d_corr_im); d_corr_im = nullptr;
    cudaFree(d_diag_re); d_diag_re = nullptr;
    cudaFree(d_diag_im); d_diag_im = nullptr;
    cudaFree(d_near_re); d_near_re = nullptr;
    cudaFree(d_near_im); d_near_im = nullptr;
    cudaFree(d_neural_blk); d_neural_blk = nullptr;
    cudaFree(d_neural_coarse_q); d_neural_coarse_q = nullptr;
    cudaFree(d_neural_coarse_update); d_neural_coarse_update = nullptr;
    cudaFree(d_neural_coarse_coeff); d_neural_coarse_coeff = nullptr;
    cudaFree(d_near_row_ptr); d_near_row_ptr = nullptr;
    cudaFree(d_near_col_idx); d_near_col_idx = nullptr;
    device_ready = false;
    device_block_count = 0;
    device_block_dim = 0;
    device_ids_count = 0;
    device_lu_count = 0;
    device_near_nnz = 0;
}

void NearFieldPrecond::apply_block_schwarz_cuda(const cdouble* r, cdouble* z) const
{
    int block = 256;
    int grid_vec = (N2 + block - 1) / block;
    CUDA_CHECK(cudaMemcpy(d_r_complex, r, N2 * sizeof(double2), cudaMemcpyHostToDevice));
    precond_split_complex_kernel<<<grid_vec, block>>>(d_r_complex, d_r_re, d_r_im, N2);
    CUDA_CHECK(cudaGetLastError());
    apply_block_schwarz_cuda_device(d_r_re, d_r_im, d_z_re, d_z_im);

    if (richardson_sweeps > 0) {
        int grid_N = (N + block - 1) / block;
        for (int sweep = 0; sweep < richardson_sweeps; sweep++) {
            precond_near_matvec_kernel<<<grid_N, block>>>(
                N, d_near_row_ptr, d_near_col_idx,
                d_diag_re, d_diag_im, d_near_re, d_near_im,
                d_z_re, d_z_im, d_Az_re, d_Az_im);
            CUDA_CHECK(cudaGetLastError());
            precond_residual_kernel<<<grid_vec, block>>>(
                d_r_re, d_r_im, d_Az_re, d_Az_im, d_err_re, d_err_im, N2);
            CUDA_CHECK(cudaGetLastError());
            apply_block_schwarz_cuda_device(d_err_re, d_err_im, d_corr_re, d_corr_im);
            precond_axpy_kernel<<<grid_vec, block>>>(
                d_z_re, d_z_im, d_corr_re, d_corr_im, richardson_omega, N2);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    precond_pack_complex_kernel<<<grid_vec, block>>>(d_z_re, d_z_im, d_z_complex, N2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(z, d_z_complex, N2 * sizeof(double2), cudaMemcpyDeviceToHost));
}

bool NearFieldPrecond::device_apply_available() const
{
    return device_ready && (block_schwarz || neural_sparse || ilu0 || mass_matrix);
}

void NearFieldPrecond::apply_mass_device(const double2* d_rhs, double2* d_z) const
{
    const int block = 256;
    const int grid_n = (N + block - 1) / block;
    const int grid_n2 = (N2 + block - 1) / block;
    double bnorm0 = 0.0, bnorm1 = 0.0;
    mass_norm_pair(*this, d_rhs, d_rhs + N, &bnorm0, &bnorm1);
    const bool active0 = bnorm0 > 1e-30;
    const bool active1 = bnorm1 > 1e-30;
    if (!active0) bnorm0 = 1.0;
    if (!active1) bnorm1 = 1.0;

    mass_jacobi_init_kernel<<<grid_n, block>>>(
        N, d_mass_inv_diag, d_rhs, d_mass_x);
    mass_spmv_pair_kernel<<<grid_n, block>>>(
        N, d_mass_row_ptr, d_mass_col_idx, d_mass_val, d_mass_x, d_mass_ap);
    mass_residual_init_kernel<<<grid_n2, block>>>(
        N, d_rhs, d_mass_ap, d_mass_r, d_mass_p);
    CUDA_CHECK(cudaGetLastError());

    double rnorm0 = 0.0, rnorm1 = 0.0;
    mass_norm_pair(*this, d_mass_r, d_mass_r + N, &rnorm0, &rnorm1);
    double rr0 = rnorm0 * rnorm0;
    double rr1 = rnorm1 * rnorm1;
    int iterations = 0;
    for (; iterations < mass_cg_max_iterations; iterations++) {
        const bool done0 = !active0 || rnorm0 <= mass_cg_tolerance * bnorm0;
        const bool done1 = !active1 || rnorm1 <= mass_cg_tolerance * bnorm1;
        if (done0 && done1)
            break;

        mass_spmv_pair_kernel<<<grid_n, block>>>(
            N, d_mass_row_ptr, d_mass_col_idx, d_mass_val, d_mass_p, d_mass_ap);
        CUDA_CHECK(cudaGetLastError());
        double2 pap0 = make_double2(0.0, 0.0);
        double2 pap1 = make_double2(0.0, 0.0);
        mass_dot_pair(*this, d_mass_p, d_mass_ap,
                      d_mass_p + N, d_mass_ap + N, &pap0, &pap1);
        const double alpha0 = (!done0 && pap0.x > 1e-300) ? rr0 / pap0.x : 0.0;
        const double alpha1 = (!done1 && pap1.x > 1e-300) ? rr1 / pap1.x : 0.0;
        mass_cg_update_kernel<<<grid_n, block>>>(
            N, alpha0, alpha1, d_mass_p, d_mass_ap, d_mass_x, d_mass_r);
        CUDA_CHECK(cudaGetLastError());

        double next_norm0 = 0.0, next_norm1 = 0.0;
        mass_norm_pair(*this, d_mass_r, d_mass_r + N,
                       &next_norm0, &next_norm1);
        const double next_rr0 = next_norm0 * next_norm0;
        const double next_rr1 = next_norm1 * next_norm1;
        const double beta0 = (!done0 && rr0 > 1e-300) ? next_rr0 / rr0 : 0.0;
        const double beta1 = (!done1 && rr1 > 1e-300) ? next_rr1 / rr1 : 0.0;
        mass_cg_direction_kernel<<<grid_n, block>>>(
            N, beta0, beta1, d_mass_r, d_mass_p);
        CUDA_CHECK(cudaGetLastError());
        rnorm0 = next_norm0;
        rnorm1 = next_norm1;
        rr0 = next_rr0;
        rr1 = next_rr1;
    }
    CUDA_CHECK(cudaMemcpy(d_z, d_mass_x, (size_t)N2 * sizeof(double2),
                          cudaMemcpyDeviceToDevice));

    const double relative0 = active0 ? rnorm0 / bnorm0 : 0.0;
    const double relative1 = active1 ? rnorm1 / bnorm1 : 0.0;
    mass_apply_count++;
    mass_iteration_count += iterations;
    mass_max_iterations_used = std::max(mass_max_iterations_used, iterations);
    mass_max_relative_residual = std::max(
        mass_max_relative_residual, std::max(relative0, relative1));
    if (mass_apply_count == 1 ||
        (iterations == mass_cg_max_iterations &&
         std::max(relative0, relative1) > 10.0 * mass_cg_tolerance)) {
        printf("  [Precond] Mass CG apply #%lld: iterations=%d rel=(%.2e, %.2e)\n",
               mass_apply_count, iterations, relative0, relative1);
        fflush(stdout);
    }
}

void NearFieldPrecond::apply_calderon_pair_device(
    const double2* d_r0, const double2* d_r1,
    double2* d_z0, double2* d_z1) const
{
    if (!calderon_rwg || !calderon_operator) {
        fprintf(stderr, "Error: RWG Calderon product requested before initialization\n");
        std::abort();
    }
    apply_mass_device(d_r0, d_calderon_mass0);
    apply_mass_device(d_r1, d_calderon_mass1);
    calderon_operator->matvec_batch2_device(
        d_calderon_mass0, d_calderon_mass1, d_calderon_op0, d_calderon_op1);
    calderon_operator_actions++;
    apply_mass_device(d_calderon_op0, d_z0);
    apply_mass_device(d_calderon_op1, d_z1);
}

void NearFieldPrecond::apply_device_complex_pair(
    const double2* d_r0, const double2* d_r1,
    double2* d_z0, double2* d_z1) const
{
    if (calderon_rwg) {
        apply_calderon_pair_device(d_r0, d_r1, d_z0, d_z1);
        return;
    }
    apply_device_complex(d_r0, d_z0);
    apply_device_complex(d_r1, d_z1);
}

void NearFieldPrecond::apply_device_complex(const double2* d_r, double2* d_z) const
{
    if (!device_apply_available()) {
        fprintf(stderr, "Error: GPU preconditioner requested before device upload\n");
        std::abort();
    }

    if (calderon_rwg) {
        apply_mass_device(d_r, d_calderon_mass0);
        calderon_operator->matvec_batch2_device(
            d_calderon_mass0, d_calderon_mass0,
            d_calderon_op0, d_calderon_op1);
        calderon_operator_actions++;
        apply_mass_device(d_calderon_op0, d_z);
        return;
    }

    if (mass_matrix) {
        apply_mass_device(d_r, d_z);
        return;
    }

    if (ilu0) {
        BEM_CUSPARSE_CHECK(cusparseDnVecSetValues(ilu_vec_in,
                                                  const_cast<double2*>(d_r)));
        BEM_CUSPARSE_CHECK(cusparseDnVecSetValues(ilu_vec_out, d_z));
        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        BEM_CUSPARSE_CHECK(cusparseSpSV_solve(
            ilu_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            ilu_mat_l, ilu_vec_in, ilu_vec_tmp, CUDA_C_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, ilu_spsv_l));
        BEM_CUSPARSE_CHECK(cusparseSpSV_solve(
            ilu_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            ilu_mat_u, ilu_vec_tmp, ilu_vec_out, CUDA_C_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, ilu_spsv_u));
        return;
    }

    if (neural_sparse) {
        int block = 256;
        int grid = (N + block - 1) / block;
        neural_sparse_inverse_kernel<<<grid, block>>>(
            N, d_near_row_ptr, d_near_col_idx, d_neural_blk, d_r, d_z);
        CUDA_CHECK(cudaGetLastError());
        if (neural_coarse_rank > 0) {
            neural_coarse_project_kernel<<<neural_coarse_rank, block,
                                           block * sizeof(double2)>>>(
                N2, d_neural_coarse_q, d_r, d_neural_coarse_coeff);
            CUDA_CHECK(cudaGetLastError());
            int coarse_grid = (N2 + block - 1) / block;
            neural_coarse_update_kernel<<<coarse_grid, block>>>(
                N2, neural_coarse_rank, d_neural_coarse_update,
                d_neural_coarse_coeff, d_z);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }

    int block = 256;
    int grid_vec = (N2 + block - 1) / block;
    precond_split_complex_kernel<<<grid_vec, block>>>(d_r, d_r_re, d_r_im, N2);
    CUDA_CHECK(cudaGetLastError());
    apply_block_schwarz_cuda_device(d_r_re, d_r_im, d_z_re, d_z_im);

    if (richardson_sweeps > 0) {
        int grid_N = (N + block - 1) / block;
        for (int sweep = 0; sweep < richardson_sweeps; sweep++) {
            precond_near_matvec_kernel<<<grid_N, block>>>(
                N, d_near_row_ptr, d_near_col_idx,
                d_diag_re, d_diag_im, d_near_re, d_near_im,
                d_z_re, d_z_im, d_Az_re, d_Az_im);
            CUDA_CHECK(cudaGetLastError());
            precond_residual_kernel<<<grid_vec, block>>>(
                d_r_re, d_r_im, d_Az_re, d_Az_im, d_err_re, d_err_im, N2);
            CUDA_CHECK(cudaGetLastError());
            apply_block_schwarz_cuda_device(d_err_re, d_err_im, d_corr_re, d_corr_im);
            precond_axpy_kernel<<<grid_vec, block>>>(
                d_z_re, d_z_im, d_corr_re, d_corr_im, richardson_omega, N2);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    precond_pack_complex_kernel<<<grid_vec, block>>>(d_z_re, d_z_im, d_z, N2);
    CUDA_CHECK(cudaGetLastError());
}

void NearFieldPrecond::apply_block_schwarz_cuda_device(const double* in_re, const double* in_im,
                                                       double* out_re, double* out_im) const
{
    CUDA_CHECK(cudaMemset(out_re, 0, N2 * sizeof(double)));
    CUDA_CHECK(cudaMemset(out_im, 0, N2 * sizeof(double)));
    if (morton_block_jacobi) {
        const int block = 256;
        const size_t shared_bytes =
            (size_t)(2 * device_block_dim + 2 * block) * sizeof(double);
        precond_mbj_kernel<<<device_block_count, block, shared_bytes>>>(
            device_block_count, N, device_block_dim,
            d_block_offsets, d_block_ids, d_block_piv,
            d_block_lu_re, d_block_lu_im,
            in_re, in_im, out_re, out_im);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    int block = 256;
    int grid_blocks = (device_block_count + block - 1) / block;
    precond_schwarz_kernel<<<grid_blocks, block>>>(
        device_block_count, N, device_block_dim,
        d_block_offsets, d_block_ids, d_block_piv,
        d_block_lu_re, d_block_lu_im,
        d_block_weight,
        in_re, in_im,
        out_re, out_im);
    CUDA_CHECK(cudaGetLastError());
}
