#include "block_gmres.h"
#include "bem_fmm.h"
#include "gpu_select.h"
#include "precond.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cuda_runtime.h>

namespace {

__device__ inline double2 d_cmul(double2 a, double2 b)
{
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ inline double2 d_cadd(double2 a, double2 b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}

__device__ inline double2 d_csub(double2 a, double2 b)
{
    return make_double2(a.x - b.x, a.y - b.y);
}

__device__ inline double2 d_cconj_mul(double2 a, double2 b)
{
    return make_double2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}

__global__ void gmres_copy_kernel(double2* dst, const double2* src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = src[i];
}

__global__ void gmres_b_minus_ax_norm_pair_reduce_kernel(const double2* b1, const double2* b2,
                                                         const double2* ax1, const double2* ax2,
                                                         double2* r1, double2* r2,
                                                         double* partial1, double* partial2,
                                                         int n)
{
    extern __shared__ double sh[];
    double* sh1 = sh;
    double* sh2 = sh + blockDim.x;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    double s1 = 0.0, s2 = 0.0;
    if (i < n) {
        double2 rv1 = make_double2(b1[i].x - ax1[i].x, b1[i].y - ax1[i].y);
        double2 rv2 = make_double2(b2[i].x - ax2[i].x, b2[i].y - ax2[i].y);
        r1[i] = rv1;
        r2[i] = rv2;
        s1 = rv1.x * rv1.x + rv1.y * rv1.y;
        s2 = rv2.x * rv2.x + rv2.y * rv2.y;
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
        partial1[blockIdx.x] = sh1[0];
        partial2[blockIdx.x] = sh2[0];
    }
}

__global__ void gmres_scale_one_kernel(double2* dst, const double2* src, double s, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = make_double2(src[i].x * s, src[i].y * s);
}

__global__ void gmres_axpy_one_kernel(double2* y, double2 alpha, const double2* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double2 v = d_cmul(alpha, x[i]);
    y[i] = make_double2(y[i].x + v.x, y[i].y + v.y);
}

__global__ void bicgstab_init_p_pair_kernel(double2* p1, double2* p2,
                                            double2* v1, double2* v2,
                                            const double2* r1, const double2* r2,
                                            int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    p1[i] = r1[i];
    p2[i] = r2[i];
    v1[i] = make_double2(0.0, 0.0);
    v2[i] = make_double2(0.0, 0.0);
}

__global__ void bicgstab_update_p_pair_kernel(double2* p1, double2* p2,
                                              const double2* r1, const double2* r2,
                                              const double2* v1, const double2* v2,
                                              double2 beta1, double2 beta2,
                                              double2 omega1, double2 omega2,
                                              int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double2 ov1 = d_cmul(omega1, v1[i]);
    double2 ov2 = d_cmul(omega2, v2[i]);
    p1[i] = d_cadd(r1[i], d_cmul(beta1, d_csub(p1[i], ov1)));
    p2[i] = d_cadd(r2[i], d_cmul(beta2, d_csub(p2[i], ov2)));
}

__global__ void bicgstab_s_pair_kernel(const double2* r1, const double2* r2,
                                       const double2* v1, const double2* v2,
                                       double2 alpha1, double2 alpha2,
                                       double2* s1, double2* s2, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    s1[i] = d_csub(r1[i], d_cmul(alpha1, v1[i]));
    s2[i] = d_csub(r2[i], d_cmul(alpha2, v2[i]));
}

__global__ void bicgstab_update_xr_pair_kernel(double2* x1, double2* x2,
                                               double2* r1, double2* r2,
                                               const double2* p1, const double2* p2,
                                               const double2* s1, const double2* s2,
                                               const double2* t1, const double2* t2,
                                               double2 alpha1, double2 alpha2,
                                               double2 omega1, double2 omega2,
                                               int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double2 ap1 = d_cmul(alpha1, p1[i]);
    double2 ap2 = d_cmul(alpha2, p2[i]);
    double2 os1 = d_cmul(omega1, s1[i]);
    double2 os2 = d_cmul(omega2, s2[i]);
    x1[i] = d_cadd(x1[i], d_cadd(ap1, os1));
    x2[i] = d_cadd(x2[i], d_cadd(ap2, os2));
    r1[i] = d_csub(s1[i], d_cmul(omega1, t1[i]));
    r2[i] = d_csub(s2[i], d_cmul(omega2, t2[i]));
}

__global__ void bicgstab_update_x_s_pair_kernel(double2* x1, double2* x2,
                                                const double2* p1, const double2* p2,
                                                double2 alpha1, double2 alpha2,
                                                int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    x1[i] = d_cadd(x1[i], d_cmul(alpha1, p1[i]));
    x2[i] = d_cadd(x2[i], d_cmul(alpha2, p2[i]));
}

__global__ void cgs_update_u_p_pair_kernel(double2* u1, double2* u2,
                                           double2* p1, double2* p2,
                                           const double2* r1, const double2* r2,
                                           const double2* q1, const double2* q2,
                                           double2 beta1, double2 beta2,
                                           int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double2 bq1 = d_cmul(beta1, q1[i]);
    double2 bq2 = d_cmul(beta2, q2[i]);
    double2 new_u1 = d_cadd(r1[i], bq1);
    double2 new_u2 = d_cadd(r2[i], bq2);
    double2 inner1 = d_cadd(q1[i], d_cmul(beta1, p1[i]));
    double2 inner2 = d_cadd(q2[i], d_cmul(beta2, p2[i]));
    u1[i] = new_u1;
    u2[i] = new_u2;
    p1[i] = d_cadd(new_u1, d_cmul(beta1, inner1));
    p2[i] = d_cadd(new_u2, d_cmul(beta2, inner2));
}

__global__ void cgs_update_q_s_x_pair_kernel(double2* q1, double2* q2,
                                             double2* s1, double2* s2,
                                             double2* x1, double2* x2,
                                             const double2* u1, const double2* u2,
                                             const double2* v1, const double2* v2,
                                             double2 alpha1, double2 alpha2,
                                             int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double2 new_q1 = d_csub(u1[i], d_cmul(alpha1, v1[i]));
    double2 new_q2 = d_csub(u2[i], d_cmul(alpha2, v2[i]));
    double2 y1 = d_cadd(u1[i], new_q1);
    double2 y2 = d_cadd(u2[i], new_q2);
    q1[i] = new_q1;
    q2[i] = new_q2;
    s1[i] = y1;
    s2[i] = y2;
    x1[i] = d_cadd(x1[i], d_cmul(alpha1, y1));
    x2[i] = d_cadd(x2[i], d_cmul(alpha2, y2));
}

__global__ void cgs_update_r_pair_kernel(double2* r1, double2* r2,
                                         const double2* ay1, const double2* ay2,
                                         double2 alpha1, double2 alpha2,
                                         int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    r1[i] = d_csub(r1[i], d_cmul(alpha1, ay1[i]));
    r2[i] = d_csub(r2[i], d_cmul(alpha2, ay2[i]));
}

__global__ void gmres_norm_pair_reduce_kernel(const double2* a1, const double2* a2,
                                              double* partial1, double* partial2, int n)
{
    extern __shared__ double sh[];
    double* sh1 = sh;
    double* sh2 = sh + blockDim.x;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    double s1 = 0.0, s2 = 0.0;
    if (i < n) {
        double2 v1 = a1[i];
        double2 v2 = a2[i];
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
        partial1[blockIdx.x] = sh1[0];
        partial2[blockIdx.x] = sh2[0];
    }
}

__global__ void gmres_dot_pair_reduce_kernel(const double2* v1, const double2* w1,
                                             const double2* v2, const double2* w2,
                                             double* pr1, double* pi1,
                                             double* pr2, double* pi2, int n)
{
    extern __shared__ double sh[];
    double* r1 = sh;
    double* i1 = sh + blockDim.x;
    double* r2 = sh + 2 * blockDim.x;
    double* i2 = sh + 3 * blockDim.x;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double ar1 = 0.0, ai1 = 0.0, ar2 = 0.0, ai2 = 0.0;
    if (idx < n) {
        double2 z1 = d_cconj_mul(v1[idx], w1[idx]);
        double2 z2 = d_cconj_mul(v2[idx], w2[idx]);
        ar1 = z1.x; ai1 = z1.y;
        ar2 = z2.x; ai2 = z2.y;
    }
    r1[tid] = ar1; i1[tid] = ai1;
    r2[tid] = ar2; i2[tid] = ai2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            r1[tid] += r1[tid + stride];
            i1[tid] += i1[tid + stride];
            r2[tid] += r2[tid + stride];
            i2[tid] += i2[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        pr1[blockIdx.x] = r1[0];
        pi1[blockIdx.x] = i1[0];
        pr2[blockIdx.x] = r2[0];
        pi2[blockIdx.x] = i2[0];
    }
}

__global__ void gmres_dot_column_pair_reduce_kernel(const double2* V1, const double2* w1,
                                                    const double2* V2, const double2* w2,
                                                    double2* partial1, double2* partial2,
                                                    int n, int red_grid)
{
    extern __shared__ double sh[];
    double* r1 = sh;
    double* i1 = sh + blockDim.x;
    double* r2 = sh + 2 * blockDim.x;
    double* i2 = sh + 3 * blockDim.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double ar1 = 0.0, ai1 = 0.0, ar2 = 0.0, ai2 = 0.0;
    if (idx < n) {
        const double2* vrow1 = V1 + (size_t)row * n;
        const double2* vrow2 = V2 + (size_t)row * n;
        double2 z1 = d_cconj_mul(vrow1[idx], w1[idx]);
        double2 z2 = d_cconj_mul(vrow2[idx], w2[idx]);
        ar1 = z1.x; ai1 = z1.y;
        ar2 = z2.x; ai2 = z2.y;
    }
    r1[tid] = ar1; i1[tid] = ai1;
    r2[tid] = ar2; i2[tid] = ai2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            r1[tid] += r1[tid + stride];
            i1[tid] += i1[tid + stride];
            r2[tid] += r2[tid + stride];
            i2[tid] += i2[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        int out = row * red_grid + blockIdx.x;
        partial1[out] = make_double2(r1[0], i1[0]);
        partial2[out] = make_double2(r2[0], i2[0]);
    }
}

__global__ void gmres_dot_column_final_kernel(const double2* partial1, const double2* partial2,
                                              double2* h1, double2* h2,
                                              int red_grid)
{
    extern __shared__ double sh[];
    double* r1 = sh;
    double* i1 = sh + blockDim.x;
    double* r2 = sh + 2 * blockDim.x;
    double* i2 = sh + 3 * blockDim.x;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    double sr1 = 0.0, si1 = 0.0, sr2 = 0.0, si2 = 0.0;
    const double2* p1 = partial1 + (size_t)row * red_grid;
    const double2* p2 = partial2 + (size_t)row * red_grid;
    for (int i = tid; i < red_grid; i += blockDim.x) {
        double2 a = p1[i];
        double2 b = p2[i];
        sr1 += a.x; si1 += a.y;
        sr2 += b.x; si2 += b.y;
    }
    r1[tid] = sr1; i1[tid] = si1;
    r2[tid] = sr2; i2[tid] = si2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            r1[tid] += r1[tid + stride];
            i1[tid] += i1[tid + stride];
            r2[tid] += r2[tid + stride];
            i2[tid] += i2[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        h1[row] = make_double2(r1[0], i1[0]);
        h2[row] = make_double2(r2[0], i2[0]);
    }
}

__global__ void gmres_axpy_column_pair_kernel(double2* w1, double2* w2,
                                              const double2* V1, const double2* V2,
                                              const double2* h1, const double2* h2,
                                              int n, int count)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
        return;
    double2 acc1 = make_double2(0.0, 0.0);
    double2 acc2 = make_double2(0.0, 0.0);
    for (int row = 0; row < count; row++) {
        double2 a1 = d_cmul(h1[row], V1[(size_t)row * n + k]);
        double2 a2 = d_cmul(h2[row], V2[(size_t)row * n + k]);
        acc1.x += a1.x; acc1.y += a1.y;
        acc2.x += a2.x; acc2.y += a2.y;
    }
    w1[k] = make_double2(w1[k].x - acc1.x, w1[k].y - acc1.y);
    w2[k] = make_double2(w2[k].x - acc2.x, w2[k].y - acc2.y);
}

__global__ void gmres_reduce2_final_kernel(const double* in1, const double* in2,
                                           double* out1, double* out2, int n)
{
    extern __shared__ double sh[];
    double* sh1 = sh;
    double* sh2 = sh + blockDim.x;
    int tid = threadIdx.x;
    double s1 = 0.0, s2 = 0.0;
    for (int i = tid; i < n; i += blockDim.x) {
        s1 += in1[i];
        s2 += in2[i];
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
        out1[0] = sh1[0];
        out2[0] = sh2[0];
    }
}

__global__ void gmres_reduce4_final_kernel(const double* in1, const double* in2,
                                           const double* in3, const double* in4,
                                           double* out1, double* out2,
                                           double* out3, double* out4, int n)
{
    extern __shared__ double sh[];
    double* sh1 = sh;
    double* sh2 = sh + blockDim.x;
    double* sh3 = sh + 2 * blockDim.x;
    double* sh4 = sh + 3 * blockDim.x;
    int tid = threadIdx.x;
    double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
    for (int i = tid; i < n; i += blockDim.x) {
        s1 += in1[i];
        s2 += in2[i];
        s3 += in3[i];
        s4 += in4[i];
    }
    sh1[tid] = s1;
    sh2[tid] = s2;
    sh3[tid] = s3;
    sh4[tid] = s4;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh1[tid] += sh1[tid + stride];
            sh2[tid] += sh2[tid + stride];
            sh3[tid] += sh3[tid + stride];
            sh4[tid] += sh4[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out1[0] = sh1[0];
        out2[0] = sh2[0];
        out3[0] = sh3[0];
        out4[0] = sh4[0];
    }
}

__global__ void bicgstab_ts_tt_pair_reduce_kernel(const double2* t1, const double2* s1,
                                                  const double2* t2, const double2* s2,
                                                  double* ts1r, double* ts1i,
                                                  double* ts2r, double* ts2i,
                                                  double* tt1, double* tt2,
                                                  int n)
{
    extern __shared__ double sh[];
    double* a = sh;
    double* b = sh + blockDim.x;
    double* c = sh + 2 * blockDim.x;
    double* d = sh + 3 * blockDim.x;
    double* e = sh + 4 * blockDim.x;
    double* f = sh + 5 * blockDim.x;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double l_a = 0.0, l_b = 0.0, l_c = 0.0, l_d = 0.0, l_e = 0.0, l_f = 0.0;
    if (idx < n) {
        double2 tv1 = t1[idx];
        double2 sv1 = s1[idx];
        double2 tv2 = t2[idx];
        double2 sv2 = s2[idx];
        double2 z1 = d_cconj_mul(tv1, sv1);
        double2 z2 = d_cconj_mul(tv2, sv2);
        l_a = z1.x;
        l_b = z1.y;
        l_c = z2.x;
        l_d = z2.y;
        l_e = tv1.x * tv1.x + tv1.y * tv1.y;
        l_f = tv2.x * tv2.x + tv2.y * tv2.y;
    }
    a[tid] = l_a; b[tid] = l_b; c[tid] = l_c;
    d[tid] = l_d; e[tid] = l_e; f[tid] = l_f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            a[tid] += a[tid + stride];
            b[tid] += b[tid + stride];
            c[tid] += c[tid + stride];
            d[tid] += d[tid + stride];
            e[tid] += e[tid + stride];
            f[tid] += f[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        ts1r[blockIdx.x] = a[0];
        ts1i[blockIdx.x] = b[0];
        ts2r[blockIdx.x] = c[0];
        ts2i[blockIdx.x] = d[0];
        tt1[blockIdx.x] = e[0];
        tt2[blockIdx.x] = f[0];
    }
}

__global__ void bicgstab_reduce6_final_kernel(const double* ts1r, const double* ts1i,
                                              const double* ts2r, const double* ts2i,
                                              const double* tt1, const double* tt2,
                                              double* out4, double* out2, int n)
{
    extern __shared__ double sh[];
    double* a = sh;
    double* b = sh + blockDim.x;
    double* c = sh + 2 * blockDim.x;
    double* d = sh + 3 * blockDim.x;
    double* e = sh + 4 * blockDim.x;
    double* f = sh + 5 * blockDim.x;
    int tid = threadIdx.x;
    double l_a = 0.0, l_b = 0.0, l_c = 0.0, l_d = 0.0, l_e = 0.0, l_f = 0.0;
    for (int i = tid; i < n; i += blockDim.x) {
        l_a += ts1r[i];
        l_b += ts1i[i];
        l_c += ts2r[i];
        l_d += ts2i[i];
        l_e += tt1[i];
        l_f += tt2[i];
    }
    a[tid] = l_a; b[tid] = l_b; c[tid] = l_c;
    d[tid] = l_d; e[tid] = l_e; f[tid] = l_f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            a[tid] += a[tid + stride];
            b[tid] += b[tid + stride];
            c[tid] += c[tid + stride];
            d[tid] += d[tid + stride];
            e[tid] += e[tid + stride];
            f[tid] += f[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out4[0] = a[0];
        out4[1] = b[0];
        out4[2] = c[0];
        out4[3] = d[0];
        out2[0] = e[0];
        out2[1] = f[0];
    }
}

__global__ void gmres_update_x_pair_kernel(double2* x1, double2* x2,
                                           const double2* V1, const double2* V2,
                                           const double2* y1, const double2* y2,
                                           int n, int restart, int m1, int m2)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
        return;
    double2 acc1 = make_double2(0.0, 0.0);
    double2 acc2 = make_double2(0.0, 0.0);
    for (int i = 0; i < m1; i++) {
        double2 yi = y1[i];
        double2 vi = V1[(size_t)i * n + k];
        double2 v = d_cmul(yi, vi);
        acc1.x += v.x; acc1.y += v.y;
    }
    for (int i = 0; i < m2; i++) {
        double2 yi = y2[i];
        double2 vi = V2[(size_t)i * n + k];
        double2 v = d_cmul(yi, vi);
        acc2.x += v.x; acc2.y += v.y;
    }
    x1[k] = make_double2(x1[k].x + acc1.x, x1[k].y + acc1.y);
    x2[k] = make_double2(x2[k].x + acc2.x, x2[k].y + acc2.y);
}

struct GmresDeviceBufferCache {
    int n = 0;
    int restart = 0;
    int red_grid = 0;
    double2 *d_b1 = nullptr, *d_b2 = nullptr, *d_x1 = nullptr, *d_x2 = nullptr;
    double2 *d_best_x1 = nullptr, *d_best_x2 = nullptr;
    double2 *d_r1 = nullptr, *d_r2 = nullptr, *d_w1 = nullptr, *d_w2 = nullptr;
    double2 *d_V1 = nullptr, *d_V2 = nullptr, *d_y1 = nullptr, *d_y2 = nullptr;
    double *d_p1 = nullptr, *d_p2 = nullptr, *d_pr1 = nullptr, *d_pi1 = nullptr;
    double *d_pr2 = nullptr, *d_pi2 = nullptr;
    double *d_reduce2 = nullptr, *d_reduce4 = nullptr;
    double2 *d_col_partial1 = nullptr, *d_col_partial2 = nullptr;
    double2 *d_hcol1 = nullptr, *d_hcol2 = nullptr;

    ~GmresDeviceBufferCache() { release(); }

    void release()
    {
        cudaFree(d_b1); cudaFree(d_b2); cudaFree(d_x1); cudaFree(d_x2);
        cudaFree(d_best_x1); cudaFree(d_best_x2);
        cudaFree(d_r1); cudaFree(d_r2); cudaFree(d_w1); cudaFree(d_w2);
        cudaFree(d_V1); cudaFree(d_V2); cudaFree(d_y1); cudaFree(d_y2);
        cudaFree(d_p1); cudaFree(d_p2); cudaFree(d_pr1); cudaFree(d_pi1);
        cudaFree(d_pr2); cudaFree(d_pi2);
        cudaFree(d_reduce2); cudaFree(d_reduce4);
        cudaFree(d_col_partial1); cudaFree(d_col_partial2);
        cudaFree(d_hcol1); cudaFree(d_hcol2);
        d_b1 = d_b2 = d_x1 = d_x2 = d_best_x1 = d_best_x2 = nullptr;
        d_r1 = d_r2 = d_w1 = d_w2 = nullptr;
        d_V1 = d_V2 = d_y1 = d_y2 = nullptr;
        d_p1 = d_p2 = d_pr1 = d_pi1 = d_pr2 = d_pi2 = nullptr;
        d_reduce2 = d_reduce4 = nullptr;
        d_col_partial1 = d_col_partial2 = d_hcol1 = d_hcol2 = nullptr;
        n = 0;
        restart = 0;
        red_grid = 0;
    }

    void ensure(int n_new, int restart_new, int red_grid_new)
    {
        if (n == n_new && restart == restart_new && red_grid == red_grid_new)
            return;
        release();
        n = n_new;
        restart = restart_new;
        red_grid = red_grid_new;
        const size_t vec_bytes = (size_t)n * sizeof(double2);
        CUDA_CHECK(cudaMalloc(&d_b1, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_b2, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_x1, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_x2, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_best_x1, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_best_x2, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_r1, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_r2, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_w1, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_w2, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_V1, (size_t)n * (restart + 1) * sizeof(double2)));
        CUDA_CHECK(cudaMalloc(&d_V2, (size_t)n * (restart + 1) * sizeof(double2)));
        CUDA_CHECK(cudaMalloc(&d_y1, (size_t)restart * sizeof(double2)));
        CUDA_CHECK(cudaMalloc(&d_y2, (size_t)restart * sizeof(double2)));
        CUDA_CHECK(cudaMalloc(&d_p1, (size_t)red_grid * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_p2, (size_t)red_grid * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pr1, (size_t)red_grid * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pi1, (size_t)red_grid * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pr2, (size_t)red_grid * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pi2, (size_t)red_grid * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_reduce2, 2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_reduce4, 4 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_col_partial1, (size_t)(restart + 1) * red_grid * sizeof(double2)));
        CUDA_CHECK(cudaMalloc(&d_col_partial2, (size_t)(restart + 1) * red_grid * sizeof(double2)));
        CUDA_CHECK(cudaMalloc(&d_hcol1, (size_t)(restart + 1) * sizeof(double2)));
        CUDA_CHECK(cudaMalloc(&d_hcol2, (size_t)(restart + 1) * sizeof(double2)));
    }
};

static GmresDeviceBufferCache& gmres_device_buffer_cache()
{
    static GmresDeviceBufferCache cache;
    return cache;
}

struct BicgstabDeviceBufferCache {
    int n = 0;
    int red_grid = 0;
    double2 *d_b1 = nullptr, *d_b2 = nullptr, *d_x1 = nullptr, *d_x2 = nullptr;
    double2 *d_best_x1 = nullptr, *d_best_x2 = nullptr;
    double2 *d_r1 = nullptr, *d_r2 = nullptr, *d_rh1 = nullptr, *d_rh2 = nullptr;
    double2 *d_p1 = nullptr, *d_p2 = nullptr, *d_v1 = nullptr, *d_v2 = nullptr;
    double2 *d_s1 = nullptr, *d_s2 = nullptr, *d_t1 = nullptr, *d_t2 = nullptr;
    double *d_np1 = nullptr, *d_np2 = nullptr;
    double *d_pr1 = nullptr, *d_pi1 = nullptr, *d_pr2 = nullptr, *d_pi2 = nullptr;
    double *d_reduce2 = nullptr, *d_reduce4 = nullptr;

    ~BicgstabDeviceBufferCache() { release(); }

    void release()
    {
        cudaFree(d_b1); cudaFree(d_b2); cudaFree(d_x1); cudaFree(d_x2);
        cudaFree(d_best_x1); cudaFree(d_best_x2);
        cudaFree(d_r1); cudaFree(d_r2); cudaFree(d_rh1); cudaFree(d_rh2);
        cudaFree(d_p1); cudaFree(d_p2); cudaFree(d_v1); cudaFree(d_v2);
        cudaFree(d_s1); cudaFree(d_s2); cudaFree(d_t1); cudaFree(d_t2);
        cudaFree(d_np1); cudaFree(d_np2);
        cudaFree(d_pr1); cudaFree(d_pi1); cudaFree(d_pr2); cudaFree(d_pi2);
        cudaFree(d_reduce2); cudaFree(d_reduce4);
        d_b1 = d_b2 = d_x1 = d_x2 = d_best_x1 = d_best_x2 = nullptr;
        d_r1 = d_r2 = d_rh1 = d_rh2 = nullptr;
        d_p1 = d_p2 = d_v1 = d_v2 = d_s1 = d_s2 = d_t1 = d_t2 = nullptr;
        d_np1 = d_np2 = d_pr1 = d_pi1 = d_pr2 = d_pi2 = nullptr;
        d_reduce2 = d_reduce4 = nullptr;
        n = 0;
        red_grid = 0;
    }

    void ensure(int n_new, int red_grid_new)
    {
        if (n == n_new && red_grid == red_grid_new)
            return;
        release();
        n = n_new;
        red_grid = red_grid_new;
        const size_t vec_bytes = (size_t)n * sizeof(double2);
        CUDA_CHECK(cudaMalloc(&d_b1, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_b2, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_x1, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_x2, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_best_x1, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_best_x2, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_r1, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_r2, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_rh1, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_rh2, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_p1, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_p2, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_v1, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_v2, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_s1, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_s2, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_t1, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_t2, vec_bytes));
        CUDA_CHECK(cudaMalloc(&d_np1, (size_t)red_grid * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_np2, (size_t)red_grid * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pr1, (size_t)red_grid * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pi1, (size_t)red_grid * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pr2, (size_t)red_grid * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pi2, (size_t)red_grid * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_reduce2, 2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_reduce4, 4 * sizeof(double)));
    }
};

static BicgstabDeviceBufferCache& bicgstab_device_buffer_cache()
{
    static BicgstabDeviceBufferCache cache;
    return cache;
}

} // namespace

static cdouble orthogonalize_against_p(const cdouble* vi, cdouble* w, int n)
{
    double sr = 0.0, si = 0.0;
    cdouble hij = cdouble(0);
    #pragma omp parallel
    {
        #pragma omp for reduction(+:sr,si) schedule(static)
        for (int i = 0; i < n; i++) {
            cdouble v = std::conj(vi[i]) * w[i];
            sr += v.real();
            si += v.imag();
        }
        #pragma omp single
        {
            hij = cdouble(sr, si);
        }
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++)
            w[i] -= hij * vi[i];
    }
    return hij;
}

static void orthogonalize_against_pair_p(const cdouble* vi1, cdouble* w1,
                                         const cdouble* vi2, cdouble* w2,
                                         int n, cdouble& hij1, cdouble& hij2)
{
    double sr1 = 0.0, si1 = 0.0;
    double sr2 = 0.0, si2 = 0.0;
    hij1 = cdouble(0);
    hij2 = cdouble(0);
    #pragma omp parallel
    {
        #pragma omp for reduction(+:sr1,si1,sr2,si2) schedule(static)
        for (int i = 0; i < n; i++) {
            cdouble v1 = std::conj(vi1[i]) * w1[i];
            cdouble v2 = std::conj(vi2[i]) * w2[i];
            sr1 += v1.real();
            si1 += v1.imag();
            sr2 += v2.real();
            si2 += v2.imag();
        }
        #pragma omp single
        {
            hij1 = cdouble(sr1, si1);
            hij2 = cdouble(sr2, si2);
        }
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            w1[i] -= hij1 * vi1[i];
            w2[i] -= hij2 * vi2[i];
        }
    }
}

static double norm_p(const cdouble* a, int n) {
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int i = 0; i < n; i++)
        s += std::norm(a[i]);
    return std::sqrt(s);
}

static void norm_pair_p(const cdouble* a1, const cdouble* a2, int n,
                        double& out1, double& out2)
{
    double s1 = 0.0, s2 = 0.0;
    #pragma omp parallel for reduction(+:s1,s2) schedule(static)
    for (int i = 0; i < n; i++) {
        s1 += std::norm(a1[i]);
        s2 += std::norm(a2[i]);
    }
    out1 = std::sqrt(s1);
    out2 = std::sqrt(s2);
}

static bool finite_complex(cdouble z)
{
    return std::isfinite(z.real()) && std::isfinite(z.imag());
}

static bool solve_gmres_y(int restart, int m,
                          const std::vector<cdouble>& H,
                          const std::vector<cdouble>& s,
                          std::vector<cdouble>& y)
{
    y.resize(m);
    for (int i = m - 1; i >= 0; i--) {
        y[i] = s[i];
        for (int k = i + 1; k < m; k++)
            y[i] -= H[i * restart + k] * y[k];
        cdouble diag = H[i * restart + i];
        if (!finite_complex(diag) || std::abs(diag) <= 1e-300)
            return false;
        y[i] /= diag;
        if (!finite_complex(y[i]))
            return false;
    }
    return true;
}

static int gmres_step_update(int n, int restart, int m,
                             const std::vector<cdouble>& H,
                             const std::vector<cdouble>& s,
                             const std::vector<cdouble>& V,
                             const std::vector<cdouble>& Z,
                             bool has_precond,
                             bool store_z,
                             NearFieldPrecond* precond,
                             cdouble* x,
                             std::vector<cdouble>& y,
                             std::vector<cdouble>& ztmp)
{
    if (m <= 0)
        return 0;

    if (!solve_gmres_y(restart, m, H, s, y))
        return 1;

    if (!has_precond || store_z) {
        const std::vector<cdouble>& basis = (has_precond && store_z) ? Z : V;
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; k++) {
            cdouble acc = cdouble(0);
            for (int i = 0; i < m; i++)
                acc += y[i] * basis[(size_t)i * n + k];
            x[k] += acc;
        }
        return 0;
    }

    ztmp.resize(n);
    for (int i = 0; i < m; i++) {
        const cdouble* vi = &V[(size_t)i * n];
        precond->apply(vi, ztmp.data());
        vi = ztmp.data();
        cdouble yi = y[i];
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; k++)
            x[k] += yi * vi[k];
    }
    return 0;
}

static int gmres_step_update_pair(int n, int restart, int m1, int m2,
                                  const std::vector<cdouble>& H1,
                                  const std::vector<cdouble>& H2,
                                  const std::vector<cdouble>& s1,
                                  const std::vector<cdouble>& s2,
                                  const std::vector<cdouble>& V1,
                                  const std::vector<cdouble>& V2,
                                  const std::vector<cdouble>& Z1,
                                  const std::vector<cdouble>& Z2,
                                  bool has_precond,
                                  bool store_z,
                                  NearFieldPrecond* precond,
                                  cdouble* x1,
                                  cdouble* x2,
                                  std::vector<cdouble>& y1,
                                  std::vector<cdouble>& y2,
                                  std::vector<cdouble>& ztmp,
                                  std::vector<cdouble>& ztmp2)
{
    bool fused_update = bem_env_flag_enabled("BEM_GMRES_FUSED_UPDATE", true);
    if (!fused_update) {
        int rc1 = gmres_step_update(n, restart, m1, H1, s1, V1, Z1,
                                    has_precond, store_z, precond, x1, y1, ztmp);
        int rc2 = gmres_step_update(n, restart, m2, H2, s2, V2, Z2,
                                    has_precond, store_z, precond, x2, y2, ztmp2);
        if (rc1 != 0 || rc2 != 0)
            return 1;
        return 0;
    }

    if (m1 <= 0 && m2 <= 0)
        return 0;

    if (!solve_gmres_y(restart, m1, H1, s1, y1))
        return 1;
    if (!solve_gmres_y(restart, m2, H2, s2, y2))
        return 1;

    if (has_precond && !store_z) {
        ztmp.resize(n);
        ztmp2.resize(n);
        int mm = std::max(m1, m2);
        for (int i = 0; i < mm; i++) {
            const bool do1 = i < m1;
            const bool do2 = i < m2;
            if (do1)
                precond->apply(&V1[(size_t)i * n], ztmp.data());
            if (do2)
                precond->apply(&V2[(size_t)i * n], ztmp2.data());
            cdouble yi1 = do1 ? y1[i] : cdouble(0);
            cdouble yi2 = do2 ? y2[i] : cdouble(0);
            const cdouble* z1p = ztmp.data();
            const cdouble* z2p = ztmp2.data();
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < n; k++) {
                if (do1)
                    x1[k] += yi1 * z1p[k];
                if (do2)
                    x2[k] += yi2 * z2p[k];
            }
        }
        return 0;
    }

    const std::vector<cdouble>& basis1 = (has_precond && store_z) ? Z1 : V1;
    const std::vector<cdouble>& basis2 = (has_precond && store_z) ? Z2 : V2;
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        cdouble acc1 = cdouble(0);
        cdouble acc2 = cdouble(0);
        for (int i = 0; i < m1; i++)
            acc1 += y1[i] * basis1[(size_t)i * n + k];
        for (int i = 0; i < m2; i++)
            acc2 += y2[i] * basis2[(size_t)i * n + k];
        x1[k] += acc1;
        x2[k] += acc2;
    }
    return 0;
}

int gmres_solve_paired(BemFmmOperator& op,
                       const cdouble* b1, const cdouble* b2,
                       cdouble* x1, cdouble* x2,
                       int restart, double tol, int maxiter,
                       bool verbose, NearFieldPrecond* precond)
{
    GmresPairedWorkspace ws;
    return gmres_solve_paired_ws(op, b1, b2, x1, x2,
                                 restart, tol, maxiter, verbose, precond, ws);
}

static void gpu_norm_pair(double2* d_a1, double2* d_a2, int n,
                          double* d_p1, double* d_p2,
                          double* d_reduce2,
                          double& out1, double& out2)
{
    const int block = 256;
    int grid = (n + block - 1) / block;
    gmres_norm_pair_reduce_kernel<<<grid, block, 2 * block * sizeof(double)>>>(
        d_a1, d_a2, d_p1, d_p2, n);
    CUDA_CHECK(cudaGetLastError());
    gmres_reduce2_final_kernel<<<1, block, 2 * block * sizeof(double)>>>(
        d_p1, d_p2, d_reduce2, d_reduce2 + 1, grid);
    CUDA_CHECK(cudaGetLastError());
    double h_reduce[2];
    CUDA_CHECK(cudaMemcpy(h_reduce, d_reduce2, sizeof(h_reduce), cudaMemcpyDeviceToHost));
    out1 = std::sqrt(h_reduce[0]);
    out2 = std::sqrt(h_reduce[1]);
}

static void gpu_b_minus_ax_norm_pair(const double2* d_b1, const double2* d_b2,
                                     const double2* d_ax1, const double2* d_ax2,
                                     double2* d_r1, double2* d_r2, int n,
                                     double* d_p1, double* d_p2,
                                     double* d_reduce2,
                                     double& out1, double& out2)
{
    const int block = 256;
    int grid = (n + block - 1) / block;
    gmres_b_minus_ax_norm_pair_reduce_kernel<<<grid, block, 2 * block * sizeof(double)>>>(
        d_b1, d_b2, d_ax1, d_ax2, d_r1, d_r2, d_p1, d_p2, n);
    CUDA_CHECK(cudaGetLastError());
    gmres_reduce2_final_kernel<<<1, block, 2 * block * sizeof(double)>>>(
        d_p1, d_p2, d_reduce2, d_reduce2 + 1, grid);
    CUDA_CHECK(cudaGetLastError());
    double h_reduce[2];
    CUDA_CHECK(cudaMemcpy(h_reduce, d_reduce2, sizeof(h_reduce), cudaMemcpyDeviceToHost));
    out1 = std::sqrt(h_reduce[0]);
    out2 = std::sqrt(h_reduce[1]);
}

static double gpu_true_pair_rel_from_host(BemFmmOperator& op,
                                          const cdouble* b1, const cdouble* b2,
                                          const cdouble* x1, const cdouble* x2,
                                          int& matvecs)
{
    const int n = op.system_size;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    const int red_grid = grid;
    const size_t vec_bytes = (size_t)n * sizeof(double2);

    BicgstabDeviceBufferCache& cache = bicgstab_device_buffer_cache();
    cache.ensure(n, red_grid);
    double2 *d_b1 = cache.d_b1, *d_b2 = cache.d_b2;
    double2 *d_x1 = cache.d_x1, *d_x2 = cache.d_x2;
    double2 *d_r1 = cache.d_r1, *d_r2 = cache.d_r2;
    double2 *d_v1 = cache.d_v1, *d_v2 = cache.d_v2;
    double *d_np1 = cache.d_np1, *d_np2 = cache.d_np2;
    double *d_reduce2 = cache.d_reduce2;

    CUDA_CHECK(cudaMemcpy(d_b1, reinterpret_cast<const double2*>(b1), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, reinterpret_cast<const double2*>(b2), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x1, reinterpret_cast<const double2*>(x1), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2, reinterpret_cast<const double2*>(x2), vec_bytes, cudaMemcpyHostToDevice));

    double bnorm1 = 0.0, bnorm2 = 0.0;
    gpu_norm_pair(d_b1, d_b2, n, d_np1, d_np2, d_reduce2, bnorm1, bnorm2);
    if (bnorm1 < 1e-30) bnorm1 = 1.0;
    if (bnorm2 < 1e-30) bnorm2 = 1.0;

    double xnorm1 = 0.0, xnorm2 = 0.0;
    gpu_norm_pair(d_x1, d_x2, n, d_np1, d_np2, d_reduce2, xnorm1, xnorm2);
    const bool warm1 = (xnorm1 > 1e-30);
    const bool warm2 = (xnorm2 > 1e-30);
    if (!warm1 && !warm2)
        return 1.0;

    op.matvec_batch2_device(d_x1, d_x2, d_v1, d_v2);
    matvecs++;
    double rnorm1 = bnorm1, rnorm2 = bnorm2;
    gpu_b_minus_ax_norm_pair(d_b1, d_b2, d_v1, d_v2, d_r1, d_r2, n,
                             d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
    return std::max(warm1 ? rnorm1 / bnorm1 : 1.0,
                    warm2 ? rnorm2 / bnorm2 : 1.0);
}

static void gpu_dot_pair(double2* d_v1, double2* d_w1,
                         double2* d_v2, double2* d_w2, int n,
                         double* d_pr1, double* d_pi1,
                         double* d_pr2, double* d_pi2,
                         double* d_reduce4,
                         cdouble& h1, cdouble& h2)
{
    const int block = 256;
    int grid = (n + block - 1) / block;
    gmres_dot_pair_reduce_kernel<<<grid, block, 4 * block * sizeof(double)>>>(
        d_v1, d_w1, d_v2, d_w2, d_pr1, d_pi1, d_pr2, d_pi2, n);
    CUDA_CHECK(cudaGetLastError());
    gmres_reduce4_final_kernel<<<1, block, 4 * block * sizeof(double)>>>(
        d_pr1, d_pi1, d_pr2, d_pi2,
        d_reduce4, d_reduce4 + 1, d_reduce4 + 2, d_reduce4 + 3, grid);
    CUDA_CHECK(cudaGetLastError());
    double h_reduce[4];
    CUDA_CHECK(cudaMemcpy(h_reduce, d_reduce4, sizeof(h_reduce), cudaMemcpyDeviceToHost));
    h1 = cdouble(h_reduce[0], h_reduce[1]);
    h2 = cdouble(h_reduce[2], h_reduce[3]);
}

static void gpu_bicgstab_ts_tt_pair(double2* d_t1, double2* d_s1,
                                    double2* d_t2, double2* d_s2, int n,
                                    double* d_pr1, double* d_pi1,
                                    double* d_pr2, double* d_pi2,
                                    double* d_np1, double* d_np2,
                                    double* d_reduce4, double* d_reduce2,
                                    cdouble& ts1, cdouble& ts2,
                                    cdouble& tt1, cdouble& tt2)
{
    const int block = 256;
    int grid = (n + block - 1) / block;
    bicgstab_ts_tt_pair_reduce_kernel<<<grid, block, 6 * block * sizeof(double)>>>(
        d_t1, d_s1, d_t2, d_s2,
        d_pr1, d_pi1, d_pr2, d_pi2, d_np1, d_np2, n);
    CUDA_CHECK(cudaGetLastError());
    bicgstab_reduce6_final_kernel<<<1, block, 6 * block * sizeof(double)>>>(
        d_pr1, d_pi1, d_pr2, d_pi2, d_np1, d_np2,
        d_reduce4, d_reduce2, grid);
    CUDA_CHECK(cudaGetLastError());
    double h_reduce4[4];
    double h_reduce2[2];
    CUDA_CHECK(cudaMemcpy(h_reduce4, d_reduce4, sizeof(h_reduce4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_reduce2, d_reduce2, sizeof(h_reduce2), cudaMemcpyDeviceToHost));
    ts1 = cdouble(h_reduce4[0], h_reduce4[1]);
    ts2 = cdouble(h_reduce4[2], h_reduce4[3]);
    tt1 = cdouble(h_reduce2[0], 0.0);
    tt2 = cdouble(h_reduce2[1], 0.0);
}

static void gpu_orthogonalize_column_pair(double2* d_V1, double2* d_w1,
                                          double2* d_V2, double2* d_w2,
                                          int n, int red_grid, int count,
                                          double2* d_col_partial1, double2* d_col_partial2,
                                          double2* d_hcol1, double2* d_hcol2,
                                          std::vector<cdouble>& htmp1,
                                          std::vector<cdouble>& htmp2)
{
    if (count <= 0)
        return;
    const int block = 256;
    dim3 grid(red_grid, count);
    gmres_dot_column_pair_reduce_kernel<<<grid, block, 4 * block * sizeof(double)>>>(
        d_V1, d_w1, d_V2, d_w2, d_col_partial1, d_col_partial2, n, red_grid);
    CUDA_CHECK(cudaGetLastError());
    gmres_dot_column_final_kernel<<<count, block, 4 * block * sizeof(double)>>>(
        d_col_partial1, d_col_partial2, d_hcol1, d_hcol2, red_grid);
    CUDA_CHECK(cudaGetLastError());
    htmp1.resize(count);
    htmp2.resize(count);
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(htmp1.data()), d_hcol1,
                          (size_t)count * sizeof(double2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(htmp2.data()), d_hcol2,
                          (size_t)count * sizeof(double2), cudaMemcpyDeviceToHost));
    int vec_grid = (n + block - 1) / block;
    gmres_axpy_column_pair_kernel<<<vec_grid, block>>>(d_w1, d_w2, d_V1, d_V2,
                                                       d_hcol1, d_hcol2, n, count);
    CUDA_CHECK(cudaGetLastError());
}

static int gmres_solve_paired_device_ws(BemFmmOperator& op,
                                        const cdouble* b1, const cdouble* b2,
                                        cdouble* x1, cdouble* x2,
                                        int restart, double tol, int maxiter,
                                        bool verbose, GmresPairedWorkspace& ws,
                                        NearFieldPrecond* precond = nullptr)
{
    const int n = op.system_size;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    const int red_grid = grid;
    const size_t vec_bytes = (size_t)n * sizeof(double2);
    int ret_matvecs = 0;

    ws.final_relres1 = 0.0;
    ws.final_relres2 = 0.0;
    ws.converged1 = false;
    ws.converged2 = false;
    ws.stopped_stagnant = false;
    ws.numerical_breakdown = false;
    ws.restored_best_iterate = false;
    ws.reached_max_cycles = false;

    GmresDeviceBufferCache& cache = gmres_device_buffer_cache();
    cache.ensure(n, restart, red_grid);
    double2 *d_b1 = cache.d_b1, *d_b2 = cache.d_b2;
    double2 *d_x1 = cache.d_x1, *d_x2 = cache.d_x2;
    double2 *d_best_x1 = cache.d_best_x1, *d_best_x2 = cache.d_best_x2;
    double2 *d_r1 = cache.d_r1, *d_r2 = cache.d_r2;
    double2 *d_w1 = cache.d_w1, *d_w2 = cache.d_w2;
    double2 *d_V1 = cache.d_V1, *d_V2 = cache.d_V2;
    double2 *d_y1 = cache.d_y1, *d_y2 = cache.d_y2;
    double *d_p1 = cache.d_p1, *d_p2 = cache.d_p2;
    double *d_pr1 = cache.d_pr1, *d_pi1 = cache.d_pi1;
    double *d_pr2 = cache.d_pr2, *d_pi2 = cache.d_pi2;
    double *d_reduce2 = cache.d_reduce2, *d_reduce4 = cache.d_reduce4;
    double2 *d_col_partial1 = cache.d_col_partial1, *d_col_partial2 = cache.d_col_partial2;
    double2 *d_hcol1 = cache.d_hcol1, *d_hcol2 = cache.d_hcol2;
    const bool left_precond =
        precond != nullptr &&
        precond->device_apply_available() &&
        bem_env_flag_enabled("BEM_GMRES_DEVICE_PREC", true);

    auto& H1 = ws.H1; auto& H2 = ws.H2;
    auto& cs1 = ws.cs1; auto& sn1 = ws.sn1; auto& s1 = ws.s1;
    auto& cs2 = ws.cs2; auto& sn2 = ws.sn2; auto& s2 = ws.s2;
    auto& ytmp = ws.ytmp; auto& ytmp2 = ws.ytmp2;
    auto& hcol1 = ws.ztmp; auto& hcol2 = ws.ztmp2;
    H1.resize((restart + 1) * restart);
    H2.resize((restart + 1) * restart);
    cs1.resize(restart); sn1.resize(restart); s1.resize(restart + 1);
    cs2.resize(restart); sn2.resize(restart); s2.resize(restart + 1);

    CUDA_CHECK(cudaMemcpy(d_b1, reinterpret_cast<const double2*>(b1), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, reinterpret_cast<const double2*>(b2), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x1, reinterpret_cast<const double2*>(x1), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2, reinterpret_cast<const double2*>(x2), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));

    double true_bnorm1, true_bnorm2;
    gpu_norm_pair(d_b1, d_b2, n, d_p1, d_p2, d_reduce2, true_bnorm1, true_bnorm2);
    if (true_bnorm1 < 1e-30) true_bnorm1 = 1.0;
    if (true_bnorm2 < 1e-30) true_bnorm2 = 1.0;

    double bnorm1 = true_bnorm1, bnorm2 = true_bnorm2;
    if (left_precond) {
        precond->apply_device_complex(d_b1, d_w1);
        precond->apply_device_complex(d_b2, d_w2);
        gpu_norm_pair(d_w1, d_w2, n, d_p1, d_p2, d_reduce2, bnorm1, bnorm2);
        if (bnorm1 < 1e-30) bnorm1 = 1.0;
        if (bnorm2 < 1e-30) bnorm2 = 1.0;
    }

    double xnorm1, xnorm2;
    gpu_norm_pair(d_x1, d_x2, n, d_p1, d_p2, d_reduce2, xnorm1, xnorm2);
    bool warm1 = (xnorm1 > 1e-30);
    bool warm2 = (xnorm2 > 1e-30);
    double rnorm1 = 0.0, rnorm2 = 0.0;
    bool have_rnorm = false;
    if (warm1 || warm2) {
        op.matvec_batch2_device(d_x1, d_x2, d_w1, d_w2);
        ret_matvecs++;
        gpu_b_minus_ax_norm_pair(d_b1, d_b2, d_w1, d_w2, d_r1, d_r2, n,
                                 d_p1, d_p2, d_reduce2, rnorm1, rnorm2);
        have_rnorm = true;
    } else {
        gmres_copy_kernel<<<grid, block>>>(d_r1, d_b1, n);
        gmres_copy_kernel<<<grid, block>>>(d_r2, d_b2, n);
        CUDA_CHECK(cudaGetLastError());
    }

    if (!have_rnorm)
        gpu_norm_pair(d_r1, d_r2, n, d_p1, d_p2, d_reduce2, rnorm1, rnorm2);
    double true_rel1 = rnorm1 / true_bnorm1;
    double true_rel2 = rnorm2 / true_bnorm2;

    if (left_precond) {
        precond->apply_device_complex(d_r1, d_w1);
        precond->apply_device_complex(d_r2, d_w2);
        CUDA_CHECK(cudaMemcpy(d_r1, d_w1, vec_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_r2, d_w2, vec_bytes, cudaMemcpyDeviceToDevice));
        gpu_norm_pair(d_r1, d_r2, n, d_p1, d_p2, d_reduce2, rnorm1, rnorm2);
    }
    double last_rel1 = left_precond ? (rnorm1 / bnorm1) : true_rel1;
    double last_rel2 = left_precond ? (rnorm2 / bnorm2) : true_rel2;
    double warm_max_rel = bem_env_double("BEM_GMRES_WARM_MAX_REL", 1.05);
    if ((warm1 || warm2) && warm_max_rel > 0.0 &&
        std::max(last_rel1, last_rel2) > warm_max_rel) {
        CUDA_CHECK(cudaMemset(d_x1, 0, vec_bytes));
        CUDA_CHECK(cudaMemset(d_x2, 0, vec_bytes));
        CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));
        gmres_copy_kernel<<<grid, block>>>(d_r1, d_b1, n);
        gmres_copy_kernel<<<grid, block>>>(d_r2, d_b2, n);
        CUDA_CHECK(cudaGetLastError());
        true_rel1 = 1.0;
        true_rel2 = 1.0;
        if (left_precond) {
            precond->apply_device_complex(d_r1, d_w1);
            precond->apply_device_complex(d_r2, d_w2);
            CUDA_CHECK(cudaMemcpy(d_r1, d_w1, vec_bytes, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_r2, d_w2, vec_bytes, cudaMemcpyDeviceToDevice));
            gpu_norm_pair(d_r1, d_r2, n, d_p1, d_p2, d_reduce2, rnorm1, rnorm2);
            last_rel1 = rnorm1 / bnorm1;
            last_rel2 = rnorm2 / bnorm2;
        } else {
            rnorm1 = true_bnorm1;
            rnorm2 = true_bnorm2;
            last_rel1 = 1.0;
            last_rel2 = 1.0;
        }
        if (verbose) {
            printf("  [GMRES-paired-GPU] dropped warm start: initial max rel exceeded %.2e\n",
                   warm_max_rel);
            fflush(stdout);
        }
    }
    bool conv1 = (true_rel1 < tol);
    bool conv2 = (true_rel2 < tol);
    double best_rel1 = true_rel1, best_rel2 = true_rel2;

    if (verbose) {
        printf("  [GMRES-paired-GPU] start: res1=%.2e res2=%.2e%s%s\n",
               true_rel1, true_rel2, (warm1 || warm2) ? " (warm)" : "",
               left_precond ? " (left-preconditioned)" : "");
        fflush(stdout);
    }

    bool numerical_breakdown = false;
    bool reached_max_cycles = false;
    for (int cycle = 0; cycle < maxiter && !(conv1 && conv2); cycle++) {
        std::fill(H1.begin(), H1.end(), cdouble(0));
        std::fill(H2.begin(), H2.end(), cdouble(0));
        std::fill(s1.begin(), s1.end(), cdouble(0));
        std::fill(s2.begin(), s2.end(), cdouble(0));

        if (!conv1) {
            gmres_scale_one_kernel<<<grid, block>>>(d_V1, d_r1, 1.0 / rnorm1, n);
            s1[0] = cdouble(rnorm1);
        }
        if (!conv2) {
            gmres_scale_one_kernel<<<grid, block>>>(d_V2, d_r2, 1.0 / rnorm2, n);
            s2[0] = cdouble(rnorm2);
        }
        CUDA_CHECK(cudaGetLastError());

        int m1 = 0, m2 = 0;
        for (int j = 0; j < restart && !(conv1 && conv2); j++) {
            double2* Vj1 = d_V1 + (size_t)j * n;
            double2* Vj2 = d_V2 + (size_t)j * n;
            if (!conv1 && !conv2) {
                op.matvec_batch2_device(Vj1, Vj2, d_w1, d_w2);
                if (left_precond) {
                    precond->apply_device_complex(d_w1, d_r1);
                    precond->apply_device_complex(d_w2, d_r2);
                    CUDA_CHECK(cudaMemcpy(d_w1, d_r1, vec_bytes, cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(d_w2, d_r2, vec_bytes, cudaMemcpyDeviceToDevice));
                }
                ret_matvecs++;
            } else if (!conv1) {
                op.matvec_batch2_device(Vj1, Vj1, d_w1, d_w2);
                if (left_precond) {
                    precond->apply_device_complex(d_w1, d_r1);
                    CUDA_CHECK(cudaMemcpy(d_w1, d_r1, vec_bytes, cudaMemcpyDeviceToDevice));
                }
                ret_matvecs++;
            } else {
                op.matvec_batch2_device(Vj2, Vj2, d_w2, d_w1);
                if (left_precond) {
                    precond->apply_device_complex(d_w2, d_r2);
                    CUDA_CHECK(cudaMemcpy(d_w2, d_r2, vec_bytes, cudaMemcpyDeviceToDevice));
                }
                ret_matvecs++;
            }

            bool column_ortho = bem_env_flag_enabled("BEM_GMRES_COLUMN_ORTHO", true);
            if (column_ortho && !conv1 && !conv2) {
                gpu_orthogonalize_column_pair(d_V1, d_w1, d_V2, d_w2, n, red_grid, j + 1,
                                              d_col_partial1, d_col_partial2, d_hcol1, d_hcol2,
                                              hcol1, hcol2);
                for (int i = 0; i <= j; i++) {
                    H1[i * restart + j] += hcol1[i];
                    H2[i * restart + j] += hcol2[i];
                }
            } else {
                for (int i = 0; i <= j; i++) {
                    cdouble hij1(0), hij2(0);
                    double2* Vi1 = d_V1 + (size_t)i * n;
                    double2* Vi2 = d_V2 + (size_t)i * n;
                    if (!conv1 && !conv2) {
                        gpu_dot_pair(Vi1, d_w1, Vi2, d_w2, n,
                                     d_pr1, d_pi1, d_pr2, d_pi2,
                                     d_reduce4, hij1, hij2);
                    } else if (!conv1) {
                        gpu_dot_pair(Vi1, d_w1, Vi1, d_w1, n,
                                     d_pr1, d_pi1, d_pr2, d_pi2,
                                     d_reduce4, hij1, hij2);
                    } else {
                        gpu_dot_pair(Vi2, d_w2, Vi2, d_w2, n,
                                     d_pr1, d_pi1, d_pr2, d_pi2,
                                     d_reduce4, hij1, hij2);
                    }
                    if (!conv1) {
                        H1[i * restart + j] += hij1;
                        gmres_axpy_one_kernel<<<grid, block>>>(d_w1, make_double2(-hij1.real(), -hij1.imag()), Vi1, n);
                    }
                    if (!conv2) {
                        H2[i * restart + j] += hij2;
                        gmres_axpy_one_kernel<<<grid, block>>>(d_w2, make_double2(-hij2.real(), -hij2.imag()), Vi2, n);
                    }
                    CUDA_CHECK(cudaGetLastError());
                }
            }
            if (bem_env_flag_enabled("BEM_GMRES_REORTH", true)) {
                if (column_ortho && !conv1 && !conv2) {
                    gpu_orthogonalize_column_pair(d_V1, d_w1, d_V2, d_w2, n, red_grid, j + 1,
                                                  d_col_partial1, d_col_partial2, d_hcol1, d_hcol2,
                                                  hcol1, hcol2);
                    for (int i = 0; i <= j; i++) {
                        H1[i * restart + j] += hcol1[i];
                        H2[i * restart + j] += hcol2[i];
                    }
                } else {
                    for (int i = 0; i <= j; i++) {
                        cdouble hij1(0), hij2(0);
                        double2* Vi1 = d_V1 + (size_t)i * n;
                        double2* Vi2 = d_V2 + (size_t)i * n;
                        if (!conv1 && !conv2) {
                            gpu_dot_pair(Vi1, d_w1, Vi2, d_w2, n,
                                         d_pr1, d_pi1, d_pr2, d_pi2,
                                         d_reduce4, hij1, hij2);
                        } else if (!conv1) {
                            gpu_dot_pair(Vi1, d_w1, Vi1, d_w1, n,
                                         d_pr1, d_pi1, d_pr2, d_pi2,
                                         d_reduce4, hij1, hij2);
                        } else {
                            gpu_dot_pair(Vi2, d_w2, Vi2, d_w2, n,
                                         d_pr1, d_pi1, d_pr2, d_pi2,
                                         d_reduce4, hij1, hij2);
                        }
                        if (!conv1) {
                            H1[i * restart + j] += hij1;
                            gmres_axpy_one_kernel<<<grid, block>>>(d_w1, make_double2(-hij1.real(), -hij1.imag()), Vi1, n);
                        }
                        if (!conv2) {
                            H2[i * restart + j] += hij2;
                            gmres_axpy_one_kernel<<<grid, block>>>(d_w2, make_double2(-hij2.real(), -hij2.imag()), Vi2, n);
                        }
                        CUDA_CHECK(cudaGetLastError());
                    }
                }
            }

            double wn1 = 0.0, wn2 = 0.0;
            if (!conv1 && !conv2) {
                gpu_norm_pair(d_w1, d_w2, n, d_p1, d_p2, d_reduce2, wn1, wn2);
            } else if (!conv1) {
                gpu_norm_pair(d_w1, d_w1, n, d_p1, d_p2, d_reduce2, wn1, wn2);
            } else {
                gpu_norm_pair(d_w2, d_w2, n, d_p1, d_p2, d_reduce2, wn1, wn2);
            }
            if (!conv1) {
                H1[(j + 1) * restart + j] = cdouble(wn1);
                if (wn1 > 1e-30)
                    gmres_scale_one_kernel<<<grid, block>>>(d_V1 + (size_t)(j + 1) * n, d_w1, 1.0 / wn1, n);
            }
            if (!conv2) {
                H2[(j + 1) * restart + j] = cdouble(wn2);
                if (wn2 > 1e-30)
                    gmres_scale_one_kernel<<<grid, block>>>(d_V2 + (size_t)(j + 1) * n, d_w2, 1.0 / wn2, n);
            }
            CUDA_CHECK(cudaGetLastError());

            auto givens_step = [&](std::vector<cdouble>& H, std::vector<cdouble>& cs,
                                   std::vector<cdouble>& sn, std::vector<cdouble>& s,
                                   double bnorm, bool& conv, int& m, double& last_rel) {
                if (conv)
                    return;
                for (int i = 0; i < j; i++) {
                    cdouble h0 = H[i * restart + j];
                    cdouble h1 = H[(i + 1) * restart + j];
                    H[i * restart + j]       = std::conj(cs[i]) * h0 + std::conj(sn[i]) * h1;
                    H[(i + 1) * restart + j] = -sn[i] * h0 + cs[i] * h1;
                }
                cdouble h0 = H[j * restart + j];
                cdouble h1 = H[(j + 1) * restart + j];
                double den = std::sqrt(std::norm(h0) + std::norm(h1));
                cs[j] = (den > 1e-30) ? h0 / den : cdouble(1);
                sn[j] = (den > 1e-30) ? h1 / den : cdouble(0);
                H[j * restart + j] = std::conj(cs[j]) * h0 + std::conj(sn[j]) * h1;
                H[(j + 1) * restart + j] = cdouble(0);
                cdouble s0 = s[j];
                s[j] = std::conj(cs[j]) * s0;
                s[j + 1] = -sn[j] * s0;
                m = j + 1;
                last_rel = std::abs(s[j + 1]) / bnorm;
                if (last_rel < tol)
                    conv = true;
            };
            givens_step(H1, cs1, sn1, s1, bnorm1, conv1, m1, last_rel1);
            givens_step(H2, cs2, sn2, s2, bnorm2, conv2, m2, last_rel2);

            if (verbose && (ret_matvecs <= 3 || ret_matvecs % 10 == 0)) {
                printf("    GMRES-GPU iter %d: rel1=%.2e rel2=%.2e%s%s\n",
                       ret_matvecs, conv1 ? 0.0 : last_rel1, conv2 ? 0.0 : last_rel2,
                       conv1 ? " [1:done]" : "", conv2 ? " [2:done]" : "");
                fflush(stdout);
            }
        }

        if (!solve_gmres_y(restart, m1, H1, s1, ytmp) ||
            !solve_gmres_y(restart, m2, H2, s2, ytmp2)) {
            numerical_breakdown = true;
            break;
        }
        if (m1 > 0) {
            CUDA_CHECK(cudaMemcpy(d_y1, reinterpret_cast<const double2*>(ytmp.data()),
                                  (size_t)m1 * sizeof(double2), cudaMemcpyHostToDevice));
        }
        if (m2 > 0) {
            CUDA_CHECK(cudaMemcpy(d_y2, reinterpret_cast<const double2*>(ytmp2.data()),
                                  (size_t)m2 * sizeof(double2), cudaMemcpyHostToDevice));
        }
        gmres_update_x_pair_kernel<<<grid, block>>>(d_x1, d_x2, d_V1, d_V2,
                                                    d_y1, d_y2, n, restart, m1, m2);
        CUDA_CHECK(cudaGetLastError());

        op.matvec_batch2_device(d_x1, d_x2, d_w1, d_w2);
        ret_matvecs++;
        gpu_b_minus_ax_norm_pair(d_b1, d_b2, d_w1, d_w2, d_r1, d_r2, n,
                                 d_p1, d_p2, d_reduce2, rnorm1, rnorm2);
        true_rel1 = rnorm1 / true_bnorm1;
        true_rel2 = rnorm2 / true_bnorm2;
        last_rel1 = true_rel1;
        last_rel2 = true_rel2;
        conv1 = (true_rel1 < tol);
        conv2 = (true_rel2 < tol);

        if (true_rel1 <= best_rel1) {
            best_rel1 = true_rel1;
            CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
        }
        if (true_rel2 <= best_rel2) {
            best_rel2 = true_rel2;
            CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));
        }
        if (verbose) {
            printf("  [GMRES-paired-GPU] restart %d: true rel1=%.2e true rel2=%.2e\n",
                   cycle + 1, true_rel1, true_rel2);
            fflush(stdout);
        }

        if (left_precond && !(conv1 && conv2)) {
            if (!conv1)
                precond->apply_device_complex(d_r1, d_w1);
            if (!conv2)
                precond->apply_device_complex(d_r2, d_w2);
            if (!conv1 && !conv2) {
                CUDA_CHECK(cudaMemcpy(d_r1, d_w1, vec_bytes, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_r2, d_w2, vec_bytes, cudaMemcpyDeviceToDevice));
                gpu_norm_pair(d_r1, d_r2, n, d_p1, d_p2, d_reduce2, rnorm1, rnorm2);
            } else if (!conv1) {
                CUDA_CHECK(cudaMemcpy(d_r1, d_w1, vec_bytes, cudaMemcpyDeviceToDevice));
                gpu_norm_pair(d_r1, d_r1, n, d_p1, d_p2, d_reduce2, rnorm1, rnorm2);
            } else if (!conv2) {
                CUDA_CHECK(cudaMemcpy(d_r2, d_w2, vec_bytes, cudaMemcpyDeviceToDevice));
                gpu_norm_pair(d_r2, d_r2, n, d_p1, d_p2, d_reduce2, rnorm1, rnorm2);
            }
        }
    }

    reached_max_cycles = !(conv1 && conv2) && !numerical_breakdown;
    bool restore1 = !conv1 && best_rel1 < last_rel1;
    bool restore2 = !conv2 && best_rel2 < last_rel2;
    if (restore1 || restore2) {
        if (!restore1)
            CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
        if (!restore2)
            CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x1), d_best_x1, vec_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x2), d_best_x2, vec_bytes, cudaMemcpyDeviceToHost));
        ws.restored_best_iterate = true;
        if (restore1)
            last_rel1 = best_rel1;
        if (restore2)
            last_rel2 = best_rel2;
    } else {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x1), d_x1, vec_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x2), d_x2, vec_bytes, cudaMemcpyDeviceToHost));
    }

    ws.final_relres1 = last_rel1;
    ws.final_relres2 = last_rel2;
    ws.converged1 = (last_rel1 < tol);
    ws.converged2 = (last_rel2 < tol);
    ws.numerical_breakdown = numerical_breakdown;
    ws.reached_max_cycles = reached_max_cycles;
    if (verbose) {
        printf("  [GMRES-paired-GPU] %s, %d matvec evaluations, res1=%.2e res2=%.2e\n",
               (ws.converged1 && ws.converged2) ? "Both converged" : "NOT fully converged",
               ret_matvecs, last_rel1, last_rel2);
        fflush(stdout);
    }
    return ret_matvecs;
}

static inline bool usable_divisor(cdouble z)
{
    return finite_complex(z) && std::norm(z) > 1e-300;
}

static inline double2 to_double2(cdouble z)
{
    return make_double2(z.real(), z.imag());
}

int bicgstab_solve_paired_device_ws(BemFmmOperator& op,
                                    const cdouble* b1, const cdouble* b2,
                                    cdouble* x1, cdouble* x2,
                                    int restart, double tol, int maxiter,
                                    bool verbose, GmresPairedWorkspace& ws)
{
    const int n = op.system_size;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    const int red_grid = grid;
    const size_t vec_bytes = (size_t)n * sizeof(double2);
    const int max_steps = std::max(1, restart) * std::max(1, maxiter);
    int true_every = bem_env_int("BEM_BICGSTAB_TRUE_EVERY", 10);
    true_every = std::max(1, true_every);

    ws.final_relres1 = 0.0;
    ws.final_relres2 = 0.0;
    ws.converged1 = false;
    ws.converged2 = false;
    ws.stopped_stagnant = false;
    ws.numerical_breakdown = false;
    ws.restored_best_iterate = false;
    ws.reached_max_cycles = false;

    BicgstabDeviceBufferCache& cache = bicgstab_device_buffer_cache();
    cache.ensure(n, red_grid);
    double2 *d_b1 = cache.d_b1, *d_b2 = cache.d_b2;
    double2 *d_x1 = cache.d_x1, *d_x2 = cache.d_x2;
    double2 *d_best_x1 = cache.d_best_x1, *d_best_x2 = cache.d_best_x2;
    double2 *d_r1 = cache.d_r1, *d_r2 = cache.d_r2;
    double2 *d_rh1 = cache.d_rh1, *d_rh2 = cache.d_rh2;
    double2 *d_p1 = cache.d_p1, *d_p2 = cache.d_p2;
    double2 *d_v1 = cache.d_v1, *d_v2 = cache.d_v2;
    double2 *d_s1 = cache.d_s1, *d_s2 = cache.d_s2;
    double2 *d_t1 = cache.d_t1, *d_t2 = cache.d_t2;
    double *d_np1 = cache.d_np1, *d_np2 = cache.d_np2;
    double *d_pr1 = cache.d_pr1, *d_pi1 = cache.d_pi1;
    double *d_pr2 = cache.d_pr2, *d_pi2 = cache.d_pi2;
    double *d_reduce2 = cache.d_reduce2, *d_reduce4 = cache.d_reduce4;

    CUDA_CHECK(cudaMemcpy(d_b1, reinterpret_cast<const double2*>(b1), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, reinterpret_cast<const double2*>(b2), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x1, reinterpret_cast<const double2*>(x1), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2, reinterpret_cast<const double2*>(x2), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));

    double bnorm1 = 0.0, bnorm2 = 0.0;
    gpu_norm_pair(d_b1, d_b2, n, d_np1, d_np2, d_reduce2, bnorm1, bnorm2);
    if (bnorm1 < 1e-30) bnorm1 = 1.0;
    if (bnorm2 < 1e-30) bnorm2 = 1.0;

    double xnorm1 = 0.0, xnorm2 = 0.0;
    gpu_norm_pair(d_x1, d_x2, n, d_np1, d_np2, d_reduce2, xnorm1, xnorm2);
    bool warm1 = (xnorm1 > 1e-30);
    bool warm2 = (xnorm2 > 1e-30);
    int matvecs = 0;
    double rnorm1 = 0.0, rnorm2 = 0.0;
    bool have_rnorm = false;
    if (warm1 || warm2) {
        op.matvec_batch2_device(d_x1, d_x2, d_v1, d_v2);
        matvecs++;
        gpu_b_minus_ax_norm_pair(d_b1, d_b2, d_v1, d_v2, d_r1, d_r2, n,
                                 d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
        have_rnorm = true;
    } else {
        gmres_copy_kernel<<<grid, block>>>(d_r1, d_b1, n);
        gmres_copy_kernel<<<grid, block>>>(d_r2, d_b2, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy(d_rh1, d_r1, vec_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_rh2, d_r2, vec_bytes, cudaMemcpyDeviceToDevice));
    bicgstab_init_p_pair_kernel<<<grid, block>>>(d_p1, d_p2, d_v1, d_v2, d_r1, d_r2, n);
    CUDA_CHECK(cudaGetLastError());

    if (!have_rnorm)
        gpu_norm_pair(d_r1, d_r2, n, d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
    double rel1 = rnorm1 / bnorm1;
    double rel2 = rnorm2 / bnorm2;
    double warm_max_rel = bem_env_double("BEM_GMRES_WARM_MAX_REL", 1.05);
    if ((warm1 || warm2) && warm_max_rel > 0.0 &&
        std::max(rel1, rel2) > warm_max_rel) {
        CUDA_CHECK(cudaMemset(d_x1, 0, vec_bytes));
        CUDA_CHECK(cudaMemset(d_x2, 0, vec_bytes));
        CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));
        gmres_copy_kernel<<<grid, block>>>(d_r1, d_b1, n);
        gmres_copy_kernel<<<grid, block>>>(d_r2, d_b2, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(d_rh1, d_r1, vec_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_rh2, d_r2, vec_bytes, cudaMemcpyDeviceToDevice));
        bicgstab_init_p_pair_kernel<<<grid, block>>>(d_p1, d_p2, d_v1, d_v2, d_r1, d_r2, n);
        CUDA_CHECK(cudaGetLastError());
        rnorm1 = bnorm1;
        rnorm2 = bnorm2;
        rel1 = 1.0;
        rel2 = 1.0;
        if (verbose) {
            printf("  [BiCGSTAB-paired-GPU] dropped warm start: initial max rel exceeded %.2e\n",
                   warm_max_rel);
            fflush(stdout);
        }
    }
    double best_rel1 = rel1;
    double best_rel2 = rel2;
    bool conv1 = rel1 < tol;
    bool conv2 = rel2 < tol;
    cdouble rho_old1(1.0, 0.0), rho_old2(1.0, 0.0);
    cdouble alpha1(1.0, 0.0), alpha2(1.0, 0.0);
    cdouble omega1(1.0, 0.0), omega2(1.0, 0.0);

    if (verbose) {
        printf("  [BiCGSTAB-paired-GPU] start: res1=%.2e res2=%.2e%s\n",
               rel1, rel2, (warm1 || warm2) ? " (warm)" : "");
        fflush(stdout);
    }

    bool numerical_breakdown = false;
    int iter = 0;
    for (; iter < max_steps && !(conv1 && conv2); iter++) {
        cdouble rho1, rho2;
        gpu_dot_pair(d_rh1, d_r1, d_rh2, d_r2, n,
                     d_pr1, d_pi1, d_pr2, d_pi2, d_reduce4,
                     rho1, rho2);
        if ((!conv1 && !usable_divisor(rho1)) ||
            (!conv2 && !usable_divisor(rho2)) ||
            (!conv1 && !usable_divisor(omega1)) ||
            (!conv2 && !usable_divisor(omega2))) {
            numerical_breakdown = true;
            break;
        }

        if (iter > 0) {
            cdouble beta1 = conv1 ? cdouble(0.0) : (rho1 / rho_old1) * (alpha1 / omega1);
            cdouble beta2 = conv2 ? cdouble(0.0) : (rho2 / rho_old2) * (alpha2 / omega2);
            bicgstab_update_p_pair_kernel<<<grid, block>>>(
                d_p1, d_p2, d_r1, d_r2, d_v1, d_v2,
                to_double2(beta1), to_double2(beta2),
                to_double2(omega1), to_double2(omega2), n);
            CUDA_CHECK(cudaGetLastError());
        }

        op.matvec_batch2_device(d_p1, d_p2, d_v1, d_v2);
        matvecs++;
        cdouble rhat_v1, rhat_v2;
        gpu_dot_pair(d_rh1, d_v1, d_rh2, d_v2, n,
                     d_pr1, d_pi1, d_pr2, d_pi2, d_reduce4,
                     rhat_v1, rhat_v2);
        if ((!conv1 && !usable_divisor(rhat_v1)) ||
            (!conv2 && !usable_divisor(rhat_v2))) {
            numerical_breakdown = true;
            break;
        }
        if (!conv1) alpha1 = rho1 / rhat_v1;
        if (!conv2) alpha2 = rho2 / rhat_v2;
        bicgstab_s_pair_kernel<<<grid, block>>>(
            d_r1, d_r2, d_v1, d_v2, to_double2(alpha1), to_double2(alpha2), d_s1, d_s2, n);
        CUDA_CHECK(cudaGetLastError());

        double snorm1 = 0.0, snorm2 = 0.0;
        gpu_norm_pair(d_s1, d_s2, n, d_np1, d_np2, d_reduce2, snorm1, snorm2);
        bool sconv1 = conv1 || (snorm1 / bnorm1 < tol);
        bool sconv2 = conv2 || (snorm2 / bnorm2 < tol);
        if (sconv1 && sconv2) {
            bicgstab_update_x_s_pair_kernel<<<grid, block>>>(
                d_x1, d_x2, d_p1, d_p2, to_double2(alpha1), to_double2(alpha2), n);
            CUDA_CHECK(cudaGetLastError());
            op.matvec_batch2_device(d_x1, d_x2, d_v1, d_v2);
            matvecs++;
            gpu_b_minus_ax_norm_pair(d_b1, d_b2, d_v1, d_v2, d_r1, d_r2, n,
                                     d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
            rel1 = rnorm1 / bnorm1;
            rel2 = rnorm2 / bnorm2;
            conv1 = rel1 < tol;
            conv2 = rel2 < tol;
            if (rel1 <= best_rel1) {
                best_rel1 = rel1;
                CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
            }
            if (rel2 <= best_rel2) {
                best_rel2 = rel2;
                CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));
            }
            break;
        }

        op.matvec_batch2_device(d_s1, d_s2, d_t1, d_t2);
        matvecs++;
        cdouble ts1, ts2, tt1, tt2;
        gpu_bicgstab_ts_tt_pair(d_t1, d_s1, d_t2, d_s2, n,
                                d_pr1, d_pi1, d_pr2, d_pi2, d_np1, d_np2,
                                d_reduce4, d_reduce2, ts1, ts2, tt1, tt2);
        if ((!conv1 && !usable_divisor(tt1)) ||
            (!conv2 && !usable_divisor(tt2))) {
            numerical_breakdown = true;
            break;
        }
        if (!conv1) omega1 = ts1 / tt1;
        if (!conv2) omega2 = ts2 / tt2;

        bicgstab_update_xr_pair_kernel<<<grid, block>>>(
            d_x1, d_x2, d_r1, d_r2,
            d_p1, d_p2, d_s1, d_s2, d_t1, d_t2,
            to_double2(alpha1), to_double2(alpha2),
            to_double2(omega1), to_double2(omega2), n);
        CUDA_CHECK(cudaGetLastError());
        gpu_norm_pair(d_r1, d_r2, n, d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
        rel1 = rnorm1 / bnorm1;
        rel2 = rnorm2 / bnorm2;

        bool check_true = ((iter + 1) % true_every == 0) || rel1 < tol || rel2 < tol;
        if (check_true) {
            op.matvec_batch2_device(d_x1, d_x2, d_t1, d_t2);
            matvecs++;
            gpu_b_minus_ax_norm_pair(d_b1, d_b2, d_t1, d_t2, d_r1, d_r2, n,
                                     d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
            rel1 = rnorm1 / bnorm1;
            rel2 = rnorm2 / bnorm2;
        }

        conv1 = rel1 < tol;
        conv2 = rel2 < tol;
        if (rel1 <= best_rel1) {
            best_rel1 = rel1;
            CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
        }
        if (rel2 <= best_rel2) {
            best_rel2 = rel2;
            CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));
        }
        rho_old1 = rho1;
        rho_old2 = rho2;

        if (verbose && (iter < 3 || (iter + 1) % 10 == 0)) {
            printf("    BiCGSTAB-GPU iter %d: rel1=%.2e rel2=%.2e matvecs=%d%s%s\n",
                   iter + 1, conv1 ? 0.0 : rel1, conv2 ? 0.0 : rel2, matvecs,
                   conv1 ? " [1:done]" : "", conv2 ? " [2:done]" : "");
            fflush(stdout);
        }
    }

    op.matvec_batch2_device(d_x1, d_x2, d_t1, d_t2);
    matvecs++;
    gpu_b_minus_ax_norm_pair(d_b1, d_b2, d_t1, d_t2, d_r1, d_r2, n,
                             d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
    rel1 = rnorm1 / bnorm1;
    rel2 = rnorm2 / bnorm2;
    conv1 = rel1 < tol;
    conv2 = rel2 < tol;

    bool restore1 = !conv1 && best_rel1 < rel1;
    bool restore2 = !conv2 && best_rel2 < rel2;
    if (restore1 || restore2) {
        if (!restore1)
            CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
        if (!restore2)
            CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x1), d_best_x1, vec_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x2), d_best_x2, vec_bytes, cudaMemcpyDeviceToHost));
        ws.restored_best_iterate = true;
        if (restore1)
            rel1 = best_rel1;
        if (restore2)
            rel2 = best_rel2;
        conv1 = rel1 < tol;
        conv2 = rel2 < tol;
    } else {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x1), d_x1, vec_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x2), d_x2, vec_bytes, cudaMemcpyDeviceToHost));
    }

    ws.final_relres1 = rel1;
    ws.final_relres2 = rel2;
    ws.converged1 = conv1;
    ws.converged2 = conv2;
    ws.numerical_breakdown = numerical_breakdown;
    ws.reached_max_cycles = !(conv1 && conv2) && !numerical_breakdown && iter >= max_steps;

    if (verbose) {
        printf("  [BiCGSTAB-paired-GPU] %s, %d iterations, %d matvec evaluations, res1=%.2e res2=%.2e\n",
               (conv1 && conv2) ? "Both converged" : "NOT fully converged",
               iter, matvecs, rel1, rel2);
        fflush(stdout);
    }
    return matvecs;
}

int bicgstab_rr_solve_paired_device_ws(BemFmmOperator& op,
                                       const cdouble* b1, const cdouble* b2,
                                       cdouble* x1, cdouble* x2,
                                       int restart, double tol, int maxiter,
                                       bool verbose, GmresPairedWorkspace& ws)
{
    const int n = op.system_size;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    const int red_grid = grid;
    const size_t vec_bytes = (size_t)n * sizeof(double2);
    const int max_steps = std::max(1, restart) * std::max(1, maxiter);
    int true_every = bem_env_int("BEM_BICGSTAB_RR_EVERY", 4);
    true_every = std::max(1, true_every);

    ws.final_relres1 = 0.0;
    ws.final_relres2 = 0.0;
    ws.converged1 = false;
    ws.converged2 = false;
    ws.stopped_stagnant = false;
    ws.numerical_breakdown = false;
    ws.restored_best_iterate = false;
    ws.reached_max_cycles = false;

    BicgstabDeviceBufferCache& cache = bicgstab_device_buffer_cache();
    cache.ensure(n, red_grid);
    double2 *d_b1 = cache.d_b1, *d_b2 = cache.d_b2;
    double2 *d_x1 = cache.d_x1, *d_x2 = cache.d_x2;
    double2 *d_best_x1 = cache.d_best_x1, *d_best_x2 = cache.d_best_x2;
    double2 *d_r1 = cache.d_r1, *d_r2 = cache.d_r2;
    double2 *d_rh1 = cache.d_rh1, *d_rh2 = cache.d_rh2;
    double2 *d_p1 = cache.d_p1, *d_p2 = cache.d_p2;
    double2 *d_v1 = cache.d_v1, *d_v2 = cache.d_v2;
    double2 *d_s1 = cache.d_s1, *d_s2 = cache.d_s2;
    double2 *d_t1 = cache.d_t1, *d_t2 = cache.d_t2;
    double *d_np1 = cache.d_np1, *d_np2 = cache.d_np2;
    double *d_pr1 = cache.d_pr1, *d_pi1 = cache.d_pi1;
    double *d_pr2 = cache.d_pr2, *d_pi2 = cache.d_pi2;
    double *d_reduce2 = cache.d_reduce2, *d_reduce4 = cache.d_reduce4;

    CUDA_CHECK(cudaMemcpy(d_b1, reinterpret_cast<const double2*>(b1), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, reinterpret_cast<const double2*>(b2), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x1, reinterpret_cast<const double2*>(x1), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2, reinterpret_cast<const double2*>(x2), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));

    double bnorm1 = 0.0, bnorm2 = 0.0;
    gpu_norm_pair(d_b1, d_b2, n, d_np1, d_np2, d_reduce2, bnorm1, bnorm2);
    if (bnorm1 < 1e-30) bnorm1 = 1.0;
    if (bnorm2 < 1e-30) bnorm2 = 1.0;

    double xnorm1 = 0.0, xnorm2 = 0.0;
    gpu_norm_pair(d_x1, d_x2, n, d_np1, d_np2, d_reduce2, xnorm1, xnorm2);
    bool warm1 = (xnorm1 > 1e-30);
    bool warm2 = (xnorm2 > 1e-30);
    int matvecs = 0;
    double rnorm1 = 0.0, rnorm2 = 0.0;
    bool have_rnorm = false;
    if (warm1 || warm2) {
        op.matvec_batch2_device(d_x1, d_x2, d_v1, d_v2);
        matvecs++;
        gpu_b_minus_ax_norm_pair(d_b1, d_b2, d_v1, d_v2, d_r1, d_r2, n,
                                 d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
        have_rnorm = true;
    } else {
        gmres_copy_kernel<<<grid, block>>>(d_r1, d_b1, n);
        gmres_copy_kernel<<<grid, block>>>(d_r2, d_b2, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy(d_rh1, d_r1, vec_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_rh2, d_r2, vec_bytes, cudaMemcpyDeviceToDevice));
    bicgstab_init_p_pair_kernel<<<grid, block>>>(d_p1, d_p2, d_v1, d_v2, d_r1, d_r2, n);
    CUDA_CHECK(cudaGetLastError());

    if (!have_rnorm)
        gpu_norm_pair(d_r1, d_r2, n, d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
    double rel1 = rnorm1 / bnorm1;
    double rel2 = rnorm2 / bnorm2;
    double warm_max_rel = bem_env_double("BEM_GMRES_WARM_MAX_REL", 1.05);
    if ((warm1 || warm2) && warm_max_rel > 0.0 &&
        std::max(rel1, rel2) > warm_max_rel) {
        CUDA_CHECK(cudaMemset(d_x1, 0, vec_bytes));
        CUDA_CHECK(cudaMemset(d_x2, 0, vec_bytes));
        CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));
        gmres_copy_kernel<<<grid, block>>>(d_r1, d_b1, n);
        gmres_copy_kernel<<<grid, block>>>(d_r2, d_b2, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(d_rh1, d_r1, vec_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_rh2, d_r2, vec_bytes, cudaMemcpyDeviceToDevice));
        bicgstab_init_p_pair_kernel<<<grid, block>>>(d_p1, d_p2, d_v1, d_v2, d_r1, d_r2, n);
        CUDA_CHECK(cudaGetLastError());
        rnorm1 = bnorm1;
        rnorm2 = bnorm2;
        rel1 = 1.0;
        rel2 = 1.0;
        if (verbose) {
            printf("  [BiCGSTAB-RR-paired-GPU] dropped warm start: initial max rel exceeded %.2e\n",
                   warm_max_rel);
            fflush(stdout);
        }
    }
    double best_rel1 = rel1;
    double best_rel2 = rel2;
    bool conv1 = rel1 < tol;
    bool conv2 = rel2 < tol;
    cdouble rho_old1(1.0, 0.0), rho_old2(1.0, 0.0);
    cdouble alpha1(1.0, 0.0), alpha2(1.0, 0.0);
    cdouble omega1(1.0, 0.0), omega2(1.0, 0.0);

    if (verbose) {
        printf("  [BiCGSTAB-RR-paired-GPU] start: res1=%.2e res2=%.2e%s, rr_every=%d\n",
               rel1, rel2, (warm1 || warm2) ? " (warm)" : "", true_every);
        fflush(stdout);
    }

    bool numerical_breakdown = false;
    int iter = 0;
    int last_rr_iter = 0;
    const bool auto_krylov_requested =
        std::getenv("BEM_KRYLOV") && strcmp(std::getenv("BEM_KRYLOV"), "auto") == 0;
    int stagnation_windows = bem_env_int(
        "BEM_BICGSTAB_RR_STAGNATION_WINDOWS",
        auto_krylov_requested ? 2 : 0);
    stagnation_windows = std::max(0, stagnation_windows);
    const int stagnation_min_iter = std::max(
        1, bem_env_int("BEM_BICGSTAB_RR_STAGNATION_MIN_ITER",
                       auto_krylov_requested ? 12 : 0));
    const double stagnation_min_gain = std::min(
        0.999999, std::max(0.0, bem_env_double("BEM_BICGSTAB_RR_STAGNATION_MIN_GAIN", 0.995)));
    double best_true_max_rel = std::max(rel1, rel2);
    int stagnant_true_checks = 0;
    for (; iter < max_steps && !(conv1 && conv2); iter++) {
        cdouble rho1, rho2;
        gpu_dot_pair(d_rh1, d_r1, d_rh2, d_r2, n,
                     d_pr1, d_pi1, d_pr2, d_pi2, d_reduce4,
                     rho1, rho2);
        if ((!conv1 && !usable_divisor(rho1)) ||
            (!conv2 && !usable_divisor(rho2)) ||
            (!conv1 && !usable_divisor(omega1)) ||
            (!conv2 && !usable_divisor(omega2))) {
            numerical_breakdown = true;
            break;
        }

        if (iter > last_rr_iter) {
            cdouble beta1 = conv1 ? cdouble(0.0) : (rho1 / rho_old1) * (alpha1 / omega1);
            cdouble beta2 = conv2 ? cdouble(0.0) : (rho2 / rho_old2) * (alpha2 / omega2);
            bicgstab_update_p_pair_kernel<<<grid, block>>>(
                d_p1, d_p2, d_r1, d_r2, d_v1, d_v2,
                to_double2(beta1), to_double2(beta2),
                to_double2(omega1), to_double2(omega2), n);
            CUDA_CHECK(cudaGetLastError());
        }

        op.matvec_batch2_device(d_p1, d_p2, d_v1, d_v2);
        matvecs++;
        cdouble rhat_v1, rhat_v2;
        gpu_dot_pair(d_rh1, d_v1, d_rh2, d_v2, n,
                     d_pr1, d_pi1, d_pr2, d_pi2, d_reduce4,
                     rhat_v1, rhat_v2);
        if ((!conv1 && !usable_divisor(rhat_v1)) ||
            (!conv2 && !usable_divisor(rhat_v2))) {
            numerical_breakdown = true;
            break;
        }
        if (!conv1) alpha1 = rho1 / rhat_v1;
        if (!conv2) alpha2 = rho2 / rhat_v2;
        bicgstab_s_pair_kernel<<<grid, block>>>(
            d_r1, d_r2, d_v1, d_v2, to_double2(alpha1), to_double2(alpha2), d_s1, d_s2, n);
        CUDA_CHECK(cudaGetLastError());

        double snorm1 = 0.0, snorm2 = 0.0;
        gpu_norm_pair(d_s1, d_s2, n, d_np1, d_np2, d_reduce2, snorm1, snorm2);
        bool sconv1 = conv1 || (snorm1 / bnorm1 < tol);
        bool sconv2 = conv2 || (snorm2 / bnorm2 < tol);
        if (sconv1 && sconv2) {
            bicgstab_update_x_s_pair_kernel<<<grid, block>>>(
                d_x1, d_x2, d_p1, d_p2, to_double2(alpha1), to_double2(alpha2), n);
            CUDA_CHECK(cudaGetLastError());
            op.matvec_batch2_device(d_x1, d_x2, d_v1, d_v2);
            matvecs++;
            gpu_b_minus_ax_norm_pair(d_b1, d_b2, d_v1, d_v2, d_r1, d_r2, n,
                                     d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
            rel1 = rnorm1 / bnorm1;
            rel2 = rnorm2 / bnorm2;
            conv1 = rel1 < tol;
            conv2 = rel2 < tol;
            if (rel1 <= best_rel1) {
                best_rel1 = rel1;
                CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
            }
            if (rel2 <= best_rel2) {
                best_rel2 = rel2;
                CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));
            }
            break;
        }

        op.matvec_batch2_device(d_s1, d_s2, d_t1, d_t2);
        matvecs++;
        cdouble ts1, ts2, tt1, tt2;
        gpu_bicgstab_ts_tt_pair(d_t1, d_s1, d_t2, d_s2, n,
                                d_pr1, d_pi1, d_pr2, d_pi2, d_np1, d_np2,
                                d_reduce4, d_reduce2, ts1, ts2, tt1, tt2);
        if ((!conv1 && !usable_divisor(tt1)) ||
            (!conv2 && !usable_divisor(tt2))) {
            numerical_breakdown = true;
            break;
        }
        if (!conv1) omega1 = ts1 / tt1;
        if (!conv2) omega2 = ts2 / tt2;

        bicgstab_update_xr_pair_kernel<<<grid, block>>>(
            d_x1, d_x2, d_r1, d_r2,
            d_p1, d_p2, d_s1, d_s2, d_t1, d_t2,
            to_double2(alpha1), to_double2(alpha2),
            to_double2(omega1), to_double2(omega2), n);
        CUDA_CHECK(cudaGetLastError());
        gpu_norm_pair(d_r1, d_r2, n, d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
        rel1 = rnorm1 / bnorm1;
        rel2 = rnorm2 / bnorm2;

        bool check_true = ((iter + 1) % true_every == 0) || rel1 < tol || rel2 < tol;
        bool did_rr = false;
        if (check_true) {
            op.matvec_batch2_device(d_x1, d_x2, d_t1, d_t2);
            matvecs++;
            gpu_b_minus_ax_norm_pair(d_b1, d_b2, d_t1, d_t2, d_r1, d_r2, n,
                                     d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
            rel1 = rnorm1 / bnorm1;
            rel2 = rnorm2 / bnorm2;
            conv1 = rel1 < tol;
            conv2 = rel2 < tol;
            const double true_max_rel = std::max(rel1, rel2);
            if (true_max_rel < best_true_max_rel * stagnation_min_gain) {
                best_true_max_rel = true_max_rel;
                stagnant_true_checks = 0;
            } else if (!(conv1 && conv2) && stagnation_windows > 0 &&
                       iter + 1 >= stagnation_min_iter) {
                stagnant_true_checks++;
                if (stagnant_true_checks >= stagnation_windows) {
                    ws.stopped_stagnant = true;
                    if (verbose) {
                        printf("    BiCGSTAB-RR-GPU stopped early: true residual stagnated at %.2e after %d checked windows\n",
                               true_max_rel, stagnant_true_checks);
                        fflush(stdout);
                    }
                    break;
                }
            }
            if (!(conv1 && conv2)) {
                CUDA_CHECK(cudaMemcpy(d_rh1, d_r1, vec_bytes, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_rh2, d_r2, vec_bytes, cudaMemcpyDeviceToDevice));
                bicgstab_init_p_pair_kernel<<<grid, block>>>(d_p1, d_p2, d_v1, d_v2, d_r1, d_r2, n);
                CUDA_CHECK(cudaGetLastError());
                rho_old1 = cdouble(1.0, 0.0);
                rho_old2 = cdouble(1.0, 0.0);
                alpha1 = cdouble(1.0, 0.0);
                alpha2 = cdouble(1.0, 0.0);
                omega1 = cdouble(1.0, 0.0);
                omega2 = cdouble(1.0, 0.0);
                last_rr_iter = iter + 1;
                did_rr = true;
            }
        } else {
            conv1 = rel1 < tol;
            conv2 = rel2 < tol;
        }

        if (rel1 <= best_rel1) {
            best_rel1 = rel1;
            CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
        }
        if (rel2 <= best_rel2) {
            best_rel2 = rel2;
            CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));
        }
        if (!did_rr) {
            rho_old1 = rho1;
            rho_old2 = rho2;
        }

        if (verbose && (iter < 3 || (iter + 1) % 10 == 0 || did_rr)) {
            printf("    BiCGSTAB-RR-GPU iter %d: rel1=%.2e rel2=%.2e matvecs=%d%s%s%s\n",
                   iter + 1, conv1 ? 0.0 : rel1, conv2 ? 0.0 : rel2, matvecs,
                   conv1 ? " [1:done]" : "", conv2 ? " [2:done]" : "",
                   did_rr ? " [rr]" : "");
            fflush(stdout);
        }
    }

    op.matvec_batch2_device(d_x1, d_x2, d_t1, d_t2);
    matvecs++;
    gpu_b_minus_ax_norm_pair(d_b1, d_b2, d_t1, d_t2, d_r1, d_r2, n,
                             d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
    rel1 = rnorm1 / bnorm1;
    rel2 = rnorm2 / bnorm2;
    conv1 = rel1 < tol;
    conv2 = rel2 < tol;

    bool restore1 = !conv1 && best_rel1 < rel1;
    bool restore2 = !conv2 && best_rel2 < rel2;
    if (restore1 || restore2) {
        if (!restore1)
            CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
        if (!restore2)
            CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x1), d_best_x1, vec_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x2), d_best_x2, vec_bytes, cudaMemcpyDeviceToHost));
        ws.restored_best_iterate = true;
        if (restore1)
            rel1 = best_rel1;
        if (restore2)
            rel2 = best_rel2;
        conv1 = rel1 < tol;
        conv2 = rel2 < tol;
    } else {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x1), d_x1, vec_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x2), d_x2, vec_bytes, cudaMemcpyDeviceToHost));
    }

    ws.final_relres1 = rel1;
    ws.final_relres2 = rel2;
    ws.converged1 = conv1;
    ws.converged2 = conv2;
    ws.numerical_breakdown = numerical_breakdown;
    ws.reached_max_cycles = !(conv1 && conv2) && !numerical_breakdown && iter >= max_steps;

    if (verbose) {
        printf("  [BiCGSTAB-RR-paired-GPU] %s, %d iterations, %d matvec evaluations, res1=%.2e res2=%.2e\n",
               (conv1 && conv2) ? "Both converged" : "NOT fully converged",
               iter, matvecs, rel1, rel2);
        fflush(stdout);
    }
    return matvecs;
}

int cgs_rr_solve_paired_device_ws(BemFmmOperator& op,
                                  const cdouble* b1, const cdouble* b2,
                                  cdouble* x1, cdouble* x2,
                                  int restart, double tol, int maxiter,
                                  bool verbose, GmresPairedWorkspace& ws)
{
    const int n = op.system_size;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    const int red_grid = grid;
    const size_t vec_bytes = (size_t)n * sizeof(double2);
    const int max_steps = std::max(1, restart) * std::max(1, maxiter);
    int true_every = bem_env_int("BEM_CGS_RR_EVERY", 4);
    true_every = std::max(1, true_every);

    ws.final_relres1 = 0.0;
    ws.final_relres2 = 0.0;
    ws.converged1 = false;
    ws.converged2 = false;
    ws.stopped_stagnant = false;
    ws.numerical_breakdown = false;
    ws.restored_best_iterate = false;
    ws.reached_max_cycles = false;

    BicgstabDeviceBufferCache& cache = bicgstab_device_buffer_cache();
    cache.ensure(n, red_grid);
    double2 *d_b1 = cache.d_b1, *d_b2 = cache.d_b2;
    double2 *d_x1 = cache.d_x1, *d_x2 = cache.d_x2;
    double2 *d_best_x1 = cache.d_best_x1, *d_best_x2 = cache.d_best_x2;
    double2 *d_r1 = cache.d_r1, *d_r2 = cache.d_r2;
    double2 *d_rh1 = cache.d_rh1, *d_rh2 = cache.d_rh2;
    double2 *d_p1 = cache.d_p1, *d_p2 = cache.d_p2;
    double2 *d_v1 = cache.d_v1, *d_v2 = cache.d_v2;
    double2 *d_u1 = cache.d_s1, *d_u2 = cache.d_s2;
    double2 *d_q1 = cache.d_t1, *d_q2 = cache.d_t2;
    double *d_np1 = cache.d_np1, *d_np2 = cache.d_np2;
    double *d_pr1 = cache.d_pr1, *d_pi1 = cache.d_pi1;
    double *d_pr2 = cache.d_pr2, *d_pi2 = cache.d_pi2;
    double *d_reduce2 = cache.d_reduce2, *d_reduce4 = cache.d_reduce4;

    CUDA_CHECK(cudaMemcpy(d_b1, reinterpret_cast<const double2*>(b1), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, reinterpret_cast<const double2*>(b2), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x1, reinterpret_cast<const double2*>(x1), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2, reinterpret_cast<const double2*>(x2), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));

    double bnorm1 = 0.0, bnorm2 = 0.0;
    gpu_norm_pair(d_b1, d_b2, n, d_np1, d_np2, d_reduce2, bnorm1, bnorm2);
    if (bnorm1 < 1e-30) bnorm1 = 1.0;
    if (bnorm2 < 1e-30) bnorm2 = 1.0;

    double xnorm1 = 0.0, xnorm2 = 0.0;
    gpu_norm_pair(d_x1, d_x2, n, d_np1, d_np2, d_reduce2, xnorm1, xnorm2);
    bool warm1 = (xnorm1 > 1e-30);
    bool warm2 = (xnorm2 > 1e-30);
    int matvecs = 0;
    double rnorm1 = 0.0, rnorm2 = 0.0;
    if (warm1 || warm2) {
        op.matvec_batch2_device(d_x1, d_x2, d_v1, d_v2);
        matvecs++;
        gpu_b_minus_ax_norm_pair(d_b1, d_b2, d_v1, d_v2, d_r1, d_r2, n,
                                 d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
    } else {
        gmres_copy_kernel<<<grid, block>>>(d_r1, d_b1, n);
        gmres_copy_kernel<<<grid, block>>>(d_r2, d_b2, n);
        CUDA_CHECK(cudaGetLastError());
        gpu_norm_pair(d_r1, d_r2, n, d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
    }

    double rel1 = rnorm1 / bnorm1;
    double rel2 = rnorm2 / bnorm2;
    double warm_max_rel = bem_env_double("BEM_GMRES_WARM_MAX_REL", 1.05);
    if ((warm1 || warm2) && warm_max_rel > 0.0 &&
        std::max(rel1, rel2) > warm_max_rel) {
        CUDA_CHECK(cudaMemset(d_x1, 0, vec_bytes));
        CUDA_CHECK(cudaMemset(d_x2, 0, vec_bytes));
        CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));
        gmres_copy_kernel<<<grid, block>>>(d_r1, d_b1, n);
        gmres_copy_kernel<<<grid, block>>>(d_r2, d_b2, n);
        CUDA_CHECK(cudaGetLastError());
        rnorm1 = bnorm1;
        rnorm2 = bnorm2;
        rel1 = 1.0;
        rel2 = 1.0;
        if (verbose) {
            printf("  [CGS-RR-paired-GPU] dropped warm start: initial max rel exceeded %.2e\n",
                   warm_max_rel);
            fflush(stdout);
        }
    }

    CUDA_CHECK(cudaMemcpy(d_rh1, d_r1, vec_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_rh2, d_r2, vec_bytes, cudaMemcpyDeviceToDevice));
    gmres_copy_kernel<<<grid, block>>>(d_u1, d_r1, n);
    gmres_copy_kernel<<<grid, block>>>(d_u2, d_r2, n);
    gmres_copy_kernel<<<grid, block>>>(d_p1, d_r1, n);
    gmres_copy_kernel<<<grid, block>>>(d_p2, d_r2, n);
    CUDA_CHECK(cudaMemset(d_q1, 0, vec_bytes));
    CUDA_CHECK(cudaMemset(d_q2, 0, vec_bytes));
    CUDA_CHECK(cudaGetLastError());

    double best_rel1 = rel1;
    double best_rel2 = rel2;
    bool conv1 = rel1 < tol;
    bool conv2 = rel2 < tol;
    cdouble rho_old1(1.0, 0.0), rho_old2(1.0, 0.0);

    if (verbose) {
        printf("  [CGS-RR-paired-GPU] start: res1=%.2e res2=%.2e%s, rr_every=%d\n",
               rel1, rel2, (warm1 || warm2) ? " (warm)" : "", true_every);
        fflush(stdout);
    }

    bool numerical_breakdown = false;
    int iter = 0;
    for (; iter < max_steps && !(conv1 && conv2); iter++) {
        cdouble rho1, rho2;
        gpu_dot_pair(d_rh1, d_r1, d_rh2, d_r2, n,
                     d_pr1, d_pi1, d_pr2, d_pi2, d_reduce4,
                     rho1, rho2);
        if ((!conv1 && !usable_divisor(rho1)) ||
            (!conv2 && !usable_divisor(rho2))) {
            numerical_breakdown = true;
            break;
        }

        if (iter > 0) {
            cdouble beta1 = conv1 ? cdouble(0.0) : rho1 / rho_old1;
            cdouble beta2 = conv2 ? cdouble(0.0) : rho2 / rho_old2;
            cgs_update_u_p_pair_kernel<<<grid, block>>>(
                d_u1, d_u2, d_p1, d_p2, d_r1, d_r2, d_q1, d_q2,
                to_double2(beta1), to_double2(beta2), n);
            CUDA_CHECK(cudaGetLastError());
        }

        op.matvec_batch2_device(d_p1, d_p2, d_v1, d_v2);
        matvecs++;
        cdouble sigma1, sigma2;
        gpu_dot_pair(d_rh1, d_v1, d_rh2, d_v2, n,
                     d_pr1, d_pi1, d_pr2, d_pi2, d_reduce4,
                     sigma1, sigma2);
        if ((!conv1 && !usable_divisor(sigma1)) ||
            (!conv2 && !usable_divisor(sigma2))) {
            numerical_breakdown = true;
            break;
        }

        cdouble alpha1 = conv1 ? cdouble(0.0) : rho1 / sigma1;
        cdouble alpha2 = conv2 ? cdouble(0.0) : rho2 / sigma2;
        cgs_update_q_s_x_pair_kernel<<<grid, block>>>(
            d_q1, d_q2, d_u1, d_u2, d_x1, d_x2,
            d_u1, d_u2, d_v1, d_v2,
            to_double2(alpha1), to_double2(alpha2), n);
        CUDA_CHECK(cudaGetLastError());

        op.matvec_batch2_device(d_u1, d_u2, d_v1, d_v2);
        matvecs++;
        cgs_update_r_pair_kernel<<<grid, block>>>(
            d_r1, d_r2, d_v1, d_v2, to_double2(alpha1), to_double2(alpha2), n);
        CUDA_CHECK(cudaGetLastError());
        gpu_norm_pair(d_r1, d_r2, n, d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
        rel1 = rnorm1 / bnorm1;
        rel2 = rnorm2 / bnorm2;

        bool check_true = ((iter + 1) % true_every == 0) || rel1 < tol || rel2 < tol;
        bool did_rr = false;
        if (check_true) {
            op.matvec_batch2_device(d_x1, d_x2, d_v1, d_v2);
            matvecs++;
            gpu_b_minus_ax_norm_pair(d_b1, d_b2, d_v1, d_v2, d_r1, d_r2, n,
                                     d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
            rel1 = rnorm1 / bnorm1;
            rel2 = rnorm2 / bnorm2;
            conv1 = rel1 < tol;
            conv2 = rel2 < tol;
            if (!(conv1 && conv2)) {
                CUDA_CHECK(cudaMemcpy(d_rh1, d_r1, vec_bytes, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_rh2, d_r2, vec_bytes, cudaMemcpyDeviceToDevice));
                gmres_copy_kernel<<<grid, block>>>(d_u1, d_r1, n);
                gmres_copy_kernel<<<grid, block>>>(d_u2, d_r2, n);
                gmres_copy_kernel<<<grid, block>>>(d_p1, d_r1, n);
                gmres_copy_kernel<<<grid, block>>>(d_p2, d_r2, n);
                CUDA_CHECK(cudaMemset(d_q1, 0, vec_bytes));
                CUDA_CHECK(cudaMemset(d_q2, 0, vec_bytes));
                CUDA_CHECK(cudaGetLastError());
                rho_old1 = cdouble(1.0, 0.0);
                rho_old2 = cdouble(1.0, 0.0);
                did_rr = true;
            }
        } else {
            conv1 = rel1 < tol;
            conv2 = rel2 < tol;
        }

        if (rel1 <= best_rel1) {
            best_rel1 = rel1;
            CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
        }
        if (rel2 <= best_rel2) {
            best_rel2 = rel2;
            CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));
        }
        if (!did_rr) {
            rho_old1 = rho1;
            rho_old2 = rho2;
        }

        if (verbose && (iter < 3 || (iter + 1) % 10 == 0 || did_rr)) {
            printf("    CGS-RR-GPU iter %d: rel1=%.2e rel2=%.2e matvecs=%d%s%s%s\n",
                   iter + 1, conv1 ? 0.0 : rel1, conv2 ? 0.0 : rel2, matvecs,
                   conv1 ? " [1:done]" : "", conv2 ? " [2:done]" : "",
                   did_rr ? " [rr]" : "");
            fflush(stdout);
        }
    }

    op.matvec_batch2_device(d_x1, d_x2, d_v1, d_v2);
    matvecs++;
    gpu_b_minus_ax_norm_pair(d_b1, d_b2, d_v1, d_v2, d_r1, d_r2, n,
                             d_np1, d_np2, d_reduce2, rnorm1, rnorm2);
    rel1 = rnorm1 / bnorm1;
    rel2 = rnorm2 / bnorm2;
    conv1 = rel1 < tol;
    conv2 = rel2 < tol;

    bool restore1 = !conv1 && best_rel1 < rel1;
    bool restore2 = !conv2 && best_rel2 < rel2;
    if (restore1 || restore2) {
        if (!restore1)
            CUDA_CHECK(cudaMemcpy(d_best_x1, d_x1, vec_bytes, cudaMemcpyDeviceToDevice));
        if (!restore2)
            CUDA_CHECK(cudaMemcpy(d_best_x2, d_x2, vec_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x1), d_best_x1, vec_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x2), d_best_x2, vec_bytes, cudaMemcpyDeviceToHost));
        ws.restored_best_iterate = true;
        if (restore1)
            rel1 = best_rel1;
        if (restore2)
            rel2 = best_rel2;
        conv1 = rel1 < tol;
        conv2 = rel2 < tol;
    } else {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x1), d_x1, vec_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(x2), d_x2, vec_bytes, cudaMemcpyDeviceToHost));
    }

    ws.final_relres1 = rel1;
    ws.final_relres2 = rel2;
    ws.converged1 = conv1;
    ws.converged2 = conv2;
    ws.numerical_breakdown = numerical_breakdown;
    ws.reached_max_cycles = !(conv1 && conv2) && !numerical_breakdown && iter >= max_steps;

    if (verbose) {
        printf("  [CGS-RR-paired-GPU] %s, %d iterations, %d matvec evaluations, res1=%.2e res2=%.2e\n",
               (conv1 && conv2) ? "Both converged" : "NOT fully converged",
               iter, matvecs, rel1, rel2);
        fflush(stdout);
    }
    return matvecs;
}

int gmres_solve_paired_ws(BemFmmOperator& op,
                          const cdouble* b1, const cdouble* b2,
                          cdouble* x1, cdouble* x2,
                          int restart, double tol, int maxiter,
                          bool verbose, NearFieldPrecond* precond,
                          GmresPairedWorkspace& ws)
{
    int n = op.system_size;
    ws.final_relres1 = 0.0;
    ws.final_relres2 = 0.0;
    ws.converged1 = false;
    ws.converged2 = false;
    ws.stopped_stagnant = false;
    ws.numerical_breakdown = false;
    ws.restored_best_iterate = false;
    ws.reached_max_cycles = false;

    const char* krylov = std::getenv("BEM_KRYLOV");
    bool auto_krylov_requested = krylov && strcmp(krylov, "auto") == 0;
    bool bicgstab_rr_requested = krylov &&
        (strcmp(krylov, "bicgstab-rr") == 0 || strcmp(krylov, "bicgstab_rr") == 0 ||
         strcmp(krylov, "bcgstab-rr") == 0 || strcmp(krylov, "bcgstab_rr") == 0);
    bool cgs_rr_requested = krylov &&
        (strcmp(krylov, "cgs-rr") == 0 || strcmp(krylov, "cgs_rr") == 0 ||
         strcmp(krylov, "cgs") == 0 || strcmp(krylov, "CGS") == 0);
    bool hybrid_requested = krylov &&
        (strcmp(krylov, "hybrid") == 0 ||
         strcmp(krylov, "gpu-hybrid") == 0 ||
         strcmp(krylov, "gpu_hybrid") == 0 ||
         strcmp(krylov, "gpu-native") == 0 ||
         strcmp(krylov, "gpu_native") == 0);
    const char* auto_solver = std::getenv("BEM_KRYLOV_AUTO_SOLVER");
    bool auto_best_requested = auto_krylov_requested &&
        (!auto_solver ||
         strcmp(auto_solver, "best") == 0 ||
         strcmp(auto_solver, "multi") == 0 ||
         strcmp(auto_solver, "adaptive") == 0 ||
         strcmp(auto_solver, "gpu-native") == 0 ||
         strcmp(auto_solver, "gpu_native") == 0);
    bool auto_cgs_requested = auto_krylov_requested && auto_solver &&
        (strcmp(auto_solver, "cgs") == 0 ||
         strcmp(auto_solver, "cgs-rr") == 0 ||
         strcmp(auto_solver, "cgs_rr") == 0);
    bool bicgstab_requested = krylov &&
        (auto_krylov_requested ||
         bicgstab_rr_requested ||
         strcmp(krylov, "bcgstab") == 0 || strcmp(krylov, "bicgstab") == 0 ||
         strcmp(krylov, "BiCGSTAB") == 0);

    struct AutoKrylovState {
        std::vector<cdouble> initial_x1;
        std::vector<cdouble> initial_x2;
        std::vector<cdouble> best_x1;
        std::vector<cdouble> best_x2;
        double initial_rel = 1.0;
        double best_rel = 1.0;
        int probe_matvecs = 0;
        bool best_is_initial = true;
    };

    auto make_auto_state = [&]() -> AutoKrylovState {
        AutoKrylovState st;
        st.initial_x1.assign(x1, x1 + n);
        st.initial_x2.assign(x2, x2 + n);
        st.best_x1 = st.initial_x1;
        st.best_x2 = st.initial_x2;
        st.initial_rel = gpu_true_pair_rel_from_host(op, b1, b2, x1, x2,
                                                     st.probe_matvecs);
        if (!std::isfinite(st.initial_rel))
            st.initial_rel = 1.0;
        st.best_rel = st.initial_rel;
        return st;
    };

    auto update_auto_best = [&](AutoKrylovState& st) {
        if (!std::isfinite(ws.final_relres1) || !std::isfinite(ws.final_relres2))
            return;
        const double cur_rel = std::max(ws.final_relres1, ws.final_relres2);
        if (cur_rel < st.best_rel) {
            st.best_rel = cur_rel;
            st.best_x1.assign(x1, x1 + n);
            st.best_x2.assign(x2, x2 + n);
            st.best_is_initial = false;
        }
    };

    auto restore_auto_best = [&](const AutoKrylovState& st) {
        const std::vector<cdouble>& rx1 = st.best_is_initial ? st.initial_x1 : st.best_x1;
        const std::vector<cdouble>& rx2 = st.best_is_initial ? st.initial_x2 : st.best_x2;
        std::copy(rx1.begin(), rx1.end(), x1);
        std::copy(rx2.begin(), rx2.end(), x2);
        if (!st.best_is_initial)
            ws.restored_best_iterate = true;
    };

    auto continue_with_device_gmres = [&](int matvecs_done,
                                          const char* label) -> int {
        if (verbose) {
            printf("  [%s] did not reach tol %.2e (res1=%.2e res2=%.2e); continuing with GMRES warm start\n",
                   label, tol, ws.final_relres1, ws.final_relres2);
            fflush(stdout);
        }
        GmresPairedWorkspace gmres_ws;
        int gmres_matvecs = gmres_solve_paired_device_ws(op, b1, b2, x1, x2,
                                                          restart, tol, maxiter,
                                                          verbose, gmres_ws);
        matvecs_done += gmres_matvecs;
        gmres_ws.restored_best_iterate =
            gmres_ws.restored_best_iterate || ws.restored_best_iterate;
        ws = gmres_ws;
        return matvecs_done;
    };

    auto auto_window_is_useful = [&](double prev_rel, int done_steps) -> bool {
        if (!std::isfinite(ws.final_relres1) || !std::isfinite(ws.final_relres2))
            return false;
        if (ws.numerical_breakdown || ws.stopped_stagnant)
            return false;
        double cur_rel = std::max(ws.final_relres1, ws.final_relres2);
        double min_gain = bem_env_double("BEM_KRYLOV_AUTO_MIN_GAIN", 0.75);
        min_gain = std::min(0.999999, std::max(0.0, min_gain));
        double accept_rel = bem_env_double("BEM_KRYLOV_AUTO_ACCEPT_REL", 0.20);
        accept_rel = std::max(0.0, accept_rel);
        (void)done_steps;
        return cur_rel <= accept_rel || cur_rel < prev_rel * min_gain;
    };

    if (precond == nullptr && (hybrid_requested || auto_best_requested)) {
        if (op.device_matvec_available()) {
            int probe_steps = bem_env_int("BEM_KRYLOV_HYBRID_PROBE_STEPS",
                                          bem_env_int("BEM_KRYLOV_AUTO_WINDOW_STEPS", 12));
            probe_steps = std::max(1, probe_steps);
            int max_steps = bem_env_int("BEM_KRYLOV_HYBRID_MAX_STEPS",
                                        bem_env_int("BEM_KRYLOV_AUTO_MAX_STEPS", 48));
            max_steps = std::max(probe_steps, max_steps);
            double min_gain = bem_env_double("BEM_KRYLOV_HYBRID_MIN_GAIN",
                                             bem_env_double("BEM_KRYLOV_AUTO_MIN_GAIN", 0.75));
            min_gain = std::min(0.999999, std::max(0.0, min_gain));
            if (verbose) {
                printf("  [%s] GPU probe: BiCGSTAB-RR vs CGS-RR, probe_steps=%d max_steps=%d\n",
                       auto_best_requested ? "Krylov-auto" : "Krylov-gpu-native",
                       probe_steps, max_steps);
                fflush(stdout);
            }

            AutoKrylovState auto_state = make_auto_state();
            int matvecs = auto_state.probe_matvecs;
            const double initial_rel = auto_state.initial_rel;
            std::vector<cdouble> base_x1 = auto_state.initial_x1;
            std::vector<cdouble> base_x2 = auto_state.initial_x2;

            GmresPairedWorkspace bicg_ws;
            std::copy(base_x1.begin(), base_x1.end(), x1);
            std::copy(base_x2.begin(), base_x2.end(), x2);
            int bicg_mv = bicgstab_rr_solve_paired_device_ws(op, b1, b2, x1, x2,
                                                             1, tol, probe_steps,
                                                             verbose, bicg_ws);
            std::vector<cdouble> bicg_x1(x1, x1 + n);
            std::vector<cdouble> bicg_x2(x2, x2 + n);
            double bicg_rel = (std::isfinite(bicg_ws.final_relres1) &&
                               std::isfinite(bicg_ws.final_relres2))
                ? std::max(bicg_ws.final_relres1, bicg_ws.final_relres2)
                : std::numeric_limits<double>::infinity();

            GmresPairedWorkspace cgs_ws;
            std::copy(base_x1.begin(), base_x1.end(), x1);
            std::copy(base_x2.begin(), base_x2.end(), x2);
            int cgs_mv = cgs_rr_solve_paired_device_ws(op, b1, b2, x1, x2,
                                                       1, tol, probe_steps,
                                                       verbose, cgs_ws);
            std::vector<cdouble> cgs_x1(x1, x1 + n);
            std::vector<cdouble> cgs_x2(x2, x2 + n);
            double cgs_rel = (std::isfinite(cgs_ws.final_relres1) &&
                              std::isfinite(cgs_ws.final_relres2))
                ? std::max(cgs_ws.final_relres1, cgs_ws.final_relres2)
                : std::numeric_limits<double>::infinity();

            matvecs += bicg_mv + cgs_mv;
            const bool choose_cgs = cgs_rel < bicg_rel;
            const double chosen_rel = choose_cgs ? cgs_rel : bicg_rel;
            if (choose_cgs) {
                std::copy(cgs_x1.begin(), cgs_x1.end(), x1);
                std::copy(cgs_x2.begin(), cgs_x2.end(), x2);
                ws = cgs_ws;
            } else {
                std::copy(bicg_x1.begin(), bicg_x1.end(), x1);
                std::copy(bicg_x2.begin(), bicg_x2.end(), x2);
                ws = bicg_ws;
            }
            update_auto_best(auto_state);

            if (verbose) {
                printf("  [%s] probe result: BiCGSTAB-RR %.2e, CGS-RR %.2e; selected %s\n",
                       auto_best_requested ? "Krylov-auto" : "Krylov-gpu-native",
                       bicg_rel, cgs_rel, choose_cgs ? "CGS-RR" : "BiCGSTAB-RR");
                fflush(stdout);
            }
            if (ws.converged1 && ws.converged2)
                return matvecs;

            const bool useful = std::isfinite(chosen_rel) &&
                                chosen_rel < initial_rel * min_gain;
            int remaining = max_steps - probe_steps;
            if (useful && remaining > 0) {
                int extra_mv = choose_cgs
                    ? cgs_rr_solve_paired_device_ws(op, b1, b2, x1, x2,
                                                    1, tol, remaining,
                                                    verbose, ws)
                    : bicgstab_rr_solve_paired_device_ws(op, b1, b2, x1, x2,
                                                         1, tol, remaining,
                                                         verbose, ws);
                matvecs += extra_mv;
                update_auto_best(auto_state);
                if (ws.converged1 && ws.converged2)
                    return matvecs;
            }

            restore_auto_best(auto_state);
            return continue_with_device_gmres(
                matvecs,
                auto_best_requested ? "Krylov-auto" : "Krylov-gpu-native");
        }
        if (verbose) {
            printf("  [%s] requested, but current backend has no device matvec; using GMRES\n",
                   auto_best_requested ? "Krylov-auto" : "Krylov-gpu-native");
            fflush(stdout);
        }
    }

    if (precond == nullptr && (cgs_rr_requested || auto_cgs_requested)) {
        if (op.device_matvec_available()) {
            if (auto_cgs_requested) {
                int auto_max_steps = bem_env_int(
                    "BEM_KRYLOV_AUTO_CGS_STEPS",
                    bem_env_int("BEM_KRYLOV_AUTO_MAX_STEPS",
                                bem_env_int("BEM_KRYLOV_AUTO_BICGSTAB_STEPS", 48)));
                auto_max_steps = std::max(1, auto_max_steps);
                int auto_window = bem_env_int("BEM_KRYLOV_AUTO_WINDOW_STEPS", 12);
                auto_window = std::max(1, std::min(auto_window, auto_max_steps));
                if (verbose) {
                    printf("  [Krylov-auto] CGS-RR GPU windowed prepass: max_steps=%d window=%d\n",
                           auto_max_steps, auto_window);
                    fflush(stdout);
                }
                AutoKrylovState auto_state = make_auto_state();
                int matvecs = auto_state.probe_matvecs;
                int done_steps = 0;
                double prev_rel = auto_state.initial_rel;
                while (done_steps < auto_max_steps) {
                    int steps = std::min(auto_window, auto_max_steps - done_steps);
                    matvecs += cgs_rr_solve_paired_device_ws(op, b1, b2, x1, x2,
                                                             1, tol, steps,
                                                             verbose, ws);
                    update_auto_best(auto_state);
                    if (ws.converged1 && ws.converged2)
                        return matvecs;
                    if (!auto_window_is_useful(prev_rel, done_steps))
                        break;
                    prev_rel = std::max(ws.final_relres1, ws.final_relres2);
                    done_steps += steps;
                }
                restore_auto_best(auto_state);
                return continue_with_device_gmres(matvecs, "Krylov-auto CGS-RR");
            }
            int cgs_restart = restart;
            int cgs_maxiter = maxiter;
            int matvecs = cgs_rr_solve_paired_device_ws(op, b1, b2, x1, x2,
                                                        cgs_restart, tol, cgs_maxiter,
                                                        verbose, ws);
            const bool fallback_enabled =
                bem_env_flag_enabled("BEM_CGS_FALLBACK_GMRES");
            if (fallback_enabled && !(ws.converged1 && ws.converged2)) {
                return continue_with_device_gmres(matvecs, "CGS-RR-paired-GPU");
            }
            return matvecs;
        }
        if (verbose) {
            printf("  [CGS-RR-paired-GPU] BEM_KRYLOV=%s requested, but current backend has no device matvec; using GMRES\n",
                   krylov ? krylov : "");
            fflush(stdout);
        }
    }
    if (precond == nullptr && bicgstab_requested) {
        if (op.device_matvec_available()) {
            const bool auto_use_rr =
                auto_krylov_requested && bem_env_flag_enabled("BEM_KRYLOV_AUTO_RR", true);
            if (auto_krylov_requested) {
                if (verbose) {
                    printf("  [Krylov-auto] using %s GPU windowed prepass (BEM_KRYLOV_AUTO_RR=%s)\n",
                           auto_use_rr ? "BiCGSTAB-RR" : "BiCGSTAB",
                           auto_use_rr ? "1" : "0");
                    fflush(stdout);
                }
                int auto_max_steps = bem_env_int(
                    "BEM_KRYLOV_AUTO_BICGSTAB_STEPS",
                    bem_env_int("BEM_KRYLOV_AUTO_MAX_STEPS", 48));
                auto_max_steps = std::max(1, auto_max_steps);
                int auto_window = bem_env_int("BEM_KRYLOV_AUTO_WINDOW_STEPS", 12);
                auto_window = std::max(1, std::min(auto_window, auto_max_steps));
                AutoKrylovState auto_state = make_auto_state();
                int matvecs = auto_state.probe_matvecs;
                int done_steps = 0;
                double prev_rel = auto_state.initial_rel;
                while (done_steps < auto_max_steps) {
                    int steps = std::min(auto_window, auto_max_steps - done_steps);
                    matvecs += auto_use_rr ?
                        bicgstab_rr_solve_paired_device_ws(op, b1, b2, x1, x2,
                                                           1, tol, steps,
                                                           verbose, ws) :
                        bicgstab_solve_paired_device_ws(op, b1, b2, x1, x2,
                                                        1, tol, steps,
                                                        verbose, ws);
                    update_auto_best(auto_state);
                    if (ws.converged1 && ws.converged2)
                        return matvecs;
                    if (!auto_window_is_useful(prev_rel, done_steps))
                        break;
                    prev_rel = std::max(ws.final_relres1, ws.final_relres2);
                    done_steps += steps;
                }
                restore_auto_best(auto_state);
                return continue_with_device_gmres(matvecs, "Krylov-auto BiCGSTAB");
            }
            int bicgstab_restart = restart;
            int bicgstab_maxiter = maxiter;
            int matvecs = (bicgstab_rr_requested || auto_use_rr) ?
                bicgstab_rr_solve_paired_device_ws(op, b1, b2, x1, x2,
                                                   bicgstab_restart, tol, bicgstab_maxiter,
                                                   verbose, ws) :
                bicgstab_solve_paired_device_ws(op, b1, b2, x1, x2,
                                                bicgstab_restart, tol, bicgstab_maxiter,
                                                verbose, ws);
            const bool fallback_enabled =
                bem_env_flag_enabled("BEM_BICGSTAB_FALLBACK_GMRES");
            if (fallback_enabled && !(ws.converged1 && ws.converged2)) {
                return continue_with_device_gmres(matvecs, "BiCGSTAB-paired-GPU");
            }
            return matvecs;
        }
        if (verbose) {
            printf("  [BiCGSTAB-paired-GPU] BEM_KRYLOV=%s requested, but current backend has no device matvec; using GMRES\n",
                   krylov ? krylov : "");
            fflush(stdout);
        }
    }

    if (bem_env_flag_enabled("BEM_GMRES_DEVICE")) {
        if (op.device_matvec_available()) {
            if (precond == nullptr ||
                (precond->device_apply_available() &&
                 bem_env_flag_enabled("BEM_GMRES_DEVICE_PREC", true))) {
                return gmres_solve_paired_device_ws(op, b1, b2, x1, x2,
                                                    restart, tol, maxiter, verbose, ws,
                                                    precond);
            }
        }
        printf("  [GMRES-paired] BEM_GMRES_DEVICE requested, but current backend/preconditioner cannot stay on device; using CPU-resident GMRES\n");
        fflush(stdout);
    }

    bool has_precond = (precond != nullptr);
    bool reorth = bem_env_flag_enabled("BEM_GMRES_REORTH", true);
    bool pair_arnoldi = bem_env_flag_enabled("BEM_GMRES_PAIR_ARNOLDI", true);
    bool store_z = false;
    if (has_precond) {
        store_z = bem_env_flag_enabled("BEM_GMRES_STORE_Z");
        if (verbose && store_z) {
            double store_z_mb = (2.0 * (double)n * (double)restart * sizeof(cdouble)) /
                                (1024.0 * 1024.0);
            printf("  [GMRES-paired] STORE_Z enabled (%.1f MB)\n", store_z_mb);
            fflush(stdout);
        }
    }
    int stagnation_cycles = 0;
    stagnation_cycles = std::max(0, bem_env_int("BEM_GMRES_STAGNATION_CYCLES", stagnation_cycles));
    double stagnation_rel = 0.01;
    stagnation_rel = std::max(0.0, bem_env_double("BEM_GMRES_STAGNATION_REL", stagnation_rel));
    int inner_stagnation_window = 0;
    inner_stagnation_window = std::max(0, bem_env_int("BEM_GMRES_INNER_STAGNATION_WINDOW", inner_stagnation_window));
    double inner_stagnation_rel = 0.05;
    inner_stagnation_rel = std::max(0.0, bem_env_double("BEM_GMRES_INNER_STAGNATION_REL", inner_stagnation_rel));
    int inner_stagnation_min_iter = 300;
    inner_stagnation_min_iter = std::max(0, bem_env_int("BEM_GMRES_INNER_STAGNATION_MIN_ITER", inner_stagnation_min_iter));

    auto& r1 = ws.r1; auto& r2 = ws.r2;
    auto& w1 = ws.w1; auto& w2 = ws.w2;
    auto& z1 = ws.z1; auto& z2 = ws.z2;
    auto& V1 = ws.V1; auto& V2 = ws.V2;
    auto& Z1 = ws.Z1; auto& Z2 = ws.Z2;
    auto& H1 = ws.H1; auto& H2 = ws.H2;
    auto& cs1 = ws.cs1; auto& sn1 = ws.sn1; auto& s1 = ws.s1;
    auto& cs2 = ws.cs2; auto& sn2 = ws.sn2; auto& s2 = ws.s2;
    auto& ytmp = ws.ytmp; auto& ytmp2 = ws.ytmp2; auto& ztmp = ws.ztmp; auto& ztmp2 = ws.ztmp2;

    r1.assign(b1, b1 + n);
    r2.assign(b2, b2 + n);
    w1.resize(n);
    w2.resize(n);
    if (has_precond) {
        z1.resize(n);
        z2.resize(n);
    } else {
        z1.clear();
        z2.clear();
    }
    V1.resize((size_t)n * (restart + 1));
    V2.resize((size_t)n * (restart + 1));
    if (has_precond && store_z) {
        Z1.resize((size_t)n * restart);
        Z2.resize((size_t)n * restart);
    } else {
        Z1.clear();
        Z2.clear();
    }

    H1.resize((restart + 1) * restart);
    H2.resize((restart + 1) * restart);
    cs1.resize(restart); sn1.resize(restart); s1.resize(restart + 1);
    cs2.resize(restart); sn2.resize(restart); s2.resize(restart + 1);

    double bnorm1, bnorm2;
    norm_pair_p(b1, b2, n, bnorm1, bnorm2);
    if (bnorm1 < 1e-30) bnorm1 = 1.0;
    if (bnorm2 < 1e-30) bnorm2 = 1.0;

    double xnorm1, xnorm2;
    norm_pair_p(x1, x2, n, xnorm1, xnorm2);
    bool warm1 = (xnorm1 > 1e-30);
    bool warm2 = (xnorm2 > 1e-30);
    int total_matvecs = 0;

    if (warm1 && warm2) {
        op.matvec_batch2(x1, x2, r1.data(), r2.data());
        total_matvecs++;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            r1[i] = b1[i] - r1[i];
            r2[i] = b2[i] - r2[i];
        }
    } else if (warm1) {
        op.matvec(x1, r1.data());
        total_matvecs++;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            r1[i] = b1[i] - r1[i];
    } else if (warm2) {
        op.matvec(x2, r2.data());
        total_matvecs++;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            r2[i] = b2[i] - r2[i];
    }

    double rnorm1, rnorm2;
    norm_pair_p(r1.data(), r2.data(), n, rnorm1, rnorm2);

    bool conv1 = (rnorm1 / bnorm1 < tol);
    bool conv2 = (rnorm2 / bnorm2 < tol);
    double last_rel1 = rnorm1 / bnorm1;
    double last_rel2 = rnorm2 / bnorm2;
    double best_rel = std::max(conv1 ? 0.0 : rnorm1 / bnorm1,
                               conv2 ? 0.0 : rnorm2 / bnorm2);
    double best_true_rel1 = last_rel1;
    double best_true_rel2 = last_rel2;
    double best_pair_rel = std::max(last_rel1, last_rel2);
    double best_inner_rel = best_pair_rel;
    int best_inner_iter = total_matvecs;
    std::vector<cdouble> best_x1(x1, x1 + n);
    std::vector<cdouble> best_x2(x2, x2 + n);
    int stagnant_restart_count = 0;
    bool stopped_stagnant = false;
    bool numerical_breakdown = false;

    if (verbose) {
        printf("  [GMRES-paired] start: res1=%.2e res2=%.2e%s\n",
               rnorm1 / bnorm1, rnorm2 / bnorm2,
               (warm1 || warm2) ? " (warm)" : "");
        fflush(stdout);
    }

    for (int cycle = 0; cycle < maxiter && !(conv1 && conv2); cycle++) {
        std::fill(H1.begin(), H1.end(), cdouble(0));
        std::fill(H2.begin(), H2.end(), cdouble(0));
        std::fill(s1.begin(), s1.end(), cdouble(0));
        std::fill(s2.begin(), s2.end(), cdouble(0));

        if (!conv1) {
            double inv = 1.0 / rnorm1;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++)
                V1[i] = r1[i] * inv;
            s1[0] = cdouble(rnorm1);
        }
        if (!conv2) {
            double inv = 1.0 / rnorm2;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++)
                V2[i] = r2[i] * inv;
            s2[0] = cdouble(rnorm2);
        }

        int m1 = 0, m2 = 0;
        for (int j = 0; j < restart && !(conv1 && conv2) && !stopped_stagnant; j++) {
            if (!conv1 && !conv2) {
                const cdouble* in1 = &V1[(size_t)j * n];
                const cdouble* in2 = &V2[(size_t)j * n];
                if (has_precond) {
                    precond->apply_pair(in1, in2, z1.data(), z2.data());
                    if (store_z) {
                        memcpy(&Z1[(size_t)j * n], z1.data(), n * sizeof(cdouble));
                        memcpy(&Z2[(size_t)j * n], z2.data(), n * sizeof(cdouble));
                    }
                    op.matvec_batch2(z1.data(), z2.data(), w1.data(), w2.data());
                } else {
                    op.matvec_batch2(in1, in2, w1.data(), w2.data());
                }
                total_matvecs++;
            } else if (!conv1) {
                const cdouble* in = &V1[(size_t)j * n];
                if (has_precond) {
                    precond->apply(in, z1.data());
                    if (store_z)
                        memcpy(&Z1[(size_t)j * n], z1.data(), n * sizeof(cdouble));
                    op.matvec(z1.data(), w1.data());
                } else {
                    op.matvec(in, w1.data());
                }
                total_matvecs++;
            } else if (!conv2) {
                const cdouble* in = &V2[(size_t)j * n];
                if (has_precond) {
                    precond->apply(in, z2.data());
                    if (store_z)
                        memcpy(&Z2[(size_t)j * n], z2.data(), n * sizeof(cdouble));
                    op.matvec(z2.data(), w2.data());
                } else {
                    op.matvec(in, w2.data());
                }
                total_matvecs++;
            }

            auto arnoldi = [&](std::vector<cdouble>& V, std::vector<cdouble>& H,
                               std::vector<cdouble>& cs, std::vector<cdouble>& sn,
                               std::vector<cdouble>& s, std::vector<cdouble>& w,
                               double bnorm, bool& conv, int& m, double& last_rel) {
                if (conv)
                    return;
                for (int i = 0; i <= j; i++) {
                    cdouble* vi = &V[(size_t)i * n];
                    cdouble hij = orthogonalize_against_p(vi, w.data(), n);
                    H[i * restart + j] = hij;
                }
                if (reorth) {
                    for (int i = 0; i <= j; i++) {
                        cdouble* vi = &V[(size_t)i * n];
                        cdouble hij = orthogonalize_against_p(vi, w.data(), n);
                        H[i * restart + j] += hij;
                    }
                }
                double wn = norm_p(w.data(), n);
                H[(j + 1) * restart + j] = cdouble(wn);
                if (wn > 1e-30) {
                    cdouble* vnext = &V[(size_t)(j + 1) * n];
                    double inv = 1.0 / wn;
                    #pragma omp parallel for schedule(static)
                    for (int k = 0; k < n; k++)
                        vnext[k] = w[k] * inv;
                }
                for (int i = 0; i < j; i++) {
                    cdouble h0 = H[i * restart + j];
                    cdouble h1 = H[(i + 1) * restart + j];
                    H[i * restart + j]       = std::conj(cs[i]) * h0 + std::conj(sn[i]) * h1;
                    H[(i + 1) * restart + j] = -sn[i] * h0 + cs[i] * h1;
                }
                cdouble h0 = H[j * restart + j];
                cdouble h1 = H[(j + 1) * restart + j];
                double den = std::sqrt(std::norm(h0) + std::norm(h1));
                cs[j] = (den > 1e-30) ? h0 / den : cdouble(1);
                sn[j] = (den > 1e-30) ? h1 / den : cdouble(0);
                H[j * restart + j] = std::conj(cs[j]) * h0 + std::conj(sn[j]) * h1;
                H[(j + 1) * restart + j] = cdouble(0);
                cdouble s0 = s[j];
                s[j] = std::conj(cs[j]) * s0;
                s[j + 1] = -sn[j] * s0;
                m = j + 1;
                last_rel = std::abs(s[j + 1]) / bnorm;
                if (last_rel < tol)
                    conv = true;
            };

            auto arnoldi_pair = [&]() {
                for (int i = 0; i <= j; i++) {
                    cdouble* vi1 = &V1[(size_t)i * n];
                    cdouble* vi2 = &V2[(size_t)i * n];
                    cdouble hij1, hij2;
                    orthogonalize_against_pair_p(vi1, w1.data(), vi2, w2.data(), n, hij1, hij2);
                    H1[i * restart + j] = hij1;
                    H2[i * restart + j] = hij2;
                }
                if (reorth) {
                    for (int i = 0; i <= j; i++) {
                        cdouble* vi1 = &V1[(size_t)i * n];
                        cdouble* vi2 = &V2[(size_t)i * n];
                        cdouble hij1, hij2;
                        orthogonalize_against_pair_p(vi1, w1.data(), vi2, w2.data(), n, hij1, hij2);
                        H1[i * restart + j] += hij1;
                        H2[i * restart + j] += hij2;
                    }
                }

                double wn1, wn2;
                norm_pair_p(w1.data(), w2.data(), n, wn1, wn2);
                H1[(j + 1) * restart + j] = cdouble(wn1);
                H2[(j + 1) * restart + j] = cdouble(wn2);
                if (wn1 > 1e-30 && wn2 > 1e-30) {
                    cdouble* vnext1 = &V1[(size_t)(j + 1) * n];
                    cdouble* vnext2 = &V2[(size_t)(j + 1) * n];
                    double inv1 = 1.0 / wn1;
                    double inv2 = 1.0 / wn2;
                    #pragma omp parallel for schedule(static)
                    for (int k = 0; k < n; k++) {
                        vnext1[k] = w1[k] * inv1;
                        vnext2[k] = w2[k] * inv2;
                    }
                } else {
                    if (wn1 > 1e-30) {
                        cdouble* vnext1 = &V1[(size_t)(j + 1) * n];
                        double inv1 = 1.0 / wn1;
                        #pragma omp parallel for schedule(static)
                        for (int k = 0; k < n; k++)
                            vnext1[k] = w1[k] * inv1;
                    }
                    if (wn2 > 1e-30) {
                        cdouble* vnext2 = &V2[(size_t)(j + 1) * n];
                        double inv2 = 1.0 / wn2;
                        #pragma omp parallel for schedule(static)
                        for (int k = 0; k < n; k++)
                            vnext2[k] = w2[k] * inv2;
                    }
                }

                for (int i = 0; i < j; i++) {
                    cdouble h10 = H1[i * restart + j];
                    cdouble h11 = H1[(i + 1) * restart + j];
                    H1[i * restart + j]       = std::conj(cs1[i]) * h10 + std::conj(sn1[i]) * h11;
                    H1[(i + 1) * restart + j] = -sn1[i] * h10 + cs1[i] * h11;

                    cdouble h20 = H2[i * restart + j];
                    cdouble h21 = H2[(i + 1) * restart + j];
                    H2[i * restart + j]       = std::conj(cs2[i]) * h20 + std::conj(sn2[i]) * h21;
                    H2[(i + 1) * restart + j] = -sn2[i] * h20 + cs2[i] * h21;
                }

                cdouble h10 = H1[j * restart + j];
                cdouble h11 = H1[(j + 1) * restart + j];
                double den1 = std::sqrt(std::norm(h10) + std::norm(h11));
                cs1[j] = (den1 > 1e-30) ? h10 / den1 : cdouble(1);
                sn1[j] = (den1 > 1e-30) ? h11 / den1 : cdouble(0);
                H1[j * restart + j] = std::conj(cs1[j]) * h10 + std::conj(sn1[j]) * h11;
                H1[(j + 1) * restart + j] = cdouble(0);
                cdouble s10 = s1[j];
                s1[j] = std::conj(cs1[j]) * s10;
                s1[j + 1] = -sn1[j] * s10;
                m1 = j + 1;
                last_rel1 = std::abs(s1[j + 1]) / bnorm1;
                if (last_rel1 < tol)
                    conv1 = true;

                cdouble h20 = H2[j * restart + j];
                cdouble h21 = H2[(j + 1) * restart + j];
                double den2 = std::sqrt(std::norm(h20) + std::norm(h21));
                cs2[j] = (den2 > 1e-30) ? h20 / den2 : cdouble(1);
                sn2[j] = (den2 > 1e-30) ? h21 / den2 : cdouble(0);
                H2[j * restart + j] = std::conj(cs2[j]) * h20 + std::conj(sn2[j]) * h21;
                H2[(j + 1) * restart + j] = cdouble(0);
                cdouble s20 = s2[j];
                s2[j] = std::conj(cs2[j]) * s20;
                s2[j + 1] = -sn2[j] * s20;
                m2 = j + 1;
                last_rel2 = std::abs(s2[j + 1]) / bnorm2;
                if (last_rel2 < tol)
                    conv2 = true;
            };

            if (pair_arnoldi && !conv1 && !conv2)
                arnoldi_pair();
            else {
                arnoldi(V1, H1, cs1, sn1, s1, w1, bnorm1, conv1, m1, last_rel1);
                arnoldi(V2, H2, cs2, sn2, s2, w2, bnorm2, conv2, m2, last_rel2);
            }

            if (verbose && (total_matvecs <= 3 || total_matvecs % 10 == 0)) {
                double rel1 = conv1 ? 0.0 : std::abs(s1[j + 1]) / bnorm1;
                double rel2 = conv2 ? 0.0 : std::abs(s2[j + 1]) / bnorm2;
                printf("    GMRES iter %d: rel1=%.2e rel2=%.2e%s%s\n",
                       total_matvecs, rel1, rel2,
                       conv1 ? " [1:done]" : "", conv2 ? " [2:done]" : "");
                fflush(stdout);
            }

            if (inner_stagnation_window > 0 && !(conv1 && conv2)) {
                double rel1 = conv1 ? 0.0 : last_rel1;
                double rel2 = conv2 ? 0.0 : last_rel2;
                double pair_est = std::max(rel1, rel2);
                double required = best_inner_rel * (1.0 - inner_stagnation_rel);
                if (pair_est < required) {
                    best_inner_rel = pair_est;
                    best_inner_iter = total_matvecs;
                } else if (total_matvecs >= inner_stagnation_min_iter &&
                           total_matvecs - best_inner_iter >= inner_stagnation_window) {
                    stopped_stagnant = true;
                    if (verbose) {
                        printf("  [GMRES-paired] Inner stagnation stop at iter %d: best_est=%.2e current_est=%.2e threshold=%.2g window=%d\n",
                               total_matvecs, best_inner_rel, pair_est,
                               inner_stagnation_rel, inner_stagnation_window);
                        fflush(stdout);
                    }
                    break;
                }
            }
        }

        if (gmres_step_update_pair(n, restart, m1, m2, H1, H2, s1, s2, V1, V2, Z1, Z2,
                                   has_precond, store_z, precond, x1, x2, ytmp, ytmp2, ztmp, ztmp2) != 0) {
            numerical_breakdown = true;
            if (verbose) {
                printf("  [GMRES-paired] numerical breakdown while solving Hessenberg least-squares update\n");
                fflush(stdout);
            }
            break;
        }

        // The Arnoldi/Givens residual is an estimate.  Recompute the true
        // residual after each update before accepting convergence; otherwise
        // restarted/preconditioned GMRES can report a solved system while
        // ||b - A*x|| is still above tolerance.
        op.matvec_batch2(x1, x2, r1.data(), r2.data());
        total_matvecs++;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            r1[i] = b1[i] - r1[i];
            r2[i] = b2[i] - r2[i];
        }

        rnorm1 = norm_p(r1.data(), n);
        rnorm2 = norm_p(r2.data(), n);
        last_rel1 = rnorm1 / bnorm1;
        last_rel2 = rnorm2 / bnorm2;
        conv1 = (last_rel1 < tol);
        conv2 = (last_rel2 < tol);
        double pair_rel = std::max(last_rel1, last_rel2);
        if (pair_rel < best_pair_rel) {
            best_pair_rel = pair_rel;
            best_true_rel1 = last_rel1;
            best_true_rel2 = last_rel2;
            std::copy(x1, x1 + n, best_x1.begin());
            std::copy(x2, x2 + n, best_x2.begin());
        }

        if (verbose) {
            printf("  [GMRES-paired] restart %d: true rel1=%.2e true rel2=%.2e\n",
                   cycle + 1, last_rel1, last_rel2);
            fflush(stdout);
        }

        if (stopped_stagnant)
            break;

        if (stagnation_cycles > 0 && !(conv1 && conv2)) {
            double rel_now = std::max(conv1 ? 0.0 : rnorm1 / bnorm1,
                                      conv2 ? 0.0 : rnorm2 / bnorm2);
            double required = best_rel * (1.0 - stagnation_rel);
            if (rel_now < required) {
                best_rel = rel_now;
                best_inner_rel = rel_now;
                best_inner_iter = total_matvecs;
                stagnant_restart_count = 0;
            } else {
                stagnant_restart_count++;
            }
            if (stagnant_restart_count >= stagnation_cycles) {
                stopped_stagnant = true;
                if (verbose) {
                    printf("  [GMRES-paired] Stagnation stop after %d restarts: best=%.2e current=%.2e threshold=%.2g\n",
                           cycle + 1, best_rel, rel_now, stagnation_rel);
                    fflush(stdout);
                }
                break;
            }
        }
    }

    double final_pair_rel = std::max(last_rel1, last_rel2);
    bool reached_max_cycles = !(conv1 && conv2) && !stopped_stagnant && !numerical_breakdown;
    if (!(conv1 && conv2) && best_pair_rel < final_pair_rel) {
        std::copy(best_x1.begin(), best_x1.end(), x1);
        std::copy(best_x2.begin(), best_x2.end(), x2);
        last_rel1 = best_true_rel1;
        last_rel2 = best_true_rel2;
        conv1 = (last_rel1 < tol);
        conv2 = (last_rel2 < tol);
        ws.restored_best_iterate = true;
        if (verbose) {
            printf("  [GMRES-paired] restored best true-residual iterate: res1=%.2e res2=%.2e\n",
                   last_rel1, last_rel2);
            fflush(stdout);
        }
    }

    if (verbose) {
        if (conv1 && conv2)
            printf("  [GMRES-paired] Both converged, %d matvec evaluations\n", total_matvecs);
        else if (stopped_stagnant)
            printf("  [GMRES-paired] STOPPED by stagnation guard, %d matvecs, res1=%.2e res2=%.2e\n",
                   total_matvecs,
                   conv1 ? 0.0 : rnorm1 / bnorm1,
                   conv2 ? 0.0 : rnorm2 / bnorm2);
        else if (numerical_breakdown)
            printf("  [GMRES-paired] NOT fully converged: numerical breakdown, %d matvecs, res1=%.2e res2=%.2e\n",
                   total_matvecs,
                   conv1 ? 0.0 : rnorm1 / bnorm1,
                   conv2 ? 0.0 : rnorm2 / bnorm2);
        else if (reached_max_cycles)
            printf("  [GMRES-paired] NOT fully converged: reached max cycles, %d matvecs, res1=%.2e res2=%.2e\n",
                   total_matvecs,
                   conv1 ? 0.0 : last_rel1,
                   conv2 ? 0.0 : last_rel2);
        else
            printf("  [GMRES-paired] NOT fully converged (%s%s), %d matvecs, res1=%.2e res2=%.2e\n",
                   conv1 ? "" : "sys1 ", conv2 ? "" : "sys2 ", total_matvecs,
                   conv1 ? 0.0 : rnorm1 / bnorm1,
                   conv2 ? 0.0 : rnorm2 / bnorm2);
        fflush(stdout);
    }

    ws.final_relres1 = last_rel1;
    ws.final_relres2 = last_rel2;
    ws.converged1 = conv1;
    ws.converged2 = conv2;
    ws.stopped_stagnant = stopped_stagnant;
    ws.numerical_breakdown = numerical_breakdown;
    ws.reached_max_cycles = reached_max_cycles;

    return total_matvecs;
}
