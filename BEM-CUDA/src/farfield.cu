#include "farfield.h"
#include "gpu_select.h"
#include "quadrature.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <complex>
#include <cuda_runtime.h>

static void release_host_double_buffer(double*& ptr, bool& pinned)
{
    if (!ptr)
        return;
    if (pinned)
        cudaFreeHost(ptr);
    else
        std::free(ptr);
    ptr = nullptr;
    pinned = false;
}

static void allocate_host_double_buffer(double*& ptr, bool& pinned, size_t count, const char* label)
{
    release_host_double_buffer(ptr, pinned);
    if (count == 0)
        return;
    cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&ptr), count * sizeof(double), cudaHostAllocDefault);
    if (err == cudaSuccess) {
        pinned = true;
        return;
    }
    cudaGetLastError();
    ptr = static_cast<double*>(std::malloc(count * sizeof(double)));
    pinned = false;
    if (!ptr) {
        fprintf(stderr, "Error: failed to allocate %s host workspace (%.1f MB)\n",
                label, (double)(count * sizeof(double)) / (1024.0 * 1024.0));
        std::abort();
    }
    if (bem_env_flag_enabled("BEM_FF_VERBOSE")) {
        fprintf(stderr, "Warning: cudaHostAlloc failed for %s, using pageable host memory\n",
                label);
    }
}

static double farfield_m_scale()
{
    return bem_env_double("BEM_FF_M_SIGN", -1.0);
}

static double farfield_j_scale()
{
    return bem_env_double("BEM_FF_J_SCALE", 1.0);
}

static double farfield_phase_sign()
{
    return bem_env_double("BEM_FF_PHASE_SIGN", -1.0);
}

// ======== FFCache CPU init (unchanged) ========

void FFCache::init(const RWG& rwg, const Mesh& mesh, int quad_order) {
    TriQuad quad = tri_quadrature(quad_order);
    N = rwg.N;
    Nq = quad.npts;
    int total = 2 * N * Nq;

    qpts.resize(total * 3);
    fvals.resize(total * 3);
    jw.resize(total);

    std::vector<double> lam0(Nq);
    for (int q = 0; q < Nq; q++)
        lam0[q] = 1.0 - quad.pts[q][0] - quad.pts[q][1];

    for (int half = 0; half < 2; half++) {
        int sign = (half == 0) ? +1 : -1;
        int offset = half * N * Nq;

        for (int n = 0; n < N; n++) {
            int ti = (sign > 0) ? rwg.tri_p[n] : rwg.tri_m[n];
            Vec3 free_v = (sign > 0) ? rwg.free_p[n] : rwg.free_m[n];
            double area = (sign > 0) ? rwg.area_p[n] : rwg.area_m[n];
            double coeff = sign * rwg.length[n] / (2.0 * area);

            Vec3 v0, v1, v2;
            mesh.tri_verts(ti, v0, v1, v2);

            for (int q = 0; q < Nq; q++) {
                double l0 = lam0[q], l1 = quad.pts[q][0], l2 = quad.pts[q][1];
                Vec3 rr = v0 * l0 + v1 * l1 + v2 * l2;
                Vec3 fv = (rr - free_v) * coeff;

                int idx = offset + n * Nq + q;
                qpts[idx*3]   = rr.x;
                qpts[idx*3+1] = rr.y;
                qpts[idx*3+2] = rr.z;
                fvals[idx*3]   = fv.x;
                fvals[idx*3+1] = fv.y;
                fvals[idx*3+2] = fv.z;
                jw[idx] = area * quad.wts[q];
            }
        }
    }
}


// ======== FFCacheGPU: upload / free ========

void FFCacheGPU::upload(const FFCache& cache) {
    N = cache.N;
    Nq = cache.Nq;
    int total = 2 * N * Nq;

    CUDA_CHECK(cudaMalloc(&d_qpts,  total * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fvals, total * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_jw,    total * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_qpts,  cache.qpts.data(),  total * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fvals, cache.fvals.data(), total * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_jw,    cache.jw.data(),    total * sizeof(double), cudaMemcpyHostToDevice));

    initialized = true;
    printf("  FFCacheGPU: uploaded %d quad points (%.1f MB)\n",
           total, (total * 7.0 * sizeof(double)) / (1024.0 * 1024.0));
}

void FFCacheGPU::free() {
    if (initialized) {
        cudaFree(d_qpts);
        cudaFree(d_fvals);
        cudaFree(d_jw);
        d_qpts = d_fvals = d_jw = 0;
        initialized = false;
    }
}

FFCacheGPU::~FFCacheGPU() {
    free();
}


// ======== CUDA kernel ========
//
// Grid: (n_calls, ndir)
// Block: BLOCK_SIZE threads
//
// Each block computes Fv[call_idx, dir_idx, 0:3] by reducing over 2*N*Nq quad points.
// Threads cooperatively accumulate Jt[3] and Mt[3] (real + imag = 12 doubles) via shared memory.

#define FF_BLOCK 256

__global__ void unpack_complex_coeffs_kernel(
    const double2* __restrict__ Jz,
    const double2* __restrict__ Mz,
    int total,
    double* __restrict__ J_re,
    double* __restrict__ J_im,
    double* __restrict__ M_re,
    double* __restrict__ M_im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    double2 j = Jz[i];
    double2 m = Mz[i];
    J_re[i] = j.x;
    J_im[i] = j.y;
    M_re[i] = m.x;
    M_im[i] = m.y;
}

static void upload_complex_coeffs(FFBatchWorkspace& workspace,
                                  const std::complex<double>* coeffs_J,
                                  const std::complex<double>* coeffs_M,
                                  int total_coeffs)
{
    bool host_pack = bem_env_flag_enabled("BEM_FF_HOST_PACK");
    int gpu_pack_min = 262144;
    gpu_pack_min = std::max(0, bem_env_int("BEM_FF_GPU_PACK_MIN", gpu_pack_min));
    if (!host_pack && total_coeffs >= gpu_pack_min && sizeof(std::complex<double>) == sizeof(double2)) {
        if (total_coeffs > workspace.cap_coeffs_z) {
            cudaFree(workspace.d_cJ_z);
            cudaFree(workspace.d_cM_z);
            CUDA_CHECK(cudaMalloc(&workspace.d_cJ_z, (size_t)total_coeffs * sizeof(double2)));
            CUDA_CHECK(cudaMalloc(&workspace.d_cM_z, (size_t)total_coeffs * sizeof(double2)));
            workspace.cap_coeffs_z = total_coeffs;
        }
        CUDA_CHECK(cudaMemcpyAsync(workspace.d_cJ_z, coeffs_J,
                                   (size_t)total_coeffs * sizeof(double2), cudaMemcpyHostToDevice,
                                   workspace.stream));
        CUDA_CHECK(cudaMemcpyAsync(workspace.d_cM_z, coeffs_M,
                                   (size_t)total_coeffs * sizeof(double2), cudaMemcpyHostToDevice,
                                   workspace.stream));
        int block = 256;
        int grid = (total_coeffs + block - 1) / block;
        unpack_complex_coeffs_kernel<<<grid, block, 0, workspace.stream>>>(
            static_cast<const double2*>(workspace.d_cJ_z),
            static_cast<const double2*>(workspace.d_cM_z),
            total_coeffs,
            workspace.d_cJ_re, workspace.d_cJ_im,
            workspace.d_cM_re, workspace.d_cM_im);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    workspace.reserve_host_coeffs(total_coeffs);
    for (int i = 0; i < total_coeffs; i++) {
        workspace.h_cJ_re[i] = coeffs_J[i].real();
        workspace.h_cJ_im[i] = coeffs_J[i].imag();
        workspace.h_cM_re[i] = coeffs_M[i].real();
        workspace.h_cM_im[i] = coeffs_M[i].imag();
    }
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_cJ_re, workspace.h_cJ_re,
                               (size_t)total_coeffs * sizeof(double), cudaMemcpyHostToDevice,
                               workspace.stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_cJ_im, workspace.h_cJ_im,
                               (size_t)total_coeffs * sizeof(double), cudaMemcpyHostToDevice,
                               workspace.stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_cM_re, workspace.h_cM_re,
                               (size_t)total_coeffs * sizeof(double), cudaMemcpyHostToDevice,
                               workspace.stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_cM_im, workspace.h_cM_im,
                               (size_t)total_coeffs * sizeof(double), cudaMemcpyHostToDevice,
                               workspace.stream));
}

__global__ void farfield_batch_kernel(
    const double* __restrict__ qpts,    // (total, 3)
    const double* __restrict__ fvals,   // (total, 3)
    const double* __restrict__ jw,      // (total)
    const double* __restrict__ coeffs_J_re, // (n_calls, N) re
    const double* __restrict__ coeffs_J_im, // (n_calls, N) im
    const double* __restrict__ coeffs_M_re, // (n_calls, N) re
    const double* __restrict__ coeffs_M_im, // (n_calls, N) im
    const double* __restrict__ r_hats,      // (n_orient, ndir, 3)
    double k_re, double k_im, double eta_ext, double j_scale, double m_sign, double phase_sign,
    int N, int Nq, int n_calls, int n_orient, int ndir,
    double* __restrict__ Fv_re,  // (n_calls, ndir, 3)
    double* __restrict__ Fv_im)  // (n_calls, ndir, 3)
{
    int call_idx = blockIdx.x;
    int dir_idx  = blockIdx.y;
    if (call_idx >= n_calls || dir_idx >= ndir) return;

    int tid = threadIdx.x;
    int total = 2 * N * Nq;

    // r_hat for this direction: orient_idx = call_idx / 2
    int orient_idx = call_idx / 2;
    int rhat_base = (orient_idx * ndir + dir_idx) * 3;
    double rx = r_hats[rhat_base];
    double ry = r_hats[rhat_base + 1];
    double rz = r_hats[rhat_base + 2];

    // Coefficient base for this call
    int coeff_base = call_idx * N;

    // Thread-local accumulators: Jt(re,im)[3], Mt(re,im)[3]
    double jt_re0 = 0, jt_re1 = 0, jt_re2 = 0;
    double jt_im0 = 0, jt_im1 = 0, jt_im2 = 0;
    double mt_re0 = 0, mt_re1 = 0, mt_re2 = 0;
    double mt_im0 = 0, mt_im1 = 0, mt_im2 = 0;

    // Each thread processes quad points: tid, tid+BLOCK, tid+2*BLOCK, ...
    for (int i = tid; i < total; i += FF_BLOCK) {
        // Which basis function does this quad point belong to?
        int n = (i % (N * Nq)) / Nq;  // basis fn index within half

        double px = qpts[i * 3];
        double py = qpts[i * 3 + 1];
        double pz = qpts[i * 3 + 2];
        double fx = fvals[i * 3];
        double fy = fvals[i * 3 + 1];
        double fz = fvals[i * 3 + 2];
        double w  = jw[i];

        // Phase: exp(i * phase_sign * k * r_hat . r'), default phase_sign=-1.
        double rdot = rx * px + ry * py + rz * pz;
        double arg = phase_sign * k_re * rdot;
        double sn, cs;
        sincos(arg, &sn, &cs);
        double ea = (k_im == 0.0) ? 1.0 : exp(-phase_sign * k_im * rdot);
        double c = cs * ea;       // Re(phase)
        double s = sn * ea;       // Im(phase)

        // phase * w * f
        double pw_re = c * w;
        double pw_im = s * w;
        double ifx_re = fx * pw_re;
        double ifx_im = fx * pw_im;
        double ify_re = fy * pw_re;
        double ify_im = fy * pw_im;
        double ifz_re = fz * pw_re;
        double ifz_im = fz * pw_im;

        // Multiply by coefficients (complex): coeff * integ
        double cJ_re = coeffs_J_re[coeff_base + n];
        double cJ_im = coeffs_J_im[coeff_base + n];
        double cM_re = coeffs_M_re[coeff_base + n];
        double cM_im = coeffs_M_im[coeff_base + n];

        // J contrib: cJ * integ (complex multiply)
        jt_re0 += cJ_re * ifx_re - cJ_im * ifx_im;
        jt_im0 += cJ_re * ifx_im + cJ_im * ifx_re;
        jt_re1 += cJ_re * ify_re - cJ_im * ify_im;
        jt_im1 += cJ_re * ify_im + cJ_im * ify_re;
        jt_re2 += cJ_re * ifz_re - cJ_im * ifz_im;
        jt_im2 += cJ_re * ifz_im + cJ_im * ifz_re;

        // M contrib: cM * integ
        mt_re0 += cM_re * ifx_re - cM_im * ifx_im;
        mt_im0 += cM_re * ifx_im + cM_im * ifx_re;
        mt_re1 += cM_re * ify_re - cM_im * ify_im;
        mt_im1 += cM_re * ify_im + cM_im * ify_re;
        mt_re2 += cM_re * ifz_re - cM_im * ifz_im;
        mt_im2 += cM_re * ifz_im + cM_im * ifz_re;
    }

    // Shared memory reduction for 12 values
    __shared__ double smem[12 * FF_BLOCK];
    smem[0  * FF_BLOCK + tid] = jt_re0;
    smem[1  * FF_BLOCK + tid] = jt_im0;
    smem[2  * FF_BLOCK + tid] = jt_re1;
    smem[3  * FF_BLOCK + tid] = jt_im1;
    smem[4  * FF_BLOCK + tid] = jt_re2;
    smem[5  * FF_BLOCK + tid] = jt_im2;
    smem[6  * FF_BLOCK + tid] = mt_re0;
    smem[7  * FF_BLOCK + tid] = mt_im0;
    smem[8  * FF_BLOCK + tid] = mt_re1;
    smem[9  * FF_BLOCK + tid] = mt_im1;
    smem[10 * FF_BLOCK + tid] = mt_re2;
    smem[11 * FF_BLOCK + tid] = mt_im2;
    __syncthreads();

    // Tree reduction
    for (int stride = FF_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int k = 0; k < 12; k++)
                smem[k * FF_BLOCK + tid] += smem[k * FF_BLOCK + tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 finalizes
    if (tid == 0) {
        double jr0 = smem[0], ji0 = smem[1 * FF_BLOCK];
        double jr1 = smem[2 * FF_BLOCK], ji1 = smem[3 * FF_BLOCK];
        double jr2 = smem[4 * FF_BLOCK], ji2 = smem[5 * FF_BLOCK];
        double mr0 = smem[6 * FF_BLOCK], mi0 = smem[7 * FF_BLOCK];
        double mr1 = smem[8 * FF_BLOCK], mi1 = smem[9 * FF_BLOCK];
        double mr2 = smem[10 * FF_BLOCK], mi2 = smem[11 * FF_BLOCK];

        // Jp = J - r_hat * (r_hat . J)
        double rdotJ_re = rx * jr0 + ry * jr1 + rz * jr2;
        double rdotJ_im = rx * ji0 + ry * ji1 + rz * ji2;
        double jp_re0 = jr0 - rx * rdotJ_re, jp_im0 = ji0 - rx * rdotJ_im;
        double jp_re1 = jr1 - ry * rdotJ_re, jp_im1 = ji1 - ry * rdotJ_im;
        double jp_re2 = jr2 - rz * rdotJ_re, jp_im2 = ji2 - rz * rdotJ_im;

        // Mc = r_hat x M
        double mc_re0 = ry * mr2 - rz * mr1, mc_im0 = ry * mi2 - rz * mi1;
        double mc_re1 = rz * mr0 - rx * mr2, mc_im1 = rz * mi0 - rx * mi2;
        double mc_re2 = rx * mr1 - ry * mr0, mc_im2 = rx * mi1 - ry * mi0;

        // F = prefac * (sJ * eta * Jp + sM * Mc)
        // prefac = -ik/(4pi), sM = -1
        // prefac = (-i) * (k_re + i*k_im) / (4pi)
        //        = (k_im - i*k_re) / (4pi)
        double inv4pi = 1.0 / (4.0 * M_PI);
        double pf_re = k_im * inv4pi;
        double pf_im = -k_re * inv4pi;
        double sM = m_sign;

        // v = eta*Jp + sM*Mc for each component
        double eJ = j_scale * eta_ext;
        double v_re0 = eJ * jp_re0 + sM * mc_re0;
        double v_im0 = eJ * jp_im0 + sM * mc_im0;
        double v_re1 = eJ * jp_re1 + sM * mc_re1;
        double v_im1 = eJ * jp_im1 + sM * mc_im1;
        double v_re2 = eJ * jp_re2 + sM * mc_re2;
        double v_im2 = eJ * jp_im2 + sM * mc_im2;

        // F = prefac * v (complex multiply)
        int out_base = (call_idx * ndir + dir_idx) * 3;
        Fv_re[out_base]   = pf_re * v_re0 - pf_im * v_im0;
        Fv_im[out_base]   = pf_re * v_im0 + pf_im * v_re0;
        Fv_re[out_base+1] = pf_re * v_re1 - pf_im * v_im1;
        Fv_im[out_base+1] = pf_re * v_im1 + pf_im * v_re1;
        Fv_re[out_base+2] = pf_re * v_re2 - pf_im * v_im2;
        Fv_im[out_base+2] = pf_re * v_im2 + pf_im * v_re2;
    }
}

__device__ inline void cmul(double ar, double ai, double br, double bi,
                            double& rr, double& ri)
{
    rr = ar * br - ai * bi;
    ri = ar * bi + ai * br;
}

__device__ inline double cnorm2(double ar, double ai)
{
    return ar * ar + ai * ai;
}

__device__ inline void c_mul_conj(double ar, double ai, double br, double bi,
                                  double& rr, double& ri)
{
    rr = ar * br + ai * bi;
    ri = ai * br - ar * bi;
}

__global__ void mueller_accum_kernel(
    const double* __restrict__ Fv_re,
    const double* __restrict__ Fv_im,
    const double* __restrict__ e_par,
    const double* __restrict__ e_perp,
    const double* __restrict__ weights,
    double ik_re, double ik_im, double inv_k2,
    int n_orient, int ndir,
    double* __restrict__ M_accum)
{
    int oi = blockIdx.x;
    int t = blockIdx.y * blockDim.x + threadIdx.x;
    if (oi >= n_orient || t >= ndir)
        return;

    int fv_par = ((2 * oi) * ndir + t) * 3;
    int fv_perp = ((2 * oi + 1) * ndir + t) * 3;
    int ebase = (oi * ndir + t) * 3;
    double epx = e_par[ebase], epy = e_par[ebase + 1], epz = e_par[ebase + 2];
    double eqx = e_perp[ebase], eqy = e_perp[ebase + 1], eqz = e_perp[ebase + 2];

    double fpp_re = Fv_re[fv_par] * epx + Fv_re[fv_par + 1] * epy + Fv_re[fv_par + 2] * epz;
    double fpp_im = Fv_im[fv_par] * epx + Fv_im[fv_par + 1] * epy + Fv_im[fv_par + 2] * epz;
    double fpq_re = Fv_re[fv_par] * eqx + Fv_re[fv_par + 1] * eqy + Fv_re[fv_par + 2] * eqz;
    double fpq_im = Fv_im[fv_par] * eqx + Fv_im[fv_par + 1] * eqy + Fv_im[fv_par + 2] * eqz;
    double fqp_re = Fv_re[fv_perp] * epx + Fv_re[fv_perp + 1] * epy + Fv_re[fv_perp + 2] * epz;
    double fqp_im = Fv_im[fv_perp] * epx + Fv_im[fv_perp + 1] * epy + Fv_im[fv_perp + 2] * epz;
    double fqq_re = Fv_re[fv_perp] * eqx + Fv_re[fv_perp + 1] * eqy + Fv_re[fv_perp + 2] * eqz;
    double fqq_im = Fv_im[fv_perp] * eqx + Fv_im[fv_perp + 1] * eqy + Fv_im[fv_perp + 2] * eqz;

    double S1r, S1i, S2r, S2i, S3r, S3i, S4r, S4i;
    cmul(ik_re, ik_im, fqq_re, fqq_im, S1r, S1i);
    cmul(ik_re, ik_im, fpp_re, fpp_im, S2r, S2i);
    cmul(ik_re, ik_im, fqp_re, fqp_im, S3r, S3i);
    cmul(ik_re, ik_im, fpq_re, fpq_im, S4r, S4i);

    double as1 = cnorm2(S1r, S1i), as2 = cnorm2(S2r, S2i);
    double as3 = cnorm2(S3r, S3i), as4 = cnorm2(S4r, S4i);

    double s23r, s23i, s14r, s14i, s24r, s24i, s13r, s13i;
    double s12r, s12i, s34r, s34i, s21r, s21i, s42r, s42i, s43r, s43i;
    c_mul_conj(S2r, S2i, S3r, S3i, s23r, s23i);
    c_mul_conj(S1r, S1i, S4r, S4i, s14r, s14i);
    c_mul_conj(S2r, S2i, S4r, S4i, s24r, s24i);
    c_mul_conj(S1r, S1i, S3r, S3i, s13r, s13i);
    c_mul_conj(S1r, S1i, S2r, S2i, s12r, s12i);
    c_mul_conj(S3r, S3i, S4r, S4i, s34r, s34i);
    c_mul_conj(S2r, S2i, S1r, S1i, s21r, s21i);
    c_mul_conj(S4r, S4i, S2r, S2i, s42r, s42i);
    c_mul_conj(S4r, S4i, S3r, S3i, s43r, s43i);

    double scale = weights[oi] * inv_k2;
    #define MADD(i,j,val) atomicAdd(&M_accum[((i)*4+(j))*ndir + t], scale * (val))
    MADD(0,0, 0.5 * (as1 + as2 + as3 + as4));
    MADD(0,1, 0.5 * (as2 - as1 + as4 - as3));
    MADD(0,2, s23r + s14r);
    MADD(0,3, s23i - s14i);
    MADD(1,0, 0.5 * (as2 - as1 - as4 + as3));
    MADD(1,1, 0.5 * (as2 + as1 - as4 - as3));
    MADD(1,2, s23r - s14r);
    MADD(1,3, s23i + s14i);
    MADD(2,0, s24r + s13r);
    MADD(2,1, s24r - s13r);
    MADD(2,2, s12r + s34r);
    MADD(2,3, s21i + s43i);
    MADD(3,0, s42i + s13i);
    MADD(3,1, s42i - s13i);
    MADD(3,2, s12i - s34i);
    MADD(3,3, s12r - s34r);
    #undef MADD
}

#define FF_DIRECT_BLOCK 128

__global__ void farfield_mueller_direct_kernel(
    const double* __restrict__ qpts,
    const double* __restrict__ fvals,
    const double* __restrict__ jw,
    const double* __restrict__ coeffs_J_re,
    const double* __restrict__ coeffs_J_im,
    const double* __restrict__ coeffs_M_re,
    const double* __restrict__ coeffs_M_im,
    const double* __restrict__ r_hats,
    const double* __restrict__ e_par,
    const double* __restrict__ e_perp,
    const double* __restrict__ weights,
    double k_re, double k_im, double eta_ext, double j_scale, double m_sign, double phase_sign,
    double ik_re, double ik_im, double inv_k2,
    int N, int Nq, int n_orient, int ndir,
    double* __restrict__ M_accum)
{
    int oi = blockIdx.x;
    int dir_idx = blockIdx.y;
    if (oi >= n_orient || dir_idx >= ndir) return;

    int tid = threadIdx.x;
    int total = 2 * N * Nq;

    int rhat_base = (oi * ndir + dir_idx) * 3;
    double rx = r_hats[rhat_base];
    double ry = r_hats[rhat_base + 1];
    double rz = r_hats[rhat_base + 2];

    int coeff_base0 = (2 * oi) * N;
    int coeff_base1 = (2 * oi + 1) * N;

    double a[24];
    #pragma unroll
    for (int k = 0; k < 24; k++) a[k] = 0.0;

    for (int i = tid; i < total; i += FF_DIRECT_BLOCK) {
        int n = (i % (N * Nq)) / Nq;

        double px = qpts[i * 3];
        double py = qpts[i * 3 + 1];
        double pz = qpts[i * 3 + 2];
        double fx = fvals[i * 3];
        double fy = fvals[i * 3 + 1];
        double fz = fvals[i * 3 + 2];
        double w  = jw[i];

        double rdot = rx * px + ry * py + rz * pz;
        double arg = phase_sign * k_re * rdot;
        double sn, cs;
        sincos(arg, &sn, &cs);
        double ea = (k_im == 0.0) ? 1.0 : exp(-phase_sign * k_im * rdot);
        double pw_re = cs * ea * w;
        double pw_im = sn * ea * w;

        double ifx_re = fx * pw_re, ifx_im = fx * pw_im;
        double ify_re = fy * pw_re, ify_im = fy * pw_im;
        double ifz_re = fz * pw_re, ifz_im = fz * pw_im;

        double cJ0_re = coeffs_J_re[coeff_base0 + n];
        double cJ0_im = coeffs_J_im[coeff_base0 + n];
        double cM0_re = coeffs_M_re[coeff_base0 + n];
        double cM0_im = coeffs_M_im[coeff_base0 + n];
        double cJ1_re = coeffs_J_re[coeff_base1 + n];
        double cJ1_im = coeffs_J_im[coeff_base1 + n];
        double cM1_re = coeffs_M_re[coeff_base1 + n];
        double cM1_im = coeffs_M_im[coeff_base1 + n];

        a[0]  += cJ0_re * ifx_re - cJ0_im * ifx_im;
        a[1]  += cJ0_re * ifx_im + cJ0_im * ifx_re;
        a[2]  += cJ0_re * ify_re - cJ0_im * ify_im;
        a[3]  += cJ0_re * ify_im + cJ0_im * ify_re;
        a[4]  += cJ0_re * ifz_re - cJ0_im * ifz_im;
        a[5]  += cJ0_re * ifz_im + cJ0_im * ifz_re;
        a[6]  += cM0_re * ifx_re - cM0_im * ifx_im;
        a[7]  += cM0_re * ifx_im + cM0_im * ifx_re;
        a[8]  += cM0_re * ify_re - cM0_im * ify_im;
        a[9]  += cM0_re * ify_im + cM0_im * ify_re;
        a[10] += cM0_re * ifz_re - cM0_im * ifz_im;
        a[11] += cM0_re * ifz_im + cM0_im * ifz_re;

        a[12] += cJ1_re * ifx_re - cJ1_im * ifx_im;
        a[13] += cJ1_re * ifx_im + cJ1_im * ifx_re;
        a[14] += cJ1_re * ify_re - cJ1_im * ify_im;
        a[15] += cJ1_re * ify_im + cJ1_im * ify_re;
        a[16] += cJ1_re * ifz_re - cJ1_im * ifz_im;
        a[17] += cJ1_re * ifz_im + cJ1_im * ifz_re;
        a[18] += cM1_re * ifx_re - cM1_im * ifx_im;
        a[19] += cM1_re * ifx_im + cM1_im * ifx_re;
        a[20] += cM1_re * ify_re - cM1_im * ify_im;
        a[21] += cM1_re * ify_im + cM1_im * ify_re;
        a[22] += cM1_re * ifz_re - cM1_im * ifz_im;
        a[23] += cM1_re * ifz_im + cM1_im * ifz_re;
    }

    __shared__ double smem[24 * FF_DIRECT_BLOCK];
    #pragma unroll
    for (int k = 0; k < 24; k++)
        smem[k * FF_DIRECT_BLOCK + tid] = a[k];
    __syncthreads();

    for (int stride = FF_DIRECT_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int k = 0; k < 24; k++)
                smem[k * FF_DIRECT_BLOCK + tid] += smem[k * FF_DIRECT_BLOCK + tid + stride];
        }
        __syncthreads();
    }

    if (tid != 0) return;

    double jr0[2] = {smem[0],  smem[12 * FF_DIRECT_BLOCK]};
    double ji0[2] = {smem[1 * FF_DIRECT_BLOCK],  smem[13 * FF_DIRECT_BLOCK]};
    double jr1[2] = {smem[2 * FF_DIRECT_BLOCK],  smem[14 * FF_DIRECT_BLOCK]};
    double ji1[2] = {smem[3 * FF_DIRECT_BLOCK],  smem[15 * FF_DIRECT_BLOCK]};
    double jr2[2] = {smem[4 * FF_DIRECT_BLOCK],  smem[16 * FF_DIRECT_BLOCK]};
    double ji2[2] = {smem[5 * FF_DIRECT_BLOCK],  smem[17 * FF_DIRECT_BLOCK]};
    double mr0[2] = {smem[6 * FF_DIRECT_BLOCK],  smem[18 * FF_DIRECT_BLOCK]};
    double mi0[2] = {smem[7 * FF_DIRECT_BLOCK],  smem[19 * FF_DIRECT_BLOCK]};
    double mr1[2] = {smem[8 * FF_DIRECT_BLOCK],  smem[20 * FF_DIRECT_BLOCK]};
    double mi1[2] = {smem[9 * FF_DIRECT_BLOCK],  smem[21 * FF_DIRECT_BLOCK]};
    double mr2[2] = {smem[10 * FF_DIRECT_BLOCK], smem[22 * FF_DIRECT_BLOCK]};
    double mi2[2] = {smem[11 * FF_DIRECT_BLOCK], smem[23 * FF_DIRECT_BLOCK]};

    double fre[6], fim[6];
    double pf_re = k_im / (4.0 * M_PI);
    double pf_im = -k_re / (4.0 * M_PI);
    #pragma unroll
    for (int c = 0; c < 2; c++) {
        double rdotJ_re = rx * jr0[c] + ry * jr1[c] + rz * jr2[c];
        double rdotJ_im = rx * ji0[c] + ry * ji1[c] + rz * ji2[c];
        double jp_re0 = jr0[c] - rx * rdotJ_re, jp_im0 = ji0[c] - rx * rdotJ_im;
        double jp_re1 = jr1[c] - ry * rdotJ_re, jp_im1 = ji1[c] - ry * rdotJ_im;
        double jp_re2 = jr2[c] - rz * rdotJ_re, jp_im2 = ji2[c] - rz * rdotJ_im;

        double mc_re0 = ry * mr2[c] - rz * mr1[c], mc_im0 = ry * mi2[c] - rz * mi1[c];
        double mc_re1 = rz * mr0[c] - rx * mr2[c], mc_im1 = rz * mi0[c] - rx * mi2[c];
        double mc_re2 = rx * mr1[c] - ry * mr0[c], mc_im2 = rx * mi1[c] - ry * mi0[c];

        double eJ = j_scale * eta_ext;
        double v_re0 = eJ * jp_re0 + m_sign * mc_re0;
        double v_im0 = eJ * jp_im0 + m_sign * mc_im0;
        double v_re1 = eJ * jp_re1 + m_sign * mc_re1;
        double v_im1 = eJ * jp_im1 + m_sign * mc_im1;
        double v_re2 = eJ * jp_re2 + m_sign * mc_re2;
        double v_im2 = eJ * jp_im2 + m_sign * mc_im2;

        int o = 3 * c;
        fre[o]     = pf_re * v_re0 - pf_im * v_im0;
        fim[o]     = pf_re * v_im0 + pf_im * v_re0;
        fre[o + 1] = pf_re * v_re1 - pf_im * v_im1;
        fim[o + 1] = pf_re * v_im1 + pf_im * v_re1;
        fre[o + 2] = pf_re * v_re2 - pf_im * v_im2;
        fim[o + 2] = pf_re * v_im2 + pf_im * v_re2;
    }

    int ebase = (oi * ndir + dir_idx) * 3;
    double epx = e_par[ebase], epy = e_par[ebase + 1], epz = e_par[ebase + 2];
    double eqx = e_perp[ebase], eqy = e_perp[ebase + 1], eqz = e_perp[ebase + 2];

    double fpp_re = fre[0] * epx + fre[1] * epy + fre[2] * epz;
    double fpp_im = fim[0] * epx + fim[1] * epy + fim[2] * epz;
    double fpq_re = fre[0] * eqx + fre[1] * eqy + fre[2] * eqz;
    double fpq_im = fim[0] * eqx + fim[1] * eqy + fim[2] * eqz;
    double fqp_re = fre[3] * epx + fre[4] * epy + fre[5] * epz;
    double fqp_im = fim[3] * epx + fim[4] * epy + fim[5] * epz;
    double fqq_re = fre[3] * eqx + fre[4] * eqy + fre[5] * eqz;
    double fqq_im = fim[3] * eqx + fim[4] * eqy + fim[5] * eqz;

    double S1r, S1i, S2r, S2i, S3r, S3i, S4r, S4i;
    cmul(ik_re, ik_im, fqq_re, fqq_im, S1r, S1i);
    cmul(ik_re, ik_im, fpp_re, fpp_im, S2r, S2i);
    cmul(ik_re, ik_im, fqp_re, fqp_im, S3r, S3i);
    cmul(ik_re, ik_im, fpq_re, fpq_im, S4r, S4i);

    double as1 = cnorm2(S1r, S1i), as2 = cnorm2(S2r, S2i);
    double as3 = cnorm2(S3r, S3i), as4 = cnorm2(S4r, S4i);
    double s23r, s23i, s14r, s14i, s24r, s24i, s13r, s13i;
    double s12r, s12i, s34r, s34i, s21r, s21i, s42r, s42i, s43r, s43i;
    c_mul_conj(S2r, S2i, S3r, S3i, s23r, s23i);
    c_mul_conj(S1r, S1i, S4r, S4i, s14r, s14i);
    c_mul_conj(S2r, S2i, S4r, S4i, s24r, s24i);
    c_mul_conj(S1r, S1i, S3r, S3i, s13r, s13i);
    c_mul_conj(S1r, S1i, S2r, S2i, s12r, s12i);
    c_mul_conj(S3r, S3i, S4r, S4i, s34r, s34i);
    c_mul_conj(S2r, S2i, S1r, S1i, s21r, s21i);
    c_mul_conj(S4r, S4i, S2r, S2i, s42r, s42i);
    c_mul_conj(S4r, S4i, S3r, S3i, s43r, s43i);

    double scale = weights[oi] * inv_k2;
    int t = dir_idx;
    #define MADD_DIRECT(i,j,val) atomicAdd(&M_accum[((i)*4+(j))*ndir + t], scale * (val))
    MADD_DIRECT(0,0, 0.5 * (as1 + as2 + as3 + as4));
    MADD_DIRECT(0,1, 0.5 * (as2 - as1 + as4 - as3));
    MADD_DIRECT(0,2, s23r + s14r);
    MADD_DIRECT(0,3, s23i - s14i);
    MADD_DIRECT(1,0, 0.5 * (as2 - as1 - as4 + as3));
    MADD_DIRECT(1,1, 0.5 * (as2 + as1 - as4 - as3));
    MADD_DIRECT(1,2, s23r - s14r);
    MADD_DIRECT(1,3, s23i + s14i);
    MADD_DIRECT(2,0, s24r + s13r);
    MADD_DIRECT(2,1, s24r - s13r);
    MADD_DIRECT(2,2, s12r + s34r);
    MADD_DIRECT(2,3, s21i + s43i);
    MADD_DIRECT(3,0, s42i + s13i);
    MADD_DIRECT(3,1, s42i - s13i);
    MADD_DIRECT(3,2, s12i - s34i);
    MADD_DIRECT(3,3, s12r - s34r);
    #undef MADD_DIRECT
}

__global__ void farfield_mueller_alpha_kernel(
    const double* __restrict__ qpts,
    const double* __restrict__ fvals,
    const double* __restrict__ jw,
    const double* __restrict__ coeffs_J_re,
    const double* __restrict__ coeffs_J_im,
    const double* __restrict__ coeffs_M_re,
    const double* __restrict__ coeffs_M_im,
    const double* __restrict__ r_hats,
    const double* __restrict__ e_par,
    const double* __restrict__ e_perp,
    const double* __restrict__ RT_mats,
    const double* __restrict__ rhat_lab,
    const double* __restrict__ etheta_lab,
    double ephi_x, double ephi_y, double ephi_z,
    const double* __restrict__ weights,
    const double* __restrict__ alpha_cos,
    const double* __restrict__ alpha_sin,
    double k_re, double k_im, double eta_ext, double j_scale, double m_sign, double phase_sign,
    double ik_re, double ik_im, double inv_k2,
    int N, int Nq, int n_samples, int alpha_avg, int ndir, int geom_mode, int weights_by_base,
    int partial_groups,
    double* __restrict__ M_accum)
{
    int sample_idx = blockIdx.x;
    int dir_idx = blockIdx.y;
    if (sample_idx >= n_samples || dir_idx >= ndir) return;

    int tid = threadIdx.x;
    int total = 2 * N * Nq;
    int base_idx = sample_idx / alpha_avg;
    int alpha_idx = sample_idx - base_idx * alpha_avg;
    double ca = alpha_cos[alpha_idx];
    double sa = alpha_sin[alpha_idx];

    double rx, ry, rz;
    double epx, epy, epz;
    double eqx, eqy, eqz;
    if (geom_mode) {
        const double* RT = RT_mats + base_idx * 9;
        const double* rh0 = rhat_lab + dir_idx * 3;
        const double* ep0 = etheta_lab + dir_idx * 3;

        double x = ca * rh0[0] + sa * rh0[1];
        double y = -sa * rh0[0] + ca * rh0[1];
        double z = rh0[2];
        rx = RT[0] * x + RT[1] * y + RT[2] * z;
        ry = RT[3] * x + RT[4] * y + RT[5] * z;
        rz = RT[6] * x + RT[7] * y + RT[8] * z;

        x = ca * ep0[0] + sa * ep0[1];
        y = -sa * ep0[0] + ca * ep0[1];
        z = ep0[2];
        epx = RT[0] * x + RT[1] * y + RT[2] * z;
        epy = RT[3] * x + RT[4] * y + RT[5] * z;
        epz = RT[6] * x + RT[7] * y + RT[8] * z;

        x = ca * ephi_x + sa * ephi_y;
        y = -sa * ephi_x + ca * ephi_y;
        z = ephi_z;
        eqx = RT[0] * x + RT[1] * y + RT[2] * z;
        eqy = RT[3] * x + RT[4] * y + RT[5] * z;
        eqz = RT[6] * x + RT[7] * y + RT[8] * z;
    } else {
        int ebase = (sample_idx * ndir + dir_idx) * 3;
        rx = r_hats[ebase];
        ry = r_hats[ebase + 1];
        rz = r_hats[ebase + 2];
        epx = e_par[ebase];
        epy = e_par[ebase + 1];
        epz = e_par[ebase + 2];
        eqx = e_perp[ebase];
        eqy = e_perp[ebase + 1];
        eqz = e_perp[ebase + 2];
    }

    int coeff_base_par = (2 * base_idx) * N;
    int coeff_base_perp = (2 * base_idx + 1) * N;

    double a[24];
    #pragma unroll
    for (int k = 0; k < 24; k++) a[k] = 0.0;

    for (int i = tid; i < total; i += FF_DIRECT_BLOCK) {
        int n = (i % (N * Nq)) / Nq;

        double px = qpts[i * 3];
        double py = qpts[i * 3 + 1];
        double pz = qpts[i * 3 + 2];
        double fx = fvals[i * 3];
        double fy = fvals[i * 3 + 1];
        double fz = fvals[i * 3 + 2];
        double w  = jw[i];

        double rdot = rx * px + ry * py + rz * pz;
        double arg = phase_sign * k_re * rdot;
        double sn, cs;
        sincos(arg, &sn, &cs);
        double ea = (k_im == 0.0) ? 1.0 : exp(-phase_sign * k_im * rdot);
        double pw_re = cs * ea * w;
        double pw_im = sn * ea * w;

        double ifx_re = fx * pw_re, ifx_im = fx * pw_im;
        double ify_re = fy * pw_re, ify_im = fy * pw_im;
        double ifz_re = fz * pw_re, ifz_im = fz * pw_im;

        double jp_re = coeffs_J_re[coeff_base_par + n];
        double jp_im = coeffs_J_im[coeff_base_par + n];
        double mp_re = coeffs_M_re[coeff_base_par + n];
        double mp_im = coeffs_M_im[coeff_base_par + n];
        double ju_re = coeffs_J_re[coeff_base_perp + n];
        double ju_im = coeffs_J_im[coeff_base_perp + n];
        double mu_re = coeffs_M_re[coeff_base_perp + n];
        double mu_im = coeffs_M_im[coeff_base_perp + n];

        bool yz_plane = fabs(ephi_x) > 0.5;
        double cJ0_re, cJ0_im, cM0_re, cM0_im;
        double cJ1_re, cJ1_im, cM1_re, cM1_im;
        if (yz_plane) {
            cJ0_re = ca * jp_re + sa * ju_re;
            cJ0_im = ca * jp_im + sa * ju_im;
            cM0_re = ca * mp_re + sa * mu_re;
            cM0_im = ca * mp_im + sa * mu_im;
            cJ1_re = -sa * jp_re + ca * ju_re;
            cJ1_im = -sa * jp_im + ca * ju_im;
            cM1_re = -sa * mp_re + ca * mu_re;
            cM1_im = -sa * mp_im + ca * mu_im;
        } else {
            cJ0_re = ca * jp_re - sa * ju_re;
            cJ0_im = ca * jp_im - sa * ju_im;
            cM0_re = ca * mp_re - sa * mu_re;
            cM0_im = ca * mp_im - sa * mu_im;
            cJ1_re = -sa * jp_re - ca * ju_re;
            cJ1_im = -sa * jp_im - ca * ju_im;
            cM1_re = -sa * mp_re - ca * mu_re;
            cM1_im = -sa * mp_im - ca * mu_im;
        }

        a[0]  += cJ0_re * ifx_re - cJ0_im * ifx_im;
        a[1]  += cJ0_re * ifx_im + cJ0_im * ifx_re;
        a[2]  += cJ0_re * ify_re - cJ0_im * ify_im;
        a[3]  += cJ0_re * ify_im + cJ0_im * ify_re;
        a[4]  += cJ0_re * ifz_re - cJ0_im * ifz_im;
        a[5]  += cJ0_re * ifz_im + cJ0_im * ifz_re;
        a[6]  += cM0_re * ifx_re - cM0_im * ifx_im;
        a[7]  += cM0_re * ifx_im + cM0_im * ifx_re;
        a[8]  += cM0_re * ify_re - cM0_im * ify_im;
        a[9]  += cM0_re * ify_im + cM0_im * ify_re;
        a[10] += cM0_re * ifz_re - cM0_im * ifz_im;
        a[11] += cM0_re * ifz_im + cM0_im * ifz_re;

        a[12] += cJ1_re * ifx_re - cJ1_im * ifx_im;
        a[13] += cJ1_re * ifx_im + cJ1_im * ifx_re;
        a[14] += cJ1_re * ify_re - cJ1_im * ify_im;
        a[15] += cJ1_re * ify_im + cJ1_im * ify_re;
        a[16] += cJ1_re * ifz_re - cJ1_im * ifz_im;
        a[17] += cJ1_re * ifz_im + cJ1_im * ifz_re;
        a[18] += cM1_re * ifx_re - cM1_im * ifx_im;
        a[19] += cM1_re * ifx_im + cM1_im * ifx_re;
        a[20] += cM1_re * ify_re - cM1_im * ify_im;
        a[21] += cM1_re * ify_im + cM1_im * ify_re;
        a[22] += cM1_re * ifz_re - cM1_im * ifz_im;
        a[23] += cM1_re * ifz_im + cM1_im * ifz_re;
    }

    __shared__ double smem[24 * FF_DIRECT_BLOCK];
    #pragma unroll
    for (int k = 0; k < 24; k++)
        smem[k * FF_DIRECT_BLOCK + tid] = a[k];
    __syncthreads();

    for (int stride = FF_DIRECT_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int k = 0; k < 24; k++)
                smem[k * FF_DIRECT_BLOCK + tid] += smem[k * FF_DIRECT_BLOCK + tid + stride];
        }
        __syncthreads();
    }

    if (tid != 0) return;

    double jr0[2] = {smem[0],  smem[12 * FF_DIRECT_BLOCK]};
    double ji0[2] = {smem[1 * FF_DIRECT_BLOCK],  smem[13 * FF_DIRECT_BLOCK]};
    double jr1[2] = {smem[2 * FF_DIRECT_BLOCK],  smem[14 * FF_DIRECT_BLOCK]};
    double ji1[2] = {smem[3 * FF_DIRECT_BLOCK],  smem[15 * FF_DIRECT_BLOCK]};
    double jr2[2] = {smem[4 * FF_DIRECT_BLOCK],  smem[16 * FF_DIRECT_BLOCK]};
    double ji2[2] = {smem[5 * FF_DIRECT_BLOCK],  smem[17 * FF_DIRECT_BLOCK]};
    double mr0[2] = {smem[6 * FF_DIRECT_BLOCK],  smem[18 * FF_DIRECT_BLOCK]};
    double mi0[2] = {smem[7 * FF_DIRECT_BLOCK],  smem[19 * FF_DIRECT_BLOCK]};
    double mr1[2] = {smem[8 * FF_DIRECT_BLOCK],  smem[20 * FF_DIRECT_BLOCK]};
    double mi1[2] = {smem[9 * FF_DIRECT_BLOCK],  smem[21 * FF_DIRECT_BLOCK]};
    double mr2[2] = {smem[10 * FF_DIRECT_BLOCK], smem[22 * FF_DIRECT_BLOCK]};
    double mi2[2] = {smem[11 * FF_DIRECT_BLOCK], smem[23 * FF_DIRECT_BLOCK]};

    double fre[6], fim[6];
    double pf_re = k_im / (4.0 * M_PI);
    double pf_im = -k_re / (4.0 * M_PI);
    #pragma unroll
    for (int c = 0; c < 2; c++) {
        double rdotJ_re = rx * jr0[c] + ry * jr1[c] + rz * jr2[c];
        double rdotJ_im = rx * ji0[c] + ry * ji1[c] + rz * ji2[c];
        double jp_re0 = jr0[c] - rx * rdotJ_re, jp_im0 = ji0[c] - rx * rdotJ_im;
        double jp_re1 = jr1[c] - ry * rdotJ_re, jp_im1 = ji1[c] - ry * rdotJ_im;
        double jp_re2 = jr2[c] - rz * rdotJ_re, jp_im2 = ji2[c] - rz * rdotJ_im;

        double mc_re0 = ry * mr2[c] - rz * mr1[c], mc_im0 = ry * mi2[c] - rz * mi1[c];
        double mc_re1 = rz * mr0[c] - rx * mr2[c], mc_im1 = rz * mi0[c] - rx * mi2[c];
        double mc_re2 = rx * mr1[c] - ry * mr0[c], mc_im2 = rx * mi1[c] - ry * mi0[c];

        double eJ = j_scale * eta_ext;
        double v_re0 = eJ * jp_re0 + m_sign * mc_re0;
        double v_im0 = eJ * jp_im0 + m_sign * mc_im0;
        double v_re1 = eJ * jp_re1 + m_sign * mc_re1;
        double v_im1 = eJ * jp_im1 + m_sign * mc_im1;
        double v_re2 = eJ * jp_re2 + m_sign * mc_re2;
        double v_im2 = eJ * jp_im2 + m_sign * mc_im2;

        int o = 3 * c;
        fre[o]     = pf_re * v_re0 - pf_im * v_im0;
        fim[o]     = pf_re * v_im0 + pf_im * v_re0;
        fre[o + 1] = pf_re * v_re1 - pf_im * v_im1;
        fim[o + 1] = pf_re * v_im1 + pf_im * v_re1;
        fre[o + 2] = pf_re * v_re2 - pf_im * v_im2;
        fim[o + 2] = pf_re * v_im2 + pf_im * v_re2;
    }

    double fpp_re = fre[0] * epx + fre[1] * epy + fre[2] * epz;
    double fpp_im = fim[0] * epx + fim[1] * epy + fim[2] * epz;
    double fpq_re = fre[0] * eqx + fre[1] * eqy + fre[2] * eqz;
    double fpq_im = fim[0] * eqx + fim[1] * eqy + fim[2] * eqz;
    double fqp_re = fre[3] * epx + fre[4] * epy + fre[5] * epz;
    double fqp_im = fim[3] * epx + fim[4] * epy + fim[5] * epz;
    double fqq_re = fre[3] * eqx + fre[4] * eqy + fre[5] * eqz;
    double fqq_im = fim[3] * eqx + fim[4] * eqy + fim[5] * eqz;

    double S1r, S1i, S2r, S2i, S3r, S3i, S4r, S4i;
    cmul(ik_re, ik_im, fqq_re, fqq_im, S1r, S1i);
    cmul(ik_re, ik_im, fpp_re, fpp_im, S2r, S2i);
    cmul(ik_re, ik_im, fqp_re, fqp_im, S3r, S3i);
    cmul(ik_re, ik_im, fpq_re, fpq_im, S4r, S4i);

    double as1 = cnorm2(S1r, S1i), as2 = cnorm2(S2r, S2i);
    double as3 = cnorm2(S3r, S3i), as4 = cnorm2(S4r, S4i);
    double s23r, s23i, s14r, s14i, s24r, s24i, s13r, s13i;
    double s12r, s12i, s34r, s34i, s21r, s21i, s42r, s42i, s43r, s43i;
    c_mul_conj(S2r, S2i, S3r, S3i, s23r, s23i);
    c_mul_conj(S1r, S1i, S4r, S4i, s14r, s14i);
    c_mul_conj(S2r, S2i, S4r, S4i, s24r, s24i);
    c_mul_conj(S1r, S1i, S3r, S3i, s13r, s13i);
    c_mul_conj(S1r, S1i, S2r, S2i, s12r, s12i);
    c_mul_conj(S3r, S3i, S4r, S4i, s34r, s34i);
    c_mul_conj(S2r, S2i, S1r, S1i, s21r, s21i);
    c_mul_conj(S4r, S4i, S2r, S2i, s42r, s42i);
    c_mul_conj(S4r, S4i, S3r, S3i, s43r, s43i);

    double scale = (weights_by_base ? weights[base_idx] / (double)alpha_avg : weights[sample_idx]) * inv_k2;
    int t = dir_idx;
    int group = (partial_groups > 1) ? (sample_idx % partial_groups) : 0;
    double* M_base = M_accum + (size_t)group * 16 * ndir;
    #define MADD_ALPHA(i,j,val) atomicAdd(&M_base[((i)*4+(j))*ndir + t], scale * (val))
    MADD_ALPHA(0,0, 0.5 * (as1 + as2 + as3 + as4));
    MADD_ALPHA(0,1, 0.5 * (as2 - as1 + as4 - as3));
    MADD_ALPHA(0,2, s23r + s14r);
    MADD_ALPHA(0,3, s23i - s14i);
    MADD_ALPHA(1,0, 0.5 * (as2 - as1 - as4 + as3));
    MADD_ALPHA(1,1, 0.5 * (as2 + as1 - as4 - as3));
    MADD_ALPHA(1,2, s23r - s14r);
    MADD_ALPHA(1,3, s23i + s14i);
    MADD_ALPHA(2,0, s24r + s13r);
    MADD_ALPHA(2,1, s24r - s13r);
    MADD_ALPHA(2,2, s12r + s34r);
    MADD_ALPHA(2,3, s21i + s43i);
    MADD_ALPHA(3,0, s42i + s13i);
    MADD_ALPHA(3,1, s42i - s13i);
    MADD_ALPHA(3,2, s12i - s34i);
    MADD_ALPHA(3,3, s12r - s34r);
    #undef MADD_ALPHA
}

__global__ void reduce_mueller_partials_kernel(
    const double* __restrict__ partial,
    int groups, int ndir,
    double* __restrict__ accum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 16 * ndir;
    if (idx >= total) return;
    double sum = 0.0;
    for (int g = 0; g < groups; g++)
        sum += partial[(size_t)g * total + idx];
    accum[idx] += sum;
}

__global__ void alpha_geometry_kernel(
    const double* __restrict__ RT_mats,
    const double* __restrict__ rhat_lab,
    const double* __restrict__ etheta_lab,
    double ephi_x, double ephi_y, double ephi_z,
    const double* __restrict__ alpha_cos,
    const double* __restrict__ alpha_sin,
    int n_samples, int alpha_avg, int ndir,
    double* __restrict__ r_hats,
    double* __restrict__ e_par,
    double* __restrict__ e_perp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_samples * ndir;
    if (idx >= total) return;

    int sample_idx = idx / ndir;
    int dir_idx = idx - sample_idx * ndir;
    int base_idx = sample_idx / alpha_avg;
    int alpha_idx = sample_idx - base_idx * alpha_avg;
    double ca = alpha_cos[alpha_idx];
    double sa = alpha_sin[alpha_idx];

    const double* RT = RT_mats + base_idx * 9;
    const double* rh0 = rhat_lab + dir_idx * 3;
    const double* ep0 = etheta_lab + dir_idx * 3;

    double x = ca * rh0[0] + sa * rh0[1];
    double y = -sa * rh0[0] + ca * rh0[1];
    double z = rh0[2];
    int out = idx * 3;
    r_hats[out]     = RT[0] * x + RT[1] * y + RT[2] * z;
    r_hats[out + 1] = RT[3] * x + RT[4] * y + RT[5] * z;
    r_hats[out + 2] = RT[6] * x + RT[7] * y + RT[8] * z;

    x = ca * ep0[0] + sa * ep0[1];
    y = -sa * ep0[0] + ca * ep0[1];
    z = ep0[2];
    e_par[out]     = RT[0] * x + RT[1] * y + RT[2] * z;
    e_par[out + 1] = RT[3] * x + RT[4] * y + RT[5] * z;
    e_par[out + 2] = RT[6] * x + RT[7] * y + RT[8] * z;

    x = ca * ephi_x + sa * ephi_y;
    y = -sa * ephi_x + ca * ephi_y;
    z = ephi_z;
    e_perp[out]     = RT[0] * x + RT[1] * y + RT[2] * z;
    e_perp[out + 1] = RT[3] * x + RT[4] * y + RT[5] * z;
    e_perp[out + 2] = RT[6] * x + RT[7] * y + RT[8] * z;
}


FFBatchWorkspace::FFBatchWorkspace()
    : d_cJ_re(0), d_cJ_im(0), d_cM_re(0), d_cM_im(0),
      d_cJ_z(0), d_cM_z(0),
      d_r_hats(0), d_RT(0), d_rhat_lab(0), d_etheta_lab(0), d_Fv_re(0), d_Fv_im(0),
      d_e_par(0), d_e_perp(0), d_weights(0),
      d_alpha_cos(0), d_alpha_sin(0), d_M_accum(0), d_M_partial(0),
      h_cJ_re(0), h_cJ_im(0), h_cM_re(0), h_cM_im(0), h_fv_re(0), h_fv_im(0), h_M_accum(0),
      stream(0),
      h_cJ_re_pinned(false), h_cJ_im_pinned(false), h_cM_re_pinned(false), h_cM_im_pinned(false),
      h_fv_re_pinned(false), h_fv_im_pinned(false), h_mueller_pinned(false),
      cap_coeffs(0), cap_coeffs_z(0), cap_rhat(0), cap_rt(0), cap_lab_dirs(0), cap_fv(0),
      cap_evec(0), cap_weight(0), cap_alpha(0), cap_mueller(0), cap_mueller_partial(0),
      cap_host_coeffs(0), cap_host_fv(0), cap_host_mueller(0),
      cached_lab_dirs(0), cached_alpha(0)
{
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}

void FFBatchWorkspace::reserve(int total_coeffs, int total_rhat, int total_fv)
{
    if (total_coeffs > cap_coeffs) {
        cudaFree(d_cJ_re); cudaFree(d_cJ_im);
        cudaFree(d_cM_re); cudaFree(d_cM_im);
        CUDA_CHECK(cudaMalloc(&d_cJ_re, total_coeffs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cJ_im, total_coeffs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cM_re, total_coeffs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cM_im, total_coeffs * sizeof(double)));
        cap_coeffs = total_coeffs;
    }
    if (total_rhat > cap_rhat) {
        cudaFree(d_r_hats);
        CUDA_CHECK(cudaMalloc(&d_r_hats, total_rhat * sizeof(double)));
        cap_rhat = total_rhat;
    }
    if (total_fv > cap_fv) {
        cudaFree(d_Fv_re); cudaFree(d_Fv_im);
        CUDA_CHECK(cudaMalloc(&d_Fv_re, total_fv * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Fv_im, total_fv * sizeof(double)));
        cap_fv = total_fv;
    }
}

void FFBatchWorkspace::reserve_host_coeffs(int total_coeffs)
{
    if (total_coeffs <= cap_host_coeffs)
        return;
    release_host_double_buffer(h_cJ_re, h_cJ_re_pinned);
    release_host_double_buffer(h_cJ_im, h_cJ_im_pinned);
    release_host_double_buffer(h_cM_re, h_cM_re_pinned);
    release_host_double_buffer(h_cM_im, h_cM_im_pinned);
    allocate_host_double_buffer(h_cJ_re, h_cJ_re_pinned, total_coeffs, "far-field J real pack");
    allocate_host_double_buffer(h_cJ_im, h_cJ_im_pinned, total_coeffs, "far-field J imag pack");
    allocate_host_double_buffer(h_cM_re, h_cM_re_pinned, total_coeffs, "far-field M real pack");
    allocate_host_double_buffer(h_cM_im, h_cM_im_pinned, total_coeffs, "far-field M imag pack");
    cap_host_coeffs = total_coeffs;
}

void FFBatchWorkspace::reserve_host_fv(int total_fv)
{
    if (total_fv <= cap_host_fv)
        return;
    release_host_double_buffer(h_fv_re, h_fv_re_pinned);
    release_host_double_buffer(h_fv_im, h_fv_im_pinned);
    allocate_host_double_buffer(h_fv_re, h_fv_re_pinned, total_fv, "far-field real output");
    allocate_host_double_buffer(h_fv_im, h_fv_im_pinned, total_fv, "far-field imag output");
    cap_host_fv = total_fv;
}

void FFBatchWorkspace::reserve_host_mueller(int total_mueller)
{
    if (total_mueller <= cap_host_mueller)
        return;
    release_host_double_buffer(h_M_accum, h_mueller_pinned);
    bool pinned = false;
    allocate_host_double_buffer(h_M_accum, pinned, total_mueller, "far-field Mueller output");
    h_mueller_pinned = pinned;
    cap_host_mueller = total_mueller;
}

void FFBatchWorkspace::reserve_alpha(int n_alpha)
{
    if (n_alpha <= cap_alpha)
        return;
    cudaFree(d_alpha_cos);
    cudaFree(d_alpha_sin);
    CUDA_CHECK(cudaMalloc(&d_alpha_cos, (size_t)n_alpha * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha_sin, (size_t)n_alpha * sizeof(double)));
    cap_alpha = n_alpha;
    cached_alpha = 0;
}

void FFBatchWorkspace::reserve_mueller_accum(int ndir)
{
    int total_mueller = 16 * ndir;
    if (total_mueller > cap_mueller) {
        cudaFree(d_M_accum);
        CUDA_CHECK(cudaMalloc(&d_M_accum, (size_t)total_mueller * sizeof(double)));
        cap_mueller = total_mueller;
    }
}

void FFBatchWorkspace::reserve_mueller(int n_orient, int ndir)
{
    int total_evec = n_orient * ndir * 3;
    if (total_evec > cap_evec) {
        cudaFree(d_e_par);
        cudaFree(d_e_perp);
        CUDA_CHECK(cudaMalloc(&d_e_par, total_evec * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_e_perp, total_evec * sizeof(double)));
        cap_evec = total_evec;
    }
    if (n_orient > cap_weight) {
        cudaFree(d_weights);
        CUDA_CHECK(cudaMalloc(&d_weights, n_orient * sizeof(double)));
        cap_weight = n_orient;
    }
    reserve_mueller_accum(ndir);
}

void FFBatchWorkspace::reserve_mueller_partials(int groups, int ndir)
{
    int total_mueller = 16 * ndir;
    int total_partial = std::max(1, groups) * total_mueller;
    if (total_partial > cap_mueller_partial) {
        cudaFree(d_M_partial);
        CUDA_CHECK(cudaMalloc(&d_M_partial, (size_t)total_partial * sizeof(double)));
        cap_mueller_partial = total_partial;
    }
}

void FFBatchWorkspace::zero_mueller(int ndir)
{
    reserve_mueller_accum(ndir);
    CUDA_CHECK(cudaMemsetAsync(d_M_accum, 0, 16 * ndir * sizeof(double), stream));
}

void FFBatchWorkspace::zero_mueller_partials(int groups, int ndir)
{
    reserve_mueller_partials(groups, ndir);
    CUDA_CHECK(cudaMemsetAsync(d_M_partial, 0, (size_t)std::max(1, groups) * 16 * ndir * sizeof(double), stream));
}

void FFBatchWorkspace::download_mueller(double* M_out, int ndir)
{
    int total_mueller = 16 * ndir;
    reserve_host_mueller(total_mueller);
    CUDA_CHECK(cudaMemcpyAsync(h_M_accum, d_M_accum,
                               total_mueller * sizeof(double), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < total_mueller; i++)
        M_out[i] = h_M_accum[i];
}

void FFBatchWorkspace::free()
{
    if (stream)
        cudaStreamSynchronize(stream);
    cudaFree(d_cJ_re); cudaFree(d_cJ_im);
    cudaFree(d_cM_re); cudaFree(d_cM_im);
    cudaFree(d_cJ_z); cudaFree(d_cM_z);
    cudaFree(d_r_hats);
    cudaFree(d_RT); cudaFree(d_rhat_lab); cudaFree(d_etheta_lab);
    cudaFree(d_Fv_re); cudaFree(d_Fv_im);
    cudaFree(d_e_par); cudaFree(d_e_perp);
    cudaFree(d_weights);
    cudaFree(d_alpha_cos); cudaFree(d_alpha_sin);
    cudaFree(d_M_accum);
    cudaFree(d_M_partial);
    release_host_double_buffer(h_cJ_re, h_cJ_re_pinned);
    release_host_double_buffer(h_cJ_im, h_cJ_im_pinned);
    release_host_double_buffer(h_cM_re, h_cM_re_pinned);
    release_host_double_buffer(h_cM_im, h_cM_im_pinned);
    release_host_double_buffer(h_fv_re, h_fv_re_pinned);
    release_host_double_buffer(h_fv_im, h_fv_im_pinned);
    release_host_double_buffer(h_M_accum, h_mueller_pinned);
    if (stream) {
        cudaStreamDestroy(stream);
        stream = 0;
    }
    d_cJ_re = d_cJ_im = d_cM_re = d_cM_im = 0;
    d_cJ_z = d_cM_z = 0;
    d_r_hats = d_RT = d_rhat_lab = d_etheta_lab = d_Fv_re = d_Fv_im = 0;
    d_e_par = d_e_perp = d_weights = d_alpha_cos = d_alpha_sin = d_M_accum = d_M_partial = 0;
    h_cJ_re = h_cJ_im = h_cM_re = h_cM_im = h_fv_re = h_fv_im = h_M_accum = 0;
    cap_coeffs = cap_coeffs_z = cap_rhat = cap_rt = cap_lab_dirs = cap_fv = 0;
    cap_evec = cap_weight = cap_alpha = cap_mueller = cap_mueller_partial = 0;
    cap_host_coeffs = cap_host_fv = cap_host_mueller = 0;
    cached_lab_dirs = cached_alpha = 0;
}

FFBatchWorkspace::~FFBatchWorkspace()
{
    free();
}


// ======== Host wrapper: batched GPU far-field ========

void compute_farfield_batch_cuda_ws(
    const FFCacheGPU& gpu_cache,
    FFBatchWorkspace& workspace,
    const std::complex<double>* coeffs_J,  // (n_calls * N) host
    const std::complex<double>* coeffs_M,  // (n_calls * N) host
    const double* r_hats,                  // (n_orient * ndir * 3) host
    std::complex<double> k_ext, double eta_ext,
    int n_calls, int n_orient, int ndir,
    std::complex<double>* Fv_out)          // (n_calls * ndir * 3) host
{
    Timer timer;
    int N = gpu_cache.N;
    int total_coeffs = n_calls * N;
    int total_rhat = n_orient * ndir * 3;
    int total_fv = n_calls * ndir * 3;
    workspace.reserve(total_coeffs, total_rhat, total_fv);
    workspace.reserve_host_fv(total_fv);

    upload_complex_coeffs(workspace, coeffs_J, coeffs_M, total_coeffs);
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_r_hats, r_hats, total_rhat * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));

    // Launch kernel
    dim3 grid(n_calls, ndir);
    dim3 block(FF_BLOCK);

    // Shared memory: 12 * FF_BLOCK doubles
    size_t smem_size = 12 * FF_BLOCK * sizeof(double);
    const double m_sign = farfield_m_scale();
    const double j_scale = farfield_j_scale();
    const double phase_sign = farfield_phase_sign();

    farfield_batch_kernel<<<grid, block, smem_size, workspace.stream>>>(
        gpu_cache.d_qpts, gpu_cache.d_fvals, gpu_cache.d_jw,
        workspace.d_cJ_re, workspace.d_cJ_im, workspace.d_cM_re, workspace.d_cM_im,
        workspace.d_r_hats,
        k_ext.real(), k_ext.imag(), eta_ext, j_scale, m_sign, phase_sign,
        N, gpu_cache.Nq, n_calls, n_orient, ndir,
        workspace.d_Fv_re, workspace.d_Fv_im);

    CUDA_CHECK(cudaGetLastError());

    // Download results
    CUDA_CHECK(cudaMemcpyAsync(workspace.h_fv_re, workspace.d_Fv_re, total_fv * sizeof(double),
                               cudaMemcpyDeviceToHost, workspace.stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.h_fv_im, workspace.d_Fv_im, total_fv * sizeof(double),
                               cudaMemcpyDeviceToHost, workspace.stream));
    CUDA_CHECK(cudaStreamSynchronize(workspace.stream));

    // Pack into complex output
    for (int i = 0; i < total_fv; i++)
        Fv_out[i] = std::complex<double>(workspace.h_fv_re[i], workspace.h_fv_im[i]);

    if (bem_env_flag_enabled("BEM_FF_VERBOSE"))
        printf("  GPU far-field (%d calls x %d dirs): %.2fs\n",
               n_calls, ndir, timer.elapsed_s());
}

void accumulate_farfield_mueller_batch_cuda_ws(
    const FFCacheGPU& gpu_cache,
    FFBatchWorkspace& workspace,
    const std::complex<double>* coeffs_J,
    const std::complex<double>* coeffs_M,
    const double* r_hats,
    const double* e_par,
    const double* e_perp,
    const double* weights,
    std::complex<double> k_ext, double eta_ext,
    int n_calls, int n_orient, int ndir)
{
    Timer timer;
    int N = gpu_cache.N;
    int total_coeffs = n_calls * N;
    int total_rhat = n_orient * ndir * 3;
    int total_fv = n_calls * ndir * 3;
    int total_evec = n_orient * ndir * 3;
    workspace.reserve(total_coeffs, total_rhat, total_fv);
    workspace.reserve_mueller(n_orient, ndir);

    upload_complex_coeffs(workspace, coeffs_J, coeffs_M, total_coeffs);
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_r_hats, r_hats, total_rhat * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_e_par, e_par, total_evec * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_e_perp, e_perp, total_evec * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_weights, weights, n_orient * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));

    dim3 grid(n_calls, ndir);
    dim3 block(FF_BLOCK);
    size_t smem_size = 12 * FF_BLOCK * sizeof(double);
    const double m_sign = farfield_m_scale();
    const double j_scale = farfield_j_scale();
    const double phase_sign = farfield_phase_sign();

    farfield_batch_kernel<<<grid, block, smem_size, workspace.stream>>>(
        gpu_cache.d_qpts, gpu_cache.d_fvals, gpu_cache.d_jw,
        workspace.d_cJ_re, workspace.d_cJ_im, workspace.d_cM_re, workspace.d_cM_im,
        workspace.d_r_hats,
        k_ext.real(), k_ext.imag(), eta_ext, j_scale, m_sign, phase_sign,
        N, gpu_cache.Nq, n_calls, n_orient, ndir,
        workspace.d_Fv_re, workspace.d_Fv_im);
    CUDA_CHECK(cudaGetLastError());

    dim3 mblock(128);
    dim3 mgrid(n_orient, (ndir + (int)mblock.x - 1) / (int)mblock.x);
    std::complex<double> ik_val = std::complex<double>(0, -1) * k_ext;
    double inv_k2 = 1.0;
    mueller_accum_kernel<<<mgrid, mblock, 0, workspace.stream>>>(
        workspace.d_Fv_re, workspace.d_Fv_im,
        workspace.d_e_par, workspace.d_e_perp, workspace.d_weights,
        ik_val.real(), ik_val.imag(), inv_k2,
        n_orient, ndir, workspace.d_M_accum);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(workspace.stream));

    if (bem_env_flag_enabled("BEM_FF_VERBOSE"))
        printf("  GPU far-field+Mueller (%d calls x %d dirs): %.2fs\n",
               n_calls, ndir, timer.elapsed_s());
}

void accumulate_farfield_mueller_direct_cuda_ws(
    const FFCacheGPU& gpu_cache,
    FFBatchWorkspace& workspace,
    const std::complex<double>* coeffs_J,
    const std::complex<double>* coeffs_M,
    const double* r_hats,
    const double* e_par,
    const double* e_perp,
    const double* weights,
    std::complex<double> k_ext, double eta_ext,
    int n_calls, int n_orient, int ndir)
{
    Timer timer;
    int N = gpu_cache.N;
    if (n_calls != 2 * n_orient) {
        accumulate_farfield_mueller_batch_cuda_ws(
            gpu_cache, workspace, coeffs_J, coeffs_M, r_hats,
            e_par, e_perp, weights, k_ext, eta_ext,
            n_calls, n_orient, ndir);
        return;
    }

    int total_coeffs = n_calls * N;
    int total_rhat = n_orient * ndir * 3;
    int total_evec = n_orient * ndir * 3;
    workspace.reserve(total_coeffs, total_rhat, 0);
    workspace.reserve_mueller(n_orient, ndir);

    upload_complex_coeffs(workspace, coeffs_J, coeffs_M, total_coeffs);
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_r_hats, r_hats, total_rhat * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_e_par, e_par, total_evec * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_e_perp, e_perp, total_evec * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_weights, weights, n_orient * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));

    dim3 grid(n_orient, ndir);
    dim3 block(FF_DIRECT_BLOCK);
    std::complex<double> ik_val = std::complex<double>(0, -1) * k_ext;
    double inv_k2 = 1.0;
    const double m_sign = farfield_m_scale();
    const double j_scale = farfield_j_scale();
    const double phase_sign = farfield_phase_sign();
    farfield_mueller_direct_kernel<<<grid, block, 0, workspace.stream>>>(
        gpu_cache.d_qpts, gpu_cache.d_fvals, gpu_cache.d_jw,
        workspace.d_cJ_re, workspace.d_cJ_im, workspace.d_cM_re, workspace.d_cM_im,
        workspace.d_r_hats, workspace.d_e_par, workspace.d_e_perp, workspace.d_weights,
        k_ext.real(), k_ext.imag(), eta_ext, j_scale, m_sign, phase_sign,
        ik_val.real(), ik_val.imag(), inv_k2,
        N, gpu_cache.Nq, n_orient, ndir,
        workspace.d_M_accum);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(workspace.stream));

    if (bem_env_flag_enabled("BEM_FF_VERBOSE"))
        printf("  GPU direct far-field+Mueller (%d samples x %d dirs): %.2fs\n",
               n_orient, ndir, timer.elapsed_s());
}

void accumulate_farfield_mueller_alpha_cuda_ws(
    const FFCacheGPU& gpu_cache,
    FFBatchWorkspace& workspace,
    const std::complex<double>* coeffs_J,
    const std::complex<double>* coeffs_M,
    const double* r_hats,
    const double* e_par,
    const double* e_perp,
    const double* weights,
    const double* alpha_cos,
    const double* alpha_sin,
    std::complex<double> k_ext, double eta_ext,
    int n_base_orient, int alpha_avg, int ndir)
{
    Timer timer;
    int N = gpu_cache.N;
    int n_samples = n_base_orient * alpha_avg;
    if (n_base_orient <= 0 || alpha_avg <= 0 || n_samples <= 0)
        return;
    if (alpha_avg == 1) {
        accumulate_farfield_mueller_direct_cuda_ws(
            gpu_cache, workspace, coeffs_J, coeffs_M, r_hats,
            e_par, e_perp, weights, k_ext, eta_ext,
            n_base_orient * 2, n_base_orient, ndir);
        return;
    }

    int total_coeffs = 2 * n_base_orient * N;
    int total_rhat = n_samples * ndir * 3;
    int total_evec = n_samples * ndir * 3;
    workspace.reserve(total_coeffs, total_rhat, 0);
    workspace.reserve_mueller(n_samples, ndir);
    workspace.reserve_alpha(alpha_avg);
    int partial_groups = 16;
    partial_groups = std::max(1, bem_env_int("BEM_FF_PARTIAL_GROUPS", partial_groups));
    partial_groups = std::max(1, std::min(partial_groups, n_samples));
    double* M_target = workspace.d_M_accum;
    if (partial_groups > 1) {
        workspace.zero_mueller_partials(partial_groups, ndir);
        M_target = workspace.d_M_partial;
    }

    upload_complex_coeffs(workspace, coeffs_J, coeffs_M, total_coeffs);
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_r_hats, r_hats, total_rhat * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_e_par, e_par, total_evec * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_e_perp, e_perp, total_evec * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_weights, weights, n_samples * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));
    if (workspace.cached_alpha != alpha_avg) {
        CUDA_CHECK(cudaMemcpyAsync(workspace.d_alpha_cos, alpha_cos, alpha_avg * sizeof(double),
                                   cudaMemcpyHostToDevice, workspace.stream));
        CUDA_CHECK(cudaMemcpyAsync(workspace.d_alpha_sin, alpha_sin, alpha_avg * sizeof(double),
                                   cudaMemcpyHostToDevice, workspace.stream));
        workspace.cached_alpha = alpha_avg;
    }

    dim3 grid(n_samples, ndir);
    dim3 block(FF_DIRECT_BLOCK);
    std::complex<double> ik_val = std::complex<double>(0, -1) * k_ext;
    double inv_k2 = 1.0;
    const double m_sign = farfield_m_scale();
    const double j_scale = farfield_j_scale();
    const double phase_sign = farfield_phase_sign();
    farfield_mueller_alpha_kernel<<<grid, block, 0, workspace.stream>>>(
        gpu_cache.d_qpts, gpu_cache.d_fvals, gpu_cache.d_jw,
        workspace.d_cJ_re, workspace.d_cJ_im, workspace.d_cM_re, workspace.d_cM_im,
        workspace.d_r_hats, workspace.d_e_par, workspace.d_e_perp,
        0, 0, 0, 0.0, 0.0, 0.0, workspace.d_weights,
        workspace.d_alpha_cos, workspace.d_alpha_sin,
        k_ext.real(), k_ext.imag(), eta_ext, j_scale, m_sign, phase_sign,
        ik_val.real(), ik_val.imag(), inv_k2,
        N, gpu_cache.Nq, n_samples, alpha_avg, ndir, 0, 0,
        partial_groups, M_target);
    CUDA_CHECK(cudaGetLastError());
    if (partial_groups > 1) {
        int total_mueller = 16 * ndir;
        int block_reduce = 256;
        int grid_reduce = (total_mueller + block_reduce - 1) / block_reduce;
        reduce_mueller_partials_kernel<<<grid_reduce, block_reduce, 0, workspace.stream>>>(
            workspace.d_M_partial, partial_groups, ndir, workspace.d_M_accum);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(workspace.stream));

    if (bem_env_flag_enabled("BEM_FF_VERBOSE"))
        printf("  GPU alpha direct far-field+Mueller (%d base x %d alpha x %d dirs): %.2fs\n",
               n_base_orient, alpha_avg, ndir, timer.elapsed_s());
}

void accumulate_farfield_mueller_alpha_geom_cuda_ws(
    const FFCacheGPU& gpu_cache,
    FFBatchWorkspace& workspace,
    const std::complex<double>* coeffs_J,
    const std::complex<double>* coeffs_M,
    const double* RT_mats,
    const double* rhat_lab,
    const double* etheta_lab,
    const double* ephi_lab,
    const double* weights,
    const double* alpha_cos,
    const double* alpha_sin,
    std::complex<double> k_ext, double eta_ext,
    int n_base_orient, int alpha_avg, int ndir,
    bool sync_after)
{
    Timer timer;
    int N = gpu_cache.N;
    int n_samples = n_base_orient * alpha_avg;
    if (n_base_orient <= 0 || alpha_avg <= 0 || n_samples <= 0)
        return;

    int total_coeffs = 2 * n_base_orient * N;
    int total_rt = n_base_orient * 9;
    int total_lab = ndir * 3;
    workspace.reserve(total_coeffs, 0, 0);
    workspace.reserve_mueller_accum(ndir);
    if (n_base_orient > workspace.cap_weight) {
        cudaFree(workspace.d_weights);
        CUDA_CHECK(cudaMalloc(&workspace.d_weights, (size_t)n_base_orient * sizeof(double)));
        workspace.cap_weight = n_base_orient;
    }
    workspace.reserve_alpha(alpha_avg);
    int partial_groups = 16;
    partial_groups = std::max(1, bem_env_int("BEM_FF_PARTIAL_GROUPS", partial_groups));
    partial_groups = std::max(1, std::min(partial_groups, n_samples));
    double* M_target = workspace.d_M_accum;
    if (partial_groups > 1) {
        workspace.zero_mueller_partials(partial_groups, ndir);
        M_target = workspace.d_M_partial;
    }

    if (total_rt > workspace.cap_rt) {
        cudaFree(workspace.d_RT);
        CUDA_CHECK(cudaMalloc(&workspace.d_RT, total_rt * sizeof(double)));
        workspace.cap_rt = total_rt;
    }
    if (total_lab > workspace.cap_lab_dirs) {
        cudaFree(workspace.d_rhat_lab);
        cudaFree(workspace.d_etheta_lab);
        CUDA_CHECK(cudaMalloc(&workspace.d_rhat_lab, total_lab * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&workspace.d_etheta_lab, total_lab * sizeof(double)));
        workspace.cap_lab_dirs = total_lab;
        workspace.cached_lab_dirs = 0;
    }

    upload_complex_coeffs(workspace, coeffs_J, coeffs_M, total_coeffs);
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_RT, RT_mats, total_rt * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));
    if (workspace.cached_lab_dirs != ndir) {
        CUDA_CHECK(cudaMemcpyAsync(workspace.d_rhat_lab, rhat_lab, total_lab * sizeof(double),
                                   cudaMemcpyHostToDevice, workspace.stream));
        CUDA_CHECK(cudaMemcpyAsync(workspace.d_etheta_lab, etheta_lab, total_lab * sizeof(double),
                                   cudaMemcpyHostToDevice, workspace.stream));
        workspace.cached_lab_dirs = ndir;
    }
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_weights, weights, n_base_orient * sizeof(double),
                               cudaMemcpyHostToDevice, workspace.stream));
    if (workspace.cached_alpha != alpha_avg) {
        CUDA_CHECK(cudaMemcpyAsync(workspace.d_alpha_cos, alpha_cos, alpha_avg * sizeof(double),
                                   cudaMemcpyHostToDevice, workspace.stream));
        CUDA_CHECK(cudaMemcpyAsync(workspace.d_alpha_sin, alpha_sin, alpha_avg * sizeof(double),
                                   cudaMemcpyHostToDevice, workspace.stream));
        workspace.cached_alpha = alpha_avg;
    }

    dim3 grid(n_samples, ndir);
    dim3 block(FF_DIRECT_BLOCK);
    std::complex<double> ik_val = std::complex<double>(0, -1) * k_ext;
    double inv_k2 = 1.0;
    const double m_sign = farfield_m_scale();
    const double j_scale = farfield_j_scale();
    const double phase_sign = farfield_phase_sign();
    farfield_mueller_alpha_kernel<<<grid, block, 0, workspace.stream>>>(
        gpu_cache.d_qpts, gpu_cache.d_fvals, gpu_cache.d_jw,
        workspace.d_cJ_re, workspace.d_cJ_im, workspace.d_cM_re, workspace.d_cM_im,
        0, 0, 0,
        workspace.d_RT, workspace.d_rhat_lab, workspace.d_etheta_lab,
        ephi_lab[0], ephi_lab[1], ephi_lab[2],
        workspace.d_weights,
        workspace.d_alpha_cos, workspace.d_alpha_sin,
        k_ext.real(), k_ext.imag(), eta_ext, j_scale, m_sign, phase_sign,
        ik_val.real(), ik_val.imag(), inv_k2,
        N, gpu_cache.Nq, n_samples, alpha_avg, ndir, 1, 1,
        partial_groups, M_target);
    CUDA_CHECK(cudaGetLastError());
    if (partial_groups > 1) {
        int total_mueller = 16 * ndir;
        int block_reduce = 256;
        int grid_reduce = (total_mueller + block_reduce - 1) / block_reduce;
        reduce_mueller_partials_kernel<<<grid_reduce, block_reduce, 0, workspace.stream>>>(
            workspace.d_M_partial, partial_groups, ndir, workspace.d_M_accum);
        CUDA_CHECK(cudaGetLastError());
    }
    if (sync_after)
        CUDA_CHECK(cudaStreamSynchronize(workspace.stream));

    if (bem_env_flag_enabled("BEM_FF_VERBOSE"))
        printf("  GPU alpha geom+direct far-field+Mueller (%d base x %d alpha x %d dirs): %.2fs\n",
               n_base_orient, alpha_avg, ndir, timer.elapsed_s());
}

void compute_farfield_batch_cuda(
    const FFCacheGPU& gpu_cache,
    const std::complex<double>* coeffs_J,
    const std::complex<double>* coeffs_M,
    const double* r_hats,
    std::complex<double> k_ext, double eta_ext,
    int n_calls, int n_orient, int ndir,
    std::complex<double>* Fv_out)
{
    FFBatchWorkspace workspace;
    compute_farfield_batch_cuda_ws(gpu_cache, workspace, coeffs_J, coeffs_M,
                                   r_hats, k_ext, eta_ext,
                                   n_calls, n_orient, ndir, Fv_out);
}


// ======== CPU functions (kept for --single mode and fallback) ========

void compute_far_field_vec_batch_cpu(const FFCache& cache,
                                     const std::complex<double>* coeffs_J,
                                     const std::complex<double>* coeffs_M,
                                     std::complex<double> k_ext, double eta_ext,
                                     const Vec3* r_hats, int ndir,
                                     std::complex<double>* Fv_out)
{
    int N = cache.N;
    int Nq = cache.Nq;
    double k_re = k_ext.real(), k_im = k_ext.imag();
    double phase_sign = farfield_phase_sign();

    std::vector<std::complex<double>> Jt(ndir * 3, 0);
    std::vector<std::complex<double>> Mt(ndir * 3, 0);

    for (int half = 0; half < 2; half++) {
        int offset = half * N * Nq;

        for (int n = 0; n < N; n++) {
            std::vector<std::complex<double>> integ(ndir * 3, 0);
            int base = (offset + n * Nq);

            for (int q = 0; q < Nq; q++) {
                int idx = base + q;
                double rx = cache.qpts[idx*3];
                double ry = cache.qpts[idx*3+1];
                double rz = cache.qpts[idx*3+2];
                double fx = cache.fvals[idx*3];
                double fy = cache.fvals[idx*3+1];
                double fz = cache.fvals[idx*3+2];
                double w = cache.jw[idx];

                for (int d = 0; d < ndir; d++) {
                    double rdot = r_hats[d].x * rx + r_hats[d].y * ry + r_hats[d].z * rz;
                    double arg = phase_sign * k_re * rdot;
                    double ea = exp(-phase_sign * k_im * rdot);
                    double c = cos(arg) * ea;
                    double s = sin(arg) * ea;
                    double pw_re = c * w;
                    double pw_im = s * w;

                    integ[d*3]   += std::complex<double>(fx * pw_re, fx * pw_im);
                    integ[d*3+1] += std::complex<double>(fy * pw_re, fy * pw_im);
                    integ[d*3+2] += std::complex<double>(fz * pw_re, fz * pw_im);
                }
            }

            std::complex<double> cJ = coeffs_J[n];
            std::complex<double> cM = coeffs_M[n];
            for (int d = 0; d < ndir; d++) {
                Jt[d*3]   += integ[d*3]   * cJ;
                Jt[d*3+1] += integ[d*3+1] * cJ;
                Jt[d*3+2] += integ[d*3+2] * cJ;
                Mt[d*3]   += integ[d*3]   * cM;
                Mt[d*3+1] += integ[d*3+1] * cM;
                Mt[d*3+2] += integ[d*3+2] * cM;
            }
        }
    }

    std::complex<double> prefac = std::complex<double>(0, -1) * k_ext / (4.0 * M_PI);
    double sM = farfield_m_scale();
    double sJ = farfield_j_scale();

    for (int d = 0; d < ndir; d++) {
        std::complex<double> jx = Jt[d*3], jy = Jt[d*3+1], jz = Jt[d*3+2];
        std::complex<double> mx = Mt[d*3], my = Mt[d*3+1], mz = Mt[d*3+2];
        double rrx = r_hats[d].x, rry = r_hats[d].y, rrz = r_hats[d].z;

        std::complex<double> rdotJ = rrx * jx + rry * jy + rrz * jz;
        std::complex<double> jpx = jx - rrx * rdotJ;
        std::complex<double> jpy = jy - rry * rdotJ;
        std::complex<double> jpz = jz - rrz * rdotJ;

        std::complex<double> mcx = rry * mz - rrz * my;
        std::complex<double> mcy = rrz * mx - rrx * mz;
        std::complex<double> mcz = rrx * my - rry * mx;

        Fv_out[d*3]   = prefac * (sJ * eta_ext * jpx + sM * mcx);
        Fv_out[d*3+1] = prefac * (sJ * eta_ext * jpy + sM * mcy);
        Fv_out[d*3+2] = prefac * (sJ * eta_ext * jpz + sM * mcz);
    }
}


// CPU: single-orient phi=0 mode
void compute_far_field(const FFCache& cache,
                       const std::complex<double>* coeffs_J,
                       const std::complex<double>* coeffs_M,
                       std::complex<double> k_ext, double eta_ext,
                       const double* theta_arr, int ntheta,
                       bool scattering_plane_yz,
                       std::complex<double>* F_theta,
                       std::complex<double>* F_phi)
{
    std::vector<Vec3> r_hats(ntheta);
    std::vector<Vec3> theta_hats(ntheta);
    Vec3 phi_hat = scattering_plane_yz ? Vec3(1.0, 0.0, 0.0) : Vec3(0.0, 1.0, 0.0);
    for (int it = 0; it < ntheta; it++) {
        double ct = cos(theta_arr[it]), st = sin(theta_arr[it]);
        if (scattering_plane_yz) {
            r_hats[it] = Vec3(0.0, st, ct);
            theta_hats[it] = Vec3(0.0, ct, -st);
        } else {
            r_hats[it] = Vec3(st, 0.0, ct);
            theta_hats[it] = Vec3(ct, 0.0, -st);
        }
    }

    std::vector<std::complex<double>> Fv(ntheta * 3);
    compute_far_field_vec_batch_cpu(cache, coeffs_J, coeffs_M,
                                    k_ext, eta_ext, r_hats.data(), ntheta, Fv.data());

    for (int it = 0; it < ntheta; it++) {
        F_theta[it] = Fv[it*3] * theta_hats[it].x + Fv[it*3+1] * theta_hats[it].y + Fv[it*3+2] * theta_hats[it].z;
        F_phi[it]   = Fv[it*3] * phi_hat.x   + Fv[it*3+1] * phi_hat.y   + Fv[it*3+2] * phi_hat.z;
    }
}


// Mueller matrix from amplitude matrix (unchanged)
void amplitude_to_mueller(const std::complex<double>* S1,
                          const std::complex<double>* S2,
                          const std::complex<double>* S3,
                          const std::complex<double>* S4,
                          int ntheta, double* M)
{
    memset(M, 0, 16 * ntheta * sizeof(double));

    for (int t = 0; t < ntheta; t++) {
        double as1 = std::norm(S1[t]), as2 = std::norm(S2[t]);
        double as3 = std::norm(S3[t]), as4 = std::norm(S4[t]);

        std::complex<double> s2s3c = S2[t] * std::conj(S3[t]);
        std::complex<double> s1s4c = S1[t] * std::conj(S4[t]);
        std::complex<double> s2s4c = S2[t] * std::conj(S4[t]);
        std::complex<double> s1s3c = S1[t] * std::conj(S3[t]);
        std::complex<double> s1s2c = S1[t] * std::conj(S2[t]);
        std::complex<double> s3s4c = S3[t] * std::conj(S4[t]);
        std::complex<double> s2s1c = S2[t] * std::conj(S1[t]);
        std::complex<double> s4s2c = S4[t] * std::conj(S2[t]);
        std::complex<double> s4s3c = S4[t] * std::conj(S3[t]);

        #define MI(i,j) M[((i)*4+(j))*ntheta + t]
        MI(0,0) = 0.5 * (as1 + as2 + as3 + as4);
        MI(0,1) = 0.5 * (as2 - as1 + as4 - as3);
        MI(0,2) = s2s3c.real() + s1s4c.real();
        MI(0,3) = s2s3c.imag() - s1s4c.imag();

        MI(1,0) = 0.5 * (as2 - as1 - as4 + as3);
        MI(1,1) = 0.5 * (as2 + as1 - as4 - as3);
        MI(1,2) = s2s3c.real() - s1s4c.real();
        MI(1,3) = s2s3c.imag() + s1s4c.imag();

        MI(2,0) = s2s4c.real() + s1s3c.real();
        MI(2,1) = s2s4c.real() - s1s3c.real();
        MI(2,2) = s1s2c.real() + s3s4c.real();
        MI(2,3) = s2s1c.imag() + s4s3c.imag();

        MI(3,0) = s4s2c.imag() + s1s3c.imag();
        MI(3,1) = s4s2c.imag() - s1s3c.imag();
        MI(3,2) = s1s2c.imag() - s3s4c.imag();
        MI(3,3) = s1s2c.real() - s3s4c.real();
        #undef MI
    }
}
