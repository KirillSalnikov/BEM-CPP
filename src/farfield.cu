#include "farfield.h"
#include "quadrature.h"
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

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

__global__ void farfield_batch_kernel(
    const double* __restrict__ qpts,    // (total, 3)
    const double* __restrict__ fvals,   // (total, 3)
    const double* __restrict__ jw,      // (total)
    const double* __restrict__ coeffs_J_re, // (n_calls, N) re
    const double* __restrict__ coeffs_J_im, // (n_calls, N) im
    const double* __restrict__ coeffs_M_re, // (n_calls, N) re
    const double* __restrict__ coeffs_M_im, // (n_calls, N) im
    const double* __restrict__ r_hats,      // (n_orient, ndir, 3)
    double k_re, double k_im, double eta_ext,
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

        // Phase: exp(-ik * r_hat . r')
        double rdot = rx * px + ry * py + rz * pz;
        double arg = k_re * rdot;
        double ea = exp(k_im * rdot);
        double c = cos(arg) * ea;     // Re(phase)
        double s = -sin(arg) * ea;    // Im(phase)

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

        // F = prefac * (eta * Jp + sM * Mc)
        // prefac = -ik/(4pi), sM = -1
        // prefac = (-i) * (k_re + i*k_im) / (4pi)
        //        = (k_im - i*k_re) / (4pi)
        double inv4pi = 1.0 / (4.0 * M_PI);
        double pf_re = k_im * inv4pi;
        double pf_im = -k_re * inv4pi;
        double sM = -1.0;

        // v = eta*Jp + sM*Mc for each component
        double v_re0 = eta_ext * jp_re0 + sM * mc_re0;
        double v_im0 = eta_ext * jp_im0 + sM * mc_im0;
        double v_re1 = eta_ext * jp_re1 + sM * mc_re1;
        double v_im1 = eta_ext * jp_im1 + sM * mc_im1;
        double v_re2 = eta_ext * jp_re2 + sM * mc_re2;
        double v_im2 = eta_ext * jp_im2 + sM * mc_im2;

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


// ======== Host wrapper: batched GPU far-field ========

void compute_farfield_batch_cuda(
    const FFCacheGPU& gpu_cache,
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

    // Split complex coefficients into separate re/im arrays for GPU
    std::vector<double> cJ_re(total_coeffs), cJ_im(total_coeffs);
    std::vector<double> cM_re(total_coeffs), cM_im(total_coeffs);
    for (int i = 0; i < total_coeffs; i++) {
        cJ_re[i] = coeffs_J[i].real();
        cJ_im[i] = coeffs_J[i].imag();
        cM_re[i] = coeffs_M[i].real();
        cM_im[i] = coeffs_M[i].imag();
    }

    // Allocate device memory
    double *d_cJ_re, *d_cJ_im, *d_cM_re, *d_cM_im;
    double *d_r_hats;
    double *d_Fv_re, *d_Fv_im;

    CUDA_CHECK(cudaMalloc(&d_cJ_re, total_coeffs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cJ_im, total_coeffs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cM_re, total_coeffs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cM_im, total_coeffs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_hats, total_rhat * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Fv_re, total_fv * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Fv_im, total_fv * sizeof(double)));

    // Upload
    CUDA_CHECK(cudaMemcpy(d_cJ_re, cJ_re.data(), total_coeffs * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cJ_im, cJ_im.data(), total_coeffs * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cM_re, cM_re.data(), total_coeffs * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cM_im, cM_im.data(), total_coeffs * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r_hats, r_hats, total_rhat * sizeof(double), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 grid(n_calls, ndir);
    dim3 block(FF_BLOCK);

    // Shared memory: 12 * FF_BLOCK doubles
    size_t smem_size = 12 * FF_BLOCK * sizeof(double);

    farfield_batch_kernel<<<grid, block, smem_size>>>(
        gpu_cache.d_qpts, gpu_cache.d_fvals, gpu_cache.d_jw,
        d_cJ_re, d_cJ_im, d_cM_re, d_cM_im,
        d_r_hats,
        k_ext.real(), k_ext.imag(), eta_ext,
        N, gpu_cache.Nq, n_calls, n_orient, ndir,
        d_Fv_re, d_Fv_im);

    CUDA_CHECK(cudaGetLastError());

    // Download results
    std::vector<double> fv_re(total_fv), fv_im(total_fv);
    CUDA_CHECK(cudaMemcpy(fv_re.data(), d_Fv_re, total_fv * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fv_im.data(), d_Fv_im, total_fv * sizeof(double), cudaMemcpyDeviceToHost));

    // Pack into complex output
    for (int i = 0; i < total_fv; i++)
        Fv_out[i] = std::complex<double>(fv_re[i], fv_im[i]);

    // Free device memory (except gpu_cache which persists)
    cudaFree(d_cJ_re); cudaFree(d_cJ_im);
    cudaFree(d_cM_re); cudaFree(d_cM_im);
    cudaFree(d_r_hats);
    cudaFree(d_Fv_re); cudaFree(d_Fv_im);

    printf("  GPU far-field (%d calls x %d dirs): %.2fs\n",
           n_calls, ndir, timer.elapsed_s());
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
                    double arg = k_re * rdot;
                    double ea = exp(k_im * rdot);
                    double c = cos(arg) * ea;
                    double s = -sin(arg) * ea;
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
    double sM = -1.0;

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

        Fv_out[d*3]   = prefac * (eta_ext * jpx + sM * mcx);
        Fv_out[d*3+1] = prefac * (eta_ext * jpy + sM * mcy);
        Fv_out[d*3+2] = prefac * (eta_ext * jpz + sM * mcz);
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
