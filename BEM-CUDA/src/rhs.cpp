#include "rhs.h"
#include "farfield.h"
#include "gpu_select.h"
#include "quadrature.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>

namespace {
double rhs_h_sign()
{
    double s = bem_env_double("BEM_RHS_H_SIGN", 1.0);
    return (s < 0.0) ? -1.0 : 1.0;
}
}

void compute_rhs_planewave(const RWG& rwg, const Mesh& mesh,
                           std::complex<double> k_ext, double eta_ext,
                           const Vec3& E0, const Vec3& k_hat,
                           int quad_order,
                           std::complex<double>* b)
{
    int N = rwg.N;
    TriQuad quad = tri_quadrature(quad_order);
    int Nq = quad.npts;

    // H0 = k_hat x E0 / eta_ext
    Vec3 H0 = k_hat.cross(E0) * (rhs_h_sign() / eta_ext);

    std::vector<double> lam0(Nq);
    for (int q = 0; q < Nq; q++)
        lam0[q] = 1.0 - quad.pts[q][0] - quad.pts[q][1];

    std::fill_n(b, 2 * N, std::complex<double>(0.0, 0.0));

    // Two halves: plus (+1) and minus (-1)
    for (int half = 0; half < 2; half++) {
        int sign = (half == 0) ? +1 : -1;

        for (int n = 0; n < N; n++) {
            int ti = (sign > 0) ? rwg.tri_p[n] : rwg.tri_m[n];
            Vec3 free_v = (sign > 0) ? rwg.free_p[n] : rwg.free_m[n];
            double area = (sign > 0) ? rwg.area_p[n] : rwg.area_m[n];
            double len = rwg.length[n];
            double coeff = sign * len / (2.0 * area);

            Vec3 v0, v1, v2;
            mesh.tri_verts(ti, v0, v1, v2);

            std::complex<double> bE(0), bH(0);

            for (int q = 0; q < Nq; q++) {
                double l0 = lam0[q], l1 = quad.pts[q][0], l2 = quad.pts[q][1];
                Vec3 rr = v0 * l0 + v1 * l1 + v2 * l2;

                // Basis function value
                Vec3 fv = (rr - free_v) * coeff;

                // Phase: exp(i * k_ext * k_hat . r)
                double phase_arg = k_ext.real() * k_hat.dot(rr);
                double phase_arg_im = k_ext.imag() * k_hat.dot(rr);
                std::complex<double> phase = std::exp(
                    std::complex<double>(-phase_arg_im, phase_arg));

                double jw = area * quad.wts[q];

                bE += fv.dot(E0) * phase * jw;
                bH += fv.dot(H0) * phase * jw;
            }

            b[n]     += bE;
            b[N + n] += bH;
        }
    }
}

void compute_rhs_planewave_pair(const RWG& rwg, const Mesh& mesh,
                                std::complex<double> k_ext, double eta_ext,
                                const Vec3& E0_a, const Vec3& E0_b,
                                const Vec3& k_hat, int quad_order,
                                std::complex<double>* b_a,
                                std::complex<double>* b_b)
{
    int N = rwg.N;
    TriQuad quad = tri_quadrature(quad_order);
    int Nq = quad.npts;

    const double hsign = rhs_h_sign();
    Vec3 H0_a = k_hat.cross(E0_a) * (hsign / eta_ext);
    Vec3 H0_b = k_hat.cross(E0_b) * (hsign / eta_ext);

    std::vector<double> lam0(Nq);
    for (int q = 0; q < Nq; q++)
        lam0[q] = 1.0 - quad.pts[q][0] - quad.pts[q][1];

    std::fill_n(b_a, 2 * N, std::complex<double>(0.0, 0.0));
    std::fill_n(b_b, 2 * N, std::complex<double>(0.0, 0.0));

    for (int half = 0; half < 2; half++) {
        int sign = (half == 0) ? +1 : -1;

        for (int n = 0; n < N; n++) {
            int ti = (sign > 0) ? rwg.tri_p[n] : rwg.tri_m[n];
            Vec3 free_v = (sign > 0) ? rwg.free_p[n] : rwg.free_m[n];
            double area = (sign > 0) ? rwg.area_p[n] : rwg.area_m[n];
            double len = rwg.length[n];
            double coeff = sign * len / (2.0 * area);

            Vec3 v0, v1, v2;
            mesh.tri_verts(ti, v0, v1, v2);

            std::complex<double> bEa(0), bHa(0), bEb(0), bHb(0);

            for (int q = 0; q < Nq; q++) {
                double l0 = lam0[q], l1 = quad.pts[q][0], l2 = quad.pts[q][1];
                Vec3 rr = v0 * l0 + v1 * l1 + v2 * l2;
                Vec3 fv = (rr - free_v) * coeff;

                double kr = k_hat.dot(rr);
                double phase_arg = k_ext.real() * kr;
                double phase_arg_im = k_ext.imag() * kr;
                std::complex<double> phase = std::exp(
                    std::complex<double>(-phase_arg_im, phase_arg));
                double jw = area * quad.wts[q];
                std::complex<double> pw = phase * jw;

                bEa += fv.dot(E0_a) * pw;
                bHa += fv.dot(H0_a) * pw;
                bEb += fv.dot(E0_b) * pw;
                bHb += fv.dot(H0_b) * pw;
            }

            b_a[n]     += bEa;
            b_a[N + n] += bHa;
            b_b[n]     += bEb;
            b_b[N + n] += bHb;
        }
    }
}

void compute_rhs_planewave_pair_cached(const FFCache& cache,
                                       std::complex<double> k_ext, double eta_ext,
                                       const Vec3& E0_a, const Vec3& E0_b,
                                       const Vec3& k_hat,
                                       std::complex<double>* b_a,
                                       std::complex<double>* b_b)
{
    int N = cache.N;
    int Nq = cache.Nq;
    const double hsign = rhs_h_sign();
    Vec3 H0_a = k_hat.cross(E0_a) * (hsign / eta_ext);
    Vec3 H0_b = k_hat.cross(E0_b) * (hsign / eta_ext);

    std::fill_n(b_a, 2 * N, std::complex<double>(0.0, 0.0));
    std::fill_n(b_b, 2 * N, std::complex<double>(0.0, 0.0));

    int total = 2 * N * Nq;
    int half_stride = N * Nq;
    for (int i = 0; i < total; i++) {
        int n = (i % half_stride) / Nq;
        double px = cache.qpts[i * 3];
        double py = cache.qpts[i * 3 + 1];
        double pz = cache.qpts[i * 3 + 2];
        double fx = cache.fvals[i * 3];
        double fy = cache.fvals[i * 3 + 1];
        double fz = cache.fvals[i * 3 + 2];
        double jw = cache.jw[i];

        double kr = k_hat.x * px + k_hat.y * py + k_hat.z * pz;
        double phase_arg = k_ext.real() * kr;
        double phase_arg_im = k_ext.imag() * kr;
        std::complex<double> pw = std::exp(
            std::complex<double>(-phase_arg_im, phase_arg)) * jw;

        double fEa = fx * E0_a.x + fy * E0_a.y + fz * E0_a.z;
        double fHa = fx * H0_a.x + fy * H0_a.y + fz * H0_a.z;
        double fEb = fx * E0_b.x + fy * E0_b.y + fz * E0_b.z;
        double fHb = fx * H0_b.x + fy * H0_b.y + fz * H0_b.z;

        b_a[n]     += fEa * pw;
        b_a[N + n] += fHa * pw;
        b_b[n]     += fEb * pw;
        b_b[N + n] += fHb * pw;
    }
}

struct RHSOrientDevice {
    Vec3 E0_a;
    Vec3 E0_b;
    Vec3 k_hat;
};

RHSBatchWorkspace::RHSBatchWorkspace()
    : h_orient(0), h_B(0), d_orient(0), d_B(0),
      stream(0),
      h_orient_pinned(false), h_B_pinned(false),
      cap_orient(0), cap_host_rhs_elems(0), cap_rhs_elems(0)
{
}

static void release_host_workspace(void*& ptr, bool& pinned)
{
    if (!ptr)
        return;
    if (pinned)
        cudaFreeHost(ptr);
    else
        std::free(ptr);
    ptr = 0;
    pinned = false;
}

static void allocate_host_workspace(void*& ptr, bool& pinned, size_t bytes, const char* label)
{
    cudaError_t host_err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (host_err == cudaSuccess) {
        pinned = true;
        return;
    }

    cudaGetLastError();
    ptr = std::malloc(bytes);
    pinned = false;
    if (!ptr) {
        fprintf(stderr, "Error: failed to allocate %s host workspace (%zu bytes)\n",
                label, bytes);
        std::exit(1);
    }
}

void RHSBatchWorkspace::reserve(int n_orient, size_t rhs_elems, bool need_host_rhs)
{
    if (!stream)
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (n_orient > cap_orient) {
        release_host_workspace(h_orient, h_orient_pinned);
        if (d_orient)
            cudaFree(d_orient);

        size_t orient_bytes = (size_t)n_orient * sizeof(RHSOrientDevice);
        allocate_host_workspace(h_orient, h_orient_pinned, orient_bytes, "RHS orientation");
        CUDA_CHECK(cudaMalloc(&d_orient, (size_t)n_orient * sizeof(RHSOrientDevice)));
        cap_orient = n_orient;
    }
    if (need_host_rhs && rhs_elems > cap_host_rhs_elems) {
        release_host_workspace(h_B, h_B_pinned);
        allocate_host_workspace(h_B, h_B_pinned, rhs_elems * sizeof(double2), "RHS output");
        cap_host_rhs_elems = rhs_elems;
    }
    if (rhs_elems > cap_rhs_elems) {
        if (d_B)
            cudaFree(d_B);
        CUDA_CHECK(cudaMalloc(&d_B, rhs_elems * sizeof(double2)));
        cap_rhs_elems = rhs_elems;
    }
}

std::complex<double>* RHSBatchWorkspace::host_B()
{
    return reinterpret_cast<std::complex<double>*>(h_B);
}

const std::complex<double>* RHSBatchWorkspace::host_B() const
{
    return reinterpret_cast<const std::complex<double>*>(h_B);
}

void RHSBatchWorkspace::free()
{
    release_host_workspace(h_orient, h_orient_pinned);
    release_host_workspace(h_B, h_B_pinned);
    if (d_orient) {
        cudaFree(d_orient);
        d_orient = 0;
    }
    if (d_B) {
        cudaFree(d_B);
        d_B = 0;
    }
    if (stream) {
        cudaStreamDestroy(stream);
        stream = 0;
    }
    cap_orient = 0;
    cap_host_rhs_elems = 0;
    cap_rhs_elems = 0;
}

RHSBatchWorkspace::~RHSBatchWorkspace()
{
    free();
}

__global__ void rhs_pairs_cached_kernel(const double* __restrict__ qpts,
                                        const double* __restrict__ fvals,
                                        const double* __restrict__ jw,
                                        const RHSOrientDevice* __restrict__ orient,
                                        double k_re, double k_im, double inv_eta, double h_sign,
                                        double row_h_re, double row_h_im,
                                        int N, int Nq, int n_orient,
                                        double2* __restrict__ B)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int oi = blockIdx.y;
    if (n >= N || oi >= n_orient)
        return;

    RHSOrientDevice o = orient[oi];
    Vec3 ka = o.k_hat;
    Vec3 ea = o.E0_a;
    Vec3 eb = o.E0_b;
    double hax = (ka.y * ea.z - ka.z * ea.y) * inv_eta * h_sign;
    double hay = (ka.z * ea.x - ka.x * ea.z) * inv_eta * h_sign;
    double haz = (ka.x * ea.y - ka.y * ea.x) * inv_eta * h_sign;
    double hbx = (ka.y * eb.z - ka.z * eb.y) * inv_eta * h_sign;
    double hby = (ka.z * eb.x - ka.x * eb.z) * inv_eta * h_sign;
    double hbz = (ka.x * eb.y - ka.y * eb.x) * inv_eta * h_sign;

    double bEa_re = 0.0, bEa_im = 0.0;
    double bHa_re = 0.0, bHa_im = 0.0;
    double bEb_re = 0.0, bEb_im = 0.0;
    double bHb_re = 0.0, bHb_im = 0.0;

    int half_stride = N * Nq;
    for (int half = 0; half < 2; half++) {
        int base = half * half_stride + n * Nq;
        for (int q = 0; q < Nq; q++) {
            int idx = base + q;
            double px = qpts[idx * 3];
            double py = qpts[idx * 3 + 1];
            double pz = qpts[idx * 3 + 2];
            double fx = fvals[idx * 3];
            double fy = fvals[idx * 3 + 1];
            double fz = fvals[idx * 3 + 2];
            double w = jw[idx];

            double kr = ka.x * px + ka.y * py + ka.z * pz;
            double arg = k_re * kr;
            double s, c;
            sincos(arg, &s, &c);
            double amp = exp(-k_im * kr) * w;
            double pr = amp * c;
            double pi = amp * s;

            double fEa = fx * ea.x + fy * ea.y + fz * ea.z;
            double fHa = fx * hax + fy * hay + fz * haz;
            double fEb = fx * eb.x + fy * eb.y + fz * eb.z;
            double fHb = fx * hbx + fy * hby + fz * hbz;

            bEa_re += fEa * pr; bEa_im += fEa * pi;
            bHa_re += fHa * pr; bHa_im += fHa * pi;
            bEb_re += fEb * pr; bEb_im += fEb * pi;
            bHb_re += fHb * pr; bHb_im += fHb * pi;
        }
    }

    size_t off_a = ((size_t)oi * 2) * (size_t)(2 * N);
    size_t off_b = ((size_t)oi * 2 + 1) * (size_t)(2 * N);
    double sHa_re = bHa_re * row_h_re - bHa_im * row_h_im;
    double sHa_im = bHa_re * row_h_im + bHa_im * row_h_re;
    double sHb_re = bHb_re * row_h_re - bHb_im * row_h_im;
    double sHb_im = bHb_re * row_h_im + bHb_im * row_h_re;
    B[off_a + n]     = make_double2(bEa_re, bEa_im);
    B[off_a + N + n] = make_double2(sHa_re, sHa_im);
    B[off_b + n]     = make_double2(bEb_re, bEb_im);
    B[off_b + N + n] = make_double2(sHb_re, sHb_im);
}

int compute_rhs_planewave_pairs_cached_cuda(const FFCache& cache,
                                            std::complex<double> k_ext,
                                            double eta_ext,
                                            const Vec3* E0_a,
                                            const Vec3* E0_b,
                                            const Vec3* k_hat,
                                            int n_orient,
                                            std::complex<double>* B)
{
    if (n_orient <= 0)
        return 0;

    int N = cache.N;
    int Nq = cache.Nq;
    int total_quad = 2 * N * Nq;
    double *d_qpts = 0, *d_fvals = 0, *d_jw = 0;
    RHSOrientDevice* d_orient = 0;
    double2* d_B = 0;
    size_t rhs_elems = (size_t)n_orient * 2 * 2 * (size_t)N;
    std::vector<RHSOrientDevice> orient((size_t)n_orient);
    for (int i = 0; i < n_orient; i++) {
        orient[(size_t)i].E0_a = E0_a[i];
        orient[(size_t)i].E0_b = E0_b[i];
        orient[(size_t)i].k_hat = k_hat[i];
    }

    CUDA_CHECK(cudaMalloc(&d_qpts, total_quad * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fvals, total_quad * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_jw, total_quad * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_orient, n_orient * sizeof(RHSOrientDevice)));
    CUDA_CHECK(cudaMalloc(&d_B, rhs_elems * sizeof(double2)));

    CUDA_CHECK(cudaMemcpy(d_qpts, cache.qpts.data(), total_quad * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fvals, cache.fvals.data(), total_quad * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_jw, cache.jw.data(), total_quad * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_orient, orient.data(), n_orient * sizeof(RHSOrientDevice), cudaMemcpyHostToDevice));

    dim3 block(128);
    dim3 grid((N + (int)block.x - 1) / (int)block.x, n_orient);
    const double hsign = rhs_h_sign();
    rhs_pairs_cached_kernel<<<grid, block>>>(
        d_qpts, d_fvals, d_jw, d_orient,
        k_ext.real(), k_ext.imag(), 1.0 / eta_ext, hsign,
        1.0, 0.0,
        N, Nq, n_orient, d_B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(B), d_B,
                          rhs_elems * sizeof(double2), cudaMemcpyDeviceToHost));

    cudaFree(d_qpts);
    cudaFree(d_fvals);
    cudaFree(d_jw);
    cudaFree(d_orient);
    cudaFree(d_B);
    return 0;
}

int compute_rhs_planewave_pairs_cached_cuda(const FFCacheGPU& gpu_cache,
                                            std::complex<double> k_ext,
                                            double eta_ext,
                                            const Vec3* E0_a,
                                            const Vec3* E0_b,
                                            const Vec3* k_hat,
                                            int n_orient,
                                            std::complex<double>* B)
{
    RHSBatchWorkspace workspace;
    return compute_rhs_planewave_pairs_cached_cuda_ws(
        gpu_cache, workspace, k_ext, eta_ext, E0_a, E0_b, k_hat, n_orient, B);
}

int compute_rhs_planewave_pairs_cached_cuda_ws(const FFCacheGPU& gpu_cache,
                                               RHSBatchWorkspace& workspace,
                                               std::complex<double> k_ext,
                                               double eta_ext,
                                               const Vec3* E0_a,
                                               const Vec3* E0_b,
                                               const Vec3* k_hat,
                                               int n_orient,
                                               std::complex<double>* B)
{
    return compute_rhs_planewave_pairs_cached_cuda_ws_scaled(
        gpu_cache, workspace, k_ext, eta_ext, std::complex<double>(1.0, 0.0),
        E0_a, E0_b, k_hat, n_orient, B);
}

int compute_rhs_planewave_pairs_cached_cuda_ws_scaled(const FFCacheGPU& gpu_cache,
                                                      RHSBatchWorkspace& workspace,
                                                      std::complex<double> k_ext,
                                                      double eta_ext,
                                                      std::complex<double> row_h_scale,
                                                      const Vec3* E0_a,
                                                      const Vec3* E0_b,
                                                      const Vec3* k_hat,
                                                      int n_orient,
                                                      std::complex<double>* B)
{
    if (n_orient <= 0)
        return 0;

    int N = gpu_cache.N;
    int Nq = gpu_cache.Nq;
    size_t rhs_elems = (size_t)n_orient * 2 * 2 * (size_t)N;
    workspace.reserve(n_orient, rhs_elems, B == nullptr);
    RHSOrientDevice* orient = reinterpret_cast<RHSOrientDevice*>(workspace.h_orient);
    for (int i = 0; i < n_orient; i++) {
        orient[i].E0_a = E0_a[i];
        orient[i].E0_b = E0_b[i];
        orient[i].k_hat = k_hat[i];
    }

    RHSOrientDevice* d_orient = reinterpret_cast<RHSOrientDevice*>(workspace.d_orient);
    double2* d_B = reinterpret_cast<double2*>(workspace.d_B);

    CUDA_CHECK(cudaMemcpyAsync(d_orient, orient,
                               n_orient * sizeof(RHSOrientDevice),
                               cudaMemcpyHostToDevice, workspace.stream));

    dim3 block(128);
    dim3 grid((N + (int)block.x - 1) / (int)block.x, n_orient);
    const double hsign = rhs_h_sign();
    rhs_pairs_cached_kernel<<<grid, block, 0, workspace.stream>>>(
        gpu_cache.d_qpts, gpu_cache.d_fvals, gpu_cache.d_jw,
        d_orient,
        k_ext.real(), k_ext.imag(), 1.0 / eta_ext, hsign,
        row_h_scale.real(), row_h_scale.imag(),
        N, Nq, n_orient, d_B);
    CUDA_CHECK(cudaGetLastError());
    double2* h_out = B ? reinterpret_cast<double2*>(B) : reinterpret_cast<double2*>(workspace.h_B);
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_B,
                               rhs_elems * sizeof(double2),
                               cudaMemcpyDeviceToHost, workspace.stream));
    CUDA_CHECK(cudaStreamSynchronize(workspace.stream));
    return 0;
}
