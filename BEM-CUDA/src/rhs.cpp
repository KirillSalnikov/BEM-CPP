#include "rhs.h"
#include "farfield.h"
#include "quadrature.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>

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
    Vec3 H0 = k_hat.cross(E0) * (1.0 / eta_ext);

    std::vector<double> lam0(Nq);
    for (int q = 0; q < Nq; q++)
        lam0[q] = 1.0 - quad.pts[q][0] - quad.pts[q][1];

    memset(b, 0, 2 * N * sizeof(std::complex<double>));

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

    Vec3 H0_a = k_hat.cross(E0_a) * (1.0 / eta_ext);
    Vec3 H0_b = k_hat.cross(E0_b) * (1.0 / eta_ext);

    std::vector<double> lam0(Nq);
    for (int q = 0; q < Nq; q++)
        lam0[q] = 1.0 - quad.pts[q][0] - quad.pts[q][1];

    memset(b_a, 0, 2 * N * sizeof(std::complex<double>));
    memset(b_b, 0, 2 * N * sizeof(std::complex<double>));

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
    Vec3 H0_a = k_hat.cross(E0_a) * (1.0 / eta_ext);
    Vec3 H0_b = k_hat.cross(E0_b) * (1.0 / eta_ext);

    memset(b_a, 0, 2 * N * sizeof(std::complex<double>));
    memset(b_b, 0, 2 * N * sizeof(std::complex<double>));

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

__global__ void rhs_pairs_cached_kernel(const double* __restrict__ qpts,
                                        const double* __restrict__ fvals,
                                        const double* __restrict__ jw,
                                        const Vec3* __restrict__ E0_a,
                                        const Vec3* __restrict__ E0_b,
                                        const Vec3* __restrict__ k_hat,
                                        double k_re, double k_im, double inv_eta,
                                        int N, int Nq, int n_orient,
                                        double2* __restrict__ B)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int oi = blockIdx.y;
    if (n >= N || oi >= n_orient)
        return;

    Vec3 ka = k_hat[oi];
    Vec3 ea = E0_a[oi];
    Vec3 eb = E0_b[oi];
    double hax = (ka.y * ea.z - ka.z * ea.y) * inv_eta;
    double hay = (ka.z * ea.x - ka.x * ea.z) * inv_eta;
    double haz = (ka.x * ea.y - ka.y * ea.x) * inv_eta;
    double hbx = (ka.y * eb.z - ka.z * eb.y) * inv_eta;
    double hby = (ka.z * eb.x - ka.x * eb.z) * inv_eta;
    double hbz = (ka.x * eb.y - ka.y * eb.x) * inv_eta;

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
    B[off_a + n]     = make_double2(bEa_re, bEa_im);
    B[off_a + N + n] = make_double2(bHa_re, bHa_im);
    B[off_b + n]     = make_double2(bEb_re, bEb_im);
    B[off_b + N + n] = make_double2(bHb_re, bHb_im);
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
    Vec3 *d_E0_a = 0, *d_E0_b = 0, *d_k_hat = 0;
    double2* d_B = 0;
    size_t rhs_elems = (size_t)n_orient * 2 * 2 * (size_t)N;

    CUDA_CHECK(cudaMalloc(&d_qpts, total_quad * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fvals, total_quad * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_jw, total_quad * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_E0_a, n_orient * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_E0_b, n_orient * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_k_hat, n_orient * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_B, rhs_elems * sizeof(double2)));

    CUDA_CHECK(cudaMemcpy(d_qpts, cache.qpts.data(), total_quad * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fvals, cache.fvals.data(), total_quad * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_jw, cache.jw.data(), total_quad * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E0_a, E0_a, n_orient * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E0_b, E0_b, n_orient * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_hat, k_hat, n_orient * sizeof(Vec3), cudaMemcpyHostToDevice));

    dim3 block(128);
    dim3 grid((N + (int)block.x - 1) / (int)block.x, n_orient);
    rhs_pairs_cached_kernel<<<grid, block>>>(
        d_qpts, d_fvals, d_jw, d_E0_a, d_E0_b, d_k_hat,
        k_ext.real(), k_ext.imag(), 1.0 / eta_ext,
        N, Nq, n_orient, d_B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<double2*>(B), d_B,
                          rhs_elems * sizeof(double2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_qpts);
    cudaFree(d_fvals);
    cudaFree(d_jw);
    cudaFree(d_E0_a);
    cudaFree(d_E0_b);
    cudaFree(d_k_hat);
    cudaFree(d_B);
    return 0;
}
