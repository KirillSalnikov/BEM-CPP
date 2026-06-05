#include "assembly.h"
#include "graglia.h"
#include <cstring>
#include <cstdio>
#include <vector>
#include <map>
#include <cmath>

// ============================================================
// CUDA kernel: assemble L,K block (float64 compute)
// Ported from OpenCL kernel in bem_opencl.py
// ============================================================

__global__ void assemble_LK_kernel(
    const double* __restrict__ tp,       // (B*Nq, 3) test quad points
    const double* __restrict__ sq,       // (N*Nq, 3) source quad points
    const double* __restrict__ tf,       // (B*Nq, 3) test basis fn values
    const double* __restrict__ sf,       // (N*Nq, 3) source basis fn values
    const double* __restrict__ sq_x_sf,  // (N*Nq, 3) cross(sq, sf) precomputed
    const double* __restrict__ jw_t,     // (B*Nq) test Jacobian*weights
    const double* __restrict__ jw_s,     // (N*Nq) source Jacobian*weights
    const int* __restrict__ t_tri,      // (B) test triangle indices
    const int* __restrict__ s_tri,      // (N) source triangle indices
    const double* __restrict__ t_div,    // (B) test divergence
    const double* __restrict__ s_div,    // (N) source divergence
    double k_re, double k_im, double inv4pi,
    int Nq, int N_src, int B,
    double* __restrict__ L_re, double* __restrict__ L_im,
    double* __restrict__ K_re, double* __restrict__ K_im)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= B || n >= N_src) return;

    int is_sing = (t_tri[b] == s_tri[n]);

    double Lvec_re = 0.0, Lvec_im = 0.0;
    double Lscl_re = 0.0, Lscl_im = 0.0;
    double Kacc_re = 0.0, Kacc_im = 0.0;

    for (int iq = 0; iq < Nq; iq++) {
        int ti = b * Nq + iq;
        double tpx = __ldg(&tp[ti * 3 + 0]);
        double tpy = __ldg(&tp[ti * 3 + 1]);
        double tpz = __ldg(&tp[ti * 3 + 2]);
        double tfx = __ldg(&tf[ti * 3 + 0]);
        double tfy = __ldg(&tf[ti * 3 + 1]);
        double tfz = __ldg(&tf[ti * 3 + 2]);
        double jwt = __ldg(&jw_t[ti]);

        for (int jq = 0; jq < Nq; jq++) {
            int sj = n * Nq + jq;

            double sqx = __ldg(&sq[sj * 3 + 0]);
            double sqy = __ldg(&sq[sj * 3 + 1]);
            double sqz = __ldg(&sq[sj * 3 + 2]);

            double dx = tpx - sqx;
            double dy = tpy - sqy;
            double dz = tpz - sqz;
            double R = sqrt(dx*dx + dy*dy + dz*dz);
            double R_safe = fmax(R, 1e-12);

            // Green's function: G = exp(ikR) / (4piR)
            double eR = exp(-k_im * R_safe);
            double cosR = cos(k_re * R_safe);
            double sinR = sin(k_re * R_safe);
            double G_re = eR * cosR * inv4pi / R_safe;
            double G_im = eR * sinR * inv4pi / R_safe;
            double inv4piR = inv4pi / R_safe;

            double Gu_re = G_re;
            double Gu_im = G_im;
            if (is_sing) {
                Gu_re -= inv4piR;  // G_smooth = (exp(ikR) - 1) / (4piR)
            }

            double jw = jwt * __ldg(&jw_s[sj]);

            double sfx = __ldg(&sf[sj * 3 + 0]);
            double sfy = __ldg(&sf[sj * 3 + 1]);
            double sfz = __ldg(&sf[sj * 3 + 2]);
            double f_dot = tfx * sfx + tfy * sfy + tfz * sfz;

            double Gjw_re = Gu_re * jw;
            double Gjw_im = Gu_im * jw;
            Lvec_re += f_dot * Gjw_re;
            Lvec_im += f_dot * Gjw_im;
            Lscl_re += Gjw_re;
            Lscl_im += Gjw_im;

            // K operator
            if (!is_sing && R > 1e-12) {
                double a_re = (-k_im - 1.0 / R_safe) / R_safe;
                double a_im = k_re / R_safe;
                double gc_re = G_re * a_re - G_im * a_im;
                double gc_im = G_re * a_im + G_im * a_re;
                double gGjw_re = gc_re * jw;
                double gGjw_im = gc_im * jw;

                // Triple product: tf x tp . sf - tf . (sq x sf)
                double cx_val = tfy * tpz - tfz * tpy;
                double cy_val = tfz * tpx - tfx * tpz;
                double cz_val = tfx * tpy - tfy * tpx;
                double triple = cx_val * sfx + cy_val * sfy + cz_val * sfz
                    - tfx * __ldg(&sq_x_sf[sj * 3 + 0])
                    - tfy * __ldg(&sq_x_sf[sj * 3 + 1])
                    - tfz * __ldg(&sq_x_sf[sj * 3 + 2]);

                Kacc_re += gGjw_re * triple;
                Kacc_im += gGjw_im * triple;
            }
        }
    }

    // Combine: L[b,n] = ik * Lvec - (i/k) * div_prod * Lscl
    double ik_re = -k_im;
    double ik_im_f = k_re;
    double k_sq = k_re * k_re + k_im * k_im;
    double iok_re = k_im / k_sq;
    double iok_im = k_re / k_sq;
    double div_prod = __ldg(&t_div[b]) * __ldg(&s_div[n]);

    double term1_re = ik_re * Lvec_re - ik_im_f * Lvec_im;
    double term1_im = ik_re * Lvec_im + ik_im_f * Lvec_re;
    double term2_re = iok_re * div_prod * Lscl_re - iok_im * div_prod * Lscl_im;
    double term2_im = iok_re * div_prod * Lscl_im + iok_im * div_prod * Lscl_re;

    int idx = b * N_src + n;
    L_re[idx] = term1_re - term2_re;
    L_im[idx] = term1_im - term2_im;
    K_re[idx] = Kacc_re;
    K_im[idx] = Kacc_im;
}


// ============================================================
// Host: precompute quadrature data for one half (plus or minus)
// ============================================================

struct HalfData {
    std::vector<double> qpts;  // (N*Nq, 3) flat
    std::vector<double> fvals; // (N*Nq, 3) flat
    std::vector<double> jw;    // (N*Nq) flat
    std::vector<double> divs;  // (N) divergence
    std::vector<int>   tri_idx; // (N) triangle indices
};

static HalfData precompute_half(const RWG& rwg, const Mesh& mesh,
                                const TriQuad& quad, int sign) {
    int N = rwg.N;
    int Nq = quad.npts;
    HalfData h;
    h.qpts.resize(N * Nq * 3);
    h.fvals.resize(N * Nq * 3);
    h.jw.resize(N * Nq);
    h.divs.resize(N);
    h.tri_idx.resize(N);

    // Precompute barycentric lambda0
    std::vector<double> lam0(Nq);
    for (int q = 0; q < Nq; q++)
        lam0[q] = 1.0 - quad.pts[q][0] - quad.pts[q][1];

    for (int n = 0; n < N; n++) {
        int ti = (sign > 0) ? rwg.tri_p[n] : rwg.tri_m[n];
        Vec3 free_v = (sign > 0) ? rwg.free_p[n] : rwg.free_m[n];
        double area = (sign > 0) ? rwg.area_p[n] : rwg.area_m[n];
        double len = rwg.length[n];

        h.tri_idx[n] = ti;
        h.divs[n] = sign * len / area;  // +l/A or -l/A

        Vec3 v0, v1, v2;
        mesh.tri_verts(ti, v0, v1, v2);

        double coeff = sign * len / (2.0 * area);

        for (int q = 0; q < Nq; q++) {
            // Quadrature point in physical space
            double l0 = lam0[q], l1 = quad.pts[q][0], l2 = quad.pts[q][1];
            Vec3 rr = v0 * l0 + v1 * l1 + v2 * l2;

            int idx = (n * Nq + q) * 3;
            h.qpts[idx]     = rr.x;
            h.qpts[idx + 1] = rr.y;
            h.qpts[idx + 2] = rr.z;

            // Basis function value: sign * (l/2A) * (r - r_free)
            Vec3 fv = (rr - free_v) * coeff;
            h.fvals[idx]     = fv.x;
            h.fvals[idx + 1] = fv.y;
            h.fvals[idx + 2] = fv.z;

            // Jacobian * weight = area * w_q
            h.jw[n * Nq + q] = area * quad.wts[q];
        }
    }
    return h;
}


// ============================================================
// Host: apply Graglia singular corrections (CPU, float64)
// ============================================================

static void apply_singular_corrections(
    const RWG& rwg, const Mesh& mesh, const TriQuad& quad,
    std::complex<double> k, std::complex<double>* L, int N)
{
    int Nq = quad.npts;
    std::complex<double> ik(0, 1); ik *= k;       // ik = i*k
    std::complex<double> ik_inv(0, 1); ik_inv /= k; // i/k

    // Precompute barycentric lambda0
    std::vector<double> lam0(Nq);
    for (int q = 0; q < Nq; q++)
        lam0[q] = 1.0 - quad.pts[q][0] - quad.pts[q][1];

    // Build triangle → RWG mapping
    // For each triangle, store list of (rwg_index, div, coeff, free_vertex, sign)
    struct RWGEntry {
        int rwg_idx;
        double div_val;
        double coeff;  // sign * length / (2*area)
        Vec3 free_v;
        int sign;
    };
    std::map<int, std::vector<RWGEntry>> tri_to_rwg;

    for (int n = 0; n < N; n++) {
        // Plus half
        {
            RWGEntry e;
            e.rwg_idx = n;
            e.div_val = rwg.length[n] / rwg.area_p[n];
            e.coeff = rwg.length[n] / (2.0 * rwg.area_p[n]);
            e.free_v = rwg.free_p[n];
            e.sign = +1;
            tri_to_rwg[rwg.tri_p[n]].push_back(e);
        }
        // Minus half
        {
            RWGEntry e;
            e.rwg_idx = n;
            e.div_val = -rwg.length[n] / rwg.area_m[n];
            e.coeff = rwg.length[n] / (2.0 * rwg.area_m[n]);
            e.free_v = rwg.free_m[n];
            e.sign = -1;
            tri_to_rwg[rwg.tri_m[n]].push_back(e);
        }
    }

    // For each singular triangle, compute P and V at quad points
    for (auto& kv : tri_to_rwg) {
        int ti = kv.first;
        Vec3 v0, v1, v2;
        mesh.tri_verts(ti, v0, v1, v2);

        // Quadrature points on this triangle
        std::vector<Vec3> qpts(Nq);
        for (int q = 0; q < Nq; q++) {
            double l0 = lam0[q], l1 = quad.pts[q][0], l2 = quad.pts[q][1];
            qpts[q] = v0 * l0 + v1 * l1 + v2 * l2;
        }

        // Compute P and V at each quad point (analytical Graglia)
        std::vector<double> P(Nq);
        std::vector<Vec3> V(Nq);
        for (int q = 0; q < Nq; q++) {
            P[q] = potential_integral_triangle(qpts[q], v0, v1, v2);
            V[q] = vector_potential_integral_triangle(qpts[q], v0, v1, v2, quad);
        }

        // For each test RWG on this triangle
        for (auto& test_e : kv.second) {
            int m = test_e.rwg_idx;
            double t_div = test_e.div_val;
            double t_area = (test_e.sign > 0) ? rwg.area_p[m] : rwg.area_m[m];

            // Compute test function values and jw at quad points
            std::vector<Vec3> t_f(Nq);
            std::vector<double> t_jw(Nq);
            for (int q = 0; q < Nq; q++) {
                t_f[q] = (qpts[q] - test_e.free_v) * (test_e.sign * test_e.coeff);
                t_jw[q] = t_area * quad.wts[q];
            }

            // scalar_base = sum(P[q] * t_jw[q]) * inv4pi
            double scalar_base = 0;
            for (int q = 0; q < Nq; q++)
                scalar_base += P[q] * t_jw[q];
            scalar_base *= INV4PI;

            // For each source RWG on this same triangle
            for (auto& src_e : kv.second) {
                int n_idx = src_e.rwg_idx;
                double s_div = src_e.div_val;

                // L_sing_scalar = -ik_inv * t_div * s_div * scalar_base
                std::complex<double> L_sing_scalar = -ik_inv * t_div * s_div * scalar_base;

                // Vector integral: sum over quad points
                double vec_int = 0;
                for (int q = 0; q < Nq; q++) {
                    // Source basis fn/R: sign * coeff * (V - free * P)
                    Vec3 fn_over_R;
                    fn_over_R.x = src_e.sign * src_e.coeff * (V[q].x - src_e.free_v.x * P[q]);
                    fn_over_R.y = src_e.sign * src_e.coeff * (V[q].y - src_e.free_v.y * P[q]);
                    fn_over_R.z = src_e.sign * src_e.coeff * (V[q].z - src_e.free_v.z * P[q]);

                    vec_int += t_f[q].dot(fn_over_R) * t_jw[q];
                }
                vec_int *= INV4PI;

                L[m * N + n_idx] += L_sing_scalar + ik * vec_int;
            }
        }
    }
}


// ============================================================
// Main assembly routine
// ============================================================

void assemble_L_K_cuda(const RWG& rwg, const Mesh& mesh,
                       std::complex<double> k, int quad_order,
                       std::complex<double>* L, std::complex<double>* K)
{
    Timer timer;
    int N = rwg.N;
    TriQuad quad = tri_quadrature(quad_order);
    int Nq = quad.npts;

    printf("    Assembly: %d RWG, %d quad pts, k=(%.4f,%.4f)...\n",
           N, Nq, k.real(), k.imag());

    // Precompute half data
    HalfData hp = precompute_half(rwg, mesh, quad, +1);
    HalfData hm = precompute_half(rwg, mesh, quad, -1);

    // Initialize output to zero
    memset(L, 0, N * N * sizeof(std::complex<double>));
    memset(K, 0, N * N * sizeof(std::complex<double>));

    double k_re = k.real();
    double k_im = k.imag();
    double inv4pi = INV4PI;

    // 3 passes: (p,p), (p,m), (m,m). Cross-term (m,p) = transpose of (p,m).
    struct Pass {
        HalfData* test;
        HalfData* src;
        int th, sh;  // 0=plus, 1=minus
    };
    Pass passes[] = {
        {&hp, &hp, 0, 0},
        {&hp, &hm, 0, 1},
        {&hm, &hm, 1, 1},
    };

    // Allocate device memory for source arrays (reused across batches)
    int Nsq = N * Nq;

    for (int pass = 0; pass < 3; pass++) {
        HalfData& test = *passes[pass].test;
        HalfData& src  = *passes[pass].src;
        bool is_cross = (passes[pass].th != passes[pass].sh);

        // Precompute cross(sq, sf) on host
        std::vector<double> sq_x_sf(Nsq * 3);
        for (int i = 0; i < Nsq; i++) {
            double sx = src.qpts[i*3], sy = src.qpts[i*3+1], sz = src.qpts[i*3+2];
            double fx = src.fvals[i*3], fy = src.fvals[i*3+1], fz = src.fvals[i*3+2];
            sq_x_sf[i*3]   = sy*fz - sz*fy;
            sq_x_sf[i*3+1] = sz*fx - sx*fz;
            sq_x_sf[i*3+2] = sx*fy - sy*fx;
        }

        // Upload source arrays to device
        double *d_sq, *d_sf, *d_sq_x_sf, *d_jw_s;
        int *d_s_tri;
        double *d_s_div;
        CUDA_CHECK(cudaMalloc(&d_sq, Nsq * 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_sf, Nsq * 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_sq_x_sf, Nsq * 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_jw_s, Nsq * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_s_tri, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_s_div, N * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_sq, src.qpts.data(), Nsq*3*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sf, src.fvals.data(), Nsq*3*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sq_x_sf, sq_x_sf.data(), Nsq*3*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_jw_s, src.jw.data(), Nsq*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_s_tri, src.tri_idx.data(), N*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_s_div, src.divs.data(), N*sizeof(double), cudaMemcpyHostToDevice));

        // Batch processing
        // Memory budget: ~2GB for test arrays + output
        // Per element: 4 double outputs = 32 bytes, test arrays roughly BNq*24 bytes.
        int batch_size = std::min(N, std::max(1, (int)(1e9 / (N * 32 + Nq * 24))));
        // Ensure batch_size is at least 256 for GPU efficiency
        batch_size = std::max(batch_size, std::min(256, N));

        for (int b_start = 0; b_start < N; b_start += batch_size) {
            int b_end = std::min(b_start + batch_size, N);
            int B = b_end - b_start;
            int BNq = B * Nq;

            // Upload test batch to device
            double *d_tp, *d_tf, *d_jw_t;
            int *d_t_tri;
            double *d_t_div;
            double *d_L_re, *d_L_im, *d_K_re, *d_K_im;

            CUDA_CHECK(cudaMalloc(&d_tp, BNq * 3 * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_tf, BNq * 3 * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_jw_t, BNq * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_t_tri, B * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_t_div, B * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_L_re, B * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_L_im, B * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_K_re, B * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_K_im, B * N * sizeof(double)));

            CUDA_CHECK(cudaMemcpy(d_tp, &test.qpts[b_start*Nq*3], BNq*3*sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_tf, &test.fvals[b_start*Nq*3], BNq*3*sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_jw_t, &test.jw[b_start*Nq], BNq*sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_t_tri, &test.tri_idx[b_start], B*sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_t_div, &test.divs[b_start], B*sizeof(double), cudaMemcpyHostToDevice));

            // Launch kernel
            dim3 block(16, 16);
            dim3 grid((B + block.x - 1) / block.x, (N + block.y - 1) / block.y);

            assemble_LK_kernel<<<grid, block>>>(
                d_tp, d_sq, d_tf, d_sf, d_sq_x_sf,
                d_jw_t, d_jw_s, d_t_tri, d_s_tri,
                d_t_div, d_s_div,
                k_re, k_im, inv4pi, Nq, N, B,
                d_L_re, d_L_im, d_K_re, d_K_im);
            CUDA_CHECK(cudaGetLastError());

            // Download results
            std::vector<double> h_L_re(B*N), h_L_im(B*N), h_K_re(B*N), h_K_im(B*N);
            CUDA_CHECK(cudaMemcpy(h_L_re.data(), d_L_re, B*N*sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_L_im.data(), d_L_im, B*N*sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_K_re.data(), d_K_re, B*N*sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_K_im.data(), d_K_im, B*N*sizeof(double), cudaMemcpyDeviceToHost));

            // Accumulate into double-precision L, K
            for (int b = 0; b < B; b++) {
                int m = b_start + b;
                for (int n = 0; n < N; n++) {
                    int idx = b * N + n;
                    std::complex<double> lval(h_L_re[idx], h_L_im[idx]);
                    std::complex<double> kval(h_K_re[idx], h_K_im[idx]);

                    if (is_cross) {
                        // Cross-term: add to both (m,n) and (n,m)
                        L[m * N + n] += lval;
                        L[n * N + m] += lval;
                        K[m * N + n] += kval;
                        K[n * N + m] += kval;
                    } else {
                        L[m * N + n] += lval;
                        K[m * N + n] += kval;
                    }
                }
            }

            // Free test batch device memory
            cudaFree(d_tp); cudaFree(d_tf); cudaFree(d_jw_t);
            cudaFree(d_t_tri); cudaFree(d_t_div);
            cudaFree(d_L_re); cudaFree(d_L_im);
            cudaFree(d_K_re); cudaFree(d_K_im);
        }

        // Free source device memory
        cudaFree(d_sq); cudaFree(d_sf); cudaFree(d_sq_x_sf);
        cudaFree(d_jw_s); cudaFree(d_s_tri); cudaFree(d_s_div);
    }

    double t_main = timer.elapsed_s();
    printf("    Main loop: %.1fs\n", t_main);

    // Apply singular corrections (CPU, float64)
    Timer t_sing;
    apply_singular_corrections(rwg, mesh, quad, k, L, N);
    printf("    Singular corrections: %.1fs\n", t_sing.elapsed_s());

    // Symmetrize: L = (L + L^T) / 2, same for K
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            std::complex<double> ls = (L[i*N+j] + L[j*N+i]) * 0.5;
            L[i*N+j] = ls; L[j*N+i] = ls;
            std::complex<double> ks = (K[i*N+j] + K[j*N+i]) * 0.5;
            K[i*N+j] = ks; K[j*N+i] = ks;
        }
    }

    printf("    Total assembly: %.1fs\n", timer.elapsed_s());
}
