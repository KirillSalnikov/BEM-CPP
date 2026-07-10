#include "bem_fmm.h"
#include "graglia.h"
#include "gpu_select.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#include <set>
#include <tuple>
#include <algorithm>

namespace {
void register_host_vector(std::vector<cdouble>& v)
{
    if (!v.empty()) {
        CUDA_CHECK(cudaHostRegister(v.data(), v.size() * sizeof(cdouble), cudaHostRegisterDefault));
    }
}

void unregister_host_vector(std::vector<cdouble>& v)
{
    if (!v.empty()) {
        CUDA_CHECK(cudaHostUnregister(v.data()));
    }
}

bool use_pinned_matvec_stage()
{
    return bem_env_flag_enabled("BEM_PINNED_MATVEC_STAGE", true);
}

bool use_full_mvslot_memset(int n_basis)
{
    (void)n_basis;
    return bem_env_flag_enabled("BEM_FMM_MV_MEMSET", false);
}

void upload_complex_stage(double2* dst_device, double2* stage_host,
                          const cdouble* src_host, int n, bool pinned)
{
    if (pinned && stage_host) {
        std::memcpy(stage_host, src_host, (size_t)n * sizeof(double2));
        CUDA_CHECK(cudaMemcpy(dst_device, stage_host,
                              (size_t)n * sizeof(double2), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(dst_device, src_host,
                              (size_t)n * sizeof(double2), cudaMemcpyHostToDevice));
    }
}

void download_complex_stage(cdouble* dst_host, double2* stage_host,
                            const double2* src_device, int n, bool pinned)
{
    if (pinned && stage_host) {
        CUDA_CHECK(cudaMemcpy(stage_host, src_device,
                              (size_t)n * sizeof(double2), cudaMemcpyDeviceToHost));
        std::memcpy(static_cast<void*>(dst_host),
                    static_cast<const void*>(stage_host),
                    (size_t)n * sizeof(double2));
    } else {
        CUDA_CHECK(cudaMemcpy(dst_host, src_device,
                              (size_t)n * sizeof(double2), cudaMemcpyDeviceToHost));
    }
}

__global__ void bem_split_complex_kernel(const double2* in, double* out_re, double* out_im, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double2 v = in[i];
    out_re[i] = v.x;
    out_im[i] = v.y;
}

__global__ void bem_split_complex_batch2_scale_kernel(
    const double2* in1, const double2* in2,
    double* out1_re, double* out1_im,
    double* out2_re, double* out2_im,
    int system_size, int N, double m_scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= system_size) return;
    double s = (i >= N) ? m_scale : 1.0;
    double2 v1 = in1[i];
    double2 v2 = in2[i];
    out1_re[i] = s * v1.x;
    out1_im[i] = s * v1.y;
    out2_re[i] = s * v2.x;
    out2_im[i] = s * v2.y;
}

__global__ void bem_pack_complex_kernel(const double* in_re, const double* in_im, double2* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = make_double2(in_re[i], in_im[i]);
}

__global__ void bem_pack_vector_charges_kernel(
    const double* x1_re, const double* x1_im,
    const double* x2_re, const double* x2_im,
    const double* f_p, const double* f_m,
    const double* jw_p, const double* jw_m,
    int N, int Nq, int comp,
    double* q1_re, double* q1_im,
    double* q2_re, double* q2_im)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_pts = N * Nq;
    if (idx >= half_pts) return;
    int n = idx / Nq;
    double cp = f_p[idx * 3 + comp] * jw_p[idx];
    double cm = f_m[idx * 3 + comp] * jw_m[idx];
    int im = half_pts + idx;
    q1_re[idx] = cp * x1_re[n];
    q1_im[idx] = cp * x1_im[n];
    q2_re[idx] = cp * x2_re[n];
    q2_im[idx] = cp * x2_im[n];
    q1_re[im] = cm * x1_re[n];
    q1_im[im] = cm * x1_im[n];
    q2_re[im] = cm * x2_re[n];
    q2_im[im] = cm * x2_im[n];
}

__global__ void bem_pack_scalar_charges_kernel(
    const double* x1_re, const double* x1_im,
    const double* x2_re, const double* x2_im,
    const double* div_p, const double* div_m,
    const double* jw_p, const double* jw_m,
    int N, int Nq,
    double* q1_re, double* q1_im,
    double* q2_re, double* q2_im)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_pts = N * Nq;
    if (idx >= half_pts) return;
    int n = idx / Nq;
    double cp = div_p[n] * jw_p[idx];
    double cm = div_m[n] * jw_m[idx];
    int im = half_pts + idx;
    q1_re[idx] = cp * x1_re[n];
    q1_im[idx] = cp * x1_im[n];
    q2_re[idx] = cp * x2_re[n];
    q2_im[idx] = cp * x2_im[n];
    q1_re[im] = cm * x1_re[n];
    q1_im[im] = cm * x1_im[n];
    q2_re[im] = cm * x2_re[n];
    q2_im[im] = cm * x2_im[n];
}

__global__ void bem_zero_results_kernel(
    double* L1_re, double* L1_im, double* K1_re, double* K1_im,
    double* L2_re, double* L2_im, double* K2_re, double* K2_im, int N)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;
    L1_re[m] = L1_im[m] = K1_re[m] = K1_im[m] = 0.0;
    L2_re[m] = L2_im[m] = K2_re[m] = K2_im[m] = 0.0;
}

__global__ void bem_accum_L_vector_kernel(
    const double* phi1_re, const double* phi1_im,
    const double* phi2_re, const double* phi2_im,
    const double* f_p, const double* f_m,
    const double* jw_p, const double* jw_m,
    int N, int Nq, int comp, double ik_re, double ik_im,
    double* L1_re, double* L1_im,
    double* L2_re, double* L2_im)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;
    int half_pts = N * Nq;
    double a1r = 0.0, a1i = 0.0, a2r = 0.0, a2i = 0.0;
    for (int q = 0; q < Nq; q++) {
        int idx = m * Nq + q;
        int im = half_pts + idx;
        double cp = f_p[idx * 3 + comp] * jw_p[idx];
        double cm = f_m[idx * 3 + comp] * jw_m[idx];
        a1r += cp * phi1_re[idx] + cm * phi1_re[im];
        a1i += cp * phi1_im[idx] + cm * phi1_im[im];
        a2r += cp * phi2_re[idx] + cm * phi2_re[im];
        a2i += cp * phi2_im[idx] + cm * phi2_im[im];
    }
    double v1r = ik_re * a1r - ik_im * a1i;
    double v1i = ik_re * a1i + ik_im * a1r;
    double v2r = ik_re * a2r - ik_im * a2i;
    double v2i = ik_re * a2i + ik_im * a2r;
    if (comp == 0) {
        L1_re[m] = v1r;
        L1_im[m] = v1i;
        L2_re[m] = v2r;
        L2_im[m] = v2i;
    } else {
        L1_re[m] += v1r;
        L1_im[m] += v1i;
        L2_re[m] += v2r;
        L2_im[m] += v2i;
    }
}

__global__ void bem_accum_L_scalar_kernel(
    const double* phi1_re, const double* phi1_im,
    const double* phi2_re, const double* phi2_im,
    const double* div_p, const double* div_m,
    const double* jw_p, const double* jw_m,
    int N, int Nq, double iok_re, double iok_im,
    double* L1_re, double* L1_im,
    double* L2_re, double* L2_im)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;
    int half_pts = N * Nq;
    double p1r = 0.0, p1i = 0.0, m1r = 0.0, m1i = 0.0;
    double p2r = 0.0, p2i = 0.0, m2r = 0.0, m2i = 0.0;
    for (int q = 0; q < Nq; q++) {
        int idx = m * Nq + q;
        int im = half_pts + idx;
        p1r += jw_p[idx] * phi1_re[idx];
        p1i += jw_p[idx] * phi1_im[idx];
        m1r += jw_m[idx] * phi1_re[im];
        m1i += jw_m[idx] * phi1_im[im];
        p2r += jw_p[idx] * phi2_re[idx];
        p2i += jw_p[idx] * phi2_im[idx];
        m2r += jw_m[idx] * phi2_re[im];
        m2i += jw_m[idx] * phi2_im[im];
    }
    double a1r = div_p[m] * p1r + div_m[m] * m1r;
    double a1i = div_p[m] * p1i + div_m[m] * m1i;
    double a2r = div_p[m] * p2r + div_m[m] * m2r;
    double a2i = div_p[m] * p2i + div_m[m] * m2i;
    L1_re[m] -= iok_re * a1r - iok_im * a1i;
    L1_im[m] -= iok_re * a1i + iok_im * a1r;
    L2_re[m] -= iok_re * a2r - iok_im * a2i;
    L2_im[m] -= iok_re * a2i + iok_im * a2r;
}

__global__ void bem_accum_K_component_kernel(
    const double* grad1_re, const double* grad1_im,
    const double* grad2_re, const double* grad2_im,
    const double* f_p, const double* f_m,
    const double* jw_p, const double* jw_m,
    int N, int Nq, int comp,
    double* K1_re, double* K1_im,
    double* K2_re, double* K2_im)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;
    int half_pts = N * Nq;
    double a1r = 0.0, a1i = 0.0, a2r = 0.0, a2i = 0.0;
    for (int q = 0; q < Nq; q++) {
        int idx = m * Nq + q;
        for (int half = 0; half < 2; half++) {
            int base = (half == 0) ? idx : half_pts + idx;
            const double* f = (half == 0) ? f_p : f_m;
            const double* jw = (half == 0) ? jw_p : jw_m;
            double fx = f[idx * 3], fy = f[idx * 3 + 1], fz = f[idx * 3 + 2];
            double w = jw[idx];
            int x = base * 3, y = x + 1, z = x + 2;
            double c1r = 0.0, c1i = 0.0, c2r = 0.0, c2i = 0.0;
            if (comp == 0) {
                c1r = fy * grad1_re[z] - fz * grad1_re[y];
                c1i = fy * grad1_im[z] - fz * grad1_im[y];
                c2r = fy * grad2_re[z] - fz * grad2_re[y];
                c2i = fy * grad2_im[z] - fz * grad2_im[y];
            } else if (comp == 1) {
                c1r = fz * grad1_re[x] - fx * grad1_re[z];
                c1i = fz * grad1_im[x] - fx * grad1_im[z];
                c2r = fz * grad2_re[x] - fx * grad2_re[z];
                c2i = fz * grad2_im[x] - fx * grad2_im[z];
            } else {
                c1r = fx * grad1_re[y] - fy * grad1_re[x];
                c1i = fx * grad1_im[y] - fy * grad1_im[x];
                c2r = fx * grad2_re[y] - fy * grad2_re[x];
                c2i = fx * grad2_im[y] - fy * grad2_im[x];
            }
            a1r += w * c1r;
            a1i += w * c1i;
            a2r += w * c2r;
            a2i += w * c2i;
        }
    }
    if (comp == 0) {
        K1_re[m] = a1r;
        K1_im[m] = a1i;
        K2_re[m] = a2r;
        K2_im[m] = a2i;
    } else {
        K1_re[m] += a1r;
        K1_im[m] += a1i;
        K2_re[m] += a2r;
        K2_im[m] += a2i;
    }
}

__device__ inline double2 cx_mul(double ar, double ai, double br, double bi)
{
    return make_double2(ar * br - ai * bi, ar * bi + ai * br);
}

__device__ inline double2 cx_div(double ar, double ai, double br, double bi)
{
    const double den = br * br + bi * bi;
    return make_double2((ar * br + ai * bi) / den,
                        (ai * br - ar * bi) / den);
}

__global__ void bem_apply_corr_assemble_batch2_kernel(
    double* mv1_re, double* mv1_im,
    double* mv2_re, double* mv2_im,
    const double* x1_re, const double* x1_im,
    const double* x2_re, const double* x2_im,
    const int* row_ptr, const int* col_idx,
    const double* cLe_re, const double* cLe_im,
    const double* cKe_re, const double* cKe_im,
    const double* cLi_re, const double* cLi_im,
    const double* cKi_re, const double* cKi_im,
    const double* cI,
    int N, double eta_e_r, double eta_e_i, double eta_i_r, double eta_i_i,
    double unknown_m_scale, double row_h_scale_r, double row_h_scale_i,
    double int_op_sign, double k_identity, int n_form, double n_form_eps_int,
    double n_form_m_identity,
    double2* y1, double2* y2)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;

    const int slots = 10;
    double* out_re[2] = {mv1_re, mv2_re};
    double* out_im[2] = {mv1_im, mv2_im};
    const double* x_re[2] = {x1_re, x2_re};
    const double* x_im[2] = {x1_im, x2_im};

    for (int j = row_ptr[m]; j < row_ptr[m + 1]; j++) {
        int n = col_idx[j];
        for (int rhs = 0; rhs < 2; rhs++) {
            double Jr = x_re[rhs][n], Ji = x_im[rhs][n];
            double Mr = x_re[rhs][N + n] / unknown_m_scale;
            double Mi = x_im[rhs][N + n] / unknown_m_scale;
            double* r = out_re[rhs];
            double* im = out_im[rhs];
            int base = m;
            int s0 = 0 * N + base, s1 = 1 * N + base;
            int s2 = 2 * N + base, s3 = 3 * N + base;
            int s4 = 4 * N + base, s5 = 5 * N + base;
            int s6 = 6 * N + base, s7 = 7 * N + base;

            double ar, ai;
            ar = cLe_re[j] * Jr - cLe_im[j] * Ji; ai = cLe_re[j] * Ji + cLe_im[j] * Jr; r[s0] += ar; im[s0] += ai;
            ar = cLe_re[j] * Mr - cLe_im[j] * Mi; ai = cLe_re[j] * Mi + cLe_im[j] * Mr; r[s1] += ar; im[s1] += ai;
            ar = cKe_re[j] * Jr - cKe_im[j] * Ji; ai = cKe_re[j] * Ji + cKe_im[j] * Jr; r[s2] += ar; im[s2] += ai;
            ar = cKe_re[j] * Mr - cKe_im[j] * Mi; ai = cKe_re[j] * Mi + cKe_im[j] * Mr; r[s3] += ar; im[s3] += ai;
            ar = cLi_re[j] * Jr - cLi_im[j] * Ji; ai = cLi_re[j] * Ji + cLi_im[j] * Jr; r[s4] += ar; im[s4] += ai;
            ar = cLi_re[j] * Mr - cLi_im[j] * Mi; ai = cLi_re[j] * Mi + cLi_im[j] * Mr; r[s5] += ar; im[s5] += ai;
            ar = cKi_re[j] * Jr - cKi_im[j] * Ji; ai = cKi_re[j] * Ji + cKi_im[j] * Jr; r[s6] += ar; im[s6] += ai;
            ar = cKi_re[j] * Mr - cKi_im[j] * Mi; ai = cKi_re[j] * Mi + cKi_im[j] * Mr; r[s7] += ar; im[s7] += ai;
            if (n_form) {
                double mass = cI[j];
                r[8 * N + base] += mass * Jr;
                im[8 * N + base] += mass * Ji;
                r[9 * N + base] += mass * Mr;
                im[9 * N + base] += mass * Mi;
            }
        }
    }

    for (int rhs = 0; rhs < 2; rhs++) {
        double* r = out_re[rhs];
        double* im = out_im[rhs];
        int s0 = 0 * N + m, s1 = 1 * N + m;
        int s2 = 2 * N + m, s3 = 3 * N + m;
        int s4 = 4 * N + m, s5 = 5 * N + m;
        int s6 = 6 * N + m, s7 = 7 * N + m;
        double Mvar_r = x_re[rhs][N + m] / unknown_m_scale;
        double Mvar_i = x_im[rhs][N + m] / unknown_m_scale;
        double Jvar_r = x_re[rhs][m];
        double Jvar_i = x_im[rhs][m];
        int s8 = 8 * N + m, s9 = 9 * N + m;
        double IJ_r = n_form ? r[s8] : Jvar_r;
        double IJ_i = n_form ? im[s8] : Jvar_i;
        double IM_r = n_form ? r[s9] : Mvar_r;
        double IM_i = n_form ? im[s9] : Mvar_i;
        double ytop_r, ytop_i, ybot_r, ybot_i;
        if (n_form) {
            double2 le_m = cx_div(r[s1], im[s1], eta_e_r, eta_e_i);
            double2 li_m = cx_div(r[s5], im[s5], eta_i_r, eta_i_i);
            ytop_r = (r[s2] + int_op_sign * r[s6] + k_identity * IJ_r) +
                     le_m.x + int_op_sign * li_m.x;
            ytop_i = (im[s2] + int_op_sign * im[s6] + k_identity * IJ_i) +
                     le_m.y + int_op_sign * li_m.y;

            double2 eta_le_j = cx_mul(eta_e_r, eta_e_i, r[s0], im[s0]);
            double2 eta_li_j = cx_mul(eta_i_r, eta_i_i, r[s4], im[s4]);
            double inner_r = -(eta_le_j.x - n_form_eps_int * eta_li_j.x) +
                             (r[s3] - n_form_eps_int * r[s7] + n_form_m_identity * IM_r);
            double inner_i = -(eta_le_j.y - n_form_eps_int * eta_li_j.y) +
                             (im[s3] - n_form_eps_int * im[s7] + n_form_m_identity * IM_i);
            double2 ybot = cx_mul(row_h_scale_r, row_h_scale_i, inner_r, inner_i);
            ybot_r = ybot.x;
            ybot_i = ybot.y;
        } else {
            double2 eta_le_j = cx_mul(eta_e_r, eta_e_i, r[s0], im[s0]);
            double2 eta_li_j = cx_mul(eta_i_r, eta_i_i, r[s4], im[s4]);
            ytop_r = eta_le_j.x + int_op_sign * eta_li_j.x -
                     (r[s3] + int_op_sign * r[s7] + k_identity * Mvar_r);
            ytop_i = eta_le_j.y + int_op_sign * eta_li_j.y -
                     (im[s3] + int_op_sign * im[s7] + k_identity * Mvar_i);

            double2 le_m = cx_div(r[s1], im[s1], eta_e_r, eta_e_i);
            double2 li_m = cx_div(r[s5], im[s5], eta_i_r, eta_i_i);
            double inner_r = (r[s2] + int_op_sign * r[s6] + k_identity * Jvar_r) +
                             le_m.x + int_op_sign * li_m.x;
            double inner_i = (im[s2] + int_op_sign * im[s6] + k_identity * Jvar_i) +
                             le_m.y + int_op_sign * li_m.y;
            double2 ybot = cx_mul(row_h_scale_r, row_h_scale_i, inner_r, inner_i);
            ybot_r = ybot.x;
            ybot_i = ybot.y;
        }
        if (rhs == 0) {
            y1[m] = make_double2(ytop_r, ytop_i);
            y1[N + m] = make_double2(ybot_r, ybot_i);
        } else {
            y2[m] = make_double2(ytop_r, ytop_i);
            y2[N + m] = make_double2(ybot_r, ybot_i);
        }
    }
    (void)slots;
}

} // namespace

void BemFmmOperator::init(const RWG& rwg, const Mesh& mesh,
                            cdouble k_ext_, cdouble k_int_,
                            cdouble eta_ext_, cdouble eta_int_,
                            int quad_order, int fmm_digits, int max_leaf,
                            bool use_pfft_, bool use_spfft_)
{
    Timer timer;
    k_ext = k_ext_;
    k_int = k_int_;
    eta_ext = eta_ext_;
    eta_int = eta_int_;
#ifdef BEM_FMM_ONLY
    use_pfft = false;
    use_spfft = false;
    (void)use_pfft_;
    (void)use_spfft_;
#else
    use_pfft = use_pfft_;
    use_spfft = use_spfft_;
    if (use_spfft && !bem_env_flag_enabled("BEM_SPFFT_FORCE")) {
        printf("  [BEM-SurfPFFT] Disabled: direct cross-face P2P is slower than FMM; using FMM backend\n");
        use_spfft = false;
    }
#endif
    N = rwg.N;
    system_size = 2 * N;

    TriQuad quad = tri_quadrature(quad_order);
    Nq = quad.npts;

    printf("  [BEM-FMM] Init: N=%d, Nq=%d, k_ext=%.4f, k_int=%.4f+%.4fi\n",
           N, Nq, k_ext.real(), k_int.real(), k_int.imag());

    // Precompute quadrature points and RWG values
    qpts_p.resize(N * Nq * 3);
    qpts_m.resize(N * Nq * 3);
    f_p.resize(N * Nq * 3);
    f_m.resize(N * Nq * 3);
    div_p.resize(N);
    div_m.resize(N);
    jw_p.resize(N * Nq);
    jw_m.resize(N * Nq);

    for (int n = 0; n < N; n++) {
        // Plus half
        {
            Vec3 v0, v1, v2;
            mesh.tri_verts(rwg.tri_p[n], v0, v1, v2);
            double area = mesh.tri_area(rwg.tri_p[n]);
            double coeff = rwg.length[n] / (2.0 * area);
            Vec3 free_v = rwg.free_p[n];

            for (int q = 0; q < Nq; q++) {
                double l0 = 1.0 - quad.pts[q][0] - quad.pts[q][1];
                double l1 = quad.pts[q][0];
                double l2 = quad.pts[q][1];
                Vec3 r = v0 * l0 + v1 * l1 + v2 * l2;

                int idx = (n * Nq + q) * 3;
                qpts_p[idx]     = r.x;
                qpts_p[idx + 1] = r.y;
                qpts_p[idx + 2] = r.z;

                Vec3 fval = (r - free_v) * coeff;
                f_p[idx]     = fval.x;
                f_p[idx + 1] = fval.y;
                f_p[idx + 2] = fval.z;

                jw_p[n * Nq + q] = area * quad.wts[q];
            }

            div_p[n] = rwg.length[n] / area;
        }

        // Minus half
        {
            Vec3 v0, v1, v2;
            mesh.tri_verts(rwg.tri_m[n], v0, v1, v2);
            double area = mesh.tri_area(rwg.tri_m[n]);
            double coeff = rwg.length[n] / (2.0 * area);
            Vec3 free_v = rwg.free_m[n];

            for (int q = 0; q < Nq; q++) {
                double l0 = 1.0 - quad.pts[q][0] - quad.pts[q][1];
                double l1 = quad.pts[q][0];
                double l2 = quad.pts[q][1];
                Vec3 r = v0 * l0 + v1 * l1 + v2 * l2;

                int idx = (n * Nq + q) * 3;
                qpts_m[idx]     = r.x;
                qpts_m[idx + 1] = r.y;
                qpts_m[idx + 2] = r.z;

                // Minus half: negative sign
                Vec3 fval = (r - free_v) * (-coeff);
                f_m[idx]     = fval.x;
                f_m[idx + 1] = fval.y;
                f_m[idx + 2] = fval.z;

                jw_m[n * Nq + q] = area * quad.wts[q];
            }

            div_m[n] = -rwg.length[n] / area;
        }
    }

    // Combine all quad points: [plus_half; minus_half]
    int total_pts = 2 * N * Nq;
    all_pts.resize(total_pts * 3);
    memcpy(all_pts.data(), qpts_p.data(), N * Nq * 3 * sizeof(double));
    memcpy(all_pts.data() + N * Nq * 3, qpts_m.data(), N * Nq * 3 * sizeof(double));

    shared_fmm = (std::abs(k_int - k_ext) < 1e-10);

    if (use_spfft) {
#ifdef BEM_FMM_ONLY
        use_spfft = false;
        printf("  [BEM-FMM] FMM-only build: SurfPFFT unavailable, using FMM backend\n");
#else
        int n_tri = mesh.nt();
        std::vector<Vec3> grouped_normals;
        std::vector<double> grouped_d;
        std::vector<int> tri_face(n_tri, -1);
        double bbox = 0.0;
        for (const Vec3& v : mesh.verts)
            bbox = std::max(bbox, v.norm());
        double plane_tol = std::max(1e-10, 1e-8 * bbox);

        for (int t = 0; t < n_tri; t++) {
            Vec3 v0, v1, v2;
            mesh.tri_verts(t, v0, v1, v2);
            Vec3 n = (v1 - v0).cross(v2 - v0);
            double nm = n.norm();
            if (nm <= 1e-15)
                continue;
            n = n * (1.0 / nm);
            Vec3 c = (v0 + v1 + v2) * (1.0 / 3.0);
            double d = n.dot(c);

            int face = -1;
            for (int f = 0; f < (int)grouped_normals.size(); f++) {
                if (n.dot(grouped_normals[f]) > 1.0 - 1e-8 &&
                    std::abs(d - grouped_d[f]) < plane_tol) {
                    face = f;
                    break;
                }
            }
            if (face < 0) {
                face = (int)grouped_normals.size();
                grouped_normals.push_back(n);
                grouped_d.push_back(d);
            }
            tri_face[t] = face;
        }

        int n_face = (int)grouped_normals.size();
        if (n_face != 8)
            printf("  [BEM-SurfPFFT] Warning: detected %d planar faces (expected 8 for hex prism)\n", n_face);

        std::vector<double> face_normals(n_face * 3, 0.0);
        for (int f = 0; f < n_face; f++) {
            face_normals[f * 3 + 0] = grouped_normals[f].x;
            face_normals[f * 3 + 1] = grouped_normals[f].y;
            face_normals[f * 3 + 2] = grouped_normals[f].z;
        }

        std::vector<int> face_ids(total_pts, 0);
        for (int n = 0; n < N; n++) {
            int face_p = tri_face[rwg.tri_p[n]];
            int face_m = tri_face[rwg.tri_m[n]];
            if (face_p < 0) face_p = 0;
            if (face_m < 0) face_m = 0;

            for (int q = 0; q < Nq; q++) {
                face_ids[n * Nq + q] = face_p;
                face_ids[N * Nq + n * Nq + q] = face_m;
            }
        }

        printf("  [BEM-SurfPFFT] Building surface pFFT for k_ext...\n");
        spfft_ext.init(all_pts.data(), total_pts, face_ids.data(), n_face,
                       face_normals.data(), k_ext, fmm_digits);
        if (!shared_fmm) {
            printf("  [BEM-SurfPFFT] Building surface pFFT for k_int...\n");
            spfft_int.init(all_pts.data(), total_pts, face_ids.data(), n_face,
                           face_normals.data(), k_int, fmm_digits);
        }
#endif
    } else if (use_pfft) {
#ifdef BEM_FMM_ONLY
        use_pfft = false;
        printf("  [BEM-FMM] FMM-only build: pFFT unavailable, using FMM backend\n");
#else
        printf("  [BEM-pFFT] Building pFFT for k_ext...\n");
        pfft_ext.init(all_pts.data(), total_pts,
                      all_pts.data(), total_pts,
                      k_ext, fmm_digits, max_leaf);
        if (!shared_fmm) {
            printf("  [BEM-pFFT] Building pFFT for k_int...\n");
            pfft_int.init(all_pts.data(), total_pts,
                          all_pts.data(), total_pts,
                          k_int, fmm_digits, max_leaf);
        }
#endif
    }
    if (!use_spfft && !use_pfft) {
        printf("  [BEM-FMM] Building FMM for k_ext...\n");
        fmm_ext.init(all_pts.data(), total_pts,
                     all_pts.data(), total_pts,
                     k_ext, fmm_digits, max_leaf);
        if (!shared_fmm) {
            printf("  [BEM-FMM] Building FMM for k_int...\n");
            fmm_int.init(all_pts.data(), total_pts,
                         all_pts.data(), total_pts,
                         k_int, fmm_digits, max_leaf);
        }
    }

    // Precompute singular corrections
    printf("  [BEM-FMM] Computing singular corrections...\n");
    precompute_corrections(rwg, mesh, quad_order);
    init_device_workspace();

    printf("  [BEM-FMM] Init complete: %.1fs\n", timer.elapsed_s());
}

void BemFmmOperator::register_tmp_host_buffers()
{
    if (tmp_host_registered) return;
    register_host_vector(tmp_src_charges);
    register_host_vector(tmp2_src_charges);
    register_host_vector(tmp_phi);
    register_host_vector(tmp2_phi);
    for (int d = 0; d < 3; d++) {
        register_host_vector(tmp_grad[d]);
        register_host_vector(tmp2_grad[d]);
    }
    tmp_host_registered = true;
}

void BemFmmOperator::unregister_tmp_host_buffers()
{
    if (!tmp_host_registered) return;
    for (int d = 0; d < 3; d++) {
        unregister_host_vector(tmp2_grad[d]);
        unregister_host_vector(tmp_grad[d]);
    }
    unregister_host_vector(tmp2_phi);
    unregister_host_vector(tmp_phi);
    unregister_host_vector(tmp2_src_charges);
    unregister_host_vector(tmp_src_charges);
    tmp_host_registered = false;
}

void BemFmmOperator::ensure_host_workspace()
{
    if (!tmp_src_charges.empty())
        return;

    const int total_pts = 2 * N * Nq;
    tmp_src_charges.resize(total_pts);
    tmp_phi.resize(total_pts);
    for (int d = 0; d < 3; d++)
        tmp_grad[d].resize(total_pts * 3);
    tmp_L_result.resize(N);
    tmp_K_result.resize(N);
    mv_L_ext_J.resize(N);
    mv_L_ext_M.resize(N);
    mv_K_ext_J.resize(N);
    mv_K_ext_M.resize(N);
    mv_L_int_J.resize(N);
    mv_L_int_M.resize(N);
    mv_K_int_J.resize(N);
    mv_K_int_M.resize(N);
    tmp_M1_phys.resize(N);
    tmp_M2_phys.resize(N);
    register_tmp_host_buffers();
}

void BemFmmOperator::init_device_workspace()
{
    int half_pts = N * Nq;
    CUDA_CHECK(cudaMalloc(&d_f_p, half_pts * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f_m, half_pts * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_jw_p, half_pts * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_jw_m, half_pts * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_div_p, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_div_m, N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_f_p, f_p.data(), half_pts * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_m, f_m.data(), half_pts * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_jw_p, jw_p.data(), half_pts * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_jw_m, jw_m.data(), half_pts * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_div_p, div_p.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_div_m, div_m.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_x1_complex, N * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_x2_complex, N * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_x1_re, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x1_im, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x2_re, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x2_im, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_full_x1_complex, system_size * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_full_x2_complex, system_size * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_full_x1_re, system_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_full_x1_im, system_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_full_x2_re, system_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_full_x2_im, system_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_L1_re, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_L1_im, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_K1_re, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_K1_im, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_L2_re, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_L2_im, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_K2_re, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_K2_im, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out1_complex, N * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_out2_complex, N * sizeof(double2)));
    const int mv_slots_alloc = n_form ? 10 : 8;
    CUDA_CHECK(cudaMalloc(&d_mv1_re, (size_t)mv_slots_alloc * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mv1_im, (size_t)mv_slots_alloc * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mv2_re, (size_t)mv_slots_alloc * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mv2_im, (size_t)mv_slots_alloc * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y1_complex, system_size * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_y2_complex, system_size * sizeof(double2)));
    pinned_matvec_stage = false;
    if (use_pinned_matvec_stage()) {
        cudaError_t e1 = cudaHostAlloc(&h_full_x1_complex, (size_t)system_size * sizeof(double2), cudaHostAllocDefault);
        cudaError_t e2 = cudaHostAlloc(&h_full_x2_complex, (size_t)system_size * sizeof(double2), cudaHostAllocDefault);
        cudaError_t e3 = cudaHostAlloc(&h_y1_complex, (size_t)system_size * sizeof(double2), cudaHostAllocDefault);
        cudaError_t e4 = cudaHostAlloc(&h_y2_complex, (size_t)system_size * sizeof(double2), cudaHostAllocDefault);
        if (e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess && e4 == cudaSuccess) {
            pinned_matvec_stage = true;
        } else {
            fprintf(stderr, "  [BEM-FMM] pinned matvec staging unavailable; using pageable copies\n");
            if (h_full_x1_complex) cudaFreeHost(h_full_x1_complex);
            if (h_full_x2_complex) cudaFreeHost(h_full_x2_complex);
            if (h_y1_complex) cudaFreeHost(h_y1_complex);
            if (h_y2_complex) cudaFreeHost(h_y2_complex);
            h_full_x1_complex = nullptr;
            h_full_x2_complex = nullptr;
            h_y1_complex = nullptr;
            h_y2_complex = nullptr;
            cudaGetLastError();
        }
    }

    CUDA_CHECK(cudaMalloc(&d_corr_row_ptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_corr_col_idx, corr_nnz * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_corr_row_ptr, corr_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_corr_col_idx, corr_col_idx.data(), corr_nnz * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<double> tmp_re(corr_nnz), tmp_im(corr_nnz);
    auto upload_corr = [&](const std::vector<cdouble>& src, double** dst_re, double** dst_im) {
        for (int i = 0; i < corr_nnz; i++) {
            tmp_re[i] = src[i].real();
            tmp_im[i] = src[i].imag();
        }
        CUDA_CHECK(cudaMalloc(dst_re, corr_nnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(dst_im, corr_nnz * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(*dst_re, tmp_re.data(), corr_nnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*dst_im, tmp_im.data(), corr_nnz * sizeof(double), cudaMemcpyHostToDevice));
    };
    upload_corr(corr_L_ext_val, &d_corr_L_ext_re, &d_corr_L_ext_im);
    upload_corr(corr_K_ext_val, &d_corr_K_ext_re, &d_corr_K_ext_im);
    upload_corr(corr_L_int_val, &d_corr_L_int_re, &d_corr_L_int_im);
    upload_corr(corr_K_int_val, &d_corr_K_int_re, &d_corr_K_int_im);
    CUDA_CHECK(cudaMalloc(&d_corr_I, corr_nnz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_corr_I, corr_I_val.data(), corr_nnz * sizeof(double), cudaMemcpyHostToDevice));
}

void BemFmmOperator::free_device_workspace()
{
    cudaFree(d_f_p); d_f_p = nullptr;
    cudaFree(d_f_m); d_f_m = nullptr;
    cudaFree(d_jw_p); d_jw_p = nullptr;
    cudaFree(d_jw_m); d_jw_m = nullptr;
    cudaFree(d_div_p); d_div_p = nullptr;
    cudaFree(d_div_m); d_div_m = nullptr;
    cudaFree(d_x1_complex); d_x1_complex = nullptr;
    cudaFree(d_x2_complex); d_x2_complex = nullptr;
    cudaFree(d_x1_re); d_x1_re = nullptr;
    cudaFree(d_x1_im); d_x1_im = nullptr;
    cudaFree(d_x2_re); d_x2_re = nullptr;
    cudaFree(d_x2_im); d_x2_im = nullptr;
    cudaFree(d_full_x1_complex); d_full_x1_complex = nullptr;
    cudaFree(d_full_x2_complex); d_full_x2_complex = nullptr;
    cudaFree(d_full_x1_re); d_full_x1_re = nullptr;
    cudaFree(d_full_x1_im); d_full_x1_im = nullptr;
    cudaFree(d_full_x2_re); d_full_x2_re = nullptr;
    cudaFree(d_full_x2_im); d_full_x2_im = nullptr;
    cudaFree(d_L1_re); d_L1_re = nullptr;
    cudaFree(d_L1_im); d_L1_im = nullptr;
    cudaFree(d_K1_re); d_K1_re = nullptr;
    cudaFree(d_K1_im); d_K1_im = nullptr;
    cudaFree(d_L2_re); d_L2_re = nullptr;
    cudaFree(d_L2_im); d_L2_im = nullptr;
    cudaFree(d_K2_re); d_K2_re = nullptr;
    cudaFree(d_K2_im); d_K2_im = nullptr;
    cudaFree(d_out1_complex); d_out1_complex = nullptr;
    cudaFree(d_out2_complex); d_out2_complex = nullptr;
    cudaFree(d_mv1_re); d_mv1_re = nullptr;
    cudaFree(d_mv1_im); d_mv1_im = nullptr;
    cudaFree(d_mv2_re); d_mv2_re = nullptr;
    cudaFree(d_mv2_im); d_mv2_im = nullptr;
    cudaFree(d_y1_complex); d_y1_complex = nullptr;
    cudaFree(d_y2_complex); d_y2_complex = nullptr;
    if (h_full_x1_complex) cudaFreeHost(h_full_x1_complex);
    if (h_full_x2_complex) cudaFreeHost(h_full_x2_complex);
    if (h_y1_complex) cudaFreeHost(h_y1_complex);
    if (h_y2_complex) cudaFreeHost(h_y2_complex);
    h_full_x1_complex = nullptr;
    h_full_x2_complex = nullptr;
    h_y1_complex = nullptr;
    h_y2_complex = nullptr;
    pinned_matvec_stage = false;
    cudaFree(d_corr_row_ptr); d_corr_row_ptr = nullptr;
    cudaFree(d_corr_col_idx); d_corr_col_idx = nullptr;
    cudaFree(d_corr_L_ext_re); d_corr_L_ext_re = nullptr;
    cudaFree(d_corr_L_ext_im); d_corr_L_ext_im = nullptr;
    cudaFree(d_corr_K_ext_re); d_corr_K_ext_re = nullptr;
    cudaFree(d_corr_K_ext_im); d_corr_K_ext_im = nullptr;
    cudaFree(d_corr_L_int_re); d_corr_L_int_re = nullptr;
    cudaFree(d_corr_L_int_im); d_corr_L_int_im = nullptr;
    cudaFree(d_corr_K_int_re); d_corr_K_int_re = nullptr;
    cudaFree(d_corr_K_int_im); d_corr_K_int_im = nullptr;
    cudaFree(d_corr_I); d_corr_I = nullptr;
}

void BemFmmOperator::precompute_corrections(const RWG& rwg, const Mesh& mesh, int quad_order)
{
    // Build map: triangle index -> list of RWG half-info
    struct HalfInfo {
        int n;        // RWG index
        int half;     // 0=plus, 1=minus
        double div_val;
        int f_offset; // into f_p or f_m
        int jw_offset;
        double coeff;
        Vec3 free_v;
        int sign;
    };

    std::map<int, std::vector<HalfInfo>> tri_to_rwg;
    for (int n = 0; n < N; n++) {
        {
            HalfInfo info;
            info.n = n; info.half = 0; info.div_val = div_p[n];
            info.f_offset = n * Nq * 3; info.jw_offset = n * Nq;
            info.coeff = rwg.length[n] / (2.0 * rwg.area_p[n]);
            info.free_v = rwg.free_p[n]; info.sign = 1;
            tri_to_rwg[rwg.tri_p[n]].push_back(info);
        }
        {
            HalfInfo info;
            info.n = n; info.half = 1; info.div_val = div_m[n];
            info.f_offset = n * Nq * 3; info.jw_offset = n * Nq;
            info.coeff = rwg.length[n] / (2.0 * rwg.area_m[n]);
            info.free_v = rwg.free_m[n]; info.sign = -1;
            tri_to_rwg[rwg.tri_m[n]].push_back(info);
        }
    }

    const bool edge_delta_enabled =
        bem_env_flag_enabled("BEM_EDGE_CORRECTIONS", false);
    const bool explicit_local_delta =
        bem_env_flag_enabled("BEM_LOCAL_CORRECTIONS", false);
    const bool auto_local_delta =
        !bem_env_flag_enabled("BEM_NO_AUTO_LOCAL_CORRECTIONS", false) &&
        bem_env_flag_enabled("BEM_AUTO_LOCAL_CORRECTIONS", true);
    const bool local_delta_enabled = explicit_local_delta || auto_local_delta;
    const bool local_vertex_enabled =
        local_delta_enabled &&
        bem_env_flag_enabled("BEM_LOCAL_CORR_VERTEX", explicit_local_delta);

    std::map<std::pair<int, int>, std::vector<int>> edge_to_triangles;
    std::vector<std::vector<int>> vertex_to_triangles(mesh.nv());
    for (int t = 0; t < mesh.nt(); t++) {
        int a = mesh.tris[3 * t + 0];
        int b = mesh.tris[3 * t + 1];
        int c = mesh.tris[3 * t + 2];
        int tv[3] = {a, b, c};
        for (int e = 0; e < 3; e++) {
            if (tv[e] >= 0 && tv[e] < mesh.nv())
                vertex_to_triangles[tv[e]].push_back(t);
        }
        for (int e = 0; e < 3; e++) {
            int u = tv[e];
            int v = tv[(e + 1) % 3];
            if (u > v) std::swap(u, v);
            edge_to_triangles[std::make_pair(u, v)].push_back(t);
        }
    }

    std::set<std::pair<int, int>> local_tri_pairs;
    int edge_adjacent_tri_pairs = 0;
    int vertex_adjacent_tri_pairs = 0;
    int near_disjoint_tri_pairs = 0;
    for (const auto& edge_pair : edge_to_triangles) {
        const std::vector<int>& ts = edge_pair.second;
        if (ts.size() != 2)
            continue;
        edge_adjacent_tri_pairs++;
        if (edge_delta_enabled || local_delta_enabled) {
            int a = ts[0], b = ts[1];
            if (a > b) std::swap(a, b);
            local_tri_pairs.insert(std::make_pair(a, b));
        }
    }
    if (local_vertex_enabled) {
        for (const std::vector<int>& ts : vertex_to_triangles) {
            for (size_t i = 0; i < ts.size(); i++) {
                for (size_t j = i + 1; j < ts.size(); j++) {
                    int a = ts[i], b = ts[j];
                    if (a == b)
                        continue;
                    if (a > b) std::swap(a, b);
                    std::pair<int, int> p(a, b);
                    if (local_tri_pairs.find(p) == local_tri_pairs.end()) {
                        local_tri_pairs.insert(p);
                        vertex_adjacent_tri_pairs++;
                    }
                }
            }
        }
    }
    double near_factor = 0.0;
    if (local_delta_enabled) {
        near_factor = bem_env_double("BEM_LOCAL_CORR_NEAR_FACTOR",
                                     0.0);
    }
    int near_pair_limit = std::max(0, bem_env_int("BEM_LOCAL_CORR_NEAR_MAX_PAIRS", 64000));
    if (near_factor > 0.0 && near_pair_limit > 0) {
        struct TriGeom {
            Vec3 c;
            double h;
            int v[3];
        };
        const int nt = mesh.nt();
        std::vector<TriGeom> tg(nt);
        double edge_sum = 0.0;
        int edge_count = 0;
        for (int t = 0; t < nt; t++) {
            Vec3 a, b, c;
            mesh.tri_verts(t, a, b, c);
            double e0 = (b - a).norm();
            double e1 = (c - b).norm();
            double e2 = (a - c).norm();
            tg[t].c = (a + b + c) * (1.0 / 3.0);
            tg[t].h = std::max(e0, std::max(e1, e2));
            tg[t].v[0] = mesh.tris[3 * t + 0];
            tg[t].v[1] = mesh.tris[3 * t + 1];
            tg[t].v[2] = mesh.tris[3 * t + 2];
            edge_sum += e0 + e1 + e2;
            edge_count += 3;
        }
        double mean_edge = edge_count > 0 ? edge_sum / (double)edge_count : 1.0;
        double cell = std::max(1e-12, near_factor * mean_edge);
        std::map<std::tuple<int, int, int>, std::vector<int>> cells;
        auto cell_index = [&](const Vec3& p) {
            return std::make_tuple((int)std::floor(p.x / cell),
                                   (int)std::floor(p.y / cell),
                                   (int)std::floor(p.z / cell));
        };
        for (int t = 0; t < nt; t++)
            cells[cell_index(tg[t].c)].push_back(t);

        auto share_vertex = [&](int a, int b) {
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    if (tg[a].v[i] == tg[b].v[j])
                        return true;
            return false;
        };
        struct NearCandidate {
            double metric;
            int a;
            int b;
        };
        std::vector<NearCandidate> near_candidates;
        near_candidates.reserve((size_t)near_pair_limit * 2);
        for (int a = 0; a < nt; a++) {
            int ix = (int)std::floor(tg[a].c.x / cell);
            int iy = (int)std::floor(tg[a].c.y / cell);
            int iz = (int)std::floor(tg[a].c.z / cell);
            int reach = std::max(1, (int)std::ceil((near_factor * tg[a].h) / cell));
            for (int dx = -reach; dx <= reach; dx++) {
                for (int dy = -reach; dy <= reach; dy++) {
                    for (int dz = -reach; dz <= reach; dz++) {
                        auto it = cells.find(std::make_tuple(ix + dx, iy + dy, iz + dz));
                        if (it == cells.end())
                            continue;
                        for (int b : it->second) {
                            if (b <= a)
                                continue;
                            if (share_vertex(a, b))
                                continue;
                            double thresh = near_factor * std::max(tg[a].h, tg[b].h);
                            double dist2 = (tg[a].c - tg[b].c).norm2();
                            if (dist2 > thresh * thresh)
                                continue;
                            near_candidates.push_back({dist2 / (thresh * thresh), a, b});
                        }
                    }
                }
            }
        }
        std::sort(near_candidates.begin(), near_candidates.end(),
                  [](const NearCandidate& x, const NearCandidate& y) {
                      return x.metric < y.metric;
                  });
        for (const NearCandidate& cand : near_candidates) {
            std::pair<int, int> p(cand.a, cand.b);
            if (local_tri_pairs.find(p) != local_tri_pairs.end())
                continue;
            local_tri_pairs.insert(p);
            near_disjoint_tri_pairs++;
            if (near_disjoint_tri_pairs >= near_pair_limit)
                break;
        }
    }

    // Step 1: Determine sparsity pattern.  Self-triangle entries get the
    // analytic singular correction below.  Extra adjacent entries are inserted
    // only when a local correction mode is explicitly enabled; otherwise they
    // would be zero-valued CSR work in every matvec.
    std::vector<std::vector<int>> row_cols(N);
    for (auto& pair : tri_to_rwg) {
        const std::vector<HalfInfo>& rwg_list = pair.second;
        for (const HalfInfo& mi : rwg_list) {
            for (const HalfInfo& ni : rwg_list) {
                row_cols[mi.n].push_back(ni.n);
            }
        }
    }
    for (const auto& tri_pair : local_tri_pairs) {
        const std::vector<HalfInfo>& a = tri_to_rwg[tri_pair.first];
        const std::vector<HalfInfo>& b = tri_to_rwg[tri_pair.second];
        for (const HalfInfo& mi : a)
            for (const HalfInfo& ni : b) {
                row_cols[mi.n].push_back(ni.n);
                row_cols[ni.n].push_back(mi.n);
            }
    }
    // Sort and deduplicate each row
    for (int m = 0; m < N; m++) {
        std::sort(row_cols[m].begin(), row_cols[m].end());
        row_cols[m].erase(std::unique(row_cols[m].begin(), row_cols[m].end()), row_cols[m].end());
    }

    // Build CSR structure
    corr_row_ptr.resize(N + 1, 0);
    for (int m = 0; m < N; m++)
        corr_row_ptr[m + 1] = corr_row_ptr[m] + (int)row_cols[m].size();
    corr_nnz = corr_row_ptr[N];

    corr_col_idx.resize(corr_nnz);
    for (int m = 0; m < N; m++)
        for (int j = 0; j < (int)row_cols[m].size(); j++)
            corr_col_idx[corr_row_ptr[m] + j] = row_cols[m][j];

    // Build reverse lookup: for row m, col n -> position in values array
    // We'll use binary search since cols are sorted
    corr_L_ext_val.assign(corr_nnz, cdouble(0));
    corr_K_ext_val.assign(corr_nnz, cdouble(0));
    corr_L_int_val.assign(corr_nnz, cdouble(0));
    corr_K_int_val.assign(corr_nnz, cdouble(0));
    corr_I_val.assign(corr_nnz, 0.0);

    printf("  [BEM-FMM] Corrections: nnz=%d (%.1f per row, %.3f%% of %lld)\n",
           corr_nnz, (double)corr_nnz / N, 100.0 * corr_nnz / ((long long)N * N), (long long)N * N);

    // Step 2: Compute correction values
    double inv4pi = 1.0 / (4.0 * M_PI);

    auto csr_pos = [&](int m, int n_idx) -> int {
        const int* col_begin = &corr_col_idx[corr_row_ptr[m]];
        const int* col_end   = &corr_col_idx[corr_row_ptr[m + 1]];
        const int* it = std::lower_bound(col_begin, col_end, n_idx);
        return corr_row_ptr[m] + (int)(it - col_begin);
    };

    auto direct_pair_geom = [&](const HalfInfo& mi,
                                const Vec3& mv0, const Vec3& mv1, const Vec3& mv2,
                                double area_m,
                                const HalfInfo& ni,
                                const Vec3& nv0, const Vec3& nv1, const Vec3& nv2,
                                double area_n,
                           const TriQuad& tq, cdouble kv,
                           cdouble& L, cdouble& K) {
        cdouble ik = cdouble(0, 1) * kv;
        cdouble iok = cdouble(0, 1) / kv;
        L = cdouble(0);
        K = cdouble(0);
        for (int iq = 0; iq < tq.npts; iq++) {
            double ml0 = 1.0 - tq.pts[iq][0] - tq.pts[iq][1];
            Vec3 rp = mv0 * ml0 + mv1 * tq.pts[iq][0] + mv2 * tq.pts[iq][1];
            Vec3 fm = (rp - mi.free_v) * (mi.sign * mi.coeff);
            double jw_mi = area_m * tq.wts[iq];
            for (int jq = 0; jq < tq.npts; jq++) {
                double nl0 = 1.0 - tq.pts[jq][0] - tq.pts[jq][1];
                Vec3 rq = nv0 * nl0 + nv1 * tq.pts[jq][0] + nv2 * tq.pts[jq][1];
                Vec3 fn = (rq - ni.free_v) * (ni.sign * ni.coeff);
                double jw_nj = area_n * tq.wts[jq];
                Vec3 diff = rp - rq;
                double R = diff.norm();
                if (R < 1e-14)
                    continue;
                cdouble G = std::exp(ik * R) * inv4pi / R;
                double jw_prod = jw_mi * jw_nj;
                double fdot = fm.x * fn.x + fm.y * fn.y + fm.z * fn.z;
                L += (ik * fdot - iok * mi.div_val * ni.div_val) * G * jw_prod;
                cdouble grad_scalar = G * (ik - 1.0 / R) / R;
                Vec3 cross = diff.cross(fn);
                double kdot = fm.x * cross.x + fm.y * cross.y + fm.z * cross.z;
                K += grad_scalar * kdot * jw_prod;
            }
        }
    };

    auto direct_pair = [&](const HalfInfo& mi, int tri_m,
                           const HalfInfo& ni, int tri_n,
                           const TriQuad& tq, cdouble kv,
                           cdouble& L, cdouble& K) {
        Vec3 mv0, mv1, mv2, nv0, nv1, nv2;
        mesh.tri_verts(tri_m, mv0, mv1, mv2);
        mesh.tri_verts(tri_n, nv0, nv1, nv2);
        double area_m = (mi.half == 0) ? rwg.area_p[mi.n] : rwg.area_m[mi.n];
        double area_n = (ni.half == 0) ? rwg.area_p[ni.n] : rwg.area_m[ni.n];
        direct_pair_geom(mi, mv0, mv1, mv2, area_m, ni, nv0, nv1, nv2, area_n, tq, kv, L, K);
    };

    struct LocalTri {
        Vec3 a, b, c;
        double area;
    };
    auto split_local_tris = [](std::vector<LocalTri>& tris) {
        std::vector<LocalTri> out;
        out.reserve(tris.size() * 4);
        for (const LocalTri& t : tris) {
            Vec3 ab = (t.a + t.b) * 0.5;
            Vec3 bc = (t.b + t.c) * 0.5;
            Vec3 ca = (t.c + t.a) * 0.5;
            double qarea = t.area * 0.25;
            out.push_back({t.a, ab, ca, qarea});
            out.push_back({ab, t.b, bc, qarea});
            out.push_back({ca, bc, t.c, qarea});
            out.push_back({ab, bc, ca, qarea});
        }
        tris.swap(out);
    };
    auto make_local_tris = [&](int tri, int half, int rwg_idx, int levels) {
        Vec3 v0, v1, v2;
        mesh.tri_verts(tri, v0, v1, v2);
        double area = (half == 0) ? rwg.area_p[rwg_idx] : rwg.area_m[rwg_idx];
        std::vector<LocalTri> tris;
        tris.push_back({v0, v1, v2, area});
        for (int l = 0; l < levels; l++)
            split_local_tris(tris);
        return tris;
    };
    auto direct_pair_subdiv = [&](const HalfInfo& mi, int tri_m,
                                  const HalfInfo& ni, int tri_n,
                                  const TriQuad& tq, cdouble kv,
                                  int levels, cdouble& L, cdouble& K) {
        L = cdouble(0);
        K = cdouble(0);
        std::vector<LocalTri> mt = make_local_tris(tri_m, mi.half, mi.n, levels);
        std::vector<LocalTri> nt = make_local_tris(tri_n, ni.half, ni.n, levels);
        for (const LocalTri& a : mt) {
            for (const LocalTri& b : nt) {
                cdouble Lt, Kt;
                direct_pair_geom(mi, a.a, a.b, a.c, a.area,
                                 ni, b.a, b.b, b.c, b.area,
                                 tq, kv, Lt, Kt);
                L += Lt;
                K += Kt;
            }
        }
    };

    for (auto& pair : tri_to_rwg) {
        const std::vector<HalfInfo>& rwg_list = pair.second;

        Vec3 v0, v1, v2;
        mesh.tri_verts(pair.first, v0, v1, v2);

        TriQuad tq = tri_quadrature(quad_order);
        std::vector<Vec3> qpts(Nq);
        for (int q = 0; q < Nq; q++) {
            double l0 = 1.0 - tq.pts[q][0] - tq.pts[q][1];
            qpts[q] = v0 * l0 + v1 * tq.pts[q][0] + v2 * tq.pts[q][1];
        }

        std::vector<double> R(Nq * Nq);
        for (int i = 0; i < Nq; i++)
            for (int j = 0; j < Nq; j++)
                R[i*Nq+j] = (qpts[i] - qpts[j]).norm();

        std::vector<double> P_anal(Nq);
        std::vector<Vec3>   V_anal(Nq);
        for (int iq = 0; iq < Nq; iq++) {
            P_anal[iq] = potential_integral_triangle(qpts[iq], v0, v1, v2);
            V_anal[iq] = vector_potential_integral_triangle(qpts[iq], v0, v1, v2, tq);
        }

        cdouble k_vals[2] = {k_ext, k_int};
        cdouble* val_L_ptrs[2] = {corr_L_ext_val.data(), corr_L_int_val.data()};
        cdouble* val_K_ptrs[2] = {corr_K_ext_val.data(), corr_K_int_val.data()};

        for (int ki = 0; ki < 2; ki++) {
            cdouble kv = k_vals[ki];
            cdouble ik = cdouble(0, 1) * kv;
            cdouble iok = cdouble(0, 1) / kv;
            cdouble* vL = val_L_ptrs[ki];
            cdouble* vK = val_K_ptrs[ki];

            std::vector<cdouble> DG(Nq * Nq, cdouble(0));
            std::vector<cdouble> gradG_scalar(Nq * Nq, cdouble(0));

            for (int i = 0; i < Nq; i++) {
                for (int j = 0; j < Nq; j++) {
                    double Rij = R[i*Nq+j];
                    if (Rij > 1e-12) {
                        DG[i*Nq+j] = -1.0 / (4.0 * M_PI * Rij);
                        cdouble G_full = std::exp(ik * Rij) / (4.0 * M_PI * Rij);
                        gradG_scalar[i*Nq+j] = G_full * (ik - 1.0/Rij) / Rij;
                    } else {
                        DG[i*Nq+j] = ik / (4.0 * M_PI);
                    }
                }
            }

            for (const HalfInfo& mi : rwg_list) {
                for (const HalfInfo& ni : rwg_list) {
                    int m = mi.n, n_idx = ni.n;
                    const double* m_f = (mi.half == 0) ? &f_p[mi.f_offset] : &f_m[mi.f_offset];
                    const double* n_f = (ni.half == 0) ? &f_p[ni.f_offset] : &f_m[ni.f_offset];
                    const double* m_jw = (mi.half == 0) ? &jw_p[mi.jw_offset] : &jw_m[mi.jw_offset];
                    const double* n_jw = (ni.half == 0) ? &jw_p[ni.jw_offset] : &jw_m[ni.jw_offset];

                    int pos = csr_pos(m, n_idx);

                    // L correction
                    double mass_corr = 0.0;
                    cdouble DL_vec(0), DL_scl(0);
                    for (int i = 0; i < Nq; i++) {
                        double f_self = m_f[i*3]*n_f[i*3] +
                                        m_f[i*3+1]*n_f[i*3+1] +
                                        m_f[i*3+2]*n_f[i*3+2];
                        mass_corr += f_self * m_jw[i];
                        for (int j = 0; j < Nq; j++) {
                            double jw_prod = m_jw[i] * n_jw[j];
                            double f_dot = m_f[i*3]*n_f[j*3] + m_f[i*3+1]*n_f[j*3+1] + m_f[i*3+2]*n_f[j*3+2];
                            DL_vec += ik * f_dot * DG[i*Nq+j] * jw_prod;
                            DL_scl += -iok * mi.div_val * ni.div_val * DG[i*Nq+j] * jw_prod;
                        }
                    }

                    cdouble anal_vec(0), anal_scl(0);
                    for (int i = 0; i < Nq; i++) {
                        Vec3 fn_over_R = (V_anal[i] - ni.free_v * P_anal[i]) * (ni.sign * ni.coeff);
                        double f_dot_fn = m_f[i*3]*fn_over_R.x + m_f[i*3+1]*fn_over_R.y + m_f[i*3+2]*fn_over_R.z;
                        anal_vec += ik * f_dot_fn * m_jw[i] * inv4pi;
                        anal_scl += -iok * mi.div_val * ni.div_val * P_anal[i] * m_jw[i] * inv4pi;
                    }

                    vL[pos] += DL_vec + DL_scl + anal_vec + anal_scl;
                    corr_I_val[pos] += mass_corr;

                    // K correction
                    cdouble K_corr(0);
                    for (int i = 0; i < Nq; i++) {
                        for (int j = 0; j < Nq; j++) {
                            if (R[i*Nq+j] < 1e-12) continue;
                            double jw_prod = m_jw[i] * n_jw[j];
                            Vec3 diff = qpts[i] - qpts[j];
                            Vec3 fn_j(n_f[j*3], n_f[j*3+1], n_f[j*3+2]);
                            Vec3 cross = diff.cross(fn_j);
                            double dot_f_cross = m_f[i*3]*cross.x + m_f[i*3+1]*cross.y + m_f[i*3+2]*cross.z;
                            K_corr += gradG_scalar[i*Nq+j] * dot_f_cross * jw_prod;
                        }
                    }
                    vK[pos] -= K_corr;
                }
            }
        }
    }

    int local_corr_entries = 0;
    if (edge_delta_enabled || local_delta_enabled) {
        TriQuad q_low = tri_quadrature(quad_order);
        TriQuad q_high = local_delta_enabled ? tri_quadrature(quad_order) : tri_quadrature(13);
        int subdiv_levels = std::max(0, bem_env_int("BEM_LOCAL_CORR_SUBDIV", 1));
        subdiv_levels = std::min(subdiv_levels, 3);
        cdouble k_vals[2] = {k_ext, k_int};
        cdouble* val_L_ptrs[2] = {corr_L_ext_val.data(), corr_L_int_val.data()};
        cdouble* val_K_ptrs[2] = {corr_K_ext_val.data(), corr_K_int_val.data()};

        auto add_local_delta = [&](int tri_m, int tri_n) {
            const std::vector<HalfInfo>& test_list = tri_to_rwg[tri_m];
            const std::vector<HalfInfo>& src_list = tri_to_rwg[tri_n];
            for (const HalfInfo& mi : test_list) {
                for (const HalfInfo& ni : src_list) {
                    int pos = csr_pos(mi.n, ni.n);
                    for (int ki = 0; ki < 2; ki++) {
                        cdouble L_low, K_low, L_high, K_high;
                        direct_pair(mi, tri_m, ni, tri_n, q_low, k_vals[ki], L_low, K_low);
                        if (local_delta_enabled)
                            direct_pair_subdiv(mi, tri_m, ni, tri_n, q_high, k_vals[ki], subdiv_levels, L_high, K_high);
                        else
                            direct_pair(mi, tri_m, ni, tri_n, q_high, k_vals[ki], L_high, K_high);
                        val_L_ptrs[ki][pos] += L_high - L_low;
                        val_K_ptrs[ki][pos] += K_high - K_low;
                    }
                    local_corr_entries++;
                }
            }
        };

        for (const auto& tri_pair : local_tri_pairs) {
            add_local_delta(tri_pair.first, tri_pair.second);
            add_local_delta(tri_pair.second, tri_pair.first);
        }
        if (local_corr_entries > 0 && local_delta_enabled) {
            printf("  [BEM-FMM] Local subdivided q%d corrections%s: tri_pairs=%zu (edge=%d vertex_extra=%d), subdiv=%d, half-pair entries=%d\n",
                   quad_order, auto_local_delta && !explicit_local_delta ? " [auto edge]" : "",
                   local_tri_pairs.size(), edge_adjacent_tri_pairs,
                   vertex_adjacent_tri_pairs, subdiv_levels, local_corr_entries);
            if (near_disjoint_tri_pairs > 0) {
                printf("  [BEM-FMM] Local near-disjoint corrections: pairs=%d, factor=%.3g, max_pairs=%d\n",
                       near_disjoint_tri_pairs, near_factor, near_pair_limit);
            }
        }
    }
    if (local_corr_entries > 0 && edge_delta_enabled && !local_delta_enabled) {
        printf("  [BEM-FMM] Edge-adjacent q13-q%d corrections: tri_pairs=%d, half-pair entries=%d\n",
               quad_order, edge_adjacent_tri_pairs, local_corr_entries);
    }
}

void BemFmmOperator::L_operator(const cdouble* x, cdouble kv, HelmholtzFMM& fmm, cdouble* result)
{
    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;

    int total_pts = 2 * N * Nq;

    // Zero result
    std::fill_n(result, N, cdouble(0.0, 0.0));

    // --- Vector part: ik * integral(f_m . f_n . G) ---
    for (int d = 0; d < 3; d++) {
        std::fill_n(tmp_src_charges.data(), total_pts, cdouble(0.0, 0.0));

        for (int n = 0; n < N; n++) {
            cdouble xn = x[n];
            for (int q = 0; q < Nq; q++) {
                int idx = n * Nq + q;
                tmp_src_charges[idx] = f_p[idx*3 + d] * jw_p[idx] * xn;
                tmp_src_charges[N*Nq + idx] = f_m[idx*3 + d] * jw_m[idx] * xn;
            }
        }

        fmm.evaluate(tmp_src_charges.data(), tmp_phi.data());

        for (int m = 0; m < N; m++) {
            cdouble acc(0);
            for (int q = 0; q < Nq; q++) {
                int idx = m * Nq + q;
                acc += f_p[idx*3 + d] * jw_p[idx] * tmp_phi[idx];
                acc += f_m[idx*3 + d] * jw_m[idx] * tmp_phi[N*Nq + idx];
            }
            result[m] += ik * acc;
        }
    }

    // --- Scalar part: -(i/k) * integral(div_f_m * div_f_n * G) ---
    {
        std::fill_n(tmp_src_charges.data(), total_pts, cdouble(0.0, 0.0));

        for (int n = 0; n < N; n++) {
            cdouble xn = x[n];
            for (int q = 0; q < Nq; q++) {
                int idx = n * Nq + q;
                tmp_src_charges[idx] = div_p[n] * jw_p[idx] * xn;
                tmp_src_charges[N*Nq + idx] = div_m[n] * jw_m[idx] * xn;
            }
        }

        fmm.evaluate(tmp_src_charges.data(), tmp_phi.data());

        for (int m = 0; m < N; m++) {
            cdouble acc_p(0), acc_m(0);
            for (int q = 0; q < Nq; q++) {
                int idx = m * Nq + q;
                acc_p += jw_p[idx] * tmp_phi[idx];
                acc_m += jw_m[idx] * tmp_phi[N*Nq + idx];
            }
            result[m] -= iok * (div_p[m] * acc_p + div_m[m] * acc_m);
        }
    }
}

void BemFmmOperator::K_operator(const cdouble* x, cdouble kv, HelmholtzFMM& fmm, cdouble* result)
{
    int total_pts = 2 * N * Nq;
    std::fill_n(result, N, cdouble(0.0, 0.0));

    // For each source component k, compute gradient of potential
    for (int kc = 0; kc < 3; kc++) {
        // Source charges = f_n^k * jw * x[n]
        std::fill_n(tmp_src_charges.data(), total_pts, cdouble(0.0, 0.0));
        for (int n = 0; n < N; n++) {
            cdouble xn = x[n];
            for (int q = 0; q < Nq; q++) {
                int idx = n * Nq + q;
                tmp_src_charges[idx] = f_p[idx*3 + kc] * jw_p[idx] * xn;
                tmp_src_charges[N*Nq + idx] = f_m[idx*3 + kc] * jw_m[idx] * xn;
            }
        }

        // FMM gradient evaluation into pre-allocated buffer
        fmm.evaluate_gradient(tmp_src_charges.data(), tmp_grad[kc].data());
    }

    // Assemble curl:
    // curl_x = dPhi_z/dy - dPhi_y/dz = gP[2][:,1] - gP[1][:,2]
    // curl_y = dPhi_x/dz - dPhi_z/dx = gP[0][:,2] - gP[2][:,0]
    // curl_z = dPhi_y/dx - dPhi_x/dy = gP[1][:,0] - gP[0][:,1]
    // gP[k][i*3+j] = dPhi_k/dx_j at point i

    for (int m = 0; m < N; m++) {
        cdouble acc(0);

        // Plus half
        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            int i = idx;  // point index in plus half

            cdouble curl_x = tmp_grad[2][i*3+1] - tmp_grad[1][i*3+2];
            cdouble curl_y = tmp_grad[0][i*3+2] - tmp_grad[2][i*3+0];
            cdouble curl_z = tmp_grad[1][i*3+0] - tmp_grad[0][i*3+1];

            double fx = f_p[idx*3], fy = f_p[idx*3+1], fz = f_p[idx*3+2];
            acc += jw_p[idx] * (fx * curl_x + fy * curl_y + fz * curl_z);
        }

        // Minus half
        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            int i = N*Nq + idx;  // point index in minus half

            cdouble curl_x = tmp_grad[2][i*3+1] - tmp_grad[1][i*3+2];
            cdouble curl_y = tmp_grad[0][i*3+2] - tmp_grad[2][i*3+0];
            cdouble curl_z = tmp_grad[1][i*3+0] - tmp_grad[0][i*3+1];

            double fx = f_m[idx*3], fy = f_m[idx*3+1], fz = f_m[idx*3+2];
            acc += jw_m[idx] * (fx * curl_x + fy * curl_y + fz * curl_z);
        }

        result[m] = acc;
    }
}

void BemFmmOperator::LK_combined(const cdouble* x, cdouble kv, HelmholtzFMM& fmm,
                                  cdouble* L_result, cdouble* K_result)
{
    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;

    int total_pts = 2 * N * Nq;

    std::fill_n(L_result, N, cdouble(0.0, 0.0));
    std::fill_n(K_result, N, cdouble(0.0, 0.0));

    // --- Vector part: combined potential (for L) + gradient (for K) in one FMM pass ---
    for (int d = 0; d < 3; d++) {
        std::fill_n(tmp_src_charges.data(), total_pts, cdouble(0.0, 0.0));

        for (int n = 0; n < N; n++) {
            cdouble xn = x[n];
            for (int q = 0; q < Nq; q++) {
                int idx = n * Nq + q;
                tmp_src_charges[idx] = f_p[idx*3 + d] * jw_p[idx] * xn;
                tmp_src_charges[N*Nq + idx] = f_m[idx*3 + d] * jw_m[idx] * xn;
            }
        }

        // Single FMM pass: get both potential and gradient
        fmm.evaluate_pot_grad(tmp_src_charges.data(), tmp_phi.data(), tmp_grad[d].data());

        // Accumulate L vector part from potential
        for (int m = 0; m < N; m++) {
            cdouble acc(0);
            for (int q = 0; q < Nq; q++) {
                int idx = m * Nq + q;
                acc += f_p[idx*3 + d] * jw_p[idx] * tmp_phi[idx];
                acc += f_m[idx*3 + d] * jw_m[idx] * tmp_phi[N*Nq + idx];
            }
            L_result[m] += ik * acc;
        }
    }

    // --- L scalar part: potential only (no gradient needed) ---
    {
        std::fill_n(tmp_src_charges.data(), total_pts, cdouble(0.0, 0.0));

        for (int n = 0; n < N; n++) {
            cdouble xn = x[n];
            for (int q = 0; q < Nq; q++) {
                int idx = n * Nq + q;
                tmp_src_charges[idx] = div_p[n] * jw_p[idx] * xn;
                tmp_src_charges[N*Nq + idx] = div_m[n] * jw_m[idx] * xn;
            }
        }

        fmm.evaluate(tmp_src_charges.data(), tmp_phi.data());

        for (int m = 0; m < N; m++) {
            cdouble acc_p(0), acc_m(0);
            for (int q = 0; q < Nq; q++) {
                int idx = m * Nq + q;
                acc_p += jw_p[idx] * tmp_phi[idx];
                acc_m += jw_m[idx] * tmp_phi[N*Nq + idx];
            }
            L_result[m] -= iok * (div_p[m] * acc_p + div_m[m] * acc_m);
        }
    }

    // --- K: assemble curl from gradients computed above ---
    // curl_x = dPhi_z/dy - dPhi_y/dz
    // curl_y = dPhi_x/dz - dPhi_z/dx
    // curl_z = dPhi_y/dx - dPhi_x/dy
    for (int m = 0; m < N; m++) {
        cdouble acc(0);

        // Plus half
        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            int i = idx;

            cdouble curl_x = tmp_grad[2][i*3+1] - tmp_grad[1][i*3+2];
            cdouble curl_y = tmp_grad[0][i*3+2] - tmp_grad[2][i*3+0];
            cdouble curl_z = tmp_grad[1][i*3+0] - tmp_grad[0][i*3+1];

            double fx = f_p[idx*3], fy = f_p[idx*3+1], fz = f_p[idx*3+2];
            acc += jw_p[idx] * (fx * curl_x + fy * curl_y + fz * curl_z);
        }

        // Minus half
        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            int i = N*Nq + idx;

            cdouble curl_x = tmp_grad[2][i*3+1] - tmp_grad[1][i*3+2];
            cdouble curl_y = tmp_grad[0][i*3+2] - tmp_grad[2][i*3+0];
            cdouble curl_z = tmp_grad[1][i*3+0] - tmp_grad[0][i*3+1];

            double fx = f_m[idx*3], fy = f_m[idx*3+1], fz = f_m[idx*3+2];
            acc += jw_m[idx] * (fx * curl_x + fy * curl_y + fz * curl_z);
        }

        K_result[m] = acc;
    }
}

#ifndef BEM_FMM_ONLY
void BemFmmOperator::LK_combined(const cdouble* x, cdouble kv, HelmholtzPFFT& pf,
                                 cdouble* L_result, cdouble* K_result)
{
    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;
    int total_pts = 2 * N * Nq;

    std::fill_n(L_result, N, cdouble(0.0, 0.0));
    std::fill_n(K_result, N, cdouble(0.0, 0.0));

    for (int d = 0; d < 3; d++) {
        std::fill_n(tmp_src_charges.data(), total_pts, cdouble(0.0, 0.0));
        for (int n = 0; n < N; n++) {
            cdouble xn = x[n];
            for (int q = 0; q < Nq; q++) {
                int idx = n * Nq + q;
                tmp_src_charges[idx] = f_p[idx * 3 + d] * jw_p[idx] * xn;
                tmp_src_charges[N * Nq + idx] = f_m[idx * 3 + d] * jw_m[idx] * xn;
            }
        }
        pf.evaluate_pot_grad(tmp_src_charges.data(), tmp_phi.data(), tmp_grad[d].data());
        for (int m = 0; m < N; m++) {
            cdouble acc(0);
            for (int q = 0; q < Nq; q++) {
                int idx = m * Nq + q;
                acc += f_p[idx * 3 + d] * jw_p[idx] * tmp_phi[idx];
                acc += f_m[idx * 3 + d] * jw_m[idx] * tmp_phi[N * Nq + idx];
            }
            L_result[m] += ik * acc;
        }
    }

    std::fill_n(tmp_src_charges.data(), total_pts, cdouble(0.0, 0.0));
    for (int n = 0; n < N; n++) {
        cdouble xn = x[n];
        for (int q = 0; q < Nq; q++) {
            int idx = n * Nq + q;
            tmp_src_charges[idx] = div_p[n] * jw_p[idx] * xn;
            tmp_src_charges[N * Nq + idx] = div_m[n] * jw_m[idx] * xn;
        }
    }
    pf.evaluate(tmp_src_charges.data(), tmp_phi.data());
    for (int m = 0; m < N; m++) {
        cdouble acc_p(0), acc_m(0);
        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            acc_p += jw_p[idx] * tmp_phi[idx];
            acc_m += jw_m[idx] * tmp_phi[N * Nq + idx];
        }
        L_result[m] -= iok * (div_p[m] * acc_p + div_m[m] * acc_m);
    }

    for (int m = 0; m < N; m++) {
        cdouble acc(0);
        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            int i = idx;
            cdouble curl_x = tmp_grad[2][i * 3 + 1] - tmp_grad[1][i * 3 + 2];
            cdouble curl_y = tmp_grad[0][i * 3 + 2] - tmp_grad[2][i * 3 + 0];
            cdouble curl_z = tmp_grad[1][i * 3 + 0] - tmp_grad[0][i * 3 + 1];
            acc += jw_p[idx] * (f_p[idx * 3] * curl_x + f_p[idx * 3 + 1] * curl_y + f_p[idx * 3 + 2] * curl_z);
        }
        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            int i = N * Nq + idx;
            cdouble curl_x = tmp_grad[2][i * 3 + 1] - tmp_grad[1][i * 3 + 2];
            cdouble curl_y = tmp_grad[0][i * 3 + 2] - tmp_grad[2][i * 3 + 0];
            cdouble curl_z = tmp_grad[1][i * 3 + 0] - tmp_grad[0][i * 3 + 1];
            acc += jw_m[idx] * (f_m[idx * 3] * curl_x + f_m[idx * 3 + 1] * curl_y + f_m[idx * 3 + 2] * curl_z);
        }
        K_result[m] = acc;
    }
}

void BemFmmOperator::LK_combined(const cdouble* x, cdouble kv, HelmholtzSurfacePFFT& spf,
                                  cdouble* L_result, cdouble* K_result)
{
    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;
    int total_pts = 2 * N * Nq;

    std::fill_n(L_result, N, cdouble(0.0, 0.0));
    std::fill_n(K_result, N, cdouble(0.0, 0.0));

    cdouble* src[4] = { tmp_src_charges.data(), tmp2_src_charges.data(), b4_src2.data(), b4_src3.data() };
    cdouble* pot[4] = { tmp_phi.data(), tmp2_phi.data(), b4_pot2.data(), b4_pot3.data() };

    for (int d = 0; d < 3; d++) {
        std::fill_n(src[d], total_pts, cdouble(0.0, 0.0));
        for (int n = 0; n < N; n++) {
            cdouble xn = x[n];
            for (int q = 0; q < Nq; q++) {
                int idx = n * Nq + q;
                src[d][idx] = f_p[idx * 3 + d] * jw_p[idx] * xn;
                src[d][N * Nq + idx] = f_m[idx * 3 + d] * jw_m[idx] * xn;
            }
        }
    }
    std::fill_n(src[3], total_pts, cdouble(0.0, 0.0));
    for (int n = 0; n < N; n++) {
        cdouble xn = x[n];
        for (int q = 0; q < Nq; q++) {
            int idx = n * Nq + q;
            src[3][idx] = div_p[n] * jw_p[idx] * xn;
            src[3][N * Nq + idx] = div_m[n] * jw_m[idx] * xn;
        }
    }

    spf.evaluate_batch4(src[0], src[1], src[2], src[3],
                        pot[0], pot[1], pot[2], pot[3],
                        tmp_grad[0].data(), tmp_grad[1].data(), tmp_grad[2].data());

    for (int d = 0; d < 3; d++) {
        for (int m = 0; m < N; m++) {
            cdouble acc(0);
            for (int q = 0; q < Nq; q++) {
                int idx = m * Nq + q;
                acc += f_p[idx * 3 + d] * jw_p[idx] * pot[d][idx];
                acc += f_m[idx * 3 + d] * jw_m[idx] * pot[d][N * Nq + idx];
            }
            L_result[m] += ik * acc;
        }
    }

    for (int m = 0; m < N; m++) {
        cdouble acc_p(0), acc_m(0);
        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            acc_p += jw_p[idx] * pot[3][idx];
            acc_m += jw_m[idx] * pot[3][N * Nq + idx];
        }
        L_result[m] -= iok * (div_p[m] * acc_p + div_m[m] * acc_m);
    }

    for (int m = 0; m < N; m++) {
        cdouble acc(0);
        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            int i = idx;
            cdouble curl_x = tmp_grad[2][i * 3 + 1] - tmp_grad[1][i * 3 + 2];
            cdouble curl_y = tmp_grad[0][i * 3 + 2] - tmp_grad[2][i * 3 + 0];
            cdouble curl_z = tmp_grad[1][i * 3 + 0] - tmp_grad[0][i * 3 + 1];
            acc += jw_p[idx] * (f_p[idx * 3] * curl_x + f_p[idx * 3 + 1] * curl_y + f_p[idx * 3 + 2] * curl_z);
        }
        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            int i = N * Nq + idx;
            cdouble curl_x = tmp_grad[2][i * 3 + 1] - tmp_grad[1][i * 3 + 2];
            cdouble curl_y = tmp_grad[0][i * 3 + 2] - tmp_grad[2][i * 3 + 0];
            cdouble curl_z = tmp_grad[1][i * 3 + 0] - tmp_grad[0][i * 3 + 1];
            acc += jw_m[idx] * (f_m[idx * 3] * curl_x + f_m[idx * 3 + 1] * curl_y + f_m[idx * 3 + 2] * curl_z);
        }
        K_result[m] = acc;
    }
}

void BemFmmOperator::LK_combined_batch2(
    const cdouble* x1, const cdouble* x2,
    cdouble kv, HelmholtzSurfacePFFT& spf,
    cdouble* L_result1, cdouble* K_result1,
    cdouble* L_result2, cdouble* K_result2)
{
    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;

    int half_pts = N * Nq;
    int total_pts = 2 * half_pts;
    int block = 256;
    int grid_N = (N + block - 1) / block;
    int grid_half = (half_pts + block - 1) / block;

    CUDA_CHECK(cudaMemcpy(d_x1_complex, x1, N * sizeof(double2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2_complex, x2, N * sizeof(double2), cudaMemcpyHostToDevice));
    bem_split_complex_kernel<<<grid_N, block>>>(d_x1_complex, d_x1_re, d_x1_im, N);
    bem_split_complex_kernel<<<grid_N, block>>>(d_x2_complex, d_x2_re, d_x2_im, N);
    CUDA_CHECK(cudaGetLastError());

    for (int d = 0; d < 3; d++) {
        bem_pack_vector_charges_kernel<<<grid_half, block>>>(
            d_x1_re, d_x1_im, d_x2_re, d_x2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m, N, Nq, d,
            spf.d_batch8_re + d * total_pts, spf.d_batch8_im + d * total_pts,
            spf.d_batch8_re + (4 + d) * total_pts, spf.d_batch8_im + (4 + d) * total_pts);
        CUDA_CHECK(cudaGetLastError());
    }

    bem_pack_scalar_charges_kernel<<<grid_half, block>>>(
        d_x1_re, d_x1_im, d_x2_re, d_x2_im,
        d_div_p, d_div_m, d_jw_p, d_jw_m, N, Nq,
        spf.d_batch8_re + 3 * total_pts, spf.d_batch8_im + 3 * total_pts,
        spf.d_batch8_re + 7 * total_pts, spf.d_batch8_im + 7 * total_pts);
    CUDA_CHECK(cudaGetLastError());

    spf.evaluate_batch8_gpu();

    bem_zero_results_kernel<<<grid_N, block>>>(
        d_L1_re, d_L1_im, d_K1_re, d_K1_im,
        d_L2_re, d_L2_im, d_K2_re, d_K2_im, N);
    CUDA_CHECK(cudaGetLastError());

    for (int d = 0; d < 3; d++) {
        bem_accum_L_vector_kernel<<<grid_N, block>>>(
            spf.d_bp_res_re[d], spf.d_bp_res_im[d],
            spf.d_bp_res_re[4 + d], spf.d_bp_res_im[4 + d],
            d_f_p, d_f_m, d_jw_p, d_jw_m,
            N, Nq, d, ik.real(), ik.imag(),
            d_L1_re, d_L1_im, d_L2_re, d_L2_im);
        CUDA_CHECK(cudaGetLastError());

        bem_accum_K_component_kernel<<<grid_N, block>>>(
            spf.d_bp_grd_re[d], spf.d_bp_grd_im[d],
            spf.d_bp_grd_re[3 + d], spf.d_bp_grd_im[3 + d],
            d_f_p, d_f_m, d_jw_p, d_jw_m, N, Nq, d,
            d_K1_re, d_K1_im, d_K2_re, d_K2_im);
        CUDA_CHECK(cudaGetLastError());
    }

    bem_accum_L_scalar_kernel<<<grid_N, block>>>(
        spf.d_bp_res_re[3], spf.d_bp_res_im[3],
        spf.d_bp_res_re[7], spf.d_bp_res_im[7],
        d_div_p, d_div_m, d_jw_p, d_jw_m,
        N, Nq, iok.real(), iok.imag(),
        d_L1_re, d_L1_im, d_L2_re, d_L2_im);
    CUDA_CHECK(cudaGetLastError());

    bem_pack_complex_kernel<<<grid_N, block>>>(d_L1_re, d_L1_im, d_out1_complex, N);
    bem_pack_complex_kernel<<<grid_N, block>>>(d_L2_re, d_L2_im, d_out2_complex, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(L_result1, d_out1_complex, N * sizeof(double2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(L_result2, d_out2_complex, N * sizeof(double2), cudaMemcpyDeviceToHost));

    bem_pack_complex_kernel<<<grid_N, block>>>(d_K1_re, d_K1_im, d_out1_complex, N);
    bem_pack_complex_kernel<<<grid_N, block>>>(d_K2_re, d_K2_im, d_out2_complex, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(K_result1, d_out1_complex, N * sizeof(double2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(K_result2, d_out2_complex, N * sizeof(double2), cudaMemcpyDeviceToHost));
}
#endif

void BemFmmOperator::LK_combined_batch2(
    const cdouble* x1, const cdouble* x2,
    cdouble kv, HelmholtzFMM& fmm,
    cdouble* L_result1, cdouble* K_result1,
    cdouble* L_result2, cdouble* K_result2)
{
    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;

    int half_pts = N * Nq;
    int block = 256;
    int grid_N = (N + block - 1) / block;
    int grid_half = (half_pts + block - 1) / block;

    CUDA_CHECK(cudaMemcpy(d_x1_complex, x1, N * sizeof(double2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2_complex, x2, N * sizeof(double2), cudaMemcpyHostToDevice));
    bem_split_complex_kernel<<<grid_N, block>>>(d_x1_complex, d_x1_re, d_x1_im, N);
    bem_split_complex_kernel<<<grid_N, block>>>(d_x2_complex, d_x2_re, d_x2_im, N);
    CUDA_CHECK(cudaGetLastError());

    bem_zero_results_kernel<<<grid_N, block>>>(
        d_L1_re, d_L1_im, d_K1_re, d_K1_im,
        d_L2_re, d_L2_im, d_K2_re, d_K2_im, N);
    CUDA_CHECK(cudaGetLastError());

    for (int d = 0; d < 3; d++) {
        bem_pack_vector_charges_kernel<<<grid_half, block>>>(
            d_x1_re, d_x1_im, d_x2_re, d_x2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m, N, Nq, d,
            fmm.d_charges_re, fmm.d_charges_im,
            fmm.d_charges2_re, fmm.d_charges2_im);
        CUDA_CHECK(cudaGetLastError());

        fmm.evaluate_pot_grad_batch2_uploaded();

        bem_accum_L_vector_kernel<<<grid_N, block>>>(
            fmm.d_result_re, fmm.d_result_im,
            fmm.d_result2_re, fmm.d_result2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m,
            N, Nq, d, ik.real(), ik.imag(),
            d_L1_re, d_L1_im, d_L2_re, d_L2_im);
        CUDA_CHECK(cudaGetLastError());

        bem_accum_K_component_kernel<<<grid_N, block>>>(
            fmm.d_grad_re, fmm.d_grad_im,
            fmm.d_grad2_re, fmm.d_grad2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m, N, Nq, d,
            d_K1_re, d_K1_im, d_K2_re, d_K2_im);
        CUDA_CHECK(cudaGetLastError());
    }

    bem_pack_scalar_charges_kernel<<<grid_half, block>>>(
        d_x1_re, d_x1_im, d_x2_re, d_x2_im,
        d_div_p, d_div_m, d_jw_p, d_jw_m, N, Nq,
        fmm.d_charges_re, fmm.d_charges_im,
        fmm.d_charges2_re, fmm.d_charges2_im);
    CUDA_CHECK(cudaGetLastError());

    fmm.evaluate_batch2_uploaded();

    bem_accum_L_scalar_kernel<<<grid_N, block>>>(
        fmm.d_result_re, fmm.d_result_im,
        fmm.d_result2_re, fmm.d_result2_im,
        d_div_p, d_div_m, d_jw_p, d_jw_m,
        N, Nq, iok.real(), iok.imag(),
        d_L1_re, d_L1_im, d_L2_re, d_L2_im);
    CUDA_CHECK(cudaGetLastError());

    bem_pack_complex_kernel<<<grid_N, block>>>(d_L1_re, d_L1_im, d_out1_complex, N);
    bem_pack_complex_kernel<<<grid_N, block>>>(d_L2_re, d_L2_im, d_out2_complex, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(L_result1, d_out1_complex, N * sizeof(double2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(L_result2, d_out2_complex, N * sizeof(double2), cudaMemcpyDeviceToHost));

    bem_pack_complex_kernel<<<grid_N, block>>>(d_K1_re, d_K1_im, d_out1_complex, N);
    bem_pack_complex_kernel<<<grid_N, block>>>(d_K2_re, d_K2_im, d_out2_complex, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(K_result1, d_out1_complex, N * sizeof(double2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(K_result2, d_out2_complex, N * sizeof(double2), cudaMemcpyDeviceToHost));
}

void BemFmmOperator::LK_combined_batch2_device(
    const cdouble* x1, const cdouble* x2,
    cdouble kv, HelmholtzFMM& fmm,
    int L_slot, int K_slot)
{
    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;

    int half_pts = N * Nq;
    int block = 256;
    int grid_N = (N + block - 1) / block;
    int grid_half = (half_pts + block - 1) / block;

    CUDA_CHECK(cudaMemcpy(d_x1_complex, x1, N * sizeof(double2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2_complex, x2, N * sizeof(double2), cudaMemcpyHostToDevice));
    bem_split_complex_kernel<<<grid_N, block>>>(d_x1_complex, d_x1_re, d_x1_im, N);
    bem_split_complex_kernel<<<grid_N, block>>>(d_x2_complex, d_x2_re, d_x2_im, N);
    CUDA_CHECK(cudaGetLastError());

    bem_zero_results_kernel<<<grid_N, block>>>(
        d_L1_re, d_L1_im, d_K1_re, d_K1_im,
        d_L2_re, d_L2_im, d_K2_re, d_K2_im, N);
    CUDA_CHECK(cudaGetLastError());

    for (int d = 0; d < 3; d++) {
        bem_pack_vector_charges_kernel<<<grid_half, block>>>(
            d_x1_re, d_x1_im, d_x2_re, d_x2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m, N, Nq, d,
            fmm.d_charges_re, fmm.d_charges_im,
            fmm.d_charges2_re, fmm.d_charges2_im);
        CUDA_CHECK(cudaGetLastError());

        fmm.evaluate_pot_grad_batch2_uploaded();

        bem_accum_L_vector_kernel<<<grid_N, block>>>(
            fmm.d_result_re, fmm.d_result_im,
            fmm.d_result2_re, fmm.d_result2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m,
            N, Nq, d, ik.real(), ik.imag(),
            d_L1_re, d_L1_im, d_L2_re, d_L2_im);
        CUDA_CHECK(cudaGetLastError());

        bem_accum_K_component_kernel<<<grid_N, block>>>(
            fmm.d_grad_re, fmm.d_grad_im,
            fmm.d_grad2_re, fmm.d_grad2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m, N, Nq, d,
            d_K1_re, d_K1_im, d_K2_re, d_K2_im);
        CUDA_CHECK(cudaGetLastError());
    }

    bem_pack_scalar_charges_kernel<<<grid_half, block>>>(
        d_x1_re, d_x1_im, d_x2_re, d_x2_im,
        d_div_p, d_div_m, d_jw_p, d_jw_m, N, Nq,
        fmm.d_charges_re, fmm.d_charges_im,
        fmm.d_charges2_re, fmm.d_charges2_im);
    CUDA_CHECK(cudaGetLastError());

    fmm.evaluate_batch2_uploaded();

    bem_accum_L_scalar_kernel<<<grid_N, block>>>(
        fmm.d_result_re, fmm.d_result_im,
        fmm.d_result2_re, fmm.d_result2_im,
        d_div_p, d_div_m, d_jw_p, d_jw_m,
        N, Nq, iok.real(), iok.imag(),
        d_L1_re, d_L1_im, d_L2_re, d_L2_im);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(d_mv1_re + L_slot * N, d_L1_re, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_mv1_im + L_slot * N, d_L1_im, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_mv2_re + L_slot * N, d_L2_re, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_mv2_im + L_slot * N, d_L2_im, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_mv1_re + K_slot * N, d_K1_re, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_mv1_im + K_slot * N, d_K1_im, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_mv2_re + K_slot * N, d_K2_re, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_mv2_im + K_slot * N, d_K2_im, N * sizeof(double), cudaMemcpyDeviceToDevice));
}

void BemFmmOperator::LK_combined_batch2_device_split(
    const double* x1_re, const double* x1_im,
    const double* x2_re, const double* x2_im,
    cdouble kv, HelmholtzFMM& fmm,
    int L_slot, int K_slot)
{
    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;

    int half_pts = N * Nq;
    int block = 256;
    int grid_N = (N + block - 1) / block;
    int grid_half = (half_pts + block - 1) / block;
    double* L1_re = d_mv1_re + L_slot * N;
    double* L1_im = d_mv1_im + L_slot * N;
    double* K1_re = d_mv1_re + K_slot * N;
    double* K1_im = d_mv1_im + K_slot * N;
    double* L2_re = d_mv2_re + L_slot * N;
    double* L2_im = d_mv2_im + L_slot * N;
    double* K2_re = d_mv2_re + K_slot * N;
    double* K2_im = d_mv2_im + K_slot * N;

    for (int d = 0; d < 3; d++) {
        bem_pack_vector_charges_kernel<<<grid_half, block>>>(
            x1_re, x1_im, x2_re, x2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m, N, Nq, d,
            fmm.d_charges_re, fmm.d_charges_im,
            fmm.d_charges2_re, fmm.d_charges2_im);
        CUDA_CHECK(cudaGetLastError());

        fmm.evaluate_pot_grad_batch2_uploaded();

        bem_accum_L_vector_kernel<<<grid_N, block>>>(
            fmm.d_result_re, fmm.d_result_im,
            fmm.d_result2_re, fmm.d_result2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m,
            N, Nq, d, ik.real(), ik.imag(),
            L1_re, L1_im, L2_re, L2_im);
        CUDA_CHECK(cudaGetLastError());

        bem_accum_K_component_kernel<<<grid_N, block>>>(
            fmm.d_grad_re, fmm.d_grad_im,
            fmm.d_grad2_re, fmm.d_grad2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m, N, Nq, d,
            K1_re, K1_im, K2_re, K2_im);
        CUDA_CHECK(cudaGetLastError());
    }

    bem_pack_scalar_charges_kernel<<<grid_half, block>>>(
        x1_re, x1_im, x2_re, x2_im,
        d_div_p, d_div_m, d_jw_p, d_jw_m, N, Nq,
        fmm.d_charges_re, fmm.d_charges_im,
        fmm.d_charges2_re, fmm.d_charges2_im);
    CUDA_CHECK(cudaGetLastError());

    fmm.evaluate_batch2_uploaded();

    bem_accum_L_scalar_kernel<<<grid_N, block>>>(
        fmm.d_result_re, fmm.d_result_im,
        fmm.d_result2_re, fmm.d_result2_im,
        d_div_p, d_div_m, d_jw_p, d_jw_m,
        N, Nq, iok.real(), iok.imag(),
        L1_re, L1_im, L2_re, L2_im);
    CUDA_CHECK(cudaGetLastError());
}

void BemFmmOperator::LK_combined_batch4_jm_device_split(
    const double* J1_re, const double* J1_im,
    const double* J2_re, const double* J2_im,
    const double* M1_re, const double* M1_im,
    const double* M2_re, const double* M2_im,
    cdouble kv, HelmholtzFMM& fmm,
    int LJ_slot, int KJ_slot,
    int LM_slot, int KM_slot)
{
    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;

    int half_pts = N * Nq;
    int block = 256;
    int grid_N = (N + block - 1) / block;
    int grid_half = (half_pts + block - 1) / block;

    double* LJ1_re = d_mv1_re + LJ_slot * N;
    double* LJ1_im = d_mv1_im + LJ_slot * N;
    double* KJ1_re = d_mv1_re + KJ_slot * N;
    double* KJ1_im = d_mv1_im + KJ_slot * N;
    double* LM1_re = d_mv1_re + LM_slot * N;
    double* LM1_im = d_mv1_im + LM_slot * N;
    double* KM1_re = d_mv1_re + KM_slot * N;
    double* KM1_im = d_mv1_im + KM_slot * N;

    double* LJ2_re = d_mv2_re + LJ_slot * N;
    double* LJ2_im = d_mv2_im + LJ_slot * N;
    double* KJ2_re = d_mv2_re + KJ_slot * N;
    double* KJ2_im = d_mv2_im + KJ_slot * N;
    double* LM2_re = d_mv2_re + LM_slot * N;
    double* LM2_im = d_mv2_im + LM_slot * N;
    double* KM2_re = d_mv2_re + KM_slot * N;
    double* KM2_im = d_mv2_im + KM_slot * N;

    for (int d = 0; d < 3; d++) {
        bem_pack_vector_charges_kernel<<<grid_half, block>>>(
            J1_re, J1_im, J2_re, J2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m, N, Nq, d,
            fmm.d_charges_re, fmm.d_charges_im,
            fmm.d_charges2_re, fmm.d_charges2_im);
        CUDA_CHECK(cudaGetLastError());
        bem_pack_vector_charges_kernel<<<grid_half, block>>>(
            M1_re, M1_im, M2_re, M2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m, N, Nq, d,
            fmm.d_charges3_re, fmm.d_charges3_im,
            fmm.d_charges4_re, fmm.d_charges4_im);
        CUDA_CHECK(cudaGetLastError());

        fmm.evaluate_pot_grad_batch4_uploaded();

        bem_accum_L_vector_kernel<<<grid_N, block>>>(
            fmm.d_result_re, fmm.d_result_im,
            fmm.d_result2_re, fmm.d_result2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m,
            N, Nq, d, ik.real(), ik.imag(),
            LJ1_re, LJ1_im, LJ2_re, LJ2_im);
        CUDA_CHECK(cudaGetLastError());
        bem_accum_L_vector_kernel<<<grid_N, block>>>(
            fmm.d_result3_re, fmm.d_result3_im,
            fmm.d_result4_re, fmm.d_result4_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m,
            N, Nq, d, ik.real(), ik.imag(),
            LM1_re, LM1_im, LM2_re, LM2_im);
        CUDA_CHECK(cudaGetLastError());

        bem_accum_K_component_kernel<<<grid_N, block>>>(
            fmm.d_grad_re, fmm.d_grad_im,
            fmm.d_grad2_re, fmm.d_grad2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m, N, Nq, d,
            KJ1_re, KJ1_im, KJ2_re, KJ2_im);
        CUDA_CHECK(cudaGetLastError());
        bem_accum_K_component_kernel<<<grid_N, block>>>(
            fmm.d_grad3_re, fmm.d_grad3_im,
            fmm.d_grad4_re, fmm.d_grad4_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m, N, Nq, d,
            KM1_re, KM1_im, KM2_re, KM2_im);
        CUDA_CHECK(cudaGetLastError());
    }

    bem_pack_scalar_charges_kernel<<<grid_half, block>>>(
        J1_re, J1_im, J2_re, J2_im,
        d_div_p, d_div_m, d_jw_p, d_jw_m, N, Nq,
        fmm.d_charges_re, fmm.d_charges_im,
        fmm.d_charges2_re, fmm.d_charges2_im);
    CUDA_CHECK(cudaGetLastError());
    bem_pack_scalar_charges_kernel<<<grid_half, block>>>(
        M1_re, M1_im, M2_re, M2_im,
        d_div_p, d_div_m, d_jw_p, d_jw_m, N, Nq,
        fmm.d_charges3_re, fmm.d_charges3_im,
        fmm.d_charges4_re, fmm.d_charges4_im);
    CUDA_CHECK(cudaGetLastError());

    fmm.evaluate_batch4_uploaded();

    bem_accum_L_scalar_kernel<<<grid_N, block>>>(
        fmm.d_result_re, fmm.d_result_im,
        fmm.d_result2_re, fmm.d_result2_im,
        d_div_p, d_div_m, d_jw_p, d_jw_m,
        N, Nq, iok.real(), iok.imag(),
        LJ1_re, LJ1_im, LJ2_re, LJ2_im);
    CUDA_CHECK(cudaGetLastError());
    bem_accum_L_scalar_kernel<<<grid_N, block>>>(
        fmm.d_result3_re, fmm.d_result3_im,
        fmm.d_result4_re, fmm.d_result4_im,
        d_div_p, d_div_m, d_jw_p, d_jw_m,
        N, Nq, iok.real(), iok.imag(),
        LM1_re, LM1_im, LM2_re, LM2_im);
    CUDA_CHECK(cudaGetLastError());
}

#ifndef BEM_FMM_ONLY
void BemFmmOperator::LK_combined_batch2_spfft_device_split(
    const double* x1_re, const double* x1_im,
    const double* x2_re, const double* x2_im,
    cdouble kv, HelmholtzSurfacePFFT& spf,
    int L_slot, int K_slot)
{
    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;

    int half_pts = N * Nq;
    int total_pts = 2 * half_pts;
    int block = 256;
    int grid_N = (N + block - 1) / block;
    int grid_half = (half_pts + block - 1) / block;
    double* L1_re = d_mv1_re + L_slot * N;
    double* L1_im = d_mv1_im + L_slot * N;
    double* K1_re = d_mv1_re + K_slot * N;
    double* K1_im = d_mv1_im + K_slot * N;
    double* L2_re = d_mv2_re + L_slot * N;
    double* L2_im = d_mv2_im + L_slot * N;
    double* K2_re = d_mv2_re + K_slot * N;
    double* K2_im = d_mv2_im + K_slot * N;

    for (int d = 0; d < 3; d++) {
        bem_pack_vector_charges_kernel<<<grid_half, block>>>(
            x1_re, x1_im, x2_re, x2_im,
            d_f_p, d_f_m, d_jw_p, d_jw_m, N, Nq, d,
            spf.d_batch8_re + d * total_pts, spf.d_batch8_im + d * total_pts,
            spf.d_batch8_re + (4 + d) * total_pts, spf.d_batch8_im + (4 + d) * total_pts);
        CUDA_CHECK(cudaGetLastError());
    }

    bem_pack_scalar_charges_kernel<<<grid_half, block>>>(
        x1_re, x1_im, x2_re, x2_im,
        d_div_p, d_div_m, d_jw_p, d_jw_m, N, Nq,
        spf.d_batch8_re + 3 * total_pts, spf.d_batch8_im + 3 * total_pts,
        spf.d_batch8_re + 7 * total_pts, spf.d_batch8_im + 7 * total_pts);
    CUDA_CHECK(cudaGetLastError());

    spf.evaluate_batch8_gpu();

    for (int d = 0; d < 3; d++) {
        bem_accum_L_vector_kernel<<<grid_N, block>>>(
            spf.d_bp_res_re[d], spf.d_bp_res_im[d],
            spf.d_bp_res_re[4 + d], spf.d_bp_res_im[4 + d],
            d_f_p, d_f_m, d_jw_p, d_jw_m,
            N, Nq, d, ik.real(), ik.imag(),
            L1_re, L1_im, L2_re, L2_im);
        CUDA_CHECK(cudaGetLastError());

        bem_accum_K_component_kernel<<<grid_N, block>>>(
            spf.d_bp_grd_re[d], spf.d_bp_grd_im[d],
            spf.d_bp_grd_re[3 + d], spf.d_bp_grd_im[3 + d],
            d_f_p, d_f_m, d_jw_p, d_jw_m, N, Nq, d,
            K1_re, K1_im, K2_re, K2_im);
        CUDA_CHECK(cudaGetLastError());
    }

    bem_accum_L_scalar_kernel<<<grid_N, block>>>(
        spf.d_bp_res_re[3], spf.d_bp_res_im[3],
        spf.d_bp_res_re[7], spf.d_bp_res_im[7],
        d_div_p, d_div_m, d_jw_p, d_jw_m,
        N, Nq, iok.real(), iok.imag(),
        L1_re, L1_im, L2_re, L2_im);
    CUDA_CHECK(cudaGetLastError());
}
#endif

bool BemFmmOperator::device_matvec_available() const
{
#ifndef BEM_FMM_ONLY
    if (use_pfft || use_spfft)
        return false;
#endif
    return true;
}

void BemFmmOperator::matvec_batch2_device(const double2* d_x1_full_in,
                                          const double2* d_x2_full_in,
                                          double2* d_y1_out,
                                          double2* d_y2_out)
{
#ifndef BEM_FMM_ONLY
    if (use_pfft) {
        fprintf(stderr, "Error: matvec_batch2_device is not implemented for PFFT backend\n");
        std::exit(1);
        return;
    }
#endif

    if (use_spfft) {
#ifdef BEM_FMM_ONLY
        use_spfft = false;
#else
        fprintf(stderr, "Error: matvec_batch2_device is not implemented for SurfPFFT backend\n");
        std::exit(1);
        return;
#endif
    }

    HelmholtzFMM& fmm_i = shared_fmm ? fmm_ext : fmm_int;

    int block = 256;
    int grid_N = (N + block - 1) / block;
    int grid_sys = (system_size + block - 1) / block;

    double assembly_unknown_m_scale = unknown_m_scale;
    double split_m_scale = 1.0;
    if (unknown_m_scale != 1.0) {
        split_m_scale = 1.0 / unknown_m_scale;
        assembly_unknown_m_scale = 1.0;
    }
    bem_split_complex_batch2_scale_kernel<<<grid_sys, block>>>(
        d_x1_full_in, d_x2_full_in,
        d_full_x1_re, d_full_x1_im,
        d_full_x2_re, d_full_x2_im,
        system_size, N, split_m_scale);
    CUDA_CHECK(cudaGetLastError());

    if (use_full_mvslot_memset(N)) {
        const int mv_slots = n_form ? 10 : 8;
        CUDA_CHECK(cudaMemset(d_mv1_re, 0, mv_slots * N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_mv1_im, 0, mv_slots * N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_mv2_re, 0, mv_slots * N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_mv2_im, 0, mv_slots * N * sizeof(double)));
    } else if (n_form) {
        CUDA_CHECK(cudaMemset(d_mv1_re + 8 * N, 0, 2 * N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_mv1_im + 8 * N, 0, 2 * N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_mv2_re + 8 * N, 0, 2 * N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_mv2_im + 8 * N, 0, 2 * N * sizeof(double)));
    }

    bool use_batch4 = (bem_env_flag_enabled("BEM_FMM_BATCH4") &&
                       fmm_ext.batch4_allocated && fmm_i.batch4_allocated);
    if (use_batch4) {
        LK_combined_batch4_jm_device_split(d_full_x1_re, d_full_x1_im,
                                           d_full_x2_re, d_full_x2_im,
                                           d_full_x1_re + N, d_full_x1_im + N,
                                           d_full_x2_re + N, d_full_x2_im + N,
                                           k_ext, fmm_ext, 0, 2, 1, 3);
        LK_combined_batch4_jm_device_split(d_full_x1_re, d_full_x1_im,
                                           d_full_x2_re, d_full_x2_im,
                                           d_full_x1_re + N, d_full_x1_im + N,
                                           d_full_x2_re + N, d_full_x2_im + N,
                                           k_int, fmm_i, 4, 6, 5, 7);
    } else {
        LK_combined_batch2_device_split(d_full_x1_re, d_full_x1_im,
                                        d_full_x2_re, d_full_x2_im,
                                        k_ext, fmm_ext, 0, 2);
        LK_combined_batch2_device_split(d_full_x1_re + N, d_full_x1_im + N,
                                        d_full_x2_re + N, d_full_x2_im + N,
                                        k_ext, fmm_ext, 1, 3);
        LK_combined_batch2_device_split(d_full_x1_re, d_full_x1_im,
                                        d_full_x2_re, d_full_x2_im,
                                        k_int, fmm_i, 4, 6);
        LK_combined_batch2_device_split(d_full_x1_re + N, d_full_x1_im + N,
                                        d_full_x2_re + N, d_full_x2_im + N,
                                        k_int, fmm_i, 5, 7);
    }

    bem_apply_corr_assemble_batch2_kernel<<<grid_N, block>>>(
        d_mv1_re, d_mv1_im, d_mv2_re, d_mv2_im,
        d_full_x1_re, d_full_x1_im, d_full_x2_re, d_full_x2_im,
        d_corr_row_ptr, d_corr_col_idx,
        d_corr_L_ext_re, d_corr_L_ext_im,
        d_corr_K_ext_re, d_corr_K_ext_im,
        d_corr_L_int_re, d_corr_L_int_im,
        d_corr_K_int_re, d_corr_K_int_im,
        d_corr_I,
        N, eta_ext.real(), eta_ext.imag(), eta_int.real(), eta_int.imag(),
        assembly_unknown_m_scale, row_h_scale.real(), row_h_scale.imag(),
        int_op_sign, k_identity,
        n_form ? 1 : 0, n_form_eps_int, n_form_m_identity,
        d_y1_out, d_y2_out);
    CUDA_CHECK(cudaGetLastError());
}

void BemFmmOperator::matvec_batch2(const cdouble* x1_full, const cdouble* x2_full,
                                    cdouble* y1, cdouble* y2)
{
#ifndef BEM_FMM_ONLY
    if (use_pfft) {
        matvec(x1_full, y1);
        matvec(x2_full, y2);
        return;
    }
    if (use_spfft) {
        HelmholtzSurfacePFFT& sp_i = shared_fmm ? spfft_ext : spfft_int;

        int block = 256;
        int grid_N = (N + block - 1) / block;
        int grid_sys = (system_size + block - 1) / block;

        upload_complex_stage(d_full_x1_complex, h_full_x1_complex, x1_full,
                             system_size, pinned_matvec_stage);
        upload_complex_stage(d_full_x2_complex, h_full_x2_complex, x2_full,
                             system_size, pinned_matvec_stage);
        double assembly_unknown_m_scale = unknown_m_scale;
        double split_m_scale = 1.0;
        if (unknown_m_scale != 1.0) {
            split_m_scale = 1.0 / unknown_m_scale;
            assembly_unknown_m_scale = 1.0;
        }
        bem_split_complex_batch2_scale_kernel<<<grid_sys, block>>>(
            d_full_x1_complex, d_full_x2_complex,
            d_full_x1_re, d_full_x1_im,
            d_full_x2_re, d_full_x2_im,
            system_size, N, split_m_scale);
        CUDA_CHECK(cudaGetLastError());

        if (use_full_mvslot_memset(N)) {
            const int mv_slots = n_form ? 10 : 8;
            CUDA_CHECK(cudaMemset(d_mv1_re, 0, mv_slots * N * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_mv1_im, 0, mv_slots * N * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_mv2_re, 0, mv_slots * N * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_mv2_im, 0, mv_slots * N * sizeof(double)));
        } else if (n_form) {
            CUDA_CHECK(cudaMemset(d_mv1_re + 8 * N, 0, 2 * N * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_mv1_im + 8 * N, 0, 2 * N * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_mv2_re + 8 * N, 0, 2 * N * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_mv2_im + 8 * N, 0, 2 * N * sizeof(double)));
        }

        LK_combined_batch2_spfft_device_split(d_full_x1_re, d_full_x1_im,
                                              d_full_x2_re, d_full_x2_im,
                                              k_ext, spfft_ext, 0, 2);
        LK_combined_batch2_spfft_device_split(d_full_x1_re + N, d_full_x1_im + N,
                                              d_full_x2_re + N, d_full_x2_im + N,
                                              k_ext, spfft_ext, 1, 3);
        LK_combined_batch2_spfft_device_split(d_full_x1_re, d_full_x1_im,
                                              d_full_x2_re, d_full_x2_im,
                                              k_int, sp_i,      4, 6);
        LK_combined_batch2_spfft_device_split(d_full_x1_re + N, d_full_x1_im + N,
                                              d_full_x2_re + N, d_full_x2_im + N,
                                              k_int, sp_i,      5, 7);

        bem_apply_corr_assemble_batch2_kernel<<<grid_N, block>>>(
            d_mv1_re, d_mv1_im, d_mv2_re, d_mv2_im,
            d_full_x1_re, d_full_x1_im, d_full_x2_re, d_full_x2_im,
            d_corr_row_ptr, d_corr_col_idx,
            d_corr_L_ext_re, d_corr_L_ext_im,
            d_corr_K_ext_re, d_corr_K_ext_im,
            d_corr_L_int_re, d_corr_L_int_im,
            d_corr_K_int_re, d_corr_K_int_im,
            d_corr_I,
            N, eta_ext.real(), eta_ext.imag(), eta_int.real(), eta_int.imag(),
            assembly_unknown_m_scale, row_h_scale.real(), row_h_scale.imag(),
            int_op_sign, k_identity,
            n_form ? 1 : 0, n_form_eps_int, n_form_m_identity,
            d_y1_complex, d_y2_complex);
        CUDA_CHECK(cudaGetLastError());

        download_complex_stage(y1, h_y1_complex, d_y1_complex,
                               system_size, pinned_matvec_stage);
        download_complex_stage(y2, h_y2_complex, d_y2_complex,
                               system_size, pinned_matvec_stage);
        return;
    }
#endif

    upload_complex_stage(d_full_x1_complex, h_full_x1_complex, x1_full,
                         system_size, pinned_matvec_stage);
    upload_complex_stage(d_full_x2_complex, h_full_x2_complex, x2_full,
                         system_size, pinned_matvec_stage);
    matvec_batch2_device(d_full_x1_complex, d_full_x2_complex, d_y1_complex, d_y2_complex);
    download_complex_stage(y1, h_y1_complex, d_y1_complex,
                           system_size, pinned_matvec_stage);
    download_complex_stage(y2, h_y2_complex, d_y2_complex,
                           system_size, pinned_matvec_stage);
}

void BemFmmOperator::matvec(const cdouble* x_full, cdouble* y)
{
#ifdef BEM_FMM_ONLY
    tmp_single_y.resize(system_size);
    matvec_batch2(x_full, x_full, y, tmp_single_y.data());
    return;
#else
    if (!use_pfft) {
        tmp_single_y.resize(system_size);
        matvec_batch2(x_full, x_full, y, tmp_single_y.data());
        return;
    }
#endif

    ensure_host_workspace();

    const cdouble* J = x_full;
    const cdouble* M = x_full + N;
    if (unknown_m_scale != 1.0) {
        double inv_s = 1.0 / unknown_m_scale;
        tmp_M1_phys.resize(N);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            tmp_M1_phys[i] = M[i] * inv_s;
        M = tmp_M1_phys.data();
    }

    // Combined L+K: 4 FMM passes each (3 pot+grad + 1 pot) instead of 7 (4L + 3K)
    if (use_spfft) {
#ifdef BEM_FMM_ONLY
        use_spfft = false;
#else
        HelmholtzSurfacePFFT& sp_i = shared_fmm ? spfft_ext : spfft_int;
        LK_combined(J, k_ext, spfft_ext, mv_L_ext_J.data(), mv_K_ext_J.data());
        LK_combined(M, k_ext, spfft_ext, mv_L_ext_M.data(), mv_K_ext_M.data());
        LK_combined(J, k_int, sp_i,      mv_L_int_J.data(), mv_K_int_J.data());
        LK_combined(M, k_int, sp_i,      mv_L_int_M.data(), mv_K_int_M.data());
#endif
    } else if (use_pfft) {
#ifdef BEM_FMM_ONLY
        use_pfft = false;
#else
        HelmholtzPFFT& pf_i = shared_fmm ? pfft_ext : pfft_int;
        LK_combined(J, k_ext, pfft_ext, mv_L_ext_J.data(), mv_K_ext_J.data());
        LK_combined(M, k_ext, pfft_ext, mv_L_ext_M.data(), mv_K_ext_M.data());
        LK_combined(J, k_int, pf_i,     mv_L_int_J.data(), mv_K_int_J.data());
        LK_combined(M, k_int, pf_i,     mv_L_int_M.data(), mv_K_int_M.data());
#endif
    } else {
        HelmholtzFMM& fmm_i = shared_fmm ? fmm_ext : fmm_int;
        LK_combined(J, k_ext, fmm_ext, mv_L_ext_J.data(), mv_K_ext_J.data());
        LK_combined(M, k_ext, fmm_ext, mv_L_ext_M.data(), mv_K_ext_M.data());
        LK_combined(J, k_int, fmm_i,   mv_L_int_J.data(), mv_K_int_J.data());
        LK_combined(M, k_int, fmm_i,   mv_L_int_M.data(), mv_K_int_M.data());
    }

    // Apply singular corrections (sparse CSR)
    for (int m = 0; m < N; m++) {
        for (int j = corr_row_ptr[m]; j < corr_row_ptr[m + 1]; j++) {
            int n = corr_col_idx[j];
            mv_L_ext_J[m] += corr_L_ext_val[j] * J[n];
            mv_L_ext_M[m] += corr_L_ext_val[j] * M[n];
            mv_K_ext_J[m] += corr_K_ext_val[j] * J[n];
            mv_K_ext_M[m] += corr_K_ext_val[j] * M[n];

            mv_L_int_J[m] += corr_L_int_val[j] * J[n];
            mv_L_int_M[m] += corr_L_int_val[j] * M[n];
            mv_K_int_J[m] += corr_K_int_val[j] * J[n];
            mv_K_int_M[m] += corr_K_int_val[j] * M[n];
        }
    }

    // Assemble system blocks
    for (int m = 0; m < N; m++) {
        cdouble IJ = J[m];
        cdouble IM = M[m];
        if (n_form) {
            IJ = cdouble(0);
            IM = cdouble(0);
            for (int j = corr_row_ptr[m]; j < corr_row_ptr[m + 1]; j++) {
                int n = corr_col_idx[j];
                IJ += corr_I_val[j] * J[n];
                IM += corr_I_val[j] * M[n];
            }
        }
        cdouble K_sum_J = mv_K_ext_J[m] + int_op_sign * mv_K_int_J[m] + k_identity * IJ;
        cdouble K_sum_M = mv_K_ext_M[m] + int_op_sign * mv_K_int_M[m] + k_identity * IM;

        if (n_form) {
            y[m] = K_sum_J + (mv_L_ext_M[m] / eta_ext +
                              int_op_sign * mv_L_int_M[m] / eta_int);
            cdouble K_sum_M_n = mv_K_ext_M[m] - n_form_eps_int * mv_K_int_M[m] +
                                n_form_m_identity * IM;
            y[N + m] = row_h_scale *
                (-(eta_ext * mv_L_ext_J[m] - n_form_eps_int * eta_int * mv_L_int_J[m]) +
                 K_sum_M_n);
        } else {
            y[m] = (eta_ext * mv_L_ext_J[m] + int_op_sign * eta_int * mv_L_int_J[m]) - K_sum_M;
            y[N + m] = row_h_scale * (K_sum_J +
                                      (mv_L_ext_M[m] / eta_ext +
                                       int_op_sign * mv_L_int_M[m] / eta_int));
        }
    }
}

void BemFmmOperator::cleanup()
{
    free_device_workspace();
    unregister_tmp_host_buffers();
#ifndef BEM_FMM_ONLY
    if (use_spfft) {
        spfft_ext.cleanup();
        if (!shared_fmm) spfft_int.cleanup();
    } else if (use_pfft) {
        pfft_ext.cleanup();
        if (!shared_fmm) pfft_int.cleanup();
    } else {
#endif
        fmm_ext.cleanup();
        if (!shared_fmm) fmm_int.cleanup();
#ifndef BEM_FMM_ONLY
    }
#endif
}
