#include "bem_fmm.h"
#include "graglia.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
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

__global__ void bem_split_complex_kernel(const double2* in, double* out_re, double* out_im, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double2 v = in[i];
    out_re[i] = v.x;
    out_im[i] = v.y;
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
    L1_re[m] += ik_re * a1r - ik_im * a1i;
    L1_im[m] += ik_re * a1i + ik_im * a1r;
    L2_re[m] += ik_re * a2r - ik_im * a2i;
    L2_im[m] += ik_re * a2i + ik_im * a2r;
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
    K1_re[m] += a1r;
    K1_im[m] += a1i;
    K2_re[m] += a2r;
    K2_im[m] += a2i;
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
    int N, double eta_e, double eta_i, double unknown_m_scale, double row_h_scale,
    double int_op_sign, double k_identity, int n_form, double n_form_eps_int,
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
            double mass = cI[j];
            r[8 * N + base] += mass * Jr;
            im[8 * N + base] += mass * Ji;
            r[9 * N + base] += mass * Mr;
            im[9 * N + base] += mass * Mi;
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
            ytop_r = (r[s2] + int_op_sign * r[s6] + k_identity * IJ_r) +
                     (r[s1] / eta_e + int_op_sign * r[s5] / eta_i);
            ytop_i = (im[s2] + int_op_sign * im[s6] + k_identity * IJ_i) +
                     (im[s1] / eta_e + int_op_sign * im[s5] / eta_i);
            double kid_bot = -0.5 * (1.0 + n_form_eps_int);
            ybot_r = row_h_scale * (-(eta_e * r[s0] - n_form_eps_int * eta_i * r[s4]) +
                                    (r[s3] - n_form_eps_int * r[s7] + kid_bot * IM_r));
            ybot_i = row_h_scale * (-(eta_e * im[s0] - n_form_eps_int * eta_i * im[s4]) +
                                    (im[s3] - n_form_eps_int * im[s7] + kid_bot * IM_i));
        } else {
            ytop_r = eta_e * r[s0] + int_op_sign * eta_i * r[s4] -
                            (r[s3] + int_op_sign * r[s7] + k_identity * Mvar_r);
            ytop_i = eta_e * im[s0] + int_op_sign * eta_i * im[s4] -
                            (im[s3] + int_op_sign * im[s7] + k_identity * Mvar_i);
            ybot_r = row_h_scale * ((r[s2] + int_op_sign * r[s6] + k_identity * Jvar_r) +
                                           r[s1] / eta_e + int_op_sign * r[s5] / eta_i);
            ybot_i = row_h_scale * ((im[s2] + int_op_sign * im[s6] + k_identity * Jvar_i) +
                                           im[s1] / eta_e + int_op_sign * im[s5] / eta_i);
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
                            double eta_ext_, double eta_int_,
                            int quad_order, int fmm_digits, int max_leaf,
                            bool use_pfft_, bool use_spfft_)
{
    Timer timer;
    k_ext = k_ext_;
    k_int = k_int_;
    eta_ext = cdouble(eta_ext_);
    eta_int = cdouble(eta_int_);
#ifdef BEM_FMM_ONLY
    use_pfft = false;
    use_spfft = false;
    (void)use_pfft_;
    (void)use_spfft_;
#else
    use_pfft = use_pfft_;
    use_spfft = use_spfft_;
    if (use_spfft) {
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

    // Pre-allocate temporary buffers used in matvec (avoid malloc/free per iteration)
    tmp_src_charges.resize(total_pts);
    tmp_phi.resize(total_pts);
    for (int d = 0; d < 3; d++)
        tmp_grad[d].resize(total_pts * 3);
    tmp_L_result.resize(N);
    tmp_K_result.resize(N);
    mv_L_ext_J.resize(N); mv_L_ext_M.resize(N);
    mv_K_ext_J.resize(N); mv_K_ext_M.resize(N);
    mv_L_int_J.resize(N); mv_L_int_M.resize(N);
    mv_K_int_J.resize(N); mv_K_int_M.resize(N);

    // Batch-2 workspace
    tmp2_src_charges.resize(total_pts);
    tmp2_phi.resize(total_pts);
    for (int d = 0; d < 3; d++)
        tmp2_grad[d].resize(total_pts * 3);
    mv2_L_ext_J.resize(N, 0); mv2_L_ext_M.resize(N, 0);
    mv2_K_ext_J.resize(N, 0); mv2_K_ext_M.resize(N, 0);
    mv2_L_int_J.resize(N, 0); mv2_L_int_M.resize(N, 0);
    mv2_K_int_J.resize(N, 0); mv2_K_int_M.resize(N, 0);
    b4_src2.resize(total_pts);
    b4_src3.resize(total_pts);
    b4_pot2.resize(total_pts);
    b4_pot3.resize(total_pts);
    tmp_M1_phys.resize(N);
    tmp_M2_phys.resize(N);
    register_tmp_host_buffers();

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
    CUDA_CHECK(cudaMalloc(&d_mv1_re, 10 * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mv1_im, 10 * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mv2_re, 10 * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mv2_im, 10 * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y1_complex, system_size * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_y2_complex, system_size * sizeof(double2)));

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

    // Step 1: Determine sparsity pattern — collect (row, col) pairs with shared triangles
    // Use a set per row to avoid duplicates (a pair can share 2 triangles)
    std::vector<std::vector<int>> row_cols(N);
    for (auto& pair : tri_to_rwg) {
        const std::vector<HalfInfo>& rwg_list = pair.second;
        for (const HalfInfo& mi : rwg_list) {
            for (const HalfInfo& ni : rwg_list) {
                row_cols[mi.n].push_back(ni.n);
            }
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

                    // Find position in CSR
                    const int* col_begin = &corr_col_idx[corr_row_ptr[m]];
                    const int* col_end   = &corr_col_idx[corr_row_ptr[m + 1]];
                    const int* it = std::lower_bound(col_begin, col_end, n_idx);
                    int pos = corr_row_ptr[m] + (int)(it - col_begin);

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
}

void BemFmmOperator::L_operator(const cdouble* x, cdouble kv, HelmholtzFMM& fmm, cdouble* result)
{
    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;

    int total_pts = 2 * N * Nq;

    // Zero result
    memset(result, 0, N * sizeof(cdouble));

    // --- Vector part: ik * integral(f_m . f_n . G) ---
    for (int d = 0; d < 3; d++) {
        memset(tmp_src_charges.data(), 0, total_pts * sizeof(cdouble));

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
        memset(tmp_src_charges.data(), 0, total_pts * sizeof(cdouble));

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
    memset(result, 0, N * sizeof(cdouble));

    // For each source component k, compute gradient of potential
    for (int kc = 0; kc < 3; kc++) {
        // Source charges = f_n^k * jw * x[n]
        memset(tmp_src_charges.data(), 0, total_pts * sizeof(cdouble));
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

    memset(L_result, 0, N * sizeof(cdouble));
    memset(K_result, 0, N * sizeof(cdouble));

    // --- Vector part: combined potential (for L) + gradient (for K) in one FMM pass ---
    for (int d = 0; d < 3; d++) {
        memset(tmp_src_charges.data(), 0, total_pts * sizeof(cdouble));

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
        memset(tmp_src_charges.data(), 0, total_pts * sizeof(cdouble));

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

    memset(L_result, 0, N * sizeof(cdouble));
    memset(K_result, 0, N * sizeof(cdouble));

    for (int d = 0; d < 3; d++) {
        memset(tmp_src_charges.data(), 0, total_pts * sizeof(cdouble));
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

    memset(tmp_src_charges.data(), 0, total_pts * sizeof(cdouble));
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

    memset(L_result, 0, N * sizeof(cdouble));
    memset(K_result, 0, N * sizeof(cdouble));

    cdouble* src[4] = { tmp_src_charges.data(), tmp2_src_charges.data(), b4_src2.data(), b4_src3.data() };
    cdouble* pot[4] = { tmp_phi.data(), tmp2_phi.data(), b4_pot2.data(), b4_pot3.data() };

    for (int d = 0; d < 3; d++) {
        memset(src[d], 0, total_pts * sizeof(cdouble));
        for (int n = 0; n < N; n++) {
            cdouble xn = x[n];
            for (int q = 0; q < Nq; q++) {
                int idx = n * Nq + q;
                src[d][idx] = f_p[idx * 3 + d] * jw_p[idx] * xn;
                src[d][N * Nq + idx] = f_m[idx * 3 + d] * jw_m[idx] * xn;
            }
        }
    }
    memset(src[3], 0, total_pts * sizeof(cdouble));
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
void BemFmmOperator::LK_combined_batch2_spfft_device(
    const cdouble* x1, const cdouble* x2,
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

    CUDA_CHECK(cudaMemcpy(d_mv1_re + L_slot * N, d_L1_re, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_mv1_im + L_slot * N, d_L1_im, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_mv2_re + L_slot * N, d_L2_re, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_mv2_im + L_slot * N, d_L2_im, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_mv1_re + K_slot * N, d_K1_re, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_mv1_im + K_slot * N, d_K1_im, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_mv2_re + K_slot * N, d_K2_re, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_mv2_im + K_slot * N, d_K2_im, N * sizeof(double), cudaMemcpyDeviceToDevice));
}
#endif

void BemFmmOperator::matvec_batch2(const cdouble* x1_full, const cdouble* x2_full,
                                    cdouble* y1, cdouble* y2)
{
#ifndef BEM_FMM_ONLY
    if (use_pfft) {
        matvec(x1_full, y1);
        matvec(x2_full, y2);
        return;
    }
#endif

    const cdouble* J1 = x1_full;
    const cdouble* M1 = x1_full + N;
    const cdouble* J2 = x2_full;
    const cdouble* M2 = x2_full + N;
    if (unknown_m_scale != 1.0) {
        double inv_s = 1.0 / unknown_m_scale;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            tmp_M1_phys[i] = M1[i] * inv_s;
            tmp_M2_phys[i] = M2[i] * inv_s;
        }
        M1 = tmp_M1_phys.data();
        M2 = tmp_M2_phys.data();
    }

    if (use_spfft) {
#ifdef BEM_FMM_ONLY
        use_spfft = false;
#else
        HelmholtzSurfacePFFT& sp_i = shared_fmm ? spfft_ext : spfft_int;

        int block = 256;
        int grid_N = (N + block - 1) / block;
        int grid_sys = (system_size + block - 1) / block;

        CUDA_CHECK(cudaMemcpy(d_full_x1_complex, x1_full, system_size * sizeof(double2), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_full_x2_complex, x2_full, system_size * sizeof(double2), cudaMemcpyHostToDevice));
        bem_split_complex_kernel<<<grid_sys, block>>>(d_full_x1_complex, d_full_x1_re, d_full_x1_im, system_size);
        bem_split_complex_kernel<<<grid_sys, block>>>(d_full_x2_complex, d_full_x2_re, d_full_x2_im, system_size);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(d_mv1_re, 0, 10 * N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_mv1_im, 0, 10 * N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_mv2_re, 0, 10 * N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_mv2_im, 0, 10 * N * sizeof(double)));

        LK_combined_batch2_spfft_device(J1, J2, k_ext, spfft_ext, 0, 2);
        LK_combined_batch2_spfft_device(M1, M2, k_ext, spfft_ext, 1, 3);
        LK_combined_batch2_spfft_device(J1, J2, k_int, sp_i,      4, 6);
        LK_combined_batch2_spfft_device(M1, M2, k_int, sp_i,      5, 7);

        bem_apply_corr_assemble_batch2_kernel<<<grid_N, block>>>(
            d_mv1_re, d_mv1_im, d_mv2_re, d_mv2_im,
            d_full_x1_re, d_full_x1_im, d_full_x2_re, d_full_x2_im,
            d_corr_row_ptr, d_corr_col_idx,
            d_corr_L_ext_re, d_corr_L_ext_im,
            d_corr_K_ext_re, d_corr_K_ext_im,
            d_corr_L_int_re, d_corr_L_int_im,
            d_corr_K_int_re, d_corr_K_int_im,
            d_corr_I,
            N, eta_ext.real(), eta_int.real(), unknown_m_scale, row_h_scale, int_op_sign, k_identity,
            n_form ? 1 : 0, n_form_eps_int,
            d_y1_complex, d_y2_complex);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(y1, d_y1_complex, system_size * sizeof(double2), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y2, d_y2_complex, system_size * sizeof(double2), cudaMemcpyDeviceToHost));
        return;
#endif
    }

    HelmholtzFMM& fmm_i = shared_fmm ? fmm_ext : fmm_int;

    int block = 256;
    int grid_N = (N + block - 1) / block;
    int grid_sys = (system_size + block - 1) / block;

    CUDA_CHECK(cudaMemcpy(d_full_x1_complex, x1_full, system_size * sizeof(double2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_full_x2_complex, x2_full, system_size * sizeof(double2), cudaMemcpyHostToDevice));
    bem_split_complex_kernel<<<grid_sys, block>>>(d_full_x1_complex, d_full_x1_re, d_full_x1_im, system_size);
    bem_split_complex_kernel<<<grid_sys, block>>>(d_full_x2_complex, d_full_x2_re, d_full_x2_im, system_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(d_mv1_re, 0, 10 * N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mv1_im, 0, 10 * N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mv2_re, 0, 10 * N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mv2_im, 0, 10 * N * sizeof(double)));

    bool use_batch4 = (unknown_m_scale == 1.0 && std::getenv("BEM_FMM_BATCH4"));
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
    } else if (unknown_m_scale == 1.0) {
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
    } else {
        LK_combined_batch2_device(J1, J2, k_ext, fmm_ext, 0, 2);
        LK_combined_batch2_device(M1, M2, k_ext, fmm_ext, 1, 3);
        LK_combined_batch2_device(J1, J2, k_int, fmm_i,   4, 6);
        LK_combined_batch2_device(M1, M2, k_int, fmm_i,   5, 7);
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
        N, eta_ext.real(), eta_int.real(), unknown_m_scale, row_h_scale, int_op_sign, k_identity,
        n_form ? 1 : 0, n_form_eps_int,
        d_y1_complex, d_y2_complex);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(y1, d_y1_complex, system_size * sizeof(double2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y2, d_y2_complex, system_size * sizeof(double2), cudaMemcpyDeviceToHost));
}

void BemFmmOperator::matvec(const cdouble* x_full, cdouble* y)
{
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
            cdouble K_sum_M_n = mv_K_ext_M[m] - n_form_eps_int * mv_K_int_M[m] -
                                0.5 * (1.0 + n_form_eps_int) * IM;
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
