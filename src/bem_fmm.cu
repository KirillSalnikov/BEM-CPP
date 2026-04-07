#include "bem_fmm.h"
#include "graglia.h"
#include <cstring>
#include <cstdio>
#include <cmath>
#include <map>
#include <algorithm>
#include <cuda_runtime.h>

// ============================================================
// GPU charge packing kernels
// Pack RWG coefficient x[n] into FMM source charges on GPU
// ============================================================

// Vector charge: charges[n*Nq+q] = f[idx*3+d] * jw[idx] * x_re/im[n]
// for both plus (offset 0) and minus (offset N*Nq) halves
__global__ void pack_charges_vector_kernel(
    const double* __restrict__ x_re,
    const double* __restrict__ x_im,
    const double* __restrict__ f_p,     // (N*Nq*3) basis values, plus half
    const double* __restrict__ f_m,     // (N*Nq*3) basis values, minus half
    const double* __restrict__ jw_p,    // (N*Nq) Jacobian weights, plus half
    const double* __restrict__ jw_m,    // (N*Nq) Jacobian weights, minus half
    double* __restrict__ charges_re,    // (2*N*Nq) output
    double* __restrict__ charges_im,
    int N, int Nq, int d)              // d = component (0,1,2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Nq;
    if (tid >= total) return;

    int n = tid / Nq;
    double xr = x_re[n], xi = x_im[n];

    // Plus half
    double fp = f_p[tid * 3 + d] * jw_p[tid];
    charges_re[tid] = fp * xr;
    charges_im[tid] = fp * xi;

    // Minus half
    double fm = f_m[tid * 3 + d] * jw_m[tid];
    charges_re[total + tid] = fm * xr;
    charges_im[total + tid] = fm * xi;
}

// Scalar charge: charges[n*Nq+q] = div[n] * jw[idx] * x_re/im[n]
__global__ void pack_charges_scalar_kernel(
    const double* __restrict__ x_re,
    const double* __restrict__ x_im,
    const double* __restrict__ div_p,   // (N) divergence, plus half
    const double* __restrict__ div_m,   // (N) divergence, minus half
    const double* __restrict__ jw_p,    // (N*Nq)
    const double* __restrict__ jw_m,
    double* __restrict__ charges_re,
    double* __restrict__ charges_im,
    int N, int Nq)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Nq;
    if (tid >= total) return;

    int n = tid / Nq;
    double xr = x_re[n], xi = x_im[n];

    double dp = div_p[n] * jw_p[tid];
    charges_re[tid] = dp * xr;
    charges_im[tid] = dp * xi;

    double dm = div_m[n] * jw_m[tid];
    charges_re[total + tid] = dm * xr;
    charges_im[total + tid] = dm * xi;
}

// Accumulate L vector part: L_result[m] += ik * sum_q(f[idx*3+d]*jw[idx]*phi[idx])
__global__ void accum_L_vector_kernel(
    const double* __restrict__ phi_re,
    const double* __restrict__ phi_im,
    const double* __restrict__ f_p,
    const double* __restrict__ f_m,
    const double* __restrict__ jw_p,
    const double* __restrict__ jw_m,
    double* __restrict__ L_re,          // (N) output, accumulated
    double* __restrict__ L_im,
    double ik_re, double ik_im,         // ik = i*k
    int N, int Nq, int d)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;

    double acc_re = 0.0, acc_im = 0.0;
    for (int q = 0; q < Nq; q++) {
        int idx = m * Nq + q;
        double w;

        // Plus half
        w = f_p[idx * 3 + d] * jw_p[idx];
        acc_re += w * phi_re[idx];
        acc_im += w * phi_im[idx];

        // Minus half
        int idx2 = N * Nq + idx;
        w = f_m[idx * 3 + d] * jw_m[idx];
        acc_re += w * phi_re[idx2];
        acc_im += w * phi_im[idx2];
    }

    // Multiply by ik: (ik_re + i*ik_im) * (acc_re + i*acc_im)
    L_re[m] += ik_re * acc_re - ik_im * acc_im;
    L_im[m] += ik_re * acc_im + ik_im * acc_re;
}

// Accumulate L scalar part: L_result[m] -= iok * (div_p[m]*acc_p + div_m[m]*acc_m)
__global__ void accum_L_scalar_kernel(
    const double* __restrict__ phi_re,
    const double* __restrict__ phi_im,
    const double* __restrict__ div_p,
    const double* __restrict__ div_m,
    const double* __restrict__ jw_p,
    const double* __restrict__ jw_m,
    double* __restrict__ L_re,
    double* __restrict__ L_im,
    double iok_re, double iok_im,       // iok = i/k
    int N, int Nq)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;

    double acc_p_re = 0.0, acc_p_im = 0.0;
    double acc_m_re = 0.0, acc_m_im = 0.0;
    for (int q = 0; q < Nq; q++) {
        int idx = m * Nq + q;
        acc_p_re += jw_p[idx] * phi_re[idx];
        acc_p_im += jw_p[idx] * phi_im[idx];

        int idx2 = N * Nq + idx;
        acc_m_re += jw_m[idx] * phi_re[idx2];
        acc_m_im += jw_m[idx] * phi_im[idx2];
    }

    double dp = div_p[m], dm = div_m[m];
    double total_re = dp * acc_p_re + dm * acc_m_re;
    double total_im = dp * acc_p_im + dm * acc_m_im;

    // Subtract iok * total: -(iok_re + i*iok_im) * (total_re + i*total_im)
    L_re[m] -= iok_re * total_re - iok_im * total_im;
    L_im[m] -= iok_re * total_im + iok_im * total_re;
}

// K operator: curl(grad) dot f, accumulated from 3 gradient arrays (one per source component d)
// grad_d[i*3+c] = gradient of G with source charges along component d, target i, direction c
// K[m] = sum_q jw * (f · curl), where curl = cross(grad) of the 3 source components
// grad_d is stored as separate re/im for each of d=0,1,2 → 6 arrays total
// Each grad_d has layout (2*N*Nq, 3) interleaved [gx,gy,gz, gx,gy,gz, ...]
__global__ void accum_K_curl_kernel(
    const double* __restrict__ grad0_re,  // (2*N*Nq*3) gradient from d=0 source charges
    const double* __restrict__ grad0_im,
    const double* __restrict__ grad1_re,  // (2*N*Nq*3) gradient from d=1
    const double* __restrict__ grad1_im,
    const double* __restrict__ grad2_re,  // (2*N*Nq*3) gradient from d=2
    const double* __restrict__ grad2_im,
    const double* __restrict__ f_p,       // (N*Nq*3)
    const double* __restrict__ f_m,       // (N*Nq*3)
    const double* __restrict__ jw_p,      // (N*Nq)
    const double* __restrict__ jw_m,      // (N*Nq)
    double* __restrict__ K_re,            // (N) output
    double* __restrict__ K_im,
    int N, int Nq)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;

    double acc_re = 0.0, acc_im = 0.0;

    // Plus half
    for (int q = 0; q < Nq; q++) {
        int idx = m * Nq + q;
        int i3 = idx * 3;

        // curl_x = dPhi_z/dy - dPhi_y/dz = grad2[i*3+1] - grad1[i*3+2]
        double curl_x_re = grad2_re[i3+1] - grad1_re[i3+2];
        double curl_x_im = grad2_im[i3+1] - grad1_im[i3+2];
        // curl_y = dPhi_x/dz - dPhi_z/dx = grad0[i*3+2] - grad2[i*3+0]
        double curl_y_re = grad0_re[i3+2] - grad2_re[i3+0];
        double curl_y_im = grad0_im[i3+2] - grad2_im[i3+0];
        // curl_z = dPhi_y/dx - dPhi_x/dy = grad1[i*3+0] - grad0[i*3+1]
        double curl_z_re = grad1_re[i3+0] - grad0_re[i3+1];
        double curl_z_im = grad1_im[i3+0] - grad0_im[i3+1];

        double fx = f_p[idx*3], fy = f_p[idx*3+1], fz = f_p[idx*3+2];
        double w = jw_p[idx];
        acc_re += w * (fx * curl_x_re + fy * curl_y_re + fz * curl_z_re);
        acc_im += w * (fx * curl_x_im + fy * curl_y_im + fz * curl_z_im);
    }

    // Minus half
    int offset = N * Nq;
    for (int q = 0; q < Nq; q++) {
        int idx = m * Nq + q;
        int i = offset + idx;
        int i3 = i * 3;

        double curl_x_re = grad2_re[i3+1] - grad1_re[i3+2];
        double curl_x_im = grad2_im[i3+1] - grad1_im[i3+2];
        double curl_y_re = grad0_re[i3+2] - grad2_re[i3+0];
        double curl_y_im = grad0_im[i3+2] - grad2_im[i3+0];
        double curl_z_re = grad1_re[i3+0] - grad0_re[i3+1];
        double curl_z_im = grad1_im[i3+0] - grad0_im[i3+1];

        double fx = f_m[idx*3], fy = f_m[idx*3+1], fz = f_m[idx*3+2];
        double w = jw_m[idx];
        acc_re += w * (fx * curl_x_re + fy * curl_y_re + fz * curl_z_re);
        acc_im += w * (fx * curl_x_im + fy * curl_y_im + fz * curl_z_im);
    }

    K_re[m] = acc_re;
    K_im[m] = acc_im;
}

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
    use_pfft = use_pfft_;
    use_spfft = use_spfft_;
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

    // Upload basis data to GPU for charge packing kernels
    {
        gpu_pack_ready = false;
        cudaMalloc(&d_f_p, N * Nq * 3 * sizeof(double));
        cudaMalloc(&d_f_m, N * Nq * 3 * sizeof(double));
        cudaMalloc(&d_jw_p, N * Nq * sizeof(double));
        cudaMalloc(&d_jw_m, N * Nq * sizeof(double));
        cudaMalloc(&d_div_p, N * sizeof(double));
        cudaMalloc(&d_div_m, N * sizeof(double));
        cudaMemcpy(d_f_p, f_p.data(), N * Nq * 3 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_f_m, f_m.data(), N * Nq * 3 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_jw_p, jw_p.data(), N * Nq * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_jw_m, jw_m.data(), N * Nq * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_p, div_p.data(), N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_div_m, div_m.data(), N * sizeof(double), cudaMemcpyHostToDevice);
        // Workspace
        cudaMalloc(&d_x_re, N * sizeof(double));
        cudaMalloc(&d_x_im, N * sizeof(double));
        cudaMalloc(&d_src_re, total_pts * sizeof(double));
        cudaMalloc(&d_src_im, total_pts * sizeof(double));
        cudaMalloc(&d_phi_re, total_pts * sizeof(double));
        cudaMalloc(&d_phi_im, total_pts * sizeof(double));
        cudaMalloc(&d_L_re, N * sizeof(double));
        cudaMalloc(&d_L_im, N * sizeof(double));
        cudaMalloc(&d_K_re, N * sizeof(double));
        cudaMalloc(&d_K_im, N * sizeof(double));
        // Gradient buffers for K operator (3 sets × re/im × Nt*3)
        int grad_size = total_pts * 3;
        for (int d = 0; d < 3; d++) {
            cudaMalloc(&d_grad_buf_re[d], grad_size * sizeof(double));
            cudaMalloc(&d_grad_buf_im[d], grad_size * sizeof(double));
        }
        gpu_pack_ready = true;
        h_x_split_re.resize(N);
        h_x_split_im.resize(N);
        double gpu_mb = (N * Nq * 3 * 2 + N * Nq * 2 + N * 2 + N + total_pts * 2 +
                         total_pts * 2 + N * 2 + N * 2 + grad_size * 6) * 8.0 / 1e6;
        printf("  [BEM-FMM] GPU charge packing arrays uploaded: %.1f MB\n", gpu_mb);
    }

    // Build accelerator (FMM or pFFT)
    // Targets = sources = all quad points (self-interaction)
    shared_fmm = (std::abs(k_int - k_ext) < 1e-10);

    if (use_spfft) {
        // Surface pFFT: classify quad points by face
        // Hex prism: 24 base triangles, base_tri = t >> (2*ref), base_type = base_tri % 4
        // 0=top, 1=bottom, 2,3=side (6 sides, pair index = base_tri/4)
        int n_tri = mesh.nt();
        int ref = 0;
        { int nt = n_tri; while (nt > 24) { nt /= 4; ref++; } }

        // Determine face normals from mesh
        int n_face = 8;  // top, bottom, 6 sides
        std::vector<double> face_normals(n_face * 3, 0.0);
        // Top face (face 0): average normal of top triangles
        // Bottom face (face 1): average normal of bottom triangles
        // Side faces 2..7: average normal of each side pair
        for (int t = 0; t < n_tri; t++) {
            int base_tri = t >> (2 * ref);
            int base_type = base_tri % 4;
            int face_idx;
            if (base_type == 0) face_idx = 0;       // top
            else if (base_type == 1) face_idx = 1;   // bottom
            else face_idx = 2 + base_tri / 4;        // side

            Vec3 v0, v1, v2;
            mesh.tri_verts(t, v0, v1, v2);
            Vec3 n = (v1 - v0).cross(v2 - v0);
            double nm = n.norm();
            if (nm > 1e-15) {
                face_normals[face_idx*3+0] += n.x;
                face_normals[face_idx*3+1] += n.y;
                face_normals[face_idx*3+2] += n.z;
            }
        }
        // Normalize
        for (int f = 0; f < n_face; f++) {
            double nn = sqrt(face_normals[f*3+0]*face_normals[f*3+0] +
                             face_normals[f*3+1]*face_normals[f*3+1] +
                             face_normals[f*3+2]*face_normals[f*3+2]);
            if (nn > 1e-15) {
                face_normals[f*3+0] /= nn;
                face_normals[f*3+1] /= nn;
                face_normals[f*3+2] /= nn;
            }
        }

        // Classify each quad point by face
        // all_pts layout: [qpts_p (N*Nq); qpts_m (N*Nq)]
        // Point (n,q) in plus half -> triangle rwg.tri_p[n]
        // Point (n,q) in minus half -> triangle rwg.tri_m[n]
        std::vector<int> face_ids(total_pts);
        for (int n = 0; n < N; n++) {
            int tri_p = rwg.tri_p[n];
            int base_tri_p = tri_p >> (2 * ref);
            int base_type_p = base_tri_p % 4;
            int face_p = (base_type_p == 0) ? 0 : (base_type_p == 1) ? 1 : 2 + base_tri_p / 4;

            int tri_m = rwg.tri_m[n];
            int base_tri_m = tri_m >> (2 * ref);
            int base_type_m = base_tri_m % 4;
            int face_m = (base_type_m == 0) ? 0 : (base_type_m == 1) ? 1 : 2 + base_tri_m / 4;

            for (int q = 0; q < Nq; q++) {
                face_ids[n * Nq + q] = face_p;
                face_ids[N * Nq + n * Nq + q] = face_m;
            }
        }

        printf("  [BEM-SurfPFFT] Building surface pFFT for k_ext...\n");
        spfft_ext.init(all_pts.data(), total_pts,
                       face_ids.data(), n_face,
                       face_normals.data(),
                       k_ext, fmm_digits);
        if (!shared_fmm) {
            printf("  [BEM-SurfPFFT] Building surface pFFT for k_int...\n");
            spfft_int.init(all_pts.data(), total_pts,
                           face_ids.data(), n_face,
                           face_normals.data(),
                           k_int, fmm_digits);
        }
    } else if (use_pfft) {
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
    } else {
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

    // Batch-4 workspace (extra buffers for batched P2P — reuse tmp_src_charges, tmp2_src_charges, tmp_phi, tmp2_phi for batches 0,1)
    b4_src2.resize(total_pts);
    b4_src3.resize(total_pts);
    b4_pot2.resize(total_pts);
    b4_pot3.resize(total_pts);
    mv2_L_ext_J.resize(N, 0); mv2_L_ext_M.resize(N, 0);
    mv2_K_ext_J.resize(N, 0); mv2_K_ext_M.resize(N, 0);
    mv2_L_int_J.resize(N, 0); mv2_L_int_M.resize(N, 0);
    mv2_K_int_J.resize(N, 0); mv2_K_int_M.resize(N, 0);

    // Batch-8 workspace for LK_combined_batch2(spfft)
    for (int b = 0; b < 8; b++) b8_src[b].resize(total_pts);
    for (int b = 0; b < 8; b++) b8_pot[b].resize(total_pts);
    for (int b = 0; b < 6; b++) b8_grad[b].resize(total_pts * 3);

    // Precompute singular corrections
    printf("  [BEM-FMM] Computing singular corrections...\n");
    precompute_corrections(rwg, mesh, quad_order);

    printf("  [BEM-FMM] Init complete: %.1fs\n", timer.elapsed_s());
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
                    cdouble DL_vec(0), DL_scl(0);
                    for (int i = 0; i < Nq; i++) {
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

    if (gpu_pack_ready) {
        // === GPU-accelerated path: charge packing + accumulation on GPU ===

        // Upload coefficient vector x to GPU (N complex values → split re/im)
        for (int n = 0; n < N; n++) {
            fmm.h_q_re_buf[n] = x[n].real();
            fmm.h_q_im_buf[n] = x[n].imag();
        }
        cudaMemcpy(d_x_re, fmm.h_q_re_buf.data(), N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_im, fmm.h_q_im_buf.data(), N * sizeof(double), cudaMemcpyHostToDevice);

        // Zero L accumulator on GPU
        cudaMemset(d_L_re, 0, N * sizeof(double));
        cudaMemset(d_L_im, 0, N * sizeof(double));

        int pack_block = 256;
        int pack_grid = (N * Nq + pack_block - 1) / pack_block;
        int accum_block = 256;
        int accum_grid = (N + accum_block - 1) / accum_block;

        // --- Vector part: d=0,1,2 ---
        for (int d = 0; d < 3; d++) {
            // GPU charge packing
            pack_charges_vector_kernel<<<pack_grid, pack_block>>>(
                d_x_re, d_x_im, d_f_p, d_f_m, d_jw_p, d_jw_m,
                d_src_re, d_src_im, N, Nq, d);

            // GPU-resident FMM: pot+grad, results stay on device
            fmm.evaluate_pot_grad_gpu(d_src_re, d_src_im,
                d_phi_re, d_phi_im,
                d_grad_buf_re[d], d_grad_buf_im[d]);

            // GPU L accumulation from potential
            accum_L_vector_kernel<<<accum_grid, accum_block>>>(
                d_phi_re, d_phi_im, d_f_p, d_f_m, d_jw_p, d_jw_m,
                d_L_re, d_L_im,
                ik.real(), ik.imag(), N, Nq, d);
        }

        // --- Scalar part ---
        pack_charges_scalar_kernel<<<pack_grid, pack_block>>>(
            d_x_re, d_x_im, d_div_p, d_div_m, d_jw_p, d_jw_m,
            d_src_re, d_src_im, N, Nq);

        fmm.evaluate_gpu(d_src_re, d_src_im, d_phi_re, d_phi_im);

        accum_L_scalar_kernel<<<accum_grid, accum_block>>>(
            d_phi_re, d_phi_im, d_div_p, d_div_m, d_jw_p, d_jw_m,
            d_L_re, d_L_im,
            iok.real(), iok.imag(), N, Nq);

        // --- K: curl assembly from 3 gradient buffers on GPU ---
        accum_K_curl_kernel<<<accum_grid, accum_block>>>(
            d_grad_buf_re[0], d_grad_buf_im[0],
            d_grad_buf_re[1], d_grad_buf_im[1],
            d_grad_buf_re[2], d_grad_buf_im[2],
            d_f_p, d_f_m, d_jw_p, d_jw_m,
            d_K_re, d_K_im, N, Nq);
        cudaDeviceSynchronize();

        // Download L and K results (small: N values each)
        std::vector<double> h_L_re(N), h_L_im(N), h_K_re(N), h_K_im(N);
        cudaMemcpy(h_L_re.data(), d_L_re, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_L_im.data(), d_L_im, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_K_re.data(), d_K_re, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_K_im.data(), d_K_im, N * sizeof(double), cudaMemcpyDeviceToHost);
        for (int m = 0; m < N; m++) {
            L_result[m] = cdouble(h_L_re[m], h_L_im[m]);
            K_result[m] = cdouble(h_K_re[m], h_K_im[m]);
        }
    } else {
        // === CPU fallback path (original implementation) ===

        // --- Vector part ---
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
            fmm.evaluate_pot_grad(tmp_src_charges.data(), tmp_phi.data(), tmp_grad[d].data());
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

        // --- Scalar part ---
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

        // --- K: curl assembly ---
        for (int m = 0; m < N; m++) {
            cdouble acc(0);
            for (int q = 0; q < Nq; q++) {
                int idx = m * Nq + q;
                int i = idx;
                cdouble curl_x = tmp_grad[2][i*3+1] - tmp_grad[1][i*3+2];
                cdouble curl_y = tmp_grad[0][i*3+2] - tmp_grad[2][i*3+0];
                cdouble curl_z = tmp_grad[1][i*3+0] - tmp_grad[0][i*3+1];
                double fx = f_p[idx*3], fy = f_p[idx*3+1], fz = f_p[idx*3+2];
                acc += jw_p[idx] * (fx * curl_x + fy * curl_y + fz * curl_z);
            }
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
}

// pFFT variant of LK_combined: identical algorithm, but calls pfft instead of fmm
void BemFmmOperator::LK_combined(const cdouble* x, cdouble kv, HelmholtzPFFT& pf,
                                  cdouble* L_result, cdouble* K_result)
{
    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;

    int total_pts = 2 * N * Nq;

    memset(L_result, 0, N * sizeof(cdouble));
    memset(K_result, 0, N * sizeof(cdouble));

    // --- Vector part: combined potential (for L) + gradient (for K) ---
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

        pf.evaluate_pot_grad(tmp_src_charges.data(), tmp_phi.data(), tmp_grad[d].data());

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

    // --- L scalar part ---
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

        pf.evaluate(tmp_src_charges.data(), tmp_phi.data());

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

    // --- K: assemble curl from gradients ---
    for (int m = 0; m < N; m++) {
        cdouble acc(0);

        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            int i = idx;
            cdouble curl_x = tmp_grad[2][i*3+1] - tmp_grad[1][i*3+2];
            cdouble curl_y = tmp_grad[0][i*3+2] - tmp_grad[2][i*3+0];
            cdouble curl_z = tmp_grad[1][i*3+0] - tmp_grad[0][i*3+1];
            double fx = f_p[idx*3], fy = f_p[idx*3+1], fz = f_p[idx*3+2];
            acc += jw_p[idx] * (fx * curl_x + fy * curl_y + fz * curl_z);
        }

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

// Surface pFFT variant of LK_combined: uses batched P2P (4 charge vectors in 1 pass)
void BemFmmOperator::LK_combined(const cdouble* x, cdouble kv, HelmholtzSurfacePFFT& spf,
                                  cdouble* L_result, cdouble* K_result)
{
    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;

    int total_pts = 2 * N * Nq;

    memset(L_result, 0, N * sizeof(cdouble));
    memset(K_result, 0, N * sizeof(cdouble));

    // Use 4 buffers: tmp_src_charges (b0), tmp2_src_charges (b1), b4_src2 (b2), b4_src3 (b3)
    cdouble* src[4] = { tmp_src_charges.data(), tmp2_src_charges.data(), b4_src2.data(), b4_src3.data() };
    cdouble* pot[4] = { tmp_phi.data(), tmp2_phi.data(), b4_pot2.data(), b4_pot3.data() };

    // --- Pack all 4 charge vectors ---
    for (int d = 0; d < 3; d++) {
        memset(src[d], 0, total_pts * sizeof(cdouble));
        for (int n = 0; n < N; n++) {
            cdouble xn = x[n];
            for (int q = 0; q < Nq; q++) {
                int idx = n * Nq + q;
                src[d][idx] = f_p[idx*3 + d] * jw_p[idx] * xn;
                src[d][N*Nq + idx] = f_m[idx*3 + d] * jw_m[idx] * xn;
            }
        }
    }
    // Scalar (div) charges
    memset(src[3], 0, total_pts * sizeof(cdouble));
    for (int n = 0; n < N; n++) {
        cdouble xn = x[n];
        for (int q = 0; q < Nq; q++) {
            int idx = n * Nq + q;
            src[3][idx] = div_p[n] * jw_p[idx] * xn;
            src[3][N*Nq + idx] = div_m[n] * jw_m[idx] * xn;
        }
    }

    // --- Batched evaluate: 3× pot+grad + 1× pot in single P2P pass ---
    spf.evaluate_batch4(src[0], src[1], src[2], src[3],
                        pot[0], pot[1], pot[2], pot[3],
                        tmp_grad[0].data(), tmp_grad[1].data(), tmp_grad[2].data());

    // --- Accumulate L: vector part from pot[0..2] ---
    for (int d = 0; d < 3; d++) {
        for (int m = 0; m < N; m++) {
            cdouble acc(0);
            for (int q = 0; q < Nq; q++) {
                int idx = m * Nq + q;
                acc += f_p[idx*3 + d] * jw_p[idx] * pot[d][idx];
                acc += f_m[idx*3 + d] * jw_m[idx] * pot[d][N*Nq + idx];
            }
            L_result[m] += ik * acc;
        }
    }

    // --- L scalar part from pot[3] ---
    for (int m = 0; m < N; m++) {
        cdouble acc_p(0), acc_m(0);
        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            acc_p += jw_p[idx] * pot[3][idx];
            acc_m += jw_m[idx] * pot[3][N*Nq + idx];
        }
        L_result[m] -= iok * (div_p[m] * acc_p + div_m[m] * acc_m);
    }

    // --- K: assemble curl from gradients ---
    for (int m = 0; m < N; m++) {
        cdouble acc(0);

        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            int i = idx;
            cdouble curl_x = tmp_grad[2][i*3+1] - tmp_grad[1][i*3+2];
            cdouble curl_y = tmp_grad[0][i*3+2] - tmp_grad[2][i*3+0];
            cdouble curl_z = tmp_grad[1][i*3+0] - tmp_grad[0][i*3+1];
            double fx = f_p[idx*3], fy = f_p[idx*3+1], fz = f_p[idx*3+2];
            acc += jw_p[idx] * (fx * curl_x + fy * curl_y + fz * curl_z);
        }

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

// Surface pFFT batch2: two RHS × 4 evals = 8 charge vectors in ONE P2P pass
// GPU-native: pack + evaluate + accumulate all on GPU, only download L/K results
void BemFmmOperator::LK_combined_batch2(
    const cdouble* x1, const cdouble* x2,
    cdouble kv, HelmholtzSurfacePFFT& spf,
    cdouble* L_result1, cdouble* K_result1,
    cdouble* L_result2, cdouble* K_result2)
{
    static int lk_call_count = 0;
    bool do_timing = (lk_call_count == 0);
    lk_call_count++;
    Timer t_pack, t_eval, t_accum;

    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;
    int total_pts = 2 * N * Nq;

    memset(L_result1, 0, N * sizeof(cdouble));
    memset(K_result1, 0, N * sizeof(cdouble));
    memset(L_result2, 0, N * sizeof(cdouble));
    memset(K_result2, 0, N * sizeof(cdouble));

    if (!gpu_pack_ready) {
        // Fallback: CPU path (should not happen if init allocated GPU arrays)
        // ... (omitted for brevity — old code path)
        fprintf(stderr, "WARNING: GPU pack not ready in SurfPFFT LK_combined_batch2\n");
        return;
    }

    // === GPU-native path: pack + evaluate + accumulate all on GPU ===
    if (do_timing) t_pack.reset();

    int pack_block = 256;
    int pack_grid = (N * Nq + pack_block - 1) / pack_block;
    int accum_block = 256;
    int accum_grid = (N + accum_block - 1) / accum_block;

    // Upload x1 coefficients to GPU (N complex → split re/im, ~120KB)
    for (int n = 0; n < N; n++) {
        h_x_split_re[n] = x1[n].real();
        h_x_split_im[n] = x1[n].imag();
    }
    CUDA_CHECK(cudaMemcpy(d_x_re, h_x_split_re.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_im, h_x_split_im.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // GPU pack batches 0-2 (vector, x1) and 3 (scalar, x1) into spf.d_batch8_re/im
    for (int d = 0; d < 3; d++) {
        pack_charges_vector_kernel<<<pack_grid, pack_block>>>(
            d_x_re, d_x_im, d_f_p, d_f_m, d_jw_p, d_jw_m,
            spf.d_batch8_re + d * total_pts, spf.d_batch8_im + d * total_pts,
            N, Nq, d);
    }
    pack_charges_scalar_kernel<<<pack_grid, pack_block>>>(
        d_x_re, d_x_im, d_div_p, d_div_m, d_jw_p, d_jw_m,
        spf.d_batch8_re + 3 * total_pts, spf.d_batch8_im + 3 * total_pts,
        N, Nq);

    // Upload x2 coefficients and pack batches 4-7
    for (int n = 0; n < N; n++) {
        h_x_split_re[n] = x2[n].real();
        h_x_split_im[n] = x2[n].imag();
    }
    CUDA_CHECK(cudaMemcpy(d_x_re, h_x_split_re.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_im, h_x_split_im.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    for (int d = 0; d < 3; d++) {
        pack_charges_vector_kernel<<<pack_grid, pack_block>>>(
            d_x_re, d_x_im, d_f_p, d_f_m, d_jw_p, d_jw_m,
            spf.d_batch8_re + (4 + d) * total_pts, spf.d_batch8_im + (4 + d) * total_pts,
            N, Nq, d);
    }
    pack_charges_scalar_kernel<<<pack_grid, pack_block>>>(
        d_x_re, d_x_im, d_div_p, d_div_m, d_jw_p, d_jw_m,
        spf.d_batch8_re + 7 * total_pts, spf.d_batch8_im + 7 * total_pts,
        N, Nq);

    if (do_timing) printf("    [LK_B2-GPU] pack: %.1fms\n", t_pack.elapsed_ms());

    // GPU-native evaluate: no H2D upload, no D2H download
    if (do_timing) t_eval.reset();
    spf.evaluate_batch8_gpu();
    if (do_timing) printf("    [LK_B2-GPU] eval: %.1fms\n", t_eval.elapsed_ms());

    // GPU accumulate L+K for RHS 1 from d_bp_res[0..3], d_bp_grd[0..2]
    if (do_timing) t_accum.reset();

    // Zero L/K accumulators
    CUDA_CHECK(cudaMemset(d_L_re, 0, N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_L_im, 0, N * sizeof(double)));

    // L vector: d=0,1,2
    for (int d = 0; d < 3; d++) {
        accum_L_vector_kernel<<<accum_grid, accum_block>>>(
            spf.d_bp_res_re[d], spf.d_bp_res_im[d],
            d_f_p, d_f_m, d_jw_p, d_jw_m,
            d_L_re, d_L_im,
            ik.real(), ik.imag(), N, Nq, d);
    }
    // L scalar
    accum_L_scalar_kernel<<<accum_grid, accum_block>>>(
        spf.d_bp_res_re[3], spf.d_bp_res_im[3],
        d_div_p, d_div_m, d_jw_p, d_jw_m,
        d_L_re, d_L_im,
        iok.real(), iok.imag(), N, Nq);
    // K curl
    accum_K_curl_kernel<<<accum_grid, accum_block>>>(
        spf.d_bp_grd_re[0], spf.d_bp_grd_im[0],
        spf.d_bp_grd_re[1], spf.d_bp_grd_im[1],
        spf.d_bp_grd_re[2], spf.d_bp_grd_im[2],
        d_f_p, d_f_m, d_jw_p, d_jw_m,
        d_K_re, d_K_im, N, Nq);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Download L1/K1 (small: 4*N doubles = ~240KB)
    std::vector<double> h_L_re(N), h_L_im(N), h_K_re(N), h_K_im(N);
    CUDA_CHECK(cudaMemcpy(h_L_re.data(), d_L_re, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_L_im.data(), d_L_im, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_K_re.data(), d_K_re, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_K_im.data(), d_K_im, N * sizeof(double), cudaMemcpyDeviceToHost));
    for (int m = 0; m < N; m++) {
        L_result1[m] = cdouble(h_L_re[m], h_L_im[m]);
        K_result1[m] = cdouble(h_K_re[m], h_K_im[m]);
    }

    // GPU accumulate L+K for RHS 2 from d_bp_res[4..7], d_bp_grd[3..5]
    CUDA_CHECK(cudaMemset(d_L_re, 0, N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_L_im, 0, N * sizeof(double)));

    for (int d = 0; d < 3; d++) {
        accum_L_vector_kernel<<<accum_grid, accum_block>>>(
            spf.d_bp_res_re[4 + d], spf.d_bp_res_im[4 + d],
            d_f_p, d_f_m, d_jw_p, d_jw_m,
            d_L_re, d_L_im,
            ik.real(), ik.imag(), N, Nq, d);
    }
    accum_L_scalar_kernel<<<accum_grid, accum_block>>>(
        spf.d_bp_res_re[7], spf.d_bp_res_im[7],
        d_div_p, d_div_m, d_jw_p, d_jw_m,
        d_L_re, d_L_im,
        iok.real(), iok.imag(), N, Nq);
    accum_K_curl_kernel<<<accum_grid, accum_block>>>(
        spf.d_bp_grd_re[3], spf.d_bp_grd_im[3],
        spf.d_bp_grd_re[4], spf.d_bp_grd_im[4],
        spf.d_bp_grd_re[5], spf.d_bp_grd_im[5],
        d_f_p, d_f_m, d_jw_p, d_jw_m,
        d_K_re, d_K_im, N, Nq);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Download L2/K2
    CUDA_CHECK(cudaMemcpy(h_L_re.data(), d_L_re, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_L_im.data(), d_L_im, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_K_re.data(), d_K_re, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_K_im.data(), d_K_im, N * sizeof(double), cudaMemcpyDeviceToHost));
    for (int m = 0; m < N; m++) {
        L_result2[m] = cdouble(h_L_re[m], h_L_im[m]);
        K_result2[m] = cdouble(h_K_re[m], h_K_im[m]);
    }

    if (do_timing) printf("    [LK_B2-GPU] accum+download: %.1fms\n", t_accum.elapsed_ms());
}

void BemFmmOperator::LK_combined_batch2(
    const cdouble* x1, const cdouble* x2,
    cdouble kv, HelmholtzFMM& fmm,
    cdouble* L_result1, cdouble* K_result1,
    cdouble* L_result2, cdouble* K_result2)
{
    cdouble ik = cdouble(0, 1) * kv;
    cdouble iok = cdouble(0, 1) / kv;

    int total_pts = 2 * N * Nq;

    memset(L_result1, 0, N * sizeof(cdouble));
    memset(K_result1, 0, N * sizeof(cdouble));
    memset(L_result2, 0, N * sizeof(cdouble));
    memset(K_result2, 0, N * sizeof(cdouble));

    // --- Vector part: combined potential (for L) + gradient (for K) in one FMM pass ---
    for (int d = 0; d < 3; d++) {
        // OMP-parallel charge packing for both vectors
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            cdouble xn1 = x1[n];
            cdouble xn2 = x2[n];
            for (int q = 0; q < Nq; q++) {
                int idx = n * Nq + q;
                double fp_d = f_p[idx*3 + d];
                double fm_d = f_m[idx*3 + d];
                double jwp = jw_p[idx];
                double jwm = jw_m[idx];

                tmp_src_charges[idx]        = fp_d * jwp * xn1;
                tmp_src_charges[N*Nq + idx] = fm_d * jwm * xn1;

                tmp2_src_charges[idx]        = fp_d * jwp * xn2;
                tmp2_src_charges[N*Nq + idx] = fm_d * jwm * xn2;
            }
        }

        // Batched FMM pass: get potential and gradient for both charge vectors
        fmm.evaluate_pot_grad_batch2(tmp_src_charges.data(), tmp2_src_charges.data(),
                                      tmp_phi.data(), tmp_grad[d].data(),
                                      tmp2_phi.data(), tmp2_grad[d].data());

        // OMP-parallel L accumulation for both RHS
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < N; m++) {
            cdouble acc1(0), acc2(0);
            for (int q = 0; q < Nq; q++) {
                int idx = m * Nq + q;
                double fp_d = f_p[idx*3 + d];
                double fm_d = f_m[idx*3 + d];
                double jwp = jw_p[idx];
                double jwm = jw_m[idx];

                acc1 += fp_d * jwp * tmp_phi[idx];
                acc1 += fm_d * jwm * tmp_phi[N*Nq + idx];

                acc2 += fp_d * jwp * tmp2_phi[idx];
                acc2 += fm_d * jwm * tmp2_phi[N*Nq + idx];
            }
            L_result1[m] += ik * acc1;
            L_result2[m] += ik * acc2;
        }
    }

    // --- L scalar part: potential only (no gradient needed) ---
    {
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            cdouble xn1 = x1[n];
            cdouble xn2 = x2[n];
            double dp = div_p[n];
            double dm_v = div_m[n];
            for (int q = 0; q < Nq; q++) {
                int idx = n * Nq + q;
                double jwp = jw_p[idx];
                double jwm = jw_m[idx];

                tmp_src_charges[idx]        = dp * jwp * xn1;
                tmp_src_charges[N*Nq + idx] = dm_v * jwm * xn1;

                tmp2_src_charges[idx]        = dp * jwp * xn2;
                tmp2_src_charges[N*Nq + idx] = dm_v * jwm * xn2;
            }
        }

        fmm.evaluate_batch2(tmp_src_charges.data(), tmp2_src_charges.data(),
                             tmp_phi.data(), tmp2_phi.data());

        #pragma omp parallel for schedule(static)
        for (int m = 0; m < N; m++) {
            cdouble acc_p1(0), acc_m1(0);
            cdouble acc_p2(0), acc_m2(0);
            for (int q = 0; q < Nq; q++) {
                int idx = m * Nq + q;
                double jwp = jw_p[idx];
                double jwm = jw_m[idx];

                acc_p1 += jwp * tmp_phi[idx];
                acc_m1 += jwm * tmp_phi[N*Nq + idx];

                acc_p2 += jwp * tmp2_phi[idx];
                acc_m2 += jwm * tmp2_phi[N*Nq + idx];
            }
            L_result1[m] -= iok * (div_p[m] * acc_p1 + div_m[m] * acc_m1);
            L_result2[m] -= iok * (div_p[m] * acc_p2 + div_m[m] * acc_m2);
        }
    }

    // --- K: assemble curl from gradients computed above ---
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < N; m++) {
        cdouble acc1(0), acc2(0);

        // Plus half
        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            int i = idx;

            cdouble curl_x1 = tmp_grad[2][i*3+1] - tmp_grad[1][i*3+2];
            cdouble curl_y1 = tmp_grad[0][i*3+2] - tmp_grad[2][i*3+0];
            cdouble curl_z1 = tmp_grad[1][i*3+0] - tmp_grad[0][i*3+1];

            cdouble curl_x2 = tmp2_grad[2][i*3+1] - tmp2_grad[1][i*3+2];
            cdouble curl_y2 = tmp2_grad[0][i*3+2] - tmp2_grad[2][i*3+0];
            cdouble curl_z2 = tmp2_grad[1][i*3+0] - tmp2_grad[0][i*3+1];

            double fx = f_p[idx*3], fy = f_p[idx*3+1], fz = f_p[idx*3+2];
            acc1 += jw_p[idx] * (fx * curl_x1 + fy * curl_y1 + fz * curl_z1);
            acc2 += jw_p[idx] * (fx * curl_x2 + fy * curl_y2 + fz * curl_z2);
        }

        // Minus half
        for (int q = 0; q < Nq; q++) {
            int idx = m * Nq + q;
            int i = N*Nq + idx;

            cdouble curl_x1 = tmp_grad[2][i*3+1] - tmp_grad[1][i*3+2];
            cdouble curl_y1 = tmp_grad[0][i*3+2] - tmp_grad[2][i*3+0];
            cdouble curl_z1 = tmp_grad[1][i*3+0] - tmp_grad[0][i*3+1];

            cdouble curl_x2 = tmp2_grad[2][i*3+1] - tmp2_grad[1][i*3+2];
            cdouble curl_y2 = tmp2_grad[0][i*3+2] - tmp2_grad[2][i*3+0];
            cdouble curl_z2 = tmp2_grad[1][i*3+0] - tmp2_grad[0][i*3+1];

            double fx = f_m[idx*3], fy = f_m[idx*3+1], fz = f_m[idx*3+2];
            acc1 += jw_m[idx] * (fx * curl_x1 + fy * curl_y1 + fz * curl_z1);
            acc2 += jw_m[idx] * (fx * curl_x2 + fy * curl_y2 + fz * curl_z2);
        }

        K_result1[m] = acc1;
        K_result2[m] = acc2;
    }
}

void BemFmmOperator::matvec_batch2(const cdouble* x1_full, const cdouble* x2_full,
                                    cdouble* y1, cdouble* y2)
{
    const cdouble* J1 = x1_full;
    const cdouble* M1 = x1_full + N;
    const cdouble* J2 = x2_full;
    const cdouble* M2 = x2_full + N;

    // Clear all output buffers
    std::fill(mv_L_ext_J.begin(), mv_L_ext_J.end(), cdouble(0));
    std::fill(mv_L_ext_M.begin(), mv_L_ext_M.end(), cdouble(0));
    std::fill(mv_K_ext_J.begin(), mv_K_ext_J.end(), cdouble(0));
    std::fill(mv_K_ext_M.begin(), mv_K_ext_M.end(), cdouble(0));
    std::fill(mv_L_int_J.begin(), mv_L_int_J.end(), cdouble(0));
    std::fill(mv_L_int_M.begin(), mv_L_int_M.end(), cdouble(0));
    std::fill(mv_K_int_J.begin(), mv_K_int_J.end(), cdouble(0));
    std::fill(mv_K_int_M.begin(), mv_K_int_M.end(), cdouble(0));

    std::fill(mv2_L_ext_J.begin(), mv2_L_ext_J.end(), cdouble(0));
    std::fill(mv2_L_ext_M.begin(), mv2_L_ext_M.end(), cdouble(0));
    std::fill(mv2_K_ext_J.begin(), mv2_K_ext_J.end(), cdouble(0));
    std::fill(mv2_K_ext_M.begin(), mv2_K_ext_M.end(), cdouble(0));
    std::fill(mv2_L_int_J.begin(), mv2_L_int_J.end(), cdouble(0));
    std::fill(mv2_L_int_M.begin(), mv2_L_int_M.end(), cdouble(0));
    std::fill(mv2_K_int_J.begin(), mv2_K_int_J.end(), cdouble(0));
    std::fill(mv2_K_int_M.begin(), mv2_K_int_M.end(), cdouble(0));

    // Batched calls: batch8 for spfft (4 calls instead of 8), batch2 for FMM
    if (use_spfft) {
        HelmholtzSurfacePFFT& sp_i = shared_fmm ? spfft_ext : spfft_int;
        LK_combined_batch2(J1, J2, k_ext, spfft_ext,
                            mv_L_ext_J.data(), mv_K_ext_J.data(),
                            mv2_L_ext_J.data(), mv2_K_ext_J.data());
        LK_combined_batch2(M1, M2, k_ext, spfft_ext,
                            mv_L_ext_M.data(), mv_K_ext_M.data(),
                            mv2_L_ext_M.data(), mv2_K_ext_M.data());
        LK_combined_batch2(J1, J2, k_int, sp_i,
                            mv_L_int_J.data(), mv_K_int_J.data(),
                            mv2_L_int_J.data(), mv2_K_int_J.data());
        LK_combined_batch2(M1, M2, k_int, sp_i,
                            mv_L_int_M.data(), mv_K_int_M.data(),
                            mv2_L_int_M.data(), mv2_K_int_M.data());
    } else if (use_pfft) {
        HelmholtzPFFT& pf_i = shared_fmm ? pfft_ext : pfft_int;
        LK_combined(J1, k_ext, pfft_ext, mv_L_ext_J.data(), mv_K_ext_J.data());
        LK_combined(J2, k_ext, pfft_ext, mv2_L_ext_J.data(), mv2_K_ext_J.data());
        LK_combined(M1, k_ext, pfft_ext, mv_L_ext_M.data(), mv_K_ext_M.data());
        LK_combined(M2, k_ext, pfft_ext, mv2_L_ext_M.data(), mv2_K_ext_M.data());
        LK_combined(J1, k_int, pf_i,     mv_L_int_J.data(), mv_K_int_J.data());
        LK_combined(J2, k_int, pf_i,     mv2_L_int_J.data(), mv2_K_int_J.data());
        LK_combined(M1, k_int, pf_i,     mv_L_int_M.data(), mv_K_int_M.data());
        LK_combined(M2, k_int, pf_i,     mv2_L_int_M.data(), mv2_K_int_M.data());
    } else {
        HelmholtzFMM& fmm_i = shared_fmm ? fmm_ext : fmm_int;
        LK_combined_batch2(J1, J2, k_ext, fmm_ext,
                            mv_L_ext_J.data(), mv_K_ext_J.data(),
                            mv2_L_ext_J.data(), mv2_K_ext_J.data());
        LK_combined_batch2(M1, M2, k_ext, fmm_ext,
                            mv_L_ext_M.data(), mv_K_ext_M.data(),
                            mv2_L_ext_M.data(), mv2_K_ext_M.data());
        LK_combined_batch2(J1, J2, k_int, fmm_i,
                            mv_L_int_J.data(), mv_K_int_J.data(),
                            mv2_L_int_J.data(), mv2_K_int_J.data());
        LK_combined_batch2(M1, M2, k_int, fmm_i,
                            mv_L_int_M.data(), mv_K_int_M.data(),
                            mv2_L_int_M.data(), mv2_K_int_M.data());
    }

    // Apply singular corrections (sparse CSR) — for both RHS vectors
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < N; m++) {
        for (int j = corr_row_ptr[m]; j < corr_row_ptr[m + 1]; j++) {
            int n = corr_col_idx[j];

            // RHS 1
            mv_L_ext_J[m] += corr_L_ext_val[j] * J1[n];
            mv_L_ext_M[m] += corr_L_ext_val[j] * M1[n];
            mv_K_ext_J[m] += corr_K_ext_val[j] * J1[n];
            mv_K_ext_M[m] += corr_K_ext_val[j] * M1[n];

            mv_L_int_J[m] += corr_L_int_val[j] * J1[n];
            mv_L_int_M[m] += corr_L_int_val[j] * M1[n];
            mv_K_int_J[m] += corr_K_int_val[j] * J1[n];
            mv_K_int_M[m] += corr_K_int_val[j] * M1[n];

            // RHS 2
            mv2_L_ext_J[m] += corr_L_ext_val[j] * J2[n];
            mv2_L_ext_M[m] += corr_L_ext_val[j] * M2[n];
            mv2_K_ext_J[m] += corr_K_ext_val[j] * J2[n];
            mv2_K_ext_M[m] += corr_K_ext_val[j] * M2[n];

            mv2_L_int_J[m] += corr_L_int_val[j] * J2[n];
            mv2_L_int_M[m] += corr_L_int_val[j] * M2[n];
            mv2_K_int_J[m] += corr_K_int_val[j] * J2[n];
            mv2_K_int_M[m] += corr_K_int_val[j] * M2[n];
        }
    }

    // Assemble PMCHWT blocks for y1
    for (int m = 0; m < N; m++) {
        cdouble K_sum_J = mv_K_ext_J[m] + mv_K_int_J[m];
        cdouble K_sum_M = mv_K_ext_M[m] + mv_K_int_M[m];

        // y[:N] = (eta_ext*L_ext + eta_int*L_int)*J - (K_ext+K_int)*M
        y1[m] = (eta_ext * mv_L_ext_J[m] + eta_int * mv_L_int_J[m]) - K_sum_M;

        // y[N:] = (K_ext+K_int)*J + (L_ext/eta_ext + L_int/eta_int)*M
        y1[N + m] = K_sum_J + (mv_L_ext_M[m] / eta_ext + mv_L_int_M[m] / eta_int);
    }

    // Assemble PMCHWT blocks for y2
    for (int m = 0; m < N; m++) {
        cdouble K_sum_J = mv2_K_ext_J[m] + mv2_K_int_J[m];
        cdouble K_sum_M = mv2_K_ext_M[m] + mv2_K_int_M[m];

        y2[m] = (eta_ext * mv2_L_ext_J[m] + eta_int * mv2_L_int_J[m]) - K_sum_M;

        y2[N + m] = K_sum_J + (mv2_L_ext_M[m] / eta_ext + mv2_L_int_M[m] / eta_int);
    }
}

void BemFmmOperator::matvec(const cdouble* x_full, cdouble* y)
{
    const cdouble* J = x_full;
    const cdouble* M = x_full + N;

    // Combined L+K: 4 passes each (3 pot+grad + 1 pot) instead of 7 (4L + 3K)
    if (use_spfft) {
        HelmholtzSurfacePFFT& sp_i = shared_fmm ? spfft_ext : spfft_int;
        LK_combined(J, k_ext, spfft_ext, mv_L_ext_J.data(), mv_K_ext_J.data());
        LK_combined(M, k_ext, spfft_ext, mv_L_ext_M.data(), mv_K_ext_M.data());
        LK_combined(J, k_int, sp_i,      mv_L_int_J.data(), mv_K_int_J.data());
        LK_combined(M, k_int, sp_i,      mv_L_int_M.data(), mv_K_int_M.data());
    } else if (use_pfft) {
        HelmholtzPFFT& pf_i = shared_fmm ? pfft_ext : pfft_int;
        LK_combined(J, k_ext, pfft_ext, mv_L_ext_J.data(), mv_K_ext_J.data());
        LK_combined(M, k_ext, pfft_ext, mv_L_ext_M.data(), mv_K_ext_M.data());
        LK_combined(J, k_int, pf_i,     mv_L_int_J.data(), mv_K_int_J.data());
        LK_combined(M, k_int, pf_i,     mv_L_int_M.data(), mv_K_int_M.data());
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

    // Assemble PMCHWT blocks
    for (int m = 0; m < N; m++) {
        cdouble K_sum_J = mv_K_ext_J[m] + mv_K_int_J[m];
        cdouble K_sum_M = mv_K_ext_M[m] + mv_K_int_M[m];

        // y[:N] = (eta_ext*L_ext + eta_int*L_int)*J - (K_ext+K_int)*M
        y[m] = (eta_ext * mv_L_ext_J[m] + eta_int * mv_L_int_J[m]) - K_sum_M;

        // y[N:] = (K_ext+K_int)*J + (L_ext/eta_ext + L_int/eta_int)*M
        y[N + m] = K_sum_J + (mv_L_ext_M[m] / eta_ext + mv_L_int_M[m] / eta_int);
    }
}

void BemFmmOperator::cleanup()
{
    if (use_spfft) {
        spfft_ext.cleanup();
        if (!shared_fmm) spfft_int.cleanup();
    } else if (use_pfft) {
        pfft_ext.cleanup();
        if (!shared_fmm) pfft_int.cleanup();
    } else {
        fmm_ext.cleanup();
        if (!shared_fmm) fmm_int.cleanup();
    }
    if (gpu_pack_ready) {
        cudaFree(d_f_p); cudaFree(d_f_m);
        cudaFree(d_jw_p); cudaFree(d_jw_m);
        cudaFree(d_div_p); cudaFree(d_div_m);
        cudaFree(d_x_re); cudaFree(d_x_im);
        cudaFree(d_src_re); cudaFree(d_src_im);
        cudaFree(d_phi_re); cudaFree(d_phi_im);
        cudaFree(d_L_re); cudaFree(d_L_im);
        cudaFree(d_K_re); cudaFree(d_K_im);
        for (int d = 0; d < 3; d++) {
            cudaFree(d_grad_buf_re[d]); cudaFree(d_grad_buf_im[d]);
        }
        gpu_pack_ready = false;
    }
}
