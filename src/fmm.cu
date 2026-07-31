#include "fmm.h"
#include "gpu_select.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <map>
#include <vector>
#include <algorithm>
#include <limits>

// P2P launchers (from p2p.cu)
#include "p2p.h"

__global__ void split_complex_kernel(const double2* __restrict__ in,
                                     double* __restrict__ out_re,
                                     double* __restrict__ out_im,
                                     int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double2 v = in[i];
    out_re[i] = v.x;
    out_im[i] = v.y;
}

__global__ void pack_complex_kernel(const double* __restrict__ in_re,
                                    const double* __restrict__ in_im,
                                    double2* __restrict__ out,
                                    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = make_double2(in_re[i], in_im[i]);
}

// ============================================================
// Spherical Bessel / Hankel functions
// ============================================================

static cdouble sph_jn(int n, cdouble z) {
    if (std::abs(z) < 1e-15) return (n == 0) ? cdouble(1.0) : cdouble(0.0);
    cdouble j0 = std::sin(z) / z;
    if (n == 0) return j0;
    cdouble j1 = std::sin(z) / (z*z) - std::cos(z) / z;
    if (n == 1) return j1;
    cdouble jnm1 = j0, jn = j1;
    for (int l = 1; l < n; l++) {
        cdouble jnp1 = ((2.0*l + 1.0) / z) * jn - jnm1;
        jnm1 = jn;
        jn = jnp1;
    }
    return jn;
}

static cdouble sph_yn(int n, cdouble z) {
    if (std::abs(z) < 1e-15) return cdouble(-1e30);
    cdouble y0 = -std::cos(z) / z;
    if (n == 0) return y0;
    cdouble y1 = -std::cos(z) / (z*z) - std::sin(z) / z;
    if (n == 1) return y1;
    cdouble ynm1 = y0, yn = y1;
    for (int l = 1; l < n; l++) {
        cdouble ynp1 = ((2.0*l + 1.0) / z) * yn - ynm1;
        ynm1 = yn;
        yn = ynp1;
    }
    return yn;
}

cdouble spherical_hankel1(int n, cdouble z) {
    return sph_jn(n, z) + cdouble(0, 1) * sph_yn(n, z);
}

using cldouble = std::complex<long double>;

static cldouble sph_jn_extended(int n, cldouble z) {
    if (std::abs(z) < 1e-18L)
        return n == 0 ? cldouble(1.0L) : cldouble(0.0L);
    cldouble j0 = std::sin(z) / z;
    if (n == 0) return j0;
    cldouble j1 = std::sin(z) / (z * z) - std::cos(z) / z;
    if (n == 1) return j1;
    cldouble previous = j0;
    cldouble current = j1;
    for (int order = 1; order < n; order++) {
        const cldouble next =
            ((2.0L * order + 1.0L) / z) * current - previous;
        previous = current;
        current = next;
    }
    return current;
}

static cldouble sph_yn_extended(int n, cldouble z) {
    if (std::abs(z) < 1e-18L)
        return cldouble(-1e100L);
    cldouble y0 = -std::cos(z) / z;
    if (n == 0) return y0;
    cldouble y1 = -std::cos(z) / (z * z) - std::sin(z) / z;
    if (n == 1) return y1;
    cldouble previous = y0;
    cldouble current = y1;
    for (int order = 1; order < n; order++) {
        const cldouble next =
            ((2.0L * order + 1.0L) / z) * current - previous;
        previous = current;
        current = next;
    }
    return current;
}

static cldouble spherical_hankel1_extended(int n, cldouble z) {
    return sph_jn_extended(n, z) +
        cldouble(0.0L, 1.0L) * sph_yn_extended(n, z);
}

// ============================================================
// CUDA kernels for FMM tree operations
// All use ORIGINAL-ORDER indices for positions and charges
// ============================================================

__global__ void build_leaf_phase_cache_fp32_kernel(
    const double* __restrict__ points,
    const double* __restrict__ dirs,
    double k_re,
    float2* __restrict__ phase_cache,
    float2* __restrict__ l2p_phase_cache,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ point_id_offsets,
    const int* __restrict__ point_ids,
    const double* __restrict__ node_centers,
    int L, int point_count,
    int n_leaves)
{
    const int leaf = blockIdx.x;
    if (leaf >= n_leaves)
        return;
    const int node = leaf_indices[leaf];
    const int start = point_id_offsets[leaf];
    const int count = point_id_offsets[leaf + 1] - start;
    const double cx = node_centers[3 * node];
    const double cy = node_centers[3 * node + 1];
    const double cz = node_centers[3 * node + 2];
    const int entry_count = count * L;
    for (int entry = threadIdx.x;
         entry < entry_count;
         entry += blockDim.x) {
        const int point_in_leaf = entry / L;
        const int direction = entry - point_in_leaf * L;
        const int point = point_ids[start + point_in_leaf];
        const double rx = points[3 * point] - cx;
        const double ry = points[3 * point + 1] - cy;
        const double rz = points[3 * point + 2] - cz;
        const double dot =
            dirs[3 * direction] * rx +
            dirs[3 * direction + 1] * ry +
            dirs[3 * direction + 2] * rz;
        double phase_sine = 0;
        double phase_cosine = 0;
        sincos(k_re * dot, &phase_sine, &phase_cosine);
        const float2 value = make_float2(
            static_cast<float>(phase_cosine),
            static_cast<float>(phase_sine));
        phase_cache[(size_t)point * L + direction] = value;
        if (l2p_phase_cache != nullptr)
            l2p_phase_cache[
                (size_t)direction * point_count +
                start + point_in_leaf] = value;
    }
}

// P2M kernel: one block per leaf, threads split L directions
// src_ids: flat array of original source indices per leaf
__global__ void p2m_kernel(
    const double* __restrict__ src_pts,      // (Ns*3) original order
    const double* __restrict__ q_re,         // (Ns) original order
    const double* __restrict__ q_im,
    const double* __restrict__ dirs,         // (L*3)
    double k_re, double k_im,
    double* __restrict__ multi_re,           // (n_nodes*L)
    double* __restrict__ multi_im,
    const int* __restrict__ leaf_indices,    // (n_leaves) node index
    const int* __restrict__ src_id_offsets,  // (n_leaves+1) into src_ids
    const int* __restrict__ src_ids,         // flat original source IDs
    const double* __restrict__ node_centers, // (n_nodes*3)
    int L, int n_leaves)
{
    int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves) return;

    int node = leaf_indices[leaf_id];
    int s_start = src_id_offsets[leaf_id];
    int s_end = src_id_offsets[leaf_id + 1];
    int s_count = s_end - s_start;
    if (s_count == 0) return;

    double cx = node_centers[node*3];
    double cy = node_centers[node*3+1];
    double cz = node_centers[node*3+2];

    int base = node * L;

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        double dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
        double acc_re = 0.0, acc_im = 0.0;

        for (int s = s_start; s < s_end; s++) {
            int sid = src_ids[s];  // original source index
            double rx = src_pts[sid*3]   - cx;
            double ry = src_pts[sid*3+1] - cy;
            double rz = src_pts[sid*3+2] - cz;
            double dot = dx*rx + dy*ry + dz*rz;

            // exp(-ik * s_hat . r_rel)
            double phase_re = k_im * dot;
            double phase_im = -k_re * dot;
            double e_re = exp(phase_re) * cos(phase_im);
            double e_im = exp(phase_re) * sin(phase_im);

            double qr = q_re[sid], qi = q_im[sid];
            acc_re += e_re * qr - e_im * qi;
            acc_im += e_re * qi + e_im * qr;
        }

        multi_re[base + l] = acc_re;
        multi_im[base + l] = acc_im;
    }
}

__global__ void p2m_kernel_batch2(
    const double* __restrict__ src_pts,
    const double* __restrict__ q1_re,
    const double* __restrict__ q1_im,
    const double* __restrict__ q2_re,
    const double* __restrict__ q2_im,
    const double* __restrict__ dirs,
    double k_re, double k_im,
    double* __restrict__ multi1_re,
    double* __restrict__ multi1_im,
    double* __restrict__ multi2_re,
    double* __restrict__ multi2_im,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ src_id_offsets,
    const int* __restrict__ src_ids,
    const double* __restrict__ node_centers,
    int L, int n_leaves)
{
    int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves) return;

    int node = leaf_indices[leaf_id];
    int s_start = src_id_offsets[leaf_id];
    int s_end = src_id_offsets[leaf_id + 1];
    if (s_end == s_start) return;

    double cx = node_centers[node*3];
    double cy = node_centers[node*3+1];
    double cz = node_centers[node*3+2];
    int base = node * L;

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        double dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
        double a1_re = 0.0, a1_im = 0.0, a2_re = 0.0, a2_im = 0.0;

        for (int s = s_start; s < s_end; s++) {
            int sid = src_ids[s];
            double rx = src_pts[sid*3]   - cx;
            double ry = src_pts[sid*3+1] - cy;
            double rz = src_pts[sid*3+2] - cz;
            double dot = dx*rx + dy*ry + dz*rz;
            double phase_re = k_im * dot;
            double phase_im = -k_re * dot;
            double e_re = exp(phase_re) * cos(phase_im);
            double e_im = exp(phase_re) * sin(phase_im);

            double q1r = q1_re[sid], q1i = q1_im[sid];
            double q2r = q2_re[sid], q2i = q2_im[sid];
            a1_re += e_re * q1r - e_im * q1i;
            a1_im += e_re * q1i + e_im * q1r;
            a2_re += e_re * q2r - e_im * q2i;
            a2_im += e_re * q2i + e_im * q2r;
        }

        multi1_re[base + l] = a1_re;
        multi1_im[base + l] = a1_im;
        multi2_re[base + l] = a2_re;
        multi2_im[base + l] = a2_im;
    }
}

__global__ void p2m_kernel_batch4(
    const double* __restrict__ src_pts,
    const double* __restrict__ q1_re, const double* __restrict__ q1_im,
    const double* __restrict__ q2_re, const double* __restrict__ q2_im,
    const double* __restrict__ q3_re, const double* __restrict__ q3_im,
    const double* __restrict__ q4_re, const double* __restrict__ q4_im,
    const double* __restrict__ dirs,
    double k_re, double k_im,
    double* __restrict__ m1_re, double* __restrict__ m1_im,
    double* __restrict__ m2_re, double* __restrict__ m2_im,
    double* __restrict__ m3_re, double* __restrict__ m3_im,
    double* __restrict__ m4_re, double* __restrict__ m4_im,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ src_id_offsets,
    const int* __restrict__ src_ids,
    const double* __restrict__ node_centers,
    int L, int n_leaves)
{
    int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves) return;

    int node = leaf_indices[leaf_id];
    int s_start = src_id_offsets[leaf_id];
    int s_end = src_id_offsets[leaf_id + 1];
    if (s_end == s_start) return;

    double cx = node_centers[node*3];
    double cy = node_centers[node*3+1];
    double cz = node_centers[node*3+2];
    int base = node * L;

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        double dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
        double a1r = 0.0, a1i = 0.0, a2r = 0.0, a2i = 0.0;
        double a3r = 0.0, a3i = 0.0, a4r = 0.0, a4i = 0.0;

        for (int s = s_start; s < s_end; s++) {
            int sid = src_ids[s];
            double rx = src_pts[sid*3] - cx;
            double ry = src_pts[sid*3+1] - cy;
            double rz = src_pts[sid*3+2] - cz;
            double dot = dx*rx + dy*ry + dz*rz;
            double phase_re = k_im * dot;
            double phase_im = -k_re * dot;
            double e_re = exp(phase_re) * cos(phase_im);
            double e_im = exp(phase_re) * sin(phase_im);

            double qr = q1_re[sid], qi = q1_im[sid];
            a1r += e_re * qr - e_im * qi; a1i += e_re * qi + e_im * qr;
            qr = q2_re[sid]; qi = q2_im[sid];
            a2r += e_re * qr - e_im * qi; a2i += e_re * qi + e_im * qr;
            qr = q3_re[sid]; qi = q3_im[sid];
            a3r += e_re * qr - e_im * qi; a3i += e_re * qi + e_im * qr;
            qr = q4_re[sid]; qi = q4_im[sid];
            a4r += e_re * qr - e_im * qi; a4i += e_re * qi + e_im * qr;
        }

        m1_re[base + l] = a1r; m1_im[base + l] = a1i;
        m2_re[base + l] = a2r; m2_im[base + l] = a2i;
        m3_re[base + l] = a3r; m3_im[base + l] = a3i;
        m4_re[base + l] = a4r; m4_im[base + l] = a4i;
    }
}

__global__ void p2m_kernel_batch3(
    const double* __restrict__ src_pts,
    const double* __restrict__ q1_re, const double* __restrict__ q1_im,
    const double* __restrict__ q2_re, const double* __restrict__ q2_im,
    const double* __restrict__ q3_re, const double* __restrict__ q3_im,
    const double* __restrict__ dirs,
    double k_re, double k_im,
    double* __restrict__ m1_re, double* __restrict__ m1_im,
    double* __restrict__ m2_re, double* __restrict__ m2_im,
    double* __restrict__ m3_re, double* __restrict__ m3_im,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ src_id_offsets,
    const int* __restrict__ src_ids,
    const double* __restrict__ node_centers,
    int L, int n_leaves)
{
    const int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves)
        return;
    const int node = leaf_indices[leaf_id];
    const int start = src_id_offsets[leaf_id];
    const int end = src_id_offsets[leaf_id + 1];
    if (start == end)
        return;
    const double cx = node_centers[3 * node];
    const double cy = node_centers[3 * node + 1];
    const double cz = node_centers[3 * node + 2];
    const int base = node * L;
    const bool real_wavenumber = k_im == 0.0;
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        const double dx = dirs[3 * l];
        const double dy = dirs[3 * l + 1];
        const double dz = dirs[3 * l + 2];
        double a1r = 0.0, a1i = 0.0;
        double a2r = 0.0, a2i = 0.0;
        double a3r = 0.0, a3i = 0.0;
        for (int source = start; source < end; source++) {
            const int sid = src_ids[source];
            const double dot =
                dx * (src_pts[3 * sid] - cx) +
                dy * (src_pts[3 * sid + 1] - cy) +
                dz * (src_pts[3 * sid + 2] - cz);
            const double exponential =
                real_wavenumber ? 1.0 : exp(k_im * dot);
            const double phase = -k_re * dot;
            double phase_sine = 0;
            double phase_cosine = 0;
            sincos(phase, &phase_sine, &phase_cosine);
            const double exp_re = exponential * phase_cosine;
            const double exp_im = exponential * phase_sine;
#define ACCUMULATE_P2M(Q_RE, Q_IM, ACC_RE, ACC_IM) \
            do { \
                const double qr = (Q_RE)[sid]; \
                const double qi = (Q_IM)[sid]; \
                (ACC_RE) += exp_re * qr - exp_im * qi; \
                (ACC_IM) += exp_re * qi + exp_im * qr; \
            } while (0)
            ACCUMULATE_P2M(q1_re, q1_im, a1r, a1i);
            ACCUMULATE_P2M(q2_re, q2_im, a2r, a2i);
            ACCUMULATE_P2M(q3_re, q3_im, a3r, a3i);
#undef ACCUMULATE_P2M
        }
        m1_re[base + l] = a1r;
        m1_im[base + l] = a1i;
        m2_re[base + l] = a2r;
        m2_im[base + l] = a2i;
        m3_re[base + l] = a3r;
        m3_im[base + l] = a3i;
    }
}

// M2M kernel: propagate multipole from child to parent
__global__ void m2m_kernel(
    const int*    __restrict__ parent_idx,
    const int*    __restrict__ child_idx,
    const double* __restrict__ shift_re,
    const double* __restrict__ shift_im,
    double* __restrict__ multi_re,
    double* __restrict__ multi_im,
    int L, int n_pairs, int offset)
{
    int pair = blockIdx.x + offset;
    if (pair >= offset + n_pairs) return;

    int p = parent_idx[pair];
    int c = child_idx[pair];
    int shift_base = pair * L;
    int p_base = p * L;
    int c_base = c * L;

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        double cr = multi_re[c_base + l];
        double ci = multi_im[c_base + l];
        double sr = shift_re[shift_base + l];
        double si = shift_im[shift_base + l];
        atomicAdd(&multi_re[p_base + l], cr * sr - ci * si);
        atomicAdd(&multi_im[p_base + l], cr * si + ci * sr);
    }
}

__global__ void m2m_kernel_batch2(
    const int*    __restrict__ parent_idx,
    const int*    __restrict__ child_idx,
    const double* __restrict__ shift_re,
    const double* __restrict__ shift_im,
    double* __restrict__ multi1_re,
    double* __restrict__ multi1_im,
    double* __restrict__ multi2_re,
    double* __restrict__ multi2_im,
    int L, int n_pairs, int offset)
{
    int pair = blockIdx.x + offset;
    if (pair >= offset + n_pairs) return;

    int p = parent_idx[pair];
    int c = child_idx[pair];
    int shift_base = pair * L;
    int p_base = p * L;
    int c_base = c * L;

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        double sr = shift_re[shift_base + l], si = shift_im[shift_base + l];
        double c1r = multi1_re[c_base + l], c1i = multi1_im[c_base + l];
        double c2r = multi2_re[c_base + l], c2i = multi2_im[c_base + l];
        atomicAdd(&multi1_re[p_base + l], c1r * sr - c1i * si);
        atomicAdd(&multi1_im[p_base + l], c1r * si + c1i * sr);
        atomicAdd(&multi2_re[p_base + l], c2r * sr - c2i * si);
        atomicAdd(&multi2_im[p_base + l], c2r * si + c2i * sr);
    }
}

__global__ void m2m_kernel_batch4(
    const int* __restrict__ parent_idx, const int* __restrict__ child_idx,
    const double* __restrict__ shift_re, const double* __restrict__ shift_im,
    double* __restrict__ m1_re, double* __restrict__ m1_im,
    double* __restrict__ m2_re, double* __restrict__ m2_im,
    double* __restrict__ m3_re, double* __restrict__ m3_im,
    double* __restrict__ m4_re, double* __restrict__ m4_im,
    int L, int n_pairs, int offset)
{
    int pair = blockIdx.x + offset;
    if (pair >= offset + n_pairs) return;
    int p = parent_idx[pair], c = child_idx[pair];
    int shift_base = pair * L, p_base = p * L, c_base = c * L;
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        double sr = shift_re[shift_base + l], si = shift_im[shift_base + l];
        double cr = m1_re[c_base + l], ci = m1_im[c_base + l];
        atomicAdd(&m1_re[p_base + l], cr * sr - ci * si); atomicAdd(&m1_im[p_base + l], cr * si + ci * sr);
        cr = m2_re[c_base + l]; ci = m2_im[c_base + l];
        atomicAdd(&m2_re[p_base + l], cr * sr - ci * si); atomicAdd(&m2_im[p_base + l], cr * si + ci * sr);
        cr = m3_re[c_base + l]; ci = m3_im[c_base + l];
        atomicAdd(&m3_re[p_base + l], cr * sr - ci * si); atomicAdd(&m3_im[p_base + l], cr * si + ci * sr);
        cr = m4_re[c_base + l]; ci = m4_im[c_base + l];
        atomicAdd(&m4_re[p_base + l], cr * sr - ci * si); atomicAdd(&m4_im[p_base + l], cr * si + ci * sr);
    }
}

__global__ void m2m_kernel_batch3(
    const int* __restrict__ parent_idx,
    const int* __restrict__ child_idx,
    const double* __restrict__ shift_re,
    const double* __restrict__ shift_im,
    double* __restrict__ m1_re, double* __restrict__ m1_im,
    double* __restrict__ m2_re, double* __restrict__ m2_im,
    double* __restrict__ m3_re, double* __restrict__ m3_im,
    int L, int n_pairs, int offset)
{
    const int pair = blockIdx.x + offset;
    if (pair >= offset + n_pairs)
        return;
    const int parent_base = parent_idx[pair] * L;
    const int child_base = child_idx[pair] * L;
    const int shift_base = pair * L;
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        const double sr = shift_re[shift_base + l];
        const double si = shift_im[shift_base + l];
#define ACCUMULATE_M2M(REAL, IMAG) \
        do { \
            const double cr = (REAL)[child_base + l]; \
            const double ci = (IMAG)[child_base + l]; \
            atomicAdd(&(REAL)[parent_base + l], cr * sr - ci * si); \
            atomicAdd(&(IMAG)[parent_base + l], cr * si + ci * sr); \
        } while (0)
        ACCUMULATE_M2M(m1_re, m1_im);
        ACCUMULATE_M2M(m2_re, m2_im);
        ACCUMULATE_M2M(m3_re, m3_im);
#undef ACCUMULATE_M2M
    }
}

// M2L kernel: translate multipole to local expansion
__global__ void m2l_kernel(
    const int*    __restrict__ tgt_idx,
    const int*    __restrict__ src_idx,
    const int*    __restrict__ transfer_idx,
    const double* __restrict__ transfer_re,
    const double* __restrict__ transfer_im,
    const double* __restrict__ multi_re,
    const double* __restrict__ multi_im,
    double* __restrict__ local_re,
    double* __restrict__ local_im,
    int L, int n_pairs, int offset)
{
    int pair = blockIdx.x + offset;
    if (pair >= offset + n_pairs) return;

    int tgt = tgt_idx[pair];
    int src = src_idx[pair];
    int tidx = transfer_idx[pair];

    int t_base = tidx * L;
    int s_base = src * L;
    int l_base = tgt * L;

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        double tr = transfer_re[t_base + l];
        double ti = transfer_im[t_base + l];
        double mr = multi_re[s_base + l];
        double mi = multi_im[s_base + l];
        atomicAdd(&local_re[l_base + l], tr * mr - ti * mi);
        atomicAdd(&local_im[l_base + l], tr * mi + ti * mr);
    }
}

__global__ void m2l_kernel_batch2(
    const int*    __restrict__ tgt_idx,
    const int*    __restrict__ src_idx,
    const int*    __restrict__ transfer_idx,
    const double* __restrict__ transfer_re,
    const double* __restrict__ transfer_im,
    const double* __restrict__ multi1_re,
    const double* __restrict__ multi1_im,
    const double* __restrict__ multi2_re,
    const double* __restrict__ multi2_im,
    double* __restrict__ local1_re,
    double* __restrict__ local1_im,
    double* __restrict__ local2_re,
    double* __restrict__ local2_im,
    int L, int n_pairs, int offset)
{
    int pair = blockIdx.x + offset;
    if (pair >= offset + n_pairs) return;

    int tgt = tgt_idx[pair];
    int src = src_idx[pair];
    int tidx = transfer_idx[pair];
    int t_base = tidx * L;
    int s_base = src * L;
    int l_base = tgt * L;

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        double tr = transfer_re[t_base + l], ti = transfer_im[t_base + l];
        double m1r = multi1_re[s_base + l], m1i = multi1_im[s_base + l];
        double m2r = multi2_re[s_base + l], m2i = multi2_im[s_base + l];
        atomicAdd(&local1_re[l_base + l], tr * m1r - ti * m1i);
        atomicAdd(&local1_im[l_base + l], tr * m1i + ti * m1r);
        atomicAdd(&local2_re[l_base + l], tr * m2r - ti * m2i);
        atomicAdd(&local2_im[l_base + l], tr * m2i + ti * m2r);
    }
}

__global__ void m2l_kernel_batch4(
    const int* __restrict__ tgt_idx, const int* __restrict__ src_idx,
    const int* __restrict__ transfer_idx,
    const double* __restrict__ transfer_re, const double* __restrict__ transfer_im,
    const double* __restrict__ m1_re, const double* __restrict__ m1_im,
    const double* __restrict__ m2_re, const double* __restrict__ m2_im,
    const double* __restrict__ m3_re, const double* __restrict__ m3_im,
    const double* __restrict__ m4_re, const double* __restrict__ m4_im,
    double* __restrict__ l1_re, double* __restrict__ l1_im,
    double* __restrict__ l2_re, double* __restrict__ l2_im,
    double* __restrict__ l3_re, double* __restrict__ l3_im,
    double* __restrict__ l4_re, double* __restrict__ l4_im,
    int L, int n_pairs, int offset)
{
    int pair = blockIdx.x + offset;
    if (pair >= offset + n_pairs) return;
    int tgt = tgt_idx[pair], src = src_idx[pair], tidx = transfer_idx[pair];
    int t_base = tidx * L, s_base = src * L, l_base = tgt * L;
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        double tr = transfer_re[t_base + l], ti = transfer_im[t_base + l];
        double mr = m1_re[s_base + l], mi = m1_im[s_base + l];
        atomicAdd(&l1_re[l_base + l], tr * mr - ti * mi); atomicAdd(&l1_im[l_base + l], tr * mi + ti * mr);
        mr = m2_re[s_base + l]; mi = m2_im[s_base + l];
        atomicAdd(&l2_re[l_base + l], tr * mr - ti * mi); atomicAdd(&l2_im[l_base + l], tr * mi + ti * mr);
        mr = m3_re[s_base + l]; mi = m3_im[s_base + l];
        atomicAdd(&l3_re[l_base + l], tr * mr - ti * mi); atomicAdd(&l3_im[l_base + l], tr * mi + ti * mr);
        mr = m4_re[s_base + l]; mi = m4_im[s_base + l];
        atomicAdd(&l4_re[l_base + l], tr * mr - ti * mi); atomicAdd(&l4_im[l_base + l], tr * mi + ti * mr);
    }
}

// One block owns one target expansion. This preserves the M2L summation
// order while replacing an atomic update per interaction with one store.
__global__ void m2l_kernel_batch4_target_rows(
    const int* __restrict__ row_target,
    const int* __restrict__ row_start,
    const int* __restrict__ row_end,
    const int* __restrict__ src_idx,
    const int* __restrict__ transfer_idx,
    const double* __restrict__ transfer_re,
    const double* __restrict__ transfer_im,
    const double* __restrict__ m1_re,
    const double* __restrict__ m1_im,
    const double* __restrict__ m2_re,
    const double* __restrict__ m2_im,
    const double* __restrict__ m3_re,
    const double* __restrict__ m3_im,
    const double* __restrict__ m4_re,
    const double* __restrict__ m4_im,
    double* __restrict__ l1_re,
    double* __restrict__ l1_im,
    double* __restrict__ l2_re,
    double* __restrict__ l2_im,
    double* __restrict__ l3_re,
    double* __restrict__ l3_im,
    double* __restrict__ l4_re,
    double* __restrict__ l4_im,
    int L, int row_count, int row_offset)
{
    const int row = blockIdx.x + row_offset;
    if (row >= row_offset + row_count)
        return;
    const int local_base = row_target[row] * L;
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        double a1r = 0.0, a1i = 0.0;
        double a2r = 0.0, a2i = 0.0;
        double a3r = 0.0, a3i = 0.0;
        double a4r = 0.0, a4i = 0.0;
        for (int pair = row_start[row];
             pair < row_end[row]; pair++) {
            const int transfer_base = transfer_idx[pair] * L + l;
            const int source_base = src_idx[pair] * L + l;
            const double tr = transfer_re[transfer_base];
            const double ti = transfer_im[transfer_base];
#define ACCUMULATE_M2L(MULTI_RE, MULTI_IM, ACC_RE, ACC_IM) \
            do { \
                const double mr = (MULTI_RE)[source_base]; \
                const double mi = (MULTI_IM)[source_base]; \
                (ACC_RE) += tr * mr - ti * mi; \
                (ACC_IM) += tr * mi + ti * mr; \
            } while (0)
            ACCUMULATE_M2L(m1_re, m1_im, a1r, a1i);
            ACCUMULATE_M2L(m2_re, m2_im, a2r, a2i);
            ACCUMULATE_M2L(m3_re, m3_im, a3r, a3i);
            ACCUMULATE_M2L(m4_re, m4_im, a4r, a4i);
#undef ACCUMULATE_M2L
        }
        l1_re[local_base + l] += a1r;
        l1_im[local_base + l] += a1i;
        l2_re[local_base + l] += a2r;
        l2_im[local_base + l] += a2i;
        l3_re[local_base + l] += a3r;
        l3_im[local_base + l] += a3i;
        l4_re[local_base + l] += a4r;
        l4_im[local_base + l] += a4i;
    }
}

__global__ void m2l_kernel_batch3_target_rows(
    const int* __restrict__ row_target,
    const int* __restrict__ row_start,
    const int* __restrict__ row_end,
    const int* __restrict__ src_idx,
    const int* __restrict__ transfer_idx,
    const double* __restrict__ transfer_re,
    const double* __restrict__ transfer_im,
    const double* __restrict__ m1_re,
    const double* __restrict__ m1_im,
    const double* __restrict__ m2_re,
    const double* __restrict__ m2_im,
    const double* __restrict__ m3_re,
    const double* __restrict__ m3_im,
    double* __restrict__ l1_re,
    double* __restrict__ l1_im,
    double* __restrict__ l2_re,
    double* __restrict__ l2_im,
    double* __restrict__ l3_re,
    double* __restrict__ l3_im,
    int L, int row_count, int row_offset)
{
    const int row = blockIdx.x + row_offset;
    if (row >= row_offset + row_count)
        return;
    const int local_base = row_target[row] * L;
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        double a1r = 0.0, a1i = 0.0;
        double a2r = 0.0, a2i = 0.0;
        double a3r = 0.0, a3i = 0.0;
        for (int pair = row_start[row];
             pair < row_end[row]; pair++) {
            const int transfer_base =
                transfer_idx[pair] * L + l;
            const int source_base = src_idx[pair] * L + l;
            const double tr = transfer_re[transfer_base];
            const double ti = transfer_im[transfer_base];
#define ACCUMULATE_M2L3(MULTI_RE, MULTI_IM, ACC_RE, ACC_IM) \
            do { \
                const double mr = (MULTI_RE)[source_base]; \
                const double mi = (MULTI_IM)[source_base]; \
                (ACC_RE) += tr * mr - ti * mi; \
                (ACC_IM) += tr * mi + ti * mr; \
            } while (0)
            ACCUMULATE_M2L3(m1_re, m1_im, a1r, a1i);
            ACCUMULATE_M2L3(m2_re, m2_im, a2r, a2i);
            ACCUMULATE_M2L3(m3_re, m3_im, a3r, a3i);
#undef ACCUMULATE_M2L3
        }
        l1_re[local_base + l] += a1r;
        l1_im[local_base + l] += a1i;
        l2_re[local_base + l] += a2r;
        l2_im[local_base + l] += a2i;
        l3_re[local_base + l] += a3r;
        l3_im[local_base + l] += a3i;
    }
}

// L2L kernel
__global__ void l2l_kernel(
    const int*    __restrict__ parent_idx,
    const int*    __restrict__ child_idx,
    const double* __restrict__ shift_re,
    const double* __restrict__ shift_im,
    const double* local_re_in,
    const double* local_im_in,
    double* local_re,
    double* local_im,
    int L, int n_pairs, int offset)
{
    int pair = blockIdx.x + offset;
    if (pair >= offset + n_pairs) return;

    int p = parent_idx[pair];
    int c = child_idx[pair];
    int shift_base = pair * L;
    int p_base = p * L;
    int c_base = c * L;

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        double pr = local_re_in[p_base + l];
        double pi = local_im_in[p_base + l];
        double sr = shift_re[shift_base + l];
        double si = shift_im[shift_base + l];
        atomicAdd(&local_re[c_base + l], pr * sr - pi * si);
        atomicAdd(&local_im[c_base + l], pr * si + pi * sr);
    }
}

__global__ void l2l_kernel_batch2(
    const int*    __restrict__ parent_idx,
    const int*    __restrict__ child_idx,
    const double* __restrict__ shift_re,
    const double* __restrict__ shift_im,
    const double* local1_re_in,
    const double* local1_im_in,
    double* local1_re,
    double* local1_im,
    const double* local2_re_in,
    const double* local2_im_in,
    double* local2_re,
    double* local2_im,
    int L, int n_pairs, int offset)
{
    int pair = blockIdx.x + offset;
    if (pair >= offset + n_pairs) return;

    int p = parent_idx[pair];
    int c = child_idx[pair];
    int shift_base = pair * L;
    int p_base = p * L;
    int c_base = c * L;

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        double sr = shift_re[shift_base + l], si = shift_im[shift_base + l];
        double p1r = local1_re_in[p_base + l], p1i = local1_im_in[p_base + l];
        double p2r = local2_re_in[p_base + l], p2i = local2_im_in[p_base + l];
        atomicAdd(&local1_re[c_base + l], p1r * sr - p1i * si);
        atomicAdd(&local1_im[c_base + l], p1r * si + p1i * sr);
        atomicAdd(&local2_re[c_base + l], p2r * sr - p2i * si);
        atomicAdd(&local2_im[c_base + l], p2r * si + p2i * sr);
    }
}

__global__ void l2l_kernel_batch4(
    const int* __restrict__ parent_idx, const int* __restrict__ child_idx,
    const double* __restrict__ shift_re, const double* __restrict__ shift_im,
    const double* l1_re_in, const double* l1_im_in,
    double* l1_re, double* l1_im,
    const double* l2_re_in, const double* l2_im_in,
    double* l2_re, double* l2_im,
    const double* l3_re_in, const double* l3_im_in,
    double* l3_re, double* l3_im,
    const double* l4_re_in, const double* l4_im_in,
    double* l4_re, double* l4_im,
    int L, int n_pairs, int offset)
{
    int pair = blockIdx.x + offset;
    if (pair >= offset + n_pairs) return;
    int p = parent_idx[pair], c = child_idx[pair];
    int shift_base = pair * L, p_base = p * L, c_base = c * L;
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        double sr = shift_re[shift_base + l], si = shift_im[shift_base + l];
        double pr = l1_re_in[p_base + l], pi = l1_im_in[p_base + l];
        atomicAdd(&l1_re[c_base + l], pr * sr - pi * si); atomicAdd(&l1_im[c_base + l], pr * si + pi * sr);
        pr = l2_re_in[p_base + l]; pi = l2_im_in[p_base + l];
        atomicAdd(&l2_re[c_base + l], pr * sr - pi * si); atomicAdd(&l2_im[c_base + l], pr * si + pi * sr);
        pr = l3_re_in[p_base + l]; pi = l3_im_in[p_base + l];
        atomicAdd(&l3_re[c_base + l], pr * sr - pi * si); atomicAdd(&l3_im[c_base + l], pr * si + pi * sr);
        pr = l4_re_in[p_base + l]; pi = l4_im_in[p_base + l];
        atomicAdd(&l4_re[c_base + l], pr * sr - pi * si); atomicAdd(&l4_im[c_base + l], pr * si + pi * sr);
    }
}

__global__ void l2l_kernel_batch3(
    const int* __restrict__ parent_idx,
    const int* __restrict__ child_idx,
    const double* __restrict__ shift_re,
    const double* __restrict__ shift_im,
    const double* l1_re_in,
    const double* l1_im_in,
    double* l1_re,
    double* l1_im,
    const double* l2_re_in,
    const double* l2_im_in,
    double* l2_re,
    double* l2_im,
    const double* l3_re_in,
    const double* l3_im_in,
    double* l3_re,
    double* l3_im,
    int L, int n_pairs, int offset)
{
    const int pair = blockIdx.x + offset;
    if (pair >= offset + n_pairs)
        return;
    const int parent_base = parent_idx[pair] * L;
    const int child_base = child_idx[pair] * L;
    const int shift_base = pair * L;
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        const double sr = shift_re[shift_base + l];
        const double si = shift_im[shift_base + l];
#define ACCUMULATE_L2L3(IN_RE, IN_IM, OUT_RE, OUT_IM) \
        do { \
            const double pr = (IN_RE)[parent_base + l]; \
            const double pi = (IN_IM)[parent_base + l]; \
            (OUT_RE)[child_base + l] += pr * sr - pi * si; \
            (OUT_IM)[child_base + l] += pr * si + pi * sr; \
        } while (0)
        ACCUMULATE_L2L3(l1_re_in, l1_im_in, l1_re, l1_im);
        ACCUMULATE_L2L3(l2_re_in, l2_im_in, l2_re, l2_im);
        ACCUMULATE_L2L3(l3_re_in, l3_im_in, l3_re, l3_im);
#undef ACCUMULATE_L2L3
    }
}

template <
    bool CacheSources,
    bool CachePhase,
    typename MultiReal,
    typename ChargeReal>
__global__ void p2m_kernel_batch6_flat(
    const double* __restrict__ src_pts,
    const ChargeReal* __restrict__ q1_re,
    const ChargeReal* __restrict__ q1_im,
    const ChargeReal* __restrict__ q2_re,
    const ChargeReal* __restrict__ q2_im,
    const ChargeReal* __restrict__ q3_re,
    const ChargeReal* __restrict__ q3_im,
    const ChargeReal* __restrict__ q4_re,
    const ChargeReal* __restrict__ q4_im,
    const ChargeReal* __restrict__ q5_re,
    const ChargeReal* __restrict__ q5_im,
    const ChargeReal* __restrict__ q6_re,
    const ChargeReal* __restrict__ q6_im,
    const float2* __restrict__ phase_cache,
    const double* __restrict__ dirs,
    double k_re, double k_im,
    MultiReal* __restrict__ multi_re,
    MultiReal* __restrict__ multi_im,
    int field_stride,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ src_id_offsets,
    const int* __restrict__ src_ids,
    const double* __restrict__ node_centers,
    int source_cache_stride,
    int L, int n_leaves)
{
    const int leaf = blockIdx.x;
    if (leaf >= n_leaves)
        return;
    const int node = leaf_indices[leaf];
    const int start = src_id_offsets[leaf];
    const int end = src_id_offsets[leaf + 1];
    if (start == end)
        return;
    const double cx = node_centers[3 * node];
    const double cy = node_centers[3 * node + 1];
    const double cz = node_centers[3 * node + 2];
    const ChargeReal* charges_re[6] = {
        q1_re, q2_re, q3_re, q4_re, q5_re, q6_re
    };
    const ChargeReal* charges_im[6] = {
        q1_im, q2_im, q3_im, q4_im, q5_im, q6_im
    };
    const bool real_wavenumber = k_im == 0.0;

    extern __shared__ double source_cache[];
    if (CacheSources) {
        const int source_count = end - start;
        for (int local_source = threadIdx.x;
             local_source < source_count;
             local_source += blockDim.x) {
            const int sid = src_ids[start + local_source];
            source_cache[local_source] =
                src_pts[3 * sid] - cx;
            source_cache[source_cache_stride + local_source] =
                src_pts[3 * sid + 1] - cy;
            source_cache[2 * source_cache_stride + local_source] =
                src_pts[3 * sid + 2] - cz;
#pragma unroll
            for (int field = 0; field < 6; field++) {
                source_cache[
                    (3 + 2 * field) * source_cache_stride +
                    local_source] = charges_re[field][sid];
                source_cache[
                    (4 + 2 * field) * source_cache_stride +
                    local_source] = charges_im[field][sid];
            }
        }
        __syncthreads();
    }

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        const double dx = dirs[3 * l];
        const double dy = dirs[3 * l + 1];
        const double dz = dirs[3 * l + 2];
        MultiReal accum_re[6] = {0, 0, 0, 0, 0, 0};
        MultiReal accum_im[6] = {0, 0, 0, 0, 0, 0};
        for (int source = start; source < end; source++) {
            const int local_source = source - start;
            const int sid = src_ids[source];
            MultiReal phase_re = 0;
            MultiReal phase_im = 0;
            if (CachePhase) {
                const float2 cached =
                    phase_cache[(size_t)sid * L + l];
                phase_re = static_cast<MultiReal>(cached.x);
                phase_im = -static_cast<MultiReal>(cached.y);
            } else {
                const double rx =
                    CacheSources
                        ? source_cache[local_source]
                        : src_pts[3 * sid] - cx;
                const double ry =
                    CacheSources
                        ? source_cache[
                            source_cache_stride + local_source]
                        : src_pts[3 * sid + 1] - cy;
                const double rz =
                    CacheSources
                        ? source_cache[
                            2 * source_cache_stride + local_source]
                        : src_pts[3 * sid + 2] - cz;
                const double dot = dx * rx + dy * ry + dz * rz;
                const double attenuation =
                    real_wavenumber ? 1.0 : exp(k_im * dot);
                const double phase = -k_re * dot;
                double phase_sine = 0;
                double phase_cosine = 0;
                sincos(phase, &phase_sine, &phase_cosine);
                phase_re = static_cast<MultiReal>(
                    attenuation * phase_cosine);
                phase_im = static_cast<MultiReal>(
                    attenuation * phase_sine);
            }
#pragma unroll
            for (int field = 0; field < 6; field++) {
                const MultiReal qr = static_cast<MultiReal>(
                    CacheSources
                        ? source_cache[
                            (3 + 2 * field) *
                                source_cache_stride +
                            local_source]
                        : charges_re[field][sid]);
                const MultiReal qi = static_cast<MultiReal>(
                    CacheSources
                        ? source_cache[
                            (4 + 2 * field) *
                                source_cache_stride +
                            local_source]
                        : charges_im[field][sid]);
                accum_re[field] += phase_re * qr - phase_im * qi;
                accum_im[field] += phase_re * qi + phase_im * qr;
            }
        }
        const int index = node * L + l;
#pragma unroll
        for (int field = 0; field < 6; field++) {
            multi_re[field * field_stride + index] =
                static_cast<MultiReal>(accum_re[field]);
            multi_im[field * field_stride + index] =
                static_cast<MultiReal>(accum_im[field]);
        }
    }
}

template <typename MultiReal>
__global__ void p2m_kernel_batch12_packed_fp32(
    const double* __restrict__ src_pts,
    const float* __restrict__ packed_charges,
    int source_count,
    const float2* __restrict__ phase_cache,
    const double* __restrict__ dirs,
    double k_re, double k_im,
    MultiReal* __restrict__ multi_re,
    MultiReal* __restrict__ multi_im,
    int field_stride,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ src_id_offsets,
    const int* __restrict__ src_ids,
    const double* __restrict__ node_centers,
    int L, int n_leaves)
{
    const int leaf = blockIdx.x;
    if (leaf >= n_leaves)
        return;
    const int node = leaf_indices[leaf];
    const int start = src_id_offsets[leaf];
    const int end = src_id_offsets[leaf + 1];
    if (start == end)
        return;
    const double cx = node_centers[3 * node];
    const double cy = node_centers[3 * node + 1];
    const double cz = node_centers[3 * node + 2];
    const bool real_wavenumber = k_im == 0.0;

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        MultiReal accum_re[12] = {};
        MultiReal accum_im[12] = {};
        const double dx = dirs[3 * l];
        const double dy = dirs[3 * l + 1];
        const double dz = dirs[3 * l + 2];
        for (int source = start; source < end; source++) {
            const int sid = src_ids[source];
            MultiReal phase_re;
            MultiReal phase_im;
            if (phase_cache != nullptr) {
                const float2 cached =
                    phase_cache[(size_t)sid * L + l];
                phase_re = static_cast<MultiReal>(cached.x);
                phase_im = static_cast<MultiReal>(-cached.y);
            } else {
                const double dot =
                    dx * (src_pts[3 * sid] - cx) +
                    dy * (src_pts[3 * sid + 1] - cy) +
                    dz * (src_pts[3 * sid + 2] - cz);
                const double attenuation =
                    real_wavenumber ? 1.0 : exp(k_im * dot);
                double sine;
                double cosine;
                sincos(-k_re * dot, &sine, &cosine);
                phase_re = static_cast<MultiReal>(attenuation * cosine);
                phase_im = static_cast<MultiReal>(attenuation * sine);
            }
#pragma unroll
            for (int field = 0; field < 12; field++) {
                const MultiReal qr = static_cast<MultiReal>(
                    packed_charges[(2 * field) * source_count + sid]);
                const MultiReal qi = static_cast<MultiReal>(
                    packed_charges[(2 * field + 1) * source_count + sid]);
                accum_re[field] += phase_re * qr - phase_im * qi;
                accum_im[field] += phase_re * qi + phase_im * qr;
            }
        }
        const int index = node * L + l;
#pragma unroll
        for (int field = 0; field < 12; field++) {
            multi_re[field * field_stride + index] = accum_re[field];
            multi_im[field * field_stride + index] = accum_im[field];
        }
    }
}

template <int FieldCount, typename MultiReal>
__global__ void m2m_kernel_batch6_flat(
    const int* __restrict__ parent_idx,
    const int* __restrict__ child_idx,
    const double* __restrict__ shift_re,
    const double* __restrict__ shift_im,
    MultiReal* __restrict__ multi_re,
    MultiReal* __restrict__ multi_im,
    int field_stride,
    int L, int pair_count, int pair_offset)
{
    const int pair = blockIdx.x + pair_offset;
    if (pair >= pair_offset + pair_count)
        return;
    const int parent_base = parent_idx[pair] * L;
    const int child_base = child_idx[pair] * L;
    const int shift_base = pair * L;
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        const MultiReal sr =
            static_cast<MultiReal>(shift_re[shift_base + l]);
        const MultiReal si =
            static_cast<MultiReal>(shift_im[shift_base + l]);
#pragma unroll
        for (int field = 0; field < FieldCount; field++) {
            const int parent = field * field_stride + parent_base + l;
            const int child = field * field_stride + child_base + l;
            const MultiReal cr = multi_re[child];
            const MultiReal ci = multi_im[child];
            atomicAdd(&multi_re[parent], cr * sr - ci * si);
            atomicAdd(&multi_im[parent], cr * si + ci * sr);
        }
    }
}

template <
    int FieldCount,
    typename ComputeReal,
    typename TransferReal,
    typename MultiReal,
    typename LocalReal>
__global__ void m2l_kernel_batch6_flat_target_rows(
    const int* __restrict__ row_target,
    const int* __restrict__ row_start,
    const int* __restrict__ row_end,
    const int* __restrict__ source_idx,
    const int* __restrict__ transfer_idx,
    const TransferReal* __restrict__ transfer_re,
    const TransferReal* __restrict__ transfer_im,
    const MultiReal* __restrict__ multi_re,
    const MultiReal* __restrict__ multi_im,
    LocalReal* __restrict__ local_re,
    LocalReal* __restrict__ local_im,
    int field_offset,
    int field_stride,
    int L, int row_count, int row_offset)
{
    const int row = blockIdx.x + row_offset;
    if (row >= row_offset + row_count)
        return;
    const int local_base = row_target[row] * L;
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        ComputeReal accum_re[FieldCount] = {};
        ComputeReal accum_im[FieldCount] = {};
        for (int pair = row_start[row]; pair < row_end[row]; pair++) {
            const int transfer =
                transfer_idx[pair] * L + l;
            const int source = source_idx[pair] * L + l;
            const ComputeReal tr =
                static_cast<ComputeReal>(transfer_re[transfer]);
            const ComputeReal ti =
                static_cast<ComputeReal>(transfer_im[transfer]);
#pragma unroll
            for (int field = 0; field < FieldCount; field++) {
                const int index =
                    (field_offset + field) * field_stride + source;
                const ComputeReal mr =
                    static_cast<ComputeReal>(multi_re[index]);
                const ComputeReal mi =
                    static_cast<ComputeReal>(multi_im[index]);
                accum_re[field] += tr * mr - ti * mi;
                accum_im[field] += tr * mi + ti * mr;
            }
        }
#pragma unroll
        for (int field = 0; field < FieldCount; field++) {
            const int index =
                (field_offset + field) * field_stride + local_base + l;
            local_re[index] +=
                static_cast<LocalReal>(accum_re[field]);
            local_im[index] +=
                static_cast<LocalReal>(accum_im[field]);
        }
    }
}

template <int FieldCount, typename LocalReal>
__global__ void l2l_kernel_batch6_flat(
    const int* __restrict__ parent_idx,
    const int* __restrict__ child_idx,
    const double* __restrict__ shift_re,
    const double* __restrict__ shift_im,
    LocalReal* __restrict__ local_re,
    LocalReal* __restrict__ local_im,
    int field_stride,
    int L, int pair_count, int pair_offset)
{
    const int pair = blockIdx.x + pair_offset;
    if (pair >= pair_offset + pair_count)
        return;
    const int parent_base = parent_idx[pair] * L;
    const int child_base = child_idx[pair] * L;
    const int shift_base = pair * L;
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        const LocalReal sr =
            static_cast<LocalReal>(shift_re[shift_base + l]);
        const LocalReal si =
            static_cast<LocalReal>(shift_im[shift_base + l]);
#pragma unroll
        for (int field = 0; field < FieldCount; field++) {
            const int parent = field * field_stride + parent_base + l;
            const int child = field * field_stride + child_base + l;
            const LocalReal pr = local_re[parent];
            const LocalReal pi = local_im[parent];
            local_re[child] += pr * sr - pi * si;
            local_im[child] += pr * si + pi * sr;
        }
    }
}

// L2P kernel: evaluate local expansion at target points
// tgt_ids: flat array of original target indices per leaf
__global__ void l2p_kernel(
    const double* __restrict__ tgt_pts,      // (Nt*3) original order
    const double* __restrict__ dirs,
    const double* __restrict__ weights,
    const double* __restrict__ local_re,
    const double* __restrict__ local_im,
    double k_re, double k_im,
    double prefac_re, double prefac_im,
    double* __restrict__ out_re,             // (Nt) indexed by original tgt ID
    double* __restrict__ out_im,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ tgt_id_offsets,  // (n_leaves+1)
    const int* __restrict__ tgt_ids,         // flat original target IDs
    const double* __restrict__ node_centers,
    int L, int n_leaves)
{
    int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves) return;

    int node = leaf_indices[leaf_id];
    int t_start = tgt_id_offsets[leaf_id];
    int t_end = tgt_id_offsets[leaf_id + 1];
    int t_count = t_end - t_start;
    if (t_count == 0) return;

    double ccx = node_centers[node*3];
    double ccy = node_centers[node*3+1];
    double ccz = node_centers[node*3+2];
    int l_base = node * L;

    for (int t = threadIdx.x; t < t_count; t += blockDim.x) {
        int tid = tgt_ids[t_start + t];  // original target index
        double rx = tgt_pts[tid*3]   - ccx;
        double ry = tgt_pts[tid*3+1] - ccy;
        double rz = tgt_pts[tid*3+2] - ccz;

        double acc_re = 0.0, acc_im = 0.0;
        for (int l = 0; l < L; l++) {
            double dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
            double dot = dx*rx + dy*ry + dz*rz;
            double phase_re = -k_im * dot;
            double phase_im = k_re * dot;
            double e_re = exp(phase_re) * cos(phase_im);
            double e_im = exp(phase_re) * sin(phase_im);

            double lr = local_re[l_base + l];
            double li = local_im[l_base + l];
            double wl = weights[l];
            double wlr = wl * lr, wli = wl * li;
            acc_re += wlr * e_re - wli * e_im;
            acc_im += wlr * e_im + wli * e_re;
        }

        double final_re = prefac_re * acc_re - prefac_im * acc_im;
        double final_im = prefac_re * acc_im + prefac_im * acc_re;
        atomicAdd(&out_re[tid], final_re);
        atomicAdd(&out_im[tid], final_im);
    }
}

__global__ void l2p_kernel_batch2(
    const double* __restrict__ tgt_pts,
    const double* __restrict__ dirs,
    const double* __restrict__ weights,
    const double* __restrict__ local1_re,
    const double* __restrict__ local1_im,
    const double* __restrict__ local2_re,
    const double* __restrict__ local2_im,
    double k_re, double k_im,
    double prefac_re, double prefac_im,
    double* __restrict__ out1_re,
    double* __restrict__ out1_im,
    double* __restrict__ out2_re,
    double* __restrict__ out2_im,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ tgt_id_offsets,
    const int* __restrict__ tgt_ids,
    const double* __restrict__ node_centers,
    int L, int n_leaves)
{
    int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves) return;

    int node = leaf_indices[leaf_id];
    int t_start = tgt_id_offsets[leaf_id];
    int t_end = tgt_id_offsets[leaf_id + 1];
    int t_count = t_end - t_start;
    if (t_count == 0) return;

    double ccx = node_centers[node*3];
    double ccy = node_centers[node*3+1];
    double ccz = node_centers[node*3+2];
    int l_base = node * L;

    for (int t = threadIdx.x; t < t_count; t += blockDim.x) {
        int tid = tgt_ids[t_start + t];
        double rx = tgt_pts[tid*3]   - ccx;
        double ry = tgt_pts[tid*3+1] - ccy;
        double rz = tgt_pts[tid*3+2] - ccz;
        double a1_re = 0.0, a1_im = 0.0, a2_re = 0.0, a2_im = 0.0;

        for (int l = 0; l < L; l++) {
            double dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
            double dot = dx*rx + dy*ry + dz*rz;
            double phase_re = -k_im * dot;
            double phase_im = k_re * dot;
            double e_re = exp(phase_re) * cos(phase_im);
            double e_im = exp(phase_re) * sin(phase_im);
            double wl = weights[l];

            double l1r = wl * local1_re[l_base + l], l1i = wl * local1_im[l_base + l];
            double l2r = wl * local2_re[l_base + l], l2i = wl * local2_im[l_base + l];
            a1_re += l1r * e_re - l1i * e_im;
            a1_im += l1r * e_im + l1i * e_re;
            a2_re += l2r * e_re - l2i * e_im;
            a2_im += l2r * e_im + l2i * e_re;
        }

        atomicAdd(&out1_re[tid], prefac_re * a1_re - prefac_im * a1_im);
        atomicAdd(&out1_im[tid], prefac_re * a1_im + prefac_im * a1_re);
        atomicAdd(&out2_re[tid], prefac_re * a2_re - prefac_im * a2_im);
        atomicAdd(&out2_im[tid], prefac_re * a2_im + prefac_im * a2_re);
    }
}

__global__ void l2p_kernel_batch4(
    const double* __restrict__ tgt_pts,
    const double* __restrict__ dirs,
    const double* __restrict__ weights,
    const double* __restrict__ l1_re, const double* __restrict__ l1_im,
    const double* __restrict__ l2_re, const double* __restrict__ l2_im,
    const double* __restrict__ l3_re, const double* __restrict__ l3_im,
    const double* __restrict__ l4_re, const double* __restrict__ l4_im,
    double k_re, double k_im, double prefac_re, double prefac_im,
    double* __restrict__ out1_re, double* __restrict__ out1_im,
    double* __restrict__ out2_re, double* __restrict__ out2_im,
    double* __restrict__ out3_re, double* __restrict__ out3_im,
    double* __restrict__ out4_re, double* __restrict__ out4_im,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ tgt_id_offsets,
    const int* __restrict__ tgt_ids,
    const double* __restrict__ node_centers,
    int L, int n_leaves)
{
    int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves) return;
    int node = leaf_indices[leaf_id];
    int t_start = tgt_id_offsets[leaf_id], t_end = tgt_id_offsets[leaf_id + 1];
    if (t_end == t_start) return;
    double ccx = node_centers[node*3], ccy = node_centers[node*3+1], ccz = node_centers[node*3+2];
    int l_base = node * L;

    for (int t = threadIdx.x; t < t_end - t_start; t += blockDim.x) {
        int tid = tgt_ids[t_start + t];
        double rx = tgt_pts[tid*3] - ccx, ry = tgt_pts[tid*3+1] - ccy, rz = tgt_pts[tid*3+2] - ccz;
        double a1r = 0.0, a1i = 0.0, a2r = 0.0, a2i = 0.0, a3r = 0.0, a3i = 0.0, a4r = 0.0, a4i = 0.0;
        for (int l = 0; l < L; l++) {
            double dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
            double dot = dx*rx + dy*ry + dz*rz;
            double e_re = exp(-k_im * dot) * cos(k_re * dot);
            double e_im = exp(-k_im * dot) * sin(k_re * dot);
            double wl = weights[l];
            double lr = wl * l1_re[l_base + l], li = wl * l1_im[l_base + l];
            a1r += lr * e_re - li * e_im; a1i += lr * e_im + li * e_re;
            lr = wl * l2_re[l_base + l]; li = wl * l2_im[l_base + l];
            a2r += lr * e_re - li * e_im; a2i += lr * e_im + li * e_re;
            lr = wl * l3_re[l_base + l]; li = wl * l3_im[l_base + l];
            a3r += lr * e_re - li * e_im; a3i += lr * e_im + li * e_re;
            lr = wl * l4_re[l_base + l]; li = wl * l4_im[l_base + l];
            a4r += lr * e_re - li * e_im; a4i += lr * e_im + li * e_re;
        }
        atomicAdd(&out1_re[tid], prefac_re * a1r - prefac_im * a1i);
        atomicAdd(&out1_im[tid], prefac_re * a1i + prefac_im * a1r);
        atomicAdd(&out2_re[tid], prefac_re * a2r - prefac_im * a2i);
        atomicAdd(&out2_im[tid], prefac_re * a2i + prefac_im * a2r);
        atomicAdd(&out3_re[tid], prefac_re * a3r - prefac_im * a3i);
        atomicAdd(&out3_im[tid], prefac_re * a3i + prefac_im * a3r);
        atomicAdd(&out4_re[tid], prefac_re * a4r - prefac_im * a4i);
        atomicAdd(&out4_im[tid], prefac_re * a4i + prefac_im * a4r);
    }
}

// Repack gradient from 6 separate component arrays into interleaved xyz format
// Output: out_re[i*3+0]=gx_re[i], out_re[i*3+1]=gy_re[i], out_re[i*3+2]=gz_re[i], same for im
__global__ void repack_gradient_kernel(
    const double* __restrict__ gx_re, const double* __restrict__ gx_im,
    const double* __restrict__ gy_re, const double* __restrict__ gy_im,
    const double* __restrict__ gz_re, const double* __restrict__ gz_im,
    double* __restrict__ out_re, double* __restrict__ out_im,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    out_re[i*3]   = gx_re[i];
    out_re[i*3+1] = gy_re[i];
    out_re[i*3+2] = gz_re[i];

    out_im[i*3]   = gx_im[i];
    out_im[i*3+1] = gy_im[i];
    out_im[i*3+2] = gz_im[i];
}

// L2P gradient kernel
__global__ void l2p_gradient_kernel(
    const double* __restrict__ tgt_pts,
    const double* __restrict__ dirs,
    const double* __restrict__ weights,
    const double* __restrict__ local_re,
    const double* __restrict__ local_im,
    double k_re, double k_im,
    double prefac_re, double prefac_im,
    double ik2_re, double ik2_im,
    double* __restrict__ gx_re, double* __restrict__ gx_im,
    double* __restrict__ gy_re, double* __restrict__ gy_im,
    double* __restrict__ gz_re, double* __restrict__ gz_im,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ tgt_id_offsets,
    const int* __restrict__ tgt_ids,
    const double* __restrict__ node_centers,
    int L, int n_leaves)
{
    int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves) return;

    int node = leaf_indices[leaf_id];
    int t_start = tgt_id_offsets[leaf_id];
    int t_end = tgt_id_offsets[leaf_id + 1];
    int t_count = t_end - t_start;
    if (t_count == 0) return;

    double ccx = node_centers[node*3];
    double ccy = node_centers[node*3+1];
    double ccz = node_centers[node*3+2];
    int l_base = node * L;

    for (int t = threadIdx.x; t < t_count; t += blockDim.x) {
        int tid = tgt_ids[t_start + t];
        double rx = tgt_pts[tid*3]   - ccx;
        double ry = tgt_pts[tid*3+1] - ccy;
        double rz = tgt_pts[tid*3+2] - ccz;

        double ax_re = 0, ax_im = 0, ay_re = 0, ay_im = 0, az_re = 0, az_im = 0;

        for (int l = 0; l < L; l++) {
            double dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
            double dot = dx*rx + dy*ry + dz*rz;
            double phase_re = -k_im * dot;
            double phase_im = k_re * dot;
            double e_re = exp(phase_re) * cos(phase_im);
            double e_im = exp(phase_re) * sin(phase_im);

            double lr = local_re[l_base + l], li = local_im[l_base + l];
            double wl = weights[l];
            double wlr = wl * lr, wli = wl * li;
            double pr = wlr * e_re - wli * e_im;
            double pi = wlr * e_im + wli * e_re;

            double ikpr = ik2_re * pr - ik2_im * pi;
            double ikpi = ik2_re * pi + ik2_im * pr;

            ax_re += ikpr * dx; ax_im += ikpi * dx;
            ay_re += ikpr * dy; ay_im += ikpi * dy;
            az_re += ikpr * dz; az_im += ikpi * dz;
        }

        atomicAdd(&gx_re[tid], prefac_re*ax_re - prefac_im*ax_im);
        atomicAdd(&gx_im[tid], prefac_re*ax_im + prefac_im*ax_re);
        atomicAdd(&gy_re[tid], prefac_re*ay_re - prefac_im*ay_im);
        atomicAdd(&gy_im[tid], prefac_re*ay_im + prefac_im*ay_re);
        atomicAdd(&gz_re[tid], prefac_re*az_re - prefac_im*az_im);
        atomicAdd(&gz_im[tid], prefac_re*az_im + prefac_im*az_re);
    }
}

// L2P symmetric Hessian in xx,xy,xz,yy,yz,zz order.
__global__ void l2p_hessian_kernel(
    const double* __restrict__ tgt_pts,
    const double* __restrict__ dirs,
    const double* __restrict__ weights,
    const double* __restrict__ local_re,
    const double* __restrict__ local_im,
    double k_re, double k_im,
    double prefac_re, double prefac_im,
    double neg_k2_re, double neg_k2_im,
    double* __restrict__ hess_re,
    double* __restrict__ hess_im,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ tgt_id_offsets,
    const int* __restrict__ tgt_ids,
    const double* __restrict__ node_centers,
    int L, int n_leaves)
{
    const int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves) return;
    const int node = leaf_indices[leaf_id];
    const int start = tgt_id_offsets[leaf_id];
    const int end = tgt_id_offsets[leaf_id + 1];
    const double cx = node_centers[3 * node];
    const double cy = node_centers[3 * node + 1];
    const double cz = node_centers[3 * node + 2];
    const int local_base = node * L;
    const int row[6] = {0, 0, 0, 1, 1, 2};
    const int col[6] = {0, 1, 2, 1, 2, 2};

    for (int target = start + threadIdx.x;
         target < end; target += blockDim.x) {
        const int tid = tgt_ids[target];
        const double position[3] = {
            tgt_pts[3 * tid] - cx,
            tgt_pts[3 * tid + 1] - cy,
            tgt_pts[3 * tid + 2] - cz
        };
        double acc_re[6] = {0, 0, 0, 0, 0, 0};
        double acc_im[6] = {0, 0, 0, 0, 0, 0};
        for (int l = 0; l < L; l++) {
            const double direction[3] = {
                dirs[3 * l], dirs[3 * l + 1], dirs[3 * l + 2]
            };
            const double dot =
                direction[0] * position[0] +
                direction[1] * position[1] +
                direction[2] * position[2];
            const double exponential =
                exp(-k_im * dot);
            const double phase = k_re * dot;
            const double exp_re = exponential * cos(phase);
            const double exp_im = exponential * sin(phase);
            const double weighted_re =
                weights[l] * local_re[local_base + l];
            const double weighted_im =
                weights[l] * local_im[local_base + l];
            const double plane_re =
                weighted_re * exp_re - weighted_im * exp_im;
            const double plane_im =
                weighted_re * exp_im + weighted_im * exp_re;
            const double second_re =
                neg_k2_re * plane_re - neg_k2_im * plane_im;
            const double second_im =
                neg_k2_re * plane_im + neg_k2_im * plane_re;
            for (int component = 0; component < 6; component++) {
                const double scale =
                    direction[row[component]] * direction[col[component]];
                acc_re[component] += second_re * scale;
                acc_im[component] += second_im * scale;
            }
        }
        for (int component = 0; component < 6; component++) {
            const double value_re =
                prefac_re * acc_re[component] -
                prefac_im * acc_im[component];
            const double value_im =
                prefac_re * acc_im[component] +
                prefac_im * acc_re[component];
            atomicAdd(&hess_re[6 * tid + component], value_re);
            atomicAdd(&hess_im[6 * tid + component], value_im);
        }
    }
}

__global__ void l2p_hessian_kernel_batch3(
    const double* __restrict__ tgt_pts,
    const double* __restrict__ dirs,
    const double* __restrict__ weights,
    const double* __restrict__ local1_re,
    const double* __restrict__ local1_im,
    const double* __restrict__ local2_re,
    const double* __restrict__ local2_im,
    const double* __restrict__ local3_re,
    const double* __restrict__ local3_im,
    double k_re, double k_im,
    double prefac_re, double prefac_im,
    double neg_k2_re, double neg_k2_im,
    double* __restrict__ hess1_re,
    double* __restrict__ hess1_im,
    double* __restrict__ hess2_re,
    double* __restrict__ hess2_im,
    double* __restrict__ hess3_re,
    double* __restrict__ hess3_im,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ tgt_id_offsets,
    const int* __restrict__ tgt_ids,
    const double* __restrict__ node_centers,
    int L, int n_leaves)
{
    const int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves) return;
    const int node = leaf_indices[leaf_id];
    const int start = tgt_id_offsets[leaf_id];
    const int end = tgt_id_offsets[leaf_id + 1];
    const double cx = node_centers[3 * node];
    const double cy = node_centers[3 * node + 1];
    const double cz = node_centers[3 * node + 2];
    const int local_base = node * L;
    const int row[6] = {0, 0, 0, 1, 1, 2};
    const int col[6] = {0, 1, 2, 1, 2, 2};

    for (int target = start + threadIdx.x;
         target < end; target += blockDim.x) {
        const int tid = tgt_ids[target];
        const double position[3] = {
            tgt_pts[3 * tid] - cx,
            tgt_pts[3 * tid + 1] - cy,
            tgt_pts[3 * tid + 2] - cz
        };
        double acc1_re[6] = {0, 0, 0, 0, 0, 0};
        double acc1_im[6] = {0, 0, 0, 0, 0, 0};
        double acc2_re[6] = {0, 0, 0, 0, 0, 0};
        double acc2_im[6] = {0, 0, 0, 0, 0, 0};
        double acc3_re[6] = {0, 0, 0, 0, 0, 0};
        double acc3_im[6] = {0, 0, 0, 0, 0, 0};
        for (int l = 0; l < L; l++) {
            const double direction[3] = {
                dirs[3 * l], dirs[3 * l + 1], dirs[3 * l + 2]
            };
            const double dot =
                direction[0] * position[0] +
                direction[1] * position[1] +
                direction[2] * position[2];
            const double exponential = exp(-k_im * dot);
            const double phase = k_re * dot;
            const double exp_re = exponential * cos(phase);
            const double exp_im = exponential * sin(phase);
            const double wl = weights[l];
            const double local_re[3] = {
                wl * local1_re[local_base + l],
                wl * local2_re[local_base + l],
                wl * local3_re[local_base + l]
            };
            const double local_im[3] = {
                wl * local1_im[local_base + l],
                wl * local2_im[local_base + l],
                wl * local3_im[local_base + l]
            };
            const double plane_re[3] = {
                local_re[0] * exp_re - local_im[0] * exp_im,
                local_re[1] * exp_re - local_im[1] * exp_im,
                local_re[2] * exp_re - local_im[2] * exp_im
            };
            const double plane_im[3] = {
                local_re[0] * exp_im + local_im[0] * exp_re,
                local_re[1] * exp_im + local_im[1] * exp_re,
                local_re[2] * exp_im + local_im[2] * exp_re
            };
            const double second_re[3] = {
                neg_k2_re * plane_re[0] - neg_k2_im * plane_im[0],
                neg_k2_re * plane_re[1] - neg_k2_im * plane_im[1],
                neg_k2_re * plane_re[2] - neg_k2_im * plane_im[2]
            };
            const double second_im[3] = {
                neg_k2_re * plane_im[0] + neg_k2_im * plane_re[0],
                neg_k2_re * plane_im[1] + neg_k2_im * plane_re[1],
                neg_k2_re * plane_im[2] + neg_k2_im * plane_re[2]
            };
            for (int component = 0; component < 6; component++) {
                const double scale =
                    direction[row[component]] * direction[col[component]];
                acc1_re[component] += second_re[0] * scale;
                acc1_im[component] += second_im[0] * scale;
                acc2_re[component] += second_re[1] * scale;
                acc2_im[component] += second_im[1] * scale;
                acc3_re[component] += second_re[2] * scale;
                acc3_im[component] += second_im[2] * scale;
            }
        }
#define STORE_HESSIAN_BATCH(OUT_RE, OUT_IM, ACC_RE, ACC_IM) \
        do { \
            for (int component = 0; component < 6; component++) { \
                const double value_re = \
                    prefac_re * (ACC_RE)[component] - \
                    prefac_im * (ACC_IM)[component]; \
                const double value_im = \
                    prefac_re * (ACC_IM)[component] + \
                    prefac_im * (ACC_RE)[component]; \
                atomicAdd(&(OUT_RE)[6 * tid + component], value_re); \
                atomicAdd(&(OUT_IM)[6 * tid + component], value_im); \
            } \
        } while (0)
        STORE_HESSIAN_BATCH(hess1_re, hess1_im, acc1_re, acc1_im);
        STORE_HESSIAN_BATCH(hess2_re, hess2_im, acc2_re, acc2_im);
        STORE_HESSIAN_BATCH(hess3_re, hess3_im, acc3_re, acc3_im);
#undef STORE_HESSIAN_BATCH
    }
}

// Directly contract the derivatives needed by the Muller current operator.
// This avoids writing three full gradients and three full Hessians.
__global__ void l2p_vector_actions_kernel_batch3(
    const double* __restrict__ tgt_pts,
    const double* __restrict__ dirs,
    const double* __restrict__ weights,
    const double* __restrict__ local_x_re,
    const double* __restrict__ local_x_im,
    const double* __restrict__ local_y_re,
    const double* __restrict__ local_y_im,
    const double* __restrict__ local_z_re,
    const double* __restrict__ local_z_im,
    double k_re, double k_im,
    double prefac_re, double prefac_im,
    double ik_re, double ik_im,
    double neg_k2_re, double neg_k2_im,
    double* __restrict__ curl_re,
    double* __restrict__ curl_im,
    double* __restrict__ hessian_action_re,
    double* __restrict__ hessian_action_im,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ tgt_id_offsets,
    const int* __restrict__ tgt_ids,
    const double* __restrict__ node_centers,
    int L, int n_leaves)
{
    const int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves)
        return;
    const int node = leaf_indices[leaf_id];
    const int start = tgt_id_offsets[leaf_id];
    const int end = tgt_id_offsets[leaf_id + 1];
    const double cx = node_centers[3 * node];
    const double cy = node_centers[3 * node + 1];
    const double cz = node_centers[3 * node + 2];
    const int local_base = node * L;
    const bool real_wavenumber = k_im == 0.0;

    for (int target = start + threadIdx.x;
         target < end; target += blockDim.x) {
        const int tid = tgt_ids[target];
        const double rx = tgt_pts[3 * tid] - cx;
        const double ry = tgt_pts[3 * tid + 1] - cy;
        const double rz = tgt_pts[3 * tid + 2] - cz;
        double curl_acc_re[3] = {0, 0, 0};
        double curl_acc_im[3] = {0, 0, 0};
        double action_acc_re[3] = {0, 0, 0};
        double action_acc_im[3] = {0, 0, 0};

        for (int l = 0; l < L; l++) {
            const double dx = dirs[3 * l];
            const double dy = dirs[3 * l + 1];
            const double dz = dirs[3 * l + 2];
            const double dot = dx * rx + dy * ry + dz * rz;
            const double exponential =
                real_wavenumber ? 1.0 : exp(-k_im * dot);
            const double phase = k_re * dot;
            double phase_sine = 0;
            double phase_cosine = 0;
            sincos(phase, &phase_sine, &phase_cosine);
            const double exp_re = exponential * phase_cosine;
            const double exp_im = exponential * phase_sine;
            const double weight = weights[l];
            const double weighted_re[3] = {
                weight * local_x_re[local_base + l],
                weight * local_y_re[local_base + l],
                weight * local_z_re[local_base + l]
            };
            const double weighted_im[3] = {
                weight * local_x_im[local_base + l],
                weight * local_y_im[local_base + l],
                weight * local_z_im[local_base + l]
            };
            double plane_re[3];
            double plane_im[3];
            double gradient_re[3];
            double gradient_im[3];
            double second_re[3];
            double second_im[3];
            for (int component = 0; component < 3; component++) {
                plane_re[component] =
                    weighted_re[component] * exp_re -
                    weighted_im[component] * exp_im;
                plane_im[component] =
                    weighted_re[component] * exp_im +
                    weighted_im[component] * exp_re;
                gradient_re[component] =
                    ik_re * plane_re[component] -
                    ik_im * plane_im[component];
                gradient_im[component] =
                    ik_re * plane_im[component] +
                    ik_im * plane_re[component];
                second_re[component] =
                    neg_k2_re * plane_re[component] -
                    neg_k2_im * plane_im[component];
                second_im[component] =
                    neg_k2_re * plane_im[component] +
                    neg_k2_im * plane_re[component];
            }

            curl_acc_re[0] +=
                dy * gradient_re[0] - dx * gradient_re[1];
            curl_acc_im[0] +=
                dy * gradient_im[0] - dx * gradient_im[1];
            curl_acc_re[1] +=
                dz * gradient_re[0] - dx * gradient_re[2];
            curl_acc_im[1] +=
                dz * gradient_im[0] - dx * gradient_im[2];
            curl_acc_re[2] +=
                dz * gradient_re[1] - dy * gradient_re[2];
            curl_acc_im[2] +=
                dz * gradient_im[1] - dy * gradient_im[2];

            action_acc_re[0] +=
                -(dy * dy + dz * dz) * second_re[0] +
                dx * dy * second_re[1] +
                dx * dz * second_re[2];
            action_acc_im[0] +=
                -(dy * dy + dz * dz) * second_im[0] +
                dx * dy * second_im[1] +
                dx * dz * second_im[2];
            action_acc_re[1] +=
                dx * dy * second_re[0] -
                (dx * dx + dz * dz) * second_re[1] +
                dy * dz * second_re[2];
            action_acc_im[1] +=
                dx * dy * second_im[0] -
                (dx * dx + dz * dz) * second_im[1] +
                dy * dz * second_im[2];
            action_acc_re[2] +=
                dx * dz * second_re[0] +
                dy * dz * second_re[1] -
                (dx * dx + dy * dy) * second_re[2];
            action_acc_im[2] +=
                dx * dz * second_im[0] +
                dy * dz * second_im[1] -
                (dx * dx + dy * dy) * second_im[2];
        }

        for (int component = 0; component < 3; component++) {
            const int offset = 3 * tid + component;
            curl_re[offset] =
                prefac_re * curl_acc_re[component] -
                prefac_im * curl_acc_im[component];
            curl_im[offset] =
                prefac_re * curl_acc_im[component] +
                prefac_im * curl_acc_re[component];
            hessian_action_re[offset] =
                prefac_re * action_acc_re[component] -
                prefac_im * action_acc_im[component];
            hessian_action_im[offset] =
                prefac_re * action_acc_im[component] +
                prefac_im * action_acc_re[component];
        }
    }
}

__device__ __forceinline__ float fmm_l2p_attenuation(float value)
{
    return expf(value);
}

__device__ __forceinline__ double fmm_l2p_attenuation(double value)
{
    return exp(value);
}

__device__ __forceinline__ void fmm_l2p_sine_cosine(
    float value, float* sine, float* cosine)
{
    sincosf(value, sine, cosine);
}

__device__ __forceinline__ void fmm_l2p_sine_cosine(
    double value, double* sine, double* cosine)
{
    sincos(value, sine, cosine);
}

template <typename ComputeReal>
__device__ __forceinline__ void accumulate_l2p_vector_field(
    ComputeReal weight,
    ComputeReal exp_re,
    ComputeReal exp_im,
    ComputeReal dx,
    ComputeReal dy,
    ComputeReal dz,
    ComputeReal local_x_re,
    ComputeReal local_x_im,
    ComputeReal local_y_re,
    ComputeReal local_y_im,
    ComputeReal local_z_re,
    ComputeReal local_z_im,
    ComputeReal curl_acc_re[3],
    ComputeReal curl_acc_im[3],
    ComputeReal action_acc_re[3],
    ComputeReal action_acc_im[3])
{
    const ComputeReal weighted_re[3] = {
        weight * local_x_re,
        weight * local_y_re,
        weight * local_z_re
    };
    const ComputeReal weighted_im[3] = {
        weight * local_x_im,
        weight * local_y_im,
        weight * local_z_im
    };
    ComputeReal plane_re[3];
    ComputeReal plane_im[3];
#pragma unroll
    for (int component = 0; component < 3; component++) {
        plane_re[component] =
            weighted_re[component] * exp_re -
            weighted_im[component] * exp_im;
        plane_im[component] =
            weighted_re[component] * exp_im +
            weighted_im[component] * exp_re;
    }
    curl_acc_re[0] +=
        dy * plane_re[0] - dx * plane_re[1];
    curl_acc_im[0] +=
        dy * plane_im[0] - dx * plane_im[1];
    curl_acc_re[1] +=
        dz * plane_re[0] - dx * plane_re[2];
    curl_acc_im[1] +=
        dz * plane_im[0] - dx * plane_im[2];
    curl_acc_re[2] +=
        dz * plane_re[1] - dy * plane_re[2];
    curl_acc_im[2] +=
        dz * plane_im[1] - dy * plane_im[2];

    const ComputeReal direction_norm_squared =
        dx * dx + dy * dy + dz * dz;
    const ComputeReal direction_dot_re =
        dx * plane_re[0] +
        dy * plane_re[1] +
        dz * plane_re[2];
    const ComputeReal direction_dot_im =
        dx * plane_im[0] +
        dy * plane_im[1] +
        dz * plane_im[2];
    const ComputeReal direction[3] = {dx, dy, dz};
#pragma unroll
    for (int component = 0; component < 3; component++) {
        action_acc_re[component] +=
            direction[component] * direction_dot_re -
            direction_norm_squared * plane_re[component];
        action_acc_im[component] +=
            direction[component] * direction_dot_im -
            direction_norm_squared * plane_im[component];
    }
}

template <
    bool CacheLocal,
    bool CachePhase,
    typename ComputeReal,
    typename LocalReal,
    typename DirectionReal = double,
    bool TransposedPhase = false>
__global__ void l2p_vector_actions_pair_batch3_kernel(
    const double* __restrict__ tgt_pts,
    const DirectionReal* __restrict__ dirs,
    const DirectionReal* __restrict__ weights,
    const float2* __restrict__ phase_cache,
    const LocalReal* __restrict__ first_local_re,
    const LocalReal* __restrict__ first_local_im,
    const LocalReal* __restrict__ second_local_x_re,
    const LocalReal* __restrict__ second_local_x_im,
    const LocalReal* __restrict__ second_local_y_re,
    const LocalReal* __restrict__ second_local_y_im,
    const LocalReal* __restrict__ second_local_z_re,
    const LocalReal* __restrict__ second_local_z_im,
    int local_field_stride,
    int phase_stride,
    double k_re, double k_im,
    double prefac_re, double prefac_im,
    double ik_re, double ik_im,
    double neg_k2_re, double neg_k2_im,
    double* __restrict__ first_curl_re,
    double* __restrict__ first_curl_im,
    double* __restrict__ first_action_re,
    double* __restrict__ first_action_im,
    double* __restrict__ second_curl_re,
    double* __restrict__ second_curl_im,
    double* __restrict__ second_action_re,
    double* __restrict__ second_action_im,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ tgt_id_offsets,
    const int* __restrict__ tgt_ids,
    const double* __restrict__ node_centers,
    int L, int n_leaves)
{
    const int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves)
        return;
    const int node = leaf_indices[leaf_id];
    const int start = tgt_id_offsets[leaf_id];
    const int end = tgt_id_offsets[leaf_id + 1];
    const double cx = node_centers[3 * node];
    const double cy = node_centers[3 * node + 1];
    const double cz = node_centers[3 * node + 2];
    const int local_base = node * L;
    const bool real_wavenumber = k_im == 0.0;
    const ComputeReal real_k =
        static_cast<ComputeReal>(k_re);
    const ComputeReal imaginary_k =
        static_cast<ComputeReal>(k_im);
    const ComputeReal curl_scale_re =
        static_cast<ComputeReal>(
            prefac_re * ik_re - prefac_im * ik_im);
    const ComputeReal curl_scale_im =
        static_cast<ComputeReal>(
            prefac_re * ik_im + prefac_im * ik_re);
    const ComputeReal action_scale_re =
        static_cast<ComputeReal>(
            prefac_re * neg_k2_re - prefac_im * neg_k2_im);
    const ComputeReal action_scale_im =
        static_cast<ComputeReal>(
            prefac_re * neg_k2_im + prefac_im * neg_k2_re);

    extern __shared__ unsigned char local_cache_bytes[];
    LocalReal* local_cache =
        reinterpret_cast<LocalReal*>(local_cache_bytes);
    const LocalReal* eval_first_x_re = first_local_re + local_base;
    const LocalReal* eval_first_x_im = first_local_im + local_base;
    const LocalReal* eval_first_y_re =
        first_local_re + local_field_stride + local_base;
    const LocalReal* eval_first_y_im =
        first_local_im + local_field_stride + local_base;
    const LocalReal* eval_first_z_re =
        first_local_re + 2 * local_field_stride + local_base;
    const LocalReal* eval_first_z_im =
        first_local_im + 2 * local_field_stride + local_base;
    const LocalReal* eval_second_x_re =
        second_local_x_re + local_base;
    const LocalReal* eval_second_x_im =
        second_local_x_im + local_base;
    const LocalReal* eval_second_y_re =
        second_local_y_re + local_base;
    const LocalReal* eval_second_y_im =
        second_local_y_im + local_base;
    const LocalReal* eval_second_z_re =
        second_local_z_re + local_base;
    const LocalReal* eval_second_z_im =
        second_local_z_im + local_base;

    if (CacheLocal) {
        for (int l = threadIdx.x; l < L; l += blockDim.x) {
            local_cache[l] = eval_first_x_re[l];
            local_cache[L + l] = eval_first_x_im[l];
            local_cache[2 * L + l] = eval_first_y_re[l];
            local_cache[3 * L + l] = eval_first_y_im[l];
            local_cache[4 * L + l] = eval_first_z_re[l];
            local_cache[5 * L + l] = eval_first_z_im[l];
            local_cache[6 * L + l] = eval_second_x_re[l];
            local_cache[7 * L + l] = eval_second_x_im[l];
            local_cache[8 * L + l] = eval_second_y_re[l];
            local_cache[9 * L + l] = eval_second_y_im[l];
            local_cache[10 * L + l] = eval_second_z_re[l];
            local_cache[11 * L + l] = eval_second_z_im[l];
        }
        __syncthreads();
        eval_first_x_re = local_cache;
        eval_first_x_im = local_cache + L;
        eval_first_y_re = local_cache + 2 * L;
        eval_first_y_im = local_cache + 3 * L;
        eval_first_z_re = local_cache + 4 * L;
        eval_first_z_im = local_cache + 5 * L;
        eval_second_x_re = local_cache + 6 * L;
        eval_second_x_im = local_cache + 7 * L;
        eval_second_y_re = local_cache + 8 * L;
        eval_second_y_im = local_cache + 9 * L;
        eval_second_z_re = local_cache + 10 * L;
        eval_second_z_im = local_cache + 11 * L;
    }

    for (int target = start + threadIdx.x;
         target < end; target += blockDim.x) {
        const int tid = tgt_ids[target];
        const ComputeReal rx = static_cast<ComputeReal>(
            tgt_pts[3 * tid] - cx);
        const ComputeReal ry = static_cast<ComputeReal>(
            tgt_pts[3 * tid + 1] - cy);
        const ComputeReal rz = static_cast<ComputeReal>(
            tgt_pts[3 * tid + 2] - cz);
        ComputeReal first_curl_acc_re[3] = {0, 0, 0};
        ComputeReal first_curl_acc_im[3] = {0, 0, 0};
        ComputeReal first_action_acc_re[3] = {0, 0, 0};
        ComputeReal first_action_acc_im[3] = {0, 0, 0};
        ComputeReal second_curl_acc_re[3] = {0, 0, 0};
        ComputeReal second_curl_acc_im[3] = {0, 0, 0};
        ComputeReal second_action_acc_re[3] = {0, 0, 0};
        ComputeReal second_action_acc_im[3] = {0, 0, 0};

        for (int l = 0; l < L; l++) {
            const ComputeReal dx =
                static_cast<ComputeReal>(dirs[3 * l]);
            const ComputeReal dy =
                static_cast<ComputeReal>(dirs[3 * l + 1]);
            const ComputeReal dz =
                static_cast<ComputeReal>(dirs[3 * l + 2]);
            ComputeReal exp_re = 0;
            ComputeReal exp_im = 0;
            if (CachePhase) {
                const float2 cached = TransposedPhase
                    ? phase_cache[
                        (size_t)l * phase_stride + target]
                    : phase_cache[(size_t)tid * L + l];
                exp_re = static_cast<ComputeReal>(cached.x);
                exp_im = static_cast<ComputeReal>(cached.y);
            } else {
                const ComputeReal dot =
                    dx * rx + dy * ry + dz * rz;
                const ComputeReal exponential =
                    real_wavenumber
                        ? static_cast<ComputeReal>(1)
                        : fmm_l2p_attenuation(
                            -imaginary_k * dot);
                const ComputeReal phase = real_k * dot;
                ComputeReal phase_sine = 0;
                ComputeReal phase_cosine = 0;
                fmm_l2p_sine_cosine(
                    phase, &phase_sine, &phase_cosine);
                exp_re = exponential * phase_cosine;
                exp_im = exponential * phase_sine;
            }
            accumulate_l2p_vector_field<ComputeReal>(
                static_cast<ComputeReal>(weights[l]),
                exp_re, exp_im,
                dx, dy, dz,
                static_cast<ComputeReal>(eval_first_x_re[l]),
                static_cast<ComputeReal>(eval_first_x_im[l]),
                static_cast<ComputeReal>(eval_first_y_re[l]),
                static_cast<ComputeReal>(eval_first_y_im[l]),
                static_cast<ComputeReal>(eval_first_z_re[l]),
                static_cast<ComputeReal>(eval_first_z_im[l]),
                first_curl_acc_re, first_curl_acc_im,
                first_action_acc_re, first_action_acc_im);
            accumulate_l2p_vector_field<ComputeReal>(
                static_cast<ComputeReal>(weights[l]),
                exp_re, exp_im,
                dx, dy, dz,
                static_cast<ComputeReal>(eval_second_x_re[l]),
                static_cast<ComputeReal>(eval_second_x_im[l]),
                static_cast<ComputeReal>(eval_second_y_re[l]),
                static_cast<ComputeReal>(eval_second_y_im[l]),
                static_cast<ComputeReal>(eval_second_z_re[l]),
                static_cast<ComputeReal>(eval_second_z_im[l]),
                second_curl_acc_re, second_curl_acc_im,
                second_action_acc_re, second_action_acc_im);
        }

#define STORE_VECTOR_ACTION(CURL_RE, CURL_IM, ACTION_RE, ACTION_IM, \
                            CURL_ACC_RE, CURL_ACC_IM, \
                            ACTION_ACC_RE, ACTION_ACC_IM) \
        do { \
            for (int component = 0; component < 3; component++) { \
                const int offset = 3 * tid + component; \
                (CURL_RE)[offset] = \
                    curl_scale_re * (CURL_ACC_RE)[component] - \
                    curl_scale_im * (CURL_ACC_IM)[component]; \
                (CURL_IM)[offset] = \
                    curl_scale_re * (CURL_ACC_IM)[component] + \
                    curl_scale_im * (CURL_ACC_RE)[component]; \
                (ACTION_RE)[offset] = \
                    action_scale_re * (ACTION_ACC_RE)[component] - \
                    action_scale_im * (ACTION_ACC_IM)[component]; \
                (ACTION_IM)[offset] = \
                    action_scale_re * (ACTION_ACC_IM)[component] + \
                    action_scale_im * (ACTION_ACC_RE)[component]; \
            } \
        } while (0)
        STORE_VECTOR_ACTION(
            first_curl_re, first_curl_im,
            first_action_re, first_action_im,
            first_curl_acc_re, first_curl_acc_im,
            first_action_acc_re, first_action_acc_im);
        STORE_VECTOR_ACTION(
            second_curl_re, second_curl_im,
            second_action_re, second_action_im,
            second_curl_acc_re, second_curl_acc_im,
            second_action_acc_re, second_action_acc_im);
#undef STORE_VECTOR_ACTION
    }
}

__device__ __forceinline__ float warp_sum_float(float value)
{
    constexpr unsigned full_warp = 0xffffffffu;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(full_warp, value, offset);
    return value;
}

// A warp cooperates on one target. The target-major phase cache and the local
// coefficients are consequently read in contiguous 32-value segments.
__global__ void l2p_vector_actions_pair_batch3_warp_fp32_kernel(
    const float* __restrict__ dirs,
    const float* __restrict__ weights,
    const float2* __restrict__ phase_cache,
    const float* __restrict__ first_local_re,
    const float* __restrict__ first_local_im,
    const float* __restrict__ second_local_x_re,
    const float* __restrict__ second_local_x_im,
    const float* __restrict__ second_local_y_re,
    const float* __restrict__ second_local_y_im,
    const float* __restrict__ second_local_z_re,
    const float* __restrict__ second_local_z_im,
    int local_field_stride,
    float curl_scale_re,
    float curl_scale_im,
    float action_scale_re,
    float action_scale_im,
    double* __restrict__ first_curl_re,
    double* __restrict__ first_curl_im,
    double* __restrict__ first_action_re,
    double* __restrict__ first_action_im,
    double* __restrict__ second_curl_re,
    double* __restrict__ second_curl_im,
    double* __restrict__ second_action_re,
    double* __restrict__ second_action_im,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ tgt_id_offsets,
    const int* __restrict__ tgt_ids,
    int L,
    int n_leaves)
{
    const int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves)
        return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int node = leaf_indices[leaf_id];
    const int start = tgt_id_offsets[leaf_id];
    const int end = tgt_id_offsets[leaf_id + 1];
    const int local_base = node * L;

    const float* eval_first_x_re = first_local_re + local_base;
    const float* eval_first_x_im = first_local_im + local_base;
    const float* eval_first_y_re =
        first_local_re + local_field_stride + local_base;
    const float* eval_first_y_im =
        first_local_im + local_field_stride + local_base;
    const float* eval_first_z_re =
        first_local_re + 2 * local_field_stride + local_base;
    const float* eval_first_z_im =
        first_local_im + 2 * local_field_stride + local_base;
    const float* eval_second_x_re = second_local_x_re + local_base;
    const float* eval_second_x_im = second_local_x_im + local_base;
    const float* eval_second_y_re = second_local_y_re + local_base;
    const float* eval_second_y_im = second_local_y_im + local_base;
    const float* eval_second_z_re = second_local_z_re + local_base;
    const float* eval_second_z_im = second_local_z_im + local_base;

    for (int target = start + warp; target < end;
         target += warps_per_block) {
        const int tid = tgt_ids[target];
        float first_curl_acc_re[3] = {0, 0, 0};
        float first_curl_acc_im[3] = {0, 0, 0};
        float first_action_acc_re[3] = {0, 0, 0};
        float first_action_acc_im[3] = {0, 0, 0};
        float second_curl_acc_re[3] = {0, 0, 0};
        float second_curl_acc_im[3] = {0, 0, 0};
        float second_action_acc_re[3] = {0, 0, 0};
        float second_action_acc_im[3] = {0, 0, 0};

        for (int l = lane; l < L; l += 32) {
            const float dx = dirs[3 * l];
            const float dy = dirs[3 * l + 1];
            const float dz = dirs[3 * l + 2];
            const float2 phase = phase_cache[(size_t)tid * L + l];
            accumulate_l2p_vector_field<float>(
                weights[l], phase.x, phase.y, dx, dy, dz,
                eval_first_x_re[l], eval_first_x_im[l],
                eval_first_y_re[l], eval_first_y_im[l],
                eval_first_z_re[l], eval_first_z_im[l],
                first_curl_acc_re, first_curl_acc_im,
                first_action_acc_re, first_action_acc_im);
            accumulate_l2p_vector_field<float>(
                weights[l], phase.x, phase.y, dx, dy, dz,
                eval_second_x_re[l], eval_second_x_im[l],
                eval_second_y_re[l], eval_second_y_im[l],
                eval_second_z_re[l], eval_second_z_im[l],
                second_curl_acc_re, second_curl_acc_im,
                second_action_acc_re, second_action_acc_im);
        }

#pragma unroll
        for (int component = 0; component < 3; component++) {
            first_curl_acc_re[component] =
                warp_sum_float(first_curl_acc_re[component]);
            first_curl_acc_im[component] =
                warp_sum_float(first_curl_acc_im[component]);
            first_action_acc_re[component] =
                warp_sum_float(first_action_acc_re[component]);
            first_action_acc_im[component] =
                warp_sum_float(first_action_acc_im[component]);
            second_curl_acc_re[component] =
                warp_sum_float(second_curl_acc_re[component]);
            second_curl_acc_im[component] =
                warp_sum_float(second_curl_acc_im[component]);
            second_action_acc_re[component] =
                warp_sum_float(second_action_acc_re[component]);
            second_action_acc_im[component] =
                warp_sum_float(second_action_acc_im[component]);
        }

        if (lane == 0) {
#define STORE_WARP_VECTOR_ACTION(                                      \
    CURL_RE, CURL_IM, ACTION_RE, ACTION_IM, CURL_ACC_RE, CURL_ACC_IM, \
    ACTION_ACC_RE, ACTION_ACC_IM)                                      \
            do {                                                       \
                for (int component = 0; component < 3; component++) { \
                    const int offset = 3 * tid + component;            \
                    (CURL_RE)[offset] =                                \
                        curl_scale_re * (CURL_ACC_RE)[component] -     \
                        curl_scale_im * (CURL_ACC_IM)[component];      \
                    (CURL_IM)[offset] =                                \
                        curl_scale_re * (CURL_ACC_IM)[component] +     \
                        curl_scale_im * (CURL_ACC_RE)[component];      \
                    (ACTION_RE)[offset] =                              \
                        action_scale_re * (ACTION_ACC_RE)[component] - \
                        action_scale_im * (ACTION_ACC_IM)[component];  \
                    (ACTION_IM)[offset] =                              \
                        action_scale_re * (ACTION_ACC_IM)[component] + \
                        action_scale_im * (ACTION_ACC_RE)[component];  \
                }                                                      \
            } while (0)
            STORE_WARP_VECTOR_ACTION(
                first_curl_re, first_curl_im,
                first_action_re, first_action_im,
                first_curl_acc_re, first_curl_acc_im,
                first_action_acc_re, first_action_acc_im);
            STORE_WARP_VECTOR_ACTION(
                second_curl_re, second_curl_im,
                second_action_re, second_action_im,
                second_curl_acc_re, second_curl_acc_im,
                second_action_acc_re, second_action_acc_im);
#undef STORE_WARP_VECTOR_ACTION
        }
    }
}

__global__ void l2p_gradient_kernel_batch2(
    const double* __restrict__ tgt_pts,
    const double* __restrict__ dirs,
    const double* __restrict__ weights,
    const double* __restrict__ local1_re,
    const double* __restrict__ local1_im,
    const double* __restrict__ local2_re,
    const double* __restrict__ local2_im,
    double k_re, double k_im,
    double prefac_re, double prefac_im,
    double ik2_re, double ik2_im,
    double* __restrict__ gx1_re, double* __restrict__ gx1_im,
    double* __restrict__ gy1_re, double* __restrict__ gy1_im,
    double* __restrict__ gz1_re, double* __restrict__ gz1_im,
    double* __restrict__ gx2_re, double* __restrict__ gx2_im,
    double* __restrict__ gy2_re, double* __restrict__ gy2_im,
    double* __restrict__ gz2_re, double* __restrict__ gz2_im,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ tgt_id_offsets,
    const int* __restrict__ tgt_ids,
    const double* __restrict__ node_centers,
    int L, int n_leaves)
{
    int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves) return;

    int node = leaf_indices[leaf_id];
    int t_start = tgt_id_offsets[leaf_id];
    int t_end = tgt_id_offsets[leaf_id + 1];
    int t_count = t_end - t_start;
    if (t_count == 0) return;

    double ccx = node_centers[node*3];
    double ccy = node_centers[node*3+1];
    double ccz = node_centers[node*3+2];
    int l_base = node * L;

    for (int t = threadIdx.x; t < t_count; t += blockDim.x) {
        int tid = tgt_ids[t_start + t];
        double rx = tgt_pts[tid*3]   - ccx;
        double ry = tgt_pts[tid*3+1] - ccy;
        double rz = tgt_pts[tid*3+2] - ccz;

        double ax1_re = 0, ax1_im = 0, ay1_re = 0, ay1_im = 0, az1_re = 0, az1_im = 0;
        double ax2_re = 0, ax2_im = 0, ay2_re = 0, ay2_im = 0, az2_re = 0, az2_im = 0;

        for (int l = 0; l < L; l++) {
            double dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
            double dot = dx*rx + dy*ry + dz*rz;
            double phase_re = -k_im * dot;
            double phase_im = k_re * dot;
            double e_re = exp(phase_re) * cos(phase_im);
            double e_im = exp(phase_re) * sin(phase_im);
            double wl = weights[l];

            double l1r = wl * local1_re[l_base + l], l1i = wl * local1_im[l_base + l];
            double p1r = l1r * e_re - l1i * e_im;
            double p1i = l1r * e_im + l1i * e_re;
            double ikp1r = ik2_re * p1r - ik2_im * p1i;
            double ikp1i = ik2_re * p1i + ik2_im * p1r;
            ax1_re += ikp1r * dx; ax1_im += ikp1i * dx;
            ay1_re += ikp1r * dy; ay1_im += ikp1i * dy;
            az1_re += ikp1r * dz; az1_im += ikp1i * dz;

            double l2r = wl * local2_re[l_base + l], l2i = wl * local2_im[l_base + l];
            double p2r = l2r * e_re - l2i * e_im;
            double p2i = l2r * e_im + l2i * e_re;
            double ikp2r = ik2_re * p2r - ik2_im * p2i;
            double ikp2i = ik2_re * p2i + ik2_im * p2r;
            ax2_re += ikp2r * dx; ax2_im += ikp2i * dx;
            ay2_re += ikp2r * dy; ay2_im += ikp2i * dy;
            az2_re += ikp2r * dz; az2_im += ikp2i * dz;
        }

        atomicAdd(&gx1_re[tid], prefac_re*ax1_re - prefac_im*ax1_im);
        atomicAdd(&gx1_im[tid], prefac_re*ax1_im + prefac_im*ax1_re);
        atomicAdd(&gy1_re[tid], prefac_re*ay1_re - prefac_im*ay1_im);
        atomicAdd(&gy1_im[tid], prefac_re*ay1_im + prefac_im*ay1_re);
        atomicAdd(&gz1_re[tid], prefac_re*az1_re - prefac_im*az1_im);
        atomicAdd(&gz1_im[tid], prefac_re*az1_im + prefac_im*az1_re);

        atomicAdd(&gx2_re[tid], prefac_re*ax2_re - prefac_im*ax2_im);
        atomicAdd(&gx2_im[tid], prefac_re*ax2_im + prefac_im*ax2_re);
        atomicAdd(&gy2_re[tid], prefac_re*ay2_re - prefac_im*ay2_im);
        atomicAdd(&gy2_im[tid], prefac_re*ay2_im + prefac_im*ay2_re);
        atomicAdd(&gz2_re[tid], prefac_re*az2_re - prefac_im*az2_im);
        atomicAdd(&gz2_im[tid], prefac_re*az2_im + prefac_im*az2_re);
    }
}

__global__ void l2p_gradient_kernel_batch4(
    const double* __restrict__ tgt_pts,
    const double* __restrict__ dirs,
    const double* __restrict__ weights,
    const double* __restrict__ l1_re, const double* __restrict__ l1_im,
    const double* __restrict__ l2_re, const double* __restrict__ l2_im,
    const double* __restrict__ l3_re, const double* __restrict__ l3_im,
    const double* __restrict__ l4_re, const double* __restrict__ l4_im,
    double k_re, double k_im, double prefac_re, double prefac_im,
    double ik2_re, double ik2_im,
    double* __restrict__ gx1_re, double* __restrict__ gx1_im,
    double* __restrict__ gy1_re, double* __restrict__ gy1_im,
    double* __restrict__ gz1_re, double* __restrict__ gz1_im,
    double* __restrict__ gx2_re, double* __restrict__ gx2_im,
    double* __restrict__ gy2_re, double* __restrict__ gy2_im,
    double* __restrict__ gz2_re, double* __restrict__ gz2_im,
    double* __restrict__ gx3_re, double* __restrict__ gx3_im,
    double* __restrict__ gy3_re, double* __restrict__ gy3_im,
    double* __restrict__ gz3_re, double* __restrict__ gz3_im,
    double* __restrict__ gx4_re, double* __restrict__ gx4_im,
    double* __restrict__ gy4_re, double* __restrict__ gy4_im,
    double* __restrict__ gz4_re, double* __restrict__ gz4_im,
    const int* __restrict__ leaf_indices,
    const int* __restrict__ tgt_id_offsets,
    const int* __restrict__ tgt_ids,
    const double* __restrict__ node_centers,
    int L, int n_leaves)
{
    int leaf_id = blockIdx.x;
    if (leaf_id >= n_leaves) return;
    int node = leaf_indices[leaf_id];
    int t_start = tgt_id_offsets[leaf_id], t_end = tgt_id_offsets[leaf_id + 1];
    if (t_end == t_start) return;
    double ccx = node_centers[node*3], ccy = node_centers[node*3+1], ccz = node_centers[node*3+2];
    int l_base = node * L;

    for (int t = threadIdx.x; t < t_end - t_start; t += blockDim.x) {
        int tid = tgt_ids[t_start + t];
        double rx = tgt_pts[tid*3] - ccx, ry = tgt_pts[tid*3+1] - ccy, rz = tgt_pts[tid*3+2] - ccz;
        double ax1r=0, ax1i=0, ay1r=0, ay1i=0, az1r=0, az1i=0;
        double ax2r=0, ax2i=0, ay2r=0, ay2i=0, az2r=0, az2i=0;
        double ax3r=0, ax3i=0, ay3r=0, ay3i=0, az3r=0, az3i=0;
        double ax4r=0, ax4i=0, ay4r=0, ay4i=0, az4r=0, az4i=0;
        for (int l = 0; l < L; l++) {
            double dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
            double dot = dx*rx + dy*ry + dz*rz;
            double e_re = exp(-k_im * dot) * cos(k_re * dot);
            double e_im = exp(-k_im * dot) * sin(k_re * dot);
            double wl = weights[l];
#define ACC_GRAD(LR, LI, AXR, AXI, AYR, AYI, AZR, AZI) \
            do { \
                double pr = (wl * (LR)) * e_re - (wl * (LI)) * e_im; \
                double pi = (wl * (LR)) * e_im + (wl * (LI)) * e_re; \
                double ikpr = ik2_re * pr - ik2_im * pi; \
                double ikpi = ik2_re * pi + ik2_im * pr; \
                AXR += ikpr * dx; AXI += ikpi * dx; \
                AYR += ikpr * dy; AYI += ikpi * dy; \
                AZR += ikpr * dz; AZI += ikpi * dz; \
            } while (0)
            ACC_GRAD(l1_re[l_base + l], l1_im[l_base + l], ax1r, ax1i, ay1r, ay1i, az1r, az1i);
            ACC_GRAD(l2_re[l_base + l], l2_im[l_base + l], ax2r, ax2i, ay2r, ay2i, az2r, az2i);
            ACC_GRAD(l3_re[l_base + l], l3_im[l_base + l], ax3r, ax3i, ay3r, ay3i, az3r, az3i);
            ACC_GRAD(l4_re[l_base + l], l4_im[l_base + l], ax4r, ax4i, ay4r, ay4i, az4r, az4i);
#undef ACC_GRAD
        }
#define STORE_GRAD(GXR, GXI, GYR, GYI, GZR, GZI, AXR, AXI, AYR, AYI, AZR, AZI) \
        do { \
            atomicAdd(&(GXR)[tid], prefac_re*(AXR) - prefac_im*(AXI)); \
            atomicAdd(&(GXI)[tid], prefac_re*(AXI) + prefac_im*(AXR)); \
            atomicAdd(&(GYR)[tid], prefac_re*(AYR) - prefac_im*(AYI)); \
            atomicAdd(&(GYI)[tid], prefac_re*(AYI) + prefac_im*(AYR)); \
            atomicAdd(&(GZR)[tid], prefac_re*(AZR) - prefac_im*(AZI)); \
            atomicAdd(&(GZI)[tid], prefac_re*(AZI) + prefac_im*(AZR)); \
        } while (0)
        STORE_GRAD(gx1_re, gx1_im, gy1_re, gy1_im, gz1_re, gz1_im, ax1r, ax1i, ay1r, ay1i, az1r, az1i);
        STORE_GRAD(gx2_re, gx2_im, gy2_re, gy2_im, gz2_re, gz2_im, ax2r, ax2i, ay2r, ay2i, az2r, az2i);
        STORE_GRAD(gx3_re, gx3_im, gy3_re, gy3_im, gz3_re, gz3_im, ax3r, ax3i, ay3r, ay3i, az3r, az3i);
        STORE_GRAD(gx4_re, gx4_im, gy4_re, gy4_im, gz4_re, gz4_im, ax4r, ax4i, ay4r, ay4i, az4r, az4i);
#undef STORE_GRAD
    }
}

// ============================================================
// HelmholtzFMM implementation
// ============================================================

struct DisplacementKey {
    long long ix, iy, iz;
    bool operator<(const DisplacementKey& o) const {
        if (ix != o.ix) return ix < o.ix;
        if (iy != o.iy) return iy < o.iy;
        return iz < o.iz;
    }
};

static DisplacementKey make_key(const double d[3], double eps) {
    DisplacementKey key;
    key.ix = (long long)std::round(d[0] / eps);
    key.iy = (long long)std::round(d[1] / eps);
    key.iz = (long long)std::round(d[2] / eps);
    return key;
}

__global__ void double_to_float_kernel(
    const double* __restrict__ input,
    float* __restrict__ output,
    size_t count)
{
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        output[index] = static_cast<float>(input[index]);
}

void HelmholtzFMM::init(const double* targets, int n_tgt,
                          const double* sources, int n_src,
                          cdouble k_val, int digits, int max_leaf,
                          int near_radius,
                          bool request_batch4_workspace,
                          bool request_vector_pair)
{
    if (initialized)
        cleanup();
    transfer_cache.clear();
    m2l_batches.clear();
    m2m_data.clear();
    l2l_data.clear();
    m2l_level_info.clear();
    m2m_level_info.clear();
    l2l_level_info.clear();
    leaf_info.clear();
    h_leaf_indices.clear();
    h_tgt_id_offsets.clear();
    h_src_id_offsets.clear();
    h_tgt_ids_flat.clear();
    h_src_ids_flat.clear();
    h_node_centers.clear();

    Timer timer;
    k = k_val;
    Nt = n_tgt;
    Ns = n_src;

    // Build octree (uses combined point set internally)
    tree.build(
        targets, n_tgt, sources, n_src, max_leaf, near_radius);
    n_nodes = (int)tree.nodes.size();

    // Sphere quadrature
    double leaf_hs = tree.nodes[tree.leaves[0]].half_size;
    double leaf_box_size = 2.0 * leaf_hs;
    double order_box_size = leaf_box_size;
    const int order_reference_depth = bem_env_int(
        "BEM_FMM_ORDER_REFERENCE_DEPTH", 0);
    if (order_reference_depth > 0 &&
        order_reference_depth <= 6) {
        const double root_box_size =
            2.0 * tree.nodes.front().half_size;
        const double reference_box_size =
            root_box_size /
            static_cast<double>(1 << order_reference_depth);
        order_box_size =
            std::max(order_box_size, reference_box_size);
        if (order_box_size > leaf_box_size * (1.0 + 1.0e-12)) {
            std::printf(
                "  [FMM] depth-%d order floor: leaf %.6g -> %.6g\n",
                order_reference_depth,
                leaf_box_size, order_box_size);
        }
    }
    p = fmm_truncation_order(std::abs(k), order_box_size, digits);
    squad.init(p);
    L = squad.L;

    // Build leaf info with ORIGINAL IDs
    leaf_info.clear();
    for (int li : tree.leaves) {
        LeafInfo info;
        info.node_idx = li;
        info.tgt_sorted_start = 0; info.tgt_count = 0;
        info.src_sorted_start = 0; info.src_count = 0;

        const OctreeNode& leaf = tree.nodes[li];
        for (int i = leaf.pt_start; i < leaf.pt_start + leaf.pt_count; i++) {
            int orig = tree.sorted_idx[i];
            if (orig < Nt)
                info.tgt_count++;
            else
                info.src_count++;
        }
        leaf_info.push_back(info);
    }

    // Build per-leaf original ID arrays
    // tgt_ids_flat: for each leaf, list of original target IDs
    // src_ids_flat: for each leaf, list of original source IDs (0-based, i.e. orig - Nt)
    std::vector<int> tgt_ids_flat, src_ids_flat;
    std::vector<int> tgt_id_offsets_h, src_id_offsets_h;
    int n_leaves = (int)leaf_info.size();

    tgt_id_offsets_h.resize(n_leaves + 1, 0);
    src_id_offsets_h.resize(n_leaves + 1, 0);

    for (int li_idx = 0; li_idx < n_leaves; li_idx++) {
        int node_idx = leaf_info[li_idx].node_idx;
        const OctreeNode& leaf = tree.nodes[node_idx];

        tgt_id_offsets_h[li_idx] = (int)tgt_ids_flat.size();
        src_id_offsets_h[li_idx] = (int)src_ids_flat.size();

        for (int i = leaf.pt_start; i < leaf.pt_start + leaf.pt_count; i++) {
            int orig = tree.sorted_idx[i];
            if (orig < Nt)
                tgt_ids_flat.push_back(orig);
            else
                src_ids_flat.push_back(orig - Nt);
        }
    }
    tgt_id_offsets_h[n_leaves] = (int)tgt_ids_flat.size();
    src_id_offsets_h[n_leaves] = (int)src_ids_flat.size();

    // Near-field P2P is evaluated from compressed leaf-neighbor lists.
    // Do not materialize per-target source CSR: for ref6 this scales as
    // O(N_targets * sources_per_neighbor_leaf) and can exceed host memory.
    p2p_offsets.clear();
    p2p_indices.clear();
    p2p_nnz = 0;

    // Precompute M2L transfer functions
    printf("  [FMM] Precomputing M2L transfers...\n");
    std::map<DisplacementKey, int> transfer_map;
    double eps = leaf_hs * 1e-8;
    transfer_cache.clear();
    n_unique_transfers = 0;
    m2l_batches.resize(tree.max_level + 1);
    int total_m2l = 0;

    for (int level = 1; level <= tree.max_level; level++) {
        M2LBatch& batch = m2l_batches[level];
        batch.n_pairs = 0;

        for (int ni : tree.level_nodes[level]) {
            const OctreeNode& node = tree.nodes[ni];
            for (int fi = node.far_start; fi < node.far_start + node.far_count; fi++) {
                int far_ni = tree.far_list[fi];
                double d[3] = {
                    node.center[0] - tree.nodes[far_ni].center[0],
                    node.center[1] - tree.nodes[far_ni].center[1],
                    node.center[2] - tree.nodes[far_ni].center[2]
                };
                DisplacementKey key = make_key(d, eps);
                int tidx;
                auto it = transfer_map.find(key);
                if (it == transfer_map.end()) {
                    tidx = n_unique_transfers++;
                    transfer_map[key] = tidx;

                    double d_norm = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
                    const cldouble kd(
                        (long double)k.real() * d_norm,
                        (long double)k.imag() * d_norm);
                    double d_hat[3] = {d[0]/d_norm, d[1]/d_norm, d[2]/d_norm};

                    // The radial factors depend only on |d| and l. Computing
                    // spherical Hankel functions inside the angular loop made
                    // M2L setup O(n_unique * L * p) special-function calls.
                    std::vector<cldouble> radial_coeff(p + 1);
                    cldouble i_pow(1.0L, 0.0L);
                    for (int l = 0; l <= p; l++) {
                        if (l > 0)
                            i_pow *= cldouble(0.0L, 1.0L);
                        radial_coeff[l] =
                            (2.0L * l + 1.0L) * i_pow *
                            spherical_hankel1_extended(l, kd);
                    }
                    std::vector<cdouble> T(L);
                    for (int ll = 0; ll < L; ll++) {
                        const long double cos_angle =
                            (long double)squad.dirs[ll*3]*d_hat[0] +
                            (long double)squad.dirs[ll*3+1]*d_hat[1] +
                            (long double)squad.dirs[ll*3+2]*d_hat[2];
                        cldouble sum(0.0L, 0.0L);
                        long double P_prev = 1.0L;
                        long double P_curr = cos_angle;
                        sum += radial_coeff[0] * P_prev;
                        if (p >= 1)
                            sum += radial_coeff[1] * P_curr;
                        for (int l = 2; l <= p; l++) {
                            const long double P_next =
                                ((2.0L*l - 1.0L) * cos_angle * P_curr -
                                 (l - 1.0L) * P_prev) / (long double)l;
                            sum += radial_coeff[l] * P_next;
                            P_prev = P_curr;
                            P_curr = P_next;
                        }
                        T[ll] = cdouble(
                            (double)sum.real(), (double)sum.imag());
                    }
                    for (int ll = 0; ll < L; ll++)
                        transfer_cache.push_back(T[ll]);
                } else {
                    tidx = it->second;
                }
                batch.tgt_idx.push_back(ni);
                batch.src_idx.push_back(far_ni);
                batch.transfer_idx.push_back(tidx);
                batch.n_pairs++;
                total_m2l++;
            }
        }
    }
    printf("  [FMM] %d unique transfers, %d total M2L pairs\n", n_unique_transfers, total_m2l);

    // Precompute M2M shifts
    m2m_data.resize(tree.max_level + 1);
    for (int level = tree.max_level - 1; level >= 1; level--) {
        LevelShifts& data = m2m_data[level];
        for (int ni : tree.level_nodes[level]) {
            const OctreeNode& node = tree.nodes[ni];
            if (node.is_leaf) continue;
            for (int o = 0; o < 8; o++) {
                if (node.children[o] < 0) continue;
                int ci = node.children[o];
                data.pairs.push_back({ni, ci});
                const OctreeNode& child = tree.nodes[ci];
                double t[3] = {child.center[0]-node.center[0], child.center[1]-node.center[1], child.center[2]-node.center[2]};
                for (int l = 0; l < L; l++) {
                    double dot = squad.dirs[l*3]*t[0] + squad.dirs[l*3+1]*t[1] + squad.dirs[l*3+2]*t[2];
                    data.shifts.push_back(std::exp(cdouble(0, -1) * k * dot));
                }
            }
        }
    }

    // Precompute L2L shifts
    l2l_data.resize(tree.max_level + 1);
    for (int level = 2; level <= tree.max_level; level++) {
        LevelShifts& data = l2l_data[level];
        for (int ni : tree.level_nodes[level]) {
            const OctreeNode& node = tree.nodes[ni];
            if (node.parent < 0) continue;
            data.pairs.push_back({node.parent, ni});
            double t[3] = {node.center[0]-tree.nodes[node.parent].center[0],
                           node.center[1]-tree.nodes[node.parent].center[1],
                           node.center[2]-tree.nodes[node.parent].center[2]};
            for (int l = 0; l < L; l++) {
                double dot = squad.dirs[l*3]*t[0] + squad.dirs[l*3+1]*t[1] + squad.dirs[l*3+2]*t[2];
                data.shifts.push_back(std::exp(cdouble(0, 1) * k * dot));
            }
        }
    }

    // ============================================================
    // Upload everything to GPU
    // ============================================================
    printf("  [FMM] Uploading to GPU...\n");

    // Target and source positions in ORIGINAL order
    CUDA_CHECK(cudaMalloc(&d_tgt_pts, Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_src_pts, Ns * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_tgt_pts, targets, Nt * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_pts, sources, Ns * 3 * sizeof(double), cudaMemcpyHostToDevice));
    d_tgt_pts_fp32 = nullptr;
    d_src_pts_fp32 = nullptr;
    if (request_vector_pair) {
        std::vector<float> target_points_fp32((size_t)Nt * 3);
        std::vector<float> source_points_fp32((size_t)Ns * 3);
        for (size_t index = 0; index < target_points_fp32.size(); index++)
            target_points_fp32[index] =
                static_cast<float>(targets[index]);
        for (size_t index = 0; index < source_points_fp32.size(); index++)
            source_points_fp32[index] =
                static_cast<float>(sources[index]);
        CUDA_CHECK(cudaMalloc(
            &d_tgt_pts_fp32,
            target_points_fp32.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            &d_src_pts_fp32,
            source_points_fp32.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(
            d_tgt_pts_fp32, target_points_fp32.data(),
            target_points_fp32.size() * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_src_pts_fp32, source_points_fp32.data(),
            source_points_fp32.size() * sizeof(float),
            cudaMemcpyHostToDevice));
    }

    d_p2p_offsets = nullptr;
    d_p2p_indices = nullptr;

    // FMM arrays
    CUDA_CHECK(cudaMalloc(&d_multi_re, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_multi_im, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_re, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_im, n_nodes * L * sizeof(double)));

    // Transfers
    if (n_unique_transfers > 0) {
        std::vector<double> t_re(n_unique_transfers * L), t_im(n_unique_transfers * L);
        for (int i = 0; i < n_unique_transfers * L; i++) {
            t_re[i] = transfer_cache[i].real();
            t_im[i] = transfer_cache[i].imag();
        }
        CUDA_CHECK(cudaMalloc(&d_transfer_re, n_unique_transfers * L * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_transfer_im, n_unique_transfers * L * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_transfer_re, t_re.data(), n_unique_transfers * L * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_transfer_im, t_im.data(), n_unique_transfers * L * sizeof(double), cudaMemcpyHostToDevice));
    }

    // M2L batch (concat levels)
    {
        std::vector<int> all_tgt, all_src, all_tidx;
        std::vector<int> all_row_target;
        std::vector<int> all_row_start;
        std::vector<int> all_row_end;
        m2l_level_info.resize(tree.max_level + 1);
        m2l_row_level_info.resize(tree.max_level + 1);
        for (int level = 0; level <= tree.max_level; level++) {
            m2l_level_info[level].offset = (int)all_tgt.size();
            m2l_level_info[level].count = 0;
            m2l_row_level_info[level].offset =
                (int)all_row_target.size();
            m2l_row_level_info[level].count = 0;
            if (level < (int)m2l_batches.size() && m2l_batches[level].n_pairs > 0) {
                const M2LBatch& b = m2l_batches[level];
                const int pair_offset = (int)all_tgt.size();
                all_tgt.insert(all_tgt.end(), b.tgt_idx.begin(), b.tgt_idx.end());
                all_src.insert(all_src.end(), b.src_idx.begin(), b.src_idx.end());
                all_tidx.insert(all_tidx.end(), b.transfer_idx.begin(), b.transfer_idx.end());
                m2l_level_info[level].count = b.n_pairs;
                std::vector<unsigned char> seen_target(n_nodes, 0);
                int previous_target = -1;
                for (int pair = 0; pair < b.n_pairs; pair++) {
                    const int target = b.tgt_idx[pair];
                    if (target == previous_target)
                        continue;
                    if (seen_target[target])
                        throw std::runtime_error(
                            "M2L target interactions are not contiguous");
                    seen_target[target] = 1;
                    if (previous_target >= 0)
                        all_row_end.push_back(pair_offset + pair);
                    all_row_target.push_back(target);
                    all_row_start.push_back(pair_offset + pair);
                    previous_target = target;
                }
                if (previous_target >= 0)
                    all_row_end.push_back(pair_offset + b.n_pairs);
                m2l_row_level_info[level].count =
                    (int)all_row_target.size() -
                    m2l_row_level_info[level].offset;
            }
        }
        if (!all_tgt.empty()) {
            CUDA_CHECK(cudaMalloc(&d_m2l_tgt, all_tgt.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_m2l_src, all_src.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_m2l_tidx, all_tidx.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_m2l_tgt, all_tgt.data(), all_tgt.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_m2l_src, all_src.data(), all_src.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_m2l_tidx, all_tidx.data(), all_tidx.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(
                &d_m2l_row_target,
                all_row_target.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(
                &d_m2l_row_start,
                all_row_start.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(
                &d_m2l_row_end,
                all_row_end.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(
                d_m2l_row_target, all_row_target.data(),
                all_row_target.size() * sizeof(int),
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_m2l_row_start, all_row_start.data(),
                all_row_start.size() * sizeof(int),
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_m2l_row_end, all_row_end.data(),
                all_row_end.size() * sizeof(int),
                cudaMemcpyHostToDevice));
        }
    }

    // M2M shifts (concat)
    {
        std::vector<int> all_p, all_c;
        std::vector<double> all_s_re, all_s_im;
        m2m_level_info.resize(tree.max_level + 1);
        for (int level = 0; level <= tree.max_level; level++) {
            m2m_level_info[level].offset = (int)all_p.size();
            m2m_level_info[level].count = 0;
            if (level < (int)m2m_data.size() && !m2m_data[level].pairs.empty()) {
                const LevelShifts& ls = m2m_data[level];
                for (auto& pr : ls.pairs) { all_p.push_back(pr.parent); all_c.push_back(pr.child); }
                for (auto& s : ls.shifts) { all_s_re.push_back(s.real()); all_s_im.push_back(s.imag()); }
                m2m_level_info[level].count = (int)ls.pairs.size();
            }
        }
        if (!all_p.empty()) {
            CUDA_CHECK(cudaMalloc(&d_m2m_parent, all_p.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_m2m_child, all_c.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_m2m_shift_re, all_s_re.size() * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_m2m_shift_im, all_s_im.size() * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_m2m_parent, all_p.data(), all_p.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_m2m_child, all_c.data(), all_c.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_m2m_shift_re, all_s_re.data(), all_s_re.size() * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_m2m_shift_im, all_s_im.data(), all_s_im.size() * sizeof(double), cudaMemcpyHostToDevice));
        }
    }

    // L2L shifts (concat)
    {
        std::vector<int> all_p, all_c;
        std::vector<double> all_s_re, all_s_im;
        l2l_level_info.resize(tree.max_level + 1);
        for (int level = 0; level <= tree.max_level; level++) {
            l2l_level_info[level].offset = (int)all_p.size();
            l2l_level_info[level].count = 0;
            if (level < (int)l2l_data.size() && !l2l_data[level].pairs.empty()) {
                const LevelShifts& ls = l2l_data[level];
                for (auto& pr : ls.pairs) { all_p.push_back(pr.parent); all_c.push_back(pr.child); }
                for (auto& s : ls.shifts) { all_s_re.push_back(s.real()); all_s_im.push_back(s.imag()); }
                l2l_level_info[level].count = (int)ls.pairs.size();
            }
        }
        if (!all_p.empty()) {
            CUDA_CHECK(cudaMalloc(&d_l2l_parent, all_p.data() ? all_p.size() * sizeof(int) : sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_l2l_child, all_c.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_l2l_shift_re, all_s_re.size() * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_l2l_shift_im, all_s_im.size() * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_l2l_parent, all_p.data(), all_p.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_l2l_child, all_c.data(), all_c.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_l2l_shift_re, all_s_re.data(), all_s_re.size() * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_l2l_shift_im, all_s_im.data(), all_s_im.size() * sizeof(double), cudaMemcpyHostToDevice));
        }
    }

    // Charge/result buffers
    CUDA_CHECK(cudaMalloc(&d_charges_re, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_charges_im, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result_re, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result_im, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_re, Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_im, Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hess_re, Nt * 6 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hess_im, Nt * 6 * sizeof(double)));

    // Batch-2 workspace buffers
    CUDA_CHECK(cudaMalloc(&d_charges2_re, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_charges2_im, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result2_re, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result2_im, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad2_re, Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad2_im, Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_multi2_re, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_multi2_im, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local2_re, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local2_im, n_nodes * L * sizeof(double)));
    bool request_batch4 =
        request_batch4_workspace ||
        bem_env_flag_enabled("BEM_FMM_ALLOC_BATCH4") ||
        bem_env_flag_enabled("BEM_FMM_BATCH4");
    bool alloc_batch4 = request_batch4;
    batch4_allocated = false;
    auto free_batch4_workspace = [&]() {
        cudaFree(d_charges3_re); d_charges3_re = nullptr;
        cudaFree(d_charges3_im); d_charges3_im = nullptr;
        cudaFree(d_charges4_re); d_charges4_re = nullptr;
        cudaFree(d_charges4_im); d_charges4_im = nullptr;
        cudaFree(d_result3_re); d_result3_re = nullptr;
        cudaFree(d_result3_im); d_result3_im = nullptr;
        cudaFree(d_result4_re); d_result4_re = nullptr;
        cudaFree(d_result4_im); d_result4_im = nullptr;
        cudaFree(d_grad3_re); d_grad3_re = nullptr;
        cudaFree(d_grad3_im); d_grad3_im = nullptr;
        cudaFree(d_grad4_re); d_grad4_re = nullptr;
        cudaFree(d_grad4_im); d_grad4_im = nullptr;
        cudaFree(d_hess2_re); d_hess2_re = nullptr;
        cudaFree(d_hess2_im); d_hess2_im = nullptr;
        cudaFree(d_hess3_re); d_hess3_re = nullptr;
        cudaFree(d_hess3_im); d_hess3_im = nullptr;
        cudaFree(d_hess4_re); d_hess4_re = nullptr;
        cudaFree(d_hess4_im); d_hess4_im = nullptr;
        cudaFree(d_multi3_re); d_multi3_re = nullptr;
        cudaFree(d_multi3_im); d_multi3_im = nullptr;
        cudaFree(d_multi4_re); d_multi4_re = nullptr;
        cudaFree(d_multi4_im); d_multi4_im = nullptr;
        cudaFree(d_local3_re); d_local3_re = nullptr;
        cudaFree(d_local3_im); d_local3_im = nullptr;
        cudaFree(d_local4_re); d_local4_re = nullptr;
        cudaFree(d_local4_im); d_local4_im = nullptr;
        cudaFree(d_gy3_re_cached); d_gy3_re_cached = nullptr;
        cudaFree(d_gy3_im_cached); d_gy3_im_cached = nullptr;
        cudaFree(d_gz3_re_cached); d_gz3_re_cached = nullptr;
        cudaFree(d_gz3_im_cached); d_gz3_im_cached = nullptr;
        cudaFree(d_gx3_re_tmp_cached); d_gx3_re_tmp_cached = nullptr;
        cudaFree(d_gx3_im_tmp_cached); d_gx3_im_tmp_cached = nullptr;
        cudaFree(d_gy4_re_cached); d_gy4_re_cached = nullptr;
        cudaFree(d_gy4_im_cached); d_gy4_im_cached = nullptr;
        cudaFree(d_gz4_re_cached); d_gz4_re_cached = nullptr;
        cudaFree(d_gz4_im_cached); d_gz4_im_cached = nullptr;
        cudaFree(d_gx4_re_tmp_cached); d_gx4_re_tmp_cached = nullptr;
        cudaFree(d_gx4_im_tmp_cached); d_gx4_im_tmp_cached = nullptr;
        cudaGetLastError();
    };
    auto try_malloc_double = [&](double** ptr, size_t count, const char* name) {
        cudaError_t err = cudaMalloc(ptr, count * sizeof(double));
        if (err != cudaSuccess) {
            fprintf(stderr, "  [FMM] batch4 workspace unavailable at %s (%s); falling back to batch2\n",
                    name, cudaGetErrorString(err));
            *ptr = nullptr;
            cudaGetLastError();
            return false;
        }
        return true;
    };
    if (alloc_batch4) {
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_charges3_re, Ns, "charges3_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_charges3_im, Ns, "charges3_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_charges4_re, Ns, "charges4_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_charges4_im, Ns, "charges4_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_result3_re, Nt, "result3_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_result3_im, Nt, "result3_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_result4_re, Nt, "result4_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_result4_im, Nt, "result4_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_grad3_re, (size_t)Nt * 3, "grad3_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_grad3_im, (size_t)Nt * 3, "grad3_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_grad4_re, (size_t)Nt * 3, "grad4_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_grad4_im, (size_t)Nt * 3, "grad4_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_hess2_re, (size_t)Nt * 6, "hess2_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_hess2_im, (size_t)Nt * 6, "hess2_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_hess3_re, (size_t)Nt * 6, "hess3_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_hess3_im, (size_t)Nt * 6, "hess3_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_hess4_re, (size_t)Nt * 6, "hess4_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_hess4_im, (size_t)Nt * 6, "hess4_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_multi3_re, (size_t)n_nodes * L, "multi3_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_multi3_im, (size_t)n_nodes * L, "multi3_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_multi4_re, (size_t)n_nodes * L, "multi4_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_multi4_im, (size_t)n_nodes * L, "multi4_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_local3_re, (size_t)n_nodes * L, "local3_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_local3_im, (size_t)n_nodes * L, "local3_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_local4_re, (size_t)n_nodes * L, "local4_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_local4_im, (size_t)n_nodes * L, "local4_im");
        if (!alloc_batch4)
            free_batch4_workspace();
    }

    // Store leaf ID arrays on host
    h_leaf_indices.resize(n_leaves);
    for (int i = 0; i < n_leaves; i++) h_leaf_indices[i] = leaf_info[i].node_idx;
    h_tgt_id_offsets = tgt_id_offsets_h;
    h_src_id_offsets = src_id_offsets_h;
    h_tgt_ids_flat = tgt_ids_flat;
    h_src_ids_flat = src_ids_flat;

    // Node centers
    h_node_centers.resize(n_nodes * 3);
    for (int i = 0; i < n_nodes; i++) {
        h_node_centers[i*3]   = tree.nodes[i].center[0];
        h_node_centers[i*3+1] = tree.nodes[i].center[1];
        h_node_centers[i*3+2] = tree.nodes[i].center[2];
    }

    // Cached GPU arrays for run_tree() — allocated once, reused every call
    CUDA_CHECK(cudaMalloc(&d_node_centers_cached, n_nodes * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dirs_cached, L * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_weights_cached, L * sizeof(double)));
    d_dirs_fp32_cached = nullptr;
    d_weights_fp32_cached = nullptr;
    if (request_vector_pair) {
        std::vector<float> directions_fp32((size_t)L * 3);
        std::vector<float> weights_fp32(L);
        for (size_t index = 0; index < directions_fp32.size(); index++)
            directions_fp32[index] =
                static_cast<float>(squad.dirs[index]);
        for (int index = 0; index < L; index++)
            weights_fp32[index] =
                static_cast<float>(squad.weights[index]);
        CUDA_CHECK(cudaMalloc(
            &d_dirs_fp32_cached,
            directions_fp32.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            &d_weights_fp32_cached,
            weights_fp32.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(
            d_dirs_fp32_cached, directions_fp32.data(),
            directions_fp32.size() * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_weights_fp32_cached, weights_fp32.data(),
            weights_fp32.size() * sizeof(float),
            cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_leaf_idx_cached, n_leaves * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_src_id_offsets_cached, (n_leaves + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_src_ids_cached, std::max((int)h_src_ids_flat.size(), 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tgt_id_offsets_cached, (n_leaves + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tgt_ids_cached, std::max((int)h_tgt_ids_flat.size(), 1) * sizeof(int)));
    std::vector<int> node_to_leaf(n_nodes, -1);
    for (int i = 0; i < n_leaves; i++) node_to_leaf[h_leaf_indices[i]] = i;
    std::vector<int> leaf_near_offsets(n_leaves + 1, 0);
    std::vector<int> leaf_near_ids;
    for (int li_idx = 0; li_idx < n_leaves; li_idx++) {
        int node_idx = h_leaf_indices[li_idx];
        const OctreeNode& leaf = tree.nodes[node_idx];
        leaf_near_offsets[li_idx] = (int)leaf_near_ids.size();
        for (int ni = leaf.near_start; ni < leaf.near_start + leaf.near_count; ni++) {
            int nb_leaf = node_to_leaf[tree.near_list[ni]];
            if (nb_leaf >= 0)
                leaf_near_ids.push_back(nb_leaf);
        }
    }
    leaf_near_offsets[n_leaves] = (int)leaf_near_ids.size();
    CUDA_CHECK(cudaMalloc(&d_leaf_near_offsets_cached, (n_leaves + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_leaf_near_ids_cached, std::max((int)leaf_near_ids.size(), 1) * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_node_centers_cached, h_node_centers.data(), n_nodes * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dirs_cached, squad.dirs.data(), L * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights_cached, squad.weights.data(), L * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_idx_cached, h_leaf_indices.data(), n_leaves * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_id_offsets_cached, h_src_id_offsets.data(), (n_leaves + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (!h_src_ids_flat.empty())
        CUDA_CHECK(cudaMemcpy(d_src_ids_cached, h_src_ids_flat.data(), h_src_ids_flat.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt_id_offsets_cached, h_tgt_id_offsets.data(), (n_leaves + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (!h_tgt_ids_flat.empty())
        CUDA_CHECK(cudaMemcpy(d_tgt_ids_cached, h_tgt_ids_flat.data(), h_tgt_ids_flat.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_near_offsets_cached, leaf_near_offsets.data(), (n_leaves + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (!leaf_near_ids.empty())
        CUDA_CHECK(cudaMemcpy(d_leaf_near_ids_cached, leaf_near_ids.data(), leaf_near_ids.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Cached gradient workspace arrays
    CUDA_CHECK(cudaMalloc(&d_gy_re_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gy_im_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gz_re_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gz_im_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gx_re_tmp_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gx_im_tmp_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gy2_re_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gy2_im_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gz2_re_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gz2_im_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gx2_re_tmp_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gx2_im_tmp_cached, Nt * sizeof(double)));
    if (alloc_batch4) {
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_gy3_re_cached, Nt, "gy3_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_gy3_im_cached, Nt, "gy3_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_gz3_re_cached, Nt, "gz3_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_gz3_im_cached, Nt, "gz3_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_gx3_re_tmp_cached, Nt, "gx3_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_gx3_im_tmp_cached, Nt, "gx3_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_gy4_re_cached, Nt, "gy4_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_gy4_im_cached, Nt, "gy4_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_gz4_re_cached, Nt, "gz4_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_gz4_im_cached, Nt, "gz4_im");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_gx4_re_tmp_cached, Nt, "gx4_re");
        alloc_batch4 = alloc_batch4 && try_malloc_double(&d_gx4_im_tmp_cached, Nt, "gx4_im");
        if (!alloc_batch4)
            free_batch4_workspace();
    }
    batch4_allocated = alloc_batch4;
    pair_l2p_allocated = false;
    bool local_storage_fp32_default = false;
    bool local_m2l_fp32_default = false;
    bool multi_storage_fp32_default = false;
#ifdef BEM_DEFAULT_FMM_NEAR_FP32
    local_storage_fp32_default = true;
    local_m2l_fp32_default = true;
    multi_storage_fp32_default = true;
#endif
    const bool request_local_storage_fp32 =
        bem_env_flag_enabled(
            "BEM_FMM_LOCAL_STORAGE_FP32",
            local_storage_fp32_default) &&
        bem_env_flag_enabled(
            "BEM_FMM_M2L_FP32",
            local_m2l_fp32_default);
    const bool request_multi_storage_fp32 =
        bem_env_flag_enabled(
            "BEM_FMM_MULTI_STORAGE_FP32",
            multi_storage_fp32_default) &&
        bem_env_flag_enabled(
            "BEM_FMM_M2L_FP32",
            local_m2l_fp32_default);
    if (request_vector_pair && batch4_allocated) {
        const bool request_four_field =
            bem_env_flag_enabled("BEM_FMM_FOUR_FIELD", false);
        pair_workspace_fields = request_four_field ? 12 : 6;
        size_t pair_field_count =
            (size_t)pair_workspace_fields * n_nodes * L;
        cudaError_t pair_error = cudaSuccess;
        if (request_multi_storage_fp32) {
            pair_error = cudaMalloc(
                &d_pair_multi_re_fp32,
                pair_field_count * sizeof(float));
            if (pair_error == cudaSuccess) {
                pair_error = cudaMalloc(
                    &d_pair_multi_im_fp32,
                    pair_field_count * sizeof(float));
            }
        } else {
            pair_error = cudaMalloc(
                &d_pair_multi_re,
                pair_field_count * sizeof(double));
            if (pair_error == cudaSuccess) {
                pair_error = cudaMalloc(
                    &d_pair_multi_im,
                    pair_field_count * sizeof(double));
            }
        }
        if (pair_error != cudaSuccess &&
            request_multi_storage_fp32) {
            cudaFree(d_pair_multi_re_fp32);
            cudaFree(d_pair_multi_im_fp32);
            d_pair_multi_re_fp32 = nullptr;
            d_pair_multi_im_fp32 = nullptr;
            cudaGetLastError();
            pair_error = cudaMalloc(
                &d_pair_multi_re,
                pair_field_count * sizeof(double));
            if (pair_error == cudaSuccess) {
                pair_error = cudaMalloc(
                    &d_pair_multi_im,
                    pair_field_count * sizeof(double));
            }
            if (pair_error == cudaSuccess) {
                std::fprintf(
                    stderr,
                    "  [FMM] FP32 multipole storage unavailable; "
                    "using FP64 multipole storage\n");
            }
        }
        if (pair_error == cudaSuccess) {
            if (request_local_storage_fp32) {
                pair_error = cudaMalloc(
                    &d_pair_local_re_fp32,
                    pair_field_count * sizeof(float));
                if (pair_error == cudaSuccess) {
                    pair_error = cudaMalloc(
                        &d_pair_local_im_fp32,
                        pair_field_count * sizeof(float));
                }
            } else {
                pair_error = cudaMalloc(
                    &d_pair_local_re,
                    pair_field_count * sizeof(double));
                if (pair_error == cudaSuccess) {
                    pair_error = cudaMalloc(
                        &d_pair_local_im,
                        pair_field_count * sizeof(double));
                }
            }
        }
        if (pair_error != cudaSuccess &&
            request_local_storage_fp32) {
            cudaFree(d_pair_local_re_fp32);
            cudaFree(d_pair_local_im_fp32);
            d_pair_local_re_fp32 = nullptr;
            d_pair_local_im_fp32 = nullptr;
            cudaGetLastError();
            pair_error = cudaMalloc(
                &d_pair_local_re,
                pair_field_count * sizeof(double));
            if (pair_error == cudaSuccess) {
                pair_error = cudaMalloc(
                    &d_pair_local_im,
                    pair_field_count * sizeof(double));
            }
            if (pair_error == cudaSuccess) {
                std::fprintf(
                    stderr,
                    "  [FMM] FP32 local storage unavailable; "
                    "using FP64 local storage\n");
            }
        }
        if (pair_error != cudaSuccess && pair_workspace_fields == 12) {
            std::fprintf(
                stderr,
                "  [FMM] four-field workspace unavailable (%s); "
                "retrying the two-field workspace\n",
                cudaGetErrorString(pair_error));
            cudaFree(d_pair_multi_re);
            cudaFree(d_pair_multi_im);
            cudaFree(d_pair_multi_re_fp32);
            cudaFree(d_pair_multi_im_fp32);
            cudaFree(d_pair_local_re);
            cudaFree(d_pair_local_im);
            cudaFree(d_pair_local_re_fp32);
            cudaFree(d_pair_local_im_fp32);
            d_pair_multi_re = nullptr;
            d_pair_multi_im = nullptr;
            d_pair_multi_re_fp32 = nullptr;
            d_pair_multi_im_fp32 = nullptr;
            d_pair_local_re = nullptr;
            d_pair_local_im = nullptr;
            d_pair_local_re_fp32 = nullptr;
            d_pair_local_im_fp32 = nullptr;
            cudaGetLastError();
            pair_workspace_fields = 6;
            pair_field_count = (size_t)6 * n_nodes * L;
            if (request_multi_storage_fp32) {
                pair_error = cudaMalloc(
                    &d_pair_multi_re_fp32,
                    pair_field_count * sizeof(float));
                if (pair_error == cudaSuccess)
                    pair_error = cudaMalloc(
                        &d_pair_multi_im_fp32,
                        pair_field_count * sizeof(float));
            } else {
                pair_error = cudaMalloc(
                    &d_pair_multi_re,
                    pair_field_count * sizeof(double));
                if (pair_error == cudaSuccess)
                    pair_error = cudaMalloc(
                        &d_pair_multi_im,
                        pair_field_count * sizeof(double));
            }
            if (pair_error == cudaSuccess) {
                if (request_local_storage_fp32) {
                    pair_error = cudaMalloc(
                        &d_pair_local_re_fp32,
                        pair_field_count * sizeof(float));
                    if (pair_error == cudaSuccess)
                        pair_error = cudaMalloc(
                            &d_pair_local_im_fp32,
                            pair_field_count * sizeof(float));
                } else {
                    pair_error = cudaMalloc(
                        &d_pair_local_re,
                        pair_field_count * sizeof(double));
                    if (pair_error == cudaSuccess)
                        pair_error = cudaMalloc(
                            &d_pair_local_im,
                            pair_field_count * sizeof(double));
                }
            }
        }
        if (pair_error != cudaSuccess) {
            std::fprintf(
                stderr,
                "  [FMM] paired far/L2P workspace unavailable (%s); "
                "sharing P2P only\n",
                cudaGetErrorString(pair_error));
            cudaFree(d_pair_multi_re);
            cudaFree(d_pair_multi_im);
            cudaFree(d_pair_multi_re_fp32);
            cudaFree(d_pair_multi_im_fp32);
            cudaFree(d_pair_local_re);
            cudaFree(d_pair_local_im);
            cudaFree(d_pair_local_re_fp32);
            cudaFree(d_pair_local_im_fp32);
            d_pair_multi_re = nullptr;
            d_pair_multi_im = nullptr;
            d_pair_multi_re_fp32 = nullptr;
            d_pair_multi_im_fp32 = nullptr;
            d_pair_local_re = nullptr;
            d_pair_local_im = nullptr;
            d_pair_local_re_fp32 = nullptr;
            d_pair_local_im_fp32 = nullptr;
            pair_workspace_fields = 0;
            cudaGetLastError();
        } else {
            pair_l2p_allocated = true;
        }
    }
    if (pair_l2p_allocated &&
        bem_env_flag_enabled("BEM_MIXED_ITERATIVE_REFINEMENT")) {
        const size_t strict_count =
            static_cast<size_t>(6) * n_nodes * L;
        const size_t strict_bytes = strict_count * sizeof(double);
        cudaError_t strict_error = cudaMalloc(
            &d_strict_pair_multi_re, strict_bytes);
        if (strict_error == cudaSuccess)
            strict_error = cudaMalloc(
                &d_strict_pair_multi_im, strict_bytes);
        if (strict_error == cudaSuccess)
            strict_error = cudaMalloc(
                &d_strict_pair_local_re, strict_bytes);
        if (strict_error == cudaSuccess)
            strict_error = cudaMalloc(
                &d_strict_pair_local_im, strict_bytes);
        if (strict_error != cudaSuccess) {
            std::fprintf(
                stderr,
                "  [FMM] strict paired FP64 workspace unavailable (%s); "
                "using sequential strict residuals\n",
                cudaGetErrorString(strict_error));
            cudaFree(d_strict_pair_multi_re);
            cudaFree(d_strict_pair_multi_im);
            cudaFree(d_strict_pair_local_re);
            cudaFree(d_strict_pair_local_im);
            d_strict_pair_multi_re = nullptr;
            d_strict_pair_multi_im = nullptr;
            d_strict_pair_local_re = nullptr;
            d_strict_pair_local_im = nullptr;
            cudaGetLastError();
        }
    }
    bool m2l_storage_fp32_default = false;
#ifdef BEM_DEFAULT_FMM_NEAR_FP32
    m2l_storage_fp32_default = true;
#endif
    const bool request_m2l_storage_fp32 =
        pair_l2p_allocated &&
        bem_env_flag_enabled(
            "BEM_FMM_M2L_STORAGE_FP32",
            m2l_storage_fp32_default) &&
        bem_env_flag_enabled(
            "BEM_FMM_M2L_FP32",
            local_m2l_fp32_default);
    if (request_m2l_storage_fp32) {
        const size_t transfer_count =
            static_cast<size_t>(n_unique_transfers) * L;
        const size_t pair_field_count =
            static_cast<size_t>(pair_workspace_fields) * n_nodes * L;
        cudaError_t storage_error = cudaMalloc(
            &d_transfer_re_fp32,
            transfer_count * sizeof(float));
        if (storage_error == cudaSuccess) {
            storage_error = cudaMalloc(
                &d_transfer_im_fp32,
                transfer_count * sizeof(float));
        }
        const bool primary_multipoles_fp32 =
            d_pair_multi_re == nullptr &&
            d_pair_multi_im == nullptr &&
            d_pair_multi_re_fp32 != nullptr &&
            d_pair_multi_im_fp32 != nullptr;
        if (storage_error == cudaSuccess &&
            d_pair_multi_re_fp32 == nullptr) {
            storage_error = cudaMalloc(
                &d_pair_multi_re_fp32,
                pair_field_count * sizeof(float));
        }
        if (storage_error == cudaSuccess &&
            d_pair_multi_im_fp32 == nullptr) {
            storage_error = cudaMalloc(
                &d_pair_multi_im_fp32,
                pair_field_count * sizeof(float));
        }
        if (storage_error != cudaSuccess) {
            std::fprintf(
                stderr,
                "  [FMM] FP32 M2L storage unavailable (%s); "
                "using FP64 storage\n",
            cudaGetErrorString(storage_error));
            cudaFree(d_transfer_re_fp32);
            cudaFree(d_transfer_im_fp32);
            d_transfer_re_fp32 = nullptr;
            d_transfer_im_fp32 = nullptr;
            if (!primary_multipoles_fp32) {
                cudaFree(d_pair_multi_re_fp32);
                cudaFree(d_pair_multi_im_fp32);
                d_pair_multi_re_fp32 = nullptr;
                d_pair_multi_im_fp32 = nullptr;
            }
            cudaGetLastError();
        } else if (transfer_count > 0) {
            const int block = 256;
            const int grid = static_cast<int>(
                (transfer_count + block - 1) / block);
            double_to_float_kernel<<<grid, block>>>(
                d_transfer_re, d_transfer_re_fp32,
                transfer_count);
            double_to_float_kernel<<<grid, block>>>(
                d_transfer_im, d_transfer_im_fp32,
                transfer_count);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    d_pair_charges_fp32 = nullptr;
    if (request_vector_pair) {
        cudaError_t charge_error = cudaMalloc(
            &d_pair_charges_fp32,
            (size_t)2 * std::max(pair_workspace_fields, 6) * Ns *
                sizeof(float));
        if (charge_error != cudaSuccess) {
            std::fprintf(
                stderr,
                "  [FMM] FP32 paired-charge cache unavailable (%s); "
                "using FP64 source storage in P2P\n",
                cudaGetErrorString(charge_error));
            d_pair_charges_fp32 = nullptr;
            cudaGetLastError();
        }
    }

    int complex_tmp_n = std::max(Ns, 6 * Nt);
    CUDA_CHECK(cudaMalloc(&d_complex_tmp1, complex_tmp_n * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_complex_tmp2, complex_tmp_n * sizeof(double2)));

    d_phase_cache_fp32 = nullptr;
    d_l2p_phase_cache_fp32 = nullptr;
    bool matching_leaf_points =
        Nt == Ns && k.imag() == 0.0 && Nt > 0 &&
        std::memcmp(
            targets, sources,
            (size_t)Nt * 3 * sizeof(double)) == 0;
    if (matching_leaf_points) {
        std::vector<int> target_nodes(Nt, -1);
        std::vector<int> source_nodes(Ns, -1);
        for (int leaf = 0; leaf < n_leaves; leaf++) {
            const int node = h_leaf_indices[leaf];
            for (int index = h_tgt_id_offsets[leaf];
                 index < h_tgt_id_offsets[leaf + 1];
                 index++)
                target_nodes[h_tgt_ids_flat[index]] = node;
            for (int index = h_src_id_offsets[leaf];
                 index < h_src_id_offsets[leaf + 1];
                 index++)
                source_nodes[h_src_ids_flat[index]] = node;
        }
        matching_leaf_points = target_nodes == source_nodes;
    }
    bool phase_cache_default = false;
#ifdef BEM_DEFAULT_FMM_NEAR_FP32
    phase_cache_default = true;
#endif
    const bool request_phase_cache =
        request_vector_pair && matching_leaf_points &&
        bem_env_flag_enabled(
            "BEM_FMM_PHASE_CACHE", phase_cache_default);
    if (request_phase_cache) {
        const size_t phase_bytes =
            (size_t)Nt * L * sizeof(float2);
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const int reserve_mb = std::max(
            512,
            bem_env_int(
                "BEM_FMM_PHASE_CACHE_RESERVE_MB", 6144));
        const size_t reserve_bytes =
            (size_t)reserve_mb * 1024 * 1024;
        if (phase_bytes + reserve_bytes <= free_bytes) {
            const cudaError_t phase_error = cudaMalloc(
                &d_phase_cache_fp32, phase_bytes);
            if (phase_error == cudaSuccess) {
                bool transpose_default = false;
#ifdef BEM_DEFAULT_FMM_NEAR_FP32
                transpose_default = true;
#endif
                const bool request_warp_l2p =
                    bem_env_flag_enabled(
                        "BEM_FMM_L2P_WARP_PER_TARGET", false);
                const bool request_transpose =
                    !request_warp_l2p &&
                    bem_env_flag_enabled(
                        "BEM_FMM_L2P_TRANSPOSED_PHASE_CACHE",
                        transpose_default);
                if (request_transpose &&
                    2 * phase_bytes + reserve_bytes <= free_bytes) {
                    const cudaError_t transpose_error = cudaMalloc(
                        &d_l2p_phase_cache_fp32, phase_bytes);
                    if (transpose_error != cudaSuccess) {
                        d_l2p_phase_cache_fp32 = nullptr;
                        cudaGetLastError();
                    }
                }
                build_leaf_phase_cache_fp32_kernel
                    <<<n_leaves, 256>>>(
                        d_tgt_pts, d_dirs_cached, k.real(),
                        d_phase_cache_fp32,
                        d_l2p_phase_cache_fp32,
                        d_leaf_idx_cached,
                        d_tgt_id_offsets_cached,
                        d_tgt_ids_cached,
                        d_node_centers_cached,
                        L, Nt, n_leaves);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                std::printf(
                    "  [FMM] FP32 phase cache: %.1f MiB%s\n",
                    phase_bytes / (1024.0 * 1024.0),
                    d_l2p_phase_cache_fp32 != nullptr
                        ? " + transposed L2P copy" : "");
            } else {
                d_phase_cache_fp32 = nullptr;
                cudaGetLastError();
            }
        } else {
            std::fprintf(
                stderr,
                "  [FMM] phase cache needs %.1f MiB, only %.1f MiB "
                "free after reserve; using on-the-fly phases\n",
                phase_bytes / (1024.0 * 1024.0),
                free_bytes / (1024.0 * 1024.0));
        }
    }

    d_leaf_near_source_offsets_cached = nullptr;
    d_leaf_near_source_ids_cached = nullptr;
    bool flat_near_sources_default = false;
#ifdef BEM_DEFAULT_FMM_NEAR_FP32
    flat_near_sources_default = true;
#endif
    const bool request_flat_near_sources =
        request_vector_pair &&
        bem_env_flag_enabled(
            "BEM_FMM_FLAT_NEAR_SOURCES",
            flat_near_sources_default);
    if (request_flat_near_sources && n_leaves > 0) {
        std::uint64_t flat_source_count = 0;
        for (int leaf = 0; leaf < n_leaves; leaf++) {
            flat_source_count +=
                static_cast<std::uint64_t>(
                    h_src_id_offsets[leaf + 1] -
                    h_src_id_offsets[leaf]);
            for (int index = leaf_near_offsets[leaf];
                 index < leaf_near_offsets[leaf + 1]; index++) {
                const int source_leaf = leaf_near_ids[index];
                flat_source_count +=
                    static_cast<std::uint64_t>(
                        h_src_id_offsets[source_leaf + 1] -
                        h_src_id_offsets[source_leaf]);
            }
        }
        const size_t maximum_flat_sources =
            static_cast<size_t>(
                std::numeric_limits<int>::max());
        if (flat_source_count <= maximum_flat_sources) {
            const size_t offset_bytes =
                static_cast<size_t>(n_leaves + 1) * sizeof(int);
            const size_t source_bytes =
                static_cast<size_t>(flat_source_count) * sizeof(int);
            const size_t cache_bytes = offset_bytes + source_bytes;
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            const int reserve_mb = std::max(
                512,
                bem_env_int(
                    "BEM_FMM_FLAT_NEAR_RESERVE_MB", 4096));
            const size_t reserve_bytes =
                static_cast<size_t>(reserve_mb) * 1024 * 1024;
            if (cache_bytes + reserve_bytes <= free_bytes) {
                std::vector<int> flat_offsets(n_leaves + 1, 0);
                std::vector<int> flat_sources;
                flat_sources.reserve(
                    static_cast<size_t>(flat_source_count));
                for (int leaf = 0; leaf < n_leaves; leaf++) {
                    flat_offsets[leaf] =
                        static_cast<int>(flat_sources.size());
                    const auto append_leaf =
                        [&](int source_leaf) {
                            flat_sources.insert(
                                flat_sources.end(),
                                h_src_ids_flat.begin() +
                                    h_src_id_offsets[source_leaf],
                                h_src_ids_flat.begin() +
                                    h_src_id_offsets[source_leaf + 1]);
                        };
                    append_leaf(leaf);
                    for (int index = leaf_near_offsets[leaf];
                         index < leaf_near_offsets[leaf + 1]; index++)
                        append_leaf(leaf_near_ids[index]);
                }
                flat_offsets[n_leaves] =
                    static_cast<int>(flat_sources.size());
                cudaError_t offset_error = cudaMalloc(
                    &d_leaf_near_source_offsets_cached,
                    offset_bytes);
                cudaError_t source_error = cudaSuccess;
                if (offset_error == cudaSuccess)
                    source_error = cudaMalloc(
                        &d_leaf_near_source_ids_cached,
                        std::max(source_bytes, sizeof(int)));
                if (offset_error == cudaSuccess &&
                    source_error == cudaSuccess) {
                    CUDA_CHECK(cudaMemcpy(
                        d_leaf_near_source_offsets_cached,
                        flat_offsets.data(), offset_bytes,
                        cudaMemcpyHostToDevice));
                    if (!flat_sources.empty())
                        CUDA_CHECK(cudaMemcpy(
                            d_leaf_near_source_ids_cached,
                            flat_sources.data(), source_bytes,
                            cudaMemcpyHostToDevice));
                    std::printf(
                        "  [FMM] Flat near-source cache: %.1f MiB "
                        "(%zu source references)\n",
                        cache_bytes / (1024.0 * 1024.0),
                        flat_sources.size());
                } else {
                    cudaFree(d_leaf_near_source_offsets_cached);
                    cudaFree(d_leaf_near_source_ids_cached);
                    d_leaf_near_source_offsets_cached = nullptr;
                    d_leaf_near_source_ids_cached = nullptr;
                    cudaGetLastError();
                }
            } else {
                std::fprintf(
                    stderr,
                    "  [FMM] flat near-source cache needs %.1f MiB; "
                    "using compact leaf lists\n",
                    cache_bytes / (1024.0 * 1024.0));
            }
        } else {
            std::fprintf(
                stderr,
                "  [FMM] flat near-source cache exceeds 32-bit "
                "indexing; using compact leaf lists\n");
        }
    }

    initialized = true;
    printf("  [FMM] Init complete: p=%d L=%d, %d nodes, %.1fms\n",
           p, L, n_nodes, timer.elapsed_ms());
}

// Helper to run the full FMM tree traversal (shared between evaluate and evaluate_gradient)
void HelmholtzFMM::run_tree(
    const double* h_q_re, const double* h_q_im, int derivative_order)
{
    CUDA_CHECK(cudaMemcpy(d_charges_re, h_q_re, Ns * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_charges_im, h_q_im, Ns * sizeof(double), cudaMemcpyHostToDevice));
    run_tree_uploaded(derivative_order);
}

void HelmholtzFMM::run_tree_uploaded(int derivative_order)
{
    int n_leaves = (int)leaf_info.size();

    // Clear multipole/local arrays
    CUDA_CHECK(cudaMemset(d_multi_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local_im, 0, n_nodes * L * sizeof(double)));

    // Use cached GPU arrays (allocated once in init())
    double *d_node_centers = d_node_centers_cached;
    double *d_dirs = d_dirs_cached;
    double *d_weights = d_weights_cached;
    int *d_leaf_idx = d_leaf_idx_cached;
    int *d_src_id_offsets = d_src_id_offsets_cached;
    int *d_src_ids = d_src_ids_cached;
    int *d_tgt_id_offsets = d_tgt_id_offsets_cached;
    int *d_tgt_ids = d_tgt_ids_cached;

    int block_L = std::min(L, 256);

    // === P2M ===
    if (n_leaves > 0) {
        p2m_kernel<<<n_leaves, block_L>>>(
            d_src_pts, d_charges_re, d_charges_im,
            d_dirs, k.real(), k.imag(),
            d_multi_re, d_multi_im,
            d_leaf_idx, d_src_id_offsets, d_src_ids,
            d_node_centers, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
    }

    // === M2M (bottom-up) ===
    for (int level = tree.max_level - 1; level >= 1; level--) {
        if (level < (int)m2m_level_info.size() && m2m_level_info[level].count > 0) {
            int off = m2m_level_info[level].offset;
            int cnt = m2m_level_info[level].count;
            m2m_kernel<<<cnt, block_L>>>(d_m2m_parent, d_m2m_child,
                d_m2m_shift_re, d_m2m_shift_im, d_multi_re, d_multi_im, L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // === M2L ===
    for (int level = 1; level <= tree.max_level; level++) {
        if (level < (int)m2l_level_info.size() && m2l_level_info[level].count > 0) {
            int off = m2l_level_info[level].offset;
            int cnt = m2l_level_info[level].count;
            m2l_kernel<<<cnt, block_L>>>(d_m2l_tgt, d_m2l_src, d_m2l_tidx,
                d_transfer_re, d_transfer_im, d_multi_re, d_multi_im,
                d_local_re, d_local_im, L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // === L2L (top-down) ===
    for (int level = 2; level <= tree.max_level; level++) {
        if (level < (int)l2l_level_info.size() && l2l_level_info[level].count > 0) {
            int off = l2l_level_info[level].offset;
            int cnt = l2l_level_info[level].count;
            l2l_kernel<<<cnt, block_L>>>(d_l2l_parent, d_l2l_child,
                d_l2l_shift_re, d_l2l_shift_im,
                d_local_re, d_local_im, d_local_re, d_local_im, L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    cdouble ik_val = cdouble(0, 1) * k;
    cdouble prefactor = ik_val / (16.0 * M_PI * M_PI);

    if (derivative_order == 0) {
        // === L2P (potential) ===
        CUDA_CHECK(cudaMemset(d_result_re, 0, Nt * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_result_im, 0, Nt * sizeof(double)));
        if (n_leaves > 0) {
            int block_tgt = 256;
            l2p_kernel<<<n_leaves, block_tgt>>>(
                d_tgt_pts, d_dirs, d_weights,
                d_local_re, d_local_im,
                k.real(), k.imag(), prefactor.real(), prefactor.imag(),
                d_result_re, d_result_im,
                d_leaf_idx, d_tgt_id_offsets, d_tgt_ids,
                d_node_centers, L, n_leaves);
            CUDA_CHECK(cudaGetLastError());
        }

        // === P2P (potential) ===
        launch_p2p_potential_leaf(d_tgt_pts, d_src_pts,
            d_charges_re, d_charges_im,
            d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_src_id_offsets_cached, d_src_ids_cached,
            d_leaf_near_offsets_cached, d_leaf_near_ids_cached,
            n_leaves, k.real(), k.imag(),
            d_result_re, d_result_im);
        CUDA_CHECK(cudaGetLastError());
    }
    if (derivative_order == 1 || derivative_order == 3) {
        // === L2P (gradient) ===
        CUDA_CHECK(cudaMemset(d_grad_re, 0, Nt * 3 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_grad_im, 0, Nt * 3 * sizeof(double)));

        // grad uses 6 separate arrays: gx, gy, gz (re/im)
        // gx aliases d_grad_re/im; gy, gz use cached arrays
        double *d_gx_re = d_grad_re, *d_gx_im = d_grad_im;
        double *d_gy_re = d_gy_re_cached, *d_gy_im = d_gy_im_cached;
        double *d_gz_re = d_gz_re_cached, *d_gz_im = d_gz_im_cached;
        CUDA_CHECK(cudaMemset(d_gy_re, 0, Nt * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_gy_im, 0, Nt * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_gz_re, 0, Nt * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_gz_im, 0, Nt * sizeof(double)));

        if (n_leaves > 0) {
            l2p_gradient_kernel<<<n_leaves, 256>>>(
                d_tgt_pts, d_dirs, d_weights,
                d_local_re, d_local_im,
                k.real(), k.imag(), prefactor.real(), prefactor.imag(),
                ik_val.real(), ik_val.imag(),
                d_gx_re, d_gx_im, d_gy_re, d_gy_im, d_gz_re, d_gz_im,
                d_leaf_idx, d_tgt_id_offsets, d_tgt_ids,
                d_node_centers, L, n_leaves);
            CUDA_CHECK(cudaGetLastError());
        }

        // P2P gradient
        launch_p2p_gradient_leaf(d_tgt_pts, d_src_pts,
            d_charges_re, d_charges_im,
            d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_src_id_offsets_cached, d_src_ids_cached,
            d_leaf_near_offsets_cached, d_leaf_near_ids_cached,
            n_leaves, k.real(), k.imag(),
            d_gx_re, d_gx_im, d_gy_re, d_gy_im, d_gz_re, d_gz_im);
        CUDA_CHECK(cudaGetLastError());

        // Repack gradient on GPU: 6 separate arrays -> interleaved [x0,y0,z0,x1,...]
        // d_gx_re aliases d_grad_re[0..Nt-1], so we need a temp copy of gx
        // to avoid overwriting source data during interleaving.
        double *d_gx_re_tmp = d_gx_re_tmp_cached;
        double *d_gx_im_tmp = d_gx_im_tmp_cached;
        CUDA_CHECK(cudaMemcpy(d_gx_re_tmp, d_gx_re, Nt * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_gx_im_tmp, d_gx_im, Nt * sizeof(double), cudaMemcpyDeviceToDevice));

        int repack_block = 256;
        int repack_grid = (Nt + repack_block - 1) / repack_block;
        repack_gradient_kernel<<<repack_grid, repack_block>>>(
            d_gx_re_tmp, d_gx_im_tmp, d_gy_re, d_gy_im, d_gz_re, d_gz_im,
            d_grad_re, d_grad_im, Nt);
        CUDA_CHECK(cudaGetLastError());

    }
    if (derivative_order == 2 || derivative_order == 3) {
        CUDA_CHECK(cudaMemset(
            d_hess_re, 0, (size_t)Nt * 6 * sizeof(double)));
        CUDA_CHECK(cudaMemset(
            d_hess_im, 0, (size_t)Nt * 6 * sizeof(double)));
        const cdouble negative_k_squared = -k * k;
        if (n_leaves > 0) {
            l2p_hessian_kernel<<<n_leaves, 256>>>(
                d_tgt_pts, d_dirs, d_weights,
                d_local_re, d_local_im,
                k.real(), k.imag(),
                prefactor.real(), prefactor.imag(),
                negative_k_squared.real(),
                negative_k_squared.imag(),
                d_hess_re, d_hess_im,
                d_leaf_idx, d_tgt_id_offsets, d_tgt_ids,
                d_node_centers, L, n_leaves);
            CUDA_CHECK(cudaGetLastError());
        }
        launch_p2p_hessian_leaf(
            d_tgt_pts, d_src_pts,
            d_charges_re, d_charges_im,
            d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_src_id_offsets_cached, d_src_ids_cached,
            d_leaf_near_offsets_cached, d_leaf_near_ids_cached,
            n_leaves, k.real(), k.imag(),
            d_hess_re, d_hess_im);
        CUDA_CHECK(cudaGetLastError());
    }
}

void HelmholtzFMM::evaluate(const cdouble* charges, cdouble* result)
{
    int block = 256;
    int grid_src = (Ns + block - 1) / block;
    CUDA_CHECK(cudaMemcpy(d_complex_tmp1, charges, Ns * sizeof(double2), cudaMemcpyHostToDevice));
    split_complex_kernel<<<grid_src, block>>>(d_complex_tmp1, d_charges_re, d_charges_im, Ns);
    CUDA_CHECK(cudaGetLastError());

    run_tree_uploaded(false);

    int grid_tgt = (Nt + block - 1) / block;
    pack_complex_kernel<<<grid_tgt, block>>>(d_result_re, d_result_im, d_complex_tmp1, Nt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(result, d_complex_tmp1, Nt * sizeof(double2), cudaMemcpyDeviceToHost));
}

void HelmholtzFMM::evaluate_gradient(const cdouble* charges, cdouble* grad_result)
{
    int block = 256;
    int grid_src = (Ns + block - 1) / block;
    CUDA_CHECK(cudaMemcpy(d_complex_tmp1, charges, Ns * sizeof(double2), cudaMemcpyHostToDevice));
    split_complex_kernel<<<grid_src, block>>>(d_complex_tmp1, d_charges_re, d_charges_im, Ns);
    CUDA_CHECK(cudaGetLastError());

    run_tree_uploaded(true);

    int ngrad = Nt * 3;
    int grid_grad = (ngrad + block - 1) / block;
    pack_complex_kernel<<<grid_grad, block>>>(d_grad_re, d_grad_im, d_complex_tmp1, ngrad);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(grad_result, d_complex_tmp1, ngrad * sizeof(double2), cudaMemcpyDeviceToHost));
}

void HelmholtzFMM::evaluate_hessian(
    const cdouble* charges, cdouble* hessian_result)
{
    const int block = 256;
    const int source_grid = (Ns + block - 1) / block;
    CUDA_CHECK(cudaMemcpy(
        d_complex_tmp1, charges,
        (size_t)Ns * sizeof(double2), cudaMemcpyHostToDevice));
    split_complex_kernel<<<source_grid, block>>>(
        d_complex_tmp1, d_charges_re, d_charges_im, Ns);
    CUDA_CHECK(cudaGetLastError());

    run_tree_uploaded(2);

    const int output_count = 6 * Nt;
    const int target_grid = (output_count + block - 1) / block;
    pack_complex_kernel<<<target_grid, block>>>(
        d_hess_re, d_hess_im, d_complex_tmp1, output_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(
        hessian_result, d_complex_tmp1,
        (size_t)output_count * sizeof(double2), cudaMemcpyDeviceToHost));
}

void HelmholtzFMM::evaluate_grad_hessian(
    const cdouble* charges,
    cdouble* gradient_result,
    cdouble* hessian_result)
{
    const int block = 256;
    const int source_grid = (Ns + block - 1) / block;
    CUDA_CHECK(cudaMemcpy(
        d_complex_tmp1, charges,
        (size_t)Ns * sizeof(double2), cudaMemcpyHostToDevice));
    split_complex_kernel<<<source_grid, block>>>(
        d_complex_tmp1, d_charges_re, d_charges_im, Ns);
    CUDA_CHECK(cudaGetLastError());

    run_tree_uploaded(3);

    const int gradient_count = 3 * Nt;
    const int gradient_grid =
        (gradient_count + block - 1) / block;
    pack_complex_kernel<<<gradient_grid, block>>>(
        d_grad_re, d_grad_im, d_complex_tmp1, gradient_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(
        gradient_result, d_complex_tmp1,
        (size_t)gradient_count * sizeof(double2),
        cudaMemcpyDeviceToHost));

    const int hessian_count = 6 * Nt;
    const int hessian_grid =
        (hessian_count + block - 1) / block;
    pack_complex_kernel<<<hessian_grid, block>>>(
        d_hess_re, d_hess_im, d_complex_tmp1, hessian_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(
        hessian_result, d_complex_tmp1,
        (size_t)hessian_count * sizeof(double2),
        cudaMemcpyDeviceToHost));
}

void HelmholtzFMM::evaluate_grad_hessian_batch3(
    const cdouble* charges1,
    const cdouble* charges2,
    const cdouble* charges3,
    cdouble* gradient1,
    cdouble* gradient2,
    cdouble* gradient3,
    cdouble* hessian1,
    cdouble* hessian2,
    cdouble* hessian3)
{
    if (!batch4_allocated) {
        evaluate_grad_hessian(charges1, gradient1, hessian1);
        evaluate_grad_hessian(charges2, gradient2, hessian2);
        evaluate_grad_hessian(charges3, gradient3, hessian3);
        return;
    }

    const bool profile =
        bem_env_flag_enabled("BEM_FMM_PROFILE_BATCH3");
    cudaEvent_t profile_events[8] = {};
    if (profile) {
        for (cudaEvent_t& event : profile_events)
            CUDA_CHECK(cudaEventCreate(&event));
        CUDA_CHECK(cudaEventRecord(profile_events[0]));
    }

    const int block = 256;
    const int source_grid = (Ns + block - 1) / block;
    auto upload = [&](const cdouble* charges,
                      double* real, double* imaginary) {
        CUDA_CHECK(cudaMemcpy(
            d_complex_tmp1, charges,
            (size_t)Ns * sizeof(double2), cudaMemcpyHostToDevice));
        split_complex_kernel<<<source_grid, block>>>(
            d_complex_tmp1, real, imaginary, Ns);
        CUDA_CHECK(cudaGetLastError());
    };
    upload(charges1, d_charges_re, d_charges_im);
    upload(charges2, d_charges2_re, d_charges2_im);
    upload(charges3, d_charges3_re, d_charges3_im);
    if (profile)
        CUDA_CHECK(cudaEventRecord(profile_events[1]));

    // Far-field gradients and Hessians reuse one three-channel traversal;
    // their near fields are fused below.
    evaluate_batch3_far_uploaded();
    if (profile)
        CUDA_CHECK(cudaEventRecord(profile_events[2]));
    evaluate_gradient_batch4_l2p_uploaded();
    if (profile)
        CUDA_CHECK(cudaEventRecord(profile_events[3]));

    const size_t hessian_bytes = (size_t)Nt * 6 * sizeof(double);
    CUDA_CHECK(cudaMemset(d_hess_re, 0, hessian_bytes));
    CUDA_CHECK(cudaMemset(d_hess_im, 0, hessian_bytes));
    CUDA_CHECK(cudaMemset(d_hess2_re, 0, hessian_bytes));
    CUDA_CHECK(cudaMemset(d_hess2_im, 0, hessian_bytes));
    CUDA_CHECK(cudaMemset(d_hess3_re, 0, hessian_bytes));
    CUDA_CHECK(cudaMemset(d_hess3_im, 0, hessian_bytes));
    const cdouble ik_val = cdouble(0, 1) * k;
    const cdouble prefactor = ik_val / (16.0 * M_PI * M_PI);
    const cdouble negative_k_squared = -k * k;
    const int n_leaves = (int)leaf_info.size();
    if (n_leaves > 0) {
        l2p_hessian_kernel_batch3<<<n_leaves, 256>>>(
            d_tgt_pts, d_dirs_cached, d_weights_cached,
            d_local_re, d_local_im,
            d_local2_re, d_local2_im,
            d_local3_re, d_local3_im,
            k.real(), k.imag(),
            prefactor.real(), prefactor.imag(),
            negative_k_squared.real(), negative_k_squared.imag(),
            d_hess_re, d_hess_im,
            d_hess2_re, d_hess2_im,
            d_hess3_re, d_hess3_im,
            d_leaf_idx_cached, d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_node_centers_cached, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
    }
    if (profile)
        CUDA_CHECK(cudaEventRecord(profile_events[4]));
    launch_p2p_grad_hessian_batch3_leaf(
        d_tgt_pts, d_src_pts,
        d_charges_re, d_charges_im,
        d_charges2_re, d_charges2_im,
        d_charges3_re, d_charges3_im,
        d_tgt_id_offsets_cached, d_tgt_ids_cached,
        d_src_id_offsets_cached, d_src_ids_cached,
        d_leaf_near_offsets_cached, d_leaf_near_ids_cached,
        n_leaves, k.real(), k.imag(),
        d_gx_re_tmp_cached, d_gx_im_tmp_cached,
        d_gy_re_cached, d_gy_im_cached,
        d_gz_re_cached, d_gz_im_cached,
        d_gx2_re_tmp_cached, d_gx2_im_tmp_cached,
        d_gy2_re_cached, d_gy2_im_cached,
        d_gz2_re_cached, d_gz2_im_cached,
        d_gx3_re_tmp_cached, d_gx3_im_tmp_cached,
        d_gy3_re_cached, d_gy3_im_cached,
        d_gz3_re_cached, d_gz3_im_cached,
        d_hess_re, d_hess_im,
        d_hess2_re, d_hess2_im,
        d_hess3_re, d_hess3_im,
        near_field_fp32);
    CUDA_CHECK(cudaGetLastError());
    if (profile)
        CUDA_CHECK(cudaEventRecord(profile_events[5]));

    const int gradient_grid = (Nt + block - 1) / block;
    repack_gradient_kernel<<<gradient_grid, block>>>(
        d_gx_re_tmp_cached, d_gx_im_tmp_cached,
        d_gy_re_cached, d_gy_im_cached,
        d_gz_re_cached, d_gz_im_cached,
        d_grad_re, d_grad_im, Nt);
    repack_gradient_kernel<<<gradient_grid, block>>>(
        d_gx2_re_tmp_cached, d_gx2_im_tmp_cached,
        d_gy2_re_cached, d_gy2_im_cached,
        d_gz2_re_cached, d_gz2_im_cached,
        d_grad2_re, d_grad2_im, Nt);
    repack_gradient_kernel<<<gradient_grid, block>>>(
        d_gx3_re_tmp_cached, d_gx3_im_tmp_cached,
        d_gy3_re_cached, d_gy3_im_cached,
        d_gz3_re_cached, d_gz3_im_cached,
        d_grad3_re, d_grad3_im, Nt);
    CUDA_CHECK(cudaGetLastError());
    if (profile)
        CUDA_CHECK(cudaEventRecord(profile_events[6]));

    auto download = [&](const double* real, const double* imaginary,
                        cdouble* output, int count) {
        const int grid = (count + block - 1) / block;
        pack_complex_kernel<<<grid, block>>>(
            real, imaginary, d_complex_tmp1, count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(
            output, d_complex_tmp1,
            (size_t)count * sizeof(double2), cudaMemcpyDeviceToHost));
    };
    download(d_grad_re, d_grad_im, gradient1, 3 * Nt);
    download(d_grad2_re, d_grad2_im, gradient2, 3 * Nt);
    download(d_grad3_re, d_grad3_im, gradient3, 3 * Nt);
    download(d_hess_re, d_hess_im, hessian1, 6 * Nt);
    download(d_hess2_re, d_hess2_im, hessian2, 6 * Nt);
    download(d_hess3_re, d_hess3_im, hessian3, 6 * Nt);
    if (profile) {
        CUDA_CHECK(cudaEventRecord(profile_events[7]));
        CUDA_CHECK(cudaEventSynchronize(profile_events[7]));
        float milliseconds[7] = {};
        for (int phase = 0; phase < 7; phase++) {
            CUDA_CHECK(cudaEventElapsedTime(
                &milliseconds[phase],
                profile_events[phase],
                profile_events[phase + 1]));
        }
        std::printf(
            "  [FMM batch3 profile] upload=%.3fms far=%.3fms "
            "grad-L2P=%.3fms hess-L2P=%.3fms P2P=%.3fms "
            "repack=%.3fms download=%.3fms\n",
            milliseconds[0], milliseconds[1], milliseconds[2],
            milliseconds[3], milliseconds[4], milliseconds[5],
            milliseconds[6]);
        for (cudaEvent_t& event : profile_events)
            CUDA_CHECK(cudaEventDestroy(event));
    }
}

void HelmholtzFMM::evaluate_vector_actions_batch3(
    const cdouble* charges_x,
    const cdouble* charges_y,
    const cdouble* charges_z,
    cdouble* curl_result,
    cdouble* hessian_action)
{
    if (!batch4_allocated) {
        std::vector<cdouble> gradient_x((size_t)Nt * 3);
        std::vector<cdouble> gradient_y((size_t)Nt * 3);
        std::vector<cdouble> gradient_z((size_t)Nt * 3);
        std::vector<cdouble> hessian_x((size_t)Nt * 6);
        std::vector<cdouble> hessian_y((size_t)Nt * 6);
        std::vector<cdouble> hessian_z((size_t)Nt * 6);
        evaluate_grad_hessian_batch3(
            charges_x, charges_y, charges_z,
            gradient_x.data(), gradient_y.data(), gradient_z.data(),
            hessian_x.data(), hessian_y.data(), hessian_z.data());
        for (int point = 0; point < Nt; point++) {
            curl_result[3 * point] =
                gradient_x[3 * point + 1] -
                gradient_y[3 * point];
            curl_result[3 * point + 1] =
                gradient_x[3 * point + 2] -
                gradient_z[3 * point];
            curl_result[3 * point + 2] =
                gradient_y[3 * point + 2] -
                gradient_z[3 * point + 1];
            hessian_action[3 * point] =
                -hessian_x[6 * point + 3] -
                hessian_x[6 * point + 5] +
                hessian_y[6 * point + 1] +
                hessian_z[6 * point + 2];
            hessian_action[3 * point + 1] =
                hessian_x[6 * point + 1] -
                hessian_y[6 * point] -
                hessian_y[6 * point + 5] +
                hessian_z[6 * point + 4];
            hessian_action[3 * point + 2] =
                hessian_x[6 * point + 2] +
                hessian_y[6 * point + 4] -
                hessian_z[6 * point] -
                hessian_z[6 * point + 3];
        }
        return;
    }

    const bool profile =
        bem_env_flag_enabled("BEM_FMM_PROFILE_BATCH3");
    cudaEvent_t profile_events[5] = {};
    if (profile) {
        for (cudaEvent_t& event : profile_events)
            CUDA_CHECK(cudaEventCreate(&event));
        CUDA_CHECK(cudaEventRecord(profile_events[0]));
    }

    const int block = 256;
    const int source_grid = (Ns + block - 1) / block;
    auto upload = [&](const cdouble* charges,
                      double* real, double* imaginary) {
        CUDA_CHECK(cudaMemcpy(
            d_complex_tmp1, charges,
            (size_t)Ns * sizeof(double2), cudaMemcpyHostToDevice));
        split_complex_kernel<<<source_grid, block>>>(
            d_complex_tmp1, real, imaginary, Ns);
        CUDA_CHECK(cudaGetLastError());
    };
    upload(charges_x, d_charges_re, d_charges_im);
    upload(charges_y, d_charges2_re, d_charges2_im);
    upload(charges_z, d_charges3_re, d_charges3_im);
    if (profile)
        CUDA_CHECK(cudaEventRecord(profile_events[1]));

    evaluate_batch3_far_uploaded();
    if (profile)
        CUDA_CHECK(cudaEventRecord(profile_events[2]));

    const int output_count = 3 * Nt;
    const size_t output_bytes =
        (size_t)output_count * sizeof(double);
    CUDA_CHECK(cudaMemset(d_grad_re, 0, output_bytes));
    CUDA_CHECK(cudaMemset(d_grad_im, 0, output_bytes));
    CUDA_CHECK(cudaMemset(d_hess_re, 0, output_bytes));
    CUDA_CHECK(cudaMemset(d_hess_im, 0, output_bytes));
    const cdouble ik_val = cdouble(0, 1) * k;
    const cdouble prefactor =
        ik_val / (16.0 * M_PI * M_PI);
    const cdouble negative_k_squared = -k * k;
    const int n_leaves = (int)leaf_info.size();
    if (n_leaves > 0) {
        l2p_vector_actions_kernel_batch3<<<n_leaves, 256>>>(
            d_tgt_pts, d_dirs_cached, d_weights_cached,
            d_local_re, d_local_im,
            d_local2_re, d_local2_im,
            d_local3_re, d_local3_im,
            k.real(), k.imag(),
            prefactor.real(), prefactor.imag(),
            ik_val.real(), ik_val.imag(),
            negative_k_squared.real(), negative_k_squared.imag(),
            d_grad_re, d_grad_im,
            d_hess_re, d_hess_im,
            d_leaf_idx_cached, d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_node_centers_cached, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
        launch_p2p_vector_actions_batch3_leaf(
            d_tgt_pts, d_src_pts,
            d_charges_re, d_charges_im,
            d_charges2_re, d_charges2_im,
            d_charges3_re, d_charges3_im,
            d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_src_id_offsets_cached, d_src_ids_cached,
            d_leaf_near_offsets_cached, d_leaf_near_ids_cached,
            d_leaf_near_source_offsets_cached,
            d_leaf_near_source_ids_cached,
            n_leaves, k.real(), k.imag(),
            d_grad_re, d_grad_im,
            d_hess_re, d_hess_im,
            near_field_fp32);
        CUDA_CHECK(cudaGetLastError());
    }
    if (profile)
        CUDA_CHECK(cudaEventRecord(profile_events[3]));

    const int output_grid =
        (output_count + block - 1) / block;
    pack_complex_kernel<<<output_grid, block>>>(
        d_grad_re, d_grad_im, d_complex_tmp1, output_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(
        curl_result, d_complex_tmp1,
        (size_t)output_count * sizeof(double2),
        cudaMemcpyDeviceToHost));
    pack_complex_kernel<<<output_grid, block>>>(
        d_hess_re, d_hess_im, d_complex_tmp1, output_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(
        hessian_action, d_complex_tmp1,
        (size_t)output_count * sizeof(double2),
        cudaMemcpyDeviceToHost));
    if (profile) {
        CUDA_CHECK(cudaEventRecord(profile_events[4]));
        CUDA_CHECK(cudaEventSynchronize(profile_events[4]));
        float milliseconds[4] = {};
        for (int phase = 0; phase < 4; phase++) {
            CUDA_CHECK(cudaEventElapsedTime(
                &milliseconds[phase],
                profile_events[phase],
                profile_events[phase + 1]));
        }
        std::printf(
            "  [FMM vector profile] upload=%.3fms far=%.3fms "
            "contracted-L2P+P2P=%.3fms download=%.3fms\n",
            milliseconds[0], milliseconds[1],
            milliseconds[2], milliseconds[3]);
        for (cudaEvent_t& event : profile_events)
            CUDA_CHECK(cudaEventDestroy(event));
    }
}

void HelmholtzFMM::evaluate_vector_actions_batch3_device(
    const double* charges_x_re,
    const double* charges_x_im,
    const double* charges_y_re,
    const double* charges_y_im,
    const double* charges_z_re,
    const double* charges_z_im)
{
    if (!batch4_allocated)
        throw std::runtime_error(
            "device vector actions require batch3 FMM buffers");

    const bool profile =
        bem_env_flag_enabled("BEM_FMM_PROFILE_BATCH3");
    cudaEvent_t profile_events[5] = {};
    if (profile) {
        for (cudaEvent_t& event : profile_events)
            CUDA_CHECK(cudaEventCreate(&event));
        CUDA_CHECK(cudaEventRecord(profile_events[0]));
    }

    const size_t charge_bytes = (size_t)Ns * sizeof(double);
    CUDA_CHECK(cudaMemcpyAsync(
        d_charges_re, charges_x_re, charge_bytes,
        cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyAsync(
        d_charges_im, charges_x_im, charge_bytes,
        cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyAsync(
        d_charges2_re, charges_y_re, charge_bytes,
        cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyAsync(
        d_charges2_im, charges_y_im, charge_bytes,
        cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyAsync(
        d_charges3_re, charges_z_re, charge_bytes,
        cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyAsync(
        d_charges3_im, charges_z_im, charge_bytes,
        cudaMemcpyDeviceToDevice));
    if (profile)
        CUDA_CHECK(cudaEventRecord(profile_events[1]));

    evaluate_batch3_far_uploaded();
    if (profile)
        CUDA_CHECK(cudaEventRecord(profile_events[2]));

    const int output_count = 3 * Nt;
    const size_t output_bytes =
        (size_t)output_count * sizeof(double);
    CUDA_CHECK(cudaMemsetAsync(d_grad_re, 0, output_bytes));
    CUDA_CHECK(cudaMemsetAsync(d_grad_im, 0, output_bytes));
    CUDA_CHECK(cudaMemsetAsync(d_hess_re, 0, output_bytes));
    CUDA_CHECK(cudaMemsetAsync(d_hess_im, 0, output_bytes));
    const cdouble ik_val = cdouble(0, 1) * k;
    const cdouble prefactor =
        ik_val / (16.0 * M_PI * M_PI);
    const cdouble negative_k_squared = -k * k;
    const int n_leaves = (int)leaf_info.size();
    if (n_leaves > 0) {
        l2p_vector_actions_kernel_batch3<<<n_leaves, 256>>>(
            d_tgt_pts, d_dirs_cached, d_weights_cached,
            d_local_re, d_local_im,
            d_local2_re, d_local2_im,
            d_local3_re, d_local3_im,
            k.real(), k.imag(),
            prefactor.real(), prefactor.imag(),
            ik_val.real(), ik_val.imag(),
            negative_k_squared.real(), negative_k_squared.imag(),
            d_grad_re, d_grad_im,
            d_hess_re, d_hess_im,
            d_leaf_idx_cached, d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_node_centers_cached, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
        if (profile)
            CUDA_CHECK(cudaEventRecord(profile_events[3]));
        launch_p2p_vector_actions_batch3_leaf(
            d_tgt_pts, d_src_pts,
            d_charges_re, d_charges_im,
            d_charges2_re, d_charges2_im,
            d_charges3_re, d_charges3_im,
            d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_src_id_offsets_cached, d_src_ids_cached,
            d_leaf_near_offsets_cached, d_leaf_near_ids_cached,
            d_leaf_near_source_offsets_cached,
            d_leaf_near_source_ids_cached,
            n_leaves, k.real(), k.imag(),
            d_grad_re, d_grad_im,
            d_hess_re, d_hess_im,
            near_field_fp32);
        CUDA_CHECK(cudaGetLastError());
    } else if (profile) {
        CUDA_CHECK(cudaEventRecord(profile_events[3]));
    }
    if (profile) {
        CUDA_CHECK(cudaEventRecord(profile_events[4]));
        CUDA_CHECK(cudaEventSynchronize(profile_events[4]));
        float milliseconds[4] = {};
        for (int phase = 0; phase < 4; phase++) {
            CUDA_CHECK(cudaEventElapsedTime(
                &milliseconds[phase],
                profile_events[phase],
                profile_events[phase + 1]));
        }
        std::printf(
            "  [FMM vector device profile] copy=%.3fms far=%.3fms "
            "contracted-L2P=%.3fms P2P=%.3fms\n",
            milliseconds[0], milliseconds[1],
            milliseconds[2], milliseconds[3]);
        for (cudaEvent_t& event : profile_events)
            CUDA_CHECK(cudaEventDestroy(event));
    } else {
        CUDA_CHECK(cudaStreamSynchronize(0));
    }
}

__global__ void pack_pair_charges_fp32_kernel(
    const double* first_x_re,
    const double* first_x_im,
    const double* first_y_re,
    const double* first_y_im,
    const double* first_z_re,
    const double* first_z_im,
    const double* second_x_re,
    const double* second_x_im,
    const double* second_y_re,
    const double* second_y_im,
    const double* second_z_re,
    const double* second_z_im,
    float* packed,
    int source_count)
{
    const int source = blockIdx.x * blockDim.x + threadIdx.x;
    if (source >= source_count)
        return;
    const double* fields[12] = {
        first_x_re, first_x_im,
        first_y_re, first_y_im,
        first_z_re, first_z_im,
        second_x_re, second_x_im,
        second_y_re, second_y_im,
        second_z_re, second_z_im
    };
#pragma unroll
    for (int field = 0; field < 12; field++)
        packed[field * source_count + source] =
            static_cast<float>(fields[field][source]);
}

void HelmholtzFMM::evaluate_vector_actions_pair_batch3_device(
    const double* first_x_re,
    const double* first_x_im,
    const double* first_y_re,
    const double* first_y_im,
    const double* first_z_re,
    const double* first_z_im,
    const double* second_x_re,
    const double* second_x_im,
    const double* second_y_re,
    const double* second_y_im,
    const double* second_z_re,
    const double* second_z_im)
{
    if (!batch4_allocated || d_hess2_re == nullptr ||
        d_hess2_im == nullptr)
        throw std::runtime_error(
            "paired device vector actions require batch4 FMM buffers");

    const bool profile =
        bem_env_flag_enabled("BEM_FMM_PROFILE_BATCH3");
    cudaEvent_t profile_events[6] = {};
    if (profile) {
        for (cudaEvent_t& event : profile_events)
            CUDA_CHECK(cudaEventCreate(&event));
        CUDA_CHECK(cudaEventRecord(profile_events[0]));
    }

    const size_t charge_bytes = (size_t)Ns * sizeof(double);
    const int output_count = 3 * Nt;
    const size_t output_bytes =
        (size_t)output_count * sizeof(double);
    const cdouble ik_val = cdouble(0, 1) * k;
    const cdouble prefactor =
        ik_val / (16.0 * M_PI * M_PI);
    const cdouble negative_k_squared = -k * k;
    const int n_leaves = (int)leaf_info.size();
    int max_leaf_sources = 0;
    for (const LeafInfo& leaf : leaf_info)
        max_leaf_sources =
            std::max(max_leaf_sources, leaf.src_count);
    bool share_l2p = pair_l2p_allocated;
    const char* pair_l2p_override =
        std::getenv("BEM_FMM_PAIR_L2P");
    if (pair_l2p_override != nullptr)
        share_l2p =
            std::strcmp(pair_l2p_override, "0") != 0 &&
            pair_l2p_allocated;
    bool share_far = share_l2p;
    const char* pair_far_override =
        std::getenv("BEM_FMM_PAIR_FAR");
    if (pair_far_override != nullptr)
        share_far =
            std::strcmp(pair_far_override, "0") != 0 &&
            share_l2p;
    bool pair_m2l_fp32_default = false;
#ifdef BEM_DEFAULT_FMM_NEAR_FP32
    pair_m2l_fp32_default = true;
#endif
    const bool pair_m2l_fp32 =
        !force_pair_fp64 && bem_env_flag_enabled(
            "BEM_FMM_M2L_FP32", pair_m2l_fp32_default);
    const bool pair_local_storage_fp32 =
        pair_m2l_fp32 &&
        d_pair_local_re_fp32 != nullptr &&
        d_pair_local_im_fp32 != nullptr;
    const bool pair_multi_storage_fp32 =
        pair_m2l_fp32 &&
        d_pair_multi_re == nullptr &&
        d_pair_multi_im == nullptr &&
        d_pair_multi_re_fp32 != nullptr &&
        d_pair_multi_im_fp32 != nullptr;
    if (pair_local_storage_fp32 && !share_far)
        share_l2p = false;
    const bool packed_pair_charges =
        d_pair_charges_fp32 != nullptr &&
        (near_field_fp32 || pair_multi_storage_fp32);
    if (packed_pair_charges) {
        const int block = 256;
        const int grid = (Ns + block - 1) / block;
        pack_pair_charges_fp32_kernel<<<grid, block>>>(
            first_x_re, first_x_im,
            first_y_re, first_y_im,
            first_z_re, first_z_im,
            second_x_re, second_x_im,
            second_y_re, second_y_im,
            second_z_re, second_z_im,
            d_pair_charges_fp32, Ns);
        CUDA_CHECK(cudaGetLastError());
    }

    auto upload_three = [&](const double* x_re, const double* x_im,
                            const double* y_re, const double* y_im,
                            const double* z_re, const double* z_im) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_charges_re, x_re, charge_bytes,
            cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpyAsync(
            d_charges_im, x_im, charge_bytes,
            cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpyAsync(
            d_charges2_re, y_re, charge_bytes,
            cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpyAsync(
            d_charges2_im, y_im, charge_bytes,
            cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpyAsync(
            d_charges3_re, z_re, charge_bytes,
            cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpyAsync(
            d_charges3_im, z_im, charge_bytes,
            cudaMemcpyDeviceToDevice));
    };
    auto evaluate_l2p = [&](double* curl_re, double* curl_im,
                            double* action_re, double* action_im) {
        CUDA_CHECK(cudaMemsetAsync(curl_re, 0, output_bytes));
        CUDA_CHECK(cudaMemsetAsync(curl_im, 0, output_bytes));
        CUDA_CHECK(cudaMemsetAsync(action_re, 0, output_bytes));
        CUDA_CHECK(cudaMemsetAsync(action_im, 0, output_bytes));
        if (n_leaves <= 0)
            return;
        l2p_vector_actions_kernel_batch3<<<n_leaves, 256>>>(
            d_tgt_pts, d_dirs_cached, d_weights_cached,
            d_local_re, d_local_im,
            d_local2_re, d_local2_im,
            d_local3_re, d_local3_im,
            k.real(), k.imag(),
            prefactor.real(), prefactor.imag(),
            ik_val.real(), ik_val.imag(),
            negative_k_squared.real(), negative_k_squared.imag(),
            curl_re, curl_im, action_re, action_im,
            d_leaf_idx_cached, d_tgt_id_offsets_cached,
            d_tgt_ids_cached, d_node_centers_cached,
            L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
    };
    auto evaluate_pair_far = [&]() {
        const bool profile_far_detail =
            bem_env_flag_enabled("BEM_FMM_PROFILE_FAR_DETAIL");
        cudaEvent_t far_events[6] = {};
        if (profile_far_detail) {
            for (cudaEvent_t& event : far_events)
                CUDA_CHECK(cudaEventCreate(&event));
            CUDA_CHECK(cudaEventRecord(far_events[0]));
        }
        const int field_stride = n_nodes * L;
        const size_t workspace_bytes =
            (size_t)6 * field_stride * sizeof(double);
        if (pair_multi_storage_fp32) {
            const size_t multi_bytes =
                (size_t)6 * field_stride * sizeof(float);
            CUDA_CHECK(cudaMemsetAsync(
                d_pair_multi_re_fp32, 0, multi_bytes));
            CUDA_CHECK(cudaMemsetAsync(
                d_pair_multi_im_fp32, 0, multi_bytes));
        } else {
            CUDA_CHECK(cudaMemsetAsync(
                d_pair_multi_re, 0, workspace_bytes));
            CUDA_CHECK(cudaMemsetAsync(
                d_pair_multi_im, 0, workspace_bytes));
        }
        if (pair_local_storage_fp32) {
            const size_t local_bytes =
                (size_t)6 * field_stride * sizeof(float);
            CUDA_CHECK(cudaMemsetAsync(
                d_pair_local_re_fp32, 0, local_bytes));
            CUDA_CHECK(cudaMemsetAsync(
                d_pair_local_im_fp32, 0, local_bytes));
        } else {
            CUDA_CHECK(cudaMemsetAsync(
                d_pair_local_re, 0, workspace_bytes));
            CUDA_CHECK(cudaMemsetAsync(
                d_pair_local_im, 0, workspace_bytes));
        }
        if (profile_far_detail)
            CUDA_CHECK(cudaEventRecord(far_events[1]));
        int block_l = bem_env_int(
            "BEM_FMM_PAIR_FAR_THREADS", 256);
        if (block_l != 64 && block_l != 128 &&
            block_l != 256 && block_l != 512)
            block_l = 256;
        block_l = std::min(L, block_l);
        if (n_leaves > 0) {
            const size_t p2m_shared_bytes =
                (size_t)15 * max_leaf_sources * sizeof(double);
            bool cache_p2m =
                bem_env_flag_enabled(
                    "BEM_FMM_PAIR_P2M_SHARED", false);
            if (cache_p2m) {
                int device = 0;
                int max_shared_bytes = 0;
                CUDA_CHECK(cudaGetDevice(&device));
                CUDA_CHECK(cudaDeviceGetAttribute(
                    &max_shared_bytes,
                    cudaDevAttrMaxSharedMemoryPerBlock,
                    device));
                cache_p2m =
                    p2m_shared_bytes <=
                    (size_t)max_shared_bytes;
            }
#define LAUNCH_PAIR_P2M(                                             \
    CACHE_SOURCES, CACHE_PHASE, MULTI_TYPE,                           \
    MULTI_RE, MULTI_IM, SHARED_BYTES)                                \
            p2m_kernel_batch6_flat<                                  \
                CACHE_SOURCES, CACHE_PHASE, MULTI_TYPE>              \
                <<<n_leaves, block_l, SHARED_BYTES>>>(                \
                d_src_pts,                                           \
                first_x_re, first_x_im,                              \
                first_y_re, first_y_im,                              \
                first_z_re, first_z_im,                              \
                second_x_re, second_x_im,                            \
                second_y_re, second_y_im,                            \
                second_z_re, second_z_im,                            \
                d_phase_cache_fp32,                                  \
                d_dirs_cached, k.real(), k.imag(),                   \
                MULTI_RE, MULTI_IM,                                  \
                field_stride,                                       \
                d_leaf_idx_cached,                                   \
                d_src_id_offsets_cached,                             \
                d_src_ids_cached,                                    \
                d_node_centers_cached,                               \
                max_leaf_sources,                                    \
                L, n_leaves)
            const bool cache_phase =
                d_phase_cache_fp32 != nullptr;
            if (packed_pair_charges &&
                pair_multi_storage_fp32) {
#define LAUNCH_PACKED_PAIR_P2M(CACHE_SOURCES, CACHE_PHASE, SHARED_BYTES) \
                p2m_kernel_batch6_flat<                              \
                    CACHE_SOURCES, CACHE_PHASE, float, float>         \
                    <<<n_leaves, block_l, SHARED_BYTES>>>(            \
                        d_src_pts,                                    \
                        d_pair_charges_fp32,                          \
                        d_pair_charges_fp32 + Ns,                     \
                        d_pair_charges_fp32 + 2 * Ns,                 \
                        d_pair_charges_fp32 + 3 * Ns,                 \
                        d_pair_charges_fp32 + 4 * Ns,                 \
                        d_pair_charges_fp32 + 5 * Ns,                 \
                        d_pair_charges_fp32 + 6 * Ns,                 \
                        d_pair_charges_fp32 + 7 * Ns,                 \
                        d_pair_charges_fp32 + 8 * Ns,                 \
                        d_pair_charges_fp32 + 9 * Ns,                 \
                        d_pair_charges_fp32 + 10 * Ns,                \
                        d_pair_charges_fp32 + 11 * Ns,                \
                        d_phase_cache_fp32,                           \
                        d_dirs_cached, k.real(), k.imag(),            \
                        d_pair_multi_re_fp32,                         \
                        d_pair_multi_im_fp32,                         \
                        field_stride,                                \
                        d_leaf_idx_cached,                            \
                        d_src_id_offsets_cached,                      \
                        d_src_ids_cached,                             \
                        d_node_centers_cached,                        \
                        max_leaf_sources, L, n_leaves)
                if (cache_p2m && cache_phase)
                    LAUNCH_PACKED_PAIR_P2M(
                        true, true, p2m_shared_bytes);
                else if (cache_p2m)
                    LAUNCH_PACKED_PAIR_P2M(
                        true, false, p2m_shared_bytes);
                else if (cache_phase)
                    LAUNCH_PACKED_PAIR_P2M(false, true, 0);
                else
                    LAUNCH_PACKED_PAIR_P2M(false, false, 0);
#undef LAUNCH_PACKED_PAIR_P2M
            } else if (cache_p2m && cache_phase &&
                pair_multi_storage_fp32)
                LAUNCH_PAIR_P2M(
                    true, true, float,
                    d_pair_multi_re_fp32,
                    d_pair_multi_im_fp32,
                    p2m_shared_bytes);
            else if (cache_p2m && pair_multi_storage_fp32)
                LAUNCH_PAIR_P2M(
                    true, false, float,
                    d_pair_multi_re_fp32,
                    d_pair_multi_im_fp32,
                    p2m_shared_bytes);
            else if (cache_phase && pair_multi_storage_fp32)
                LAUNCH_PAIR_P2M(
                    false, true, float,
                    d_pair_multi_re_fp32,
                    d_pair_multi_im_fp32, 0);
            else if (pair_multi_storage_fp32)
                LAUNCH_PAIR_P2M(
                    false, false, float,
                    d_pair_multi_re_fp32,
                    d_pair_multi_im_fp32, 0);
            else if (cache_p2m && cache_phase)
                LAUNCH_PAIR_P2M(
                    true, true, double,
                    d_pair_multi_re, d_pair_multi_im,
                    p2m_shared_bytes);
            else if (cache_p2m)
                LAUNCH_PAIR_P2M(
                    true, false, double,
                    d_pair_multi_re, d_pair_multi_im,
                    p2m_shared_bytes);
            else if (cache_phase)
                LAUNCH_PAIR_P2M(
                    false, true, double,
                    d_pair_multi_re, d_pair_multi_im, 0);
            else
                LAUNCH_PAIR_P2M(
                    false, false, double,
                    d_pair_multi_re, d_pair_multi_im, 0);
#undef LAUNCH_PAIR_P2M
            CUDA_CHECK(cudaGetLastError());
        }
        if (profile_far_detail)
            CUDA_CHECK(cudaEventRecord(far_events[2]));
        for (int level = tree.max_level - 1;
             level >= 1; level--) {
            if (level >= (int)m2m_level_info.size() ||
                m2m_level_info[level].count <= 0)
                continue;
            const int offset = m2m_level_info[level].offset;
            const int count = m2m_level_info[level].count;
            if (pair_multi_storage_fp32) {
                m2m_kernel_batch6_flat<6, float><<<count, block_l>>>(
                    d_m2m_parent, d_m2m_child,
                    d_m2m_shift_re, d_m2m_shift_im,
                    d_pair_multi_re_fp32,
                    d_pair_multi_im_fp32,
                    field_stride, L, count, offset);
            } else {
                m2m_kernel_batch6_flat<6, double><<<count, block_l>>>(
                    d_m2m_parent, d_m2m_child,
                    d_m2m_shift_re, d_m2m_shift_im,
                    d_pair_multi_re, d_pair_multi_im,
                    field_stride, L, count, offset);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        if (profile_far_detail)
            CUDA_CHECK(cudaEventRecord(far_events[3]));
        const bool transfer_storage_fp32 =
            pair_m2l_fp32 &&
            d_transfer_re_fp32 != nullptr &&
            d_transfer_im_fp32 != nullptr;
        const bool m2l_multi_storage_fp32 =
            pair_m2l_fp32 &&
            d_pair_multi_re_fp32 != nullptr &&
            d_pair_multi_im_fp32 != nullptr;
        if (m2l_multi_storage_fp32 &&
            !pair_multi_storage_fp32) {
            const size_t value_count =
                static_cast<size_t>(6) * field_stride;
            const int conversion_block = 256;
            const int conversion_grid = static_cast<int>(
                (value_count + conversion_block - 1) /
                conversion_block);
            double_to_float_kernel
                <<<conversion_grid, conversion_block>>>(
                    d_pair_multi_re,
                    d_pair_multi_re_fp32,
                    value_count);
            double_to_float_kernel
                <<<conversion_grid, conversion_block>>>(
                    d_pair_multi_im,
                    d_pair_multi_im_fp32,
                    value_count);
            CUDA_CHECK(cudaGetLastError());
        }
        for (int level = 1; level <= tree.max_level; level++) {
            if (level >= (int)m2l_row_level_info.size() ||
                m2l_row_level_info[level].count <= 0)
                continue;
            const int offset = m2l_row_level_info[level].offset;
            const int count = m2l_row_level_info[level].count;
#define LAUNCH_PAIR_M2L(                                             \
    COMPUTE_TYPE, TRANSFER_TYPE, MULTI_TYPE, LOCAL_TYPE,             \
    TRANSFER_RE, TRANSFER_IM, MULTI_RE, MULTI_IM,                    \
    LOCAL_RE, LOCAL_IM)                                              \
            m2l_kernel_batch6_flat_target_rows<6,                   \
                COMPUTE_TYPE, TRANSFER_TYPE, MULTI_TYPE, LOCAL_TYPE> \
                <<<count, block_l>>>(                                \
                    d_m2l_row_target,                                \
                    d_m2l_row_start,                                 \
                    d_m2l_row_end,                                   \
                    d_m2l_src,                                       \
                    d_m2l_tidx,                                      \
                    TRANSFER_RE,                                     \
                    TRANSFER_IM,                                     \
                    MULTI_RE,                                        \
                    MULTI_IM,                                        \
                    LOCAL_RE, LOCAL_IM,                              \
                    0,                                               \
                    field_stride,                                    \
                    L, count, offset)
            if (transfer_storage_fp32 &&
                m2l_multi_storage_fp32 &&
                pair_local_storage_fp32)
                LAUNCH_PAIR_M2L(
                    float, float, float, float,
                    d_transfer_re_fp32, d_transfer_im_fp32,
                    d_pair_multi_re_fp32, d_pair_multi_im_fp32,
                    d_pair_local_re_fp32, d_pair_local_im_fp32);
            else if (transfer_storage_fp32 &&
                     m2l_multi_storage_fp32)
                LAUNCH_PAIR_M2L(
                    float, float, float, double,
                    d_transfer_re_fp32, d_transfer_im_fp32,
                    d_pair_multi_re_fp32, d_pair_multi_im_fp32,
                    d_pair_local_re, d_pair_local_im);
            else if (m2l_multi_storage_fp32 &&
                     pair_local_storage_fp32)
                LAUNCH_PAIR_M2L(
                    float, double, float, float,
                    d_transfer_re, d_transfer_im,
                    d_pair_multi_re_fp32, d_pair_multi_im_fp32,
                    d_pair_local_re_fp32, d_pair_local_im_fp32);
            else if (m2l_multi_storage_fp32)
                LAUNCH_PAIR_M2L(
                    float, double, float, double,
                    d_transfer_re, d_transfer_im,
                    d_pair_multi_re_fp32, d_pair_multi_im_fp32,
                    d_pair_local_re, d_pair_local_im);
            else if (pair_local_storage_fp32)
                LAUNCH_PAIR_M2L(
                    float, double, double, float,
                    d_transfer_re, d_transfer_im,
                    d_pair_multi_re, d_pair_multi_im,
                    d_pair_local_re_fp32, d_pair_local_im_fp32);
            else if (pair_m2l_fp32)
                LAUNCH_PAIR_M2L(
                    float, double, double, double,
                    d_transfer_re, d_transfer_im,
                    d_pair_multi_re, d_pair_multi_im,
                    d_pair_local_re, d_pair_local_im);
            else
                LAUNCH_PAIR_M2L(
                    double, double, double, double,
                    d_transfer_re, d_transfer_im,
                    d_pair_multi_re, d_pair_multi_im,
                    d_pair_local_re, d_pair_local_im);
#undef LAUNCH_PAIR_M2L
            CUDA_CHECK(cudaGetLastError());
        }
        if (profile_far_detail)
            CUDA_CHECK(cudaEventRecord(far_events[4]));
        for (int level = 2; level <= tree.max_level; level++) {
            if (level >= (int)l2l_level_info.size() ||
                l2l_level_info[level].count <= 0)
                continue;
            const int offset = l2l_level_info[level].offset;
            const int count = l2l_level_info[level].count;
            if (pair_local_storage_fp32) {
                l2l_kernel_batch6_flat<6, float><<<count, block_l>>>(
                    d_l2l_parent, d_l2l_child,
                    d_l2l_shift_re, d_l2l_shift_im,
                    d_pair_local_re_fp32, d_pair_local_im_fp32,
                    field_stride, L, count, offset);
            } else {
                l2l_kernel_batch6_flat<6, double><<<count, block_l>>>(
                    d_l2l_parent, d_l2l_child,
                    d_l2l_shift_re, d_l2l_shift_im,
                    d_pair_local_re, d_pair_local_im,
                    field_stride, L, count, offset);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        if (profile_far_detail) {
            CUDA_CHECK(cudaEventRecord(far_events[5]));
            CUDA_CHECK(cudaEventSynchronize(far_events[5]));
            float milliseconds[5] = {};
            for (int phase = 0; phase < 5; phase++)
                CUDA_CHECK(cudaEventElapsedTime(
                    &milliseconds[phase],
                    far_events[phase],
                    far_events[phase + 1]));
            std::printf(
                "  [FMM far detail k=%.6g] clear=%.3fms "
                "P2M=%.3fms M2M=%.3fms M2L=%.3fms L2L=%.3fms\n",
                k.real(), milliseconds[0], milliseconds[1],
                milliseconds[2], milliseconds[3],
                milliseconds[4]);
            for (cudaEvent_t& event : far_events)
                CUDA_CHECK(cudaEventDestroy(event));
        }
    };

    const size_t field_stride = (size_t)n_nodes * L;
    const double* second_local_x_re = d_local_re;
    const double* second_local_x_im = d_local_im;
    const double* second_local_y_re = d_local2_re;
    const double* second_local_y_im = d_local2_im;
    const double* second_local_z_re = d_local3_re;
    const double* second_local_z_im = d_local3_im;
    if (share_far) {
        evaluate_pair_far();
        if (!pair_local_storage_fp32) {
            second_local_x_re =
                d_pair_local_re + 3 * field_stride;
            second_local_x_im =
                d_pair_local_im + 3 * field_stride;
            second_local_y_re =
                d_pair_local_re + 4 * field_stride;
            second_local_y_im =
                d_pair_local_im + 4 * field_stride;
            second_local_z_re =
                d_pair_local_re + 5 * field_stride;
            second_local_z_im =
                d_pair_local_im + 5 * field_stride;
        }
        if (profile) {
            CUDA_CHECK(cudaEventRecord(profile_events[1]));
            CUDA_CHECK(cudaEventRecord(profile_events[2]));
            CUDA_CHECK(cudaEventRecord(profile_events[3]));
        }
    } else {
        upload_three(
            first_x_re, first_x_im,
            first_y_re, first_y_im,
            first_z_re, first_z_im);
        evaluate_batch3_far_uploaded();
        if (profile)
            CUDA_CHECK(cudaEventRecord(profile_events[1]));
        if (share_l2p) {
            const size_t field_bytes =
                field_stride * sizeof(double);
            const double* local_re[3] = {
                d_local_re, d_local2_re, d_local3_re
            };
            const double* local_im[3] = {
                d_local_im, d_local2_im, d_local3_im
            };
            for (int component = 0; component < 3; component++) {
                CUDA_CHECK(cudaMemcpyAsync(
                    d_pair_local_re + component * field_stride,
                    local_re[component], field_bytes,
                    cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpyAsync(
                    d_pair_local_im + component * field_stride,
                    local_im[component], field_bytes,
                    cudaMemcpyDeviceToDevice));
            }
        } else {
            evaluate_l2p(
                d_grad_re, d_grad_im, d_hess_re, d_hess_im);
        }
        if (profile)
            CUDA_CHECK(cudaEventRecord(profile_events[2]));
        upload_three(
            second_x_re, second_x_im,
            second_y_re, second_y_im,
            second_z_re, second_z_im);
        evaluate_batch3_far_uploaded();
        if (profile)
            CUDA_CHECK(cudaEventRecord(profile_events[3]));
    }
    if (share_l2p) {
        bool l2p_fp32_default = false;
#ifdef BEM_DEFAULT_FMM_NEAR_FP32
        l2p_fp32_default = true;
#endif
        const bool request_l2p_fp32 =
            bem_env_flag_enabled(
                "BEM_FMM_L2P_FP32", l2p_fp32_default);
        double* outputs[] = {
            d_grad_re, d_grad_im, d_hess_re, d_hess_im,
            d_grad2_re, d_grad2_im, d_hess2_re, d_hess2_im
        };
        for (double* output : outputs)
            CUDA_CHECK(cudaMemsetAsync(output, 0, output_bytes));
        int l2p_threads = bem_env_int(
            "BEM_FMM_L2P_PAIR_THREADS", 256);
        if (l2p_threads != 64 && l2p_threads != 128 &&
            l2p_threads != 256)
            l2p_threads = 256;
        if (n_leaves > 0) {
            const size_t shared_l2p_bytes =
                (size_t)12 * L * sizeof(double);
            bool cache_l2p =
                bem_env_flag_enabled("BEM_FMM_PAIR_L2P_SHARED", false);
            if (pair_local_storage_fp32)
                cache_l2p = false;
            if (cache_l2p) {
                int device = 0;
                int max_shared_bytes = 0;
                CUDA_CHECK(cudaGetDevice(&device));
                CUDA_CHECK(cudaDeviceGetAttribute(
                    &max_shared_bytes,
                    cudaDevAttrMaxSharedMemoryPerBlockOptin,
                    device));
                cache_l2p =
                    shared_l2p_bytes <=
                    (size_t)max_shared_bytes;
                if (cache_l2p) {
                    const bool cache_phase =
                        d_phase_cache_fp32 != nullptr;
                    const bool l2p_fp32 =
                        cache_phase && request_l2p_fp32;
                    const cudaError_t attribute_status =
                        l2p_fp32
                        ? cudaFuncSetAttribute(
                            l2p_vector_actions_pair_batch3_kernel<
                                true, true, float, double>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            max_shared_bytes)
                        : cache_phase
                            ? cudaFuncSetAttribute(
                            l2p_vector_actions_pair_batch3_kernel<
                                true, true, double, double>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            max_shared_bytes)
                            : cudaFuncSetAttribute(
                                l2p_vector_actions_pair_batch3_kernel<
                                    true, false, double, double>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                max_shared_bytes);
                    if (attribute_status != cudaSuccess) {
                        cudaGetLastError();
                        cache_l2p = false;
                    }
                }
            }
#define LAUNCH_PAIR_L2P(                                             \
    CACHE_LOCAL, CACHE_PHASE, COMPUTE_TYPE, LOCAL_TYPE, SHARED_BYTES,\
    FIRST_LOCAL_RE, FIRST_LOCAL_IM,                                  \
    SECOND_X_RE, SECOND_X_IM, SECOND_Y_RE, SECOND_Y_IM,              \
    SECOND_Z_RE, SECOND_Z_IM)                                        \
            l2p_vector_actions_pair_batch3_kernel<                   \
                CACHE_LOCAL, CACHE_PHASE, COMPUTE_TYPE, LOCAL_TYPE>  \
                <<<n_leaves, l2p_threads, SHARED_BYTES>>>(           \
                    d_tgt_pts, d_dirs_cached, d_weights_cached,       \
                    d_phase_cache_fp32,                              \
                    FIRST_LOCAL_RE, FIRST_LOCAL_IM,                   \
                    SECOND_X_RE, SECOND_X_IM,                         \
                    SECOND_Y_RE, SECOND_Y_IM,                         \
                    SECOND_Z_RE, SECOND_Z_IM,                         \
                    n_nodes * L, Nt,                                  \
                    k.real(), k.imag(),                               \
                    prefactor.real(), prefactor.imag(),               \
                    ik_val.real(), ik_val.imag(),                     \
                    negative_k_squared.real(),                        \
                    negative_k_squared.imag(),                        \
                    d_grad_re, d_grad_im,                             \
                    d_hess_re, d_hess_im,                             \
                    d_grad2_re, d_grad2_im,                           \
                    d_hess2_re, d_hess2_im,                           \
                    d_leaf_idx_cached,                               \
                    d_tgt_id_offsets_cached,                         \
                    d_tgt_ids_cached,                                \
                    d_node_centers_cached,                           \
                    L, n_leaves)
            const bool cache_phase =
                d_phase_cache_fp32 != nullptr;
            const bool l2p_fp32 =
                cache_phase && request_l2p_fp32;
            const bool warp_per_target =
                pair_local_storage_fp32 && l2p_fp32 &&
                d_dirs_fp32_cached != nullptr &&
                d_weights_fp32_cached != nullptr &&
                bem_env_flag_enabled(
                    "BEM_FMM_L2P_WARP_PER_TARGET",
                    d_l2p_phase_cache_fp32 == nullptr);
            if (warp_per_target)
                l2p_vector_actions_pair_batch3_warp_fp32_kernel
                    <<<n_leaves, l2p_threads>>>(
                        d_dirs_fp32_cached,
                        d_weights_fp32_cached,
                        d_phase_cache_fp32,
                        d_pair_local_re_fp32,
                        d_pair_local_im_fp32,
                        d_pair_local_re_fp32 + 3 * field_stride,
                        d_pair_local_im_fp32 + 3 * field_stride,
                        d_pair_local_re_fp32 + 4 * field_stride,
                        d_pair_local_im_fp32 + 4 * field_stride,
                        d_pair_local_re_fp32 + 5 * field_stride,
                        d_pair_local_im_fp32 + 5 * field_stride,
                        n_nodes * L,
                        static_cast<float>(
                            prefactor.real() * ik_val.real() -
                            prefactor.imag() * ik_val.imag()),
                        static_cast<float>(
                            prefactor.real() * ik_val.imag() +
                            prefactor.imag() * ik_val.real()),
                        static_cast<float>(
                            prefactor.real() *
                                negative_k_squared.real() -
                            prefactor.imag() *
                                negative_k_squared.imag()),
                        static_cast<float>(
                            prefactor.real() *
                                negative_k_squared.imag() +
                            prefactor.imag() *
                                negative_k_squared.real()),
                        d_grad_re, d_grad_im,
                        d_hess_re, d_hess_im,
                        d_grad2_re, d_grad2_im,
                        d_hess2_re, d_hess2_im,
                        d_leaf_idx_cached,
                        d_tgt_id_offsets_cached,
                        d_tgt_ids_cached,
                        L, n_leaves);
            else if (pair_local_storage_fp32 && l2p_fp32 &&
                d_l2p_phase_cache_fp32 != nullptr &&
                d_dirs_fp32_cached != nullptr &&
                d_weights_fp32_cached != nullptr)
                l2p_vector_actions_pair_batch3_kernel<
                    false, true, float, float, float, true>
                    <<<n_leaves, l2p_threads>>>(
                        d_tgt_pts,
                        d_dirs_fp32_cached,
                        d_weights_fp32_cached,
                        d_l2p_phase_cache_fp32,
                        d_pair_local_re_fp32,
                        d_pair_local_im_fp32,
                        d_pair_local_re_fp32 + 3 * field_stride,
                        d_pair_local_im_fp32 + 3 * field_stride,
                        d_pair_local_re_fp32 + 4 * field_stride,
                        d_pair_local_im_fp32 + 4 * field_stride,
                        d_pair_local_re_fp32 + 5 * field_stride,
                        d_pair_local_im_fp32 + 5 * field_stride,
                        n_nodes * L, Nt,
                        k.real(), k.imag(),
                        prefactor.real(), prefactor.imag(),
                        ik_val.real(), ik_val.imag(),
                        negative_k_squared.real(),
                        negative_k_squared.imag(),
                        d_grad_re, d_grad_im,
                        d_hess_re, d_hess_im,
                        d_grad2_re, d_grad2_im,
                        d_hess2_re, d_hess2_im,
                        d_leaf_idx_cached,
                        d_tgt_id_offsets_cached,
                        d_tgt_ids_cached,
                        d_node_centers_cached,
                        L, n_leaves);
            else if (pair_local_storage_fp32 && l2p_fp32)
                LAUNCH_PAIR_L2P(
                    false, true, float, float, 0,
                    d_pair_local_re_fp32,
                    d_pair_local_im_fp32,
                    d_pair_local_re_fp32 + 3 * field_stride,
                    d_pair_local_im_fp32 + 3 * field_stride,
                    d_pair_local_re_fp32 + 4 * field_stride,
                    d_pair_local_im_fp32 + 4 * field_stride,
                    d_pair_local_re_fp32 + 5 * field_stride,
                    d_pair_local_im_fp32 + 5 * field_stride);
            else if (pair_local_storage_fp32 && cache_phase)
                LAUNCH_PAIR_L2P(
                    false, true, double, float, 0,
                    d_pair_local_re_fp32,
                    d_pair_local_im_fp32,
                    d_pair_local_re_fp32 + 3 * field_stride,
                    d_pair_local_im_fp32 + 3 * field_stride,
                    d_pair_local_re_fp32 + 4 * field_stride,
                    d_pair_local_im_fp32 + 4 * field_stride,
                    d_pair_local_re_fp32 + 5 * field_stride,
                    d_pair_local_im_fp32 + 5 * field_stride);
            else if (pair_local_storage_fp32)
                LAUNCH_PAIR_L2P(
                    false, false, double, float, 0,
                    d_pair_local_re_fp32,
                    d_pair_local_im_fp32,
                    d_pair_local_re_fp32 + 3 * field_stride,
                    d_pair_local_im_fp32 + 3 * field_stride,
                    d_pair_local_re_fp32 + 4 * field_stride,
                    d_pair_local_im_fp32 + 4 * field_stride,
                    d_pair_local_re_fp32 + 5 * field_stride,
                    d_pair_local_im_fp32 + 5 * field_stride);
            else if (cache_l2p && l2p_fp32)
                LAUNCH_PAIR_L2P(
                    true, true, float, double, shared_l2p_bytes,
                    d_pair_local_re, d_pair_local_im,
                    second_local_x_re, second_local_x_im,
                    second_local_y_re, second_local_y_im,
                    second_local_z_re, second_local_z_im);
            else if (cache_l2p && cache_phase)
                LAUNCH_PAIR_L2P(
                    true, true, double, double, shared_l2p_bytes,
                    d_pair_local_re, d_pair_local_im,
                    second_local_x_re, second_local_x_im,
                    second_local_y_re, second_local_y_im,
                    second_local_z_re, second_local_z_im);
            else if (cache_l2p)
                LAUNCH_PAIR_L2P(
                    true, false, double, double, shared_l2p_bytes,
                    d_pair_local_re, d_pair_local_im,
                    second_local_x_re, second_local_x_im,
                    second_local_y_re, second_local_y_im,
                    second_local_z_re, second_local_z_im);
            else if (l2p_fp32)
                LAUNCH_PAIR_L2P(
                    false, true, float, double, 0,
                    d_pair_local_re, d_pair_local_im,
                    second_local_x_re, second_local_x_im,
                    second_local_y_re, second_local_y_im,
                    second_local_z_re, second_local_z_im);
            else if (cache_phase)
                LAUNCH_PAIR_L2P(
                    false, true, double, double, 0,
                    d_pair_local_re, d_pair_local_im,
                    second_local_x_re, second_local_x_im,
                    second_local_y_re, second_local_y_im,
                    second_local_z_re, second_local_z_im);
            else
                LAUNCH_PAIR_L2P(
                    false, false, double, double, 0,
                    d_pair_local_re, d_pair_local_im,
                    second_local_x_re, second_local_x_im,
                    second_local_y_re, second_local_y_im,
                    second_local_z_re, second_local_z_im);
#undef LAUNCH_PAIR_L2P
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        evaluate_l2p(
            d_grad2_re, d_grad2_im, d_hess2_re, d_hess2_im);
    }
    if (profile)
        CUDA_CHECK(cudaEventRecord(profile_events[4]));

    if (n_leaves > 0) {
        launch_p2p_vector_actions_pair_batch3_leaf(
            d_tgt_pts, d_src_pts,
            first_x_re, first_x_im,
            first_y_re, first_y_im,
            first_z_re, first_z_im,
            second_x_re, second_x_im,
            second_y_re, second_y_im,
            second_z_re, second_z_im,
            d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_src_id_offsets_cached, d_src_ids_cached,
            d_leaf_near_offsets_cached, d_leaf_near_ids_cached,
            d_leaf_near_source_offsets_cached,
            d_leaf_near_source_ids_cached,
            n_leaves, k.real(), k.imag(),
            d_grad_re, d_grad_im,
            d_hess_re, d_hess_im,
            d_grad2_re, d_grad2_im,
            d_hess2_re, d_hess2_im,
            d_tgt_pts_fp32, d_src_pts_fp32,
            d_pair_charges_fp32, Ns,
            near_field_fp32);
        CUDA_CHECK(cudaGetLastError());
    }

    if (profile) {
        CUDA_CHECK(cudaEventRecord(profile_events[5]));
        CUDA_CHECK(cudaEventSynchronize(profile_events[5]));
        float milliseconds[5] = {};
        for (int phase = 0; phase < 5; phase++) {
            CUDA_CHECK(cudaEventElapsedTime(
                &milliseconds[phase],
                profile_events[phase],
                profile_events[phase + 1]));
        }
        if (share_far) {
            std::printf(
                "  [FMM vector-pair device profile] "
                "shared-far=%.3fms shared-L2P=%.3fms "
                "shared-P2P=%.3fms\n",
                milliseconds[0], milliseconds[3],
                milliseconds[4]);
        } else {
            std::printf(
                "  [FMM vector-pair device profile] "
                "first-far=%.3fms first-save/L2P=%.3fms "
                "second-far=%.3fms shared/second-L2P=%.3fms "
                "shared-P2P=%.3fms\n",
                milliseconds[0], milliseconds[1],
                milliseconds[2], milliseconds[3],
                milliseconds[4]);
        }
        for (cudaEvent_t& event : profile_events)
            CUDA_CHECK(cudaEventDestroy(event));
    } else {
        CUDA_CHECK(cudaStreamSynchronize(0));
    }
}

bool HelmholtzFMM::strict_vector_pair_available() const
{
    const bool primary_fp64 =
        pair_workspace_fields >= 6 &&
        d_pair_multi_re != nullptr && d_pair_multi_im != nullptr &&
        d_pair_local_re != nullptr && d_pair_local_im != nullptr;
    const bool dedicated_fp64 =
        d_strict_pair_multi_re != nullptr &&
        d_strict_pair_multi_im != nullptr &&
        d_strict_pair_local_re != nullptr &&
        d_strict_pair_local_im != nullptr;
    return primary_fp64 || dedicated_fp64;
}

void HelmholtzFMM::evaluate_vector_actions_pair_batch3_device_strict(
    const double* first_x_re,
    const double* first_x_im,
    const double* first_y_re,
    const double* first_y_im,
    const double* first_z_re,
    const double* first_z_im,
    const double* second_x_re,
    const double* second_x_im,
    const double* second_y_re,
    const double* second_y_im,
    const double* second_z_re,
    const double* second_z_im)
{
    if (!strict_vector_pair_available())
        throw std::runtime_error(
            "strict paired vector actions require FP64 pair buffers");

    const bool use_dedicated =
        d_strict_pair_multi_re != nullptr;
    double* saved_multi_re = d_pair_multi_re;
    double* saved_multi_im = d_pair_multi_im;
    double* saved_local_re = d_pair_local_re;
    double* saved_local_im = d_pair_local_im;
    float* saved_multi_re_fp32 = d_pair_multi_re_fp32;
    float* saved_multi_im_fp32 = d_pair_multi_im_fp32;
    float* saved_local_re_fp32 = d_pair_local_re_fp32;
    float* saved_local_im_fp32 = d_pair_local_im_fp32;
    const bool saved_force_pair_fp64 = force_pair_fp64;
    const bool saved_near_field_fp32 = near_field_fp32;

    if (use_dedicated) {
        d_pair_multi_re = d_strict_pair_multi_re;
        d_pair_multi_im = d_strict_pair_multi_im;
        d_pair_local_re = d_strict_pair_local_re;
        d_pair_local_im = d_strict_pair_local_im;
        d_pair_multi_re_fp32 = nullptr;
        d_pair_multi_im_fp32 = nullptr;
        d_pair_local_re_fp32 = nullptr;
        d_pair_local_im_fp32 = nullptr;
    }
    force_pair_fp64 = true;
    near_field_fp32 = false;

    const auto restore = [&]() {
        force_pair_fp64 = saved_force_pair_fp64;
        near_field_fp32 = saved_near_field_fp32;
        if (use_dedicated) {
            d_pair_multi_re = saved_multi_re;
            d_pair_multi_im = saved_multi_im;
            d_pair_local_re = saved_local_re;
            d_pair_local_im = saved_local_im;
            d_pair_multi_re_fp32 = saved_multi_re_fp32;
            d_pair_multi_im_fp32 = saved_multi_im_fp32;
            d_pair_local_re_fp32 = saved_local_re_fp32;
            d_pair_local_im_fp32 = saved_local_im_fp32;
        }
    };
    try {
        evaluate_vector_actions_pair_batch3_device(
            first_x_re, first_x_im,
            first_y_re, first_y_im,
            first_z_re, first_z_im,
            second_x_re, second_x_im,
            second_y_re, second_y_im,
            second_z_re, second_z_im);
    } catch (...) {
        restore();
        throw;
    }
    restore();
}

void HelmholtzFMM::evaluate_vector_actions_quad_batch3_device(
    const double* const charges_re[12],
    const double* const charges_im[12])
{
    if (!vector_actions_quad_available())
        throw std::runtime_error(
            "four-field device vector actions require a 12-channel workspace");
    for (int field = 0; field < 12; field++) {
        if (charges_re[field] == nullptr || charges_im[field] == nullptr)
            throw std::invalid_argument("null four-field FMM charge buffer");
    }

    const bool profile =
        bem_env_flag_enabled("BEM_FMM_PROFILE_FOUR_FIELD");
    cudaEvent_t events[5] = {};
    if (profile) {
        for (cudaEvent_t& event : events)
            CUDA_CHECK(cudaEventCreate(&event));
        CUDA_CHECK(cudaEventRecord(events[0]));
    }

    const int n_leaves = static_cast<int>(leaf_info.size());
    const int field_stride = n_nodes * L;
    const size_t output_bytes = (size_t)3 * Nt * sizeof(double);
    const bool local_fp32 =
        d_pair_local_re_fp32 != nullptr &&
        d_pair_local_im_fp32 != nullptr;
    const bool multi_fp32 =
        d_pair_multi_re == nullptr && d_pair_multi_im == nullptr &&
        d_pair_multi_re_fp32 != nullptr &&
        d_pair_multi_im_fp32 != nullptr;
    const bool transfer_fp32 =
        d_transfer_re_fp32 != nullptr && d_transfer_im_fp32 != nullptr;
    if (d_pair_charges_fp32 == nullptr)
        throw std::runtime_error(
            "four-field FMM requires the packed FP32 charge cache");

    const int pack_block = 256;
    const int pack_grid = (Ns + pack_block - 1) / pack_block;
    pack_pair_charges_fp32_kernel<<<pack_grid, pack_block>>>(
        charges_re[0], charges_im[0],
        charges_re[1], charges_im[1],
        charges_re[2], charges_im[2],
        charges_re[3], charges_im[3],
        charges_re[4], charges_im[4],
        charges_re[5], charges_im[5],
        d_pair_charges_fp32, Ns);
    pack_pair_charges_fp32_kernel<<<pack_grid, pack_block>>>(
        charges_re[6], charges_im[6],
        charges_re[7], charges_im[7],
        charges_re[8], charges_im[8],
        charges_re[9], charges_im[9],
        charges_re[10], charges_im[10],
        charges_re[11], charges_im[11],
        d_pair_charges_fp32 + (size_t)12 * Ns, Ns);
    CUDA_CHECK(cudaGetLastError());

    const size_t coefficient_count = (size_t)12 * field_stride;
    if (multi_fp32) {
        CUDA_CHECK(cudaMemsetAsync(
            d_pair_multi_re_fp32, 0,
            coefficient_count * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(
            d_pair_multi_im_fp32, 0,
            coefficient_count * sizeof(float)));
    } else {
        CUDA_CHECK(cudaMemsetAsync(
            d_pair_multi_re, 0,
            coefficient_count * sizeof(double)));
        CUDA_CHECK(cudaMemsetAsync(
            d_pair_multi_im, 0,
            coefficient_count * sizeof(double)));
    }
    if (local_fp32) {
        CUDA_CHECK(cudaMemsetAsync(
            d_pair_local_re_fp32, 0,
            coefficient_count * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(
            d_pair_local_im_fp32, 0,
            coefficient_count * sizeof(float)));
    } else {
        CUDA_CHECK(cudaMemsetAsync(
            d_pair_local_re, 0,
            coefficient_count * sizeof(double)));
        CUDA_CHECK(cudaMemsetAsync(
            d_pair_local_im, 0,
            coefficient_count * sizeof(double)));
    }

    int threads = bem_env_int("BEM_FMM_FOUR_FIELD_THREADS", 256);
    if (threads != 64 && threads != 128 &&
        threads != 256 && threads != 512)
        threads = 256;
    threads = std::min(threads, L);
    if (n_leaves > 0) {
        if (multi_fp32) {
            p2m_kernel_batch12_packed_fp32<float>
                <<<n_leaves, threads>>>(
                    d_src_pts, d_pair_charges_fp32, Ns,
                    d_phase_cache_fp32, d_dirs_cached,
                    k.real(), k.imag(),
                    d_pair_multi_re_fp32,
                    d_pair_multi_im_fp32,
                    field_stride, d_leaf_idx_cached,
                    d_src_id_offsets_cached, d_src_ids_cached,
                    d_node_centers_cached, L, n_leaves);
        } else {
            p2m_kernel_batch12_packed_fp32<double>
                <<<n_leaves, threads>>>(
                    d_src_pts, d_pair_charges_fp32, Ns,
                    d_phase_cache_fp32, d_dirs_cached,
                    k.real(), k.imag(),
                    d_pair_multi_re, d_pair_multi_im,
                    field_stride, d_leaf_idx_cached,
                    d_src_id_offsets_cached, d_src_ids_cached,
                    d_node_centers_cached, L, n_leaves);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    for (int level = tree.max_level - 1; level >= 1; level--) {
        if (level >= (int)m2m_level_info.size() ||
            m2m_level_info[level].count <= 0)
            continue;
        const int offset = m2m_level_info[level].offset;
        const int count = m2m_level_info[level].count;
        if (multi_fp32) {
            m2m_kernel_batch6_flat<12, float><<<count, threads>>>(
                d_m2m_parent, d_m2m_child,
                d_m2m_shift_re, d_m2m_shift_im,
                d_pair_multi_re_fp32, d_pair_multi_im_fp32,
                field_stride, L, count, offset);
        } else {
            m2m_kernel_batch6_flat<12, double><<<count, threads>>>(
                d_m2m_parent, d_m2m_child,
                d_m2m_shift_re, d_m2m_shift_im,
                d_pair_multi_re, d_pair_multi_im,
                field_stride, L, count, offset);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    if (profile)
        CUDA_CHECK(cudaEventRecord(events[1]));

    for (int level = 1; level <= tree.max_level; level++) {
        if (level >= (int)m2l_row_level_info.size() ||
            m2l_row_level_info[level].count <= 0)
            continue;
        const int offset = m2l_row_level_info[level].offset;
        const int count = m2l_row_level_info[level].count;
#define LAUNCH_QUAD_M2L(COMPUTE, TRANSFER, MULTI, LOCAL,             \
                        TRANSFER_RE, TRANSFER_IM, MULTI_RE, MULTI_IM,\
                        LOCAL_RE, LOCAL_IM)                           \
        do {                                                          \
            m2l_kernel_batch6_flat_target_rows<6, COMPUTE, TRANSFER, \
                MULTI, LOCAL><<<count, threads>>>(                    \
                    d_m2l_row_target, d_m2l_row_start, d_m2l_row_end, \
                    d_m2l_src, d_m2l_tidx, TRANSFER_RE, TRANSFER_IM,  \
                    MULTI_RE, MULTI_IM, LOCAL_RE, LOCAL_IM,            \
                    0, field_stride, L, count, offset);               \
            m2l_kernel_batch6_flat_target_rows<6, COMPUTE, TRANSFER, \
                MULTI, LOCAL><<<count, threads>>>(                    \
                    d_m2l_row_target, d_m2l_row_start, d_m2l_row_end, \
                    d_m2l_src, d_m2l_tidx, TRANSFER_RE, TRANSFER_IM,  \
                    MULTI_RE, MULTI_IM, LOCAL_RE, LOCAL_IM,            \
                    6, field_stride, L, count, offset);               \
        } while (0)
        if (transfer_fp32 && multi_fp32 && local_fp32)
            LAUNCH_QUAD_M2L(
                float, float, float, float,
                d_transfer_re_fp32, d_transfer_im_fp32,
                d_pair_multi_re_fp32, d_pair_multi_im_fp32,
                d_pair_local_re_fp32, d_pair_local_im_fp32);
        else if (multi_fp32 && local_fp32)
            LAUNCH_QUAD_M2L(
                float, double, float, float,
                d_transfer_re, d_transfer_im,
                d_pair_multi_re_fp32, d_pair_multi_im_fp32,
                d_pair_local_re_fp32, d_pair_local_im_fp32);
        else if (transfer_fp32 && multi_fp32)
            LAUNCH_QUAD_M2L(
                float, float, float, double,
                d_transfer_re_fp32, d_transfer_im_fp32,
                d_pair_multi_re_fp32, d_pair_multi_im_fp32,
                d_pair_local_re, d_pair_local_im);
        else if (multi_fp32)
            LAUNCH_QUAD_M2L(
                float, double, float, double,
                d_transfer_re, d_transfer_im,
                d_pair_multi_re_fp32, d_pair_multi_im_fp32,
                d_pair_local_re, d_pair_local_im);
        else if (local_fp32)
            LAUNCH_QUAD_M2L(
                float, double, double, float,
                d_transfer_re, d_transfer_im,
                d_pair_multi_re, d_pair_multi_im,
                d_pair_local_re_fp32, d_pair_local_im_fp32);
        else
            LAUNCH_QUAD_M2L(
                double, double, double, double,
                d_transfer_re, d_transfer_im,
                d_pair_multi_re, d_pair_multi_im,
                d_pair_local_re, d_pair_local_im);
#undef LAUNCH_QUAD_M2L
        CUDA_CHECK(cudaGetLastError());
    }
    for (int level = 2; level <= tree.max_level; level++) {
        if (level >= (int)l2l_level_info.size() ||
            l2l_level_info[level].count <= 0)
            continue;
        const int offset = l2l_level_info[level].offset;
        const int count = l2l_level_info[level].count;
        if (local_fp32) {
            l2l_kernel_batch6_flat<12, float><<<count, threads>>>(
                d_l2l_parent, d_l2l_child,
                d_l2l_shift_re, d_l2l_shift_im,
                d_pair_local_re_fp32, d_pair_local_im_fp32,
                field_stride, L, count, offset);
        } else {
            l2l_kernel_batch6_flat<12, double><<<count, threads>>>(
                d_l2l_parent, d_l2l_child,
                d_l2l_shift_re, d_l2l_shift_im,
                d_pair_local_re, d_pair_local_im,
                field_stride, L, count, offset);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    if (profile)
        CUDA_CHECK(cudaEventRecord(events[2]));

    double* outputs[] = {
        d_grad_re, d_grad_im, d_hess_re, d_hess_im,
        d_grad2_re, d_grad2_im, d_hess2_re, d_hess2_im,
        d_grad3_re, d_grad3_im, d_hess3_re, d_hess3_im,
        d_grad4_re, d_grad4_im, d_hess4_re, d_hess4_im
    };
    for (double* output : outputs)
        CUDA_CHECK(cudaMemsetAsync(output, 0, output_bytes));

    const cdouble ik_val = cdouble(0, 1) * k;
    const cdouble prefactor = ik_val / (16.0 * M_PI * M_PI);
    const cdouble negative_k_squared = -k * k;
    auto launch_pair_l2p = [&](int field_offset,
                               double* grad_first_re,
                               double* grad_first_im,
                               double* hess_first_re,
                               double* hess_first_im,
                               double* grad_second_re,
                               double* grad_second_im,
                               double* hess_second_re,
                               double* hess_second_im) {
        if (n_leaves <= 0)
            return;
        if (local_fp32 && d_dirs_fp32_cached != nullptr &&
            d_weights_fp32_cached != nullptr &&
            d_phase_cache_fp32 != nullptr) {
            l2p_vector_actions_pair_batch3_warp_fp32_kernel
                <<<n_leaves, 256>>>(
                    d_dirs_fp32_cached, d_weights_fp32_cached,
                    d_phase_cache_fp32,
                    d_pair_local_re_fp32 +
                        (size_t)field_offset * field_stride,
                    d_pair_local_im_fp32 +
                        (size_t)field_offset * field_stride,
                    d_pair_local_re_fp32 +
                        (size_t)(field_offset + 3) * field_stride,
                    d_pair_local_im_fp32 +
                        (size_t)(field_offset + 3) * field_stride,
                    d_pair_local_re_fp32 +
                        (size_t)(field_offset + 4) * field_stride,
                    d_pair_local_im_fp32 +
                        (size_t)(field_offset + 4) * field_stride,
                    d_pair_local_re_fp32 +
                        (size_t)(field_offset + 5) * field_stride,
                    d_pair_local_im_fp32 +
                        (size_t)(field_offset + 5) * field_stride,
                    field_stride,
                    static_cast<float>(
                        prefactor.real() * ik_val.real() -
                        prefactor.imag() * ik_val.imag()),
                    static_cast<float>(
                        prefactor.real() * ik_val.imag() +
                        prefactor.imag() * ik_val.real()),
                    static_cast<float>(
                        prefactor.real() * negative_k_squared.real() -
                        prefactor.imag() * negative_k_squared.imag()),
                    static_cast<float>(
                        prefactor.real() * negative_k_squared.imag() +
                        prefactor.imag() * negative_k_squared.real()),
                    grad_first_re, grad_first_im,
                    hess_first_re, hess_first_im,
                    grad_second_re, grad_second_im,
                    hess_second_re, hess_second_im,
                    d_leaf_idx_cached, d_tgt_id_offsets_cached,
                    d_tgt_ids_cached, L, n_leaves);
        } else {
            l2p_vector_actions_pair_batch3_kernel<
                false, true, double, double>
                <<<n_leaves, 256>>>(
                    d_tgt_pts, d_dirs_cached, d_weights_cached,
                    d_phase_cache_fp32,
                    d_pair_local_re +
                        (size_t)field_offset * field_stride,
                    d_pair_local_im +
                        (size_t)field_offset * field_stride,
                    d_pair_local_re +
                        (size_t)(field_offset + 3) * field_stride,
                    d_pair_local_im +
                        (size_t)(field_offset + 3) * field_stride,
                    d_pair_local_re +
                        (size_t)(field_offset + 4) * field_stride,
                    d_pair_local_im +
                        (size_t)(field_offset + 4) * field_stride,
                    d_pair_local_re +
                        (size_t)(field_offset + 5) * field_stride,
                    d_pair_local_im +
                        (size_t)(field_offset + 5) * field_stride,
                    field_stride, Nt,
                    k.real(), k.imag(),
                    prefactor.real(), prefactor.imag(),
                    ik_val.real(), ik_val.imag(),
                    negative_k_squared.real(),
                    negative_k_squared.imag(),
                    grad_first_re, grad_first_im,
                    hess_first_re, hess_first_im,
                    grad_second_re, grad_second_im,
                    hess_second_re, hess_second_im,
                    d_leaf_idx_cached, d_tgt_id_offsets_cached,
                    d_tgt_ids_cached, d_node_centers_cached,
                    L, n_leaves);
        }
        CUDA_CHECK(cudaGetLastError());
    };
    launch_pair_l2p(
        0, d_grad_re, d_grad_im, d_hess_re, d_hess_im,
        d_grad2_re, d_grad2_im, d_hess2_re, d_hess2_im);
    launch_pair_l2p(
        6, d_grad3_re, d_grad3_im, d_hess3_re, d_hess3_im,
        d_grad4_re, d_grad4_im, d_hess4_re, d_hess4_im);
    if (profile)
        CUDA_CHECK(cudaEventRecord(events[3]));

    if (n_leaves > 0) {
        auto launch_pair_p2p = [&](int field_offset,
                                   int packed_offset,
                                   double* grad_first_re,
                                   double* grad_first_im,
                                   double* hess_first_re,
                                   double* hess_first_im,
                                   double* grad_second_re,
                                   double* grad_second_im,
                                   double* hess_second_re,
                                   double* hess_second_im) {
            launch_p2p_vector_actions_pair_batch3_leaf(
                d_tgt_pts, d_src_pts,
                charges_re[field_offset], charges_im[field_offset],
                charges_re[field_offset + 1], charges_im[field_offset + 1],
                charges_re[field_offset + 2], charges_im[field_offset + 2],
                charges_re[field_offset + 3], charges_im[field_offset + 3],
                charges_re[field_offset + 4], charges_im[field_offset + 4],
                charges_re[field_offset + 5], charges_im[field_offset + 5],
                d_tgt_id_offsets_cached, d_tgt_ids_cached,
                d_src_id_offsets_cached, d_src_ids_cached,
                d_leaf_near_offsets_cached, d_leaf_near_ids_cached,
                d_leaf_near_source_offsets_cached,
                d_leaf_near_source_ids_cached,
                n_leaves, k.real(), k.imag(),
                grad_first_re, grad_first_im,
                hess_first_re, hess_first_im,
                grad_second_re, grad_second_im,
                hess_second_re, hess_second_im,
                d_tgt_pts_fp32, d_src_pts_fp32,
                d_pair_charges_fp32 + (size_t)packed_offset * Ns,
                Ns, near_field_fp32);
        };
        launch_pair_p2p(
            0, 0, d_grad_re, d_grad_im, d_hess_re, d_hess_im,
            d_grad2_re, d_grad2_im, d_hess2_re, d_hess2_im);
        launch_pair_p2p(
            6, 12, d_grad3_re, d_grad3_im, d_hess3_re, d_hess3_im,
            d_grad4_re, d_grad4_im, d_hess4_re, d_hess4_im);
        CUDA_CHECK(cudaGetLastError());
    }
    if (profile) {
        CUDA_CHECK(cudaEventRecord(events[4]));
        CUDA_CHECK(cudaEventSynchronize(events[4]));
        float milliseconds[4] = {};
        for (int phase = 0; phase < 4; phase++)
            CUDA_CHECK(cudaEventElapsedTime(
                &milliseconds[phase], events[phase], events[phase + 1]));
        std::printf(
            "  [FMM four-field profile] upward=%.3fms "
            "M2L/L2L=%.3fms L2P=%.3fms P2P=%.3fms\n",
            milliseconds[0], milliseconds[1],
            milliseconds[2], milliseconds[3]);
        for (cudaEvent_t& event : events)
            CUDA_CHECK(cudaEventDestroy(event));
    } else {
        CUDA_CHECK(cudaStreamSynchronize(0));
    }
}

void HelmholtzFMM::evaluate_pot_grad(const cdouble* charges,
                                      cdouble* pot_result,
                                      cdouble* grad_result)
{
    int block = 256;
    int grid_src = (Ns + block - 1) / block;
    CUDA_CHECK(cudaMemcpy(d_complex_tmp1, charges, Ns * sizeof(double2), cudaMemcpyHostToDevice));
    split_complex_kernel<<<grid_src, block>>>(d_complex_tmp1, d_charges_re, d_charges_im, Ns);
    CUDA_CHECK(cudaGetLastError());

    evaluate_pot_grad_uploaded();

    int grid_tgt = (Nt + block - 1) / block;
    pack_complex_kernel<<<grid_tgt, block>>>(d_result_re, d_result_im, d_complex_tmp1, Nt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(pot_result, d_complex_tmp1, Nt * sizeof(double2), cudaMemcpyDeviceToHost));

    int ngrad = Nt * 3;
    int grid_grad = (ngrad + block - 1) / block;
    pack_complex_kernel<<<grid_grad, block>>>(d_grad_re, d_grad_im, d_complex_tmp1, ngrad);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(grad_result, d_complex_tmp1, ngrad * sizeof(double2), cudaMemcpyDeviceToHost));
}

void HelmholtzFMM::evaluate_pot_grad_uploaded()
{
    int n_leaves = (int)leaf_info.size();

    CUDA_CHECK(cudaMemset(d_multi_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local_im, 0, n_nodes * L * sizeof(double)));

    double *d_node_centers = d_node_centers_cached;
    double *d_dirs = d_dirs_cached;
    double *d_weights = d_weights_cached;
    int *d_leaf_idx = d_leaf_idx_cached;
    int *d_src_id_offsets = d_src_id_offsets_cached;
    int *d_src_ids = d_src_ids_cached;
    int *d_tgt_id_offsets = d_tgt_id_offsets_cached;
    int *d_tgt_ids = d_tgt_ids_cached;

    int block_L = std::min(L, 256);

    if (n_leaves > 0) {
        p2m_kernel<<<n_leaves, block_L>>>(
            d_src_pts, d_charges_re, d_charges_im,
            d_dirs, k.real(), k.imag(),
            d_multi_re, d_multi_im,
            d_leaf_idx, d_src_id_offsets, d_src_ids,
            d_node_centers, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
    }

    for (int level = tree.max_level - 1; level >= 1; level--) {
        if (level < (int)m2m_level_info.size() && m2m_level_info[level].count > 0) {
            int off = m2m_level_info[level].offset;
            int cnt = m2m_level_info[level].count;
            m2m_kernel<<<cnt, block_L>>>(d_m2m_parent, d_m2m_child,
                d_m2m_shift_re, d_m2m_shift_im, d_multi_re, d_multi_im, L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (int level = 1; level <= tree.max_level; level++) {
        if (level < (int)m2l_level_info.size() && m2l_level_info[level].count > 0) {
            int off = m2l_level_info[level].offset;
            int cnt = m2l_level_info[level].count;
            m2l_kernel<<<cnt, block_L>>>(d_m2l_tgt, d_m2l_src, d_m2l_tidx,
                d_transfer_re, d_transfer_im, d_multi_re, d_multi_im,
                d_local_re, d_local_im, L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (int level = 2; level <= tree.max_level; level++) {
        if (level < (int)l2l_level_info.size() && l2l_level_info[level].count > 0) {
            int off = l2l_level_info[level].offset;
            int cnt = l2l_level_info[level].count;
            l2l_kernel<<<cnt, block_L>>>(d_l2l_parent, d_l2l_child,
                d_l2l_shift_re, d_l2l_shift_im,
                d_local_re, d_local_im, d_local_re, d_local_im, L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    cdouble ik_val = cdouble(0, 1) * k;
    cdouble prefactor = ik_val / (16.0 * M_PI * M_PI);

    CUDA_CHECK(cudaMemset(d_result_re, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result_im, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gx_re_tmp_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gx_im_tmp_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy_re_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy_im_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz_re_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz_im_cached, 0, Nt * sizeof(double)));

    if (n_leaves > 0) {
        l2p_kernel<<<n_leaves, 256>>>(
            d_tgt_pts, d_dirs, d_weights,
            d_local_re, d_local_im,
            k.real(), k.imag(), prefactor.real(), prefactor.imag(),
            d_result_re, d_result_im,
            d_leaf_idx, d_tgt_id_offsets, d_tgt_ids,
            d_node_centers, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());

        l2p_gradient_kernel<<<n_leaves, 256>>>(
            d_tgt_pts, d_dirs, d_weights,
            d_local_re, d_local_im,
            k.real(), k.imag(), prefactor.real(), prefactor.imag(),
            ik_val.real(), ik_val.imag(),
            d_gx_re_tmp_cached, d_gx_im_tmp_cached,
            d_gy_re_cached, d_gy_im_cached,
            d_gz_re_cached, d_gz_im_cached,
            d_leaf_idx, d_tgt_id_offsets, d_tgt_ids,
            d_node_centers, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
    }

    launch_p2p_pot_grad_leaf(
        d_tgt_pts, d_src_pts,
        d_charges_re, d_charges_im,
        d_tgt_id_offsets_cached, d_tgt_ids_cached,
        d_src_id_offsets_cached, d_src_ids_cached,
        d_leaf_near_offsets_cached, d_leaf_near_ids_cached,
        n_leaves, k.real(), k.imag(),
        d_result_re, d_result_im,
        d_gx_re_tmp_cached, d_gx_im_tmp_cached,
        d_gy_re_cached, d_gy_im_cached,
        d_gz_re_cached, d_gz_im_cached);
    CUDA_CHECK(cudaGetLastError());

    int repack_block = 256;
    int repack_grid = (Nt + repack_block - 1) / repack_block;
    repack_gradient_kernel<<<repack_grid, repack_block>>>(
        d_gx_re_tmp_cached, d_gx_im_tmp_cached,
        d_gy_re_cached, d_gy_im_cached,
        d_gz_re_cached, d_gz_im_cached,
        d_grad_re, d_grad_im, Nt);
    CUDA_CHECK(cudaGetLastError());
}

void HelmholtzFMM::evaluate_batch2(
    const cdouble* charges1, const cdouble* charges2,
    cdouble* result1, cdouble* result2)
{
    int block = 256;
    int grid_src = (Ns + block - 1) / block;
    CUDA_CHECK(cudaMemcpy(d_complex_tmp1, charges1, Ns * sizeof(double2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_complex_tmp2, charges2, Ns * sizeof(double2), cudaMemcpyHostToDevice));
    split_complex_kernel<<<grid_src, block>>>(d_complex_tmp1, d_charges_re, d_charges_im, Ns);
    split_complex_kernel<<<grid_src, block>>>(d_complex_tmp2, d_charges2_re, d_charges2_im, Ns);
    CUDA_CHECK(cudaGetLastError());

    evaluate_batch2_uploaded();

    int grid_tgt = (Nt + block - 1) / block;
    pack_complex_kernel<<<grid_tgt, block>>>(d_result_re, d_result_im, d_complex_tmp1, Nt);
    pack_complex_kernel<<<grid_tgt, block>>>(d_result2_re, d_result2_im, d_complex_tmp2, Nt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(result1, d_complex_tmp1, Nt * sizeof(double2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result2, d_complex_tmp2, Nt * sizeof(double2), cudaMemcpyDeviceToHost));
}

void HelmholtzFMM::evaluate_batch2_uploaded()
{
    int n_leaves = (int)leaf_info.size();

    CUDA_CHECK(cudaMemset(d_multi_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi2_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi2_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local2_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local2_im, 0, n_nodes * L * sizeof(double)));

    int block_L = std::min(L, 256);
    if (n_leaves > 0) {
        p2m_kernel_batch2<<<n_leaves, block_L>>>(
            d_src_pts, d_charges_re, d_charges_im, d_charges2_re, d_charges2_im,
            d_dirs_cached, k.real(), k.imag(),
            d_multi_re, d_multi_im, d_multi2_re, d_multi2_im,
            d_leaf_idx_cached, d_src_id_offsets_cached, d_src_ids_cached,
            d_node_centers_cached, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
    }

    for (int level = tree.max_level - 1; level >= 1; level--) {
        if (level < (int)m2m_level_info.size() && m2m_level_info[level].count > 0) {
            int off = m2m_level_info[level].offset;
            int cnt = m2m_level_info[level].count;
            m2m_kernel_batch2<<<cnt, block_L>>>(d_m2m_parent, d_m2m_child,
                d_m2m_shift_re, d_m2m_shift_im,
                d_multi_re, d_multi_im, d_multi2_re, d_multi2_im, L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (int level = 1; level <= tree.max_level; level++) {
        if (level < (int)m2l_level_info.size() && m2l_level_info[level].count > 0) {
            int off = m2l_level_info[level].offset;
            int cnt = m2l_level_info[level].count;
            m2l_kernel_batch2<<<cnt, block_L>>>(d_m2l_tgt, d_m2l_src, d_m2l_tidx,
                d_transfer_re, d_transfer_im,
                d_multi_re, d_multi_im, d_multi2_re, d_multi2_im,
                d_local_re, d_local_im, d_local2_re, d_local2_im, L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (int level = 2; level <= tree.max_level; level++) {
        if (level < (int)l2l_level_info.size() && l2l_level_info[level].count > 0) {
            int off = l2l_level_info[level].offset;
            int cnt = l2l_level_info[level].count;
            l2l_kernel_batch2<<<cnt, block_L>>>(d_l2l_parent, d_l2l_child,
                d_l2l_shift_re, d_l2l_shift_im,
                d_local_re, d_local_im, d_local_re, d_local_im,
                d_local2_re, d_local2_im, d_local2_re, d_local2_im, L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    cdouble ik_val = cdouble(0, 1) * k;
    cdouble prefactor = ik_val / (16.0 * M_PI * M_PI);

    CUDA_CHECK(cudaMemset(d_result_re, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result_im, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result2_re, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result2_im, 0, Nt * sizeof(double)));
    if (n_leaves > 0) {
        l2p_kernel_batch2<<<n_leaves, 256>>>(
            d_tgt_pts, d_dirs_cached, d_weights_cached,
            d_local_re, d_local_im, d_local2_re, d_local2_im,
            k.real(), k.imag(), prefactor.real(), prefactor.imag(),
            d_result_re, d_result_im, d_result2_re, d_result2_im,
            d_leaf_idx_cached, d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_node_centers_cached, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
    }

    launch_p2p_potential_batch2_leaf(
        d_tgt_pts, d_src_pts,
        d_charges_re, d_charges_im, d_charges2_re, d_charges2_im,
        d_tgt_id_offsets_cached, d_tgt_ids_cached,
        d_src_id_offsets_cached, d_src_ids_cached,
        d_leaf_near_offsets_cached, d_leaf_near_ids_cached,
        n_leaves, k.real(), k.imag(),
        d_result_re, d_result_im, d_result2_re, d_result2_im);
    CUDA_CHECK(cudaGetLastError());
}

void HelmholtzFMM::evaluate_batch3_far_uploaded()
{
    const int n_leaves = (int)leaf_info.size();
    const size_t expansion_bytes =
        (size_t)n_nodes * L * sizeof(double);
    CUDA_CHECK(cudaMemset(d_multi_re, 0, expansion_bytes));
    CUDA_CHECK(cudaMemset(d_multi_im, 0, expansion_bytes));
    CUDA_CHECK(cudaMemset(d_local_re, 0, expansion_bytes));
    CUDA_CHECK(cudaMemset(d_local_im, 0, expansion_bytes));
    CUDA_CHECK(cudaMemset(d_multi2_re, 0, expansion_bytes));
    CUDA_CHECK(cudaMemset(d_multi2_im, 0, expansion_bytes));
    CUDA_CHECK(cudaMemset(d_local2_re, 0, expansion_bytes));
    CUDA_CHECK(cudaMemset(d_local2_im, 0, expansion_bytes));
    CUDA_CHECK(cudaMemset(d_multi3_re, 0, expansion_bytes));
    CUDA_CHECK(cudaMemset(d_multi3_im, 0, expansion_bytes));
    CUDA_CHECK(cudaMemset(d_local3_re, 0, expansion_bytes));
    CUDA_CHECK(cudaMemset(d_local3_im, 0, expansion_bytes));

    const int block_L = std::min(L, 256);
    if (n_leaves > 0) {
        p2m_kernel_batch3<<<n_leaves, block_L>>>(
            d_src_pts,
            d_charges_re, d_charges_im,
            d_charges2_re, d_charges2_im,
            d_charges3_re, d_charges3_im,
            d_dirs_cached, k.real(), k.imag(),
            d_multi_re, d_multi_im,
            d_multi2_re, d_multi2_im,
            d_multi3_re, d_multi3_im,
            d_leaf_idx_cached,
            d_src_id_offsets_cached, d_src_ids_cached,
            d_node_centers_cached, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
    }

    for (int level = tree.max_level - 1; level >= 1; level--) {
        if (level < (int)m2m_level_info.size() &&
            m2m_level_info[level].count > 0) {
            const int offset = m2m_level_info[level].offset;
            const int count = m2m_level_info[level].count;
            m2m_kernel_batch3<<<count, block_L>>>(
                d_m2m_parent, d_m2m_child,
                d_m2m_shift_re, d_m2m_shift_im,
                d_multi_re, d_multi_im,
                d_multi2_re, d_multi2_im,
                d_multi3_re, d_multi3_im,
                L, count, offset);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (int level = 1; level <= tree.max_level; level++) {
        if (level < (int)m2l_row_level_info.size() &&
            m2l_row_level_info[level].count > 0) {
            const int offset = m2l_row_level_info[level].offset;
            const int count = m2l_row_level_info[level].count;
            m2l_kernel_batch3_target_rows<<<count, block_L>>>(
                d_m2l_row_target, d_m2l_row_start, d_m2l_row_end,
                d_m2l_src, d_m2l_tidx,
                d_transfer_re, d_transfer_im,
                d_multi_re, d_multi_im,
                d_multi2_re, d_multi2_im,
                d_multi3_re, d_multi3_im,
                d_local_re, d_local_im,
                d_local2_re, d_local2_im,
                d_local3_re, d_local3_im,
                L, count, offset);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (int level = 2; level <= tree.max_level; level++) {
        if (level < (int)l2l_level_info.size() &&
            l2l_level_info[level].count > 0) {
            const int offset = l2l_level_info[level].offset;
            const int count = l2l_level_info[level].count;
            l2l_kernel_batch3<<<count, block_L>>>(
                d_l2l_parent, d_l2l_child,
                d_l2l_shift_re, d_l2l_shift_im,
                d_local_re, d_local_im,
                d_local_re, d_local_im,
                d_local2_re, d_local2_im,
                d_local2_re, d_local2_im,
                d_local3_re, d_local3_im,
                d_local3_re, d_local3_im,
                L, count, offset);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

void HelmholtzFMM::evaluate_batch4_far_uploaded(
    bool evaluate_potential)
{
    int n_leaves = (int)leaf_info.size();

    CUDA_CHECK(cudaMemset(d_multi_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi2_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi2_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local2_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local2_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi3_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi3_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local3_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local3_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi4_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi4_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local4_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local4_im, 0, n_nodes * L * sizeof(double)));

    int block_L = std::min(L, 256);
    if (n_leaves > 0) {
        p2m_kernel_batch4<<<n_leaves, block_L>>>(
            d_src_pts,
            d_charges_re, d_charges_im, d_charges2_re, d_charges2_im,
            d_charges3_re, d_charges3_im, d_charges4_re, d_charges4_im,
            d_dirs_cached, k.real(), k.imag(),
            d_multi_re, d_multi_im, d_multi2_re, d_multi2_im,
            d_multi3_re, d_multi3_im, d_multi4_re, d_multi4_im,
            d_leaf_idx_cached, d_src_id_offsets_cached, d_src_ids_cached,
            d_node_centers_cached, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
    }

    for (int level = tree.max_level - 1; level >= 1; level--) {
        if (level < (int)m2m_level_info.size() && m2m_level_info[level].count > 0) {
            int off = m2m_level_info[level].offset, cnt = m2m_level_info[level].count;
            m2m_kernel_batch4<<<cnt, block_L>>>(d_m2m_parent, d_m2m_child,
                d_m2m_shift_re, d_m2m_shift_im,
                d_multi_re, d_multi_im, d_multi2_re, d_multi2_im,
                d_multi3_re, d_multi3_im, d_multi4_re, d_multi4_im,
                L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (int level = 1; level <= tree.max_level; level++) {
        if (level < (int)m2l_row_level_info.size() &&
            m2l_row_level_info[level].count > 0) {
            const int off = m2l_row_level_info[level].offset;
            const int cnt = m2l_row_level_info[level].count;
            m2l_kernel_batch4_target_rows<<<cnt, block_L>>>(
                d_m2l_row_target, d_m2l_row_start, d_m2l_row_end,
                d_m2l_src, d_m2l_tidx,
                d_transfer_re, d_transfer_im,
                d_multi_re, d_multi_im, d_multi2_re, d_multi2_im,
                d_multi3_re, d_multi3_im, d_multi4_re, d_multi4_im,
                d_local_re, d_local_im, d_local2_re, d_local2_im,
                d_local3_re, d_local3_im, d_local4_re, d_local4_im,
                L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (int level = 2; level <= tree.max_level; level++) {
        if (level < (int)l2l_level_info.size() && l2l_level_info[level].count > 0) {
            int off = l2l_level_info[level].offset, cnt = l2l_level_info[level].count;
            l2l_kernel_batch4<<<cnt, block_L>>>(d_l2l_parent, d_l2l_child,
                d_l2l_shift_re, d_l2l_shift_im,
                d_local_re, d_local_im, d_local_re, d_local_im,
                d_local2_re, d_local2_im, d_local2_re, d_local2_im,
                d_local3_re, d_local3_im, d_local3_re, d_local3_im,
                d_local4_re, d_local4_im, d_local4_re, d_local4_im,
                L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (evaluate_potential) {
        const cdouble ik_val = cdouble(0, 1) * k;
        const cdouble prefactor =
            ik_val / (16.0 * M_PI * M_PI);
        CUDA_CHECK(cudaMemset(
            d_result_re, 0, Nt * sizeof(double)));
        CUDA_CHECK(cudaMemset(
            d_result_im, 0, Nt * sizeof(double)));
        CUDA_CHECK(cudaMemset(
            d_result2_re, 0, Nt * sizeof(double)));
        CUDA_CHECK(cudaMemset(
            d_result2_im, 0, Nt * sizeof(double)));
        CUDA_CHECK(cudaMemset(
            d_result3_re, 0, Nt * sizeof(double)));
        CUDA_CHECK(cudaMemset(
            d_result3_im, 0, Nt * sizeof(double)));
        CUDA_CHECK(cudaMemset(
            d_result4_re, 0, Nt * sizeof(double)));
        CUDA_CHECK(cudaMemset(
            d_result4_im, 0, Nt * sizeof(double)));
        if (n_leaves > 0) {
            l2p_kernel_batch4<<<n_leaves, 256>>>(
                d_tgt_pts, d_dirs_cached, d_weights_cached,
                d_local_re, d_local_im, d_local2_re, d_local2_im,
                d_local3_re, d_local3_im, d_local4_re, d_local4_im,
                k.real(), k.imag(),
                prefactor.real(), prefactor.imag(),
                d_result_re, d_result_im,
                d_result2_re, d_result2_im,
                d_result3_re, d_result3_im,
                d_result4_re, d_result4_im,
                d_leaf_idx_cached,
                d_tgt_id_offsets_cached, d_tgt_ids_cached,
                d_node_centers_cached, L, n_leaves);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

void HelmholtzFMM::evaluate_batch4_uploaded()
{
    evaluate_batch4_far_uploaded();
    int n_leaves = (int)leaf_info.size();
    launch_p2p_potential_batch4_leaf(
        d_tgt_pts, d_src_pts,
        d_charges_re, d_charges_im, d_charges2_re, d_charges2_im,
        d_charges3_re, d_charges3_im, d_charges4_re, d_charges4_im,
        d_tgt_id_offsets_cached, d_tgt_ids_cached,
        d_src_id_offsets_cached, d_src_ids_cached,
        d_leaf_near_offsets_cached, d_leaf_near_ids_cached,
        n_leaves, k.real(), k.imag(),
        d_result_re, d_result_im, d_result2_re, d_result2_im,
        d_result3_re, d_result3_im, d_result4_re, d_result4_im);
    CUDA_CHECK(cudaGetLastError());
}

void HelmholtzFMM::evaluate_pot_grad_batch2(
    const cdouble* charges1, const cdouble* charges2,
    cdouble* pot1, cdouble* grad1,
    cdouble* pot2, cdouble* grad2)
{
    int block = 256;
    int grid_src = (Ns + block - 1) / block;
    CUDA_CHECK(cudaMemcpy(d_complex_tmp1, charges1, Ns * sizeof(double2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_complex_tmp2, charges2, Ns * sizeof(double2), cudaMemcpyHostToDevice));
    split_complex_kernel<<<grid_src, block>>>(d_complex_tmp1, d_charges_re, d_charges_im, Ns);
    split_complex_kernel<<<grid_src, block>>>(d_complex_tmp2, d_charges2_re, d_charges2_im, Ns);
    CUDA_CHECK(cudaGetLastError());

    evaluate_pot_grad_batch2_uploaded();

    int grid_tgt = (Nt + block - 1) / block;
    pack_complex_kernel<<<grid_tgt, block>>>(d_result_re, d_result_im, d_complex_tmp1, Nt);
    pack_complex_kernel<<<grid_tgt, block>>>(d_result2_re, d_result2_im, d_complex_tmp2, Nt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(pot1, d_complex_tmp1, Nt * sizeof(double2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pot2, d_complex_tmp2, Nt * sizeof(double2), cudaMemcpyDeviceToHost));

    int ngrad = Nt * 3;
    int grid_grad = (ngrad + block - 1) / block;
    pack_complex_kernel<<<grid_grad, block>>>(d_grad_re, d_grad_im, d_complex_tmp1, ngrad);
    pack_complex_kernel<<<grid_grad, block>>>(d_grad2_re, d_grad2_im, d_complex_tmp2, ngrad);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(grad1, d_complex_tmp1, ngrad * sizeof(double2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grad2, d_complex_tmp2, ngrad * sizeof(double2), cudaMemcpyDeviceToHost));
}

void HelmholtzFMM::evaluate_pot_grad_batch2_uploaded()
{
    int n_leaves = (int)leaf_info.size();

    CUDA_CHECK(cudaMemset(d_multi_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi2_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_multi2_im, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local2_re, 0, n_nodes * L * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local2_im, 0, n_nodes * L * sizeof(double)));

    int block_L = std::min(L, 256);
    if (n_leaves > 0) {
        p2m_kernel_batch2<<<n_leaves, block_L>>>(
            d_src_pts, d_charges_re, d_charges_im, d_charges2_re, d_charges2_im,
            d_dirs_cached, k.real(), k.imag(),
            d_multi_re, d_multi_im, d_multi2_re, d_multi2_im,
            d_leaf_idx_cached, d_src_id_offsets_cached, d_src_ids_cached,
            d_node_centers_cached, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
    }

    for (int level = tree.max_level - 1; level >= 1; level--) {
        if (level < (int)m2m_level_info.size() && m2m_level_info[level].count > 0) {
            int off = m2m_level_info[level].offset;
            int cnt = m2m_level_info[level].count;
            m2m_kernel_batch2<<<cnt, block_L>>>(d_m2m_parent, d_m2m_child,
                d_m2m_shift_re, d_m2m_shift_im,
                d_multi_re, d_multi_im, d_multi2_re, d_multi2_im, L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (int level = 1; level <= tree.max_level; level++) {
        if (level < (int)m2l_level_info.size() && m2l_level_info[level].count > 0) {
            int off = m2l_level_info[level].offset;
            int cnt = m2l_level_info[level].count;
            m2l_kernel_batch2<<<cnt, block_L>>>(d_m2l_tgt, d_m2l_src, d_m2l_tidx,
                d_transfer_re, d_transfer_im,
                d_multi_re, d_multi_im, d_multi2_re, d_multi2_im,
                d_local_re, d_local_im, d_local2_re, d_local2_im, L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (int level = 2; level <= tree.max_level; level++) {
        if (level < (int)l2l_level_info.size() && l2l_level_info[level].count > 0) {
            int off = l2l_level_info[level].offset;
            int cnt = l2l_level_info[level].count;
            l2l_kernel_batch2<<<cnt, block_L>>>(d_l2l_parent, d_l2l_child,
                d_l2l_shift_re, d_l2l_shift_im,
                d_local_re, d_local_im, d_local_re, d_local_im,
                d_local2_re, d_local2_im, d_local2_re, d_local2_im, L, cnt, off);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    cdouble ik_val = cdouble(0, 1) * k;
    cdouble prefactor = ik_val / (16.0 * M_PI * M_PI);

    CUDA_CHECK(cudaMemset(d_result_re, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result_im, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result2_re, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result2_im, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gx_re_tmp_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gx_im_tmp_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy_re_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy_im_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz_re_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz_im_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gx2_re_tmp_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gx2_im_tmp_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy2_re_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy2_im_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz2_re_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz2_im_cached, 0, Nt * sizeof(double)));

    if (n_leaves > 0) {
        l2p_kernel_batch2<<<n_leaves, 256>>>(
            d_tgt_pts, d_dirs_cached, d_weights_cached,
            d_local_re, d_local_im, d_local2_re, d_local2_im,
            k.real(), k.imag(), prefactor.real(), prefactor.imag(),
            d_result_re, d_result_im, d_result2_re, d_result2_im,
            d_leaf_idx_cached, d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_node_centers_cached, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());

        l2p_gradient_kernel_batch2<<<n_leaves, 256>>>(
            d_tgt_pts, d_dirs_cached, d_weights_cached,
            d_local_re, d_local_im, d_local2_re, d_local2_im,
            k.real(), k.imag(), prefactor.real(), prefactor.imag(),
            ik_val.real(), ik_val.imag(),
            d_gx_re_tmp_cached, d_gx_im_tmp_cached, d_gy_re_cached, d_gy_im_cached, d_gz_re_cached, d_gz_im_cached,
            d_gx2_re_tmp_cached, d_gx2_im_tmp_cached, d_gy2_re_cached, d_gy2_im_cached, d_gz2_re_cached, d_gz2_im_cached,
            d_leaf_idx_cached, d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_node_centers_cached, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
    }

    launch_p2p_pot_grad_batch2_leaf(
        d_tgt_pts, d_src_pts,
        d_charges_re, d_charges_im, d_charges2_re, d_charges2_im,
        d_tgt_id_offsets_cached, d_tgt_ids_cached,
        d_src_id_offsets_cached, d_src_ids_cached,
        d_leaf_near_offsets_cached, d_leaf_near_ids_cached,
        n_leaves, k.real(), k.imag(),
        d_result_re, d_result_im,
        d_gx_re_tmp_cached, d_gx_im_tmp_cached, d_gy_re_cached, d_gy_im_cached, d_gz_re_cached, d_gz_im_cached,
        d_result2_re, d_result2_im,
        d_gx2_re_tmp_cached, d_gx2_im_tmp_cached, d_gy2_re_cached, d_gy2_im_cached, d_gz2_re_cached, d_gz2_im_cached);
    CUDA_CHECK(cudaGetLastError());

    int repack_block = 256;
    int repack_grid = (Nt + repack_block - 1) / repack_block;
    repack_gradient_kernel<<<repack_grid, repack_block>>>(
        d_gx_re_tmp_cached, d_gx_im_tmp_cached,
        d_gy_re_cached, d_gy_im_cached, d_gz_re_cached, d_gz_im_cached,
        d_grad_re, d_grad_im, Nt);
    CUDA_CHECK(cudaGetLastError());
    repack_gradient_kernel<<<repack_grid, repack_block>>>(
        d_gx2_re_tmp_cached, d_gx2_im_tmp_cached,
        d_gy2_re_cached, d_gy2_im_cached, d_gz2_re_cached, d_gz2_im_cached,
        d_grad2_re, d_grad2_im, Nt);
    CUDA_CHECK(cudaGetLastError());
}

void HelmholtzFMM::evaluate_gradient_batch4_l2p_uploaded()
{
    int n_leaves = (int)leaf_info.size();
    cdouble ik_val = cdouble(0, 1) * k;
    cdouble prefactor = ik_val / (16.0 * M_PI * M_PI);

    CUDA_CHECK(cudaMemset(d_gx_re_tmp_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gx_im_tmp_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy_re_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy_im_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz_re_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz_im_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gx2_re_tmp_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gx2_im_tmp_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy2_re_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy2_im_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz2_re_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz2_im_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gx3_re_tmp_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gx3_im_tmp_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy3_re_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy3_im_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz3_re_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz3_im_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gx4_re_tmp_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gx4_im_tmp_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy4_re_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy4_im_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz4_re_cached, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz4_im_cached, 0, Nt * sizeof(double)));

    if (n_leaves > 0) {
        l2p_gradient_kernel_batch4<<<n_leaves, 256>>>(
            d_tgt_pts, d_dirs_cached, d_weights_cached,
            d_local_re, d_local_im, d_local2_re, d_local2_im,
            d_local3_re, d_local3_im, d_local4_re, d_local4_im,
            k.real(), k.imag(), prefactor.real(), prefactor.imag(),
            ik_val.real(), ik_val.imag(),
            d_gx_re_tmp_cached, d_gx_im_tmp_cached, d_gy_re_cached, d_gy_im_cached, d_gz_re_cached, d_gz_im_cached,
            d_gx2_re_tmp_cached, d_gx2_im_tmp_cached, d_gy2_re_cached, d_gy2_im_cached, d_gz2_re_cached, d_gz2_im_cached,
            d_gx3_re_tmp_cached, d_gx3_im_tmp_cached, d_gy3_re_cached, d_gy3_im_cached, d_gz3_re_cached, d_gz3_im_cached,
            d_gx4_re_tmp_cached, d_gx4_im_tmp_cached, d_gy4_re_cached, d_gy4_im_cached, d_gz4_re_cached, d_gz4_im_cached,
            d_leaf_idx_cached, d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_node_centers_cached, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
    }
}

void HelmholtzFMM::evaluate_pot_grad_batch4_uploaded()
{
    evaluate_batch4_far_uploaded();
    evaluate_gradient_batch4_l2p_uploaded();

    int n_leaves = (int)leaf_info.size();

    launch_p2p_pot_grad_batch4_leaf(
        d_tgt_pts, d_src_pts,
        d_charges_re, d_charges_im, d_charges2_re, d_charges2_im,
        d_charges3_re, d_charges3_im, d_charges4_re, d_charges4_im,
        d_tgt_id_offsets_cached, d_tgt_ids_cached,
        d_src_id_offsets_cached, d_src_ids_cached,
        d_leaf_near_offsets_cached, d_leaf_near_ids_cached,
        n_leaves, k.real(), k.imag(),
        d_result_re, d_result_im,
        d_gx_re_tmp_cached, d_gx_im_tmp_cached, d_gy_re_cached, d_gy_im_cached, d_gz_re_cached, d_gz_im_cached,
        d_result2_re, d_result2_im,
        d_gx2_re_tmp_cached, d_gx2_im_tmp_cached, d_gy2_re_cached, d_gy2_im_cached, d_gz2_re_cached, d_gz2_im_cached,
        d_result3_re, d_result3_im,
        d_gx3_re_tmp_cached, d_gx3_im_tmp_cached, d_gy3_re_cached, d_gy3_im_cached, d_gz3_re_cached, d_gz3_im_cached,
        d_result4_re, d_result4_im,
        d_gx4_re_tmp_cached, d_gx4_im_tmp_cached, d_gy4_re_cached, d_gy4_im_cached, d_gz4_re_cached, d_gz4_im_cached);
    CUDA_CHECK(cudaGetLastError());

    int block = 256;
    int grid = (Nt + block - 1) / block;
    repack_gradient_kernel<<<grid, block>>>(d_gx_re_tmp_cached, d_gx_im_tmp_cached,
        d_gy_re_cached, d_gy_im_cached, d_gz_re_cached, d_gz_im_cached, d_grad_re, d_grad_im, Nt);
    repack_gradient_kernel<<<grid, block>>>(d_gx2_re_tmp_cached, d_gx2_im_tmp_cached,
        d_gy2_re_cached, d_gy2_im_cached, d_gz2_re_cached, d_gz2_im_cached, d_grad2_re, d_grad2_im, Nt);
    repack_gradient_kernel<<<grid, block>>>(d_gx3_re_tmp_cached, d_gx3_im_tmp_cached,
        d_gy3_re_cached, d_gy3_im_cached, d_gz3_re_cached, d_gz3_im_cached, d_grad3_re, d_grad3_im, Nt);
    repack_gradient_kernel<<<grid, block>>>(d_gx4_re_tmp_cached, d_gx4_im_tmp_cached,
        d_gy4_re_cached, d_gy4_im_cached, d_gz4_re_cached, d_gz4_im_cached, d_grad4_re, d_grad4_im, Nt);
    CUDA_CHECK(cudaGetLastError());
}

void HelmholtzFMM::cleanup()
{
    if (!initialized) return;
    cudaFree(d_tgt_pts); cudaFree(d_src_pts);
    cudaFree(d_tgt_pts_fp32); cudaFree(d_src_pts_fp32);
    cudaFree(d_dirs_fp32_cached); cudaFree(d_weights_fp32_cached);
    cudaFree(d_phase_cache_fp32);
    cudaFree(d_l2p_phase_cache_fp32);
    cudaFree(d_p2p_offsets); cudaFree(d_p2p_indices);
    cudaFree(d_multi_re); cudaFree(d_multi_im);
    cudaFree(d_local_re); cudaFree(d_local_im);
    cudaFree(d_transfer_re); cudaFree(d_transfer_im);
    cudaFree(d_transfer_re_fp32); cudaFree(d_transfer_im_fp32);
    cudaFree(d_m2l_tgt); cudaFree(d_m2l_src); cudaFree(d_m2l_tidx);
    cudaFree(d_m2l_row_target);
    cudaFree(d_m2l_row_start);
    cudaFree(d_m2l_row_end);
    cudaFree(d_m2m_shift_re); cudaFree(d_m2m_shift_im);
    cudaFree(d_m2m_parent); cudaFree(d_m2m_child);
    cudaFree(d_l2l_shift_re); cudaFree(d_l2l_shift_im);
    cudaFree(d_l2l_parent); cudaFree(d_l2l_child);
    cudaFree(d_charges_re); cudaFree(d_charges_im);
    cudaFree(d_result_re); cudaFree(d_result_im);
    cudaFree(d_grad_re); cudaFree(d_grad_im);
    cudaFree(d_hess_re); cudaFree(d_hess_im);
    if (d_charges2_re) cudaFree(d_charges2_re);
    if (d_charges2_im) cudaFree(d_charges2_im);
    if (d_result2_re) cudaFree(d_result2_re);
    if (d_result2_im) cudaFree(d_result2_im);
    if (d_grad2_re) cudaFree(d_grad2_re);
    if (d_grad2_im) cudaFree(d_grad2_im);
    if (d_multi2_re) cudaFree(d_multi2_re);
    if (d_multi2_im) cudaFree(d_multi2_im);
    if (d_local2_re) cudaFree(d_local2_re);
    if (d_local2_im) cudaFree(d_local2_im);
    if (d_charges3_re) cudaFree(d_charges3_re);
    if (d_charges3_im) cudaFree(d_charges3_im);
    if (d_charges4_re) cudaFree(d_charges4_re);
    if (d_charges4_im) cudaFree(d_charges4_im);
    if (d_result3_re) cudaFree(d_result3_re);
    if (d_result3_im) cudaFree(d_result3_im);
    if (d_result4_re) cudaFree(d_result4_re);
    if (d_result4_im) cudaFree(d_result4_im);
    if (d_grad3_re) cudaFree(d_grad3_re);
    if (d_grad3_im) cudaFree(d_grad3_im);
    if (d_grad4_re) cudaFree(d_grad4_re);
    if (d_grad4_im) cudaFree(d_grad4_im);
    if (d_hess2_re) cudaFree(d_hess2_re);
    if (d_hess2_im) cudaFree(d_hess2_im);
    if (d_hess3_re) cudaFree(d_hess3_re);
    if (d_hess3_im) cudaFree(d_hess3_im);
    if (d_hess4_re) cudaFree(d_hess4_re);
    if (d_hess4_im) cudaFree(d_hess4_im);
    if (d_multi3_re) cudaFree(d_multi3_re);
    if (d_multi3_im) cudaFree(d_multi3_im);
    if (d_multi4_re) cudaFree(d_multi4_re);
    if (d_multi4_im) cudaFree(d_multi4_im);
    if (d_local3_re) cudaFree(d_local3_re);
    if (d_local3_im) cudaFree(d_local3_im);
    if (d_local4_re) cudaFree(d_local4_re);
    if (d_local4_im) cudaFree(d_local4_im);
    if (d_pair_multi_re) cudaFree(d_pair_multi_re);
    if (d_pair_multi_im) cudaFree(d_pair_multi_im);
    if (d_pair_multi_re_fp32) cudaFree(d_pair_multi_re_fp32);
    if (d_pair_multi_im_fp32) cudaFree(d_pair_multi_im_fp32);
    if (d_pair_local_re) cudaFree(d_pair_local_re);
    if (d_pair_local_im) cudaFree(d_pair_local_im);
    if (d_pair_local_re_fp32) cudaFree(d_pair_local_re_fp32);
    if (d_pair_local_im_fp32) cudaFree(d_pair_local_im_fp32);
    if (d_strict_pair_multi_re) cudaFree(d_strict_pair_multi_re);
    if (d_strict_pair_multi_im) cudaFree(d_strict_pair_multi_im);
    if (d_strict_pair_local_re) cudaFree(d_strict_pair_local_re);
    if (d_strict_pair_local_im) cudaFree(d_strict_pair_local_im);
    if (d_pair_charges_fp32) cudaFree(d_pair_charges_fp32);
    cudaFree(d_node_centers_cached); cudaFree(d_dirs_cached); cudaFree(d_weights_cached);
    cudaFree(d_leaf_idx_cached);
    cudaFree(d_src_id_offsets_cached); cudaFree(d_src_ids_cached);
    cudaFree(d_tgt_id_offsets_cached); cudaFree(d_tgt_ids_cached);
    cudaFree(d_leaf_near_offsets_cached); cudaFree(d_leaf_near_ids_cached);
    cudaFree(d_leaf_near_source_offsets_cached);
    cudaFree(d_leaf_near_source_ids_cached);
    cudaFree(d_gy_re_cached); cudaFree(d_gy_im_cached);
    cudaFree(d_gz_re_cached); cudaFree(d_gz_im_cached);
    cudaFree(d_gx_re_tmp_cached); cudaFree(d_gx_im_tmp_cached);
    cudaFree(d_gy2_re_cached); cudaFree(d_gy2_im_cached);
    cudaFree(d_gz2_re_cached); cudaFree(d_gz2_im_cached);
    cudaFree(d_gx2_re_tmp_cached); cudaFree(d_gx2_im_tmp_cached);
    cudaFree(d_gy3_re_cached); cudaFree(d_gy3_im_cached);
    cudaFree(d_gz3_re_cached); cudaFree(d_gz3_im_cached);
    cudaFree(d_gx3_re_tmp_cached); cudaFree(d_gx3_im_tmp_cached);
    cudaFree(d_gy4_re_cached); cudaFree(d_gy4_im_cached);
    cudaFree(d_gz4_re_cached); cudaFree(d_gz4_im_cached);
    cudaFree(d_gx4_re_tmp_cached); cudaFree(d_gx4_im_tmp_cached);
    cudaFree(d_complex_tmp1); cudaFree(d_complex_tmp2);
    d_charges2_re = nullptr;
    d_charges2_im = nullptr;
    d_result2_re = nullptr;
    d_result2_im = nullptr;
    d_grad2_re = nullptr;
    d_grad2_im = nullptr;
    d_multi2_re = nullptr;
    d_multi2_im = nullptr;
    d_local2_re = nullptr;
    d_local2_im = nullptr;
    d_charges3_re = nullptr;
    d_charges3_im = nullptr;
    d_charges4_re = nullptr;
    d_charges4_im = nullptr;
    d_result3_re = nullptr;
    d_result3_im = nullptr;
    d_result4_re = nullptr;
    d_result4_im = nullptr;
    d_grad3_re = nullptr;
    d_grad3_im = nullptr;
    d_grad4_re = nullptr;
    d_grad4_im = nullptr;
    d_hess2_re = nullptr;
    d_hess2_im = nullptr;
    d_hess3_re = nullptr;
    d_hess3_im = nullptr;
    d_hess4_re = nullptr;
    d_hess4_im = nullptr;
    d_multi3_re = nullptr;
    d_multi3_im = nullptr;
    d_multi4_re = nullptr;
    d_multi4_im = nullptr;
    d_local3_re = nullptr;
    d_local3_im = nullptr;
    d_local4_re = nullptr;
    d_local4_im = nullptr;
    d_pair_multi_re = nullptr;
    d_pair_multi_im = nullptr;
    d_pair_multi_re_fp32 = nullptr;
    d_pair_multi_im_fp32 = nullptr;
    d_pair_local_re = nullptr;
    d_pair_local_im = nullptr;
    d_pair_local_re_fp32 = nullptr;
    d_pair_local_im_fp32 = nullptr;
    d_pair_charges_fp32 = nullptr;
    d_tgt_pts_fp32 = nullptr;
    d_src_pts_fp32 = nullptr;
    d_dirs_fp32_cached = nullptr;
    d_weights_fp32_cached = nullptr;
    d_phase_cache_fp32 = nullptr;
    d_l2p_phase_cache_fp32 = nullptr;
    d_m2l_row_target = nullptr;
    d_m2l_row_start = nullptr;
    d_m2l_row_end = nullptr;
    d_transfer_re_fp32 = nullptr;
    d_transfer_im_fp32 = nullptr;
    d_leaf_near_source_offsets_cached = nullptr;
    d_leaf_near_source_ids_cached = nullptr;
    initialized = false;
    batch4_allocated = false;
    pair_l2p_allocated = false;
    pair_workspace_fields = 0;
}
