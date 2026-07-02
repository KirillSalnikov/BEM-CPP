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

// ============================================================
// CUDA kernels for FMM tree operations
// All use ORIGINAL-ORDER indices for positions and charges
// ============================================================

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

void HelmholtzFMM::init(const double* targets, int n_tgt,
                          const double* sources, int n_src,
                          cdouble k_val, int digits, int max_leaf)
{
    Timer timer;
    k = k_val;
    Nt = n_tgt;
    Ns = n_src;

    // Build octree (uses combined point set internally)
    tree.build(targets, n_tgt, sources, n_src, max_leaf);
    n_nodes = (int)tree.nodes.size();

    // Sphere quadrature
    double leaf_hs = tree.nodes[tree.leaves[0]].half_size;
    double leaf_box_size = 2.0 * leaf_hs;
    p = fmm_truncation_order(std::abs(k), leaf_box_size, digits);
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
                    cdouble kd = k * d_norm;
                    double d_hat[3] = {d[0]/d_norm, d[1]/d_norm, d[2]/d_norm};

                    std::vector<cdouble> T(L);
                    for (int ll = 0; ll < L; ll++) {
                        double cos_angle = squad.dirs[ll*3]*d_hat[0] +
                                           squad.dirs[ll*3+1]*d_hat[1] +
                                           squad.dirs[ll*3+2]*d_hat[2];
                        cdouble sum(0, 0);
                        double P_prev = 1.0, P_curr = cos_angle;
                        sum += 1.0 * spherical_hankel1(0, kd) * P_prev;
                        if (p >= 1)
                            sum += 3.0 * cdouble(0, 1) * spherical_hankel1(1, kd) * P_curr;
                        cdouble i_pow(0, 1);
                        for (int l = 2; l <= p; l++) {
                            double P_next = ((2*l - 1) * cos_angle * P_curr - (l - 1) * P_prev) / l;
                            i_pow *= cdouble(0, 1);
                            sum += (2.0*l + 1.0) * i_pow * spherical_hankel1(l, kd) * P_next;
                            P_prev = P_curr;
                            P_curr = P_next;
                        }
                        T[ll] = sum;
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
        m2l_level_info.resize(tree.max_level + 1);
        for (int level = 0; level <= tree.max_level; level++) {
            m2l_level_info[level].offset = (int)all_tgt.size();
            m2l_level_info[level].count = 0;
            if (level < (int)m2l_batches.size() && m2l_batches[level].n_pairs > 0) {
                const M2LBatch& b = m2l_batches[level];
                all_tgt.insert(all_tgt.end(), b.tgt_idx.begin(), b.tgt_idx.end());
                all_src.insert(all_src.end(), b.src_idx.begin(), b.src_idx.end());
                all_tidx.insert(all_tidx.end(), b.transfer_idx.begin(), b.transfer_idx.end());
                m2l_level_info[level].count = b.n_pairs;
            }
        }
        if (!all_tgt.empty()) {
            CUDA_CHECK(cudaMalloc(&d_m2l_tgt, all_tgt.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_m2l_src, all_src.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_m2l_tidx, all_tidx.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_m2l_tgt, all_tgt.data(), all_tgt.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_m2l_src, all_src.data(), all_src.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_m2l_tidx, all_tidx.data(), all_tidx.size() * sizeof(int), cudaMemcpyHostToDevice));
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
    bool request_batch4 = bem_env_flag_enabled("BEM_FMM_ALLOC_BATCH4") ||
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

    int complex_tmp_n = std::max(Ns, 3 * Nt);
    CUDA_CHECK(cudaMalloc(&d_complex_tmp1, complex_tmp_n * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_complex_tmp2, complex_tmp_n * sizeof(double2)));

    initialized = true;
    printf("  [FMM] Init complete: p=%d L=%d, %d nodes, %.1fms\n",
           p, L, n_nodes, timer.elapsed_ms());
}

// Helper to run the full FMM tree traversal (shared between evaluate and evaluate_gradient)
void HelmholtzFMM::run_tree(const double* h_q_re, const double* h_q_im, bool need_grad)
{
    CUDA_CHECK(cudaMemcpy(d_charges_re, h_q_re, Ns * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_charges_im, h_q_im, Ns * sizeof(double), cudaMemcpyHostToDevice));
    run_tree_uploaded(need_grad);
}

void HelmholtzFMM::run_tree_uploaded(bool need_grad)
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

    if (!need_grad) {
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
    } else {
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

void HelmholtzFMM::evaluate_batch4_far_uploaded()
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
        if (level < (int)m2l_level_info.size() && m2l_level_info[level].count > 0) {
            int off = m2l_level_info[level].offset, cnt = m2l_level_info[level].count;
            m2l_kernel_batch4<<<cnt, block_L>>>(d_m2l_tgt, d_m2l_src, d_m2l_tidx,
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

    cdouble ik_val = cdouble(0, 1) * k;
    cdouble prefactor = ik_val / (16.0 * M_PI * M_PI);

    CUDA_CHECK(cudaMemset(d_result_re, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result_im, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result2_re, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result2_im, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result3_re, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result3_im, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result4_re, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result4_im, 0, Nt * sizeof(double)));
    if (n_leaves > 0) {
        l2p_kernel_batch4<<<n_leaves, 256>>>(
            d_tgt_pts, d_dirs_cached, d_weights_cached,
            d_local_re, d_local_im, d_local2_re, d_local2_im,
            d_local3_re, d_local3_im, d_local4_re, d_local4_im,
            k.real(), k.imag(), prefactor.real(), prefactor.imag(),
            d_result_re, d_result_im, d_result2_re, d_result2_im,
            d_result3_re, d_result3_im, d_result4_re, d_result4_im,
            d_leaf_idx_cached, d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_node_centers_cached, L, n_leaves);
        CUDA_CHECK(cudaGetLastError());
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

void HelmholtzFMM::evaluate_pot_grad_batch4_uploaded()
{
    evaluate_batch4_far_uploaded();

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
    cudaFree(d_p2p_offsets); cudaFree(d_p2p_indices);
    cudaFree(d_multi_re); cudaFree(d_multi_im);
    cudaFree(d_local_re); cudaFree(d_local_im);
    cudaFree(d_transfer_re); cudaFree(d_transfer_im);
    cudaFree(d_m2l_tgt); cudaFree(d_m2l_src); cudaFree(d_m2l_tidx);
    cudaFree(d_m2m_shift_re); cudaFree(d_m2m_shift_im);
    cudaFree(d_m2m_parent); cudaFree(d_m2m_child);
    cudaFree(d_l2l_shift_re); cudaFree(d_l2l_shift_im);
    cudaFree(d_l2l_parent); cudaFree(d_l2l_child);
    cudaFree(d_charges_re); cudaFree(d_charges_im);
    cudaFree(d_result_re); cudaFree(d_result_im);
    cudaFree(d_grad_re); cudaFree(d_grad_im);
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
    if (d_multi3_re) cudaFree(d_multi3_re);
    if (d_multi3_im) cudaFree(d_multi3_im);
    if (d_multi4_re) cudaFree(d_multi4_re);
    if (d_multi4_im) cudaFree(d_multi4_im);
    if (d_local3_re) cudaFree(d_local3_re);
    if (d_local3_im) cudaFree(d_local3_im);
    if (d_local4_re) cudaFree(d_local4_re);
    if (d_local4_im) cudaFree(d_local4_im);
    cudaFree(d_node_centers_cached); cudaFree(d_dirs_cached); cudaFree(d_weights_cached);
    cudaFree(d_leaf_idx_cached);
    cudaFree(d_src_id_offsets_cached); cudaFree(d_src_ids_cached);
    cudaFree(d_tgt_id_offsets_cached); cudaFree(d_tgt_ids_cached);
    cudaFree(d_leaf_near_offsets_cached); cudaFree(d_leaf_near_ids_cached);
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
    initialized = false;
    batch4_allocated = false;
}
