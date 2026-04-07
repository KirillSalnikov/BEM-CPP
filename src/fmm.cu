#include "fmm.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <map>
#include <vector>
#include <algorithm>

// P2P launchers (from p2p.cu)
#include "p2p.h"

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
// CUDA kernels for FMM tree operations (float32 internal precision)
// All use ORIGINAL-ORDER indices for positions and charges
// ============================================================

// P2M kernel: double positions/charges in, float32 multipole out
__global__ void p2m_kernel(
    const double* __restrict__ src_pts,
    const double* __restrict__ q_re,
    const double* __restrict__ q_im,
    const fmm_real* __restrict__ dirs,
    float k_re, float k_im,
    fmm_real* __restrict__ multi_re,
    fmm_real* __restrict__ multi_im,
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
    int s_count = s_end - s_start;
    if (s_count == 0) return;

    float cx = (float)node_centers[node*3];
    float cy = (float)node_centers[node*3+1];
    float cz = (float)node_centers[node*3+2];

    int base = node * L;

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        float dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
        float acc_re = 0.0f, acc_im = 0.0f;

        for (int s = s_start; s < s_end; s++) {
            int sid = src_ids[s];
            float rx = (float)(src_pts[sid*3]   - (double)cx);
            float ry = (float)(src_pts[sid*3+1] - (double)cy);
            float rz = (float)(src_pts[sid*3+2] - (double)cz);
            float dot = dx*rx + dy*ry + dz*rz;

            float phase_re = k_im * dot;
            float phase_im = -k_re * dot;
            float e_re = expf(phase_re) * cosf(phase_im);
            float e_im = expf(phase_re) * sinf(phase_im);

            float qr = (float)q_re[sid], qi = (float)q_im[sid];
            acc_re += e_re * qr - e_im * qi;
            acc_im += e_re * qi + e_im * qr;
        }

        multi_re[base + l] = acc_re;
        multi_im[base + l] = acc_im;
    }
}

// M2M kernel (all float32)
__global__ void m2m_kernel(
    const int*      __restrict__ parent_idx,
    const int*      __restrict__ child_idx,
    const fmm_real* __restrict__ shift_re,
    const fmm_real* __restrict__ shift_im,
    fmm_real* __restrict__ multi_re,
    fmm_real* __restrict__ multi_im,
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
        float cr = multi_re[c_base + l];
        float ci = multi_im[c_base + l];
        float sr = shift_re[shift_base + l];
        float si = shift_im[shift_base + l];
        atomicAdd(&multi_re[p_base + l], cr * sr - ci * si);
        atomicAdd(&multi_im[p_base + l], cr * si + ci * sr);
    }
}

// M2L legacy kernel (float32, kept for batch2 fallback)
__global__ void m2l_kernel(
    const int*      __restrict__ tgt_idx,
    const int*      __restrict__ src_idx,
    const int*      __restrict__ transfer_idx,
    const fmm_real* __restrict__ transfer_re,
    const fmm_real* __restrict__ transfer_im,
    const fmm_real* __restrict__ multi_re,
    const fmm_real* __restrict__ multi_im,
    fmm_real* __restrict__ local_re,
    fmm_real* __restrict__ local_im,
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
        float tr = transfer_re[t_base + l];
        float ti = transfer_im[t_base + l];
        float mr = multi_re[s_base + l];
        float mi = multi_im[s_base + l];
        atomicAdd(&local_re[l_base + l], tr * mr - ti * mi);
        atomicAdd(&local_im[l_base + l], tr * mi + ti * mr);
    }
}

// ============================================================
// Optimized M2L kernel (opt3): one block per unique target node.
// CSR structure, shared memory for transfer coefficients,
// register accumulators (no atomicAdd). Float32 precision.
// Uses float2 vectorized loads for interleaved transfer data.
// ============================================================
__global__ void m2l_target_kernel(
    const int*      __restrict__ csr_offsets,
    const int*      __restrict__ csr_tgt_nodes,
    const int*      __restrict__ csr_src,
    const int*      __restrict__ csr_tidx,
    const fmm_real* __restrict__ transfer_ri,
    const fmm_real* __restrict__ multi_re,
    const fmm_real* __restrict__ multi_im,
    fmm_real* __restrict__ local_re,
    fmm_real* __restrict__ local_im,
    int L, int n_targets, int tgt_offset, int pair_offset)
{
    int tgt_id = blockIdx.x + tgt_offset;
    if (tgt_id >= tgt_offset + n_targets) return;

    int tgt_node = csr_tgt_nodes[tgt_id];
    int pair_start = csr_offsets[tgt_id] + pair_offset;
    int pair_end   = csr_offsets[tgt_id + 1] + pair_offset;
    int n_src = pair_end - pair_start;
    if (n_src == 0) return;

    int l_base = tgt_node * L;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    extern __shared__ fmm_real smem[];  // size = L * 2 fmm_reals

    int n_slots = (L + nthreads - 1) / nthreads;
    const int CHUNK = 8;
    int n_chunks = (n_slots + CHUNK - 1) / CHUNK;

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int slot_start = chunk * CHUNK;
        int slot_end = slot_start + CHUNK;
        if (slot_end > n_slots) slot_end = n_slots;
        int n_active = slot_end - slot_start;

        float acc_re[CHUNK], acc_im[CHUNK];
        for (int s = 0; s < n_active; s++) {
            acc_re[s] = 0.0f;
            acc_im[s] = 0.0f;
        }

        for (int p = pair_start; p < pair_end; p++) {
            int src_node = csr_src[p];
            int tidx_val = csr_tidx[p];

            int t_base = tidx_val * L * 2;
            for (int i = tid; i < L; i += nthreads) {
                float2 val = *reinterpret_cast<const float2*>(&transfer_ri[t_base + i * 2]);
                smem[i * 2]     = val.x;
                smem[i * 2 + 1] = val.y;
            }
            __syncthreads();

            int s_base = src_node * L;
            for (int s = 0; s < n_active; s++) {
                int l = (slot_start + s) * nthreads + tid;
                if (l < L) {
                    float tr = smem[l * 2];
                    float ti = smem[l * 2 + 1];
                    float mr = multi_re[s_base + l];
                    float mi = multi_im[s_base + l];
                    acc_re[s] += tr * mr - ti * mi;
                    acc_im[s] += tr * mi + ti * mr;
                }
            }

            __syncthreads();
        }

        for (int s = 0; s < n_active; s++) {
            int l = (slot_start + s) * nthreads + tid;
            if (l < L) {
                local_re[l_base + l] += acc_re[s];
                local_im[l_base + l] += acc_im[s];
            }
        }
    }
}

// L2L kernel (all float32)
__global__ void l2l_kernel(
    const int*      __restrict__ parent_idx,
    const int*      __restrict__ child_idx,
    const fmm_real* __restrict__ shift_re,
    const fmm_real* __restrict__ shift_im,
    const fmm_real* __restrict__ local_re_in,
    const fmm_real* __restrict__ local_im_in,
    fmm_real* __restrict__ local_re,
    fmm_real* __restrict__ local_im,
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
        float pr = local_re_in[p_base + l];
        float pi = local_im_in[p_base + l];
        float sr = shift_re[shift_base + l];
        float si = shift_im[shift_base + l];
        atomicAdd(&local_re[c_base + l], pr * sr - pi * si);
        atomicAdd(&local_im[c_base + l], pr * si + pi * sr);
    }
}

// L2P kernel: float32 local -> double64 output
__global__ void l2p_kernel(
    const double*   __restrict__ tgt_pts,
    const fmm_real* __restrict__ dirs,
    const fmm_real* __restrict__ weights,
    const fmm_real* __restrict__ local_re,
    const fmm_real* __restrict__ local_im,
    float k_re, float k_im,
    double prefac_re, double prefac_im,
    double* __restrict__ out_re,
    double* __restrict__ out_im,
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

    float ccx = (float)node_centers[node*3];
    float ccy = (float)node_centers[node*3+1];
    float ccz = (float)node_centers[node*3+2];
    int l_base = node * L;

    for (int t = threadIdx.x; t < t_count; t += blockDim.x) {
        int tid = tgt_ids[t_start + t];
        float rx = (float)(tgt_pts[tid*3]   - (double)ccx);
        float ry = (float)(tgt_pts[tid*3+1] - (double)ccy);
        float rz = (float)(tgt_pts[tid*3+2] - (double)ccz);

        float acc_re = 0.0f, acc_im = 0.0f;
        for (int l = 0; l < L; l++) {
            float dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
            float dot = dx*rx + dy*ry + dz*rz;
            float phase_re = -k_im * dot;
            float phase_im = k_re * dot;
            float e_re = expf(phase_re) * cosf(phase_im);
            float e_im = expf(phase_re) * sinf(phase_im);

            float lr = local_re[l_base + l];
            float li = local_im[l_base + l];
            float wl = weights[l];
            float wlr = wl * lr, wli = wl * li;
            acc_re += wlr * e_re - wli * e_im;
            acc_im += wlr * e_im + wli * e_re;
        }

        double final_re = prefac_re * (double)acc_re - prefac_im * (double)acc_im;
        double final_im = prefac_re * (double)acc_im + prefac_im * (double)acc_re;
        atomicAdd(&out_re[tid], final_re);
        atomicAdd(&out_im[tid], final_im);
    }
}

// Repack gradient from 6 separate arrays into interleaved xyz
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

// Vector addition kernel for merging P2P results into FMM results
__global__ void vector_add_kernel(double* __restrict__ dst, const double* __restrict__ src, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] += src[i];
}

// L2P gradient kernel: float32 local -> double64 gradient
__global__ void l2p_gradient_kernel(
    const double*   __restrict__ tgt_pts,
    const fmm_real* __restrict__ dirs,
    const fmm_real* __restrict__ weights,
    const fmm_real* __restrict__ local_re,
    const fmm_real* __restrict__ local_im,
    float k_re, float k_im,
    double prefac_re, double prefac_im,
    float ik2_re, float ik2_im,
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

    float ccx = (float)node_centers[node*3];
    float ccy = (float)node_centers[node*3+1];
    float ccz = (float)node_centers[node*3+2];
    int l_base = node * L;

    for (int t = threadIdx.x; t < t_count; t += blockDim.x) {
        int tid = tgt_ids[t_start + t];
        float rx = (float)(tgt_pts[tid*3]   - (double)ccx);
        float ry = (float)(tgt_pts[tid*3+1] - (double)ccy);
        float rz = (float)(tgt_pts[tid*3+2] - (double)ccz);

        float ax_re = 0, ax_im = 0, ay_re = 0, ay_im = 0, az_re = 0, az_im = 0;

        for (int l = 0; l < L; l++) {
            float dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
            float dot = dx*rx + dy*ry + dz*rz;
            float phase_re = -k_im * dot;
            float phase_im = k_re * dot;
            float e_re = expf(phase_re) * cosf(phase_im);
            float e_im = expf(phase_re) * sinf(phase_im);

            float lr = local_re[l_base + l], li = local_im[l_base + l];
            float wl = weights[l];
            float wlr = wl * lr, wli = wl * li;
            float pr = wlr * e_re - wli * e_im;
            float pi = wlr * e_im + wli * e_re;

            float ikpr = ik2_re * pr - ik2_im * pi;
            float ikpi = ik2_re * pi + ik2_im * pr;

            ax_re += ikpr * dx; ax_im += ikpi * dx;
            ay_re += ikpr * dy; ay_im += ikpi * dy;
            az_re += ikpr * dz; az_im += ikpi * dz;
        }

        atomicAdd(&gx_re[tid], prefac_re*(double)ax_re - prefac_im*(double)ax_im);
        atomicAdd(&gx_im[tid], prefac_re*(double)ax_im + prefac_im*(double)ax_re);
        atomicAdd(&gy_re[tid], prefac_re*(double)ay_re - prefac_im*(double)ay_im);
        atomicAdd(&gy_im[tid], prefac_re*(double)ay_im + prefac_im*(double)ay_re);
        atomicAdd(&gz_re[tid], prefac_re*(double)az_re - prefac_im*(double)az_im);
        atomicAdd(&gz_im[tid], prefac_re*(double)az_im + prefac_im*(double)az_re);
    }
}

// ============================================================
// Batch-2 FMM kernels: two charge vectors, single tree traversal
// Phase/shift factors computed once, applied to both charge sets
// Float32 for FMM internal (dirs, multipole, local, shifts, transfers)
// Double64 for positions, charges, results
// ============================================================

// P2M batch-2: float32 multipole output, double charges
__global__ void p2m_batch2_kernel(
    const double* __restrict__ src_pts,
    const double* __restrict__ q1_re,
    const double* __restrict__ q1_im,
    const double* __restrict__ q2_re,
    const double* __restrict__ q2_im,
    const fmm_real* __restrict__ dirs,
    float k_re, float k_im,
    fmm_real* __restrict__ multi1_re,
    fmm_real* __restrict__ multi1_im,
    fmm_real* __restrict__ multi2_re,
    fmm_real* __restrict__ multi2_im,
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
    int s_count = s_end - s_start;
    if (s_count == 0) return;

    float ccx = (float)node_centers[node*3];
    float ccy = (float)node_centers[node*3+1];
    float ccz = (float)node_centers[node*3+2];

    int base = node * L;

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        float dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
        float acc1_re = 0.0f, acc1_im = 0.0f;
        float acc2_re = 0.0f, acc2_im = 0.0f;

        for (int s = s_start; s < s_end; s++) {
            int sid = src_ids[s];
            float rx = (float)(src_pts[sid*3]   - (double)ccx);
            float ry = (float)(src_pts[sid*3+1] - (double)ccy);
            float rz = (float)(src_pts[sid*3+2] - (double)ccz);
            float dot = dx*rx + dy*ry + dz*rz;

            float phase_re = k_im * dot;
            float phase_im = -k_re * dot;
            float e_re = expf(phase_re) * cosf(phase_im);
            float e_im = expf(phase_re) * sinf(phase_im);

            float qr1 = (float)q1_re[sid], qi1 = (float)q1_im[sid];
            acc1_re += e_re * qr1 - e_im * qi1;
            acc1_im += e_re * qi1 + e_im * qr1;

            float qr2 = (float)q2_re[sid], qi2 = (float)q2_im[sid];
            acc2_re += e_re * qr2 - e_im * qi2;
            acc2_im += e_re * qi2 + e_im * qr2;
        }

        multi1_re[base + l] = acc1_re;
        multi1_im[base + l] = acc1_im;
        multi2_re[base + l] = acc2_re;
        multi2_im[base + l] = acc2_im;
    }
}

// M2M batch-2 (float32)
__global__ void m2m_batch2_kernel(
    const int*      __restrict__ parent_idx,
    const int*      __restrict__ child_idx,
    const fmm_real* __restrict__ shift_re,
    const fmm_real* __restrict__ shift_im,
    fmm_real* __restrict__ multi1_re,
    fmm_real* __restrict__ multi1_im,
    fmm_real* __restrict__ multi2_re,
    fmm_real* __restrict__ multi2_im,
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
        float sr = shift_re[shift_base + l];
        float si = shift_im[shift_base + l];

        float cr1 = multi1_re[c_base + l];
        float ci1 = multi1_im[c_base + l];
        atomicAdd(&multi1_re[p_base + l], cr1 * sr - ci1 * si);
        atomicAdd(&multi1_im[p_base + l], cr1 * si + ci1 * sr);

        float cr2 = multi2_re[c_base + l];
        float ci2 = multi2_im[c_base + l];
        atomicAdd(&multi2_re[p_base + l], cr2 * sr - ci2 * si);
        atomicAdd(&multi2_im[p_base + l], cr2 * si + ci2 * sr);
    }
}

// M2L batch-2 legacy (float32)
__global__ void m2l_batch2_kernel(
    const int*      __restrict__ tgt_idx,
    const int*      __restrict__ src_idx,
    const int*      __restrict__ transfer_idx,
    const fmm_real* __restrict__ transfer_re,
    const fmm_real* __restrict__ transfer_im,
    const fmm_real* __restrict__ multi1_re,
    const fmm_real* __restrict__ multi1_im,
    const fmm_real* __restrict__ multi2_re,
    const fmm_real* __restrict__ multi2_im,
    fmm_real* __restrict__ local1_re,
    fmm_real* __restrict__ local1_im,
    fmm_real* __restrict__ local2_re,
    fmm_real* __restrict__ local2_im,
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
        float tr = transfer_re[t_base + l];
        float ti = transfer_im[t_base + l];

        float mr1 = multi1_re[s_base + l];
        float mi1 = multi1_im[s_base + l];
        atomicAdd(&local1_re[l_base + l], tr * mr1 - ti * mi1);
        atomicAdd(&local1_im[l_base + l], tr * mi1 + ti * mr1);

        float mr2 = multi2_re[s_base + l];
        float mi2 = multi2_im[s_base + l];
        atomicAdd(&local2_re[l_base + l], tr * mr2 - ti * mi2);
        atomicAdd(&local2_im[l_base + l], tr * mi2 + ti * mr2);
    }
}

// Optimized CSR M2L batch-2: one block per unique target, no atomicAdd
// Same algorithm as m2l_target_kernel but processes two multipole/local vectors
__global__ void m2l_target_batch2_kernel(
    const int*      __restrict__ csr_offsets,
    const int*      __restrict__ csr_tgt_nodes,
    const int*      __restrict__ csr_src,
    const int*      __restrict__ csr_tidx,
    const fmm_real* __restrict__ transfer_ri,
    const fmm_real* __restrict__ multi1_re,
    const fmm_real* __restrict__ multi1_im,
    const fmm_real* __restrict__ multi2_re,
    const fmm_real* __restrict__ multi2_im,
    fmm_real* __restrict__ local1_re,
    fmm_real* __restrict__ local1_im,
    fmm_real* __restrict__ local2_re,
    fmm_real* __restrict__ local2_im,
    int L, int n_targets, int tgt_offset, int pair_offset)
{
    int tgt_id = blockIdx.x + tgt_offset;
    if (tgt_id >= tgt_offset + n_targets) return;

    int tgt_node = csr_tgt_nodes[tgt_id];
    int pair_start = csr_offsets[tgt_id] + pair_offset;
    int pair_end   = csr_offsets[tgt_id + 1] + pair_offset;
    int n_src = pair_end - pair_start;
    if (n_src == 0) return;

    int l_base = tgt_node * L;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    extern __shared__ fmm_real smem[];  // size = L * 2

    int n_slots = (L + nthreads - 1) / nthreads;
    const int CHUNK = 4;  // smaller chunk than single (4 accumulators per slot vs 2)
    int n_chunks = (n_slots + CHUNK - 1) / CHUNK;

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int slot_start = chunk * CHUNK;
        int slot_end = slot_start + CHUNK;
        if (slot_end > n_slots) slot_end = n_slots;
        int n_active = slot_end - slot_start;

        float acc1_re[CHUNK], acc1_im[CHUNK];
        float acc2_re[CHUNK], acc2_im[CHUNK];
        for (int s = 0; s < n_active; s++) {
            acc1_re[s] = 0.0f; acc1_im[s] = 0.0f;
            acc2_re[s] = 0.0f; acc2_im[s] = 0.0f;
        }

        for (int p = pair_start; p < pair_end; p++) {
            int src_node = csr_src[p];
            int tidx_val = csr_tidx[p];

            // Load transfer function into shared memory (float2 vectorized)
            int t_base = tidx_val * L * 2;
            for (int i = tid; i < L; i += nthreads) {
                float2 val = *reinterpret_cast<const float2*>(&transfer_ri[t_base + i * 2]);
                smem[i * 2]     = val.x;
                smem[i * 2 + 1] = val.y;
            }
            __syncthreads();

            int s_base = src_node * L;
            for (int s = 0; s < n_active; s++) {
                int l = (slot_start + s) * nthreads + tid;
                if (l < L) {
                    float tr = smem[l * 2];
                    float ti = smem[l * 2 + 1];
                    float mr1 = multi1_re[s_base + l];
                    float mi1 = multi1_im[s_base + l];
                    acc1_re[s] += tr * mr1 - ti * mi1;
                    acc1_im[s] += tr * mi1 + ti * mr1;
                    float mr2 = multi2_re[s_base + l];
                    float mi2 = multi2_im[s_base + l];
                    acc2_re[s] += tr * mr2 - ti * mi2;
                    acc2_im[s] += tr * mi2 + ti * mr2;
                }
            }
            __syncthreads();
        }

        for (int s = 0; s < n_active; s++) {
            int l = (slot_start + s) * nthreads + tid;
            if (l < L) {
                local1_re[l_base + l] += acc1_re[s];
                local1_im[l_base + l] += acc1_im[s];
                local2_re[l_base + l] += acc2_re[s];
                local2_im[l_base + l] += acc2_im[s];
            }
        }
    }
}

// L2L batch-2 (float32)
__global__ void l2l_batch2_kernel(
    const int*      __restrict__ parent_idx,
    const int*      __restrict__ child_idx,
    const fmm_real* __restrict__ shift_re,
    const fmm_real* __restrict__ shift_im,
    const fmm_real* __restrict__ local1_re_in,
    const fmm_real* __restrict__ local1_im_in,
    const fmm_real* __restrict__ local2_re_in,
    const fmm_real* __restrict__ local2_im_in,
    fmm_real* __restrict__ local1_re,
    fmm_real* __restrict__ local1_im,
    fmm_real* __restrict__ local2_re,
    fmm_real* __restrict__ local2_im,
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
        float sr = shift_re[shift_base + l];
        float si = shift_im[shift_base + l];

        float pr1 = local1_re_in[p_base + l];
        float pi1 = local1_im_in[p_base + l];
        atomicAdd(&local1_re[c_base + l], pr1 * sr - pi1 * si);
        atomicAdd(&local1_im[c_base + l], pr1 * si + pi1 * sr);

        float pr2 = local2_re_in[p_base + l];
        float pi2 = local2_im_in[p_base + l];
        atomicAdd(&local2_re[c_base + l], pr2 * sr - pi2 * si);
        atomicAdd(&local2_im[c_base + l], pr2 * si + pi2 * sr);
    }
}

// L2P batch-2 potential: float32 local -> double output
__global__ void l2p_batch2_kernel(
    const double*   __restrict__ tgt_pts,
    const fmm_real* __restrict__ dirs,
    const fmm_real* __restrict__ weights,
    const fmm_real* __restrict__ local1_re,
    const fmm_real* __restrict__ local1_im,
    const fmm_real* __restrict__ local2_re,
    const fmm_real* __restrict__ local2_im,
    float k_re, float k_im,
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

    float ccx = (float)node_centers[node*3];
    float ccy = (float)node_centers[node*3+1];
    float ccz = (float)node_centers[node*3+2];
    int l_base = node * L;

    for (int t = threadIdx.x; t < t_count; t += blockDim.x) {
        int tid = tgt_ids[t_start + t];
        float rx = (float)(tgt_pts[tid*3]   - (double)ccx);
        float ry = (float)(tgt_pts[tid*3+1] - (double)ccy);
        float rz = (float)(tgt_pts[tid*3+2] - (double)ccz);

        float acc1_re = 0.0f, acc1_im = 0.0f;
        float acc2_re = 0.0f, acc2_im = 0.0f;

        for (int l = 0; l < L; l++) {
            float dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
            float dot = dx*rx + dy*ry + dz*rz;

            float phase_re = -k_im * dot;
            float phase_im = k_re * dot;
            float e_re = expf(phase_re) * cosf(phase_im);
            float e_im = expf(phase_re) * sinf(phase_im);

            float wl = weights[l];

            float lr1 = local1_re[l_base + l];
            float li1 = local1_im[l_base + l];
            float wlr1 = wl * lr1, wli1 = wl * li1;
            acc1_re += wlr1 * e_re - wli1 * e_im;
            acc1_im += wlr1 * e_im + wli1 * e_re;

            float lr2 = local2_re[l_base + l];
            float li2 = local2_im[l_base + l];
            float wlr2 = wl * lr2, wli2 = wl * li2;
            acc2_re += wlr2 * e_re - wli2 * e_im;
            acc2_im += wlr2 * e_im + wli2 * e_re;
        }

        double f1_re = prefac_re * (double)acc1_re - prefac_im * (double)acc1_im;
        double f1_im = prefac_re * (double)acc1_im + prefac_im * (double)acc1_re;
        atomicAdd(&out1_re[tid], f1_re);
        atomicAdd(&out1_im[tid], f1_im);

        double f2_re = prefac_re * (double)acc2_re - prefac_im * (double)acc2_im;
        double f2_im = prefac_re * (double)acc2_im + prefac_im * (double)acc2_re;
        atomicAdd(&out2_re[tid], f2_re);
        atomicAdd(&out2_im[tid], f2_im);
    }
}

// L2P batch-2 gradient: float32 local -> double gradient
__global__ void l2p_gradient_batch2_kernel(
    const double*   __restrict__ tgt_pts,
    const fmm_real* __restrict__ dirs,
    const fmm_real* __restrict__ weights,
    const fmm_real* __restrict__ local1_re,
    const fmm_real* __restrict__ local1_im,
    const fmm_real* __restrict__ local2_re,
    const fmm_real* __restrict__ local2_im,
    float k_re, float k_im,
    double prefac_re, double prefac_im,
    float ik2_re, float ik2_im,
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

    float ccx = (float)node_centers[node*3];
    float ccy = (float)node_centers[node*3+1];
    float ccz = (float)node_centers[node*3+2];
    int l_base = node * L;

    for (int t = threadIdx.x; t < t_count; t += blockDim.x) {
        int tid = tgt_ids[t_start + t];
        float rx = (float)(tgt_pts[tid*3]   - (double)ccx);
        float ry = (float)(tgt_pts[tid*3+1] - (double)ccy);
        float rz = (float)(tgt_pts[tid*3+2] - (double)ccz);

        float ax1_re = 0, ax1_im = 0, ay1_re = 0, ay1_im = 0, az1_re = 0, az1_im = 0;
        float ax2_re = 0, ax2_im = 0, ay2_re = 0, ay2_im = 0, az2_re = 0, az2_im = 0;

        for (int l = 0; l < L; l++) {
            float dx = dirs[l*3], dy = dirs[l*3+1], dz = dirs[l*3+2];
            float dot = dx*rx + dy*ry + dz*rz;

            float phase_re = -k_im * dot;
            float phase_im = k_re * dot;
            float e_re = expf(phase_re) * cosf(phase_im);
            float e_im = expf(phase_re) * sinf(phase_im);

            float wl = weights[l];

            float lr1 = local1_re[l_base + l], li1 = local1_im[l_base + l];
            float wlr1 = wl * lr1, wli1 = wl * li1;
            float pr1 = wlr1 * e_re - wli1 * e_im;
            float pi1 = wlr1 * e_im + wli1 * e_re;
            float ikpr1 = ik2_re * pr1 - ik2_im * pi1;
            float ikpi1 = ik2_re * pi1 + ik2_im * pr1;
            ax1_re += ikpr1 * dx; ax1_im += ikpi1 * dx;
            ay1_re += ikpr1 * dy; ay1_im += ikpi1 * dy;
            az1_re += ikpr1 * dz; az1_im += ikpi1 * dz;

            float lr2 = local2_re[l_base + l], li2 = local2_im[l_base + l];
            float wlr2 = wl * lr2, wli2 = wl * li2;
            float pr2 = wlr2 * e_re - wli2 * e_im;
            float pi2 = wlr2 * e_im + wli2 * e_re;
            float ikpr2 = ik2_re * pr2 - ik2_im * pi2;
            float ikpi2 = ik2_re * pi2 + ik2_im * pr2;
            ax2_re += ikpr2 * dx; ax2_im += ikpi2 * dx;
            ay2_re += ikpr2 * dy; ay2_im += ikpi2 * dy;
            az2_re += ikpr2 * dz; az2_im += ikpi2 * dz;
        }

        atomicAdd(&gx1_re[tid], prefac_re*(double)ax1_re - prefac_im*(double)ax1_im);
        atomicAdd(&gx1_im[tid], prefac_re*(double)ax1_im + prefac_im*(double)ax1_re);
        atomicAdd(&gy1_re[tid], prefac_re*(double)ay1_re - prefac_im*(double)ay1_im);
        atomicAdd(&gy1_im[tid], prefac_re*(double)ay1_im + prefac_im*(double)ay1_re);
        atomicAdd(&gz1_re[tid], prefac_re*(double)az1_re - prefac_im*(double)az1_im);
        atomicAdd(&gz1_im[tid], prefac_re*(double)az1_im + prefac_im*(double)az1_re);

        atomicAdd(&gx2_re[tid], prefac_re*(double)ax2_re - prefac_im*(double)ax2_im);
        atomicAdd(&gx2_im[tid], prefac_re*(double)ax2_im + prefac_im*(double)ax2_re);
        atomicAdd(&gy2_re[tid], prefac_re*(double)ay2_re - prefac_im*(double)ay2_im);
        atomicAdd(&gy2_im[tid], prefac_re*(double)ay2_im + prefac_im*(double)ay2_re);
        atomicAdd(&gz2_re[tid], prefac_re*(double)az2_re - prefac_im*(double)az2_im);
        atomicAdd(&gz2_im[tid], prefac_re*(double)az2_im + prefac_im*(double)az2_re);
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

    // Sphere quadrature — L based on COARSEST M2L level for angular resolution.
    // Transfer function truncation is per-level (see M2L precompute below).
    double leaf_hs = tree.nodes[tree.leaves[0]].half_size;
    double leaf_box_size = 2.0 * leaf_hs;
    // Quadrature p must resolve multipoles at the coarsest M2L level (level 2).
    // p_quad = truncation_order(k, coarsest_box) ensures angular resolution.
    double coarsest_m2l_box = leaf_box_size;
    if (tree.max_level > 2)
        coarsest_m2l_box = leaf_box_size * (1 << (tree.max_level - 2));
    p = fmm_truncation_order(std::abs(k), coarsest_m2l_box, digits);
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

    // Build P2P CSR with ORIGINAL indices
    std::vector<std::vector<int>> tgt_src_lists(Nt);

    for (int li_idx = 0; li_idx < n_leaves; li_idx++) {
        int node_idx = leaf_info[li_idx].node_idx;
        const OctreeNode& leaf = tree.nodes[node_idx];

        std::vector<int> all_src_orig;
        for (int i = leaf.pt_start; i < leaf.pt_start + leaf.pt_count; i++) {
            int orig = tree.sorted_idx[i];
            if (orig >= Nt)
                all_src_orig.push_back(orig - Nt);
        }
        for (int ni = leaf.near_start; ni < leaf.near_start + leaf.near_count; ni++) {
            int nb_node = tree.near_list[ni];
            const OctreeNode& nb = tree.nodes[nb_node];
            for (int i = nb.pt_start; i < nb.pt_start + nb.pt_count; i++) {
                int orig = tree.sorted_idx[i];
                if (orig >= Nt)
                    all_src_orig.push_back(orig - Nt);
            }
        }

        for (int i = leaf.pt_start; i < leaf.pt_start + leaf.pt_count; i++) {
            int orig = tree.sorted_idx[i];
            if (orig < Nt)
                tgt_src_lists[orig] = all_src_orig;
        }
    }

    p2p_offsets.resize(Nt + 1, 0);
    p2p_indices.clear();
    for (int t = 0; t < Nt; t++) {
        p2p_offsets[t + 1] = p2p_offsets[t] + (int)tgt_src_lists[t].size();
        for (int si : tgt_src_lists[t])
            p2p_indices.push_back(si);
    }
    p2p_nnz = (int)p2p_indices.size();

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

        // Per-level transfer truncation order:
        // At coarser levels, boxes are bigger → need more Legendre terms.
        // This is safe because kd_min = 2*ka_box grows proportionally,
        // so the Hankel series converges and |T| stays bounded.
        double level_box_size = leaf_box_size * (1 << (tree.max_level - level));
        int p_transfer = fmm_truncation_order(std::abs(k), level_box_size, digits);
        // p_transfer is naturally correct for each level (including leaf)

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
                        if (p_transfer >= 1)
                            sum += 3.0 * cdouble(0, 1) * spherical_hankel1(1, kd) * P_curr;
                        cdouble i_pow(0, 1);
                        for (int l = 2; l <= p_transfer; l++) {
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
    // Print per-level transfer info
    for (int level = 2; level <= tree.max_level; level++) {
        double lvl_box = leaf_box_size * (1 << (tree.max_level - level));
        int pt = fmm_truncation_order(std::abs(k), lvl_box, digits);
        if (m2l_batches[level].n_pairs > 0)
            printf("  [FMM] Level %d: box=%.4f, ka_box=%.2f, p_transfer=%d, %d pairs\n",
                   level, lvl_box, std::abs(k)*lvl_box, pt, m2l_batches[level].n_pairs);
    }

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

    // Target and source positions in ORIGINAL order (double)
    CUDA_CHECK(cudaMalloc(&d_tgt_pts, Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_src_pts, Ns * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_tgt_pts, targets, Nt * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_pts, sources, Ns * 3 * sizeof(double), cudaMemcpyHostToDevice));

    // P2P CSR
    CUDA_CHECK(cudaMalloc(&d_p2p_offsets, (Nt + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_p2p_offsets, p2p_offsets.data(), (Nt + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (p2p_nnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_p2p_indices, p2p_nnz * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_p2p_indices, p2p_indices.data(), p2p_nnz * sizeof(int), cudaMemcpyHostToDevice));
    }

    // FMM arrays (float32)
    CUDA_CHECK(cudaMalloc(&d_multi_re, n_nodes * L * sizeof(fmm_real)));
    CUDA_CHECK(cudaMalloc(&d_multi_im, n_nodes * L * sizeof(fmm_real)));
    CUDA_CHECK(cudaMalloc(&d_local_re, n_nodes * L * sizeof(fmm_real)));
    CUDA_CHECK(cudaMalloc(&d_local_im, n_nodes * L * sizeof(fmm_real)));

    // Transfers (float32, separate re/im for legacy M2L + interleaved for CSR M2L)
    if (n_unique_transfers > 0) {
        int tc_size = n_unique_transfers * L;
        std::vector<fmm_real> t_re(tc_size), t_im(tc_size);
        for (int i = 0; i < tc_size; i++) {
            t_re[i] = (fmm_real)transfer_cache[i].real();
            t_im[i] = (fmm_real)transfer_cache[i].imag();
        }
        CUDA_CHECK(cudaMalloc(&d_transfer_re, tc_size * sizeof(fmm_real)));
        CUDA_CHECK(cudaMalloc(&d_transfer_im, tc_size * sizeof(fmm_real)));
        CUDA_CHECK(cudaMemcpy(d_transfer_re, t_re.data(), tc_size * sizeof(fmm_real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_transfer_im, t_im.data(), tc_size * sizeof(fmm_real), cudaMemcpyHostToDevice));

        // Interleaved [re,im] for CSR M2L kernel (float32, float2 vectorized loads)
        std::vector<fmm_real> t_ri(tc_size * 2);
        for (int i = 0; i < tc_size; i++) {
            t_ri[i * 2]     = (fmm_real)transfer_cache[i].real();
            t_ri[i * 2 + 1] = (fmm_real)transfer_cache[i].imag();
        }
        CUDA_CHECK(cudaMalloc(&d_transfer_ri, tc_size * 2 * sizeof(fmm_real)));
        CUDA_CHECK(cudaMemcpy(d_transfer_ri, t_ri.data(), tc_size * 2 * sizeof(fmm_real), cudaMemcpyHostToDevice));
    }

    // Optimized M2L: build target-sorted CSR structure per level
    {
        std::vector<int> all_csr_offsets, all_csr_tgt_nodes, all_csr_src, all_csr_tidx;
        m2l_csr_level_info.resize(tree.max_level + 1);
        m2l_level_info.resize(tree.max_level + 1);

        for (int level = 0; level <= tree.max_level; level++) {
            m2l_csr_level_info[level].offsets_start = (int)all_csr_offsets.size();
            m2l_csr_level_info[level].nodes_start = (int)all_csr_tgt_nodes.size();
            m2l_csr_level_info[level].pair_offset = (int)all_csr_src.size();
            m2l_csr_level_info[level].n_targets = 0;

            m2l_level_info[level].offset = 0;
            m2l_level_info[level].count = 0;

            if (level >= (int)m2l_batches.size() || m2l_batches[level].n_pairs == 0)
                continue;

            const M2LBatch& b = m2l_batches[level];

            std::map<int, std::vector<std::pair<int,int>>> tgt_map;
            for (int i = 0; i < b.n_pairs; i++) {
                tgt_map[b.tgt_idx[i]].push_back({b.src_idx[i], b.transfer_idx[i]});
            }

            int level_pair_base = (int)all_csr_src.size();
            for (auto& kv : tgt_map) {
                int tgt_node = kv.first;
                const auto& pairs = kv.second;

                all_csr_offsets.push_back((int)all_csr_src.size() - level_pair_base);
                all_csr_tgt_nodes.push_back(tgt_node);

                for (auto& pr : pairs) {
                    all_csr_src.push_back(pr.first);
                    all_csr_tidx.push_back(pr.second);
                }
            }
            all_csr_offsets.push_back((int)all_csr_src.size() - level_pair_base);

            m2l_csr_level_info[level].n_targets = (int)tgt_map.size();
        }

        if (!all_csr_tgt_nodes.empty()) {
            CUDA_CHECK(cudaMalloc(&d_m2l_csr_offsets, all_csr_offsets.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_m2l_csr_tgt_nodes, all_csr_tgt_nodes.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_m2l_csr_src, all_csr_src.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_m2l_csr_tidx, all_csr_tidx.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_m2l_csr_offsets, all_csr_offsets.data(),
                                  all_csr_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_m2l_csr_tgt_nodes, all_csr_tgt_nodes.data(),
                                  all_csr_tgt_nodes.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_m2l_csr_src, all_csr_src.data(),
                                  all_csr_src.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_m2l_csr_tidx, all_csr_tidx.data(),
                                  all_csr_tidx.size() * sizeof(int), cudaMemcpyHostToDevice));

            int total_targets = 0;
            for (int level = 0; level <= tree.max_level; level++)
                total_targets += m2l_csr_level_info[level].n_targets;
            printf("  [FMM] M2L CSR: %d unique targets, %d total pairs\n",
                   total_targets, (int)all_csr_src.size());
        }
    }

    // Legacy M2L batch arrays (kept for batch2 M2L fallback)
    {
        std::vector<int> all_tgt, all_src, all_tidx;
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

    // M2M shifts (concat, float32)
    {
        std::vector<int> all_p, all_c;
        std::vector<fmm_real> all_s_re, all_s_im;
        m2m_level_info.resize(tree.max_level + 1);
        for (int level = 0; level <= tree.max_level; level++) {
            m2m_level_info[level].offset = (int)all_p.size();
            m2m_level_info[level].count = 0;
            if (level < (int)m2m_data.size() && !m2m_data[level].pairs.empty()) {
                const LevelShifts& ls = m2m_data[level];
                for (auto& pr : ls.pairs) { all_p.push_back(pr.parent); all_c.push_back(pr.child); }
                for (auto& s : ls.shifts) { all_s_re.push_back((fmm_real)s.real()); all_s_im.push_back((fmm_real)s.imag()); }
                m2m_level_info[level].count = (int)ls.pairs.size();
            }
        }
        if (!all_p.empty()) {
            CUDA_CHECK(cudaMalloc(&d_m2m_parent, all_p.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_m2m_child, all_c.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_m2m_shift_re, all_s_re.size() * sizeof(fmm_real)));
            CUDA_CHECK(cudaMalloc(&d_m2m_shift_im, all_s_im.size() * sizeof(fmm_real)));
            CUDA_CHECK(cudaMemcpy(d_m2m_parent, all_p.data(), all_p.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_m2m_child, all_c.data(), all_c.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_m2m_shift_re, all_s_re.data(), all_s_re.size() * sizeof(fmm_real), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_m2m_shift_im, all_s_im.data(), all_s_im.size() * sizeof(fmm_real), cudaMemcpyHostToDevice));
        }
    }

    // L2L shifts (concat, float32)
    {
        std::vector<int> all_p, all_c;
        std::vector<fmm_real> all_s_re, all_s_im;
        l2l_level_info.resize(tree.max_level + 1);
        for (int level = 0; level <= tree.max_level; level++) {
            l2l_level_info[level].offset = (int)all_p.size();
            l2l_level_info[level].count = 0;
            if (level < (int)l2l_data.size() && !l2l_data[level].pairs.empty()) {
                const LevelShifts& ls = l2l_data[level];
                for (auto& pr : ls.pairs) { all_p.push_back(pr.parent); all_c.push_back(pr.child); }
                for (auto& s : ls.shifts) { all_s_re.push_back((fmm_real)s.real()); all_s_im.push_back((fmm_real)s.imag()); }
                l2l_level_info[level].count = (int)ls.pairs.size();
            }
        }
        if (!all_p.empty()) {
            CUDA_CHECK(cudaMalloc(&d_l2l_parent, all_p.data() ? all_p.size() * sizeof(int) : sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_l2l_child, all_c.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_l2l_shift_re, all_s_re.size() * sizeof(fmm_real)));
            CUDA_CHECK(cudaMalloc(&d_l2l_shift_im, all_s_im.size() * sizeof(fmm_real)));
            CUDA_CHECK(cudaMemcpy(d_l2l_parent, all_p.data(), all_p.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_l2l_child, all_c.data(), all_c.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_l2l_shift_re, all_s_re.data(), all_s_re.size() * sizeof(fmm_real), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_l2l_shift_im, all_s_im.data(), all_s_im.size() * sizeof(fmm_real), cudaMemcpyHostToDevice));
        }
    }

    // Charge/result buffers (double)
    CUDA_CHECK(cudaMalloc(&d_charges_re, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_charges_im, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result_re, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result_im, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_re, Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_im, Nt * 3 * sizeof(double)));

    // Batch-2 workspace (charges/results double, multipole/local float32)
    CUDA_CHECK(cudaMalloc(&d_charges2_re, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_charges2_im, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result2_re, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result2_im, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_multi2_re, n_nodes * L * sizeof(fmm_real)));
    CUDA_CHECK(cudaMalloc(&d_multi2_im, n_nodes * L * sizeof(fmm_real)));
    CUDA_CHECK(cudaMalloc(&d_local2_re, n_nodes * L * sizeof(fmm_real)));
    CUDA_CHECK(cudaMalloc(&d_local2_im, n_nodes * L * sizeof(fmm_real)));
    CUDA_CHECK(cudaMalloc(&d_grad2_re, Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad2_im, Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gy2_re_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gy2_im_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gz2_re_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gz2_im_cached, Nt * sizeof(double)));

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

    // Cached GPU arrays
    CUDA_CHECK(cudaMalloc(&d_node_centers_cached, n_nodes * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dirs_cached, L * 3 * sizeof(fmm_real)));
    CUDA_CHECK(cudaMalloc(&d_weights_cached, L * sizeof(fmm_real)));
    CUDA_CHECK(cudaMalloc(&d_leaf_idx_cached, n_leaves * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_src_id_offsets_cached, (n_leaves + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_src_ids_cached, std::max((int)h_src_ids_flat.size(), 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tgt_id_offsets_cached, (n_leaves + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tgt_ids_cached, std::max((int)h_tgt_ids_flat.size(), 1) * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_node_centers_cached, h_node_centers.data(), n_nodes * 3 * sizeof(double), cudaMemcpyHostToDevice));
    // Convert dirs/weights from double to float for upload
    {
        std::vector<fmm_real> dirs_f(L * 3), weights_f(L);
        for (int i = 0; i < L * 3; i++) dirs_f[i] = (fmm_real)squad.dirs[i];
        for (int i = 0; i < L; i++) weights_f[i] = (fmm_real)squad.weights[i];
        CUDA_CHECK(cudaMemcpy(d_dirs_cached, dirs_f.data(), L * 3 * sizeof(fmm_real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights_cached, weights_f.data(), L * sizeof(fmm_real), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_leaf_idx_cached, h_leaf_indices.data(), n_leaves * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_id_offsets_cached, h_src_id_offsets.data(), (n_leaves + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (!h_src_ids_flat.empty())
        CUDA_CHECK(cudaMemcpy(d_src_ids_cached, h_src_ids_flat.data(), h_src_ids_flat.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt_id_offsets_cached, h_tgt_id_offsets.data(), (n_leaves + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (!h_tgt_ids_flat.empty())
        CUDA_CHECK(cudaMemcpy(d_tgt_ids_cached, h_tgt_ids_flat.data(), h_tgt_ids_flat.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Cached gradient workspace arrays
    CUDA_CHECK(cudaMalloc(&d_gy_re_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gy_im_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gz_re_cached, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gz_im_cached, Nt * sizeof(double)));

    // Pre-allocated temp buffers for gradient repack
    CUDA_CHECK(cudaMalloc(&d_gx_re_tmp, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gx_im_tmp, Nt * sizeof(double)));

    // Pre-allocated host-side re/im split buffers
    h_q_re_buf.resize(Ns); h_q_im_buf.resize(Ns);
    h_q2_re_buf.resize(Ns); h_q2_im_buf.resize(Ns);
    int max_buf = std::max(Nt, Nt * 3);
    h_res_re_buf.resize(max_buf); h_res_im_buf.resize(max_buf);
    h_res2_re_buf.resize(max_buf); h_res2_im_buf.resize(max_buf);

    // CUDA streams for P2P/FMM pipeline overlap
    CUDA_CHECK(cudaStreamCreate(&stream_fmm));
    CUDA_CHECK(cudaStreamCreate(&stream_p2p));

    // P2P output buffers (separate from FMM to allow concurrent execution)
    CUDA_CHECK(cudaMalloc(&d_p2p_pot_re, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_pot_im, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_pot2_re, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_pot2_im, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_gx_re, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_gx_im, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_gy_re, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_gy_im, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_gz_re, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_gz_im, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_gx2_re, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_gx2_im, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_gy2_re, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_gy2_im, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_gz2_re, Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p2p_gz2_im, Nt * sizeof(double)));

    initialized = true;
    printf("  [FMM] Init complete: p=%d L=%d, %d nodes, %.1fms\n",
           p, L, n_nodes, timer.elapsed_ms());
}

// Helper to run the full FMM tree traversal (float32 FMM + CSR M2L)
// P2P runs concurrently with FMM tree on separate CUDA stream
void HelmholtzFMM::run_tree(const double* h_q_re, const double* h_q_im, bool need_grad)
{
    int n_leaves = (int)leaf_info.size();

    // Upload charges (synchronous — data visible to all streams after return)
    CUDA_CHECK(cudaMemcpy(d_charges_re, h_q_re, Ns * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_charges_im, h_q_im, Ns * sizeof(double), cudaMemcpyHostToDevice));

    // Use cached GPU arrays
    double *d_node_centers = d_node_centers_cached;
    fmm_real *d_dirs = d_dirs_cached;
    fmm_real *d_weights = d_weights_cached;
    int *d_leaf_idx = d_leaf_idx_cached;
    int *d_src_id_offsets = d_src_id_offsets_cached;
    int *d_src_ids = d_src_ids_cached;
    int *d_tgt_id_offsets = d_tgt_id_offsets_cached;
    int *d_tgt_ids = d_tgt_ids_cached;

    float k_re_f = (float)k.real(), k_im_f = (float)k.imag();
    int block_L = std::min(L, 256);

    // === Launch P2P on stream_p2p (concurrent with FMM tree) ===
    if (!need_grad) {
        cudaMemsetAsync(d_p2p_pot_re, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_pot_im, 0, Nt * sizeof(double), stream_p2p);
        launch_p2p_potential(d_tgt_pts, d_src_pts,
            d_charges_re, d_charges_im,
            d_p2p_offsets, d_p2p_indices,
            k.real(), k.imag(),
            d_p2p_pot_re, d_p2p_pot_im, Nt, stream_p2p);
    } else {
        cudaMemsetAsync(d_p2p_gx_re, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gx_im, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gy_re, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gy_im, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gz_re, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gz_im, 0, Nt * sizeof(double), stream_p2p);
        launch_p2p_gradient(d_tgt_pts, d_src_pts,
            d_charges_re, d_charges_im,
            d_p2p_offsets, d_p2p_indices,
            k.real(), k.imag(),
            d_p2p_gx_re, d_p2p_gx_im,
            d_p2p_gy_re, d_p2p_gy_im,
            d_p2p_gz_re, d_p2p_gz_im, Nt, stream_p2p);
    }

    // === FMM tree on stream_fmm (concurrent with P2P) ===
    // Clear multipole/local arrays
    cudaMemsetAsync(d_multi_re, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);
    cudaMemsetAsync(d_multi_im, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);
    cudaMemsetAsync(d_local_re, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);
    cudaMemsetAsync(d_local_im, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);

    // P2M
    if (n_leaves > 0) {
        p2m_kernel<<<n_leaves, block_L, 0, stream_fmm>>>(
            d_src_pts, d_charges_re, d_charges_im,
            d_dirs, k_re_f, k_im_f,
            d_multi_re, d_multi_im,
            d_leaf_idx, d_src_id_offsets, d_src_ids,
            d_node_centers, L, n_leaves);
    }

    // M2M (bottom-up) — stream ordering guarantees sequential execution
    for (int level = tree.max_level - 1; level >= 1; level--) {
        if (level < (int)m2m_level_info.size() && m2m_level_info[level].count > 0) {
            int off = m2m_level_info[level].offset;
            int cnt = m2m_level_info[level].count;
            m2m_kernel<<<cnt, block_L, 0, stream_fmm>>>(d_m2m_parent, d_m2m_child,
                d_m2m_shift_re, d_m2m_shift_im, d_multi_re, d_multi_im, L, cnt, off);
        }
    }

    // M2L (optimized CSR)
    {
        int smem_size = L * 2 * sizeof(fmm_real);
        for (int level = 1; level <= tree.max_level; level++) {
            if (level < (int)m2l_csr_level_info.size() && m2l_csr_level_info[level].n_targets > 0) {
                int n_tgts = m2l_csr_level_info[level].n_targets;
                int off_off = m2l_csr_level_info[level].offsets_start;
                int nodes_off = m2l_csr_level_info[level].nodes_start;
                int pair_off = m2l_csr_level_info[level].pair_offset;
                m2l_target_kernel<<<n_tgts, block_L, smem_size, stream_fmm>>>(
                    d_m2l_csr_offsets + off_off,
                    d_m2l_csr_tgt_nodes + nodes_off,
                    d_m2l_csr_src,
                    d_m2l_csr_tidx,
                    d_transfer_ri,
                    d_multi_re, d_multi_im,
                    d_local_re, d_local_im,
                    L, n_tgts, 0, pair_off);
            }
        }
    }

    // L2L (top-down)
    for (int level = 2; level <= tree.max_level; level++) {
        if (level < (int)l2l_level_info.size() && l2l_level_info[level].count > 0) {
            int off = l2l_level_info[level].offset;
            int cnt = l2l_level_info[level].count;
            l2l_kernel<<<cnt, block_L, 0, stream_fmm>>>(d_l2l_parent, d_l2l_child,
                d_l2l_shift_re, d_l2l_shift_im,
                d_local_re, d_local_im, d_local_re, d_local_im, L, cnt, off);
        }
    }

    cdouble ik_val = cdouble(0, 1) * k;
    cdouble prefactor = ik_val / (16.0 * M_PI * M_PI);
    int merge_grid = (Nt + 255) / 256;

    if (!need_grad) {
        // L2P (potential) on stream_fmm
        cudaMemsetAsync(d_result_re, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_result_im, 0, Nt * sizeof(double), stream_fmm);
        if (n_leaves > 0) {
            l2p_kernel<<<n_leaves, 256, 0, stream_fmm>>>(
                d_tgt_pts, d_dirs, d_weights,
                d_local_re, d_local_im,
                k_re_f, k_im_f, prefactor.real(), prefactor.imag(),
                d_result_re, d_result_im,
                d_leaf_idx, d_tgt_id_offsets, d_tgt_ids,
                d_node_centers, L, n_leaves);
        }

        // Wait for both streams
        CUDA_CHECK(cudaStreamSynchronize(stream_fmm));
        CUDA_CHECK(cudaStreamSynchronize(stream_p2p));

        // Merge P2P results into FMM results
        vector_add_kernel<<<merge_grid, 256>>>(d_result_re, d_p2p_pot_re, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_result_im, d_p2p_pot_im, Nt);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        // L2P (gradient) on stream_fmm
        double *d_gx_re = d_grad_re, *d_gx_im = d_grad_im;
        double *d_gy_re = d_gy_re_cached, *d_gy_im = d_gy_im_cached;
        double *d_gz_re = d_gz_re_cached, *d_gz_im = d_gz_im_cached;
        cudaMemsetAsync(d_gx_re, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gx_im, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gy_re, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gy_im, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gz_re, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gz_im, 0, Nt * sizeof(double), stream_fmm);

        if (n_leaves > 0) {
            l2p_gradient_kernel<<<n_leaves, 256, 0, stream_fmm>>>(
                d_tgt_pts, d_dirs, d_weights,
                d_local_re, d_local_im,
                k_re_f, k_im_f, prefactor.real(), prefactor.imag(),
                (float)ik_val.real(), (float)ik_val.imag(),
                d_gx_re, d_gx_im, d_gy_re, d_gy_im, d_gz_re, d_gz_im,
                d_leaf_idx, d_tgt_id_offsets, d_tgt_ids,
                d_node_centers, L, n_leaves);
        }

        // Wait for both streams
        CUDA_CHECK(cudaStreamSynchronize(stream_fmm));
        CUDA_CHECK(cudaStreamSynchronize(stream_p2p));

        // Merge P2P gradient results
        vector_add_kernel<<<merge_grid, 256>>>(d_gx_re, d_p2p_gx_re, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gx_im, d_p2p_gx_im, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gy_re, d_p2p_gy_re, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gy_im, d_p2p_gy_im, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gz_re, d_p2p_gz_re, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gz_im, d_p2p_gz_im, Nt);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Repack gradient [gx,gy,gz] -> interleaved [x0,y0,z0,x1,y1,z1,...]
        // Use pre-allocated temp buffers (allocated in init())
        CUDA_CHECK(cudaMemcpy(this->d_gx_re_tmp, d_gx_re, Nt * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(this->d_gx_im_tmp, d_gx_im, Nt * sizeof(double), cudaMemcpyDeviceToDevice));

        int repack_block = 256;
        int repack_grid = (Nt + repack_block - 1) / repack_block;
        repack_gradient_kernel<<<repack_grid, repack_block>>>(
            this->d_gx_re_tmp, this->d_gx_im_tmp, d_gy_re, d_gy_im, d_gz_re, d_gz_im,
            d_grad_re, d_grad_im, Nt);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Fused batch-2 tree traversal: two charge vectors, single tree walk (float32 FMM)
// Fused batch-2 tree traversal: two charge vectors, single tree walk (float32 FMM)
// P2P runs concurrently with FMM tree on separate CUDA stream
void HelmholtzFMM::run_tree_batch2(
    const double* h_q1_re, const double* h_q1_im,
    const double* h_q2_re, const double* h_q2_im,
    bool need_grad)
{
    // Profiling: set BEM_PROFILE=1 to print per-stage timing (first 3 calls)
    static int profile_calls = 0;
    static bool profile_enabled = false;
    static bool profile_checked = false;
    if (!profile_checked) {
        const char* env = getenv("BEM_PROFILE");
        profile_enabled = (env && atoi(env) > 0);
        profile_checked = true;
    }
    bool do_profile = profile_enabled && (profile_calls < 3);
    if (do_profile) profile_calls++;

    cudaEvent_t ev_start, ev_upload, ev_p2p_launch, ev_p2m, ev_m2m, ev_m2l, ev_l2l, ev_l2p, ev_merge, ev_end;
    if (do_profile) {
        cudaEventCreate(&ev_start); cudaEventCreate(&ev_upload);
        cudaEventCreate(&ev_p2p_launch); cudaEventCreate(&ev_p2m);
        cudaEventCreate(&ev_m2m); cudaEventCreate(&ev_m2l);
        cudaEventCreate(&ev_l2l); cudaEventCreate(&ev_l2p);
        cudaEventCreate(&ev_merge); cudaEventCreate(&ev_end);
        cudaEventRecord(ev_start);
    }

    int n_leaves = (int)leaf_info.size();

    // Upload both charge vectors (synchronous — data visible to all streams)
    CUDA_CHECK(cudaMemcpy(d_charges_re, h_q1_re, Ns * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_charges_im, h_q1_im, Ns * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_charges2_re, h_q2_re, Ns * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_charges2_im, h_q2_im, Ns * sizeof(double), cudaMemcpyHostToDevice));
    if (do_profile) cudaEventRecord(ev_upload);

    // Cached GPU array aliases
    double *d_node_centers = d_node_centers_cached;
    fmm_real *d_dirs = d_dirs_cached;
    fmm_real *d_weights = d_weights_cached;
    int *d_leaf_idx = d_leaf_idx_cached;
    int *d_src_id_offsets = d_src_id_offsets_cached;
    int *d_src_ids = d_src_ids_cached;
    int *d_tgt_id_offsets = d_tgt_id_offsets_cached;
    int *d_tgt_ids = d_tgt_ids_cached;

    float k_re_f = (float)k.real(), k_im_f = (float)k.imag();
    int block_L = std::min(L, 256);

    // === Launch P2P on stream_p2p (concurrent with FMM tree) ===
    if (!need_grad) {
        // P2P batch-2 potential into separate buffers
        cudaMemsetAsync(d_p2p_pot_re, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_pot_im, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_pot2_re, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_pot2_im, 0, Nt * sizeof(double), stream_p2p);
        launch_p2p_potential_batch2(
            d_tgt_pts, d_src_pts,
            d_charges_re, d_charges_im,
            d_charges2_re, d_charges2_im,
            d_p2p_offsets, d_p2p_indices,
            k.real(), k.imag(),
            d_p2p_pot_re, d_p2p_pot_im,
            d_p2p_pot2_re, d_p2p_pot2_im, Nt, stream_p2p);
    } else {
        // P2P batch-2 pot+grad into separate buffers
        cudaMemsetAsync(d_p2p_pot_re, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_pot_im, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_pot2_re, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_pot2_im, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gx_re, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gx_im, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gy_re, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gy_im, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gz_re, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gz_im, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gx2_re, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gx2_im, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gy2_re, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gy2_im, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gz2_re, 0, Nt * sizeof(double), stream_p2p);
        cudaMemsetAsync(d_p2p_gz2_im, 0, Nt * sizeof(double), stream_p2p);
        launch_p2p_pot_grad_batch2(Nt,
            d_tgt_pts, d_src_pts,
            d_charges_re, d_charges_im,
            d_charges2_re, d_charges2_im,
            d_p2p_offsets, d_p2p_indices,
            k.real(), k.imag(),
            d_p2p_pot_re, d_p2p_pot_im,
            d_p2p_pot2_re, d_p2p_pot2_im,
            d_p2p_gx_re, d_p2p_gx_im, d_p2p_gy_re, d_p2p_gy_im, d_p2p_gz_re, d_p2p_gz_im,
            d_p2p_gx2_re, d_p2p_gx2_im, d_p2p_gy2_re, d_p2p_gy2_im, d_p2p_gz2_re, d_p2p_gz2_im,
            stream_p2p);
    }
    if (do_profile) cudaEventRecord(ev_p2p_launch);

    // === FMM tree on stream_fmm (concurrent with P2P) ===
    // Clear multipole/local arrays
    cudaMemsetAsync(d_multi_re, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);
    cudaMemsetAsync(d_multi_im, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);
    cudaMemsetAsync(d_local_re, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);
    cudaMemsetAsync(d_local_im, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);
    cudaMemsetAsync(d_multi2_re, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);
    cudaMemsetAsync(d_multi2_im, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);
    cudaMemsetAsync(d_local2_re, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);
    cudaMemsetAsync(d_local2_im, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);

    // P2M batch-2
    if (n_leaves > 0) {
        p2m_batch2_kernel<<<n_leaves, block_L, 0, stream_fmm>>>(
            d_src_pts,
            d_charges_re, d_charges_im,
            d_charges2_re, d_charges2_im,
            d_dirs, k_re_f, k_im_f,
            d_multi_re, d_multi_im,
            d_multi2_re, d_multi2_im,
            d_leaf_idx, d_src_id_offsets, d_src_ids,
            d_node_centers, L, n_leaves);
    }
    if (do_profile) { cudaEventRecord(ev_p2m, stream_fmm); }

    // M2M batch-2 (bottom-up) — stream ordering guarantees sequential execution
    for (int level = tree.max_level - 1; level >= 1; level--) {
        if (level < (int)m2m_level_info.size() && m2m_level_info[level].count > 0) {
            int off = m2m_level_info[level].offset;
            int cnt = m2m_level_info[level].count;
            m2m_batch2_kernel<<<cnt, block_L, 0, stream_fmm>>>(
                d_m2m_parent, d_m2m_child,
                d_m2m_shift_re, d_m2m_shift_im,
                d_multi_re, d_multi_im,
                d_multi2_re, d_multi2_im,
                L, cnt, off);
        }
    }

    if (do_profile) { cudaEventRecord(ev_m2m, stream_fmm); }

    // M2L batch-2 (optimized CSR kernel)
    {
        int smem_size = L * 2 * sizeof(fmm_real);
        for (int level = 1; level <= tree.max_level; level++) {
            if (level < (int)m2l_csr_level_info.size() && m2l_csr_level_info[level].n_targets > 0) {
                int n_tgts = m2l_csr_level_info[level].n_targets;
                int off_off = m2l_csr_level_info[level].offsets_start;
                int nodes_off = m2l_csr_level_info[level].nodes_start;
                int pair_off = m2l_csr_level_info[level].pair_offset;
                m2l_target_batch2_kernel<<<n_tgts, block_L, smem_size, stream_fmm>>>(
                    d_m2l_csr_offsets + off_off,
                    d_m2l_csr_tgt_nodes + nodes_off,
                    d_m2l_csr_src,
                    d_m2l_csr_tidx,
                    d_transfer_ri,
                    d_multi_re, d_multi_im,
                    d_multi2_re, d_multi2_im,
                    d_local_re, d_local_im,
                    d_local2_re, d_local2_im,
                    L, n_tgts, 0, pair_off);
            }
        }
    }
    if (do_profile) { cudaEventRecord(ev_m2l, stream_fmm); }

    // L2L batch-2 (top-down)
    for (int level = 2; level <= tree.max_level; level++) {
        if (level < (int)l2l_level_info.size() && l2l_level_info[level].count > 0) {
            int off = l2l_level_info[level].offset;
            int cnt = l2l_level_info[level].count;
            l2l_batch2_kernel<<<cnt, block_L, 0, stream_fmm>>>(
                d_l2l_parent, d_l2l_child,
                d_l2l_shift_re, d_l2l_shift_im,
                d_local_re, d_local_im,
                d_local2_re, d_local2_im,
                d_local_re, d_local_im,
                d_local2_re, d_local2_im,
                L, cnt, off);
        }
    }

    if (do_profile) { cudaEventRecord(ev_l2l, stream_fmm); }

    cdouble ik_val = cdouble(0, 1) * k;
    cdouble prefactor = ik_val / (16.0 * M_PI * M_PI);
    int merge_grid = (Nt + 255) / 256;

    if (!need_grad) {
        // L2P batch-2 (potential) on stream_fmm
        cudaMemsetAsync(d_result_re, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_result_im, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_result2_re, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_result2_im, 0, Nt * sizeof(double), stream_fmm);
        if (n_leaves > 0) {
            l2p_batch2_kernel<<<n_leaves, 256, 0, stream_fmm>>>(
                d_tgt_pts, d_dirs, d_weights,
                d_local_re, d_local_im,
                d_local2_re, d_local2_im,
                k_re_f, k_im_f, prefactor.real(), prefactor.imag(),
                d_result_re, d_result_im,
                d_result2_re, d_result2_im,
                d_leaf_idx, d_tgt_id_offsets, d_tgt_ids,
                d_node_centers, L, n_leaves);
        }

        if (do_profile) { cudaEventRecord(ev_l2p, stream_fmm); }

        // Wait for both streams
        CUDA_CHECK(cudaStreamSynchronize(stream_fmm));
        CUDA_CHECK(cudaStreamSynchronize(stream_p2p));
        if (do_profile) { cudaEventRecord(ev_merge); }

        // Merge P2P results
        vector_add_kernel<<<merge_grid, 256>>>(d_result_re, d_p2p_pot_re, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_result_im, d_p2p_pot_im, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_result2_re, d_p2p_pot2_re, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_result2_im, d_p2p_pot2_im, Nt);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (do_profile) { cudaEventRecord(ev_end); cudaEventSynchronize(ev_end); }
    } else {
        // L2P batch-2 (potential) on stream_fmm
        cudaMemsetAsync(d_result_re, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_result_im, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_result2_re, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_result2_im, 0, Nt * sizeof(double), stream_fmm);
        if (n_leaves > 0) {
            l2p_batch2_kernel<<<n_leaves, 256, 0, stream_fmm>>>(
                d_tgt_pts, d_dirs, d_weights,
                d_local_re, d_local_im,
                d_local2_re, d_local2_im,
                k_re_f, k_im_f, prefactor.real(), prefactor.imag(),
                d_result_re, d_result_im,
                d_result2_re, d_result2_im,
                d_leaf_idx, d_tgt_id_offsets, d_tgt_ids,
                d_node_centers, L, n_leaves);
        }

        // L2P batch-2 (gradient) on stream_fmm
        double *d_gx1_re = d_grad_re, *d_gx1_im = d_grad_im;
        double *d_gy1_re = d_gy_re_cached, *d_gy1_im = d_gy_im_cached;
        double *d_gz1_re = d_gz_re_cached, *d_gz1_im = d_gz_im_cached;
        double *d_gx2_re = d_grad2_re, *d_gx2_im = d_grad2_im;
        double *d_gy2_re = d_gy2_re_cached, *d_gy2_im = d_gy2_im_cached;
        double *d_gz2_re = d_gz2_re_cached, *d_gz2_im = d_gz2_im_cached;

        cudaMemsetAsync(d_gx1_re, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gx1_im, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gy1_re, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gy1_im, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gz1_re, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gz1_im, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gx2_re, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gx2_im, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gy2_re, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gy2_im, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gz2_re, 0, Nt * sizeof(double), stream_fmm);
        cudaMemsetAsync(d_gz2_im, 0, Nt * sizeof(double), stream_fmm);

        if (n_leaves > 0) {
            l2p_gradient_batch2_kernel<<<n_leaves, 256, 0, stream_fmm>>>(
                d_tgt_pts, d_dirs, d_weights,
                d_local_re, d_local_im,
                d_local2_re, d_local2_im,
                k_re_f, k_im_f, prefactor.real(), prefactor.imag(),
                (float)ik_val.real(), (float)ik_val.imag(),
                d_gx1_re, d_gx1_im, d_gy1_re, d_gy1_im, d_gz1_re, d_gz1_im,
                d_gx2_re, d_gx2_im, d_gy2_re, d_gy2_im, d_gz2_re, d_gz2_im,
                d_leaf_idx, d_tgt_id_offsets, d_tgt_ids,
                d_node_centers, L, n_leaves);
        }

        if (do_profile) { cudaEventRecord(ev_l2p, stream_fmm); }

        // Wait for both streams
        CUDA_CHECK(cudaStreamSynchronize(stream_fmm));
        CUDA_CHECK(cudaStreamSynchronize(stream_p2p));
        if (do_profile) { cudaEventRecord(ev_merge); }

        // Merge P2P results: potential
        vector_add_kernel<<<merge_grid, 256>>>(d_result_re, d_p2p_pot_re, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_result_im, d_p2p_pot_im, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_result2_re, d_p2p_pot2_re, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_result2_im, d_p2p_pot2_im, Nt);
        // Merge P2P results: gradient vec 1
        vector_add_kernel<<<merge_grid, 256>>>(d_gx1_re, d_p2p_gx_re, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gx1_im, d_p2p_gx_im, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gy1_re, d_p2p_gy_re, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gy1_im, d_p2p_gy_im, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gz1_re, d_p2p_gz_re, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gz1_im, d_p2p_gz_im, Nt);
        // Merge P2P results: gradient vec 2
        vector_add_kernel<<<merge_grid, 256>>>(d_gx2_re, d_p2p_gx2_re, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gx2_im, d_p2p_gx2_im, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gy2_re, d_p2p_gy2_re, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gy2_im, d_p2p_gy2_im, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gz2_re, d_p2p_gz2_re, Nt);
        vector_add_kernel<<<merge_grid, 256>>>(d_gz2_im, d_p2p_gz2_im, Nt);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Repack gradients for both vectors (reuse pre-allocated temp buffers)
        {
            int repack_block = 256;
            int repack_grid2 = (Nt + repack_block - 1) / repack_block;

            // Vec 1: repack using pre-allocated d_gx_re_tmp/d_gx_im_tmp
            CUDA_CHECK(cudaMemcpy(this->d_gx_re_tmp, d_gx1_re, Nt * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(this->d_gx_im_tmp, d_gx1_im, Nt * sizeof(double), cudaMemcpyDeviceToDevice));
            repack_gradient_kernel<<<repack_grid2, repack_block>>>(
                this->d_gx_re_tmp, this->d_gx_im_tmp, d_gy1_re, d_gy1_im, d_gz1_re, d_gz1_im,
                d_grad_re, d_grad_im, Nt);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Vec 2: reuse same temp buffers
            CUDA_CHECK(cudaMemcpy(this->d_gx_re_tmp, d_gx2_re, Nt * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(this->d_gx_im_tmp, d_gx2_im, Nt * sizeof(double), cudaMemcpyDeviceToDevice));
            repack_gradient_kernel<<<repack_grid2, repack_block>>>(
                this->d_gx_re_tmp, this->d_gx_im_tmp, d_gy2_re, d_gy2_im, d_gz2_re, d_gz2_im,
                d_grad2_re, d_grad2_im, Nt);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        if (do_profile) { cudaEventRecord(ev_end); cudaEventSynchronize(ev_end); }
    }

    // Print profiling results
    if (do_profile) {
        float t_upload, t_p2p_launch, t_p2m, t_m2m, t_m2l, t_l2l, t_l2p, t_merge, t_end;
        cudaEventElapsedTime(&t_upload, ev_start, ev_upload);
        cudaEventElapsedTime(&t_p2p_launch, ev_upload, ev_p2p_launch);
        cudaEventElapsedTime(&t_p2m, ev_p2p_launch, ev_p2m);
        cudaEventElapsedTime(&t_m2m, ev_p2m, ev_m2m);
        cudaEventElapsedTime(&t_m2l, ev_m2m, ev_m2l);
        cudaEventElapsedTime(&t_l2l, ev_m2l, ev_l2l);
        cudaEventElapsedTime(&t_l2p, ev_l2l, ev_l2p);
        cudaEventElapsedTime(&t_merge, ev_l2p, ev_merge);
        cudaEventElapsedTime(&t_end, ev_merge, ev_end);
        float total;
        cudaEventElapsedTime(&total, ev_start, ev_end);
        // Also measure total P2P time (from upload end to merge start includes P2P running concurrently)
        float t_p2p_total;
        cudaEventElapsedTime(&t_p2p_total, ev_upload, ev_merge);
        printf("  [PROFILE run_tree_batch2 #%d] total=%.1fms\n", profile_calls, total);
        printf("    upload=%.1f  P2P_launch=%.1f  P2M=%.1f  M2M=%.1f  M2L=%.1f  L2L=%.1f  L2P=%.1f  sync+merge=%.1f+%.1f\n",
               t_upload, t_p2p_launch, t_p2m, t_m2m, t_m2l, t_l2l, t_l2p, t_merge, t_end);
        printf("    FMM_tree(P2M+M2M+M2L+L2L+L2P)=%.1f  P2P_concurrent=%.1f\n",
               t_p2m + t_m2m + t_m2l + t_l2l + t_l2p, t_p2p_total);

        cudaEventDestroy(ev_start); cudaEventDestroy(ev_upload);
        cudaEventDestroy(ev_p2p_launch); cudaEventDestroy(ev_p2m);
        cudaEventDestroy(ev_m2m); cudaEventDestroy(ev_m2l);
        cudaEventDestroy(ev_l2l); cudaEventDestroy(ev_l2p);
        cudaEventDestroy(ev_merge); cudaEventDestroy(ev_end);
    }
}

void HelmholtzFMM::evaluate(const cdouble* charges, cdouble* result)
{
    for (int i = 0; i < Ns; i++) { h_q_re_buf[i] = charges[i].real(); h_q_im_buf[i] = charges[i].imag(); }

    run_tree(h_q_re_buf.data(), h_q_im_buf.data(), false);

    // Download results
    CUDA_CHECK(cudaMemcpy(h_res_re_buf.data(), d_result_re, Nt * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_im_buf.data(), d_result_im, Nt * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < Nt; i++)
        result[i] = cdouble(h_res_re_buf[i], h_res_im_buf[i]);
}

void HelmholtzFMM::evaluate_gradient(const cdouble* charges, cdouble* grad_result)
{
    for (int i = 0; i < Ns; i++) { h_q_re_buf[i] = charges[i].real(); h_q_im_buf[i] = charges[i].imag(); }

    run_tree(h_q_re_buf.data(), h_q_im_buf.data(), true);

    // Download interleaved gradient
    CUDA_CHECK(cudaMemcpy(h_res_re_buf.data(), d_grad_re, Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_im_buf.data(), d_grad_im, Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < Nt * 3; i++)
        grad_result[i] = cdouble(h_res_re_buf[i], h_res_im_buf[i]);
}

// GPU-resident evaluate: charges already on device, results stay on device
// Avoids all host<->device transfers for charges and results
void HelmholtzFMM::evaluate_gpu(const double* d_q_re, const double* d_q_im,
                                 double* d_res_re, double* d_res_im)
{
    int n_leaves = (int)leaf_info.size();

    // Copy charges to internal buffer (device-to-device)
    CUDA_CHECK(cudaMemcpy(d_charges_re, d_q_re, Ns * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_charges_im, d_q_im, Ns * sizeof(double), cudaMemcpyDeviceToDevice));

    float k_re_f = (float)k.real(), k_im_f = (float)k.imag();
    int block_L = std::min(L, 256);

    // Clear multipole/local
    CUDA_CHECK(cudaMemset(d_multi_re, 0, n_nodes * L * sizeof(fmm_real)));
    CUDA_CHECK(cudaMemset(d_multi_im, 0, n_nodes * L * sizeof(fmm_real)));
    CUDA_CHECK(cudaMemset(d_local_re, 0, n_nodes * L * sizeof(fmm_real)));
    CUDA_CHECK(cudaMemset(d_local_im, 0, n_nodes * L * sizeof(fmm_real)));

    // P2P on separate stream (potential only)
    cudaMemsetAsync(d_p2p_pot_re, 0, Nt * sizeof(double), stream_p2p);
    cudaMemsetAsync(d_p2p_pot_im, 0, Nt * sizeof(double), stream_p2p);
    launch_p2p_potential(d_tgt_pts, d_src_pts,
        d_charges_re, d_charges_im,
        d_p2p_offsets, d_p2p_indices,
        k.real(), k.imag(),
        d_p2p_pot_re, d_p2p_pot_im, Nt, stream_p2p);

    // P2M
    if (n_leaves > 0) {
        p2m_kernel<<<n_leaves, block_L, 0, stream_fmm>>>(
            d_src_pts, d_charges_re, d_charges_im,
            d_dirs_cached, k_re_f, k_im_f,
            d_multi_re, d_multi_im,
            d_leaf_idx_cached, d_src_id_offsets_cached, d_src_ids_cached,
            d_node_centers_cached, L, n_leaves);
    }

    // M2M (bottom-up)
    for (int level = tree.max_level - 1; level >= 1; level--) {
        if (level < (int)m2m_level_info.size() && m2m_level_info[level].count > 0) {
            int off = m2m_level_info[level].offset;
            int cnt = m2m_level_info[level].count;
            m2m_kernel<<<cnt, block_L, 0, stream_fmm>>>(d_m2m_parent, d_m2m_child,
                d_m2m_shift_re, d_m2m_shift_im, d_multi_re, d_multi_im, L, cnt, off);
        }
    }

    // M2L (CSR)
    {
        int smem_size = L * 2 * sizeof(fmm_real);
        for (int level = 1; level <= tree.max_level; level++) {
            if (level < (int)m2l_csr_level_info.size() && m2l_csr_level_info[level].n_targets > 0) {
                int n_tgts = m2l_csr_level_info[level].n_targets;
                int pair_off = m2l_csr_level_info[level].pair_offset;
                m2l_target_kernel<<<n_tgts, block_L, smem_size, stream_fmm>>>(
                    d_m2l_csr_offsets + m2l_csr_level_info[level].offsets_start,
                    d_m2l_csr_tgt_nodes + m2l_csr_level_info[level].nodes_start,
                    d_m2l_csr_src, d_m2l_csr_tidx,
                    d_transfer_ri,
                    d_multi_re, d_multi_im, d_local_re, d_local_im,
                    L, n_tgts, 0, pair_off);
            }
        }
    }

    // L2L (top-down)
    for (int level = 2; level <= tree.max_level; level++) {
        if (level < (int)l2l_level_info.size() && l2l_level_info[level].count > 0) {
            int off = l2l_level_info[level].offset;
            int cnt = l2l_level_info[level].count;
            l2l_kernel<<<cnt, block_L, 0, stream_fmm>>>(d_l2l_parent, d_l2l_child,
                d_l2l_shift_re, d_l2l_shift_im,
                d_local_re, d_local_im, d_local_re, d_local_im, L, cnt, off);
        }
    }

    // L2P (potential only)
    cdouble ik_val = cdouble(0, 1) * k;
    cdouble prefactor = ik_val / (16.0 * M_PI * M_PI);
    CUDA_CHECK(cudaMemset(d_result_re, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result_im, 0, Nt * sizeof(double)));
    if (n_leaves > 0) {
        l2p_kernel<<<n_leaves, 256, 0, stream_fmm>>>(
            d_tgt_pts, d_dirs_cached, d_weights_cached,
            d_local_re, d_local_im,
            k_re_f, k_im_f, prefactor.real(), prefactor.imag(),
            d_result_re, d_result_im,
            d_leaf_idx_cached, d_tgt_id_offsets_cached, d_tgt_ids_cached,
            d_node_centers_cached, L, n_leaves);
    }

    // Wait for both streams
    cudaStreamSynchronize(stream_fmm);
    cudaStreamSynchronize(stream_p2p);

    // Merge FMM + P2P results directly into output
    int block = 256, grid = (Nt + block - 1) / block;
    vector_add_kernel<<<grid, block>>>(d_result_re, d_p2p_pot_re, Nt);
    vector_add_kernel<<<grid, block>>>(d_result_im, d_p2p_pot_im, Nt);

    // Copy to output
    CUDA_CHECK(cudaMemcpy(d_res_re, d_result_re, Nt * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_res_im, d_result_im, Nt * sizeof(double), cudaMemcpyDeviceToDevice));
}

void HelmholtzFMM::evaluate_pot_grad_gpu(const double* d_q_re, const double* d_q_im,
                                          double* d_pot_re, double* d_pot_im,
                                          double* d_grad_re_out, double* d_grad_im_out)
{
    // Full GPU-resident pot+grad: no H2D/D2H copies, everything stays on device
    int n_leaves = (int)leaf_info.size();

    // Copy input charges to FMM internal buffers (D2D)
    CUDA_CHECK(cudaMemcpy(d_charges_re, d_q_re, Ns * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_charges_im, d_q_im, Ns * sizeof(double), cudaMemcpyDeviceToDevice));

    // Cached GPU array aliases
    double *d_node_centers = d_node_centers_cached;
    fmm_real *d_dirs = d_dirs_cached;
    fmm_real *d_weights = d_weights_cached;
    int *d_leaf_idx = d_leaf_idx_cached;
    int *d_src_id_offsets = d_src_id_offsets_cached;
    int *d_src_ids = d_src_ids_cached;
    int *d_tgt_id_offsets = d_tgt_id_offsets_cached;
    int *d_tgt_ids = d_tgt_ids_cached;

    float k_re_f = (float)k.real(), k_im_f = (float)k.imag();
    int block_L = std::min(L, 256);

    // === P2P pot+grad on stream_p2p (concurrent with FMM tree) ===
    cudaMemsetAsync(d_p2p_gx_re, 0, Nt * sizeof(double), stream_p2p);
    cudaMemsetAsync(d_p2p_gx_im, 0, Nt * sizeof(double), stream_p2p);
    cudaMemsetAsync(d_p2p_gy_re, 0, Nt * sizeof(double), stream_p2p);
    cudaMemsetAsync(d_p2p_gy_im, 0, Nt * sizeof(double), stream_p2p);
    cudaMemsetAsync(d_p2p_gz_re, 0, Nt * sizeof(double), stream_p2p);
    cudaMemsetAsync(d_p2p_gz_im, 0, Nt * sizeof(double), stream_p2p);
    launch_p2p_gradient(d_tgt_pts, d_src_pts,
        d_charges_re, d_charges_im,
        d_p2p_offsets, d_p2p_indices,
        k.real(), k.imag(),
        d_p2p_gx_re, d_p2p_gx_im,
        d_p2p_gy_re, d_p2p_gy_im,
        d_p2p_gz_re, d_p2p_gz_im, Nt, stream_p2p);

    // === FMM tree on stream_fmm ===
    cudaMemsetAsync(d_multi_re, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);
    cudaMemsetAsync(d_multi_im, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);
    cudaMemsetAsync(d_local_re, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);
    cudaMemsetAsync(d_local_im, 0, n_nodes * L * sizeof(fmm_real), stream_fmm);

    // P2M
    if (n_leaves > 0) {
        p2m_kernel<<<n_leaves, block_L, 0, stream_fmm>>>(
            d_src_pts, d_charges_re, d_charges_im,
            d_dirs, k_re_f, k_im_f,
            d_multi_re, d_multi_im,
            d_leaf_idx, d_src_id_offsets, d_src_ids,
            d_node_centers, L, n_leaves);
    }

    // M2M (bottom-up)
    for (int level = tree.max_level - 1; level >= 1; level--) {
        if (level < (int)m2m_level_info.size() && m2m_level_info[level].count > 0) {
            int off = m2m_level_info[level].offset;
            int cnt = m2m_level_info[level].count;
            m2m_kernel<<<cnt, block_L, 0, stream_fmm>>>(d_m2m_parent, d_m2m_child,
                d_m2m_shift_re, d_m2m_shift_im, d_multi_re, d_multi_im, L, cnt, off);
        }
    }

    // M2L (optimized CSR)
    {
        int smem_size = L * 2 * sizeof(fmm_real);
        for (int level = 1; level <= tree.max_level; level++) {
            if (level < (int)m2l_csr_level_info.size() && m2l_csr_level_info[level].n_targets > 0) {
                int n_tgts = m2l_csr_level_info[level].n_targets;
                int off_off = m2l_csr_level_info[level].offsets_start;
                int nodes_off = m2l_csr_level_info[level].nodes_start;
                int pair_off = m2l_csr_level_info[level].pair_offset;
                m2l_target_kernel<<<n_tgts, block_L, smem_size, stream_fmm>>>(
                    d_m2l_csr_offsets + off_off,
                    d_m2l_csr_tgt_nodes + nodes_off,
                    d_m2l_csr_src, d_m2l_csr_tidx,
                    d_transfer_ri,
                    d_multi_re, d_multi_im,
                    d_local_re, d_local_im,
                    L, n_tgts, 0, pair_off);
            }
        }
    }

    // L2L (top-down)
    for (int level = 2; level <= tree.max_level; level++) {
        if (level < (int)l2l_level_info.size() && l2l_level_info[level].count > 0) {
            int off = l2l_level_info[level].offset;
            int cnt = l2l_level_info[level].count;
            l2l_kernel<<<cnt, block_L, 0, stream_fmm>>>(d_l2l_parent, d_l2l_child,
                d_l2l_shift_re, d_l2l_shift_im,
                d_local_re, d_local_im, d_local_re, d_local_im, L, cnt, off);
        }
    }

    cdouble ik_val = cdouble(0, 1) * k;
    cdouble prefactor = ik_val / (16.0 * M_PI * M_PI);
    int merge_grid = (Nt + 255) / 256;

    // L2P potential
    cudaMemsetAsync(d_result_re, 0, Nt * sizeof(double), stream_fmm);
    cudaMemsetAsync(d_result_im, 0, Nt * sizeof(double), stream_fmm);
    if (n_leaves > 0) {
        l2p_kernel<<<n_leaves, 256, 0, stream_fmm>>>(
            d_tgt_pts, d_dirs, d_weights,
            d_local_re, d_local_im,
            k_re_f, k_im_f, prefactor.real(), prefactor.imag(),
            d_result_re, d_result_im,
            d_leaf_idx, d_tgt_id_offsets, d_tgt_ids,
            d_node_centers, L, n_leaves);
    }

    // L2P gradient
    double *d_gx_re = d_grad_re, *d_gx_im = d_grad_im;
    double *d_gy_re = d_gy_re_cached, *d_gy_im = d_gy_im_cached;
    double *d_gz_re = d_gz_re_cached, *d_gz_im = d_gz_im_cached;
    cudaMemsetAsync(d_gx_re, 0, Nt * sizeof(double), stream_fmm);
    cudaMemsetAsync(d_gx_im, 0, Nt * sizeof(double), stream_fmm);
    cudaMemsetAsync(d_gy_re, 0, Nt * sizeof(double), stream_fmm);
    cudaMemsetAsync(d_gy_im, 0, Nt * sizeof(double), stream_fmm);
    cudaMemsetAsync(d_gz_re, 0, Nt * sizeof(double), stream_fmm);
    cudaMemsetAsync(d_gz_im, 0, Nt * sizeof(double), stream_fmm);

    if (n_leaves > 0) {
        l2p_gradient_kernel<<<n_leaves, 256, 0, stream_fmm>>>(
            d_tgt_pts, d_dirs, d_weights,
            d_local_re, d_local_im,
            k_re_f, k_im_f, prefactor.real(), prefactor.imag(),
            (float)ik_val.real(), (float)ik_val.imag(),
            d_gx_re, d_gx_im, d_gy_re, d_gy_im, d_gz_re, d_gz_im,
            d_leaf_idx, d_tgt_id_offsets, d_tgt_ids,
            d_node_centers, L, n_leaves);
    }

    // Wait for both streams
    CUDA_CHECK(cudaStreamSynchronize(stream_fmm));
    CUDA_CHECK(cudaStreamSynchronize(stream_p2p));

    // Merge P2P gradient results
    vector_add_kernel<<<merge_grid, 256>>>(d_gx_re, d_p2p_gx_re, Nt);
    vector_add_kernel<<<merge_grid, 256>>>(d_gx_im, d_p2p_gx_im, Nt);
    vector_add_kernel<<<merge_grid, 256>>>(d_gy_re, d_p2p_gy_re, Nt);
    vector_add_kernel<<<merge_grid, 256>>>(d_gy_im, d_p2p_gy_im, Nt);
    vector_add_kernel<<<merge_grid, 256>>>(d_gz_re, d_p2p_gz_re, Nt);
    vector_add_kernel<<<merge_grid, 256>>>(d_gz_im, d_p2p_gz_im, Nt);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy potential to output (D2D)
    CUDA_CHECK(cudaMemcpy(d_pot_re, d_result_re, Nt * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_pot_im, d_result_im, Nt * sizeof(double), cudaMemcpyDeviceToDevice));

    // Repack gradient [gx,gy,gz] -> interleaved [x0,y0,z0,...] into output
    CUDA_CHECK(cudaMemcpy(d_gx_re_tmp, d_gx_re, Nt * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_gx_im_tmp, d_gx_im, Nt * sizeof(double), cudaMemcpyDeviceToDevice));
    int repack_block = 256;
    int repack_grid = (Nt + repack_block - 1) / repack_block;
    repack_gradient_kernel<<<repack_grid, repack_block>>>(
        d_gx_re_tmp, d_gx_im_tmp, d_gy_re, d_gy_im, d_gz_re, d_gz_im,
        d_grad_re_out, d_grad_im_out, Nt);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void HelmholtzFMM::evaluate_pot_grad(const cdouble* charges,
                                      cdouble* pot_result,
                                      cdouble* grad_result)
{
    int n_leaves = (int)leaf_info.size();

    // Split charges into re/im (using pre-allocated buffers)
    for (int i = 0; i < Ns; i++) {
        h_q_re_buf[i] = charges[i].real();
        h_q_im_buf[i] = charges[i].imag();
    }

    // Upload charges (double)
    CUDA_CHECK(cudaMemcpy(d_charges_re, h_q_re_buf.data(), Ns * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_charges_im, h_q_im_buf.data(), Ns * sizeof(double), cudaMemcpyHostToDevice));

    // Clear multipole/local arrays (float32)
    CUDA_CHECK(cudaMemset(d_multi_re, 0, n_nodes * L * sizeof(fmm_real)));
    CUDA_CHECK(cudaMemset(d_multi_im, 0, n_nodes * L * sizeof(fmm_real)));
    CUDA_CHECK(cudaMemset(d_local_re, 0, n_nodes * L * sizeof(fmm_real)));
    CUDA_CHECK(cudaMemset(d_local_im, 0, n_nodes * L * sizeof(fmm_real)));

    // Cached GPU array aliases
    double *d_node_centers = d_node_centers_cached;
    fmm_real *d_dirs = d_dirs_cached;
    fmm_real *d_weights = d_weights_cached;
    int *d_leaf_idx = d_leaf_idx_cached;
    int *d_src_id_offsets = d_src_id_offsets_cached;
    int *d_src_ids = d_src_ids_cached;
    int *d_tgt_id_offsets = d_tgt_id_offsets_cached;
    int *d_tgt_ids = d_tgt_ids_cached;

    float k_re_f = (float)k.real(), k_im_f = (float)k.imag();
    int block_L = std::min(L, 256);

    // === P2M ===
    if (n_leaves > 0) {
        p2m_kernel<<<n_leaves, block_L>>>(
            d_src_pts, d_charges_re, d_charges_im,
            d_dirs, k_re_f, k_im_f,
            d_multi_re, d_multi_im,
            d_leaf_idx, d_src_id_offsets, d_src_ids,
            d_node_centers, L, n_leaves);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // === M2M (bottom-up) ===
    for (int level = tree.max_level - 1; level >= 1; level--) {
        if (level < (int)m2m_level_info.size() && m2m_level_info[level].count > 0) {
            int off = m2m_level_info[level].offset;
            int cnt = m2m_level_info[level].count;
            m2m_kernel<<<cnt, block_L>>>(d_m2m_parent, d_m2m_child,
                d_m2m_shift_re, d_m2m_shift_im, d_multi_re, d_multi_im, L, cnt, off);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // === M2L (optimized CSR) ===
    {
        int smem_size = L * 2 * sizeof(fmm_real);
        for (int level = 1; level <= tree.max_level; level++) {
            if (level < (int)m2l_csr_level_info.size() && m2l_csr_level_info[level].n_targets > 0) {
                int n_tgts = m2l_csr_level_info[level].n_targets;
                int off_off = m2l_csr_level_info[level].offsets_start;
                int nodes_off = m2l_csr_level_info[level].nodes_start;
                int pair_off = m2l_csr_level_info[level].pair_offset;
                m2l_target_kernel<<<n_tgts, block_L, smem_size>>>(
                    d_m2l_csr_offsets + off_off,
                    d_m2l_csr_tgt_nodes + nodes_off,
                    d_m2l_csr_src,
                    d_m2l_csr_tidx,
                    d_transfer_ri,
                    d_multi_re, d_multi_im,
                    d_local_re, d_local_im,
                    L, n_tgts, 0, pair_off);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
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
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    cdouble ik_val = cdouble(0, 1) * k;
    cdouble prefactor = ik_val / (16.0 * M_PI * M_PI);

    // === L2P potential ===
    CUDA_CHECK(cudaMemset(d_result_re, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result_im, 0, Nt * sizeof(double)));
    if (n_leaves > 0) {
        l2p_kernel<<<n_leaves, 256>>>(
            d_tgt_pts, d_dirs, d_weights,
            d_local_re, d_local_im,
            k_re_f, k_im_f, prefactor.real(), prefactor.imag(),
            d_result_re, d_result_im,
            d_leaf_idx, d_tgt_id_offsets, d_tgt_ids,
            d_node_centers, L, n_leaves);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // === L2P gradient ===
    double *d_gx_re = d_grad_re, *d_gx_im = d_grad_im;
    double *d_gy_re = d_gy_re_cached, *d_gy_im = d_gy_im_cached;
    double *d_gz_re = d_gz_re_cached, *d_gz_im = d_gz_im_cached;
    CUDA_CHECK(cudaMemset(d_gx_re, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gx_im, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy_re, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gy_im, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz_re, 0, Nt * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_gz_im, 0, Nt * sizeof(double)));

    if (n_leaves > 0) {
        l2p_gradient_kernel<<<n_leaves, 256>>>(
            d_tgt_pts, d_dirs, d_weights,
            d_local_re, d_local_im,
            k_re_f, k_im_f, prefactor.real(), prefactor.imag(),
            (float)ik_val.real(), (float)ik_val.imag(),
            d_gx_re, d_gx_im, d_gy_re, d_gy_im, d_gz_re, d_gz_im,
            d_leaf_idx, d_tgt_id_offsets, d_tgt_ids,
            d_node_centers, L, n_leaves);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // === P2P combined potential + gradient (stays double) ===
    launch_p2p_pot_grad(Nt,
        d_tgt_pts, d_src_pts,
        d_charges_re, d_charges_im,
        d_p2p_offsets, d_p2p_indices,
        k.real(), k.imag(),
        d_result_re, d_result_im,
        d_gx_re, d_gx_im, d_gy_re, d_gy_im, d_gz_re, d_gz_im);
    CUDA_CHECK(cudaDeviceSynchronize());

    // === Download potential ===
    {
        CUDA_CHECK(cudaMemcpy(h_res_re_buf.data(), d_result_re, Nt * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_res_im_buf.data(), d_result_im, Nt * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < Nt; i++)
            pot_result[i] = cdouble(h_res_re_buf[i], h_res_im_buf[i]);
    }

    // === Repack gradient (using pre-allocated temp buffers) ===
    {
        CUDA_CHECK(cudaMemcpy(d_gx_re_tmp, d_gx_re, Nt * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_gx_im_tmp, d_gx_im, Nt * sizeof(double), cudaMemcpyDeviceToDevice));

        int repack_block = 256;
        int repack_grid = (Nt + repack_block - 1) / repack_block;
        repack_gradient_kernel<<<repack_grid, repack_block>>>(
            d_gx_re_tmp, d_gx_im_tmp, d_gy_re, d_gy_im, d_gz_re, d_gz_im,
            d_grad_re, d_grad_im, Nt);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_res_re_buf.data(), d_grad_re, Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_res_im_buf.data(), d_grad_im, Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < Nt * 3; i++)
            grad_result[i] = cdouble(h_res_re_buf[i], h_res_im_buf[i]);
    }
}

void HelmholtzFMM::evaluate_batch2(
    const cdouble* charges1, const cdouble* charges2,
    cdouble* result1, cdouble* result2)
{
    for (int i = 0; i < Ns; i++) {
        h_q_re_buf[i] = charges1[i].real(); h_q_im_buf[i] = charges1[i].imag();
        h_q2_re_buf[i] = charges2[i].real(); h_q2_im_buf[i] = charges2[i].imag();
    }

    run_tree_batch2(h_q_re_buf.data(), h_q_im_buf.data(), h_q2_re_buf.data(), h_q2_im_buf.data(), false);

    CUDA_CHECK(cudaMemcpy(h_res_re_buf.data(), d_result_re, Nt * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_im_buf.data(), d_result_im, Nt * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res2_re_buf.data(), d_result2_re, Nt * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res2_im_buf.data(), d_result2_im, Nt * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < Nt; i++) {
        result1[i] = cdouble(h_res_re_buf[i], h_res_im_buf[i]);
        result2[i] = cdouble(h_res2_re_buf[i], h_res2_im_buf[i]);
    }
}

void HelmholtzFMM::evaluate_pot_grad_batch2(
    const cdouble* charges1, const cdouble* charges2,
    cdouble* pot1, cdouble* grad1,
    cdouble* pot2, cdouble* grad2)
{
    for (int i = 0; i < Ns; i++) {
        h_q_re_buf[i] = charges1[i].real(); h_q_im_buf[i] = charges1[i].imag();
        h_q2_re_buf[i] = charges2[i].real(); h_q2_im_buf[i] = charges2[i].imag();
    }

    run_tree_batch2(h_q_re_buf.data(), h_q_im_buf.data(), h_q2_re_buf.data(), h_q2_im_buf.data(), true);

    // Download potential for both vectors
    {
        CUDA_CHECK(cudaMemcpy(h_res_re_buf.data(), d_result_re, Nt * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_res_im_buf.data(), d_result_im, Nt * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < Nt; i++)
            pot1[i] = cdouble(h_res_re_buf[i], h_res_im_buf[i]);

        CUDA_CHECK(cudaMemcpy(h_res_re_buf.data(), d_result2_re, Nt * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_res_im_buf.data(), d_result2_im, Nt * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < Nt; i++)
            pot2[i] = cdouble(h_res_re_buf[i], h_res_im_buf[i]);
    }

    // Download gradient for both vectors
    {
        CUDA_CHECK(cudaMemcpy(h_res_re_buf.data(), d_grad_re, Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_res_im_buf.data(), d_grad_im, Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < Nt * 3; i++)
            grad1[i] = cdouble(h_res_re_buf[i], h_res_im_buf[i]);

        CUDA_CHECK(cudaMemcpy(h_res_re_buf.data(), d_grad2_re, Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_res_im_buf.data(), d_grad2_im, Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < Nt * 3; i++)
            grad2[i] = cdouble(h_res_re_buf[i], h_res_im_buf[i]);
    }
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
    if (d_m2l_csr_offsets) cudaFree(d_m2l_csr_offsets);
    if (d_m2l_csr_tgt_nodes) cudaFree(d_m2l_csr_tgt_nodes);
    if (d_m2l_csr_src) cudaFree(d_m2l_csr_src);
    if (d_m2l_csr_tidx) cudaFree(d_m2l_csr_tidx);
    if (d_transfer_ri) cudaFree(d_transfer_ri);
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
    if (d_multi2_re) cudaFree(d_multi2_re);
    if (d_multi2_im) cudaFree(d_multi2_im);
    if (d_local2_re) cudaFree(d_local2_re);
    if (d_local2_im) cudaFree(d_local2_im);
    if (d_grad2_re) cudaFree(d_grad2_re);
    if (d_grad2_im) cudaFree(d_grad2_im);
    if (d_gy2_re_cached) cudaFree(d_gy2_re_cached);
    if (d_gy2_im_cached) cudaFree(d_gy2_im_cached);
    if (d_gz2_re_cached) cudaFree(d_gz2_re_cached);
    if (d_gz2_im_cached) cudaFree(d_gz2_im_cached);
    cudaFree(d_node_centers_cached); cudaFree(d_dirs_cached); cudaFree(d_weights_cached);
    cudaFree(d_leaf_idx_cached);
    cudaFree(d_src_id_offsets_cached); cudaFree(d_src_ids_cached);
    cudaFree(d_tgt_id_offsets_cached); cudaFree(d_tgt_ids_cached);
    cudaFree(d_gy_re_cached); cudaFree(d_gy_im_cached);
    cudaFree(d_gz_re_cached); cudaFree(d_gz_im_cached);
    if (d_gx_re_tmp) cudaFree(d_gx_re_tmp);
    if (d_gx_im_tmp) cudaFree(d_gx_im_tmp);
    // P2P/FMM overlap resources
    if (stream_fmm) cudaStreamDestroy(stream_fmm);
    if (stream_p2p) cudaStreamDestroy(stream_p2p);
    cudaFree(d_p2p_pot_re); cudaFree(d_p2p_pot_im);
    cudaFree(d_p2p_pot2_re); cudaFree(d_p2p_pot2_im);
    cudaFree(d_p2p_gx_re); cudaFree(d_p2p_gx_im);
    cudaFree(d_p2p_gy_re); cudaFree(d_p2p_gy_im);
    cudaFree(d_p2p_gz_re); cudaFree(d_p2p_gz_im);
    cudaFree(d_p2p_gx2_re); cudaFree(d_p2p_gx2_im);
    cudaFree(d_p2p_gy2_re); cudaFree(d_p2p_gy2_im);
    cudaFree(d_p2p_gz2_re); cudaFree(d_p2p_gz2_im);
    initialized = false;
}
