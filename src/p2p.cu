#include "fmm.h"
#include <cuda_runtime.h>
#include <cstdio>

// ============================================================
// Leaf-to-leaf P2P near-field CUDA kernels
//
// Instead of a per-target CSR (which costs ~6 GB for large meshes),
// we store a per-leaf neighbor list (~1 MB) and iterate over
// sources in neighbor leaves at runtime.
//
// Optimizations:
//   1. Warp-per-target: 32 threads cooperatively process each target.
//   2. Float32 transcendentals: exp/sincos in float (5x faster on SM_86).
//   3. __ldg() for read-only global memory (texture cache path).
//   4. Two-level loop: outer over neighbor leaves, inner over sources.
// ============================================================

static const int WARP_SIZE = 32;
static const int WARPS_PER_BLOCK = 4;
static const int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;  // 128
static const unsigned FULL_MASK = 0xFFFFFFFFu;

// ============================================================
// Potential-only kernel (leaf-to-leaf)
// ============================================================
__global__ void p2p_potential_kernel(
    const double* __restrict__ tgt_xyz,
    const double* __restrict__ src_xyz,
    const double* __restrict__ q_re,
    const double* __restrict__ q_im,
    const int*    __restrict__ tgt_to_leaf,
    const int*    __restrict__ leaf_near_offsets,
    const int*    __restrict__ leaf_near_nbrs,
    const int*    __restrict__ src_id_offsets,
    const int*    __restrict__ src_ids,
    double k_re, double k_im,
    double* __restrict__ out_re,
    double* __restrict__ out_im,
    int Nt)
{
    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    const int tid    = blockIdx.x * WARPS_PER_BLOCK + warpId;

    if (tid >= Nt) return;

    const int t3 = tid * 3;
    const double tx = __ldg(&tgt_xyz[t3]);
    const double ty = __ldg(&tgt_xyz[t3 + 1]);
    const double tz = __ldg(&tgt_xyz[t3 + 2]);

    double acc_re = 0.0, acc_im = 0.0;
    const double inv4pi = 0.07957747154594767;
    const float k_re_f = (float)k_re;
    const float k_im_f = (float)k_im;

    const int leaf = __ldg(&tgt_to_leaf[tid]);
    const int nb_start = __ldg(&leaf_near_offsets[leaf]);
    const int nb_end   = __ldg(&leaf_near_offsets[leaf + 1]);

    for (int ni = nb_start; ni < nb_end; ni++) {
        int nb_leaf = __ldg(&leaf_near_nbrs[ni]);
        int s_start = __ldg(&src_id_offsets[nb_leaf]);
        int s_end   = __ldg(&src_id_offsets[nb_leaf + 1]);

        for (int j = s_start + lane; j < s_end; j += WARP_SIZE) {
            int sid = __ldg(&src_ids[j]);
            int sid3 = sid * 3;
            double dx = tx - __ldg(&src_xyz[sid3]);
            double dy = ty - __ldg(&src_xyz[sid3 + 1]);
            double dz = tz - __ldg(&src_xyz[sid3 + 2]);
            double R2 = dx*dx + dy*dy + dz*dz;

            if (R2 >= 1e-24) {
                double inv_R = rsqrt(R2);
                float Rf = (float)(R2 * inv_R);
                float cpf, spf;
                __sincosf(k_re_f * Rf, &spf, &cpf);
                float eR = __expf(-k_im_f * Rf);
                double Gfac = (double)eR * inv4pi * inv_R;
                double G_re = Gfac * (double)cpf;
                double G_im = Gfac * (double)spf;

                double qr = __ldg(&q_re[sid]);
                double qi = __ldg(&q_im[sid]);
                acc_re += G_re * qr - G_im * qi;
                acc_im += G_re * qi + G_im * qr;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_re += __shfl_down_sync(FULL_MASK, acc_re, offset);
        acc_im += __shfl_down_sync(FULL_MASK, acc_im, offset);
    }

    if (lane == 0) {
        out_re[tid] += acc_re;
        out_im[tid] += acc_im;
    }
}

// ============================================================
// Gradient-only kernel (leaf-to-leaf)
// ============================================================
__global__ void p2p_gradient_kernel(
    const double* __restrict__ tgt_xyz,
    const double* __restrict__ src_xyz,
    const double* __restrict__ q_re,
    const double* __restrict__ q_im,
    const int*    __restrict__ tgt_to_leaf,
    const int*    __restrict__ leaf_near_offsets,
    const int*    __restrict__ leaf_near_nbrs,
    const int*    __restrict__ src_id_offsets,
    const int*    __restrict__ src_ids,
    double k_re, double k_im,
    double* __restrict__ gx_re, double* __restrict__ gx_im,
    double* __restrict__ gy_re, double* __restrict__ gy_im,
    double* __restrict__ gz_re, double* __restrict__ gz_im,
    int Nt)
{
    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    const int tid    = blockIdx.x * WARPS_PER_BLOCK + warpId;

    if (tid >= Nt) return;

    const int t3 = tid * 3;
    const double tx = __ldg(&tgt_xyz[t3]);
    const double ty = __ldg(&tgt_xyz[t3 + 1]);
    const double tz = __ldg(&tgt_xyz[t3 + 2]);

    double ax_re = 0.0, ax_im = 0.0;
    double ay_re = 0.0, ay_im = 0.0;
    double az_re = 0.0, az_im = 0.0;

    const double inv4pi = 0.07957747154594767;
    const float k_re_f = (float)k_re;
    const float k_im_f = (float)k_im;

    const int leaf = __ldg(&tgt_to_leaf[tid]);
    const int nb_start = __ldg(&leaf_near_offsets[leaf]);
    const int nb_end   = __ldg(&leaf_near_offsets[leaf + 1]);

    for (int ni = nb_start; ni < nb_end; ni++) {
        int nb_leaf = __ldg(&leaf_near_nbrs[ni]);
        int s_start = __ldg(&src_id_offsets[nb_leaf]);
        int s_end   = __ldg(&src_id_offsets[nb_leaf + 1]);

        for (int j = s_start + lane; j < s_end; j += WARP_SIZE) {
            int sid = __ldg(&src_ids[j]);
            int sid3 = sid * 3;
            double dx = tx - __ldg(&src_xyz[sid3]);
            double dy = ty - __ldg(&src_xyz[sid3 + 1]);
            double dz = tz - __ldg(&src_xyz[sid3 + 2]);
            double R2 = dx*dx + dy*dy + dz*dz;

            if (R2 >= 1e-24) {
                double inv_R = rsqrt(R2);
                double R = R2 * inv_R;
                float Rf = (float)R;
                float cpf, spf;
                __sincosf(k_re_f * Rf, &spf, &cpf);
                float eR = __expf(-k_im_f * Rf);
                double Gfac = (double)eR * inv4pi * inv_R;
                double G_re = Gfac * (double)cpf;
                double G_im = Gfac * (double)spf;

                double fac_re = (-k_im - inv_R) * inv_R;
                double fac_im = k_re * inv_R;

                double gG_re = G_re * fac_re - G_im * fac_im;
                double gG_im = G_re * fac_im + G_im * fac_re;

                double qr = __ldg(&q_re[sid]);
                double qi = __ldg(&q_im[sid]);
                double gq_re = gG_re * qr - gG_im * qi;
                double gq_im = gG_re * qi + gG_im * qr;

                ax_re += gq_re * dx; ax_im += gq_im * dx;
                ay_re += gq_re * dy; ay_im += gq_im * dy;
                az_re += gq_re * dz; az_im += gq_im * dz;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        ax_re += __shfl_down_sync(FULL_MASK, ax_re, offset);
        ax_im += __shfl_down_sync(FULL_MASK, ax_im, offset);
        ay_re += __shfl_down_sync(FULL_MASK, ay_re, offset);
        ay_im += __shfl_down_sync(FULL_MASK, ay_im, offset);
        az_re += __shfl_down_sync(FULL_MASK, az_re, offset);
        az_im += __shfl_down_sync(FULL_MASK, az_im, offset);
    }

    if (lane == 0) {
        gx_re[tid] += ax_re; gx_im[tid] += ax_im;
        gy_re[tid] += ay_re; gy_im[tid] += ay_im;
        gz_re[tid] += az_re; gz_im[tid] += az_im;
    }
}

// ============================================================
// Combined potential + gradient kernel (leaf-to-leaf)
// ============================================================
__global__ void p2p_pot_grad_kernel(
    int Nt,
    const double* __restrict__ tgt_xyz,
    const double* __restrict__ src_xyz,
    const double* __restrict__ q_re,
    const double* __restrict__ q_im,
    const int*    __restrict__ tgt_to_leaf,
    const int*    __restrict__ leaf_near_offsets,
    const int*    __restrict__ leaf_near_nbrs,
    const int*    __restrict__ src_id_offsets,
    const int*    __restrict__ src_ids,
    double k_re, double k_im,
    double* __restrict__ pot_re,
    double* __restrict__ pot_im,
    double* __restrict__ gx_re, double* __restrict__ gx_im,
    double* __restrict__ gy_re, double* __restrict__ gy_im,
    double* __restrict__ gz_re, double* __restrict__ gz_im)
{
    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    const int tid    = blockIdx.x * WARPS_PER_BLOCK + warpId;

    if (tid >= Nt) return;

    const int t3 = tid * 3;
    const double tx = __ldg(&tgt_xyz[t3]);
    const double ty = __ldg(&tgt_xyz[t3 + 1]);
    const double tz = __ldg(&tgt_xyz[t3 + 2]);

    double p_re = 0.0, p_im = 0.0;
    double ax_re = 0.0, ax_im = 0.0;
    double ay_re = 0.0, ay_im = 0.0;
    double az_re = 0.0, az_im = 0.0;

    const double inv4pi = 0.07957747154594767;
    const float k_re_f = (float)k_re;
    const float k_im_f = (float)k_im;

    const int leaf = __ldg(&tgt_to_leaf[tid]);
    const int nb_start = __ldg(&leaf_near_offsets[leaf]);
    const int nb_end   = __ldg(&leaf_near_offsets[leaf + 1]);

    for (int ni = nb_start; ni < nb_end; ni++) {
        int nb_leaf = __ldg(&leaf_near_nbrs[ni]);
        int s_start = __ldg(&src_id_offsets[nb_leaf]);
        int s_end   = __ldg(&src_id_offsets[nb_leaf + 1]);

        for (int j = s_start + lane; j < s_end; j += WARP_SIZE) {
            int sid = __ldg(&src_ids[j]);
            int sid3 = sid * 3;
            double dx = tx - __ldg(&src_xyz[sid3]);
            double dy = ty - __ldg(&src_xyz[sid3 + 1]);
            double dz = tz - __ldg(&src_xyz[sid3 + 2]);
            double R2 = dx*dx + dy*dy + dz*dz;

            if (R2 >= 1e-24) {
                double inv_R = rsqrt(R2);
                double R = R2 * inv_R;
                float Rf = (float)R;
                float cpf, spf;
                __sincosf(k_re_f * Rf, &spf, &cpf);
                float eR = __expf(-k_im_f * Rf);
                double Gfac = (double)eR * inv4pi * inv_R;
                double G_re = Gfac * (double)cpf;
                double G_im = Gfac * (double)spf;

                double qr = __ldg(&q_re[sid]);
                double qi = __ldg(&q_im[sid]);

                p_re += G_re * qr - G_im * qi;
                p_im += G_re * qi + G_im * qr;

                double fac_re = (-k_im - inv_R) * inv_R;
                double fac_im = k_re * inv_R;

                double gG_re = G_re * fac_re - G_im * fac_im;
                double gG_im = G_re * fac_im + G_im * fac_re;

                double gq_re = gG_re * qr - gG_im * qi;
                double gq_im = gG_re * qi + gG_im * qr;

                ax_re += gq_re * dx; ax_im += gq_im * dx;
                ay_re += gq_re * dy; ay_im += gq_im * dy;
                az_re += gq_re * dz; az_im += gq_im * dz;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        p_re  += __shfl_down_sync(FULL_MASK, p_re,  offset);
        p_im  += __shfl_down_sync(FULL_MASK, p_im,  offset);
        ax_re += __shfl_down_sync(FULL_MASK, ax_re, offset);
        ax_im += __shfl_down_sync(FULL_MASK, ax_im, offset);
        ay_re += __shfl_down_sync(FULL_MASK, ay_re, offset);
        ay_im += __shfl_down_sync(FULL_MASK, ay_im, offset);
        az_re += __shfl_down_sync(FULL_MASK, az_re, offset);
        az_im += __shfl_down_sync(FULL_MASK, az_im, offset);
    }

    if (lane == 0) {
        pot_re[tid] += p_re; pot_im[tid] += p_im;
        gx_re[tid] += ax_re; gx_im[tid] += ax_im;
        gy_re[tid] += ay_re; gy_im[tid] += ay_im;
        gz_re[tid] += az_re; gz_im[tid] += az_im;
    }
}

// ============================================================
// Host wrappers
// ============================================================

void launch_p2p_potential(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_tgt_to_leaf, const int* d_leaf_near_offsets,
    const int* d_leaf_near_nbrs, const int* d_src_id_offsets,
    const int* d_src_ids,
    double k_re, double k_im,
    double* d_out_re, double* d_out_im,
    int Nt, cudaStream_t stream)
{
    int grid = (Nt + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    p2p_potential_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_tgt, d_src, d_q_re, d_q_im,
        d_tgt_to_leaf, d_leaf_near_offsets, d_leaf_near_nbrs,
        d_src_id_offsets, d_src_ids,
        k_re, k_im, d_out_re, d_out_im, Nt);
}

void launch_p2p_gradient(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_tgt_to_leaf, const int* d_leaf_near_offsets,
    const int* d_leaf_near_nbrs, const int* d_src_id_offsets,
    const int* d_src_ids,
    double k_re, double k_im,
    double* d_gx_re, double* d_gx_im,
    double* d_gy_re, double* d_gy_im,
    double* d_gz_re, double* d_gz_im,
    int Nt, cudaStream_t stream)
{
    int grid = (Nt + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    p2p_gradient_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_tgt, d_src, d_q_re, d_q_im,
        d_tgt_to_leaf, d_leaf_near_offsets, d_leaf_near_nbrs,
        d_src_id_offsets, d_src_ids,
        k_re, k_im,
        d_gx_re, d_gx_im, d_gy_re, d_gy_im, d_gz_re, d_gz_im,
        Nt);
}

void launch_p2p_pot_grad(
    int Nt,
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_tgt_to_leaf, const int* d_leaf_near_offsets,
    const int* d_leaf_near_nbrs, const int* d_src_id_offsets,
    const int* d_src_ids,
    double k_re, double k_im,
    double* d_pot_re, double* d_pot_im,
    double* d_gx_re, double* d_gx_im,
    double* d_gy_re, double* d_gy_im,
    double* d_gz_re, double* d_gz_im,
    cudaStream_t stream)
{
    int grid = (Nt + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    p2p_pot_grad_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        Nt, d_tgt, d_src, d_q_re, d_q_im,
        d_tgt_to_leaf, d_leaf_near_offsets, d_leaf_near_nbrs,
        d_src_id_offsets, d_src_ids,
        k_re, k_im,
        d_pot_re, d_pot_im,
        d_gx_re, d_gx_im, d_gy_re, d_gy_im, d_gz_re, d_gz_im);
}

// ============================================================
// Batch-2 P2P kernels: two charge vectors, single leaf traversal
// ============================================================

__global__ void p2p_potential_batch2_kernel(
    const double* __restrict__ tgt_xyz,
    const double* __restrict__ src_xyz,
    const double* __restrict__ q1_re,
    const double* __restrict__ q1_im,
    const double* __restrict__ q2_re,
    const double* __restrict__ q2_im,
    const int*    __restrict__ tgt_to_leaf,
    const int*    __restrict__ leaf_near_offsets,
    const int*    __restrict__ leaf_near_nbrs,
    const int*    __restrict__ src_id_offsets,
    const int*    __restrict__ src_ids,
    double k_re, double k_im,
    double* __restrict__ out1_re,
    double* __restrict__ out1_im,
    double* __restrict__ out2_re,
    double* __restrict__ out2_im,
    int Nt)
{
    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    const int tid    = blockIdx.x * WARPS_PER_BLOCK + warpId;

    if (tid >= Nt) return;

    const int t3 = tid * 3;
    const double tx = __ldg(&tgt_xyz[t3]);
    const double ty = __ldg(&tgt_xyz[t3 + 1]);
    const double tz = __ldg(&tgt_xyz[t3 + 2]);

    double acc1_re = 0.0, acc1_im = 0.0;
    double acc2_re = 0.0, acc2_im = 0.0;
    const double inv4pi = 0.07957747154594767;
    const float k_re_f = (float)k_re;
    const float k_im_f = (float)k_im;

    const int leaf = __ldg(&tgt_to_leaf[tid]);
    const int nb_start = __ldg(&leaf_near_offsets[leaf]);
    const int nb_end   = __ldg(&leaf_near_offsets[leaf + 1]);

    for (int ni = nb_start; ni < nb_end; ni++) {
        int nb_leaf = __ldg(&leaf_near_nbrs[ni]);
        int s_start = __ldg(&src_id_offsets[nb_leaf]);
        int s_end   = __ldg(&src_id_offsets[nb_leaf + 1]);

        for (int j = s_start + lane; j < s_end; j += WARP_SIZE) {
            int sid = __ldg(&src_ids[j]);
            int sid3 = sid * 3;
            double dx = tx - __ldg(&src_xyz[sid3]);
            double dy = ty - __ldg(&src_xyz[sid3 + 1]);
            double dz = tz - __ldg(&src_xyz[sid3 + 2]);
            double R2 = dx*dx + dy*dy + dz*dz;

            if (R2 >= 1e-24) {
                double inv_R = rsqrt(R2);
                float Rf = (float)(R2 * inv_R);
                float cpf, spf;
                __sincosf(k_re_f * Rf, &spf, &cpf);
                float eR = __expf(-k_im_f * Rf);
                double Gfac = (double)eR * inv4pi * inv_R;
                double G_re = Gfac * (double)cpf;
                double G_im = Gfac * (double)spf;

                double q1r = __ldg(&q1_re[sid]), q1i = __ldg(&q1_im[sid]);
                acc1_re += G_re * q1r - G_im * q1i;
                acc1_im += G_re * q1i + G_im * q1r;

                double q2r = __ldg(&q2_re[sid]), q2i = __ldg(&q2_im[sid]);
                acc2_re += G_re * q2r - G_im * q2i;
                acc2_im += G_re * q2i + G_im * q2r;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc1_re += __shfl_down_sync(FULL_MASK, acc1_re, offset);
        acc1_im += __shfl_down_sync(FULL_MASK, acc1_im, offset);
        acc2_re += __shfl_down_sync(FULL_MASK, acc2_re, offset);
        acc2_im += __shfl_down_sync(FULL_MASK, acc2_im, offset);
    }

    if (lane == 0) {
        out1_re[tid] += acc1_re; out1_im[tid] += acc1_im;
        out2_re[tid] += acc2_re; out2_im[tid] += acc2_im;
    }
}

void launch_p2p_potential_batch2(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const int* d_tgt_to_leaf, const int* d_leaf_near_offsets,
    const int* d_leaf_near_nbrs, const int* d_src_id_offsets,
    const int* d_src_ids,
    double k_re, double k_im,
    double* d_out1_re, double* d_out1_im,
    double* d_out2_re, double* d_out2_im, int Nt,
    cudaStream_t stream)
{
    int grid = (Nt + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    p2p_potential_batch2_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_tgt, d_src,
        d_q1_re, d_q1_im, d_q2_re, d_q2_im,
        d_tgt_to_leaf, d_leaf_near_offsets, d_leaf_near_nbrs,
        d_src_id_offsets, d_src_ids,
        k_re, k_im,
        d_out1_re, d_out1_im, d_out2_re, d_out2_im, Nt);
}

// Batch-2 P2P combined potential + gradient (leaf-to-leaf)
__global__ void p2p_pot_grad_batch2_kernel(
    int Nt,
    const double* __restrict__ tgt_xyz,
    const double* __restrict__ src_xyz,
    const double* __restrict__ q1_re,
    const double* __restrict__ q1_im,
    const double* __restrict__ q2_re,
    const double* __restrict__ q2_im,
    const int*    __restrict__ tgt_to_leaf,
    const int*    __restrict__ leaf_near_offsets,
    const int*    __restrict__ leaf_near_nbrs,
    const int*    __restrict__ src_id_offsets,
    const int*    __restrict__ src_ids,
    double k_re, double k_im,
    double* __restrict__ pot1_re, double* __restrict__ pot1_im,
    double* __restrict__ pot2_re, double* __restrict__ pot2_im,
    double* __restrict__ gx1_re, double* __restrict__ gx1_im,
    double* __restrict__ gy1_re, double* __restrict__ gy1_im,
    double* __restrict__ gz1_re, double* __restrict__ gz1_im,
    double* __restrict__ gx2_re, double* __restrict__ gx2_im,
    double* __restrict__ gy2_re, double* __restrict__ gy2_im,
    double* __restrict__ gz2_re, double* __restrict__ gz2_im)
{
    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    const int tid    = blockIdx.x * WARPS_PER_BLOCK + warpId;

    if (tid >= Nt) return;

    const int t3 = tid * 3;
    const double tx = __ldg(&tgt_xyz[t3]);
    const double ty = __ldg(&tgt_xyz[t3 + 1]);
    const double tz = __ldg(&tgt_xyz[t3 + 2]);

    double p1_re = 0.0, p1_im = 0.0;
    double p2_re = 0.0, p2_im = 0.0;
    double ax1_re = 0.0, ax1_im = 0.0, ay1_re = 0.0, ay1_im = 0.0, az1_re = 0.0, az1_im = 0.0;
    double ax2_re = 0.0, ax2_im = 0.0, ay2_re = 0.0, ay2_im = 0.0, az2_re = 0.0, az2_im = 0.0;

    const double inv4pi = 0.07957747154594767;
    const float k_re_f = (float)k_re;
    const float k_im_f = (float)k_im;

    const int leaf = __ldg(&tgt_to_leaf[tid]);
    const int nb_start = __ldg(&leaf_near_offsets[leaf]);
    const int nb_end   = __ldg(&leaf_near_offsets[leaf + 1]);

    for (int ni = nb_start; ni < nb_end; ni++) {
        int nb_leaf = __ldg(&leaf_near_nbrs[ni]);
        int s_start = __ldg(&src_id_offsets[nb_leaf]);
        int s_end   = __ldg(&src_id_offsets[nb_leaf + 1]);

        for (int j = s_start + lane; j < s_end; j += WARP_SIZE) {
            int sid = __ldg(&src_ids[j]);
            int sid3 = sid * 3;
            double dx = tx - __ldg(&src_xyz[sid3]);
            double dy = ty - __ldg(&src_xyz[sid3 + 1]);
            double dz = tz - __ldg(&src_xyz[sid3 + 2]);
            double R2 = dx*dx + dy*dy + dz*dz;

            if (R2 >= 1e-24) {
                double inv_R = rsqrt(R2);
                double R = R2 * inv_R;
                float Rf = (float)R;
                float cpf, spf;
                __sincosf(k_re_f * Rf, &spf, &cpf);
                float eR = __expf(-k_im_f * Rf);
                double Gfac = (double)eR * inv4pi * inv_R;
                double G_re = Gfac * (double)cpf;
                double G_im = Gfac * (double)spf;

                double fac_re = (-k_im - inv_R) * inv_R;
                double fac_im = k_re * inv_R;
                double gG_re = G_re * fac_re - G_im * fac_im;
                double gG_im = G_re * fac_im + G_im * fac_re;

                double q1r = __ldg(&q1_re[sid]), q1i = __ldg(&q1_im[sid]);
                p1_re += G_re * q1r - G_im * q1i;
                p1_im += G_re * q1i + G_im * q1r;
                double gq1_re = gG_re * q1r - gG_im * q1i;
                double gq1_im = gG_re * q1i + gG_im * q1r;
                ax1_re += gq1_re * dx; ax1_im += gq1_im * dx;
                ay1_re += gq1_re * dy; ay1_im += gq1_im * dy;
                az1_re += gq1_re * dz; az1_im += gq1_im * dz;

                double q2r = __ldg(&q2_re[sid]), q2i = __ldg(&q2_im[sid]);
                p2_re += G_re * q2r - G_im * q2i;
                p2_im += G_re * q2i + G_im * q2r;
                double gq2_re = gG_re * q2r - gG_im * q2i;
                double gq2_im = gG_re * q2i + gG_im * q2r;
                ax2_re += gq2_re * dx; ax2_im += gq2_im * dx;
                ay2_re += gq2_re * dy; ay2_im += gq2_im * dy;
                az2_re += gq2_re * dz; az2_im += gq2_im * dz;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        p1_re  += __shfl_down_sync(FULL_MASK, p1_re,  offset);
        p1_im  += __shfl_down_sync(FULL_MASK, p1_im,  offset);
        p2_re  += __shfl_down_sync(FULL_MASK, p2_re,  offset);
        p2_im  += __shfl_down_sync(FULL_MASK, p2_im,  offset);
        ax1_re += __shfl_down_sync(FULL_MASK, ax1_re, offset);
        ax1_im += __shfl_down_sync(FULL_MASK, ax1_im, offset);
        ay1_re += __shfl_down_sync(FULL_MASK, ay1_re, offset);
        ay1_im += __shfl_down_sync(FULL_MASK, ay1_im, offset);
        az1_re += __shfl_down_sync(FULL_MASK, az1_re, offset);
        az1_im += __shfl_down_sync(FULL_MASK, az1_im, offset);
        ax2_re += __shfl_down_sync(FULL_MASK, ax2_re, offset);
        ax2_im += __shfl_down_sync(FULL_MASK, ax2_im, offset);
        ay2_re += __shfl_down_sync(FULL_MASK, ay2_re, offset);
        ay2_im += __shfl_down_sync(FULL_MASK, ay2_im, offset);
        az2_re += __shfl_down_sync(FULL_MASK, az2_re, offset);
        az2_im += __shfl_down_sync(FULL_MASK, az2_im, offset);
    }

    if (lane == 0) {
        pot1_re[tid] += p1_re; pot1_im[tid] += p1_im;
        pot2_re[tid] += p2_re; pot2_im[tid] += p2_im;
        gx1_re[tid] += ax1_re; gx1_im[tid] += ax1_im;
        gy1_re[tid] += ay1_re; gy1_im[tid] += ay1_im;
        gz1_re[tid] += az1_re; gz1_im[tid] += az1_im;
        gx2_re[tid] += ax2_re; gx2_im[tid] += ax2_im;
        gy2_re[tid] += ay2_re; gy2_im[tid] += ay2_im;
        gz2_re[tid] += az2_re; gz2_im[tid] += az2_im;
    }
}

void launch_p2p_pot_grad_batch2(
    int Nt,
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const int* d_tgt_to_leaf, const int* d_leaf_near_offsets,
    const int* d_leaf_near_nbrs, const int* d_src_id_offsets,
    const int* d_src_ids,
    double k_re, double k_im,
    double* d_pot1_re, double* d_pot1_im,
    double* d_pot2_re, double* d_pot2_im,
    double* d_gx1_re, double* d_gx1_im,
    double* d_gy1_re, double* d_gy1_im,
    double* d_gz1_re, double* d_gz1_im,
    double* d_gx2_re, double* d_gx2_im,
    double* d_gy2_re, double* d_gy2_im,
    double* d_gz2_re, double* d_gz2_im,
    cudaStream_t stream)
{
    int grid = (Nt + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    p2p_pot_grad_batch2_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        Nt, d_tgt, d_src,
        d_q1_re, d_q1_im, d_q2_re, d_q2_im,
        d_tgt_to_leaf, d_leaf_near_offsets, d_leaf_near_nbrs,
        d_src_id_offsets, d_src_ids,
        k_re, k_im,
        d_pot1_re, d_pot1_im, d_pot2_re, d_pot2_im,
        d_gx1_re, d_gx1_im, d_gy1_re, d_gy1_im, d_gz1_re, d_gz1_im,
        d_gx2_re, d_gx2_im, d_gy2_re, d_gy2_im, d_gz2_re, d_gz2_im);
}
