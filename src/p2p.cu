#include "fmm.h"
#include <cuda_runtime.h>
#include <cstdio>

// ============================================================
// P2P near-field CUDA kernels (CSR format)
// ============================================================

// P2P scalar potential: phi_i = sum_j G(r_i, r_j) * q_j
// G = exp(ikR) / (4*pi*R)
// Each thread handles one target, loops over CSR source neighbors.
__global__ void p2p_potential_kernel(
    const double* __restrict__ tgt_xyz,   // (Nt*3)
    const double* __restrict__ src_xyz,   // (Ns*3)
    const double* __restrict__ q_re,      // (Ns)
    const double* __restrict__ q_im,      // (Ns)
    const int*    __restrict__ offsets,    // (Nt+1)
    const int*    __restrict__ src_idx,   // (nnz)
    double k_re, double k_im,
    double* __restrict__ out_re,          // (Nt)
    double* __restrict__ out_im,          // (Nt)
    int Nt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Nt) return;

    double tx = tgt_xyz[tid * 3];
    double ty = tgt_xyz[tid * 3 + 1];
    double tz = tgt_xyz[tid * 3 + 2];

    double acc_re = 0.0, acc_im = 0.0;
    const double inv4pi = 0.07957747154594767;

    int jstart = offsets[tid];
    int jend   = offsets[tid + 1];

    for (int j = jstart; j < jend; j++) {
        int sid = src_idx[j];
        double dx = tx - src_xyz[sid * 3];
        double dy = ty - src_xyz[sid * 3 + 1];
        double dz = tz - src_xyz[sid * 3 + 2];
        double R = sqrt(dx*dx + dy*dy + dz*dz);
        if (R < 1e-12) continue;

        double inv_R = 1.0 / R;
        double eR = exp(-k_im * R);
        double phase = k_re * R;
        double G_re = eR * cos(phase) * inv4pi * inv_R;
        double G_im = eR * sin(phase) * inv4pi * inv_R;

        double qr = q_re[sid];
        double qi = q_im[sid];
        acc_re += G_re * qr - G_im * qi;
        acc_im += G_re * qi + G_im * qr;
    }

    out_re[tid] += acc_re;
    out_im[tid] += acc_im;
}

// P2P gradient: (grad_phi)_i = sum_j nabla_G(r_i, r_j) * q_j
// nabla_G = G * (ik - 1/R) / R * (r_i - r_j)
__global__ void p2p_gradient_kernel(
    const double* __restrict__ tgt_xyz,
    const double* __restrict__ src_xyz,
    const double* __restrict__ q_re,
    const double* __restrict__ q_im,
    const int*    __restrict__ offsets,
    const int*    __restrict__ src_idx,
    double k_re, double k_im,
    double* __restrict__ gx_re, double* __restrict__ gx_im,
    double* __restrict__ gy_re, double* __restrict__ gy_im,
    double* __restrict__ gz_re, double* __restrict__ gz_im,
    int Nt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Nt) return;

    double tx = tgt_xyz[tid * 3];
    double ty = tgt_xyz[tid * 3 + 1];
    double tz = tgt_xyz[tid * 3 + 2];

    double ax_re = 0.0, ax_im = 0.0;
    double ay_re = 0.0, ay_im = 0.0;
    double az_re = 0.0, az_im = 0.0;

    const double inv4pi = 0.07957747154594767;

    int jstart = offsets[tid];
    int jend   = offsets[tid + 1];

    for (int j = jstart; j < jend; j++) {
        int sid = src_idx[j];
        double dx = tx - src_xyz[sid * 3];
        double dy = ty - src_xyz[sid * 3 + 1];
        double dz = tz - src_xyz[sid * 3 + 2];
        double R = sqrt(dx*dx + dy*dy + dz*dz);
        if (R < 1e-12) continue;

        double inv_R = 1.0 / R;
        double eR = exp(-k_im * R);
        double phase = k_re * R;
        double cp = cos(phase), sp = sin(phase);
        double G_re = eR * cp * inv4pi * inv_R;
        double G_im = eR * sp * inv4pi * inv_R;

        // factor = (ik - 1/R) / R  where ik = -k_im + i*k_re
        double fac_re = (-k_im - inv_R) * inv_R;
        double fac_im = k_re * inv_R;

        // gradG_scalar = G * factor
        double gG_re = G_re * fac_re - G_im * fac_im;
        double gG_im = G_re * fac_im + G_im * fac_re;

        double qr = q_re[sid];
        double qi = q_im[sid];
        double gq_re = gG_re * qr - gG_im * qi;
        double gq_im = gG_re * qi + gG_im * qr;

        ax_re += gq_re * dx; ax_im += gq_im * dx;
        ay_re += gq_re * dy; ay_im += gq_im * dy;
        az_re += gq_re * dz; az_im += gq_im * dz;
    }

    gx_re[tid] += ax_re; gx_im[tid] += ax_im;
    gy_re[tid] += ay_re; gy_im[tid] += ay_im;
    gz_re[tid] += az_re; gz_im[tid] += az_im;
}

// Host wrappers
void launch_p2p_potential(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_out_re, double* d_out_im,
    int Nt)
{
    int block = 256;
    int grid = (Nt + block - 1) / block;
    p2p_potential_kernel<<<grid, block>>>(
        d_tgt, d_src, d_q_re, d_q_im,
        d_offsets, d_indices,
        k_re, k_im, d_out_re, d_out_im, Nt);
}

__global__ void p2p_potential_batch2_kernel(
    const double* __restrict__ tgt_xyz,
    const double* __restrict__ src_xyz,
    const double* __restrict__ q1_re,
    const double* __restrict__ q1_im,
    const double* __restrict__ q2_re,
    const double* __restrict__ q2_im,
    const int*    __restrict__ offsets,
    const int*    __restrict__ src_idx,
    double k_re, double k_im,
    double* __restrict__ out1_re,
    double* __restrict__ out1_im,
    double* __restrict__ out2_re,
    double* __restrict__ out2_im,
    int Nt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Nt) return;

    double tx = tgt_xyz[tid * 3];
    double ty = tgt_xyz[tid * 3 + 1];
    double tz = tgt_xyz[tid * 3 + 2];

    double a1_re = 0.0, a1_im = 0.0;
    double a2_re = 0.0, a2_im = 0.0;
    const double inv4pi = 0.07957747154594767;

    int jstart = offsets[tid];
    int jend   = offsets[tid + 1];

    for (int j = jstart; j < jend; j++) {
        int sid = src_idx[j];
        double dx = tx - src_xyz[sid * 3];
        double dy = ty - src_xyz[sid * 3 + 1];
        double dz = tz - src_xyz[sid * 3 + 2];
        double R = sqrt(dx*dx + dy*dy + dz*dz);
        if (R < 1e-12) continue;

        double inv_R = 1.0 / R;
        double eR = exp(-k_im * R);
        double phase = k_re * R;
        double G_re = eR * cos(phase) * inv4pi * inv_R;
        double G_im = eR * sin(phase) * inv4pi * inv_R;

        double q1r = q1_re[sid], q1i = q1_im[sid];
        double q2r = q2_re[sid], q2i = q2_im[sid];
        a1_re += G_re * q1r - G_im * q1i;
        a1_im += G_re * q1i + G_im * q1r;
        a2_re += G_re * q2r - G_im * q2i;
        a2_im += G_re * q2i + G_im * q2r;
    }

    out1_re[tid] += a1_re; out1_im[tid] += a1_im;
    out2_re[tid] += a2_re; out2_im[tid] += a2_im;
}

void launch_p2p_potential_batch2(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_out1_re, double* d_out1_im,
    double* d_out2_re, double* d_out2_im,
    int Nt)
{
    int block = 256;
    int grid = (Nt + block - 1) / block;
    p2p_potential_batch2_kernel<<<grid, block>>>(
        d_tgt, d_src, d_q1_re, d_q1_im, d_q2_re, d_q2_im,
        d_offsets, d_indices, k_re, k_im,
        d_out1_re, d_out1_im, d_out2_re, d_out2_im, Nt);
}

__global__ void p2p_potential_batch4_kernel(
    const double* __restrict__ tgt_xyz, const double* __restrict__ src_xyz,
    const double* __restrict__ q1_re, const double* __restrict__ q1_im,
    const double* __restrict__ q2_re, const double* __restrict__ q2_im,
    const double* __restrict__ q3_re, const double* __restrict__ q3_im,
    const double* __restrict__ q4_re, const double* __restrict__ q4_im,
    const int* __restrict__ offsets, const int* __restrict__ src_idx,
    double k_re, double k_im,
    double* __restrict__ out1_re, double* __restrict__ out1_im,
    double* __restrict__ out2_re, double* __restrict__ out2_im,
    double* __restrict__ out3_re, double* __restrict__ out3_im,
    double* __restrict__ out4_re, double* __restrict__ out4_im,
    int Nt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Nt) return;
    double tx = tgt_xyz[tid*3], ty = tgt_xyz[tid*3+1], tz = tgt_xyz[tid*3+2];
    double a1r=0,a1i=0,a2r=0,a2i=0,a3r=0,a3i=0,a4r=0,a4i=0;
    const double inv4pi = 0.07957747154594767;
    for (int j = offsets[tid]; j < offsets[tid + 1]; j++) {
        int sid = src_idx[j];
        double dx = tx - src_xyz[sid*3], dy = ty - src_xyz[sid*3+1], dz = tz - src_xyz[sid*3+2];
        double R = sqrt(dx*dx + dy*dy + dz*dz);
        if (R < 1e-12) continue;
        double inv_R = 1.0 / R;
        double eR = exp(-k_im * R);
        double phase = k_re * R;
        double G_re = eR * cos(phase) * inv4pi * inv_R;
        double G_im = eR * sin(phase) * inv4pi * inv_R;
#define ACC_POT(QR, QI, AR, AI) do { double qr=(QR), qi=(QI); AR += G_re*qr - G_im*qi; AI += G_re*qi + G_im*qr; } while (0)
        ACC_POT(q1_re[sid], q1_im[sid], a1r, a1i);
        ACC_POT(q2_re[sid], q2_im[sid], a2r, a2i);
        ACC_POT(q3_re[sid], q3_im[sid], a3r, a3i);
        ACC_POT(q4_re[sid], q4_im[sid], a4r, a4i);
#undef ACC_POT
    }
    out1_re[tid] += a1r; out1_im[tid] += a1i;
    out2_re[tid] += a2r; out2_im[tid] += a2i;
    out3_re[tid] += a3r; out3_im[tid] += a3i;
    out4_re[tid] += a4r; out4_im[tid] += a4i;
}

void launch_p2p_potential_batch4(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const double* d_q3_re, const double* d_q3_im,
    const double* d_q4_re, const double* d_q4_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_out1_re, double* d_out1_im,
    double* d_out2_re, double* d_out2_im,
    double* d_out3_re, double* d_out3_im,
    double* d_out4_re, double* d_out4_im,
    int Nt)
{
    int block = 256;
    int grid = (Nt + block - 1) / block;
    p2p_potential_batch4_kernel<<<grid, block>>>(
        d_tgt, d_src,
        d_q1_re, d_q1_im, d_q2_re, d_q2_im, d_q3_re, d_q3_im, d_q4_re, d_q4_im,
        d_offsets, d_indices, k_re, k_im,
        d_out1_re, d_out1_im, d_out2_re, d_out2_im,
        d_out3_re, d_out3_im, d_out4_re, d_out4_im, Nt);
}

void launch_p2p_gradient(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_gx_re, double* d_gx_im,
    double* d_gy_re, double* d_gy_im,
    double* d_gz_re, double* d_gz_im,
    int Nt)
{
    int block = 256;
    int grid = (Nt + block - 1) / block;
    p2p_gradient_kernel<<<grid, block>>>(
        d_tgt, d_src, d_q_re, d_q_im,
        d_offsets, d_indices,
        k_re, k_im,
        d_gx_re, d_gx_im, d_gy_re, d_gy_im, d_gz_re, d_gz_im,
        Nt);
}

// P2P combined potential + gradient in a single pass.
// Computes both phi_i = sum_j G * q_j  and  grad_phi_i = sum_j nabla_G * q_j
// avoiding redundant distance/Green's function evaluations.
__global__ void p2p_pot_grad_kernel(
    int Nt,
    const double* __restrict__ tgt_xyz,
    const double* __restrict__ src_xyz,
    const double* __restrict__ q_re,
    const double* __restrict__ q_im,
    const int*    __restrict__ offsets,
    const int*    __restrict__ src_idx,
    double k_re, double k_im,
    double* __restrict__ pot_re,
    double* __restrict__ pot_im,
    double* __restrict__ gx_re, double* __restrict__ gx_im,
    double* __restrict__ gy_re, double* __restrict__ gy_im,
    double* __restrict__ gz_re, double* __restrict__ gz_im)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Nt) return;

    double tx = tgt_xyz[tid * 3];
    double ty = tgt_xyz[tid * 3 + 1];
    double tz = tgt_xyz[tid * 3 + 2];

    double p_re = 0.0, p_im = 0.0;
    double ax_re = 0.0, ax_im = 0.0;
    double ay_re = 0.0, ay_im = 0.0;
    double az_re = 0.0, az_im = 0.0;

    const double inv4pi = 0.07957747154594767;

    int jstart = offsets[tid];
    int jend   = offsets[tid + 1];

    for (int j = jstart; j < jend; j++) {
        int sid = src_idx[j];
        double dx = tx - src_xyz[sid * 3];
        double dy = ty - src_xyz[sid * 3 + 1];
        double dz = tz - src_xyz[sid * 3 + 2];
        double R = sqrt(dx*dx + dy*dy + dz*dz);
        if (R < 1e-12) continue;

        double inv_R = 1.0 / R;
        double eR = exp(-k_im * R);
        double phase = k_re * R;
        double cp = cos(phase), sp = sin(phase);
        double G_re = eR * cp * inv4pi * inv_R;
        double G_im = eR * sp * inv4pi * inv_R;

        double qr = q_re[sid];
        double qi = q_im[sid];

        // Potential: q * G
        p_re += G_re * qr - G_im * qi;
        p_im += G_re * qi + G_im * qr;

        // Gradient: q * nabla_G, where nabla_G = G * (ik - 1/R) / R * d
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

    pot_re[tid] += p_re; pot_im[tid] += p_im;
    gx_re[tid] += ax_re; gx_im[tid] += ax_im;
    gy_re[tid] += ay_re; gy_im[tid] += ay_im;
    gz_re[tid] += az_re; gz_im[tid] += az_im;
}

void launch_p2p_pot_grad(
    int Nt,
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_pot_re, double* d_pot_im,
    double* d_gx_re, double* d_gx_im,
    double* d_gy_re, double* d_gy_im,
    double* d_gz_re, double* d_gz_im)
{
    int block = 256;
    int grid = (Nt + block - 1) / block;
    p2p_pot_grad_kernel<<<grid, block>>>(
        Nt, d_tgt, d_src, d_q_re, d_q_im,
        d_offsets, d_indices,
        k_re, k_im,
        d_pot_re, d_pot_im,
        d_gx_re, d_gx_im, d_gy_re, d_gy_im, d_gz_re, d_gz_im);
}

__global__ void p2p_pot_grad_batch2_kernel(
    int Nt,
    const double* __restrict__ tgt_xyz,
    const double* __restrict__ src_xyz,
    const double* __restrict__ q1_re,
    const double* __restrict__ q1_im,
    const double* __restrict__ q2_re,
    const double* __restrict__ q2_im,
    const int*    __restrict__ offsets,
    const int*    __restrict__ src_idx,
    double k_re, double k_im,
    double* __restrict__ pot1_re,
    double* __restrict__ pot1_im,
    double* __restrict__ gx1_re, double* __restrict__ gx1_im,
    double* __restrict__ gy1_re, double* __restrict__ gy1_im,
    double* __restrict__ gz1_re, double* __restrict__ gz1_im,
    double* __restrict__ pot2_re,
    double* __restrict__ pot2_im,
    double* __restrict__ gx2_re, double* __restrict__ gx2_im,
    double* __restrict__ gy2_re, double* __restrict__ gy2_im,
    double* __restrict__ gz2_re, double* __restrict__ gz2_im)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Nt) return;

    double tx = tgt_xyz[tid * 3];
    double ty = tgt_xyz[tid * 3 + 1];
    double tz = tgt_xyz[tid * 3 + 2];

    double p1_re = 0.0, p1_im = 0.0, p2_re = 0.0, p2_im = 0.0;
    double ax1_re = 0.0, ax1_im = 0.0, ay1_re = 0.0, ay1_im = 0.0, az1_re = 0.0, az1_im = 0.0;
    double ax2_re = 0.0, ax2_im = 0.0, ay2_re = 0.0, ay2_im = 0.0, az2_re = 0.0, az2_im = 0.0;
    const double inv4pi = 0.07957747154594767;

    int jstart = offsets[tid];
    int jend   = offsets[tid + 1];

    for (int j = jstart; j < jend; j++) {
        int sid = src_idx[j];
        double dx = tx - src_xyz[sid * 3];
        double dy = ty - src_xyz[sid * 3 + 1];
        double dz = tz - src_xyz[sid * 3 + 2];
        double R = sqrt(dx*dx + dy*dy + dz*dz);
        if (R < 1e-12) continue;

        double inv_R = 1.0 / R;
        double eR = exp(-k_im * R);
        double phase = k_re * R;
        double cp = cos(phase), sp = sin(phase);
        double G_re = eR * cp * inv4pi * inv_R;
        double G_im = eR * sp * inv4pi * inv_R;

        double q1r = q1_re[sid], q1i = q1_im[sid];
        double q2r = q2_re[sid], q2i = q2_im[sid];

        p1_re += G_re * q1r - G_im * q1i;
        p1_im += G_re * q1i + G_im * q1r;
        p2_re += G_re * q2r - G_im * q2i;
        p2_im += G_re * q2i + G_im * q2r;

        double fac_re = (-k_im - inv_R) * inv_R;
        double fac_im = k_re * inv_R;
        double gG_re = G_re * fac_re - G_im * fac_im;
        double gG_im = G_re * fac_im + G_im * fac_re;

        double gq1_re = gG_re * q1r - gG_im * q1i;
        double gq1_im = gG_re * q1i + gG_im * q1r;
        double gq2_re = gG_re * q2r - gG_im * q2i;
        double gq2_im = gG_re * q2i + gG_im * q2r;

        ax1_re += gq1_re * dx; ax1_im += gq1_im * dx;
        ay1_re += gq1_re * dy; ay1_im += gq1_im * dy;
        az1_re += gq1_re * dz; az1_im += gq1_im * dz;
        ax2_re += gq2_re * dx; ax2_im += gq2_im * dx;
        ay2_re += gq2_re * dy; ay2_im += gq2_im * dy;
        az2_re += gq2_re * dz; az2_im += gq2_im * dz;
    }

    pot1_re[tid] += p1_re; pot1_im[tid] += p1_im;
    gx1_re[tid] += ax1_re; gx1_im[tid] += ax1_im;
    gy1_re[tid] += ay1_re; gy1_im[tid] += ay1_im;
    gz1_re[tid] += az1_re; gz1_im[tid] += az1_im;

    pot2_re[tid] += p2_re; pot2_im[tid] += p2_im;
    gx2_re[tid] += ax2_re; gx2_im[tid] += ax2_im;
    gy2_re[tid] += ay2_re; gy2_im[tid] += ay2_im;
    gz2_re[tid] += az2_re; gz2_im[tid] += az2_im;
}

void launch_p2p_pot_grad_batch2(
    int Nt,
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_pot1_re, double* d_pot1_im,
    double* d_gx1_re, double* d_gx1_im,
    double* d_gy1_re, double* d_gy1_im,
    double* d_gz1_re, double* d_gz1_im,
    double* d_pot2_re, double* d_pot2_im,
    double* d_gx2_re, double* d_gx2_im,
    double* d_gy2_re, double* d_gy2_im,
    double* d_gz2_re, double* d_gz2_im)
{
    int block = 256;
    int grid = (Nt + block - 1) / block;
    p2p_pot_grad_batch2_kernel<<<grid, block>>>(
        Nt, d_tgt, d_src, d_q1_re, d_q1_im, d_q2_re, d_q2_im,
        d_offsets, d_indices, k_re, k_im,
        d_pot1_re, d_pot1_im,
        d_gx1_re, d_gx1_im, d_gy1_re, d_gy1_im, d_gz1_re, d_gz1_im,
        d_pot2_re, d_pot2_im,
        d_gx2_re, d_gx2_im, d_gy2_re, d_gy2_im, d_gz2_re, d_gz2_im);
}

__global__ void p2p_pot_grad_batch4_kernel(
    int Nt,
    const double* __restrict__ tgt_xyz, const double* __restrict__ src_xyz,
    const double* __restrict__ q1_re, const double* __restrict__ q1_im,
    const double* __restrict__ q2_re, const double* __restrict__ q2_im,
    const double* __restrict__ q3_re, const double* __restrict__ q3_im,
    const double* __restrict__ q4_re, const double* __restrict__ q4_im,
    const int* __restrict__ offsets, const int* __restrict__ src_idx,
    double k_re, double k_im,
    double* __restrict__ p1_re, double* __restrict__ p1_im,
    double* __restrict__ gx1_re, double* __restrict__ gx1_im,
    double* __restrict__ gy1_re, double* __restrict__ gy1_im,
    double* __restrict__ gz1_re, double* __restrict__ gz1_im,
    double* __restrict__ p2_re, double* __restrict__ p2_im,
    double* __restrict__ gx2_re, double* __restrict__ gx2_im,
    double* __restrict__ gy2_re, double* __restrict__ gy2_im,
    double* __restrict__ gz2_re, double* __restrict__ gz2_im,
    double* __restrict__ p3_re, double* __restrict__ p3_im,
    double* __restrict__ gx3_re, double* __restrict__ gx3_im,
    double* __restrict__ gy3_re, double* __restrict__ gy3_im,
    double* __restrict__ gz3_re, double* __restrict__ gz3_im,
    double* __restrict__ p4_re, double* __restrict__ p4_im,
    double* __restrict__ gx4_re, double* __restrict__ gx4_im,
    double* __restrict__ gy4_re, double* __restrict__ gy4_im,
    double* __restrict__ gz4_re, double* __restrict__ gz4_im)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Nt) return;
    double tx = tgt_xyz[tid*3], ty = tgt_xyz[tid*3+1], tz = tgt_xyz[tid*3+2];
    double p1r=0,p1i=0,p2r=0,p2i=0,p3r=0,p3i=0,p4r=0,p4i=0;
    double ax1r=0,ax1i=0,ay1r=0,ay1i=0,az1r=0,az1i=0;
    double ax2r=0,ax2i=0,ay2r=0,ay2i=0,az2r=0,az2i=0;
    double ax3r=0,ax3i=0,ay3r=0,ay3i=0,az3r=0,az3i=0;
    double ax4r=0,ax4i=0,ay4r=0,ay4i=0,az4r=0,az4i=0;
    const double inv4pi = 0.07957747154594767;
    for (int j = offsets[tid]; j < offsets[tid + 1]; j++) {
        int sid = src_idx[j];
        double dx = tx - src_xyz[sid*3], dy = ty - src_xyz[sid*3+1], dz = tz - src_xyz[sid*3+2];
        double R = sqrt(dx*dx + dy*dy + dz*dz);
        if (R < 1e-12) continue;
        double inv_R = 1.0 / R;
        double eR = exp(-k_im * R);
        double phase = k_re * R;
        double cp = cos(phase), sp = sin(phase);
        double G_re = eR * cp * inv4pi * inv_R;
        double G_im = eR * sp * inv4pi * inv_R;
        double fac_re = (-k_im - inv_R) * inv_R;
        double fac_im = k_re * inv_R;
        double gG_re = G_re * fac_re - G_im * fac_im;
        double gG_im = G_re * fac_im + G_im * fac_re;
#define ACC_PG(QR, QI, PR, PI, AXR, AXI, AYR, AYI, AZR, AZI) \
        do { \
            double qr=(QR), qi=(QI); \
            PR += G_re*qr - G_im*qi; PI += G_re*qi + G_im*qr; \
            double gr = gG_re*qr - gG_im*qi; \
            double gi = gG_re*qi + gG_im*qr; \
            AXR += gr*dx; AXI += gi*dx; AYR += gr*dy; AYI += gi*dy; AZR += gr*dz; AZI += gi*dz; \
        } while (0)
        ACC_PG(q1_re[sid], q1_im[sid], p1r, p1i, ax1r, ax1i, ay1r, ay1i, az1r, az1i);
        ACC_PG(q2_re[sid], q2_im[sid], p2r, p2i, ax2r, ax2i, ay2r, ay2i, az2r, az2i);
        ACC_PG(q3_re[sid], q3_im[sid], p3r, p3i, ax3r, ax3i, ay3r, ay3i, az3r, az3i);
        ACC_PG(q4_re[sid], q4_im[sid], p4r, p4i, ax4r, ax4i, ay4r, ay4i, az4r, az4i);
#undef ACC_PG
    }
#define STORE_PG(PR, PI, GXR, GXI, GYR, GYI, GZR, GZI, PVR, PVI, AXR, AXI, AYR, AYI, AZR, AZI) \
    do { PR[tid] += PVR; PI[tid] += PVI; GXR[tid] += AXR; GXI[tid] += AXI; GYR[tid] += AYR; GYI[tid] += AYI; GZR[tid] += AZR; GZI[tid] += AZI; } while (0)
    STORE_PG(p1_re,p1_im,gx1_re,gx1_im,gy1_re,gy1_im,gz1_re,gz1_im,p1r,p1i,ax1r,ax1i,ay1r,ay1i,az1r,az1i);
    STORE_PG(p2_re,p2_im,gx2_re,gx2_im,gy2_re,gy2_im,gz2_re,gz2_im,p2r,p2i,ax2r,ax2i,ay2r,ay2i,az2r,az2i);
    STORE_PG(p3_re,p3_im,gx3_re,gx3_im,gy3_re,gy3_im,gz3_re,gz3_im,p3r,p3i,ax3r,ax3i,ay3r,ay3i,az3r,az3i);
    STORE_PG(p4_re,p4_im,gx4_re,gx4_im,gy4_re,gy4_im,gz4_re,gz4_im,p4r,p4i,ax4r,ax4i,ay4r,ay4i,az4r,az4i);
#undef STORE_PG
}

void launch_p2p_pot_grad_batch4(
    int Nt,
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const double* d_q3_re, const double* d_q3_im,
    const double* d_q4_re, const double* d_q4_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_pot1_re, double* d_pot1_im,
    double* d_gx1_re, double* d_gx1_im,
    double* d_gy1_re, double* d_gy1_im,
    double* d_gz1_re, double* d_gz1_im,
    double* d_pot2_re, double* d_pot2_im,
    double* d_gx2_re, double* d_gx2_im,
    double* d_gy2_re, double* d_gy2_im,
    double* d_gz2_re, double* d_gz2_im,
    double* d_pot3_re, double* d_pot3_im,
    double* d_gx3_re, double* d_gx3_im,
    double* d_gy3_re, double* d_gy3_im,
    double* d_gz3_re, double* d_gz3_im,
    double* d_pot4_re, double* d_pot4_im,
    double* d_gx4_re, double* d_gx4_im,
    double* d_gy4_re, double* d_gy4_im,
    double* d_gz4_re, double* d_gz4_im)
{
    int block = 256;
    int grid = (Nt + block - 1) / block;
    p2p_pot_grad_batch4_kernel<<<grid, block>>>(
        Nt, d_tgt, d_src,
        d_q1_re, d_q1_im, d_q2_re, d_q2_im, d_q3_re, d_q3_im, d_q4_re, d_q4_im,
        d_offsets, d_indices, k_re, k_im,
        d_pot1_re, d_pot1_im, d_gx1_re, d_gx1_im, d_gy1_re, d_gy1_im, d_gz1_re, d_gz1_im,
        d_pot2_re, d_pot2_im, d_gx2_re, d_gx2_im, d_gy2_re, d_gy2_im, d_gz2_re, d_gz2_im,
        d_pot3_re, d_pot3_im, d_gx3_re, d_gx3_im, d_gy3_re, d_gy3_im, d_gz3_re, d_gz3_im,
        d_pot4_re, d_pot4_im, d_gx4_re, d_gx4_im, d_gy4_re, d_gy4_im, d_gz4_re, d_gz4_im);
}
