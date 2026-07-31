#include "fmm.h"
#include "gpu_select.h"
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

template <int BATCH, bool NEED_GRAD>
__global__ void p2p_leaf_kernel(
    const double* __restrict__ tgt_xyz,
    const double* __restrict__ src_xyz,
    const double* __restrict__ q1_re, const double* __restrict__ q1_im,
    const double* __restrict__ q2_re, const double* __restrict__ q2_im,
    const int* __restrict__ tgt_offsets,
    const int* __restrict__ tgt_ids,
    const int* __restrict__ src_offsets,
    const int* __restrict__ src_ids,
    const int* __restrict__ near_offsets,
    const int* __restrict__ near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* __restrict__ pot1_re, double* __restrict__ pot1_im,
    double* __restrict__ gx1_re, double* __restrict__ gx1_im,
    double* __restrict__ gy1_re, double* __restrict__ gy1_im,
    double* __restrict__ gz1_re, double* __restrict__ gz1_im,
    double* __restrict__ pot2_re, double* __restrict__ pot2_im,
    double* __restrict__ gx2_re, double* __restrict__ gx2_im,
    double* __restrict__ gy2_re, double* __restrict__ gy2_im,
    double* __restrict__ gz2_re, double* __restrict__ gz2_im)
{
    int leaf = blockIdx.x;
    if (leaf >= n_leaves) return;

    int t0 = tgt_offsets[leaf];
    int t1 = tgt_offsets[leaf + 1];
    for (int ti = t0 + threadIdx.x; ti < t1; ti += blockDim.x) {
        int tid = tgt_ids[ti];
        double tx = tgt_xyz[tid * 3];
        double ty = tgt_xyz[tid * 3 + 1];
        double tz = tgt_xyz[tid * 3 + 2];

        double p1r = 0.0, p1i = 0.0, p2r = 0.0, p2i = 0.0;
        double ax1r = 0.0, ax1i = 0.0, ay1r = 0.0, ay1i = 0.0, az1r = 0.0, az1i = 0.0;
        double ax2r = 0.0, ax2i = 0.0, ay2r = 0.0, ay2i = 0.0, az2r = 0.0, az2i = 0.0;
        const double inv4pi = 0.07957747154594767;

        int n0 = near_offsets[leaf];
        int n1 = near_offsets[leaf + 1];
        for (int pass = -1; pass < n1 - n0; pass++) {
            int src_leaf = (pass < 0) ? leaf : near_leaf_ids[n0 + pass];
            int s0 = src_offsets[src_leaf];
            int s1 = src_offsets[src_leaf + 1];
            for (int si = s0; si < s1; si++) {
                int sid = src_ids[si];
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
                p1r += G_re * q1r - G_im * q1i;
                p1i += G_re * q1i + G_im * q1r;

                double gG_re = 0.0, gG_im = 0.0;
                if (NEED_GRAD) {
                    double fac_re = (-k_im - inv_R) * inv_R;
                    double fac_im = k_re * inv_R;
                    gG_re = G_re * fac_re - G_im * fac_im;
                    gG_im = G_re * fac_im + G_im * fac_re;
                    double gq_re = gG_re * q1r - gG_im * q1i;
                    double gq_im = gG_re * q1i + gG_im * q1r;
                    ax1r += gq_re * dx; ax1i += gq_im * dx;
                    ay1r += gq_re * dy; ay1i += gq_im * dy;
                    az1r += gq_re * dz; az1i += gq_im * dz;
                }

                if (BATCH >= 2) {
                    double q2r = q2_re[sid], q2i = q2_im[sid];
                    p2r += G_re * q2r - G_im * q2i;
                    p2i += G_re * q2i + G_im * q2r;
                    if (NEED_GRAD) {
                        double gq_re = gG_re * q2r - gG_im * q2i;
                        double gq_im = gG_re * q2i + gG_im * q2r;
                        ax2r += gq_re * dx; ax2i += gq_im * dx;
                        ay2r += gq_re * dy; ay2i += gq_im * dy;
                        az2r += gq_re * dz; az2i += gq_im * dz;
                    }
                }
            }
        }

        if (pot1_re) { pot1_re[tid] += p1r; pot1_im[tid] += p1i; }
        if (NEED_GRAD) {
            gx1_re[tid] += ax1r; gx1_im[tid] += ax1i;
            gy1_re[tid] += ay1r; gy1_im[tid] += ay1i;
            gz1_re[tid] += az1r; gz1_im[tid] += az1i;
        }
        if (BATCH >= 2) {
            if (pot2_re) { pot2_re[tid] += p2r; pot2_im[tid] += p2i; }
            if (NEED_GRAD) {
                gx2_re[tid] += ax2r; gx2_im[tid] += ax2i;
                gy2_re[tid] += ay2r; gy2_im[tid] += ay2i;
                gz2_re[tid] += az2r; gz2_im[tid] += az2i;
            }
        }
    }
}

void launch_p2p_potential_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_out_re, double* d_out_im)
{
    p2p_leaf_kernel<1, false><<<n_leaves, 128>>>(
        d_tgt, d_src, d_q_re, d_q_im, nullptr, nullptr,
        d_tgt_offsets, d_tgt_ids, d_src_offsets, d_src_ids,
        d_near_offsets, d_near_leaf_ids, n_leaves, k_re, k_im,
        d_out_re, d_out_im, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

void launch_p2p_potential_batch2_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_out1_re, double* d_out1_im,
    double* d_out2_re, double* d_out2_im)
{
    p2p_leaf_kernel<2, false><<<n_leaves, 128>>>(
        d_tgt, d_src, d_q1_re, d_q1_im, d_q2_re, d_q2_im,
        d_tgt_offsets, d_tgt_ids, d_src_offsets, d_src_ids,
        d_near_offsets, d_near_leaf_ids, n_leaves, k_re, k_im,
        d_out1_re, d_out1_im, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        d_out2_re, d_out2_im, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

template <bool NEED_GRAD>
__global__ void p2p_leaf_batch4_kernel(
    const double* __restrict__ tgt_xyz,
    const double* __restrict__ src_xyz,
    const double* __restrict__ q1_re, const double* __restrict__ q1_im,
    const double* __restrict__ q2_re, const double* __restrict__ q2_im,
    const double* __restrict__ q3_re, const double* __restrict__ q3_im,
    const double* __restrict__ q4_re, const double* __restrict__ q4_im,
    const int* __restrict__ tgt_offsets,
    const int* __restrict__ tgt_ids,
    const int* __restrict__ src_offsets,
    const int* __restrict__ src_ids,
    const int* __restrict__ near_offsets,
    const int* __restrict__ near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* __restrict__ pot1_re, double* __restrict__ pot1_im,
    double* __restrict__ gx1_re, double* __restrict__ gx1_im,
    double* __restrict__ gy1_re, double* __restrict__ gy1_im,
    double* __restrict__ gz1_re, double* __restrict__ gz1_im,
    double* __restrict__ pot2_re, double* __restrict__ pot2_im,
    double* __restrict__ gx2_re, double* __restrict__ gx2_im,
    double* __restrict__ gy2_re, double* __restrict__ gy2_im,
    double* __restrict__ gz2_re, double* __restrict__ gz2_im,
    double* __restrict__ pot3_re, double* __restrict__ pot3_im,
    double* __restrict__ gx3_re, double* __restrict__ gx3_im,
    double* __restrict__ gy3_re, double* __restrict__ gy3_im,
    double* __restrict__ gz3_re, double* __restrict__ gz3_im,
    double* __restrict__ pot4_re, double* __restrict__ pot4_im,
    double* __restrict__ gx4_re, double* __restrict__ gx4_im,
    double* __restrict__ gy4_re, double* __restrict__ gy4_im,
    double* __restrict__ gz4_re, double* __restrict__ gz4_im)
{
    int leaf = blockIdx.x;
    if (leaf >= n_leaves) return;

    int t0 = tgt_offsets[leaf];
    int t1 = tgt_offsets[leaf + 1];
    for (int ti = t0 + threadIdx.x; ti < t1; ti += blockDim.x) {
        int tid = tgt_ids[ti];
        double tx = tgt_xyz[tid * 3];
        double ty = tgt_xyz[tid * 3 + 1];
        double tz = tgt_xyz[tid * 3 + 2];

        double p1r = 0.0, p1i = 0.0, p2r = 0.0, p2i = 0.0;
        double p3r = 0.0, p3i = 0.0, p4r = 0.0, p4i = 0.0;
        double ax1r = 0.0, ax1i = 0.0, ay1r = 0.0, ay1i = 0.0, az1r = 0.0, az1i = 0.0;
        double ax2r = 0.0, ax2i = 0.0, ay2r = 0.0, ay2i = 0.0, az2r = 0.0, az2i = 0.0;
        double ax3r = 0.0, ax3i = 0.0, ay3r = 0.0, ay3i = 0.0, az3r = 0.0, az3i = 0.0;
        double ax4r = 0.0, ax4i = 0.0, ay4r = 0.0, ay4i = 0.0, az4r = 0.0, az4i = 0.0;
        const double inv4pi = 0.07957747154594767;

        int n0 = near_offsets[leaf];
        int n1 = near_offsets[leaf + 1];
        for (int pass = -1; pass < n1 - n0; pass++) {
            int src_leaf = (pass < 0) ? leaf : near_leaf_ids[n0 + pass];
            int s0 = src_offsets[src_leaf];
            int s1 = src_offsets[src_leaf + 1];
            for (int si = s0; si < s1; si++) {
                int sid = src_ids[si];
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
                double gG_re = 0.0, gG_im = 0.0;
                if (NEED_GRAD) {
                    double fac_re = (-k_im - inv_R) * inv_R;
                    double fac_im = k_re * inv_R;
                    gG_re = G_re * fac_re - G_im * fac_im;
                    gG_im = G_re * fac_im + G_im * fac_re;
                }
#define ACC4(QR, QI, PR, PI, AXR, AXI, AYR, AYI, AZR, AZI) \
                do { \
                    double qr = (QR), qi = (QI); \
                    PR += G_re * qr - G_im * qi; \
                    PI += G_re * qi + G_im * qr; \
                    if (NEED_GRAD) { \
                        double gr = gG_re * qr - gG_im * qi; \
                        double gi = gG_re * qi + gG_im * qr; \
                        AXR += gr * dx; AXI += gi * dx; \
                        AYR += gr * dy; AYI += gi * dy; \
                        AZR += gr * dz; AZI += gi * dz; \
                    } \
                } while (0)
                ACC4(q1_re[sid], q1_im[sid], p1r, p1i, ax1r, ax1i, ay1r, ay1i, az1r, az1i);
                ACC4(q2_re[sid], q2_im[sid], p2r, p2i, ax2r, ax2i, ay2r, ay2i, az2r, az2i);
                ACC4(q3_re[sid], q3_im[sid], p3r, p3i, ax3r, ax3i, ay3r, ay3i, az3r, az3i);
                ACC4(q4_re[sid], q4_im[sid], p4r, p4i, ax4r, ax4i, ay4r, ay4i, az4r, az4i);
#undef ACC4
            }
        }

#define STORE4(PR, PI, GXR, GXI, GYR, GYI, GZR, GZI, PV_R, PV_I, AXR, AXI, AYR, AYI, AZR, AZI) \
        do { \
            if (PR) { PR[tid] += PV_R; PI[tid] += PV_I; } \
            if (NEED_GRAD) { \
                GXR[tid] += AXR; GXI[tid] += AXI; \
                GYR[tid] += AYR; GYI[tid] += AYI; \
                GZR[tid] += AZR; GZI[tid] += AZI; \
            } \
        } while (0)
        STORE4(pot1_re, pot1_im, gx1_re, gx1_im, gy1_re, gy1_im, gz1_re, gz1_im,
               p1r, p1i, ax1r, ax1i, ay1r, ay1i, az1r, az1i);
        STORE4(pot2_re, pot2_im, gx2_re, gx2_im, gy2_re, gy2_im, gz2_re, gz2_im,
               p2r, p2i, ax2r, ax2i, ay2r, ay2i, az2r, az2i);
        STORE4(pot3_re, pot3_im, gx3_re, gx3_im, gy3_re, gy3_im, gz3_re, gz3_im,
               p3r, p3i, ax3r, ax3i, ay3r, ay3i, az3r, az3i);
        STORE4(pot4_re, pot4_im, gx4_re, gx4_im, gy4_re, gy4_im, gz4_re, gz4_im,
               p4r, p4i, ax4r, ax4i, ay4r, ay4i, az4r, az4i);
#undef STORE4
    }
}

void launch_p2p_potential_batch4_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const double* d_q3_re, const double* d_q3_im,
    const double* d_q4_re, const double* d_q4_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_out1_re, double* d_out1_im,
    double* d_out2_re, double* d_out2_im,
    double* d_out3_re, double* d_out3_im,
    double* d_out4_re, double* d_out4_im)
{
    p2p_leaf_batch4_kernel<false><<<n_leaves, 128>>>(
        d_tgt, d_src,
        d_q1_re, d_q1_im, d_q2_re, d_q2_im, d_q3_re, d_q3_im, d_q4_re, d_q4_im,
        d_tgt_offsets, d_tgt_ids, d_src_offsets, d_src_ids,
        d_near_offsets, d_near_leaf_ids, n_leaves, k_re, k_im,
        d_out1_re, d_out1_im, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        d_out2_re, d_out2_im, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        d_out3_re, d_out3_im, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        d_out4_re, d_out4_im, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

void launch_p2p_gradient_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_gx_re, double* d_gx_im,
    double* d_gy_re, double* d_gy_im,
    double* d_gz_re, double* d_gz_im)
{
    p2p_leaf_kernel<1, true><<<n_leaves, 128>>>(
        d_tgt, d_src, d_q_re, d_q_im, nullptr, nullptr,
        d_tgt_offsets, d_tgt_ids, d_src_offsets, d_src_ids,
        d_near_offsets, d_near_leaf_ids, n_leaves, k_re, k_im,
        nullptr, nullptr, d_gx_re, d_gx_im, d_gy_re, d_gy_im, d_gz_re, d_gz_im,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

__global__ void p2p_hessian_leaf_kernel(
    const double* __restrict__ tgt_xyz,
    const double* __restrict__ src_xyz,
    const double* __restrict__ q_re,
    const double* __restrict__ q_im,
    const int* __restrict__ tgt_offsets,
    const int* __restrict__ tgt_ids,
    const int* __restrict__ src_offsets,
    const int* __restrict__ src_ids,
    const int* __restrict__ near_offsets,
    const int* __restrict__ near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* __restrict__ hess_re,
    double* __restrict__ hess_im)
{
    const int leaf = blockIdx.x;
    if (leaf >= n_leaves) return;
    const double inv4pi = 0.07957747154594767;
    const double k2_re = k_re * k_re - k_im * k_im;
    const double k2_im = 2.0 * k_re * k_im;

    for (int ti = tgt_offsets[leaf] + threadIdx.x;
         ti < tgt_offsets[leaf + 1]; ti += blockDim.x) {
        const int tid = tgt_ids[ti];
        const double tx = tgt_xyz[3 * tid];
        const double ty = tgt_xyz[3 * tid + 1];
        const double tz = tgt_xyz[3 * tid + 2];
        double acc_re[6] = {0, 0, 0, 0, 0, 0};
        double acc_im[6] = {0, 0, 0, 0, 0, 0};

        const int near_begin = near_offsets[leaf];
        const int near_end = near_offsets[leaf + 1];
        for (int pass = -1; pass < near_end - near_begin; pass++) {
            const int source_leaf =
                pass < 0 ? leaf : near_leaf_ids[near_begin + pass];
            for (int si = src_offsets[source_leaf];
                 si < src_offsets[source_leaf + 1]; si++) {
                const int sid = src_ids[si];
                const double dx = tx - src_xyz[3 * sid];
                const double dy = ty - src_xyz[3 * sid + 1];
                const double dz = tz - src_xyz[3 * sid + 2];
                const double radius =
                    sqrt(dx * dx + dy * dy + dz * dz);
                if (radius < 1.0e-12) continue;
                const double inv_r = 1.0 / radius;
                const double inv_r2 = inv_r * inv_r;
                const double attenuation = exp(-k_im * radius);
                const double phase = k_re * radius;
                const double green_re =
                    attenuation * cos(phase) * inv4pi * inv_r;
                const double green_im =
                    attenuation * sin(phase) * inv4pi * inv_r;

                // A = 3/r^2 - 3 i k/r - k^2,
                // B = 1/r^2 - i k/r, and
                // Hess(G) = G [A rhat rhat^T - B I].
                const double a_re =
                    3.0 * inv_r2 + 3.0 * k_im * inv_r - k2_re;
                const double a_im =
                    -3.0 * k_re * inv_r - k2_im;
                const double b_re = inv_r2 + k_im * inv_r;
                const double b_im = -k_re * inv_r;
                const double ga_re =
                    green_re * a_re - green_im * a_im;
                const double ga_im =
                    green_re * a_im + green_im * a_re;
                const double gb_re =
                    green_re * b_re - green_im * b_im;
                const double gb_im =
                    green_re * b_im + green_im * b_re;
                const double unit[3] = {
                    dx * inv_r, dy * inv_r, dz * inv_r
                };
                const double qr = q_re[sid];
                const double qi = q_im[sid];
                const int row[6] = {0, 0, 0, 1, 1, 2};
                const int col[6] = {0, 1, 2, 1, 2, 2};
                for (int component = 0; component < 6; component++) {
                    double hr =
                        ga_re * unit[row[component]] * unit[col[component]];
                    double hi =
                        ga_im * unit[row[component]] * unit[col[component]];
                    if (row[component] == col[component]) {
                        hr -= gb_re;
                        hi -= gb_im;
                    }
                    acc_re[component] += hr * qr - hi * qi;
                    acc_im[component] += hr * qi + hi * qr;
                }
            }
        }
        for (int component = 0; component < 6; component++) {
            hess_re[6 * tid + component] += acc_re[component];
            hess_im[6 * tid + component] += acc_im[component];
        }
    }
}

__device__ inline float p2p_inverse_sqrt(float value)
{
    return rsqrtf(value);
}

__device__ inline double p2p_inverse_sqrt(double value)
{
    return 1.0 / sqrt(value);
}

__device__ inline float p2p_attenuation(float value)
{
    return expf(value);
}

__device__ inline double p2p_attenuation(double value)
{
    return exp(value);
}

__device__ inline void p2p_sine_cosine(
    float value, float* sine, float* cosine)
{
    sincosf(value, sine, cosine);
}

__device__ inline void p2p_sine_cosine(
    double value, double* sine, double* cosine)
{
    sincos(value, sine, cosine);
}

template <bool FastTrig>
__device__ inline void p2p_pair_sine_cosine(
    float value, float* sine, float* cosine)
{
    if (FastTrig)
        __sincosf(value, sine, cosine);
    else
        sincosf(value, sine, cosine);
}

template <bool FastTrig>
__device__ inline void p2p_pair_sine_cosine(
    double value, double* sine, double* cosine)
{
    sincos(value, sine, cosine);
}

template <typename ComputeReal>
__global__ void p2p_grad_hessian_batch3_leaf_kernel(
    const double* __restrict__ tgt_xyz,
    const double* __restrict__ src_xyz,
    const double* __restrict__ q1_re,
    const double* __restrict__ q1_im,
    const double* __restrict__ q2_re,
    const double* __restrict__ q2_im,
    const double* __restrict__ q3_re,
    const double* __restrict__ q3_im,
    const int* __restrict__ tgt_offsets,
    const int* __restrict__ tgt_ids,
    const int* __restrict__ src_offsets,
    const int* __restrict__ src_ids,
    const int* __restrict__ near_offsets,
    const int* __restrict__ near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* __restrict__ gx1_re,
    double* __restrict__ gx1_im,
    double* __restrict__ gy1_re,
    double* __restrict__ gy1_im,
    double* __restrict__ gz1_re,
    double* __restrict__ gz1_im,
    double* __restrict__ gx2_re,
    double* __restrict__ gx2_im,
    double* __restrict__ gy2_re,
    double* __restrict__ gy2_im,
    double* __restrict__ gz2_re,
    double* __restrict__ gz2_im,
    double* __restrict__ gx3_re,
    double* __restrict__ gx3_im,
    double* __restrict__ gy3_re,
    double* __restrict__ gy3_im,
    double* __restrict__ gz3_re,
    double* __restrict__ gz3_im,
    double* __restrict__ hess1_re,
    double* __restrict__ hess1_im,
    double* __restrict__ hess2_re,
    double* __restrict__ hess2_im,
    double* __restrict__ hess3_re,
    double* __restrict__ hess3_im)
{
    const int leaf = blockIdx.x;
    if (leaf >= n_leaves) return;
    const ComputeReal real_k = static_cast<ComputeReal>(k_re);
    const ComputeReal imaginary_k = static_cast<ComputeReal>(k_im);
    const ComputeReal inv4pi =
        static_cast<ComputeReal>(0.07957747154594767);
    const ComputeReal k2_re =
        real_k * real_k - imaginary_k * imaginary_k;
    const ComputeReal k2_im =
        static_cast<ComputeReal>(2.0) * real_k * imaginary_k;
    const int row[6] = {0, 0, 0, 1, 1, 2};
    const int col[6] = {0, 1, 2, 1, 2, 2};

    for (int ti = tgt_offsets[leaf] + threadIdx.x;
        ti < tgt_offsets[leaf + 1]; ti += blockDim.x) {
        const int tid = tgt_ids[ti];
        const ComputeReal tx =
            static_cast<ComputeReal>(tgt_xyz[3 * tid]);
        const ComputeReal ty =
            static_cast<ComputeReal>(tgt_xyz[3 * tid + 1]);
        const ComputeReal tz =
            static_cast<ComputeReal>(tgt_xyz[3 * tid + 2]);
        ComputeReal acc1_re[6] = {0, 0, 0, 0, 0, 0};
        ComputeReal acc1_im[6] = {0, 0, 0, 0, 0, 0};
        ComputeReal acc2_re[6] = {0, 0, 0, 0, 0, 0};
        ComputeReal acc2_im[6] = {0, 0, 0, 0, 0, 0};
        ComputeReal acc3_re[6] = {0, 0, 0, 0, 0, 0};
        ComputeReal acc3_im[6] = {0, 0, 0, 0, 0, 0};
        ComputeReal grad1_re[3] = {0, 0, 0};
        ComputeReal grad1_im[3] = {0, 0, 0};
        ComputeReal grad2_re[3] = {0, 0, 0};
        ComputeReal grad2_im[3] = {0, 0, 0};
        ComputeReal grad3_re[3] = {0, 0, 0};
        ComputeReal grad3_im[3] = {0, 0, 0};

        const int near_begin = near_offsets[leaf];
        const int near_end = near_offsets[leaf + 1];
        for (int pass = -1; pass < near_end - near_begin; pass++) {
            const int source_leaf =
                pass < 0 ? leaf : near_leaf_ids[near_begin + pass];
            for (int si = src_offsets[source_leaf];
                si < src_offsets[source_leaf + 1]; si++) {
                const int sid = src_ids[si];
                const ComputeReal dx = tx -
                    static_cast<ComputeReal>(src_xyz[3 * sid]);
                const ComputeReal dy = ty -
                    static_cast<ComputeReal>(src_xyz[3 * sid + 1]);
                const ComputeReal dz = tz -
                    static_cast<ComputeReal>(src_xyz[3 * sid + 2]);
                const ComputeReal radius_squared =
                    dx * dx + dy * dy + dz * dz;
                if (radius_squared <
                    static_cast<ComputeReal>(1.0e-24))
                    continue;
                const ComputeReal inv_r =
                    p2p_inverse_sqrt(radius_squared);
                const ComputeReal radius = radius_squared * inv_r;
                const ComputeReal inv_r2 = inv_r * inv_r;
                const ComputeReal attenuation =
                    imaginary_k == static_cast<ComputeReal>(0)
                        ? static_cast<ComputeReal>(1)
                        : p2p_attenuation(-imaginary_k * radius);
                const ComputeReal phase = real_k * radius;
                ComputeReal sine = 0;
                ComputeReal cosine = 0;
                p2p_sine_cosine(phase, &sine, &cosine);
                const ComputeReal green_re =
                    attenuation * cosine * inv4pi * inv_r;
                const ComputeReal green_im =
                    attenuation * sine * inv4pi * inv_r;
                const ComputeReal gradient_factor_re =
                    (-imaginary_k - inv_r) * inv_r;
                const ComputeReal gradient_factor_im = real_k * inv_r;
                const ComputeReal gradient_green_re =
                    green_re * gradient_factor_re -
                    green_im * gradient_factor_im;
                const ComputeReal gradient_green_im =
                    green_re * gradient_factor_im +
                    green_im * gradient_factor_re;
                const ComputeReal a_re =
                    static_cast<ComputeReal>(3.0) * inv_r2 +
                    static_cast<ComputeReal>(3.0) *
                        imaginary_k * inv_r - k2_re;
                const ComputeReal a_im =
                    -static_cast<ComputeReal>(3.0) *
                        real_k * inv_r - k2_im;
                const ComputeReal b_re =
                    inv_r2 + imaginary_k * inv_r;
                const ComputeReal b_im = -real_k * inv_r;
                const ComputeReal ga_re =
                    green_re * a_re - green_im * a_im;
                const ComputeReal ga_im =
                    green_re * a_im + green_im * a_re;
                const ComputeReal gb_re =
                    green_re * b_re - green_im * b_im;
                const ComputeReal gb_im =
                    green_re * b_im + green_im * b_re;
                const ComputeReal unit[3] = {
                    dx * inv_r, dy * inv_r, dz * inv_r
                };
                const ComputeReal qr[3] = {
                    static_cast<ComputeReal>(q1_re[sid]),
                    static_cast<ComputeReal>(q2_re[sid]),
                    static_cast<ComputeReal>(q3_re[sid])
                };
                const ComputeReal qi[3] = {
                    static_cast<ComputeReal>(q1_im[sid]),
                    static_cast<ComputeReal>(q2_im[sid]),
                    static_cast<ComputeReal>(q3_im[sid])
                };
                const ComputeReal displacement[3] = {dx, dy, dz};
                for (int axis = 0; axis < 3; axis++) {
#define ACCUMULATE_GRADIENT(GRAD_RE, GRAD_IM, INDEX) \
                    do { \
                        const ComputeReal value_re = \
                            gradient_green_re * qr[INDEX] - \
                            gradient_green_im * qi[INDEX]; \
                        const ComputeReal value_im = \
                            gradient_green_re * qi[INDEX] + \
                            gradient_green_im * qr[INDEX]; \
                        (GRAD_RE)[axis] += value_re * displacement[axis]; \
                        (GRAD_IM)[axis] += value_im * displacement[axis]; \
                    } while (0)
                    ACCUMULATE_GRADIENT(grad1_re, grad1_im, 0);
                    ACCUMULATE_GRADIENT(grad2_re, grad2_im, 1);
                    ACCUMULATE_GRADIENT(grad3_re, grad3_im, 2);
#undef ACCUMULATE_GRADIENT
                }
                for (int component = 0; component < 6; component++) {
                    ComputeReal hr =
                        ga_re * unit[row[component]] * unit[col[component]];
                    ComputeReal hi =
                        ga_im * unit[row[component]] * unit[col[component]];
                    if (row[component] == col[component]) {
                        hr -= gb_re;
                        hi -= gb_im;
                    }
#define ACCUMULATE_HESSIAN(ACC_RE, ACC_IM, INDEX) \
                    do { \
                        (ACC_RE)[component] += hr * qr[INDEX] - hi * qi[INDEX]; \
                        (ACC_IM)[component] += hr * qi[INDEX] + hi * qr[INDEX]; \
                    } while (0)
                    ACCUMULATE_HESSIAN(acc1_re, acc1_im, 0);
                    ACCUMULATE_HESSIAN(acc2_re, acc2_im, 1);
                    ACCUMULATE_HESSIAN(acc3_re, acc3_im, 2);
#undef ACCUMULATE_HESSIAN
                }
            }
        }
        for (int component = 0; component < 6; component++) {
            const int offset = 6 * tid + component;
            hess1_re[offset] += acc1_re[component];
            hess1_im[offset] += acc1_im[component];
            hess2_re[offset] += acc2_re[component];
            hess2_im[offset] += acc2_im[component];
            hess3_re[offset] += acc3_re[component];
            hess3_im[offset] += acc3_im[component];
        }
        gx1_re[tid] += grad1_re[0]; gx1_im[tid] += grad1_im[0];
        gy1_re[tid] += grad1_re[1]; gy1_im[tid] += grad1_im[1];
        gz1_re[tid] += grad1_re[2]; gz1_im[tid] += grad1_im[2];
        gx2_re[tid] += grad2_re[0]; gx2_im[tid] += grad2_im[0];
        gy2_re[tid] += grad2_re[1]; gy2_im[tid] += grad2_im[1];
        gz2_re[tid] += grad2_re[2]; gz2_im[tid] += grad2_im[2];
        gx3_re[tid] += grad3_re[0]; gx3_im[tid] += grad3_im[0];
        gy3_re[tid] += grad3_re[1]; gy3_im[tid] += grad3_im[1];
        gz3_re[tid] += grad3_re[2]; gz3_im[tid] += grad3_im[2];
    }
}

template <typename ComputeReal>
__device__ __forceinline__ void accumulate_vector_values(
    ComputeReal dx,
    ComputeReal dy,
    ComputeReal dz,
    const ComputeReal unit[3],
    ComputeReal gradient_green_re,
    ComputeReal gradient_green_im,
    ComputeReal ga_re,
    ComputeReal ga_im,
    ComputeReal gb_re,
    ComputeReal gb_im,
    const ComputeReal qr[3],
    const ComputeReal qi[3],
    ComputeReal curl_acc_re[3],
    ComputeReal curl_acc_im[3],
    ComputeReal action_acc_re[3],
    ComputeReal action_acc_im[3])
{
    ComputeReal gradient_q_re[3];
    ComputeReal gradient_q_im[3];
#pragma unroll
    for (int source = 0; source < 3; source++) {
        gradient_q_re[source] =
            gradient_green_re * qr[source] -
            gradient_green_im * qi[source];
        gradient_q_im[source] =
            gradient_green_re * qi[source] +
            gradient_green_im * qr[source];
    }
    curl_acc_re[0] +=
        dy * gradient_q_re[0] - dx * gradient_q_re[1];
    curl_acc_im[0] +=
        dy * gradient_q_im[0] - dx * gradient_q_im[1];
    curl_acc_re[1] +=
        dz * gradient_q_re[0] - dx * gradient_q_re[2];
    curl_acc_im[1] +=
        dz * gradient_q_im[0] - dx * gradient_q_im[2];
    curl_acc_re[2] +=
        dz * gradient_q_re[1] - dy * gradient_q_re[2];
    curl_acc_im[2] +=
        dz * gradient_q_im[1] - dy * gradient_q_im[2];

    ComputeReal dot_q_re = 0;
    ComputeReal dot_q_im = 0;
#pragma unroll
    for (int source = 0; source < 3; source++) {
        dot_q_re += unit[source] * qr[source];
        dot_q_im += unit[source] * qi[source];
    }
    const ComputeReal isotropic_re =
        static_cast<ComputeReal>(2.0) * gb_re - ga_re;
    const ComputeReal isotropic_im =
        static_cast<ComputeReal>(2.0) * gb_im - ga_im;
    const ComputeReal radial_re =
        ga_re * dot_q_re - ga_im * dot_q_im;
    const ComputeReal radial_im =
        ga_re * dot_q_im + ga_im * dot_q_re;
#pragma unroll
    for (int component = 0; component < 3; component++) {
        action_acc_re[component] +=
            unit[component] * radial_re +
            isotropic_re * qr[component] -
            isotropic_im * qi[component];
        action_acc_im[component] +=
            unit[component] * radial_im +
            isotropic_re * qi[component] +
            isotropic_im * qr[component];
    }
}

template <typename ComputeReal, typename ChargeReal>
__device__ __forceinline__ void accumulate_vector_source_action(
    ComputeReal dx,
    ComputeReal dy,
    ComputeReal dz,
    const ComputeReal unit[3],
    ComputeReal gradient_green_re,
    ComputeReal gradient_green_im,
    ComputeReal ga_re,
    ComputeReal ga_im,
    ComputeReal gb_re,
    ComputeReal gb_im,
    const ChargeReal* __restrict__ qx_re,
    const ChargeReal* __restrict__ qx_im,
    const ChargeReal* __restrict__ qy_re,
    const ChargeReal* __restrict__ qy_im,
    const ChargeReal* __restrict__ qz_re,
    const ChargeReal* __restrict__ qz_im,
    int sid,
    ComputeReal curl_acc_re[3],
    ComputeReal curl_acc_im[3],
    ComputeReal action_acc_re[3],
    ComputeReal action_acc_im[3])
{
    const ComputeReal qr[3] = {
        static_cast<ComputeReal>(qx_re[sid]),
        static_cast<ComputeReal>(qy_re[sid]),
        static_cast<ComputeReal>(qz_re[sid])
    };
    const ComputeReal qi[3] = {
        static_cast<ComputeReal>(qx_im[sid]),
        static_cast<ComputeReal>(qy_im[sid]),
        static_cast<ComputeReal>(qz_im[sid])
    };
    accumulate_vector_values(
        dx, dy, dz, unit,
        gradient_green_re, gradient_green_im,
        ga_re, ga_im, gb_re, gb_im,
        qr, qi,
        curl_acc_re, curl_acc_im,
        action_acc_re, action_acc_im);
}

template <typename ComputeReal>
__global__ void p2p_vector_actions_batch3_leaf_kernel(
    const double* __restrict__ tgt_xyz,
    const double* __restrict__ src_xyz,
    const double* __restrict__ qx_re,
    const double* __restrict__ qx_im,
    const double* __restrict__ qy_re,
    const double* __restrict__ qy_im,
    const double* __restrict__ qz_re,
    const double* __restrict__ qz_im,
    const int* __restrict__ tgt_offsets,
    const int* __restrict__ tgt_ids,
    const int* __restrict__ src_offsets,
    const int* __restrict__ src_ids,
    const int* __restrict__ near_offsets,
    const int* __restrict__ near_leaf_ids,
    const int* __restrict__ near_source_offsets,
    const int* __restrict__ near_source_ids,
    int n_leaves, int leaf_split, double k_re, double k_im,
    double* __restrict__ curl_re,
    double* __restrict__ curl_im,
    double* __restrict__ hessian_action_re,
    double* __restrict__ hessian_action_im)
{
    const int leaf = blockIdx.x / leaf_split;
    const int leaf_part = blockIdx.x - leaf * leaf_split;
    if (leaf >= n_leaves)
        return;
    const ComputeReal real_k = static_cast<ComputeReal>(k_re);
    const ComputeReal imaginary_k = static_cast<ComputeReal>(k_im);
    const ComputeReal inv4pi =
        static_cast<ComputeReal>(0.07957747154594767);
    const ComputeReal k2_re =
        real_k * real_k - imaginary_k * imaginary_k;
    const ComputeReal k2_im =
        static_cast<ComputeReal>(2.0) * real_k * imaginary_k;

    for (int ti = tgt_offsets[leaf] + threadIdx.x;
         ti < tgt_offsets[leaf + 1]; ti += blockDim.x) {
        const int tid = tgt_ids[ti];
        const ComputeReal tx =
            static_cast<ComputeReal>(tgt_xyz[3 * tid]);
        const ComputeReal ty =
            static_cast<ComputeReal>(tgt_xyz[3 * tid + 1]);
        const ComputeReal tz =
            static_cast<ComputeReal>(tgt_xyz[3 * tid + 2]);
        ComputeReal curl_acc_re[3] = {0, 0, 0};
        ComputeReal curl_acc_im[3] = {0, 0, 0};
        ComputeReal action_acc_re[3] = {0, 0, 0};
        ComputeReal action_acc_im[3] = {0, 0, 0};

        const bool flat_sources =
            near_source_offsets != nullptr &&
            near_source_ids != nullptr;
        const int near_begin = near_offsets[leaf];
        const int near_end = near_offsets[leaf + 1];
        const int pass_count = 1 + near_end - near_begin;
        int pass_index = leaf_part;
        int source_index = 0;
        int source_end = 0;
        const int* selected_source_ids = src_ids;
        if (flat_sources) {
            const int flat_begin = near_source_offsets[leaf];
            const int flat_count =
                near_source_offsets[leaf + 1] - flat_begin;
            source_index = flat_begin +
                static_cast<int>(
                    static_cast<long long>(flat_count) *
                    leaf_part / leaf_split);
            source_end = flat_begin +
                static_cast<int>(
                    static_cast<long long>(flat_count) *
                    (leaf_part + 1) / leaf_split);
            selected_source_ids = near_source_ids;
        } else if (pass_index < pass_count) {
            const int source_leaf =
                pass_index == 0
                    ? leaf
                    : near_leaf_ids[near_begin + pass_index - 1];
            source_index = src_offsets[source_leaf];
            source_end = src_offsets[source_leaf + 1];
        }
        while (true) {
            if (source_index >= source_end) {
                if (flat_sources)
                    break;
                pass_index += leaf_split;
                if (pass_index >= pass_count)
                    break;
                const int source_leaf =
                    pass_index == 0
                        ? leaf
                        : near_leaf_ids[
                            near_begin + pass_index - 1];
                source_index = src_offsets[source_leaf];
                source_end = src_offsets[source_leaf + 1];
                continue;
            }
                const int sid =
                    selected_source_ids[source_index++];
                const ComputeReal dx = tx -
                    static_cast<ComputeReal>(src_xyz[3 * sid]);
                const ComputeReal dy = ty -
                    static_cast<ComputeReal>(src_xyz[3 * sid + 1]);
                const ComputeReal dz = tz -
                    static_cast<ComputeReal>(src_xyz[3 * sid + 2]);
                const ComputeReal radius_squared =
                    dx * dx + dy * dy + dz * dz;
                if (radius_squared <
                    static_cast<ComputeReal>(1.0e-24))
                    continue;
                const ComputeReal inv_r =
                    p2p_inverse_sqrt(radius_squared);
                const ComputeReal radius = radius_squared * inv_r;
                const ComputeReal inv_r2 = inv_r * inv_r;
                const ComputeReal attenuation =
                    imaginary_k == static_cast<ComputeReal>(0)
                        ? static_cast<ComputeReal>(1)
                        : p2p_attenuation(-imaginary_k * radius);
                const ComputeReal phase = real_k * radius;
                ComputeReal sine = 0;
                ComputeReal cosine = 0;
                p2p_sine_cosine(phase, &sine, &cosine);
                const ComputeReal green_re =
                    attenuation * cosine * inv4pi * inv_r;
                const ComputeReal green_im =
                    attenuation * sine * inv4pi * inv_r;
                const ComputeReal gradient_factor_re =
                    (-imaginary_k - inv_r) * inv_r;
                const ComputeReal gradient_factor_im = real_k * inv_r;
                const ComputeReal gradient_green_re =
                    green_re * gradient_factor_re -
                    green_im * gradient_factor_im;
                const ComputeReal gradient_green_im =
                    green_re * gradient_factor_im +
                    green_im * gradient_factor_re;
                const ComputeReal a_re =
                    static_cast<ComputeReal>(3.0) * inv_r2 +
                    static_cast<ComputeReal>(3.0) *
                        imaginary_k * inv_r - k2_re;
                const ComputeReal a_im =
                    -static_cast<ComputeReal>(3.0) *
                        real_k * inv_r - k2_im;
                const ComputeReal b_re =
                    inv_r2 + imaginary_k * inv_r;
                const ComputeReal b_im = -real_k * inv_r;
                const ComputeReal ga_re =
                    green_re * a_re - green_im * a_im;
                const ComputeReal ga_im =
                    green_re * a_im + green_im * a_re;
                const ComputeReal gb_re =
                    green_re * b_re - green_im * b_im;
                const ComputeReal gb_im =
                    green_re * b_im + green_im * b_re;
                const ComputeReal unit[3] = {
                    dx * inv_r, dy * inv_r, dz * inv_r
                };
                accumulate_vector_source_action(
                    dx, dy, dz, unit,
                    gradient_green_re, gradient_green_im,
                    ga_re, ga_im, gb_re, gb_im,
                    qx_re, qx_im, qy_re, qy_im, qz_re, qz_im,
                    sid,
                    curl_acc_re, curl_acc_im,
                    action_acc_re, action_acc_im);
        }
        for (int component = 0; component < 3; component++) {
            const int offset = 3 * tid + component;
            if (leaf_split == 1) {
                curl_re[offset] += curl_acc_re[component];
                curl_im[offset] += curl_acc_im[component];
                hessian_action_re[offset] +=
                    action_acc_re[component];
                hessian_action_im[offset] +=
                    action_acc_im[component];
            } else {
                atomicAdd(
                    curl_re + offset,
                    static_cast<double>(curl_acc_re[component]));
                atomicAdd(
                    curl_im + offset,
                    static_cast<double>(curl_acc_im[component]));
                atomicAdd(
                    hessian_action_re + offset,
                    static_cast<double>(action_acc_re[component]));
                atomicAdd(
                    hessian_action_im + offset,
                    static_cast<double>(action_acc_im[component]));
            }
        }
    }
}

template <
    typename ComputeReal,
    typename CoordinateReal,
    typename ChargeReal,
    bool RealWaveNumber,
    bool FastTrig>
__global__ void p2p_vector_actions_pair_batch3_leaf_kernel(
    const CoordinateReal* __restrict__ tgt_xyz,
    const CoordinateReal* __restrict__ src_xyz,
    const ChargeReal* __restrict__ first_x_re,
    const ChargeReal* __restrict__ first_x_im,
    const ChargeReal* __restrict__ first_y_re,
    const ChargeReal* __restrict__ first_y_im,
    const ChargeReal* __restrict__ first_z_re,
    const ChargeReal* __restrict__ first_z_im,
    const ChargeReal* __restrict__ second_x_re,
    const ChargeReal* __restrict__ second_x_im,
    const ChargeReal* __restrict__ second_y_re,
    const ChargeReal* __restrict__ second_y_im,
    const ChargeReal* __restrict__ second_z_re,
    const ChargeReal* __restrict__ second_z_im,
    const int* __restrict__ tgt_offsets,
    const int* __restrict__ tgt_ids,
    const int* __restrict__ src_offsets,
    const int* __restrict__ src_ids,
    const int* __restrict__ near_offsets,
    const int* __restrict__ near_leaf_ids,
    const int* __restrict__ near_source_offsets,
    const int* __restrict__ near_source_ids,
    int n_leaves, int leaf_split, double k_re, double k_im,
    double* __restrict__ first_curl_re,
    double* __restrict__ first_curl_im,
    double* __restrict__ first_hessian_action_re,
    double* __restrict__ first_hessian_action_im,
    double* __restrict__ second_curl_re,
    double* __restrict__ second_curl_im,
    double* __restrict__ second_hessian_action_re,
    double* __restrict__ second_hessian_action_im)
{
    const int leaf = blockIdx.x / leaf_split;
    const int leaf_part = blockIdx.x - leaf * leaf_split;
    if (leaf >= n_leaves)
        return;
    const ComputeReal real_k = static_cast<ComputeReal>(k_re);
    const ComputeReal imaginary_k = static_cast<ComputeReal>(k_im);
    const ComputeReal inv4pi =
        static_cast<ComputeReal>(0.07957747154594767);
    const ComputeReal k2_re =
        real_k * real_k - imaginary_k * imaginary_k;
    const ComputeReal k2_im =
        static_cast<ComputeReal>(2.0) * real_k * imaginary_k;

    for (int ti = tgt_offsets[leaf] + threadIdx.x;
         ti < tgt_offsets[leaf + 1]; ti += blockDim.x) {
        const int tid = tgt_ids[ti];
        const ComputeReal tx =
            static_cast<ComputeReal>(tgt_xyz[3 * tid]);
        const ComputeReal ty =
            static_cast<ComputeReal>(tgt_xyz[3 * tid + 1]);
        const ComputeReal tz =
            static_cast<ComputeReal>(tgt_xyz[3 * tid + 2]);
        ComputeReal first_curl_acc_re[3] = {0, 0, 0};
        ComputeReal first_curl_acc_im[3] = {0, 0, 0};
        ComputeReal first_action_acc_re[3] = {0, 0, 0};
        ComputeReal first_action_acc_im[3] = {0, 0, 0};
        ComputeReal second_curl_acc_re[3] = {0, 0, 0};
        ComputeReal second_curl_acc_im[3] = {0, 0, 0};
        ComputeReal second_action_acc_re[3] = {0, 0, 0};
        ComputeReal second_action_acc_im[3] = {0, 0, 0};

        const bool flat_sources =
            near_source_offsets != nullptr &&
            near_source_ids != nullptr;
        const int near_begin = near_offsets[leaf];
        const int near_end = near_offsets[leaf + 1];
        const int pass_count = 1 + near_end - near_begin;
        int pass_index = leaf_part;
        int source_index = 0;
        int source_end = 0;
        const int* selected_source_ids = src_ids;
        if (flat_sources) {
            const int flat_begin = near_source_offsets[leaf];
            const int flat_count =
                near_source_offsets[leaf + 1] - flat_begin;
            source_index = flat_begin +
                static_cast<int>(
                    static_cast<long long>(flat_count) *
                    leaf_part / leaf_split);
            source_end = flat_begin +
                static_cast<int>(
                    static_cast<long long>(flat_count) *
                    (leaf_part + 1) / leaf_split);
            selected_source_ids = near_source_ids;
        } else if (pass_index < pass_count) {
            const int source_leaf =
                pass_index == 0
                    ? leaf
                    : near_leaf_ids[near_begin + pass_index - 1];
            source_index = src_offsets[source_leaf];
            source_end = src_offsets[source_leaf + 1];
        }
        while (true) {
            if (source_index >= source_end) {
                if (flat_sources)
                    break;
                pass_index += leaf_split;
                if (pass_index >= pass_count)
                    break;
                const int source_leaf =
                    pass_index == 0
                        ? leaf
                        : near_leaf_ids[
                            near_begin + pass_index - 1];
                source_index = src_offsets[source_leaf];
                source_end = src_offsets[source_leaf + 1];
                continue;
            }
                const int sid =
                    selected_source_ids[source_index++];
                const ComputeReal dx = tx -
                    static_cast<ComputeReal>(src_xyz[3 * sid]);
                const ComputeReal dy = ty -
                    static_cast<ComputeReal>(src_xyz[3 * sid + 1]);
                const ComputeReal dz = tz -
                    static_cast<ComputeReal>(src_xyz[3 * sid + 2]);
                const ComputeReal radius_squared =
                    dx * dx + dy * dy + dz * dz;
                if (radius_squared <
                    static_cast<ComputeReal>(1.0e-24))
                    continue;
                const ComputeReal inv_r =
                    p2p_inverse_sqrt(radius_squared);
                const ComputeReal radius = radius_squared * inv_r;
                const ComputeReal inv_r2 = inv_r * inv_r;
                const ComputeReal attenuation = RealWaveNumber
                    ? static_cast<ComputeReal>(1)
                    : p2p_attenuation(-imaginary_k * radius);
                const ComputeReal phase = real_k * radius;
                ComputeReal sine = 0;
                ComputeReal cosine = 0;
                p2p_pair_sine_cosine<FastTrig>(
                    phase, &sine, &cosine);
                const ComputeReal green_re =
                    attenuation * cosine * inv4pi * inv_r;
                const ComputeReal green_im =
                    attenuation * sine * inv4pi * inv_r;
                const ComputeReal gradient_factor_re = RealWaveNumber
                    ? -inv_r2
                    : (-imaginary_k - inv_r) * inv_r;
                const ComputeReal gradient_factor_im = real_k * inv_r;
                const ComputeReal gradient_green_re =
                    green_re * gradient_factor_re -
                    green_im * gradient_factor_im;
                const ComputeReal gradient_green_im =
                    green_re * gradient_factor_im +
                    green_im * gradient_factor_re;
                const ComputeReal a_re = RealWaveNumber
                    ? static_cast<ComputeReal>(3.0) * inv_r2 - k2_re
                    : static_cast<ComputeReal>(3.0) * inv_r2 +
                        static_cast<ComputeReal>(3.0) *
                            imaginary_k * inv_r - k2_re;
                const ComputeReal a_im =
                    -static_cast<ComputeReal>(3.0) *
                        real_k * inv_r - k2_im;
                const ComputeReal b_re = RealWaveNumber
                    ? inv_r2
                    : inv_r2 + imaginary_k * inv_r;
                const ComputeReal b_im = -real_k * inv_r;
                const ComputeReal ga_re =
                    green_re * a_re - green_im * a_im;
                const ComputeReal ga_im =
                    green_re * a_im + green_im * a_re;
                const ComputeReal gb_re =
                    green_re * b_re - green_im * b_im;
                const ComputeReal gb_im =
                    green_re * b_im + green_im * b_re;
                const ComputeReal unit[3] = {
                    dx * inv_r, dy * inv_r, dz * inv_r
                };
                accumulate_vector_source_action(
                    dx, dy, dz, unit,
                    gradient_green_re, gradient_green_im,
                    ga_re, ga_im, gb_re, gb_im,
                    first_x_re, first_x_im,
                    first_y_re, first_y_im,
                    first_z_re, first_z_im,
                    sid,
                    first_curl_acc_re, first_curl_acc_im,
                    first_action_acc_re, first_action_acc_im);
                accumulate_vector_source_action(
                    dx, dy, dz, unit,
                    gradient_green_re, gradient_green_im,
                    ga_re, ga_im, gb_re, gb_im,
                    second_x_re, second_x_im,
                    second_y_re, second_y_im,
                    second_z_re, second_z_im,
                    sid,
                    second_curl_acc_re, second_curl_acc_im,
                    second_action_acc_re, second_action_acc_im);
        }
        for (int component = 0; component < 3; component++) {
            const int offset = 3 * tid + component;
            if (leaf_split == 1) {
                first_curl_re[offset] += first_curl_acc_re[component];
                first_curl_im[offset] += first_curl_acc_im[component];
                first_hessian_action_re[offset] +=
                    first_action_acc_re[component];
                first_hessian_action_im[offset] +=
                    first_action_acc_im[component];
                second_curl_re[offset] += second_curl_acc_re[component];
                second_curl_im[offset] += second_curl_acc_im[component];
                second_hessian_action_re[offset] +=
                    second_action_acc_re[component];
                second_hessian_action_im[offset] +=
                    second_action_acc_im[component];
            } else {
                atomicAdd(
                    first_curl_re + offset,
                    static_cast<double>(first_curl_acc_re[component]));
                atomicAdd(
                    first_curl_im + offset,
                    static_cast<double>(first_curl_acc_im[component]));
                atomicAdd(
                    first_hessian_action_re + offset,
                    static_cast<double>(first_action_acc_re[component]));
                atomicAdd(
                    first_hessian_action_im + offset,
                    static_cast<double>(first_action_acc_im[component]));
                atomicAdd(
                    second_curl_re + offset,
                    static_cast<double>(second_curl_acc_re[component]));
                atomicAdd(
                    second_curl_im + offset,
                    static_cast<double>(second_curl_acc_im[component]));
                atomicAdd(
                    second_hessian_action_re + offset,
                    static_cast<double>(second_action_acc_re[component]));
                atomicAdd(
                    second_hessian_action_im + offset,
                    static_cast<double>(second_action_acc_im[component]));
            }
        }
    }
}

void launch_p2p_hessian_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_hess_re, double* d_hess_im)
{
    p2p_hessian_leaf_kernel<<<n_leaves, 128>>>(
        d_tgt, d_src, d_q_re, d_q_im,
        d_tgt_offsets, d_tgt_ids, d_src_offsets, d_src_ids,
        d_near_offsets, d_near_leaf_ids,
        n_leaves, k_re, k_im, d_hess_re, d_hess_im);
}

void launch_p2p_grad_hessian_batch3_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const double* d_q3_re, const double* d_q3_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_gx1_re, double* d_gx1_im,
    double* d_gy1_re, double* d_gy1_im,
    double* d_gz1_re, double* d_gz1_im,
    double* d_gx2_re, double* d_gx2_im,
    double* d_gy2_re, double* d_gy2_im,
    double* d_gz2_re, double* d_gz2_im,
    double* d_gx3_re, double* d_gx3_im,
    double* d_gy3_re, double* d_gy3_im,
    double* d_gz3_re, double* d_gz3_im,
    double* d_hess1_re, double* d_hess1_im,
    double* d_hess2_re, double* d_hess2_im,
    double* d_hess3_re, double* d_hess3_im,
    bool fp32_compute)
{
    if (fp32_compute) {
        p2p_grad_hessian_batch3_leaf_kernel<float><<<n_leaves, 128>>>(
            d_tgt, d_src,
            d_q1_re, d_q1_im,
            d_q2_re, d_q2_im,
            d_q3_re, d_q3_im,
            d_tgt_offsets, d_tgt_ids, d_src_offsets, d_src_ids,
            d_near_offsets, d_near_leaf_ids,
            n_leaves, k_re, k_im,
            d_gx1_re, d_gx1_im,
            d_gy1_re, d_gy1_im,
            d_gz1_re, d_gz1_im,
            d_gx2_re, d_gx2_im,
            d_gy2_re, d_gy2_im,
            d_gz2_re, d_gz2_im,
            d_gx3_re, d_gx3_im,
            d_gy3_re, d_gy3_im,
            d_gz3_re, d_gz3_im,
            d_hess1_re, d_hess1_im,
            d_hess2_re, d_hess2_im,
            d_hess3_re, d_hess3_im);
        return;
    }
    p2p_grad_hessian_batch3_leaf_kernel<double><<<n_leaves, 128>>>(
        d_tgt, d_src,
        d_q1_re, d_q1_im,
        d_q2_re, d_q2_im,
        d_q3_re, d_q3_im,
        d_tgt_offsets, d_tgt_ids, d_src_offsets, d_src_ids,
        d_near_offsets, d_near_leaf_ids,
        n_leaves, k_re, k_im,
        d_gx1_re, d_gx1_im,
        d_gy1_re, d_gy1_im,
        d_gz1_re, d_gz1_im,
        d_gx2_re, d_gx2_im,
        d_gy2_re, d_gy2_im,
        d_gz2_re, d_gz2_im,
        d_gx3_re, d_gx3_im,
        d_gy3_re, d_gy3_im,
        d_gz3_re, d_gz3_im,
        d_hess1_re, d_hess1_im,
        d_hess2_re, d_hess2_im,
        d_hess3_re, d_hess3_im);
}

void launch_p2p_vector_actions_batch3_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_qx_re, const double* d_qx_im,
    const double* d_qy_re, const double* d_qy_im,
    const double* d_qz_re, const double* d_qz_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    const int* d_near_source_offsets,
    const int* d_near_source_ids,
    int n_leaves, double k_re, double k_im,
    double* d_curl_re, double* d_curl_im,
    double* d_hessian_action_re, double* d_hessian_action_im,
    bool fp32_compute)
{
#ifdef BEM_DEFAULT_FMM_NEAR_FP32
    const int default_threads = 512;
#else
    const int default_threads = 128;
#endif
    int threads = bem_env_int(
        "BEM_FMM_P2P_THREADS", default_threads);
    if (threads != 64 && threads != 128 &&
        threads != 256 && threads != 512 && threads != 1024)
        threads = default_threads;
#ifdef BEM_DEFAULT_FMM_NEAR_FP32
    const int default_leaf_split =
        d_near_source_offsets != nullptr ? 32 : 8;
#else
    const int default_leaf_split = 1;
#endif
    int leaf_split = bem_env_int(
        "BEM_FMM_P2P_LEAF_SPLIT", default_leaf_split);
    if (leaf_split != 1 && leaf_split != 2 &&
        leaf_split != 4 && leaf_split != 8 &&
        leaf_split != 16 && leaf_split != 32 &&
        leaf_split != 64 && leaf_split != 128)
        leaf_split = 1;
    if (fp32_compute) {
        p2p_vector_actions_batch3_leaf_kernel<float>
            <<<n_leaves * leaf_split, threads>>>(
            d_tgt, d_src,
            d_qx_re, d_qx_im,
            d_qy_re, d_qy_im,
            d_qz_re, d_qz_im,
            d_tgt_offsets, d_tgt_ids, d_src_offsets, d_src_ids,
            d_near_offsets, d_near_leaf_ids,
            d_near_source_offsets, d_near_source_ids,
            n_leaves, leaf_split, k_re, k_im,
            d_curl_re, d_curl_im,
            d_hessian_action_re, d_hessian_action_im);
        return;
    }
    p2p_vector_actions_batch3_leaf_kernel<double>
        <<<n_leaves * leaf_split, threads>>>(
        d_tgt, d_src,
        d_qx_re, d_qx_im,
        d_qy_re, d_qy_im,
        d_qz_re, d_qz_im,
        d_tgt_offsets, d_tgt_ids, d_src_offsets, d_src_ids,
        d_near_offsets, d_near_leaf_ids,
        d_near_source_offsets, d_near_source_ids,
        n_leaves, leaf_split, k_re, k_im,
        d_curl_re, d_curl_im,
        d_hessian_action_re, d_hessian_action_im);
}

void launch_p2p_vector_actions_pair_batch3_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_first_x_re, const double* d_first_x_im,
    const double* d_first_y_re, const double* d_first_y_im,
    const double* d_first_z_re, const double* d_first_z_im,
    const double* d_second_x_re, const double* d_second_x_im,
    const double* d_second_y_re, const double* d_second_y_im,
    const double* d_second_z_re, const double* d_second_z_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    const int* d_near_source_offsets,
    const int* d_near_source_ids,
    int n_leaves, double k_re, double k_im,
    double* d_first_curl_re, double* d_first_curl_im,
    double* d_first_hessian_action_re,
    double* d_first_hessian_action_im,
    double* d_second_curl_re, double* d_second_curl_im,
    double* d_second_hessian_action_re,
    double* d_second_hessian_action_im,
    const float* d_tgt_fp32, const float* d_src_fp32,
    const float* d_packed_charges_fp32, int source_count,
    bool fp32_compute)
{
#ifdef BEM_DEFAULT_FMM_NEAR_FP32
    const int default_threads = 512;
#else
    const int default_threads = 128;
#endif
    int threads = bem_env_int(
        "BEM_FMM_P2P_PAIR_THREADS", default_threads);
    if (threads != 64 && threads != 128 &&
        threads != 256 && threads != 512)
        threads = default_threads;
#ifdef BEM_DEFAULT_FMM_NEAR_FP32
    const int default_leaf_split =
        d_near_source_offsets != nullptr ? 32 : 8;
#else
    const int default_leaf_split = 1;
#endif
    int leaf_split = bem_env_int(
        "BEM_FMM_P2P_LEAF_SPLIT", default_leaf_split);
    if (leaf_split != 1 && leaf_split != 2 &&
        leaf_split != 4 && leaf_split != 8 &&
        leaf_split != 16 && leaf_split != 32 &&
        leaf_split != 64 && leaf_split != 128)
        leaf_split = 1;
#define LAUNCH_PAIR_KERNEL( \
    TYPE, COORD_TYPE, CHARGE_TYPE, REAL_K, FAST_TRIG, TARGETS, SOURCES, \
    FIRST_X_RE, FIRST_X_IM, FIRST_Y_RE, FIRST_Y_IM, \
    FIRST_Z_RE, FIRST_Z_IM, SECOND_X_RE, SECOND_X_IM, \
    SECOND_Y_RE, SECOND_Y_IM, SECOND_Z_RE, SECOND_Z_IM) \
    p2p_vector_actions_pair_batch3_leaf_kernel< \
        TYPE, COORD_TYPE, CHARGE_TYPE, REAL_K, FAST_TRIG> \
        <<<n_leaves * leaf_split, threads>>>( \
            TARGETS, SOURCES, \
            FIRST_X_RE, FIRST_X_IM, \
            FIRST_Y_RE, FIRST_Y_IM, \
            FIRST_Z_RE, FIRST_Z_IM, \
            SECOND_X_RE, SECOND_X_IM, \
            SECOND_Y_RE, SECOND_Y_IM, \
            SECOND_Z_RE, SECOND_Z_IM, \
            d_tgt_offsets, d_tgt_ids, \
            d_src_offsets, d_src_ids, \
            d_near_offsets, d_near_leaf_ids, \
            d_near_source_offsets, d_near_source_ids, \
            n_leaves, leaf_split, k_re, k_im, \
            d_first_curl_re, d_first_curl_im, \
            d_first_hessian_action_re, \
            d_first_hessian_action_im, \
            d_second_curl_re, d_second_curl_im, \
            d_second_hessian_action_re, \
            d_second_hessian_action_im)
    const bool fast_trig =
        fp32_compute &&
        bem_env_flag_enabled("BEM_FMM_P2P_FAST_TRIG", true);
    const bool real_wave_number = k_im == 0.0;
    if (fp32_compute && d_tgt_fp32 != nullptr &&
        d_src_fp32 != nullptr &&
        d_packed_charges_fp32 != nullptr &&
        source_count > 0) {
        if (real_wave_number && fast_trig) {
            LAUNCH_PAIR_KERNEL(
                float, float, float, true, true,
                d_tgt_fp32, d_src_fp32,
                d_packed_charges_fp32,
                d_packed_charges_fp32 + source_count,
                d_packed_charges_fp32 + 2 * source_count,
                d_packed_charges_fp32 + 3 * source_count,
                d_packed_charges_fp32 + 4 * source_count,
                d_packed_charges_fp32 + 5 * source_count,
                d_packed_charges_fp32 + 6 * source_count,
                d_packed_charges_fp32 + 7 * source_count,
                d_packed_charges_fp32 + 8 * source_count,
                d_packed_charges_fp32 + 9 * source_count,
                d_packed_charges_fp32 + 10 * source_count,
                d_packed_charges_fp32 + 11 * source_count);
        } else if (real_wave_number) {
            LAUNCH_PAIR_KERNEL(
                float, float, float, true, false,
                d_tgt_fp32, d_src_fp32,
                d_packed_charges_fp32,
                d_packed_charges_fp32 + source_count,
                d_packed_charges_fp32 + 2 * source_count,
                d_packed_charges_fp32 + 3 * source_count,
                d_packed_charges_fp32 + 4 * source_count,
                d_packed_charges_fp32 + 5 * source_count,
                d_packed_charges_fp32 + 6 * source_count,
                d_packed_charges_fp32 + 7 * source_count,
                d_packed_charges_fp32 + 8 * source_count,
                d_packed_charges_fp32 + 9 * source_count,
                d_packed_charges_fp32 + 10 * source_count,
                d_packed_charges_fp32 + 11 * source_count);
        } else {
            LAUNCH_PAIR_KERNEL(
                float, float, float, false, false,
                d_tgt_fp32, d_src_fp32,
                d_packed_charges_fp32,
                d_packed_charges_fp32 + source_count,
                d_packed_charges_fp32 + 2 * source_count,
                d_packed_charges_fp32 + 3 * source_count,
                d_packed_charges_fp32 + 4 * source_count,
                d_packed_charges_fp32 + 5 * source_count,
                d_packed_charges_fp32 + 6 * source_count,
                d_packed_charges_fp32 + 7 * source_count,
                d_packed_charges_fp32 + 8 * source_count,
                d_packed_charges_fp32 + 9 * source_count,
                d_packed_charges_fp32 + 10 * source_count,
                d_packed_charges_fp32 + 11 * source_count);
        }
    } else if (fp32_compute) {
        LAUNCH_PAIR_KERNEL(
            float, double, double, false, false, d_tgt, d_src,
            d_first_x_re, d_first_x_im,
            d_first_y_re, d_first_y_im,
            d_first_z_re, d_first_z_im,
            d_second_x_re, d_second_x_im,
            d_second_y_re, d_second_y_im,
            d_second_z_re, d_second_z_im);
    } else {
        LAUNCH_PAIR_KERNEL(
            double, double, double, false, false, d_tgt, d_src,
            d_first_x_re, d_first_x_im,
            d_first_y_re, d_first_y_im,
            d_first_z_re, d_first_z_im,
            d_second_x_re, d_second_x_im,
            d_second_y_re, d_second_y_im,
            d_second_z_re, d_second_z_im);
    }
#undef LAUNCH_PAIR_KERNEL
}

void launch_p2p_pot_grad_batch2_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_pot1_re, double* d_pot1_im,
    double* d_gx1_re, double* d_gx1_im,
    double* d_gy1_re, double* d_gy1_im,
    double* d_gz1_re, double* d_gz1_im,
    double* d_pot2_re, double* d_pot2_im,
    double* d_gx2_re, double* d_gx2_im,
    double* d_gy2_re, double* d_gy2_im,
    double* d_gz2_re, double* d_gz2_im)
{
    p2p_leaf_kernel<2, true><<<n_leaves, 128>>>(
        d_tgt, d_src, d_q1_re, d_q1_im, d_q2_re, d_q2_im,
        d_tgt_offsets, d_tgt_ids, d_src_offsets, d_src_ids,
        d_near_offsets, d_near_leaf_ids, n_leaves, k_re, k_im,
        d_pot1_re, d_pot1_im, d_gx1_re, d_gx1_im, d_gy1_re, d_gy1_im, d_gz1_re, d_gz1_im,
        d_pot2_re, d_pot2_im, d_gx2_re, d_gx2_im, d_gy2_re, d_gy2_im, d_gz2_re, d_gz2_im);
}

void launch_p2p_pot_grad_batch4_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const double* d_q3_re, const double* d_q3_im,
    const double* d_q4_re, const double* d_q4_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
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
    p2p_leaf_batch4_kernel<true><<<n_leaves, 128>>>(
        d_tgt, d_src,
        d_q1_re, d_q1_im, d_q2_re, d_q2_im, d_q3_re, d_q3_im, d_q4_re, d_q4_im,
        d_tgt_offsets, d_tgt_ids, d_src_offsets, d_src_ids,
        d_near_offsets, d_near_leaf_ids, n_leaves, k_re, k_im,
        d_pot1_re, d_pot1_im, d_gx1_re, d_gx1_im, d_gy1_re, d_gy1_im, d_gz1_re, d_gz1_im,
        d_pot2_re, d_pot2_im, d_gx2_re, d_gx2_im, d_gy2_re, d_gy2_im, d_gz2_re, d_gz2_im,
        d_pot3_re, d_pot3_im, d_gx3_re, d_gx3_im, d_gy3_re, d_gy3_im, d_gz3_re, d_gz3_im,
        d_pot4_re, d_pot4_im, d_gx4_re, d_gx4_im, d_gy4_re, d_gy4_im, d_gz4_re, d_gz4_im);
}

void launch_p2p_pot_grad_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_pot_re, double* d_pot_im,
    double* d_gx_re, double* d_gx_im,
    double* d_gy_re, double* d_gy_im,
    double* d_gz_re, double* d_gz_im)
{
    p2p_leaf_kernel<1, true><<<n_leaves, 128>>>(
        d_tgt, d_src, d_q_re, d_q_im, nullptr, nullptr,
        d_tgt_offsets, d_tgt_ids, d_src_offsets, d_src_ids,
        d_near_offsets, d_near_leaf_ids, n_leaves, k_re, k_im,
        d_pot_re, d_pot_im, d_gx_re, d_gx_im, d_gy_re, d_gy_im, d_gz_re, d_gz_im,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}
