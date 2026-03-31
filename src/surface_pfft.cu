// Surface pFFT: 2D FFT per flat face + inter-face P2P
// For hex prisms: 8 faces (top, bottom, 6 sides), each coplanar
// Intra-face: 2D Toeplitz -> circulant -> FFT (FP64)
// Inter-face: direct P2P on GPU (FP32 for throughput)

#include "surface_pfft.h"
#include <cufft.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>

#define CUFFT_CHECK(call) do { \
    cufftResult err = (call); \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error at %s:%d: %d\n", __FILE__, __LINE__, (int)err); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// CUDA kernels
// ============================================================================

// 2D anterpolate: scatter charges to FP32 staging grid (fast FP32 atomicAdd)
__global__ void kernel_anterpolate_2d_f32(
    const double* __restrict__ q_re,
    const double* __restrict__ q_im,
    const int*    __restrict__ stencil_idx,
    const double* __restrict__ stencil_wt,
    int N, int stencil_size,
    float* __restrict__ grid_re,
    float* __restrict__ grid_im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float qr = (float)q_re[i], qi = (float)q_im[i];
    const int*    idx = stencil_idx + (long long)i * stencil_size;
    const double* wt  = stencil_wt  + (long long)i * stencil_size;

    for (int s = 0; s < stencil_size; s++) {
        float w = (float)wt[s];
        int gi = idx[s];
        atomicAdd(&grid_re[gi], w * qr);
        atomicAdd(&grid_im[gi], w * qi);
    }
}

// Convert FP32 staging grid to FP32 cuFFT complex grid
__global__ void kernel_f32_to_c2c(
    const float* __restrict__ re,
    const float* __restrict__ im,
    cufftComplex* __restrict__ out,
    long long N)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i].x = re[i];
    out[i].y = im[i];
}

// 2D interpolate: gather from FP32 grid to FP64 output
__global__ void kernel_interpolate_2d(
    const cufftComplex* __restrict__ grid,
    const int*    __restrict__ stencil_idx,
    const double* __restrict__ stencil_wt,
    int N, int stencil_size,
    double* __restrict__ out_re,
    double* __restrict__ out_im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int*    idx = stencil_idx + (long long)i * stencil_size;
    const double* wt  = stencil_wt  + (long long)i * stencil_size;

    double vr = 0.0, vi = 0.0;
    for (int s = 0; s < stencil_size; s++) {
        double w = wt[s];
        cufftComplex g = grid[idx[s]];
        vr += w * (double)g.x;
        vi += w * (double)g.y;
    }
    out_re[i] = vr;
    out_im[i] = vi;
}

// Pointwise complex multiply (FP32)
__global__ void kernel_pw_mul_2d(
    const cufftComplex* __restrict__ a,
    const cufftComplex* __restrict__ b,
    cufftComplex* __restrict__ c,
    long long N, float scale)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float ar = a[i].x, ai = a[i].y;
    float br = b[i].x, bi = b[i].y;
    c[i].x = (ar * br - ai * bi) * scale;
    c[i].y = (ar * bi + ai * br) * scale;
}

// Near-field correction (CSR)
__global__ void kernel_near_corr_2d(
    const int*    __restrict__ row_ptr,
    const int*    __restrict__ col_idx,
    const double* __restrict__ corr_re,
    const double* __restrict__ corr_im,
    const double* __restrict__ q_re,
    const double* __restrict__ q_im,
    double* __restrict__ out_re,
    double* __restrict__ out_im,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double vr = 0.0, vi = 0.0;
    for (int p = row_ptr[i]; p < row_ptr[i+1]; p++) {
        int j = col_idx[p];
        double cr = corr_re[p], ci = corr_im[p];
        double jr = q_re[j], ji = q_im[j];
        vr += cr * jr - ci * ji;
        vi += cr * ji + ci * jr;
    }
    out_re[i] += vr;
    out_im[i] += vi;
}

// Zero a 2D grid (FP32)
__global__ void kernel_zero_2d(cufftComplex* grid, long long N)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    grid[i].x = 0.0f;
    grid[i].y = 0.0f;
}

#define P2P_TILE 256

// Macro for P2P Green's function inner loop body (shared by pot and pot_grad)
#define P2P_GREEN_BODY(j) \
    float dx = tx - s_px[j]; \
    float dy = ty - s_py[j]; \
    float dz = tz - s_pz[j]; \
    float R2 = dx*dx + dy*dy + dz*dz; \
    if (R2 < 1e-30f) continue; \
    float invR = rsqrtf(R2); \
    float R = R2 * invR; \
    float kR_re = k_re * R; \
    float kR_im = k_im * R; \
    float edr = expf(-kR_im); \
    float cs, sn; \
    sincosf(kR_re, &sn, &cs); \
    float G_re = edr * cs * inv4pi * invR; \
    float G_im = edr * sn * inv4pi * invR; \
    float jr = s_qr[j], ji = s_qi[j]; \
    acc_re += G_re * jr - G_im * ji; \
    acc_im += G_re * ji + G_im * jr;

// Unified inter-face P2P: potential only (FP32, tiled, single kernel for all faces)
// Each thread processes one target point, loops over ALL non-same-face sources
__global__ void kernel_p2p_unified_pot_f32(
    const float* __restrict__ pts,
    const float* __restrict__ q_re,
    const float* __restrict__ q_im,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    int N,
    const int* __restrict__ face_offsets,
    int n_faces,
    float k_re, float k_im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float tx, ty, tz;
    int my_start = 0, my_end = 0;
    if (i < N) {
        tx = pts[i*3]; ty = pts[i*3+1]; tz = pts[i*3+2];
        for (int f = 0; f < n_faces; f++) {
            if (i < face_offsets[f+1]) {
                my_start = face_offsets[f];
                my_end = face_offsets[f+1];
                break;
            }
        }
    }

    float acc_re = 0.0f, acc_im = 0.0f;
    float inv4pi = 0.07957747154594767f;

    __shared__ float s_px[P2P_TILE], s_py[P2P_TILE], s_pz[P2P_TILE];
    __shared__ float s_qr[P2P_TILE], s_qi[P2P_TILE];

    for (int tile = 0; tile < N; tile += P2P_TILE) {
        int tile_size = min(P2P_TILE, N - tile);
        if ((int)threadIdx.x < tile_size) {
            int j = tile + threadIdx.x;
            s_px[threadIdx.x] = pts[j*3+0];
            s_py[threadIdx.x] = pts[j*3+1];
            s_pz[threadIdx.x] = pts[j*3+2];
            s_qr[threadIdx.x] = q_re[j];
            s_qi[threadIdx.x] = q_im[j];
        }
        __syncthreads();

        if (i < N) {
            int tile_end_idx = tile + tile_size;
            bool tile_overlaps = (tile_end_idx > my_start && tile < my_end);

            if (!tile_overlaps) {
                // Fast path: no same-face sources in this tile
                for (int j = 0; j < tile_size; j++) {
                    P2P_GREEN_BODY(j)
                }
            } else {
                // Slow path: skip same-face sources
                for (int j = 0; j < tile_size; j++) {
                    int gj = tile + j;
                    if (gj >= my_start && gj < my_end) continue;
                    P2P_GREEN_BODY(j)
                }
            }
        }
        __syncthreads();
    }
    if (i < N) {
        out_re[i] += acc_re;
        out_im[i] += acc_im;
    }
}

// Unified inter-face P2P: potential + gradient (FP32, tiled, single kernel)
__global__ void kernel_p2p_unified_pot_grad_f32(
    const float* __restrict__ pts,
    const float* __restrict__ q_re,
    const float* __restrict__ q_im,
    float* __restrict__ pot_re,
    float* __restrict__ pot_im,
    float* __restrict__ grad_re,
    float* __restrict__ grad_im,
    int N,
    const int* __restrict__ face_offsets,
    int n_faces,
    float k_re, float k_im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float tx, ty, tz;
    int my_start = 0, my_end = 0;
    if (i < N) {
        tx = pts[i*3]; ty = pts[i*3+1]; tz = pts[i*3+2];
        for (int f = 0; f < n_faces; f++) {
            if (i < face_offsets[f+1]) {
                my_start = face_offsets[f];
                my_end = face_offsets[f+1];
                break;
            }
        }
    }

    float acc_re = 0.0f, acc_im = 0.0f;
    float gx_re = 0.0f, gx_im = 0.0f;
    float gy_re = 0.0f, gy_im = 0.0f;
    float gz_re = 0.0f, gz_im = 0.0f;
    float inv4pi = 0.07957747154594767f;

    __shared__ float s_px[P2P_TILE], s_py[P2P_TILE], s_pz[P2P_TILE];
    __shared__ float s_qr[P2P_TILE], s_qi[P2P_TILE];

    #define P2P_GRAD_BODY(j) \
        float dx = tx - s_px[j]; \
        float dy = ty - s_py[j]; \
        float dz = tz - s_pz[j]; \
        float R2 = dx*dx + dy*dy + dz*dz; \
        if (R2 < 1e-30f) continue; \
        float invR = rsqrtf(R2); \
        float R = R2 * invR; \
        float kR_re = k_re * R; \
        float kR_im = k_im * R; \
        float edr = expf(-kR_im); \
        float cs, sn; \
        sincosf(kR_re, &sn, &cs); \
        float G_re = edr * cs * inv4pi * invR; \
        float G_im = edr * sn * inv4pi * invR; \
        float jr = s_qr[j], ji = s_qi[j]; \
        acc_re += G_re * jr - G_im * ji; \
        acc_im += G_re * ji + G_im * jr; \
        float fac_a = (-k_im - invR); \
        float fac_b = k_re; \
        float fg_re = (fac_a * G_re - fac_b * G_im) * invR; \
        float fg_im = (fac_a * G_im + fac_b * G_re) * invR; \
        float dg_re, dg_im, dGq_re, dGq_im; \
        dg_re = fg_re * dx; dg_im = fg_im * dx; \
        gx_re += dg_re * jr - dg_im * ji; gx_im += dg_re * ji + dg_im * jr; \
        dg_re = fg_re * dy; dg_im = fg_im * dy; \
        gy_re += dg_re * jr - dg_im * ji; gy_im += dg_re * ji + dg_im * jr; \
        dg_re = fg_re * dz; dg_im = fg_im * dz; \
        gz_re += dg_re * jr - dg_im * ji; gz_im += dg_re * ji + dg_im * jr;

    for (int tile = 0; tile < N; tile += P2P_TILE) {
        int tile_size = min(P2P_TILE, N - tile);
        if ((int)threadIdx.x < tile_size) {
            int j = tile + threadIdx.x;
            s_px[threadIdx.x] = pts[j*3+0];
            s_py[threadIdx.x] = pts[j*3+1];
            s_pz[threadIdx.x] = pts[j*3+2];
            s_qr[threadIdx.x] = q_re[j];
            s_qi[threadIdx.x] = q_im[j];
        }
        __syncthreads();

        if (i < N) {
            int tile_end_idx = tile + tile_size;
            bool tile_overlaps = (tile_end_idx > my_start && tile < my_end);

            if (!tile_overlaps) {
                for (int j = 0; j < tile_size; j++) {
                    P2P_GRAD_BODY(j)
                }
            } else {
                for (int j = 0; j < tile_size; j++) {
                    int gj = tile + j;
                    if (gj >= my_start && gj < my_end) continue;
                    P2P_GRAD_BODY(j)
                }
            }
        }
        __syncthreads();
    }
    #undef P2P_GRAD_BODY

    if (i < N) {
        pot_re[i] += acc_re; pot_im[i] += acc_im;
        grad_re[i*3+0] += gx_re; grad_im[i*3+0] += gx_im;
        grad_re[i*3+1] += gy_re; grad_im[i*3+1] += gy_im;
        grad_re[i*3+2] += gz_re; grad_im[i*3+2] += gz_im;
    }
}
#undef P2P_GREEN_BODY

// Batched inter-face P2P: 4 charge vectors, pot+grad for first 3, pot-only for 4th.
// Computes geometry (dx,dy,dz,R,G,∇G) once per pair, applies to all 4 charges.
// Saves ~50% FLOPs vs 4 separate P2P kernel launches.
// Layout: q_re/im[b*N + i] for batch b, point i. grad_re/im[b*N*3 + i*3+d].
__global__ void kernel_p2p_batch4_f32(
    const float* __restrict__ pts,
    const float* __restrict__ q_re,      // [4*N] packed charges
    const float* __restrict__ q_im,
    float* __restrict__ pot_re,          // [4*N] packed potential output
    float* __restrict__ pot_im,
    float* __restrict__ grad_re,         // [3*N*3] packed gradient output (batches 0-2)
    float* __restrict__ grad_im,
    int N,
    const int* __restrict__ face_offsets,
    int n_faces,
    float k_re, float k_im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float tx, ty, tz;
    int my_start = 0, my_end = 0;
    if (i < N) {
        tx = pts[i*3]; ty = pts[i*3+1]; tz = pts[i*3+2];
        for (int f = 0; f < n_faces; f++) {
            if (i < face_offsets[f+1]) {
                my_start = face_offsets[f];
                my_end = face_offsets[f+1];
                break;
            }
        }
    }

    // 4 potential accumulators (re,im)
    float a0r=0,a0i=0, a1r=0,a1i=0, a2r=0,a2i=0, a3r=0,a3i=0;
    // 3 × 3-component gradient accumulators (batches 0-2)
    float g0xr=0,g0xi=0,g0yr=0,g0yi=0,g0zr=0,g0zi=0;
    float g1xr=0,g1xi=0,g1yr=0,g1yi=0,g1zr=0,g1zi=0;
    float g2xr=0,g2xi=0,g2yr=0,g2yi=0,g2zr=0,g2zi=0;
    float inv4pi = 0.07957747154594767f;

    // Shared memory: positions + 4 charge vectors
    __shared__ float s_px[P2P_TILE], s_py[P2P_TILE], s_pz[P2P_TILE];
    __shared__ float s_q0r[P2P_TILE], s_q0i[P2P_TILE];
    __shared__ float s_q1r[P2P_TILE], s_q1i[P2P_TILE];
    __shared__ float s_q2r[P2P_TILE], s_q2i[P2P_TILE];
    __shared__ float s_q3r[P2P_TILE], s_q3i[P2P_TILE];

    for (int tile = 0; tile < N; tile += P2P_TILE) {
        int tile_size = min(P2P_TILE, N - tile);
        if ((int)threadIdx.x < tile_size) {
            int j = tile + threadIdx.x;
            s_px[threadIdx.x] = pts[j*3+0];
            s_py[threadIdx.x] = pts[j*3+1];
            s_pz[threadIdx.x] = pts[j*3+2];
            s_q0r[threadIdx.x] = q_re[j];         s_q0i[threadIdx.x] = q_im[j];
            s_q1r[threadIdx.x] = q_re[N+j];       s_q1i[threadIdx.x] = q_im[N+j];
            s_q2r[threadIdx.x] = q_re[2*N+j];     s_q2i[threadIdx.x] = q_im[2*N+j];
            s_q3r[threadIdx.x] = q_re[3*N+j];     s_q3i[threadIdx.x] = q_im[3*N+j];
        }
        __syncthreads();

        if (i < N) {
            int tile_end = tile + tile_size;
            bool overlaps = (tile_end > my_start && tile < my_end);

            for (int j = 0; j < tile_size; j++) {
                if (overlaps) {
                    int gj = tile + j;
                    if (gj >= my_start && gj < my_end) continue;
                }

                float dx = tx - s_px[j];
                float dy = ty - s_py[j];
                float dz = tz - s_pz[j];
                float R2 = dx*dx + dy*dy + dz*dz;
                if (R2 < 1e-30f) continue;
                float invR = rsqrtf(R2);
                float R = R2 * invR;
                float kR_re = k_re * R;
                float kR_im = k_im * R;
                float edr = expf(-kR_im);
                float cs, sn;
                sincosf(kR_re, &sn, &cs);
                float G_re = edr * cs * inv4pi * invR;
                float G_im = edr * sn * inv4pi * invR;

                // Gradient prefactor: ∇G = G * (ik - 1/R) / R * (r - r')
                float fac_a = (-k_im - invR);
                float fac_b = k_re;
                float fg_re = (fac_a * G_re - fac_b * G_im) * invR;
                float fg_im = (fac_a * G_im + fac_b * G_re) * invR;

                // Precompute dG components (shared across batches)
                float dGx_re = fg_re * dx, dGx_im = fg_im * dx;
                float dGy_re = fg_re * dy, dGy_im = fg_im * dy;
                float dGz_re = fg_re * dz, dGz_im = fg_im * dz;

                float jr, ji;

                // Batch 0: pot + grad
                jr = s_q0r[j]; ji = s_q0i[j];
                a0r += G_re*jr - G_im*ji; a0i += G_re*ji + G_im*jr;
                g0xr += dGx_re*jr - dGx_im*ji; g0xi += dGx_re*ji + dGx_im*jr;
                g0yr += dGy_re*jr - dGy_im*ji; g0yi += dGy_re*ji + dGy_im*jr;
                g0zr += dGz_re*jr - dGz_im*ji; g0zi += dGz_re*ji + dGz_im*jr;

                // Batch 1: pot + grad
                jr = s_q1r[j]; ji = s_q1i[j];
                a1r += G_re*jr - G_im*ji; a1i += G_re*ji + G_im*jr;
                g1xr += dGx_re*jr - dGx_im*ji; g1xi += dGx_re*ji + dGx_im*jr;
                g1yr += dGy_re*jr - dGy_im*ji; g1yi += dGy_re*ji + dGy_im*jr;
                g1zr += dGz_re*jr - dGz_im*ji; g1zi += dGz_re*ji + dGz_im*jr;

                // Batch 2: pot + grad
                jr = s_q2r[j]; ji = s_q2i[j];
                a2r += G_re*jr - G_im*ji; a2i += G_re*ji + G_im*jr;
                g2xr += dGx_re*jr - dGx_im*ji; g2xi += dGx_re*ji + dGx_im*jr;
                g2yr += dGy_re*jr - dGy_im*ji; g2yi += dGy_re*ji + dGy_im*jr;
                g2zr += dGz_re*jr - dGz_im*ji; g2zi += dGz_re*ji + dGz_im*jr;

                // Batch 3: pot only
                jr = s_q3r[j]; ji = s_q3i[j];
                a3r += G_re*jr - G_im*ji; a3i += G_re*ji + G_im*jr;
            }
        }
        __syncthreads();
    }

    if (i < N) {
        // Write potentials (4 batches)
        pot_re[i]     += a0r; pot_im[i]     += a0i;
        pot_re[N+i]   += a1r; pot_im[N+i]   += a1i;
        pot_re[2*N+i] += a2r; pot_im[2*N+i] += a2i;
        pot_re[3*N+i] += a3r; pot_im[3*N+i] += a3i;
        // Write gradients (batches 0-2)
        grad_re[i*3+0] += g0xr; grad_im[i*3+0] += g0xi;
        grad_re[i*3+1] += g0yr; grad_im[i*3+1] += g0yi;
        grad_re[i*3+2] += g0zr; grad_im[i*3+2] += g0zi;
        int o1 = N*3;
        grad_re[o1+i*3+0] += g1xr; grad_im[o1+i*3+0] += g1xi;
        grad_re[o1+i*3+1] += g1yr; grad_im[o1+i*3+1] += g1yi;
        grad_re[o1+i*3+2] += g1zr; grad_im[o1+i*3+2] += g1zi;
        int o2 = 2*N*3;
        grad_re[o2+i*3+0] += g2xr; grad_im[o2+i*3+0] += g2xi;
        grad_re[o2+i*3+1] += g2yr; grad_im[o2+i*3+1] += g2yi;
        grad_re[o2+i*3+2] += g2zr; grad_im[o2+i*3+2] += g2zi;
    }
}

// Batched inter-face P2P: 8 charge vectors (2 × LK_combined), geometry once per pair.
// Charges 0-2,4-6 get pot+grad; charges 3,7 get pot only.
// ~30% fewer FLOPs than 2× batch4 (geometry computed once instead of twice).
__global__ void kernel_p2p_batch8_f32(
    const float* __restrict__ pts,
    const float* __restrict__ q_re,      // [8*N] packed
    const float* __restrict__ q_im,
    float* __restrict__ pot_re,          // [8*N] packed
    float* __restrict__ pot_im,
    float* __restrict__ grad_re,         // [6*N*3] packed
    float* __restrict__ grad_im,
    int N,
    const int* __restrict__ face_offsets,
    int n_faces,
    float k_re, float k_im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float tx, ty, tz;
    int my_start = 0, my_end = 0;
    if (i < N) {
        tx = pts[i*3]; ty = pts[i*3+1]; tz = pts[i*3+2];
        for (int f = 0; f < n_faces; f++) {
            if (i < face_offsets[f+1]) {
                my_start = face_offsets[f];
                my_end = face_offsets[f+1];
                break;
            }
        }
    }

    // 8 potential accumulators
    float a0r=0,a0i=0, a1r=0,a1i=0, a2r=0,a2i=0, a3r=0,a3i=0;
    float a4r=0,a4i=0, a5r=0,a5i=0, a6r=0,a6i=0, a7r=0,a7i=0;
    // 6 × 3-component gradient accumulators (batches 0-2 and 4-6)
    float g0xr=0,g0xi=0,g0yr=0,g0yi=0,g0zr=0,g0zi=0;
    float g1xr=0,g1xi=0,g1yr=0,g1yi=0,g1zr=0,g1zi=0;
    float g2xr=0,g2xi=0,g2yr=0,g2yi=0,g2zr=0,g2zi=0;
    float g3xr=0,g3xi=0,g3yr=0,g3yi=0,g3zr=0,g3zi=0;
    float g4xr=0,g4xi=0,g4yr=0,g4yi=0,g4zr=0,g4zi=0;
    float g5xr=0,g5xi=0,g5yr=0,g5yi=0,g5zr=0,g5zi=0;
    float inv4pi = 0.07957747154594767f;

    __shared__ float s_px[P2P_TILE], s_py[P2P_TILE], s_pz[P2P_TILE];
    __shared__ float s_qr[8*P2P_TILE], s_qi[8*P2P_TILE];

    for (int tile = 0; tile < N; tile += P2P_TILE) {
        int tile_size = min(P2P_TILE, N - tile);
        if ((int)threadIdx.x < tile_size) {
            int j = tile + threadIdx.x;
            s_px[threadIdx.x] = pts[j*3+0];
            s_py[threadIdx.x] = pts[j*3+1];
            s_pz[threadIdx.x] = pts[j*3+2];
            for (int b = 0; b < 8; b++) {
                s_qr[b*P2P_TILE + threadIdx.x] = q_re[b*N + j];
                s_qi[b*P2P_TILE + threadIdx.x] = q_im[b*N + j];
            }
        }
        __syncthreads();

        if (i < N) {
            int tile_end = tile + tile_size;
            bool overlaps = (tile_end > my_start && tile < my_end);

            for (int j = 0; j < tile_size; j++) {
                if (overlaps) {
                    int gj = tile + j;
                    if (gj >= my_start && gj < my_end) continue;
                }

                float dx = tx - s_px[j];
                float dy = ty - s_py[j];
                float dz = tz - s_pz[j];
                float R2 = dx*dx + dy*dy + dz*dz;
                if (R2 < 1e-30f) continue;
                float invR = rsqrtf(R2);
                float R = R2 * invR;
                float edr = expf(-k_im * R);
                float cs, sn;
                sincosf(k_re * R, &sn, &cs);
                float G_re = edr * cs * inv4pi * invR;
                float G_im = edr * sn * inv4pi * invR;
                float fac_a = (-k_im - invR);
                float fg_re = (fac_a * G_re - k_re * G_im) * invR;
                float fg_im = (fac_a * G_im + k_re * G_re) * invR;
                float dGx_re = fg_re*dx, dGx_im = fg_im*dx;
                float dGy_re = fg_re*dy, dGy_im = fg_im*dy;
                float dGz_re = fg_re*dz, dGz_im = fg_im*dz;

                float jr, ji;

                // Macro: apply Green's function potential to charge
                #define B8_POT(B, AR, AI) \
                    jr = s_qr[B*P2P_TILE+j]; ji = s_qi[B*P2P_TILE+j]; \
                    AR += G_re*jr - G_im*ji; AI += G_re*ji + G_im*jr;

                // Macro: apply potential + gradient
                #define B8_PG(B, AR, AI, GXR,GXI,GYR,GYI,GZR,GZI) \
                    jr = s_qr[B*P2P_TILE+j]; ji = s_qi[B*P2P_TILE+j]; \
                    AR += G_re*jr - G_im*ji; AI += G_re*ji + G_im*jr; \
                    GXR += dGx_re*jr - dGx_im*ji; GXI += dGx_re*ji + dGx_im*jr; \
                    GYR += dGy_re*jr - dGy_im*ji; GYI += dGy_re*ji + dGy_im*jr; \
                    GZR += dGz_re*jr - dGz_im*ji; GZI += dGz_re*ji + dGz_im*jr;

                B8_PG(0, a0r,a0i, g0xr,g0xi,g0yr,g0yi,g0zr,g0zi)
                B8_PG(1, a1r,a1i, g1xr,g1xi,g1yr,g1yi,g1zr,g1zi)
                B8_PG(2, a2r,a2i, g2xr,g2xi,g2yr,g2yi,g2zr,g2zi)
                B8_POT(3, a3r,a3i)
                B8_PG(4, a4r,a4i, g3xr,g3xi,g3yr,g3yi,g3zr,g3zi)
                B8_PG(5, a5r,a5i, g4xr,g4xi,g4yr,g4yi,g4zr,g4zi)
                B8_PG(6, a6r,a6i, g5xr,g5xi,g5yr,g5yi,g5zr,g5zi)
                B8_POT(7, a7r,a7i)

                #undef B8_POT
                #undef B8_PG
            }
        }
        __syncthreads();
    }

    if (i < N) {
        // Write 8 potentials
        pot_re[0*N+i]+=a0r; pot_im[0*N+i]+=a0i;
        pot_re[1*N+i]+=a1r; pot_im[1*N+i]+=a1i;
        pot_re[2*N+i]+=a2r; pot_im[2*N+i]+=a2i;
        pot_re[3*N+i]+=a3r; pot_im[3*N+i]+=a3i;
        pot_re[4*N+i]+=a4r; pot_im[4*N+i]+=a4i;
        pot_re[5*N+i]+=a5r; pot_im[5*N+i]+=a5i;
        pot_re[6*N+i]+=a6r; pot_im[6*N+i]+=a6i;
        pot_re[7*N+i]+=a7r; pot_im[7*N+i]+=a7i;
        // Write 6 gradients
        #define WG(G, OFF, XR,XI,YR,YI,ZR,ZI) \
            grad_re[OFF+i*3+0]+=XR; grad_im[OFF+i*3+0]+=XI; \
            grad_re[OFF+i*3+1]+=YR; grad_im[OFF+i*3+1]+=YI; \
            grad_re[OFF+i*3+2]+=ZR; grad_im[OFF+i*3+2]+=ZI;
        WG(0, 0*N*3, g0xr,g0xi,g0yr,g0yi,g0zr,g0zi)
        WG(1, 1*N*3, g1xr,g1xi,g1yr,g1yi,g1zr,g1zi)
        WG(2, 2*N*3, g2xr,g2xi,g2yr,g2yi,g2zr,g2zi)
        WG(3, 3*N*3, g3xr,g3xi,g3yr,g3yi,g3zr,g3zi)
        WG(4, 4*N*3, g4xr,g4xi,g4yr,g4yi,g4zr,g4zi)
        WG(5, 5*N*3, g5xr,g5xi,g5yr,g5yi,g5zr,g5zi)
        #undef WG
    }
}

// GPU gather/scatter kernels for eliminating host<->device copies per face

// Gather charges from global GPU buffer to per-face local buffer
__global__ void kernel_gather_face(
    const double* __restrict__ global_re,
    const double* __restrict__ global_im,
    const int*    __restrict__ local_to_global,
    int n_local,
    double* __restrict__ face_re,
    double* __restrict__ face_im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_local) return;
    int gi = local_to_global[i];
    face_re[i] = global_re[gi];
    face_im[i] = global_im[gi];
}

// Scatter-add per-face results to global result buffer
__global__ void kernel_scatter_add(
    const double* __restrict__ face_re,
    const double* __restrict__ face_im,
    const int*    __restrict__ local_to_global,
    int n_local,
    double* __restrict__ global_re,
    double* __restrict__ global_im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_local) return;
    int gi = local_to_global[i];
    // No race: each point belongs to exactly one face
    global_re[gi] += face_re[i];
    global_im[gi] += face_im[i];
}

// Scatter-add local gradient component (u or v) to global xyz gradient
// grad_xyz[gi*3+d] += face_val * axis[d]
__global__ void kernel_scatter_grad_component(
    const double* __restrict__ face_re,
    const double* __restrict__ face_im,
    const int*    __restrict__ local_to_global,
    int n_local,
    double ax, double ay, double az,
    double* __restrict__ grad_re,
    double* __restrict__ grad_im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_local) return;
    int gi = local_to_global[i];
    double vr = face_re[i], vi = face_im[i];
    grad_re[gi*3+0] += vr * ax;
    grad_im[gi*3+0] += vi * ax;
    grad_re[gi*3+1] += vr * ay;
    grad_im[gi*3+1] += vi * ay;
    grad_re[gi*3+2] += vr * az;
    grad_im[gi*3+2] += vi * az;
}

// Sort charges to P2P order and convert FP64->FP32
__global__ void kernel_sort_charges_f32(
    const double* __restrict__ charges_re,
    const double* __restrict__ charges_im,
    const int*    __restrict__ sort_order,
    int N,
    float* __restrict__ sorted_re,
    float* __restrict__ sorted_im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int gi = sort_order[i];
    sorted_re[i] = (float)charges_re[gi];
    sorted_im[i] = (float)charges_im[gi];
}

// Unsort-add P2P potential results (FP32->FP64) to global buffer
__global__ void kernel_unsort_add_pot(
    const float* __restrict__ p2p_re,
    const float* __restrict__ p2p_im,
    const int*   __restrict__ sort_order,
    int N,
    double* __restrict__ global_re,
    double* __restrict__ global_im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int gi = sort_order[i];
    global_re[gi] += (double)p2p_re[i];
    global_im[gi] += (double)p2p_im[i];
}

// Unsort-add P2P gradient results (FP32->FP64) to global buffer
__global__ void kernel_unsort_add_grad(
    const float* __restrict__ p2p_grad_re,
    const float* __restrict__ p2p_grad_im,
    const int*   __restrict__ sort_order,
    int N,
    double* __restrict__ grad_re,
    double* __restrict__ grad_im)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int gi = sort_order[i];
    grad_re[gi*3+0] += (double)p2p_grad_re[i*3+0];
    grad_im[gi*3+0] += (double)p2p_grad_im[i*3+0];
    grad_re[gi*3+1] += (double)p2p_grad_re[i*3+1];
    grad_im[gi*3+1] += (double)p2p_grad_im[i*3+1];
    grad_re[gi*3+2] += (double)p2p_grad_re[i*3+2];
    grad_im[gi*3+2] += (double)p2p_grad_im[i*3+2];
}

// ============================================================================
// Host helpers
// ============================================================================

static void lagrange_weights_1d(double x, double x0, double h, int p1, double* w)
{
    for (int i = 0; i < p1; i++) {
        double xi = x0 + i * h;
        w[i] = 1.0;
        for (int j = 0; j < p1; j++) {
            if (j != i) {
                double xj = x0 + j * h;
                w[i] *= (x - xj) / (xi - xj);
            }
        }
    }
}

// Round up to nearest FFT-friendly size (factors of 2, 3, 5, 7 only)
static int round_fft(int n) {
    if (n <= 1) return 1;
    // Generate all smooth numbers (7-smooth) up to 2*n, pick smallest >= n
    int best = 1;
    while (best < n) best *= 2;  // fallback: pure power of 2
    for (int p7 = 1; p7 <= best; p7 *= 7)
        for (int p5 = p7; p5 <= best; p5 *= 5)
            for (int p3 = p5; p3 <= best; p3 *= 3)
                for (int p2 = p3; p2 <= best; p2 *= 2)
                    if (p2 >= n && p2 < best) best = p2;
    return best;
}

// ============================================================================
// HelmholtzSurfacePFFT implementation
// ============================================================================

void HelmholtzSurfacePFFT::init(const double* points, int n_pts,
                                 const int* face_ids, int n_faces_,
                                 const double* face_normals,
                                 cdouble k_val, int digits)
{
    Timer timer;
    k = k_val;
    Nt = Ns = n_pts;
    n_faces = n_faces_;
    interp_p = std::max(2, digits);
    int p1 = interp_p + 1;
    stencil_2d = p1 * p1;

    printf("  [SurfPFFT] N=%d, n_faces=%d, k=(%.4f,%.4f), interp_order=%d\n",
           Nt, n_faces, k.real(), k.imag(), interp_p);

    // Store face assignment
    point_face.assign(face_ids, face_ids + n_pts);

    // --- Step 1: Classify points per face, build local coordinate systems ---
    for (int f = 0; f < n_faces; f++) {
        FaceGrid& fg = faces[f];
        fg.face_id = f;
        fg.normal[0] = face_normals[f*3+0];
        fg.normal[1] = face_normals[f*3+1];
        fg.normal[2] = face_normals[f*3+2];

        // Collect points on this face
        fg.local_to_global.clear();
        for (int i = 0; i < n_pts; i++) {
            if (face_ids[i] == f)
                fg.local_to_global.push_back(i);
        }
        fg.n_pts = (int)fg.local_to_global.size();
        if (fg.n_pts == 0) continue;

        // Build local coordinate system: pick u perpendicular to normal
        double nx = fg.normal[0], ny = fg.normal[1], nz = fg.normal[2];
        // Choose u_axis: cross normal with (1,0,0) or (0,1,0)
        double ux, uy, uz;
        if (fabs(nx) < 0.9) {
            // cross with (1,0,0)
            ux = 0; uy = nz; uz = -ny;
        } else {
            // cross with (0,1,0)
            ux = -nz; uy = 0; uz = nx;
        }
        double unorm = sqrt(ux*ux + uy*uy + uz*uz);
        ux /= unorm; uy /= unorm; uz /= unorm;
        fg.u_axis[0] = ux; fg.u_axis[1] = uy; fg.u_axis[2] = uz;

        // v = n x u
        fg.v_axis[0] = ny*uz - nz*uy;
        fg.v_axis[1] = nz*ux - nx*uz;
        fg.v_axis[2] = nx*uy - ny*ux;

        // Origin: first point on face
        int gi0 = fg.local_to_global[0];
        fg.origin_3d[0] = points[gi0*3+0];
        fg.origin_3d[1] = points[gi0*3+1];
        fg.origin_3d[2] = points[gi0*3+2];
    }

    // --- Step 2: For each face, build 2D grid and Green's FFT ---
    double lambda_val = 2.0 * M_PI / std::max(std::abs(k), 0.01);

    for (int f = 0; f < n_faces; f++) {
        FaceGrid& fg = faces[f];
        if (fg.n_pts == 0) continue;

        // Project points to local (u,v) coords
        std::vector<double> local_u(fg.n_pts), local_v(fg.n_pts);
        double umin = 1e30, umax = -1e30, vmin = 1e30, vmax = -1e30;

        for (int li = 0; li < fg.n_pts; li++) {
            int gi = fg.local_to_global[li];
            double dx = points[gi*3+0] - fg.origin_3d[0];
            double dy = points[gi*3+1] - fg.origin_3d[1];
            double dz = points[gi*3+2] - fg.origin_3d[2];
            double u = dx*fg.u_axis[0] + dy*fg.u_axis[1] + dz*fg.u_axis[2];
            double v = dx*fg.v_axis[0] + dy*fg.v_axis[1] + dz*fg.v_axis[2];
            local_u[li] = u;
            local_v[li] = v;
            umin = std::min(umin, u); umax = std::max(umax, u);
            vmin = std::min(vmin, v); vmax = std::max(vmax, v);
        }

        // Grid spacing
        double diameter = std::max(umax - umin, vmax - vmin);
        if (diameter < 1e-15) diameter = 1e-6;
        double h_wave = lambda_val / (2.0 * interp_p);
        double h_geom = diameter / 40.0;
        fg.h = std::min(h_wave, h_geom);

        // Density-based refinement: keep correction matrix sparse
        // Target: ~4 points per grid cell => correction ~81 cells × 4 = 324/pt
        double face_area = (umax - umin) * (vmax - vmin);
        if (face_area > 1e-20) {
            double density = fg.n_pts / face_area;
            double h_density = sqrt(4.0 / density);
            fg.h = std::min(fg.h, h_density);
        }

        fg.h = std::max(fg.h, diameter / 500.0);

        double pad = (interp_p + 2) * fg.h;
        fg.grid_origin[0] = umin - pad;
        fg.grid_origin[1] = vmin - pad;

        fg.Mu = (int)ceil((umax - fg.grid_origin[0] + pad) / fg.h) + 1;
        fg.Mv = (int)ceil((vmax - fg.grid_origin[1] + pad) / fg.h) + 1;
        fg.M2u = round_fft(2 * fg.Mu);
        fg.M2v = round_fft(2 * fg.Mv);
        fg.grid_total = (long long)fg.M2u * fg.M2v;

        double pts_per_cell = (face_area > 1e-20) ? fg.n_pts * fg.h * fg.h / face_area : 0;
        printf("  [SurfPFFT] Face %d: %d pts, grid %dx%d (FFT %dx%d), h=%.4f, pts/cell=%.1f\n",
               f, fg.n_pts, fg.Mu, fg.Mv, fg.M2u, fg.M2v, fg.h, pts_per_cell);

        // --- 2D Green's function FFT (compute FP64, store FP32) ---
        std::vector<cufftComplex> h_G(fg.grid_total);
        std::vector<cufftComplex> h_dGdu(fg.grid_total);
        std::vector<cufftComplex> h_dGdv(fg.grid_total);

        for (int iu = 0; iu < fg.M2u; iu++) {
            bool zu = (iu >= fg.Mu && iu <= fg.M2u - fg.Mu);
            double du = (iu < fg.Mu) ? iu * fg.h : (iu - fg.M2u) * fg.h;
            for (int iv = 0; iv < fg.M2v; iv++) {
                bool zv = (iv >= fg.Mv && iv <= fg.M2v - fg.Mv);
                double dv = (iv < fg.Mv) ? iv * fg.h : (iv - fg.M2v) * fg.h;
                long long idx = (long long)iu * fg.M2v + iv;

                double R = sqrt(du*du + dv*dv);

                if (R < 1e-30 || zu || zv) {
                    h_G[idx] = {0.0f, 0.0f};
                    h_dGdu[idx] = {0.0f, 0.0f};
                    h_dGdv[idx] = {0.0f, 0.0f};
                } else {
                    cdouble ikR = k * R;
                    cdouble expikR = std::exp(cdouble(0, 1) * ikR);
                    cdouble G = expikR * INV4PI / R;

                    h_G[idx] = {(float)G.real(), (float)G.imag()};

                    cdouble factor = (cdouble(0,1) * k - 1.0/R) * G / R;
                    cdouble gu = factor * du;
                    cdouble gv = factor * dv;

                    h_dGdu[idx] = {(float)gu.real(), (float)gu.imag()};
                    h_dGdv[idx] = {(float)gv.real(), (float)gv.imag()};
                }
            }
        }

        // Allocate GPU buffers and FFT (FP32 C2C)
        CUDA_CHECK(cudaMalloc(&fg.d_G_hat, fg.grid_total * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&fg.d_dGdu_hat, fg.grid_total * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&fg.d_dGdv_hat, fg.grid_total * sizeof(cufftComplex)));

        CUFFT_CHECK(cufftPlan2d(&fg.plan_fwd, fg.M2u, fg.M2v, CUFFT_C2C));
        CUFFT_CHECK(cufftPlan2d(&fg.plan_inv, fg.M2u, fg.M2v, CUFFT_C2C));

        // Create per-face CUDA stream
        CUDA_CHECK(cudaStreamCreate(&fg.stream));

        auto fft_green = [&](std::vector<cufftComplex>& h_data, cufftComplex* d_hat) {
            CUDA_CHECK(cudaMemcpy(d_hat, h_data.data(),
                                  fg.grid_total * sizeof(cufftComplex),
                                  cudaMemcpyHostToDevice));
            CUFFT_CHECK(cufftExecC2C(fg.plan_fwd, d_hat, d_hat, CUFFT_FORWARD));
        };
        fft_green(h_G, fg.d_G_hat);
        fft_green(h_dGdu, fg.d_dGdu_hat);
        fft_green(h_dGdv, fg.d_dGdv_hat);

        // --- 2D interpolation stencils ---
        std::vector<int>    h_idx(fg.n_pts * stencil_2d);
        std::vector<double> h_wt(fg.n_pts * stencil_2d);

        double wu[8], wv[8];  // max p+1 = 8
        for (int li = 0; li < fg.n_pts; li++) {
            double pu = local_u[li], pv = local_v[li];
            double fu = (pu - fg.grid_origin[0]) / fg.h;
            double fv = (pv - fg.grid_origin[1]) / fg.h;

            int iu0 = (int)floor(fu) - (interp_p - 1) / 2;
            int iv0 = (int)floor(fv) - (interp_p - 1) / 2;
            iu0 = std::max(0, std::min(iu0, fg.Mu - p1));
            iv0 = std::max(0, std::min(iv0, fg.Mv - p1));

            lagrange_weights_1d(pu, fg.grid_origin[0] + iu0 * fg.h, fg.h, p1, wu);
            lagrange_weights_1d(pv, fg.grid_origin[1] + iv0 * fg.h, fg.h, p1, wv);

            int s = 0;
            for (int a = 0; a < p1; a++) {
                for (int b = 0; b < p1; b++) {
                    int giu = iu0 + a;
                    int giv = iv0 + b;
                    long long gi = (long long)giu * fg.M2v + giv;
                    h_idx[li * stencil_2d + s] = (int)gi;
                    h_wt[li * stencil_2d + s] = wu[a] * wv[b];
                    s++;
                }
            }
        }

        CUDA_CHECK(cudaMalloc(&fg.d_stencil_idx, (long long)fg.n_pts * stencil_2d * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&fg.d_stencil_wt,  (long long)fg.n_pts * stencil_2d * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(fg.d_stencil_idx, h_idx.data(),
                              (long long)fg.n_pts * stencil_2d * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(fg.d_stencil_wt, h_wt.data(),
                              (long long)fg.n_pts * stencil_2d * sizeof(double), cudaMemcpyHostToDevice));

        // --- Intra-face near-field correction ---
        double near_radius = (interp_p + 2) * fg.h;

        // Build near list in local face indices
        std::vector<int> h_row_ptr(fg.n_pts + 1, 0);
        std::vector<int> h_col_idx;
        double r2 = near_radius * near_radius;

        for (int li = 0; li < fg.n_pts; li++) {
            for (int lj = 0; lj < fg.n_pts; lj++) {
                double du2 = local_u[li] - local_u[lj];
                double dv2 = local_v[li] - local_v[lj];
                if (du2*du2 + dv2*dv2 < r2 || li == lj) {
                    h_col_idx.push_back(lj);
                }
            }
            h_row_ptr[li+1] = (int)h_col_idx.size();
        }
        fg.corr_nnz = (int)h_col_idx.size();

        // Compute correction values: G_exact - G_grid
        std::vector<double> h_cG_re(fg.corr_nnz), h_cG_im(fg.corr_nnz);
        std::vector<double> h_cdu_re(fg.corr_nnz), h_cdu_im(fg.corr_nnz);
        std::vector<double> h_cdv_re(fg.corr_nnz), h_cdv_im(fg.corr_nnz);

        int grange = (int)ceil(near_radius / fg.h) + interp_p + 1;
        int gspan = 2 * grange + 1;
        long long gsize = (long long)gspan * gspan;
        std::vector<cdouble> G_local(gsize);
        std::vector<cdouble> dGdu_local(gsize), dGdv_local(gsize);

        for (int di = -grange; di <= grange; di++) {
            for (int dj = -grange; dj <= grange; dj++) {
                double ddu = di * fg.h, ddv = dj * fg.h;
                double R = sqrt(ddu*ddu + ddv*ddv);
                long long li2 = (long long)(di+grange)*gspan + (dj+grange);
                if (R < 1e-30) {
                    G_local[li2] = 0;
                    dGdu_local[li2] = dGdv_local[li2] = 0;
                } else {
                    cdouble expikR = std::exp(cdouble(0, 1) * k * R);
                    G_local[li2] = expikR * INV4PI / R;
                    cdouble factor = (cdouble(0,1) * k - 1.0/R) * G_local[li2] / R;
                    dGdu_local[li2] = factor * ddu;
                    dGdv_local[li2] = factor * ddv;
                }
            }
        }

        #pragma omp parallel for schedule(dynamic, 64)
        for (int li = 0; li < fg.n_pts; li++) {
            for (int pp = h_row_ptr[li]; pp < h_row_ptr[li+1]; pp++) {
                int lj = h_col_idx[pp];
                double du_ex = local_u[li] - local_u[lj];
                double dv_ex = local_v[li] - local_v[lj];
                double R = sqrt(du_ex*du_ex + dv_ex*dv_ex);

                cdouble G_exact(0, 0), dGu_exact(0, 0), dGv_exact(0, 0);
                if (R > 1e-30) {
                    cdouble expikR = std::exp(cdouble(0, 1) * k * R);
                    G_exact = expikR * INV4PI / R;
                    cdouble factor = (cdouble(0,1) * k - 1.0/R) * G_exact / R;
                    dGu_exact = factor * du_ex;
                    dGv_exact = factor * dv_ex;
                }

                // Grid-mediated G
                cdouble G_grid(0, 0), dGu_grid(0, 0), dGv_grid(0, 0);
                int base_i = li * stencil_2d;
                int base_j = lj * stencil_2d;

                for (int a = 0; a < stencil_2d; a++) {
                    double wa = h_wt[base_i + a];
                    if (fabs(wa) < 1e-15) continue;
                    int ga = h_idx[base_i + a];
                    int ga_u = ga / fg.M2v;
                    int ga_v = ga % fg.M2v;

                    for (int b = 0; b < stencil_2d; b++) {
                        double wb = h_wt[base_j + b];
                        if (fabs(wb) < 1e-15) continue;
                        int gb = h_idx[base_j + b];
                        int gb_u = gb / fg.M2v;
                        int gb_v = gb % fg.M2v;

                        int ddi = ga_u - gb_u;
                        int ddj = ga_v - gb_v;

                        if (abs(ddi) <= grange && abs(ddj) <= grange) {
                            long long lli = (long long)(ddi+grange)*gspan + (ddj+grange);
                            double ww = wa * wb;
                            G_grid   += ww * G_local[lli];
                            dGu_grid += ww * dGdu_local[lli];
                            dGv_grid += ww * dGdv_local[lli];
                        }
                    }
                }

                cdouble cG  = G_exact - G_grid;
                cdouble cdu = dGu_exact - dGu_grid;
                cdouble cdv = dGv_exact - dGv_grid;

                h_cG_re[pp]  = cG.real();   h_cG_im[pp]  = cG.imag();
                h_cdu_re[pp] = cdu.real();  h_cdu_im[pp] = cdu.imag();
                h_cdv_re[pp] = cdv.real();  h_cdv_im[pp] = cdv.imag();
            }
        }

        // Upload correction to GPU
        CUDA_CHECK(cudaMalloc(&fg.d_corr_row_ptr, (fg.n_pts+1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&fg.d_corr_col_idx, fg.corr_nnz * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(fg.d_corr_row_ptr, h_row_ptr.data(),
                              (fg.n_pts+1)*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(fg.d_corr_col_idx, h_col_idx.data(),
                              fg.corr_nnz*sizeof(int), cudaMemcpyHostToDevice));

        auto upload_corr = [&](std::vector<double>& h_re, std::vector<double>& h_im,
                               double*& d_re, double*& d_im) {
            CUDA_CHECK(cudaMalloc(&d_re, fg.corr_nnz * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_im, fg.corr_nnz * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_re, h_re.data(), fg.corr_nnz*sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_im, h_im.data(), fg.corr_nnz*sizeof(double), cudaMemcpyHostToDevice));
        };
        upload_corr(h_cG_re,  h_cG_im,  fg.d_corr_G_re,   fg.d_corr_G_im);
        upload_corr(h_cdu_re, h_cdu_im, fg.d_corr_dGdu_re, fg.d_corr_dGdu_im);
        upload_corr(h_cdv_re, h_cdv_im, fg.d_corr_dGdv_re, fg.d_corr_dGdv_im);

        // Allocate per-face work buffers
        CUDA_CHECK(cudaMalloc(&fg.d_charges_re, fg.n_pts * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&fg.d_charges_im, fg.n_pts * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&fg.d_result_re,  fg.n_pts * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&fg.d_result_im,  fg.n_pts * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&fg.d_work, fg.grid_total * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&fg.d_charges_hat, fg.grid_total * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&fg.d_stage_re, fg.grid_total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&fg.d_stage_im, fg.grid_total * sizeof(float)));

        // Upload local_to_global mapping to GPU
        CUDA_CHECK(cudaMalloc(&fg.d_local_to_global, fg.n_pts * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(fg.d_local_to_global, fg.local_to_global.data(),
                              fg.n_pts * sizeof(int), cudaMemcpyHostToDevice));

        printf("  [SurfPFFT] Face %d: corr_nnz=%d (%.1f/pt), grid_mem=%.1fMB\n",
               f, fg.corr_nnz, (double)fg.corr_nnz / fg.n_pts,
               fg.grid_total * 8.0 * 4 / 1e6);  // 4 FP32 complex buffers
    }

    printf("  [SurfPFFT] Per-face init done: %.1fms\n", timer.elapsed_ms());
    timer.reset();

    // --- Step 3: Inter-face P2P setup ---
    // Sort points by face, build CSR face_offsets
    std::vector<int> sort_order(n_pts);
    for (int i = 0; i < n_pts; i++) sort_order[i] = i;
    std::sort(sort_order.begin(), sort_order.end(),
              [&](int a, int b) { return face_ids[a] < face_ids[b]; });

    face_offsets_host.resize(n_faces + 1, 0);
    for (int i = 0; i < n_pts; i++)
        face_offsets_host[face_ids[sort_order[i]] + 1]++;
    for (int f = 0; f < n_faces; f++)
        face_offsets_host[f+1] += face_offsets_host[f];

    // Upload FP32 points in sorted order
    std::vector<float> h_pts_f(n_pts * 3);
    for (int i = 0; i < n_pts; i++) {
        int gi = sort_order[i];
        h_pts_f[i*3+0] = (float)points[gi*3+0];
        h_pts_f[i*3+1] = (float)points[gi*3+1];
        h_pts_f[i*3+2] = (float)points[gi*3+2];
    }

    CUDA_CHECK(cudaMalloc(&d_pts_f, n_pts * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pts_f, h_pts_f.data(), n_pts * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_face_offsets, (n_faces+1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_face_offsets, face_offsets_host.data(),
                          (n_faces+1) * sizeof(int), cudaMemcpyHostToDevice));

    // Upload sort_order to GPU (persistent, used for charge sorting/unsorting)
    CUDA_CHECK(cudaMalloc(&d_sort_order, n_pts * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sort_order, sort_order.data(),
                          n_pts * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate P2P buffers
    CUDA_CHECK(cudaMalloc(&d_p2p_q_re,    n_pts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p2p_q_im,    n_pts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p2p_out_re,  n_pts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p2p_out_im,  n_pts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p2p_grad_re, n_pts * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p2p_grad_im, n_pts * 3 * sizeof(float)));

    // Global result buffers (FP64)
    CUDA_CHECK(cudaMalloc(&d_charges_re, n_pts * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_charges_im, n_pts * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result_re,  n_pts * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result_im,  n_pts * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_re,    n_pts * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_im,    n_pts * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pts, n_pts * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_pts, points, n_pts * 3 * sizeof(double), cudaMemcpyHostToDevice));

    // Pre-allocate host staging buffers (avoid heap alloc per evaluate)
    h_stage_re.resize(n_pts);
    h_stage_im.resize(n_pts);
    h_out_re.resize(n_pts);
    h_out_im.resize(n_pts);
    h_grad_re.resize(n_pts * 3);
    h_grad_im.resize(n_pts * 3);

    // Batched P2P buffers (FP32, packed — sized for batch8)
    CUDA_CHECK(cudaMalloc(&d_bp_q_re,    8 * n_pts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bp_q_im,    8 * n_pts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bp_pot_re,  8 * n_pts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bp_pot_im,  8 * n_pts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bp_grad_re, 6 * n_pts * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bp_grad_im, 6 * n_pts * 3 * sizeof(float)));

    // Batched global result buffers (FP64)
    for (int b = 0; b < 8; b++) {
        CUDA_CHECK(cudaMalloc(&d_bp_res_re[b], n_pts * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_bp_res_im[b], n_pts * sizeof(double)));
    }
    for (int b = 0; b < 6; b++) {
        CUDA_CHECK(cudaMalloc(&d_bp_grd_re[b], n_pts * 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_bp_grd_im[b], n_pts * 3 * sizeof(double)));
    }

    long long bp_mem = (long long)(8+8+6*3+6*3) * n_pts * sizeof(float)
                     + (long long)(8+8) * n_pts * sizeof(double)
                     + (long long)(6+6) * n_pts * 3 * sizeof(double);
    printf("  [SurfPFFT] Batched P2P buffers: %.1f MB\n", bp_mem / 1e6);

    initialized = true;

    printf("  [SurfPFFT] Total init: %.1fms\n", timer.elapsed_ms());

    // Print summary
    long long total_grid_mem = 0;
    for (int f = 0; f < n_faces; f++)
        if (faces[f].n_pts > 0)
            total_grid_mem += faces[f].grid_total * 8 * 4;  // 4 FP32 complex buffers
    printf("  [SurfPFFT] Total grid GPU memory: %.1f MB\n", total_grid_mem / 1e6);
}

// Anterpolate charges to FP32 staging grid, convert to FP32 complex, forward FFT
static void anterpolate_and_fft(FaceGrid& fg, int stencil_2d,
                                 const double* d_q_re, const double* d_q_im)
{
    int block = 256;
    int grid_blocks = (int)((fg.grid_total + block - 1) / block);

    // Zero FP32 staging grid (async on face stream)
    CUDA_CHECK(cudaMemsetAsync(fg.d_stage_re, 0, fg.grid_total * sizeof(float), fg.stream));
    CUDA_CHECK(cudaMemsetAsync(fg.d_stage_im, 0, fg.grid_total * sizeof(float), fg.stream));

    // Anterpolate with FP32 atomicAdd
    kernel_anterpolate_2d_f32<<<(fg.n_pts + block - 1) / block, block, 0, fg.stream>>>(
        d_q_re, d_q_im,
        fg.d_stencil_idx, fg.d_stencil_wt,
        fg.n_pts, stencil_2d,
        fg.d_stage_re, fg.d_stage_im);

    // Convert FP32 staging -> FP32 cuFFT complex
    kernel_f32_to_c2c<<<grid_blocks, block, 0, fg.stream>>>(
        fg.d_stage_re, fg.d_stage_im, fg.d_charges_hat, fg.grid_total);

    // Forward FFT -> charges_hat (FP32 C2C)
    CUFFT_CHECK(cufftSetStream(fg.plan_fwd, fg.stream));
    CUFFT_CHECK(cufftExecC2C(fg.plan_fwd, fg.d_charges_hat, fg.d_charges_hat, CUFFT_FORWARD));
}

// Apply kernel convolution using pre-computed charges_hat, + near correction
static void apply_kernel_and_correct(
    FaceGrid& fg, int stencil_2d,
    cufftComplex* d_kernel_hat,
    double* d_corr_re, double* d_corr_im,
    const double* d_q_re, const double* d_q_im,
    double* d_out_re, double* d_out_im)
{
    if (fg.n_pts == 0) return;
    int block = 256;
    float inv_N = 1.0f / fg.grid_total;

    // Pointwise multiply charges_hat * kernel_hat -> d_work (FP32)
    kernel_pw_mul_2d<<<(int)((fg.grid_total + block - 1) / block), block, 0, fg.stream>>>(
        fg.d_charges_hat, d_kernel_hat, fg.d_work, fg.grid_total, inv_N);

    // Inverse FFT (FP32 C2C)
    CUFFT_CHECK(cufftSetStream(fg.plan_inv, fg.stream));
    CUFFT_CHECK(cufftExecC2C(fg.plan_inv, fg.d_work, fg.d_work, CUFFT_INVERSE));

    // Interpolate (FP32 grid -> FP64 output)
    kernel_interpolate_2d<<<(fg.n_pts + block - 1) / block, block, 0, fg.stream>>>(
        fg.d_work,
        fg.d_stencil_idx, fg.d_stencil_wt,
        fg.n_pts, stencil_2d, d_out_re, d_out_im);

    // Near correction (FP64)
    if (fg.corr_nnz > 0) {
        kernel_near_corr_2d<<<(fg.n_pts + block - 1) / block, block, 0, fg.stream>>>(
            fg.d_corr_row_ptr, fg.d_corr_col_idx,
            d_corr_re, d_corr_im,
            d_q_re, d_q_im,
            d_out_re, d_out_im, fg.n_pts);
    }
}

// Full convolution: anterpolate + FFT + multiply + IFFT + interpolate + correction
static void convolve_face_2d(
    FaceGrid& fg, int stencil_2d,
    const double* d_q_re, const double* d_q_im,
    cufftComplex* d_kernel_hat,
    double* d_corr_re, double* d_corr_im,
    double* d_out_re, double* d_out_im)
{
    if (fg.n_pts == 0) return;
    anterpolate_and_fft(fg, stencil_2d, d_q_re, d_q_im);
    apply_kernel_and_correct(fg, stencil_2d, d_kernel_hat,
                             d_corr_re, d_corr_im, d_q_re, d_q_im,
                             d_out_re, d_out_im);
}

void HelmholtzSurfacePFFT::evaluate(const cdouble* charges, cdouble* result)
{
    int block = 256;

    // Split complex -> re/im and upload to GPU
    for (int i = 0; i < Nt; i++) {
        h_stage_re[i] = charges[i].real();
        h_stage_im[i] = charges[i].imag();
    }
    CUDA_CHECK(cudaMemcpy(d_charges_re, h_stage_re.data(), Nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_charges_im, h_stage_im.data(), Nt*sizeof(double), cudaMemcpyHostToDevice));

    // Zero global result
    CUDA_CHECK(cudaMemset(d_result_re, 0, Nt*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result_im, 0, Nt*sizeof(double)));

    // === Intra-face: 2D FFT per face (async on per-face streams) ===
    for (int f = 0; f < n_faces; f++) {
        FaceGrid& fg = faces[f];
        if (fg.n_pts == 0) continue;

        kernel_gather_face<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
            d_charges_re, d_charges_im, fg.d_local_to_global,
            fg.n_pts, fg.d_charges_re, fg.d_charges_im);

        CUDA_CHECK(cudaMemsetAsync(fg.d_result_re, 0, fg.n_pts*sizeof(double), fg.stream));
        CUDA_CHECK(cudaMemsetAsync(fg.d_result_im, 0, fg.n_pts*sizeof(double), fg.stream));

        convolve_face_2d(fg, stencil_2d,
                         fg.d_charges_re, fg.d_charges_im,
                         fg.d_G_hat,
                         fg.d_corr_G_re, fg.d_corr_G_im,
                         fg.d_result_re, fg.d_result_im);

        kernel_scatter_add<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
            fg.d_result_re, fg.d_result_im, fg.d_local_to_global,
            fg.n_pts, d_result_re, d_result_im);
    }

    // Sync all face streams before P2P
    CUDA_CHECK(cudaDeviceSynchronize());

    // === Inter-face P2P (single unified kernel, default stream) ===
    kernel_sort_charges_f32<<<(Nt+block-1)/block, block>>>(
        d_charges_re, d_charges_im, d_sort_order, Nt,
        d_p2p_q_re, d_p2p_q_im);
    CUDA_CHECK(cudaMemset(d_p2p_out_re, 0, Nt*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_p2p_out_im, 0, Nt*sizeof(float)));

    kernel_p2p_unified_pot_f32<<<(Nt+block-1)/block, block>>>(
        d_pts_f, d_p2p_q_re, d_p2p_q_im,
        d_p2p_out_re, d_p2p_out_im,
        Nt, d_face_offsets, n_faces,
        (float)k.real(), (float)k.imag());

    kernel_unsort_add_pot<<<(Nt+block-1)/block, block>>>(
        d_p2p_out_re, d_p2p_out_im, d_sort_order, Nt,
        d_result_re, d_result_im);

    // Download result
    CUDA_CHECK(cudaMemcpy(h_out_re.data(), d_result_re, Nt*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_im.data(), d_result_im, Nt*sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < Nt; i++)
        result[i] = cdouble(h_out_re[i], h_out_im[i]);
}

void HelmholtzSurfacePFFT::evaluate_pot_grad(const cdouble* charges,
                                              cdouble* pot_result, cdouble* grad_result)
{
    int block = 256;

    // Split complex -> re/im and upload to GPU (single H->D transfer)
    for (int i = 0; i < Nt; i++) {
        h_stage_re[i] = charges[i].real();
        h_stage_im[i] = charges[i].imag();
    }
    CUDA_CHECK(cudaMemcpy(d_charges_re, h_stage_re.data(), Nt*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_charges_im, h_stage_im.data(), Nt*sizeof(double), cudaMemcpyHostToDevice));

    // Zero global result and gradient on GPU
    CUDA_CHECK(cudaMemset(d_result_re, 0, Nt*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result_im, 0, Nt*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grad_re,   0, Nt*3*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grad_im,   0, Nt*3*sizeof(double)));

    // === Intra-face: 2D FFT per face (async on per-face streams) ===
    // Optimization: anterpolate+FFT once, reuse charges_hat for G/dGdu/dGdv
    for (int f = 0; f < n_faces; f++) {
        FaceGrid& fg = faces[f];
        if (fg.n_pts == 0) continue;

        // GPU gather: global charges -> per-face charges
        kernel_gather_face<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
            d_charges_re, d_charges_im, fg.d_local_to_global,
            fg.n_pts, fg.d_charges_re, fg.d_charges_im);

        // Anterpolate + forward FFT once -> charges_hat (reused 3 times)
        anterpolate_and_fft(fg, stencil_2d, fg.d_charges_re, fg.d_charges_im);

        // Potential: multiply + IFFT + interpolate + correction, scatter
        CUDA_CHECK(cudaMemsetAsync(fg.d_result_re, 0, fg.n_pts*sizeof(double), fg.stream));
        CUDA_CHECK(cudaMemsetAsync(fg.d_result_im, 0, fg.n_pts*sizeof(double), fg.stream));
        apply_kernel_and_correct(fg, stencil_2d, fg.d_G_hat,
                                 fg.d_corr_G_re, fg.d_corr_G_im,
                                 fg.d_charges_re, fg.d_charges_im,
                                 fg.d_result_re, fg.d_result_im);
        kernel_scatter_add<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
            fg.d_result_re, fg.d_result_im, fg.d_local_to_global,
            fg.n_pts, d_result_re, d_result_im);

        // Gradient u: reuse charges_hat
        CUDA_CHECK(cudaMemsetAsync(fg.d_result_re, 0, fg.n_pts*sizeof(double), fg.stream));
        CUDA_CHECK(cudaMemsetAsync(fg.d_result_im, 0, fg.n_pts*sizeof(double), fg.stream));
        apply_kernel_and_correct(fg, stencil_2d, fg.d_dGdu_hat,
                                 fg.d_corr_dGdu_re, fg.d_corr_dGdu_im,
                                 fg.d_charges_re, fg.d_charges_im,
                                 fg.d_result_re, fg.d_result_im);
        kernel_scatter_grad_component<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
            fg.d_result_re, fg.d_result_im, fg.d_local_to_global,
            fg.n_pts, fg.u_axis[0], fg.u_axis[1], fg.u_axis[2],
            d_grad_re, d_grad_im);

        // Gradient v: reuse charges_hat
        CUDA_CHECK(cudaMemsetAsync(fg.d_result_re, 0, fg.n_pts*sizeof(double), fg.stream));
        CUDA_CHECK(cudaMemsetAsync(fg.d_result_im, 0, fg.n_pts*sizeof(double), fg.stream));
        apply_kernel_and_correct(fg, stencil_2d, fg.d_dGdv_hat,
                                 fg.d_corr_dGdv_re, fg.d_corr_dGdv_im,
                                 fg.d_charges_re, fg.d_charges_im,
                                 fg.d_result_re, fg.d_result_im);
        kernel_scatter_grad_component<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
            fg.d_result_re, fg.d_result_im, fg.d_local_to_global,
            fg.n_pts, fg.v_axis[0], fg.v_axis[1], fg.v_axis[2],
            d_grad_re, d_grad_im);
    }

    // Sync all face streams before P2P
    CUDA_CHECK(cudaDeviceSynchronize());

    // === Inter-face P2P (single unified kernel, default stream) ===
    kernel_sort_charges_f32<<<(Nt+block-1)/block, block>>>(
        d_charges_re, d_charges_im, d_sort_order, Nt,
        d_p2p_q_re, d_p2p_q_im);
    CUDA_CHECK(cudaMemset(d_p2p_out_re, 0, Nt*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_p2p_out_im, 0, Nt*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_p2p_grad_re, 0, Nt*3*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_p2p_grad_im, 0, Nt*3*sizeof(float)));

    kernel_p2p_unified_pot_grad_f32<<<(Nt+block-1)/block, block>>>(
        d_pts_f, d_p2p_q_re, d_p2p_q_im,
        d_p2p_out_re, d_p2p_out_im,
        d_p2p_grad_re, d_p2p_grad_im,
        Nt, d_face_offsets, n_faces,
        (float)k.real(), (float)k.imag());

    kernel_unsort_add_pot<<<(Nt+block-1)/block, block>>>(
        d_p2p_out_re, d_p2p_out_im, d_sort_order, Nt,
        d_result_re, d_result_im);
    kernel_unsort_add_grad<<<(Nt+block-1)/block, block>>>(
        d_p2p_grad_re, d_p2p_grad_im, d_sort_order, Nt,
        d_grad_re, d_grad_im);

    // Download results (single D->H transfer for pot + grad)
    CUDA_CHECK(cudaMemcpy(h_out_re.data(), d_result_re, Nt*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_im.data(), d_result_im, Nt*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_re.data(), d_grad_re, Nt*3*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_im.data(), d_grad_im, Nt*3*sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < Nt; i++) {
        pot_result[i] = cdouble(h_out_re[i], h_out_im[i]);
        grad_result[i*3+0] = cdouble(h_grad_re[i*3+0], h_grad_im[i*3+0]);
        grad_result[i*3+1] = cdouble(h_grad_re[i*3+1], h_grad_im[i*3+1]);
        grad_result[i*3+2] = cdouble(h_grad_re[i*3+2], h_grad_im[i*3+2]);
    }
}

void HelmholtzSurfacePFFT::evaluate_batch4(
    const cdouble* charges0, const cdouble* charges1,
    const cdouble* charges2, const cdouble* charges3,
    cdouble* pot0, cdouble* pot1, cdouble* pot2, cdouble* pot3,
    cdouble* grad0, cdouble* grad1, cdouble* grad2)
{
    int block = 256;
    const cdouble* all_charges[4] = { charges0, charges1, charges2, charges3 };

    // Zero all batch4 result buffers
    for (int b = 0; b < 4; b++) {
        CUDA_CHECK(cudaMemset(d_bp_res_re[b], 0, Nt*sizeof(double)));
        CUDA_CHECK(cudaMemset(d_bp_res_im[b], 0, Nt*sizeof(double)));
    }
    for (int b = 0; b < 3; b++) {
        CUDA_CHECK(cudaMemset(d_bp_grd_re[b], 0, Nt*3*sizeof(double)));
        CUDA_CHECK(cudaMemset(d_bp_grd_im[b], 0, Nt*3*sizeof(double)));
    }

    // Process each of the 4 charge vectors through intra-face FFT
    for (int batch = 0; batch < 4; batch++) {
        // Upload charges to d_charges_re/im (reused between batches)
        for (int i = 0; i < Nt; i++) {
            h_stage_re[i] = all_charges[batch][i].real();
            h_stage_im[i] = all_charges[batch][i].imag();
        }
        CUDA_CHECK(cudaMemcpy(d_charges_re, h_stage_re.data(), Nt*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_charges_im, h_stage_im.data(), Nt*sizeof(double), cudaMemcpyHostToDevice));

        // Sort charges for P2P into batch slot [batch*Nt .. (batch+1)*Nt - 1]
        kernel_sort_charges_f32<<<(Nt+block-1)/block, block>>>(
            d_charges_re, d_charges_im, d_sort_order, Nt,
            d_bp_q_re + batch*Nt, d_bp_q_im + batch*Nt);

        // Intra-face processing (per-face streams)
        if (batch < 3) {
            // Batches 0-2: potential + gradient
            for (int f = 0; f < n_faces; f++) {
                FaceGrid& fg = faces[f];
                if (fg.n_pts == 0) continue;

                kernel_gather_face<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
                    d_charges_re, d_charges_im, fg.d_local_to_global,
                    fg.n_pts, fg.d_charges_re, fg.d_charges_im);

                anterpolate_and_fft(fg, stencil_2d, fg.d_charges_re, fg.d_charges_im);

                // Potential
                CUDA_CHECK(cudaMemsetAsync(fg.d_result_re, 0, fg.n_pts*sizeof(double), fg.stream));
                CUDA_CHECK(cudaMemsetAsync(fg.d_result_im, 0, fg.n_pts*sizeof(double), fg.stream));
                apply_kernel_and_correct(fg, stencil_2d, fg.d_G_hat,
                    fg.d_corr_G_re, fg.d_corr_G_im,
                    fg.d_charges_re, fg.d_charges_im,
                    fg.d_result_re, fg.d_result_im);
                kernel_scatter_add<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
                    fg.d_result_re, fg.d_result_im, fg.d_local_to_global,
                    fg.n_pts, d_bp_res_re[batch], d_bp_res_im[batch]);

                // Gradient u
                CUDA_CHECK(cudaMemsetAsync(fg.d_result_re, 0, fg.n_pts*sizeof(double), fg.stream));
                CUDA_CHECK(cudaMemsetAsync(fg.d_result_im, 0, fg.n_pts*sizeof(double), fg.stream));
                apply_kernel_and_correct(fg, stencil_2d, fg.d_dGdu_hat,
                    fg.d_corr_dGdu_re, fg.d_corr_dGdu_im,
                    fg.d_charges_re, fg.d_charges_im,
                    fg.d_result_re, fg.d_result_im);
                kernel_scatter_grad_component<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
                    fg.d_result_re, fg.d_result_im, fg.d_local_to_global,
                    fg.n_pts, fg.u_axis[0], fg.u_axis[1], fg.u_axis[2],
                    d_bp_grd_re[batch], d_bp_grd_im[batch]);

                // Gradient v
                CUDA_CHECK(cudaMemsetAsync(fg.d_result_re, 0, fg.n_pts*sizeof(double), fg.stream));
                CUDA_CHECK(cudaMemsetAsync(fg.d_result_im, 0, fg.n_pts*sizeof(double), fg.stream));
                apply_kernel_and_correct(fg, stencil_2d, fg.d_dGdv_hat,
                    fg.d_corr_dGdv_re, fg.d_corr_dGdv_im,
                    fg.d_charges_re, fg.d_charges_im,
                    fg.d_result_re, fg.d_result_im);
                kernel_scatter_grad_component<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
                    fg.d_result_re, fg.d_result_im, fg.d_local_to_global,
                    fg.n_pts, fg.v_axis[0], fg.v_axis[1], fg.v_axis[2],
                    d_bp_grd_re[batch], d_bp_grd_im[batch]);
            }
        } else {
            // Batch 3: potential only (no gradient)
            for (int f = 0; f < n_faces; f++) {
                FaceGrid& fg = faces[f];
                if (fg.n_pts == 0) continue;

                kernel_gather_face<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
                    d_charges_re, d_charges_im, fg.d_local_to_global,
                    fg.n_pts, fg.d_charges_re, fg.d_charges_im);

                // Anterpolate + FFT once, then apply G kernel only
                anterpolate_and_fft(fg, stencil_2d, fg.d_charges_re, fg.d_charges_im);

                CUDA_CHECK(cudaMemsetAsync(fg.d_result_re, 0, fg.n_pts*sizeof(double), fg.stream));
                CUDA_CHECK(cudaMemsetAsync(fg.d_result_im, 0, fg.n_pts*sizeof(double), fg.stream));
                apply_kernel_and_correct(fg, stencil_2d, fg.d_G_hat,
                    fg.d_corr_G_re, fg.d_corr_G_im,
                    fg.d_charges_re, fg.d_charges_im,
                    fg.d_result_re, fg.d_result_im);
                kernel_scatter_add<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
                    fg.d_result_re, fg.d_result_im, fg.d_local_to_global,
                    fg.n_pts, d_bp_res_re[3], d_bp_res_im[3]);
            }
        }

        // Sync all face streams before next batch (face buffers are reused)
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // === Batched inter-face P2P: 4 charge vectors in ONE kernel ===
    CUDA_CHECK(cudaMemset(d_bp_pot_re,  0, 4*Nt*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bp_pot_im,  0, 4*Nt*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bp_grad_re, 0, 3*Nt*3*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bp_grad_im, 0, 3*Nt*3*sizeof(float)));

    kernel_p2p_batch4_f32<<<(Nt+block-1)/block, block>>>(
        d_pts_f, d_bp_q_re, d_bp_q_im,
        d_bp_pot_re, d_bp_pot_im,
        d_bp_grad_re, d_bp_grad_im,
        Nt, d_face_offsets, n_faces,
        (float)k.real(), (float)k.imag());

    // Unsort P2P potential results for all 4 batches
    for (int b = 0; b < 4; b++) {
        kernel_unsort_add_pot<<<(Nt+block-1)/block, block>>>(
            d_bp_pot_re + b*Nt, d_bp_pot_im + b*Nt, d_sort_order, Nt,
            d_bp_res_re[b], d_bp_res_im[b]);
    }
    // Unsort P2P gradient results for batches 0-2
    for (int b = 0; b < 3; b++) {
        kernel_unsort_add_grad<<<(Nt+block-1)/block, block>>>(
            d_bp_grad_re + b*Nt*3, d_bp_grad_im + b*Nt*3, d_sort_order, Nt,
            d_bp_grd_re[b], d_bp_grd_im[b]);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Download all results
    cdouble* all_pots[4] = { pot0, pot1, pot2, pot3 };
    cdouble* all_grads[3] = { grad0, grad1, grad2 };

    for (int b = 0; b < 4; b++) {
        CUDA_CHECK(cudaMemcpy(h_out_re.data(), d_bp_res_re[b], Nt*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_out_im.data(), d_bp_res_im[b], Nt*sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < Nt; i++)
            all_pots[b][i] = cdouble(h_out_re[i], h_out_im[i]);
    }
    for (int b = 0; b < 3; b++) {
        CUDA_CHECK(cudaMemcpy(h_grad_re.data(), d_bp_grd_re[b], Nt*3*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_grad_im.data(), d_bp_grd_im[b], Nt*3*sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < Nt; i++) {
            all_grads[b][i*3+0] = cdouble(h_grad_re[i*3+0], h_grad_im[i*3+0]);
            all_grads[b][i*3+1] = cdouble(h_grad_re[i*3+1], h_grad_im[i*3+1]);
            all_grads[b][i*3+2] = cdouble(h_grad_re[i*3+2], h_grad_im[i*3+2]);
        }
    }
}

void HelmholtzSurfacePFFT::evaluate_batch8(
    const cdouble* charges[8],
    cdouble* pots[8],
    cdouble* grads[6])
{
    static int eval_call_count = 0;
    bool do_timing = (eval_call_count == 0);
    eval_call_count++;
    Timer t_fft, t_p2p, t_dl;

    int block = 256;

    // Zero all batch8 result buffers
    for (int b = 0; b < 8; b++) {
        CUDA_CHECK(cudaMemset(d_bp_res_re[b], 0, Nt*sizeof(double)));
        CUDA_CHECK(cudaMemset(d_bp_res_im[b], 0, Nt*sizeof(double)));
    }
    for (int b = 0; b < 6; b++) {
        CUDA_CHECK(cudaMemset(d_bp_grd_re[b], 0, Nt*3*sizeof(double)));
        CUDA_CHECK(cudaMemset(d_bp_grd_im[b], 0, Nt*3*sizeof(double)));
    }

    // Process each of 8 charge vectors through intra-face FFT
    // Gradient index: charges 0-2 → grads 0-2, charges 4-6 → grads 3-5
    // Charges 3,7 → pot only (no gradient)
    if (do_timing) t_fft.reset();
    for (int batch = 0; batch < 8; batch++) {
        // Upload charges (split complex → re/im)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nt; i++) {
            h_stage_re[i] = charges[batch][i].real();
            h_stage_im[i] = charges[batch][i].imag();
        }
        CUDA_CHECK(cudaMemcpy(d_charges_re, h_stage_re.data(), Nt*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_charges_im, h_stage_im.data(), Nt*sizeof(double), cudaMemcpyHostToDevice));

        // Sort charges for P2P into batch slot
        kernel_sort_charges_f32<<<(Nt+block-1)/block, block>>>(
            d_charges_re, d_charges_im, d_sort_order, Nt,
            d_bp_q_re + batch*Nt, d_bp_q_im + batch*Nt);

        // Determine if this batch needs gradient
        bool need_grad = (batch != 3 && batch != 7);
        // Map batch to gradient slot: 0→0, 1→1, 2→2, 4→3, 5→4, 6→5
        int grad_slot = -1;
        if (batch < 3) grad_slot = batch;
        else if (batch >= 4 && batch <= 6) grad_slot = batch - 1;

        if (need_grad) {
            for (int f = 0; f < n_faces; f++) {
                FaceGrid& fg = faces[f];
                if (fg.n_pts == 0) continue;

                kernel_gather_face<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
                    d_charges_re, d_charges_im, fg.d_local_to_global,
                    fg.n_pts, fg.d_charges_re, fg.d_charges_im);

                anterpolate_and_fft(fg, stencil_2d, fg.d_charges_re, fg.d_charges_im);

                // Potential
                CUDA_CHECK(cudaMemsetAsync(fg.d_result_re, 0, fg.n_pts*sizeof(double), fg.stream));
                CUDA_CHECK(cudaMemsetAsync(fg.d_result_im, 0, fg.n_pts*sizeof(double), fg.stream));
                apply_kernel_and_correct(fg, stencil_2d, fg.d_G_hat,
                    fg.d_corr_G_re, fg.d_corr_G_im,
                    fg.d_charges_re, fg.d_charges_im,
                    fg.d_result_re, fg.d_result_im);
                kernel_scatter_add<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
                    fg.d_result_re, fg.d_result_im, fg.d_local_to_global,
                    fg.n_pts, d_bp_res_re[batch], d_bp_res_im[batch]);

                // Gradient u
                CUDA_CHECK(cudaMemsetAsync(fg.d_result_re, 0, fg.n_pts*sizeof(double), fg.stream));
                CUDA_CHECK(cudaMemsetAsync(fg.d_result_im, 0, fg.n_pts*sizeof(double), fg.stream));
                apply_kernel_and_correct(fg, stencil_2d, fg.d_dGdu_hat,
                    fg.d_corr_dGdu_re, fg.d_corr_dGdu_im,
                    fg.d_charges_re, fg.d_charges_im,
                    fg.d_result_re, fg.d_result_im);
                kernel_scatter_grad_component<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
                    fg.d_result_re, fg.d_result_im, fg.d_local_to_global,
                    fg.n_pts, fg.u_axis[0], fg.u_axis[1], fg.u_axis[2],
                    d_bp_grd_re[grad_slot], d_bp_grd_im[grad_slot]);

                // Gradient v
                CUDA_CHECK(cudaMemsetAsync(fg.d_result_re, 0, fg.n_pts*sizeof(double), fg.stream));
                CUDA_CHECK(cudaMemsetAsync(fg.d_result_im, 0, fg.n_pts*sizeof(double), fg.stream));
                apply_kernel_and_correct(fg, stencil_2d, fg.d_dGdv_hat,
                    fg.d_corr_dGdv_re, fg.d_corr_dGdv_im,
                    fg.d_charges_re, fg.d_charges_im,
                    fg.d_result_re, fg.d_result_im);
                kernel_scatter_grad_component<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
                    fg.d_result_re, fg.d_result_im, fg.d_local_to_global,
                    fg.n_pts, fg.v_axis[0], fg.v_axis[1], fg.v_axis[2],
                    d_bp_grd_re[grad_slot], d_bp_grd_im[grad_slot]);
            }
        } else {
            // Pot only (batches 3 and 7)
            for (int f = 0; f < n_faces; f++) {
                FaceGrid& fg = faces[f];
                if (fg.n_pts == 0) continue;

                kernel_gather_face<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
                    d_charges_re, d_charges_im, fg.d_local_to_global,
                    fg.n_pts, fg.d_charges_re, fg.d_charges_im);

                anterpolate_and_fft(fg, stencil_2d, fg.d_charges_re, fg.d_charges_im);

                CUDA_CHECK(cudaMemsetAsync(fg.d_result_re, 0, fg.n_pts*sizeof(double), fg.stream));
                CUDA_CHECK(cudaMemsetAsync(fg.d_result_im, 0, fg.n_pts*sizeof(double), fg.stream));
                apply_kernel_and_correct(fg, stencil_2d, fg.d_G_hat,
                    fg.d_corr_G_re, fg.d_corr_G_im,
                    fg.d_charges_re, fg.d_charges_im,
                    fg.d_result_re, fg.d_result_im);
                kernel_scatter_add<<<(fg.n_pts+block-1)/block, block, 0, fg.stream>>>(
                    fg.d_result_re, fg.d_result_im, fg.d_local_to_global,
                    fg.n_pts, d_bp_res_re[batch], d_bp_res_im[batch]);
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (do_timing) printf("      [B8] FFT+upload: %.1fms\n", t_fft.elapsed_ms());

    // === Batched inter-face P2P: 8 charge vectors in ONE kernel ===
    if (do_timing) t_p2p.reset();
    CUDA_CHECK(cudaMemset(d_bp_pot_re,  0, 8*Nt*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bp_pot_im,  0, 8*Nt*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bp_grad_re, 0, 6*Nt*3*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bp_grad_im, 0, 6*Nt*3*sizeof(float)));

    kernel_p2p_batch8_f32<<<(Nt+block-1)/block, block>>>(
        d_pts_f, d_bp_q_re, d_bp_q_im,
        d_bp_pot_re, d_bp_pot_im,
        d_bp_grad_re, d_bp_grad_im,
        Nt, d_face_offsets, n_faces,
        (float)k.real(), (float)k.imag());

    // Unsort P2P potential results for all 8 batches
    for (int b = 0; b < 8; b++) {
        kernel_unsort_add_pot<<<(Nt+block-1)/block, block>>>(
            d_bp_pot_re + b*Nt, d_bp_pot_im + b*Nt, d_sort_order, Nt,
            d_bp_res_re[b], d_bp_res_im[b]);
    }
    // Unsort P2P gradient results for 6 grad batches
    for (int b = 0; b < 6; b++) {
        kernel_unsort_add_grad<<<(Nt+block-1)/block, block>>>(
            d_bp_grad_re + b*Nt*3, d_bp_grad_im + b*Nt*3, d_sort_order, Nt,
            d_bp_grd_re[b], d_bp_grd_im[b]);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    if (do_timing) printf("      [B8] P2P+unsort: %.1fms\n", t_p2p.elapsed_ms());

    // Download all results
    if (do_timing) t_dl.reset();
    for (int b = 0; b < 8; b++) {
        CUDA_CHECK(cudaMemcpy(h_out_re.data(), d_bp_res_re[b], Nt*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_out_im.data(), d_bp_res_im[b], Nt*sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < Nt; i++)
            pots[b][i] = cdouble(h_out_re[i], h_out_im[i]);
    }
    for (int b = 0; b < 6; b++) {
        CUDA_CHECK(cudaMemcpy(h_grad_re.data(), d_bp_grd_re[b], Nt*3*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_grad_im.data(), d_bp_grd_im[b], Nt*3*sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < Nt; i++) {
            grads[b][i*3+0] = cdouble(h_grad_re[i*3+0], h_grad_im[i*3+0]);
            grads[b][i*3+1] = cdouble(h_grad_re[i*3+1], h_grad_im[i*3+1]);
            grads[b][i*3+2] = cdouble(h_grad_re[i*3+2], h_grad_im[i*3+2]);
        }
    }
    if (do_timing) printf("      [B8] download: %.1fms\n", t_dl.elapsed_ms());
}

void HelmholtzSurfacePFFT::cleanup()
{
    if (!initialized) return;

    auto safe_free = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };

    for (int f = 0; f < n_faces; f++) {
        FaceGrid& fg = faces[f];
        if (fg.n_pts == 0) continue;
        if (fg.stream) { cudaStreamDestroy(fg.stream); fg.stream = 0; }
        cufftDestroy(fg.plan_fwd);
        cufftDestroy(fg.plan_inv);
        safe_free((void*&)fg.d_G_hat);
        safe_free((void*&)fg.d_dGdu_hat);
        safe_free((void*&)fg.d_dGdv_hat);
        safe_free((void*&)fg.d_stencil_idx);
        safe_free((void*&)fg.d_stencil_wt);
        safe_free((void*&)fg.d_corr_row_ptr);
        safe_free((void*&)fg.d_corr_col_idx);
        safe_free((void*&)fg.d_corr_G_re);
        safe_free((void*&)fg.d_corr_G_im);
        safe_free((void*&)fg.d_corr_dGdu_re);
        safe_free((void*&)fg.d_corr_dGdu_im);
        safe_free((void*&)fg.d_corr_dGdv_re);
        safe_free((void*&)fg.d_corr_dGdv_im);
        safe_free((void*&)fg.d_charges_re);
        safe_free((void*&)fg.d_charges_im);
        safe_free((void*&)fg.d_result_re);
        safe_free((void*&)fg.d_result_im);
        safe_free((void*&)fg.d_work);
        safe_free((void*&)fg.d_charges_hat);
        safe_free((void*&)fg.d_stage_re);
        safe_free((void*&)fg.d_stage_im);
        safe_free((void*&)fg.d_local_to_global);
    }

    safe_free((void*&)d_pts_f);
    safe_free((void*&)d_face_offsets);
    safe_free((void*&)d_p2p_q_re);
    safe_free((void*&)d_p2p_q_im);
    safe_free((void*&)d_p2p_out_re);
    safe_free((void*&)d_p2p_out_im);
    safe_free((void*&)d_p2p_grad_re);
    safe_free((void*&)d_p2p_grad_im);
    safe_free((void*&)d_charges_re);
    safe_free((void*&)d_charges_im);
    safe_free((void*&)d_result_re);
    safe_free((void*&)d_result_im);
    safe_free((void*&)d_grad_re);
    safe_free((void*&)d_grad_im);
    safe_free((void*&)d_pts);
    safe_free((void*&)d_sort_order);

    // Batched P2P buffers
    safe_free((void*&)d_bp_q_re);
    safe_free((void*&)d_bp_q_im);
    safe_free((void*&)d_bp_pot_re);
    safe_free((void*&)d_bp_pot_im);
    safe_free((void*&)d_bp_grad_re);
    safe_free((void*&)d_bp_grad_im);
    for (int b = 0; b < 8; b++) {
        safe_free((void*&)d_bp_res_re[b]);
        safe_free((void*&)d_bp_res_im[b]);
    }
    for (int b = 0; b < 6; b++) {
        safe_free((void*&)d_bp_grd_re[b]);
        safe_free((void*&)d_bp_grd_im[b]);
    }

    initialized = false;
}
