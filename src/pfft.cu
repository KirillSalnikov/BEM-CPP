// Pre-corrected FFT (pFFT) for Helmholtz Green's function
// Drop-in replacement for FMM with O(N log N) via FFT on regular grid

#include "pfft.h"
#include <cufft.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>

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

// Anterpolate charges from irregular points to regular grid (scatter + atomicAdd)
__global__ void kernel_anterpolate(
    const double* __restrict__ q_re,
    const double* __restrict__ q_im,
    const int*    __restrict__ stencil_idx,
    const double* __restrict__ stencil_wt,
    int N, int stencil_size,
    cufftDoubleComplex* __restrict__ grid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double qr = q_re[i], qi = q_im[i];
    const int*    idx = stencil_idx + (long long)i * stencil_size;
    const double* wt  = stencil_wt  + (long long)i * stencil_size;

    for (int s = 0; s < stencil_size; s++) {
        double w = wt[s];
        int gi = idx[s];
        atomicAdd(&grid[gi].x, w * qr);
        atomicAdd(&grid[gi].y, w * qi);
    }
}

// Interpolate from regular grid to irregular target points (gather)
__global__ void kernel_interpolate(
    const cufftDoubleComplex* __restrict__ grid,
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
        cufftDoubleComplex g = grid[idx[s]];
        vr += w * g.x;
        vi += w * g.y;
    }
    out_re[i] = vr;
    out_im[i] = vi;
}

// Pointwise complex multiply: c = a * b (element-wise)
__global__ void kernel_pointwise_mul(
    const cufftDoubleComplex* __restrict__ a,
    const cufftDoubleComplex* __restrict__ b,
    cufftDoubleComplex* __restrict__ c,
    long long N, double scale)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double ar = a[i].x, ai = a[i].y;
    double br = b[i].x, bi = b[i].y;
    c[i].x = (ar * br - ai * bi) * scale;
    c[i].y = (ar * bi + ai * br) * scale;
}

// Apply near-field correction (CSR sparse)
__global__ void kernel_near_correction(
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

// Zero a grid buffer
__global__ void kernel_zero_grid(cufftDoubleComplex* grid, long long N)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    grid[i].x = 0.0;
    grid[i].y = 0.0;
}

// ============================================================================
// Host helpers
// ============================================================================

// 1D Lagrange interpolation weights for point x among nodes x0, x0+h, ..., x0+p*h
static void lagrange_weights(double x, double x0, double h, int p1, double* w)
{
    // p1 = number of nodes = interp_order + 1
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

// Build near-list: for each target, find all sources within radius
// Returns CSR structure
static void build_near_list(const double* tgt, int Nt,
                            const double* src, int Ns,
                            double radius,
                            std::vector<int>& row_ptr,
                            std::vector<int>& col_idx)
{
    row_ptr.resize(Nt + 1);
    col_idx.clear();
    double r2 = radius * radius;

    // Simple O(N^2) but with spatial hashing for efficiency
    // For BEM, Nt = Ns = 2*N*Nq ~ 500k -> need spatial hash

    // Build spatial hash grid
    double bmin[3] = {1e30, 1e30, 1e30};
    double bmax[3] = {-1e30, -1e30, -1e30};
    for (int i = 0; i < Ns; i++) {
        for (int d = 0; d < 3; d++) {
            bmin[d] = std::min(bmin[d], src[i*3+d]);
            bmax[d] = std::max(bmax[d], src[i*3+d]);
        }
    }
    double cell = radius;
    int nx = std::max(1, (int)ceil((bmax[0]-bmin[0])/cell)) + 1;
    int ny = std::max(1, (int)ceil((bmax[1]-bmin[1])/cell)) + 1;
    int nz = std::max(1, (int)ceil((bmax[2]-bmin[2])/cell)) + 1;

    // Limit grid to avoid excessive memory
    long long ncells = (long long)nx * ny * nz;
    bool use_hash = (ncells < 10000000LL);

    if (use_hash) {
        std::vector<std::vector<int>> cells(ncells);
        for (int i = 0; i < Ns; i++) {
            int cx = (int)((src[i*3+0]-bmin[0])/cell);
            int cy = (int)((src[i*3+1]-bmin[1])/cell);
            int cz = (int)((src[i*3+2]-bmin[2])/cell);
            cx = std::max(0, std::min(cx, nx-1));
            cy = std::max(0, std::min(cy, ny-1));
            cz = std::max(0, std::min(cz, nz-1));
            cells[(long long)cx * ny * nz + cy * nz + cz].push_back(i);
        }

        row_ptr[0] = 0;
        for (int i = 0; i < Nt; i++) {
            double tx = tgt[i*3+0], ty = tgt[i*3+1], tz = tgt[i*3+2];
            int cx = (int)((tx-bmin[0])/cell);
            int cy = (int)((ty-bmin[1])/cell);
            int cz = (int)((tz-bmin[2])/cell);

            for (int dx = -1; dx <= 1; dx++) {
                int ccx = cx + dx;
                if (ccx < 0 || ccx >= nx) continue;
                for (int dy = -1; dy <= 1; dy++) {
                    int ccy = cy + dy;
                    if (ccy < 0 || ccy >= ny) continue;
                    for (int dz = -1; dz <= 1; dz++) {
                        int ccz = cz + dz;
                        if (ccz < 0 || ccz >= nz) continue;
                        auto& c = cells[(long long)ccx * ny * nz + ccy * nz + ccz];
                        for (int j : c) {
                            double dx2 = tx - src[j*3+0];
                            double dy2 = ty - src[j*3+1];
                            double dz2 = tz - src[j*3+2];
                            if (dx2*dx2 + dy2*dy2 + dz2*dz2 < r2 || i == j) {
                                col_idx.push_back(j);
                            }
                        }
                    }
                }
            }
            row_ptr[i+1] = (int)col_idx.size();
        }
    } else {
        // Fallback: O(N^2) brute force (should not happen for reasonable grids)
        row_ptr[0] = 0;
        for (int i = 0; i < Nt; i++) {
            double tx = tgt[i*3+0], ty = tgt[i*3+1], tz = tgt[i*3+2];
            for (int j = 0; j < Ns; j++) {
                double dx = tx - src[j*3+0];
                double dy = ty - src[j*3+1];
                double dz = tz - src[j*3+2];
                if (dx*dx + dy*dy + dz*dz < r2 || i == j)
                    col_idx.push_back(j);
            }
            row_ptr[i+1] = (int)col_idx.size();
        }
    }
}

// ============================================================================
// HelmholtzPFFT implementation
// ============================================================================

void HelmholtzPFFT::init(const double* targets, int n_tgt,
                          const double* sources, int n_src,
                          cdouble k_val, int digits, int /* max_leaf */)
{
    Timer timer;
    k = k_val;
    Nt = n_tgt;
    Ns = n_src;

    // Interpolation order: digits+1 gives good accuracy
    interp_p = std::max(2, digits);   // order 2 for 2 digits, 3 for 3 digits
    int p1 = interp_p + 1;           // nodes per dimension
    stencil = p1 * p1 * p1;

    printf("  [pFFT] N_tgt=%d, N_src=%d, k=(%.4f,%.4f), interp_order=%d\n",
           Nt, Ns, k.real(), k.imag(), interp_p);

    // --- Step 1: Determine grid ---
    double bmin[3] = {1e30, 1e30, 1e30};
    double bmax[3] = {-1e30, -1e30, -1e30};
    for (int i = 0; i < n_tgt; i++) {
        for (int d = 0; d < 3; d++) {
            bmin[d] = std::min(bmin[d], targets[i*3+d]);
            bmax[d] = std::max(bmax[d], targets[i*3+d]);
        }
    }
    for (int i = 0; i < n_src; i++) {
        for (int d = 0; d < 3; d++) {
            bmin[d] = std::min(bmin[d], sources[i*3+d]);
            bmax[d] = std::max(bmax[d], sources[i*3+d]);
        }
    }

    double diameter = 0;
    for (int d = 0; d < 3; d++)
        diameter = std::max(diameter, bmax[d] - bmin[d]);

    // Grid spacing: balance interpolation accuracy and grid size
    // For p-order Lagrange, error ~ O(h^{p+1}).
    // Choose h so that ~10-20 grid points span the particle per dimension
    // and h < lambda/(2*p) for wave resolution
    double lambda_val = 2.0 * M_PI / std::max(std::abs(k), 0.01);
    double h_wave = lambda_val / (2.0 * interp_p);
    double h_geom = diameter / 40.0;  // at least 40 points across particle
    h = std::min(h_wave, h_geom);
    h = std::max(h, diameter / 200.0);  // cap grid size at 200^3

    // Padding: interp_p+1 cells on each side
    double pad = (interp_p + 2) * h;
    for (int d = 0; d < 3; d++)
        origin[d] = bmin[d] - pad;

    Mx = (int)ceil((bmax[0] - origin[0] + pad) / h) + 1;
    My = (int)ceil((bmax[1] - origin[1] + pad) / h) + 1;
    Mz = (int)ceil((bmax[2] - origin[2] + pad) / h) + 1;

    // Round up to nice FFT sizes (7-smooth: factors of 2,3,5,7 only)
    auto round_fft = [](int n) {
        if (n <= 1) return 1;
        int best = 1;
        while (best < n) best *= 2;
        for (int p7 = 1; p7 <= best; p7 *= 7)
            for (int p5 = p7; p5 <= best; p5 *= 5)
                for (int p3 = p5; p3 <= best; p3 *= 3)
                    for (int p2 = p3; p2 <= best; p2 *= 2)
                        if (p2 >= n && p2 < best) best = p2;
        return best;
    };
    M2x = round_fft(2 * Mx);
    M2y = round_fft(2 * My);
    M2z = round_fft(2 * Mz);
    grid_total = (long long)M2x * M2y * M2z;

    printf("  [pFFT] Grid: %d×%d×%d (physical), %d×%d×%d (FFT), h=%.4f\n",
           Mx, My, Mz, M2x, M2y, M2z, h);
    printf("  [pFFT] Grid memory: %.1f MB per buffer\n",
           grid_total * 16.0 / 1e6);

    // --- Step 2: Precompute Green's function FFT ---
    printf("  [pFFT] Precomputing Green's function FFTs...\n");

    std::vector<cufftDoubleComplex> h_G(grid_total);
    std::vector<cufftDoubleComplex> h_dGdx(grid_total);
    std::vector<cufftDoubleComplex> h_dGdy(grid_total);
    std::vector<cufftDoubleComplex> h_dGdz(grid_total);

    for (int ix = 0; ix < M2x; ix++) {
        // Circulant embedding: zero zone for ix in [Mx, M2x-Mx]
        bool zx = (ix >= Mx && ix <= M2x - Mx);
        double dx = (ix < Mx) ? ix * h : (ix - M2x) * h;
        for (int iy = 0; iy < M2y; iy++) {
            bool zy = (iy >= My && iy <= M2y - My);
            double dy = (iy < My) ? iy * h : (iy - M2y) * h;
            for (int iz = 0; iz < M2z; iz++) {
                bool zz = (iz >= Mz && iz <= M2z - Mz);
                double dz = (iz < Mz) ? iz * h : (iz - M2z) * h;
                long long idx = (long long)ix * M2y * M2z + iy * M2z + iz;

                double R = sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-30 || zx || zy || zz) {
                    // Zero zone (circulant padding) or self-interaction
                    h_G[idx] = {0.0, 0.0};
                    h_dGdx[idx] = {0.0, 0.0};
                    h_dGdy[idx] = {0.0, 0.0};
                    h_dGdz[idx] = {0.0, 0.0};
                } else {
                    // G = exp(ikR) / (4*pi*R)
                    cdouble ikR = k * R;
                    cdouble expikR = std::exp(cdouble(0, 1) * ikR);
                    cdouble G = expikR * INV4PI / R;

                    h_G[idx] = {G.real(), G.imag()};

                    // dG/dx = (ik - 1/R) * G * dx/R
                    cdouble factor = (cdouble(0,1) * k - 1.0/R) * G / R;
                    cdouble gx = factor * dx;
                    cdouble gy = factor * dy;
                    cdouble gz = factor * dz;

                    h_dGdx[idx] = {gx.real(), gx.imag()};
                    h_dGdy[idx] = {gy.real(), gy.imag()};
                    h_dGdz[idx] = {gz.real(), gz.imag()};
                }
            }
        }
    }

    // Allocate GPU buffers for Green's FFTs
    CUDA_CHECK(cudaMalloc(&d_G_hat,    grid_total * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_dGdx_hat, grid_total * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_dGdy_hat, grid_total * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_dGdz_hat, grid_total * sizeof(cufftDoubleComplex)));

    // Create cuFFT plans
    CUFFT_CHECK(cufftPlan3d(&plan_fwd, M2x, M2y, M2z, CUFFT_Z2Z));
    CUFFT_CHECK(cufftPlan3d(&plan_inv, M2x, M2y, M2z, CUFFT_Z2Z));

    // Upload and transform Green's functions
    auto fft_green = [&](std::vector<cufftDoubleComplex>& h_data, cufftDoubleComplex* d_hat) {
        CUDA_CHECK(cudaMemcpy(d_hat, h_data.data(),
                              grid_total * sizeof(cufftDoubleComplex),
                              cudaMemcpyHostToDevice));
        CUFFT_CHECK(cufftExecZ2Z(plan_fwd, d_hat, d_hat, CUFFT_FORWARD));
    };

    fft_green(h_G, d_G_hat);
    fft_green(h_dGdx, d_dGdx_hat);
    fft_green(h_dGdy, d_dGdy_hat);
    fft_green(h_dGdz, d_dGdz_hat);

    printf("  [pFFT] Green's FFTs done: %.1fms\n", timer.elapsed_ms());
    timer.reset();

    // --- Step 3: Compute interpolation stencils ---
    printf("  [pFFT] Computing interpolation stencils...\n");

    std::vector<int>    h_src_idx(Ns * stencil);
    std::vector<double> h_src_wt(Ns * stencil);
    std::vector<int>    h_tgt_idx(Nt * stencil);
    std::vector<double> h_tgt_wt(Nt * stencil);

    auto compute_stencil = [&](const double* pts, int N,
                               std::vector<int>& idx_out,
                               std::vector<double>& wt_out) {
        double wx[8], wy[8], wz[8];  // max p+1 = 8
        for (int i = 0; i < N; i++) {
            double px = pts[i*3+0], py = pts[i*3+1], pz = pts[i*3+2];

            // Grid cell containing the point
            double fx = (px - origin[0]) / h;
            double fy = (py - origin[1]) / h;
            double fz = (pz - origin[2]) / h;

            // Stencil start: center the p+1 nodes around the point
            int ix0 = (int)floor(fx) - (interp_p - 1) / 2;
            int iy0 = (int)floor(fy) - (interp_p - 1) / 2;
            int iz0 = (int)floor(fz) - (interp_p - 1) / 2;

            // Clamp to grid
            ix0 = std::max(0, std::min(ix0, Mx - p1));
            iy0 = std::max(0, std::min(iy0, My - p1));
            iz0 = std::max(0, std::min(iz0, Mz - p1));

            // Compute 1D Lagrange weights
            lagrange_weights(px, origin[0] + ix0 * h, h, p1, wx);
            lagrange_weights(py, origin[1] + iy0 * h, h, p1, wy);
            lagrange_weights(pz, origin[2] + iz0 * h, h, p1, wz);

            // Fill stencil (tensor product)
            int s = 0;
            for (int a = 0; a < p1; a++) {
                for (int b = 0; b < p1; b++) {
                    for (int c = 0; c < p1; c++) {
                        int gix = ix0 + a;
                        int giy = iy0 + b;
                        int giz = iz0 + c;
                        // Linear index in doubled grid (physical part only)
                        long long gi = (long long)gix * M2y * M2z + giy * M2z + giz;
                        idx_out[(long long)i * stencil + s] = (int)gi;
                        wt_out[(long long)i * stencil + s] = wx[a] * wy[b] * wz[c];
                        s++;
                    }
                }
            }
        }
    };

    compute_stencil(sources, Ns, h_src_idx, h_src_wt);
    compute_stencil(targets, Nt, h_tgt_idx, h_tgt_wt);

    // Upload stencils to GPU
    CUDA_CHECK(cudaMalloc(&d_src_stencil_idx, (long long)Ns * stencil * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_src_stencil_wt,  (long long)Ns * stencil * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tgt_stencil_idx, (long long)Nt * stencil * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tgt_stencil_wt,  (long long)Nt * stencil * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_src_stencil_idx, h_src_idx.data(),
                          (long long)Ns * stencil * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_stencil_wt,  h_src_wt.data(),
                          (long long)Ns * stencil * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt_stencil_idx, h_tgt_idx.data(),
                          (long long)Nt * stencil * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt_stencil_wt,  h_tgt_wt.data(),
                          (long long)Nt * stencil * sizeof(double), cudaMemcpyHostToDevice));

    printf("  [pFFT] Stencils done: %.1fms\n", timer.elapsed_ms());
    timer.reset();

    // --- Step 4: Build near-field correction ---
    printf("  [pFFT] Building near-field correction...\n");

    double near_radius = (interp_p + 2) * h;  // correction radius
    std::vector<int> h_row_ptr, h_col_idx;
    build_near_list(targets, Nt, sources, Ns, near_radius, h_row_ptr, h_col_idx);
    corr_nnz = (int)h_col_idx.size();

    printf("  [pFFT] Near pairs: %d (%.1f per target, %.3f%% density)\n",
           corr_nnz, (double)corr_nnz / Nt,
           100.0 * corr_nnz / ((double)Nt * Ns));

    // Compute correction values: C[i,j] = G_exact(ri,rj) - G_grid(ri,rj)
    std::vector<double> h_cG_re(corr_nnz), h_cG_im(corr_nnz);
    std::vector<double> h_cdx_re(corr_nnz), h_cdx_im(corr_nnz);
    std::vector<double> h_cdy_re(corr_nnz), h_cdy_im(corr_nnz);
    std::vector<double> h_cdz_re(corr_nnz), h_cdz_im(corr_nnz);

    // Precompute Green's function values on local grid region (for grid-mediated G)
    // G_grid_local[di][dj][dk] for small di,dj,dk
    int grange = (int)ceil(near_radius / h) + interp_p + 1;  // max stencil displacement
    int gspan = 2 * grange + 1;
    long long gsize = (long long)gspan * gspan * gspan;
    std::vector<cdouble> G_local(gsize);
    std::vector<cdouble> dGdx_local(gsize), dGdy_local(gsize), dGdz_local(gsize);
    for (int di = -grange; di <= grange; di++) {
        for (int dj = -grange; dj <= grange; dj++) {
            for (int dk = -grange; dk <= grange; dk++) {
                double dx = di * h, dy = dj * h, dz = dk * h;
                double R = sqrt(dx*dx + dy*dy + dz*dz);
                long long li = (long long)(di+grange)*gspan*gspan + (dj+grange)*gspan + (dk+grange);
                if (R < 1e-30) {
                    G_local[li] = 0;
                    dGdx_local[li] = dGdy_local[li] = dGdz_local[li] = 0;
                } else {
                    cdouble expikR = std::exp(cdouble(0, 1) * k * R);
                    G_local[li] = expikR * INV4PI / R;
                    cdouble factor = (cdouble(0,1) * k - 1.0/R) * G_local[li] / R;
                    dGdx_local[li] = factor * dx;
                    dGdy_local[li] = factor * dy;
                    dGdz_local[li] = factor * dz;
                }
            }
        }
    }

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < Nt; i++) {
        for (int p = h_row_ptr[i]; p < h_row_ptr[i+1]; p++) {
            int j = h_col_idx[p];
            double tx = targets[i*3+0], ty = targets[i*3+1], tz = targets[i*3+2];
            double sx = sources[j*3+0], sy = sources[j*3+1], sz = sources[j*3+2];

            // Exact Green's function
            double ddx = tx - sx, ddy = ty - sy, ddz = tz - sz;
            double R = sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
            cdouble G_exact(0, 0), dGx_exact(0, 0), dGy_exact(0, 0), dGz_exact(0, 0);
            if (R > 1e-30) {
                cdouble expikR = std::exp(cdouble(0, 1) * k * R);
                G_exact = expikR * INV4PI / R;
                cdouble factor = (cdouble(0,1) * k - 1.0/R) * G_exact / R;
                dGx_exact = factor * ddx;
                dGy_exact = factor * ddy;
                dGz_exact = factor * ddz;
            }

            // Grid-mediated Green's function:
            // G_grid(i,j) = sum_a sum_b w_tgt[i,a] * G_grid[a-b] * w_src[j,b]
            cdouble G_grid(0, 0), dGx_grid(0, 0), dGy_grid(0, 0), dGz_grid(0, 0);

            int base_i = (long long)i * stencil;
            int base_j = (long long)j * stencil;

            for (int a = 0; a < stencil; a++) {
                double wa = h_tgt_wt[base_i + a];
                if (fabs(wa) < 1e-15) continue;
                int ga = h_tgt_idx[base_i + a];
                // Decompose ga into grid coordinates
                int ga_x = ga / (M2y * M2z);
                int ga_y = (ga % (M2y * M2z)) / M2z;
                int ga_z = ga % M2z;

                for (int b = 0; b < stencil; b++) {
                    double wb = h_src_wt[base_j + b];
                    if (fabs(wb) < 1e-15) continue;
                    int gb = h_src_idx[base_j + b];
                    int gb_x = gb / (M2y * M2z);
                    int gb_y = (gb % (M2y * M2z)) / M2z;
                    int gb_z = gb % M2z;

                    int di = ga_x - gb_x;
                    int dj = ga_y - gb_y;
                    int dk = ga_z - gb_z;

                    // Look up in local table
                    if (abs(di) <= grange && abs(dj) <= grange && abs(dk) <= grange) {
                        long long li = (long long)(di+grange)*gspan*gspan +
                                       (dj+grange)*gspan + (dk+grange);
                        double ww = wa * wb;
                        G_grid    += ww * G_local[li];
                        dGx_grid  += ww * dGdx_local[li];
                        dGy_grid  += ww * dGdy_local[li];
                        dGz_grid  += ww * dGdz_local[li];
                    }
                }
            }

            // Correction = exact - grid
            cdouble cG  = G_exact - G_grid;
            cdouble cdx = dGx_exact - dGx_grid;
            cdouble cdy = dGy_exact - dGy_grid;
            cdouble cdz = dGz_exact - dGz_grid;

            h_cG_re[p]  = cG.real();   h_cG_im[p]  = cG.imag();
            h_cdx_re[p] = cdx.real();  h_cdx_im[p] = cdx.imag();
            h_cdy_re[p] = cdy.real();  h_cdy_im[p] = cdy.imag();
            h_cdz_re[p] = cdz.real();  h_cdz_im[p] = cdz.imag();
        }
    }

    // Upload correction data to GPU
    CUDA_CHECK(cudaMalloc(&d_corr_row_ptr, (Nt+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_corr_col_idx, corr_nnz * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_corr_row_ptr, h_row_ptr.data(), (Nt+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_corr_col_idx, h_col_idx.data(), corr_nnz*sizeof(int), cudaMemcpyHostToDevice));

    auto upload_corr = [&](std::vector<double>& h_re, std::vector<double>& h_im,
                           double*& d_re, double*& d_im) {
        CUDA_CHECK(cudaMalloc(&d_re, corr_nnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_im, corr_nnz * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_re, h_re.data(), corr_nnz*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_im, h_im.data(), corr_nnz*sizeof(double), cudaMemcpyHostToDevice));
    };
    upload_corr(h_cG_re,  h_cG_im,  d_corr_G_re,    d_corr_G_im);
    upload_corr(h_cdx_re, h_cdx_im, d_corr_dGdx_re, d_corr_dGdx_im);
    upload_corr(h_cdy_re, h_cdy_im, d_corr_dGdy_re, d_corr_dGdy_im);
    upload_corr(h_cdz_re, h_cdz_im, d_corr_dGdz_re, d_corr_dGdz_im);

    printf("  [pFFT] Near-field correction done: %.1fms\n", timer.elapsed_ms());
    timer.reset();

    // --- Step 5: Allocate work buffers ---
    CUDA_CHECK(cudaMalloc(&d_work_a, grid_total * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_work_b, grid_total * sizeof(cufftDoubleComplex)));

    CUDA_CHECK(cudaMalloc(&d_charges_re, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_charges_im, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result_re,  Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result_im,  Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_re,    Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_im,    Nt * 3 * sizeof(double)));

    // Batch-2 buffers
    CUDA_CHECK(cudaMalloc(&d_charges2_re, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_charges2_im, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result2_re,  Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result2_im,  Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad2_re,    Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad2_im,    Nt * 3 * sizeof(double)));

    // Upload point positions
    CUDA_CHECK(cudaMalloc(&d_src_pts, Ns * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tgt_pts, Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_src_pts, sources, Ns*3*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt_pts, targets, Nt*3*sizeof(double), cudaMemcpyHostToDevice));

    initialized = true;
    printf("  [pFFT] Total init: %.1fms, GPU memory: ~%.1f MB\n",
           timer.elapsed_ms(),
           (4 * grid_total * 16.0 + 2 * grid_total * 16.0 +
            (long long)(Ns + Nt) * stencil * 12.0 +
            corr_nnz * 8.0 * 8 +
            (Ns + Nt) * (5 * 8.0 + 3 * 8.0 * 2)) / 1e6);
}

// Core: convolve charges with kernel and add near-field correction
void HelmholtzPFFT::convolve_and_correct(
    const double* d_q_re, const double* d_q_im,
    const cufftDoubleComplex* d_kernel_hat,
    double* d_out_re, double* d_out_im)
{
    int block = 256;
    double inv_N = 1.0 / grid_total;  // FFT normalization

    // 1. Zero grid
    kernel_zero_grid<<<((int)((grid_total+block-1)/block)), block>>>(d_work_a, grid_total);

    // 2. Anterpolate: scatter charges to grid
    kernel_anterpolate<<<(Ns+block-1)/block, block>>>(
        d_q_re, d_q_im,
        d_src_stencil_idx, d_src_stencil_wt,
        Ns, stencil, d_work_a);

    // 3. Forward FFT
    CUFFT_CHECK(cufftExecZ2Z(plan_fwd, d_work_a, d_work_a, CUFFT_FORWARD));

    // 4. Pointwise multiply by kernel
    kernel_pointwise_mul<<<((int)((grid_total+block-1)/block)), block>>>(
        d_work_a, d_kernel_hat, d_work_b, grid_total, inv_N);

    // 5. Inverse FFT
    CUFFT_CHECK(cufftExecZ2Z(plan_inv, d_work_b, d_work_b, CUFFT_INVERSE));

    // 6. Interpolate: gather from grid to target points
    kernel_interpolate<<<(Nt+block-1)/block, block>>>(
        d_work_b,
        d_tgt_stencil_idx, d_tgt_stencil_wt,
        Nt, stencil, d_out_re, d_out_im);

    // 7. Near-field correction
    if (corr_nnz > 0) {
        kernel_near_correction<<<(Nt+block-1)/block, block>>>(
            d_corr_row_ptr, d_corr_col_idx,
            (d_kernel_hat == d_G_hat) ? d_corr_G_re :
            (d_kernel_hat == d_dGdx_hat) ? d_corr_dGdx_re :
            (d_kernel_hat == d_dGdy_hat) ? d_corr_dGdy_re : d_corr_dGdz_re,
            (d_kernel_hat == d_G_hat) ? d_corr_G_im :
            (d_kernel_hat == d_dGdx_hat) ? d_corr_dGdx_im :
            (d_kernel_hat == d_dGdy_hat) ? d_corr_dGdy_im : d_corr_dGdz_im,
            d_q_re, d_q_im,
            d_out_re, d_out_im, Nt);
    }
}

// ============================================================================
// Public evaluate methods
// ============================================================================

void HelmholtzPFFT::evaluate(const cdouble* charges, cdouble* result)
{
    // Split complex -> real/imag and upload
    std::vector<double> h_re(Ns), h_im(Ns);
    for (int i = 0; i < Ns; i++) {
        h_re[i] = charges[i].real();
        h_im[i] = charges[i].imag();
    }
    CUDA_CHECK(cudaMemcpy(d_charges_re, h_re.data(), Ns*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_charges_im, h_im.data(), Ns*sizeof(double), cudaMemcpyHostToDevice));

    // Zero result
    CUDA_CHECK(cudaMemset(d_result_re, 0, Nt*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result_im, 0, Nt*sizeof(double)));

    // Convolve with G
    convolve_and_correct(d_charges_re, d_charges_im, d_G_hat, d_result_re, d_result_im);

    // Download
    std::vector<double> r_re(Nt), r_im(Nt);
    CUDA_CHECK(cudaMemcpy(r_re.data(), d_result_re, Nt*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r_im.data(), d_result_im, Nt*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < Nt; i++)
        result[i] = cdouble(r_re[i], r_im[i]);
}

void HelmholtzPFFT::evaluate_gradient(const cdouble* charges, cdouble* grad_result)
{
    std::vector<double> h_re(Ns), h_im(Ns);
    for (int i = 0; i < Ns; i++) {
        h_re[i] = charges[i].real();
        h_im[i] = charges[i].imag();
    }
    CUDA_CHECK(cudaMemcpy(d_charges_re, h_re.data(), Ns*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_charges_im, h_im.data(), Ns*sizeof(double), cudaMemcpyHostToDevice));

    // Allocate temp buffers for each gradient component
    CUDA_CHECK(cudaMemset(d_grad_re, 0, Nt*3*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grad_im, 0, Nt*3*sizeof(double)));

    // 3 convolutions: dG/dx, dG/dy, dG/dz
    convolve_and_correct(d_charges_re, d_charges_im, d_dGdx_hat,
                         d_grad_re, d_grad_im);              // offset 0: gx
    convolve_and_correct(d_charges_re, d_charges_im, d_dGdy_hat,
                         d_grad_re + Nt, d_grad_im + Nt);    // offset Nt: gy
    convolve_and_correct(d_charges_re, d_charges_im, d_dGdz_hat,
                         d_grad_re + 2*Nt, d_grad_im + 2*Nt); // offset 2*Nt: gz

    // Download (layout: [gx0,gx1,...,gxN, gy0,...gyN, gz0,...gzN])
    // BEM expects interleaved [gx0,gy0,gz0, gx1,gy1,gz1, ...]
    std::vector<double> g_re(Nt*3), g_im(Nt*3);
    CUDA_CHECK(cudaMemcpy(g_re.data(), d_grad_re, Nt*3*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_im.data(), d_grad_im, Nt*3*sizeof(double), cudaMemcpyDeviceToHost));

    // Transpose from [comp][point] to [point][comp]
    for (int i = 0; i < Nt; i++) {
        grad_result[i*3+0] = cdouble(g_re[i],      g_im[i]);        // gx
        grad_result[i*3+1] = cdouble(g_re[Nt+i],   g_im[Nt+i]);    // gy
        grad_result[i*3+2] = cdouble(g_re[2*Nt+i], g_im[2*Nt+i]);  // gz
    }
}

void HelmholtzPFFT::evaluate_pot_grad(const cdouble* charges,
                                       cdouble* pot_result, cdouble* grad_result)
{
    std::vector<double> h_re(Ns), h_im(Ns);
    for (int i = 0; i < Ns; i++) {
        h_re[i] = charges[i].real();
        h_im[i] = charges[i].imag();
    }
    CUDA_CHECK(cudaMemcpy(d_charges_re, h_re.data(), Ns*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_charges_im, h_im.data(), Ns*sizeof(double), cudaMemcpyHostToDevice));

    // Potential
    CUDA_CHECK(cudaMemset(d_result_re, 0, Nt*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result_im, 0, Nt*sizeof(double)));
    convolve_and_correct(d_charges_re, d_charges_im, d_G_hat, d_result_re, d_result_im);

    std::vector<double> r_re(Nt), r_im(Nt);
    CUDA_CHECK(cudaMemcpy(r_re.data(), d_result_re, Nt*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r_im.data(), d_result_im, Nt*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < Nt; i++)
        pot_result[i] = cdouble(r_re[i], r_im[i]);

    // Gradient (3 convolutions)
    CUDA_CHECK(cudaMemset(d_grad_re, 0, Nt*3*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grad_im, 0, Nt*3*sizeof(double)));

    convolve_and_correct(d_charges_re, d_charges_im, d_dGdx_hat,
                         d_grad_re, d_grad_im);
    convolve_and_correct(d_charges_re, d_charges_im, d_dGdy_hat,
                         d_grad_re + Nt, d_grad_im + Nt);
    convolve_and_correct(d_charges_re, d_charges_im, d_dGdz_hat,
                         d_grad_re + 2*Nt, d_grad_im + 2*Nt);

    std::vector<double> g_re(Nt*3), g_im(Nt*3);
    CUDA_CHECK(cudaMemcpy(g_re.data(), d_grad_re, Nt*3*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_im.data(), d_grad_im, Nt*3*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < Nt; i++) {
        grad_result[i*3+0] = cdouble(g_re[i],      g_im[i]);
        grad_result[i*3+1] = cdouble(g_re[Nt+i],   g_im[Nt+i]);
        grad_result[i*3+2] = cdouble(g_re[2*Nt+i], g_im[2*Nt+i]);
    }
}

void HelmholtzPFFT::evaluate_batch2(const cdouble* charges1, const cdouble* charges2,
                                     cdouble* result1, cdouble* result2)
{
    // For now: two separate evaluations.
    // TODO: fuse anterpolation and share FFT when possible
    evaluate(charges1, result1);
    evaluate(charges2, result2);
}

void HelmholtzPFFT::evaluate_pot_grad_batch2(
    const cdouble* charges1, const cdouble* charges2,
    cdouble* pot1, cdouble* grad1,
    cdouble* pot2, cdouble* grad2)
{
    evaluate_pot_grad(charges1, pot1, grad1);
    evaluate_pot_grad(charges2, pot2, grad2);
}

void HelmholtzPFFT::cleanup()
{
    if (!initialized) return;

    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);

    auto safe_free = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    safe_free((void*&)d_G_hat);
    safe_free((void*&)d_dGdx_hat);
    safe_free((void*&)d_dGdy_hat);
    safe_free((void*&)d_dGdz_hat);
    safe_free((void*&)d_src_stencil_idx);
    safe_free((void*&)d_src_stencil_wt);
    safe_free((void*&)d_tgt_stencil_idx);
    safe_free((void*&)d_tgt_stencil_wt);
    safe_free((void*&)d_corr_row_ptr);
    safe_free((void*&)d_corr_col_idx);
    safe_free((void*&)d_corr_G_re);
    safe_free((void*&)d_corr_G_im);
    safe_free((void*&)d_corr_dGdx_re);
    safe_free((void*&)d_corr_dGdx_im);
    safe_free((void*&)d_corr_dGdy_re);
    safe_free((void*&)d_corr_dGdy_im);
    safe_free((void*&)d_corr_dGdz_re);
    safe_free((void*&)d_corr_dGdz_im);
    safe_free((void*&)d_work_a);
    safe_free((void*&)d_work_b);
    safe_free((void*&)d_charges_re);
    safe_free((void*&)d_charges_im);
    safe_free((void*&)d_result_re);
    safe_free((void*&)d_result_im);
    safe_free((void*&)d_grad_re);
    safe_free((void*&)d_grad_im);
    safe_free((void*&)d_charges2_re);
    safe_free((void*&)d_charges2_im);
    safe_free((void*&)d_result2_re);
    safe_free((void*&)d_result2_im);
    safe_free((void*&)d_grad2_re);
    safe_free((void*&)d_grad2_im);
    safe_free((void*&)d_src_pts);
    safe_free((void*&)d_tgt_pts);

    initialized = false;
}
