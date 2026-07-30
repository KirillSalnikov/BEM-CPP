// Pre-corrected FFT (pFFT) for Helmholtz Green's function
// Drop-in replacement for FMM with O(N log N) via FFT on regular grid

#include "pfft.h"
#include <cufft.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#define CUFFT_CHECK(call) do { \
    cufftResult err = (call); \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error at %s:%d: %d\n", __FILE__, __LINE__, (int)err); \
        exit(1); \
    } \
} while(0)

#ifdef BEM_PFFT_FP32
static constexpr cufftType PFFT_CUFFT_TYPE = CUFFT_C2C;

static cufftResult pfft_execute(
    cufftHandle plan, PfftComplex* data, int direction)
{
    return cufftExecC2C(plan, data, data, direction);
}
#else
static constexpr cufftType PFFT_CUFFT_TYPE = CUFFT_Z2Z;

static cufftResult pfft_execute(
    cufftHandle plan, PfftComplex* data, int direction)
{
    return cufftExecZ2Z(plan, data, data, direction);
}
#endif

static PfftComplex pfft_complex(double real, double imaginary)
{
    PfftComplex value;
    value.x = static_cast<decltype(value.x)>(real);
    value.y = static_cast<decltype(value.y)>(imaginary);
    return value;
}

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
    PfftComplex* __restrict__ grid)
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
    const PfftComplex* __restrict__ grid,
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
        PfftComplex g = grid[idx[s]];
        vr += w * g.x;
        vi += w * g.y;
    }
    out_re[i] = vr;
    out_im[i] = vi;
}

// Pointwise complex multiply: c = a * b (element-wise)
__global__ void kernel_pointwise_mul(
    const PfftComplex* __restrict__ a,
    const PfftComplex* __restrict__ b,
    PfftComplex* __restrict__ c,
    long long N, double scale)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double ar = a[i].x, ai = a[i].y;
    double br = b[i].x, bi = b[i].y;
    c[i].x = (ar * br - ai * bi) * scale;
    c[i].y = (ar * bi + ai * br) * scale;
}

__device__ inline void add_complex_product(
    PfftComplex left,
    PfftComplex right,
    double coefficient,
    double& real,
    double& imaginary)
{
    real += coefficient *
        (left.x * right.x - left.y * right.y);
    imaginary += coefficient *
        (left.x * right.y + left.y * right.x);
}

__device__ inline void add_split_complex_product(
    float kernel_real,
    float kernel_imaginary,
    double charge_real,
    double charge_imaginary,
    double coefficient,
    double& real,
    double& imaginary)
{
    real += coefficient *
        (kernel_real * charge_real -
         kernel_imaginary * charge_imaginary);
    imaginary += coefficient *
        (kernel_real * charge_imaginary +
         kernel_imaginary * charge_real);
}

// Antisymmetric gradient combinations in xy, xz, yz order.
__global__ void kernel_curl_spectrum(
    const PfftComplex* __restrict__ spectra,
    const PfftComplex* __restrict__ gx,
    const PfftComplex* __restrict__ gy,
    const PfftComplex* __restrict__ gz,
    PfftComplex* __restrict__ output,
    long long n,
    int component,
    double scale)
{
    const long long i =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const PfftComplex qx = spectra[i];
    const PfftComplex qy = spectra[n + i];
    const PfftComplex qz = spectra[2 * n + i];
    double real = 0.0;
    double imaginary = 0.0;
    if (component == 0) {
        add_complex_product(gy[i], qx, 1.0, real, imaginary);
        add_complex_product(gx[i], qy, -1.0, real, imaginary);
    } else if (component == 1) {
        add_complex_product(gz[i], qx, 1.0, real, imaginary);
        add_complex_product(gx[i], qz, -1.0, real, imaginary);
    } else {
        add_complex_product(gz[i], qy, 1.0, real, imaginary);
        add_complex_product(gy[i], qz, -1.0, real, imaginary);
    }
    output[i].x = scale * real;
    output[i].y = scale * imaginary;
}

// H*q - trace(H)*q, where H is the Hessian convolution tensor.
__global__ void kernel_hessian_action_spectrum(
    const PfftComplex* __restrict__ spectra,
    const PfftComplex* __restrict__ hxx,
    const PfftComplex* __restrict__ hxy,
    const PfftComplex* __restrict__ hxz,
    const PfftComplex* __restrict__ hyy,
    const PfftComplex* __restrict__ hyz,
    const PfftComplex* __restrict__ hzz,
    PfftComplex* __restrict__ output,
    long long n,
    int component,
    double scale)
{
    const long long i =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const PfftComplex qx = spectra[i];
    const PfftComplex qy = spectra[n + i];
    const PfftComplex qz = spectra[2 * n + i];
    double real = 0.0;
    double imaginary = 0.0;
    if (component == 0) {
        add_complex_product(hyy[i], qx, -1.0, real, imaginary);
        add_complex_product(hzz[i], qx, -1.0, real, imaginary);
        add_complex_product(hxy[i], qy, 1.0, real, imaginary);
        add_complex_product(hxz[i], qz, 1.0, real, imaginary);
    } else if (component == 1) {
        add_complex_product(hxy[i], qx, 1.0, real, imaginary);
        add_complex_product(hxx[i], qy, -1.0, real, imaginary);
        add_complex_product(hzz[i], qy, -1.0, real, imaginary);
        add_complex_product(hyz[i], qz, 1.0, real, imaginary);
    } else {
        add_complex_product(hxz[i], qx, 1.0, real, imaginary);
        add_complex_product(hyz[i], qy, 1.0, real, imaginary);
        add_complex_product(hxx[i], qz, -1.0, real, imaginary);
        add_complex_product(hyy[i], qz, -1.0, real, imaginary);
    }
    output[i].x = scale * real;
    output[i].y = scale * imaginary;
}

__global__ void kernel_curl_near_correction(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ dx_re,
    const float* __restrict__ dx_im,
    const float* __restrict__ dy_re,
    const float* __restrict__ dy_im,
    const float* __restrict__ dz_re,
    const float* __restrict__ dz_im,
    const double* __restrict__ qx_re,
    const double* __restrict__ qx_im,
    const double* __restrict__ qy_re,
    const double* __restrict__ qy_im,
    const double* __restrict__ qz_re,
    const double* __restrict__ qz_im,
    double* __restrict__ out_re,
    double* __restrict__ out_im,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double real[3] = {0.0, 0.0, 0.0};
    double imaginary[3] = {0.0, 0.0, 0.0};
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++) {
        const int j = col_idx[p];
        add_split_complex_product(
            dy_re[p], dy_im[p], qx_re[j], qx_im[j],
            1.0, real[0], imaginary[0]);
        add_split_complex_product(
            dx_re[p], dx_im[p], qy_re[j], qy_im[j],
            -1.0, real[0], imaginary[0]);
        add_split_complex_product(
            dz_re[p], dz_im[p], qx_re[j], qx_im[j],
            1.0, real[1], imaginary[1]);
        add_split_complex_product(
            dx_re[p], dx_im[p], qz_re[j], qz_im[j],
            -1.0, real[1], imaginary[1]);
        add_split_complex_product(
            dz_re[p], dz_im[p], qy_re[j], qy_im[j],
            1.0, real[2], imaginary[2]);
        add_split_complex_product(
            dy_re[p], dy_im[p], qz_re[j], qz_im[j],
            -1.0, real[2], imaginary[2]);
    }
    for (int component = 0; component < 3; component++) {
        out_re[component * n + i] += real[component];
        out_im[component * n + i] += imaginary[component];
    }
}

__global__ void kernel_hessian_action_near_correction(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ hxx_re,
    const float* __restrict__ hxx_im,
    const float* __restrict__ hxy_re,
    const float* __restrict__ hxy_im,
    const float* __restrict__ hxz_re,
    const float* __restrict__ hxz_im,
    const float* __restrict__ hyy_re,
    const float* __restrict__ hyy_im,
    const float* __restrict__ hyz_re,
    const float* __restrict__ hyz_im,
    const float* __restrict__ hzz_re,
    const float* __restrict__ hzz_im,
    const double* __restrict__ qx_re,
    const double* __restrict__ qx_im,
    const double* __restrict__ qy_re,
    const double* __restrict__ qy_im,
    const double* __restrict__ qz_re,
    const double* __restrict__ qz_im,
    double* __restrict__ out_re,
    double* __restrict__ out_im,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double real[3] = {0.0, 0.0, 0.0};
    double imaginary[3] = {0.0, 0.0, 0.0};
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++) {
        const int j = col_idx[p];
        add_split_complex_product(
            hyy_re[p], hyy_im[p], qx_re[j], qx_im[j],
            -1.0, real[0], imaginary[0]);
        add_split_complex_product(
            hzz_re[p], hzz_im[p], qx_re[j], qx_im[j],
            -1.0, real[0], imaginary[0]);
        add_split_complex_product(
            hxy_re[p], hxy_im[p], qy_re[j], qy_im[j],
            1.0, real[0], imaginary[0]);
        add_split_complex_product(
            hxz_re[p], hxz_im[p], qz_re[j], qz_im[j],
            1.0, real[0], imaginary[0]);

        add_split_complex_product(
            hxy_re[p], hxy_im[p], qx_re[j], qx_im[j],
            1.0, real[1], imaginary[1]);
        add_split_complex_product(
            hxx_re[p], hxx_im[p], qy_re[j], qy_im[j],
            -1.0, real[1], imaginary[1]);
        add_split_complex_product(
            hzz_re[p], hzz_im[p], qy_re[j], qy_im[j],
            -1.0, real[1], imaginary[1]);
        add_split_complex_product(
            hyz_re[p], hyz_im[p], qz_re[j], qz_im[j],
            1.0, real[1], imaginary[1]);

        add_split_complex_product(
            hxz_re[p], hxz_im[p], qx_re[j], qx_im[j],
            1.0, real[2], imaginary[2]);
        add_split_complex_product(
            hyz_re[p], hyz_im[p], qy_re[j], qy_im[j],
            1.0, real[2], imaginary[2]);
        add_split_complex_product(
            hxx_re[p], hxx_im[p], qz_re[j], qz_im[j],
            -1.0, real[2], imaginary[2]);
        add_split_complex_product(
            hyy_re[p], hyy_im[p], qz_re[j], qz_im[j],
            -1.0, real[2], imaginary[2]);
    }
    for (int component = 0; component < 3; component++) {
        out_re[component * n + i] += real[component];
        out_im[component * n + i] += imaginary[component];
    }
}

// Apply near-field correction (CSR sparse)
__global__ void kernel_near_correction(
    const int*    __restrict__ row_ptr,
    const int*    __restrict__ col_idx,
    const float*  __restrict__ corr_re,
    const float*  __restrict__ corr_im,
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
__global__ void kernel_zero_grid(PfftComplex* grid, long long N)
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

double HelmholtzPFFT::grid_spacing_for_diameter(
    double diameter, cdouble wave_number,
    int interpolation_order)
{
    double geometry_cells = 40.0;
    const char* geometry_cells_env =
        std::getenv("BEM_PFFT_GRID_CELLS");
    if (geometry_cells_env) {
        geometry_cells = std::atof(geometry_cells_env);
        if (geometry_cells < 16.0 || geometry_cells > 200.0) {
            std::fprintf(
                stderr,
                "BEM_PFFT_GRID_CELLS must be in [16,200]\n");
            std::exit(2);
        }
    }
    const double lambda =
        2.0 * M_PI / std::max(std::abs(wave_number), 0.01);
    const double h_wave =
        lambda / (2.0 * interpolation_order);
    const double h_geometry = diameter / geometry_cells;
    return std::max(
        std::min(h_wave, h_geometry),
        diameter / 200.0);
}

void HelmholtzPFFT::init(const double* targets, int n_tgt,
                          const double* sources, int n_src,
                          cdouble k_val, int digits, int /* max_leaf */,
                          double grid_spacing,
                          double correction_radius_cells)
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
    h = grid_spacing > 0.0
        ? grid_spacing
        : grid_spacing_for_diameter(diameter, k, interp_p);

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
           grid_total * sizeof(PfftComplex) / 1e6);

    // --- Step 2: Precompute Green's function FFT ---
    printf("  [pFFT] Precomputing Green's function FFTs...\n");

    std::vector<PfftComplex> h_G(grid_total);
    std::vector<PfftComplex> h_dGdx(grid_total);
    std::vector<PfftComplex> h_dGdy(grid_total);
    std::vector<PfftComplex> h_dGdz(grid_total);
    std::array<std::vector<PfftComplex>, 6> h_d2G;
    for (int component = 0; component < 6; component++)
        h_d2G[component].resize(grid_total);

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
                    for (int component = 0; component < 6; component++)
                        h_d2G[component][idx] = {0.0, 0.0};
                } else {
                    // G = exp(ikR) / (4*pi*R)
                    cdouble ikR = k * R;
                    cdouble expikR = std::exp(cdouble(0, 1) * ikR);
                    cdouble G = expikR * INV4PI / R;

                    h_G[idx] = pfft_complex(G.real(), G.imag());

                    // dG/dx = (ik - 1/R) * G * dx/R
                    cdouble factor = (cdouble(0,1) * k - 1.0/R) * G / R;
                    cdouble gx = factor * dx;
                    cdouble gy = factor * dy;
                    cdouble gz = factor * dz;

                    h_dGdx[idx] =
                        pfft_complex(gx.real(), gx.imag());
                    h_dGdy[idx] =
                        pfft_complex(gy.real(), gy.imag());
                    h_dGdz[idx] =
                        pfft_complex(gz.real(), gz.imag());

                    const double invR = 1.0 / R;
                    const cdouble a =
                        3.0 * invR * invR -
                        3.0 * cdouble(0.0, 1.0) * k * invR -
                        k * k;
                    const cdouble b =
                        invR * invR -
                        cdouble(0.0, 1.0) * k * invR;
                    const double displacement[3] = {dx, dy, dz};
                    const int row[6] = {0, 0, 0, 1, 1, 2};
                    const int column[6] = {0, 1, 2, 1, 2, 2};
                    for (int component = 0; component < 6; component++) {
                        cdouble value =
                            G * a *
                            displacement[row[component]] *
                            displacement[column[component]] *
                            invR * invR;
                        if (row[component] == column[component])
                            value -= G * b;
                        h_d2G[component][idx] = pfft_complex(
                            value.real(), value.imag());
                    }
                }
            }
        }
    }

    // Allocate GPU buffers for Green's FFTs
    CUDA_CHECK(cudaMalloc(&d_G_hat,    grid_total * sizeof(PfftComplex)));
    CUDA_CHECK(cudaMalloc(&d_dGdx_hat, grid_total * sizeof(PfftComplex)));
    CUDA_CHECK(cudaMalloc(&d_dGdy_hat, grid_total * sizeof(PfftComplex)));
    CUDA_CHECK(cudaMalloc(&d_dGdz_hat, grid_total * sizeof(PfftComplex)));
    PfftComplex** hessian_hat[6] = {
        &d_d2Gxx_hat, &d_d2Gxy_hat, &d_d2Gxz_hat,
        &d_d2Gyy_hat, &d_d2Gyz_hat, &d_d2Gzz_hat
    };
    for (int component = 0; component < 6; component++) {
        CUDA_CHECK(cudaMalloc(
            hessian_hat[component],
            grid_total * sizeof(PfftComplex)));
    }

    // Create cuFFT plans
    CUFFT_CHECK(cufftPlan3d(
        &plan_fwd, M2x, M2y, M2z, PFFT_CUFFT_TYPE));
    CUFFT_CHECK(cufftPlan3d(
        &plan_inv, M2x, M2y, M2z, PFFT_CUFFT_TYPE));

    // Upload and transform Green's functions
    auto fft_green = [&](std::vector<PfftComplex>& h_data, PfftComplex* d_hat) {
        CUDA_CHECK(cudaMemcpy(d_hat, h_data.data(),
                              grid_total * sizeof(PfftComplex),
                              cudaMemcpyHostToDevice));
        CUFFT_CHECK(pfft_execute(plan_fwd, d_hat, CUFFT_FORWARD));
    };

    fft_green(h_G, d_G_hat);
    fft_green(h_dGdx, d_dGdx_hat);
    fft_green(h_dGdy, d_dGdy_hat);
    fft_green(h_dGdz, d_dGdz_hat);
    for (int component = 0; component < 6; component++)
        fft_green(h_d2G[component], *hessian_hat[component]);

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

    double near_radius =
        correction_radius_cells >= 0.0
        ? correction_radius_cells * h
        : (interp_p + 2) * h;
    const char* correction_radius_env =
        std::getenv("BEM_PFFT_CORR_RADIUS_H");
    if (correction_radius_cells < 0.0 && correction_radius_env) {
        const double radius_in_grid_cells =
            std::atof(correction_radius_env);
        if (radius_in_grid_cells < 0.0) {
            std::fprintf(
                stderr,
                "BEM_PFFT_CORR_RADIUS_H must be non-negative\n");
            std::exit(2);
        }
        near_radius = radius_in_grid_cells * h;
    }
    std::vector<int> h_row_ptr, h_col_idx;
    if (near_radius > 0.0) {
        build_near_list(
            targets, Nt, sources, Ns,
            near_radius, h_row_ptr, h_col_idx);
    } else {
        h_row_ptr.assign(Nt + 1, 0);
    }
    corr_nnz = (int)h_col_idx.size();

    printf("  [pFFT] Near pairs: %d (%.1f per target, %.3f%% density)\n",
           corr_nnz, (double)corr_nnz / Nt,
           100.0 * corr_nnz / ((double)Nt * Ns));

    // Compute correction values: C[i,j] = G_exact(ri,rj) - G_grid(ri,rj)
    std::vector<float> h_cG_re(corr_nnz), h_cG_im(corr_nnz);
    std::vector<float> h_cdx_re(corr_nnz), h_cdx_im(corr_nnz);
    std::vector<float> h_cdy_re(corr_nnz), h_cdy_im(corr_nnz);
    std::vector<float> h_cdz_re(corr_nnz), h_cdz_im(corr_nnz);
    std::array<std::vector<float>, 6> h_c2_re;
    std::array<std::vector<float>, 6> h_c2_im;
    for (int component = 0; component < 6; component++) {
        h_c2_re[component].resize(corr_nnz);
        h_c2_im[component].resize(corr_nnz);
    }

    // Precompute Green's function values on local grid region (for grid-mediated G)
    // G_grid_local[di][dj][dk] for small di,dj,dk
    int grange = (int)ceil(near_radius / h) + interp_p + 1;  // max stencil displacement
    int gspan = 2 * grange + 1;
    long long gsize = (long long)gspan * gspan * gspan;
    std::vector<cdouble> G_local(gsize);
    std::vector<cdouble> dGdx_local(gsize), dGdy_local(gsize), dGdz_local(gsize);
    std::array<std::vector<cdouble>, 6> d2G_local;
    for (int component = 0; component < 6; component++)
        d2G_local[component].resize(gsize);
    for (int di = -grange; di <= grange; di++) {
        for (int dj = -grange; dj <= grange; dj++) {
            for (int dk = -grange; dk <= grange; dk++) {
                double dx = di * h, dy = dj * h, dz = dk * h;
                double R = sqrt(dx*dx + dy*dy + dz*dz);
                long long li = (long long)(di+grange)*gspan*gspan + (dj+grange)*gspan + (dk+grange);
                if (R < 1e-30) {
                    G_local[li] = 0;
                    dGdx_local[li] = dGdy_local[li] = dGdz_local[li] = 0;
                    for (int component = 0; component < 6; component++)
                        d2G_local[component][li] = 0;
                } else {
                    cdouble expikR = std::exp(cdouble(0, 1) * k * R);
                    G_local[li] = expikR * INV4PI / R;
                    cdouble factor = (cdouble(0,1) * k - 1.0/R) * G_local[li] / R;
                    dGdx_local[li] = factor * dx;
                    dGdy_local[li] = factor * dy;
                    dGdz_local[li] = factor * dz;
                    const double invR = 1.0 / R;
                    const cdouble a =
                        3.0 * invR * invR -
                        3.0 * cdouble(0.0, 1.0) * k * invR -
                        k * k;
                    const cdouble b =
                        invR * invR -
                        cdouble(0.0, 1.0) * k * invR;
                    const double displacement[3] = {dx, dy, dz};
                    const int row[6] = {0, 0, 0, 1, 1, 2};
                    const int column[6] = {0, 1, 2, 1, 2, 2};
                    for (int component = 0; component < 6; component++) {
                        cdouble value =
                            G_local[li] * a *
                            displacement[row[component]] *
                            displacement[column[component]] *
                            invR * invR;
                        if (row[component] == column[component])
                            value -= G_local[li] * b;
                        d2G_local[component][li] = value;
                    }
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
            cdouble d2G_exact[6] = {};
            if (R > 1e-30) {
                cdouble expikR = std::exp(cdouble(0, 1) * k * R);
                G_exact = expikR * INV4PI / R;
                cdouble factor = (cdouble(0,1) * k - 1.0/R) * G_exact / R;
                dGx_exact = factor * ddx;
                dGy_exact = factor * ddy;
                dGz_exact = factor * ddz;
                const double invR = 1.0 / R;
                const cdouble a =
                    3.0 * invR * invR -
                    3.0 * cdouble(0.0, 1.0) * k * invR -
                    k * k;
                const cdouble b =
                    invR * invR -
                    cdouble(0.0, 1.0) * k * invR;
                const double displacement[3] = {ddx, ddy, ddz};
                const int row[6] = {0, 0, 0, 1, 1, 2};
                const int column[6] = {0, 1, 2, 1, 2, 2};
                for (int component = 0; component < 6; component++) {
                    d2G_exact[component] =
                        G_exact * a *
                        displacement[row[component]] *
                        displacement[column[component]] *
                        invR * invR;
                    if (row[component] == column[component])
                        d2G_exact[component] -= G_exact * b;
                }
            }

            // Grid-mediated Green's function:
            // G_grid(i,j) = sum_a sum_b w_tgt[i,a] * G_grid[a-b] * w_src[j,b]
            cdouble G_grid(0, 0), dGx_grid(0, 0), dGy_grid(0, 0), dGz_grid(0, 0);
            cdouble d2G_grid[6] = {};

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
                        for (int component = 0; component < 6; component++)
                            d2G_grid[component] +=
                                ww * d2G_local[component][li];
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
            for (int component = 0; component < 6; component++) {
                const cdouble correction =
                    d2G_exact[component] - d2G_grid[component];
                h_c2_re[component][p] = correction.real();
                h_c2_im[component][p] = correction.imag();
            }
        }
    }

    // Upload correction data to GPU
    CUDA_CHECK(cudaMalloc(&d_corr_row_ptr, (Nt+1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_corr_row_ptr, h_row_ptr.data(), (Nt+1)*sizeof(int), cudaMemcpyHostToDevice));
    if (corr_nnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_corr_col_idx, corr_nnz * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
            d_corr_col_idx, h_col_idx.data(),
            corr_nnz * sizeof(int), cudaMemcpyHostToDevice));
    }

    auto upload_corr = [&](std::vector<float>& h_re, std::vector<float>& h_im,
                           float*& d_re, float*& d_im) {
        CUDA_CHECK(cudaMalloc(&d_re, corr_nnz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_im, corr_nnz * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(
            d_re, h_re.data(), corr_nnz * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_im, h_im.data(), corr_nnz * sizeof(float),
            cudaMemcpyHostToDevice));
    };
    if (corr_nnz > 0) {
        upload_corr(h_cG_re,  h_cG_im,  d_corr_G_re,    d_corr_G_im);
        upload_corr(h_cdx_re, h_cdx_im, d_corr_dGdx_re, d_corr_dGdx_im);
        upload_corr(h_cdy_re, h_cdy_im, d_corr_dGdy_re, d_corr_dGdy_im);
        upload_corr(h_cdz_re, h_cdz_im, d_corr_dGdz_re, d_corr_dGdz_im);
    }
    float** hessian_correction_re[6] = {
        &d_corr_d2Gxx_re, &d_corr_d2Gxy_re, &d_corr_d2Gxz_re,
        &d_corr_d2Gyy_re, &d_corr_d2Gyz_re, &d_corr_d2Gzz_re
    };
    float** hessian_correction_im[6] = {
        &d_corr_d2Gxx_im, &d_corr_d2Gxy_im, &d_corr_d2Gxz_im,
        &d_corr_d2Gyy_im, &d_corr_d2Gyz_im, &d_corr_d2Gzz_im
    };
    if (corr_nnz > 0) {
        for (int component = 0; component < 6; component++) {
            upload_corr(
                h_c2_re[component], h_c2_im[component],
                *hessian_correction_re[component],
                *hessian_correction_im[component]);
        }
    }

    printf("  [pFFT] Near-field correction done: %.1fms\n", timer.elapsed_ms());
    timer.reset();

    // --- Step 5: Allocate work buffers ---
    CUDA_CHECK(cudaMalloc(&d_work_a, grid_total * sizeof(PfftComplex)));
    CUDA_CHECK(cudaMalloc(&d_work_b, grid_total * sizeof(PfftComplex)));

    CUDA_CHECK(cudaMalloc(&d_charges_re, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_charges_im, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result_re,  Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result_im,  Nt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_re,    Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_im,    Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hess_re,    Nt * 6 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hess_im,    Nt * 6 * sizeof(double)));

    // Batch-2 buffers
    CUDA_CHECK(cudaMalloc(&d_charges2_re, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_charges2_im, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_charges3_re, Ns * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_charges3_im, Ns * sizeof(double)));
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
           (12 * grid_total * sizeof(PfftComplex) +
            (long long)(Ns + Nt) * stencil * 12.0 +
            corr_nnz * 20.0 * 4 +
            (Ns + Nt) * (5 * 8.0 + 3 * 8.0 * 2)) / 1e6);
}

void HelmholtzPFFT::prepare_charge_spectrum(
    const double* d_q_re, const double* d_q_im)
{
    int block = 256;

    kernel_zero_grid<<<((int)((grid_total + block - 1) / block)), block>>>(
        d_work_a, grid_total);

    kernel_anterpolate<<<(Ns + block - 1) / block, block>>>(
        d_q_re, d_q_im,
        d_src_stencil_idx, d_src_stencil_wt,
        Ns, stencil, d_work_a);

    CUFFT_CHECK(pfft_execute(
        plan_fwd, d_work_a, CUFFT_FORWARD));
}

void HelmholtzPFFT::convolve_prepared_and_correct(
    const double* d_q_re, const double* d_q_im,
    const PfftComplex* d_kernel_hat,
    double* d_out_re, double* d_out_im)
{
    int block = 256;
    double inv_N = 1.0 / grid_total;

    kernel_pointwise_mul<<<
        ((int)((grid_total + block - 1) / block)), block>>>(
        d_work_a, d_kernel_hat, d_work_b, grid_total, inv_N);

    CUFFT_CHECK(pfft_execute(
        plan_inv, d_work_b, CUFFT_INVERSE));

    kernel_interpolate<<<(Nt + block - 1) / block, block>>>(
        d_work_b,
        d_tgt_stencil_idx, d_tgt_stencil_wt,
        Nt, stencil, d_out_re, d_out_im);

    if (corr_nnz > 0) {
        const float* correction_re = d_corr_dGdz_re;
        const float* correction_im = d_corr_dGdz_im;
        if (d_kernel_hat == d_G_hat) {
            correction_re = d_corr_G_re;
            correction_im = d_corr_G_im;
        } else if (d_kernel_hat == d_dGdx_hat) {
            correction_re = d_corr_dGdx_re;
            correction_im = d_corr_dGdx_im;
        } else if (d_kernel_hat == d_dGdy_hat) {
            correction_re = d_corr_dGdy_re;
            correction_im = d_corr_dGdy_im;
        } else if (d_kernel_hat == d_d2Gxx_hat) {
            correction_re = d_corr_d2Gxx_re;
            correction_im = d_corr_d2Gxx_im;
        } else if (d_kernel_hat == d_d2Gxy_hat) {
            correction_re = d_corr_d2Gxy_re;
            correction_im = d_corr_d2Gxy_im;
        } else if (d_kernel_hat == d_d2Gxz_hat) {
            correction_re = d_corr_d2Gxz_re;
            correction_im = d_corr_d2Gxz_im;
        } else if (d_kernel_hat == d_d2Gyy_hat) {
            correction_re = d_corr_d2Gyy_re;
            correction_im = d_corr_d2Gyy_im;
        } else if (d_kernel_hat == d_d2Gyz_hat) {
            correction_re = d_corr_d2Gyz_re;
            correction_im = d_corr_d2Gyz_im;
        } else if (d_kernel_hat == d_d2Gzz_hat) {
            correction_re = d_corr_d2Gzz_re;
            correction_im = d_corr_d2Gzz_im;
        }
        kernel_near_correction<<<(Nt+block-1)/block, block>>>(
            d_corr_row_ptr, d_corr_col_idx,
            correction_re, correction_im,
            d_q_re, d_q_im,
            d_out_re, d_out_im, Nt);
    }
}

// Core: convolve charges with kernel and add near-field correction
void HelmholtzPFFT::convolve_and_correct(
    const double* d_q_re, const double* d_q_im,
    const PfftComplex* d_kernel_hat,
    double* d_out_re, double* d_out_im)
{
    prepare_charge_spectrum(d_q_re, d_q_im);
    convolve_prepared_and_correct(
        d_q_re, d_q_im, d_kernel_hat, d_out_re, d_out_im);
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
    prepare_charge_spectrum(d_charges_re, d_charges_im);

    // 3 convolutions: dG/dx, dG/dy, dG/dz
    convolve_prepared_and_correct(
        d_charges_re, d_charges_im, d_dGdx_hat,
        d_grad_re, d_grad_im);               // offset 0: gx
    convolve_prepared_and_correct(
        d_charges_re, d_charges_im, d_dGdy_hat,
        d_grad_re + Nt, d_grad_im + Nt);     // offset Nt: gy
    convolve_prepared_and_correct(
        d_charges_re, d_charges_im, d_dGdz_hat,
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

    prepare_charge_spectrum(d_charges_re, d_charges_im);

    // Potential
    CUDA_CHECK(cudaMemset(d_result_re, 0, Nt*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_result_im, 0, Nt*sizeof(double)));
    convolve_prepared_and_correct(
        d_charges_re, d_charges_im, d_G_hat,
        d_result_re, d_result_im);

    std::vector<double> r_re(Nt), r_im(Nt);
    CUDA_CHECK(cudaMemcpy(r_re.data(), d_result_re, Nt*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r_im.data(), d_result_im, Nt*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < Nt; i++)
        pot_result[i] = cdouble(r_re[i], r_im[i]);

    // Gradient (3 convolutions)
    CUDA_CHECK(cudaMemset(d_grad_re, 0, Nt*3*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grad_im, 0, Nt*3*sizeof(double)));

    convolve_prepared_and_correct(
        d_charges_re, d_charges_im, d_dGdx_hat,
        d_grad_re, d_grad_im);
    convolve_prepared_and_correct(
        d_charges_re, d_charges_im, d_dGdy_hat,
        d_grad_re + Nt, d_grad_im + Nt);
    convolve_prepared_and_correct(
        d_charges_re, d_charges_im, d_dGdz_hat,
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

void HelmholtzPFFT::evaluate_grad_hessian(
    const cdouble* charges,
    cdouble* grad_result,
    cdouble* hessian_result)
{
    std::vector<double> h_re(Ns), h_im(Ns);
    for (int i = 0; i < Ns; i++) {
        h_re[i] = charges[i].real();
        h_im[i] = charges[i].imag();
    }
    CUDA_CHECK(cudaMemcpy(
        d_charges_re, h_re.data(),
        Ns * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_charges_im, h_im.data(),
        Ns * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_grad_re, 0, Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grad_im, 0, Nt * 3 * sizeof(double)));
    prepare_charge_spectrum(d_charges_re, d_charges_im);
    const PfftComplex* gradient_kernels[3] = {
        d_dGdx_hat, d_dGdy_hat, d_dGdz_hat
    };
    for (int component = 0; component < 3; component++) {
        convolve_prepared_and_correct(
            d_charges_re, d_charges_im,
            gradient_kernels[component],
            d_grad_re + component * Nt,
            d_grad_im + component * Nt);
    }

    CUDA_CHECK(cudaMemset(d_hess_re, 0, Nt * 6 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_hess_im, 0, Nt * 6 * sizeof(double)));
    const PfftComplex* hessian_kernels[6] = {
        d_d2Gxx_hat, d_d2Gxy_hat, d_d2Gxz_hat,
        d_d2Gyy_hat, d_d2Gyz_hat, d_d2Gzz_hat
    };
    for (int component = 0; component < 6; component++) {
        convolve_prepared_and_correct(
            d_charges_re, d_charges_im,
            hessian_kernels[component],
            d_hess_re + component * Nt,
            d_hess_im + component * Nt);
    }

    std::vector<double> gradient_re(Nt * 3), gradient_im(Nt * 3);
    std::vector<double> hessian_re(Nt * 6), hessian_im(Nt * 6);
    CUDA_CHECK(cudaMemcpy(
        gradient_re.data(), d_grad_re,
        Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gradient_im.data(), d_grad_im,
        Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        hessian_re.data(), d_hess_re,
        Nt * 6 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        hessian_im.data(), d_hess_im,
        Nt * 6 * sizeof(double), cudaMemcpyDeviceToHost));
    for (int point = 0; point < Nt; point++) {
        for (int component = 0; component < 3; component++) {
            grad_result[3 * point + component] = cdouble(
                gradient_re[component * Nt + point],
                gradient_im[component * Nt + point]);
        }
        for (int component = 0; component < 6; component++) {
            hessian_result[6 * point + component] = cdouble(
                hessian_re[component * Nt + point],
                hessian_im[component * Nt + point]);
        }
    }
}

void HelmholtzPFFT::evaluate_grad_hessian_from_prepared(
    const HelmholtzPFFT& prepared_source,
    cdouble* grad_result,
    cdouble* hessian_result)
{
    if (!initialized || !prepared_source.initialized ||
        Ns != prepared_source.Ns || Nt != prepared_source.Nt ||
        grid_total != prepared_source.grid_total ||
        M2x != prepared_source.M2x ||
        M2y != prepared_source.M2y ||
        M2z != prepared_source.M2z) {
        throw std::invalid_argument(
            "pFFT prepared spectrum requires identical auxiliary grids");
    }

    CUDA_CHECK(cudaMemcpy(
        d_work_a, prepared_source.d_work_a,
        grid_total * sizeof(PfftComplex),
        cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemset(d_grad_re, 0, Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_grad_im, 0, Nt * 3 * sizeof(double)));
    const PfftComplex* gradient_kernels[3] = {
        d_dGdx_hat, d_dGdy_hat, d_dGdz_hat
    };
    for (int component = 0; component < 3; component++) {
        convolve_prepared_and_correct(
            prepared_source.d_charges_re,
            prepared_source.d_charges_im,
            gradient_kernels[component],
            d_grad_re + component * Nt,
            d_grad_im + component * Nt);
    }

    CUDA_CHECK(cudaMemset(d_hess_re, 0, Nt * 6 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_hess_im, 0, Nt * 6 * sizeof(double)));
    const PfftComplex* hessian_kernels[6] = {
        d_d2Gxx_hat, d_d2Gxy_hat, d_d2Gxz_hat,
        d_d2Gyy_hat, d_d2Gyz_hat, d_d2Gzz_hat
    };
    for (int component = 0; component < 6; component++) {
        convolve_prepared_and_correct(
            prepared_source.d_charges_re,
            prepared_source.d_charges_im,
            hessian_kernels[component],
            d_hess_re + component * Nt,
            d_hess_im + component * Nt);
    }

    std::vector<double> gradient_re(Nt * 3), gradient_im(Nt * 3);
    std::vector<double> hessian_re(Nt * 6), hessian_im(Nt * 6);
    CUDA_CHECK(cudaMemcpy(
        gradient_re.data(), d_grad_re,
        Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gradient_im.data(), d_grad_im,
        Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        hessian_re.data(), d_hess_re,
        Nt * 6 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        hessian_im.data(), d_hess_im,
        Nt * 6 * sizeof(double), cudaMemcpyDeviceToHost));
    for (int point = 0; point < Nt; point++) {
        for (int component = 0; component < 3; component++) {
            grad_result[3 * point + component] = cdouble(
                gradient_re[component * Nt + point],
                gradient_im[component * Nt + point]);
        }
        for (int component = 0; component < 6; component++) {
            hessian_result[6 * point + component] = cdouble(
                hessian_re[component * Nt + point],
                hessian_im[component * Nt + point]);
        }
    }
}

void HelmholtzPFFT::evaluate_vector_actions(
    const cdouble* charges_x,
    const cdouble* charges_y,
    const cdouble* charges_z,
    cdouble* curl_result,
    cdouble* hessian_action)
{
    if (!initialized)
        throw std::runtime_error("pFFT is not initialized");
    if (!d_vector_spectra) {
        CUDA_CHECK(cudaMalloc(
            &d_vector_spectra,
            3 * grid_total * sizeof(PfftComplex)));
    }

    std::vector<double> host_real(
        static_cast<size_t>(3) * Ns);
    std::vector<double> host_imaginary(
        static_cast<size_t>(3) * Ns);
    const cdouble* host_charges[3] = {
        charges_x, charges_y, charges_z
    };
    for (int component = 0; component < 3; component++) {
        for (int source = 0; source < Ns; source++) {
            host_real[
                static_cast<size_t>(component) * Ns + source] =
                host_charges[component][source].real();
            host_imaginary[
                static_cast<size_t>(component) * Ns + source] =
                host_charges[component][source].imag();
        }
    }
    double* device_real[3] = {
        d_charges_re, d_charges2_re, d_charges3_re
    };
    double* device_imaginary[3] = {
        d_charges_im, d_charges2_im, d_charges3_im
    };
    for (int component = 0; component < 3; component++) {
        CUDA_CHECK(cudaMemcpy(
            device_real[component],
            host_real.data() + static_cast<size_t>(component) * Ns,
            Ns * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            device_imaginary[component],
            host_imaginary.data() +
                static_cast<size_t>(component) * Ns,
            Ns * sizeof(double), cudaMemcpyHostToDevice));
        prepare_charge_spectrum(
            device_real[component], device_imaginary[component]);
        CUDA_CHECK(cudaMemcpy(
            d_vector_spectra +
                static_cast<size_t>(component) * grid_total,
            d_work_a,
            grid_total * sizeof(PfftComplex),
            cudaMemcpyDeviceToDevice));
    }

    evaluate_vector_actions_device(
        d_vector_spectra,
        d_charges_re, d_charges_im,
        d_charges2_re, d_charges2_im,
        d_charges3_re, d_charges3_im,
        curl_result, hessian_action);
}

void HelmholtzPFFT::evaluate_vector_actions_from_prepared(
    const HelmholtzPFFT& prepared_source,
    cdouble* curl_result,
    cdouble* hessian_action)
{
    if (!initialized || !prepared_source.initialized ||
        !prepared_source.d_vector_spectra ||
        Ns != prepared_source.Ns || Nt != prepared_source.Nt ||
        grid_total != prepared_source.grid_total ||
        M2x != prepared_source.M2x ||
        M2y != prepared_source.M2y ||
        M2z != prepared_source.M2z) {
        throw std::invalid_argument(
            "pFFT vector actions require identical auxiliary grids");
    }
    evaluate_vector_actions_device(
        prepared_source.d_vector_spectra,
        prepared_source.d_charges_re,
        prepared_source.d_charges_im,
        prepared_source.d_charges2_re,
        prepared_source.d_charges2_im,
        prepared_source.d_charges3_re,
        prepared_source.d_charges3_im,
        curl_result, hessian_action);
}

void HelmholtzPFFT::evaluate_vector_actions_device(
    const PfftComplex* spectra,
    const double* qx_re,
    const double* qx_im,
    const double* qy_re,
    const double* qy_im,
    const double* qz_re,
    const double* qz_im,
    cdouble* curl_result,
    cdouble* hessian_action)
{
    const int block = 256;
    const int grid_blocks = static_cast<int>(
        (grid_total + block - 1) / block);
    const int point_blocks = (Nt + block - 1) / block;
    const double inverse_grid = 1.0 / grid_total;

    CUDA_CHECK(cudaMemset(
        d_grad_re, 0, Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(
        d_grad_im, 0, Nt * 3 * sizeof(double)));
    for (int component = 0; component < 3; component++) {
        kernel_curl_spectrum<<<grid_blocks, block>>>(
            spectra, d_dGdx_hat, d_dGdy_hat, d_dGdz_hat,
            d_work_b, grid_total, component, inverse_grid);
        CUFFT_CHECK(pfft_execute(
            plan_inv, d_work_b, CUFFT_INVERSE));
        kernel_interpolate<<<point_blocks, block>>>(
            d_work_b,
            d_tgt_stencil_idx, d_tgt_stencil_wt,
            Nt, stencil,
            d_grad_re + component * Nt,
            d_grad_im + component * Nt);
    }
    if (corr_nnz > 0) {
        kernel_curl_near_correction<<<point_blocks, block>>>(
            d_corr_row_ptr, d_corr_col_idx,
            d_corr_dGdx_re, d_corr_dGdx_im,
            d_corr_dGdy_re, d_corr_dGdy_im,
            d_corr_dGdz_re, d_corr_dGdz_im,
            qx_re, qx_im, qy_re, qy_im, qz_re, qz_im,
            d_grad_re, d_grad_im, Nt);
    }

    CUDA_CHECK(cudaMemset(
        d_hess_re, 0, Nt * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(
        d_hess_im, 0, Nt * 3 * sizeof(double)));
    for (int component = 0; component < 3; component++) {
        kernel_hessian_action_spectrum<<<grid_blocks, block>>>(
            spectra,
            d_d2Gxx_hat, d_d2Gxy_hat, d_d2Gxz_hat,
            d_d2Gyy_hat, d_d2Gyz_hat, d_d2Gzz_hat,
            d_work_b, grid_total, component, inverse_grid);
        CUFFT_CHECK(pfft_execute(
            plan_inv, d_work_b, CUFFT_INVERSE));
        kernel_interpolate<<<point_blocks, block>>>(
            d_work_b,
            d_tgt_stencil_idx, d_tgt_stencil_wt,
            Nt, stencil,
            d_hess_re + component * Nt,
            d_hess_im + component * Nt);
    }
    if (corr_nnz > 0) {
        kernel_hessian_action_near_correction<<<point_blocks, block>>>(
            d_corr_row_ptr, d_corr_col_idx,
            d_corr_d2Gxx_re, d_corr_d2Gxx_im,
            d_corr_d2Gxy_re, d_corr_d2Gxy_im,
            d_corr_d2Gxz_re, d_corr_d2Gxz_im,
            d_corr_d2Gyy_re, d_corr_d2Gyy_im,
            d_corr_d2Gyz_re, d_corr_d2Gyz_im,
            d_corr_d2Gzz_re, d_corr_d2Gzz_im,
            qx_re, qx_im, qy_re, qy_im, qz_re, qz_im,
            d_hess_re, d_hess_im, Nt);
    }
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> curl_real(
        static_cast<size_t>(Nt) * 3);
    std::vector<double> curl_imaginary(
        static_cast<size_t>(Nt) * 3);
    std::vector<double> hessian_real(
        static_cast<size_t>(Nt) * 3);
    std::vector<double> hessian_imaginary(
        static_cast<size_t>(Nt) * 3);
    CUDA_CHECK(cudaMemcpy(
        curl_real.data(), d_grad_re,
        Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        curl_imaginary.data(), d_grad_im,
        Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        hessian_real.data(), d_hess_re,
        Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        hessian_imaginary.data(), d_hess_im,
        Nt * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    for (int point = 0; point < Nt; point++) {
        for (int component = 0; component < 3; component++) {
            curl_result[3 * point + component] = cdouble(
                curl_real[component * Nt + point],
                curl_imaginary[component * Nt + point]);
            hessian_action[3 * point + component] = cdouble(
                hessian_real[component * Nt + point],
                hessian_imaginary[component * Nt + point]);
        }
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
    safe_free((void*&)d_d2Gxx_hat);
    safe_free((void*&)d_d2Gxy_hat);
    safe_free((void*&)d_d2Gxz_hat);
    safe_free((void*&)d_d2Gyy_hat);
    safe_free((void*&)d_d2Gyz_hat);
    safe_free((void*&)d_d2Gzz_hat);
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
    safe_free((void*&)d_corr_d2Gxx_re);
    safe_free((void*&)d_corr_d2Gxx_im);
    safe_free((void*&)d_corr_d2Gxy_re);
    safe_free((void*&)d_corr_d2Gxy_im);
    safe_free((void*&)d_corr_d2Gxz_re);
    safe_free((void*&)d_corr_d2Gxz_im);
    safe_free((void*&)d_corr_d2Gyy_re);
    safe_free((void*&)d_corr_d2Gyy_im);
    safe_free((void*&)d_corr_d2Gyz_re);
    safe_free((void*&)d_corr_d2Gyz_im);
    safe_free((void*&)d_corr_d2Gzz_re);
    safe_free((void*&)d_corr_d2Gzz_im);
    safe_free((void*&)d_work_a);
    safe_free((void*&)d_work_b);
    safe_free((void*&)d_charges_re);
    safe_free((void*&)d_charges_im);
    safe_free((void*&)d_result_re);
    safe_free((void*&)d_result_im);
    safe_free((void*&)d_grad_re);
    safe_free((void*&)d_grad_im);
    safe_free((void*&)d_hess_re);
    safe_free((void*&)d_hess_im);
    safe_free((void*&)d_charges2_re);
    safe_free((void*&)d_charges2_im);
    safe_free((void*&)d_charges3_re);
    safe_free((void*&)d_charges3_im);
    safe_free((void*&)d_result2_re);
    safe_free((void*&)d_result2_im);
    safe_free((void*&)d_grad2_re);
    safe_free((void*&)d_grad2_im);
    safe_free((void*&)d_vector_spectra);
    safe_free((void*&)d_src_pts);
    safe_free((void*&)d_tgt_pts);

    initialized = false;
}
