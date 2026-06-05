#ifndef BEM_PFFT_H
#define BEM_PFFT_H

#include "types.h"
#include <cufft.h>
#include <complex>
#include <vector>

// Pre-corrected FFT (pFFT) accelerator for Helmholtz Green's function.
// Drop-in replacement for HelmholtzFMM with the same public interface.
//
// Algorithm:
//  1. Embed source/target points in a uniform 3D grid
//  2. Anterpolate charges to grid (Lagrange interpolation)
//  3. Convolve with Green's function via FFT (Toeplitz -> circulant embedding)
//  4. Interpolate potential/gradient back to target points
//  5. Apply near-field correction (exact - grid-mediated) for nearby pairs

struct HelmholtzPFFT {
    cdouble k;          // wavenumber
    int Nt, Ns;         // number of target / source points

    // Grid parameters
    int Mx, My, Mz;     // physical grid dimensions
    int M2x, M2y, M2z;  // doubled grid dimensions (Toeplitz embedding)
    double h;            // isotropic grid spacing
    double origin[3];    // grid origin (min corner - padding)
    int interp_p;        // interpolation order (default 3, so p+1=4 nodes/dim)
    int stencil;         // (interp_p+1)^3 stencil size

    // Precomputed FFT of Green's function (doubled grid, complex double)
    cufftDoubleComplex* d_G_hat;       // potential: exp(ikR)/(4piR)
    cufftDoubleComplex* d_dGdx_hat;    // gradient x component
    cufftDoubleComplex* d_dGdy_hat;    // gradient y component
    cufftDoubleComplex* d_dGdz_hat;    // gradient z component

    // cuFFT plans (3D complex-to-complex)
    cufftHandle plan_fwd, plan_inv;
    long long grid_total;  // M2x * M2y * M2z

    // Interpolation stencils (per point, stencil entries each)
    // Grid-linear index + weight for each stencil node
    int*    d_src_stencil_idx;   // (Ns * stencil)
    double* d_src_stencil_wt;    // (Ns * stencil)
    int*    d_tgt_stencil_idx;   // (Nt * stencil)
    double* d_tgt_stencil_wt;    // (Nt * stencil)

    // Near-field correction (sparse CSR)
    // Correction = G_exact(ri, rj) - G_grid_mediated(ri, rj)
    int*    d_corr_row_ptr;      // (Nt + 1)
    int*    d_corr_col_idx;      // (nnz)
    double* d_corr_G_re;        // (nnz) potential correction (real part)
    double* d_corr_G_im;        // (nnz) potential correction (imag part)
    double* d_corr_dGdx_re;     // (nnz) grad-x correction
    double* d_corr_dGdx_im;
    double* d_corr_dGdy_re;     // (nnz) grad-y correction
    double* d_corr_dGdy_im;
    double* d_corr_dGdz_re;     // (nnz) grad-z correction
    double* d_corr_dGdz_im;
    int corr_nnz;

    // Work buffers on GPU (doubled grid, complex)
    cufftDoubleComplex* d_work_a;    // FFT workspace A
    cufftDoubleComplex* d_work_b;    // FFT workspace B

    // Charge / result buffers on GPU
    double* d_charges_re;      // (Ns)
    double* d_charges_im;
    double* d_result_re;       // (Nt)
    double* d_result_im;
    double* d_grad_re;         // (Nt*3) interleaved [gx0,gy0,gz0,gx1,...]
    double* d_grad_im;

    // Batch-2 buffers
    double* d_charges2_re;
    double* d_charges2_im;
    double* d_result2_re;
    double* d_result2_im;
    double* d_grad2_re;
    double* d_grad2_im;

    // Source/target positions on GPU
    double* d_src_pts;         // (Ns*3)
    double* d_tgt_pts;         // (Nt*3)

    bool initialized;

    HelmholtzPFFT() : initialized(false),
        d_G_hat(0), d_dGdx_hat(0), d_dGdy_hat(0), d_dGdz_hat(0),
        d_src_stencil_idx(0), d_src_stencil_wt(0),
        d_tgt_stencil_idx(0), d_tgt_stencil_wt(0),
        d_corr_row_ptr(0), d_corr_col_idx(0),
        d_corr_G_re(0), d_corr_G_im(0),
        d_corr_dGdx_re(0), d_corr_dGdx_im(0),
        d_corr_dGdy_re(0), d_corr_dGdy_im(0),
        d_corr_dGdz_re(0), d_corr_dGdz_im(0),
        d_work_a(0), d_work_b(0),
        d_charges_re(0), d_charges_im(0),
        d_result_re(0), d_result_im(0),
        d_grad_re(0), d_grad_im(0),
        d_charges2_re(0), d_charges2_im(0),
        d_result2_re(0), d_result2_im(0),
        d_grad2_re(0), d_grad2_im(0),
        d_src_pts(0), d_tgt_pts(0) {}

    // Initialize: build grid, precompute Green's FFT, interpolation stencils,
    // near-field corrections
    void init(const double* targets, int n_tgt,
              const double* sources, int n_src,
              cdouble k_val, int digits = 3, int max_leaf = 64);

    // Evaluate: result[i] = sum_j G(r_i, r_j) * charges[j]
    void evaluate(const cdouble* charges, cdouble* result);

    // Evaluate gradient: grad[i*3+d] = sum_j dG/dx_d(r_i, r_j) * charges[j]
    void evaluate_gradient(const cdouble* charges, cdouble* grad_result);

    // Evaluate both potential and gradient
    void evaluate_pot_grad(const cdouble* charges, cdouble* pot_result, cdouble* grad_result);

    // Batch-2: two charge vectors, single FFT pipeline
    void evaluate_batch2(const cdouble* charges1, const cdouble* charges2,
                         cdouble* result1, cdouble* result2);

    // Batch-2 pot+grad
    void evaluate_pot_grad_batch2(const cdouble* charges1, const cdouble* charges2,
                                   cdouble* pot1, cdouble* grad1,
                                   cdouble* pot2, cdouble* grad2);

    void cleanup();
    ~HelmholtzPFFT() { if (initialized) cleanup(); }

private:
    // Core FFT-based convolution:
    // 1. Anterpolate charges -> grid
    // 2. FFT forward
    // 3. Pointwise multiply by kernel_hat
    // 4. FFT inverse
    // 5. Interpolate grid -> targets
    // 6. Add near-field correction
    void convolve_and_correct(const double* d_q_re, const double* d_q_im,
                              const cufftDoubleComplex* d_kernel_hat,
                              double* d_out_re, double* d_out_im);
};

#endif // BEM_PFFT_H
