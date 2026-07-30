#ifndef BEM_PFFT_H
#define BEM_PFFT_H

#include "types.h"
#include <cufft.h>
#include <complex>
#include <vector>

#ifdef BEM_PFFT_FP32
using PfftComplex = cufftComplex;
#else
using PfftComplex = cufftDoubleComplex;
#endif

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

    // Precomputed Fourier-space Green kernels.
    PfftComplex* d_G_hat;       // potential: exp(ikR)/(4piR)
    PfftComplex* d_dGdx_hat;    // gradient x component
    PfftComplex* d_dGdy_hat;    // gradient y component
    PfftComplex* d_dGdz_hat;    // gradient z component
    PfftComplex* d_d2Gxx_hat;   // Hessian: xx, xy, xz, yy, yz, zz
    PfftComplex* d_d2Gxy_hat;
    PfftComplex* d_d2Gxz_hat;
    PfftComplex* d_d2Gyy_hat;
    PfftComplex* d_d2Gyz_hat;
    PfftComplex* d_d2Gzz_hat;

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
    float* d_corr_G_re;        // (nnz) approximate correction, FP32
    float* d_corr_G_im;
    float* d_corr_dGdx_re;
    float* d_corr_dGdx_im;
    float* d_corr_dGdy_re;
    float* d_corr_dGdy_im;
    float* d_corr_dGdz_re;
    float* d_corr_dGdz_im;
    float* d_corr_d2Gxx_re;
    float* d_corr_d2Gxx_im;
    float* d_corr_d2Gxy_re;
    float* d_corr_d2Gxy_im;
    float* d_corr_d2Gxz_re;
    float* d_corr_d2Gxz_im;
    float* d_corr_d2Gyy_re;
    float* d_corr_d2Gyy_im;
    float* d_corr_d2Gyz_re;
    float* d_corr_d2Gyz_im;
    float* d_corr_d2Gzz_re;
    float* d_corr_d2Gzz_im;
    int corr_nnz;

    // Work buffers on GPU (doubled grid, complex)
    PfftComplex* d_work_a;    // FFT workspace A
    PfftComplex* d_work_b;    // FFT workspace B

    // Charge / result buffers on GPU
    double* d_charges_re;      // (Ns)
    double* d_charges_im;
    double* d_result_re;       // (Nt)
    double* d_result_im;
    double* d_grad_re;         // (Nt*3) interleaved [gx0,gy0,gz0,gx1,...]
    double* d_grad_im;
    double* d_hess_re;         // (Nt*6): xx, xy, xz, yy, yz, zz
    double* d_hess_im;

    // Batch-2 buffers
    double* d_charges2_re;
    double* d_charges2_im;
    double* d_charges3_re;
    double* d_charges3_im;
    double* d_result2_re;
    double* d_result2_im;
    double* d_grad2_re;
    double* d_grad2_im;

    // Three prepared charge spectra for contracted vector derivatives.
    // Allocated lazily because scalar pFFT users do not need them.
    PfftComplex* d_vector_spectra;

    // Source/target positions on GPU
    double* d_src_pts;         // (Ns*3)
    double* d_tgt_pts;         // (Nt*3)

    bool initialized;

    HelmholtzPFFT() : d_G_hat(0), d_dGdx_hat(0),
        d_dGdy_hat(0), d_dGdz_hat(0),
        d_d2Gxx_hat(0), d_d2Gxy_hat(0), d_d2Gxz_hat(0),
        d_d2Gyy_hat(0), d_d2Gyz_hat(0), d_d2Gzz_hat(0),
        d_src_stencil_idx(0), d_src_stencil_wt(0),
        d_tgt_stencil_idx(0), d_tgt_stencil_wt(0),
        d_corr_row_ptr(0), d_corr_col_idx(0),
        d_corr_G_re(0), d_corr_G_im(0),
        d_corr_dGdx_re(0), d_corr_dGdx_im(0),
        d_corr_dGdy_re(0), d_corr_dGdy_im(0),
        d_corr_dGdz_re(0), d_corr_dGdz_im(0),
        d_corr_d2Gxx_re(0), d_corr_d2Gxx_im(0),
        d_corr_d2Gxy_re(0), d_corr_d2Gxy_im(0),
        d_corr_d2Gxz_re(0), d_corr_d2Gxz_im(0),
        d_corr_d2Gyy_re(0), d_corr_d2Gyy_im(0),
        d_corr_d2Gyz_re(0), d_corr_d2Gyz_im(0),
        d_corr_d2Gzz_re(0), d_corr_d2Gzz_im(0),
        d_work_a(0), d_work_b(0),
        d_charges_re(0), d_charges_im(0),
        d_result_re(0), d_result_im(0),
        d_grad_re(0), d_grad_im(0),
        d_hess_re(0), d_hess_im(0),
        d_charges2_re(0), d_charges2_im(0),
        d_charges3_re(0), d_charges3_im(0),
        d_result2_re(0), d_result2_im(0),
        d_grad2_re(0), d_grad2_im(0),
        d_vector_spectra(0),
        d_src_pts(0), d_tgt_pts(0), initialized(false) {}

    // Initialize: build grid, precompute Green's FFT, interpolation stencils,
    // near-field corrections
    void init(const double* targets, int n_tgt,
              const double* sources, int n_src,
              cdouble k_val, int digits = 3, int max_leaf = 64,
              double grid_spacing = 0.0,
              double correction_radius_cells = -1.0);

    static double grid_spacing_for_diameter(
        double diameter, cdouble wave_number,
        int interpolation_order);

    // Evaluate: result[i] = sum_j G(r_i, r_j) * charges[j]
    void evaluate(const cdouble* charges, cdouble* result);

    // Evaluate gradient: grad[i*3+d] = sum_j dG/dx_d(r_i, r_j) * charges[j]
    void evaluate_gradient(const cdouble* charges, cdouble* grad_result);

    // Evaluate both potential and gradient
    void evaluate_pot_grad(const cdouble* charges, cdouble* pot_result, cdouble* grad_result);

    // Evaluate gradient and Hessian in the layout used by HelmholtzFMM.
    void evaluate_grad_hessian(
        const cdouble* charges,
        cdouble* grad_result,
        cdouble* hessian_result);

    // Reuse charges and their forward FFT from another pFFT instance on the
    // same auxiliary grid. Kernel transforms and near corrections still
    // belong to this instance.
    void evaluate_grad_hessian_from_prepared(
        const HelmholtzPFFT& prepared_source,
        cdouble* grad_result,
        cdouble* hessian_result);

    // Contract derivatives of a three-component charge field before the
    // inverse transforms. curl_result stores xy, xz, yz antisymmetric
    // gradient components. hessian_action stores
    // div(grad(G) q) - trace(H(G)) q component-wise.
    void evaluate_vector_actions(
        const cdouble* charges_x,
        const cdouble* charges_y,
        const cdouble* charges_z,
        cdouble* curl_result,
        cdouble* hessian_action);

    void evaluate_vector_actions_from_prepared(
        const HelmholtzPFFT& prepared_source,
        cdouble* curl_result,
        cdouble* hessian_action);

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
    // Anterpolate one charge vector and compute its spectrum once.  The
    // spectrum is reused for potential, gradient and Hessian kernels.
    void prepare_charge_spectrum(
        const double* d_q_re, const double* d_q_im);

    // Apply one Fourier-space kernel to the spectrum in d_work_a.
    void convolve_prepared_and_correct(
        const double* d_q_re, const double* d_q_im,
        const PfftComplex* d_kernel_hat,
        double* d_out_re, double* d_out_im);

    void evaluate_vector_actions_device(
        const PfftComplex* spectra,
        const double* qx_re, const double* qx_im,
        const double* qy_re, const double* qy_im,
        const double* qz_re, const double* qz_im,
        cdouble* curl_result,
        cdouble* hessian_action);

    // Core FFT-based convolution:
    // 1. Anterpolate charges -> grid
    // 2. FFT forward
    // 3. Pointwise multiply by kernel_hat
    // 4. FFT inverse
    // 5. Interpolate grid -> targets
    // 6. Add near-field correction
    void convolve_and_correct(const double* d_q_re, const double* d_q_im,
                              const PfftComplex* d_kernel_hat,
                              double* d_out_re, double* d_out_im);
};

#endif // BEM_PFFT_H
