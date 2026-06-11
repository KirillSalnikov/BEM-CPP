#ifndef BEM_FARFIELD_H
#define BEM_FARFIELD_H

#include "rwg.h"
#include <complex>
#include <vector>

// Precomputed quadrature data for far-field
struct FFCache {
    std::vector<double> qpts;  // (2*N*Nq, 3) flat
    std::vector<double> fvals; // (2*N*Nq, 3) flat
    std::vector<double> jw;    // (2*N*Nq) flat
    int N, Nq;

    void init(const RWG& rwg, const Mesh& mesh, int quad_order);
};

// GPU-resident copy of FFCache (uploaded once, reused for all orientations)
struct FFCacheGPU {
    double* d_qpts;    // (2*N*Nq, 3) device
    double* d_fvals;   // (2*N*Nq, 3) device
    double* d_jw;      // (2*N*Nq) device
    int N, Nq;
    bool initialized;

    FFCacheGPU() : d_qpts(0), d_fvals(0), d_jw(0), N(0), Nq(0), initialized(false) {}
    void upload(const FFCache& cache);
    void free();
    ~FFCacheGPU();
};

struct FFBatchWorkspace {
    double *d_cJ_re, *d_cJ_im, *d_cM_re, *d_cM_im;
    double *d_r_hats, *d_RT, *d_rhat_lab, *d_etheta_lab, *d_Fv_re, *d_Fv_im;
    double *d_e_par, *d_e_perp, *d_weights, *d_alpha_cos, *d_alpha_sin, *d_M_accum, *d_M_partial;
    int cap_coeffs, cap_rhat, cap_rt, cap_lab_dirs, cap_fv, cap_evec, cap_weight, cap_alpha, cap_mueller, cap_mueller_partial;
    std::vector<double> cJ_re, cJ_im, cM_re, cM_im, fv_re, fv_im, M_accum;

    FFBatchWorkspace();
    void reserve(int total_coeffs, int total_rhat, int total_fv);
    void reserve_mueller(int n_orient, int ndir);
    void reserve_mueller_partials(int groups, int ndir);
    void zero_mueller(int ndir);
    void zero_mueller_partials(int groups, int ndir);
    void download_mueller(double* M_out, int ndir);
    void free();
    ~FFBatchWorkspace();
};

// Batched GPU far-field: compute Fv for multiple (call, direction) pairs at once.
//
// coeffs_J, coeffs_M: (n_calls, N) complex double, host memory.
//   Call c uses coefficients at offset c*N.
// r_hats: (n_orient, ndir, 3) double, host memory.
//   r_hat for call c is at orient_idx = c/2 (par/perp share same directions).
// Fv_out: (n_calls, ndir, 3) complex double, host memory output.
//
// n_calls = 2 * n_orient (par + perp per orientation).
void compute_farfield_batch_cuda(
    const FFCacheGPU& gpu_cache,
    const std::complex<double>* coeffs_J,  // (n_calls * N) host
    const std::complex<double>* coeffs_M,  // (n_calls * N) host
    const double* r_hats,                  // (n_orient * ndir * 3) host
    std::complex<double> k_ext, double eta_ext,
    int n_calls, int n_orient, int ndir,
    std::complex<double>* Fv_out);         // (n_calls * ndir * 3) host

void compute_farfield_batch_cuda_ws(
    const FFCacheGPU& gpu_cache,
    FFBatchWorkspace& workspace,
    const std::complex<double>* coeffs_J,
    const std::complex<double>* coeffs_M,
    const double* r_hats,
    std::complex<double> k_ext, double eta_ext,
    int n_calls, int n_orient, int ndir,
    std::complex<double>* Fv_out);

void accumulate_farfield_mueller_batch_cuda_ws(
    const FFCacheGPU& gpu_cache,
    FFBatchWorkspace& workspace,
    const std::complex<double>* coeffs_J,
    const std::complex<double>* coeffs_M,
    const double* r_hats,
    const double* e_par,
    const double* e_perp,
    const double* weights,
    std::complex<double> k_ext, double eta_ext,
    int n_calls, int n_orient, int ndir);

void accumulate_farfield_mueller_direct_cuda_ws(
    const FFCacheGPU& gpu_cache,
    FFBatchWorkspace& workspace,
    const std::complex<double>* coeffs_J,
    const std::complex<double>* coeffs_M,
    const double* r_hats,
    const double* e_par,
    const double* e_perp,
    const double* weights,
    std::complex<double> k_ext, double eta_ext,
    int n_calls, int n_orient, int ndir);

void accumulate_farfield_mueller_alpha_cuda_ws(
    const FFCacheGPU& gpu_cache,
    FFBatchWorkspace& workspace,
    const std::complex<double>* coeffs_J,
    const std::complex<double>* coeffs_M,
    const double* r_hats,
    const double* e_par,
    const double* e_perp,
    const double* weights,
    const double* alpha_cos,
    const double* alpha_sin,
    std::complex<double> k_ext, double eta_ext,
    int n_base_orient, int alpha_avg, int ndir);

void accumulate_farfield_mueller_alpha_geom_cuda_ws(
    const FFCacheGPU& gpu_cache,
    FFBatchWorkspace& workspace,
    const std::complex<double>* coeffs_J,
    const std::complex<double>* coeffs_M,
    const double* RT_mats,
    const double* rhat_lab,
    const double* etheta_lab,
    const double* ephi_lab,
    const double* weights,
    const double* alpha_cos,
    const double* alpha_sin,
    std::complex<double> k_ext, double eta_ext,
    int n_base_orient, int alpha_avg, int ndir);

// CPU fallback for single-direction computation
void compute_far_field_vec_batch_cpu(const FFCache& cache,
                                     const std::complex<double>* coeffs_J,
                                     const std::complex<double>* coeffs_M,
                                     std::complex<double> k_ext, double eta_ext,
                                     const Vec3* r_hats, int ndir,
                                     std::complex<double>* Fv_out);

// CPU far-field for single-orient mode (phi=0 scattering plane)
void compute_far_field(const FFCache& cache,
                       const std::complex<double>* coeffs_J,
                       const std::complex<double>* coeffs_M,
                       std::complex<double> k_ext, double eta_ext,
                       const double* theta_arr, int ntheta,
                       bool scattering_plane_yz,
                       std::complex<double>* F_theta,
                       std::complex<double>* F_phi);

void amplitude_to_mueller(const std::complex<double>* S1,
                          const std::complex<double>* S2,
                          const std::complex<double>* S3,
                          const std::complex<double>* S4,
                          int ntheta, double* M);

#endif
