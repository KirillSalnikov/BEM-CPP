#ifndef BEM_BEM_FMM_H
#define BEM_BEM_FMM_H

#include "types.h"
#include "fmm.h"
#ifndef BEM_FMM_ONLY
#include "pfft.h"
#include "surface_pfft.h"
#endif
#include "rwg.h"
#include "mesh.h"
#include "quadrature.h"
#include <complex>
#include <vector>

struct BemFmmOperator {
    int N;             // number of RWG basis functions
    int Nq;            // quad points per triangle
    int system_size;   // 2*N

    cdouble k_ext, k_int;
    cdouble eta_ext, eta_int;

    bool use_pfft = false;
    bool use_spfft = false;
    double unknown_m_scale = 1.0;
    cdouble row_h_scale = cdouble(1.0, 0.0);
    double int_op_sign = 1.0;
    double k_identity = 0.0;
    bool n_form = false;
    double n_form_eps_int = 1.0;
    double n_form_m_identity = 0.0;

    // FMM engines (one per wavenumber)
    HelmholtzFMM fmm_ext;
    HelmholtzFMM fmm_int;
#ifndef BEM_FMM_ONLY
    HelmholtzPFFT pfft_ext;
    HelmholtzPFFT pfft_int;
    HelmholtzSurfacePFFT spfft_ext;
    HelmholtzSurfacePFFT spfft_int;
#endif
    bool shared_fmm;  // true if k_ext ≈ k_int

    // Precomputed quadrature data
    // quad points: (N, Nq, 3) for plus/minus halves
    std::vector<double> qpts_p;   // (N*Nq*3) flat
    std::vector<double> qpts_m;

    // RWG basis values: (N, Nq, 3)
    std::vector<double> f_p;      // (N*Nq*3) flat
    std::vector<double> f_m;

    // Divergences: (N)
    std::vector<double> div_p;
    std::vector<double> div_m;

    // Jacobian × weights: (N, Nq)
    std::vector<double> jw_p;
    std::vector<double> jw_m;

    // All quad points flat for FMM: (2*N*Nq, 3)
    std::vector<double> all_pts;

    // Singular correction matrices in sparse CSR format
    // All 4 matrices share the same sparsity pattern (same-triangle RWG pairs)
    std::vector<int> corr_row_ptr;       // (N+1)
    std::vector<int> corr_col_idx;       // (nnz)
    std::vector<cdouble> corr_L_ext_val; // (nnz)
    std::vector<cdouble> corr_K_ext_val;
    std::vector<cdouble> corr_L_int_val;
    std::vector<cdouble> corr_K_int_val;
    std::vector<double> corr_I_val;
    int corr_nnz;

    // Pre-allocated temporary buffers for matvec (avoid malloc/free per iteration)
    std::vector<cdouble> tmp_src_charges;   // (2*N*Nq) — source charges for FMM
    std::vector<cdouble> tmp_phi;           // (2*N*Nq) — FMM potential result
    std::vector<cdouble> tmp_grad[3];       // (2*N*Nq*3) each — FMM gradient results
    std::vector<cdouble> tmp_L_result;      // (N) — L operator result buffer
    std::vector<cdouble> tmp_K_result;      // (N) — K operator result buffer
    // Matvec output buffers: L/K × ext/int × J/M
    std::vector<cdouble> mv_L_ext_J, mv_L_ext_M, mv_K_ext_J, mv_K_ext_M;
    std::vector<cdouble> mv_L_int_J, mv_L_int_M, mv_K_int_J, mv_K_int_M;

    // Batch-2 workspace
    std::vector<cdouble> tmp2_src_charges;
    std::vector<cdouble> tmp2_phi;
    std::vector<cdouble> tmp2_grad[3];
    std::vector<cdouble> mv2_L_ext_J, mv2_L_ext_M, mv2_K_ext_J, mv2_K_ext_M;
    std::vector<cdouble> mv2_L_int_J, mv2_L_int_M, mv2_K_int_J, mv2_K_int_M;
    std::vector<cdouble> b4_src2, b4_src3;
    std::vector<cdouble> b4_pot2, b4_pot3;
    std::vector<cdouble> tmp_M1_phys, tmp_M2_phys;
    std::vector<cdouble> tmp_single_y;
    bool tmp_host_registered = false;

    double* d_f_p = nullptr;
    double* d_f_m = nullptr;
    double* d_jw_p = nullptr;
    double* d_jw_m = nullptr;
    double* d_div_p = nullptr;
    double* d_div_m = nullptr;
    double2* d_x1_complex = nullptr;
    double2* d_x2_complex = nullptr;
    double* d_x1_re = nullptr;
    double* d_x1_im = nullptr;
    double* d_x2_re = nullptr;
    double* d_x2_im = nullptr;
    double2* d_full_x1_complex = nullptr;
    double2* d_full_x2_complex = nullptr;
    double* d_full_x1_re = nullptr;
    double* d_full_x1_im = nullptr;
    double* d_full_x2_re = nullptr;
    double* d_full_x2_im = nullptr;
    double* d_L1_re = nullptr;
    double* d_L1_im = nullptr;
    double* d_K1_re = nullptr;
    double* d_K1_im = nullptr;
    double* d_L2_re = nullptr;
    double* d_L2_im = nullptr;
    double* d_K2_re = nullptr;
    double* d_K2_im = nullptr;
    double2* d_out1_complex = nullptr;
    double2* d_out2_complex = nullptr;
    double* d_mv1_re = nullptr;
    double* d_mv1_im = nullptr;
    double* d_mv2_re = nullptr;
    double* d_mv2_im = nullptr;
    double2* d_y1_complex = nullptr;
    double2* d_y2_complex = nullptr;
    double2* h_full_x1_complex = nullptr;
    double2* h_full_x2_complex = nullptr;
    double2* h_y1_complex = nullptr;
    double2* h_y2_complex = nullptr;
    bool pinned_matvec_stage = false;
    int* d_corr_row_ptr = nullptr;
    int* d_corr_col_idx = nullptr;
    double* d_corr_L_ext_re = nullptr;
    double* d_corr_L_ext_im = nullptr;
    double* d_corr_K_ext_re = nullptr;
    double* d_corr_K_ext_im = nullptr;
    double* d_corr_L_int_re = nullptr;
    double* d_corr_L_int_im = nullptr;
    double* d_corr_K_int_re = nullptr;
    double* d_corr_K_int_im = nullptr;
    double* d_corr_I = nullptr;

    // Initialize operator
    void init(const RWG& rwg, const Mesh& mesh,
              cdouble k_ext, cdouble k_int,
              cdouble eta_ext, cdouble eta_int,
              int quad_order = 7, int fmm_digits = 3, int max_leaf = 64,
              bool use_pfft_ = false, bool use_spfft_ = false);

    // Apply PMCHWT system: y = Z * x, where x and y are (2*N) vectors
    void matvec(const cdouble* x, cdouble* y);

    // Batched matvec: y1 = Z*x1, y2 = Z*x2
    void matvec_batch2(const cdouble* x1, const cdouble* x2, cdouble* y1, cdouble* y2);

    // Device-resident batched matvec for GPU GMRES.
    // Inputs and outputs are full PMCHWT vectors of length system_size on the active CUDA device.
    void matvec_batch2_device(const double2* d_x1, const double2* d_x2,
                              double2* d_y1, double2* d_y2);
    bool device_matvec_available() const;

    // Cleanup FMM resources
    void cleanup();

private:
    // Apply L operator via FMM: result = L(k) * x
    void L_operator(const cdouble* x, cdouble k, HelmholtzFMM& fmm, cdouble* result);

    // Apply K operator via FMM: result = K(k) * x
    void K_operator(const cdouble* x, cdouble k, HelmholtzFMM& fmm, cdouble* result);

    // Combined L+K operator: single FMM tree pass per vector component
    // Computes both L(k)*x and K(k)*x using evaluate_pot_grad
    void LK_combined(const cdouble* x, cdouble k, HelmholtzFMM& fmm,
                     cdouble* L_result, cdouble* K_result);

#ifndef BEM_FMM_ONLY
    void LK_combined(const cdouble* x, cdouble k, HelmholtzPFFT& pf,
                     cdouble* L_result, cdouble* K_result);

    void LK_combined(const cdouble* x, cdouble k, HelmholtzSurfacePFFT& spf,
                     cdouble* L_result, cdouble* K_result);
#endif

    // Batched combined L+K for two RHS vectors
    void LK_combined_batch2(const cdouble* x1, const cdouble* x2,
                             cdouble kv, HelmholtzFMM& fmm,
                             cdouble* L_result1, cdouble* K_result1,
                             cdouble* L_result2, cdouble* K_result2);
#ifndef BEM_FMM_ONLY
    void LK_combined_batch2(const cdouble* x1, const cdouble* x2,
                             cdouble kv, HelmholtzSurfacePFFT& spf,
                             cdouble* L_result1, cdouble* K_result1,
                             cdouble* L_result2, cdouble* K_result2);
#endif
    void LK_combined_batch2_device(const cdouble* x1, const cdouble* x2,
                                   cdouble kv, HelmholtzFMM& fmm,
                                   int L_slot, int K_slot);
    void LK_combined_batch2_device_split(const double* x1_re, const double* x1_im,
                                         const double* x2_re, const double* x2_im,
                                         cdouble kv, HelmholtzFMM& fmm,
                                         int L_slot, int K_slot);
    void LK_combined_batch4_jm_device_split(const double* J1_re, const double* J1_im,
                                            const double* J2_re, const double* J2_im,
                                            const double* M1_re, const double* M1_im,
                                            const double* M2_re, const double* M2_im,
                                            cdouble kv, HelmholtzFMM& fmm,
                                            int LJ_slot, int KJ_slot,
                                            int LM_slot, int KM_slot);
#ifndef BEM_FMM_ONLY
    void LK_combined_batch2_spfft_device_split(const double* x1_re, const double* x1_im,
                                               const double* x2_re, const double* x2_im,
                                               cdouble kv, HelmholtzSurfacePFFT& spf,
                                               int L_slot, int K_slot);
#endif

    // Precompute singular corrections
    void precompute_corrections(const RWG& rwg, const Mesh& mesh, int quad_order);

    void register_tmp_host_buffers();
    void unregister_tmp_host_buffers();
    void ensure_host_workspace();
    void init_device_workspace();
    void free_device_workspace();
};

#endif // BEM_BEM_FMM_H
