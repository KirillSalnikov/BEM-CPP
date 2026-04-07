#ifndef BEM_BEM_FMM_H
#define BEM_BEM_FMM_H

#include "types.h"
#include "fmm.h"
#include "pfft.h"
#include "surface_pfft.h"
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

    bool use_pfft;     // true = pFFT, false = FMM
    bool use_spfft;    // true = surface pFFT (2D per face)

    // FMM engines (one per wavenumber) -- used when !use_pfft
    HelmholtzFMM fmm_ext;
    HelmholtzFMM fmm_int;
    // pFFT engines -- used when use_pfft
    HelmholtzPFFT pfft_ext;
    HelmholtzPFFT pfft_int;
    // Surface pFFT engines -- used when use_spfft
    HelmholtzSurfacePFFT spfft_ext;
    HelmholtzSurfacePFFT spfft_int;
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

    // Batch workspace (extra charge/pot/grad buffers for batched P2P)
    std::vector<cdouble> b4_src2, b4_src3;  // charges for batch 2,3
    std::vector<cdouble> b4_pot2, b4_pot3;  // potentials for batch 2,3
    // Batch8 additional workspace (charges 4-7, pots 4-7, grads 3-5)
    std::vector<cdouble> b8_src[8];    // all 8 charge vectors
    std::vector<cdouble> b8_pot[8];    // all 8 potential results
    std::vector<cdouble> b8_grad[6];   // 6 gradient results

    // GPU arrays for charge packing (uploaded once in init, reused every matvec)
    double* d_f_p;       // (N*Nq*3) basis values, plus half
    double* d_f_m;       // (N*Nq*3) basis values, minus half
    double* d_jw_p;      // (N*Nq) Jacobian weights, plus half
    double* d_jw_m;      // (N*Nq)
    double* d_div_p;     // (N) divergence, plus half
    double* d_div_m;     // (N) divergence, minus half
    // GPU workspace for charge packing / result accumulation
    double* d_x_re;      // (N) input coefficient re
    double* d_x_im;      // (N) input coefficient im
    double* d_src_re;    // (2*N*Nq) packed charges re
    double* d_src_im;    // (2*N*Nq)
    double* d_phi_re;    // (2*N*Nq) FMM potential result re
    double* d_phi_im;    // (2*N*Nq)
    double* d_L_re;      // (N) L operator result re
    double* d_L_im;      // (N)
    double* d_K_re;      // (N) K operator result re
    double* d_K_im;      // (N)
    // Gradient storage for K operator (3 sets, one per source component d=0,1,2)
    double* d_grad_buf_re[3];  // each (2*N*Nq*3) = (Nt*3)
    double* d_grad_buf_im[3];
    bool gpu_pack_ready; // true after GPU arrays initialized

    // Host staging for x coefficient split (GPU-native SurfPFFT path)
    std::vector<double> h_x_split_re, h_x_split_im;  // (N)

    // Initialize operator (use_pfft_=true for pFFT acceleration)
    void init(const RWG& rwg, const Mesh& mesh,
              cdouble k_ext, cdouble k_int,
              double eta_ext, double eta_int,
              int quad_order = 7, int fmm_digits = 3, int max_leaf = 64,
              bool use_pfft_ = false, bool use_spfft_ = false);

    // Apply PMCHWT system: y = Z * x, where x and y are (2*N) vectors
    void matvec(const cdouble* x, cdouble* y);

    // Batched matvec: y1 = Z*x1, y2 = Z*x2
    void matvec_batch2(const cdouble* x1, const cdouble* x2, cdouble* y1, cdouble* y2);

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

    // pFFT variant of LK_combined
    void LK_combined(const cdouble* x, cdouble k, HelmholtzPFFT& pf,
                     cdouble* L_result, cdouble* K_result);

    // Surface pFFT variant of LK_combined
    void LK_combined(const cdouble* x, cdouble k, HelmholtzSurfacePFFT& spf,
                     cdouble* L_result, cdouble* K_result);

    // Batched combined L+K for two RHS vectors (FMM variant)
    void LK_combined_batch2(const cdouble* x1, const cdouble* x2,
                             cdouble kv, HelmholtzFMM& fmm,
                             cdouble* L_result1, cdouble* K_result1,
                             cdouble* L_result2, cdouble* K_result2);

    // Batched combined L+K for two RHS vectors (Surface pFFT variant — uses batch8 P2P)
    void LK_combined_batch2(const cdouble* x1, const cdouble* x2,
                             cdouble kv, HelmholtzSurfacePFFT& spf,
                             cdouble* L_result1, cdouble* K_result1,
                             cdouble* L_result2, cdouble* K_result2);

    // Precompute singular corrections
    void precompute_corrections(const RWG& rwg, const Mesh& mesh, int quad_order);
};

#endif // BEM_BEM_FMM_H
