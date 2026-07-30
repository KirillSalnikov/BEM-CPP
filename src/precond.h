#ifndef BEM_PRECOND_H
#define BEM_PRECOND_H

#include "types.h"
#include <cusparse.h>
#include <cstdint>
#include <vector>
#include <complex>

struct BemFmmOperator;
struct RWG;
struct Mesh;

uint64_t bem_neural_geometry_signature(const RWG& rwg);

// Near-field 2x2 block Jacobi preconditioner for PMCHWT system.
// For each RWG function m, inverts the 2x2 self-interaction block:
//   [A(m,m)  B(m,m)]^{-1}   where A = eta*L, B = -K, C = K, D = L/eta
//   [C(m,m)  D(m,m)]
// Gives ~17% speedup (12% fewer GMRES iterations).
struct NearFieldPrecond {
    struct LocalBlock {
        std::vector<int> ids;
        std::vector<cdouble> lu;
        std::vector<int> piv;
    };

    int N;      // RWG count
    int N2;     // 2*N (system size)

    // 2x2 block inverse: z[m] = a*r[m] + b*r[N+m], z[N+m] = c*r[m] + d*r[N+m]
    std::vector<cdouble> blk_inv;   // (N*4): [a,b,c,d] for each m
    std::vector<cdouble> diag_blk;  // (N*4): approximate PMCHWT diagonal block [A,B,C,D]

    // Sparse local correction PMCHWT blocks, same CSR pattern as op corrections.
    // Diagonal entries are zero here because diag_blk carries the full diagonal.
    std::vector<int> near_row_ptr;
    std::vector<int> near_col_idx;
    std::vector<cdouble> near_blk;  // (nnz*4): [A,B,C,D]
    int richardson_sweeps = 0;
    double richardson_omega = 0.8;
    bool block_schwarz = false;
    bool morton_block_jacobi = false;
    bool ilu0 = false;
    bool mass_matrix = false;
    bool calderon_rwg = false;
    bool neural_sparse = false;
    int neural_coarse_rank = 0;
    // Column-major factors of the global correction L = coarse_update * coarse_q^H.
    std::vector<cdouble> neural_coarse_q;
    std::vector<cdouble> neural_coarse_update;
    int max_block_basis = 8;
    int max_block_dim = 0;
    std::vector<LocalBlock> blocks;
    std::vector<double> block_weight;

    int* d_block_offsets = nullptr;
    int* d_block_ids = nullptr;
    int* d_block_piv = nullptr;
    double* d_block_lu_re = nullptr;
    double* d_block_lu_im = nullptr;
    double* d_block_weight = nullptr;
    double2* d_r_complex = nullptr;
    double2* d_z_complex = nullptr;
    double* d_r_re = nullptr;
    double* d_r_im = nullptr;
    double* d_z_re = nullptr;
    double* d_z_im = nullptr;
    double* d_Az_re = nullptr;
    double* d_Az_im = nullptr;
    double* d_err_re = nullptr;
    double* d_err_im = nullptr;
    double* d_corr_re = nullptr;
    double* d_corr_im = nullptr;
    double* d_diag_re = nullptr;
    double* d_diag_im = nullptr;
    double* d_near_re = nullptr;
    double* d_near_im = nullptr;
    double2* d_neural_blk = nullptr;
    float2* d_neural_coarse_q = nullptr;
    float2* d_neural_coarse_update = nullptr;
    double2* d_neural_coarse_coeff = nullptr;
    int* d_near_row_ptr = nullptr;
    int* d_near_col_idx = nullptr;
    std::vector<int> ilu_row_ptr;
    std::vector<int> ilu_col_idx;
    std::vector<cdouble> ilu_val;
    std::vector<int> ilu_diag_ptr;
    int* d_ilu_row_ptr = nullptr;
    int* d_ilu_col_idx = nullptr;
    double2* d_ilu_val = nullptr;
    double2* d_ilu_rhs = nullptr;
    double2* d_ilu_tmp = nullptr;
    double2* d_ilu_out = nullptr;
    cusparseHandle_t ilu_handle = nullptr;
    cusparseSpMatDescr_t ilu_mat_l = nullptr;
    cusparseSpMatDescr_t ilu_mat_u = nullptr;
    cusparseDnVecDescr_t ilu_vec_in = nullptr;
    cusparseDnVecDescr_t ilu_vec_tmp = nullptr;
    cusparseDnVecDescr_t ilu_vec_out = nullptr;
    cusparseSpSVDescr_t ilu_spsv_l = nullptr;
    cusparseSpSVDescr_t ilu_spsv_u = nullptr;
    void* d_ilu_buffer_l = nullptr;
    void* d_ilu_buffer_u = nullptr;
    std::vector<int> mass_row_ptr;
    std::vector<int> mass_col_idx;
    std::vector<double> mass_val;
    std::vector<double> mass_inv_diag;
    int* d_mass_row_ptr = nullptr;
    int* d_mass_col_idx = nullptr;
    double* d_mass_val = nullptr;
    double* d_mass_inv_diag = nullptr;
    double2* d_mass_x = nullptr;
    double2* d_mass_r = nullptr;
    double2* d_mass_p = nullptr;
    double2* d_mass_ap = nullptr;
    double2* d_calderon_mass0 = nullptr;
    double2* d_calderon_mass1 = nullptr;
    double2* d_calderon_op0 = nullptr;
    double2* d_calderon_op1 = nullptr;
    BemFmmOperator* calderon_operator = nullptr;
    double* d_mass_norm_sum0 = nullptr;
    double* d_mass_norm_sum1 = nullptr;
    double2* d_mass_dot_sum0 = nullptr;
    double2* d_mass_dot_sum1 = nullptr;
    int mass_reduction_blocks = 0;
    mutable std::vector<double> mass_host_norm0;
    mutable std::vector<double> mass_host_norm1;
    mutable std::vector<double2> mass_host_dot0;
    mutable std::vector<double2> mass_host_dot1;
    double mass_cg_tolerance = 1e-10;
    int mass_cg_max_iterations = 40;
    mutable long long mass_apply_count = 0;
    mutable long long mass_iteration_count = 0;
    mutable int mass_max_iterations_used = 0;
    mutable double mass_max_relative_residual = 0.0;
    mutable long long calderon_operator_actions = 0;
    int device_near_nnz = 0;
    int device_block_count = 0;
    int device_block_dim = 0;
    int device_ids_count = 0;
    int device_lu_count = 0;
    bool device_ready = false;
    mutable std::vector<cdouble> tmp_Az;
    mutable std::vector<cdouble> tmp_err;
    mutable std::vector<cdouble> tmp_corr;

    // Build preconditioner from near-field BEM entries
    void build(BemFmmOperator& op, const RWG* rwg_geometry = nullptr,
               const Mesh* mesh_geometry = nullptr);

    bool dump_neural_features(const char* path, const RWG& rwg, const Mesh& mesh,
                              BemFmmOperator& op, double ka, double n_re,
                              double n_im, bool balanced_system,
                              int coarse_rank = 0) const;

    // Load a neural sparse approximate inverse in BEM's C++ RWG ordering.
    bool load_neural(const char* path, int expected_n, double expected_ka,
                     double expected_n_re, double expected_n_im,
                     bool expected_balanced, uint64_t expected_geometry_signature);

    // Apply: z = M^{-1} * r
    void apply(const cdouble* r, cdouble* z) const;
    void apply_pair(const cdouble* r1, const cdouble* r2, cdouble* z1, cdouble* z2) const;

    void apply_block_inv(const cdouble* r, cdouble* z) const;
    void apply_near(const cdouble* x, cdouble* y) const;
    void apply_block_schwarz(const cdouble* r, cdouble* z) const;
    void apply_block_schwarz_cuda(const cdouble* r, cdouble* z) const;
    void apply_block_schwarz_cuda_device(const double* in_re, const double* in_im,
                                         double* out_re, double* out_im) const;
    void apply_mass_device(const double2* d_r, double2* d_z) const;
    void apply_calderon_pair_device(const double2* d_r0, const double2* d_r1,
                                    double2* d_z0, double2* d_z1) const;
    bool device_apply_available() const;
    bool uses_right_device_preconditioning() const {
        return neural_sparse || ilu0 || morton_block_jacobi;
    }
    void apply_device_complex(const double2* d_r, double2* d_z) const;
    void apply_device_complex_pair(const double2* d_r0, const double2* d_r1,
                                   double2* d_z0, double2* d_z1) const;
    long long full_operator_action_count() const { return calderon_operator_actions; }
    void upload_device();
    void cleanup_device();

    ~NearFieldPrecond() { cleanup_device(); }
};

#endif // BEM_PRECOND_H
