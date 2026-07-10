#ifndef BEM_PRECOND_H
#define BEM_PRECOND_H

#include "types.h"
#include <vector>
#include <complex>

struct BemFmmOperator;

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
    int* d_near_row_ptr = nullptr;
    int* d_near_col_idx = nullptr;
    int device_near_nnz = 0;
    int device_block_count = 0;
    int device_ids_count = 0;
    int device_lu_count = 0;
    bool device_ready = false;
    mutable std::vector<cdouble> tmp_Az;
    mutable std::vector<cdouble> tmp_err;
    mutable std::vector<cdouble> tmp_corr;

    // Build preconditioner from near-field BEM entries
    void build(BemFmmOperator& op);

    // Apply: z = M^{-1} * r
    void apply(const cdouble* r, cdouble* z) const;
    void apply_pair(const cdouble* r1, const cdouble* r2, cdouble* z1, cdouble* z2) const;

    void apply_block_inv(const cdouble* r, cdouble* z) const;
    void apply_near(const cdouble* x, cdouble* y) const;
    void apply_block_schwarz(const cdouble* r, cdouble* z) const;
    void apply_block_schwarz_cuda(const cdouble* r, cdouble* z) const;
    void apply_block_schwarz_cuda_device(const double* in_re, const double* in_im,
                                         double* out_re, double* out_im) const;
    bool device_apply_available() const;
    void apply_device_complex(const double2* d_r, double2* d_z) const;
    void upload_device();
    void cleanup_device();

    ~NearFieldPrecond() { cleanup_device(); }
};

#endif // BEM_PRECOND_H
