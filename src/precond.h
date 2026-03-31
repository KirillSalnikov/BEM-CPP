#ifndef BEM_PRECOND_H
#define BEM_PRECOND_H

#include "types.h"
#include <vector>
#include <complex>

struct BemFmmOperator;

enum PrecondMode {
    PREC_NONE = 0,
    PREC_DIAG = 1,      // Diagonal scaling (Z_diag^{-1})
    PREC_ILU0 = 2,      // ILU(0) on near-field sparse matrix
    PREC_NEARLU = 3,     // Full LU on near-field sparse matrix (small N only)
    PREC_BLOCKJ = 4      // Block-Jacobi: spatial cell blocks with dense LU
};

// Preconditioner for PMCHWT BEM system.
//
// The 2N×2N system has structure:
//   [ eta_e*L_ext + eta_i*L_int    -(K_ext + K_int)        ] [J]
//   [  K_ext + K_int           L_ext/eta_e + L_int/eta_i   ] [M]
//
// Right-preconditioning in GMRES: solve Z*M^{-1} * (M*x) = b.
struct NearFieldPrecond {
    int N;      // RWG count
    int N2;     // 2*N (system size)
    PrecondMode mode;

    // Diagonal preconditioner (PREC_DIAG)
    std::vector<cdouble> diag_val;         // (2N) diagonal entries of near-field Z

    // Sparse 2N×2N matrix in CSR format
    std::vector<int> csr_row_ptr;       // (2N+1)
    std::vector<int> csr_col_idx;       // (nnz_total)
    std::vector<cdouble> csr_val;       // (nnz_total)

    // For each row i, index into csr_col_idx where the diagonal element is
    std::vector<int> diag_ptr;          // (2N)

    // Full LU factorization (PREC_NEARLU, small N only)
    std::vector<cdouble> lu_dense;      // (2N × 2N) column-major
    std::vector<int> lu_piv;            // (2N) pivot indices

    // Block-Jacobi (PREC_BLOCKJ): spatial cell blocks
    int n_blocks;
    std::vector<int> block_sizes;           // (n_blocks) RWG count per block
    std::vector<std::vector<int>> block_rwg;// (n_blocks) global RWG indices per block
    std::vector<std::vector<cdouble>> block_lu; // (n_blocks) LU-factored 2B×2B dense (col-major)
    std::vector<std::vector<int>> block_piv;    // (n_blocks) pivot arrays (2B)
    std::vector<int> rwg_block;             // (N) which block each RWG belongs to
    std::vector<int> rwg_local;             // (N) local index within block

    // GPU Block-Jacobi data (uploaded by upload_to_gpu)
    cuDoubleComplex* d_lu_flat;    // row-major LU factors, all blocks concatenated
    int* d_piv_flat;               // pivots, all blocks concatenated
    int* d_rwg_flat;               // RWG indices, all blocks concatenated
    int* d_blk_B;                  // block sizes (B per block)
    int* d_lu_off;                 // element offset into d_lu_flat per block
    int* d_piv_off;                // element offset into d_piv_flat per block
    int* d_rwg_off;                // element offset into d_rwg_flat per block
    cuDoubleComplex* d_buf_r;      // (N2) GPU buffer for apply
    cuDoubleComplex* d_buf_z;      // (N2) GPU buffer for apply
    bool gpu_ready;
    int max_B2;                    // largest 2*B across all blocks

    // Build preconditioner from near-field BEM entries
    // radius_mult: cell_size = radius_mult * avg_extent (default 2.0)
    // max_block_rwg: max RWG per block-Jacobi block (default 1500)
    void build(BemFmmOperator& op, PrecondMode mode, double radius_mult = 2.0,
               int max_block_rwg = 1500);

    // Apply: z = M^{-1} * r  (host pointers; uses GPU internally if available)
    void apply(const cdouble* r, cdouble* z) const;

    // GPU apply: z = M^{-1} * r  (device pointers, no transfers)
    void apply_gpu(const cuDoubleComplex* d_r, cuDoubleComplex* d_z) const;

    // Upload Block-Jacobi data to GPU (called at end of build)
    void upload_to_gpu();

    // Free GPU memory
    void cleanup_gpu();
};

#endif // BEM_PRECOND_H
