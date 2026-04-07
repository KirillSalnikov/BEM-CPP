#ifndef BEM_PRECOND_H
#define BEM_PRECOND_H

#include "types.h"
#include <vector>
#include <complex>

struct BemFmmOperator;

enum PrecondMode {
    PREC_NONE = 0,
    PREC_ILU0 = 1,      // ILU(0) on near-field sparse matrix
    PREC_BLOCKJ = 2,     // Block-Jacobi: spatial cell blocks with dense LU
    PREC_DIAG = 3,       // 2x2 block-diagonal (self-interaction only)
    PREC_NEARLU = 4      // Reserved
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

    // Sparse 2N×2N matrix in CSR format
    std::vector<int> csr_row_ptr;       // (2N+1)
    std::vector<int> csr_col_idx;       // (nnz_total)
    std::vector<cdouble> csr_val;       // (nnz_total)

    // For each row i, index into csr_col_idx where the diagonal element is
    std::vector<int> diag_ptr;          // (2N)

    // Block-Jacobi (PREC_BLOCKJ): spatial cell blocks
    int n_blocks;
    std::vector<int> block_sizes;           // (n_blocks) RWG count per block (own only)
    std::vector<std::vector<int>> block_rwg;// (n_blocks) global RWG indices per block (own only)
    std::vector<std::vector<cdouble>> block_lu; // (n_blocks) LU-factored dense (col-major)
    std::vector<std::vector<int>> block_piv;    // (n_blocks) pivot arrays
    std::vector<int> rwg_block;             // (N) which block each RWG belongs to
    std::vector<int> rwg_local;             // (N) local index within block

    // Overlap (RAS): extended blocks include neighboring RWGs
    int overlap_layers;                     // 0 = standard BlockJ, >0 = RAS overlap
    std::vector<int> block_sizes_ext;       // (n_blocks) extended RWG count (own + overlap)
    std::vector<std::vector<int>> block_rwg_ext; // (n_blocks) extended RWG indices (own first, then overlap)

    // 2x2 block-diagonal preconditioner (PREC_DIAG)
    // For each RWG i, stores inverted 2x2 block: [A,B;C,D]^{-1}
    std::vector<cdouble> diag_inv;  // (N*4) flat: [inv_A, inv_B, inv_C, inv_D] per RWG

    // GPU Block-Jacobi data (uploaded by upload_to_gpu)
    cuDoubleComplex* d_lu_flat;    // row-major LU factors, all blocks concatenated
    int* d_piv_flat;               // pivots, all blocks concatenated
    int* d_rwg_flat;               // RWG indices (extended), all blocks concatenated
    int* d_blk_B;                  // own block sizes (B_own per block)
    int* d_blk_B_ext;             // extended block sizes (B_ext per block, = B_own when no overlap)
    int* d_lu_off;                 // element offset into d_lu_flat per block
    int* d_piv_off;                // element offset into d_piv_flat per block
    int* d_rwg_off;                // element offset into d_rwg_flat per block
    cuDoubleComplex* d_buf_r;      // (N2) GPU buffer for apply
    cuDoubleComplex* d_buf_z;      // (N2) GPU buffer for apply
    cuDoubleComplex* d_workspace;  // (n_blocks * max_B2) workspace for LU solve
    cuDoubleComplex* d_buf_z2;     // (N2) second output buffer for paired apply
    cuDoubleComplex* d_workspace2; // second workspace for paired apply
    cudaStream_t precond_stream2;  // second CUDA stream for paired apply
    bool gpu_ready;
    int max_B2;                    // largest 2*B_ext across all blocks

    // GPU diagonal preconditioner data
    cuDoubleComplex* d_diag_inv;   // (N*4) on GPU

    // Build preconditioner from near-field BEM entries
    // radius_mult: cell_size = radius_mult * avg_extent (default 2.0)
    // max_block_rwg: max RWG per block-Jacobi block (default 1500)
    // overlap: number of neighbor layers for RAS overlap (default 0 = standard BlockJ)
    void build(BemFmmOperator& op, PrecondMode mode, double radius_mult = 2.0,
               int max_block_rwg = 1500, int overlap = 0);

    // Apply: z = M^{-1} * r  (host pointers; uses GPU internally if available)
    void apply(const cdouble* r, cdouble* z) const;

    // GPU apply: z = M^{-1} * r  (device pointers, no transfers)
    void apply_gpu(const cuDoubleComplex* d_r, cuDoubleComplex* d_z) const;

    // Paired GPU apply: z1 = M^{-1}*r1, z2 = M^{-1}*r2 concurrently (two CUDA streams)
    void apply_gpu_paired(const cuDoubleComplex* d_r1, cuDoubleComplex* d_z1,
                          const cuDoubleComplex* d_r2, cuDoubleComplex* d_z2) const;

    // Upload Block-Jacobi data to GPU (called at end of build)
    void upload_to_gpu();

    // Free GPU memory
    void cleanup_gpu();
};

// Auto-select optimal preconditioner parameters based on N and ka
struct AutoPrecondParams {
    PrecondMode mode;
    int block_size;
    int overlap;
    double radius;

    static AutoPrecondParams compute(int N, double ka);
};

#endif // BEM_PRECOND_H
