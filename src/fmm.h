#ifndef BEM_FMM_H
#define BEM_FMM_H

#include "types.h"
#include "octree.h"
#include "sphere_quad.h"
#include <complex>
#include <vector>

struct HelmholtzFMM {
    Octree tree;
    SphereQuad squad;

    cdouble k;          // wavenumber
    int Nt, Ns;         // number of target / source points
    int p;              // truncation order
    int L;              // number of plane-wave directions

    // M2L transfer function cache
    // Indexed by unique displacement vector hash
    struct M2LPair { int tgt_node, src_node, transfer_idx; };
    std::vector<cdouble>   transfer_cache;  // (n_unique * L) flat
    int n_unique_transfers;

    // Per-level M2L batch data
    struct M2LBatch {
        std::vector<int> tgt_idx, src_idx, transfer_idx;
        int n_pairs;
    };
    std::vector<M2LBatch> m2l_batches;  // indexed by level

    // M2M / L2L shift vectors: per parent-child pair
    struct ShiftPair { int parent, child; };
    struct LevelShifts {
        std::vector<ShiftPair> pairs;
        std::vector<cdouble>   shifts;  // (n_pairs * L) flat
    };
    std::vector<LevelShifts> m2m_data;  // indexed by level
    std::vector<LevelShifts> l2l_data;  // indexed by level

    // P2P CSR structure (for near-field)
    std::vector<int> p2p_offsets;  // (Nt+1)
    std::vector<int> p2p_indices;  // (nnz)
    int p2p_nnz;

    // GPU arrays (device pointers)
    // Allocated in init(), freed in cleanup()
    double* d_tgt_pts;        // (Nt*3)
    double* d_src_pts;        // (Ns*3)
    int*    d_p2p_offsets;    // (Nt+1)
    int*    d_p2p_indices;    // (nnz)

    // FMM workspace on GPU (float32 -- sufficient for ~3-digit FMM accuracy)
    fmm_real* d_multi_re;       // (n_nodes * L)
    fmm_real* d_multi_im;
    fmm_real* d_local_re;       // (n_nodes * L)
    fmm_real* d_local_im;

    // M2L transfers on GPU (float32)
    fmm_real* d_transfer_re;    // (n_unique * L)
    fmm_real* d_transfer_im;
    int*    d_m2l_tgt;        // per-level batch arrays (concatenated) [legacy, kept for cleanup]
    int*    d_m2l_src;
    int*    d_m2l_tidx;

    // Optimized M2L: target-sorted CSR structure
    // For each unique target node, stores (src_node, transfer_idx) pairs contiguously
    int*    d_m2l_csr_offsets;   // (n_unique_targets+1) per level, concatenated
    int*    d_m2l_csr_tgt_nodes; // (n_unique_targets) target node index
    int*    d_m2l_csr_src;       // (n_total_pairs) source node indices, sorted by target
    int*    d_m2l_csr_tidx;      // (n_total_pairs) transfer type indices, sorted by target
    fmm_real* d_transfer_ri;     // (n_unique * L * 2) interleaved [re,im,re,im,...] for vectorized loads

    // Per-level info for optimized M2L kernel
    struct M2LCSRLevelInfo {
        int offsets_start;  // index into d_m2l_csr_offsets (n_targets+1 entries)
        int nodes_start;    // index into d_m2l_csr_tgt_nodes (n_targets entries)
        int pair_offset;    // index into d_m2l_csr_src / d_m2l_csr_tidx
        int n_targets;
    };
    std::vector<M2LCSRLevelInfo> m2l_csr_level_info;

    // M2M/L2L shifts on GPU (float32)
    fmm_real* d_m2m_shift_re;   // concatenated across levels
    fmm_real* d_m2m_shift_im;
    int*    d_m2m_parent;
    int*    d_m2m_child;
    fmm_real* d_l2l_shift_re;
    fmm_real* d_l2l_shift_im;
    int*    d_l2l_parent;
    int*    d_l2l_child;

    // Per-level offset info for M2L/M2M/L2L kernel launches
    struct LevelKernelInfo { int offset, count; };
    std::vector<LevelKernelInfo> m2l_level_info;
    std::vector<LevelKernelInfo> m2m_level_info;
    std::vector<LevelKernelInfo> l2l_level_info;

    // Leaf -> target/source mapping for L2P/P2M
    struct LeafInfo {
        int node_idx;
        int tgt_sorted_start, tgt_count;  // in sorted array
        int src_sorted_start, src_count;
    };
    std::vector<LeafInfo> leaf_info;

    // Per-leaf original ID arrays (for P2M/L2P kernels)
    std::vector<int> h_leaf_indices;      // (n_leaves) node index per leaf
    std::vector<int> h_tgt_id_offsets;    // (n_leaves+1) offsets into h_tgt_ids_flat
    std::vector<int> h_src_id_offsets;    // (n_leaves+1) offsets into h_src_ids_flat
    std::vector<int> h_tgt_ids_flat;      // flat array of original target IDs
    std::vector<int> h_src_ids_flat;      // flat array of original source IDs
    std::vector<double> h_node_centers;   // (n_nodes*3) node centers

    int n_nodes;

    // Charge/result buffers on GPU (per-evaluation, reused)
    double* d_charges_re;     // (Ns)
    double* d_charges_im;
    double* d_result_re;      // (Nt)
    double* d_result_im;
    double* d_grad_re;        // (Nt*3) for gradient mode
    double* d_grad_im;

    // Batch-2 workspace (second charge vector)
    double* d_charges2_re = nullptr;
    double* d_charges2_im = nullptr;
    double* d_result2_re = nullptr;
    double* d_result2_im = nullptr;
    fmm_real* d_multi2_re = nullptr;
    fmm_real* d_multi2_im = nullptr;
    fmm_real* d_local2_re = nullptr;
    fmm_real* d_local2_im = nullptr;
    double* d_grad2_re = nullptr;       // (Nt*3) gradient for second charge vector
    double* d_grad2_im = nullptr;
    double* d_gy2_re_cached = nullptr;  // (Nt) grad y/z workspace for second vector
    double* d_gy2_im_cached = nullptr;
    double* d_gz2_re_cached = nullptr;
    double* d_gz2_im_cached = nullptr;

    // Cached GPU arrays for run_tree() -- allocated once in init(), reused every call
    double* d_node_centers_cached;   // (n_nodes*3) -- double for position accuracy
    fmm_real* d_dirs_cached;         // (L*3) -- float32 for FMM kernels
    fmm_real* d_weights_cached;      // (L) -- float32 for FMM kernels
    int*    d_leaf_idx_cached;       // (n_leaves)
    int*    d_src_id_offsets_cached;  // (n_leaves+1)
    int*    d_src_ids_cached;        // (h_src_ids_flat.size())
    int*    d_tgt_id_offsets_cached;  // (n_leaves+1)
    int*    d_tgt_ids_cached;        // (h_tgt_ids_flat.size())

    // Cached gradient workspace arrays
    double* d_gy_re_cached;          // (Nt) for gradient y component
    double* d_gy_im_cached;
    double* d_gz_re_cached;          // (Nt) for gradient z component
    double* d_gz_im_cached;

    // Pre-allocated temp buffers for gradient repack (avoid per-call cudaMalloc)
    double* d_gx_re_tmp;             // (Nt) for repack_gradient_kernel
    double* d_gx_im_tmp;

    // Pre-allocated host-side re/im split buffers (avoid per-call std::vector alloc)
    std::vector<double> h_q_re_buf, h_q_im_buf;       // (Ns) for charges
    std::vector<double> h_q2_re_buf, h_q2_im_buf;     // (Ns) for batch-2 charges
    std::vector<double> h_res_re_buf, h_res_im_buf;   // (max(Nt, Nt*3)) for results
    std::vector<double> h_res2_re_buf, h_res2_im_buf; // (max(Nt, Nt*3)) for batch-2 results

    // CUDA streams for P2P/FMM pipeline overlap
    cudaStream_t stream_fmm;
    cudaStream_t stream_p2p;

    // Separate P2P output buffers for concurrent execution with FMM tree
    double* d_p2p_pot_re;      // (Nt) potential
    double* d_p2p_pot_im;
    double* d_p2p_pot2_re;     // (Nt) batch-2 second potential
    double* d_p2p_pot2_im;
    double* d_p2p_gx_re;      // (Nt) gradient x
    double* d_p2p_gx_im;
    double* d_p2p_gy_re;      // (Nt) gradient y
    double* d_p2p_gy_im;
    double* d_p2p_gz_re;      // (Nt) gradient z
    double* d_p2p_gz_im;
    double* d_p2p_gx2_re;     // (Nt) batch-2 gradient x
    double* d_p2p_gx2_im;
    double* d_p2p_gy2_re;     // (Nt) batch-2 gradient y
    double* d_p2p_gy2_im;
    double* d_p2p_gz2_re;     // (Nt) batch-2 gradient z
    double* d_p2p_gz2_im;

    bool initialized;

    HelmholtzFMM() : initialized(false),
        d_tgt_pts(0), d_src_pts(0), d_p2p_offsets(0), d_p2p_indices(0),
        d_multi_re(0), d_multi_im(0), d_local_re(0), d_local_im(0),
        d_transfer_re(0), d_transfer_im(0),
        d_m2l_tgt(0), d_m2l_src(0), d_m2l_tidx(0),
        d_m2l_csr_offsets(0), d_m2l_csr_tgt_nodes(0),
        d_m2l_csr_src(0), d_m2l_csr_tidx(0), d_transfer_ri(0),
        d_m2m_shift_re(0), d_m2m_shift_im(0), d_m2m_parent(0), d_m2m_child(0),
        d_l2l_shift_re(0), d_l2l_shift_im(0), d_l2l_parent(0), d_l2l_child(0),
        d_charges_re(0), d_charges_im(0), d_result_re(0), d_result_im(0),
        d_grad_re(0), d_grad_im(0),
        d_charges2_re(0), d_charges2_im(0), d_result2_re(0), d_result2_im(0),
        d_multi2_re(0), d_multi2_im(0), d_local2_re(0), d_local2_im(0),
        d_grad2_re(0), d_grad2_im(0),
        d_gy2_re_cached(0), d_gy2_im_cached(0), d_gz2_re_cached(0), d_gz2_im_cached(0),
        d_node_centers_cached(0), d_dirs_cached(0), d_weights_cached(0),
        d_leaf_idx_cached(0), d_src_id_offsets_cached(0), d_src_ids_cached(0),
        d_tgt_id_offsets_cached(0), d_tgt_ids_cached(0),
        d_gy_re_cached(0), d_gy_im_cached(0), d_gz_re_cached(0), d_gz_im_cached(0),
        d_gx_re_tmp(0), d_gx_im_tmp(0),
        stream_fmm(0), stream_p2p(0),
        d_p2p_pot_re(0), d_p2p_pot_im(0), d_p2p_pot2_re(0), d_p2p_pot2_im(0),
        d_p2p_gx_re(0), d_p2p_gx_im(0), d_p2p_gy_re(0), d_p2p_gy_im(0),
        d_p2p_gz_re(0), d_p2p_gz_im(0),
        d_p2p_gx2_re(0), d_p2p_gx2_im(0), d_p2p_gy2_re(0), d_p2p_gy2_im(0),
        d_p2p_gz2_re(0), d_p2p_gz2_im(0) {}

    // Initialize: build tree, precompute transfers, upload to GPU
    void init(const double* targets, int n_tgt,
              const double* sources, int n_src,
              cdouble k_val, int digits = 3, int max_leaf = 64);

    // Evaluate: y[i] = sum_j G(r_i, r_j) * q[j]
    // charges: host array (Ns), result: host array (Nt)
    void evaluate(const cdouble* charges, cdouble* result);

    // Zero-copy evaluate: charges already on GPU as split re/im, results stay on GPU
    // d_q_re/im: device (Ns), d_res_re/im: device (Nt) — output added to existing values
    void evaluate_gpu(const double* d_q_re, const double* d_q_im,
                      double* d_res_re, double* d_res_im);

    // Zero-copy pot+grad: charges on GPU, results on GPU
    // d_q_re/im: device (Ns), d_pot_re/im: device (Nt), d_grad_re/im: device (Nt*3)
    void evaluate_pot_grad_gpu(const double* d_q_re, const double* d_q_im,
                                double* d_pot_re, double* d_pot_im,
                                double* d_grad_re, double* d_grad_im);

    // Evaluate gradient: grad[i] = sum_j nabla_G(r_i, r_j) * q[j]
    // charges: host array (Ns), grad_result: host array (Nt*3) [x0,y0,z0,x1,y1,z1,...]
    void evaluate_gradient(const cdouble* charges, cdouble* grad_result);

    // Evaluate both potential and gradient in a single tree traversal.
    // charges: host (Ns), pot_result: host (Nt), grad_result: host (Nt*3)
    void evaluate_pot_grad(const cdouble* charges, cdouble* pot_result, cdouble* grad_result);

    // Batched evaluate: two charge vectors, two result vectors, single tree traversal
    void evaluate_batch2(const cdouble* charges1, const cdouble* charges2,
                         cdouble* result1, cdouble* result2);

    // Batched pot+grad: two charge vectors, two pot + two grad results, single tree traversal
    void evaluate_pot_grad_batch2(const cdouble* charges1, const cdouble* charges2,
                                   cdouble* pot1, cdouble* grad1,
                                   cdouble* pot2, cdouble* grad2);

    // Run FMM tree traversal (P2M->M2M->M2L->L2L->L2P/P2P)
    void run_tree(const double* h_q_re, const double* h_q_im, bool need_grad);

    // Run fused batch-2 tree traversal: two charge vectors share a single tree walk
    void run_tree_batch2(const double* h_q1_re, const double* h_q1_im,
                         const double* h_q2_re, const double* h_q2_im,
                         bool need_grad);

    // Free GPU memory
    void cleanup();

    ~HelmholtzFMM() { if (initialized) cleanup(); }
};

// Spherical Hankel function of the first kind: h_n^(1)(z) = j_n(z) + i*y_n(z)
cdouble spherical_hankel1(int n, cdouble z);

#endif // BEM_FMM_H
