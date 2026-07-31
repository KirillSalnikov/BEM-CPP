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
    float*  d_tgt_pts_fp32;   // mixed-precision near-field copy
    float*  d_src_pts_fp32;
    float*  d_dirs_fp32_cached;
    float*  d_weights_fp32_cached;
    float2* d_phase_cache_fp32; // reusable real-k leaf phases, Nt*L
    float2* d_l2p_phase_cache_fp32; // direction-major phases, L*Nt
    int*    d_p2p_offsets;    // (Nt+1)
    int*    d_p2p_indices;    // (nnz)

    // FMM workspace on GPU
    double* d_multi_re;       // (n_nodes * L)
    double* d_multi_im;
    double* d_local_re;       // (n_nodes * L)
    double* d_local_im;

    // M2L transfers on GPU
    double* d_transfer_re;    // (n_unique * L)
    double* d_transfer_im;
    float*  d_transfer_re_fp32;
    float*  d_transfer_im_fp32;
    int*    d_m2l_tgt;        // per-level batch arrays (concatenated)
    int*    d_m2l_src;
    int*    d_m2l_tidx;
    int*    d_m2l_row_target; // target-grouped M2L rows
    int*    d_m2l_row_start;
    int*    d_m2l_row_end;

    // M2M/L2L shifts on GPU
    double* d_m2m_shift_re;   // concatenated across levels
    double* d_m2m_shift_im;
    int*    d_m2m_parent;
    int*    d_m2m_child;
    double* d_l2l_shift_re;
    double* d_l2l_shift_im;
    int*    d_l2l_parent;
    int*    d_l2l_child;

    // Per-level offset info for M2L/M2M/L2L kernel launches
    struct LevelKernelInfo { int offset, count; };
    std::vector<LevelKernelInfo> m2l_level_info;
    std::vector<LevelKernelInfo> m2l_row_level_info;
    std::vector<LevelKernelInfo> m2m_level_info;
    std::vector<LevelKernelInfo> l2l_level_info;

    // Leaf → target/source mapping for L2P/P2M
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
    double* d_hess_re;        // (Nt*6): xx, xy, xz, yy, yz, zz
    double* d_hess_im;

    // Batch-2 workspace (second charge vector)
    double* d_charges2_re = nullptr;
    double* d_charges2_im = nullptr;
    double* d_result2_re = nullptr;
    double* d_result2_im = nullptr;
    double* d_grad2_re = nullptr;
    double* d_grad2_im = nullptr;
    double* d_multi2_re = nullptr;
    double* d_multi2_im = nullptr;
    double* d_local2_re = nullptr;
    double* d_local2_im = nullptr;
    double* d_charges3_re = nullptr;
    double* d_charges3_im = nullptr;
    double* d_charges4_re = nullptr;
    double* d_charges4_im = nullptr;
    double* d_result3_re = nullptr;
    double* d_result3_im = nullptr;
    double* d_result4_re = nullptr;
    double* d_result4_im = nullptr;
    double* d_grad3_re = nullptr;
    double* d_grad3_im = nullptr;
    double* d_grad4_re = nullptr;
    double* d_grad4_im = nullptr;
    double* d_hess2_re = nullptr;
    double* d_hess2_im = nullptr;
    double* d_hess3_re = nullptr;
    double* d_hess3_im = nullptr;
    double* d_hess4_re = nullptr;
    double* d_hess4_im = nullptr;
    double* d_multi3_re = nullptr;
    double* d_multi3_im = nullptr;
    double* d_multi4_re = nullptr;
    double* d_multi4_im = nullptr;
    double* d_local3_re = nullptr;
    double* d_local3_im = nullptr;
    double* d_local4_re = nullptr;
    double* d_local4_im = nullptr;
    double* d_pair_multi_re = nullptr; // six vector components, 6*n_nodes*L
    double* d_pair_multi_im = nullptr;
    float* d_pair_multi_re_fp32 = nullptr;
    float* d_pair_multi_im_fp32 = nullptr;
    double* d_pair_local_re = nullptr;
    double* d_pair_local_im = nullptr;
    float* d_pair_local_re_fp32 = nullptr;
    float* d_pair_local_im_fp32 = nullptr;
    double* d_strict_pair_multi_re = nullptr;
    double* d_strict_pair_multi_im = nullptr;
    double* d_strict_pair_local_re = nullptr;
    double* d_strict_pair_local_im = nullptr;
    float* d_pair_charges_fp32 = nullptr; // 12 split-complex source arrays
    int pair_workspace_fields = 0;
    bool force_pair_fp64 = false;

    // Cached GPU arrays for run_tree() — allocated once in init(), reused every call
    double* d_node_centers_cached;   // (n_nodes*3)
    double* d_dirs_cached;           // (L*3)
    double* d_weights_cached;        // (L)
    int*    d_leaf_idx_cached;       // (n_leaves)
    int*    d_src_id_offsets_cached;  // (n_leaves+1)
    int*    d_src_ids_cached;        // (h_src_ids_flat.size())
    int*    d_tgt_id_offsets_cached;  // (n_leaves+1)
    int*    d_tgt_ids_cached;        // (h_tgt_ids_flat.size())
    int*    d_leaf_near_offsets_cached; // (n_leaves+1), neighbor leaf ordinals
    int*    d_leaf_near_ids_cached;     // flat neighbor leaf ordinals
    int*    d_leaf_near_source_offsets_cached; // (n_leaves+1), expanded source IDs
    int*    d_leaf_near_source_ids_cached;     // self + near sources per target leaf

    // Cached gradient workspace arrays
    double* d_gy_re_cached;          // (Nt) for gradient y component
    double* d_gy_im_cached;
    double* d_gz_re_cached;          // (Nt) for gradient z component
    double* d_gz_im_cached;
    double* d_gx_re_tmp_cached;      // (Nt) temp for interleaving gradient
    double* d_gx_im_tmp_cached;
    double* d_gy2_re_cached;         // (Nt) batch-2 gradient y component
    double* d_gy2_im_cached;
    double* d_gz2_re_cached;         // (Nt) batch-2 gradient z component
    double* d_gz2_im_cached;
    double* d_gx2_re_tmp_cached;     // (Nt) batch-2 temp for interleaving gradient
    double* d_gx2_im_tmp_cached;
    double* d_gy3_re_cached;
    double* d_gy3_im_cached;
    double* d_gz3_re_cached;
    double* d_gz3_im_cached;
    double* d_gx3_re_tmp_cached;
    double* d_gx3_im_tmp_cached;
    double* d_gy4_re_cached;
    double* d_gy4_im_cached;
    double* d_gz4_re_cached;
    double* d_gz4_im_cached;
    double* d_gx4_re_tmp_cached;
    double* d_gx4_im_tmp_cached;
    double2* d_complex_tmp1;         // max(Ns, 3*Nt) complex staging buffer
    double2* d_complex_tmp2;

    bool initialized;
    bool batch4_allocated;
    bool pair_l2p_allocated;
    bool near_field_fp32;

    HelmholtzFMM() : d_tgt_pts(0), d_src_pts(0),
        d_tgt_pts_fp32(0), d_src_pts_fp32(0),
        d_dirs_fp32_cached(0), d_weights_fp32_cached(0),
        d_phase_cache_fp32(0), d_l2p_phase_cache_fp32(0),
        d_p2p_offsets(0), d_p2p_indices(0),
        d_multi_re(0), d_multi_im(0), d_local_re(0), d_local_im(0),
        d_transfer_re(0), d_transfer_im(0),
        d_transfer_re_fp32(0), d_transfer_im_fp32(0),
        d_m2l_tgt(0), d_m2l_src(0), d_m2l_tidx(0),
        d_m2l_row_target(0), d_m2l_row_start(0), d_m2l_row_end(0),
        d_m2m_shift_re(0), d_m2m_shift_im(0), d_m2m_parent(0), d_m2m_child(0),
        d_l2l_shift_re(0), d_l2l_shift_im(0), d_l2l_parent(0), d_l2l_child(0),
        d_charges_re(0), d_charges_im(0), d_result_re(0), d_result_im(0),
        d_grad_re(0), d_grad_im(0), d_hess_re(0), d_hess_im(0),
        d_charges2_re(0), d_charges2_im(0), d_result2_re(0), d_result2_im(0),
        d_grad2_re(0), d_grad2_im(0),
        d_multi2_re(0), d_multi2_im(0), d_local2_re(0), d_local2_im(0),
        d_charges3_re(0), d_charges3_im(0), d_charges4_re(0), d_charges4_im(0),
        d_result3_re(0), d_result3_im(0), d_result4_re(0), d_result4_im(0),
        d_grad3_re(0), d_grad3_im(0), d_grad4_re(0), d_grad4_im(0),
        d_multi3_re(0), d_multi3_im(0), d_multi4_re(0), d_multi4_im(0),
        d_local3_re(0), d_local3_im(0), d_local4_re(0), d_local4_im(0),
        d_pair_multi_re(0), d_pair_multi_im(0),
        d_pair_multi_re_fp32(0), d_pair_multi_im_fp32(0),
        d_pair_local_re(0), d_pair_local_im(0),
        d_pair_local_re_fp32(0), d_pair_local_im_fp32(0),
        d_strict_pair_multi_re(0), d_strict_pair_multi_im(0),
        d_strict_pair_local_re(0), d_strict_pair_local_im(0),
        d_pair_charges_fp32(0), pair_workspace_fields(0),
        force_pair_fp64(false),
        d_node_centers_cached(0), d_dirs_cached(0), d_weights_cached(0),
        d_leaf_idx_cached(0), d_src_id_offsets_cached(0), d_src_ids_cached(0),
        d_tgt_id_offsets_cached(0), d_tgt_ids_cached(0),
        d_leaf_near_offsets_cached(0), d_leaf_near_ids_cached(0),
        d_leaf_near_source_offsets_cached(0),
        d_leaf_near_source_ids_cached(0),
        d_gy_re_cached(0), d_gy_im_cached(0), d_gz_re_cached(0), d_gz_im_cached(0),
        d_gx_re_tmp_cached(0), d_gx_im_tmp_cached(0),
        d_gy2_re_cached(0), d_gy2_im_cached(0), d_gz2_re_cached(0), d_gz2_im_cached(0),
        d_gx2_re_tmp_cached(0), d_gx2_im_tmp_cached(0),
        d_gy3_re_cached(0), d_gy3_im_cached(0), d_gz3_re_cached(0), d_gz3_im_cached(0),
        d_gx3_re_tmp_cached(0), d_gx3_im_tmp_cached(0),
        d_gy4_re_cached(0), d_gy4_im_cached(0), d_gz4_re_cached(0), d_gz4_im_cached(0),
        d_gx4_re_tmp_cached(0), d_gx4_im_tmp_cached(0),
        d_complex_tmp1(0), d_complex_tmp2(0), initialized(false),
        batch4_allocated(false), pair_l2p_allocated(false),
        near_field_fp32(false) {}

    // Initialize: build tree, precompute transfers, upload to GPU
    void init(const double* targets, int n_tgt,
              const double* sources, int n_src,
              cdouble k_val, int digits = 3, int max_leaf = 64,
              int near_radius = 1, bool request_batch4 = false,
              bool request_vector_pair = false);

    // Evaluate: y[i] = sum_j G(r_i, r_j) * q[j]
    // charges: host array (Ns), result: host array (Nt)
    void evaluate(const cdouble* charges, cdouble* result);

    // Evaluate gradient: grad[i] = sum_j nabla_G(r_i, r_j) * q[j]
    // charges: host array (Ns), grad_result: host array (Nt*3) [x0,y0,z0,x1,y1,z1,...]
    void evaluate_gradient(const cdouble* charges, cdouble* grad_result);

    // Evaluate the symmetric Hessian in xx,xy,xz,yy,yz,zz order.
    void evaluate_hessian(const cdouble* charges, cdouble* hessian_result);

    // Evaluate gradient and Hessian from one multipole/local traversal.
    void evaluate_grad_hessian(
        const cdouble* charges,
        cdouble* gradient_result,
        cdouble* hessian_result);
    // Evaluate three vector components in one shared FMM traversal.
    void evaluate_grad_hessian_batch3(
        const cdouble* charges1,
        const cdouble* charges2,
        const cdouble* charges3,
        cdouble* gradient1,
        cdouble* gradient2,
        cdouble* gradient3,
        cdouble* hessian1,
        cdouble* hessian2,
        cdouble* hessian3);

    // Contract the derivatives of a three-component field during L2P/P2P.
    // curl_result stores xy, xz, yz antisymmetric gradient components.
    // hessian_action stores H*q - trace(H)*q.
    void evaluate_vector_actions_batch3(
        const cdouble* charges_x,
        const cdouble* charges_y,
        const cdouble* charges_z,
        cdouble* curl_result,
        cdouble* hessian_action);

    // Keep the contracted vector action on the device. The source arrays are
    // split-complex device buffers. Results remain in d_grad_{re,im} (curl)
    // and d_hess_{re,im} (H*q-tr(H)q), so a caller can fuse medium
    // combination and Galerkin testing without a host round trip.
    void evaluate_vector_actions_batch3_device(
        const double* charges_x_re,
        const double* charges_x_im,
        const double* charges_y_re,
        const double* charges_y_im,
        const double* charges_z_re,
        const double* charges_z_im);

    // Evaluate two three-component source fields while sharing the far
    // traversal, L2P, and geometry-heavy near interaction. The first result
    // remains in d_grad/d_hess and the second in d_grad2/d_hess2.
    void evaluate_vector_actions_pair_batch3_device(
        const double* first_x_re,
        const double* first_x_im,
        const double* first_y_re,
        const double* first_y_im,
        const double* first_z_re,
        const double* first_z_im,
        const double* second_x_re,
        const double* second_x_im,
        const double* second_y_re,
        const double* second_y_im,
        const double* second_z_re,
        const double* second_z_im);

    bool strict_vector_pair_available() const;
    void evaluate_vector_actions_pair_batch3_device_strict(
        const double* first_x_re,
        const double* first_x_im,
        const double* first_y_re,
        const double* first_y_im,
        const double* first_z_re,
        const double* first_z_im,
        const double* second_x_re,
        const double* second_x_im,
        const double* second_y_re,
        const double* second_y_im,
        const double* second_z_re,
        const double* second_z_im);

    // Evaluate the J/M fields for two independent right-hand sides in one
    // 12-channel far traversal. Results use grad/hess slots 1 through 4.
    void evaluate_vector_actions_quad_batch3_device(
        const double* const charges_re[12],
        const double* const charges_im[12]);

    bool vector_actions_pair_available() const {
        return batch4_allocated;
    }

    bool vector_actions_quad_available() const {
        return batch4_allocated && pair_workspace_fields >= 12 &&
            d_hess4_re != nullptr && d_hess4_im != nullptr;
    }

    // Evaluate both potential and gradient in a single tree traversal.
    // charges: host (Ns), pot_result: host (Nt), grad_result: host (Nt*3)
    void evaluate_pot_grad(const cdouble* charges, cdouble* pot_result, cdouble* grad_result);

    // Batched evaluate: two charge vectors, two result vectors, single tree traversal
    void evaluate_batch2(const cdouble* charges1, const cdouble* charges2,
                         cdouble* result1, cdouble* result2);
    void evaluate_batch2_uploaded();

    // Batched pot+grad: two charge vectors, two pot + two grad results, single tree traversal
    void evaluate_pot_grad_batch2(const cdouble* charges1, const cdouble* charges2,
                                   cdouble* pot1, cdouble* grad1,
                                   cdouble* pot2, cdouble* grad2);
    void evaluate_pot_grad_uploaded();
    void evaluate_pot_grad_batch2_uploaded();
    void evaluate_batch4_uploaded();
    void evaluate_batch3_far_uploaded();
    void evaluate_batch4_far_uploaded(bool evaluate_potential = true);
    void evaluate_gradient_batch4_l2p_uploaded();
    void evaluate_pot_grad_batch4_uploaded();

    // Run FMM tree traversal (P2M→M2M→M2L→L2L→L2P/P2P)
    // derivative_order: 0 potential, 1 gradient, 2 Hessian,
    // 3 gradient and Hessian.
    void run_tree(const double* h_q_re, const double* h_q_im,
                  int derivative_order);
    void run_tree_uploaded(int derivative_order);

    // Free GPU memory
    void cleanup();

    ~HelmholtzFMM() { if (initialized) cleanup(); }
};

// Spherical Hankel function of the first kind: h_n^(1)(z) = j_n(z) + i*y_n(z)
cdouble spherical_hankel1(int n, cdouble z);

#endif // BEM_FMM_H
