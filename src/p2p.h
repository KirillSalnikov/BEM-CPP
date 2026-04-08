#ifndef P2P_H
#define P2P_H

#include <cuda_runtime.h>

// Leaf-to-leaf P2P near-field kernel launch functions (definitions in p2p.cu)
// Uses leaf neighbor lists instead of per-point CSR for ~6 GB memory savings.

// Scalar potential only: phi_i = sum_j G(r_i, r_j) * q_j
void launch_p2p_potential(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_tgt_to_leaf, const int* d_leaf_near_offsets,
    const int* d_leaf_near_nbrs, const int* d_src_id_offsets,
    const int* d_src_ids,
    double k_re, double k_im,
    double* d_out_re, double* d_out_im, int Nt,
    cudaStream_t stream = 0);

// Gradient only: grad_phi_i = sum_j nabla_G(r_i, r_j) * q_j
void launch_p2p_gradient(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_tgt_to_leaf, const int* d_leaf_near_offsets,
    const int* d_leaf_near_nbrs, const int* d_src_id_offsets,
    const int* d_src_ids,
    double k_re, double k_im,
    double* d_gx_re, double* d_gx_im,
    double* d_gy_re, double* d_gy_im,
    double* d_gz_re, double* d_gz_im, int Nt,
    cudaStream_t stream = 0);

// Combined potential + gradient in a single pass
void launch_p2p_pot_grad(
    int Nt,
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_tgt_to_leaf, const int* d_leaf_near_offsets,
    const int* d_leaf_near_nbrs, const int* d_src_id_offsets,
    const int* d_src_ids,
    double k_re, double k_im,
    double* d_pot_re, double* d_pot_im,
    double* d_gx_re, double* d_gx_im,
    double* d_gy_re, double* d_gy_im,
    double* d_gz_re, double* d_gz_im,
    cudaStream_t stream = 0);

// Batch-2 potential: two charge vectors, single leaf traversal
void launch_p2p_potential_batch2(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const int* d_tgt_to_leaf, const int* d_leaf_near_offsets,
    const int* d_leaf_near_nbrs, const int* d_src_id_offsets,
    const int* d_src_ids,
    double k_re, double k_im,
    double* d_out1_re, double* d_out1_im,
    double* d_out2_re, double* d_out2_im, int Nt,
    cudaStream_t stream = 0);

// Batch-2 combined potential + gradient
void launch_p2p_pot_grad_batch2(
    int Nt,
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const int* d_tgt_to_leaf, const int* d_leaf_near_offsets,
    const int* d_leaf_near_nbrs, const int* d_src_id_offsets,
    const int* d_src_ids,
    double k_re, double k_im,
    double* d_pot1_re, double* d_pot1_im,
    double* d_pot2_re, double* d_pot2_im,
    double* d_gx1_re, double* d_gx1_im,
    double* d_gy1_re, double* d_gy1_im,
    double* d_gz1_re, double* d_gz1_im,
    double* d_gx2_re, double* d_gx2_im,
    double* d_gy2_re, double* d_gy2_im,
    double* d_gz2_re, double* d_gz2_im,
    cudaStream_t stream = 0);

#endif // P2P_H
