#ifndef P2P_H
#define P2P_H

// P2P near-field kernel launch functions (definitions in p2p.cu)

// Scalar potential only: phi_i = sum_j G(r_i, r_j) * q_j
void launch_p2p_potential(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_out_re, double* d_out_im, int Nt);

void launch_p2p_potential_batch2(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_out1_re, double* d_out1_im,
    double* d_out2_re, double* d_out2_im,
    int Nt);

void launch_p2p_potential_batch4(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const double* d_q3_re, const double* d_q3_im,
    const double* d_q4_re, const double* d_q4_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_out1_re, double* d_out1_im,
    double* d_out2_re, double* d_out2_im,
    double* d_out3_re, double* d_out3_im,
    double* d_out4_re, double* d_out4_im,
    int Nt);

// Gradient only: grad_phi_i = sum_j nabla_G(r_i, r_j) * q_j
void launch_p2p_gradient(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_gx_re, double* d_gx_im,
    double* d_gy_re, double* d_gy_im,
    double* d_gz_re, double* d_gz_im, int Nt);

// Combined potential + gradient in a single pass (avoids redundant work)
void launch_p2p_pot_grad(
    int Nt,
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_pot_re, double* d_pot_im,
    double* d_gx_re, double* d_gx_im,
    double* d_gy_re, double* d_gy_im,
    double* d_gz_re, double* d_gz_im);

void launch_p2p_pot_grad_batch2(
    int Nt,
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_pot1_re, double* d_pot1_im,
    double* d_gx1_re, double* d_gx1_im,
    double* d_gy1_re, double* d_gy1_im,
    double* d_gz1_re, double* d_gz1_im,
    double* d_pot2_re, double* d_pot2_im,
    double* d_gx2_re, double* d_gx2_im,
    double* d_gy2_re, double* d_gy2_im,
    double* d_gz2_re, double* d_gz2_im);

void launch_p2p_pot_grad_batch4(
    int Nt,
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const double* d_q3_re, const double* d_q3_im,
    const double* d_q4_re, const double* d_q4_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_pot1_re, double* d_pot1_im,
    double* d_gx1_re, double* d_gx1_im,
    double* d_gy1_re, double* d_gy1_im,
    double* d_gz1_re, double* d_gz1_im,
    double* d_pot2_re, double* d_pot2_im,
    double* d_gx2_re, double* d_gx2_im,
    double* d_gy2_re, double* d_gy2_im,
    double* d_gz2_re, double* d_gz2_im,
    double* d_pot3_re, double* d_pot3_im,
    double* d_gx3_re, double* d_gx3_im,
    double* d_gy3_re, double* d_gy3_im,
    double* d_gz3_re, double* d_gz3_im,
    double* d_pot4_re, double* d_pot4_im,
    double* d_gx4_re, double* d_gx4_im,
    double* d_gy4_re, double* d_gy4_im,
    double* d_gz4_re, double* d_gz4_im);

// Compressed near-field layout: one block per target leaf, sources are read
// from the leaf itself and its neighbor leaves. Avoids per-target CSR storage.
void launch_p2p_potential_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_out_re, double* d_out_im);

void launch_p2p_potential_batch2_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_out1_re, double* d_out1_im,
    double* d_out2_re, double* d_out2_im);

void launch_p2p_potential_batch4_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const double* d_q3_re, const double* d_q3_im,
    const double* d_q4_re, const double* d_q4_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_out1_re, double* d_out1_im,
    double* d_out2_re, double* d_out2_im,
    double* d_out3_re, double* d_out3_im,
    double* d_out4_re, double* d_out4_im);

void launch_p2p_gradient_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_gx_re, double* d_gx_im,
    double* d_gy_re, double* d_gy_im,
    double* d_gz_re, double* d_gz_im);

// Symmetric Hessian in interleaved xx,xy,xz,yy,yz,zz order.
void launch_p2p_hessian_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_hess_re, double* d_hess_im);

void launch_p2p_grad_hessian_batch3_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const double* d_q3_re, const double* d_q3_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_gx1_re, double* d_gx1_im,
    double* d_gy1_re, double* d_gy1_im,
    double* d_gz1_re, double* d_gz1_im,
    double* d_gx2_re, double* d_gx2_im,
    double* d_gy2_re, double* d_gy2_im,
    double* d_gz2_re, double* d_gz2_im,
    double* d_gx3_re, double* d_gx3_im,
    double* d_gy3_re, double* d_gy3_im,
    double* d_gz3_re, double* d_gz3_im,
    double* d_hess1_re, double* d_hess1_im,
    double* d_hess2_re, double* d_hess2_im,
    double* d_hess3_re, double* d_hess3_im,
    bool fp32_compute = false);

// Contract the derivatives of a three-component source field directly.
// curl uses xy, xz, yz antisymmetric gradient components. hessian_action is
// H*q - trace(H)*q. Both outputs are interleaved xyz arrays.
void launch_p2p_vector_actions_batch3_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_qx_re, const double* d_qx_im,
    const double* d_qy_re, const double* d_qy_im,
    const double* d_qz_re, const double* d_qz_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    const int* d_near_source_offsets,
    const int* d_near_source_ids,
    int n_leaves, double k_re, double k_im,
    double* d_curl_re, double* d_curl_im,
    double* d_hessian_action_re, double* d_hessian_action_im,
    bool fp32_compute = false);

// Contract two independent three-component source fields while sharing all
// target/source geometry and Green-function evaluations.
void launch_p2p_vector_actions_pair_batch3_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_first_x_re, const double* d_first_x_im,
    const double* d_first_y_re, const double* d_first_y_im,
    const double* d_first_z_re, const double* d_first_z_im,
    const double* d_second_x_re, const double* d_second_x_im,
    const double* d_second_y_re, const double* d_second_y_im,
    const double* d_second_z_re, const double* d_second_z_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    const int* d_near_source_offsets,
    const int* d_near_source_ids,
    int n_leaves, double k_re, double k_im,
    double* d_first_curl_re, double* d_first_curl_im,
    double* d_first_hessian_action_re,
    double* d_first_hessian_action_im,
    double* d_second_curl_re, double* d_second_curl_im,
    double* d_second_hessian_action_re,
    double* d_second_hessian_action_im,
    const float* d_tgt_fp32, const float* d_src_fp32,
    const float* d_packed_charges_fp32, int source_count,
    bool fp32_compute = false);

void launch_p2p_pot_grad_batch2_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_pot1_re, double* d_pot1_im,
    double* d_gx1_re, double* d_gx1_im,
    double* d_gy1_re, double* d_gy1_im,
    double* d_gz1_re, double* d_gz1_im,
    double* d_pot2_re, double* d_pot2_im,
    double* d_gx2_re, double* d_gx2_im,
    double* d_gy2_re, double* d_gy2_im,
    double* d_gz2_re, double* d_gz2_im);

void launch_p2p_pot_grad_batch4_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const double* d_q3_re, const double* d_q3_im,
    const double* d_q4_re, const double* d_q4_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_pot1_re, double* d_pot1_im,
    double* d_gx1_re, double* d_gx1_im,
    double* d_gy1_re, double* d_gy1_im,
    double* d_gz1_re, double* d_gz1_im,
    double* d_pot2_re, double* d_pot2_im,
    double* d_gx2_re, double* d_gx2_im,
    double* d_gy2_re, double* d_gy2_im,
    double* d_gz2_re, double* d_gz2_im,
    double* d_pot3_re, double* d_pot3_im,
    double* d_gx3_re, double* d_gx3_im,
    double* d_gy3_re, double* d_gy3_im,
    double* d_gz3_re, double* d_gz3_im,
    double* d_pot4_re, double* d_pot4_im,
    double* d_gx4_re, double* d_gx4_im,
    double* d_gy4_re, double* d_gy4_im,
    double* d_gz4_re, double* d_gz4_im);

void launch_p2p_pot_grad_leaf(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_tgt_offsets, const int* d_tgt_ids,
    const int* d_src_offsets, const int* d_src_ids,
    const int* d_near_offsets, const int* d_near_leaf_ids,
    int n_leaves, double k_re, double k_im,
    double* d_pot_re, double* d_pot_im,
    double* d_gx_re, double* d_gx_im,
    double* d_gy_re, double* d_gy_im,
    double* d_gz_re, double* d_gz_im);

#endif // P2P_H
