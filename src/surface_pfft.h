#ifndef BEM_SURFACE_PFFT_H
#define BEM_SURFACE_PFFT_H

#include "types.h"
#include <cufft.h>
#include <complex>
#include <vector>

// Surface pFFT: 2D FFT on each flat face + inter-face P2P.
// Drop-in replacement for HelmholtzPFFT on hex prisms.
//
// Intra-face: Toeplitz structure in 2D -> circulant embedding -> 2D FFT
// Inter-face: direct GPU P2P (FP32 for throughput)

struct FaceGrid {
    int face_id;
    int n_pts;              // number of quad points on this face

    // Local coordinate system
    double normal[3];       // outward normal
    double u_axis[3];       // local u direction
    double v_axis[3];       // local v direction
    double origin_3d[3];    // a point on the face (for plane equation)

    // 2D grid parameters
    int Mu, Mv;             // physical grid dimensions
    int M2u, M2v;           // doubled (FFT) grid
    double h;               // grid spacing (same as 3D for wavelength compat)
    double grid_origin[2];  // grid origin in local (u,v) coords
    long long grid_total;   // M2u * M2v

    // Mapping: local point index -> global point index in all_pts
    std::vector<int> local_to_global;

    // GPU: 2D Green's FFT (potential + gradient in local coords) — FP32 for speed
    cufftComplex* d_G_hat;       // potential
    cufftComplex* d_dGdu_hat;    // gradient in u-direction (local)
    cufftComplex* d_dGdv_hat;    // gradient in v-direction (local)

    // cuFFT 2D plans
    cufftHandle plan_fwd, plan_inv;

    // 2D interpolation stencils (per point, (p+1)^2 entries)
    int*    d_stencil_idx;  // (n_pts * stencil_2d)
    double* d_stencil_wt;   // (n_pts * stencil_2d)

    // Intra-face near-field correction (sparse CSR)
    int*    d_corr_row_ptr;
    int*    d_corr_col_idx;
    double* d_corr_G_re,  *d_corr_G_im;
    double* d_corr_dGdu_re, *d_corr_dGdu_im;
    double* d_corr_dGdv_re, *d_corr_dGdv_im;
    int corr_nnz;

    // GPU: local charges/results for 2D FFT
    double* d_charges_re;
    double* d_charges_im;
    double* d_result_re;
    double* d_result_im;

    // Work buffer for FFT (FP32)
    cufftComplex* d_work;
    // Forward-FFT'd charges (reusable across G/dGdu/dGdv convolutions, FP32)
    cufftComplex* d_charges_hat;

    // FP32 staging grid for anterpolation (fast FP32 atomicAdd)
    float* d_stage_re;
    float* d_stage_im;

    // GPU copy of local_to_global mapping for gather/scatter kernels
    int* d_local_to_global;

    // CUDA stream for async execution
    cudaStream_t stream;

    FaceGrid() : d_G_hat(0), d_dGdu_hat(0), d_dGdv_hat(0),
                 d_stencil_idx(0), d_stencil_wt(0),
                 d_corr_row_ptr(0), d_corr_col_idx(0),
                 d_corr_G_re(0), d_corr_G_im(0),
                 d_corr_dGdu_re(0), d_corr_dGdu_im(0),
                 d_corr_dGdv_re(0), d_corr_dGdv_im(0),
                 d_charges_re(0), d_charges_im(0),
                 d_result_re(0), d_result_im(0),
                 d_work(0), d_charges_hat(0),
                 d_stage_re(0), d_stage_im(0),
                 d_local_to_global(0), stream(0) {}
};

struct HelmholtzSurfacePFFT {
    cdouble k;
    int Nt, Ns;         // total points (same as all_pts size)
    int interp_p;       // interpolation order (default 3)
    int stencil_2d;     // (interp_p+1)^2
    int n_faces;        // 8 for hex prism

    FaceGrid faces[8];

    // Face assignment per global point
    std::vector<int> point_face;    // (Nt) face_id for each point

    // Inter-face P2P: points sorted by face on GPU (FP32 for speed)
    float*  d_pts_f;          // (Nt * 3) all points in FP32
    int*    d_face_offsets;   // (n_faces + 1) CSR-style face boundaries
    std::vector<int> face_offsets_host;  // host copy

    // Charge/result buffers for P2P (FP32)
    float*  d_p2p_q_re;      // (Nt) charges for P2P
    float*  d_p2p_q_im;
    float*  d_p2p_out_re;    // (Nt) P2P result
    float*  d_p2p_out_im;
    float*  d_p2p_grad_re;   // (Nt*3) P2P gradient result
    float*  d_p2p_grad_im;

    // Global charge/result buffers (FP64, for assembling final result)
    double* d_charges_re;
    double* d_charges_im;
    double* d_result_re;
    double* d_result_im;
    double* d_grad_re;       // (Nt*3)
    double* d_grad_im;

    // Source/target positions on GPU (FP64, for stencil computation)
    double* d_pts;            // (Nt * 3)

    // Precomputed sort order on GPU (for P2P charge sorting/unsorting)
    int* d_sort_order;        // (Nt) sorted_idx -> original_idx

    // Pre-allocated host staging buffers (avoid heap alloc per evaluate call)
    std::vector<double> h_stage_re, h_stage_im;   // (Nt) charge split
    std::vector<double> h_out_re, h_out_im;       // (Nt) result download
    std::vector<double> h_grad_re, h_grad_im;     // (Nt*3) gradient download

    // Batched inter-face P2P buffers (FP32, sized for batch8)
    float*  d_bp_q_re;        // [8*Nt] sorted charges (up to 8 vectors packed)
    float*  d_bp_q_im;
    float*  d_bp_pot_re;      // [8*Nt] sorted potential results
    float*  d_bp_pot_im;
    float*  d_bp_grad_re;     // [6*Nt*3] sorted gradient results
    float*  d_bp_grad_im;

    // Batched global result buffers (FP64)
    double* d_bp_res_re[8];   // [Nt] each — potential per batch
    double* d_bp_res_im[8];
    double* d_bp_grd_re[6];   // [Nt*3] each — gradient per batch
    double* d_bp_grd_im[6];

    bool initialized;

    HelmholtzSurfacePFFT() : initialized(false),
        d_pts_f(0), d_face_offsets(0),
        d_p2p_q_re(0), d_p2p_q_im(0),
        d_p2p_out_re(0), d_p2p_out_im(0),
        d_p2p_grad_re(0), d_p2p_grad_im(0),
        d_charges_re(0), d_charges_im(0),
        d_result_re(0), d_result_im(0),
        d_grad_re(0), d_grad_im(0),
        d_pts(0), d_sort_order(0),
        d_bp_q_re(0), d_bp_q_im(0),
        d_bp_pot_re(0), d_bp_pot_im(0),
        d_bp_grad_re(0), d_bp_grad_im(0)
    {
        for (int i = 0; i < 8; i++) { d_bp_res_re[i]=0; d_bp_res_im[i]=0; }
        for (int i = 0; i < 6; i++) { d_bp_grd_re[i]=0; d_bp_grd_im[i]=0; }
    }

    // Initialize with face classification
    // face_ids[i] = face index (0..n_faces-1) for point i
    // face_normals[f*3..f*3+2] = outward normal of face f
    void init(const double* points, int n_pts,
              const int* face_ids, int n_faces,
              const double* face_normals,
              cdouble k_val, int digits = 3);

    // Same evaluate interface as HelmholtzPFFT
    void evaluate(const cdouble* charges, cdouble* result);
    void evaluate_pot_grad(const cdouble* charges, cdouble* pot_result, cdouble* grad_result);

    // Batched evaluate: 3× evaluate_pot_grad + 1× evaluate in single P2P pass.
    // charges[0..2] get both potential and gradient, charges[3] gets potential only.
    // Computes Green's function geometry once per source-target pair instead of 4×.
    void evaluate_batch4(
        const cdouble* charges0, const cdouble* charges1,
        const cdouble* charges2, const cdouble* charges3,
        cdouble* pot0, cdouble* pot1, cdouble* pot2, cdouble* pot3,
        cdouble* grad0, cdouble* grad1, cdouble* grad2);

    // Double-batched: 2 × (3 pot+grad + 1 pot) = 8 charge vectors in ONE P2P pass.
    // charges 0-2,4-6 get pot+grad; charges 3,7 get pot only.
    // For LK_combined_batch2: 2 coefficient vectors × 4 charges each.
    void evaluate_batch8(
        const cdouble* charges[8],
        cdouble* pots[8],
        cdouble* grads[6]);  // grads[0-2] for charges 0-2, grads[3-5] for charges 4-6

    void cleanup();
    ~HelmholtzSurfacePFFT() { if (initialized) cleanup(); }
};

#endif // BEM_SURFACE_PFFT_H
