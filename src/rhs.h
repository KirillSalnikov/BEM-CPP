#ifndef BEM_RHS_H
#define BEM_RHS_H

#include "rwg.h"
#include <cuda_runtime.h>
#include <complex>
#include <cstddef>

struct FFCache;
struct FFCacheGPU;

struct RHSBatchWorkspace {
    void* h_orient;
    void* h_B;
    void* d_orient;
    void* d_B;
    cudaStream_t stream;
    bool h_orient_pinned;
    bool h_B_pinned;
    int cap_orient;
    size_t cap_host_rhs_elems;
    size_t cap_rhs_elems;

    RHSBatchWorkspace();
    void reserve(int n_orient, size_t rhs_elems, bool need_host_rhs = true);
    std::complex<double>* host_B();
    const std::complex<double>* host_B() const;
    void free();
    ~RHSBatchWorkspace();
};

// Compute PMCHWT RHS for plane wave incidence.
// b: output array of size 2*N.
// E0: incident E-field polarization (3-vector).
// k_hat: incident wave direction (unit 3-vector).
void compute_rhs_planewave(const RWG& rwg, const Mesh& mesh,
                           std::complex<double> k_ext, double eta_ext,
                           const Vec3& E0, const Vec3& k_hat,
                           int quad_order,
                           std::complex<double>* b);

void compute_rhs_planewave_pair(const RWG& rwg, const Mesh& mesh,
                                std::complex<double> k_ext, double eta_ext,
                                const Vec3& E0_a, const Vec3& E0_b,
                                const Vec3& k_hat, int quad_order,
                                std::complex<double>* b_a,
                                std::complex<double>* b_b);

void compute_rhs_planewave_pair_cached(const FFCache& cache,
                                       std::complex<double> k_ext, double eta_ext,
                                       const Vec3& E0_a, const Vec3& E0_b,
                                       const Vec3& k_hat,
                                       std::complex<double>* b_a,
                                       std::complex<double>* b_b);

int compute_rhs_planewave_pairs_cached_cuda(const FFCache& cache,
                                            std::complex<double> k_ext,
                                            double eta_ext,
                                            const Vec3* E0_a,
                                            const Vec3* E0_b,
                                            const Vec3* k_hat,
                                            int n_orient,
                                            std::complex<double>* B);

int compute_rhs_planewave_pairs_cached_cuda(const FFCacheGPU& gpu_cache,
                                            std::complex<double> k_ext,
                                            double eta_ext,
                                            const Vec3* E0_a,
                                            const Vec3* E0_b,
                                            const Vec3* k_hat,
                                            int n_orient,
                                            std::complex<double>* B);

int compute_rhs_planewave_pairs_cached_cuda_ws(const FFCacheGPU& gpu_cache,
                                               RHSBatchWorkspace& workspace,
                                               std::complex<double> k_ext,
                                               double eta_ext,
                                               const Vec3* E0_a,
                                               const Vec3* E0_b,
                                               const Vec3* k_hat,
                                               int n_orient,
                                               // If B is null, output is written to workspace.host_B().
                                               std::complex<double>* B);

int compute_rhs_planewave_pairs_cached_cuda_ws_scaled(const FFCacheGPU& gpu_cache,
                                                      RHSBatchWorkspace& workspace,
                                                      std::complex<double> k_ext,
                                                      double eta_ext,
                                                      std::complex<double> row_h_scale,
                                                      const Vec3* E0_a,
                                                      const Vec3* E0_b,
                                                      const Vec3* k_hat,
                                                      int n_orient,
                                                      // If B is null, output is written to workspace.host_B().
                                                      std::complex<double>* B);

#endif
