#ifndef BEM_RHS_H
#define BEM_RHS_H

#include "rwg.h"
#include <complex>

struct FFCache;

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

#endif
