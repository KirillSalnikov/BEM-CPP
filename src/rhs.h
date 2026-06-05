#ifndef BEM_RHS_H
#define BEM_RHS_H

#include "rwg.h"
#include <complex>

// Compute PMCHWT RHS for plane wave incidence.
// b: output array of size 2*N.
// E0: incident E-field polarization (3-vector).
// k_hat: incident wave direction (unit 3-vector).
void compute_rhs_planewave(const RWG& rwg, const Mesh& mesh,
                           std::complex<double> k_ext, double eta_ext,
                           const Vec3& E0, const Vec3& k_hat,
                           int quad_order,
                           std::complex<double>* b);

#endif
