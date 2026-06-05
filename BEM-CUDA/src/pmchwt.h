#ifndef BEM_PMCHWT_H
#define BEM_PMCHWT_H

#include "rwg.h"
#include "assembly.h"
#include <complex>

// Assemble full PMCHWT system matrix Z (2N x 2N).
// Also returns L_ext, K_ext for preconditioner.
// Z layout: [[eta_ext*L_ext + eta_int*L_int, -(K_ext+K_int)],
//            [K_ext+K_int, L_ext/eta_ext + L_int/eta_int]]
void assemble_pmchwt(const RWG& rwg, const Mesh& mesh,
                     std::complex<double> k_ext, std::complex<double> k_int,
                     double eta_ext, double eta_int,
                     int quad_order,
                     std::complex<double>* Z,      // (2N x 2N) output
                     std::complex<double>* L_ext,   // (N x N) output, can be NULL
                     std::complex<double>* K_ext);  // (N x N) output, can be NULL

#endif
