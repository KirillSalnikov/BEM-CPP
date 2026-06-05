#ifndef BEM_ASSEMBLY_H
#define BEM_ASSEMBLY_H

#include "rwg.h"
#include "quadrature.h"
#include <complex>

// Assemble L and K operator matrices using CUDA.
// L, K: output arrays of size N*N (row-major), allocated by caller.
// k: wavenumber (complex for absorbing media).
void assemble_L_K_cuda(const RWG& rwg, const Mesh& mesh,
                       std::complex<double> k, int quad_order,
                       std::complex<double>* L, std::complex<double>* K);

#endif // BEM_ASSEMBLY_H
