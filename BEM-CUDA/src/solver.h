#ifndef BEM_SOLVER_H
#define BEM_SOLVER_H

#include "types.h"
#include <complex>

// LU factorization + solve.
// Z is (n x n) complex double, overwritten with LU factors.
// ipiv is output pivot array (size n), allocated by caller.
// Returns 0 on success.
int lu_factorize_cuda(std::complex<double>* Z, int n, int* ipiv);

// Solve with precomputed LU: Z * X = B.
// Z and ipiv from lu_factorize_cuda.
// B is (n x nrhs) column-major, overwritten with solution X.
int lu_solve_cuda(const std::complex<double>* Z, const int* ipiv,
                  int n, std::complex<double>* B, int nrhs);

// Combined: factorize Z, solve Z*X = B.
// Z is overwritten. B is overwritten with X.
int lu_solve_full(std::complex<double>* Z, int n,
                  std::complex<double>* B, int nrhs);

#endif
