#ifndef BLOCK_GMRES_H
#define BLOCK_GMRES_H

#include <complex>
typedef std::complex<double> cdouble;

class BemFmmOperator;
class NearFieldPrecond;

// Solve Z*x1=b1 and Z*x2=b2 simultaneously using paired GMRES
// Both systems share the same operator Z, using batched matvec
// Returns total number of matvec evaluations
int gmres_solve_paired(BemFmmOperator& op,
    const cdouble* b1, const cdouble* b2,
    cdouble* x1, cdouble* x2,
    int restart = 100, double tol = 1e-4, int maxiter = 300,
    bool verbose = true, NearFieldPrecond* precond = nullptr);

#endif
