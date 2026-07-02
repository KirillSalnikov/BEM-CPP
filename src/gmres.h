#ifndef BEM_GMRES_H
#define BEM_GMRES_H

#include "bem_fmm.h"
#include "precond.h"

// GMRES(m) solver for PMCHWT system with FMM-accelerated matvec.
//
// op: BEM-FMM operator (provides matvec)
// b: RHS vector (2*N)
// x: solution vector (2*N), initialized to zero
// restart: GMRES restart parameter (default 100)
// tol: relative tolerance (default 1e-4)
// maxiter: max number of restart cycles (default 300)
// verbose: print convergence info
// precond: optional right block-Jacobi preconditioner
//
// Returns: 0 if converged, 1 if not
int gmres_solve(BemFmmOperator& op, const cdouble* b, cdouble* x,
                int restart = 100, double tol = 1e-4, int maxiter = 300,
                bool verbose = true, NearFieldPrecond* precond = nullptr);

#endif // BEM_GMRES_H
