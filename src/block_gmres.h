#ifndef BLOCK_GMRES_H
#define BLOCK_GMRES_H

#include <complex>
#include <vector>
typedef std::complex<double> cdouble;

class BemFmmOperator;
class NearFieldPrecond;

struct GmresPairedWorkspace {
    std::vector<cdouble> r1, r2, w1, w2, z1, z2;
    std::vector<cdouble> V1, V2, Z1, Z2;
    std::vector<cdouble> H1, H2, cs1, sn1, s1, cs2, sn2, s2;
    std::vector<cdouble> ytmp, ytmp2, ztmp, ztmp2;
    double final_relres1 = 0.0;
    double final_relres2 = 0.0;
    bool converged1 = false;
    bool converged2 = false;
    bool stopped_stagnant = false;
    bool numerical_breakdown = false;
    bool restored_best_iterate = false;
    bool reached_max_cycles = false;
};

// Solve Z*x1=b1 and Z*x2=b2 simultaneously using paired GMRES
// Both systems share the same operator Z, using batched matvec
// Returns total number of matvec evaluations
int gmres_solve_paired(BemFmmOperator& op,
    const cdouble* b1, const cdouble* b2,
    cdouble* x1, cdouble* x2,
    int restart = 100, double tol = 1e-4, int maxiter = 300,
    bool verbose = true, NearFieldPrecond* precond = nullptr);

int gmres_solve_paired_ws(BemFmmOperator& op,
    const cdouble* b1, const cdouble* b2,
    cdouble* x1, cdouble* x2,
    int restart, double tol, int maxiter,
    bool verbose, NearFieldPrecond* precond,
    GmresPairedWorkspace& ws);

#endif
