#ifndef BEM_MULLER_PAIRED_GMRES_H
#define BEM_MULLER_PAIRED_GMRES_H

#include <complex>

struct MullerFmmOperator;
struct MullerMbjPreconditioner;

struct MullerPairedGmresResult {
    int iterations = 0;
    int operator_evaluations = 0;
    double initial_residual_x = 1.0;
    double initial_residual_y = 1.0;
    double final_residual_x = 1.0;
    double final_residual_y = 1.0;
    double seconds = 0.0;
    bool converged_x = false;
    bool converged_y = false;
};

MullerPairedGmresResult solve_muller_paired_gmres_device(
    MullerFmmOperator& op,
    const MullerMbjPreconditioner& preconditioner,
    const std::complex<double>* rhs_x,
    const std::complex<double>* rhs_y,
    std::complex<double>* solution_x,
    std::complex<double>* solution_y,
    int restart,
    double tolerance,
    int maximum_iterations,
    bool verbose = true);

#endif
