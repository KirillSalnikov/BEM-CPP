#ifndef BEM_OUTPUT_H
#define BEM_OUTPUT_H

#include <cstdio>

// Write Mueller matrix and metadata to JSON file.
// M: [16 * ntheta] array, layout M[(i*4+j)*ntheta + t]
// theta: [ntheta] array in radians
void write_json(const char* filename,
                const double* M, const double* theta, int ntheta,
                double ka, double n_re, double n_im, int refinements,
                int n_alpha, int n_beta, int n_gamma,
                double time_assembly, double time_solve, double time_farfield,
                double time_total);

#endif
