#include "output.h"
#include <cstdio>
#include <cmath>

void write_json(const char* filename,
                const double* M, const double* theta, int ntheta,
                double ka, double n_re, double n_im, int refinements,
                int n_alpha, int n_beta, int n_gamma,
                double time_assembly, double time_solve, double time_farfield,
                double time_total)
{
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Cannot open %s for writing\n", filename);
        return;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"ka\": %.6f,\n", ka);
    fprintf(f, "  \"ri\": [%.6f, %.6f],\n", n_re, n_im);
    fprintf(f, "  \"refinements\": %d,\n", refinements);
    fprintf(f, "  \"orient\": [%d, %d, %d],\n", n_alpha, n_beta, n_gamma);
    fprintf(f, "  \"ntheta\": %d,\n", ntheta);

    // Timing
    fprintf(f, "  \"timing\": {\n");
    fprintf(f, "    \"assembly_s\": %.2f,\n", time_assembly);
    fprintf(f, "    \"solve_s\": %.2f,\n", time_solve);
    fprintf(f, "    \"farfield_s\": %.2f,\n", time_farfield);
    fprintf(f, "    \"total_s\": %.2f\n", time_total);
    fprintf(f, "  },\n");

    // Theta array (degrees)
    fprintf(f, "  \"theta\": [");
    for (int t = 0; t < ntheta; t++) {
        fprintf(f, "%.4f", theta[t] * 180.0 / M_PI);
        if (t < ntheta - 1) fprintf(f, ", ");
    }
    fprintf(f, "],\n");

    // Mueller matrix: M[i][j][theta]
    fprintf(f, "  \"mueller\": [\n");
    for (int i = 0; i < 4; i++) {
        fprintf(f, "    [\n");
        for (int j = 0; j < 4; j++) {
            fprintf(f, "      [");
            for (int t = 0; t < ntheta; t++) {
                fprintf(f, "%.8e", M[(i*4+j)*ntheta + t]);
                if (t < ntheta - 1) fprintf(f, ", ");
            }
            fprintf(f, "]");
            if (j < 3) fprintf(f, ",");
            fprintf(f, "\n");
        }
        fprintf(f, "    ]");
        if (i < 3) fprintf(f, ",");
        fprintf(f, "\n");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    fclose(f);
    printf("Results written to %s\n", filename);
}
