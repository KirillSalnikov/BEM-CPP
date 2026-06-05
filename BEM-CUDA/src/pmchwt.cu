#include "pmchwt.h"
#include <cstdio>
#include <cstring>
#include <vector>

void assemble_pmchwt(const RWG& rwg, const Mesh& mesh,
                     std::complex<double> k_ext, std::complex<double> k_int,
                     double eta_ext, double eta_int,
                     int quad_order,
                     std::complex<double>* Z,
                     std::complex<double>* L_ext_out,
                     std::complex<double>* K_ext_out)
{
    int N = rwg.N;
    int N2 = 2 * N;
    printf("  Assembling %dx%d PMCHWT matrix (%d RWG functions)...\n", N2, N2, N);

    std::vector<std::complex<double>> L_ext(N*N), K_ext(N*N);
    std::vector<std::complex<double>> L_int(N*N), K_int(N*N);

    // Assemble exterior operators
    Timer t0;
    printf("  Exterior operators (k=%.4f+%.4fi)...\n", k_ext.real(), k_ext.imag());
    assemble_L_K_cuda(rwg, mesh, k_ext, quad_order, L_ext.data(), K_ext.data());

    printf("  Interior operators (k=%.4f+%.4fi)...\n", k_int.real(), k_int.imag());
    assemble_L_K_cuda(rwg, mesh, k_int, quad_order, L_int.data(), K_int.data());
    printf("  Total PMCHWT assembly: %.1fs\n", t0.elapsed_s());

    // Form Z matrix
    std::complex<double> ce(eta_ext), ci(eta_int);
    std::complex<double> inv_ce(1.0 / eta_ext), inv_ci(1.0 / eta_int);

    memset(Z, 0, N2 * N2 * sizeof(std::complex<double>));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::complex<double> le = L_ext[i*N+j], li = L_int[i*N+j];
            std::complex<double> ke = K_ext[i*N+j], ki = K_int[i*N+j];
            std::complex<double> K_sum = ke + ki;

            // Top-left: eta_ext*L_ext + eta_int*L_int
            Z[i * N2 + j] = ce * le + ci * li;
            // Top-right: -(K_ext + K_int)
            Z[i * N2 + (N + j)] = -K_sum;
            // Bottom-left: K_ext + K_int
            Z[(N + i) * N2 + j] = K_sum;
            // Bottom-right: L_ext/eta_ext + L_int/eta_int
            Z[(N + i) * N2 + (N + j)] = inv_ce * le + inv_ci * li;
        }
    }

    // Copy L_ext, K_ext to output if requested
    if (L_ext_out) memcpy(L_ext_out, L_ext.data(), N*N*sizeof(std::complex<double>));
    if (K_ext_out) memcpy(K_ext_out, K_ext.data(), N*N*sizeof(std::complex<double>));
}
