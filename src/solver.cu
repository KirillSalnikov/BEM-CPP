#include "solver.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int lu_factorize_cuda(std::complex<double>* Z, int n, int* ipiv) {
    Timer timer;

    for (int k = 0; k < n; k++) {
        int piv = k;
        double best = std::abs(Z[k * n + k]);
        for (int i = k + 1; i < n; i++) {
            double v = std::abs(Z[i * n + k]);
            if (v > best) {
                best = v;
                piv = i;
            }
        }

        ipiv[k] = piv;
        if (best < 1e-300) {
            fprintf(stderr, "  LU factorization failed: singular pivot at %d\n", k);
            return k + 1;
        }

        if (piv != k) {
            for (int j = 0; j < n; j++)
                std::swap(Z[k * n + j], Z[piv * n + j]);
        }

        std::complex<double> pivot = Z[k * n + k];
        for (int i = k + 1; i < n; i++) {
            Z[i * n + k] /= pivot;
            std::complex<double> lik = Z[i * n + k];
            for (int j = k + 1; j < n; j++)
                Z[i * n + j] -= lik * Z[k * n + j];
        }
    }

    printf("  LU factorization CPU fallback (%dx%d): %.1fs\n", n, n, timer.elapsed_s());
    return 0;
}

int lu_solve_cuda(const std::complex<double>* Z, const int* ipiv,
                  int n, std::complex<double>* B, int nrhs) {
    Timer timer;

    for (int rhs = 0; rhs < nrhs; rhs++) {
        std::complex<double>* b = B + (size_t)rhs * n;

        for (int k = 0; k < n; k++) {
            if (ipiv[k] != k)
                std::swap(b[k], b[ipiv[k]]);
        }

        for (int i = 1; i < n; i++) {
            std::complex<double> sum = b[i];
            for (int j = 0; j < i; j++)
                sum -= Z[i * n + j] * b[j];
            b[i] = sum;
        }

        for (int i = n - 1; i >= 0; i--) {
            std::complex<double> sum = b[i];
            for (int j = i + 1; j < n; j++)
                sum -= Z[i * n + j] * b[j];
            b[i] = sum / Z[i * n + i];
        }
    }

    printf("  LU solve CPU fallback (%dx%d, %d RHS): %.2fs\n", n, n, nrhs, timer.elapsed_s());
    return 0;
}

int lu_solve_full(std::complex<double>* Z, int n,
                  std::complex<double>* B, int nrhs) {
    Timer timer;
    std::vector<int> ipiv(n);
    int info = lu_factorize_cuda(Z, n, ipiv.data());
    if (info != 0)
        return info;
    info = lu_solve_cuda(Z, ipiv.data(), n, B, nrhs);
    printf("  Total factorize+solve CPU fallback: %.1fs\n", timer.elapsed_s());
    return info;
}
