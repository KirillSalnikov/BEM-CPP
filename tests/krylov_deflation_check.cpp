#include "krylov_deflation.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

int main()
{
    using bem_krylov::Complex;
    const int size = 4;
    const int stride = size;
    std::vector<Complex> hessenberg(
        static_cast<std::size_t>(size + 1) * stride, Complex(0.0));
    hessenberg[0 * stride + 0] = 0.1;
    hessenberg[1 * stride + 1] = 0.5;
    hessenberg[2 * stride + 2] = 2.0;
    hessenberg[3 * stride + 3] = 4.0;

    std::vector<Complex> coefficients;
    const int rank = bem_krylov::harmonic_ritz_coefficients(
        hessenberg, stride, size, 2, coefficients);
    if (rank != 2) {
        std::fprintf(stderr, "expected rank 2, got %d\n", rank);
        return 1;
    }
    double slow_overlap = 0.0;
    double fast_overlap = 0.0;
    double orthogonality = 0.0;
    for (int column = 0; column < rank; column++) {
        slow_overlap +=
            std::norm(coefficients[column * size + 0]) +
            std::norm(coefficients[column * size + 1]);
        fast_overlap +=
            std::norm(coefficients[column * size + 2]) +
            std::norm(coefficients[column * size + 3]);
    }
    for (int row = 0; row < size; row++) {
        orthogonality += std::real(
            std::conj(coefficients[row]) * coefficients[size + row]);
    }
    if (slow_overlap < 1.999999 || fast_overlap > 1.0e-6 ||
        std::abs(orthogonality) > 1.0e-10) {
        std::fprintf(
            stderr,
            "unexpected harmonic Ritz subspace: slow %.12g fast %.12g "
            "orthogonality %.12g\n",
            slow_overlap, fast_overlap, orthogonality);
        return 1;
    }
    std::printf("krylov deflation check passed\n");
    return 0;
}
