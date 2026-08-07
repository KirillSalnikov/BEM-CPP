#ifndef BEM_KRYLOV_DEFLATION_H
#define BEM_KRYLOV_DEFLATION_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace bem_krylov {

using Complex = std::complex<double>;

inline bool dense_lu_factor(
    std::vector<Complex>& matrix,
    int size,
    std::vector<int>& pivots)
{
    pivots.resize(size);
    for (int column = 0; column < size; column++) {
        int pivot = column;
        double pivot_magnitude = std::abs(
            matrix[static_cast<std::size_t>(column) * size + column]);
        for (int row = column + 1; row < size; row++) {
            const double magnitude = std::abs(
                matrix[static_cast<std::size_t>(row) * size + column]);
            if (magnitude > pivot_magnitude) {
                pivot = row;
                pivot_magnitude = magnitude;
            }
        }
        if (pivot_magnitude <= 1.0e-14)
            return false;
        pivots[column] = pivot;
        if (pivot != column) {
            for (int entry = 0; entry < size; entry++) {
                std::swap(
                    matrix[static_cast<std::size_t>(column) * size + entry],
                    matrix[static_cast<std::size_t>(pivot) * size + entry]);
            }
        }
        const Complex diagonal =
            matrix[static_cast<std::size_t>(column) * size + column];
        for (int row = column + 1; row < size; row++) {
            Complex& multiplier =
                matrix[static_cast<std::size_t>(row) * size + column];
            multiplier /= diagonal;
            for (int entry = column + 1; entry < size; entry++) {
                matrix[static_cast<std::size_t>(row) * size + entry] -=
                    multiplier *
                    matrix[static_cast<std::size_t>(column) * size + entry];
            }
        }
    }
    return true;
}

inline void dense_lu_solve(
    const std::vector<Complex>& factor,
    const std::vector<int>& pivots,
    int size,
    Complex* right_hand_side)
{
    for (int column = 0; column < size; column++) {
        if (pivots[column] != column) {
            std::swap(
                right_hand_side[column],
                right_hand_side[pivots[column]]);
        }
    }
    for (int row = 1; row < size; row++) {
        for (int column = 0; column < row; column++) {
            right_hand_side[row] -=
                factor[static_cast<std::size_t>(row) * size + column] *
                right_hand_side[column];
        }
    }
    for (int row = size - 1; row >= 0; row--) {
        for (int column = row + 1; column < size; column++) {
            right_hand_side[row] -=
                factor[static_cast<std::size_t>(row) * size + column] *
                right_hand_side[column];
        }
        right_hand_side[row] /=
            factor[static_cast<std::size_t>(row) * size + row];
    }
}

inline int orthonormalize_columns(
    std::vector<Complex>& columns,
    int rows,
    int count)
{
    int accepted = 0;
    for (int candidate = 0; candidate < count; candidate++) {
        Complex* vector =
            columns.data() + static_cast<std::size_t>(candidate) * rows;
        for (int pass = 0; pass < 2; pass++) {
            for (int previous = 0; previous < accepted; previous++) {
                const Complex* basis = columns.data() +
                    static_cast<std::size_t>(previous) * rows;
                Complex coefficient(0.0);
                for (int row = 0; row < rows; row++)
                    coefficient += std::conj(basis[row]) * vector[row];
                for (int row = 0; row < rows; row++)
                    vector[row] -= coefficient * basis[row];
            }
        }
        double squared_norm = 0.0;
        for (int row = 0; row < rows; row++)
            squared_norm += std::norm(vector[row]);
        if (squared_norm <= 1.0e-24)
            continue;
        const double inverse_norm = 1.0 / std::sqrt(squared_norm);
        for (int row = 0; row < rows; row++)
            vector[row] *= inverse_norm;
        if (accepted != candidate) {
            std::copy(
                vector, vector + rows,
                columns.begin() + static_cast<std::size_t>(accepted) * rows);
        }
        accepted++;
    }
    columns.resize(static_cast<std::size_t>(accepted) * rows);
    return accepted;
}

// Return a coefficient basis for the harmonic Ritz vectors associated with
// the smallest-magnitude eigenvalues of an Arnoldi relation.
inline int harmonic_ritz_coefficients(
    const std::vector<Complex>& hessenberg,
    int stride,
    int arnoldi_size,
    int requested_rank,
    std::vector<Complex>& coefficients)
{
    const int rank = std::min(requested_rank, arnoldi_size);
    coefficients.clear();
    if (rank <= 0 || stride < arnoldi_size)
        return 0;

    std::vector<Complex> projected(
        static_cast<std::size_t>(arnoldi_size) * arnoldi_size);
    for (int row = 0; row < arnoldi_size; row++) {
        for (int column = 0; column < arnoldi_size; column++) {
            projected[static_cast<std::size_t>(row) * arnoldi_size + column] =
                hessenberg[static_cast<std::size_t>(row) * stride + column];
        }
    }

    // Harmonic Ritz matrix:
    // T = H + |h_(m+1,m)|^2 H^(-H) e_m e_m^H.
    std::vector<Complex> adjoint(
        static_cast<std::size_t>(arnoldi_size) * arnoldi_size);
    for (int row = 0; row < arnoldi_size; row++) {
        for (int column = 0; column < arnoldi_size; column++) {
            adjoint[static_cast<std::size_t>(row) * arnoldi_size + column] =
                std::conj(
                    projected[
                        static_cast<std::size_t>(column) * arnoldi_size + row]);
        }
    }
    std::vector<int> adjoint_pivots;
    if (!dense_lu_factor(adjoint, arnoldi_size, adjoint_pivots))
        return 0;
    std::vector<Complex> correction(arnoldi_size, Complex(0.0));
    correction[arnoldi_size - 1] = Complex(1.0);
    dense_lu_solve(
        adjoint, adjoint_pivots, arnoldi_size, correction.data());
    const double trailing_squared = std::norm(
        hessenberg[
            static_cast<std::size_t>(arnoldi_size) * stride +
            arnoldi_size - 1]);
    for (int row = 0; row < arnoldi_size; row++) {
        projected[
            static_cast<std::size_t>(row) * arnoldi_size +
            arnoldi_size - 1] += trailing_squared * correction[row];
    }

    std::vector<Complex> inverse_factor = projected;
    std::vector<int> inverse_pivots;
    if (!dense_lu_factor(
            inverse_factor, arnoldi_size, inverse_pivots)) {
        return 0;
    }

    coefficients.resize(
        static_cast<std::size_t>(rank) * arnoldi_size);
    const double normalization =
        1.0 / std::sqrt(static_cast<double>(arnoldi_size));
    for (int column = 0; column < rank; column++) {
        for (int row = 0; row < arnoldi_size; row++) {
            const double phase =
                2.0 * 3.14159265358979323846 *
                static_cast<double>((row + 1) * (column + 1)) /
                static_cast<double>(arnoldi_size + 1);
            coefficients[
                static_cast<std::size_t>(column) * arnoldi_size + row] =
                normalization * Complex(std::cos(phase), std::sin(phase));
        }
    }
    int accepted = orthonormalize_columns(
        coefficients, arnoldi_size, rank);
    for (int iteration = 0; iteration < 20 && accepted > 0; iteration++) {
        for (int column = 0; column < accepted; column++) {
            dense_lu_solve(
                inverse_factor, inverse_pivots, arnoldi_size,
                coefficients.data() +
                    static_cast<std::size_t>(column) * arnoldi_size);
        }
        accepted = orthonormalize_columns(
            coefficients, arnoldi_size, accepted);
    }
    return accepted;
}

}  // namespace bem_krylov

#endif
