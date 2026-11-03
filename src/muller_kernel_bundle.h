#ifndef BEM_MULLER_KERNEL_BUNDLE_H
#define BEM_MULLER_KERNEL_BUNDLE_H

#include "muller_nodal.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bem_muller {

struct KernelValues {
    std::complex<double> k1_scalar = 0.0;
    MullerTensor3 hessian;
    std::array<std::complex<double>, 3> gradient_epsilon;
    std::array<std::complex<double>, 3> gradient_mu;
};

// All three Galerkin kernels share the two material Green functions. Keep
// the cancellation-safe Taylor formulas for small k*r; away from that
// branch compute each complex exponential once rather than repeatedly in
// the Hessian and the two composite gradients.
inline KernelValues evaluate_kernel_values(
    std::complex<double> k_exterior,
    std::complex<double> k_interior,
    std::complex<double> epsilon_exterior,
    std::complex<double> epsilon_interior,
    std::complex<double> mu_exterior,
    std::complex<double> mu_interior,
    const Vec3& displacement)
{
    using C = std::complex<double>;
    const double radius = displacement.norm();
    if (radius <= 0.0)
        throw std::runtime_error("Muller kernel values at zero distance");
    const C imaginary(0.0, 1.0);
    const C exterior = std::exp(imaginary * k_exterior * radius) * INV4PI;
    const C interior = std::exp(imaginary * k_interior * radius) * INV4PI;
    const C phi_exterior = exterior / radius;
    const C phi_interior = interior / radius;
    KernelValues values;
    values.k1_scalar = k_exterior * k_exterior * phi_exterior -
                       k_interior * k_interior * phi_interior;
    if (std::max(std::abs(k_exterior), std::abs(k_interior)) * radius < 1.0e-2) {
        values.hessian = muller_hessian_difference(
            k_exterior, k_interior, displacement);
        values.gradient_epsilon = muller_composite_gradient(
            epsilon_exterior, epsilon_interior,
            k_exterior, k_interior, displacement);
        values.gradient_mu = muller_composite_gradient(
            mu_exterior, mu_interior,
            k_exterior, k_interior, displacement);
        return values;
    }
    const double radius3 = std::pow(radius, 3);
    const C h_exterior = exterior / radius3;
    const C h_interior = interior / radius3;
    const C f = h_exterior * (3.0 - 3.0 * imaginary * k_exterior * radius -
                              k_exterior * k_exterior * radius * radius) -
                h_interior * (3.0 - 3.0 * imaginary * k_interior * radius -
                              k_interior * k_interior * radius * radius);
    const C g = h_exterior * (1.0 - imaginary * k_exterior * radius) -
                h_interior * (1.0 - imaginary * k_interior * radius);
    const C radial_exterior = phi_exterior * (imaginary * k_exterior - 1.0 / radius);
    const C radial_interior = phi_interior * (imaginary * k_interior - 1.0 / radius);
    const C radial_epsilon = epsilon_exterior * radial_exterior -
                             epsilon_interior * radial_interior;
    const C radial_mu = mu_exterior * radial_exterior - mu_interior * radial_interior;
    const double unit[3] = {displacement.x / radius, displacement.y / radius,
                            displacement.z / radius};
    for (int row = 0; row < 3; ++row) {
        values.gradient_epsilon[row] = radial_epsilon * unit[row];
        values.gradient_mu[row] = radial_mu * unit[row];
        for (int column = 0; column < 3; ++column)
            values.hessian[3 * row + column] =
                f * unit[row] * unit[column] - g * (row == column ? 1.0 : 0.0);
    }
    return values;
}

} // namespace bem_muller

#endif
