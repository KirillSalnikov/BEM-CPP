#include "pfft.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using cdouble = std::complex<double>;

void direct_derivatives(
    const std::vector<double>& targets,
    const std::vector<double>& sources,
    const std::vector<cdouble>& charges,
    cdouble k,
    std::vector<cdouble>& gradient,
    std::vector<cdouble>& hessian)
{
    const int nt = static_cast<int>(targets.size()) / 3;
    const int ns = static_cast<int>(sources.size()) / 3;
    const cdouble imaginary(0.0, 1.0);
    const int row[6] = {0, 0, 0, 1, 1, 2};
    const int column[6] = {0, 1, 2, 1, 2, 2};
    gradient.assign(static_cast<size_t>(nt) * 3, cdouble(0.0));
    hessian.assign(static_cast<size_t>(nt) * 6, cdouble(0.0));
    for (int target = 0; target < nt; target++) {
        for (int source = 0; source < ns; source++) {
            const double displacement[3] = {
                targets[3 * target] - sources[3 * source],
                targets[3 * target + 1] - sources[3 * source + 1],
                targets[3 * target + 2] - sources[3 * source + 2]
            };
            const double radius = std::sqrt(
                displacement[0] * displacement[0] +
                displacement[1] * displacement[1] +
                displacement[2] * displacement[2]);
            const double inv_radius = 1.0 / radius;
            const cdouble green =
                std::exp(imaginary * k * radius) *
                (0.07957747154594767 * inv_radius);
            const cdouble radial =
                green * (imaginary * k - inv_radius) *
                inv_radius * charges[source];
            for (int axis = 0; axis < 3; axis++) {
                gradient[3 * target + axis] +=
                    radial * displacement[axis];
            }
            const cdouble a =
                3.0 * inv_radius * inv_radius -
                3.0 * imaginary * k * inv_radius - k * k;
            const cdouble b =
                inv_radius * inv_radius - imaginary * k * inv_radius;
            for (int component = 0; component < 6; component++) {
                cdouble value =
                    green * a *
                    displacement[row[component]] *
                    displacement[column[component]] *
                    inv_radius * inv_radius;
                if (row[component] == column[component])
                    value -= green * b;
                hessian[6 * target + component] +=
                    value * charges[source];
            }
        }
    }
}

double relative_l2(
    const std::vector<cdouble>& actual,
    const std::vector<cdouble>& expected)
{
    double difference = 0.0;
    double reference = 0.0;
    for (size_t i = 0; i < actual.size(); i++) {
        difference += std::norm(actual[i] - expected[i]);
        reference += std::norm(expected[i]);
    }
    return std::sqrt(difference / reference);
}

void contract_vector_derivatives(
    const std::array<std::vector<cdouble>, 3>& gradients,
    const std::array<std::vector<cdouble>, 3>& hessians,
    std::vector<cdouble>& curl,
    std::vector<cdouble>& hessian_action)
{
    const size_t point_count = gradients[0].size() / 3;
    curl.resize(point_count * 3);
    hessian_action.resize(point_count * 3);
    for (size_t point = 0; point < point_count; point++) {
        curl[3 * point] =
            gradients[0][3 * point + 1] -
            gradients[1][3 * point];
        curl[3 * point + 1] =
            gradients[0][3 * point + 2] -
            gradients[2][3 * point];
        curl[3 * point + 2] =
            gradients[1][3 * point + 2] -
            gradients[2][3 * point + 1];

        hessian_action[3 * point] =
            -hessians[0][6 * point + 3] -
            hessians[0][6 * point + 5] +
            hessians[1][6 * point + 1] +
            hessians[2][6 * point + 2];
        hessian_action[3 * point + 1] =
            hessians[0][6 * point + 1] -
            hessians[1][6 * point] -
            hessians[1][6 * point + 5] +
            hessians[2][6 * point + 4];
        hessian_action[3 * point + 2] =
            hessians[0][6 * point + 2] +
            hessians[1][6 * point + 4] -
            hessians[2][6 * point] -
            hessians[2][6 * point + 3];
    }
}

} // namespace

int main()
{
    const int nt = 64;
    const int ns = 128;
    std::mt19937 generator(260727u);
    std::uniform_real_distribution<double> coordinate(-1.0, 1.0);
    std::vector<double> targets(static_cast<size_t>(nt) * 3);
    std::vector<double> sources(static_cast<size_t>(ns) * 3);
    std::vector<cdouble> charges(ns);
    for (double& value : targets)
        value = coordinate(generator);
    for (double& value : sources)
        value = coordinate(generator);
    for (cdouble& value : charges)
        value = cdouble(coordinate(generator), coordinate(generator));

    const cdouble k(2.3, 0.08);
    HelmholtzPFFT pfft;
    pfft.init(
        targets.data(), nt, sources.data(), ns,
        k, 2, 64);
    std::vector<cdouble> actual_gradient(static_cast<size_t>(nt) * 3);
    std::vector<cdouble> actual_hessian(static_cast<size_t>(nt) * 6);
    std::vector<cdouble> expected_gradient;
    std::vector<cdouble> expected_hessian;
    pfft.evaluate_grad_hessian(
        charges.data(),
        actual_gradient.data(),
        actual_hessian.data());
    direct_derivatives(
        targets, sources, charges, k,
        expected_gradient, expected_hessian);

    const double gradient_error =
        relative_l2(actual_gradient, expected_gradient);
    const double hessian_error =
        relative_l2(actual_hessian, expected_hessian);
    const cdouble second_k(3.1, 0.04);
    HelmholtzPFFT second_pfft;
    second_pfft.init(
        targets.data(), nt, sources.data(), ns,
        second_k, 2, 64, pfft.h);
    std::vector<cdouble> independent_gradient(
        static_cast<size_t>(nt) * 3);
    std::vector<cdouble> independent_hessian(
        static_cast<size_t>(nt) * 6);
    std::vector<cdouble> shared_gradient(
        static_cast<size_t>(nt) * 3);
    std::vector<cdouble> shared_hessian(
        static_cast<size_t>(nt) * 6);
    second_pfft.evaluate_grad_hessian(
        charges.data(),
        independent_gradient.data(),
        independent_hessian.data());
    second_pfft.evaluate_grad_hessian_from_prepared(
        pfft, shared_gradient.data(), shared_hessian.data());
    const double shared_gradient_error =
        relative_l2(shared_gradient, independent_gradient);
    const double shared_hessian_error =
        relative_l2(shared_hessian, independent_hessian);

    std::array<std::vector<cdouble>, 3> vector_charges;
    std::array<std::vector<cdouble>, 3> component_gradients;
    std::array<std::vector<cdouble>, 3> component_hessians;
    for (int component = 0; component < 3; component++) {
        vector_charges[component].resize(ns);
        component_gradients[component].resize(
            static_cast<size_t>(nt) * 3);
        component_hessians[component].resize(
            static_cast<size_t>(nt) * 6);
        for (cdouble& value : vector_charges[component]) {
            value = cdouble(
                coordinate(generator), coordinate(generator));
        }
        pfft.evaluate_grad_hessian(
            vector_charges[component].data(),
            component_gradients[component].data(),
            component_hessians[component].data());
    }
    std::vector<cdouble> expected_curl;
    std::vector<cdouble> expected_hessian_action;
    contract_vector_derivatives(
        component_gradients, component_hessians,
        expected_curl, expected_hessian_action);
    std::vector<cdouble> actual_curl(
        static_cast<size_t>(nt) * 3);
    std::vector<cdouble> actual_hessian_action(
        static_cast<size_t>(nt) * 3);
    pfft.evaluate_vector_actions(
        vector_charges[0].data(),
        vector_charges[1].data(),
        vector_charges[2].data(),
        actual_curl.data(),
        actual_hessian_action.data());
    const double vector_curl_error =
        relative_l2(actual_curl, expected_curl);
    const double vector_hessian_error =
        relative_l2(
            actual_hessian_action,
            expected_hessian_action);

    std::vector<cdouble> independent_curl(
        static_cast<size_t>(nt) * 3);
    std::vector<cdouble> independent_hessian_action(
        static_cast<size_t>(nt) * 3);
    std::vector<cdouble> shared_curl(
        static_cast<size_t>(nt) * 3);
    std::vector<cdouble> shared_hessian_action(
        static_cast<size_t>(nt) * 3);
    second_pfft.evaluate_vector_actions(
        vector_charges[0].data(),
        vector_charges[1].data(),
        vector_charges[2].data(),
        independent_curl.data(),
        independent_hessian_action.data());
    second_pfft.evaluate_vector_actions_from_prepared(
        pfft, shared_curl.data(), shared_hessian_action.data());
    const double shared_vector_curl_error =
        relative_l2(shared_curl, independent_curl);
    const double shared_vector_hessian_error =
        relative_l2(
            shared_hessian_action,
            independent_hessian_action);
    std::printf(
        "pFFT derivative check: gradient_l2=%.3e hessian_l2=%.3e "
        "shared_gradient=%.3e shared_hessian=%.3e "
        "vector_curl=%.3e vector_hessian=%.3e "
        "shared_vector_curl=%.3e shared_vector_hessian=%.3e\n",
        gradient_error, hessian_error,
        shared_gradient_error, shared_hessian_error,
        vector_curl_error, vector_hessian_error,
        shared_vector_curl_error,
        shared_vector_hessian_error);
    second_pfft.cleanup();
    pfft.cleanup();
    return gradient_error < 5.0e-2 && hessian_error < 8.0e-2 &&
           shared_gradient_error < 1.0e-12 &&
           shared_hessian_error < 1.0e-12 &&
           vector_curl_error < 1.0e-12 &&
           vector_hessian_error < 1.0e-12 &&
           shared_vector_curl_error < 1.0e-12 &&
           shared_vector_hessian_error < 1.0e-12
        ? 0 : 1;
}
