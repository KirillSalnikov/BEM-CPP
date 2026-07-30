#include "fmm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

using cdouble = std::complex<double>;

void direct_hessian(
    const std::vector<double>& targets,
    const std::vector<double>& sources,
    const std::vector<cdouble>& charges,
    cdouble k,
    std::vector<cdouble>& result)
{
    const int nt = (int)targets.size() / 3;
    const int ns = (int)sources.size() / 3;
    const cdouble imaginary(0.0, 1.0);
    const int row[6] = {0, 0, 0, 1, 1, 2};
    const int col[6] = {0, 1, 2, 1, 2, 2};
    result.assign((size_t)nt * 6, cdouble(0.0));
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
            if (radius < 1.0e-13)
                continue;
            const double inv_radius = 1.0 / radius;
            const cdouble green =
                std::exp(imaginary * k * radius) *
                (0.07957747154594767 * inv_radius);
            const cdouble a =
                3.0 * inv_radius * inv_radius -
                3.0 * imaginary * k * inv_radius - k * k;
            const cdouble b =
                inv_radius * inv_radius -
                imaginary * k * inv_radius;
            for (int component = 0; component < 6; component++) {
                cdouble hessian = green * a *
                    displacement[row[component]] *
                    displacement[col[component]] *
                    inv_radius * inv_radius;
                if (row[component] == col[component])
                    hessian -= green * b;
                result[6 * target + component] +=
                    hessian * charges[source];
            }
        }
    }
}

void direct_gradient(
    const std::vector<double>& targets,
    const std::vector<double>& sources,
    const std::vector<cdouble>& charges,
    cdouble k,
    std::vector<cdouble>& result)
{
    const int nt = (int)targets.size() / 3;
    const int ns = (int)sources.size() / 3;
    const cdouble imaginary(0.0, 1.0);
    result.assign((size_t)nt * 3, cdouble(0.0));
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
            if (radius < 1.0e-13)
                continue;
            const double inv_radius = 1.0 / radius;
            const cdouble green =
                std::exp(imaginary * k * radius) *
                (0.07957747154594767 * inv_radius);
            const cdouble radial =
                green * (imaginary * k - inv_radius) *
                inv_radius * charges[source];
            for (int axis = 0; axis < 3; axis++)
                result[3 * target + axis] +=
                    radial * displacement[axis];
        }
    }
}

void contract_vector_derivatives(
    const std::vector<cdouble>& gradient_x,
    const std::vector<cdouble>& gradient_y,
    const std::vector<cdouble>& gradient_z,
    const std::vector<cdouble>& hessian_x,
    const std::vector<cdouble>& hessian_y,
    const std::vector<cdouble>& hessian_z,
    std::vector<cdouble>& curl,
    std::vector<cdouble>& hessian_action)
{
    const int point_count = (int)gradient_x.size() / 3;
    curl.resize((size_t)point_count * 3);
    hessian_action.resize((size_t)point_count * 3);
    for (int point = 0; point < point_count; point++) {
        curl[3 * point] =
            gradient_x[3 * point + 1] -
            gradient_y[3 * point];
        curl[3 * point + 1] =
            gradient_x[3 * point + 2] -
            gradient_z[3 * point];
        curl[3 * point + 2] =
            gradient_y[3 * point + 2] -
            gradient_z[3 * point + 1];
        hessian_action[3 * point] =
            -hessian_x[6 * point + 3] -
            hessian_x[6 * point + 5] +
            hessian_y[6 * point + 1] +
            hessian_z[6 * point + 2];
        hessian_action[3 * point + 1] =
            hessian_x[6 * point + 1] -
            hessian_y[6 * point] -
            hessian_y[6 * point + 5] +
            hessian_z[6 * point + 4];
        hessian_action[3 * point + 2] =
            hessian_x[6 * point + 2] +
            hessian_y[6 * point + 4] -
            hessian_z[6 * point] -
            hessian_z[6 * point + 3];
    }
}

} // namespace

int main(int argc, char** argv)
{
    bool surface = false;
    int surface_points = 560;
    int near_radius = 2;
    int max_leaf = 128;
    int digits = 8;
    bool near_fp32 = false;
    for (int argument = 1; argument < argc; argument++) {
        const std::string option(argv[argument]);
        if (option == "--surface")
            surface = true;
        else if (option == "--points" && argument + 1 < argc)
            surface_points = std::atoi(argv[++argument]);
        else if (option == "--near-radius" && argument + 1 < argc)
            near_radius = std::atoi(argv[++argument]);
        else if (option == "--max-leaf" && argument + 1 < argc)
            max_leaf = std::atoi(argv[++argument]);
        else if (option == "--digits" && argument + 1 < argc)
            digits = std::atoi(argv[++argument]);
        else if (option == "--near-fp32")
            near_fp32 = true;
    }
    const int nt = surface ? surface_points : 256;
    const int ns = surface ? nt : 768;
    std::mt19937 generator(260421181u);
    std::uniform_real_distribution<double> coordinate(-1.0, 1.0);
    std::vector<double> targets((size_t)nt * 3);
    std::vector<double> sources((size_t)ns * 3);
    std::vector<cdouble> charges(ns);
    std::vector<cdouble> charges2(ns);
    std::vector<cdouble> charges3(ns);
    if (surface) {
        const double golden_angle =
            M_PI * (3.0 - std::sqrt(5.0));
        for (int point = 0; point < nt; point++) {
            const double z =
                1.0 - 2.0 * (point + 0.5) / nt;
            const double radius =
                std::sqrt(std::max(0.0, 1.0 - z * z));
            const double azimuth = golden_angle * point;
            targets[3 * point] = radius * std::cos(azimuth);
            targets[3 * point + 1] = radius * std::sin(azimuth);
            targets[3 * point + 2] = z;
        }
        sources = targets;
    } else {
        for (double& value : targets)
            value = coordinate(generator);
        for (double& value : sources)
            value = coordinate(generator);
    }
    for (int source = 0; source < ns; source++) {
        charges[source] = cdouble(
            coordinate(generator), coordinate(generator));
        charges2[source] = cdouble(
            coordinate(generator), coordinate(generator));
        charges3[source] = cdouble(
            coordinate(generator), coordinate(generator));
    }

    const cdouble k =
        surface ? cdouble(1.3, 0.0) : cdouble(2.3, 0.08);
    HelmholtzFMM fmm;
    fmm.init(
        targets.data(), nt, sources.data(), ns,
        k, surface ? digits : 7, surface ? max_leaf : 24,
        surface ? near_radius : 1, true);
    fmm.near_field_fp32 = near_fp32;
    std::vector<cdouble> actual((size_t)nt * 6);
    std::vector<cdouble> combined((size_t)nt * 6);
    std::vector<cdouble> gradient((size_t)nt * 3);
    std::vector<cdouble> expected;
    std::vector<cdouble> expected_gradient;
    std::vector<cdouble> batch_gradient1((size_t)nt * 3);
    std::vector<cdouble> batch_gradient2((size_t)nt * 3);
    std::vector<cdouble> batch_gradient3((size_t)nt * 3);
    std::vector<cdouble> batch_hessian1((size_t)nt * 6);
    std::vector<cdouble> batch_hessian2((size_t)nt * 6);
    std::vector<cdouble> batch_hessian3((size_t)nt * 6);
    std::vector<cdouble> expected_hessian2;
    std::vector<cdouble> expected_hessian3;
    std::vector<cdouble> expected_gradient2;
    std::vector<cdouble> expected_gradient3;
    std::vector<cdouble> contracted_curl((size_t)nt * 3);
    std::vector<cdouble> contracted_hessian_action((size_t)nt * 3);
    std::vector<cdouble> reference_curl;
    std::vector<cdouble> reference_hessian_action;
    fmm.evaluate_hessian(charges.data(), actual.data());
    fmm.evaluate_grad_hessian(
        charges.data(), gradient.data(), combined.data());
    direct_hessian(targets, sources, charges, k, expected);
    direct_gradient(
        targets, sources, charges, k, expected_gradient);
    const auto batch_start = std::chrono::steady_clock::now();
    fmm.evaluate_grad_hessian_batch3(
        charges.data(), charges2.data(), charges3.data(),
        batch_gradient1.data(),
        batch_gradient2.data(),
        batch_gradient3.data(),
        batch_hessian1.data(),
        batch_hessian2.data(),
        batch_hessian3.data());
    const double batch_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - batch_start).count();
    direct_hessian(targets, sources, charges2, k, expected_hessian2);
    direct_hessian(targets, sources, charges3, k, expected_hessian3);
    direct_gradient(targets, sources, charges2, k, expected_gradient2);
    direct_gradient(targets, sources, charges3, k, expected_gradient3);
    fmm.evaluate_vector_actions_batch3(
        charges.data(), charges2.data(), charges3.data(),
        contracted_curl.data(), contracted_hessian_action.data());
    contract_vector_derivatives(
        batch_gradient1, batch_gradient2, batch_gradient3,
        batch_hessian1, batch_hessian2, batch_hessian3,
        reference_curl, reference_hessian_action);

    double error_squared = 0.0;
    double reference_squared = 0.0;
    double combined_difference_squared = 0.0;
    double gradient_error_squared = 0.0;
    double gradient_reference_squared = 0.0;
    double maximum_relative = 0.0;
    double batch_error_squared = 0.0;
    double batch_reference_squared = 0.0;
    double contracted_error_squared = 0.0;
    double contracted_reference_squared = 0.0;
    for (size_t i = 0; i < actual.size(); i++) {
        error_squared += std::norm(actual[i] - expected[i]);
        reference_squared += std::norm(expected[i]);
        combined_difference_squared +=
            std::norm(actual[i] - combined[i]);
        if (std::abs(expected[i]) > 1.0e-8)
            maximum_relative = std::max(
                maximum_relative,
                std::abs(actual[i] - expected[i]) /
                    std::abs(expected[i]));
        batch_error_squared +=
            std::norm(batch_hessian1[i] - expected[i]) +
            std::norm(batch_hessian2[i] - expected_hessian2[i]) +
            std::norm(batch_hessian3[i] - expected_hessian3[i]);
        batch_reference_squared +=
            std::norm(expected[i]) +
            std::norm(expected_hessian2[i]) +
            std::norm(expected_hessian3[i]);
    }
    for (size_t i = 0; i < gradient.size(); i++) {
        gradient_error_squared +=
            std::norm(gradient[i] - expected_gradient[i]);
        gradient_reference_squared +=
            std::norm(expected_gradient[i]);
        batch_error_squared +=
            std::norm(batch_gradient1[i] - expected_gradient[i]) +
            std::norm(batch_gradient2[i] - expected_gradient2[i]) +
            std::norm(batch_gradient3[i] - expected_gradient3[i]);
        batch_reference_squared +=
            std::norm(expected_gradient[i]) +
            std::norm(expected_gradient2[i]) +
            std::norm(expected_gradient3[i]);
        contracted_error_squared +=
            std::norm(contracted_curl[i] - reference_curl[i]) +
            std::norm(
                contracted_hessian_action[i] -
                reference_hessian_action[i]);
        contracted_reference_squared +=
            std::norm(reference_curl[i]) +
            std::norm(reference_hessian_action[i]);
    }
    const double relative_l2 =
        std::sqrt(error_squared / reference_squared);
    const double combined_relative_l2 =
        std::sqrt(combined_difference_squared / reference_squared);
    const double gradient_relative_l2 =
        std::sqrt(
            gradient_error_squared / gradient_reference_squared);
    const double batch_relative_l2 =
        std::sqrt(batch_error_squared / batch_reference_squared);
    const double contracted_relative_l2 =
        std::sqrt(
            contracted_error_squared /
            contracted_reference_squared);
    std::printf(
        "FMM Hessian check (%s): relative_l2=%.3e "
        "gradient_l2=%.3e combined_difference=%.3e "
        "batch3_l2=%.3e contracted_l2=%.3e "
        "max_relative=%.3e batch3_s=%.6f "
        "near_precision=%s\n",
        surface ? "surface" : "volume",
        relative_l2, gradient_relative_l2,
        combined_relative_l2, batch_relative_l2,
        contracted_relative_l2, maximum_relative, batch_seconds,
        near_fp32 ? "fp32" : "fp64");
    fmm.cleanup();
    if (surface) {
        std::vector<cdouble> repeated((size_t)nt * 6);
        fmm.init(
            targets.data(), nt, sources.data(), ns,
            k, digits, max_leaf, near_radius);
        fmm.near_field_fp32 = near_fp32;
        fmm.evaluate_hessian(charges.data(), repeated.data());
        double repeated_error_squared = 0.0;
        for (size_t i = 0; i < repeated.size(); i++)
            repeated_error_squared +=
                std::norm(repeated[i] - expected[i]);
        const double repeated_relative_l2 =
            std::sqrt(repeated_error_squared / reference_squared);
        std::printf(
            "FMM reinitialization check: relative_l2=%.3e\n",
            repeated_relative_l2);
        fmm.cleanup();
        if (repeated_relative_l2 >= 5.0e-3)
            return 1;
    }
    return relative_l2 < 5.0e-3 &&
           gradient_relative_l2 < 5.0e-3 &&
           combined_relative_l2 < 1.0e-7 &&
           batch_relative_l2 < 5.0e-3 &&
           contracted_relative_l2 <
               (near_fp32 ? 1.0e-6 : 1.0e-7) ? 0 : 1;
}
