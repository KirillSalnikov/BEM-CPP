#include "muller_dense.h"
#include "muller_mbj.h"
#include "types.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cdouble = std::complex<double>;

struct GmresResult {
    int iterations = 0;
    double relative_residual = 1.0;
    double seconds = 0.0;
};

double seconds_since(const std::chrono::steady_clock::time_point& start);

cdouble dot_product(
    const cdouble* first, const cdouble* second, int n)
{
    cdouble result(0.0);
    for (int i = 0; i < n; i++)
        result += std::conj(first[i]) * second[i];
    return result;
}

double vector_norm(const cdouble* vector, int n)
{
    double result = 0.0;
    for (int i = 0; i < n; i++)
        result += std::norm(vector[i]);
    return std::sqrt(result);
}

void dense_matvec(
    const std::vector<cdouble>& matrix,
    const cdouble* input,
    cdouble* output,
    int n)
{
    for (int row = 0; row < n; row++) {
        cdouble value(0.0);
        for (int col = 0; col < n; col++)
            value += matrix[(size_t)row * n + col] * input[col];
        output[row] = value;
    }
}

GmresResult solve_gmres(
    const std::vector<cdouble>& matrix,
    const cdouble* rhs,
    int n,
    double tolerance,
    int maximum_iterations,
    const MullerMbjPreconditioner* preconditioner)
{
    const auto start = std::chrono::steady_clock::now();
    const int maximum = std::min(n, maximum_iterations);
    std::vector<cdouble> basis((size_t)(maximum + 1) * n);
    std::vector<cdouble> hessenberg(
        (size_t)(maximum + 1) * maximum, cdouble(0.0));
    std::vector<cdouble> cosine(maximum);
    std::vector<cdouble> sine(maximum);
    std::vector<cdouble> projected(maximum + 1, cdouble(0.0));
    std::vector<cdouble> work(n), preconditioned(n);
    std::vector<cdouble> solution(n, cdouble(0.0));
    const double rhs_norm = std::max(vector_norm(rhs, n), 1.0e-300);
    for (int i = 0; i < n; i++)
        basis[i] = rhs[i] / rhs_norm;
    projected[0] = rhs_norm;

    int iterations = 0;
    for (int column = 0; column < maximum; column++) {
        const cdouble* vector =
            basis.data() + (size_t)column * n;
        if (preconditioner) {
            preconditioner->apply(
                vector, preconditioned.data());
            dense_matvec(
                matrix, preconditioned.data(), work.data(), n);
        } else {
            dense_matvec(matrix, vector, work.data(), n);
        }
        for (int row = 0; row <= column; row++) {
            const cdouble* previous =
                basis.data() + (size_t)row * n;
            const cdouble value =
                dot_product(previous, work.data(), n);
            hessenberg[(size_t)row * maximum + column] = value;
            for (int i = 0; i < n; i++)
                work[i] -= value * previous[i];
        }
        // A second pass keeps the dense validation solve insensitive
        // to loss of orthogonality.
        for (int row = 0; row <= column; row++) {
            const cdouble* previous =
                basis.data() + (size_t)row * n;
            const cdouble correction =
                dot_product(previous, work.data(), n);
            hessenberg[(size_t)row * maximum + column] +=
                correction;
            for (int i = 0; i < n; i++)
                work[i] -= correction * previous[i];
        }
        const double next_norm = vector_norm(work.data(), n);
        hessenberg[
            (size_t)(column + 1) * maximum + column] =
            cdouble(next_norm, 0.0);
        if (next_norm > 1.0e-30) {
            cdouble* next =
                basis.data() + (size_t)(column + 1) * n;
            for (int i = 0; i < n; i++)
                next[i] = work[i] / next_norm;
        }

        for (int row = 0; row < column; row++) {
            const cdouble first =
                hessenberg[(size_t)row * maximum + column];
            const cdouble second =
                hessenberg[(size_t)(row + 1) * maximum + column];
            hessenberg[(size_t)row * maximum + column] =
                std::conj(cosine[row]) * first +
                std::conj(sine[row]) * second;
            hessenberg[(size_t)(row + 1) * maximum + column] =
                -sine[row] * first + cosine[row] * second;
        }
        const cdouble first =
            hessenberg[(size_t)column * maximum + column];
        const cdouble second =
            hessenberg[(size_t)(column + 1) * maximum + column];
        const double denominator =
            std::sqrt(std::norm(first) + std::norm(second));
        cosine[column] = denominator > 1.0e-30
            ? first / denominator
            : cdouble(1.0, 0.0);
        sine[column] = denominator > 1.0e-30
            ? second / denominator
            : cdouble(0.0, 0.0);
        hessenberg[(size_t)column * maximum + column] =
            std::conj(cosine[column]) * first +
            std::conj(sine[column]) * second;
        hessenberg[
            (size_t)(column + 1) * maximum + column] =
            cdouble(0.0);
        projected[column + 1] =
            -sine[column] * projected[column];
        projected[column] =
            std::conj(cosine[column]) * projected[column];
        iterations = column + 1;
        if (std::abs(projected[column + 1]) / rhs_norm <
            tolerance) {
            break;
        }
    }

    std::vector<cdouble> coefficients(iterations);
    for (int row = iterations - 1; row >= 0; row--) {
        cdouble value = projected[row];
        for (int col = row + 1; col < iterations; col++)
            value -=
                hessenberg[(size_t)row * maximum + col] *
                coefficients[col];
        coefficients[row] =
            value /
            hessenberg[(size_t)row * maximum + row];
    }
    std::fill(work.begin(), work.end(), cdouble(0.0));
    for (int column = 0; column < iterations; column++) {
        const cdouble* vector =
            basis.data() + (size_t)column * n;
        for (int i = 0; i < n; i++)
            work[i] += coefficients[column] * vector[i];
    }
    if (preconditioner)
        preconditioner->apply(work.data(), solution.data());
    else
        solution = work;
    dense_matvec(matrix, solution.data(), work.data(), n);
    double residual_norm2 = 0.0;
    for (int i = 0; i < n; i++)
        residual_norm2 += std::norm(rhs[i] - work[i]);

    GmresResult result;
    result.iterations = iterations;
    result.relative_residual =
        std::sqrt(residual_norm2) / rhs_norm;
    result.seconds = seconds_since(start);
    return result;
}

void solve_dense(
    std::vector<cdouble>& matrix,
    int n,
    std::vector<cdouble>& rhs,
    int right_hand_sides)
{
    for (int k = 0; k < n; k++) {
        int pivot = k;
        double pivot_norm = std::abs(matrix[(size_t)k * n + k]);
        for (int row = k + 1; row < n; row++) {
            const double candidate =
                std::abs(matrix[(size_t)row * n + k]);
            if (candidate > pivot_norm) {
                pivot = row;
                pivot_norm = candidate;
            }
        }
        if (pivot_norm < 1.0e-24)
            throw std::runtime_error("singular dense Muller matrix");
        if (pivot != k) {
            for (int col = 0; col < n; col++)
                std::swap(
                    matrix[(size_t)k * n + col],
                    matrix[(size_t)pivot * n + col]);
            for (int r = 0; r < right_hand_sides; r++)
                std::swap(rhs[(size_t)r * n + k],
                          rhs[(size_t)r * n + pivot]);
        }
        const cdouble diagonal = matrix[(size_t)k * n + k];
        for (int row = k + 1; row < n; row++) {
            const cdouble factor =
                matrix[(size_t)row * n + k] / diagonal;
            matrix[(size_t)row * n + k] = factor;
            for (int col = k + 1; col < n; col++)
                matrix[(size_t)row * n + col] -=
                    factor * matrix[(size_t)k * n + col];
            for (int r = 0; r < right_hand_sides; r++)
                rhs[(size_t)r * n + row] -=
                    factor * rhs[(size_t)r * n + k];
        }
    }
    for (int r = 0; r < right_hand_sides; r++) {
        cdouble* x = rhs.data() + (size_t)r * n;
        for (int row = n - 1; row >= 0; row--) {
            cdouble value = x[row];
            for (int col = row + 1; col < n; col++)
                value -= matrix[(size_t)row * n + col] * x[col];
            x[row] = value / matrix[(size_t)row * n + row];
        }
    }
}

double relative_residual(
    const std::vector<cdouble>& matrix,
    const cdouble* solution,
    const cdouble* rhs,
    int n)
{
    double numerator = 0.0;
    double denominator = 0.0;
    for (int row = 0; row < n; row++) {
        cdouble value(0.0);
        for (int col = 0; col < n; col++)
            value += matrix[(size_t)row * n + col] * solution[col];
        numerator += std::norm(value - rhs[row]);
        denominator += std::norm(rhs[row]);
    }
    return std::sqrt(numerator / std::max(denominator, 1.0e-300));
}

void amplitude_to_mueller(
    const std::vector<cdouble>& s1,
    const std::vector<cdouble>& s2,
    const std::vector<cdouble>& s3,
    const std::vector<cdouble>& s4,
    std::vector<double>& mueller)
{
    const int ntheta = (int)s1.size();
    mueller.assign((size_t)16 * ntheta, 0.0);
    auto at = [&](int i, int j, int t) -> double& {
        return mueller[((size_t)i * 4 + j) * ntheta + t];
    };
    for (int t = 0; t < ntheta; t++) {
        const double a1 = std::norm(s1[t]);
        const double a2 = std::norm(s2[t]);
        const double a3 = std::norm(s3[t]);
        const double a4 = std::norm(s4[t]);
        const cdouble s2s3 = s2[t] * std::conj(s3[t]);
        const cdouble s1s4 = s1[t] * std::conj(s4[t]);
        const cdouble s2s4 = s2[t] * std::conj(s4[t]);
        const cdouble s1s3 = s1[t] * std::conj(s3[t]);
        const cdouble s1s2 = s1[t] * std::conj(s2[t]);
        const cdouble s3s4 = s3[t] * std::conj(s4[t]);
        at(0, 0, t) = 0.5 * (a1 + a2 + a3 + a4);
        at(0, 1, t) = 0.5 * (a2 - a1 + a4 - a3);
        at(1, 0, t) = 0.5 * (a2 - a1 - a4 + a3);
        at(1, 1, t) = 0.5 * (a2 + a1 - a4 - a3);
        at(0, 2, t) = s2s3.real() + s1s4.real();
        at(0, 3, t) = s2s3.imag() - s1s4.imag();
        at(1, 2, t) = s2s3.real() - s1s4.real();
        at(1, 3, t) = s2s3.imag() + s1s4.imag();
        at(2, 0, t) = s2s4.real() + s1s3.real();
        at(2, 1, t) = s2s4.real() - s1s3.real();
        at(2, 2, t) = s1s2.real() + s3s4.real();
        at(2, 3, t) = -s1s2.imag() - s3s4.imag();
        at(3, 0, t) =
            (s4[t] * std::conj(s2[t])).imag() + s1s3.imag();
        at(3, 1, t) =
            (s4[t] * std::conj(s2[t])).imag() - s1s3.imag();
        at(3, 2, t) = s1s2.imag() - s3s4.imag();
        at(3, 3, t) = s1s2.real() - s3s4.real();
    }
}

double seconds_since(
    const std::chrono::steady_clock::time_point& start)
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
}

} // namespace

int main(int argc, char** argv)
{
    double ka = 1.0;
    double n_re = 1.3;
    double n_im = 0.0;
    int refinement = 0;
    int ntheta = 37;
    int regular_quadrature = 7;
    int duffy_order = 4;
    bool benchmark_gmres = false;
    int mbj_nodes = 50;
    std::string shape = "sphere";
    int prism_sides = 6;
    double prism_aspect = 1.0;
    int edge_refine = 0;
    double feature_angle = 45.0;
    bool edge_mode_explicit = false;
    MullerEdgeMode edge_mode = MullerEdgeMode::Smooth;
    std::string output = "runs/muller_nodal_demo.json";
    for (int i = 1; i < argc; i++) {
        const std::string argument = argv[i];
        if (argument == "--ka" && i + 1 < argc)
            ka = std::atof(argv[++i]);
        else if (argument == "--ri" && i + 2 < argc) {
            n_re = std::atof(argv[++i]);
            n_im = std::atof(argv[++i]);
        } else if (argument == "--ref" && i + 1 < argc)
            refinement = std::atoi(argv[++i]);
        else if (argument == "--ntheta" && i + 1 < argc)
            ntheta = std::atoi(argv[++i]);
        else if (argument == "--regular-quad" && i + 1 < argc)
            regular_quadrature = std::atoi(argv[++i]);
        else if (argument == "--duffy-order" && i + 1 < argc)
            duffy_order = std::atoi(argv[++i]);
        else if (argument == "--benchmark-gmres")
            benchmark_gmres = true;
        else if (argument == "--mbj-nodes" && i + 1 < argc)
            mbj_nodes = std::atoi(argv[++i]);
        else if (argument == "--shape" && i + 1 < argc)
            shape = argv[++i];
        else if (argument == "--sides" && i + 1 < argc)
            prism_sides = std::atoi(argv[++i]);
        else if (argument == "--aspect" && i + 1 < argc)
            prism_aspect = std::atof(argv[++i]);
        else if (argument == "--edge-refine" && i + 1 < argc)
            edge_refine = std::atoi(argv[++i]);
        else if (argument == "--feature-angle" && i + 1 < argc)
            feature_angle = std::atof(argv[++i]);
        else if (argument == "--edge-mode" && i + 1 < argc) {
            const std::string mode = argv[++i];
            edge_mode_explicit = true;
            if (mode == "smooth")
                edge_mode = MullerEdgeMode::Smooth;
            else if (mode == "split")
                edge_mode = MullerEdgeMode::SplitFeatureEdges;
            else {
                std::fprintf(
                    stderr, "--edge-mode must be smooth or split\n");
                return 1;
            }
        }
        else if (argument == "--out" && i + 1 < argc)
            output = argv[++i];
        else {
            std::fprintf(stderr, "unknown or incomplete argument: %s\n",
                         argument.c_str());
            return 1;
        }
    }
    if (shape != "sphere" && shape != "prism") {
        std::fprintf(stderr, "--shape must be sphere or prism\n");
        return 1;
    }
    const bool prism_mode = shape == "prism";
    if (prism_mode && !edge_mode_explicit)
        edge_mode = MullerEdgeMode::SplitFeatureEdges;

    const auto total_start = std::chrono::steady_clock::now();
    const Mesh geometry = prism_mode
        ? regular_prism(
              prism_sides, prism_aspect, refinement,
              1.0, edge_refine)
        : icosphere(1.0, refinement);
    MullerP2BuildOptions build_options;
    build_options.project_edge_nodes_to_sphere = !prism_mode;
    build_options.edge_mode = edge_mode;
    build_options.feature_angle_degrees = feature_angle;
    const auto assembly_start = std::chrono::steady_clock::now();
    MullerDenseSystem system = assemble_muller_nodal_dense(
        geometry, cdouble(ka, 0.0), cdouble(n_re, n_im),
        build_options,
        regular_quadrature, duffy_order);
    const double assembly_seconds = seconds_since(assembly_start);

    const Vec3 propagation(0.0, 0.0, 1.0);
    const std::vector<cdouble> rhs_parallel =
        muller_nodal_planewave_rhs(
            system.mesh, cdouble(ka, 0.0),
            Vec3(0.0, 1.0, 0.0), propagation);
    const std::vector<cdouble> rhs_perpendicular =
        muller_nodal_planewave_rhs(
            system.mesh, cdouble(ka, 0.0),
            Vec3(1.0, 0.0, 0.0), propagation);
    GmresResult baseline_gmres;
    GmresResult mbj_gmres;
    double mbj_setup_seconds = 0.0;
    double mbj_storage_megabytes = 0.0;
    int mbj_blocks = 0;
    if (benchmark_gmres) {
        baseline_gmres = solve_gmres(
            system.matrix, rhs_parallel.data(),
            system.system_dofs, 1.0e-8, 1000, nullptr);
        const auto mbj_start = std::chrono::steady_clock::now();
        MullerMbjPreconditioner mbj;
        mbj.build(system, mbj_nodes);
        mbj_setup_seconds = seconds_since(mbj_start);
        mbj_storage_megabytes = mbj.storage_megabytes();
        mbj_blocks = (int)mbj.blocks.size();
        mbj_gmres = solve_gmres(
            system.matrix, rhs_parallel.data(),
            system.system_dofs, 1.0e-8, 1000, &mbj);
    }
    std::vector<cdouble> solutions(
        (size_t)2 * system.system_dofs);
    std::copy(
        rhs_parallel.begin(), rhs_parallel.end(),
        solutions.begin());
    std::copy(
        rhs_perpendicular.begin(), rhs_perpendicular.end(),
        solutions.begin() + system.system_dofs);
    std::vector<cdouble> factors = system.matrix;
    const auto solve_start = std::chrono::steady_clock::now();
    solve_dense(factors, system.system_dofs, solutions, 2);
    const double solve_seconds = seconds_since(solve_start);
    const double residual_parallel = relative_residual(
        system.matrix, solutions.data(), rhs_parallel.data(),
        system.system_dofs);
    const double residual_perpendicular = relative_residual(
        system.matrix,
        solutions.data() + system.system_dofs,
        rhs_perpendicular.data(), system.system_dofs);

    std::vector<double> theta(ntheta);
    std::vector<Vec3> directions(ntheta);
    std::vector<Vec3> theta_hat(ntheta);
    for (int t = 0; t < ntheta; t++) {
        theta[t] = 180.0 * t / (ntheta - 1);
        const double radians = theta[t] * M_PI / 180.0;
        directions[t] =
            Vec3(0.0, std::sin(radians), std::cos(radians));
        theta_hat[t] =
            Vec3(0.0, std::cos(radians), -std::sin(radians));
    }
    std::vector<cdouble> field_parallel;
    std::vector<cdouble> field_perpendicular;
    const int current_dofs = system.current_dofs;
    muller_nodal_farfield(
        system.mesh,
        solutions.data(),
        solutions.data() + current_dofs,
        cdouble(ka, 0.0), directions, field_parallel);
    muller_nodal_farfield(
        system.mesh,
        solutions.data() + system.system_dofs,
        solutions.data() + system.system_dofs + current_dofs,
        cdouble(ka, 0.0), directions, field_perpendicular);

    std::vector<cdouble> s1(ntheta), s2(ntheta);
    std::vector<cdouble> s3(ntheta), s4(ntheta);
    const cdouble amplitude_scale(0.0, -ka);
    for (int t = 0; t < ntheta; t++) {
        const cdouble parallel_theta =
            field_parallel[3 * t] * theta_hat[t].x +
            field_parallel[3 * t + 1] * theta_hat[t].y +
            field_parallel[3 * t + 2] * theta_hat[t].z;
        const cdouble perpendicular_theta =
            field_perpendicular[3 * t] * theta_hat[t].x +
            field_perpendicular[3 * t + 1] * theta_hat[t].y +
            field_perpendicular[3 * t + 2] * theta_hat[t].z;
        s2[t] = amplitude_scale * parallel_theta;
        s4[t] = amplitude_scale * field_parallel[3 * t];
        s3[t] = amplitude_scale * perpendicular_theta;
        s1[t] = amplitude_scale * field_perpendicular[3 * t];
    }
    std::vector<double> mueller;
    amplitude_to_mueller(s1, s2, s3, s4, mueller);

    std::ofstream stream(output);
    if (!stream) {
        std::fprintf(stderr, "cannot create %s\n", output.c_str());
        return 1;
    }
    stream << std::setprecision(17);
    stream << "{\n";
    stream << "  \"solver\": \"muller_nodal_p2_dense\",\n";
    stream << "  \"shape\": \"" << shape << "\",\n";
    stream << "  \"ka\": " << ka << ",\n";
    stream << "  \"ri\": [" << n_re << ", " << n_im << "],\n";
    stream << "  \"refinements\": " << refinement << ",\n";
    stream << "  \"p2_nodes\": " << system.mesh.scalar_nodes() << ",\n";
    stream << "  \"system_dofs\": " << system.system_dofs << ",\n";
    stream << "  \"edge_mode\": \""
           << (edge_mode == MullerEdgeMode::SplitFeatureEdges
                   ? "split" : "smooth")
           << "\",\n";
    stream << "  \"feature_angle_degrees\": "
           << feature_angle << ",\n";
    stream << "  \"feature_edge_segments\": "
           << system.mesh.feature_edges << ",\n";
    stream << "  \"smooth_patches\": "
           << system.mesh.smooth_patches << ",\n";
    stream << "  \"duplicated_edge_nodes\": "
           << system.mesh.duplicated_corner_nodes +
                  system.mesh.duplicated_midpoint_nodes
           << ",\n";
    stream << "  \"regular_quadrature\": "
           << regular_quadrature << ",\n";
    stream << "  \"duffy_order\": " << duffy_order << ",\n";
    if (benchmark_gmres) {
        stream << "  \"gmres_benchmark\": {\n";
        stream << "    \"tolerance\": 1e-8,\n";
        stream << "    \"baseline_iterations\": "
               << baseline_gmres.iterations << ",\n";
        stream << "    \"baseline_relative_residual\": "
               << baseline_gmres.relative_residual << ",\n";
        stream << "    \"baseline_s\": "
               << baseline_gmres.seconds << ",\n";
        stream << "    \"mbj_scalar_nodes_per_block\": "
               << mbj_nodes << ",\n";
        stream << "    \"mbj_blocks\": " << mbj_blocks << ",\n";
        stream << "    \"mbj_setup_s\": "
               << mbj_setup_seconds << ",\n";
        stream << "    \"mbj_storage_mb\": "
               << mbj_storage_megabytes << ",\n";
        stream << "    \"mbj_iterations\": "
               << mbj_gmres.iterations << ",\n";
        stream << "    \"mbj_relative_residual\": "
               << mbj_gmres.relative_residual << ",\n";
        stream << "    \"mbj_s\": " << mbj_gmres.seconds << "\n";
        stream << "  },\n";
    }
    stream << "  \"relative_residual\": "
           << std::max(residual_parallel, residual_perpendicular) << ",\n";
    stream << "  \"timing\": {\"assembly_s\": " << assembly_seconds
           << ", \"solve_s\": " << solve_seconds
           << ", \"total_s\": " << seconds_since(total_start) << "},\n";
    stream << "  \"theta\": [";
    for (int t = 0; t < ntheta; t++) {
        if (t) stream << ", ";
        stream << theta[t];
    }
    stream << "],\n  \"mueller\": [\n";
    for (int i = 0; i < 4; i++) {
        stream << "    [\n";
        for (int j = 0; j < 4; j++) {
            stream << "      [";
            for (int t = 0; t < ntheta; t++) {
                if (t) stream << ", ";
                stream << mueller[((i * 4 + j) * ntheta) + t];
            }
            stream << "]" << (j == 3 ? "\n" : ",\n");
        }
        stream << "    ]" << (i == 3 ? "\n" : ",\n");
    }
    stream << "  ],\n  \"amplitudes\": {\n";
    const char* names[4] = {"S1", "S2", "S3", "S4"};
    const std::vector<cdouble>* amplitudes[4] = {
        &s1, &s2, &s3, &s4
    };
    for (int component = 0; component < 4; component++) {
        stream << "    \"" << names[component] << "\": [";
        for (int t = 0; t < ntheta; t++) {
            if (t) stream << ", ";
            const cdouble value = (*amplitudes[component])[t];
            stream << "[" << value.real() << ", " << value.imag() << "]";
        }
        stream << "]" << (component == 3 ? "\n" : ",\n");
    }
    stream << "  }\n}\n";

    std::printf(
        "Muller nodal demo: shape=%s ref=%d nodes=%d dofs=%d "
        "feature_edges=%d "
        "assembly=%.3fs solve=%.3fs residual=%.3e out=%s\n",
        shape.c_str(), refinement, system.mesh.scalar_nodes(),
        system.system_dofs, system.mesh.feature_edges,
        assembly_seconds, solve_seconds,
        std::max(residual_parallel, residual_perpendicular),
        output.c_str());
    if (benchmark_gmres) {
        std::printf(
            "  GMRES(1e-8): baseline=%d (%.3fs, rel=%.2e), "
            "MBJ=%d (setup %.3fs + solve %.3fs, rel=%.2e, %.2fMB)\n",
            baseline_gmres.iterations, baseline_gmres.seconds,
            baseline_gmres.relative_residual,
            mbj_gmres.iterations, mbj_setup_seconds,
            mbj_gmres.seconds, mbj_gmres.relative_residual,
            mbj_storage_megabytes);
    }
    return 0;
}
