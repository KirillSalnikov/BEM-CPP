#include "muller_dense.h"
#include "muller_fmm.h"
#include "muller_mbj.h"

#include <cmath>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    using cdouble = std::complex<double>;
    int refinement = 0;
    int digits = 5;
    int max_leaf = 512;
    double ka = 1.0;
    double refractive_real = 1.3;
    double tolerance = 2.0e-4;
    const char* shape = "sphere";
    int prism_sides = 6;
    double prism_aspect = 1.0;
    double feature_angle = 45.0;
    int near_radius = 3;
    bool near_fp32 = false;
    MullerEdgeMode edge_mode = MullerEdgeMode::SplitFeatureEdges;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--ref") == 0 && i + 1 < argc)
            refinement = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--digits") == 0 && i + 1 < argc)
            digits = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--max-leaf") == 0 && i + 1 < argc)
            max_leaf = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ka") == 0 && i + 1 < argc)
            ka = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ri") == 0 && i + 1 < argc)
            refractive_real = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--tol") == 0 && i + 1 < argc)
            tolerance = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--shape") == 0 && i + 1 < argc)
            shape = argv[++i];
        else if (std::strcmp(argv[i], "--sides") == 0 && i + 1 < argc)
            prism_sides = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--aspect") == 0 && i + 1 < argc)
            prism_aspect = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--feature-angle") == 0 &&
                 i + 1 < argc)
            feature_angle = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--near-radius") == 0 &&
                 i + 1 < argc)
            near_radius = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--near-fp32") == 0)
            near_fp32 = true;
        else if (std::strcmp(argv[i], "--edge-mode") == 0 &&
                 i + 1 < argc) {
            const char* mode = argv[++i];
            if (std::strcmp(mode, "smooth") == 0)
                edge_mode = MullerEdgeMode::Smooth;
            else if (std::strcmp(mode, "split") == 0)
                edge_mode = MullerEdgeMode::SplitFeatureEdges;
            else if (std::strcmp(mode, "hdiv") == 0)
                edge_mode = MullerEdgeMode::HDivBdm1;
            else {
                std::fprintf(stderr, "invalid edge mode\n");
                return 2;
            }
        }
    }
    const bool prism_mode = std::strcmp(shape, "prism") == 0;
    if (!prism_mode && std::strcmp(shape, "sphere") != 0) {
        std::fprintf(stderr, "shape must be sphere or prism\n");
        return 2;
    }
    const Mesh geometry = prism_mode
        ? regular_prism(
              prism_sides, prism_aspect, refinement, 1.0, 0)
        : icosphere(1.0, refinement);
    MullerP2BuildOptions build_options;
    build_options.project_edge_nodes_to_sphere = !prism_mode;
    build_options.edge_mode = edge_mode;
    build_options.feature_angle_degrees = feature_angle;
    const cdouble k(ka, 0.0);
    const cdouble refractive_index(refractive_real, 0.0);
    const char* correction_cache_path =
        "/tmp/bem_muller_near_correction_check.bin";
    std::remove(correction_cache_path);
    const MullerDenseSystem dense =
        assemble_muller_nodal_dense(
            geometry, k, refractive_index, build_options, 7, 4);
    MullerFmmOperator fmm;
    fmm.init(
        geometry, k, refractive_index, build_options,
        7, 4, digits, max_leaf,
        false, 2, 2.0, 0.96, correction_cache_path,
        near_radius);
    fmm.set_fmm_near_fp32(near_fp32);
    if (fmm.near_correction_cache_hit) {
        std::fprintf(
            stderr, "fresh near-correction cache reported a hit\n");
        return 1;
    }
    if (fmm.system_dofs != dense.system_dofs) {
        std::fprintf(stderr, "Muller FMM dimension mismatch\n");
        return 1;
    }

    std::vector<cdouble> input(dense.system_dofs);
    std::vector<cdouble> expected(
        dense.system_dofs, cdouble(0.0));
    std::vector<cdouble> actual(
        dense.system_dofs, cdouble(0.0));
    std::vector<cdouble> direct(
        dense.system_dofs, cdouble(0.0));
    for (int i = 0; i < dense.system_dofs; i++)
        input[i] = cdouble(
            std::sin(0.19 * i),
            0.7 * std::cos(0.13 * i));
    for (int row = 0; row < dense.system_dofs; row++) {
        for (int column = 0;
             column < dense.system_dofs; column++) {
            expected[row] += dense.matrix[
                (size_t)row * dense.system_dofs + column] *
                input[column];
        }
    }
    const char* pair_currents_environment =
        std::getenv("BEM_FMM_PAIR_CURRENTS");
    const char* pair_far_environment =
        std::getenv("BEM_FMM_PAIR_FAR");
    const bool had_pair_currents_environment =
        pair_currents_environment != nullptr;
    const bool had_pair_far_environment =
        pair_far_environment != nullptr;
    const std::string saved_pair_currents =
        had_pair_currents_environment
            ? pair_currents_environment : "";
    const std::string saved_pair_far =
        had_pair_far_environment ? pair_far_environment : "";
    setenv("BEM_FMM_PAIR_CURRENTS", "1", 1);
    setenv("BEM_FMM_PAIR_FAR", "1", 1);
    const auto fmm_start = std::chrono::steady_clock::now();
    fmm.matvec(input.data(), actual.data());
    const double fmm_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - fmm_start).count();
    setenv("BEM_FMM_PAIR_CURRENTS", "0", 1);
    std::vector<cdouble> separate_current_actual(
        dense.system_dofs, cdouble(0.0));
    fmm.matvec(input.data(), separate_current_actual.data());
    if (had_pair_currents_environment)
        setenv(
            "BEM_FMM_PAIR_CURRENTS",
            saved_pair_currents.c_str(), 1);
    else
        unsetenv("BEM_FMM_PAIR_CURRENTS");
    if (had_pair_far_environment)
        setenv("BEM_FMM_PAIR_FAR", saved_pair_far.c_str(), 1);
    else
        unsetenv("BEM_FMM_PAIR_FAR");
    double pair_difference_squared = 0.0;
    double pair_reference_squared = 0.0;
    for (int i = 0; i < dense.system_dofs; i++) {
        pair_difference_squared += std::norm(
            actual[i] - separate_current_actual[i]);
        pair_reference_squared += std::norm(
            separate_current_actual[i]);
    }
    const double pair_relative_error = std::sqrt(
        pair_difference_squared / pair_reference_squared);
    double template_relative_error = 0.0;
    {
        MullerFmmOperator legacy_fmm;
        legacy_fmm.init(
            geometry, k, refractive_index, build_options,
            7, 4, digits, max_leaf,
            false, 2, 2.0, 0.96, nullptr,
            near_radius, false);
        legacy_fmm.set_fmm_near_fp32(near_fp32);
        std::vector<cdouble> legacy_actual(
            dense.system_dofs, cdouble(0.0));
        legacy_fmm.matvec(input.data(), legacy_actual.data());
        double difference_squared = 0.0;
        double template_reference_squared = 0.0;
        for (int i = 0; i < dense.system_dofs; i++) {
            difference_squared +=
                std::norm(actual[i] - legacy_actual[i]);
            template_reference_squared +=
                std::norm(legacy_actual[i]);
        }
        template_relative_error = std::sqrt(
            difference_squared / template_reference_squared);
        legacy_fmm.cleanup();
    }
    const auto direct_start = std::chrono::steady_clock::now();
    fmm.matvec_direct_reference(input.data(), direct.data());
    const double direct_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - direct_start).count();
    double farfield_relative_error = 0.0;
    if (fmm.gpu_operator_assembly) {
        std::vector<Vec3> directions = {
            Vec3(0.0, 0.0, 1.0),
            Vec3(0.0, std::sqrt(0.5), std::sqrt(0.5)),
            Vec3(1.0, 0.0, 0.0)
        };
        std::vector<cdouble> farfield_cpu;
        std::vector<cdouble> farfield_gpu;
        std::vector<cdouble> second_input(input.size());
        for (size_t index = 0; index < input.size(); index++)
            second_input[index] =
                cdouble(0.37, -0.21) * input[index];
        std::vector<cdouble> farfield_second;
        std::vector<cdouble> farfield_cpu_pair_first;
        std::vector<cdouble> farfield_cpu_pair_second;
        std::vector<cdouble> farfield_pair_first;
        std::vector<cdouble> farfield_pair_second;
        muller_nodal_farfield(
            fmm.mesh,
            input.data(),
            input.data() + fmm.current_dofs,
            k, directions, farfield_cpu);
        muller_nodal_farfield_pair(
            fmm.mesh,
            input.data(), input.data() + fmm.current_dofs,
            second_input.data(),
            second_input.data() + fmm.current_dofs,
            k, directions,
            farfield_cpu_pair_first, farfield_cpu_pair_second);
        fmm.farfield(input.data(), directions, farfield_gpu);
        fmm.farfield(
            second_input.data(), directions, farfield_second);
        fmm.farfield_pair(
            input.data(), second_input.data(), directions,
            farfield_pair_first, farfield_pair_second);
        double difference_squared = 0.0;
        double field_squared = 0.0;
        for (size_t index = 0;
             index < farfield_cpu.size(); index++) {
            difference_squared += std::norm(
                farfield_gpu[index] - farfield_cpu[index]);
            difference_squared += std::norm(
                farfield_cpu_pair_first[index] - farfield_cpu[index]);
            difference_squared += std::norm(
                farfield_cpu_pair_second[index] -
                cdouble(0.37, -0.21) * farfield_cpu[index]);
            field_squared += std::norm(farfield_cpu[index]);
        }
        for (size_t index = 0;
             index < farfield_gpu.size(); index++) {
            difference_squared += std::norm(
                farfield_pair_first[index] - farfield_gpu[index]);
            difference_squared += std::norm(
                farfield_pair_second[index] - farfield_second[index]);
            field_squared += std::norm(farfield_gpu[index]);
            field_squared += std::norm(farfield_second[index]);
        }
        farfield_relative_error = std::sqrt(
            difference_squared / field_squared);
    }
    MullerFmmOperator cached_fmm;
    cached_fmm.init(
        geometry, k, refractive_index, build_options,
        7, 4, digits, max_leaf,
        false, 2, 2.0, 0.96, correction_cache_path,
        near_radius);
    cached_fmm.set_fmm_near_fp32(near_fp32);
    if (!cached_fmm.near_correction_cache_hit) {
        std::fprintf(
            stderr, "matching near-correction cache was not loaded\n");
        return 1;
    }
    std::vector<cdouble> cached_actual(
        dense.system_dofs, cdouble(0.0));
    cached_fmm.matvec(input.data(), cached_actual.data());
    double cached_error_squared = 0.0;
    double cached_reference_squared = 0.0;
    for (int i = 0; i < dense.system_dofs; i++) {
        cached_error_squared +=
            std::norm(cached_actual[i] - actual[i]);
        cached_reference_squared += std::norm(actual[i]);
    }
    const double cached_relative_error = std::sqrt(
        cached_error_squared / cached_reference_squared);
    cached_fmm.cleanup();

    MullerFmmOperator incompatible_fmm;
    incompatible_fmm.init(
        geometry, k * 1.01, refractive_index, build_options,
        7, 4, digits, max_leaf,
        false, 2, 2.0, 0.96, correction_cache_path,
        near_radius);
    const bool incompatible_cache_rejected =
        !incompatible_fmm.near_correction_cache_hit;
    incompatible_fmm.cleanup();
    std::remove(correction_cache_path);
    MullerMbjPreconditioner dense_mbj;
    MullerMbjPreconditioner local_mbj;
    dense_mbj.build(dense, 10, 3);
    local_mbj.build(fmm, 10, 3);
    const char* mbj_cache_path =
        "/tmp/bem_muller_mbj_check.bin";
    std::remove(mbj_cache_path);
    MullerMbjPreconditioner stored_mbj;
    MullerMbjPreconditioner cached_mbj;
    stored_mbj.build_cached(fmm, 10, 3, mbj_cache_path);
    cached_mbj.build_cached(fmm, 10, 3, mbj_cache_path);
    const bool mbj_cache_ok =
        !stored_mbj.cache_hit && cached_mbj.cache_hit;
    std::vector<cdouble> dense_preconditioned(
        dense.system_dofs);
    std::vector<cdouble> local_preconditioned(
        dense.system_dofs);
    std::vector<cdouble> cached_preconditioned(
        dense.system_dofs);
    dense_mbj.apply(input.data(), dense_preconditioned.data());
    local_mbj.apply(input.data(), local_preconditioned.data());
    cached_mbj.apply(input.data(), cached_preconditioned.data());
    std::remove(mbj_cache_path);
    double preconditioner_difference_squared = 0.0;
    double preconditioner_reference_squared = 0.0;
    double cached_preconditioner_difference_squared = 0.0;
    for (int i = 0; i < dense.system_dofs; i++) {
        preconditioner_difference_squared += std::norm(
            local_preconditioned[i] - dense_preconditioned[i]);
        preconditioner_reference_squared +=
            std::norm(dense_preconditioned[i]);
        cached_preconditioner_difference_squared += std::norm(
            cached_preconditioned[i] - local_preconditioned[i]);
    }
    const double preconditioner_relative_error = std::sqrt(
        preconditioner_difference_squared /
        preconditioner_reference_squared);
    const double cached_preconditioner_relative_error = std::sqrt(
        cached_preconditioner_difference_squared /
        preconditioner_reference_squared);

    double error_squared = 0.0;
    double reference_squared = 0.0;
    double electric_error_squared = 0.0;
    double electric_reference_squared = 0.0;
    double magnetic_error_squared = 0.0;
    double magnetic_reference_squared = 0.0;
    double direct_error_squared = 0.0;
    for (int i = 0; i < dense.system_dofs; i++) {
        error_squared += std::norm(actual[i] - expected[i]);
        reference_squared += std::norm(expected[i]);
        direct_error_squared +=
            std::norm(direct[i] - expected[i]);
        if (i < dense.current_dofs) {
            electric_error_squared +=
                std::norm(actual[i] - expected[i]);
            electric_reference_squared += std::norm(expected[i]);
        } else {
            magnetic_error_squared +=
                std::norm(actual[i] - expected[i]);
            magnetic_reference_squared += std::norm(expected[i]);
        }
    }
    const double relative_error =
        std::sqrt(error_squared / reference_squared);
    std::printf(
        "Muller FMM operator check: shape=%s dofs=%d quadrature=%zu "
        "feature_edges=%d split_nodes=%d "
        "correction_nnz=%zu relative_error=%.3e "
        "direct_error=%.3e electric=%.3e magnetic=%.3e "
        "mbj_error=%.3e mbj_cache_error=%.3e mbj_cache=%s "
        "pair_error=%.3e "
        "template_error=%.3e "
        "cache_error=%.3e cache_rejected=%s "
        "farfield_error=%.3e fmm_s=%.3f direct_s=%.3f "
        "near_precision=%s\n",
        shape, fmm.system_dofs, fmm.quadrature.size(),
        fmm.mesh.feature_edges,
        fmm.mesh.duplicated_corner_nodes +
            fmm.mesh.duplicated_midpoint_nodes,
        fmm.correction.entries.size(), relative_error,
        std::sqrt(direct_error_squared / reference_squared),
        std::sqrt(
            electric_error_squared / electric_reference_squared),
        std::sqrt(
            magnetic_error_squared / magnetic_reference_squared),
        preconditioner_relative_error,
        cached_preconditioner_relative_error,
        mbj_cache_ok ? "yes" : "no",
        pair_relative_error,
        template_relative_error,
        cached_relative_error,
        incompatible_cache_rejected ? "yes" : "no",
        farfield_relative_error,
        fmm_seconds, direct_seconds,
        near_fp32 ? "fp32" : "fp64");
    fmm.cleanup();
    return relative_error < tolerance &&
           direct_error_squared / reference_squared < 1.0e-20 &&
           preconditioner_relative_error < 1.0e-10 &&
           cached_preconditioner_relative_error < 1.0e-14 &&
           pair_relative_error <
               (near_fp32 ? 3.0e-6 : 1.0e-11) &&
           template_relative_error < 1.0e-11 &&
           mbj_cache_ok &&
           cached_relative_error <
               std::max(1.0e-12, 10.0 * relative_error) &&
           farfield_relative_error < 1.0e-12 &&
           incompatible_cache_rejected
        ? 0 : 1;
}
