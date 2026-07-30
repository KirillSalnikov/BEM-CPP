#include "muller_dense.h"
#include "muller_mbj.h"

#include <cmath>
#include <complex>
#include <cstdio>

int main()
{
    const Mesh sphere = icosphere(1.0, 0);
    const MullerDenseSystem system = assemble_muller_nodal_dense(
        sphere,
        std::complex<double>(1.0, 0.0),
        std::complex<double>(1.3, 0.0),
        true, 7, 4);
    if (system.mesh.scalar_nodes() != 42 ||
        system.current_dofs != 84 ||
        system.system_dofs != 168 ||
        system.matrix.size() != (size_t)168 * 168) {
        std::fprintf(stderr, "unexpected dense Muller dimensions\n");
        return 1;
    }
    const long long element_pairs =
        system.stats.regular_element_pairs +
        system.stats.coincident_element_pairs +
        system.stats.edge_adjacent_element_pairs +
        system.stats.vertex_adjacent_element_pairs;
    if (element_pairs != (long long)sphere.nt() * sphere.nt()) {
        std::fprintf(stderr, "incomplete element-pair classification\n");
        return 1;
    }
    double matrix_norm2 = 0.0;
    for (const auto& value : system.matrix) {
        if (!std::isfinite(value.real()) ||
            !std::isfinite(value.imag())) {
            std::fprintf(stderr, "non-finite dense Muller entry\n");
            return 1;
        }
        matrix_norm2 += std::norm(value);
    }
    if (matrix_norm2 <= 0.0) {
        std::fprintf(stderr, "empty dense Muller matrix\n");
        return 1;
    }

    const std::vector<std::complex<double>> rhs =
        muller_nodal_planewave_rhs(
            system.mesh,
            std::complex<double>(1.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 1.0));
    if ((int)rhs.size() != system.system_dofs) {
        std::fprintf(stderr, "unexpected dense Muller RHS size\n");
        return 1;
    }
    double rhs_norm2 = 0.0;
    for (const auto& value : rhs)
        rhs_norm2 += std::norm(value);
    if (rhs_norm2 <= 0.0) {
        std::fprintf(stderr, "empty dense Muller RHS\n");
        return 1;
    }

    MullerMbjPreconditioner mbj;
    mbj.build(system, 10);
    std::vector<std::complex<double>> expected(
        system.system_dofs);
    for (int i = 0; i < system.system_dofs; i++)
        expected[i] = std::complex<double>(
            std::sin(0.17 * i), std::cos(0.11 * i));
    std::vector<std::complex<double>> block_rhs(
        system.system_dofs, std::complex<double>(0.0));
    for (const MullerMbjBlock& block : mbj.blocks) {
        for (int local_row = 0;
             local_row < (int)block.dofs.size();
             local_row++) {
            const int row = block.dofs[local_row];
            for (int local_col = 0;
                 local_col < (int)block.dofs.size();
                 local_col++) {
                const int col = block.dofs[local_col];
                block_rhs[row] +=
                    system.matrix[
                        (size_t)row * system.system_dofs + col] *
                    expected[col];
            }
        }
    }
    std::vector<std::complex<double>> recovered(
        system.system_dofs);
    mbj.apply(block_rhs.data(), recovered.data());
    std::vector<std::complex<double>> reconstructed_rhs(
        system.system_dofs, std::complex<double>(0.0));
    for (const MullerMbjBlock& block : mbj.blocks) {
        for (int local_row = 0;
             local_row < (int)block.dofs.size();
             local_row++) {
            const int row = block.dofs[local_row];
            for (int local_col = 0;
                 local_col < (int)block.dofs.size();
                 local_col++) {
                const int col = block.dofs[local_col];
                reconstructed_rhs[row] +=
                    system.matrix[
                        (size_t)row * system.system_dofs + col] *
                    recovered[col];
            }
        }
    }
    double mbj_residual2 = 0.0;
    double block_rhs_norm2 = 0.0;
    for (int i = 0; i < system.system_dofs; i++) {
        mbj_residual2 +=
            std::norm(reconstructed_rhs[i] - block_rhs[i]);
        block_rhs_norm2 += std::norm(block_rhs[i]);
    }
    const double mbj_residual =
        std::sqrt(mbj_residual2 / block_rhs_norm2);
    if (mbj_residual > 1.0e-10) {
        std::fprintf(
            stderr, "MBJ block residual mismatch: %.3e\n",
            mbj_residual);
        return 1;
    }

    MullerP2BuildOptions edge_options;
    edge_options.edge_mode = MullerEdgeMode::SplitFeatureEdges;
    edge_options.feature_angle_degrees = 45.0;
    const Mesh prism = regular_prism(6, 1.0, 0, 1.0, 0);
    const MullerDenseSystem prism_system =
        assemble_muller_nodal_dense(
            prism,
            std::complex<double>(1.0, 0.0),
            std::complex<double>(1.3, 0.0),
            edge_options, 3, 2);
    const long long prism_element_pairs =
        prism_system.stats.regular_element_pairs +
        prism_system.stats.coincident_element_pairs +
        prism_system.stats.edge_adjacent_element_pairs +
        prism_system.stats.vertex_adjacent_element_pairs;
    if (prism_element_pairs !=
            (long long)prism.nt() * prism.nt() ||
        prism_system.stats.edge_adjacent_element_pairs !=
            3LL * prism.nt()) {
        std::fprintf(
            stderr,
            "split-edge geometry lost singular adjacency: "
            "pairs=%lld edge_pairs=%lld expected=%d\n",
            prism_element_pairs,
            prism_system.stats.edge_adjacent_element_pairs,
            3 * prism.nt());
        return 1;
    }
    for (const auto& value : prism_system.matrix) {
        if (!std::isfinite(value.real()) ||
            !std::isfinite(value.imag())) {
            std::fprintf(
                stderr, "non-finite split-edge Muller entry\n");
            return 1;
        }
    }

    MullerP2BuildOptions hdiv_options;
    hdiv_options.edge_mode = MullerEdgeMode::HDivBdm1;
    const MullerDenseSystem hdiv_prism_system =
        assemble_muller_nodal_dense(
            prism,
            std::complex<double>(1.0, 0.0),
            std::complex<double>(1.3, 0.0),
            hdiv_options, 3, 2);
    if (hdiv_prism_system.mesh.basis_kind !=
            MullerBasisKind::HDivBdm1 ||
        hdiv_prism_system.current_dofs !=
            2 * hdiv_prism_system.mesh.topology_edge_count ||
        hdiv_prism_system.system_dofs !=
            2 * hdiv_prism_system.current_dofs) {
        std::fprintf(stderr, "unexpected dense H(div) dimensions\n");
        return 1;
    }
    double hdiv_matrix_norm2 = 0.0;
    for (const auto& value : hdiv_prism_system.matrix) {
        if (!std::isfinite(value.real()) ||
            !std::isfinite(value.imag())) {
            std::fprintf(stderr, "non-finite H(div) Muller entry\n");
            return 1;
        }
        hdiv_matrix_norm2 += std::norm(value);
    }
    const std::vector<std::complex<double>> hdiv_rhs =
        muller_nodal_planewave_rhs(
            hdiv_prism_system.mesh,
            std::complex<double>(1.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 1.0));
    double hdiv_rhs_norm2 = 0.0;
    for (const auto& value : hdiv_rhs)
        hdiv_rhs_norm2 += std::norm(value);
    if (hdiv_matrix_norm2 <= 0.0 || hdiv_rhs_norm2 <= 0.0 ||
        (int)hdiv_rhs.size() != hdiv_prism_system.system_dofs) {
        std::fprintf(stderr, "empty dense H(div) system\n");
        return 1;
    }

    std::printf(
        "Muller dense assembly check: elements=%d nodes=%d dofs=%d "
        "quadrature_pairs=%lld mbj_residual=%.3e "
        "prism_dofs=%d hdiv_prism_dofs=%d "
        "prism_edge_pairs=%lld\n",
        sphere.nt(), system.mesh.scalar_nodes(), system.system_dofs,
        system.stats.quadrature_pairs, mbj_residual,
        prism_system.system_dofs, hdiv_prism_system.system_dofs,
        prism_system.stats.edge_adjacent_element_pairs);
    return 0;
}
