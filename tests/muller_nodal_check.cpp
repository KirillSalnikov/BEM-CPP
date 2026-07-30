#include "muller_nodal.h"
#include "muller_duffy.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <map>
#include <vector>

namespace {

double tensor_relative_error(
    const MullerTensor3& first, const MullerTensor3& second)
{
    double numerator = 0.0;
    double denominator = 0.0;
    for (int i = 0; i < 9; i++) {
        numerator += std::norm(first[i] - second[i]);
        denominator += std::norm(first[i]);
    }
    return std::sqrt(numerator / std::max(denominator, 1.0e-300));
}

Vec3 triangle_normal(
    const Mesh& mesh, const MullerP2Element& element)
{
    const Vec3& first =
        mesh.verts[element.topology_vertices[0]];
    const Vec3& second =
        mesh.verts[element.topology_vertices[1]];
    const Vec3& third =
        mesh.verts[element.topology_vertices[2]];
    const Vec3 cross = (second - first).cross(third - first);
    return cross * (1.0 / cross.norm());
}

int common_topology_vertices(
    const MullerP2Element& first,
    const MullerP2Element& second,
    int shared[2])
{
    int count = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (first.topology_vertices[i] ==
                second.topology_vertices[j]) {
                if (count < 2)
                    shared[count] = first.topology_vertices[i];
                count++;
            }
        }
    }
    return count;
}

int edge_midpoint_node(
    const MullerP2Element& element, int first, int second)
{
    const int edge_corners[3][2] = {
        {0, 1}, {1, 2}, {2, 0}
    };
    for (int edge = 0; edge < 3; edge++) {
        const int a =
            element.topology_vertices[edge_corners[edge][0]];
        const int b =
            element.topology_vertices[edge_corners[edge][1]];
        if ((a == first && b == second) ||
            (a == second && b == first))
            return element.nodes[3 + edge];
    }
    return -1;
}

} // namespace

int main()
{
    for (MullerDuffyAdjacency adjacency : {
             MullerDuffyAdjacency::Coincident,
             MullerDuffyAdjacency::EdgeAdjacent,
             MullerDuffyAdjacency::VertexAdjacent}) {
        const std::vector<MullerDuffyPoint> rule =
            muller_duffy_rule(4, adjacency);
        double weight_sum = 0.0;
        for (const MullerDuffyPoint& point : rule) {
            weight_sum += point.weight;
            if (point.test_xi < 0.0 || point.test_eta < 0.0 ||
                point.test_xi + point.test_eta > 1.0 ||
                point.trial_xi < 0.0 || point.trial_eta < 0.0 ||
                point.trial_xi + point.trial_eta > 1.0) {
                std::fprintf(stderr, "Duffy point outside triangle\n");
                return 1;
            }
        }
        if (std::abs(weight_sum - 0.25) > 1.0e-12) {
            std::fprintf(
                stderr, "Duffy constant integral mismatch: %.16g\n",
                weight_sum);
            return 1;
        }
    }

    const Mesh linear = icosphere(1.0, 1);
    const MullerP2Mesh mesh = build_muller_p2_mesh(linear, true);
    const int unique_edges = 3 * linear.nt() / 2;
    if (mesh.scalar_nodes() != linear.nv() + unique_edges) {
        std::fprintf(stderr, "unexpected P2 node count\n");
        return 1;
    }
    if (mesh.system_dofs() != 4 * mesh.scalar_nodes()) {
        std::fprintf(stderr, "unexpected Muller system size\n");
        return 1;
    }

    const Mesh prism = regular_prism(6, 1.0, 0, 1.0, 0);
    const MullerP2Mesh smooth_prism =
        build_muller_p2_mesh(prism, false);
    MullerP2BuildOptions edge_options;
    edge_options.edge_mode = MullerEdgeMode::SplitFeatureEdges;
    edge_options.feature_angle_degrees = 45.0;
    const MullerP2Mesh split_prism =
        build_muller_p2_mesh(prism, edge_options);
    if (split_prism.feature_edges != 24 ||
        split_prism.smooth_patches != 8 ||
        split_prism.scalar_nodes() <= smooth_prism.scalar_nodes() ||
        split_prism.duplicated_corner_nodes <= 0 ||
        split_prism.duplicated_midpoint_nodes !=
            split_prism.feature_edges) {
        std::fprintf(
            stderr,
            "unexpected prism edge split: feature=%d patches=%d "
            "nodes=%d/%d duplicate_corner=%d duplicate_mid=%d\n",
            split_prism.feature_edges,
            split_prism.smooth_patches,
            smooth_prism.scalar_nodes(),
            split_prism.scalar_nodes(),
            split_prism.duplicated_corner_nodes,
            split_prism.duplicated_midpoint_nodes);
        return 1;
    }
    for (int element_index = 0;
         element_index < (int)split_prism.elements.size();
         element_index++) {
        const MullerP2Element& element =
            split_prism.elements[element_index];
        const Vec3 face_normal = triangle_normal(prism, element);
        for (int local = 0; local < 6; local++) {
            if (split_prism.normals[element.nodes[local]].dot(
                    face_normal) < 1.0 - 1.0e-11) {
                std::fprintf(
                    stderr,
                    "prism nodal normal crosses a feature edge\n");
                return 1;
            }
        }
    }
    int checked_feature_pairs = 0;
    int checked_smooth_pairs = 0;
    for (int first = 0;
         first < (int)split_prism.elements.size(); first++) {
        for (int second = first + 1;
             second < (int)split_prism.elements.size(); second++) {
            int shared[2] = {-1, -1};
            if (common_topology_vertices(
                    split_prism.elements[first],
                    split_prism.elements[second], shared) != 2)
                continue;
            const bool feature =
                triangle_normal(
                    prism, split_prism.elements[first]).dot(
                    triangle_normal(
                        prism, split_prism.elements[second])) <
                std::cos(45.0 * std::acos(-1.0) / 180.0);
            const int first_midpoint = edge_midpoint_node(
                split_prism.elements[first], shared[0], shared[1]);
            const int second_midpoint = edge_midpoint_node(
                split_prism.elements[second], shared[0], shared[1]);
            if (first_midpoint < 0 || second_midpoint < 0 ||
                (feature && first_midpoint == second_midpoint) ||
                (!feature && first_midpoint != second_midpoint)) {
                std::fprintf(
                    stderr,
                    "incorrect P2 midpoint sharing at prism edge\n");
                return 1;
            }
            if (feature)
                checked_feature_pairs++;
            else
                checked_smooth_pairs++;
        }
    }
    if (checked_feature_pairs != split_prism.feature_edges ||
        checked_smooth_pairs == 0) {
        std::fprintf(stderr, "prism edge-pair coverage mismatch\n");
        return 1;
    }

    MullerP2BuildOptions hdiv_options;
    hdiv_options.edge_mode = MullerEdgeMode::HDivBdm1;
    const MullerP2Mesh hdiv_prism =
        build_muller_p2_mesh(prism, hdiv_options);
    if (hdiv_prism.basis_kind != MullerBasisKind::HDivBdm1 ||
        hdiv_prism.current_dofs() !=
            2 * hdiv_prism.topology_edge_count ||
        (int)hdiv_prism.current_dof_points.size() !=
            hdiv_prism.current_dofs()) {
        std::fprintf(stderr, "unexpected H(div) Muller dimensions\n");
        return 1;
    }
    std::map<int, std::vector<std::pair<int, int>>> edge_elements;
    for (int element = 0;
         element < (int)hdiv_prism.elements.size(); element++) {
        for (int local_edge = 0; local_edge < 3; local_edge++) {
            edge_elements[
                hdiv_prism.elements[element]
                    .topology_edges[local_edge]
            ].push_back(std::make_pair(element, local_edge));
        }
    }
    double maximum_hdiv_flux_jump = 0.0;
    const double global_parameters[3] = {0.17, 0.51, 0.83};
    for (const auto& entry : edge_elements) {
        if (entry.second.size() != 2) {
            std::fprintf(stderr, "non-manifold H(div) test edge\n");
            return 1;
        }
        for (int moment = 0; moment < 2; moment++) {
            const int global_dof = 2 * entry.first + moment;
            for (double global_parameter : global_parameters) {
                double signed_flux_sum = 0.0;
                for (const auto& side : entry.second) {
                    const int element_index = side.first;
                    const int local_edge = side.second;
                    const MullerP2Element& element =
                        hdiv_prism.elements[element_index];
                    const double local_parameter =
                        element.edge_orientations[local_edge] > 0
                            ? global_parameter
                            : 1.0 - global_parameter;
                    double xi = 0.0;
                    double eta = 0.0;
                    if (local_edge == 0) {
                        xi = local_parameter;
                    } else if (local_edge == 1) {
                        xi = 1.0 - local_parameter;
                        eta = local_parameter;
                    } else {
                        eta = 1.0 - local_parameter;
                    }
                    const MullerFrameSample frame =
                        evaluate_muller_frame(
                            hdiv_prism, element_index, xi, eta);
                    const MullerBasisSample basis =
                        evaluate_muller_basis(
                            hdiv_prism, element_index, frame);
                    Vec3 value;
                    bool found = false;
                    for (int local = 0;
                         local < basis.count; local++) {
                        if (basis.dofs[local] == global_dof) {
                            value = basis.values[local];
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        std::fprintf(
                            stderr, "missing H(div) edge basis\n");
                        return 1;
                    }
                    Vec3 boundary_tangent;
                    if (local_edge == 0)
                        boundary_tangent = frame.derivative_xi;
                    else if (local_edge == 1)
                        boundary_tangent =
                            frame.derivative_eta -
                            frame.derivative_xi;
                    else
                        boundary_tangent =
                            frame.derivative_eta * -1.0;
                    boundary_tangent =
                        boundary_tangent *
                        (1.0 / boundary_tangent.norm());
                    const Vec3 outward_conormal =
                        boundary_tangent.cross(frame.normal);
                    signed_flux_sum +=
                        value.dot(outward_conormal);
                }
                maximum_hdiv_flux_jump = std::max(
                    maximum_hdiv_flux_jump,
                    std::abs(signed_flux_sum));
            }
        }
    }
    if (maximum_hdiv_flux_jump > 1.0e-11) {
        std::fprintf(
            stderr, "H(div) co-normal flux jump: %.3e\n",
            maximum_hdiv_flux_jump);
        return 1;
    }

    for (int i = 0; i < mesh.scalar_nodes(); i++) {
        const Vec3& n = mesh.normals[i];
        const Vec3& t1 = mesh.tangent1[i];
        const Vec3& t2 = mesh.tangent2[i];
        const double error = std::max({
            std::abs(n.norm() - 1.0),
            std::abs(t1.norm() - 1.0),
            std::abs(t2.norm() - 1.0),
            std::abs(n.dot(t1)),
            std::abs(n.dot(t2)),
            std::abs(t1.dot(t2))
        });
        if (error > 1.0e-11) {
            std::fprintf(stderr, "invalid nodal frame: %.3e\n", error);
            return 1;
        }
    }

    const MullerFrameSample sample =
        evaluate_muller_frame(mesh, 0, 0.27, 0.31);
    const double frame_error = std::max({
        std::abs(sample.normal.norm() - 1.0),
        std::abs(sample.tangent1.norm() - 1.0),
        std::abs(sample.tangent2.norm() - 1.0),
        std::abs(sample.normal.dot(sample.tangent1)),
        std::abs(sample.normal.dot(sample.tangent2)),
        std::abs(sample.tangent1.dot(sample.tangent2))
    });
    if (frame_error > 1.0e-11 || sample.jacobian <= 0.0) {
        std::fprintf(stderr, "invalid interpolated frame: %.3e\n", frame_error);
        return 1;
    }

    const std::complex<double> ka(3.0, 0.0);
    const std::complex<double> ki(4.5, 0.0);
    const Vec3 displacement(0.0012, -0.0007, 0.0009);
    const MullerTensor3 exact =
        muller_hessian_difference(ka, ki, displacement, 0.0);
    const MullerTensor3 taylor =
        muller_hessian_difference(ka, ki, displacement, 1.0);
    const double hessian_error = tensor_relative_error(exact, taylor);
    if (hessian_error > 2.0e-5) {
        std::fprintf(
            stderr, "Hessian Taylor mismatch: %.3e\n", hessian_error);
        return 1;
    }

    const Vec3 near_displacement(1.0e-10, -2.0e-10, 1.5e-10);
    const MullerTensor3 near =
        muller_hessian_difference(ka, ki, near_displacement);
    double scaled_norm = 0.0;
    for (const auto& value : near)
        scaled_norm += std::norm(
            value * near_displacement.norm());
    if (!std::isfinite(scaled_norm) || scaled_norm <= 0.0) {
        std::fprintf(stderr, "unstable near Hessian difference\n");
        return 1;
    }

    const auto gradient = muller_composite_gradient(
        std::complex<double>(1.0, 0.0),
        std::complex<double>(2.25, 0.0),
        ka, ki, near_displacement);
    for (const auto& value : gradient) {
        if (!std::isfinite(value.real()) ||
            !std::isfinite(value.imag())) {
            std::fprintf(stderr, "unstable composite gradient\n");
            return 1;
        }
    }

    const Vec3 rotated_test =
        sample.tangent1.cross(sample.normal);
    const std::complex<double> k1 = muller_k1_kernel(
        ka, ki, near_displacement,
        rotated_test, sample.tangent2);
    const std::complex<double> k2 = muller_k2_kernel(
        std::complex<double>(1.0, 0.0),
        std::complex<double>(2.25, 0.0),
        ka, ki, near_displacement,
        sample.tangent1, sample.normal, sample.tangent2);
    if (!std::isfinite(k1.real()) || !std::isfinite(k1.imag()) ||
        !std::isfinite(k2.real()) || !std::isfinite(k2.imag())) {
        std::fprintf(stderr, "unstable Muller scalar kernels\n");
        return 1;
    }

    std::printf(
        "Muller P2 geometry/kernel check: nodes=%d dofs=%d "
        "frame_error=%.3e hessian_error=%.3e "
        "prism_feature_edges=%d prism_split_nodes=%d\n",
        mesh.scalar_nodes(), mesh.system_dofs(),
        frame_error, hessian_error,
        split_prism.feature_edges, split_prism.scalar_nodes());
    return 0;
}
