#include "muller_nodal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace {

using cdouble = std::complex<double>;

Vec3 project_to_radius(const Vec3& point, double radius)
{
    const double norm = point.norm();
    if (norm <= 0.0)
        throw std::runtime_error("cannot project a zero P2 node");
    return point * (radius / norm);
}

Vec3 interpolate_vec(
    const std::array<double, 6>& shape,
    const MullerP2Element& element,
    const std::vector<Vec3>& values)
{
    Vec3 result;
    for (int local = 0; local < 6; local++)
        result = result + values[element.nodes[local]] * shape[local];
    return result;
}

Vec3 normalized_checked(const Vec3& value, const char* what)
{
    const double norm = value.norm();
    if (norm < 1.0e-14)
        throw std::runtime_error(what);
    return value * (1.0 / norm);
}

cdouble factorial(int n)
{
    double result = 1.0;
    for (int i = 2; i <= n; i++)
        result *= (double)i;
    return cdouble(result, 0.0);
}

struct DisjointSet {
    explicit DisjointSet(int size) : parent(size), rank(size, 0)
    {
        for (int i = 0; i < size; i++)
            parent[i] = i;
    }

    int find(int value)
    {
        if (parent[value] != value)
            parent[value] = find(parent[value]);
        return parent[value];
    }

    void unite(int first, int second)
    {
        first = find(first);
        second = find(second);
        if (first == second)
            return;
        if (rank[first] < rank[second])
            std::swap(first, second);
        parent[second] = first;
        if (rank[first] == rank[second])
            rank[first]++;
    }

    std::vector<int> parent;
    std::vector<int> rank;
};

int triangle_local_vertex(
    const Mesh& mesh, int triangle, int vertex)
{
    for (int local = 0; local < 3; local++) {
        if (mesh.tris[3 * triangle + local] == vertex)
            return local;
    }
    throw std::runtime_error("edge vertex is absent from adjacent triangle");
}

} // namespace

void muller_p2_shape(
    double xi, double eta,
    std::array<double, 6>& shape,
    std::array<double, 6>& dshape_dxi,
    std::array<double, 6>& dshape_deta)
{
    const double l0 = 1.0 - xi - eta;
    const double l1 = xi;
    const double l2 = eta;

    shape = {
        l0 * (2.0 * l0 - 1.0),
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        4.0 * l0 * l1,
        4.0 * l1 * l2,
        4.0 * l2 * l0
    };
    dshape_dxi = {
        1.0 - 4.0 * l0,
        4.0 * l1 - 1.0,
        0.0,
        4.0 * (l0 - l1),
        4.0 * l2,
        -4.0 * l2
    };
    dshape_deta = {
        1.0 - 4.0 * l0,
        0.0,
        4.0 * l2 - 1.0,
        -4.0 * l1,
        4.0 * l1,
        4.0 * (l0 - l2)
    };
}

MullerFrameSample evaluate_muller_frame(
    const MullerP2Mesh& mesh, int element_index, double xi, double eta)
{
    if (element_index < 0 || element_index >= (int)mesh.elements.size())
        throw std::out_of_range("invalid Muller P2 element");
    const MullerP2Element& element = mesh.elements[element_index];
    std::array<double, 6> shape, dxi, deta;
    muller_p2_shape(xi, eta, shape, dxi, deta);

    Vec3 position;
    Vec3 a1;
    Vec3 a2;
    for (int local = 0; local < 6; local++) {
        const Vec3& node = mesh.nodes[element.nodes[local]];
        position = position + node * shape[local];
        a1 = a1 + node * dxi[local];
        a2 = a2 + node * deta[local];
    }
    const Vec3 cross = a1.cross(a2);
    const double jacobian = cross.norm();
    if (jacobian < 1.0e-14)
        throw std::runtime_error("degenerate Muller P2 element");
    const Vec3 normal = cross * (1.0 / jacobian);

    Vec3 raw_t1 = interpolate_vec(shape, element, mesh.tangent1);
    Vec3 raw_t2 = interpolate_vec(shape, element, mesh.tangent2);
    raw_t1 = raw_t1 - normal * raw_t1.dot(normal);
    const Vec3 tangent1 =
        normalized_checked(raw_t1, "degenerate interpolated tangent1");
    raw_t2 = raw_t2 - normal * raw_t2.dot(normal);
    raw_t2 = raw_t2 - tangent1 * raw_t2.dot(tangent1);
    const Vec3 tangent2 =
        normalized_checked(raw_t2, "degenerate interpolated tangent2");

    MullerFrameSample result;
    result.position = position;
    result.normal = normal;
    result.tangent1 = tangent1;
    result.tangent2 = tangent2;
    result.derivative_xi = a1;
    result.derivative_eta = a2;
    result.reference_xi = xi;
    result.reference_eta = eta;
    result.jacobian = jacobian;
    result.shape = shape;
    return result;
}

MullerBasisSample evaluate_muller_basis(
    const MullerP2Mesh& mesh,
    int element_index,
    const MullerFrameSample& frame)
{
    if (element_index < 0 || element_index >= (int)mesh.elements.size())
        throw std::out_of_range("invalid Muller basis element");
    const MullerP2Element& element = mesh.elements[element_index];
    MullerBasisSample result;
    if (mesh.basis_kind == MullerBasisKind::NodalP2) {
        result.count = 12;
        for (int local = 0; local < 6; local++) {
            result.dofs[2 * local] = 2 * element.nodes[local];
            result.dofs[2 * local + 1] =
                2 * element.nodes[local] + 1;
            result.values[2 * local] =
                frame.tangent1 * frame.shape[local];
            result.values[2 * local + 1] =
                frame.tangent2 * frame.shape[local];
        }
        return result;
    }

    // BDM1 on the reference triangle. The six functions are dual to
    // P0/P1 moments of the outward co-normal trace on edges
    // (v0,v1), (v1,v2), and (v2,v0), respectively.
    const double xi = frame.reference_xi;
    const double eta = frame.reference_eta;
    const double reference[6][2] = {
        {xi, eta - 1.0},
        {3.0 * xi, -6.0 * xi - 3.0 * eta + 3.0},
        {xi, eta},
        {-3.0 * xi, 3.0 * eta},
        {xi - 1.0, eta},
        {3.0 * xi + 6.0 * eta - 3.0, -3.0 * eta}
    };
    result.count = 6;
    for (int local_edge = 0; local_edge < 3; local_edge++) {
        for (int moment = 0; moment < 2; moment++) {
            const int local = 2 * local_edge + moment;
            const int orientation =
                moment == 0
                    ? element.edge_orientations[local_edge]
                    : 1;
            result.dofs[local] =
                2 * element.topology_edges[local_edge] + moment;
            result.values[local] =
                (frame.derivative_xi * reference[local][0] +
                 frame.derivative_eta * reference[local][1]) *
                ((double)orientation / frame.jacobian);
        }
    }
    return result;
}

MullerP2Mesh build_muller_p2_mesh(
    const Mesh& mesh, const MullerP2BuildOptions& options)
{
    if (mesh.nt() == 0 || mesh.nv() == 0)
        throw std::runtime_error("empty mesh for Muller P2 conversion");
    if (options.feature_angle_degrees <= 0.0 ||
        options.feature_angle_degrees >= 180.0) {
        throw std::invalid_argument(
            "Muller feature angle must be between 0 and 180 degrees");
    }

    MullerP2Mesh result;
    result.edge_mode = options.edge_mode;
    result.basis_kind =
        options.edge_mode == MullerEdgeMode::HDivBdm1
            ? MullerBasisKind::HDivBdm1
            : MullerBasisKind::NodalP2;
    result.feature_angle_degrees = options.feature_angle_degrees;
    double radius = 0.0;
    if (options.project_edge_nodes_to_sphere) {
        for (const Vec3& vertex : mesh.verts)
            radius += vertex.norm();
        radius /= (double)mesh.nv();
    }

    using Edge = std::pair<int, int>;
    std::map<Edge, std::vector<int>> edge_triangles;
    std::set<int> used_vertices;
    std::vector<Vec3> triangle_normals(mesh.nt());
    for (int triangle = 0; triangle < mesh.nt(); triangle++) {
        const int v[3] = {
            mesh.tris[3 * triangle],
            mesh.tris[3 * triangle + 1],
            mesh.tris[3 * triangle + 2]
        };
        const Vec3 cross =
            (mesh.verts[v[1]] - mesh.verts[v[0]]).cross(
                mesh.verts[v[2]] - mesh.verts[v[0]]);
        triangle_normals[triangle] = normalized_checked(
            cross, "degenerate triangle in Muller edge detection");
        used_vertices.insert(v[0]);
        used_vertices.insert(v[1]);
        used_vertices.insert(v[2]);
        for (int edge = 0; edge < 3; edge++) {
            const int first = v[edge];
            const int second = v[(edge + 1) % 3];
            edge_triangles[Edge(
                std::min(first, second),
                std::max(first, second))].push_back(triangle);
        }
    }

    const double pi = std::acos(-1.0);
    const double feature_cosine = std::cos(
        options.feature_angle_degrees * pi / 180.0);
    std::map<Edge, bool> feature_edges;
    std::map<Edge, int> topology_edges;
    DisjointSet triangle_patches(mesh.nt());
    DisjointSet corner_patches(3 * mesh.nt());
    for (const auto& entry : edge_triangles) {
        const Edge& edge = entry.first;
        const std::vector<int>& adjacent = entry.second;
        bool feature = adjacent.size() != 2;
        if (!feature) {
            double cosine = triangle_normals[adjacent[0]].dot(
                triangle_normals[adjacent[1]]);
            cosine = std::max(-1.0, std::min(1.0, cosine));
            feature = cosine < feature_cosine;
        }
        feature_edges[edge] = feature;
        if (feature) {
            result.feature_edges++;
            continue;
        }
        triangle_patches.unite(adjacent[0], adjacent[1]);
        for (int endpoint : {edge.first, edge.second}) {
            const int local_first = triangle_local_vertex(
                mesh, adjacent[0], endpoint);
            const int local_second = triangle_local_vertex(
                mesh, adjacent[1], endpoint);
            corner_patches.unite(
                3 * adjacent[0] + local_first,
                3 * adjacent[1] + local_second);
        }
    }
    for (const auto& entry : edge_triangles) {
        topology_edges.emplace(
            entry.first, (int)topology_edges.size());
    }
    result.topology_edge_count = (int)topology_edges.size();
    std::set<int> patch_roots;
    for (int triangle = 0; triangle < mesh.nt(); triangle++)
        patch_roots.insert(triangle_patches.find(triangle));
    result.smooth_patches = (int)patch_roots.size();

    const bool split_edges =
        options.edge_mode == MullerEdgeMode::SplitFeatureEdges;
    std::map<Edge, int> shared_edge_nodes;
    std::map<int, int> split_corner_nodes;
    std::map<std::tuple<int, int, int>, int> split_edge_nodes;
    if (!split_edges)
        result.nodes = mesh.verts;

    result.elements.reserve(mesh.nt());
    for (int triangle = 0; triangle < mesh.nt(); triangle++) {
        const int v[3] = {
            mesh.tris[3 * triangle],
            mesh.tris[3 * triangle + 1],
            mesh.tris[3 * triangle + 2]
        };
        MullerP2Element element;
        element.topology_vertices = {v[0], v[1], v[2]};
        for (int local = 0; local < 3; local++) {
            if (!split_edges) {
                element.nodes[local] = v[local];
                continue;
            }
            const int root =
                corner_patches.find(3 * triangle + local);
            auto inserted = split_corner_nodes.emplace(
                root, (int)result.nodes.size());
            if (inserted.second)
                result.nodes.push_back(mesh.verts[v[local]]);
            element.nodes[local] = inserted.first->second;
        }
        const int edge_vertices[3][2] = {
            {v[0], v[1]}, {v[1], v[2]}, {v[2], v[0]}
        };
        for (int edge = 0; edge < 3; edge++) {
            const Edge key(
                std::min(edge_vertices[edge][0], edge_vertices[edge][1]),
                std::max(edge_vertices[edge][0], edge_vertices[edge][1]));
            element.topology_edges[edge] = topology_edges.at(key);
            element.edge_orientations[edge] =
                edge_vertices[edge][0] < edge_vertices[edge][1]
                    ? 1 : -1;
            int node = -1;
            if (!split_edges) {
                auto inserted = shared_edge_nodes.emplace(
                    key, (int)result.nodes.size());
                node = inserted.first->second;
                if (inserted.second) {
                    Vec3 midpoint =
                        (mesh.verts[key.first] +
                         mesh.verts[key.second]) * 0.5;
                    if (options.project_edge_nodes_to_sphere)
                        midpoint = project_to_radius(midpoint, radius);
                    result.nodes.push_back(midpoint);
                }
            } else {
                const int side =
                    feature_edges[key] ? triangle : -1;
                const std::tuple<int, int, int> split_key(
                    key.first, key.second, side);
                auto inserted = split_edge_nodes.emplace(
                    split_key, (int)result.nodes.size());
                node = inserted.first->second;
                if (inserted.second) {
                    Vec3 midpoint =
                        (mesh.verts[key.first] +
                         mesh.verts[key.second]) * 0.5;
                    if (options.project_edge_nodes_to_sphere)
                        midpoint = project_to_radius(midpoint, radius);
                    result.nodes.push_back(midpoint);
                }
            }
            element.nodes[3 + edge] = node;
        }
        result.elements.push_back(element);
    }
    if (split_edges) {
        result.duplicated_corner_nodes =
            (int)split_corner_nodes.size() -
            (int)used_vertices.size();
        result.duplicated_midpoint_nodes =
            (int)split_edge_nodes.size() -
            (int)edge_triangles.size();
    }

    result.normals.assign(result.nodes.size(), Vec3());
    const double local_coords[6][2] = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0},
        {0.5, 0.0}, {0.5, 0.5}, {0.0, 0.5}
    };
    for (int element_index = 0;
         element_index < (int)result.elements.size();
         element_index++) {
        const MullerP2Element& element = result.elements[element_index];
        for (int local_node = 0; local_node < 6; local_node++) {
            std::array<double, 6> shape, dxi, deta;
            muller_p2_shape(
                local_coords[local_node][0],
                local_coords[local_node][1],
                shape, dxi, deta);
            Vec3 a1;
            Vec3 a2;
            for (int local = 0; local < 6; local++) {
                const Vec3& node = result.nodes[element.nodes[local]];
                a1 = a1 + node * dxi[local];
                a2 = a2 + node * deta[local];
            }
            const double g11 = a1.dot(a1);
            const double g22 = a2.dot(a2);
            if (g11 * g22 < 1.0e-28)
                throw std::runtime_error(
                    "degenerate metric in Muller normal construction");
            result.normals[element.nodes[local_node]] =
                result.normals[element.nodes[local_node]] +
                a1.cross(a2) * (1.0 / (g11 * g22));
        }
    }

    result.tangent1.resize(result.nodes.size());
    result.tangent2.resize(result.nodes.size());
    for (int node = 0; node < (int)result.nodes.size(); node++) {
        const Vec3 normal = normalized_checked(
            result.normals[node], "degenerate Muller nodal normal");
        result.normals[node] = normal;
        Vec3 reference =
            std::abs(normal.z) > 0.9 ? Vec3(1.0, 0.0, 0.0)
                                     : Vec3(0.0, 0.0, 1.0);
        if (options.azimuthal_tangent_frame) {
            reference = Vec3(0.0, 0.0, 1.0);
            if (reference.cross(normal).norm() < 1.0e-12)
                reference = Vec3(1.0, 0.0, 0.0);
        }
        result.tangent1[node] = normalized_checked(
            reference.cross(normal), "degenerate Muller tangent frame");
        result.tangent2[node] =
            normal.cross(result.tangent1[node]);
    }
    result.current_dof_points.resize(result.current_dofs());
    if (result.basis_kind == MullerBasisKind::NodalP2) {
        for (int node = 0; node < result.scalar_nodes(); node++) {
            result.current_dof_points[2 * node] = result.nodes[node];
            result.current_dof_points[2 * node + 1] = result.nodes[node];
        }
    } else {
        for (const auto& entry : topology_edges) {
            const Vec3 midpoint =
                (mesh.verts[entry.first.first] +
                 mesh.verts[entry.first.second]) * 0.5;
            result.current_dof_points[2 * entry.second] = midpoint;
            result.current_dof_points[2 * entry.second + 1] = midpoint;
        }
    }
    return result;
}

MullerP2Mesh build_muller_p2_mesh(
    const Mesh& mesh, bool project_edge_nodes_to_sphere)
{
    MullerP2BuildOptions options;
    options.project_edge_nodes_to_sphere =
        project_edge_nodes_to_sphere;
    return build_muller_p2_mesh(mesh, options);
}

MullerTensor3 muller_hessian_difference(
    cdouble k_exterior,
    cdouble k_interior,
    const Vec3& displacement,
    double taylor_switch)
{
    const double radius = displacement.norm();
    if (radius <= 0.0)
        throw std::runtime_error(
            "Muller Hessian difference is undefined at zero distance");
    const double unit[3] = {
        displacement.x / radius,
        displacement.y / radius,
        displacement.z / radius
    };
    MullerTensor3 result;
    result.fill(cdouble(0.0));

    const double scaled_radius =
        std::max(std::abs(k_exterior), std::abs(k_interior)) * radius;
    if (scaled_radius < taylor_switch) {
        const cdouble imaginary(0.0, 1.0);
        for (int n = 2; n <= 6; n++) {
            const cdouble coefficient =
                std::pow(imaginary, n) *
                (std::pow(k_exterior, n) - std::pow(k_interior, n)) /
                factorial(n) *
                std::pow(radius, n - 3) * INV4PI;
            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 3; col++) {
                    const double identity = row == col ? 1.0 : 0.0;
                    result[3 * row + col] += coefficient *
                        ((n - 1.0) * identity +
                         (n - 3.0) * unit[row] * unit[col]);
                }
            }
        }
        return result;
    }

    const cdouble imaginary(0.0, 1.0);
    auto f = [&](cdouble k) {
        return std::exp(imaginary * k * radius) * INV4PI /
            std::pow(radius, 3) *
            (3.0 - 3.0 * imaginary * k * radius -
             k * k * radius * radius);
    };
    auto g = [&](cdouble k) {
        return std::exp(imaginary * k * radius) * INV4PI /
            std::pow(radius, 3) *
            (1.0 - imaginary * k * radius);
    };
    const cdouble f_difference = f(k_exterior) - f(k_interior);
    const cdouble g_difference = g(k_exterior) - g(k_interior);
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            const double identity = row == col ? 1.0 : 0.0;
            result[3 * row + col] =
                f_difference * unit[row] * unit[col] -
                g_difference * identity;
        }
    }
    return result;
}

std::array<cdouble, 3> muller_composite_gradient(
    cdouble chi_exterior,
    cdouble chi_interior,
    cdouble k_exterior,
    cdouble k_interior,
    const Vec3& displacement,
    double taylor_switch)
{
    const double radius = displacement.norm();
    if (radius <= 0.0)
        throw std::runtime_error(
            "Muller composite gradient is undefined at zero distance");
    const double unit[3] = {
        displacement.x / radius,
        displacement.y / radius,
        displacement.z / radius
    };
    const double scaled_radius =
        std::max(std::abs(k_exterior), std::abs(k_interior)) * radius;
    cdouble radial(0.0);
    const cdouble imaginary(0.0, 1.0);
    if (scaled_radius < taylor_switch) {
        for (int n = 0; n <= 6; n++) {
            if (n == 1)
                continue;
            const cdouble material_coefficient =
                chi_exterior * std::pow(k_exterior, n) -
                chi_interior * std::pow(k_interior, n);
            radial += (n - 1.0) * std::pow(imaginary, n) *
                material_coefficient / factorial(n) *
                std::pow(radius, n - 2) * INV4PI;
        }
    } else {
        auto gradient_radial = [&](cdouble k) {
            const cdouble green =
                std::exp(imaginary * k * radius) * INV4PI / radius;
            return green * (imaginary * k - 1.0 / radius);
        };
        radial =
            chi_exterior * gradient_radial(k_exterior) -
            chi_interior * gradient_radial(k_interior);
    }
    return {
        radial * unit[0],
        radial * unit[1],
        radial * unit[2]
    };
}

cdouble muller_k1_kernel(
    cdouble k_exterior,
    cdouble k_interior,
    const Vec3& displacement,
    const Vec3& rotated_test_tangent,
    const Vec3& source_tangent,
    double taylor_switch)
{
    const double radius = displacement.norm();
    if (radius <= 0.0)
        throw std::runtime_error("Muller K1 kernel at zero distance");
    const cdouble imaginary(0.0, 1.0);
    const cdouble phi_exterior =
        std::exp(imaginary * k_exterior * radius) * INV4PI / radius;
    const cdouble phi_interior =
        std::exp(imaginary * k_interior * radius) * INV4PI / radius;
    cdouble result =
        (k_exterior * k_exterior * phi_exterior -
         k_interior * k_interior * phi_interior) *
        rotated_test_tangent.dot(source_tangent);

    const MullerTensor3 hessian = muller_hessian_difference(
        k_exterior, k_interior, displacement, taylor_switch);
    const double source[3] = {
        source_tangent.x, source_tangent.y, source_tangent.z
    };
    const double test[3] = {
        rotated_test_tangent.x,
        rotated_test_tangent.y,
        rotated_test_tangent.z
    };
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            result += test[row] * hessian[3 * row + col] * source[col];
    return result;
}

cdouble muller_k2_kernel(
    cdouble chi_exterior,
    cdouble chi_interior,
    cdouble k_exterior,
    cdouble k_interior,
    const Vec3& displacement,
    const Vec3& test_tangent,
    const Vec3& observation_normal,
    const Vec3& source_tangent,
    double taylor_switch)
{
    const std::array<cdouble, 3> gradient =
        muller_composite_gradient(
            chi_exterior, chi_interior,
            k_exterior, k_interior,
            displacement, taylor_switch);
    const double test[3] = {
        test_tangent.x, test_tangent.y, test_tangent.z
    };
    const double normal[3] = {
        observation_normal.x,
        observation_normal.y,
        observation_normal.z
    };
    cdouble test_dot_gradient(0.0);
    cdouble normal_dot_gradient(0.0);
    for (int axis = 0; axis < 3; axis++) {
        test_dot_gradient += test[axis] * gradient[axis];
        normal_dot_gradient += normal[axis] * gradient[axis];
    }
    return test_tangent.dot(source_tangent) * normal_dot_gradient -
           observation_normal.dot(source_tangent) * test_dot_gradient;
}
