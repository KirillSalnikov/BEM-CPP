#ifndef BEM_MULLER_NODAL_H
#define BEM_MULLER_NODAL_H

#include "mesh.h"
#include <array>
#include <complex>
#include <vector>

struct MullerP2Element {
    std::array<int, 6> nodes;
    // Original linear-mesh vertices. These remain shared across feature
    // edges even when the P2 degrees of freedom are split per smooth patch.
    std::array<int, 3> topology_vertices;
    // Global topological edges for the local directed edges
    // (v0,v1), (v1,v2), and (v2,v0).
    std::array<int, 3> topology_edges;
    // Orientation of each local directed edge relative to the global
    // low-vertex-to-high-vertex direction.
    std::array<int, 3> edge_orientations;
};

enum class MullerEdgeMode {
    Smooth,
    SplitFeatureEdges,
    HDivBdm1
};

enum class MullerBasisKind {
    NodalP2,
    HDivBdm1
};

struct MullerP2BuildOptions {
    bool project_edge_nodes_to_sphere = false;
    bool azimuthal_tangent_frame = false;
    MullerEdgeMode edge_mode = MullerEdgeMode::Smooth;
    double feature_angle_degrees = 45.0;
};

struct MullerFrameSample {
    Vec3 position;
    Vec3 normal;
    Vec3 tangent1;
    Vec3 tangent2;
    Vec3 derivative_xi;
    Vec3 derivative_eta;
    double reference_xi = 0.0;
    double reference_eta = 0.0;
    double jacobian = 0.0;
    std::array<double, 6> shape;
};

struct MullerBasisSample {
    int count = 0;
    std::array<int, 12> dofs;
    std::array<Vec3, 12> values;
};

struct MullerP2Mesh {
    std::vector<Vec3> nodes;
    std::vector<MullerP2Element> elements;
    std::vector<Vec3> normals;
    std::vector<Vec3> tangent1;
    std::vector<Vec3> tangent2;
    std::vector<Vec3> current_dof_points;
    MullerEdgeMode edge_mode = MullerEdgeMode::Smooth;
    MullerBasisKind basis_kind = MullerBasisKind::NodalP2;
    double feature_angle_degrees = 45.0;
    int feature_edges = 0;
    int topology_edge_count = 0;
    int smooth_patches = 1;
    int duplicated_corner_nodes = 0;
    int duplicated_midpoint_nodes = 0;

    int scalar_nodes() const { return (int)nodes.size(); }
    int current_dofs() const {
        return basis_kind == MullerBasisKind::HDivBdm1
            ? 2 * topology_edge_count : 2 * scalar_nodes();
    }
    int system_dofs() const { return 2 * current_dofs(); }
};

MullerP2Mesh build_muller_p2_mesh(
    const Mesh& mesh, bool project_edge_nodes_to_sphere = false);

MullerP2Mesh build_muller_p2_mesh(
    const Mesh& mesh, const MullerP2BuildOptions& options);

void muller_p2_shape(
    double xi, double eta,
    std::array<double, 6>& shape,
    std::array<double, 6>& dshape_dxi,
    std::array<double, 6>& dshape_deta);

MullerFrameSample evaluate_muller_frame(
    const MullerP2Mesh& mesh, int element, double xi, double eta);

MullerBasisSample evaluate_muller_basis(
    const MullerP2Mesh& mesh,
    int element,
    const MullerFrameSample& frame);

using MullerTensor3 = std::array<std::complex<double>, 9>;

MullerTensor3 muller_hessian_difference(
    std::complex<double> k_exterior,
    std::complex<double> k_interior,
    const Vec3& displacement,
    double taylor_switch = 1.0e-2);

std::array<std::complex<double>, 3> muller_composite_gradient(
    std::complex<double> chi_exterior,
    std::complex<double> chi_interior,
    std::complex<double> k_exterior,
    std::complex<double> k_interior,
    const Vec3& displacement,
    double taylor_switch = 1.0e-2);

std::complex<double> muller_k1_kernel(
    std::complex<double> k_exterior,
    std::complex<double> k_interior,
    const Vec3& displacement,
    const Vec3& rotated_test_tangent,
    const Vec3& source_tangent,
    double taylor_switch = 1.0e-2);

std::complex<double> muller_k2_kernel(
    std::complex<double> chi_exterior,
    std::complex<double> chi_interior,
    std::complex<double> k_exterior,
    std::complex<double> k_interior,
    const Vec3& displacement,
    const Vec3& test_tangent,
    const Vec3& observation_normal,
    const Vec3& source_tangent,
    double taylor_switch = 1.0e-2);

#endif
