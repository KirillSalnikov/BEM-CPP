#include "muller_dense.h"

#include "muller_duffy.h"
#include "quadrature.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

using cdouble = std::complex<double>;

struct ElementAdjacency {
    int shared_count = 0;
    int test_local[3] = {-1, -1, -1};
    int trial_local[3] = {-1, -1, -1};
};

ElementAdjacency classify_elements(
    const MullerP2Element& test,
    const MullerP2Element& trial)
{
    ElementAdjacency result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (test.topology_vertices[i] ==
                trial.topology_vertices[j]) {
                const int slot = result.shared_count++;
                result.test_local[slot] = i;
                result.trial_local[slot] = j;
            }
        }
    }
    return result;
}

void add_operator_sample(
    const MullerP2Mesh& mesh,
    int test_element,
    int trial_element,
    const MullerFrameSample& test,
    const MullerFrameSample& trial,
    cdouble k_exterior,
    cdouble k_interior,
    cdouble epsilon_exterior,
    cdouble epsilon_interior,
    cdouble mu_exterior,
    cdouble mu_interior,
    double weight,
    std::vector<cdouble>& k1_matrix,
    std::vector<cdouble>& k2_epsilon_matrix,
    std::vector<cdouble>& k2_mu_matrix)
{
    const int current_dofs = mesh.current_dofs();
    const Vec3 displacement = test.position - trial.position;
    const double physical_weight =
        weight * test.jacobian * trial.jacobian;
    const MullerBasisSample test_basis =
        evaluate_muller_basis(mesh, test_element, test);
    const MullerBasisSample trial_basis =
        evaluate_muller_basis(mesh, trial_element, trial);

    for (int alpha = 0; alpha < test_basis.count; alpha++) {
        const Vec3& test_tangent = test_basis.values[alpha];
        const Vec3 rotated_test = test_tangent.cross(test.normal);
        for (int beta = 0; beta < trial_basis.count; beta++) {
            const Vec3& source_tangent = trial_basis.values[beta];
            const cdouble k1 = muller_k1_kernel(
                k_exterior, k_interior, displacement,
                rotated_test, source_tangent);
            const cdouble k2_epsilon = muller_k2_kernel(
                epsilon_exterior, epsilon_interior,
                k_exterior, k_interior, displacement,
                test_tangent, test.normal, source_tangent);
            const cdouble k2_mu = muller_k2_kernel(
                mu_exterior, mu_interior,
                k_exterior, k_interior, displacement,
                test_tangent, test.normal, source_tangent);

            const int row = test_basis.dofs[alpha];
            const int col = trial_basis.dofs[beta];
            const size_t index =
                (size_t)row * current_dofs + col;
            k1_matrix[index] += physical_weight * k1;
            k2_epsilon_matrix[index] +=
                physical_weight * k2_epsilon;
            k2_mu_matrix[index] +=
                physical_weight * k2_mu;
        }
    }
}

void assemble_mass(
    const MullerP2Mesh& mesh,
    int quadrature_order,
    std::vector<cdouble>& mass)
{
    const TriQuad quadrature =
        tri_quadrature(quadrature_order);
    const int current_dofs = mesh.current_dofs();
    for (int element_index = 0;
         element_index < (int)mesh.elements.size();
         element_index++) {
        for (int q = 0; q < quadrature.npts; q++) {
            const MullerFrameSample sample =
                evaluate_muller_frame(
                    mesh, element_index,
                    quadrature.pts[q][0],
                    quadrature.pts[q][1]);
            const double weight =
                0.5 * quadrature.wts[q] * sample.jacobian;
            const MullerBasisSample basis =
                evaluate_muller_basis(mesh, element_index, sample);
            for (int i = 0; i < basis.count; i++) {
                const int row = basis.dofs[i];
                for (int j = 0; j < basis.count; j++) {
                    const int col = basis.dofs[j];
                    mass[(size_t)row * current_dofs + col] +=
                        weight * basis.values[i].dot(
                            basis.values[j]);
                }
            }
        }
    }
}

void remap_singular_point(
    MullerDuffyPoint& point,
    const ElementAdjacency& adjacency)
{
    if (adjacency.shared_count == 1) {
        muller_duffy_remap_shared_vertex(
            point.test_xi, point.test_eta,
            adjacency.test_local[0]);
        muller_duffy_remap_shared_vertex(
            point.trial_xi, point.trial_eta,
            adjacency.trial_local[0]);
    } else if (adjacency.shared_count == 2) {
        int test_first = adjacency.test_local[0];
        int test_second = adjacency.test_local[1];
        int trial_first = adjacency.trial_local[0];
        int trial_second = adjacency.trial_local[1];
        muller_duffy_remap_shared_edge(
            point.test_xi, point.test_eta,
            test_first, test_second);
        muller_duffy_remap_shared_edge(
            point.trial_xi, point.trial_eta,
            trial_first, trial_second);
    }
}

std::array<cdouble, 3> current_at_sample(
    const MullerP2Mesh& mesh,
    int element,
    const MullerFrameSample& sample,
    const cdouble* coefficients)
{
    std::array<cdouble, 3> current = {
        cdouble(0.0), cdouble(0.0), cdouble(0.0)
    };
    const MullerBasisSample basis =
        evaluate_muller_basis(mesh, element, sample);
    for (int local = 0; local < basis.count; local++) {
        const cdouble coefficient =
            coefficients[basis.dofs[local]];
        const Vec3& tangent = basis.values[local];
        current[0] += coefficient * tangent.x;
        current[1] += coefficient * tangent.y;
        current[2] += coefficient * tangent.z;
    }
    return current;
}

} // namespace

MullerDenseSystem assemble_muller_nodal_dense(
    const Mesh& linear_mesh,
    cdouble k_exterior,
    cdouble refractive_index,
    bool project_edge_nodes_to_sphere,
    int regular_quadrature_order,
    int duffy_order)
{
    MullerP2BuildOptions options;
    options.project_edge_nodes_to_sphere =
        project_edge_nodes_to_sphere;
    return assemble_muller_nodal_dense(
        linear_mesh, k_exterior, refractive_index, options,
        regular_quadrature_order, duffy_order);
}

MullerDenseSystem assemble_muller_nodal_dense(
    const Mesh& linear_mesh,
    cdouble k_exterior,
    cdouble refractive_index,
    const MullerP2BuildOptions& build_options,
    int regular_quadrature_order,
    int duffy_order)
{
    MullerDenseSystem result;
    result.mesh = build_muller_p2_mesh(
        linear_mesh, build_options);
    result.current_dofs = result.mesh.current_dofs();
    result.system_dofs = result.mesh.system_dofs();
    if (result.system_dofs > 5000)
        throw std::runtime_error(
            "dense Muller validation path is limited to 5000 DOFs");

    const int n = result.current_dofs;
    const size_t operator_size = (size_t)n * n;
    std::vector<cdouble> mass(operator_size, cdouble(0.0));
    std::vector<cdouble> k1(operator_size, cdouble(0.0));
    std::vector<cdouble> k2_epsilon(
        operator_size, cdouble(0.0));
    std::vector<cdouble> k2_mu(
        operator_size, cdouble(0.0));

    const cdouble epsilon_exterior(1.0, 0.0);
    const cdouble epsilon_interior =
        refractive_index * refractive_index;
    const cdouble mu_exterior(1.0, 0.0);
    const cdouble mu_interior(1.0, 0.0);
    const cdouble k_interior =
        k_exterior * refractive_index;

    assemble_mass(result.mesh, 13, mass);
    const TriQuad regular =
        tri_quadrature(regular_quadrature_order);
    const std::vector<MullerDuffyPoint> coincident_rule =
        muller_duffy_rule(
            duffy_order, MullerDuffyAdjacency::Coincident);
    const std::vector<MullerDuffyPoint> edge_rule =
        muller_duffy_rule(
            duffy_order, MullerDuffyAdjacency::EdgeAdjacent);
    const std::vector<MullerDuffyPoint> vertex_rule =
        muller_duffy_rule(
            duffy_order, MullerDuffyAdjacency::VertexAdjacent);

    const int element_count =
        (int)result.mesh.elements.size();
    for (int test_element_index = 0;
         test_element_index < element_count;
         test_element_index++) {
        const MullerP2Element& test_element =
            result.mesh.elements[test_element_index];
        for (int trial_element_index = 0;
             trial_element_index < element_count;
             trial_element_index++) {
            const MullerP2Element& trial_element =
                result.mesh.elements[trial_element_index];
            const ElementAdjacency adjacency =
                classify_elements(test_element, trial_element);

            const std::vector<MullerDuffyPoint>* singular_rule =
                nullptr;
            if (test_element_index == trial_element_index) {
                singular_rule = &coincident_rule;
                result.stats.coincident_element_pairs++;
            } else if (adjacency.shared_count == 2) {
                singular_rule = &edge_rule;
                result.stats.edge_adjacent_element_pairs++;
            } else if (adjacency.shared_count == 1) {
                singular_rule = &vertex_rule;
                result.stats.vertex_adjacent_element_pairs++;
            } else {
                result.stats.regular_element_pairs++;
            }

            if (singular_rule) {
                for (MullerDuffyPoint point : *singular_rule) {
                    remap_singular_point(point, adjacency);
                    const MullerFrameSample test =
                        evaluate_muller_frame(
                            result.mesh, test_element_index,
                            point.test_xi, point.test_eta);
                    const MullerFrameSample trial =
                        evaluate_muller_frame(
                            result.mesh, trial_element_index,
                            point.trial_xi, point.trial_eta);
                    add_operator_sample(
                        result.mesh,
                        test_element_index, trial_element_index,
                        test, trial,
                        k_exterior, k_interior,
                        epsilon_exterior, epsilon_interior,
                        mu_exterior, mu_interior,
                        point.weight,
                        k1, k2_epsilon, k2_mu);
                    result.stats.quadrature_pairs++;
                }
            } else {
                for (int qx = 0; qx < regular.npts; qx++) {
                    const MullerFrameSample test =
                        evaluate_muller_frame(
                            result.mesh, test_element_index,
                            regular.pts[qx][0],
                            regular.pts[qx][1]);
                    for (int qy = 0; qy < regular.npts; qy++) {
                        const MullerFrameSample trial =
                            evaluate_muller_frame(
                                result.mesh, trial_element_index,
                                regular.pts[qy][0],
                                regular.pts[qy][1]);
                        const double weight =
                            0.25 * regular.wts[qx] *
                            regular.wts[qy];
                        add_operator_sample(
                            result.mesh,
                            test_element_index, trial_element_index,
                            test, trial,
                            k_exterior, k_interior,
                            epsilon_exterior, epsilon_interior,
                            mu_exterior, mu_interior,
                            weight,
                            k1, k2_epsilon, k2_mu);
                        result.stats.quadrature_pairs++;
                    }
                }
            }
        }
    }

    const int system_dofs = result.system_dofs;
    result.matrix.assign(
        (size_t)system_dofs * system_dofs, cdouble(0.0));
    const cdouble imaginary(0.0, 1.0);
    const cdouble omega = k_exterior;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            const size_t source = (size_t)row * n + col;
            result.matrix[(size_t)row * system_dofs + col] =
                imaginary / omega * k1[source];
            result.matrix[
                (size_t)row * system_dofs + n + col] =
                0.5 * (epsilon_interior + epsilon_exterior) *
                    mass[source] +
                k2_epsilon[source];
            result.matrix[
                (size_t)(n + row) * system_dofs + col] =
                0.5 * (mu_interior + mu_exterior) *
                    mass[source] +
                k2_mu[source];
            result.matrix[
                (size_t)(n + row) * system_dofs + n + col] =
                -imaginary / omega * k1[source];
        }
    }
    return result;
}

std::vector<cdouble> muller_nodal_planewave_rhs(
    const MullerP2Mesh& mesh,
    cdouble k_exterior,
    const Vec3& electric_field,
    const Vec3& propagation_direction,
    int quadrature_order)
{
    const int current_dofs = mesh.current_dofs();
    std::vector<cdouble> rhs(
        2 * current_dofs, cdouble(0.0));
    const Vec3 magnetic_field =
        propagation_direction.cross(electric_field);
    const TriQuad quadrature =
        tri_quadrature(quadrature_order);
    const cdouble imaginary(0.0, 1.0);
    for (int element_index = 0;
         element_index < (int)mesh.elements.size();
         element_index++) {
        for (int q = 0; q < quadrature.npts; q++) {
            const MullerFrameSample sample =
                evaluate_muller_frame(
                    mesh, element_index,
                    quadrature.pts[q][0],
                    quadrature.pts[q][1]);
            const cdouble phase = std::exp(
                imaginary * k_exterior *
                propagation_direction.dot(sample.position));
            const double weight =
                0.5 * quadrature.wts[q] * sample.jacobian;
            const MullerBasisSample basis =
                evaluate_muller_basis(mesh, element_index, sample);
            for (int alpha = 0; alpha < basis.count; alpha++) {
                const Vec3& tangent = basis.values[alpha];
                const Vec3 rotated =
                    tangent.cross(sample.normal);
                const cdouble b_e =
                    -rotated.dot(electric_field) * phase;
                const cdouble b_h =
                    rotated.dot(magnetic_field) * phase;
                const int dof = basis.dofs[alpha];
                rhs[dof] += weight * b_e;
                rhs[current_dofs + dof] += weight * b_h;
            }
        }
    }
    return rhs;
}

void muller_nodal_farfield(
    const MullerP2Mesh& mesh,
    const cdouble* electric_current,
    const cdouble* magnetic_current,
    cdouble k_exterior,
    const std::vector<Vec3>& directions,
    std::vector<cdouble>& field,
    int quadrature_order)
{
    field.assign(
        (size_t)directions.size() * 3, cdouble(0.0));
    std::vector<cdouble> transformed_j(
        (size_t)directions.size() * 3, cdouble(0.0));
    std::vector<cdouble> transformed_m(
        (size_t)directions.size() * 3, cdouble(0.0));
    const TriQuad quadrature =
        tri_quadrature(quadrature_order);
    const cdouble imaginary(0.0, 1.0);

    for (int element_index = 0;
         element_index < (int)mesh.elements.size();
         element_index++) {
        for (int q = 0; q < quadrature.npts; q++) {
            const MullerFrameSample sample =
                evaluate_muller_frame(
                    mesh, element_index,
                    quadrature.pts[q][0],
                    quadrature.pts[q][1]);
            const double weight =
                0.5 * quadrature.wts[q] * sample.jacobian;
            const std::array<cdouble, 3> current_j =
                current_at_sample(
                    mesh, element_index, sample, electric_current);
            const std::array<cdouble, 3> current_m =
                current_at_sample(
                    mesh, element_index, sample, magnetic_current);
            for (int direction = 0;
                 direction < (int)directions.size();
                 direction++) {
                const cdouble phase = std::exp(
                    -imaginary * k_exterior *
                    directions[direction].dot(sample.position));
                const cdouble weighted_phase = weight * phase;
                for (int axis = 0; axis < 3; axis++) {
                    transformed_j[3 * direction + axis] +=
                        weighted_phase * current_j[axis];
                    transformed_m[3 * direction + axis] +=
                        weighted_phase * current_m[axis];
                }
            }
        }
    }

    const cdouble prefactor =
        -imaginary * k_exterior * INV4PI;
    for (int direction = 0;
         direction < (int)directions.size();
         direction++) {
        const Vec3& r = directions[direction];
        const cdouble j[3] = {
            transformed_j[3 * direction],
            transformed_j[3 * direction + 1],
            transformed_j[3 * direction + 2]
        };
        const cdouble m[3] = {
            transformed_m[3 * direction],
            transformed_m[3 * direction + 1],
            transformed_m[3 * direction + 2]
        };
        const cdouble r_dot_j =
            r.x * j[0] + r.y * j[1] + r.z * j[2];
        const cdouble j_perpendicular[3] = {
            j[0] - r.x * r_dot_j,
            j[1] - r.y * r_dot_j,
            j[2] - r.z * r_dot_j
        };
        const cdouble r_cross_m[3] = {
            r.y * m[2] - r.z * m[1],
            r.z * m[0] - r.x * m[2],
            r.x * m[1] - r.y * m[0]
        };
        for (int axis = 0; axis < 3; axis++) {
            field[3 * direction + axis] =
                prefactor *
                (j_perpendicular[axis] - r_cross_m[axis]);
        }
    }
}
