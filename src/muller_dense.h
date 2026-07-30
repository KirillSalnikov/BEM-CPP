#ifndef BEM_MULLER_DENSE_H
#define BEM_MULLER_DENSE_H

#include "muller_nodal.h"
#include <complex>
#include <vector>

struct MullerDenseAssemblyStats {
    long long regular_element_pairs = 0;
    long long coincident_element_pairs = 0;
    long long edge_adjacent_element_pairs = 0;
    long long vertex_adjacent_element_pairs = 0;
    long long quadrature_pairs = 0;
};

struct MullerDenseSystem {
    MullerP2Mesh mesh;
    int current_dofs = 0;
    int system_dofs = 0;
    std::vector<std::complex<double>> matrix;
    MullerDenseAssemblyStats stats;
};

MullerDenseSystem assemble_muller_nodal_dense(
    const Mesh& linear_mesh,
    std::complex<double> k_exterior,
    std::complex<double> refractive_index,
    bool project_edge_nodes_to_sphere,
    int regular_quadrature_order = 7,
    int duffy_order = 4);

MullerDenseSystem assemble_muller_nodal_dense(
    const Mesh& linear_mesh,
    std::complex<double> k_exterior,
    std::complex<double> refractive_index,
    const MullerP2BuildOptions& build_options,
    int regular_quadrature_order = 7,
    int duffy_order = 4);

std::vector<std::complex<double>> muller_nodal_planewave_rhs(
    const MullerP2Mesh& mesh,
    std::complex<double> k_exterior,
    const Vec3& electric_field,
    const Vec3& propagation_direction,
    int quadrature_order = 13);

void muller_nodal_farfield(
    const MullerP2Mesh& mesh,
    const std::complex<double>* electric_current,
    const std::complex<double>* magnetic_current,
    std::complex<double> k_exterior,
    const std::vector<Vec3>& directions,
    std::vector<std::complex<double>>& field,
    int quadrature_order = 13);

#endif
