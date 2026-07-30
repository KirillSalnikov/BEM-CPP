#include "muller_dense.h"
#include "muller_fmm.h"
#include "muller_mbj.h"
#include "orient.h"
#include "solver_policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>

namespace {

using cdouble = std::complex<double>;
using Matvec = std::function<void(const cdouble*, cdouble*)>;
using FlexiblePreconditioner =
    std::function<void(const cdouble*, cdouble*)>;

std::ofstream iteration_log;
int iteration_log_every = 1;

void log_iteration(
    const char* label,
    const char* event,
    int iteration,
    double projected_residual,
    double operator_residual,
    double matvec_seconds,
    double preconditioner_seconds,
    double orthogonalization_seconds,
    double elapsed_seconds)
{
    if (!iteration_log ||
        (std::strcmp(event, "iteration") == 0 &&
         iteration % iteration_log_every != 0)) {
        return;
    }
    iteration_log
        << std::setprecision(12)
        << label << ',' << event << ',' << iteration << ','
        << projected_residual << ',' << operator_residual << ','
        << matvec_seconds << ',' << preconditioner_seconds << ','
        << orthogonalization_seconds << ',' << elapsed_seconds
        << '\n';
    iteration_log.flush();
}

struct SymmetryNodeKey {
    std::array<long long, 6> value;

    bool operator==(const SymmetryNodeKey& other) const
    {
        return value == other.value;
    }
};

struct SymmetryNodeKeyHash {
    size_t operator()(const SymmetryNodeKey& key) const
    {
        size_t result = 1469598103934665603ULL;
        for (long long value : key.value) {
            result ^= std::hash<long long>()(value);
            result *= 1099511628211ULL;
        }
        return result;
    }
};

struct ElementNodeSetHash {
    size_t operator()(const std::array<int, 6>& nodes) const
    {
        size_t result = 1469598103934665603ULL;
        for (int node : nodes) {
            result ^= std::hash<int>()(node);
            result *= 1099511628211ULL;
        }
        return result;
    }
};

struct GmresResult {
    int iterations = 0;
    int resumed_iterations = 0;
    double initial_operator_residual = 1.0;
    double projected_residual = 1.0;
    double operator_residual = 1.0;
    double seconds = 0.0;
    double recycle_seconds = 0.0;
    std::vector<cdouble> solution;
};

struct SolverCheckpointOptions {
    std::string path;
    std::uint64_t signature = 0;
    bool allow_signature_mismatch = false;
};

const std::uint64_t FNV_OFFSET = 1469598103934665603ULL;
const std::uint64_t FNV_PRIME = 1099511628211ULL;

void hash_bytes(
    std::uint64_t& hash, const void* data, std::size_t size)
{
    const unsigned char* bytes =
        static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }
}

std::uint64_t vector_hash(const cdouble* values, int count)
{
    std::uint64_t hash = FNV_OFFSET;
    hash_bytes(
        hash, values,
        static_cast<std::size_t>(count) * sizeof(cdouble));
    return hash;
}

std::string checkpoint_stage_path(
    const std::string& base, const char* label)
{
    if (base.empty())
        return std::string();
    std::string suffix;
    for (const char* p = label; *p; p++) {
        const unsigned char value =
            static_cast<unsigned char>(*p);
        suffix.push_back(
            (value >= 'a' && value <= 'z') ||
                    (value >= 'A' && value <= 'Z') ||
                    (value >= '0' && value <= '9')
                ? static_cast<char>(value)
                : '_');
    }
    return base + "." + suffix + ".bin";
}

bool load_solver_checkpoint(
    const SolverCheckpointOptions& options,
    std::uint64_t rhs_hash,
    int expected_size,
    int& iterations,
    double& residual,
    std::vector<cdouble>& solution)
{
    if (options.path.empty())
        return false;
    std::ifstream input(options.path.c_str(), std::ios::binary);
    if (!input)
        return false;

    char magic[16] = {};
    std::uint64_t version = 0;
    std::uint64_t signature = 0;
    std::uint64_t stored_rhs_hash = 0;
    std::uint64_t size = 0;
    std::uint64_t stored_iterations = 0;
    input.read(magic, sizeof(magic));
    input.read(
        reinterpret_cast<char*>(&version), sizeof(version));
    input.read(
        reinterpret_cast<char*>(&signature), sizeof(signature));
    input.read(
        reinterpret_cast<char*>(&stored_rhs_hash),
        sizeof(stored_rhs_hash));
    input.read(reinterpret_cast<char*>(&size), sizeof(size));
    input.read(
        reinterpret_cast<char*>(&stored_iterations),
        sizeof(stored_iterations));
    input.read(
        reinterpret_cast<char*>(&residual), sizeof(residual));
    const char expected_magic[16] = "BEM_FGMRES_CP1";
    const bool signature_matches =
        signature == options.signature;
    if (!input ||
        std::memcmp(magic, expected_magic, sizeof(magic)) != 0 ||
        version != 1 ||
        (!signature_matches &&
         !options.allow_signature_mismatch) ||
        stored_rhs_hash != rhs_hash ||
        size != static_cast<std::uint64_t>(expected_size) ||
        stored_iterations >
            static_cast<std::uint64_t>(expected_size)) {
        std::fprintf(
            stderr,
            "  [checkpoint] incompatible checkpoint ignored: %s\n",
            options.path.c_str());
        return false;
    }
    if (!signature_matches) {
        std::fprintf(
            stderr,
            "  [checkpoint] explicitly migrating operator signature: "
            "%s\n",
            options.path.c_str());
    }

    solution.resize(expected_size);
    input.read(
        reinterpret_cast<char*>(solution.data()),
        static_cast<std::streamsize>(
            static_cast<std::size_t>(expected_size) *
            sizeof(cdouble)));
    if (!input) {
        std::fprintf(
            stderr,
            "  [checkpoint] truncated checkpoint ignored: %s\n",
            options.path.c_str());
        solution.clear();
        return false;
    }
    iterations = static_cast<int>(stored_iterations);
    return true;
}

bool load_checkpoint_solution(
    const char* path, std::vector<cdouble>& solution)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return false;
    char magic[16] = {};
    std::uint64_t version = 0;
    std::uint64_t signature = 0;
    std::uint64_t rhs_hash = 0;
    std::uint64_t size = 0;
    std::uint64_t iterations = 0;
    double residual = 0.0;
    input.read(magic, sizeof(magic));
    input.read(reinterpret_cast<char*>(&version), sizeof(version));
    input.read(reinterpret_cast<char*>(&signature), sizeof(signature));
    input.read(reinterpret_cast<char*>(&rhs_hash), sizeof(rhs_hash));
    input.read(reinterpret_cast<char*>(&size), sizeof(size));
    input.read(reinterpret_cast<char*>(&iterations), sizeof(iterations));
    input.read(reinterpret_cast<char*>(&residual), sizeof(residual));
    const char expected_magic[16] = "BEM_FGMRES_CP1";
    if (!input ||
        std::memcmp(magic, expected_magic, sizeof(magic)) != 0 ||
        version != 1 || size > 1000000000ULL) {
        return false;
    }
    solution.resize(static_cast<std::size_t>(size));
    input.read(
        reinterpret_cast<char*>(solution.data()),
        static_cast<std::streamsize>(
            solution.size() * sizeof(cdouble)));
    return static_cast<bool>(input);
}

void save_solver_checkpoint(
    const SolverCheckpointOptions& options,
    std::uint64_t rhs_hash,
    int iterations,
    double residual,
    const std::vector<cdouble>& solution)
{
    if (options.path.empty())
        return;
    const std::string temporary = options.path + ".tmp";
    std::ofstream output(
        temporary.c_str(),
        std::ios::binary | std::ios::out | std::ios::trunc);
    if (!output)
        throw std::runtime_error(
            "cannot create solver checkpoint " + temporary);

    const char magic[16] = "BEM_FGMRES_CP1";
    const std::uint64_t version = 1;
    const std::uint64_t size =
        static_cast<std::uint64_t>(solution.size());
    const std::uint64_t stored_iterations =
        static_cast<std::uint64_t>(iterations);
    output.write(magic, sizeof(magic));
    output.write(
        reinterpret_cast<const char*>(&version), sizeof(version));
    output.write(
        reinterpret_cast<const char*>(&options.signature),
        sizeof(options.signature));
    output.write(
        reinterpret_cast<const char*>(&rhs_hash), sizeof(rhs_hash));
    output.write(
        reinterpret_cast<const char*>(&size), sizeof(size));
    output.write(
        reinterpret_cast<const char*>(&stored_iterations),
        sizeof(stored_iterations));
    output.write(
        reinterpret_cast<const char*>(&residual), sizeof(residual));
    output.write(
        reinterpret_cast<const char*>(solution.data()),
        static_cast<std::streamsize>(
            solution.size() * sizeof(cdouble)));
    output.flush();
    if (!output)
        throw std::runtime_error(
            "cannot write solver checkpoint " + temporary);
    output.close();
    if (std::rename(
            temporary.c_str(), options.path.c_str()) != 0) {
        std::remove(temporary.c_str());
        throw std::runtime_error(
            "cannot atomically replace solver checkpoint " +
            options.path);
    }
}

void solve_projected_system(
    const std::vector<cdouble>& hessenberg,
    int stride,
    const std::vector<cdouble>& projected,
    int count,
    std::vector<cdouble>& coefficients)
{
    coefficients.assign(count, cdouble(0.0));
    for (int row = count - 1; row >= 0; row--) {
        cdouble value = projected[row];
        for (int column = row + 1; column < count; column++) {
            value -=
                hessenberg[
                    static_cast<std::size_t>(row) * stride +
                    column] *
                coefficients[column];
        }
        coefficients[row] =
            value /
            hessenberg[
                static_cast<std::size_t>(row) * stride + row];
    }
}

Vec3 rotate_about_z(const Vec3& value, double cosine, double sine)
{
    return Vec3(
        cosine * value.x - sine * value.y,
        sine * value.x + cosine * value.y,
        value.z);
}

Vec3 rotate_axis_to_z(const Vec3& value, const Vec3& source_axis)
{
    const Vec3 unit = source_axis.normalized();
    const Vec3 target(0.0, 0.0, 1.0);
    const double cosine = std::max(
        -1.0, std::min(1.0, unit.dot(target)));
    const Vec3 cross = unit.cross(target);
    const double sine = cross.norm();
    if (sine < 1.0e-15) {
        if (cosine > 0.0)
            return value;
        return Vec3(value.x, -value.y, -value.z);
    }
    const Vec3 axis = cross * (1.0 / sine);
    return value * cosine +
        axis.cross(value) * sine +
        axis * (axis.dot(value) * (1.0 - cosine));
}

Vec3 reflect_about_vertical_plane(
    const Vec3& value, double cosine_twice, double sine_twice)
{
    return Vec3(
        cosine_twice * value.x + sine_twice * value.y,
        sine_twice * value.x - cosine_twice * value.y,
        value.z);
}

SymmetryNodeKey symmetry_node_key(
    const Vec3& position, const Vec3& normal)
{
    const double scale = 1.0e9;
    SymmetryNodeKey key;
    key.value = {
        std::llround(scale * position.x),
        std::llround(scale * position.y),
        std::llround(scale * position.z),
        std::llround(scale * normal.x),
        std::llround(scale * normal.y),
        std::llround(scale * normal.z)
    };
    return key;
}

std::vector<int> rotation_source_for_target(
    const MullerP2Mesh& mesh, double angle)
{
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const bool require_matching_normal =
        mesh.basis_kind != MullerBasisKind::HDivBdm1;
    std::unordered_map<
        SymmetryNodeKey, int, SymmetryNodeKeyHash> node_by_key;
    node_by_key.reserve(2 * mesh.nodes.size());
    for (int node = 0; node < mesh.scalar_nodes(); node++) {
        const Vec3 key_normal =
            require_matching_normal ? mesh.normals[node] : Vec3();
        const auto inserted = node_by_key.emplace(
            symmetry_node_key(mesh.nodes[node], key_normal),
            node);
        if (!inserted.second)
            throw std::runtime_error(
                "cyclic symmetry found duplicate position/normal node");
    }

    std::vector<int> source_for_target(mesh.scalar_nodes(), -1);
    for (int source = 0; source < mesh.scalar_nodes(); source++) {
        const Vec3 rotated_position = rotate_about_z(
            mesh.nodes[source], cosine, sine);
        const Vec3 rotated_normal = require_matching_normal
            ? rotate_about_z(mesh.normals[source], cosine, sine)
            : Vec3();
        const auto target = node_by_key.find(
            symmetry_node_key(rotated_position, rotated_normal));
        if (target == node_by_key.end())
            throw std::runtime_error(
                "mesh is not invariant under the requested cyclic rotation");
        if (source_for_target[target->second] != -1)
            throw std::runtime_error(
                "cyclic symmetry node mapping is not one-to-one");
        source_for_target[target->second] = source;
    }
    return source_for_target;
}

std::vector<int> reflection_source_for_target(
    const MullerP2Mesh& mesh, double axis_angle,
    bool require_matching_normal = true)
{
    const double cosine_twice = std::cos(2.0 * axis_angle);
    const double sine_twice = std::sin(2.0 * axis_angle);
    std::unordered_map<
        SymmetryNodeKey, int, SymmetryNodeKeyHash> node_by_key;
    node_by_key.reserve(2 * mesh.nodes.size());
    for (int node = 0; node < mesh.scalar_nodes(); node++) {
        const Vec3 key_normal =
            require_matching_normal ? mesh.normals[node] : Vec3();
        const auto inserted = node_by_key.emplace(
            symmetry_node_key(mesh.nodes[node], key_normal),
            node);
        if (!inserted.second)
            throw std::runtime_error(
                "mirror symmetry found duplicate position/normal node");
    }

    std::vector<int> source_for_target(mesh.scalar_nodes(), -1);
    for (int source = 0; source < mesh.scalar_nodes(); source++) {
        const Vec3 reflected_position = reflect_about_vertical_plane(
            mesh.nodes[source], cosine_twice, sine_twice);
        const Vec3 reflected_normal = require_matching_normal
            ? reflect_about_vertical_plane(
                  mesh.normals[source], cosine_twice, sine_twice)
            : Vec3();
        const auto target = node_by_key.find(
            symmetry_node_key(reflected_position, reflected_normal));
        if (target == node_by_key.end())
            throw std::runtime_error(
                "mesh is not invariant under the requested reflection");
        if (source_for_target[target->second] != -1)
            throw std::runtime_error(
                "mirror symmetry node mapping is not one-to-one");
        source_for_target[target->second] = source;
    }
    return source_for_target;
}

double symmetry_element_match_fraction(
    const MullerP2Mesh& mesh,
    const std::vector<int>& source_for_target)
{
    std::vector<int> target_for_source(
        mesh.scalar_nodes(), -1);
    for (int target = 0; target < mesh.scalar_nodes(); target++)
        target_for_source[source_for_target[target]] = target;
    std::unordered_map<
        std::array<int, 6>, int, ElementNodeSetHash> element_sets;
    element_sets.reserve(2 * mesh.elements.size());
    for (int element = 0;
         element < static_cast<int>(mesh.elements.size());
         element++) {
        std::array<int, 6> nodes = mesh.elements[element].nodes;
        std::sort(nodes.begin(), nodes.end());
        element_sets.emplace(nodes, element);
    }
    int matches = 0;
    for (const MullerP2Element& element : mesh.elements) {
        std::array<int, 6> transformed;
        for (int local = 0; local < 6; local++)
            transformed[local] =
                target_for_source[element.nodes[local]];
        std::sort(transformed.begin(), transformed.end());
        if (element_sets.find(transformed) != element_sets.end())
            matches++;
    }
    return static_cast<double>(matches) /
        static_cast<double>(mesh.elements.size());
}

struct HDivEdgeSymmetryMap {
    std::vector<int> source_edge_for_target;
    std::vector<int> moment0_sign;
};

HDivEdgeSymmetryMap hdiv_edge_symmetry_map(
    const MullerP2Mesh& mesh,
    const std::vector<int>& source_for_target,
    const char* operation)
{
    std::vector<int> target_for_source(mesh.scalar_nodes(), -1);
    for (int target = 0; target < mesh.scalar_nodes(); target++)
        target_for_source[source_for_target[target]] = target;

    std::vector<std::array<int, 2>> edge_vertices(
        mesh.topology_edge_count, {{-1, -1}});
    std::map<std::pair<int, int>, int> edge_by_vertices;
    const int local_vertices[3][2] = {
        {0, 1}, {1, 2}, {2, 0}
    };
    for (const MullerP2Element& element : mesh.elements) {
        for (int local_edge = 0; local_edge < 3; local_edge++) {
            int first = element.topology_vertices[
                local_vertices[local_edge][0]];
            int second = element.topology_vertices[
                local_vertices[local_edge][1]];
            if (first > second)
                std::swap(first, second);
            const int edge = element.topology_edges[local_edge];
            edge_vertices[edge] = {{first, second}};
            edge_by_vertices[std::make_pair(first, second)] = edge;
        }
    }

    HDivEdgeSymmetryMap result;
    result.source_edge_for_target.assign(
        mesh.topology_edge_count, -1);
    result.moment0_sign.assign(mesh.topology_edge_count, 0);
    for (int source_edge = 0;
         source_edge < mesh.topology_edge_count; source_edge++) {
        const int source_first = edge_vertices[source_edge][0];
        const int source_second = edge_vertices[source_edge][1];
        if (source_first < 0 || source_second < 0)
            throw std::runtime_error(
                "incomplete H(div) edge topology");
        const int target_first = target_for_source[source_first];
        const int target_second = target_for_source[source_second];
        const auto target = edge_by_vertices.find(
            std::make_pair(
                std::min(target_first, target_second),
                std::max(target_first, target_second)));
        if (target == edge_by_vertices.end())
            throw std::runtime_error(
                std::string(operation) +
                " does not preserve H(div) edges");
        if (result.source_edge_for_target[target->second] >= 0)
            throw std::runtime_error(
                std::string(operation) +
                " H(div) edge mapping is not one-to-one");
        result.source_edge_for_target[target->second] = source_edge;
        result.moment0_sign[target->second] =
            target_first < target_second ? 1 : -1;
    }
    return result;
}

void rotate_muller_solution(
    const MullerP2Mesh& mesh,
    const std::vector<cdouble>& source,
    const std::vector<int>& source_for_target,
    double angle,
    std::vector<cdouble>& rotated)
{
    if (mesh.basis_kind == MullerBasisKind::HDivBdm1) {
        const HDivEdgeSymmetryMap edge_map =
            hdiv_edge_symmetry_map(
                mesh, source_for_target, "cyclic rotation");

        const int current_dofs = mesh.current_dofs();
        rotated.resize(source.size());
        for (int current = 0; current < 2; current++) {
            const int offset = current * current_dofs;
#pragma omp parallel for schedule(static)
            for (int target_edge = 0;
                 target_edge < mesh.topology_edge_count; target_edge++) {
                const int source_edge =
                    edge_map.source_edge_for_target[target_edge];
                rotated[offset + 2 * target_edge] =
                    (double)edge_map.moment0_sign[target_edge] *
                    source[offset + 2 * source_edge];
                rotated[offset + 2 * target_edge + 1] =
                    source[offset + 2 * source_edge + 1];
            }
        }
        return;
    }

    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const int current_dofs = mesh.current_dofs();
    rotated.resize(source.size());
    for (int current = 0; current < 2; current++) {
        const int offset = current * current_dofs;
#pragma omp parallel for schedule(static)
        for (int target = 0; target < mesh.scalar_nodes(); target++) {
            const int source_node = source_for_target[target];
            const cdouble first = source[offset + 2 * source_node];
            const cdouble second = source[offset + 2 * source_node + 1];
            const Vec3& source_t1 = mesh.tangent1[source_node];
            const Vec3& source_t2 = mesh.tangent2[source_node];
            const cdouble global_x =
                first * source_t1.x + second * source_t2.x;
            const cdouble global_y =
                first * source_t1.y + second * source_t2.y;
            const cdouble global_z =
                first * source_t1.z + second * source_t2.z;
            const cdouble rotated_x =
                cosine * global_x - sine * global_y;
            const cdouble rotated_y =
                sine * global_x + cosine * global_y;
            const Vec3& target_t1 = mesh.tangent1[target];
            const Vec3& target_t2 = mesh.tangent2[target];
            rotated[offset + 2 * target] =
                rotated_x * target_t1.x +
                rotated_y * target_t1.y +
                global_z * target_t1.z;
            rotated[offset + 2 * target + 1] =
                rotated_x * target_t2.x +
                rotated_y * target_t2.y +
                global_z * target_t2.z;
        }
    }
}

void reflect_nodal_solution(
    const MullerP2Mesh& mesh,
    const std::vector<cdouble>& source,
    const std::vector<int>& source_for_target,
    double axis_angle,
    std::vector<cdouble>& reflected)
{
    if (mesh.basis_kind == MullerBasisKind::HDivBdm1) {
        const HDivEdgeSymmetryMap edge_map =
            hdiv_edge_symmetry_map(
                mesh, source_for_target, "mirror reflection");
        const int current_dofs = mesh.current_dofs();
        reflected.resize(source.size());
        for (int current = 0; current < 2; current++) {
            const int offset = current * current_dofs;
            const double parity = current == 0 ? -1.0 : 1.0;
#pragma omp parallel for schedule(static)
            for (int target_edge = 0;
                 target_edge < mesh.topology_edge_count; target_edge++) {
                const int source_edge =
                    edge_map.source_edge_for_target[target_edge];
                reflected[offset + 2 * target_edge] =
                    parity *
                    (double)edge_map.moment0_sign[target_edge] *
                    source[offset + 2 * source_edge];
                reflected[offset + 2 * target_edge + 1] =
                    parity * source[offset + 2 * source_edge + 1];
            }
        }
        return;
    }

    const double cosine_twice = std::cos(2.0 * axis_angle);
    const double sine_twice = std::sin(2.0 * axis_angle);
    const int current_dofs = mesh.current_dofs();
    reflected.resize(source.size());
    for (int current = 0; current < 2; current++) {
        const int offset = current * current_dofs;
        const double parity = current == 0 ? 1.0 : -1.0;
#pragma omp parallel for schedule(static)
        for (int target = 0; target < mesh.scalar_nodes(); target++) {
            const int source_node = source_for_target[target];
            const cdouble first = source[offset + 2 * source_node];
            const cdouble second = source[offset + 2 * source_node + 1];
            const Vec3& source_t1 = mesh.tangent1[source_node];
            const Vec3& source_t2 = mesh.tangent2[source_node];
            const cdouble global_x =
                first * source_t1.x + second * source_t2.x;
            const cdouble global_y =
                first * source_t1.y + second * source_t2.y;
            const cdouble global_z =
                first * source_t1.z + second * source_t2.z;
            const cdouble reflected_x = parity *
                (cosine_twice * global_x +
                 sine_twice * global_y);
            const cdouble reflected_y = parity *
                (sine_twice * global_x -
                 cosine_twice * global_y);
            const cdouble reflected_z = parity * global_z;
            const Vec3& target_t1 = mesh.tangent1[target];
            const Vec3& target_t2 = mesh.tangent2[target];
            reflected[offset + 2 * target] =
                reflected_x * target_t1.x +
                reflected_y * target_t1.y +
                reflected_z * target_t1.z;
            reflected[offset + 2 * target + 1] =
                reflected_x * target_t2.x +
                reflected_y * target_t2.y +
                reflected_z * target_t2.z;
        }
    }
}

double maximum_mesh_edge(const Mesh& mesh)
{
    double result = 0.0;
    for (int element = 0; element < mesh.nt(); element++) {
        const Vec3& a = mesh.verts[mesh.tris[3 * element]];
        const Vec3& b = mesh.verts[mesh.tris[3 * element + 1]];
        const Vec3& c = mesh.verts[mesh.tris[3 * element + 2]];
        result = std::max(result, (a - b).norm());
        result = std::max(result, (b - c).norm());
        result = std::max(result, (c - a).norm());
    }
    return result;
}

const char* muller_edge_mode_name(MullerEdgeMode mode)
{
    switch (mode) {
        case MullerEdgeMode::Smooth: return "smooth";
        case MullerEdgeMode::SplitFeatureEdges: return "split";
        case MullerEdgeMode::HDivBdm1: return "hdiv-bdm1";
    }
    return "unknown";
}

void rotate_mesh_about_z(Mesh& mesh, double angle)
{
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    for (Vec3& vertex : mesh.verts)
        vertex = rotate_about_z(vertex, cosine, sine);
}

void align_icosphere_fivefold_axis_to_z(Mesh& mesh)
{
    if (mesh.verts.empty())
        throw std::runtime_error("cannot align an empty icosphere");
    const Vec3 fivefold_axis = mesh.verts.front();
    for (Vec3& vertex : mesh.verts)
        vertex = rotate_axis_to_z(vertex, fivefold_axis);
}

std::vector<cdouble> prolong_icosphere_p2_solution(
    const MullerP2Mesh& coarse,
    const MullerP2Mesh& fine,
    const std::vector<cdouble>& coarse_solution)
{
    if (fine.elements.size() != 4 * coarse.elements.size() ||
        coarse_solution.size() !=
            static_cast<std::size_t>(coarse.system_dofs()) ||
        coarse.basis_kind != MullerBasisKind::NodalP2 ||
        fine.basis_kind != MullerBasisKind::NodalP2) {
        throw std::invalid_argument(
            "sphere prolongation requires one nested nodal P2 refinement");
    }
    const double child_vertices[4][3][2] = {
        {{0.0, 0.0}, {0.5, 0.0}, {0.0, 0.5}},
        {{1.0, 0.0}, {0.5, 0.5}, {0.5, 0.0}},
        {{0.0, 1.0}, {0.0, 0.5}, {0.5, 0.5}},
        {{0.5, 0.0}, {0.5, 0.5}, {0.0, 0.5}}
    };
    const double local_coordinates[6][2] = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0},
        {0.5, 0.0}, {0.5, 0.5}, {0.0, 0.5}
    };
    std::vector<cdouble> fine_solution(
        fine.system_dofs(), cdouble(0.0));
    std::vector<unsigned char> assigned(fine.scalar_nodes(), 0);
    const int coarse_current_dofs = coarse.current_dofs();
    const int fine_current_dofs = fine.current_dofs();
    for (int parent = 0;
         parent < static_cast<int>(coarse.elements.size());
         parent++) {
        const MullerP2Element& coarse_element =
            coarse.elements[parent];
        for (int child = 0; child < 4; child++) {
            const MullerP2Element& fine_element =
                fine.elements[4 * parent + child];
            const double* p0 = child_vertices[child][0];
            const double* p1 = child_vertices[child][1];
            const double* p2 = child_vertices[child][2];
            for (int local = 0; local < 6; local++) {
                const int fine_node = fine_element.nodes[local];
                if (assigned[fine_node])
                    continue;
                const double child_xi = local_coordinates[local][0];
                const double child_eta = local_coordinates[local][1];
                const double parent_xi =
                    p0[0] + child_xi * (p1[0] - p0[0]) +
                    child_eta * (p2[0] - p0[0]);
                const double parent_eta =
                    p0[1] + child_xi * (p1[1] - p0[1]) +
                    child_eta * (p2[1] - p0[1]);
                std::array<double, 6> shape, dxi, deta;
                muller_p2_shape(
                    parent_xi, parent_eta, shape, dxi, deta);
                for (int current = 0; current < 2; current++) {
                    for (int component = 0;
                         component < 2; component++) {
                        cdouble value(0.0);
                        for (int coarse_local = 0;
                             coarse_local < 6; coarse_local++) {
                            const int coarse_node =
                                coarse_element.nodes[coarse_local];
                            value += shape[coarse_local] *
                                coarse_solution[
                                    current * coarse_current_dofs +
                                    2 * coarse_node + component];
                        }
                        fine_solution[
                            current * fine_current_dofs +
                            2 * fine_node + component] = value;
                    }
                }
                assigned[fine_node] = 1;
            }
        }
    }
    if (std::find(
            assigned.begin(), assigned.end(),
            static_cast<unsigned char>(0)) != assigned.end()) {
        throw std::runtime_error(
            "sphere prolongation did not cover every fine P2 node");
    }
    return fine_solution;
}

void amplitude_to_mueller(
    const std::vector<cdouble>& s1,
    const std::vector<cdouble>& s2,
    const std::vector<cdouble>& s3,
    const std::vector<cdouble>& s4,
    std::vector<double>& mueller)
{
    const int ntheta = static_cast<int>(s1.size());
    mueller.assign(static_cast<size_t>(16) * ntheta, 0.0);
    const auto index = [ntheta](int row, int column, int angle) {
        return (static_cast<size_t>(row) * 4 + column) * ntheta + angle;
    };
    for (int angle = 0; angle < ntheta; angle++) {
        const double a1 = std::norm(s1[angle]);
        const double a2 = std::norm(s2[angle]);
        const double a3 = std::norm(s3[angle]);
        const double a4 = std::norm(s4[angle]);
        const cdouble s2s3 = s2[angle] * std::conj(s3[angle]);
        const cdouble s1s4 = s1[angle] * std::conj(s4[angle]);
        const cdouble s2s4 = s2[angle] * std::conj(s4[angle]);
        const cdouble s1s3 = s1[angle] * std::conj(s3[angle]);
        const cdouble s1s2 = s1[angle] * std::conj(s2[angle]);
        const cdouble s3s4 = s3[angle] * std::conj(s4[angle]);
        mueller[index(0, 0, angle)] = 0.5 * (a1 + a2 + a3 + a4);
        mueller[index(0, 1, angle)] = 0.5 * (a2 - a1 + a4 - a3);
        mueller[index(1, 0, angle)] = 0.5 * (a2 - a1 - a4 + a3);
        mueller[index(1, 1, angle)] = 0.5 * (a2 + a1 - a4 - a3);
        mueller[index(0, 2, angle)] = s2s3.real() + s1s4.real();
        mueller[index(0, 3, angle)] = s2s3.imag() - s1s4.imag();
        mueller[index(1, 2, angle)] = s2s3.real() - s1s4.real();
        mueller[index(1, 3, angle)] = s2s3.imag() + s1s4.imag();
        mueller[index(2, 0, angle)] = s2s4.real() + s1s3.real();
        mueller[index(2, 1, angle)] = s2s4.real() - s1s3.real();
        mueller[index(2, 2, angle)] = s1s2.real() + s3s4.real();
        mueller[index(2, 3, angle)] = -s1s2.imag() - s3s4.imag();
        mueller[index(3, 0, angle)] =
            (s4[angle] * std::conj(s2[angle])).imag() + s1s3.imag();
        mueller[index(3, 1, angle)] =
            (s4[angle] * std::conj(s2[angle])).imag() - s1s3.imag();
        mueller[index(3, 2, angle)] = s1s2.imag() - s3s4.imag();
        mueller[index(3, 3, angle)] = s1s2.real() - s3s4.real();
    }
}

double norm(const cdouble* vector, int n)
{
    double value = 0.0;
#pragma omp parallel for reduction(+:value) schedule(static)
    for (int i = 0; i < n; i++)
        value += std::norm(vector[i]);
    return std::sqrt(value);
}

cdouble orthogonalize_against(
    const cdouble* basis, cdouble* work, int n)
{
    double real = 0.0;
    double imaginary = 0.0;
    cdouble projection(0.0);
#pragma omp parallel
    {
#pragma omp for reduction(+:real,imaginary) schedule(static)
        for (int i = 0; i < n; i++) {
            const cdouble value = std::conj(basis[i]) * work[i];
            real += value.real();
            imaginary += value.imag();
        }
#pragma omp single
        projection = cdouble(real, imaginary);
#pragma omp for schedule(static)
        for (int i = 0; i < n; i++)
            work[i] -= projection * basis[i];
    }
    return projection;
}

cdouble inner_product(
    const cdouble* left, const cdouble* right, int n)
{
    double real = 0.0;
    double imaginary = 0.0;
#pragma omp parallel for reduction(+:real,imaginary) schedule(static)
    for (int i = 0; i < n; i++) {
        const cdouble value = std::conj(left[i]) * right[i];
        real += value.real();
        imaginary += value.imag();
    }
    return cdouble(real, imaginary);
}

GmresResult solve_gmres(
    const Matvec& matvec,
    const cdouble* rhs,
    int n,
    double tolerance,
    int maximum_iterations,
    int restart,
    const MullerMbjPreconditioner* preconditioner,
    const char* label,
    const std::vector<cdouble>* initial_guess = nullptr,
    const cdouble* recycle_rhs = nullptr,
    std::vector<cdouble>* recycle_guess = nullptr,
    const std::function<void(
        const std::vector<cdouble>&,
        std::vector<cdouble>&)>* recycle_base_transform = nullptr,
    const SolverCheckpointOptions& checkpoint =
        SolverCheckpointOptions())
{
    const auto start = std::chrono::steady_clock::now();
    const int maximum = std::min(n, maximum_iterations);
    const int cycle_limit =
        restart > 0 ? std::min(restart, maximum) : maximum;
    std::vector<cdouble> basis((size_t)(cycle_limit + 1) * n);
    std::vector<cdouble> hessenberg(
        (size_t)(cycle_limit + 1) * cycle_limit, cdouble(0.0));
    std::vector<cdouble> cosine(cycle_limit);
    std::vector<cdouble> sine(cycle_limit);
    std::vector<cdouble> projected(cycle_limit + 1, cdouble(0.0));
    std::vector<cdouble> work(n), preconditioned(n), residual(n);
    const double rhs_norm = std::max(norm(rhs, n), 1.0e-300);
    const std::uint64_t rhs_hash = vector_hash(rhs, n);
    GmresResult result;
    std::vector<cdouble> checkpoint_solution;
    int checkpoint_iterations = 0;
    double checkpoint_residual = 1.0;
    const bool resumed = load_solver_checkpoint(
        checkpoint, rhs_hash, n, checkpoint_iterations,
        checkpoint_residual, checkpoint_solution);
    const std::vector<cdouble>* starting_guess =
        resumed ? &checkpoint_solution : initial_guess;
    if (starting_guess) {
        if (starting_guess->size() != static_cast<size_t>(n))
            throw std::invalid_argument(
                "GMRES initial guess has the wrong size");
        result.solution = *starting_guess;
        matvec(result.solution.data(), work.data());
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            residual[i] = rhs[i] - work[i];
    } else {
        result.solution.assign(n, cdouble(0.0));
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            residual[i] = rhs[i];
    }

    int total_iterations = resumed ? checkpoint_iterations : 0;
    result.resumed_iterations = total_iterations;
    int final_cycle_iterations = 0;
    double projected_relative = 1.0;
    double operator_relative = norm(residual.data(), n) / rhs_norm;
    result.initial_operator_residual = operator_relative;
    if (resumed) {
        std::printf(
            "  [%s checkpoint] resumed at iteration %d; "
            "stored residual %.3e, verified residual %.3e\n",
            label, total_iterations, checkpoint_residual,
            operator_relative);
        std::fflush(stdout);
    } else if (starting_guess) {
        std::printf(
            "  [%s] initial operator residual %.3e\n",
            label, operator_relative);
        std::fflush(stdout);
    } else if (!checkpoint.path.empty()) {
        std::printf(
            "  [%s checkpoint] autosave after every outer iteration: %s\n",
            label, checkpoint.path.c_str());
        std::fflush(stdout);
    }
    log_iteration(
        label, "initial", 0, projected_relative, operator_relative,
        0.0, 0.0, 0.0,
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count());
    while (total_iterations < maximum && operator_relative >= tolerance) {
        const int cycle_maximum = std::min(
            cycle_limit, maximum - total_iterations);
        std::fill(hessenberg.begin(), hessenberg.end(), cdouble(0.0));
        std::fill(cosine.begin(), cosine.end(), cdouble(0.0));
        std::fill(sine.begin(), sine.end(), cdouble(0.0));
        std::fill(projected.begin(), projected.end(), cdouble(0.0));
        const double residual_norm = norm(residual.data(), n);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            basis[i] = residual[i] / residual_norm;
        projected[0] = residual_norm;

        int cycle_iterations = 0;
        for (int column = 0; column < cycle_maximum; column++) {
            const auto iteration_start =
                std::chrono::steady_clock::now();
            const cdouble* vector =
                basis.data() + (size_t)column * n;
            double preconditioner_seconds = 0.0;
            const cdouble* action_input = vector;
            if (preconditioner) {
                const auto preconditioner_start =
                    std::chrono::steady_clock::now();
                preconditioner->apply(
                    vector, preconditioned.data());
                preconditioner_seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        preconditioner_start).count();
                action_input = preconditioned.data();
            }
            const auto matvec_start =
                std::chrono::steady_clock::now();
            matvec(action_input, work.data());
            const double matvec_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    matvec_start).count();

            const auto orthogonalization_start =
                std::chrono::steady_clock::now();
            for (int row = 0; row <= column; row++) {
                const cdouble* previous =
                    basis.data() + (size_t)row * n;
                const cdouble value = orthogonalize_against(
                    previous, work.data(), n);
                hessenberg[
                    (size_t)row * cycle_limit + column] = value;
            }
            for (int row = 0; row <= column; row++) {
                const cdouble* previous =
                    basis.data() + (size_t)row * n;
                const cdouble correction = orthogonalize_against(
                    previous, work.data(), n);
                hessenberg[
                    (size_t)row * cycle_limit + column] += correction;
            }
            const double orthogonalization_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    orthogonalization_start).count();

            const double next_norm = norm(work.data(), n);
            hessenberg[
                (size_t)(column + 1) * cycle_limit + column] =
                cdouble(next_norm, 0.0);
            if (next_norm > 1.0e-30) {
                cdouble* next =
                    basis.data() + (size_t)(column + 1) * n;
#pragma omp parallel for schedule(static)
                for (int i = 0; i < n; i++)
                    next[i] = work[i] / next_norm;
            }
            for (int row = 0; row < column; row++) {
                const cdouble first =
                    hessenberg[(size_t)row * cycle_limit + column];
                const cdouble second =
                    hessenberg[
                        (size_t)(row + 1) * cycle_limit + column];
                hessenberg[(size_t)row * cycle_limit + column] =
                    std::conj(cosine[row]) * first +
                    std::conj(sine[row]) * second;
                hessenberg[
                    (size_t)(row + 1) * cycle_limit + column] =
                    -sine[row] * first + cosine[row] * second;
            }
            const cdouble first =
                hessenberg[(size_t)column * cycle_limit + column];
            const cdouble second =
                hessenberg[
                    (size_t)(column + 1) * cycle_limit + column];
            const double denominator =
                std::sqrt(std::norm(first) + std::norm(second));
            cosine[column] = denominator > 1.0e-30
                ? first / denominator : cdouble(1.0);
            sine[column] = denominator > 1.0e-30
                ? second / denominator : cdouble(0.0);
            hessenberg[(size_t)column * cycle_limit + column] =
                std::conj(cosine[column]) * first +
                std::conj(sine[column]) * second;
            hessenberg[
                (size_t)(column + 1) * cycle_limit + column] = 0.0;
            projected[column + 1] =
                -sine[column] * projected[column];
            projected[column] =
                std::conj(cosine[column]) * projected[column];
            cycle_iterations = column + 1;
            total_iterations++;
            projected_relative =
                std::abs(projected[column + 1]) / rhs_norm;

            const double iteration_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    iteration_start).count();
            log_iteration(
                label, "iteration", total_iterations,
                projected_relative, operator_relative,
                matvec_seconds, preconditioner_seconds,
                orthogonalization_seconds,
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count());
            if (total_iterations == 1 ||
                total_iterations % 25 == 0) {
                std::printf(
                    "  [%s %d] projected residual %.3e; "
                    "matvec %.3fs, preconditioner %.3fs, "
                    "orthogonalization %.3fs, iteration %.3fs\n",
                    label, total_iterations, projected_relative,
                    matvec_seconds, preconditioner_seconds,
                    orthogonalization_seconds, iteration_seconds);
                std::fflush(stdout);
            }
            if (!checkpoint.path.empty()) {
                std::vector<cdouble> checkpoint_coefficients;
                solve_projected_system(
                    hessenberg, cycle_limit, projected,
                    cycle_iterations, checkpoint_coefficients);
#pragma omp parallel for schedule(static)
                for (int i = 0; i < n; i++) {
                    cdouble value(0.0);
                    for (int inner_column = 0;
                         inner_column < cycle_iterations;
                         inner_column++) {
                        value +=
                            checkpoint_coefficients[inner_column] *
                            basis[
                                static_cast<size_t>(
                                    inner_column) * n + i];
                    }
                    work[i] = value;
                }
                if (preconditioner) {
                    preconditioner->apply(
                        work.data(), preconditioned.data());
                } else {
                    preconditioned = work;
                }
                std::vector<cdouble> candidate(n);
#pragma omp parallel for schedule(static)
                for (int i = 0; i < n; i++)
                    candidate[i] =
                        result.solution[i] + preconditioned[i];
                save_solver_checkpoint(
                    checkpoint, rhs_hash, total_iterations,
                    projected_relative, candidate);
                std::printf(
                    "  [%s checkpoint] saved iteration %d\n",
                    label, total_iterations);
                std::fflush(stdout);
            }
            if (projected_relative < tolerance)
                break;
        }

        std::vector<cdouble> coefficients;
        solve_projected_system(
            hessenberg, cycle_limit, projected,
            cycle_iterations, coefficients);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            cdouble value(0.0);
            for (int column = 0; column < cycle_iterations; column++)
                value += coefficients[column] *
                    basis[(size_t)column * n + i];
            work[i] = value;
        }

        if (preconditioner)
            preconditioner->apply(work.data(), preconditioned.data());
        else
            preconditioned = work;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            result.solution[i] += preconditioned[i];

        const auto exact_matvec_start =
            std::chrono::steady_clock::now();
        matvec(result.solution.data(), work.data());
        const double exact_matvec_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                exact_matvec_start).count();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            residual[i] = rhs[i] - work[i];
        operator_relative = norm(residual.data(), n) / rhs_norm;
        final_cycle_iterations = cycle_iterations;
        log_iteration(
            label, "cycle", total_iterations, projected_relative,
            operator_relative, exact_matvec_seconds, 0.0, 0.0,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count());
        std::printf(
            "  [%s %d] exact operator residual %.3e; "
            "verification matvec %.3fs\n",
            label, total_iterations, operator_relative,
            exact_matvec_seconds);
        save_solver_checkpoint(
            checkpoint, rhs_hash, total_iterations,
            operator_relative, result.solution);
        std::fflush(stdout);
    }

    result.iterations = total_iterations;
    result.projected_residual = projected_relative;
    result.operator_residual = operator_relative;
    result.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    log_iteration(
        label, "final", total_iterations, projected_relative,
        operator_relative, 0.0, 0.0, 0.0, result.seconds);
    if (recycle_rhs && recycle_guess && final_cycle_iterations > 0) {
        const auto recycle_start = std::chrono::steady_clock::now();
        if (recycle_base_transform) {
            (*recycle_base_transform)(
                result.solution, *recycle_guess);
            matvec(recycle_guess->data(), work.data());
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++)
                residual[i] = recycle_rhs[i] - work[i];
        } else {
            recycle_guess->assign(n, cdouble(0.0));
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++)
                residual[i] = recycle_rhs[i];
        }
        std::vector<cdouble> projected_recycle(
            final_cycle_iterations + 1);
        for (int row = 0; row <= final_cycle_iterations; row++) {
            projected_recycle[row] = inner_product(
                basis.data() + (size_t)row * n, residual.data(), n);
        }
        for (int row = 0; row < final_cycle_iterations; row++) {
            const cdouble first = projected_recycle[row];
            const cdouble second = projected_recycle[row + 1];
            projected_recycle[row] =
                std::conj(cosine[row]) * first +
                std::conj(sine[row]) * second;
            projected_recycle[row + 1] =
                -sine[row] * first + cosine[row] * second;
        }
        std::vector<cdouble> coefficients(final_cycle_iterations);
        for (int row = final_cycle_iterations - 1; row >= 0; row--) {
            cdouble value = projected_recycle[row];
            for (int column = row + 1;
                 column < final_cycle_iterations; column++) {
                value -=
                    hessenberg[
                        (size_t)row * cycle_limit + column] *
                    coefficients[column];
            }
            coefficients[row] =
                value /
                hessenberg[(size_t)row * cycle_limit + row];
        }
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            cdouble value(0.0);
            for (int column = 0;
                 column < final_cycle_iterations; column++) {
                value += coefficients[column] *
                    basis[(size_t)column * n + i];
            }
            work[i] = value;
        }
        if (preconditioner) {
            preconditioner->apply(
                work.data(), preconditioned.data());
        } else {
            preconditioned = work;
        }
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            (*recycle_guess)[i] += preconditioned[i];
        result.recycle_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            recycle_start).count();
    }
    return result;
}

GmresResult solve_flexible_gmres(
    const Matvec& matvec,
    const FlexiblePreconditioner& precondition,
    const cdouble* rhs,
    int n,
    double tolerance,
    int maximum_iterations,
    int restart,
    const char* label,
    const std::vector<cdouble>* initial_guess = nullptr,
    const cdouble* recycle_rhs = nullptr,
    std::vector<cdouble>* recycle_guess = nullptr,
    const std::function<void(
        const std::vector<cdouble>&,
        std::vector<cdouble>&)>* recycle_transform = nullptr,
    const SolverCheckpointOptions& checkpoint =
        SolverCheckpointOptions())
{
    const auto start = std::chrono::steady_clock::now();
    const int maximum = std::min(n, maximum_iterations);
    const int cycle_limit = std::min(
        restart > 0 ? restart : 12, maximum);
    std::vector<cdouble> basis(
        static_cast<size_t>(cycle_limit + 1) * n);
    std::vector<cdouble> preconditioned_basis(
        static_cast<size_t>(cycle_limit) * n);
    std::vector<cdouble> hessenberg(
        static_cast<size_t>(cycle_limit + 1) * cycle_limit,
        cdouble(0.0));
    std::vector<cdouble> cosine(cycle_limit);
    std::vector<cdouble> sine(cycle_limit);
    std::vector<cdouble> projected(
        cycle_limit + 1, cdouble(0.0));
    std::vector<cdouble> work(n), residual(n);
    const double rhs_norm = std::max(norm(rhs, n), 1.0e-300);
    const std::uint64_t rhs_hash = vector_hash(rhs, n);

    GmresResult result;
    std::vector<cdouble> checkpoint_solution;
    int checkpoint_iterations = 0;
    double checkpoint_residual = 1.0;
    const bool resumed = load_solver_checkpoint(
        checkpoint, rhs_hash, n, checkpoint_iterations,
        checkpoint_residual, checkpoint_solution);
    const std::vector<cdouble>* starting_guess =
        resumed ? &checkpoint_solution : initial_guess;
    if (starting_guess) {
        if (starting_guess->size() != static_cast<size_t>(n))
            throw std::invalid_argument(
                "FGMRES initial guess has the wrong size");
        result.solution = *starting_guess;
        matvec(result.solution.data(), work.data());
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            residual[i] = rhs[i] - work[i];
    } else {
        result.solution.assign(n, cdouble(0.0));
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            residual[i] = rhs[i];
    }
    result.initial_operator_residual =
        norm(residual.data(), n) / rhs_norm;
    double operator_relative = result.initial_operator_residual;
    double projected_relative = operator_relative;
    int total_iterations = resumed ? checkpoint_iterations : 0;
    int final_cycle_iterations = 0;
    result.resumed_iterations = total_iterations;
    if (resumed) {
        std::printf(
            "  [%s checkpoint] resumed at iteration %d; "
            "stored residual %.3e, verified residual %.3e\n",
            label, total_iterations, checkpoint_residual,
            operator_relative);
        std::fflush(stdout);
    } else if (starting_guess) {
        std::printf(
            "  [%s] initial operator residual %.3e\n",
            label, operator_relative);
        std::fflush(stdout);
    } else if (!checkpoint.path.empty()) {
        std::printf(
            "  [%s checkpoint] autosave after every outer iteration: %s\n",
            label, checkpoint.path.c_str());
        std::fflush(stdout);
    }
    log_iteration(
        label, "initial", total_iterations, projected_relative,
        operator_relative, 0.0, 0.0, 0.0,
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count());

    while (total_iterations < maximum &&
           operator_relative >= tolerance) {
        const int cycle_maximum = std::min(
            cycle_limit, maximum - total_iterations);
        std::fill(
            hessenberg.begin(), hessenberg.end(), cdouble(0.0));
        std::fill(cosine.begin(), cosine.end(), cdouble(0.0));
        std::fill(sine.begin(), sine.end(), cdouble(0.0));
        std::fill(projected.begin(), projected.end(), cdouble(0.0));

        const double residual_norm = norm(residual.data(), n);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            basis[i] = residual[i] / residual_norm;
        projected[0] = residual_norm;

        int cycle_iterations = 0;
        for (int column = 0; column < cycle_maximum; column++) {
            const cdouble* vector =
                basis.data() + static_cast<size_t>(column) * n;
            cdouble* flexible_vector =
                preconditioned_basis.data() +
                static_cast<size_t>(column) * n;
            const auto preconditioner_start =
                std::chrono::steady_clock::now();
            precondition(vector, flexible_vector);
            const double preconditioner_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    preconditioner_start).count();
            const auto matvec_start =
                std::chrono::steady_clock::now();
            matvec(flexible_vector, work.data());
            const double matvec_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    matvec_start).count();

            const auto orthogonalization_start =
                std::chrono::steady_clock::now();
            for (int pass = 0; pass < 2; pass++) {
                for (int row = 0; row <= column; row++) {
                    const cdouble* previous =
                        basis.data() + static_cast<size_t>(row) * n;
                    const cdouble value = orthogonalize_against(
                        previous, work.data(), n);
                    hessenberg[
                        static_cast<size_t>(row) * cycle_limit +
                        column] += value;
                }
            }
            const double orthogonalization_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    orthogonalization_start).count();
            const double next_norm = norm(work.data(), n);
            hessenberg[
                static_cast<size_t>(column + 1) * cycle_limit +
                column] = cdouble(next_norm, 0.0);
            if (next_norm > 1.0e-30) {
                cdouble* next =
                    basis.data() +
                    static_cast<size_t>(column + 1) * n;
#pragma omp parallel for schedule(static)
                for (int i = 0; i < n; i++)
                    next[i] = work[i] / next_norm;
            }

            for (int row = 0; row < column; row++) {
                const cdouble first =
                    hessenberg[
                        static_cast<size_t>(row) * cycle_limit +
                        column];
                const cdouble second =
                    hessenberg[
                        static_cast<size_t>(row + 1) * cycle_limit +
                        column];
                hessenberg[
                    static_cast<size_t>(row) * cycle_limit + column] =
                    std::conj(cosine[row]) * first +
                    std::conj(sine[row]) * second;
                hessenberg[
                    static_cast<size_t>(row + 1) * cycle_limit +
                    column] =
                    -sine[row] * first + cosine[row] * second;
            }
            const cdouble first =
                hessenberg[
                    static_cast<size_t>(column) * cycle_limit +
                    column];
            const cdouble second =
                hessenberg[
                    static_cast<size_t>(column + 1) * cycle_limit +
                    column];
            const double denominator =
                std::sqrt(std::norm(first) + std::norm(second));
            cosine[column] = denominator > 1.0e-30
                ? first / denominator : cdouble(1.0);
            sine[column] = denominator > 1.0e-30
                ? second / denominator : cdouble(0.0);
            hessenberg[
                static_cast<size_t>(column) * cycle_limit + column] =
                std::conj(cosine[column]) * first +
                std::conj(sine[column]) * second;
            hessenberg[
                static_cast<size_t>(column + 1) * cycle_limit +
                column] = 0.0;
            projected[column + 1] =
                -sine[column] * projected[column];
            projected[column] =
                std::conj(cosine[column]) * projected[column];
            cycle_iterations = column + 1;
            total_iterations++;
            projected_relative =
                std::abs(projected[column + 1]) / rhs_norm;
            std::printf(
                "  [%s %d] projected residual %.3e\n",
                label, total_iterations, projected_relative);
            log_iteration(
                label, "iteration", total_iterations,
                projected_relative, operator_relative,
                matvec_seconds, preconditioner_seconds,
                orthogonalization_seconds,
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count());
            if (!checkpoint.path.empty()) {
                std::vector<cdouble> checkpoint_coefficients;
                solve_projected_system(
                    hessenberg, cycle_limit, projected,
                    cycle_iterations, checkpoint_coefficients);
                std::vector<cdouble> candidate(n);
#pragma omp parallel for schedule(static)
                for (int i = 0; i < n; i++) {
                    cdouble update(0.0);
                    for (int inner_column = 0;
                         inner_column < cycle_iterations;
                         inner_column++) {
                        update +=
                            checkpoint_coefficients[inner_column] *
                            preconditioned_basis[
                                static_cast<size_t>(
                                    inner_column) * n + i];
                    }
                    candidate[i] = result.solution[i] + update;
                }
                save_solver_checkpoint(
                    checkpoint, rhs_hash, total_iterations,
                    projected_relative, candidate);
                std::printf(
                    "  [%s checkpoint] saved iteration %d\n",
                    label, total_iterations);
                std::fflush(stdout);
            }
            if (projected_relative < tolerance)
                break;
        }

        std::vector<cdouble> coefficients;
        solve_projected_system(
            hessenberg, cycle_limit, projected,
            cycle_iterations, coefficients);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            cdouble update(0.0);
            for (int column = 0;
                 column < cycle_iterations; column++) {
                update += coefficients[column] *
                    preconditioned_basis[
                        static_cast<size_t>(column) * n + i];
            }
            result.solution[i] += update;
        }

        matvec(result.solution.data(), work.data());
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            residual[i] = rhs[i] - work[i];
        operator_relative = norm(residual.data(), n) / rhs_norm;
        final_cycle_iterations = cycle_iterations;
        log_iteration(
            label, "cycle", total_iterations, projected_relative,
            operator_relative, 0.0, 0.0, 0.0,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count());
        std::printf(
            "  [%s %d] exact operator residual %.3e\n",
            label, total_iterations, operator_relative);
        save_solver_checkpoint(
            checkpoint, rhs_hash, total_iterations,
            operator_relative, result.solution);
        std::fflush(stdout);
    }

    result.iterations = total_iterations;
    result.projected_residual = projected_relative;
    result.operator_residual = operator_relative;
    result.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    log_iteration(
        label, "final", total_iterations, projected_relative,
        operator_relative, 0.0, 0.0, 0.0, result.seconds);
    if (recycle_rhs && recycle_guess && final_cycle_iterations > 0) {
        const auto recycle_start = std::chrono::steady_clock::now();
        std::vector<cdouble> transformed(n);
        if (recycle_transform) {
            (*recycle_transform)(result.solution, *recycle_guess);
            matvec(recycle_guess->data(), work.data());
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++)
                residual[i] = recycle_rhs[i] - work[i];
        } else {
            recycle_guess->assign(n, cdouble(0.0));
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++)
                residual[i] = recycle_rhs[i];
        }

        std::vector<cdouble> projected_recycle(
            final_cycle_iterations + 1);
        for (int row = 0; row <= final_cycle_iterations; row++) {
            const cdouble* source =
                basis.data() + static_cast<size_t>(row) * n;
            if (recycle_transform) {
                std::vector<cdouble> source_vector(source, source + n);
                (*recycle_transform)(source_vector, transformed);
                source = transformed.data();
            }
            projected_recycle[row] =
                inner_product(source, residual.data(), n);
        }
        for (int row = 0; row < final_cycle_iterations; row++) {
            const cdouble first = projected_recycle[row];
            const cdouble second = projected_recycle[row + 1];
            projected_recycle[row] =
                std::conj(cosine[row]) * first +
                std::conj(sine[row]) * second;
            projected_recycle[row + 1] =
                -sine[row] * first + cosine[row] * second;
        }
        std::vector<cdouble> coefficients(final_cycle_iterations);
        for (int row = final_cycle_iterations - 1; row >= 0; row--) {
            cdouble value = projected_recycle[row];
            for (int column = row + 1;
                 column < final_cycle_iterations; column++) {
                value -=
                    hessenberg[
                        static_cast<size_t>(row) * cycle_limit +
                        column] *
                    coefficients[column];
            }
            coefficients[row] =
                value /
                hessenberg[
                    static_cast<size_t>(row) * cycle_limit + row];
        }
        for (int column = 0;
             column < final_cycle_iterations; column++) {
            const cdouble* update =
                preconditioned_basis.data() +
                static_cast<size_t>(column) * n;
            if (recycle_transform) {
                std::vector<cdouble> update_vector(update, update + n);
                (*recycle_transform)(update_vector, transformed);
                update = transformed.data();
            }
            const cdouble coefficient = coefficients[column];
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++)
                (*recycle_guess)[i] += coefficient * update[i];
        }
        result.recycle_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            recycle_start).count();
    }
    return result;
}

double dense_residual(
    const MullerDenseSystem& system,
    const std::vector<cdouble>& solution,
    const std::vector<cdouble>& rhs)
{
    double residual_squared = 0.0;
    double rhs_squared = 0.0;
    for (int row = 0; row < system.system_dofs; row++) {
        cdouble action(0.0);
        for (int column = 0;
             column < system.system_dofs; column++) {
            action += system.matrix[
                (size_t)row * system.system_dofs + column] *
                solution[column];
        }
        residual_squared += std::norm(rhs[row] - action);
        rhs_squared += std::norm(rhs[row]);
    }
    return std::sqrt(residual_squared / rhs_squared);
}

struct AveragingOrientation {
    Mat3 RT;
    double beta = 0.0;
    double gamma = 0.0;
    double weight = 0.0;
};

double rotation_distance_squared(const Mat3& first, const Mat3& second)
{
    double result = 0.0;
    for (int row = 0; row < 3; row++)
        for (int column = 0; column < 3; column++) {
            const double difference =
                first.m[row][column] - second.m[row][column];
            result += difference * difference;
        }
    return result;
}

std::vector<AveragingOrientation> averaging_orientations(
    int beta_count, int gamma_count, int symmetry_order)
{
    std::vector<double> beta_nodes;
    std::vector<double> beta_weights;
    gauss_legendre(beta_count, beta_nodes, beta_weights);
    std::vector<AveragingOrientation> unordered;
    unordered.reserve(beta_count * gamma_count);
    const double gamma_period =
        2.0 * M_PI / static_cast<double>(symmetry_order);
    for (int beta_index = 0;
         beta_index < beta_count; beta_index++) {
        const double beta = std::acos(beta_nodes[beta_index]);
        for (int gamma_index = 0;
             gamma_index < gamma_count; gamma_index++) {
            const double gamma =
                gamma_period *
                (gamma_index + 0.5) /
                static_cast<double>(gamma_count);
            AveragingOrientation orientation;
            orientation.RT =
                euler_rotation(0.0, beta, gamma).T();
            orientation.beta = beta;
            orientation.gamma = gamma;
            orientation.weight =
                0.5 * beta_weights[beta_index] /
                static_cast<double>(gamma_count);
            unordered.push_back(orientation);
        }
    }

    std::vector<AveragingOrientation> ordered;
    ordered.reserve(unordered.size());
    std::vector<unsigned char> used(unordered.size(), 0);
    size_t current = 0;
    for (size_t count = 0; count < unordered.size(); count++) {
        ordered.push_back(unordered[current]);
        used[current] = 1;
        size_t nearest = unordered.size();
        double nearest_distance = std::numeric_limits<double>::max();
        for (size_t candidate = 0;
             candidate < unordered.size(); candidate++) {
            if (used[candidate])
                continue;
            const double distance = rotation_distance_squared(
                unordered[current].RT, unordered[candidate].RT);
            if (distance < nearest_distance) {
                nearest_distance = distance;
                nearest = candidate;
            }
        }
        if (nearest == unordered.size())
            break;
        current = nearest;
    }
    return ordered;
}

std::array<cdouble, 3> rotate_complex_vector(
    const Mat3& matrix,
    const cdouble* vector)
{
    std::array<cdouble, 3> result;
    for (int row = 0; row < 3; row++) {
        result[row] =
            matrix.m[row][0] * vector[0] +
            matrix.m[row][1] * vector[1] +
            matrix.m[row][2] * vector[2];
    }
    return result;
}

struct OrientationCheckpointHeader {
    char magic[16];
    std::uint32_t version = 1;
    std::uint64_t signature = 0;
    std::int32_t next_orientation = 0;
    std::int32_t ntheta = 0;
    std::int32_t system_dofs = 0;
    std::int32_t total_iterations = 0;
    std::int32_t maximum_orientation_iterations = 0;
    std::int32_t warm_started_solves = 0;
    std::int32_t inner_applications = 0;
    std::int32_t inner_iterations = 0;
    double maximum_residual = 0.0;
    double solve_seconds = 0.0;
    double farfield_seconds = 0.0;
    double inner_seconds = 0.0;
    double loop_seconds = 0.0;
};

bool load_orientation_checkpoint(
    const std::string& path,
    std::uint64_t signature,
    int ntheta,
    int system_dofs,
    OrientationCheckpointHeader& header,
    std::vector<double>& averaged_mueller,
    std::vector<cdouble>& previous_x,
    std::vector<cdouble>& previous_y)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return false;
    input.read(
        reinterpret_cast<char*>(&header), sizeof(header));
    const char expected_magic[16] = "BEM_ORIENT_CP1";
    if (!input ||
        std::memcmp(header.magic, expected_magic, 16) != 0 ||
        header.version != 1 ||
        header.signature != signature ||
        header.ntheta != ntheta ||
        header.system_dofs != system_dofs ||
        header.next_orientation < 0)
        return false;
    averaged_mueller.resize(static_cast<size_t>(16) * ntheta);
    previous_x.resize(system_dofs);
    previous_y.resize(system_dofs);
    input.read(
        reinterpret_cast<char*>(averaged_mueller.data()),
        averaged_mueller.size() * sizeof(double));
    input.read(
        reinterpret_cast<char*>(previous_x.data()),
        previous_x.size() * sizeof(cdouble));
    input.read(
        reinterpret_cast<char*>(previous_y.data()),
        previous_y.size() * sizeof(cdouble));
    return static_cast<bool>(input);
}

void save_orientation_checkpoint(
    const std::string& path,
    const OrientationCheckpointHeader& header,
    const std::vector<double>& averaged_mueller,
    const std::vector<cdouble>& previous_x,
    const std::vector<cdouble>& previous_y)
{
    const std::string temporary = path + ".tmp";
    std::ofstream output(
        temporary, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error(
            "cannot create orientation checkpoint " + temporary);
    output.write(
        reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(
        reinterpret_cast<const char*>(averaged_mueller.data()),
        averaged_mueller.size() * sizeof(double));
    output.write(
        reinterpret_cast<const char*>(previous_x.data()),
        previous_x.size() * sizeof(cdouble));
    output.write(
        reinterpret_cast<const char*>(previous_y.data()),
        previous_y.size() * sizeof(cdouble));
    output.close();
    if (!output)
        throw std::runtime_error(
            "failed to write orientation checkpoint " + temporary);
    if (std::rename(temporary.c_str(), path.c_str()) != 0)
        throw std::runtime_error(
            "cannot publish orientation checkpoint " + path);
}

std::uint64_t muller_operator_hash(const MullerFmmOperator& fmm)
{
    std::uint64_t hash = FNV_OFFSET;
    hash_bytes(hash, &fmm.k_exterior, sizeof(fmm.k_exterior));
    hash_bytes(hash, &fmm.k_interior, sizeof(fmm.k_interior));
    hash_bytes(
        hash, &fmm.epsilon_exterior, sizeof(fmm.epsilon_exterior));
    hash_bytes(
        hash, &fmm.epsilon_interior, sizeof(fmm.epsilon_interior));
    hash_bytes(hash, &fmm.mu_exterior, sizeof(fmm.mu_exterior));
    hash_bytes(hash, &fmm.mu_interior, sizeof(fmm.mu_interior));
    hash_bytes(
        hash, &fmm.quadrature_order, sizeof(fmm.quadrature_order));
    hash_bytes(
        hash, &fmm.fmm_near_radius, sizeof(fmm.fmm_near_radius));
    hash_bytes(
        hash, &fmm.fmm_near_fp32, sizeof(fmm.fmm_near_fp32));
    hash_bytes(
        hash, &fmm.mesh.basis_kind, sizeof(fmm.mesh.basis_kind));
    for (const Vec3& node : fmm.mesh.nodes) {
        hash_bytes(hash, &node.x, sizeof(node.x));
        hash_bytes(hash, &node.y, sizeof(node.y));
        hash_bytes(hash, &node.z, sizeof(node.z));
    }
    for (const Vec3& tangent : fmm.mesh.tangent1) {
        hash_bytes(hash, &tangent.x, sizeof(tangent.x));
        hash_bytes(hash, &tangent.y, sizeof(tangent.y));
        hash_bytes(hash, &tangent.z, sizeof(tangent.z));
    }
    for (const Vec3& tangent : fmm.mesh.tangent2) {
        hash_bytes(hash, &tangent.x, sizeof(tangent.x));
        hash_bytes(hash, &tangent.y, sizeof(tangent.y));
        hash_bytes(hash, &tangent.z, sizeof(tangent.z));
    }
    for (const MullerP2Element& element : fmm.mesh.elements) {
        hash_bytes(
            hash, element.nodes.data(),
            element.nodes.size() * sizeof(int));
        hash_bytes(
            hash, element.topology_vertices.data(),
            element.topology_vertices.size() * sizeof(int));
        hash_bytes(
            hash, element.topology_edges.data(),
            element.topology_edges.size() * sizeof(int));
        hash_bytes(
            hash, element.edge_orientations.data(),
            element.edge_orientations.size() * sizeof(int));
    }
    return hash;
}

int run_orientation_average(
    MullerFmmOperator& fmm,
    MullerMbjPreconditioner& mbj,
    cdouble wave_number,
    double ka,
    double refractive_real,
    int refinement,
    int prism_sides,
    int alpha_count,
    int beta_count,
    int gamma_count,
    int symmetry_order,
    bool warm_start,
    double warm_start_max_angle_degrees,
    bool pfft_fgmres,
    int digits,
    int max_leaf,
    double pfft_inner_tolerance,
    int pfft_inner_iterations,
    int pfft_outer_restart,
    double tolerance,
    int maximum_iterations,
    int gmres_restart,
    int ntheta,
    double setup_seconds,
    double mbj_setup_seconds,
    bool checkpoint_enabled,
    const char* output_path)
{
    const auto total_start = std::chrono::steady_clock::now();
    const Matvec exact_action =
        [&](const cdouble* input, cdouble* output) {
            fmm.select_fmm_backend();
            fmm.matvec(input, output);
        };
    const Matvec current_action =
        [&](const cdouble* input, cdouble* output) {
            fmm.matvec(input, output);
        };
    const Matvec pfft_action =
        [&](const cdouble* input, cdouble* output) {
            fmm.select_pfft_backend();
            fmm.matvec(input, output);
        };
    double fmm_switch_seconds = 0.0;
    int inner_applications = 0;
    int inner_iterations = 0;
    double inner_seconds = 0.0;
    FlexiblePreconditioner pfft_inverse;
    if (pfft_fgmres) {
        fmm_switch_seconds =
            fmm.switch_pfft_to_fmm(digits, max_leaf, true);
        pfft_inverse =
            [&](const cdouble* inner_rhs, cdouble* output) {
                GmresResult inner = solve_gmres(
                    pfft_action, inner_rhs, fmm.system_dofs,
                    pfft_inner_tolerance, pfft_inner_iterations,
                    pfft_inner_iterations, &mbj,
                    "orient-pFFT-inner");
                std::copy(
                    inner.solution.begin(),
                    inner.solution.end(),
                    output);
                inner_applications++;
                inner_iterations += inner.iterations;
                inner_seconds += inner.seconds;
            };
    }
    if (!fmm.gpu_operator_assembly)
        throw std::runtime_error(
            "orientation averaging requires GPU operator assembly");

    const std::vector<AveragingOrientation> orientations =
        averaging_orientations(
            beta_count, gamma_count, symmetry_order);
    std::vector<double> theta(ntheta);
    std::vector<Vec3> laboratory_directions(ntheta);
    std::vector<Vec3> laboratory_theta_hat(ntheta);
    for (int angle = 0; angle < ntheta; angle++) {
        theta[angle] =
            180.0 * angle / static_cast<double>(ntheta - 1);
        const double radians = theta[angle] * M_PI / 180.0;
        laboratory_directions[angle] =
            Vec3(0.0, std::sin(radians), std::cos(radians));
        laboratory_theta_hat[angle] =
            Vec3(0.0, std::cos(radians), -std::sin(radians));
    }

    std::vector<double> averaged_mueller(
        static_cast<size_t>(16) * ntheta, 0.0);
    std::vector<cdouble> previous_x;
    std::vector<cdouble> previous_y;
    Mat3 previous_rotation = {};
    bool have_previous_rotation = false;
    int total_iterations = 0;
    int maximum_orientation_iterations = 0;
    double maximum_residual = 0.0;
    double solve_seconds = 0.0;
    double farfield_seconds = 0.0;
    int warm_started_solves = 0;
    int represented_orientations =
        alpha_count * beta_count * gamma_count * symmetry_order;
    std::uint64_t checkpoint_signature = FNV_OFFSET;
    const std::uint64_t operator_signature =
        muller_operator_hash(fmm);
    hash_bytes(
        checkpoint_signature, &operator_signature,
        sizeof(operator_signature));
    hash_bytes(checkpoint_signature, &ka, sizeof(ka));
    hash_bytes(
        checkpoint_signature, &refractive_real,
        sizeof(refractive_real));
    hash_bytes(checkpoint_signature, &refinement, sizeof(refinement));
    hash_bytes(checkpoint_signature, &prism_sides, sizeof(prism_sides));
    hash_bytes(checkpoint_signature, &alpha_count, sizeof(alpha_count));
    hash_bytes(checkpoint_signature, &beta_count, sizeof(beta_count));
    hash_bytes(checkpoint_signature, &gamma_count, sizeof(gamma_count));
    hash_bytes(
        checkpoint_signature, &symmetry_order,
        sizeof(symmetry_order));
    hash_bytes(
        checkpoint_signature, &warm_start_max_angle_degrees,
        sizeof(warm_start_max_angle_degrees));
    hash_bytes(
        checkpoint_signature, &warm_start, sizeof(warm_start));
    hash_bytes(
        checkpoint_signature, &pfft_fgmres, sizeof(pfft_fgmres));
    hash_bytes(
        checkpoint_signature, &pfft_inner_tolerance,
        sizeof(pfft_inner_tolerance));
    hash_bytes(
        checkpoint_signature, &pfft_inner_iterations,
        sizeof(pfft_inner_iterations));
    hash_bytes(
        checkpoint_signature, &pfft_outer_restart,
        sizeof(pfft_outer_restart));
    hash_bytes(checkpoint_signature, &tolerance, sizeof(tolerance));
    hash_bytes(
        checkpoint_signature, &maximum_iterations,
        sizeof(maximum_iterations));
    hash_bytes(
        checkpoint_signature, &gmres_restart,
        sizeof(gmres_restart));
    hash_bytes(checkpoint_signature, &ntheta, sizeof(ntheta));
    hash_bytes(
        checkpoint_signature, &fmm.system_dofs,
        sizeof(fmm.system_dofs));
    const std::string orientation_checkpoint_path =
        std::string(output_path) + ".orient.checkpoint";
    int orientation_start = 0;
    double previous_loop_seconds = 0.0;
    bool resumed = false;
    if (checkpoint_enabled) {
        OrientationCheckpointHeader checkpoint;
        if (load_orientation_checkpoint(
                orientation_checkpoint_path,
                checkpoint_signature,
                ntheta,
                fmm.system_dofs,
                checkpoint,
                averaged_mueller,
                previous_x,
                previous_y) &&
            checkpoint.next_orientation <=
                static_cast<int>(orientations.size())) {
            orientation_start = checkpoint.next_orientation;
            total_iterations = checkpoint.total_iterations;
            maximum_orientation_iterations =
                checkpoint.maximum_orientation_iterations;
            warm_started_solves = checkpoint.warm_started_solves;
            inner_applications = checkpoint.inner_applications;
            inner_iterations = checkpoint.inner_iterations;
            maximum_residual = checkpoint.maximum_residual;
            solve_seconds = checkpoint.solve_seconds;
            farfield_seconds = checkpoint.farfield_seconds;
            inner_seconds = checkpoint.inner_seconds;
            previous_loop_seconds = checkpoint.loop_seconds;
            if (orientation_start > 0) {
                previous_rotation =
                    orientations[orientation_start - 1].RT;
                have_previous_rotation = true;
            }
            resumed = orientation_start > 0;
            std::printf(
                "  [orientation checkpoint] resumed at %d/%zu from %s\n",
                orientation_start, orientations.size(),
                orientation_checkpoint_path.c_str());
        }
    }

    for (size_t orientation_index =
             static_cast<size_t>(orientation_start);
         orientation_index < orientations.size();
         orientation_index++) {
        const AveragingOrientation& orientation =
            orientations[orientation_index];
        const Vec3 propagation =
            orientation.RT * Vec3(0.0, 0.0, 1.0);
        const Vec3 electric_x =
            orientation.RT * Vec3(1.0, 0.0, 0.0);
        const Vec3 electric_y =
            orientation.RT * Vec3(0.0, 1.0, 0.0);
        const std::vector<cdouble> rhs_x =
            muller_nodal_planewave_rhs(
                fmm.mesh, wave_number,
                electric_x, propagation, 13);
        const std::vector<cdouble> rhs_y =
            muller_nodal_planewave_rhs(
                fmm.mesh, wave_number,
                electric_y, propagation, 13);
        const double neighbor_distance =
            have_previous_rotation
                ? rotation_distance_squared(
                      previous_rotation, orientation.RT)
                : std::numeric_limits<double>::max();
        const double neighbor_angle =
            have_previous_rotation
                ? 2.0 * std::asin(std::min(
                      1.0,
                      std::sqrt(std::max(0.0, neighbor_distance) / 8.0)))
                : M_PI;
        const bool use_neighbor_guess =
            warm_start && have_previous_rotation &&
            neighbor_angle <=
                warm_start_max_angle_degrees * M_PI / 180.0;
        const std::vector<cdouble>* guess_x =
            use_neighbor_guess && previous_x.size() ==
                static_cast<size_t>(fmm.system_dofs)
                ? &previous_x : nullptr;
        const std::vector<cdouble>* guess_y =
            use_neighbor_guess && previous_y.size() ==
                static_cast<size_t>(fmm.system_dofs)
                ? &previous_y : nullptr;
        warm_started_solves += (guess_x != nullptr) + (guess_y != nullptr);

        char label_x[96];
        char label_y[96];
        std::snprintf(
            label_x, sizeof(label_x),
            "orientation-%04zu-x", orientation_index);
        std::snprintf(
            label_y, sizeof(label_y),
            "orientation-%04zu-y", orientation_index);
        GmresResult solution_x;
        GmresResult solution_y;
        if (pfft_fgmres) {
            solution_x = solve_flexible_gmres(
                exact_action, pfft_inverse,
                rhs_x.data(), fmm.system_dofs,
                tolerance, maximum_iterations,
                pfft_outer_restart, label_x, guess_x);
            solution_y = solve_flexible_gmres(
                exact_action, pfft_inverse,
                rhs_y.data(), fmm.system_dofs,
                tolerance, maximum_iterations,
                pfft_outer_restart, label_y, guess_y);
            fmm.select_fmm_backend();
        } else {
            solution_x = solve_gmres(
                current_action, rhs_x.data(), fmm.system_dofs,
                tolerance, maximum_iterations, gmres_restart,
                &mbj, label_x, guess_x);
            solution_y = solve_gmres(
                current_action, rhs_y.data(), fmm.system_dofs,
                tolerance, maximum_iterations, gmres_restart,
                &mbj, label_y, guess_y);
        }
        previous_x = solution_x.solution;
        previous_y = solution_y.solution;
        previous_rotation = orientation.RT;
        have_previous_rotation = true;
        const int orientation_iterations =
            solution_x.iterations + solution_y.iterations;
        total_iterations += orientation_iterations;
        maximum_orientation_iterations = std::max(
            maximum_orientation_iterations,
            orientation_iterations);
        maximum_residual = std::max(
            maximum_residual,
            std::max(
                solution_x.operator_residual,
                solution_y.operator_residual));
        solve_seconds += solution_x.seconds + solution_y.seconds;

        const auto farfield_start =
            std::chrono::steady_clock::now();
        std::vector<Vec3> particle_directions(
            static_cast<size_t>(alpha_count) * ntheta);
        std::vector<Mat3> rotations(alpha_count);
        for (int alpha_index = 0;
             alpha_index < alpha_count; alpha_index++) {
            const double alpha =
                2.0 * M_PI * alpha_index /
                static_cast<double>(alpha_count);
            const Mat3 rotation =
                euler_rotation(
                    alpha, orientation.beta, orientation.gamma);
            rotations[alpha_index] = rotation;
            const Mat3 inverse = rotation.T();
            for (int angle = 0; angle < ntheta; angle++) {
                particle_directions[
                    static_cast<size_t>(alpha_index) * ntheta +
                    angle] =
                    inverse * laboratory_directions[angle];
            }
        }
        std::vector<cdouble> field_x;
        std::vector<cdouble> field_y;
        fmm.farfield_pair(
            solution_x.solution.data(),
            solution_y.solution.data(),
            particle_directions, field_x, field_y);

        const cdouble amplitude_scale(0.0, -ka);
        for (int alpha_index = 0;
             alpha_index < alpha_count; alpha_index++) {
            const double alpha =
                2.0 * M_PI * alpha_index /
                static_cast<double>(alpha_count);
            const double cosine = std::cos(alpha);
            const double sine = std::sin(alpha);
            std::vector<cdouble> s1(ntheta);
            std::vector<cdouble> s2(ntheta);
            std::vector<cdouble> s3(ntheta);
            std::vector<cdouble> s4(ntheta);
            for (int angle = 0; angle < ntheta; angle++) {
                const size_t direction =
                    static_cast<size_t>(alpha_index) * ntheta + angle;
                cdouble particle_x[3];
                cdouble particle_y[3];
                for (int axis = 0; axis < 3; axis++) {
                    particle_x[axis] =
                        cosine * field_x[3 * direction + axis] -
                        sine * field_y[3 * direction + axis];
                    particle_y[axis] =
                        sine * field_x[3 * direction + axis] +
                        cosine * field_y[3 * direction + axis];
                }
                const std::array<cdouble, 3> laboratory_x =
                    rotate_complex_vector(
                        rotations[alpha_index], particle_x);
                const std::array<cdouble, 3> laboratory_y =
                    rotate_complex_vector(
                        rotations[alpha_index], particle_y);
                const Vec3& theta_hat =
                    laboratory_theta_hat[angle];
                const cdouble theta_x =
                    theta_hat.x * laboratory_x[0] +
                    theta_hat.y * laboratory_x[1] +
                    theta_hat.z * laboratory_x[2];
                const cdouble theta_y =
                    theta_hat.x * laboratory_y[0] +
                    theta_hat.y * laboratory_y[1] +
                    theta_hat.z * laboratory_y[2];
                s1[angle] = amplitude_scale * laboratory_x[0];
                s2[angle] = amplitude_scale * theta_y;
                s3[angle] = amplitude_scale * theta_x;
                s4[angle] = amplitude_scale * laboratory_y[0];
            }
            std::vector<double> sample_mueller;
            amplitude_to_mueller(
                s1, s2, s3, s4, sample_mueller);
            const double sample_weight =
                orientation.weight /
                static_cast<double>(alpha_count);
            for (size_t index = 0;
                 index < averaged_mueller.size(); index++) {
                averaged_mueller[index] +=
                    sample_weight * sample_mueller[index];
            }
        }
        farfield_seconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                farfield_start).count();
        std::printf(
            "  [orientation %zu/%zu] beta=%.3f gamma=%.3f deg, "
            "%d iterations, residual %.3e\n",
            orientation_index + 1, orientations.size(),
            orientation.beta * 180.0 / M_PI,
            orientation.gamma * 180.0 / M_PI,
            orientation_iterations,
            std::max(
                solution_x.operator_residual,
                solution_y.operator_residual));
        std::fflush(stdout);
        if (checkpoint_enabled) {
            OrientationCheckpointHeader checkpoint = {};
            const char magic[16] = "BEM_ORIENT_CP1";
            std::memcpy(checkpoint.magic, magic, 16);
            checkpoint.version = 1;
            checkpoint.signature = checkpoint_signature;
            checkpoint.next_orientation =
                static_cast<int>(orientation_index + 1);
            checkpoint.ntheta = ntheta;
            checkpoint.system_dofs = fmm.system_dofs;
            checkpoint.total_iterations = total_iterations;
            checkpoint.maximum_orientation_iterations =
                maximum_orientation_iterations;
            checkpoint.warm_started_solves = warm_started_solves;
            checkpoint.inner_applications = inner_applications;
            checkpoint.inner_iterations = inner_iterations;
            checkpoint.maximum_residual = maximum_residual;
            checkpoint.solve_seconds = solve_seconds;
            checkpoint.farfield_seconds = farfield_seconds;
            checkpoint.inner_seconds = inner_seconds;
            checkpoint.loop_seconds =
                previous_loop_seconds +
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    total_start).count();
            save_orientation_checkpoint(
                orientation_checkpoint_path,
                checkpoint,
                averaged_mueller,
                previous_x,
                previous_y);
        }
    }

    const double total_seconds =
        previous_loop_seconds +
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            total_start).count();
    std::ofstream output(output_path);
    output << std::setprecision(17)
           << "{\n"
           << "  \"solver\": \"muller_orientation_average\",\n"
           << "  \"ka\": " << ka << ",\n"
           << "  \"ri\": " << refractive_real << ",\n"
           << "  \"refinements\": " << refinement << ",\n"
           << "  \"system_dofs\": " << fmm.system_dofs << ",\n"
           << "  \"tolerance\": " << tolerance << ",\n"
           << "  \"gpu_operator_assembly\": true,\n"
           << "  \"orientation\": {\"alpha_samples\": "
           << alpha_count
           << ", \"beta_nodes\": " << beta_count
           << ", \"gamma_nodes_in_fundamental_sector\": "
           << gamma_count
           << ", \"rotational_symmetry_order\": "
           << symmetry_order
           << ", \"solved_base_orientations\": "
           << orientations.size()
           << ", \"represented_full_orientations\": "
           << represented_orientations
           << ", \"warm_start\": "
           << (warm_start ? "true" : "false")
           << ", \"warm_start_max_neighbor_angle_degrees\": "
           << warm_start_max_angle_degrees
           << ", \"warm_started_solves\": "
           << warm_started_solves
           << ", \"resumed\": "
           << (resumed ? "true" : "false")
           << ", \"checkpoint\": ";
    if (checkpoint_enabled)
        output << "\"" << orientation_checkpoint_path << "\"";
    else
        output << "null";
    output << "},\n"
           << "  \"iterations\": {\"total\": "
           << total_iterations
           << ", \"mean_per_polarization\": "
           << static_cast<double>(total_iterations) /
                  (2.0 * orientations.size())
           << ", \"maximum_per_orientation\": "
           << maximum_orientation_iterations
           << ", \"maximum_residual\": "
           << maximum_residual << "},\n"
           << "  \"pfft_inner\": {\"enabled\": "
           << (pfft_fgmres ? "true" : "false")
           << ", \"applications\": " << inner_applications
           << ", \"iterations\": " << inner_iterations
           << ", \"seconds\": " << inner_seconds << "},\n"
           << "  \"timing\": {\"operator_setup_s\": "
           << setup_seconds
           << ", \"mbj_setup_s\": " << mbj_setup_seconds
           << ", \"fmm_switch_s\": " << fmm_switch_seconds
           << ", \"solve_s\": " << solve_seconds
           << ", \"farfield_s\": " << farfield_seconds
           << ", \"average_loop_s\": " << total_seconds
           << ", \"total_with_setup_s\": "
           << setup_seconds + mbj_setup_seconds + total_seconds
           << "},\n"
           << "  \"theta_degrees\": [";
    for (int angle = 0; angle < ntheta; angle++) {
        if (angle)
            output << ", ";
        output << theta[angle];
    }
    output << "],\n  \"mueller\": [\n";
    for (int row = 0; row < 4; row++) {
        output << "    [";
        for (int column = 0; column < 4; column++) {
            if (column)
                output << ", ";
            output << "[";
            for (int angle = 0; angle < ntheta; angle++) {
                if (angle)
                    output << ", ";
                output << averaged_mueller[
                    (static_cast<size_t>(row) * 4 + column) *
                        ntheta + angle];
            }
            output << "]";
        }
        output << "]" << (row == 3 ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    output.close();
    std::printf(
        "Muller orientation average: %zu base orientations represent "
        "%d full samples, %d iterations, %.3fs solve, %.3fs far field, "
        "max residual %.3e, out=%s\n",
        orientations.size(), represented_orientations,
        total_iterations, solve_seconds, farfield_seconds,
        maximum_residual, output_path);
    fmm.cleanup();
    return maximum_residual <= 2.0 * tolerance ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    int refinement = 2;
    int digits = 5;
    int regular_quadrature = 7;
    int duffy_order = 4;
    int max_leaf = 64;
    int fmm_near_radius = 3;
    int pfft_order = 2;
    double pfft_correction_radius = 2.0;
    bool pfft_correction_radius_explicit = false;
    double pfft_grid_safety = 0.96;
    double hybrid_pfft_tolerance = 1.0e-2;
    double pfft_inner_tolerance = 1.0e-1;
    int pfft_inner_iterations = 20;
    bool pfft_inner_iterations_auto = true;
    int pfft_outer_restart = 12;
    int mbj_nodes = 50;
    bool mbj_nodes_explicit = false;
    int mbj_overlap = 0;
    int mbj_coarse_rank = 0;
    int maximum_iterations = 500;
    int gmres_restart = 0;
    double ka = 6.0;
    double refractive_real = 1.5;
    double tolerance = 1.0e-5;
    bool dense_validation = true;
    bool mbj_only = false;
    bool setup_only = false;
    bool physical_check = false;
    bool cyclic_polarization = false;
    bool cyclic_exact_geometry = false;
    bool mirror_polarization = false;
    bool auto_polarization_symmetry = false;
    bool mirror_symmetric_mesh = false;
    bool sphere_fivefold_axis = false;
    bool sphere_fivefold_polarization = false;
    bool sphere_rotational_farfield = false;
    bool use_pfft = false;
    bool hybrid_pfft_fmm = false;
    bool pfft_fgmres = false;
    bool near_template_reuse = true;
    int orient_average_alpha = 0;
    int orient_average_beta = 0;
    int orient_average_gamma = 0;
    int orient_symmetry_order = 0;
    bool orient_warm_start = true;
    double orient_warm_max_angle_degrees = 25.0;
#ifdef BEM_DEFAULT_FMM_NEAR_FP32
    bool fmm_near_fp32 = true;
#else
    bool fmm_near_fp32 = false;
#endif
    int ntheta = 73;
    const char* iteration_log_path = nullptr;
    const char* checkpoint_path = nullptr;
    bool checkpoint_disabled = false;
    bool allow_checkpoint_migration = false;
    const char* neural_preconditioner_path = nullptr;
    const char* near_correction_cache_path = nullptr;
    const char* mbj_cache_path = nullptr;
    const char* coarse_checkpoint_path = nullptr;
    const char* shape = "sphere";
    const char* obj_file = nullptr;
    int prism_sides = 6;
    double prism_aspect = 1.0;
    double prism_azimuth_degrees = 0.0;
    bool prism_azimuth_explicit = false;
    int edge_refine = 0;
    double feature_angle = 45.0;
    bool edge_mode_explicit = false;
    MullerEdgeMode edge_mode = MullerEdgeMode::Smooth;
    const char* output_path =
        "runs/muller_nodal_fmm_benchmark.json";
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--ref") == 0 && i + 1 < argc)
            refinement = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ka") == 0 && i + 1 < argc)
            ka = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ri") == 0 && i + 1 < argc)
            refractive_real = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--tol") == 0 && i + 1 < argc)
            tolerance = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--digits") == 0 && i + 1 < argc)
            digits = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--quad") == 0 && i + 1 < argc)
            regular_quadrature = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--duffy-order") == 0 &&
                 i + 1 < argc)
            duffy_order = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--max-leaf") == 0 && i + 1 < argc)
            max_leaf = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--fmm-near-radius") == 0 &&
                 i + 1 < argc)
            fmm_near_radius = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--pfft-order") == 0 &&
                 i + 1 < argc)
            pfft_order = std::atoi(argv[++i]);
        else if (std::strcmp(
                     argv[i], "--pfft-correction-radius") == 0 &&
                 i + 1 < argc) {
            pfft_correction_radius = std::atof(argv[++i]);
            pfft_correction_radius_explicit = true;
        }
        else if (std::strcmp(
                     argv[i], "--pfft-grid-safety") == 0 &&
                 i + 1 < argc)
            pfft_grid_safety = std::atof(argv[++i]);
        else if (std::strcmp(
                     argv[i], "--near-correction-cache") == 0 &&
                 i + 1 < argc)
            near_correction_cache_path = argv[++i];
        else if (std::strcmp(
                     argv[i], "--no-near-template-reuse") == 0)
            near_template_reuse = false;
        else if (std::strcmp(argv[i], "--mbj-cache") == 0 &&
                 i + 1 < argc)
            mbj_cache_path = argv[++i];
        else if (std::strcmp(argv[i], "--coarse-checkpoint") == 0 &&
                 i + 1 < argc)
            coarse_checkpoint_path = argv[++i];
        else if (std::strcmp(
                     argv[i], "--hybrid-pfft-tol") == 0 &&
                 i + 1 < argc)
            hybrid_pfft_tolerance = std::atof(argv[++i]);
        else if (std::strcmp(
                     argv[i], "--pfft-inner-tol") == 0 &&
                 i + 1 < argc)
            pfft_inner_tolerance = std::atof(argv[++i]);
        else if (std::strcmp(
                     argv[i], "--pfft-inner-iters") == 0 &&
                 i + 1 < argc) {
            const char* value = argv[++i];
            if (std::strcmp(value, "auto") == 0) {
                pfft_inner_iterations_auto = true;
            } else {
                pfft_inner_iterations = std::atoi(value);
                pfft_inner_iterations_auto = false;
            }
        }
        else if (std::strcmp(
                     argv[i], "--pfft-outer-restart") == 0 &&
                 i + 1 < argc)
            pfft_outer_restart = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--pfft") == 0)
            use_pfft = true;
        else if (std::strcmp(argv[i], "--hybrid-pfft-fmm") == 0) {
            hybrid_pfft_fmm = true;
            use_pfft = true;
        }
        else if (std::strcmp(argv[i], "--pfft-fgmres") == 0) {
            pfft_fgmres = true;
            use_pfft = true;
        }
        else if (std::strcmp(argv[i], "--operator-backend") == 0 &&
                 i + 1 < argc) {
            const char* backend = argv[++i];
            if (std::strcmp(backend, "fmm") == 0)
                use_pfft = false;
            else if (std::strcmp(backend, "pfft") == 0)
                use_pfft = true;
            else {
                std::fprintf(
                    stderr,
                    "--operator-backend must be fmm or pfft\n");
                return 2;
            }
        }
        else if (std::strcmp(argv[i], "--fmm-near-fp32") == 0)
            fmm_near_fp32 = true;
        else if (std::strcmp(argv[i], "--fmm-near-fp64") == 0)
            fmm_near_fp32 = false;
        else if (std::strcmp(argv[i], "--mbj-nodes") == 0 &&
                 i + 1 < argc) {
            mbj_nodes = std::atoi(argv[++i]);
            mbj_nodes_explicit = true;
        }
        else if (std::strcmp(argv[i], "--mbj-overlap") == 0 &&
                 i + 1 < argc)
            mbj_overlap = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--mbj-coarse-rank") == 0 &&
                 i + 1 < argc)
            mbj_coarse_rank = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--max-iters") == 0 && i + 1 < argc)
            maximum_iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--gmres-restart") == 0 &&
                 i + 1 < argc)
            gmres_restart = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iteration-log") == 0 &&
                 i + 1 < argc)
            iteration_log_path = argv[++i];
        else if (std::strcmp(argv[i], "--iteration-log-every") == 0 &&
                 i + 1 < argc)
            iteration_log_every = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--checkpoint") == 0 &&
                 i + 1 < argc)
            checkpoint_path = argv[++i];
        else if (std::strcmp(argv[i], "--no-checkpoint") == 0)
            checkpoint_disabled = true;
        else if (std::strcmp(
                     argv[i], "--allow-checkpoint-migration") == 0)
            allow_checkpoint_migration = true;
        else if (std::strcmp(argv[i], "--no-dense-validation") == 0)
            dense_validation = false;
        else if (std::strcmp(argv[i], "--mbj-only") == 0)
            mbj_only = true;
        else if (std::strcmp(argv[i], "--setup-only") == 0)
            setup_only = true;
        else if (std::strcmp(argv[i], "--physical-check") == 0)
            physical_check = true;
        else if (std::strcmp(argv[i], "--orient-average") == 0 &&
                 i + 3 < argc) {
            orient_average_alpha = std::atoi(argv[++i]);
            orient_average_beta = std::atoi(argv[++i]);
            orient_average_gamma = std::atoi(argv[++i]);
            physical_check = true;
        }
        else if (std::strcmp(
                     argv[i], "--orient-symmetry-order") == 0 &&
                 i + 1 < argc)
            orient_symmetry_order = std::atoi(argv[++i]);
        else if (std::strcmp(
                     argv[i], "--orient-zero-start") == 0)
            orient_warm_start = false;
        else if (std::strcmp(
                     argv[i], "--orient-warm-max-angle") == 0 &&
                 i + 1 < argc)
            orient_warm_max_angle_degrees = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--cyclic-polarization") == 0)
            cyclic_polarization = true;
        else if (std::strcmp(
                     argv[i], "--cyclic-exact-geometry") == 0)
            cyclic_exact_geometry = true;
        else if (std::strcmp(argv[i], "--mirror-polarization") == 0)
            mirror_polarization = true;
        else if (std::strcmp(
                     argv[i], "--auto-polarization-symmetry") == 0)
            auto_polarization_symmetry = true;
        else if (std::strcmp(
                     argv[i], "--mirror-symmetric-mesh") == 0)
            mirror_symmetric_mesh = true;
        else if (std::strcmp(
                     argv[i], "--sphere-fivefold-axis") == 0)
            sphere_fivefold_axis = true;
        else if (std::strcmp(
                     argv[i], "--sphere-fivefold-polarization") == 0) {
            sphere_fivefold_axis = true;
            sphere_fivefold_polarization = true;
        }
        else if (std::strcmp(
                     argv[i], "--sphere-rotational-farfield") == 0) {
            sphere_fivefold_axis = true;
            sphere_rotational_farfield = true;
        }
        else if (std::strcmp(argv[i], "--ntheta") == 0 &&
                 i + 1 < argc)
            ntheta = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--neural-prec") == 0 &&
                 i + 1 < argc)
            neural_preconditioner_path = argv[++i];
        else if (std::strcmp(argv[i], "--shape") == 0 && i + 1 < argc)
            shape = argv[++i];
        else if (std::strcmp(argv[i], "--obj") == 0 && i + 1 < argc) {
            obj_file = argv[++i];
            shape = "obj";
        }
        else if (std::strcmp(argv[i], "--sides") == 0 && i + 1 < argc)
            prism_sides = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--aspect") == 0 && i + 1 < argc)
            prism_aspect = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--prism-azimuth-deg") == 0 &&
                 i + 1 < argc) {
            prism_azimuth_degrees = std::atof(argv[++i]);
            prism_azimuth_explicit = true;
        }
        else if (std::strcmp(argv[i], "--edge-refine") == 0 &&
                 i + 1 < argc)
            edge_refine = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--feature-angle") == 0 &&
                 i + 1 < argc)
            feature_angle = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--edge-mode") == 0 &&
                 i + 1 < argc) {
            edge_mode_explicit = true;
            const char* mode = argv[++i];
            if (std::strcmp(mode, "smooth") == 0)
                edge_mode = MullerEdgeMode::Smooth;
            else if (std::strcmp(mode, "split") == 0)
                edge_mode = MullerEdgeMode::SplitFeatureEdges;
            else if (std::strcmp(mode, "hdiv") == 0 ||
                     std::strcmp(mode, "hdiv-bdm1") == 0)
                edge_mode = MullerEdgeMode::HDivBdm1;
            else {
                std::fprintf(
                    stderr,
                    "--edge-mode must be smooth, split, or hdiv\n");
                return 2;
            }
        }
        else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc)
            output_path = argv[++i];
    }

    const std::string checkpoint_base = checkpoint_disabled
        ? std::string()
        : (checkpoint_path
               ? std::string(checkpoint_path)
               : std::string(output_path) + ".checkpoint");
    const bool prism_mode = std::strcmp(shape, "prism") == 0;
    const bool cube_mode = std::strcmp(shape, "cube") == 0;
    const bool sphere_mode = std::strcmp(shape, "sphere") == 0;
    const bool obj_mode = std::strcmp(shape, "obj") == 0;
    const bool polygon_symmetry_mode = prism_mode || cube_mode;
    const bool sharp_mode = polygon_symmetry_mode || obj_mode;
    if (cube_mode) {
        prism_sides = 4;
        prism_aspect = 1.0;
    }
    if (hybrid_pfft_fmm || pfft_fgmres)
        use_pfft = true;
    if (!prism_mode && !cube_mode && !sphere_mode && !obj_mode) {
        std::fprintf(stderr, "--shape must be sphere, cube, prism, or obj\n");
        return 2;
    }
    if (auto_polarization_symmetry) {
        if (!physical_check) {
            std::fprintf(
                stderr,
                "--auto-polarization-symmetry requires "
                "--physical-check\n");
            return 2;
        }
        if (cyclic_polarization || cyclic_exact_geometry ||
            mirror_polarization || sphere_fivefold_axis ||
            sphere_fivefold_polarization ||
            sphere_rotational_farfield) {
            std::fprintf(
                stderr,
                "--auto-polarization-symmetry cannot be combined "
                "with an explicit polarization-symmetry mode\n");
            return 2;
        }
        if (cube_mode) {
            cyclic_polarization = true;
            cyclic_exact_geometry = true;
        } else if (prism_mode) {
            mirror_polarization = true;
        } else if (sphere_mode) {
            sphere_fivefold_axis = true;
            sphere_fivefold_polarization = true;
        } else {
            std::printf(
                "  [auto symmetry] OBJ has no declared exact "
                "polarization symmetry; solving both polarizations\n");
        }
    }
    if (obj_mode && !obj_file) {
        std::fprintf(stderr, "--shape obj requires --obj FILE\n");
        return 2;
    }
    if (obj_mode && pfft_fgmres) {
        if (!pfft_correction_radius_explicit)
            pfft_correction_radius = 0.0;
        if (!mbj_nodes_explicit)
            mbj_nodes = 8;
        std::printf(
            "  [OBJ strict pFFT defaults] correction radius %.1f, "
            "MBJ block %d%s\n",
            pfft_correction_radius, mbj_nodes,
            (pfft_correction_radius_explicit || mbj_nodes_explicit)
                ? " (explicit values preserved)" : "");
    }
    if (cyclic_polarization &&
        (!polygon_symmetry_mode || prism_sides < 3 || !physical_check)) {
        std::fprintf(
            stderr,
            "--cyclic-polarization requires --physical-check and a "
            "regular prism with at least three sides\n");
        return 2;
    }
    if (mirror_polarization &&
        (!polygon_symmetry_mode || prism_sides < 3 || !physical_check)) {
        std::fprintf(
            stderr,
            "--mirror-polarization requires --physical-check and a "
            "regular prism with at least three sides\n");
        return 2;
    }
    if (mirror_polarization && cyclic_polarization) {
        std::fprintf(
            stderr,
            "--mirror-polarization and --cyclic-polarization are "
            "mutually exclusive\n");
        return 2;
    }
    if (sphere_fivefold_axis && (!sphere_mode || !physical_check)) {
        std::fprintf(
            stderr,
            "--sphere-fivefold-axis requires --shape sphere and "
            "--physical-check\n");
        return 2;
    }
    if (sphere_fivefold_polarization &&
        (cyclic_polarization || mirror_polarization)) {
        std::fprintf(
            stderr,
            "--sphere-fivefold-polarization cannot be combined with "
            "prism polarization symmetry\n");
        return 2;
    }
    if (sphere_rotational_farfield &&
        (sphere_fivefold_polarization ||
         cyclic_polarization || mirror_polarization)) {
        std::fprintf(
            stderr,
            "--sphere-rotational-farfield cannot be combined with "
            "a second-polarization reconstruction mode\n");
        return 2;
    }
    if (coarse_checkpoint_path &&
        (!sphere_mode || refinement < 1 ||
         edge_mode == MullerEdgeMode::HDivBdm1)) {
        std::fprintf(
            stderr,
            "--coarse-checkpoint requires a nodal sphere with ref >= 1\n");
        return 2;
    }
    if (cyclic_exact_geometry && !cyclic_polarization) {
        std::fprintf(
            stderr,
            "--cyclic-exact-geometry requires --cyclic-polarization\n");
        return 2;
    }
    if (cube_mode && edge_refine != 0) {
        std::fprintf(
            stderr,
            "--edge-refine is intentionally disabled for --shape cube "
            "because it would destroy the structured face grid\n");
        return 2;
    }
    if (mbj_nodes < 1 || mbj_overlap < 0 || gmres_restart < 0 ||
        iteration_log_every < 1 ||
        mbj_coarse_rank < 0 || mbj_coarse_rank > 64) {
        std::fprintf(
            stderr,
            "--mbj-nodes must be positive and "
            "--iteration-log-every must be positive; "
            "--mbj-overlap/--mbj-coarse-rank/--gmres-restart must be "
            "non-negative; "
            "coarse rank is limited to 64\n");
        return 2;
    }
    if (ntheta < 2) {
        std::fprintf(stderr, "--ntheta must be at least 2\n");
        return 2;
    }
    const bool orientation_average =
        orient_average_alpha > 0 ||
        orient_average_beta > 0 ||
        orient_average_gamma > 0;
    if (orientation_average &&
        (orient_average_alpha < 1 ||
         orient_average_beta < 1 ||
         orient_average_gamma < 1 ||
         orient_warm_max_angle_degrees <= 0.0 ||
         hybrid_pfft_fmm ||
         !mbj_only ||
         dense_validation ||
         neural_preconditioner_path)) {
        std::fprintf(
            stderr,
            "--orient-average Na Nb Ng requires positive counts, "
            "--mbj-only, --no-dense-validation, no neural preconditioner, "
            "and does not support --hybrid-pfft-fmm\n");
        return 2;
    }
    if (orientation_average && orient_symmetry_order <= 0)
        orient_symmetry_order =
            polygon_symmetry_mode ? prism_sides : 1;
    if (orientation_average &&
        (orient_symmetry_order < 1 ||
         (polygon_symmetry_mode
              ? prism_sides % orient_symmetry_order != 0
              : orient_symmetry_order != 1))) {
        std::fprintf(
            stderr,
            "orientation symmetry order must be one for sphere/OBJ or "
            "divide the regular-prism side count\n");
        return 2;
    }
    if (pfft_order < 2 || pfft_order > 5) {
        std::fprintf(stderr, "--pfft-order must be in [2,5]\n");
        return 2;
    }
    if (pfft_correction_radius < 0.0 ||
        pfft_grid_safety <= 0.5 || pfft_grid_safety > 1.0 ||
        hybrid_pfft_tolerance <= 0.0 ||
        hybrid_pfft_tolerance >= 1.0 ||
        pfft_inner_tolerance <= 0.0 ||
        pfft_inner_tolerance >= 1.0 ||
        (!pfft_inner_iterations_auto &&
         pfft_inner_iterations < 1) ||
        pfft_outer_restart < 1) {
        std::fprintf(
            stderr,
            "--pfft-correction-radius must be non-negative and "
            "--pfft-grid-safety must be in (0.5,1]; "
            "--hybrid-pfft-tol/--pfft-inner-tol must be in (0,1); "
            "--pfft-inner-iters must be auto or positive; "
            "--pfft-outer-restart must be positive\n");
        return 2;
    }
    if ((hybrid_pfft_fmm || pfft_fgmres) &&
        (!mbj_only || setup_only || neural_preconditioner_path)) {
        std::fprintf(
            stderr,
            "hybrid pFFT modes currently require --mbj-only and do "
            "not support --setup-only or --neural-prec\n");
        return 2;
    }
    if (hybrid_pfft_fmm && pfft_fgmres) {
        std::fprintf(
            stderr,
            "--hybrid-pfft-fmm and --pfft-fgmres are mutually "
            "exclusive\n");
        return 2;
    }
    if (regular_quadrature != 4 && regular_quadrature != 7 &&
        regular_quadrature != 13) {
        std::fprintf(stderr, "--quad must be 4, 7, or 13\n");
        return 2;
    }
    if (sharp_mode && !edge_mode_explicit)
        edge_mode = MullerEdgeMode::HDivBdm1;
    if (sharp_mode &&
        edge_mode != MullerEdgeMode::HDivBdm1) {
        std::fprintf(
            stderr,
            "Warning: the nodal P2 Muller sharp-edge treatment is "
            "experimental and is not H(div)-conforming; mesh "
            "self-convergence is not an independent physical "
            "validation.\n");
    } else if (sharp_mode) {
        std::fprintf(
            stderr,
            "Info: using the H(div)-conforming BDM1 edge basis; "
            "sharp-edge physical convergence still requires "
            "independent validation.\n");
    }
    if (mirror_polarization && !prism_azimuth_explicit)
        prism_azimuth_degrees =
            90.0 / static_cast<double>(prism_sides);
    if (mirror_polarization)
        mirror_symmetric_mesh = true;
    if (iteration_log_path) {
        bool iteration_log_has_content = false;
        {
            std::ifstream existing_log(
                iteration_log_path,
                std::ios::binary | std::ios::ate);
            iteration_log_has_content =
                existing_log &&
                existing_log.tellg() > std::streampos(0);
        }
        iteration_log.open(
            iteration_log_path, std::ios::out | std::ios::app);
        if (!iteration_log) {
            std::fprintf(
                stderr, "failed to open iteration log: %s\n",
                iteration_log_path);
            return 2;
        }
        if (!iteration_log_has_content) {
            iteration_log
                << "solver,event,iteration,projected_residual,"
                   "operator_residual,matvec_s,preconditioner_s,"
                   "orthogonalization_s,elapsed_s\n";
        }
        iteration_log.flush();
    }
    Mesh mesh;
    if (cube_mode) {
        mesh = structured_cube(refinement, 1.0);
    } else if (prism_mode) {
        mesh = regular_prism(
            prism_sides, prism_aspect, refinement,
            1.0, edge_refine, mirror_symmetric_mesh);
    } else if (obj_mode) {
        mesh = load_obj(obj_file);
        normalize_mesh(mesh);
        for (int subdivision = 0; subdivision < refinement; subdivision++)
            mesh = subdivide_flat(mesh);
    } else {
        mesh = icosphere(1.0, refinement);
    }
    if (sphere_fivefold_axis)
        align_icosphere_fivefold_axis_to_z(mesh);
    if (polygon_symmetry_mode && prism_azimuth_degrees != 0.0) {
        rotate_mesh_about_z(
            mesh, prism_azimuth_degrees * M_PI / 180.0);
    }
    {
        double coordinate_min[3] = {1.0e300, 1.0e300, 1.0e300};
        double coordinate_max[3] = {-1.0e300, -1.0e300, -1.0e300};
        for (const Vec3& vertex : mesh.verts) {
            coordinate_min[0] = std::min(coordinate_min[0], vertex.x);
            coordinate_min[1] = std::min(coordinate_min[1], vertex.y);
            coordinate_min[2] = std::min(coordinate_min[2], vertex.z);
            coordinate_max[0] = std::max(coordinate_max[0], vertex.x);
            coordinate_max[1] = std::max(coordinate_max[1], vertex.y);
            coordinate_max[2] = std::max(coordinate_max[2], vertex.z);
        }
        const double root_box_size = 1.001 * std::max(
            coordinate_max[0] - coordinate_min[0],
            std::max(
                coordinate_max[1] - coordinate_min[1],
                coordinate_max[2] - coordinate_min[2]));
        const double largest_wave_number =
            ka * std::max(1.0, std::abs(refractive_real));
        const int depth5_order = fmm_truncation_order(
            largest_wave_number, root_box_size / 32.0, digits);
        const int depth6_order = fmm_truncation_order(
            largest_wave_number, root_box_size / 64.0, digits);
        const std::size_t combined_quadrature_points =
            2ULL * static_cast<std::size_t>(mesh.nt()) *
            static_cast<std::size_t>(regular_quadrature);
        const std::size_t depth5_boxes = 1ULL << 15;
        const int depth5_safe_leaf = static_cast<int>(
            (combined_quadrature_points + depth5_boxes - 1) /
            depth5_boxes);
        if (depth6_order < depth5_order &&
            max_leaf < depth5_safe_leaf) {
            std::printf(
                "  [FMM accuracy guard] max-leaf %d -> %d to keep "
                "depth <= 5 and truncation order p >= %d "
                "(depth-6 estimate p=%d)\n",
                max_leaf, depth5_safe_leaf,
                depth5_order, depth6_order);
            max_leaf = depth5_safe_leaf;
        }
    }
    MullerP2BuildOptions build_options;
    build_options.project_edge_nodes_to_sphere = sphere_mode;
    build_options.azimuthal_tangent_frame =
        sphere_fivefold_polarization;
    build_options.edge_mode = edge_mode;
    build_options.feature_angle_degrees = feature_angle;
    const cdouble wave_number(ka, 0.0);
    const cdouble refractive_index(refractive_real, 0.0);
    const double max_element_edge = maximum_mesh_edge(mesh);
    const double ka_h_element = ka * max_element_edge;
    const double p2_nodes_per_wavelength =
        ka_h_element > 0.0 ? 4.0 * M_PI / ka_h_element : 0.0;
    size_t gpu_free_before = 0;
    size_t gpu_total_bytes = 0;
    const bool gpu_memory_before_valid =
        cudaMemGetInfo(&gpu_free_before, &gpu_total_bytes) ==
        cudaSuccess;
    MullerFmmOperator fmm;
    const auto fmm_setup_start = std::chrono::steady_clock::now();
    fmm.init(
        mesh, wave_number, refractive_index, build_options,
        regular_quadrature, duffy_order, digits, max_leaf,
        use_pfft, pfft_order, pfft_correction_radius,
        pfft_grid_safety, near_correction_cache_path,
        fmm_near_radius, near_template_reuse);
    fmm.set_fmm_near_fp32(fmm_near_fp32);
    if (pfft_inner_iterations_auto) {
        const size_t point_count = fmm.quadrature.size();
        if (point_count <= 16384)
            pfft_inner_iterations = 12;
        else if (point_count <= 65536)
            pfft_inner_iterations = 20;
        else
            pfft_inner_iterations = 24;
        if (pfft_fgmres) {
            std::printf(
                "  [pFFT-inner] auto limit: %d iterations "
                "for %zu quadrature points\n",
                pfft_inner_iterations, point_count);
        }
    }
    const double fmm_setup_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - fmm_setup_start).count();
    std::uint64_t checkpoint_signature = FNV_OFFSET;
    hash_bytes(
        checkpoint_signature, shape, std::strlen(shape));
    if (obj_file)
        hash_bytes(
            checkpoint_signature, obj_file, std::strlen(obj_file));
    if (!mesh.verts.empty())
        hash_bytes(
            checkpoint_signature, mesh.verts.data(),
            mesh.verts.size() * sizeof(mesh.verts.front()));
    if (!mesh.tris.empty())
        hash_bytes(
            checkpoint_signature, mesh.tris.data(),
            mesh.tris.size() * sizeof(mesh.tris.front()));
    hash_bytes(
        checkpoint_signature, &refinement, sizeof(refinement));
    hash_bytes(checkpoint_signature, &ka, sizeof(ka));
    hash_bytes(
        checkpoint_signature, &refractive_real,
        sizeof(refractive_real));
    hash_bytes(
        checkpoint_signature, &regular_quadrature,
        sizeof(regular_quadrature));
    hash_bytes(
        checkpoint_signature, &duffy_order,
        sizeof(duffy_order));
    hash_bytes(checkpoint_signature, &digits, sizeof(digits));
    hash_bytes(
        checkpoint_signature, &max_leaf, sizeof(max_leaf));
    hash_bytes(
        checkpoint_signature, &fmm_near_radius,
        sizeof(fmm_near_radius));
    hash_bytes(
        checkpoint_signature, &fmm_near_fp32,
        sizeof(fmm_near_fp32));
    hash_bytes(
        checkpoint_signature, &prism_sides,
        sizeof(prism_sides));
    hash_bytes(
        checkpoint_signature, &prism_aspect,
        sizeof(prism_aspect));
    hash_bytes(
        checkpoint_signature, &prism_azimuth_degrees,
        sizeof(prism_azimuth_degrees));
    hash_bytes(
        checkpoint_signature, &edge_refine,
        sizeof(edge_refine));
    const int edge_mode_value = static_cast<int>(edge_mode);
    hash_bytes(
        checkpoint_signature, &edge_mode_value,
        sizeof(edge_mode_value));
    hash_bytes(
        checkpoint_signature, &fmm.system_dofs,
        sizeof(fmm.system_dofs));
    const std::size_t quadrature_points =
        fmm.quadrature.size();
    hash_bytes(
        checkpoint_signature, &quadrature_points,
        sizeof(quadrature_points));
    for (const Vec3& vertex : mesh.verts) {
        hash_bytes(
            checkpoint_signature, &vertex.x,
            sizeof(vertex.x));
        hash_bytes(
            checkpoint_signature, &vertex.y,
            sizeof(vertex.y));
        hash_bytes(
            checkpoint_signature, &vertex.z,
            sizeof(vertex.z));
    }
    if (!mesh.tris.empty()) {
        hash_bytes(
            checkpoint_signature, mesh.tris.data(),
            mesh.tris.size() * sizeof(mesh.tris[0]));
    }
    const auto solver_checkpoint =
        [&](const char* label) {
            SolverCheckpointOptions options;
            options.path =
                checkpoint_stage_path(checkpoint_base, label);
            options.signature = checkpoint_signature;
            options.allow_signature_mismatch =
                allow_checkpoint_migration;
            return options;
        };
    MullerMbjPreconditioner mbj;
    const auto mbj_setup_start = std::chrono::steady_clock::now();
    if (mbj_cache_path) {
        mbj.build_cached(
            fmm, mbj_nodes, mbj_overlap, mbj_cache_path);
    } else {
        mbj.build(fmm, mbj_nodes, mbj_overlap);
    }
    const double mbj_local_setup_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - mbj_setup_start).count();
    mbj.build_coarse(fmm, mbj_coarse_rank);
    const double mbj_coarse_setup_seconds = mbj.coarse_setup_seconds;
    size_t gpu_free_after = 0;
    size_t gpu_total_after = 0;
    const bool gpu_memory_after_valid =
        cudaMemGetInfo(&gpu_free_after, &gpu_total_after) ==
        cudaSuccess;
    const double gpu_memory_delta_mb =
        gpu_memory_before_valid && gpu_memory_after_valid &&
                gpu_free_before >= gpu_free_after
            ? static_cast<double>(gpu_free_before - gpu_free_after) /
                  (1024.0 * 1024.0)
            : -1.0;

    if (setup_only) {
        std::ofstream output(output_path);
        output << std::setprecision(17)
               << "{\n"
               << "  \"solver\": \"muller_"
               << (edge_mode == MullerEdgeMode::HDivBdm1
                       ? "hdiv_bdm1_" : "nodal_p2_")
               << fmm.backend_name() << "\",\n"
               << "  \"operator_backend\": \""
               << fmm.backend_name() << "\",\n"
#ifdef BEM_PFFT_FP32
               << "  \"pfft_fft_precision\": \"fp32\",\n"
#else
               << "  \"pfft_fft_precision\": \"fp64\",\n"
#endif
               << "  \"pfft_order\": " << pfft_order << ",\n"
               << "  \"pfft_correction_radius_cells\": "
               << pfft_correction_radius << ",\n"
               << "  \"pfft_grid_safety\": "
               << pfft_grid_safety << ",\n"
               << "  \"hybrid_pfft_tolerance\": "
               << hybrid_pfft_tolerance << ",\n"
               << "  \"setup_only\": true,\n"
               << "  \"shape\": \"" << shape << "\",\n"
               << "  \"obj_file\": "
               << (obj_file
                       ? std::string("\"") + obj_file + "\""
                       : std::string("null"))
               << ",\n"
               << "  \"edge_mode\": \""
               << muller_edge_mode_name(edge_mode) << "\",\n"
               << "  \"hdiv_conforming\": "
               << (edge_mode == MullerEdgeMode::HDivBdm1
                       ? "true" : "false") << ",\n"
               << "  \"sharp_edge_formulation_validated\": "
               << (sharp_mode ? "false" : "true") << ",\n"
               << "  \"prism_azimuth_degrees\": "
               << prism_azimuth_degrees << ",\n"
               << "  \"mirror_symmetric_mesh\": "
               << (mirror_symmetric_mesh ? "true" : "false")
               << ",\n"
               << "  \"ka\": " << ka << ",\n"
               << "  \"ri\": " << refractive_real << ",\n"
               << "  \"refinements\": " << refinement << ",\n"
               << "  \"edge_refine_requested\": "
               << mesh.edge_refine_requested << ",\n"
               << "  \"edge_refine_applied\": "
               << mesh.edge_refine_applied << ",\n"
               << "  \"edge_refine_uniform_fallback\": "
               << (mesh.edge_refine_uniform_fallback ? "true" : "false")
               << ",\n"
               << "  \"system_dofs\": " << fmm.system_dofs << ",\n"
               << "  \"quadrature_points\": "
               << fmm.quadrature.size() << ",\n"
               << "  \"regular_quadrature\": "
               << regular_quadrature << ",\n"
               << "  \"duffy_order\": " << duffy_order << ",\n"
               << "  \"fmm_digits\": "
               << std::min(digits, 5) << ",\n"
               << "  \"fmm_digits_requested\": " << digits << ",\n"
               << "  \"fmm_max_leaf_points\": " << max_leaf << ",\n"
               << "  \"fmm_near_radius\": "
               << fmm_near_radius << ",\n"
               << "  \"fmm_near_precision\": \""
               << (fmm_near_fp32 ? "fp32" : "fp64") << "\",\n"
               << "  \"gmres_restart\": " << gmres_restart << ",\n"
               << "  \"cyclic_polarization_requested\": "
               << (cyclic_polarization ? "true" : "false") << ",\n"
               << "  \"auto_polarization_symmetry_requested\": "
               << (auto_polarization_symmetry
                       ? "true" : "false") << ",\n"
               << "  \"sphere_fivefold_axis\": "
               << (sphere_fivefold_axis ? "true" : "false") << ",\n"
               << "  \"sphere_fivefold_polarization_requested\": "
               << (sphere_fivefold_polarization ? "true" : "false")
               << ",\n"
               << "  \"sphere_rotational_farfield\": "
               << (sphere_rotational_farfield ? "true" : "false")
               << ",\n"
               << "  \"max_element_edge\": "
               << max_element_edge << ",\n"
               << "  \"ka_h_element\": " << ka_h_element << ",\n"
               << "  \"p2_nodes_per_wavelength_min\": "
               << p2_nodes_per_wavelength << ",\n"
               << "  \"fmm_setup_s\": " << fmm_setup_seconds << ",\n"
               << "  \"fmm_setup_breakdown\": {\"geometry_s\": "
               << fmm.geometry_setup_seconds
               << ", \"near_correction_s\": "
               << fmm.near_correction_setup_seconds
               << ", \"engines_s\": "
               << fmm.fmm_engine_setup_seconds
               << ", \"near_correction_colors\": "
               << fmm.near_correction_colors
               << ", \"near_correction_pairs\": "
               << fmm.near_correction_pairs
               << ", \"near_correction_unique_templates\": "
               << fmm.near_correction_unique_templates
               << ", \"near_correction_template_reuse\": "
               << (fmm.near_correction_template_reuse
                       ? "true" : "false")
               << "},\n"
               << "  \"near_correction_cache\": {\"enabled\": "
               << (near_correction_cache_path ? "true" : "false")
               << ", \"hit\": "
               << (fmm.near_correction_cache_hit
                       ? "true" : "false")
               << ", \"entries\": "
               << fmm.correction.entries.size() << "},\n"
               << "  \"mbj_local_setup_s\": "
               << mbj_local_setup_seconds << ",\n"
               << "  \"mbj_setup_breakdown\": {\"ordering_s\": "
               << mbj.ordering_seconds
               << ", \"assembly_s\": " << mbj.assembly_seconds
               << ", \"factorization_s\": "
               << mbj.factorization_seconds
               << ", \"threads\": " << mbj.setup_threads
               << ", \"cache_enabled\": "
               << (mbj_cache_path ? "true" : "false")
               << ", \"cache_hit\": "
               << (mbj.cache_hit ? "true" : "false")
               << ", \"cache_io_s\": "
               << mbj.cache_io_seconds << "},\n"
               << "  \"mbj\": {\"nodes_per_block\": " << mbj_nodes
               << ", \"overlap_nodes\": " << mbj_overlap
               << ", \"coarse_rank\": " << mbj.coarse_rank
               << ", \"coarse_setup_s\": "
               << mbj_coarse_setup_seconds
               << ", \"storage_mb\": " << mbj.storage_megabytes()
               << "},\n"
               << "  \"gpu_total_mb\": "
               << (gpu_memory_before_valid
                       ? static_cast<double>(gpu_total_bytes) /
                             (1024.0 * 1024.0)
                       : -1.0)
               << ",\n"
               << "  \"gpu_memory_delta_mb\": "
               << gpu_memory_delta_mb << "\n"
               << "}\n";
        output.close();
        std::printf(
            "Muller setup only: ref=%d, dofs=%d, quadrature=%zu, "
            "%s %.3fs, MBJ %.3fs, GPU delta %.1f MiB, out=%s\n",
            refinement, fmm.system_dofs, fmm.quadrature.size(),
            fmm.backend_name(),
            fmm_setup_seconds, mbj_local_setup_seconds,
            gpu_memory_delta_mb, output_path);
        fmm.cleanup();
        return 0;
    }

    MullerMbjPreconditioner neural;
    double neural_load_seconds = 0.0;
    if (neural_preconditioner_path) {
        const auto neural_load_start =
            std::chrono::steady_clock::now();
        neural.load_neural(fmm, neural_preconditioner_path);
        neural_load_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                neural_load_start).count();
    }
    std::vector<cdouble> rhs = muller_nodal_planewave_rhs(
        fmm.mesh, wave_number,
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 1.0), 13);
    std::vector<cdouble> rhs_parallel;
    if (physical_check) {
        rhs_parallel = muller_nodal_planewave_rhs(
            fmm.mesh, wave_number,
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 0.0, 1.0), 13);
    }
    std::vector<cdouble> coarse_initial_guess;
    if (coarse_checkpoint_path) {
        Mesh coarse_mesh = icosphere(1.0, refinement - 1);
        if (sphere_fivefold_axis)
            align_icosphere_fivefold_axis_to_z(coarse_mesh);
        const MullerP2Mesh coarse_p2 =
            build_muller_p2_mesh(coarse_mesh, build_options);
        std::vector<cdouble> coarse_solution;
        if (!load_checkpoint_solution(
                coarse_checkpoint_path, coarse_solution)) {
            throw std::runtime_error(
                std::string("cannot load coarse checkpoint ") +
                coarse_checkpoint_path);
        }
        coarse_initial_guess = prolong_icosphere_p2_solution(
            coarse_p2, fmm.mesh, coarse_solution);
        std::printf(
            "  [nested sphere warm start] prolonged ref=%d checkpoint "
            "to ref=%d (%zu -> %zu unknowns)\n",
            refinement - 1, refinement, coarse_solution.size(),
            coarse_initial_guess.size());
    }
    const Matvec action = [&](const cdouble* input, cdouble* output) {
        fmm.matvec(input, output);
    };
    const Matvec exact_action =
        [&](const cdouble* input, cdouble* output) {
            fmm.select_fmm_backend();
            fmm.matvec(input, output);
        };
    const Matvec pfft_action =
        [&](const cdouble* input, cdouble* output) {
            fmm.select_pfft_backend();
            fmm.matvec(input, output);
        };
    if (orientation_average) {
        return run_orientation_average(
            fmm, mbj, wave_number, ka, refractive_real,
            refinement, prism_sides,
            orient_average_alpha,
            orient_average_beta,
            orient_average_gamma,
            orient_symmetry_order,
            orient_warm_start,
            orient_warm_max_angle_degrees,
            pfft_fgmres,
            digits,
            max_leaf,
            pfft_inner_tolerance,
            pfft_inner_iterations,
            pfft_outer_restart,
            tolerance,
            maximum_iterations,
            gmres_restart,
            ntheta,
            fmm_setup_seconds,
            mbj_local_setup_seconds,
            !checkpoint_disabled,
            output_path);
    }
    const bool rotation_polarization =
        cyclic_polarization || sphere_fivefold_polarization;
    const bool symmetry_polarization =
        rotation_polarization || mirror_polarization;
    const char* symmetry_name =
        mirror_polarization
            ? "mirror"
            : (sphere_fivefold_polarization ? "sphere C5" : "cyclic");
    const double mirror_axis_angle =
        M_PI / static_cast<double>(prism_sides) +
        prism_azimuth_degrees * M_PI / 180.0;
    const double symmetry_angle = mirror_polarization
        ? 2.0 * mirror_axis_angle
        : (sphere_fivefold_polarization
               ? 2.0 * M_PI / 5.0
               : (cyclic_polarization
               ? 2.0 * M_PI / static_cast<double>(prism_sides)
               : 0.0));
    std::vector<int> symmetry_source_for_target;
    std::function<void(
        const std::vector<cdouble>&,
        std::vector<cdouble>&)> symmetry_solution_vector_transform;
    std::function<void(
        const std::vector<cdouble>&,
        std::vector<cdouble>&)> symmetry_rhs_transform;
    std::function<void(
        const std::vector<cdouble>&,
        std::vector<cdouble>&)> symmetry_solution_transform;
    if (symmetry_polarization) {
        if (rotation_polarization) {
            symmetry_source_for_target =
                rotation_source_for_target(
                    fmm.mesh, symmetry_angle);
            std::printf(
                "  [%s symmetry] mapped P2 elements %.2f%%\n",
                symmetry_name,
                100.0 * symmetry_element_match_fraction(
                    fmm.mesh, symmetry_source_for_target));
            symmetry_solution_vector_transform =
                [&](const std::vector<cdouble>& source,
                    std::vector<cdouble>& transformed) {
                    rotate_muller_solution(
                        fmm.mesh, source,
                        symmetry_source_for_target,
                        symmetry_angle, transformed);
                };
            symmetry_rhs_transform =
                symmetry_solution_vector_transform;
        } else {
            symmetry_source_for_target =
                reflection_source_for_target(
                    fmm.mesh, mirror_axis_angle,
                    fmm.mesh.basis_kind !=
                        MullerBasisKind::HDivBdm1);
            std::printf(
                "  [mirror symmetry] mapped P2 elements %.2f%%\n",
                100.0 * symmetry_element_match_fraction(
                    fmm.mesh, symmetry_source_for_target));
            symmetry_solution_vector_transform =
                [&, mirror_axis_angle](
                    const std::vector<cdouble>& source,
                    std::vector<cdouble>& transformed) {
                    reflect_nodal_solution(
                        fmm.mesh, source,
                        symmetry_source_for_target,
                        mirror_axis_angle, transformed);
                };
            symmetry_rhs_transform =
                [&](const std::vector<cdouble>& source,
                    std::vector<cdouble>& transformed) {
                    symmetry_solution_vector_transform(
                        source, transformed);
#pragma omp parallel for schedule(static)
                    for (int i = 0; i < fmm.system_dofs; i++)
                        transformed[i] = -transformed[i];
                };
        }
        symmetry_solution_transform =
            [&](const std::vector<cdouble>& source,
                std::vector<cdouble>& parallel) {
                std::vector<cdouble> transformed;
                symmetry_solution_vector_transform(
                    source, transformed);
                const double cosine = std::cos(symmetry_angle);
                const double sine = std::sin(symmetry_angle);
                parallel.resize(fmm.system_dofs);
#pragma omp parallel for schedule(static)
                for (int i = 0; i < fmm.system_dofs; i++) {
                    parallel[i] =
                        (transformed[i] - cosine * source[i]) / sine;
                }
            };
    }

    GmresResult baseline;
    if (!mbj_only) {
        baseline = solve_gmres(
            action, rhs.data(), fmm.system_dofs,
            tolerance, maximum_iterations, gmres_restart,
            nullptr, "baseline", nullptr, nullptr, nullptr, nullptr,
            solver_checkpoint("baseline"));
    }
    std::vector<cdouble> recycled_parallel_guess;
    GmresResult hybrid_pfft_result;
    double hybrid_fmm_switch_setup_seconds = 0.0;
    double hybrid_gpu_memory_delta_mb = -1.0;
    int pfft_inner_applications = 0;
    int pfft_inner_total_iterations = 0;
    double pfft_inner_total_seconds = 0.0;
    int first_pfft_inner_applications = 0;
    int first_pfft_inner_iterations = 0;
    double first_pfft_inner_seconds = 0.0;
    int parallel_pfft_inner_applications = 0;
    int parallel_pfft_inner_iterations = 0;
    double parallel_pfft_inner_seconds = 0.0;
    FlexiblePreconditioner pfft_inverse;
    GmresResult preconditioned;
    if (hybrid_pfft_fmm) {
        hybrid_pfft_result = solve_gmres(
            action, rhs.data(), fmm.system_dofs,
            hybrid_pfft_tolerance, maximum_iterations, gmres_restart,
            &mbj, "pFFT-MBJ", nullptr, nullptr, nullptr, nullptr,
            solver_checkpoint("pFFT-MBJ"));
        hybrid_fmm_switch_setup_seconds =
            fmm.switch_pfft_to_fmm(digits, max_leaf);
        size_t hybrid_gpu_free = 0;
        size_t hybrid_gpu_total = 0;
        if (gpu_memory_before_valid &&
            cudaMemGetInfo(
                &hybrid_gpu_free, &hybrid_gpu_total) == cudaSuccess &&
            gpu_free_before >= hybrid_gpu_free) {
            hybrid_gpu_memory_delta_mb =
                static_cast<double>(
                    gpu_free_before - hybrid_gpu_free) /
                (1024.0 * 1024.0);
        }
        preconditioned = solve_gmres(
            action, rhs.data(), fmm.system_dofs,
            tolerance, maximum_iterations, gmres_restart,
            &mbj, "FMM-correction",
            &hybrid_pfft_result.solution,
            symmetry_polarization && !cyclic_exact_geometry
                ? rhs_parallel.data() : nullptr,
            symmetry_polarization && !cyclic_exact_geometry
                ? &recycled_parallel_guess : nullptr,
            symmetry_polarization && !cyclic_exact_geometry
                ? &symmetry_solution_transform : nullptr,
            solver_checkpoint("FMM-correction"));
    } else if (pfft_fgmres) {
        hybrid_fmm_switch_setup_seconds =
            fmm.switch_pfft_to_fmm(digits, max_leaf, true);
        size_t hybrid_gpu_free = 0;
        size_t hybrid_gpu_total = 0;
        if (gpu_memory_before_valid &&
            cudaMemGetInfo(
                &hybrid_gpu_free, &hybrid_gpu_total) == cudaSuccess &&
            gpu_free_before >= hybrid_gpu_free) {
            hybrid_gpu_memory_delta_mb =
                static_cast<double>(
                    gpu_free_before - hybrid_gpu_free) /
                (1024.0 * 1024.0);
        }
        pfft_inverse =
            [&](const cdouble* inner_rhs, cdouble* output) {
                GmresResult inner = solve_gmres(
                    pfft_action, inner_rhs, fmm.system_dofs,
                    pfft_inner_tolerance, pfft_inner_iterations,
                    pfft_inner_iterations, &mbj, "pFFT-inner");
                std::copy(
                    inner.solution.begin(), inner.solution.end(), output);
                pfft_inner_applications++;
                pfft_inner_total_iterations += inner.iterations;
                pfft_inner_total_seconds += inner.seconds;
            };
        preconditioned = solve_flexible_gmres(
            exact_action, pfft_inverse, rhs.data(), fmm.system_dofs,
            tolerance, maximum_iterations, pfft_outer_restart,
            "FMM-pFFT-FGMRES",
            coarse_initial_guess.empty() ? nullptr : &coarse_initial_guess,
            nullptr, nullptr, nullptr,
            solver_checkpoint("FMM-pFFT-FGMRES"));
        first_pfft_inner_applications = pfft_inner_applications;
        first_pfft_inner_iterations = pfft_inner_total_iterations;
        first_pfft_inner_seconds = pfft_inner_total_seconds;
        fmm.select_fmm_backend();
    } else {
        preconditioned = solve_gmres(
            action, rhs.data(), fmm.system_dofs,
            tolerance, maximum_iterations, gmres_restart,
            &mbj, "MBJ",
            coarse_initial_guess.empty() ? nullptr : &coarse_initial_guess,
            symmetry_polarization && !cyclic_exact_geometry
                ? rhs_parallel.data() : nullptr,
            symmetry_polarization && !cyclic_exact_geometry
                ? &recycled_parallel_guess : nullptr,
            symmetry_polarization && !cyclic_exact_geometry
                ? &symmetry_solution_transform : nullptr,
            solver_checkpoint("MBJ"));
    }
    GmresResult parallel_preconditioned;
    bool cyclic_polarization_used = false;
    bool cyclic_exact_geometry_used = false;
    bool cyclic_polarization_corrected = false;
    bool cyclic_polarization_fallback = false;
    bool mirror_polarization_used = false;
    double cyclic_rhs_relative_error = -1.0;
    double symmetry_direct_residual = -1.0;
    std::vector<cdouble> symmetry_direct_solution;
    double farfield_seconds = 0.0;
    std::vector<double> theta;
    std::vector<double> mueller;
    std::vector<cdouble> s1, s2, s3, s4;
    double sphere_cross_polarization_relative = -1.0;
    if (physical_check) {
        if (sphere_rotational_farfield) {
            std::printf(
                "  [sphere rotational symmetry] skipping the independent "
                "second-polarization solve\n");
        } else if (symmetry_polarization) {
            const auto symmetry_start =
                std::chrono::steady_clock::now();
            std::vector<cdouble> transformed_rhs;
            symmetry_rhs_transform(rhs, transformed_rhs);
            double rhs_difference_squared = 0.0;
            double rhs_rotated_squared = 0.0;
#pragma omp parallel for reduction(+:rhs_difference_squared,rhs_rotated_squared) schedule(static)
            for (int i = 0; i < fmm.system_dofs; i++) {
                const cdouble expected =
                    std::cos(symmetry_angle) * rhs[i] +
                    std::sin(symmetry_angle) * rhs_parallel[i];
                rhs_difference_squared +=
                    std::norm(transformed_rhs[i] - expected);
                rhs_rotated_squared += std::norm(expected);
            }
            cyclic_rhs_relative_error = std::sqrt(
                rhs_difference_squared /
                std::max(rhs_rotated_squared, 1.0e-300));
            std::printf(
                "  [%s symmetry] RHS transform relative error %.3e\n",
                symmetry_name, cyclic_rhs_relative_error);
            symmetry_solution_transform(
                preconditioned.solution,
                parallel_preconditioned.solution);
            symmetry_direct_solution =
                parallel_preconditioned.solution;
            std::vector<cdouble> symmetry_work(fmm.system_dofs);
            action(
                parallel_preconditioned.solution.data(),
                symmetry_work.data());
            const double rhs_parallel_norm = std::max(
                norm(rhs_parallel.data(), fmm.system_dofs), 1.0e-300);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < fmm.system_dofs; i++)
                symmetry_work[i] =
                    rhs_parallel[i] - symmetry_work[i];
            parallel_preconditioned.operator_residual =
                norm(symmetry_work.data(), fmm.system_dofs) /
                rhs_parallel_norm;
            symmetry_direct_residual =
                parallel_preconditioned.operator_residual;
            parallel_preconditioned.projected_residual =
                parallel_preconditioned.operator_residual;
            parallel_preconditioned.seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    symmetry_start).count();
            std::printf(
                "  [%s symmetry] direct reconstructed residual %.3e\n",
                symmetry_name, symmetry_direct_residual);
            if (symmetry_reconstruction_meets_tolerance(
                    parallel_preconditioned.operator_residual,
                    tolerance)) {
                cyclic_polarization_used = true;
                mirror_polarization_used = mirror_polarization;
                cyclic_exact_geometry_used = cyclic_exact_geometry;
                if (cyclic_exact_geometry) {
                    std::printf(
                        "  [C%d exact geometry] rotated polarization "
                        "verified; FMM residual %.3e\n",
                        prism_sides,
                        parallel_preconditioned.operator_residual);
                } else {
                    std::printf(
                        "  [%s symmetry] reconstructed polarization "
                        "residual %.3e\n",
                        symmetry_name,
                        parallel_preconditioned.operator_residual);
                }
            } else {
                if (cyclic_exact_geometry) {
                    std::printf(
                        "  [C%d exact geometry] rotated polarization "
                        "failed the discrete-operator check (%.3e > "
                        "%.3e); solving a correction\n",
                        prism_sides,
                        parallel_preconditioned.operator_residual,
                        tolerance);
                }
                double candidate_seconds =
                    parallel_preconditioned.seconds +
                    preconditioned.recycle_seconds;
                double best_initial_residual =
                    parallel_preconditioned.operator_residual;
                std::vector<cdouble> correction_guess =
                    parallel_preconditioned.solution;
                if (!recycled_parallel_guess.empty()) {
                    action(
                        recycled_parallel_guess.data(),
                        symmetry_work.data());
#pragma omp parallel for schedule(static)
                    for (int i = 0; i < fmm.system_dofs; i++)
                        symmetry_work[i] =
                            rhs_parallel[i] - symmetry_work[i];
                    const double recycled_residual =
                        norm(symmetry_work.data(), fmm.system_dofs) /
                        rhs_parallel_norm;
                    std::printf(
                        "  [%s Krylov recycle] initial residual %.3e "
                        "(%.3fs)\n",
                        symmetry_name, recycled_residual,
                        preconditioned.recycle_seconds);
                    if (recycled_residual < best_initial_residual) {
                        best_initial_residual = recycled_residual;
                        correction_guess = recycled_parallel_guess;
                    }
                }
                const std::vector<cdouble>* selected_guess =
                    &correction_guess;
                if (best_initial_residual >= 1.0)
                    selected_guess = nullptr;
                std::printf(
                    "  [%s symmetry] best initial residual %.3e "
                    "exceeds %.3e; solving a correction\n",
                    symmetry_name,
                    best_initial_residual,
                    tolerance);
                if (pfft_fgmres) {
                    const int applications_before =
                        pfft_inner_applications;
                    const int iterations_before =
                        pfft_inner_total_iterations;
                    const double seconds_before =
                        pfft_inner_total_seconds;
                    parallel_preconditioned =
                        solve_flexible_gmres(
                            exact_action, pfft_inverse,
                            rhs_parallel.data(), fmm.system_dofs,
                            tolerance, maximum_iterations,
                            pfft_outer_restart,
                            "FMM-pFFT-symmetry-correction",
                            selected_guess,
                            nullptr, nullptr, nullptr,
                            solver_checkpoint(
                                "FMM-pFFT-symmetry-correction"));
                    parallel_pfft_inner_applications +=
                        pfft_inner_applications -
                        applications_before;
                    parallel_pfft_inner_iterations +=
                        pfft_inner_total_iterations -
                        iterations_before;
                    parallel_pfft_inner_seconds +=
                        pfft_inner_total_seconds - seconds_before;
                } else {
                    parallel_preconditioned = solve_gmres(
                        action, rhs_parallel.data(), fmm.system_dofs,
                        tolerance, maximum_iterations, gmres_restart,
                        &mbj, "MBJ-symmetry-correction",
                        selected_guess, nullptr, nullptr, nullptr,
                        solver_checkpoint(
                            "MBJ-symmetry-correction"));
                }
                parallel_preconditioned.seconds += candidate_seconds;
                if (parallel_preconditioned.operator_residual <=
                    tolerance) {
                    cyclic_polarization_used = true;
                    mirror_polarization_used = mirror_polarization;
                    cyclic_polarization_corrected = true;
                } else {
                    cyclic_polarization_fallback = true;
                    const double correction_seconds =
                        parallel_preconditioned.seconds;
                    std::printf(
                        "  [%s symmetry] correction did not converge; "
                        "falling back to zero-start GMRES\n",
                        symmetry_name);
                    parallel_preconditioned = solve_gmres(
                        action, rhs_parallel.data(), fmm.system_dofs,
                        tolerance, maximum_iterations, gmres_restart,
                        &mbj, "MBJ-parallel", nullptr, nullptr, nullptr,
                        nullptr, solver_checkpoint("MBJ-parallel"));
                    parallel_preconditioned.seconds +=
                        correction_seconds;
                }
            }
        } else {
            if (pfft_fgmres) {
                const int applications_before =
                    pfft_inner_applications;
                const int iterations_before =
                    pfft_inner_total_iterations;
                const double seconds_before =
                    pfft_inner_total_seconds;
                parallel_preconditioned =
                    solve_flexible_gmres(
                        exact_action, pfft_inverse,
                        rhs_parallel.data(), fmm.system_dofs,
                        tolerance, maximum_iterations,
                        pfft_outer_restart,
                        "FMM-pFFT-parallel", nullptr,
                        nullptr, nullptr, nullptr,
                        solver_checkpoint("FMM-pFFT-parallel"));
                parallel_pfft_inner_applications =
                    pfft_inner_applications - applications_before;
                parallel_pfft_inner_iterations =
                    pfft_inner_total_iterations - iterations_before;
                parallel_pfft_inner_seconds =
                    pfft_inner_total_seconds - seconds_before;
            } else {
                parallel_preconditioned = solve_gmres(
                    action, rhs_parallel.data(), fmm.system_dofs,
                    tolerance, maximum_iterations, gmres_restart,
                    &mbj, "MBJ-parallel", nullptr, nullptr, nullptr,
                    nullptr, solver_checkpoint("MBJ-parallel"));
            }
        }

        const auto farfield_start = std::chrono::steady_clock::now();
        theta.resize(ntheta);
        std::vector<Vec3> directions(ntheta);
        std::vector<Vec3> theta_hat(ntheta);
        std::vector<Vec3> orthogonal_directions;
        std::vector<Vec3> orthogonal_theta_hat;
        if (sphere_rotational_farfield) {
            orthogonal_directions.resize(ntheta);
            orthogonal_theta_hat.resize(ntheta);
        }
        for (int angle = 0; angle < ntheta; angle++) {
            theta[angle] = 180.0 * angle / (ntheta - 1);
            const double radians = theta[angle] * M_PI / 180.0;
            directions[angle] =
                Vec3(0.0, std::sin(radians), std::cos(radians));
            theta_hat[angle] =
                Vec3(0.0, std::cos(radians), -std::sin(radians));
            if (sphere_rotational_farfield) {
                orthogonal_directions[angle] =
                    Vec3(std::sin(radians), 0.0, std::cos(radians));
                orthogonal_theta_hat[angle] =
                    Vec3(std::cos(radians), 0.0, -std::sin(radians));
            }
        }
        std::vector<cdouble> field_parallel;
        std::vector<cdouble> field_perpendicular;
        if (sphere_rotational_farfield) {
            if (fmm.gpu_operator_assembly) {
                fmm.farfield(
                    preconditioned.solution.data(),
                    orthogonal_directions, field_parallel);
            } else {
                muller_nodal_farfield(
                    fmm.mesh,
                    preconditioned.solution.data(),
                    preconditioned.solution.data() + fmm.current_dofs,
                    wave_number, orthogonal_directions, field_parallel);
            }
        } else {
            if (fmm.gpu_operator_assembly) {
                fmm.farfield(
                    parallel_preconditioned.solution.data(),
                    directions, field_parallel);
            } else {
                muller_nodal_farfield(
                    fmm.mesh,
                    parallel_preconditioned.solution.data(),
                    parallel_preconditioned.solution.data() +
                        fmm.current_dofs,
                    wave_number, directions, field_parallel);
            }
        }
        if (fmm.gpu_operator_assembly) {
            fmm.farfield(
                preconditioned.solution.data(),
                directions, field_perpendicular);
        } else {
            muller_nodal_farfield(
                fmm.mesh,
                preconditioned.solution.data(),
                preconditioned.solution.data() + fmm.current_dofs,
                wave_number, directions, field_perpendicular);
        }
        s1.resize(ntheta);
        s2.resize(ntheta);
        s3.resize(ntheta);
        s4.resize(ntheta);
        const cdouble amplitude_scale(0.0, -ka);
        double cross_squared = 0.0;
        double copolar_squared = 0.0;
        for (int angle = 0; angle < ntheta; angle++) {
            const Vec3& parallel_hat = sphere_rotational_farfield
                ? orthogonal_theta_hat[angle] : theta_hat[angle];
            const cdouble parallel_theta =
                field_parallel[3 * angle] * parallel_hat.x +
                field_parallel[3 * angle + 1] * parallel_hat.y +
                field_parallel[3 * angle + 2] * parallel_hat.z;
            const cdouble perpendicular_theta =
                field_perpendicular[3 * angle] * theta_hat[angle].x +
                field_perpendicular[3 * angle + 1] * theta_hat[angle].y +
                field_perpendicular[3 * angle + 2] * theta_hat[angle].z;
            s2[angle] = amplitude_scale * parallel_theta;
            if (sphere_rotational_farfield) {
                const cdouble cross_parallel =
                    -amplitude_scale * field_parallel[3 * angle + 1];
                const cdouble cross_perpendicular =
                    amplitude_scale * perpendicular_theta;
                cross_squared += std::norm(cross_parallel) +
                    std::norm(cross_perpendicular);
                copolar_squared += std::norm(s2[angle]) +
                    std::norm(
                        amplitude_scale *
                        field_perpendicular[3 * angle]);
                s3[angle] = cdouble(0.0);
                s4[angle] = cdouble(0.0);
            } else {
                s4[angle] =
                    amplitude_scale * field_parallel[3 * angle];
                s3[angle] =
                    amplitude_scale * perpendicular_theta;
            }
            s1[angle] = amplitude_scale * field_perpendicular[3 * angle];
        }
        if (sphere_rotational_farfield) {
            sphere_cross_polarization_relative = std::sqrt(
                cross_squared / std::max(copolar_squared, 1.0e-300));
            std::printf(
                "  [sphere rotational symmetry] discarded cross-polar "
                "amplitude norm %.3e\n",
                sphere_cross_polarization_relative);
        }
        amplitude_to_mueller(s1, s2, s3, s4, mueller);
        farfield_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - farfield_start).count();
    }
    GmresResult neural_preconditioned;
    if (neural_preconditioner_path) {
        neural_preconditioned = solve_gmres(
            action, rhs.data(), fmm.system_dofs,
            tolerance, maximum_iterations, gmres_restart,
            &neural, "neural", nullptr, nullptr, nullptr, nullptr,
            solver_checkpoint("neural"));
    }

    MullerDenseSystem dense;
    double dense_validation_seconds = 0.0;
    double baseline_dense_residual = -1.0;
    double mbj_dense_residual = -1.0;
    double symmetry_direct_dense_residual = -1.0;
    double neural_dense_residual = -1.0;
    if (dense_validation) {
        const auto dense_start = std::chrono::steady_clock::now();
        dense = assemble_muller_nodal_dense(
            mesh, wave_number, refractive_index,
            build_options, 7, 4);
        dense_validation_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                dense_start).count();
        if (!mbj_only) {
            baseline_dense_residual =
                dense_residual(dense, baseline.solution, rhs);
        }
        mbj_dense_residual =
            dense_residual(dense, preconditioned.solution, rhs);
        if (symmetry_polarization) {
            symmetry_direct_dense_residual = dense_residual(
                dense, symmetry_direct_solution, rhs_parallel);
        }
        if (neural_preconditioner_path) {
            neural_dense_residual = dense_residual(
                dense, neural_preconditioned.solution, rhs);
        }
    }
    double difference_squared = 0.0;
    double baseline_squared = 0.0;
    if (!mbj_only) {
        for (int i = 0; i < fmm.system_dofs; i++) {
            difference_squared += std::norm(
                preconditioned.solution[i] -
                baseline.solution[i]);
            baseline_squared += std::norm(baseline.solution[i]);
        }
    }
    const double solution_difference = mbj_only ? -1.0 :
        std::sqrt(difference_squared / baseline_squared);
    const double solve_speedup = mbj_only ? -1.0 :
        baseline.seconds / preconditioned.seconds;
    const double neural_solve_speedup =
        mbj_only || !neural_preconditioner_path ? -1.0 :
        baseline.seconds / neural_preconditioned.seconds;

    std::ofstream output(output_path);
    output << std::setprecision(17)
           << "{\n"
           << "  \"solver\": \""
           << (pfft_fgmres
                   ? "muller_nodal_p2_pfft_fgmres"
                   : (hybrid_pfft_fmm
                   ? "muller_nodal_p2_hybrid_pfft_fmm"
                   : std::string("muller_nodal_p2_") +
                         fmm.backend_name()))
           << "\",\n"
           << "  \"operator_backend\": \""
           << fmm.backend_name() << "\",\n"
           << "  \"gpu_operator_assembly\": "
           << (fmm.gpu_operator_assembly ? "true" : "false")
           << ",\n"
           << "  \"hybrid_pfft_fmm\": "
           << (hybrid_pfft_fmm ? "true" : "false") << ",\n"
           << "  \"pfft_fgmres_enabled\": "
           << (pfft_fgmres ? "true" : "false") << ",\n"
#ifdef BEM_PFFT_FP32
           << "  \"pfft_fft_precision\": \"fp32\",\n"
#else
           << "  \"pfft_fft_precision\": \"fp64\",\n"
#endif
           << "  \"pfft_order\": " << pfft_order << ",\n"
           << "  \"pfft_correction_radius_cells\": "
           << pfft_correction_radius << ",\n"
           << "  \"pfft_grid_safety\": "
           << pfft_grid_safety << ",\n"
           << "  \"hybrid_pfft_tolerance\": "
           << hybrid_pfft_tolerance << ",\n"
           << "  \"pfft_inner_tolerance\": "
           << pfft_inner_tolerance << ",\n"
           << "  \"pfft_inner_max_iterations\": "
           << pfft_inner_iterations << ",\n"
           << "  \"pfft_inner_iterations_auto\": "
           << (pfft_inner_iterations_auto ? "true" : "false")
           << ",\n"
           << "  \"pfft_outer_restart\": "
           << pfft_outer_restart << ",\n"
           << "  \"solver_checkpoint\": {\"enabled\": "
           << (checkpoint_base.empty() ? "false" : "true")
           << ", \"base\": \""
           << checkpoint_base << "\"},\n"
           << "  \"shape\": \"" << shape << "\",\n"
           << "  \"prism_azimuth_degrees\": "
           << prism_azimuth_degrees << ",\n"
           << "  \"mirror_symmetric_mesh\": "
           << (mirror_symmetric_mesh ? "true" : "false")
           << ",\n"
           << "  \"ka\": " << ka << ",\n"
           << "  \"ri\": " << refractive_real << ",\n"
           << "  \"refinements\": " << refinement << ",\n"
           << "  \"edge_refine_requested\": "
           << mesh.edge_refine_requested << ",\n"
           << "  \"edge_refine_applied\": "
           << mesh.edge_refine_applied << ",\n"
           << "  \"edge_refine_uniform_fallback\": "
           << (mesh.edge_refine_uniform_fallback ? "true" : "false")
           << ",\n"
           << "  \"system_dofs\": " << fmm.system_dofs << ",\n"
           << "  \"quadrature_points\": "
           << fmm.quadrature.size() << ",\n"
           << "  \"regular_quadrature\": "
           << regular_quadrature << ",\n"
           << "  \"duffy_order\": " << duffy_order << ",\n"
           << "  \"fmm_digits\": "
           << std::min(digits, 5) << ",\n"
           << "  \"fmm_digits_requested\": " << digits << ",\n"
           << "  \"fmm_max_leaf_points\": " << max_leaf << ",\n"
           << "  \"fmm_near_radius\": "
           << fmm_near_radius << ",\n"
           << "  \"fmm_near_precision\": \""
           << (fmm_near_fp32 ? "fp32" : "fp64") << "\",\n"
           << "  \"gmres_restart\": " << gmres_restart << ",\n"
           << "  \"cyclic_polarization_requested\": "
           << (cyclic_polarization ? "true" : "false") << ",\n"
           << "  \"auto_polarization_symmetry_requested\": "
           << (auto_polarization_symmetry
                   ? "true" : "false") << ",\n"
           << "  \"sphere_fivefold_axis\": "
           << (sphere_fivefold_axis ? "true" : "false") << ",\n"
           << "  \"sphere_fivefold_polarization_requested\": "
           << (sphere_fivefold_polarization ? "true" : "false")
           << ",\n"
           << "  \"sphere_rotational_farfield\": "
           << (sphere_rotational_farfield ? "true" : "false")
           << ",\n"
           << "  \"cyclic_exact_geometry_requested\": "
           << (cyclic_exact_geometry ? "true" : "false") << ",\n"
           << "  \"mirror_polarization_requested\": "
           << (mirror_polarization ? "true" : "false") << ",\n"
           << "  \"max_element_edge\": "
           << max_element_edge << ",\n"
           << "  \"ka_h_element\": " << ka_h_element << ",\n"
           << "  \"p2_nodes_per_wavelength_min\": "
           << p2_nodes_per_wavelength << ",\n"
           << "  \"gpu_memory_delta_mb\": "
           << gpu_memory_delta_mb << ",\n"
           << "  \"edge_mode\": \""
           << muller_edge_mode_name(edge_mode)
           << "\",\n"
           << "  \"hdiv_conforming\": "
           << (edge_mode == MullerEdgeMode::HDivBdm1
                   ? "true" : "false") << ",\n"
           << "  \"sharp_edge_formulation_validated\": "
           << (sharp_mode ? "false" : "true") << ",\n"
           << "  \"feature_angle_degrees\": "
           << feature_angle << ",\n"
           << "  \"feature_edge_segments\": "
           << fmm.mesh.feature_edges << ",\n"
           << "  \"smooth_patches\": "
           << fmm.mesh.smooth_patches << ",\n"
           << "  \"duplicated_edge_nodes\": "
           << fmm.mesh.duplicated_corner_nodes +
                  fmm.mesh.duplicated_midpoint_nodes
           << ",\n"
           << "  \"tolerance\": " << tolerance << ",\n"
           << "  \"fmm_setup_s\": " << fmm_setup_seconds << ",\n"
           << "  \"fmm_setup_breakdown\": {\"geometry_s\": "
           << fmm.geometry_setup_seconds
           << ", \"near_correction_s\": "
           << fmm.near_correction_setup_seconds
           << ", \"engines_s\": "
           << fmm.fmm_engine_setup_seconds
           << ", \"near_correction_colors\": "
           << fmm.near_correction_colors
           << ", \"near_correction_pairs\": "
           << fmm.near_correction_pairs
           << ", \"near_correction_unique_templates\": "
           << fmm.near_correction_unique_templates
           << ", \"near_correction_template_reuse\": "
           << (fmm.near_correction_template_reuse
                   ? "true" : "false")
           << "},\n"
           << "  \"near_correction_cache\": {\"enabled\": "
           << (near_correction_cache_path ? "true" : "false")
           << ", \"hit\": "
           << (fmm.near_correction_cache_hit ? "true" : "false")
           << ", \"entries\": "
           << fmm.correction.entries.size() << "},\n"
           << "  \"dense_validation_s\": "
           << dense_validation_seconds << ",\n"
           << "  \"symmetry_direct_dense_residual\": ";
    if (dense_validation && symmetry_polarization)
        output << symmetry_direct_dense_residual;
    else
        output << "null";
    output << ",\n"
           << "  \"mbj_local_setup_s\": "
           << mbj_local_setup_seconds << ",\n"
           << "  \"mbj_setup_breakdown\": {\"ordering_s\": "
           << mbj.ordering_seconds
           << ", \"assembly_s\": " << mbj.assembly_seconds
           << ", \"factorization_s\": "
           << mbj.factorization_seconds
           << ", \"threads\": " << mbj.setup_threads
           << ", \"cache_enabled\": "
           << (mbj_cache_path ? "true" : "false")
           << ", \"cache_hit\": "
           << (mbj.cache_hit ? "true" : "false")
           << ", \"cache_io_s\": "
           << mbj.cache_io_seconds << "},\n"
           << "  \"baseline\": ";
    if (mbj_only) {
        output << "null,\n";
    } else {
        output << "{\"iterations\": "
               << baseline.iterations
               << ", \"resumed_iterations\": "
               << baseline.resumed_iterations
               << ", \"solve_s\": " << baseline.seconds
               << ", \"fmm_residual\": "
               << baseline.operator_residual
               << ", \"dense_residual\": ";
        if (dense_validation)
            output << baseline_dense_residual;
        else
            output << "null";
        output << "},\n";
    }
    output
           << "  \"mbj\": {\"nodes_per_block\": " << mbj_nodes
           << ", \"overlap_nodes\": " << mbj_overlap
           << ", \"coarse_rank\": " << mbj.coarse_rank
           << ", \"coarse_setup_s\": "
           << mbj_coarse_setup_seconds
           << ", \"iterations\": " << preconditioned.iterations
           << ", \"resumed_iterations\": "
           << preconditioned.resumed_iterations
           << ", \"solve_s\": " << preconditioned.seconds
           << ", \"fmm_residual\": "
           << preconditioned.operator_residual
           << ", \"dense_residual\": ";
    if (dense_validation)
        output << mbj_dense_residual;
    else
        output << "null";
    output
           << ", \"storage_mb\": " << mbj.storage_megabytes()
           << "},\n"
           << "  \"hybrid\": ";
    if (!hybrid_pfft_fmm) {
        output << "null,\n";
    } else {
        output
            << "{\"pfft_iterations\": "
            << hybrid_pfft_result.iterations
            << ", \"pfft_solve_s\": "
            << hybrid_pfft_result.seconds
            << ", \"pfft_residual\": "
            << hybrid_pfft_result.operator_residual
            << ", \"fmm_switch_setup_s\": "
            << hybrid_fmm_switch_setup_seconds
            << ", \"combined_gpu_memory_delta_mb\": "
            << hybrid_gpu_memory_delta_mb
            << ", \"initial_fmm_residual\": "
            << preconditioned.initial_operator_residual
            << ", \"fmm_correction_iterations\": "
            << preconditioned.iterations
            << ", \"fmm_correction_s\": "
            << preconditioned.seconds
            << ", \"total_solve_s\": "
            << hybrid_pfft_result.seconds +
                   hybrid_fmm_switch_setup_seconds +
                   preconditioned.seconds
            << "},\n";
    }
    output
           << "  \"pfft_fgmres\": ";
    if (!pfft_fgmres) {
        output << "null,\n";
    } else {
        output
            << "{\"fmm_switch_setup_s\": "
            << hybrid_fmm_switch_setup_seconds
            << ", \"combined_gpu_memory_delta_mb\": "
            << hybrid_gpu_memory_delta_mb
            << ", \"outer_iterations\": "
            << preconditioned.iterations
            << ", \"resumed_outer_iterations\": "
            << preconditioned.resumed_iterations
            << ", \"outer_solve_s\": "
            << preconditioned.seconds
            << ", \"fmm_residual\": "
            << preconditioned.operator_residual
            << ", \"initial_fmm_residual\": "
            << preconditioned.initial_operator_residual
            << ", \"inner_applications\": "
            << first_pfft_inner_applications
            << ", \"inner_iterations\": "
            << first_pfft_inner_iterations
            << ", \"inner_solve_s\": "
            << first_pfft_inner_seconds
            << "},\n";
    }
    output
           << "  \"coarse_initial_guess\": ";
    if (!coarse_checkpoint_path) {
        output << "null,\n";
    } else {
        output << "{\"path\": \"" << coarse_checkpoint_path
               << "\", \"source_ref\": " << refinement - 1
               << ", \"initial_fmm_residual\": "
               << preconditioned.initial_operator_residual << "},\n";
    }
    output
           << "  \"neural\": ";
    if (!neural_preconditioner_path) {
        output << "null,\n";
    } else {
        output << "{\"path\": \"" << neural_preconditioner_path
               << "\", \"load_s\": " << neural_load_seconds
               << ", \"iterations\": "
               << neural_preconditioned.iterations
               << ", \"resumed_iterations\": "
               << neural_preconditioned.resumed_iterations
               << ", \"solve_s\": "
               << neural_preconditioned.seconds
               << ", \"fmm_residual\": "
               << neural_preconditioned.operator_residual
               << ", \"dense_residual\": ";
        if (dense_validation)
            output << neural_dense_residual;
        else
            output << "null";
        output << ", \"storage_mb\": "
               << neural.storage_megabytes() << "},\n";
    }
    output
           << "  \"physical\": ";
    if (!physical_check) {
        output << "null,\n";
    } else {
        output << "{\"polarization_mode\": \""
               << (sphere_rotational_farfield
                       ? "sphere_rotational_farfield"
                       : (sphere_fivefold_polarization
                       ? (cyclic_polarization_corrected
                              ? "sphere_c5_reconstruction_corrected"
                              : (cyclic_polarization_used
                                     ? "sphere_c5_reconstruction"
                                     : "independent_gmres_fallback"))
                       : (mirror_polarization_used
                       ? (cyclic_polarization_corrected
                              ? "mirror_reconstruction_corrected"
                              : "mirror_reconstruction")
                       : (cyclic_exact_geometry_used
                       ? "cyclic_exact_geometry"
                       : (cyclic_polarization_corrected
                       ? "cyclic_reconstruction_corrected"
                       : (cyclic_polarization_used
                              ? "cyclic_reconstruction"
                       : (cyclic_polarization_fallback
                              ? "independent_gmres_fallback"
                              : "independent_gmres")))))))
               << "\", \"cyclic_order\": "
               << (sphere_fivefold_polarization
                       ? 5
                       : (cyclic_polarization ? prism_sides : 0))
               << ", \"mirror_axis_degrees\": "
               << (mirror_polarization
                       ? 180.0 / static_cast<double>(prism_sides) +
                             prism_azimuth_degrees
                       : 0.0)
               << ", \"cyclic_rhs_relative_error\": "
               << cyclic_rhs_relative_error
               << ", \"symmetry_direct_fmm_residual\": "
               << symmetry_direct_residual
               << ", \"parallel_iterations\": "
               << parallel_preconditioned.iterations
               << ", \"parallel_resumed_iterations\": "
               << parallel_preconditioned.resumed_iterations
               << ", \"parallel_s\": "
               << parallel_preconditioned.seconds
               << ", \"parallel_fmm_residual\": ";
        if (sphere_rotational_farfield)
            output << "null";
        else
            output << parallel_preconditioned.operator_residual;
        output
               << ", \"sphere_cross_polarization_relative\": "
               << sphere_cross_polarization_relative
               << ", \"parallel_pfft_inner_applications\": "
               << parallel_pfft_inner_applications
               << ", \"parallel_pfft_inner_iterations\": "
               << parallel_pfft_inner_iterations
               << ", \"parallel_pfft_inner_s\": "
               << parallel_pfft_inner_seconds
               << ", \"farfield_s\": " << farfield_seconds
               << ", \"theta_degrees\": [";
        for (int angle = 0; angle < ntheta; angle++) {
            if (angle)
                output << ", ";
            output << theta[angle];
        }
        output << "], \"mueller\": [\n";
        for (int row = 0; row < 4; row++) {
            output << "    [";
            for (int column = 0; column < 4; column++) {
                if (column)
                    output << ", ";
                output << "[";
                for (int angle = 0; angle < ntheta; angle++) {
                    if (angle)
                        output << ", ";
                    output << mueller[
                        (static_cast<size_t>(row) * 4 + column) *
                            ntheta +
                        angle];
                }
                output << "]";
            }
            output << "]" << (row == 3 ? "\n" : ",\n");
        }
        output << "  ], \"amplitudes\": {\n";
        const char* amplitude_names[4] = {"S1", "S2", "S3", "S4"};
        const std::vector<cdouble>* amplitudes[4] = {
            &s1, &s2, &s3, &s4
        };
        for (int component = 0; component < 4; component++) {
            output << "    \"" << amplitude_names[component] << "\": [";
            for (int angle = 0; angle < ntheta; angle++) {
                if (angle)
                    output << ", ";
                const cdouble value = (*amplitudes[component])[angle];
                output << "[" << value.real() << ", "
                       << value.imag() << "]";
            }
            output << "]" << (component == 3 ? "\n" : ",\n");
        }
        output << "  }},\n";
    }
    output
           << "  \"solve_speedup\": " << solve_speedup << ",\n"
           << "  \"neural_solve_speedup\": "
           << neural_solve_speedup << ",\n"
           << "  \"solution_relative_difference\": "
           << solution_difference << "\n"
           << "}\n";
    output.close();

    if (hybrid_pfft_fmm) {
        std::printf(
            "Muller hybrid solve: pFFT %d it %.3fs, "
            "FMM correction %d it %.3fs from residual %.3e, "
            "switch %.3fs\n",
            hybrid_pfft_result.iterations,
            hybrid_pfft_result.seconds,
            preconditioned.iterations,
            preconditioned.seconds,
            preconditioned.initial_operator_residual,
            hybrid_fmm_switch_setup_seconds);
    } else if (pfft_fgmres) {
        std::printf(
            "Muller pFFT-FGMRES solve: %d exact FMM it %.3fs, "
            "%d inner applications / %d pFFT it %.3fs, "
            "switch %.3fs, residual %.3e\n",
            preconditioned.iterations,
            preconditioned.seconds,
            first_pfft_inner_applications,
            first_pfft_inner_iterations,
            first_pfft_inner_seconds,
            hybrid_fmm_switch_setup_seconds,
            preconditioned.operator_residual);
    } else if (mbj_only) {
        std::printf(
            "Muller %s solve: MBJ %d it %.3fs\n",
            fmm.backend_name(),
            preconditioned.iterations,
            preconditioned.seconds);
    } else {
        std::printf(
            "Muller %s solve: baseline %d it %.3fs, "
            "MBJ %d it %.3fs, speedup %.2fx\n",
            fmm.backend_name(),
            baseline.iterations, baseline.seconds,
            preconditioned.iterations, preconditioned.seconds,
            solve_speedup);
    }
    if (neural_preconditioner_path) {
        std::printf(
            "  neural %d it %.3fs, speedup %.2fx, "
            "load %.3fs\n",
            neural_preconditioned.iterations,
            neural_preconditioned.seconds,
            neural_solve_speedup, neural_load_seconds);
    }
    if (physical_check) {
        if (sphere_rotational_farfield) {
            std::printf(
                "  sphere symmetry: one solved polarization, two "
                "far-field planes; far field %.3fs (%d angles)\n",
                farfield_seconds, ntheta);
        } else {
            std::printf(
                "  second polarization: %d it %.3fs, residual %.3e; "
                "far field %.3fs (%d angles)\n",
                parallel_preconditioned.iterations,
                parallel_preconditioned.seconds,
                parallel_preconditioned.operator_residual,
                farfield_seconds, ntheta);
        }
        if (parallel_pfft_inner_applications > 0) {
            std::printf(
                "  second-polarization inner pFFT: %d applications, "
                "%d iterations, %.3fs\n",
                parallel_pfft_inner_applications,
                parallel_pfft_inner_iterations,
                parallel_pfft_inner_seconds);
        }
    }
    std::printf(
        "  geometry: %s, feature-edge segments=%d, "
        "smooth patches=%d, duplicated P2 nodes=%d\n",
        shape, fmm.mesh.feature_edges, fmm.mesh.smooth_patches,
        fmm.mesh.duplicated_corner_nodes +
            fmm.mesh.duplicated_midpoint_nodes);
    std::printf(
        "  setup: %s %.3fs%s, local MBJ %.3fs "
        "(assembly %.3fs, LU %.3fs, coarse %.3fs rank %d, %d threads; "
        "dense validation %.3fs%s)\n",
        (hybrid_pfft_fmm || pfft_fgmres)
            ? "pfft" : fmm.backend_name(),
        fmm_setup_seconds,
        (hybrid_pfft_fmm || pfft_fgmres)
            ? " plus FMM switch above" : "",
        mbj_local_setup_seconds,
        mbj.assembly_seconds, mbj.factorization_seconds,
        mbj_coarse_setup_seconds, mbj.coarse_rank,
        mbj.setup_threads,
        dense_validation_seconds,
        dense_validation ? ", not required in production" :
                           ", disabled");
    if (dense_validation) {
        std::printf(
            "  dense residuals baseline=%.3e MBJ=%.3e, "
            "symmetry-direct=%.3e, solution difference=%.3e, out=%s\n",
            baseline_dense_residual, mbj_dense_residual,
            symmetry_direct_dense_residual,
            solution_difference, output_path);
    } else {
        std::printf("  out=%s\n", output_path);
    }
    fmm.cleanup();
    return (mbj_only ||
            baseline.operator_residual <= 2.0 * tolerance) &&
           preconditioned.operator_residual <= 2.0 * tolerance &&
           (!physical_check ||
            sphere_rotational_farfield ||
            cyclic_exact_geometry_used ||
            parallel_preconditioned.operator_residual <=
                2.0 * tolerance)
        ? 0 : 1;
}
