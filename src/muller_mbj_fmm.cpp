#include "muller_mbj.h"

#include "muller_fmm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using cdouble = std::complex<double>;

const std::uint64_t MBJ_FNV_OFFSET = 1469598103934665603ULL;
const std::uint64_t MBJ_FNV_PRIME = 1099511628211ULL;

void mbj_hash_bytes(
    std::uint64_t& hash, const void* data, std::size_t size)
{
    const unsigned char* bytes =
        static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= MBJ_FNV_PRIME;
    }
}

template <typename T>
void mbj_hash_value(std::uint64_t& hash, const T& value)
{
    mbj_hash_bytes(hash, &value, sizeof(value));
}

std::uint64_t mbj_cache_signature(
    const MullerFmmOperator& op,
    int scalar_nodes_per_block,
    int overlap_nodes)
{
    std::uint64_t hash = MBJ_FNV_OFFSET;
    mbj_hash_value(hash, scalar_nodes_per_block);
    mbj_hash_value(hash, overlap_nodes);
    mbj_hash_value(hash, op.system_dofs);
    mbj_hash_value(hash, op.current_dofs);
    mbj_hash_value(hash, op.quadrature_order);
    mbj_hash_value(hash, op.k_exterior);
    mbj_hash_value(hash, op.k_interior);
    mbj_hash_value(hash, op.epsilon_exterior);
    mbj_hash_value(hash, op.epsilon_interior);
    mbj_hash_value(hash, op.mu_exterior);
    mbj_hash_value(hash, op.mu_interior);
    const int basis_kind = static_cast<int>(op.mesh.basis_kind);
    const int edge_mode = static_cast<int>(op.mesh.edge_mode);
    mbj_hash_value(hash, basis_kind);
    mbj_hash_value(hash, edge_mode);
    mbj_hash_value(hash, op.mesh.feature_angle_degrees);
    mbj_hash_value(hash, op.mesh.topology_edge_count);
    const std::size_t node_count = op.mesh.nodes.size();
    const std::size_t element_count = op.mesh.elements.size();
    const std::size_t dof_point_count =
        op.mesh.current_dof_points.size();
    mbj_hash_value(hash, node_count);
    mbj_hash_value(hash, element_count);
    mbj_hash_value(hash, dof_point_count);
    for (const Vec3& point : op.mesh.nodes) {
        mbj_hash_value(hash, point.x);
        mbj_hash_value(hash, point.y);
        mbj_hash_value(hash, point.z);
    }
    for (const Vec3& point : op.mesh.current_dof_points) {
        mbj_hash_value(hash, point.x);
        mbj_hash_value(hash, point.y);
        mbj_hash_value(hash, point.z);
    }
    for (const MullerP2Element& element : op.mesh.elements) {
        mbj_hash_bytes(
            hash, element.nodes.data(),
            element.nodes.size() * sizeof(element.nodes[0]));
        mbj_hash_bytes(
            hash, element.topology_vertices.data(),
            element.topology_vertices.size() *
                sizeof(element.topology_vertices[0]));
        mbj_hash_bytes(
            hash, element.topology_edges.data(),
            element.topology_edges.size() *
                sizeof(element.topology_edges[0]));
        mbj_hash_bytes(
            hash, element.edge_orientations.data(),
            element.edge_orientations.size() *
                sizeof(element.edge_orientations[0]));
    }
    return hash;
}

template <typename T>
bool mbj_cache_read(std::ifstream& input, T& value)
{
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    return static_cast<bool>(input);
}

template <typename T>
void mbj_cache_write(std::ofstream& output, const T& value)
{
    output.write(
        reinterpret_cast<const char*>(&value), sizeof(value));
}

bool load_mbj_cache(
    MullerMbjPreconditioner& result,
    const MullerFmmOperator& op,
    int scalar_nodes_per_block,
    int overlap_nodes,
    const std::string& path)
{
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input)
        return false;
    char magic[8] = {};
    std::uint64_t version = 0;
    std::uint64_t signature = 0;
    std::uint64_t stored_system_dofs = 0;
    std::uint64_t block_count = 0;
    const int group_count = op.current_dofs / 2;
    const std::uint64_t expected_block_count =
        static_cast<std::uint64_t>(
            (group_count + scalar_nodes_per_block - 1) /
            scalar_nodes_per_block);
    const std::uint64_t maximum_block_dimension =
        static_cast<std::uint64_t>(
            4 * (scalar_nodes_per_block + 2 * overlap_nodes));
    input.read(magic, sizeof(magic));
    if (!mbj_cache_read(input, version) ||
        !mbj_cache_read(input, signature) ||
        !mbj_cache_read(input, stored_system_dofs) ||
        !mbj_cache_read(input, block_count) ||
        std::memcmp(magic, "MULMBJ1", 8) != 0 ||
        version != 1 ||
        signature != mbj_cache_signature(
            op, scalar_nodes_per_block, overlap_nodes) ||
        stored_system_dofs !=
            static_cast<std::uint64_t>(op.system_dofs) ||
        block_count != expected_block_count) {
        return false;
    }

    std::vector<MullerMbjBlock> blocks(
        static_cast<std::size_t>(block_count));
    for (MullerMbjBlock& block : blocks) {
        std::uint64_t dof_count = 0;
        std::uint64_t lu_count = 0;
        std::uint64_t pivot_count = 0;
        if (!mbj_cache_read(input, block.core_dof_begin) ||
            !mbj_cache_read(input, block.core_dof_end) ||
            !mbj_cache_read(input, dof_count) ||
            !mbj_cache_read(input, lu_count) ||
            !mbj_cache_read(input, pivot_count) ||
            dof_count == 0 ||
            dof_count > maximum_block_dimension ||
            lu_count != dof_count * dof_count ||
            pivot_count != dof_count ||
            block.core_dof_begin < 0 ||
            block.core_dof_end < block.core_dof_begin ||
            block.core_dof_end > static_cast<int>(dof_count)) {
            return false;
        }
        block.dofs.resize(static_cast<std::size_t>(dof_count));
        block.lu.resize(static_cast<std::size_t>(lu_count));
        block.pivots.resize(static_cast<std::size_t>(pivot_count));
        input.read(
            reinterpret_cast<char*>(block.dofs.data()),
            static_cast<std::streamsize>(
                block.dofs.size() * sizeof(block.dofs[0])));
        input.read(
            reinterpret_cast<char*>(block.lu.data()),
            static_cast<std::streamsize>(
                block.lu.size() * sizeof(block.lu[0])));
        input.read(
            reinterpret_cast<char*>(block.pivots.data()),
            static_cast<std::streamsize>(
                block.pivots.size() * sizeof(block.pivots[0])));
        if (!input)
            return false;
        for (int dof : block.dofs) {
            if (dof < 0 || dof >= op.system_dofs)
                return false;
        }
    }

    result.system_dofs = op.system_dofs;
    result.scalar_nodes_per_block = scalar_nodes_per_block;
    result.overlap_nodes = overlap_nodes;
    result.stores_inverse = false;
    result.coarse_rank = 0;
    result.blocks.swap(blocks);
    result.coarse_action.clear();
    result.coarse_update.clear();
    result.coarse_gram_lu.clear();
    result.coarse_gram_pivots.clear();
    return true;
}

void save_mbj_cache(
    const MullerMbjPreconditioner& preconditioner,
    const MullerFmmOperator& op,
    const std::string& path)
{
    const std::string temporary = path + ".tmp." +
        std::to_string(static_cast<long long>(getpid()));
    std::ofstream output(
        temporary.c_str(),
        std::ios::binary | std::ios::out | std::ios::trunc);
    if (!output)
        throw std::runtime_error(
            "cannot create Muller MBJ cache " + temporary);
    const char magic[8] = {'M', 'U', 'L', 'M', 'B', 'J', '1', '\0'};
    const std::uint64_t version = 1;
    const std::uint64_t signature = mbj_cache_signature(
        op, preconditioner.scalar_nodes_per_block,
        preconditioner.overlap_nodes);
    const std::uint64_t system_dofs =
        static_cast<std::uint64_t>(preconditioner.system_dofs);
    const std::uint64_t block_count =
        static_cast<std::uint64_t>(preconditioner.blocks.size());
    output.write(magic, sizeof(magic));
    mbj_cache_write(output, version);
    mbj_cache_write(output, signature);
    mbj_cache_write(output, system_dofs);
    mbj_cache_write(output, block_count);
    for (const MullerMbjBlock& block : preconditioner.blocks) {
        const std::uint64_t dof_count =
            static_cast<std::uint64_t>(block.dofs.size());
        const std::uint64_t lu_count =
            static_cast<std::uint64_t>(block.lu.size());
        const std::uint64_t pivot_count =
            static_cast<std::uint64_t>(block.pivots.size());
        mbj_cache_write(output, block.core_dof_begin);
        mbj_cache_write(output, block.core_dof_end);
        mbj_cache_write(output, dof_count);
        mbj_cache_write(output, lu_count);
        mbj_cache_write(output, pivot_count);
        output.write(
            reinterpret_cast<const char*>(block.dofs.data()),
            static_cast<std::streamsize>(
                block.dofs.size() * sizeof(block.dofs[0])));
        output.write(
            reinterpret_cast<const char*>(block.lu.data()),
            static_cast<std::streamsize>(
                block.lu.size() * sizeof(block.lu[0])));
        output.write(
            reinterpret_cast<const char*>(block.pivots.data()),
            static_cast<std::streamsize>(
                block.pivots.size() * sizeof(block.pivots[0])));
    }
    output.flush();
    if (!output)
        throw std::runtime_error(
            "cannot write Muller MBJ cache " + temporary);
    output.close();
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        throw std::runtime_error(
            "cannot atomically replace Muller MBJ cache " + path);
    }
}

template <typename T>
void read_value(std::ifstream& stream, T& value)
{
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!stream)
        throw std::runtime_error("truncated neural Muller preconditioner");
}

uint64_t morton_code_3d(uint32_t x, uint32_t y, uint32_t z)
{
    uint64_t code = 0;
    for (int bit = 0; bit < 21; bit++) {
        code |= (uint64_t)((x >> bit) & 1u) << (3 * bit);
        code |= (uint64_t)((y >> bit) & 1u) << (3 * bit + 1);
        code |= (uint64_t)((z >> bit) & 1u) << (3 * bit + 2);
    }
    return code;
}

std::vector<int> morton_node_order(
    const std::vector<Vec3>& nodes)
{
    double lower[3] = {
        nodes[0].x, nodes[0].y, nodes[0].z
    };
    double upper[3] = {lower[0], lower[1], lower[2]};
    for (const Vec3& node : nodes) {
        const double values[3] = {node.x, node.y, node.z};
        for (int axis = 0; axis < 3; axis++) {
            lower[axis] = std::min(lower[axis], values[axis]);
            upper[axis] = std::max(upper[axis], values[axis]);
        }
    }
    constexpr double maximum =
        (double)((1u << 21) - 1u);
    std::vector<std::pair<uint64_t, int>> coded;
    coded.reserve(nodes.size());
    for (int node = 0; node < (int)nodes.size(); node++) {
        const double values[3] = {
            nodes[node].x, nodes[node].y, nodes[node].z
        };
        uint32_t coordinate[3] = {0, 0, 0};
        for (int axis = 0; axis < 3; axis++) {
            const double span = upper[axis] - lower[axis];
            const double normalized = span > 0.0
                ? (values[axis] - lower[axis]) / span
                : 0.0;
            coordinate[axis] = (uint32_t)std::llround(
                std::max(0.0, std::min(1.0, normalized)) *
                maximum);
        }
        coded.emplace_back(
            morton_code_3d(
                coordinate[0], coordinate[1], coordinate[2]),
            node);
    }
    std::sort(coded.begin(), coded.end());
    std::vector<int> order;
    order.reserve(nodes.size());
    for (const auto& pair : coded)
        order.push_back(pair.second);
    return order;
}

std::vector<Vec3> muller_dof_group_points(
    const MullerP2Mesh& mesh)
{
    if (mesh.current_dofs() % 2 != 0 ||
        (int)mesh.current_dof_points.size() != mesh.current_dofs()) {
        throw std::logic_error(
            "Muller current DOF geometry is inconsistent");
    }
    std::vector<Vec3> points(mesh.current_dofs() / 2);
    for (int group = 0; group < (int)points.size(); group++)
        points[group] = mesh.current_dof_points[2 * group];
    return points;
}

void factorize(
    std::vector<cdouble>& matrix,
    std::vector<int>& pivots,
    int dimension)
{
    pivots.resize(dimension);
    for (int column = 0; column < dimension; column++) {
        int pivot = column;
        double pivot_norm =
            std::abs(matrix[(size_t)column * dimension + column]);
        for (int row = column + 1; row < dimension; row++) {
            const double candidate =
                std::abs(matrix[(size_t)row * dimension + column]);
            if (candidate > pivot_norm) {
                pivot = row;
                pivot_norm = candidate;
            }
        }
        if (pivot_norm < 1.0e-24)
            throw std::runtime_error("singular Muller MBJ block");
        pivots[column] = pivot;
        if (pivot != column) {
            for (int entry = 0; entry < dimension; entry++) {
                std::swap(
                    matrix[(size_t)column * dimension + entry],
                    matrix[(size_t)pivot * dimension + entry]);
            }
        }
        const cdouble diagonal =
            matrix[(size_t)column * dimension + column];
        for (int row = column + 1; row < dimension; row++) {
            const cdouble multiplier =
                matrix[(size_t)row * dimension + column] /
                diagonal;
            matrix[(size_t)row * dimension + column] = multiplier;
            for (int entry = column + 1;
                 entry < dimension; entry++) {
                matrix[(size_t)row * dimension + entry] -=
                    multiplier *
                    matrix[(size_t)column * dimension + entry];
            }
        }
    }
}

} // namespace

void MullerMbjPreconditioner::build(
    const MullerFmmOperator& op,
    int requested_scalar_nodes_per_block,
    int requested_overlap_nodes)
{
    const auto setup_start = std::chrono::steady_clock::now();
    if (op.system_dofs <= 0)
        throw std::invalid_argument(
            "invalid Muller FMM operator for MBJ");
    system_dofs = op.system_dofs;
    scalar_nodes_per_block =
        std::max(1, requested_scalar_nodes_per_block);
    overlap_nodes = std::max(0, requested_overlap_nodes);
    stores_inverse = false;
    coarse_rank = 0;
    coarse_setup_seconds = 0.0;
    coarse_action.clear();
    coarse_update.clear();
    coarse_gram_lu.clear();
    coarse_gram_pivots.clear();
    cache_hit = false;
    cache_io_seconds = 0.0;
    cache_path.clear();
    blocks.clear();
    const std::vector<Vec3> group_points =
        muller_dof_group_points(op.mesh);
    const int node_count = (int)group_points.size();
    const auto ordering_start = std::chrono::steady_clock::now();
    const std::vector<int> order =
        morton_node_order(group_points);
    ordering_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            ordering_start).count();
    assembly_seconds = 0.0;
    factorization_seconds = 0.0;
    const int block_count =
        (node_count + scalar_nodes_per_block - 1) /
        scalar_nodes_per_block;
    blocks.resize(block_count);
    std::vector<std::vector<int>> dof_groups(block_count);
    std::vector<std::vector<int>> group_support(node_count);
    for (int element = 0;
         element < (int)op.mesh.elements.size(); element++) {
        const MullerFrameSample center =
            evaluate_muller_frame(
                op.mesh, element, 1.0 / 3.0, 1.0 / 3.0);
        const MullerBasisSample basis =
            evaluate_muller_basis(op.mesh, element, center);
        std::vector<int> element_groups;
        for (int local = 0; local < basis.count; local++) {
            const int group = basis.dofs[local] / 2;
            if (std::find(
                    element_groups.begin(), element_groups.end(),
                    group) == element_groups.end()) {
                group_support[group].push_back(element);
                element_groups.push_back(group);
            }
        }
    }
    std::vector<std::vector<int>> block_support(block_count);
    for (int block_index = 0;
         block_index < block_count; block_index++) {
        const int core_begin =
            block_index * scalar_nodes_per_block;
        const int core_end = std::min(
            node_count, core_begin + scalar_nodes_per_block);
        const int begin = std::max(
            0, core_begin - overlap_nodes);
        const int end = std::min(
            node_count, core_end + overlap_nodes);
        dof_groups[block_index].reserve(end - begin);
        MullerMbjBlock& block = blocks[block_index];
        block.dofs.reserve(4 * (end - begin));
        for (int index = begin; index < end; index++) {
            const int group = order[index];
            dof_groups[block_index].push_back(group);
            block.dofs.push_back(2 * group);
            block.dofs.push_back(2 * group + 1);
            block.dofs.push_back(
                op.current_dofs + 2 * group);
            block.dofs.push_back(
                op.current_dofs + 2 * group + 1);
        }
        block.core_dof_begin = 4 * (core_begin - begin);
        block.core_dof_end =
            block.core_dof_begin + 4 * (core_end - core_begin);
        std::vector<int>& support = block_support[block_index];
        for (int group : dof_groups[block_index]) {
            support.insert(
                support.end(),
                group_support[group].begin(),
                group_support[group].end());
        }
        std::sort(support.begin(), support.end());
        support.erase(
            std::unique(support.begin(), support.end()),
            support.end());
    }
#ifdef _OPENMP
    setup_threads = std::min(block_count, omp_get_max_threads());
#else
    setup_threads = 1;
#endif

    std::exception_ptr failure;
    const auto assembly_start =
        std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1) num_threads(setup_threads)
    for (int block_index = 0;
         block_index < block_count; block_index++) {
        try {
            blocks[block_index].lu =
                assemble_muller_nodal_block(
                    op, dof_groups[block_index],
                    &block_support[block_index]);
        } catch (...) {
#pragma omp critical(muller_mbj_build_failure)
            {
                if (!failure)
                    failure = std::current_exception();
            }
        }
    }
    assembly_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            assembly_start).count();
    if (failure)
        std::rethrow_exception(failure);

    const auto factorization_start =
        std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static) num_threads(setup_threads)
    for (int block_index = 0;
         block_index < block_count; block_index++) {
        MullerMbjBlock& block = blocks[block_index];
        factorize(
            block.lu, block.pivots,
            (int)block.dofs.size());
    }
    factorization_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            factorization_start).count();
    setup_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            setup_start).count();
}

void MullerMbjPreconditioner::build_cached(
    const MullerFmmOperator& op,
    int requested_scalar_nodes_per_block,
    int requested_overlap_nodes,
    const std::string& path)
{
    if (path.empty()) {
        build(
            op, requested_scalar_nodes_per_block,
            requested_overlap_nodes);
        return;
    }
    const int nodes = std::max(
        1, requested_scalar_nodes_per_block);
    const int overlap = std::max(0, requested_overlap_nodes);
    const auto cache_start = std::chrono::steady_clock::now();
    if (load_mbj_cache(*this, op, nodes, overlap, path)) {
        cache_hit = true;
        cache_path = path;
        cache_io_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                cache_start).count();
        setup_seconds = cache_io_seconds;
        ordering_seconds = 0.0;
        assembly_seconds = 0.0;
        factorization_seconds = 0.0;
        setup_threads = 1;
        std::printf(
            "  [MBJ] Cache hit: %s (%zu blocks, %.3fs)\n",
            path.c_str(), blocks.size(), cache_io_seconds);
        std::fflush(stdout);
        return;
    }
    std::printf("  [MBJ] Cache miss: %s\n", path.c_str());
    std::fflush(stdout);
    build(op, nodes, overlap);
    const auto write_start = std::chrono::steady_clock::now();
    save_mbj_cache(*this, op, path);
    cache_io_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            write_start).count();
    cache_path = path;
    std::printf(
        "  [MBJ] Stored cache: %s (%.3fs)\n",
        path.c_str(), cache_io_seconds);
    std::fflush(stdout);
}

void MullerMbjPreconditioner::build_coarse(
    MullerFmmOperator& op,
    int requested_rank)
{
    const auto start = std::chrono::steady_clock::now();
    coarse_rank = 0;
    coarse_setup_seconds = 0.0;
    coarse_action.clear();
    coarse_update.clear();
    coarse_gram_lu.clear();
    coarse_gram_pivots.clear();
    if (requested_rank <= 0)
        return;
    if (system_dofs != op.system_dofs)
        throw std::invalid_argument("MBJ/coarse operator size mismatch");
    if (requested_rank > 64)
        throw std::invalid_argument("MBJ coarse rank must be at most 64");

    const int rank = requested_rank;
    const std::vector<Vec3> group_points =
        muller_dof_group_points(op.mesh);
    const int node_count = (int)group_points.size();
    const int n = op.system_dofs;
    std::vector<cdouble> basis((size_t)rank * n, cdouble(0.0));
    Vec3 center(0.0, 0.0, 0.0);
    for (const Vec3& point : group_points)
        center = center + point;
    center = center * (1.0 / std::max(1, node_count));
    double radius_squared = 0.0;
    for (const Vec3& point : group_points)
        radius_squared += (point - center).norm2();
    const double scale = std::max(
        std::sqrt(radius_squared / std::max(1, node_count)), 1.0e-12);

    for (int column = 0; column < rank; column++) {
        cdouble* candidate = basis.data() + (size_t)column * n;
        const int channel = column % 4;
        const int mode = column / 4;
        for (int node = 0; node < node_count; node++) {
            const Vec3 relative =
                (group_points[node] - center) * (1.0 / scale);
            cdouble value(0.0);
            switch (mode) {
                case 0: value = 1.0; break;
                case 1: value = relative.x; break;
                case 2: value = relative.y; break;
                case 3: value = relative.z; break;
                case 4:
                    value = std::exp(
                        cdouble(0.0, 1.0) *
                        op.k_exterior *
                        (group_points[node].z - center.z));
                    break;
                case 5: value = relative.x * relative.y; break;
                case 6: value = relative.x * relative.z; break;
                case 7: value = relative.y * relative.z; break;
                case 8:
                    value = relative.x * relative.x -
                            relative.y * relative.y;
                    break;
                case 9:
                    value = 3.0 * relative.z * relative.z -
                            relative.norm2();
                    break;
                default: {
                    const int harmonic = mode - 8;
                    value = std::exp(
                        cdouble(0.0, (double)harmonic) *
                        op.k_exterior *
                        (group_points[node].z - center.z));
                    break;
                }
            }
            const int current_offset = channel >= 2 ? op.current_dofs : 0;
            const int tangent = channel % 2;
            candidate[current_offset + 2 * node + tangent] = value;
        }
        for (int pass = 0; pass < 2; pass++) {
            for (int previous = 0; previous < column; previous++) {
                const cdouble* vector =
                    basis.data() + (size_t)previous * n;
                cdouble projection(0.0);
                for (int i = 0; i < n; i++)
                    projection += std::conj(vector[i]) * candidate[i];
                for (int i = 0; i < n; i++)
                    candidate[i] -= vector[i] * projection;
            }
        }
        double norm_squared = 0.0;
        for (int i = 0; i < n; i++)
            norm_squared += std::norm(candidate[i]);
        const double candidate_norm = std::sqrt(norm_squared);
        if (!(candidate_norm > 1.0e-12))
            throw std::runtime_error("dependent Muller coarse basis mode");
        for (int i = 0; i < n; i++)
            candidate[i] /= candidate_norm;
    }

    coarse_action.resize((size_t)rank * n);
    coarse_update.resize((size_t)rank * n);
    std::vector<cdouble> local_action(n);
    for (int column = 0; column < rank; column++) {
        const cdouble* vector = basis.data() + (size_t)column * n;
        cdouble* action =
            coarse_action.data() + (size_t)column * n;
        op.matvec(vector, action);
        apply(action, local_action.data());
        cdouble* update =
            coarse_update.data() + (size_t)column * n;
        for (int i = 0; i < n; i++)
            update[i] = vector[i] - local_action[i];
        std::printf(
            "  [MBJ coarse %d/%d] full FMM action complete\n",
            column + 1, rank);
        std::fflush(stdout);
    }

    coarse_gram_lu.assign((size_t)rank * rank, cdouble(0.0));
    for (int row = 0; row < rank; row++) {
        const cdouble* first =
            coarse_action.data() + (size_t)row * n;
        for (int column = 0; column < rank; column++) {
            const cdouble* second =
                coarse_action.data() + (size_t)column * n;
            cdouble value(0.0);
            for (int i = 0; i < n; i++)
                value += std::conj(first[i]) * second[i];
            coarse_gram_lu[(size_t)row * rank + column] = value;
        }
    }
    factorize(coarse_gram_lu, coarse_gram_pivots, rank);
    coarse_rank = rank;
    coarse_setup_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
}

void MullerMbjPreconditioner::load_neural(
    const MullerFmmOperator& op,
    const std::string& path)
{
    if (op.mesh.basis_kind != MullerBasisKind::NodalP2) {
        throw std::invalid_argument(
            "nodal neural Muller weights are incompatible with "
            "the H(div)-BDM1 basis");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error(
            "cannot open neural Muller preconditioner: " + path);
    char magic[8];
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, "MULPRC1\0", 8) != 0)
        throw std::runtime_error(
            "invalid neural Muller preconditioner magic");
    uint32_t version = 0;
    uint32_t total_nodes = 0;
    uint32_t requested_block_nodes = 0;
    uint32_t block_count = 0;
    read_value(input, version);
    read_value(input, total_nodes);
    read_value(input, requested_block_nodes);
    read_value(input, block_count);
    if (version != 1)
        throw std::runtime_error(
            "unsupported neural Muller preconditioner version");
    if (total_nodes != (uint32_t)op.mesh.scalar_nodes())
        throw std::runtime_error(
            "neural Muller preconditioner node count mismatch");

    system_dofs = op.system_dofs;
    scalar_nodes_per_block = (int)requested_block_nodes;
    overlap_nodes = 0;
    coarse_rank = 0;
    coarse_setup_seconds = 0.0;
    coarse_action.clear();
    coarse_update.clear();
    coarse_gram_lu.clear();
    coarse_gram_pivots.clear();
    stores_inverse = true;
    ordering_seconds = 0.0;
    assembly_seconds = 0.0;
    factorization_seconds = 0.0;
    setup_threads = 1;
    const auto setup_start = std::chrono::steady_clock::now();
    blocks.clear();
    blocks.reserve(block_count);
    const std::vector<int> order =
        morton_node_order(op.mesh.nodes);
    int begin = 0;
    for (uint32_t block_index = 0;
         block_index < block_count; block_index++) {
        uint32_t nodes = 0;
        read_value(input, nodes);
        if (nodes == 0 || begin + (int)nodes > (int)order.size())
            throw std::runtime_error(
                "invalid neural Muller block size");
        MullerMbjBlock block;
        block.dofs.reserve(4 * nodes);
        for (uint32_t index = 0; index < nodes; index++) {
            const int node = order[begin + index];
            block.dofs.push_back(2 * node);
            block.dofs.push_back(2 * node + 1);
            block.dofs.push_back(op.current_dofs + 2 * node);
            block.dofs.push_back(op.current_dofs + 2 * node + 1);
        }
        const int dimension = 4 * (int)nodes;
        block.core_dof_begin = 0;
        block.core_dof_end = dimension;
        block.lu.resize((size_t)dimension * dimension);
        for (cdouble& value : block.lu) {
            float real = 0.0f;
            float imag = 0.0f;
            read_value(input, real);
            read_value(input, imag);
            value = cdouble(real, imag);
        }
        blocks.push_back(std::move(block));
        begin += nodes;
    }
    if (begin != (int)order.size())
        throw std::runtime_error(
            "neural Muller preconditioner does not cover all nodes");
    char trailing = 0;
    if (input.read(&trailing, 1))
        throw std::runtime_error(
            "trailing data in neural Muller preconditioner");
    setup_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            setup_start).count();
}
