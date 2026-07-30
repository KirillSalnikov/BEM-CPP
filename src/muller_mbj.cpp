#include "muller_mbj.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using cdouble = std::complex<double>;

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
    if (nodes.empty())
        return {};
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
    constexpr double max_coordinate =
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
                max_coordinate);
        }
        coded.emplace_back(
            morton_code_3d(
                coordinate[0], coordinate[1], coordinate[2]),
            node);
    }
    std::sort(coded.begin(), coded.end());
    std::vector<int> result;
    result.reserve(nodes.size());
    for (const auto& pair : coded)
        result.push_back(pair.second);
    return result;
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
    int n)
{
    pivots.resize(n);
    for (int k = 0; k < n; k++) {
        int pivot = k;
        double pivot_norm =
            std::abs(matrix[(size_t)k * n + k]);
        for (int row = k + 1; row < n; row++) {
            const double candidate =
                std::abs(matrix[(size_t)row * n + k]);
            if (candidate > pivot_norm) {
                pivot = row;
                pivot_norm = candidate;
            }
        }
        if (pivot_norm < 1.0e-24)
            throw std::runtime_error("singular Muller MBJ block");
        pivots[k] = pivot;
        if (pivot != k) {
            for (int col = 0; col < n; col++)
                std::swap(
                    matrix[(size_t)k * n + col],
                    matrix[(size_t)pivot * n + col]);
        }
        const cdouble diagonal =
            matrix[(size_t)k * n + k];
        for (int row = k + 1; row < n; row++) {
            const cdouble multiplier =
                matrix[(size_t)row * n + k] / diagonal;
            matrix[(size_t)row * n + k] = multiplier;
            for (int col = k + 1; col < n; col++)
                matrix[(size_t)row * n + col] -=
                    multiplier * matrix[(size_t)k * n + col];
        }
    }
}

void solve(
    const std::vector<cdouble>& lu,
    const std::vector<int>& pivots,
    cdouble* vector,
    int n)
{
    for (int k = 0; k < n; k++) {
        if (pivots[k] != k)
            std::swap(vector[k], vector[pivots[k]]);
    }
    for (int k = 0; k < n; k++) {
        for (int row = k + 1; row < n; row++)
            vector[row] -=
                lu[(size_t)row * n + k] * vector[k];
    }
    for (int row = n - 1; row >= 0; row--) {
        cdouble value = vector[row];
        for (int col = row + 1; col < n; col++)
            value -= lu[(size_t)row * n + col] * vector[col];
        vector[row] =
            value / lu[(size_t)row * n + row];
    }
}

void solve_factorized(
    const std::vector<cdouble>& lu,
    const std::vector<int>& pivots,
    std::vector<cdouble>& vector)
{
    solve(lu, pivots, vector.data(), (int)vector.size());
}

} // namespace

void MullerMbjPreconditioner::build(
    const MullerDenseSystem& system,
    int requested_scalar_nodes_per_block,
    int requested_overlap_nodes)
{
    if (system.system_dofs <= 0 ||
        system.matrix.size() !=
            (size_t)system.system_dofs * system.system_dofs) {
        throw std::invalid_argument(
            "invalid dense Muller system for MBJ");
    }
    system_dofs = system.system_dofs;
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
    blocks.clear();

    const std::vector<Vec3> group_points =
        muller_dof_group_points(system.mesh);
    const int node_count = (int)group_points.size();
    const std::vector<int> order =
        morton_node_order(group_points);

    const int current_dofs = system.current_dofs;
    for (int core_begin = 0; core_begin < node_count;
         core_begin += scalar_nodes_per_block) {
        const int core_end = std::min(
            node_count, core_begin + scalar_nodes_per_block);
        const int begin = std::max(
            0, core_begin - overlap_nodes);
        const int end = std::min(
            node_count, core_end + overlap_nodes);
        MullerMbjBlock block;
        block.dofs.reserve(4 * (end - begin));
        for (int index = begin; index < end; index++) {
            const int node = order[index];
            block.dofs.push_back(2 * node);
            block.dofs.push_back(2 * node + 1);
            block.dofs.push_back(current_dofs + 2 * node);
            block.dofs.push_back(current_dofs + 2 * node + 1);
        }
        const int dimension = (int)block.dofs.size();
        block.core_dof_begin = 4 * (core_begin - begin);
        block.core_dof_end =
            block.core_dof_begin + 4 * (core_end - core_begin);
        block.lu.resize((size_t)dimension * dimension);
        for (int row = 0; row < dimension; row++) {
            for (int col = 0; col < dimension; col++) {
                block.lu[(size_t)row * dimension + col] =
                    system.matrix[
                        (size_t)block.dofs[row] * system_dofs +
                        block.dofs[col]];
            }
        }
        factorize(block.lu, block.pivots, dimension);
        blocks.push_back(std::move(block));
    }
}

void MullerMbjPreconditioner::apply(
    const cdouble* rhs,
    cdouble* solution) const
{
    std::fill(
        solution, solution + system_dofs, cdouble(0.0));
#ifdef _OPENMP
    const int thread_count = std::min(
        (int)blocks.size(), omp_get_max_threads());
#pragma omp parallel for schedule(static) num_threads(thread_count)
#endif
    for (int block_index = 0;
         block_index < (int)blocks.size(); block_index++) {
        const MullerMbjBlock& block = blocks[block_index];
        const int dimension = (int)block.dofs.size();
        std::vector<cdouble> local(dimension);
        for (int i = 0; i < dimension; i++)
            local[i] = rhs[block.dofs[i]];
        if (stores_inverse) {
            std::vector<cdouble> product(
                dimension, cdouble(0.0));
            for (int row = 0; row < dimension; row++) {
                for (int column = 0; column < dimension; column++) {
                    product[row] +=
                        block.lu[(size_t)row * dimension + column] *
                        local[column];
                }
            }
            local.swap(product);
        } else {
            solve(block.lu, block.pivots, local.data(), dimension);
        }
        for (int i = block.core_dof_begin;
             i < block.core_dof_end; i++)
            solution[block.dofs[i]] = local[i];
    }
    if (coarse_rank > 0) {
        std::vector<cdouble> coefficients(coarse_rank, cdouble(0.0));
#pragma omp parallel for schedule(static)
        for (int column = 0; column < coarse_rank; column++) {
            const cdouble* action =
                coarse_action.data() + (size_t)column * system_dofs;
            cdouble value(0.0);
            for (int i = 0; i < system_dofs; i++)
                value += std::conj(action[i]) * rhs[i];
            coefficients[column] = value;
        }
        solve_factorized(
            coarse_gram_lu, coarse_gram_pivots, coefficients);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < system_dofs; i++) {
            cdouble correction(0.0);
            for (int column = 0; column < coarse_rank; column++) {
                correction += coarse_update[
                    (size_t)column * system_dofs + i] *
                    coefficients[column];
            }
            solution[i] += correction;
        }
    }
}

double MullerMbjPreconditioner::storage_megabytes() const
{
    size_t bytes = 0;
    for (const MullerMbjBlock& block : blocks) {
        bytes += block.lu.size() * sizeof(cdouble);
        bytes += block.dofs.size() * sizeof(int);
        bytes += block.pivots.size() * sizeof(int);
    }
    bytes += coarse_action.size() * sizeof(cdouble);
    bytes += coarse_update.size() * sizeof(cdouble);
    bytes += coarse_gram_lu.size() * sizeof(cdouble);
    bytes += coarse_gram_pivots.size() * sizeof(int);
    return (double)bytes / (1024.0 * 1024.0);
}
