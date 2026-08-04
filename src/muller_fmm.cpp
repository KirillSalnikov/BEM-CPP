#include "muller_fmm.h"

#include "muller_duffy.h"
#include "quadrature.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>
#include <unistd.h>

namespace {

using cdouble = std::complex<double>;

struct ElementAdjacency {
    int shared_count = 0;
    int test_local[3] = {-1, -1, -1};
    int trial_local[3] = {-1, -1, -1};
};

struct CorrectionValues {
    cdouble k1 = 0.0;
    cdouble k2_epsilon = 0.0;
    cdouble k2_mu = 0.0;
};

struct LocalCorrectionBlock {
    int test_count = 0;
    int trial_count = 0;
    std::array<int, 3> test_edge_orientations{{1, 1, 1}};
    std::array<int, 3> trial_edge_orientations{{1, 1, 1}};
    std::array<CorrectionValues, 12 * 12> values;
};

struct NearPairTemplateKey {
    std::vector<std::int64_t> values;

    bool operator<(const NearPairTemplateKey& other) const
    {
        return values < other.values;
    }
};

struct CorrectionFingerprint {
    std::uint64_t first = UINT64_C(1469598103934665603);
    std::uint64_t second = UINT64_C(1099511628211);

    void add(const void* data, size_t size)
    {
        const unsigned char* bytes =
            static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < size; i++) {
            first ^= bytes[i];
            first *= UINT64_C(1099511628211);
            second ^= static_cast<std::uint64_t>(bytes[i]) +
                UINT64_C(0x9e3779b97f4a7c15) +
                (second << 6) + (second >> 2);
        }
    }

    template <typename T>
    void add_value(const T& value)
    {
        add(&value, sizeof(value));
    }

    void add_complex(const cdouble& value)
    {
        const double real = value.real();
        const double imaginary = value.imag();
        add_value(real);
        add_value(imaginary);
    }

    void add_vec3(const Vec3& value)
    {
        add_value(value.x);
        add_value(value.y);
        add_value(value.z);
    }
};

struct CorrectionCacheDiskEntry {
    std::int32_t column;
    std::uint32_t reserved;
    double k1_real;
    double k1_imaginary;
    double k2_epsilon_real;
    double k2_epsilon_imaginary;
    double k2_mu_real;
    double k2_mu_imaginary;
};

static_assert(sizeof(int) == sizeof(std::int32_t),
              "near-correction cache requires 32-bit int");
static_assert(sizeof(CorrectionCacheDiskEntry) == 56,
              "unexpected near-correction cache entry layout");

const char correction_cache_magic[16] = {
    'B', 'E', 'M', 'M', 'N', 'C', 'A', 'C',
    'H', 'E', '0', '0', '0', '1', '\r', '\n'
};
const std::uint32_t correction_cache_endian =
    UINT32_C(0x01020304);
const std::uint32_t correction_cache_algorithm_version = 2;

int muller_fmm_digits_cap()
{
    const char* value = std::getenv("BEM_MULLER_FMM_DIGITS_CAP");
    if (value == nullptr)
        return 5;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0')
        return 5;
    return std::max(1L, std::min(parsed, 10L));
}

std::vector<std::vector<int>> build_assembly_colors(
    const MullerP2Mesh& mesh)
{
    std::vector<std::vector<int>> colors;
    std::vector<std::vector<unsigned char>> occupied;
    for (int element = 0;
         element < static_cast<int>(mesh.elements.size()); element++) {
        const MullerFrameSample center = evaluate_muller_frame(
            mesh, element, 1.0 / 3.0, 1.0 / 3.0);
        const MullerBasisSample basis =
            evaluate_muller_basis(mesh, element, center);
        int selected = -1;
        for (int color = 0;
             color < static_cast<int>(colors.size()); color++) {
            bool conflict = false;
            for (int local = 0; local < basis.count; local++) {
                if (occupied[color][basis.dofs[local]]) {
                    conflict = true;
                    break;
                }
            }
            if (!conflict) {
                selected = color;
                break;
            }
        }
        if (selected < 0) {
            selected = static_cast<int>(colors.size());
            colors.emplace_back();
            occupied.emplace_back(
                mesh.current_dofs(), static_cast<unsigned char>(0));
        }
        colors[selected].push_back(element);
        for (int local = 0; local < basis.count; local++)
            occupied[selected][basis.dofs[local]] = 1;
    }
    return colors;
}

CorrectionFingerprint correction_fingerprint(
    const MullerP2Mesh& mesh,
    cdouble k_exterior,
    cdouble k_interior,
    cdouble epsilon_exterior,
    cdouble epsilon_interior,
    cdouble mu_exterior,
    cdouble mu_interior,
    int regular_quadrature_order,
    int duffy_order)
{
    CorrectionFingerprint result;
    result.add_value(correction_cache_algorithm_version);
    result.add_value(regular_quadrature_order);
    result.add_value(duffy_order);
    result.add_complex(k_exterior);
    result.add_complex(k_interior);
    result.add_complex(epsilon_exterior);
    result.add_complex(epsilon_interior);
    result.add_complex(mu_exterior);
    result.add_complex(mu_interior);
    const std::uint64_t node_count = mesh.nodes.size();
    const std::uint64_t element_count = mesh.elements.size();
    result.add_value(node_count);
    result.add_value(element_count);
    const int edge_mode = static_cast<int>(mesh.edge_mode);
    result.add_value(edge_mode);
    result.add_value(mesh.feature_angle_degrees);
    for (size_t node = 0; node < mesh.nodes.size(); node++) {
        result.add_vec3(mesh.nodes[node]);
        result.add_vec3(mesh.normals[node]);
        result.add_vec3(mesh.tangent1[node]);
        result.add_vec3(mesh.tangent2[node]);
    }
    for (const MullerP2Element& element : mesh.elements) {
        for (int node : element.nodes)
            result.add_value(node);
        for (int vertex : element.topology_vertices)
            result.add_value(vertex);
        for (int edge : element.topology_edges)
            result.add_value(edge);
        for (int orientation : element.edge_orientations)
            result.add_value(orientation);
    }
    return result;
}

template <typename T>
bool read_cache_value(std::ifstream& input, T& value)
{
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    return static_cast<bool>(input);
}

template <typename T>
bool write_cache_value(std::ofstream& output, const T& value)
{
    output.write(
        reinterpret_cast<const char*>(&value), sizeof(value));
    return static_cast<bool>(output);
}

bool load_near_correction_cache(
    const char* path,
    const CorrectionFingerprint& fingerprint,
    int current_dofs,
    MullerNearCorrection& correction,
    int& color_count,
    std::string& reason)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        reason = "not found";
        return false;
    }
    char magic[sizeof(correction_cache_magic)];
    input.read(magic, sizeof(magic));
    std::uint32_t endian = 0;
    std::uint32_t algorithm_version = 0;
    std::uint64_t first = 0;
    std::uint64_t second = 0;
    std::uint64_t row_count = 0;
    std::uint64_t entry_count = 0;
    std::int32_t stored_colors = 0;
    std::uint32_t reserved = 0;
    if (!input ||
        !read_cache_value(input, endian) ||
        !read_cache_value(input, algorithm_version) ||
        !read_cache_value(input, first) ||
        !read_cache_value(input, second) ||
        !read_cache_value(input, row_count) ||
        !read_cache_value(input, entry_count) ||
        !read_cache_value(input, stored_colors) ||
        !read_cache_value(input, reserved)) {
        reason = "truncated header";
        return false;
    }
    if (std::memcmp(
            magic, correction_cache_magic, sizeof(magic)) != 0 ||
        endian != correction_cache_endian ||
        algorithm_version != correction_cache_algorithm_version) {
        reason = "unsupported format";
        return false;
    }
    if (first != fingerprint.first ||
        second != fingerprint.second) {
        reason = "geometry or physical parameters changed";
        return false;
    }
    const std::uint64_t expected_rows =
        static_cast<std::uint64_t>(current_dofs) + 1;
    const std::uint64_t maximum_entries =
        std::max<std::uint64_t>(
            UINT64_C(1000000),
            expected_rows * UINT64_C(4096));
    if (row_count != expected_rows ||
        entry_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<int>::max()) ||
        entry_count > maximum_entries ||
        stored_colors < 0) {
        reason = "invalid dimensions";
        return false;
    }

    MullerNearCorrection loaded;
    loaded.row_offsets.resize(static_cast<size_t>(row_count));
    input.read(
        reinterpret_cast<char*>(loaded.row_offsets.data()),
        loaded.row_offsets.size() * sizeof(int));
    if (!input) {
        reason = "truncated row offsets";
        return false;
    }
    if (loaded.row_offsets.front() != 0 ||
        loaded.row_offsets.back() != static_cast<int>(entry_count)) {
        reason = "invalid row offsets";
        return false;
    }
    for (size_t row = 1; row < loaded.row_offsets.size(); row++) {
        if (loaded.row_offsets[row] <
            loaded.row_offsets[row - 1]) {
            reason = "nonmonotone row offsets";
            return false;
        }
    }

    loaded.entries.resize(static_cast<size_t>(entry_count));
    const size_t chunk_size = 32768;
    std::vector<CorrectionCacheDiskEntry> buffer(chunk_size);
    for (size_t offset = 0; offset < loaded.entries.size();
         offset += chunk_size) {
        const size_t count = std::min(
            chunk_size, loaded.entries.size() - offset);
        input.read(
            reinterpret_cast<char*>(buffer.data()),
            count * sizeof(CorrectionCacheDiskEntry));
        if (!input) {
            reason = "truncated entries";
            return false;
        }
        for (size_t index = 0; index < count; index++) {
            const CorrectionCacheDiskEntry& stored = buffer[index];
            if (stored.column < 0 ||
                stored.column >= current_dofs ||
                stored.reserved != 0) {
                reason = "invalid entry";
                return false;
            }
            MullerNearCorrectionEntry& entry =
                loaded.entries[offset + index];
            entry.column = stored.column;
            entry.k1 = cdouble(
                stored.k1_real, stored.k1_imaginary);
            entry.k2_epsilon = cdouble(
                stored.k2_epsilon_real,
                stored.k2_epsilon_imaginary);
            entry.k2_mu = cdouble(
                stored.k2_mu_real, stored.k2_mu_imaginary);
        }
    }
    if (input.peek() != std::ifstream::traits_type::eof()) {
        reason = "trailing data";
        return false;
    }
    correction = std::move(loaded);
    color_count = stored_colors;
    reason.clear();
    return true;
}

bool save_near_correction_cache(
    const char* path,
    const CorrectionFingerprint& fingerprint,
    const MullerNearCorrection& correction,
    int color_count,
    std::string& reason)
{
    const std::string temporary = std::string(path) + ".tmp." +
        std::to_string(static_cast<long long>(getpid()));
    std::ofstream output(
        temporary.c_str(),
        std::ios::binary | std::ios::trunc);
    if (!output) {
        reason = "cannot create temporary file";
        return false;
    }
    const std::uint64_t row_count =
        correction.row_offsets.size();
    const std::uint64_t entry_count =
        correction.entries.size();
    const std::int32_t stored_colors = color_count;
    const std::uint32_t reserved = 0;
    output.write(
        correction_cache_magic, sizeof(correction_cache_magic));
    if (!write_cache_value(output, correction_cache_endian) ||
        !write_cache_value(
            output, correction_cache_algorithm_version) ||
        !write_cache_value(output, fingerprint.first) ||
        !write_cache_value(output, fingerprint.second) ||
        !write_cache_value(output, row_count) ||
        !write_cache_value(output, entry_count) ||
        !write_cache_value(output, stored_colors) ||
        !write_cache_value(output, reserved)) {
        reason = "cannot write header";
        output.close();
        std::remove(temporary.c_str());
        return false;
    }
    output.write(
        reinterpret_cast<const char*>(
            correction.row_offsets.data()),
        correction.row_offsets.size() * sizeof(int));
    const size_t chunk_size = 32768;
    std::vector<CorrectionCacheDiskEntry> buffer(chunk_size);
    for (size_t offset = 0; offset < correction.entries.size();
         offset += chunk_size) {
        const size_t count = std::min(
            chunk_size, correction.entries.size() - offset);
        for (size_t index = 0; index < count; index++) {
            const MullerNearCorrectionEntry& entry =
                correction.entries[offset + index];
            CorrectionCacheDiskEntry& stored = buffer[index];
            stored.column = entry.column;
            stored.reserved = 0;
            stored.k1_real = entry.k1.real();
            stored.k1_imaginary = entry.k1.imag();
            stored.k2_epsilon_real =
                entry.k2_epsilon.real();
            stored.k2_epsilon_imaginary =
                entry.k2_epsilon.imag();
            stored.k2_mu_real = entry.k2_mu.real();
            stored.k2_mu_imaginary = entry.k2_mu.imag();
        }
        output.write(
            reinterpret_cast<const char*>(buffer.data()),
            count * sizeof(CorrectionCacheDiskEntry));
        if (!output) {
            reason = "cannot write entries";
            output.close();
            std::remove(temporary.c_str());
            return false;
        }
    }
    output.close();
    if (!output) {
        reason = "cannot finalize file";
        std::remove(temporary.c_str());
        return false;
    }
    if (std::rename(temporary.c_str(), path) != 0) {
        reason = "cannot replace cache file";
        std::remove(temporary.c_str());
        return false;
    }
    reason.clear();
    return true;
}

struct MullerKernelValues {
    cdouble k1_scalar = 0.0;
    MullerTensor3 hessian;
    std::array<cdouble, 3> gradient_epsilon;
    std::array<cdouble, 3> gradient_mu;
};

MullerKernelValues evaluate_kernel_values(
    cdouble k_exterior,
    cdouble k_interior,
    cdouble epsilon_exterior,
    cdouble epsilon_interior,
    cdouble mu_exterior,
    cdouble mu_interior,
    const Vec3& displacement)
{
    const double radius = displacement.norm();
    if (radius <= 0.0)
        throw std::runtime_error(
            "Muller kernel values at zero distance");
    const cdouble imaginary(0.0, 1.0);
    const cdouble phi_exterior =
        std::exp(imaginary * k_exterior * radius) *
        INV4PI / radius;
    const cdouble phi_interior =
        std::exp(imaginary * k_interior * radius) *
        INV4PI / radius;
    MullerKernelValues values;
    values.k1_scalar =
        k_exterior * k_exterior * phi_exterior -
        k_interior * k_interior * phi_interior;
    values.hessian = muller_hessian_difference(
        k_exterior, k_interior, displacement);
    values.gradient_epsilon = muller_composite_gradient(
        epsilon_exterior, epsilon_interior,
        k_exterior, k_interior, displacement);
    values.gradient_mu = muller_composite_gradient(
        mu_exterior, mu_interior,
        k_exterior, k_interior, displacement);
    return values;
}

cdouble contract_k1(
    const MullerKernelValues& values,
    const Vec3& rotated_test_tangent,
    const Vec3& source_tangent)
{
    cdouble result =
        values.k1_scalar *
        rotated_test_tangent.dot(source_tangent);
    const double source[3] = {
        source_tangent.x, source_tangent.y,
        source_tangent.z
    };
    const double test[3] = {
        rotated_test_tangent.x,
        rotated_test_tangent.y,
        rotated_test_tangent.z
    };
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            result += test[row] *
                values.hessian[3 * row + column] *
                source[column];
        }
    }
    return result;
}

cdouble contract_k2(
    const std::array<cdouble, 3>& gradient,
    const Vec3& test_tangent,
    const Vec3& observation_normal,
    const Vec3& source_tangent)
{
    const cdouble test_dot_gradient =
        test_tangent.x * gradient[0] +
        test_tangent.y * gradient[1] +
        test_tangent.z * gradient[2];
    const cdouble normal_dot_gradient =
        observation_normal.x * gradient[0] +
        observation_normal.y * gradient[1] +
        observation_normal.z * gradient[2];
    return
        test_tangent.dot(source_tangent) *
            normal_dot_gradient -
        observation_normal.dot(source_tangent) *
            test_dot_gradient;
}

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

NearPairTemplateKey near_pair_template_key(
    const MullerP2Mesh& mesh,
    int test_index,
    int trial_index,
    const ElementAdjacency& adjacency,
    double quantization)
{
    const MullerP2Element& test = mesh.elements[test_index];
    const MullerP2Element& trial = mesh.elements[trial_index];
    const Vec3 origin = mesh.nodes[test.nodes[0]];
    const Vec3 edge1 = mesh.nodes[test.nodes[1]] - origin;
    const Vec3 edge2 = mesh.nodes[test.nodes[2]] - origin;
    const double edge1_norm = edge1.norm();
    const Vec3 cross = edge1.cross(edge2);
    const double cross_norm = cross.norm();
    if (edge1_norm < 1.0e-14 || cross_norm < 1.0e-14)
        throw std::runtime_error(
            "degenerate element in near-correction template key");
    const Vec3 axis1 = edge1 * (1.0 / edge1_norm);
    const Vec3 normal = cross * (1.0 / cross_norm);
    const Vec3 axis2 = normal.cross(axis1);

    NearPairTemplateKey key;
    key.values.reserve(
        mesh.basis_kind == MullerBasisKind::NodalP2
            ? 12 + 12 * 12
            : 52);
    key.values.push_back(adjacency.shared_count);
    for (int local = 0; local < 3; local++) {
        key.values.push_back(adjacency.test_local[local]);
        key.values.push_back(adjacency.trial_local[local]);
    }
    auto append_node = [&](const Vec3& node) {
        const Vec3 relative = node - origin;
        key.values.push_back(static_cast<std::int64_t>(
            std::llround(relative.dot(axis1) / quantization)));
        key.values.push_back(static_cast<std::int64_t>(
            std::llround(relative.dot(axis2) / quantization)));
        key.values.push_back(static_cast<std::int64_t>(
            std::llround(relative.dot(normal) / quantization)));
    };
    auto append_direction = [&](const Vec3& direction) {
        key.values.push_back(static_cast<std::int64_t>(
            std::llround(direction.dot(axis1) / quantization)));
        key.values.push_back(static_cast<std::int64_t>(
            std::llround(direction.dot(axis2) / quantization)));
        key.values.push_back(static_cast<std::int64_t>(
            std::llround(direction.dot(normal) / quantization)));
    };
    auto append_node_and_frame = [&](int node) {
        append_node(mesh.nodes[node]);
        if (mesh.basis_kind == MullerBasisKind::NodalP2) {
            append_direction(mesh.normals[node]);
            append_direction(mesh.tangent1[node]);
            append_direction(mesh.tangent2[node]);
        }
    };
    for (int node : test.nodes)
        append_node_and_frame(node);
    for (int node : trial.nodes)
        append_node_and_frame(node);
    return key;
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
        muller_duffy_remap_shared_edge(
            point.test_xi, point.test_eta,
            adjacency.test_local[0],
            adjacency.test_local[1]);
        muller_duffy_remap_shared_edge(
            point.trial_xi, point.trial_eta,
            adjacency.trial_local[0],
            adjacency.trial_local[1]);
    }
}

void add_correction_sample(
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
    double sign,
    std::vector<std::map<int, CorrectionValues>>& rows)
{
    const Vec3 displacement = test.position - trial.position;
    if (displacement.norm() < 1.0e-13)
        return;
    const double physical_weight =
        sign * weight * test.jacobian * trial.jacobian;
    const MullerKernelValues kernels =
        evaluate_kernel_values(
            k_exterior, k_interior,
            epsilon_exterior, epsilon_interior,
            mu_exterior, mu_interior,
            displacement);
    const MullerBasisSample test_basis =
        evaluate_muller_basis(mesh, test_element, test);
    const MullerBasisSample trial_basis =
        evaluate_muller_basis(mesh, trial_element, trial);
    for (int alpha = 0; alpha < test_basis.count; alpha++) {
        const Vec3& test_tangent = test_basis.values[alpha];
        const Vec3 rotated_test =
            test_tangent.cross(test.normal);
        for (int beta = 0; beta < trial_basis.count; beta++) {
            const Vec3& source_tangent = trial_basis.values[beta];
            const cdouble k1 = contract_k1(
                kernels, rotated_test, source_tangent);
            const cdouble k2_epsilon = contract_k2(
                kernels.gradient_epsilon,
                test_tangent, test.normal, source_tangent);
            const cdouble k2_mu = contract_k2(
                kernels.gradient_mu,
                test_tangent, test.normal, source_tangent);
            const int row = test_basis.dofs[alpha];
            const int column = trial_basis.dofs[beta];
            CorrectionValues& value = rows[row][column];
            value.k1 += physical_weight * k1;
            value.k2_epsilon +=
                physical_weight * k2_epsilon;
            value.k2_mu += physical_weight * k2_mu;
        }
    }
}

void add_local_correction_sample(
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
    double sign,
    LocalCorrectionBlock& block)
{
    const Vec3 displacement = test.position - trial.position;
    if (displacement.norm() < 1.0e-13)
        return;
    const double physical_weight =
        sign * weight * test.jacobian * trial.jacobian;
    const MullerKernelValues kernels =
        evaluate_kernel_values(
            k_exterior, k_interior,
            epsilon_exterior, epsilon_interior,
            mu_exterior, mu_interior,
            displacement);
    const MullerBasisSample test_basis =
        evaluate_muller_basis(mesh, test_element, test);
    const MullerBasisSample trial_basis =
        evaluate_muller_basis(mesh, trial_element, trial);
    block.test_count = test_basis.count;
    block.trial_count = trial_basis.count;
    for (int alpha = 0; alpha < test_basis.count; alpha++) {
        const Vec3& test_tangent = test_basis.values[alpha];
        const Vec3 rotated_test =
            test_tangent.cross(test.normal);
        for (int beta = 0; beta < trial_basis.count; beta++) {
            const Vec3& source_tangent = trial_basis.values[beta];
            CorrectionValues& value =
                block.values[12 * alpha + beta];
            value.k1 += physical_weight * contract_k1(
                kernels, rotated_test, source_tangent);
            value.k2_epsilon +=
                physical_weight * contract_k2(
                    kernels.gradient_epsilon,
                    test_tangent, test.normal, source_tangent);
            value.k2_mu += physical_weight * contract_k2(
                kernels.gradient_mu,
                test_tangent, test.normal, source_tangent);
        }
    }
}

LocalCorrectionBlock build_local_correction_block(
    const MullerP2Mesh& mesh,
    int test_index,
    int trial_index,
    const ElementAdjacency& adjacency,
    cdouble k_exterior,
    cdouble k_interior,
    cdouble epsilon_exterior,
    cdouble epsilon_interior,
    cdouble mu_exterior,
    cdouble mu_interior,
    const TriQuad& regular,
    const std::vector<MullerDuffyPoint>& exact,
    const std::vector<MullerFrameSample>& regular_frames)
{
    LocalCorrectionBlock block;
    block.test_edge_orientations =
        mesh.elements[test_index].edge_orientations;
    block.trial_edge_orientations =
        mesh.elements[trial_index].edge_orientations;
    for (MullerDuffyPoint point : exact) {
        remap_singular_point(point, adjacency);
        const MullerFrameSample test =
            evaluate_muller_frame(
                mesh, test_index,
                point.test_xi, point.test_eta);
        const MullerFrameSample trial =
            evaluate_muller_frame(
                mesh, trial_index,
                point.trial_xi, point.trial_eta);
        add_local_correction_sample(
            mesh, test_index, trial_index,
            test, trial,
            k_exterior, k_interior,
            epsilon_exterior, epsilon_interior,
            mu_exterior, mu_interior,
            point.weight, 1.0, block);
    }
    for (int qx = 0; qx < regular.npts; qx++) {
        const MullerFrameSample& test =
            regular_frames[
                (size_t)test_index * regular.npts + qx];
        for (int qy = 0; qy < regular.npts; qy++) {
            const MullerFrameSample& trial =
                regular_frames[
                    (size_t)trial_index * regular.npts + qy];
            add_local_correction_sample(
                mesh, test_index, trial_index,
                test, trial,
                k_exterior, k_interior,
                epsilon_exterior, epsilon_interior,
                mu_exterior, mu_interior,
                0.25 * regular.wts[qx] * regular.wts[qy],
                -1.0, block);
        }
    }
    return block;
}

MullerNearCorrection build_near_correction(
    const MullerP2Mesh& mesh,
    cdouble k_exterior,
    cdouble k_interior,
    cdouble epsilon_exterior,
    cdouble epsilon_interior,
    cdouble mu_exterior,
    cdouble mu_interior,
    int regular_quadrature_order,
    int duffy_order,
    bool template_reuse,
    int* color_count,
    int* pair_count,
    int* unique_template_count)
{
    std::vector<std::map<int, CorrectionValues>> rows(
        mesh.current_dofs());
    const TriQuad regular =
        tri_quadrature(regular_quadrature_order);
    const std::vector<MullerDuffyPoint> coincident =
        muller_duffy_rule(
            duffy_order, MullerDuffyAdjacency::Coincident);
    const std::vector<MullerDuffyPoint> edge =
        muller_duffy_rule(
            duffy_order, MullerDuffyAdjacency::EdgeAdjacent);
    const std::vector<MullerDuffyPoint> vertex =
        muller_duffy_rule(
            duffy_order, MullerDuffyAdjacency::VertexAdjacent);

    const int element_count = (int)mesh.elements.size();
    int maximum_topology_vertex = -1;
    for (const MullerP2Element& element : mesh.elements) {
        for (int local = 0; local < 3; local++) {
            maximum_topology_vertex = std::max(
                maximum_topology_vertex,
                element.topology_vertices[local]);
        }
    }
    std::vector<std::vector<int>> vertex_elements(
        maximum_topology_vertex + 1);
    for (int element = 0; element < element_count; element++) {
        for (int local = 0; local < 3; local++) {
            vertex_elements[
                mesh.elements[element].topology_vertices[local]
            ].push_back(element);
        }
    }
    std::vector<std::vector<int>> adjacent(element_count);
    for (int test_index = 0; test_index < element_count; test_index++) {
        for (int local = 0; local < 3; local++) {
            const int vertex =
                mesh.elements[test_index].topology_vertices[local];
            adjacent[test_index].insert(
                adjacent[test_index].end(),
                vertex_elements[vertex].begin(),
                vertex_elements[vertex].end());
        }
        std::sort(
            adjacent[test_index].begin(),
            adjacent[test_index].end());
        adjacent[test_index].erase(
            std::unique(
                adjacent[test_index].begin(),
                adjacent[test_index].end()),
            adjacent[test_index].end());
    }

    std::vector<MullerFrameSample> regular_frames(
        (size_t)element_count * regular.npts);
    for (int element = 0; element < element_count; element++) {
        for (int q = 0; q < regular.npts; q++) {
            regular_frames[
                (size_t)element * regular.npts + q] =
                evaluate_muller_frame(
                    mesh, element,
                    regular.pts[q][0],
                    regular.pts[q][1]);
        }
    }

    std::vector<std::vector<int>> colors;
    std::vector<std::vector<unsigned char>> color_dofs;
    for (int element = 0; element < element_count; element++) {
        const MullerFrameSample center =
            evaluate_muller_frame(mesh, element, 1.0 / 3.0, 1.0 / 3.0);
        const MullerBasisSample basis =
            evaluate_muller_basis(mesh, element, center);
        int selected = -1;
        for (int color = 0; color < (int)colors.size(); color++) {
            bool conflict = false;
            for (int local = 0; local < basis.count; local++) {
                if (color_dofs[color][basis.dofs[local]]) {
                    conflict = true;
                    break;
                }
            }
            if (!conflict) {
                selected = color;
                break;
            }
        }
        if (selected < 0) {
            selected = (int)colors.size();
            colors.emplace_back();
            color_dofs.emplace_back(
                mesh.current_dofs(), (unsigned char)0);
        }
        colors[selected].push_back(element);
        for (int local = 0; local < basis.count; local++)
            color_dofs[selected][basis.dofs[local]] = 1;
    }
    if (color_count)
        *color_count = (int)colors.size();

    int total_pairs = 0;
    for (const std::vector<int>& trials : adjacent)
        total_pairs += static_cast<int>(trials.size());
    if (pair_count)
        *pair_count = total_pairs;

    const bool use_templates = template_reuse;
    if (use_templates) {
        Vec3 minimum = mesh.nodes.front();
        Vec3 maximum = mesh.nodes.front();
        for (const Vec3& node : mesh.nodes) {
            minimum.x = std::min(minimum.x, node.x);
            minimum.y = std::min(minimum.y, node.y);
            minimum.z = std::min(minimum.z, node.z);
            maximum.x = std::max(maximum.x, node.x);
            maximum.y = std::max(maximum.y, node.y);
            maximum.z = std::max(maximum.z, node.z);
        }
        const double geometry_scale =
            std::max((maximum - minimum).norm(), 1.0);
        const double quantization = geometry_scale * 1.0e-11;

        std::map<NearPairTemplateKey, int> template_ids;
        std::vector<std::pair<int, int>> representatives;
        std::vector<ElementAdjacency> representative_adjacency;
        std::vector<std::vector<int>> pair_templates(element_count);
        for (int test_index = 0;
             test_index < element_count; test_index++) {
            pair_templates[test_index].reserve(
                adjacent[test_index].size());
            for (int trial_index : adjacent[test_index]) {
                const ElementAdjacency adjacency =
                    classify_elements(
                        mesh.elements[test_index],
                        mesh.elements[trial_index]);
                const NearPairTemplateKey key =
                    near_pair_template_key(
                        mesh, test_index, trial_index,
                        adjacency, quantization);
                const auto inserted = template_ids.emplace(
                    key, static_cast<int>(representatives.size()));
                if (inserted.second) {
                    representatives.emplace_back(
                        test_index, trial_index);
                    representative_adjacency.push_back(adjacency);
                }
                pair_templates[test_index].push_back(
                    inserted.first->second);
            }
        }
        if (unique_template_count)
            *unique_template_count =
                static_cast<int>(representatives.size());

        std::vector<LocalCorrectionBlock> templates(
            representatives.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (int index = 0;
             index < static_cast<int>(representatives.size());
             index++) {
            const int test_index = representatives[index].first;
            const int trial_index = representatives[index].second;
            const ElementAdjacency& adjacency =
                representative_adjacency[index];
            const std::vector<MullerDuffyPoint>* exact = nullptr;
            if (test_index == trial_index)
                exact = &coincident;
            else if (adjacency.shared_count == 2)
                exact = &edge;
            else if (adjacency.shared_count == 1)
                exact = &vertex;
            if (!exact)
                continue;
            templates[index] = build_local_correction_block(
                mesh, test_index, trial_index, adjacency,
                k_exterior, k_interior,
                epsilon_exterior, epsilon_interior,
                mu_exterior, mu_interior,
                regular, *exact, regular_frames);
        }

        std::vector<MullerBasisSample> center_bases(element_count);
        for (int element = 0; element < element_count; element++) {
            const MullerFrameSample center =
                evaluate_muller_frame(
                    mesh, element, 1.0 / 3.0, 1.0 / 3.0);
            center_bases[element] =
                evaluate_muller_basis(mesh, element, center);
        }
        for (const std::vector<int>& color : colors) {
#pragma omp parallel for schedule(static)
            for (int color_index = 0;
                 color_index < static_cast<int>(color.size());
                 color_index++) {
                const int test_index = color[color_index];
                const MullerBasisSample& test_basis =
                    center_bases[test_index];
                for (size_t pair_index = 0;
                     pair_index < adjacent[test_index].size();
                     pair_index++) {
                    const int trial_index =
                        adjacent[test_index][pair_index];
                    const MullerBasisSample& trial_basis =
                        center_bases[trial_index];
                    const LocalCorrectionBlock& block =
                        templates[
                            pair_templates[test_index][pair_index]];
                    const MullerP2Element& test_element =
                        mesh.elements[test_index];
                    const MullerP2Element& trial_element =
                        mesh.elements[trial_index];
                    if (block.test_count != test_basis.count ||
                        block.trial_count != trial_basis.count) {
                        throw std::runtime_error(
                            "near-correction template basis mismatch");
                    }
                    for (int alpha = 0;
                         alpha < test_basis.count; alpha++) {
                        const int row = test_basis.dofs[alpha];
                        for (int beta = 0;
                             beta < trial_basis.count; beta++) {
                            const CorrectionValues& local =
                                block.values[12 * alpha + beta];
                            double orientation_factor = 1.0;
                            if (mesh.basis_kind ==
                                MullerBasisKind::HDivBdm1) {
                                if (alpha % 2 == 0) {
                                    const int edge = alpha / 2;
                                    orientation_factor *=
                                        test_element
                                            .edge_orientations[edge] *
                                        block.test_edge_orientations[edge];
                                }
                                if (beta % 2 == 0) {
                                    const int edge = beta / 2;
                                    orientation_factor *=
                                        trial_element
                                            .edge_orientations[edge] *
                                        block.trial_edge_orientations[edge];
                                }
                            }
                            CorrectionValues& global =
                                rows[row][trial_basis.dofs[beta]];
                            global.k1 +=
                                orientation_factor * local.k1;
                            global.k2_epsilon +=
                                orientation_factor *
                                local.k2_epsilon;
                            global.k2_mu +=
                                orientation_factor * local.k2_mu;
                        }
                    }
                }
            }
        }
    } else {
        if (unique_template_count)
            *unique_template_count = total_pairs;
        for (const std::vector<int>& color : colors) {
#pragma omp parallel for schedule(dynamic, 1)
            for (int color_index = 0;
                 color_index < (int)color.size(); color_index++) {
                const int test_index = color[color_index];
                const MullerP2Element& test_element =
                    mesh.elements[test_index];
                for (int trial_index : adjacent[test_index]) {
                    const MullerP2Element& trial_element =
                        mesh.elements[trial_index];
                    const ElementAdjacency adjacency =
                        classify_elements(test_element, trial_element);
                    const std::vector<MullerDuffyPoint>* exact = nullptr;
                    if (test_index == trial_index)
                        exact = &coincident;
                    else if (adjacency.shared_count == 2)
                        exact = &edge;
                    else if (adjacency.shared_count == 1)
                        exact = &vertex;
                    if (!exact)
                        continue;

                    for (MullerDuffyPoint point : *exact) {
                        remap_singular_point(point, adjacency);
                        const MullerFrameSample test =
                            evaluate_muller_frame(
                                mesh, test_index,
                                point.test_xi, point.test_eta);
                        const MullerFrameSample trial =
                            evaluate_muller_frame(
                                mesh, trial_index,
                                point.trial_xi, point.trial_eta);
                        add_correction_sample(
                            mesh, test_index, trial_index,
                            test, trial,
                            k_exterior, k_interior,
                            epsilon_exterior, epsilon_interior,
                            mu_exterior, mu_interior,
                            point.weight, 1.0, rows);
                    }

                    for (int qx = 0; qx < regular.npts; qx++) {
                        const MullerFrameSample& test =
                            regular_frames[
                                (size_t)test_index *
                                    regular.npts + qx];
                        for (int qy = 0; qy < regular.npts; qy++) {
                            const MullerFrameSample& trial =
                                regular_frames[
                                    (size_t)trial_index *
                                        regular.npts + qy];
                            add_correction_sample(
                                mesh, test_index, trial_index,
                                test, trial,
                                k_exterior, k_interior,
                                epsilon_exterior, epsilon_interior,
                                mu_exterior, mu_interior,
                                0.25 * regular.wts[qx] *
                                    regular.wts[qy],
                                -1.0, rows);
                        }
                    }
                }
            }
        }
    }

    MullerNearCorrection result;
    result.row_offsets.resize(mesh.current_dofs() + 1, 0);
    for (int row = 0; row < mesh.current_dofs(); row++) {
        result.row_offsets[row] = (int)result.entries.size();
        for (const auto& pair : rows[row]) {
            const CorrectionValues& value = pair.second;
            if (std::abs(value.k1) +
                std::abs(value.k2_epsilon) +
                std::abs(value.k2_mu) < 1.0e-18)
                continue;
            MullerNearCorrectionEntry entry;
            entry.column = pair.first;
            entry.k1 = value.k1;
            entry.k2_epsilon = value.k2_epsilon;
            entry.k2_mu = value.k2_mu;
            result.entries.push_back(entry);
        }
    }
    result.row_offsets[mesh.current_dofs()] =
        (int)result.entries.size();
    return result;
}

int hessian_component(int row, int column)
{
    if (row > column)
        std::swap(row, column);
    const int lookup[3][3] = {
        {0, 1, 2},
        {-1, 3, 4},
        {-1, -1, 5}
    };
    return lookup[row][column];
}

cdouble dot_real_complex(
    const Vec3& vector,
    const cdouble values[3])
{
    return vector.x * values[0] +
           vector.y * values[1] +
           vector.z * values[2];
}

void add_block_sample(
    const MullerFmmOperator& op,
    const std::vector<int>& dof_to_local,
    int test_element,
    int trial_element,
    const MullerFrameSample& test,
    const MullerFrameSample& trial,
    double weight,
    int current_dimension,
    std::vector<cdouble>& k1,
    std::vector<cdouble>& k2_epsilon,
    std::vector<cdouble>& k2_mu)
{
    const Vec3 displacement = test.position - trial.position;
    const double physical_weight =
        weight * test.jacobian * trial.jacobian;
    const MullerKernelValues kernels =
        evaluate_kernel_values(
            op.k_exterior, op.k_interior,
            op.epsilon_exterior, op.epsilon_interior,
            op.mu_exterior, op.mu_interior,
            displacement);
    const MullerBasisSample test_basis =
        evaluate_muller_basis(op.mesh, test_element, test);
    const MullerBasisSample trial_basis =
        evaluate_muller_basis(op.mesh, trial_element, trial);
    for (int alpha = 0; alpha < test_basis.count; alpha++) {
        const int row = dof_to_local[test_basis.dofs[alpha]];
        if (row < 0)
            continue;
        const Vec3& test_tangent = test_basis.values[alpha];
        const Vec3 rotated_test =
            test_tangent.cross(test.normal);
        for (int beta = 0; beta < trial_basis.count; beta++) {
            const int column =
                dof_to_local[trial_basis.dofs[beta]];
            if (column < 0)
                continue;
            const Vec3& source_tangent = trial_basis.values[beta];
            const cdouble value_k1 = contract_k1(
                kernels, rotated_test, source_tangent);
            const cdouble value_k2_epsilon = contract_k2(
                kernels.gradient_epsilon,
                test_tangent, test.normal, source_tangent);
            const cdouble value_k2_mu = contract_k2(
                kernels.gradient_mu,
                test_tangent, test.normal, source_tangent);
            const size_t index =
                (size_t)row * current_dimension + column;
            k1[index] += physical_weight * value_k1;
            k2_epsilon[index] +=
                physical_weight * value_k2_epsilon;
            k2_mu[index] +=
                physical_weight * value_k2_mu;
        }
    }
}

} // namespace

void MullerFmmOperator::init(
    const Mesh& linear_mesh,
    cdouble k_exterior_value,
    cdouble refractive_index,
    bool project_edge_nodes_to_sphere,
    int quadrature_order_value,
    int duffy_order,
    int fmm_digits,
    int max_leaf,
    bool use_pfft_value,
    int pfft_order,
    double pfft_correction_radius,
    double pfft_grid_safety_value,
    const char* correction_cache_path,
    int fmm_near_radius_value,
    bool near_template_reuse_value)
{
    MullerP2BuildOptions options;
    options.project_edge_nodes_to_sphere =
        project_edge_nodes_to_sphere;
    init(
        linear_mesh, k_exterior_value, refractive_index, options,
        quadrature_order_value, duffy_order, fmm_digits, max_leaf,
        use_pfft_value, pfft_order, pfft_correction_radius,
        pfft_grid_safety_value, correction_cache_path,
        fmm_near_radius_value, near_template_reuse_value);
}

void MullerFmmOperator::init(
    const Mesh& linear_mesh,
    cdouble k_exterior_value,
    cdouble refractive_index,
    const MullerP2BuildOptions& build_options,
    int quadrature_order_value,
    int duffy_order,
    int fmm_digits,
    int max_leaf,
    bool use_pfft_value,
    int pfft_order,
    double pfft_correction_radius,
    double pfft_grid_safety_value,
    const char* correction_cache_path,
    int fmm_near_radius_value,
    bool near_template_reuse_value)
{
    const auto geometry_start =
        std::chrono::steady_clock::now();
    cleanup();
    use_pfft = use_pfft_value;
#ifdef BEM_MULLER_GPU_ASSEMBLY_DEFAULT
    gpu_operator_assembly_requested = true;
#else
    gpu_operator_assembly_requested = false;
#endif
    if (const char* gpu_assembly_env =
            std::getenv("BEM_MULLER_GPU_ASSEMBLY")) {
        gpu_operator_assembly_requested =
            std::strcmp(gpu_assembly_env, "0") != 0;
    }
    const char* banded_split_environment =
        std::getenv("BEM_FMM_BANDED_SPLIT_DEPTH");
    banded_fmm_split_depth = banded_split_environment
        ? std::max(0, std::atoi(banded_split_environment))
        : 0;
    banded_fmm = !use_pfft && banded_fmm_split_depth > 0;
    const char* banded_leaf_environment =
        std::getenv("BEM_FMM_BANDED_COARSE_MAX_LEAF");
    banded_fmm_coarse_max_leaf = banded_leaf_environment
        ? std::max(1, std::atoi(banded_leaf_environment))
        : 4096;
    const char* banded_middle_leaf_environment =
        std::getenv("BEM_FMM_BANDED_MIDDLE_MAX_LEAF");
    banded_fmm_middle_max_leaf = banded_middle_leaf_environment
        ? std::max(1, std::atoi(banded_middle_leaf_environment))
        : std::max(1, banded_fmm_coarse_max_leaf / 8);
    if (banded_fmm && gpu_operator_assembly_requested) {
        std::fprintf(
            stderr,
            "  [Muller] Banded FMM uses host Galerkin assembly "
            "with GPU far-field projection\n");
    }
    fmm_near_radius = std::max(1, fmm_near_radius_value);
    near_correction_template_reuse = near_template_reuse_value;
    near_correction_pairs = 0;
    near_correction_unique_templates = 0;
    pfft_interpolation_order = pfft_order;
    pfft_correction_radius_cells = pfft_correction_radius;
    pfft_grid_safety = pfft_grid_safety_value;
#ifdef BEM_FMM_ONLY
    if (use_pfft)
        throw std::runtime_error(
            "pFFT is unavailable in a BEM_FMM_ONLY build");
#endif
    if (pfft_interpolation_order < 2 ||
        pfft_interpolation_order > 5)
        throw std::invalid_argument(
            "pFFT interpolation order must be in [2,5]");
    if (pfft_correction_radius_cells < 0.0 ||
        pfft_grid_safety <= 0.5 || pfft_grid_safety > 1.0)
        throw std::invalid_argument(
            "pFFT correction radius must be non-negative and "
            "grid safety must be in (0.5,1]");
    mesh = build_muller_p2_mesh(
        linear_mesh, build_options);
    current_dofs = mesh.current_dofs();
    system_dofs = mesh.system_dofs();
    quadrature_order = quadrature_order_value;
    k_exterior = k_exterior_value;
    k_interior = k_exterior * refractive_index;
    epsilon_interior = refractive_index * refractive_index;

    const TriQuad rule = tri_quadrature(quadrature_order);
    regular_points_per_element = rule.npts;
    quadrature.clear();
    quadrature.reserve(mesh.elements.size() * rule.npts);
    std::vector<double> points;
    points.reserve(mesh.elements.size() * rule.npts * 3);
    for (int element = 0;
         element < (int)mesh.elements.size(); element++) {
        for (int q = 0; q < rule.npts; q++) {
            MullerFmmQuadraturePoint point;
            point.element = element;
            point.sample = evaluate_muller_frame(
                mesh, element,
                rule.pts[q][0], rule.pts[q][1]);
            point.weight =
                0.5 * rule.wts[q] * point.sample.jacobian;
            points.push_back(point.sample.position.x);
            points.push_back(point.sample.position.y);
            points.push_back(point.sample.position.z);
            quadrature.push_back(point);
        }
    }
    const TriQuad mass_rule = tri_quadrature(13);
    mass_points_per_element = mass_rule.npts;
    mass_quadrature.clear();
    mass_quadrature.reserve(
        mesh.elements.size() * mass_rule.npts);
    for (int element = 0;
         element < (int)mesh.elements.size(); element++) {
        for (int q = 0; q < mass_rule.npts; q++) {
            MullerFmmQuadraturePoint point;
            point.element = element;
            point.sample = evaluate_muller_frame(
                mesh, element,
                mass_rule.pts[q][0], mass_rule.pts[q][1]);
            point.weight =
                0.5 * mass_rule.wts[q] *
                point.sample.jacobian;
            mass_quadrature.push_back(point);
        }
    }
    assembly_colors = build_assembly_colors(mesh);
    geometry_setup_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            geometry_start).count();

    const auto correction_start =
        std::chrono::steady_clock::now();
    near_correction_cache_path =
        correction_cache_path ? correction_cache_path : "";
    near_correction_cache_hit = false;
    const CorrectionFingerprint fingerprint =
        correction_fingerprint(
            mesh, k_exterior, k_interior,
            epsilon_exterior, epsilon_interior,
            mu_exterior, mu_interior,
            quadrature_order, duffy_order);
    std::string cache_reason;
    if (correction_cache_path &&
        correction_cache_path[0] != '\0') {
        near_correction_cache_hit =
            load_near_correction_cache(
                correction_cache_path, fingerprint,
                current_dofs, correction,
                near_correction_colors, cache_reason);
        if (near_correction_cache_hit) {
            std::printf(
                "  [Muller] Near-correction cache hit: %s "
                "(%zu entries)\n",
                correction_cache_path,
                correction.entries.size());
        } else {
            std::printf(
                "  [Muller] Near-correction cache miss: %s (%s)\n",
                correction_cache_path, cache_reason.c_str());
        }
    }
    if (!near_correction_cache_hit) {
        correction = build_near_correction(
            mesh, k_exterior, k_interior,
            epsilon_exterior, epsilon_interior,
            mu_exterior, mu_interior,
            quadrature_order, duffy_order,
            near_correction_template_reuse,
            &near_correction_colors,
            &near_correction_pairs,
            &near_correction_unique_templates);
        if (near_correction_template_reuse) {
            std::printf(
                "  [Muller] Near-correction templates: %d unique "
                "for %d adjacent pairs (%.1fx reuse)\n",
                near_correction_unique_templates,
                near_correction_pairs,
                near_correction_unique_templates > 0
                    ? static_cast<double>(near_correction_pairs) /
                        near_correction_unique_templates
                    : 1.0);
        }
        if (correction_cache_path &&
            correction_cache_path[0] != '\0') {
            if (save_near_correction_cache(
                    correction_cache_path, fingerprint,
                    correction, near_correction_colors,
                    cache_reason)) {
                std::printf(
                    "  [Muller] Stored near-correction cache: %s "
                    "(%zu entries)\n",
                    correction_cache_path,
                    correction.entries.size());
            } else {
                std::fprintf(
                    stderr,
                    "  [Muller] Warning: could not store "
                    "near-correction cache %s (%s)\n",
                    correction_cache_path,
                    cache_reason.c_str());
            }
        }
    }
    near_correction_setup_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            correction_start).count();
    const int point_count = (int)quadrature.size();
    const auto fmm_engine_start =
        std::chrono::steady_clock::now();
    if (use_pfft) {
#ifndef BEM_FMM_ONLY
        double minimum[3] = {
            points[0], points[1], points[2]
        };
        double maximum[3] = {
            points[0], points[1], points[2]
        };
        for (int point = 1; point < point_count; point++) {
            for (int axis = 0; axis < 3; axis++) {
                minimum[axis] = std::min(
                    minimum[axis], points[3 * point + axis]);
                maximum[axis] = std::max(
                    maximum[axis], points[3 * point + axis]);
            }
        }
        double diameter = 0.0;
        for (int axis = 0; axis < 3; axis++)
            diameter = std::max(
                diameter, maximum[axis] - minimum[axis]);
        const double common_grid_spacing = pfft_grid_safety * std::min(
            HelmholtzPFFT::grid_spacing_for_diameter(
                diameter, k_exterior, pfft_interpolation_order),
            HelmholtzPFFT::grid_spacing_for_diameter(
                diameter, k_interior, pfft_interpolation_order));
        pfft_exterior.init(
            points.data(), point_count,
            points.data(), point_count,
            k_exterior, pfft_interpolation_order, max_leaf,
            common_grid_spacing, pfft_correction_radius_cells);
        pfft_interior.init(
            points.data(), point_count,
            points.data(), point_count,
            k_interior, pfft_interpolation_order, max_leaf,
            common_grid_spacing, pfft_correction_radius_cells);
#endif
    } else {
        const int stable_fmm_digits =
            std::min(fmm_digits, muller_fmm_digits_cap());
        const char* pair_currents =
            std::getenv("BEM_FMM_PAIR_CURRENTS");
        const bool request_pair_workspace =
            !banded_fmm && (pair_currents == nullptr ||
            std::strcmp(pair_currents, "0") != 0);
        if (stable_fmm_digits != fmm_digits) {
            std::fprintf(
                stderr,
                "  [Muller] Requested FMM digits=%d; using digits=%d "
                "for stable surface derivatives\n",
                fmm_digits, stable_fmm_digits);
        }
        fmm_exterior.configure_interaction_band(
            1, std::numeric_limits<int>::max(), true);
        fmm_interior.configure_interaction_band(
            1, std::numeric_limits<int>::max(), true);
        if (banded_fmm) {
            const double fine_depth_ratio =
                2.0 * point_count / std::max(1, max_leaf);
            const int predicted_fine_depth = fine_depth_ratio > 1.0
                ? std::min(6, std::max(1, static_cast<int>(
                    std::ceil(
                        std::log(fine_depth_ratio) /
                        std::log(8.0)))))
                : 0;
            if (predicted_fine_depth >
                    banded_fmm_split_depth + 2) {
                throw std::runtime_error(
                    "banded FMM currently supports at most two levels "
                    "above the coarse split");
            }
            banded_fmm_middle = predicted_fine_depth ==
                banded_fmm_split_depth + 2;
            fmm_exterior.configure_interaction_band(
                banded_fmm_split_depth +
                    (banded_fmm_middle ? 2 : 1),
                std::numeric_limits<int>::max(), true);
            fmm_interior.configure_interaction_band(
                banded_fmm_split_depth +
                    (banded_fmm_middle ? 2 : 1),
                std::numeric_limits<int>::max(), true);
        }
        fmm_exterior.init(
            points.data(), point_count,
            points.data(), point_count,
            k_exterior, stable_fmm_digits, max_leaf,
            fmm_near_radius, true, request_pair_workspace);
        fmm_interior.init(
            points.data(), point_count,
            points.data(), point_count,
            k_interior, stable_fmm_digits, max_leaf,
            fmm_near_radius, true, request_pair_workspace);
        fmm_exterior.near_field_fp32 = fmm_near_fp32;
        fmm_interior.near_field_fp32 = fmm_near_fp32;
        if (banded_fmm) {
            if (fmm_exterior.tree.max_level <=
                    banded_fmm_split_depth ||
                fmm_interior.tree.max_level <=
                    banded_fmm_split_depth) {
                throw std::runtime_error(
                    "banded FMM split must be shallower than the fine tree");
            }
            fmm_exterior_coarse.configure_interaction_band(
                1, banded_fmm_split_depth, false);
            fmm_interior_coarse.configure_interaction_band(
                1, banded_fmm_split_depth, false);
            const char* coarse_order_depth = std::getenv(
                "BEM_FMM_BANDED_COARSE_ORDER_REFERENCE_DEPTH");
            if (coarse_order_depth != nullptr) {
                const int depth = std::atoi(coarse_order_depth);
                fmm_exterior_coarse.configure_order_reference_depth(depth);
                fmm_interior_coarse.configure_order_reference_depth(depth);
            }
            fmm_exterior_coarse.init(
                points.data(), point_count,
                points.data(), point_count,
                k_exterior, stable_fmm_digits,
                banded_fmm_coarse_max_leaf,
                fmm_near_radius, true, false);
            fmm_interior_coarse.init(
                points.data(), point_count,
                points.data(), point_count,
                k_interior, stable_fmm_digits,
                banded_fmm_coarse_max_leaf,
                fmm_near_radius, true, false);
            if (banded_fmm_middle) {
                fmm_exterior_middle.configure_interaction_band(
                    banded_fmm_split_depth + 1,
                    banded_fmm_split_depth + 1, false);
                fmm_interior_middle.configure_interaction_band(
                    banded_fmm_split_depth + 1,
                    banded_fmm_split_depth + 1, false);
                fmm_exterior_middle.init(
                    points.data(), point_count,
                    points.data(), point_count,
                    k_exterior, stable_fmm_digits,
                    banded_fmm_middle_max_leaf,
                    fmm_near_radius, true, false);
                fmm_interior_middle.init(
                    points.data(), point_count,
                    points.data(), point_count,
                    k_interior, stable_fmm_digits,
                    banded_fmm_middle_max_leaf,
                    fmm_near_radius, true, false);
                if (fmm_exterior_middle.tree.max_level !=
                        banded_fmm_split_depth + 1 ||
                    fmm_interior_middle.tree.max_level !=
                        banded_fmm_split_depth + 1) {
                    throw std::runtime_error(
                        "BEM_FMM_BANDED_MIDDLE_MAX_LEAF must produce "
                        "a tree one level deeper than the coarse split");
                }
            }
            if (fmm_exterior_coarse.tree.max_level !=
                    banded_fmm_split_depth ||
                fmm_interior_coarse.tree.max_level !=
                    banded_fmm_split_depth) {
                throw std::runtime_error(
                    "BEM_FMM_BANDED_COARSE_MAX_LEAF must produce a "
                    "coarse tree whose depth equals "
                    "BEM_FMM_BANDED_SPLIT_DEPTH");
            }
            if (banded_fmm_middle) {
                std::printf(
                    "  [Muller] Banded FMM: fine M2L level %d, "
                    "middle level %d, coarse levels 1..%d; "
                    "max-leaf %d/%d/%d\n",
                    fmm_exterior.tree.max_level,
                    banded_fmm_split_depth + 1,
                    banded_fmm_split_depth,
                    max_leaf, banded_fmm_middle_max_leaf,
                    banded_fmm_coarse_max_leaf);
            } else {
                std::printf(
                    "  [Muller] Banded FMM: fine M2L level %d, "
                    "coarse levels 1..%d; max-leaf %d/%d\n",
                    fmm_exterior.tree.max_level,
                    banded_fmm_split_depth,
                    max_leaf, banded_fmm_coarse_max_leaf);
            }
        }
    }
    fmm_engine_setup_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            fmm_engine_start).count();

    for (int axis = 0; axis < 3; axis++) {
        charges[axis].resize(point_count);
        gradient_exterior[axis].resize((size_t)point_count * 3);
        gradient_interior[axis].resize((size_t)point_count * 3);
        hessian_exterior[axis].resize((size_t)point_count * 6);
        hessian_interior[axis].resize((size_t)point_count * 6);
    }
    curl_exterior.resize((size_t)point_count * 3);
    curl_interior.resize((size_t)point_count * 3);
    hessian_action_exterior.resize((size_t)point_count * 3);
    hessian_action_interior.resize((size_t)point_count * 3);
    if (banded_fmm) {
        curl_exterior_coarse.resize((size_t)point_count * 3);
        curl_interior_coarse.resize((size_t)point_count * 3);
        hessian_action_exterior_coarse.resize((size_t)point_count * 3);
        hessian_action_interior_coarse.resize((size_t)point_count * 3);
        if (banded_fmm_middle) {
            curl_exterior_middle.resize((size_t)point_count * 3);
            curl_interior_middle.resize((size_t)point_count * 3);
            hessian_action_exterior_middle.resize(
                (size_t)point_count * 3);
            hessian_action_interior_middle.resize(
                (size_t)point_count * 3);
        }
    }
    mass_work.resize(current_dofs);
    k1_work.resize(current_dofs);
    k2_epsilon_work.resize(current_dofs);
    k2_mu_work.resize(current_dofs);
    if (!use_pfft && gpu_operator_assembly_requested) {
        prepare_gpu_assembly();
        if (banded_fmm) {
            // The current GPU assembly path evaluates one uniform FMM tree.
            // Keep its projection buffers for batched far fields, but retain
            // the host Galerkin path that combines all FMM bands for matvecs.
            gpu_operator_assembly = false;
            gpu_operator_assembly_requested = false;
        }
    }
}

void MullerFmmOperator::prepare_gpu_assembly()
{
    if (gpu_assembly.initialized) {
        gpu_operator_assembly = true;
        return;
    }
    std::vector<int> regular_counts(quadrature.size(), 0);
    std::vector<int> regular_dofs(quadrature.size() * 12, -1);
    std::vector<double> regular_values(
        quadrature.size() * 12 * 3, 0.0);
    std::vector<double> regular_normals(
        quadrature.size() * 3, 0.0);
    std::vector<double> regular_weights(quadrature.size(), 0.0);
    for (size_t point_index = 0;
         point_index < quadrature.size(); point_index++) {
        const MullerFmmQuadraturePoint& point =
            quadrature[point_index];
        const MullerBasisSample basis =
            evaluate_muller_basis(mesh, point.element, point.sample);
        regular_counts[point_index] = basis.count;
        regular_normals[3 * point_index] = point.sample.normal.x;
        regular_normals[3 * point_index + 1] = point.sample.normal.y;
        regular_normals[3 * point_index + 2] = point.sample.normal.z;
        regular_weights[point_index] = point.weight;
        for (int local = 0; local < basis.count; local++) {
            const size_t flat = point_index * 12 + local;
            regular_dofs[flat] = basis.dofs[local];
            regular_values[3 * flat] = basis.values[local].x;
            regular_values[3 * flat + 1] = basis.values[local].y;
            regular_values[3 * flat + 2] = basis.values[local].z;
        }
    }

    std::vector<int> mass_counts(mass_quadrature.size(), 0);
    std::vector<int> mass_dofs(mass_quadrature.size() * 12, -1);
    std::vector<double> mass_values(
        mass_quadrature.size() * 12 * 3, 0.0);
    std::vector<double> mass_positions(
        mass_quadrature.size() * 3, 0.0);
    std::vector<double> mass_weights(mass_quadrature.size(), 0.0);
    for (size_t point_index = 0;
         point_index < mass_quadrature.size(); point_index++) {
        const MullerFmmQuadraturePoint& point =
            mass_quadrature[point_index];
        const MullerBasisSample basis =
            evaluate_muller_basis(mesh, point.element, point.sample);
        mass_counts[point_index] = basis.count;
        mass_positions[3 * point_index] = point.sample.position.x;
        mass_positions[3 * point_index + 1] = point.sample.position.y;
        mass_positions[3 * point_index + 2] = point.sample.position.z;
        mass_weights[point_index] = point.weight;
        for (int local = 0; local < basis.count; local++) {
            const size_t flat = point_index * 12 + local;
            mass_dofs[flat] = basis.dofs[local];
            mass_values[3 * flat] = basis.values[local].x;
            mass_values[3 * flat + 1] = basis.values[local].y;
            mass_values[3 * flat + 2] = basis.values[local].z;
        }
    }

    std::vector<MullerGpuCorrectionValue> gpu_correction(
        correction.entries.size());
    for (size_t index = 0;
         index < correction.entries.size(); index++) {
        gpu_correction[index].column =
            correction.entries[index].column;
        gpu_correction[index].k1 =
            correction.entries[index].k1;
        gpu_correction[index].k2_epsilon =
            correction.entries[index].k2_epsilon;
        gpu_correction[index].k2_mu =
            correction.entries[index].k2_mu;
    }
    gpu_assembly.init(
        current_dofs,
        regular_counts,
        regular_dofs,
        regular_values,
        regular_normals,
        regular_weights,
        mass_counts,
        mass_dofs,
        mass_values,
        mass_positions,
        mass_weights,
        correction.row_offsets,
        gpu_correction);
    gpu_operator_assembly = true;
    std::printf(
        "  [Muller] GPU-resident operator assembly enabled "
        "(%zu regular, %zu mass points)\n",
        quadrature.size(), mass_quadrature.size());
}

void MullerFmmOperator::farfield(
    const cdouble* solution,
    const std::vector<Vec3>& directions,
    std::vector<cdouble>& field)
{
    if (!gpu_assembly.initialized)
        throw std::runtime_error(
            "GPU farfield requires GPU operator assembly");
    gpu_assembly.farfield(
        solution, k_exterior, directions, field);
}

void MullerFmmOperator::farfield_pair(
    const cdouble* solution_x,
    const cdouble* solution_y,
    const std::vector<Vec3>& directions,
    std::vector<cdouble>& field_x,
    std::vector<cdouble>& field_y)
{
    if (!gpu_assembly.initialized)
        throw std::runtime_error(
            "GPU paired farfield requires GPU operator assembly");
    gpu_assembly.farfield_pair(
        solution_x, solution_y, k_exterior,
        directions, field_x, field_y);
}

void MullerFmmOperator::apply_current_operators_gpu(
    int input_offset, int slot)
{
    gpu_assembly.project_charges_and_mass(input_offset, slot);
    const auto evaluate_exterior = [&]() {
        fmm_exterior.evaluate_vector_actions_batch3_device(
            gpu_assembly.charge_re(0, slot),
            gpu_assembly.charge_im(0, slot),
            gpu_assembly.charge_re(1, slot),
            gpu_assembly.charge_im(1, slot),
            gpu_assembly.charge_re(2, slot),
            gpu_assembly.charge_im(2, slot));
    };
    const auto evaluate_interior = [&]() {
        fmm_interior.evaluate_vector_actions_batch3_device(
            gpu_assembly.charge_re(0, slot),
            gpu_assembly.charge_im(0, slot),
            gpu_assembly.charge_re(1, slot),
            gpu_assembly.charge_im(1, slot),
            gpu_assembly.charge_re(2, slot),
            gpu_assembly.charge_im(2, slot));
    };
    const char* concurrent_media =
        std::getenv("BEM_FMM_CONCURRENT_MEDIA");
#ifdef BEM_FMM_CONCURRENT_MEDIA_DEFAULT
    bool evaluate_media_concurrently = true;
#else
    bool evaluate_media_concurrently = false;
#endif
    if (concurrent_media != nullptr)
        evaluate_media_concurrently =
            std::strcmp(concurrent_media, "0") != 0;
    if (evaluate_media_concurrently) {
#pragma omp parallel sections num_threads(2)
        {
#pragma omp section
            evaluate_exterior();
#pragma omp section
            evaluate_interior();
        }
    } else {
        evaluate_exterior();
        evaluate_interior();
    }
    gpu_assembly.assemble_media_and_correction(
        fmm_exterior,
        fmm_interior,
        epsilon_exterior,
        epsilon_interior,
        mu_exterior,
        mu_interior,
        input_offset,
        slot);
}

void MullerFmmOperator::apply_current_operator_pair_gpu(bool strict)
{
    gpu_assembly.project_charges_and_mass(0, 0);
    gpu_assembly.project_charges_and_mass(current_dofs, 1);
    const auto evaluate = [&](HelmholtzFMM& fmm) {
        const auto normal = [&]() {
            fmm.evaluate_vector_actions_pair_batch3_device(
            gpu_assembly.charge_re(0, 0),
            gpu_assembly.charge_im(0, 0),
            gpu_assembly.charge_re(1, 0),
            gpu_assembly.charge_im(1, 0),
            gpu_assembly.charge_re(2, 0),
            gpu_assembly.charge_im(2, 0),
            gpu_assembly.charge_re(0, 1),
            gpu_assembly.charge_im(0, 1),
            gpu_assembly.charge_re(1, 1),
            gpu_assembly.charge_im(1, 1),
            gpu_assembly.charge_re(2, 1),
            gpu_assembly.charge_im(2, 1));
        };
        if (!strict) {
            normal();
            return;
        }
        fmm.evaluate_vector_actions_pair_batch3_device_strict(
            gpu_assembly.charge_re(0, 0),
            gpu_assembly.charge_im(0, 0),
            gpu_assembly.charge_re(1, 0),
            gpu_assembly.charge_im(1, 0),
            gpu_assembly.charge_re(2, 0),
            gpu_assembly.charge_im(2, 0),
            gpu_assembly.charge_re(0, 1),
            gpu_assembly.charge_im(0, 1),
            gpu_assembly.charge_re(1, 1),
            gpu_assembly.charge_im(1, 1),
            gpu_assembly.charge_re(2, 1),
            gpu_assembly.charge_im(2, 1));
    };
    const auto evaluate_exterior = [&]() {
        evaluate(fmm_exterior);
    };
    const auto evaluate_interior = [&]() {
        evaluate(fmm_interior);
    };
    const char* concurrent_media =
        std::getenv("BEM_FMM_CONCURRENT_MEDIA");
#ifdef BEM_FMM_CONCURRENT_MEDIA_DEFAULT
    bool evaluate_media_concurrently = true;
#else
    bool evaluate_media_concurrently = false;
#endif
    if (concurrent_media != nullptr)
        evaluate_media_concurrently =
            std::strcmp(concurrent_media, "0") != 0;
    if (evaluate_media_concurrently) {
#pragma omp parallel sections num_threads(2)
        {
#pragma omp section
            evaluate_exterior();
#pragma omp section
            evaluate_interior();
        }
    } else {
        evaluate_exterior();
        evaluate_interior();
    }
    gpu_assembly.assemble_media_and_correction(
        fmm_exterior,
        fmm_interior,
        epsilon_exterior,
        epsilon_interior,
        mu_exterior,
        mu_interior,
        0,
        0,
        0);
    gpu_assembly.assemble_media_and_correction(
        fmm_exterior,
        fmm_interior,
        epsilon_exterior,
        epsilon_interior,
        mu_exterior,
        mu_interior,
        current_dofs,
        1,
        1);
}

void MullerFmmOperator::apply_current_operator_quad_gpu()
{
    for (int slot = 0; slot < 4; slot++)
        gpu_assembly.project_charges_and_mass(
            slot * current_dofs, slot);
    const double* charges_re[12] = {};
    const double* charges_im[12] = {};
    for (int slot = 0; slot < 4; slot++) {
        for (int component = 0; component < 3; component++) {
            const int field = 3 * slot + component;
            charges_re[field] =
                gpu_assembly.charge_re(component, slot);
            charges_im[field] =
                gpu_assembly.charge_im(component, slot);
        }
    }
    const auto evaluate_exterior = [&]() {
        fmm_exterior.evaluate_vector_actions_quad_batch3_device(
            charges_re, charges_im);
    };
    const auto evaluate_interior = [&]() {
        fmm_interior.evaluate_vector_actions_quad_batch3_device(
            charges_re, charges_im);
    };
    bool evaluate_media_concurrently = false;
#ifdef BEM_FMM_CONCURRENT_MEDIA_DEFAULT
    evaluate_media_concurrently = true;
#endif
    const char* concurrent_media =
        std::getenv("BEM_FMM_CONCURRENT_MEDIA");
    if (concurrent_media != nullptr)
        evaluate_media_concurrently =
            std::strcmp(concurrent_media, "0") != 0;
    if (evaluate_media_concurrently) {
#pragma omp parallel sections num_threads(2)
        {
#pragma omp section
            evaluate_exterior();
#pragma omp section
            evaluate_interior();
        }
    } else {
        evaluate_exterior();
        evaluate_interior();
    }
    for (int slot = 0; slot < 4; slot++) {
        gpu_assembly.assemble_media_and_correction(
            fmm_exterior,
            fmm_interior,
            epsilon_exterior,
            epsilon_interior,
            mu_exterior,
            mu_interior,
            slot * current_dofs,
            slot,
            slot);
    }
}

void MullerFmmOperator::apply_current_operators(
    const cdouble* coefficients,
    std::vector<cdouble>& mass,
    std::vector<cdouble>& k1,
    std::vector<cdouble>& k2_epsilon,
    std::vector<cdouble>& k2_mu)
{
    const int point_count = (int)quadrature.size();
    apply_mass(coefficients, mass);
    k1.assign(current_dofs, cdouble(0.0));
    k2_epsilon.assign(current_dofs, cdouble(0.0));
    k2_mu.assign(current_dofs, cdouble(0.0));

#pragma omp parallel for schedule(static)
    for (int point_index = 0;
         point_index < point_count; point_index++) {
        const MullerFmmQuadraturePoint& point =
            quadrature[point_index];
        cdouble current[3] = {
            cdouble(0.0), cdouble(0.0), cdouble(0.0)
        };
        const MullerBasisSample basis =
            evaluate_muller_basis(
                mesh, point.element, point.sample);
        for (int local = 0; local < basis.count; local++) {
            const cdouble value =
                coefficients[basis.dofs[local]];
            const Vec3& direction = basis.values[local];
            current[0] += value * direction.x;
            current[1] += value * direction.y;
            current[2] += value * direction.z;
        }
        for (int axis = 0; axis < 3; axis++)
            charges[axis][point_index] =
                point.weight * current[axis];

    }

    const bool contracted_vector_actions = true;
    if (use_pfft) {
#ifndef BEM_FMM_ONLY
        pfft_exterior.evaluate_vector_actions(
            charges[0].data(),
            charges[1].data(),
            charges[2].data(),
            curl_exterior.data(),
            hessian_action_exterior.data());
        pfft_interior.evaluate_vector_actions_from_prepared(
            pfft_exterior,
            curl_interior.data(),
            hessian_action_interior.data());
#endif
    } else {
        const auto evaluate_exterior = [&]() {
            fmm_exterior.evaluate_vector_actions_batch3(
                charges[0].data(), charges[1].data(), charges[2].data(),
                curl_exterior.data(),
                hessian_action_exterior.data());
            if (banded_fmm) {
                if (banded_fmm_middle) {
                    fmm_exterior_middle.evaluate_vector_actions_batch3(
                        charges[0].data(), charges[1].data(),
                        charges[2].data(),
                        curl_exterior_middle.data(),
                        hessian_action_exterior_middle.data());
                }
                fmm_exterior_coarse.evaluate_vector_actions_batch3(
                    charges[0].data(), charges[1].data(), charges[2].data(),
                    curl_exterior_coarse.data(),
                    hessian_action_exterior_coarse.data());
                for (size_t index = 0;
                     index < curl_exterior.size(); index++) {
                    curl_exterior[index] +=
                        curl_exterior_coarse[index];
                    hessian_action_exterior[index] +=
                        hessian_action_exterior_coarse[index];
                    if (banded_fmm_middle) {
                        curl_exterior[index] +=
                            curl_exterior_middle[index];
                        hessian_action_exterior[index] +=
                            hessian_action_exterior_middle[index];
                    }
                }
            }
        };
        const auto evaluate_interior = [&]() {
            fmm_interior.evaluate_vector_actions_batch3(
                charges[0].data(), charges[1].data(), charges[2].data(),
                curl_interior.data(),
                hessian_action_interior.data());
            if (banded_fmm) {
                if (banded_fmm_middle) {
                    fmm_interior_middle.evaluate_vector_actions_batch3(
                        charges[0].data(), charges[1].data(),
                        charges[2].data(),
                        curl_interior_middle.data(),
                        hessian_action_interior_middle.data());
                }
                fmm_interior_coarse.evaluate_vector_actions_batch3(
                    charges[0].data(), charges[1].data(), charges[2].data(),
                    curl_interior_coarse.data(),
                    hessian_action_interior_coarse.data());
                for (size_t index = 0;
                     index < curl_interior.size(); index++) {
                    curl_interior[index] += curl_interior_coarse[index];
                    hessian_action_interior[index] +=
                        hessian_action_interior_coarse[index];
                    if (banded_fmm_middle) {
                        curl_interior[index] +=
                            curl_interior_middle[index];
                        hessian_action_interior[index] +=
                            hessian_action_interior_middle[index];
                    }
                }
            }
        };
        const char* concurrent_media =
            std::getenv("BEM_FMM_CONCURRENT_MEDIA");
#ifdef BEM_FMM_CONCURRENT_MEDIA_DEFAULT
        bool evaluate_media_concurrently = true;
#else
        bool evaluate_media_concurrently = false;
#endif
        if (concurrent_media != nullptr)
            evaluate_media_concurrently =
                std::strcmp(concurrent_media, "0") != 0;
        if (evaluate_media_concurrently) {
#pragma omp parallel sections num_threads(2)
            {
#pragma omp section
                evaluate_exterior();
#pragma omp section
                evaluate_interior();
            }
        } else {
            evaluate_exterior();
            evaluate_interior();
        }
    }

    for (const std::vector<int>& color : assembly_colors) {
#pragma omp parallel for schedule(static)
        for (int color_index = 0;
             color_index < static_cast<int>(color.size());
             color_index++) {
            const int element = color[color_index];
            const int point_begin =
                element * regular_points_per_element;
            const int point_end =
                point_begin + regular_points_per_element;
            for (int point_index = point_begin;
                 point_index < point_end; point_index++) {
                const MullerFmmQuadraturePoint& point =
                    quadrature[point_index];
                cdouble k1_vector[3] = {
                    cdouble(0.0), cdouble(0.0), cdouble(0.0)
                };
                cdouble gradient_epsilon[3][3];
                cdouble gradient_mu[3][3];
                cdouble curl_epsilon[3] = {
                    cdouble(0.0), cdouble(0.0), cdouble(0.0)
                };
                cdouble curl_mu[3] = {
                    cdouble(0.0), cdouble(0.0), cdouble(0.0)
                };
                if (contracted_vector_actions) {
            for (int component = 0; component < 3; component++) {
                k1_vector[component] =
                    hessian_action_exterior[
                        3 * point_index + component] -
                    hessian_action_interior[
                        3 * point_index + component];
                curl_epsilon[component] =
                    epsilon_exterior *
                        curl_exterior[
                            3 * point_index + component] -
                    epsilon_interior *
                        curl_interior[
                            3 * point_index + component];
                curl_mu[component] =
                    mu_exterior *
                        curl_exterior[
                            3 * point_index + component] -
                    mu_interior *
                        curl_interior[
                            3 * point_index + component];
            }
                } else {
            for (int source_axis = 0;
                 source_axis < 3; source_axis++) {
                cdouble trace_exterior(0.0);
                cdouble trace_interior(0.0);
                for (int axis = 0; axis < 3; axis++) {
                    trace_exterior +=
                        hessian_exterior[source_axis][
                            6 * point_index +
                            hessian_component(axis, axis)];
                    trace_interior +=
                        hessian_interior[source_axis][
                            6 * point_index +
                            hessian_component(axis, axis)];
                }
                for (int target_axis = 0;
                     target_axis < 3; target_axis++) {
                    cdouble exterior =
                        hessian_exterior[source_axis][
                            6 * point_index +
                            hessian_component(
                                target_axis, source_axis)];
                    cdouble interior =
                        hessian_interior[source_axis][
                            6 * point_index +
                            hessian_component(
                                target_axis, source_axis)];
                    if (target_axis == source_axis) {
                        exterior -= trace_exterior;
                        interior -= trace_interior;
                    }
                    k1_vector[target_axis] +=
                        exterior - interior;
                    gradient_epsilon[source_axis][target_axis] =
                        epsilon_exterior *
                            gradient_exterior[source_axis][
                                3 * point_index + target_axis] -
                        epsilon_interior *
                            gradient_interior[source_axis][
                                3 * point_index + target_axis];
                    gradient_mu[source_axis][target_axis] =
                        mu_exterior *
                            gradient_exterior[source_axis][
                                3 * point_index + target_axis] -
                        mu_interior *
                            gradient_interior[source_axis][
                                3 * point_index + target_axis];
                }
            }
                }

                const MullerBasisSample test_basis =
                    evaluate_muller_basis(
                        mesh, point.element, point.sample);
                for (int component = 0;
                     component < test_basis.count; component++) {
            const Vec3& test_tangent =
                test_basis.values[component];
            const Vec3 rotated =
                test_tangent.cross(point.sample.normal);
            const cdouble k1_value =
                dot_real_complex(rotated, k1_vector);
            cdouble k2_epsilon_value(0.0);
            cdouble k2_mu_value(0.0);
            const double test[3] = {
                test_tangent.x,
                test_tangent.y,
                test_tangent.z
            };
            const double normal[3] = {
                point.sample.normal.x,
                point.sample.normal.y,
                point.sample.normal.z
            };
            if (contracted_vector_actions) {
                const double coefficients[3] = {
                    test[0] * normal[1] -
                        normal[0] * test[1],
                    test[0] * normal[2] -
                        normal[0] * test[2],
                    test[1] * normal[2] -
                        normal[1] * test[2]
                };
                for (int curl_component = 0;
                     curl_component < 3; curl_component++) {
                    k2_epsilon_value +=
                        coefficients[curl_component] *
                        curl_epsilon[curl_component];
                    k2_mu_value +=
                        coefficients[curl_component] *
                        curl_mu[curl_component];
                }
            } else {
                for (int source_axis = 0;
                     source_axis < 3; source_axis++) {
                    cdouble normal_gradient_epsilon(0.0);
                    cdouble tangent_gradient_epsilon(0.0);
                    cdouble normal_gradient_mu(0.0);
                    cdouble tangent_gradient_mu(0.0);
                    for (int gradient_axis = 0;
                         gradient_axis < 3; gradient_axis++) {
                        normal_gradient_epsilon +=
                            normal[gradient_axis] *
                            gradient_epsilon[
                                source_axis][gradient_axis];
                        tangent_gradient_epsilon +=
                            test[gradient_axis] *
                            gradient_epsilon[
                                source_axis][gradient_axis];
                        normal_gradient_mu +=
                            normal[gradient_axis] *
                            gradient_mu[
                                source_axis][gradient_axis];
                        tangent_gradient_mu +=
                            test[gradient_axis] *
                            gradient_mu[
                                source_axis][gradient_axis];
                    }
                    k2_epsilon_value +=
                        test[source_axis] *
                            normal_gradient_epsilon -
                        normal[source_axis] *
                            tangent_gradient_epsilon;
                    k2_mu_value +=
                        test[source_axis] * normal_gradient_mu -
                        normal[source_axis] * tangent_gradient_mu;
                }
            }
            const int row = test_basis.dofs[component];
            k1[row] += point.weight * k1_value;
            k2_epsilon[row] +=
                point.weight * k2_epsilon_value;
                    k2_mu[row] +=
                        point.weight * k2_mu_value;
                }
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (int row = 0; row < current_dofs; row++) {
        for (int index = correction.row_offsets[row];
             index < correction.row_offsets[row + 1]; index++) {
            const MullerNearCorrectionEntry& entry =
                correction.entries[index];
            const cdouble value = coefficients[entry.column];
            k1[row] += entry.k1 * value;
            k2_epsilon[row] += entry.k2_epsilon * value;
            k2_mu[row] += entry.k2_mu * value;
        }
    }
}

bool MullerFmmOperator::device_matvec_available() const
{
    return system_dofs > 0 && gpu_operator_assembly && !use_pfft;
}

void MullerFmmOperator::matvec_device(
    const void* device_input, void* device_output)
{
    if (!device_matvec_available())
        throw std::runtime_error(
            "device Muller matvec requires the FMM GPU assembly backend");
    gpu_assembly.upload_system_input_device(device_input);
    bool pair_currents =
        fmm_exterior.vector_actions_pair_available() &&
        fmm_interior.vector_actions_pair_available();
    const char* pair_current_override =
        std::getenv("BEM_FMM_PAIR_CURRENTS");
    if (pair_current_override != nullptr) {
        pair_currents =
            std::strcmp(pair_current_override, "0") != 0 &&
            fmm_exterior.vector_actions_pair_available() &&
            fmm_interior.vector_actions_pair_available();
    }
    if (pair_currents) {
        apply_current_operator_pair_gpu();
    } else {
        apply_current_operators_gpu(0, 0);
        apply_current_operators_gpu(current_dofs, 1);
    }
    gpu_assembly.combine_to_device(
        k_exterior,
        epsilon_exterior,
        epsilon_interior,
        mu_exterior,
        mu_interior,
        device_output);
}

void MullerFmmOperator::matvec_batch2_device(
    const void* device_input_x,
    const void* device_input_y,
    void* device_output_x,
    void* device_output_y)
{
    bool four_field =
        fmm_exterior.vector_actions_quad_available() &&
        fmm_interior.vector_actions_quad_available();
    const char* override = std::getenv("BEM_FMM_FOUR_FIELD");
    if (override != nullptr)
        four_field = std::strcmp(override, "0") != 0 && four_field;
    if (!four_field) {
        matvec_device(device_input_x, device_output_x);
        matvec_device(device_input_y, device_output_y);
        return;
    }
    gpu_assembly.upload_system_input_pair_device(
        device_input_x, device_input_y);
    apply_current_operator_quad_gpu();
    gpu_assembly.combine_to_device(
        k_exterior,
        epsilon_exterior,
        epsilon_interior,
        mu_exterior,
        mu_interior,
        device_output_x,
        0);
    gpu_assembly.combine_to_device(
        k_exterior,
        epsilon_exterior,
        epsilon_interior,
        mu_exterior,
        mu_interior,
        device_output_y,
        1);
}

void MullerFmmOperator::matvec_batch2_device_strict(
    const void* device_input_x,
    const void* device_input_y,
    void* device_output_x,
    void* device_output_y)
{
    if (!device_matvec_available())
        throw std::runtime_error(
            "strict device Muller matvec requires the FMM backend");
    const bool previous_near_fp32 = fmm_near_fp32;
    set_fmm_near_fp32(false);
    const bool strict_pair =
        fmm_exterior.strict_vector_pair_available() &&
        fmm_interior.strict_vector_pair_available();
    const auto apply_one = [&](const void* input, void* output) {
        gpu_assembly.upload_system_input_device(input);
        if (strict_pair) {
            apply_current_operator_pair_gpu(true);
        } else {
            apply_current_operators_gpu(0, 0);
            apply_current_operators_gpu(current_dofs, 1);
        }
        gpu_assembly.combine_to_device(
            k_exterior,
            epsilon_exterior,
            epsilon_interior,
            mu_exterior,
            mu_interior,
            output);
    };
    try {
        apply_one(device_input_x, device_output_x);
        apply_one(device_input_y, device_output_y);
    } catch (...) {
        set_fmm_near_fp32(previous_near_fp32);
        throw;
    }
    set_fmm_near_fp32(previous_near_fp32);
}

void MullerFmmOperator::matvec_strict(
    const cdouble* input, cdouble* output)
{
    if (!device_matvec_available()) {
        if (use_pfft || !fmm_exterior.initialized ||
            !fmm_interior.initialized) {
            throw std::runtime_error(
                "strict Muller matvec requires the FMM backend");
        }
        const bool previous_near_fp32 = fmm_near_fp32;
        set_fmm_near_fp32(false);
        try {
            matvec(input, output);
        } catch (...) {
            set_fmm_near_fp32(previous_near_fp32);
            throw;
        }
        set_fmm_near_fp32(previous_near_fp32);
        return;
    }
    const bool previous_near_fp32 = fmm_near_fp32;
    set_fmm_near_fp32(false);
    try {
        gpu_assembly.upload_system_input(input);
        if (fmm_exterior.strict_vector_pair_available() &&
            fmm_interior.strict_vector_pair_available()) {
            apply_current_operator_pair_gpu(true);
        } else {
            apply_current_operators_gpu(0, 0);
            apply_current_operators_gpu(current_dofs, 1);
        }
        gpu_assembly.combine_and_download(
            k_exterior,
            epsilon_exterior,
            epsilon_interior,
            mu_exterior,
            mu_interior,
            output);
    } catch (...) {
        set_fmm_near_fp32(previous_near_fp32);
        throw;
    }
    set_fmm_near_fp32(previous_near_fp32);
}

void MullerFmmOperator::matvec(
    const cdouble* input, cdouble* output)
{
    if (system_dofs <= 0)
        throw std::runtime_error(
            "Muller FMM operator is not initialized");
    if (gpu_operator_assembly && !use_pfft) {
        gpu_assembly.upload_system_input(input);
        bool pair_currents =
            fmm_exterior.vector_actions_pair_available() &&
            fmm_interior.vector_actions_pair_available();
        const char* pair_current_override =
            std::getenv("BEM_FMM_PAIR_CURRENTS");
        if (pair_current_override != nullptr)
            pair_currents =
                std::strcmp(pair_current_override, "0") != 0 &&
                fmm_exterior.vector_actions_pair_available() &&
                fmm_interior.vector_actions_pair_available();
        if (pair_currents) {
            apply_current_operator_pair_gpu();
        } else {
            apply_current_operators_gpu(0, 0);
            apply_current_operators_gpu(current_dofs, 1);
        }
        gpu_assembly.combine_and_download(
            k_exterior,
            epsilon_exterior,
            epsilon_interior,
            mu_exterior,
            mu_interior,
            output);
        return;
    }
    std::vector<cdouble> mass_j, k1_j, k2e_j, k2m_j;
    std::vector<cdouble> mass_m, k1_m, k2e_m, k2m_m;
    apply_current_operators(
        input, mass_j, k1_j, k2e_j, k2m_j);
    apply_current_operators(
        input + current_dofs,
        mass_m, k1_m, k2e_m, k2m_m);

    const cdouble imaginary(0.0, 1.0);
#pragma omp parallel for schedule(static)
    for (int row = 0; row < current_dofs; row++) {
        output[row] =
            imaginary / k_exterior * k1_j[row] +
            0.5 * (epsilon_interior + epsilon_exterior) *
                mass_m[row] +
            k2e_m[row];
        output[current_dofs + row] =
            0.5 * (mu_interior + mu_exterior) *
                mass_j[row] +
            k2m_j[row] -
            imaginary / k_exterior * k1_m[row];
    }
}

double MullerFmmOperator::switch_pfft_to_fmm(
    int fmm_digits, int max_leaf, bool keep_pfft)
{
    if (!use_pfft)
        throw std::runtime_error(
            "switch_pfft_to_fmm requires an active pFFT backend");
#ifdef BEM_FMM_ONLY
    (void)fmm_digits;
    (void)max_leaf;
    throw std::runtime_error(
        "pFFT is unavailable in a BEM_FMM_ONLY build");
#else
    const auto start = std::chrono::steady_clock::now();
    if (!keep_pfft) {
        pfft_exterior.cleanup();
        pfft_interior.cleanup();
    }

    std::vector<double> points;
    points.reserve(quadrature.size() * 3);
    for (const MullerFmmQuadraturePoint& point : quadrature) {
        points.push_back(point.sample.position.x);
        points.push_back(point.sample.position.y);
        points.push_back(point.sample.position.z);
    }
    const int point_count = static_cast<int>(quadrature.size());
    const int stable_fmm_digits =
        std::min(fmm_digits, muller_fmm_digits_cap());
    banded_fmm = banded_fmm_split_depth > 0;
    banded_fmm_middle = false;
    if (banded_fmm && gpu_operator_assembly_requested) {
        std::fprintf(
            stderr,
            "  [Muller] Banded FMM uses host Galerkin assembly "
            "with GPU far-field projection\n");
    }
    const char* pair_currents =
        std::getenv("BEM_FMM_PAIR_CURRENTS");
    const bool request_pair_workspace =
        !banded_fmm && (pair_currents == nullptr ||
        std::strcmp(pair_currents, "0") != 0);
    if (stable_fmm_digits != fmm_digits) {
        std::fprintf(
            stderr,
            "  [Muller] Requested FMM digits=%d; using digits=%d "
            "for stable surface derivatives\n",
            fmm_digits, stable_fmm_digits);
    }
    fmm_exterior.configure_interaction_band(
        1, std::numeric_limits<int>::max(), true);
    fmm_interior.configure_interaction_band(
        1, std::numeric_limits<int>::max(), true);
    if (banded_fmm) {
        const double fine_depth_ratio =
            2.0 * point_count / std::max(1, max_leaf);
        const int predicted_fine_depth = fine_depth_ratio > 1.0
            ? std::min(6, std::max(1, static_cast<int>(
                std::ceil(
                    std::log(fine_depth_ratio) /
                    std::log(8.0)))))
            : 0;
        if (predicted_fine_depth > banded_fmm_split_depth + 2) {
            throw std::runtime_error(
                "banded FMM currently supports at most two levels "
                "above the coarse split");
        }
        banded_fmm_middle = predicted_fine_depth ==
            banded_fmm_split_depth + 2;
        fmm_exterior.configure_interaction_band(
            banded_fmm_split_depth +
                (banded_fmm_middle ? 2 : 1),
            std::numeric_limits<int>::max(), true);
        fmm_interior.configure_interaction_band(
            banded_fmm_split_depth +
                (banded_fmm_middle ? 2 : 1),
            std::numeric_limits<int>::max(), true);
    }
    const auto initialize_exterior = [&]() {
        fmm_exterior.init(
            points.data(), point_count,
            points.data(), point_count,
            k_exterior, stable_fmm_digits, max_leaf,
            fmm_near_radius, true, request_pair_workspace);
    };
    const auto initialize_interior = [&]() {
        fmm_interior.init(
            points.data(), point_count,
            points.data(), point_count,
            k_interior, stable_fmm_digits, max_leaf,
            fmm_near_radius, true, request_pair_workspace);
    };
    const char* interior_first_environment =
        std::getenv("BEM_FMM_INTERIOR_FIRST");
    const bool initialize_interior_first =
        interior_first_environment != nullptr &&
        std::strcmp(interior_first_environment, "0") != 0;
    if (initialize_interior_first) {
        initialize_interior();
        initialize_exterior();
    } else {
        initialize_exterior();
        initialize_interior();
    }
    fmm_exterior.near_field_fp32 = fmm_near_fp32;
    fmm_interior.near_field_fp32 = fmm_near_fp32;
    if (banded_fmm) {
        if (fmm_exterior.tree.max_level <= banded_fmm_split_depth ||
            fmm_interior.tree.max_level <= banded_fmm_split_depth) {
            throw std::runtime_error(
                "banded FMM split must be shallower than the fine tree");
        }
        fmm_exterior_coarse.configure_interaction_band(
            1, banded_fmm_split_depth, false);
        fmm_interior_coarse.configure_interaction_band(
            1, banded_fmm_split_depth, false);
        const char* coarse_order_depth = std::getenv(
            "BEM_FMM_BANDED_COARSE_ORDER_REFERENCE_DEPTH");
        if (coarse_order_depth != nullptr) {
            const int depth = std::atoi(coarse_order_depth);
            fmm_exterior_coarse.configure_order_reference_depth(depth);
            fmm_interior_coarse.configure_order_reference_depth(depth);
        }
        fmm_exterior_coarse.init(
            points.data(), point_count,
            points.data(), point_count,
            k_exterior, stable_fmm_digits,
            banded_fmm_coarse_max_leaf,
            fmm_near_radius, true, false);
        fmm_interior_coarse.init(
            points.data(), point_count,
            points.data(), point_count,
            k_interior, stable_fmm_digits,
            banded_fmm_coarse_max_leaf,
            fmm_near_radius, true, false);
        if (banded_fmm_middle) {
            fmm_exterior_middle.configure_interaction_band(
                banded_fmm_split_depth + 1,
                banded_fmm_split_depth + 1, false);
            fmm_interior_middle.configure_interaction_band(
                banded_fmm_split_depth + 1,
                banded_fmm_split_depth + 1, false);
            fmm_exterior_middle.init(
                points.data(), point_count,
                points.data(), point_count,
                k_exterior, stable_fmm_digits,
                banded_fmm_middle_max_leaf,
                fmm_near_radius, true, false);
            fmm_interior_middle.init(
                points.data(), point_count,
                points.data(), point_count,
                k_interior, stable_fmm_digits,
                banded_fmm_middle_max_leaf,
                fmm_near_radius, true, false);
            if (fmm_exterior_middle.tree.max_level !=
                    banded_fmm_split_depth + 1 ||
                fmm_interior_middle.tree.max_level !=
                    banded_fmm_split_depth + 1) {
                throw std::runtime_error(
                    "BEM_FMM_BANDED_MIDDLE_MAX_LEAF must produce "
                    "a tree one level deeper than the coarse split");
            }
        }
        if (fmm_exterior_coarse.tree.max_level !=
                banded_fmm_split_depth ||
            fmm_interior_coarse.tree.max_level !=
                banded_fmm_split_depth) {
            throw std::runtime_error(
                "BEM_FMM_BANDED_COARSE_MAX_LEAF must produce a "
                "coarse tree whose depth equals "
                "BEM_FMM_BANDED_SPLIT_DEPTH");
        }
        curl_exterior_coarse.resize((size_t)point_count * 3);
        curl_interior_coarse.resize((size_t)point_count * 3);
        hessian_action_exterior_coarse.resize((size_t)point_count * 3);
        hessian_action_interior_coarse.resize((size_t)point_count * 3);
        if (banded_fmm_middle) {
            curl_exterior_middle.resize((size_t)point_count * 3);
            curl_interior_middle.resize((size_t)point_count * 3);
            hessian_action_exterior_middle.resize((size_t)point_count * 3);
            hessian_action_interior_middle.resize((size_t)point_count * 3);
            std::printf(
                "  [Muller] Banded FMM switch: fine M2L level %d, "
                "middle level %d, coarse levels 1..%d; "
                "max-leaf %d/%d/%d\n",
                fmm_exterior.tree.max_level,
                banded_fmm_split_depth + 1,
                banded_fmm_split_depth,
                max_leaf, banded_fmm_middle_max_leaf,
                banded_fmm_coarse_max_leaf);
        } else {
            std::printf(
                "  [Muller] Banded FMM switch: fine M2L level %d, "
                "coarse levels 1..%d; max-leaf %d/%d\n",
                fmm_exterior.tree.max_level,
                banded_fmm_split_depth,
                max_leaf, banded_fmm_coarse_max_leaf);
        }
    }
    use_pfft = false;
    if (gpu_operator_assembly_requested) {
        prepare_gpu_assembly();
        if (banded_fmm) {
            gpu_operator_assembly = false;
            gpu_operator_assembly_requested = false;
        }
    }
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
#endif
}

void MullerFmmOperator::select_pfft_backend()
{
#ifdef BEM_FMM_ONLY
    throw std::runtime_error(
        "pFFT is unavailable in a BEM_FMM_ONLY build");
#else
    if (!pfft_exterior.initialized || !pfft_interior.initialized)
        throw std::runtime_error("pFFT backend is not initialized");
    use_pfft = true;
#endif
}

void MullerFmmOperator::select_fmm_backend()
{
    if (!fmm_exterior.initialized || !fmm_interior.initialized)
        throw std::runtime_error("FMM backend is not initialized");
    use_pfft = false;
    if (gpu_operator_assembly_requested &&
        !gpu_assembly.initialized)
        prepare_gpu_assembly();
}

void MullerFmmOperator::set_fmm_near_fp32(bool enabled)
{
    fmm_near_fp32 = enabled;
    fmm_exterior.near_field_fp32 = enabled;
    fmm_interior.near_field_fp32 = enabled;
    if (banded_fmm) {
        fmm_exterior_coarse.near_field_fp32 = enabled;
        fmm_interior_coarse.near_field_fp32 = enabled;
        if (banded_fmm_middle) {
            fmm_exterior_middle.near_field_fp32 = enabled;
            fmm_interior_middle.near_field_fp32 = enabled;
        }
    }
}

void MullerFmmOperator::apply_current_operators_direct(
    const cdouble* coefficients,
    std::vector<cdouble>& mass,
    std::vector<cdouble>& k1,
    std::vector<cdouble>& k2_epsilon,
    std::vector<cdouble>& k2_mu)
{
    const int point_count = (int)quadrature.size();
    apply_mass(coefficients, mass);
    k1.assign(current_dofs, cdouble(0.0));
    k2_epsilon.assign(current_dofs, cdouble(0.0));
    k2_mu.assign(current_dofs, cdouble(0.0));
    std::vector<std::array<cdouble, 3>> currents(point_count);

    for (int point_index = 0;
         point_index < point_count; point_index++) {
        const MullerFmmQuadraturePoint& point =
            quadrature[point_index];
        currents[point_index] = {
            cdouble(0.0), cdouble(0.0), cdouble(0.0)
        };
        const MullerBasisSample basis =
            evaluate_muller_basis(
                mesh, point.element, point.sample);
        for (int local = 0; local < basis.count; local++) {
            const cdouble value =
                coefficients[basis.dofs[local]];
            const Vec3& direction = basis.values[local];
            currents[point_index][0] += value * direction.x;
            currents[point_index][1] += value * direction.y;
            currents[point_index][2] += value * direction.z;
        }
    }

    const Vec3 axes[3] = {
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0)
    };
    for (int test_index = 0; test_index < point_count; test_index++) {
        const MullerFmmQuadraturePoint& test =
            quadrature[test_index];
        const MullerBasisSample test_basis =
            evaluate_muller_basis(
                mesh, test.element, test.sample);
        for (int component = 0;
             component < test_basis.count; component++) {
            const Vec3& test_tangent =
                test_basis.values[component];
            const Vec3 rotated =
                test_tangent.cross(test.sample.normal);
            cdouble k1_value(0.0);
            cdouble k2_epsilon_value(0.0);
            cdouble k2_mu_value(0.0);
            for (int source_index = 0;
                 source_index < point_count; source_index++) {
                const MullerFmmQuadraturePoint& source =
                    quadrature[source_index];
                const Vec3 displacement =
                    test.sample.position -
                    source.sample.position;
                if (displacement.norm() < 1.0e-13)
                    continue;
                for (int axis = 0; axis < 3; axis++) {
                    const cdouble source_value =
                        source.weight *
                        currents[source_index][axis];
                    k1_value += source_value *
                        muller_k1_kernel(
                            k_exterior, k_interior,
                            displacement, rotated, axes[axis]);
                    k2_epsilon_value += source_value *
                        muller_k2_kernel(
                            epsilon_exterior,
                            epsilon_interior,
                            k_exterior, k_interior,
                            displacement,
                            test_tangent, test.sample.normal,
                            axes[axis]);
                    k2_mu_value += source_value *
                        muller_k2_kernel(
                            mu_exterior, mu_interior,
                            k_exterior, k_interior,
                            displacement,
                            test_tangent, test.sample.normal,
                            axes[axis]);
                }
            }
            const int row = test_basis.dofs[component];
            k1[row] += test.weight * k1_value;
            k2_epsilon[row] +=
                test.weight * k2_epsilon_value;
            k2_mu[row] +=
                test.weight * k2_mu_value;
        }
    }

#pragma omp parallel for schedule(static)
    for (int row = 0; row < current_dofs; row++) {
        for (int index = correction.row_offsets[row];
             index < correction.row_offsets[row + 1]; index++) {
            const MullerNearCorrectionEntry& entry =
                correction.entries[index];
            const cdouble value = coefficients[entry.column];
            k1[row] += entry.k1 * value;
            k2_epsilon[row] += entry.k2_epsilon * value;
            k2_mu[row] += entry.k2_mu * value;
        }
    }
}

void MullerFmmOperator::matvec_direct_reference(
    const cdouble* input, cdouble* output)
{
    std::vector<cdouble> mass_j, k1_j, k2e_j, k2m_j;
    std::vector<cdouble> mass_m, k1_m, k2e_m, k2m_m;
    apply_current_operators_direct(
        input, mass_j, k1_j, k2e_j, k2m_j);
    apply_current_operators_direct(
        input + current_dofs,
        mass_m, k1_m, k2e_m, k2m_m);
    const cdouble imaginary(0.0, 1.0);
    for (int row = 0; row < current_dofs; row++) {
        output[row] =
            imaginary / k_exterior * k1_j[row] +
            0.5 * (epsilon_interior + epsilon_exterior) *
                mass_m[row] +
            k2e_m[row];
        output[current_dofs + row] =
            0.5 * (mu_interior + mu_exterior) *
                mass_j[row] +
            k2m_j[row] -
            imaginary / k_exterior * k1_m[row];
    }
}

void MullerFmmOperator::apply_mass(
    const cdouble* coefficients,
    std::vector<cdouble>& mass) const
{
    mass.assign(current_dofs, cdouble(0.0));
    for (const std::vector<int>& color : assembly_colors) {
#pragma omp parallel for schedule(static)
        for (int color_index = 0;
             color_index < static_cast<int>(color.size());
             color_index++) {
            const int element = color[color_index];
            const int point_begin =
                element * mass_points_per_element;
            const int point_end =
                point_begin + mass_points_per_element;
            for (int point_index = point_begin;
                 point_index < point_end; point_index++) {
                const MullerFmmQuadraturePoint& point =
                    mass_quadrature[point_index];
                cdouble current[3] = {
                    cdouble(0.0), cdouble(0.0), cdouble(0.0)
                };
                const MullerBasisSample basis =
                    evaluate_muller_basis(
                        mesh, point.element, point.sample);
                for (int local = 0; local < basis.count; local++) {
                    const cdouble value =
                        coefficients[basis.dofs[local]];
                    const Vec3& direction = basis.values[local];
                    current[0] += value * direction.x;
                    current[1] += value * direction.y;
                    current[2] += value * direction.z;
                }
                for (int component = 0;
                     component < basis.count; component++) {
                    const Vec3& direction = basis.values[component];
                    const cdouble tested =
                        direction.x * current[0] +
                        direction.y * current[1] +
                        direction.z * current[2];
                    const int row = basis.dofs[component];
                    mass[row] += point.weight * tested;
                }
            }
        }
    }
}

void MullerFmmOperator::cleanup()
{
    gpu_assembly.cleanup();
#ifndef BEM_FMM_ONLY
    if (pfft_exterior.initialized)
        pfft_exterior.cleanup();
    if (pfft_interior.initialized)
        pfft_interior.cleanup();
#endif
    if (fmm_exterior.initialized)
        fmm_exterior.cleanup();
    if (fmm_interior.initialized)
        fmm_interior.cleanup();
    if (fmm_exterior_coarse.initialized)
        fmm_exterior_coarse.cleanup();
    if (fmm_interior_coarse.initialized)
        fmm_interior_coarse.cleanup();
    if (fmm_exterior_middle.initialized)
        fmm_exterior_middle.cleanup();
    if (fmm_interior_middle.initialized)
        fmm_interior_middle.cleanup();
    current_dofs = 0;
    system_dofs = 0;
    quadrature.clear();
    mass_quadrature.clear();
    correction = MullerNearCorrection();
    curl_exterior_coarse.clear();
    curl_interior_coarse.clear();
    hessian_action_exterior_coarse.clear();
    hessian_action_interior_coarse.clear();
    curl_exterior_middle.clear();
    curl_interior_middle.clear();
    hessian_action_exterior_middle.clear();
    hessian_action_interior_middle.clear();
    assembly_colors.clear();
    regular_points_per_element = 0;
    mass_points_per_element = 0;
    geometry_setup_seconds = 0.0;
    near_correction_setup_seconds = 0.0;
    fmm_engine_setup_seconds = 0.0;
    near_correction_colors = 0;
    near_correction_pairs = 0;
    near_correction_unique_templates = 0;
    near_correction_template_reuse = true;
    near_correction_cache_hit = false;
    near_correction_cache_path.clear();
    use_pfft = false;
    gpu_operator_assembly = false;
    gpu_operator_assembly_requested = false;
    banded_fmm = false;
    banded_fmm_middle = false;
    banded_fmm_split_depth = 0;
    banded_fmm_coarse_max_leaf = 0;
    banded_fmm_middle_max_leaf = 0;
    pfft_interpolation_order = 2;
}

std::vector<cdouble> assemble_muller_nodal_block(
    const MullerFmmOperator& op,
    const std::vector<int>& dof_groups,
    const std::vector<int>* support_elements_override)
{
    if (dof_groups.empty())
        throw std::invalid_argument("empty Muller block");
    if (op.current_dofs % 2 != 0)
        throw std::logic_error("Muller current DOFs are not paired");
    const int group_count = op.current_dofs / 2;
    std::vector<int> dof_to_local(op.current_dofs, -1);
    for (int local = 0;
         local < (int)dof_groups.size(); local++) {
        const int group = dof_groups[local];
        if (group < 0 || group >= group_count ||
            dof_to_local[2 * group] >= 0)
            throw std::invalid_argument(
                "invalid or duplicate Muller block DOF group");
        dof_to_local[2 * group] = 2 * local;
        dof_to_local[2 * group + 1] = 2 * local + 1;
    }
    std::vector<int> discovered_support_elements;
    if (!support_elements_override) {
        for (int element = 0;
             element < (int)op.mesh.elements.size(); element++) {
            bool touches_block = false;
            const MullerFrameSample center =
                evaluate_muller_frame(
                    op.mesh, element, 1.0 / 3.0, 1.0 / 3.0);
            const MullerBasisSample basis =
                evaluate_muller_basis(op.mesh, element, center);
            for (int local = 0; local < basis.count; local++) {
                if (dof_to_local[basis.dofs[local]] >= 0) {
                    touches_block = true;
                    break;
                }
            }
            if (touches_block)
                discovered_support_elements.push_back(element);
        }
    }
    const std::vector<int>& support_elements =
        support_elements_override
            ? *support_elements_override
            : discovered_support_elements;

    const int current_dimension =
        2 * (int)dof_groups.size();
    const int system_dimension =
        4 * (int)dof_groups.size();
    const size_t current_matrix_size =
        (size_t)current_dimension * current_dimension;
    std::vector<cdouble> mass(
        current_matrix_size, cdouble(0.0));
    std::vector<cdouble> k1(
        current_matrix_size, cdouble(0.0));
    std::vector<cdouble> k2_epsilon(
        current_matrix_size, cdouble(0.0));
    std::vector<cdouble> k2_mu(
        current_matrix_size, cdouble(0.0));

    const TriQuad mass_rule = tri_quadrature(13);
    std::vector<MullerFrameSample> mass_frames(
        (size_t)support_elements.size() * mass_rule.npts);
    for (int support_index = 0;
         support_index < (int)support_elements.size();
         support_index++) {
        const int element_index = support_elements[support_index];
        for (int q = 0; q < mass_rule.npts; q++) {
            mass_frames[
                (size_t)support_index * mass_rule.npts + q] =
                evaluate_muller_frame(
                    op.mesh, element_index,
                    mass_rule.pts[q][0],
                    mass_rule.pts[q][1]);
        }
    }
    for (int support_index = 0;
         support_index < (int)support_elements.size();
         support_index++) {
        const int element_index = support_elements[support_index];
        for (int q = 0; q < mass_rule.npts; q++) {
            const MullerFrameSample sample =
                mass_frames[
                    (size_t)support_index * mass_rule.npts + q];
            const double weight =
                0.5 * mass_rule.wts[q] * sample.jacobian;
            const MullerBasisSample basis =
                evaluate_muller_basis(
                    op.mesh, element_index, sample);
            for (int i = 0; i < basis.count; i++) {
                const int row = dof_to_local[basis.dofs[i]];
                if (row < 0)
                    continue;
                for (int j = 0; j < basis.count; j++) {
                    const int column =
                        dof_to_local[basis.dofs[j]];
                    if (column < 0)
                        continue;
                    mass[
                        (size_t)row * current_dimension + column] +=
                        weight * basis.values[i].dot(
                            basis.values[j]);
                }
            }
        }
    }

    const TriQuad regular =
        tri_quadrature(op.quadrature_order);
    std::vector<MullerFrameSample> regular_frames(
        (size_t)support_elements.size() * regular.npts);
    for (int support_index = 0;
         support_index < (int)support_elements.size();
         support_index++) {
        const int element_index = support_elements[support_index];
        for (int q = 0; q < regular.npts; q++) {
            regular_frames[
                (size_t)support_index * regular.npts + q] =
                evaluate_muller_frame(
                    op.mesh, element_index,
                    regular.pts[q][0],
                    regular.pts[q][1]);
        }
    }
    const std::vector<MullerDuffyPoint> coincident =
        muller_duffy_rule(
            4, MullerDuffyAdjacency::Coincident);
    const std::vector<MullerDuffyPoint> edge =
        muller_duffy_rule(
            4, MullerDuffyAdjacency::EdgeAdjacent);
    const std::vector<MullerDuffyPoint> vertex =
        muller_duffy_rule(
            4, MullerDuffyAdjacency::VertexAdjacent);
    for (int test_support = 0;
         test_support < (int)support_elements.size();
         test_support++) {
        const int test_index = support_elements[test_support];
        const MullerP2Element& test_element =
            op.mesh.elements[test_index];
        for (int trial_support = 0;
             trial_support < (int)support_elements.size();
             trial_support++) {
            const int trial_index =
                support_elements[trial_support];
            const MullerP2Element& trial_element =
                op.mesh.elements[trial_index];
            const ElementAdjacency adjacency =
                classify_elements(test_element, trial_element);
            const std::vector<MullerDuffyPoint>* singular = nullptr;
            if (test_index == trial_index)
                singular = &coincident;
            else if (adjacency.shared_count == 2)
                singular = &edge;
            else if (adjacency.shared_count == 1)
                singular = &vertex;
            if (singular) {
                for (MullerDuffyPoint point : *singular) {
                    remap_singular_point(point, adjacency);
                    const MullerFrameSample test =
                        evaluate_muller_frame(
                            op.mesh, test_index,
                            point.test_xi, point.test_eta);
                    const MullerFrameSample trial =
                        evaluate_muller_frame(
                            op.mesh, trial_index,
                            point.trial_xi, point.trial_eta);
                    add_block_sample(
                        op, dof_to_local,
                        test_index, trial_index,
                        test, trial, point.weight,
                        current_dimension,
                        k1, k2_epsilon, k2_mu);
                }
            } else {
                for (int qx = 0; qx < regular.npts; qx++) {
                    const MullerFrameSample test =
                        regular_frames[
                            (size_t)test_support *
                                regular.npts + qx];
                    for (int qy = 0; qy < regular.npts; qy++) {
                        const MullerFrameSample trial =
                            regular_frames[
                                (size_t)trial_support *
                                    regular.npts + qy];
                        add_block_sample(
                            op, dof_to_local,
                            test_index, trial_index,
                            test, trial,
                            0.25 * regular.wts[qx] *
                                regular.wts[qy],
                            current_dimension,
                            k1, k2_epsilon, k2_mu);
                    }
                }
            }
        }
    }

    std::vector<cdouble> result(
        (size_t)system_dimension * system_dimension,
        cdouble(0.0));
    const cdouble imaginary(0.0, 1.0);
    for (int row = 0; row < current_dimension; row++) {
        const int row_node = row / 2;
        const int row_component = row % 2;
        const int row_j = 4 * row_node + row_component;
        const int row_m = row_j + 2;
        for (int column = 0;
             column < current_dimension; column++) {
            const int column_node = column / 2;
            const int column_component = column % 2;
            const int column_j =
                4 * column_node + column_component;
            const int column_m = column_j + 2;
            const size_t source =
                (size_t)row * current_dimension + column;
            result[(size_t)row_j * system_dimension + column_j] =
                imaginary / op.k_exterior * k1[source];
            result[(size_t)row_j * system_dimension + column_m] =
                0.5 * (
                    op.epsilon_interior +
                    op.epsilon_exterior) * mass[source] +
                k2_epsilon[source];
            result[(size_t)row_m * system_dimension + column_j] =
                0.5 * (op.mu_interior + op.mu_exterior) *
                    mass[source] +
                k2_mu[source];
            result[(size_t)row_m * system_dimension + column_m] =
                -imaginary / op.k_exterior * k1[source];
        }
    }
    return result;
}
