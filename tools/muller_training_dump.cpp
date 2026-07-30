#include "muller_fmm.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

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

std::vector<int> morton_node_order(const std::vector<Vec3>& nodes)
{
    double lower[3] = {nodes[0].x, nodes[0].y, nodes[0].z};
    double upper[3] = {lower[0], lower[1], lower[2]};
    for (const Vec3& node : nodes) {
        const double values[3] = {node.x, node.y, node.z};
        for (int axis = 0; axis < 3; axis++) {
            lower[axis] = std::min(lower[axis], values[axis]);
            upper[axis] = std::max(upper[axis], values[axis]);
        }
    }
    constexpr double maximum = (double)((1u << 21) - 1u);
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
                ? (values[axis] - lower[axis]) / span : 0.0;
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
    for (const auto& entry : coded)
        order.push_back(entry.second);
    return order;
}

Vec3 rotate_zyz(
    const Vec3& point, double alpha, double beta, double gamma)
{
    const double ca = std::cos(alpha);
    const double sa = std::sin(alpha);
    const double cb = std::cos(beta);
    const double sb = std::sin(beta);
    const double cg = std::cos(gamma);
    const double sg = std::sin(gamma);
    const Vec3 first(
        ca * point.x - sa * point.y,
        sa * point.x + ca * point.y,
        point.z);
    const Vec3 second(
        cb * first.x + sb * first.z,
        first.y,
        -sb * first.x + cb * first.z);
    return Vec3(
        cg * second.x - sg * second.y,
        sg * second.x + cg * second.y,
        second.z);
}

template <typename T>
void write_value(std::ofstream& stream, const T& value)
{
    stream.write(
        reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_complex(std::ofstream& stream, const cdouble& value)
{
    const double real = value.real();
    const double imag = value.imag();
    write_value(stream, real);
    write_value(stream, imag);
}

void usage(const char* program)
{
    std::fprintf(
        stderr,
        "Usage: %s --out FILE [--shape sphere|ellipsoid|prism] "
        "[--ref N] [--ka F] [--ri-real F] [--ri-imag F] "
        "[--aspect F] [--sides N] [--edge-refine N] "
        "[--edge-mode smooth|split] [--feature-angle F] "
        "[--alpha F --beta F --gamma F] "
        "[--block-nodes N] [--digits N] [--max-leaf N]\n",
        program);
}

} // namespace

int main(int argc, char** argv)
{
    const char* output_path = nullptr;
    const char* shape = "sphere";
    int refinement = 1;
    int block_nodes = 50;
    int digits = 6;
    int max_leaf = 64;
    double ka = 4.0;
    double refractive_real = 1.5;
    double refractive_imag = 0.0;
    double aspect = 1.0;
    double alpha = 0.0;
    double beta = 0.0;
    double gamma = 0.0;
    int prism_sides = 6;
    int edge_refine = 0;
    double feature_angle = 45.0;
    bool edge_mode_explicit = false;
    MullerEdgeMode edge_mode = MullerEdgeMode::Smooth;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc)
            output_path = argv[++i];
        else if (std::strcmp(argv[i], "--shape") == 0 && i + 1 < argc)
            shape = argv[++i];
        else if (std::strcmp(argv[i], "--ref") == 0 && i + 1 < argc)
            refinement = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ka") == 0 && i + 1 < argc)
            ka = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ri-real") == 0 && i + 1 < argc)
            refractive_real = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--ri-imag") == 0 && i + 1 < argc)
            refractive_imag = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--aspect") == 0 && i + 1 < argc)
            aspect = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--sides") == 0 && i + 1 < argc)
            prism_sides = std::atoi(argv[++i]);
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
            else {
                usage(argv[0]);
                return 2;
            }
        }
        else if (std::strcmp(argv[i], "--alpha") == 0 && i + 1 < argc)
            alpha = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--beta") == 0 && i + 1 < argc)
            beta = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--gamma") == 0 && i + 1 < argc)
            gamma = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--block-nodes") == 0 && i + 1 < argc)
            block_nodes = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--digits") == 0 && i + 1 < argc)
            digits = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--max-leaf") == 0 && i + 1 < argc)
            max_leaf = std::atoi(argv[++i]);
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!output_path || refinement < 0 || block_nodes < 1 ||
        ka <= 0.0 || refractive_real <= 0.0 || aspect <= 0.0 ||
        (std::strcmp(shape, "sphere") != 0 &&
         std::strcmp(shape, "ellipsoid") != 0 &&
         std::strcmp(shape, "prism") != 0)) {
        usage(argv[0]);
        return 2;
    }

    try {
        const bool prism_mode =
            std::strcmp(shape, "prism") == 0;
        Mesh mesh = prism_mode
            ? regular_prism(
                  prism_sides, aspect, refinement,
                  1.0, edge_refine)
            : icosphere(1.0, refinement);
        if (std::strcmp(shape, "ellipsoid") == 0) {
            const double transverse = std::pow(aspect, -1.0 / 3.0);
            const double axial = std::pow(aspect, 2.0 / 3.0);
            for (Vec3& vertex : mesh.verts) {
                vertex.x *= transverse;
                vertex.y *= transverse;
                vertex.z *= axial;
            }
        }
        for (Vec3& vertex : mesh.verts)
            vertex = rotate_zyz(vertex, alpha, beta, gamma);

        if (prism_mode && !edge_mode_explicit)
            edge_mode = MullerEdgeMode::SplitFeatureEdges;
        MullerP2BuildOptions build_options;
        build_options.project_edge_nodes_to_sphere =
            std::strcmp(shape, "sphere") == 0;
        build_options.edge_mode = edge_mode;
        build_options.feature_angle_degrees = feature_angle;
        MullerFmmOperator op;
        op.init(
            mesh, cdouble(ka, 0.0),
            cdouble(refractive_real, refractive_imag),
            build_options,
            7, 4, digits, max_leaf);
        const std::vector<int> order =
            morton_node_order(op.mesh.nodes);
        std::vector<std::vector<int>> blocks;
        for (int begin = 0; begin < (int)order.size();
             begin += block_nodes) {
            const int end = std::min(
                (int)order.size(), begin + block_nodes);
            blocks.emplace_back(
                order.begin() + begin, order.begin() + end);
        }

        Vec3 center;
        for (const Vec3& node : op.mesh.nodes)
            center = center + node;
        center = center * (1.0 / (double)op.mesh.nodes.size());
        double scale_squared = 0.0;
        for (const Vec3& node : op.mesh.nodes)
            scale_squared += (node - center).norm2();
        const double scale = std::sqrt(
            scale_squared / (double)op.mesh.nodes.size());

        std::ofstream output(
            output_path, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("cannot open output dump");
        const char magic[8] = {
            'M', 'U', 'L', 'B', 'L', 'K', '1', '\0'
        };
        output.write(magic, sizeof(magic));
        const uint32_t version = 1;
        const uint32_t block_count = (uint32_t)blocks.size();
        const uint32_t total_nodes =
            (uint32_t)op.mesh.scalar_nodes();
        const uint32_t raw_features = 12;
        write_value(output, version);
        write_value(output, block_count);
        write_value(output, total_nodes);
        write_value(output, raw_features);
        write_value(output, ka);
        write_value(output, refractive_real);
        write_value(output, refractive_imag);
        write_value(output, center.x);
        write_value(output, center.y);
        write_value(output, center.z);
        write_value(output, scale);
        for (const std::vector<int>& nodes : blocks) {
            const uint32_t count = (uint32_t)nodes.size();
            write_value(output, count);
            for (int node : nodes) {
                const Vec3 values[4] = {
                    op.mesh.nodes[node],
                    op.mesh.normals[node],
                    op.mesh.tangent1[node],
                    op.mesh.tangent2[node]
                };
                for (const Vec3& value : values) {
                    write_value(output, value.x);
                    write_value(output, value.y);
                    write_value(output, value.z);
                }
            }
            const std::vector<cdouble> matrix =
                assemble_muller_nodal_block(op, nodes);
            for (const cdouble& value : matrix)
                write_complex(output, value);
        }
        output.close();
        op.cleanup();
        std::printf(
            "Wrote %s: shape=%s ref=%d ka=%.6g "
            "n=%.6g%+.6gi nodes=%u blocks=%u "
            "feature_edges=%d split_nodes=%d\n",
            output_path, shape, refinement, ka,
            refractive_real, refractive_imag,
            total_nodes, block_count,
            op.mesh.feature_edges,
            op.mesh.duplicated_corner_nodes +
                op.mesh.duplicated_midpoint_nodes);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "muller_training_dump: %s\n", error.what());
        return 1;
    }
}
