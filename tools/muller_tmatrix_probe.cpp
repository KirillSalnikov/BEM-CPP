// Dense reference bridge for the T-matrix experiment. Matrices are row-major
// complex128, and the input grid is float64 rows [direction, e_theta, e_phi].
#include "muller_dense.h"
#include "quadrature.h"
#include <chrono>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

using Complex = std::complex<double>;

template<class T>
void write_binary(const std::string& path, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(T));
    if (!out) throw std::runtime_error("cannot write " + path);
}

int main(int argc, char** argv) try {
    if (argc != 9) {
        std::fprintf(stderr, "Usage: %s sphere|prism|tetra ref ka ri grid.bin output_prefix quad duffy\n", argv[0]);
        return 2;
    }
    const std::string shape = argv[1], grid_path = argv[5], prefix = argv[6];
    const int ref = std::stoi(argv[2]), quad_order = std::stoi(argv[7]), duffy_order = std::stoi(argv[8]);
    const double ka = std::stod(argv[3]), ri = std::stod(argv[4]);
    std::ifstream input(grid_path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0 || static_cast<size_t>(input.tellg()) % (9*sizeof(double)))
        throw std::runtime_error("invalid grid input");
    const size_t points = static_cast<size_t>(input.tellg()) / (9*sizeof(double));
    std::vector<double> grid(points*9);
    input.seekg(0); input.read(reinterpret_cast<char*>(grid.data()), grid.size()*sizeof(double));
    Mesh mesh;
    if (shape == "sphere") mesh = icosphere(1.0, ref);
    else if (shape == "prism") mesh = regular_prism(6, 1.0, ref, 1.0);
    else if (shape == "tetra") {
        mesh.verts = {Vec3(1.2,0,0), Vec3(-0.8,0.3,0.1), Vec3(0.1,1,0.2), Vec3(0.2,0.1,1.5)};
        mesh.tris = {0,1,2, 0,3,1, 0,2,3, 1,3,2};
        Vec3 center(0,0,0);
        for (const Vec3& vertex : mesh.verts) center = center + vertex;
        center = center*(1.0/mesh.verts.size());
        for (Vec3& vertex : mesh.verts) vertex = vertex-center;
        normalize_mesh(mesh);
        for (int i=0; i<ref; ++i) mesh = subdivide_flat(mesh);
    } else throw std::runtime_error("unknown shape");
    const auto start = std::chrono::steady_clock::now();
    MullerP2BuildOptions options;
    options.project_edge_nodes_to_sphere = shape == "sphere";
    options.edge_mode = shape == "sphere" ? MullerEdgeMode::Smooth : MullerEdgeMode::HDivBdm1;
    const auto system = assemble_muller_nodal_dense(mesh, ka, ri, options, quad_order, duffy_order);
    const auto assembled = std::chrono::steady_clock::now();
    const int n = system.system_dofs, current = system.current_dofs;
    const int channels = 2*points;
    std::vector<Vec3> directions(points), theta(points), phi(points);
    for (size_t i=0; i<points; ++i) {
        const double* g = grid.data()+9*i;
        directions[i] = Vec3(g[0],g[1],g[2]);
        theta[i] = Vec3(g[3],g[4],g[5]);
        phi[i] = Vec3(g[6],g[7],g[8]);
    }
    std::vector<Complex> b(static_cast<size_t>(n)*channels);
#pragma omp parallel for schedule(static)
    for (int column=0; column<channels; ++column) {
        const size_t p = column/2;
        const auto rhs = muller_nodal_planewave_rhs(system.mesh, ka,
            column%2 ? phi[p] : theta[p], directions[p], 13);
        for (int row=0; row<n; ++row) b[static_cast<size_t>(row)*channels+column] = rhs[row];
    }
    const auto rhs_done = std::chrono::steady_clock::now();
    std::vector<Complex> d(static_cast<size_t>(channels)*n, Complex(0));
    const TriQuad quad = tri_quadrature(13);
    const Complex prefactor(0, -ka*INV4PI);
#pragma omp parallel for schedule(static)
    for (int p=0; p<static_cast<int>(points); ++p) {
        const Vec3& direction = directions[p];
        for (int e=0; e<static_cast<int>(system.mesh.elements.size()); ++e)
            for (int q=0; q<quad.npts; ++q) {
                const auto sample = evaluate_muller_frame(system.mesh,e,quad.pts[q][0],quad.pts[q][1]);
                const auto basis = evaluate_muller_basis(system.mesh,e,sample);
                const Complex phase = prefactor * (0.5*quad.wts[q]*sample.jacobian) *
                    std::exp(Complex(0,-ka*direction.dot(sample.position)));
                for (int a=0; a<basis.count; ++a) {
                    const Vec3 v = basis.values[a];
                    const Vec3 j = v - direction*direction.dot(v);
                    const Vec3 m = direction.cross(v)*(-1.0);
                    for (int pol=0; pol<2; ++pol) {
                        const Vec3& tangent = pol ? phi[p] : theta[p];
                        const size_t row = static_cast<size_t>(2*p+pol)*n;
                        d[row+basis.dofs[a]] += phase*tangent.dot(j);
                        d[row+current+basis.dofs[a]] += phase*tangent.dot(m);
                    }
                }
            }
    }
    const auto projection_done = std::chrono::steady_clock::now();
    // Independently verify the radiation matrix against the library routine.
    std::vector<Complex> x(n), field;
    for (int i=0; i<n; ++i) x[i]=Complex(std::sin(0.7*i), std::cos(0.3*i));
    muller_nodal_farfield(system.mesh,x.data(),x.data()+current,ka,directions,field,13);
    double error=0, norm=0;
    for (int row=0; row<channels; ++row) {
        Complex prediction=0;
        for (int i=0; i<n; ++i) prediction += d[static_cast<size_t>(row)*n+i]*x[i];
        const int p=row/2;
        const Vec3& tangent=row%2 ? phi[p] : theta[p];
        const Complex reference=tangent.x*field[3*p]+tangent.y*field[3*p+1]+tangent.z*field[3*p+2];
        error += std::norm(prediction-reference); norm += std::norm(reference);
    }
    const double relative_error=std::sqrt(error/norm);
    if (relative_error>1e-12) throw std::runtime_error("radiation matrix verification failed");
    write_binary(prefix+".A.bin",system.matrix);
    write_binary(prefix+".B.bin",b);
    write_binary(prefix+".D.bin",d);
    const auto seconds=[](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point z) {
        return std::chrono::duration<double>(z-a).count();
    };
    std::ofstream meta(prefix+".json");
    double radius = 0;
    for (const Vec3& vertex : mesh.verts) radius = std::max(radius, vertex.norm());
    meta << std::setprecision(17) << "{\"dofs\":" << n << ",\"points\":" << points
         << ",\"bounding_radius\":" << radius << ",\"quad\":" << quad_order << ",\"duffy\":" << duffy_order
         << ",\"ka\":" << ka << ",\"ri\":" << ri << ",\"ref\":" << ref
         << ",\"shape\":\"" << shape << "\",\"assembly_s\":" << seconds(start,assembled)
         << ",\"rhs_s\":" << seconds(assembled,rhs_done)
         << ",\"radiation_s\":" << seconds(rhs_done,projection_done)
         << ",\"radiation_check_relative_error\":" << relative_error << "}\n";
    std::printf("T-matrix bridge: %d unknowns, %zu directions, radiation error %.3e\n",n,points,relative_error);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr,"%s\n",error.what()); return 1;
}
