#include "types.h"
#include "mesh.h"
#include "rwg.h"
#include "quadrature.h"
#include "pmchwt.h"
#include "solver.h"
#include "rhs.h"
#include "farfield.h"
#include "orient.h"
#include "output.h"
#include "bem_fmm.h"
#include "gmres.h"
#include "block_gmres.h"
#include "precond.h"
#include "gpu_select.h"
#include "precond_policy.h"
#include "solver_policy.h"
#include "operator_config.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <climits>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
class PinnedHostBuffer {
public:
    PinnedHostBuffer() : ptr_(0), size_(0), cap_(0), pinned_(false) {}
    ~PinnedHostBuffer() { release(); }

    void resize(size_t n)
    {
        size_ = n;
        if (n <= cap_)
            return;
        size_t new_cap = n;
        if (cap_ > 0) {
            size_t grown = cap_ + cap_ / 2;
            if (grown > new_cap)
                new_cap = grown;
        }
        release();
        if (new_cap == 0)
            return;
        cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&ptr_), new_cap * sizeof(T), cudaHostAllocDefault);
        if (err == cudaSuccess) {
            pinned_ = true;
        } else {
            cudaGetLastError();
            ptr_ = static_cast<T*>(std::malloc(new_cap * sizeof(T)));
            pinned_ = false;
            if (!ptr_) {
                fprintf(stderr, "Error: failed to allocate pinned host buffer fallback (%.1f MB)\n",
                        (double)(new_cap * sizeof(T)) / (1024.0 * 1024.0));
                std::abort();
            }
        }
        cap_ = new_cap;
        size_ = n;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }
    size_t size() const { return size_; }
    bool pinned() const { return pinned_; }

private:
    PinnedHostBuffer(const PinnedHostBuffer&);
    PinnedHostBuffer& operator=(const PinnedHostBuffer&);

    void release()
    {
        if (!ptr_)
            return;
        if (pinned_)
            cudaFreeHost(ptr_);
        else
            std::free(ptr_);
        ptr_ = 0;
        size_ = cap_ = 0;
        pinned_ = false;
    }

    T* ptr_;
    size_t size_;
    size_t cap_;
    bool pinned_;
};

enum SolverKind {
    SOLVER_AUTO,
    SOLVER_DENSE,
    SOLVER_FMM,
    SOLVER_PFFT,
    SOLVER_SPFFT
};

static const char* solver_name(SolverKind s)
{
    switch (s) {
    case SOLVER_DENSE: return "Dense";
    case SOLVER_FMM: return "FMM";
    case SOLVER_PFFT: return "pFFT";
    case SOLVER_SPFFT: return "SurfPFFT";
    case SOLVER_AUTO:
    default: return "auto";
    }
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --ka F          Size parameter (required)\n");
    printf("  --ri RE IM      Refractive index (default: 1.3116 0)\n");
    printf("  --shape TYPE    sphere or hex_prism (default: sphere)\n");
    printf("  --obj FILE      Load OBJ mesh, normalize to unit equal-volume sphere radius\n");
    printf("  --subdiv N      Flat midpoint subdivisions for OBJ mesh (default: 0)\n");
    printf("  --prism-aspect F  Hex prism h/Dx, ADDA convention (default: 1)\n");
    printf("  --edge-refine N   Prism local edge-refinement passes (default: 0; experimental)\n");
    printf("  --ref N         Icosphere refinement level (default: 3)\n");
    printf("  --orient NA NB NG  Orientation quadrature; solves all NA*NB*NG Euler angles (default: 8 8 1)\n");
    printf("  --orient-file FILE  Read explicit alpha beta gamma [weight] orientations in degrees\n");
    printf("  --orient-bg-file FILE  Read explicit beta gamma [weight] orientations in degrees; compatible with --alpha-avg\n");
    printf("  --alpha-avg N  Fast alpha/phi far-field average only; use with --orient 1 NB NG\n");
    printf("  --orient-start I    First orientation index for chunked averaging (default: 0)\n");
    printf("  --orient-count N    Number of orientations in chunk (default: all)\n");
    printf("  --orient-split-dir DIR  Write one JSON per solved orientation in a chunk\n");
    printf("  --orient-split-indices FILE  Master indices for --orient-split-dir outputs\n");
    printf("  --orient-split-total N  Master orientation count for split JSON metadata\n");
    printf("  --ntheta N      Number of scattering angles (default: 181)\n");
    printf("  --scat-plane P  Single-orient scattering plane: yz or xz (default: yz, ADDA convention)\n");
    printf("  --quad N        Quadrature order: 4, 7, 13 (default: 7; hex_prism guarded auto)\n");
    printf("  --out FILE      Output JSON file (default: result.json)\n");
    printf("  --single        Single orientation (no averaging)\n");
    printf("  --force-orient  Force explicit orientation loop for sphere mesh\n");
    printf("  --adda-compare  Hex-prism comparison profile with guarded auto accuracy\n");
    printf("                 (quad7/digits5/tol1e-3/restart300 by default)\n");
    printf("  --accurate      Use controlled non-sphere defaults: quad7, digits6, leaf128\n");
    printf("                 (OBJ default: digits7/tol1e-5/restart1000; mesh guard: digits8/restart1400)\n");
    printf("  --fast-obj      Reproduce old fast OBJ defaults; use only for speed diagnostics\n");
    printf("  --solver TYPE   Solver backend: auto, dense, fmm, pfft, spfft (spfft falls back to fmm; default: auto)\n");
    printf("  --system TYPE   Linear system: pmchwt, balanced, muller, muller-balanced, muller2, or muller2-balanced (default: auto-balanced for |m-1|>=0.05)\n");
    printf("  --fmm           Use FMM+GMRES instead of dense LU\n");
    printf("  --fmm-digits N  FMM accuracy digits (default: 3; hex_prism auto uses 5, accurate uses 6)\n");
    printf("  --gmres-tol F   GMRES relative tolerance (default: 1e-4; hex_prism auto uses 1e-3)\n");
    printf("  --gmres-restart N  GMRES restart (default: 150; hex_prism auto uses 300)\n");
    printf("  --gmres-max-cycles N  Maximum restarted GMRES cycles (default: profile policy)\n");
    printf("  --krylov TYPE   FMM iterative method: gmres, gpu-gmres, bicgstab, bicgstab-rr, cgs-rr, gpu-adaptive, gpu-native, hybrid, or auto (default: gmres; GPU modes fall back to GMRES when needed)\n");
    printf("  --max-leaf N    FMM max particles per leaf (default: 128)\n");
    printf("  --no-prec       Disable automatic FMM preconditioner (auto skips small non-sphere cases)\n");
    printf("  --export-currents FILE  Write equivalent surface currents for the single orientation to VTK\n");
    printf("  --mesh-quality-report FILE  Write mesh quality JSON before solve\n");
    printf("  --mesh-quality-strict       Stop if mesh gate fails (closed, outward, min angle >=20 deg, aspect <=12)\n");
    printf("  --mesh-quality-only         Build mesh, write/print quality report, then exit\n");
}

struct CellCurrent {
    Vec3 J_re, J_im;
    Vec3 M_re, M_im;
};

static void add_scaled_complex_vec(Vec3& re, Vec3& im, const cdouble& c, const Vec3& v)
{
    re = re + v * c.real();
    im = im + v * c.imag();
}

static std::vector<CellCurrent> reconstruct_cell_currents(
    const RWG& rwg, const Mesh& mesh, const cdouble* J, const cdouble* M)
{
    std::vector<CellCurrent> out(mesh.nt());
    for (int n = 0; n < rwg.N; n++) {
        for (int side = 0; side < 2; side++) {
            int sign = (side == 0) ? 1 : -1;
            int ti = (sign > 0) ? rwg.tri_p[n] : rwg.tri_m[n];
            Vec3 free_v = (sign > 0) ? rwg.free_p[n] : rwg.free_m[n];
            double area = (sign > 0) ? rwg.area_p[n] : rwg.area_m[n];
            Vec3 v0, v1, v2;
            mesh.tri_verts(ti, v0, v1, v2);
            Vec3 rc = (v0 + v1 + v2) * (1.0 / 3.0);
            Vec3 f = (rc - free_v) * (sign * rwg.length[n] / (2.0 * area));
            add_scaled_complex_vec(out[ti].J_re, out[ti].J_im, J[n], f);
            add_scaled_complex_vec(out[ti].M_re, out[ti].M_im, M[n], f);
        }
    }
    return out;
}

static double vec_complex_abs(const Vec3& re, const Vec3& im)
{
    return std::sqrt(re.norm2() + im.norm2());
}

static void write_vec_array(std::ofstream& os, const char* name,
                            const std::vector<CellCurrent>& c,
                            Vec3 CellCurrent::*re_member,
                            Vec3 CellCurrent::*im_member)
{
    os << "VECTORS " << name << " double\n";
    for (const auto& ci : c) {
        const Vec3& re = ci.*re_member;
        const Vec3& im = ci.*im_member;
        Vec3 v = (re.norm2() >= im.norm2()) ? re : im;
        os << v.x << " " << v.y << " " << v.z << "\n";
    }
}

static bool export_currents_vtk(const char* path, const Mesh& mesh, const RWG& rwg,
                                const cdouble* J_par, const cdouble* M_par,
                                const cdouble* J_perp, const cdouble* M_perp)
{
    std::vector<CellCurrent> par = reconstruct_cell_currents(rwg, mesh, J_par, M_par);
    std::vector<CellCurrent> perp = reconstruct_cell_currents(rwg, mesh, J_perp, M_perp);

    std::ofstream os(path);
    if (!os) {
        fprintf(stderr, "Error: cannot write currents VTK: %s\n", path);
        return false;
    }
    os << std::setprecision(17);
    os << "# vtk DataFile Version 3.0\n";
    os << "BEM-CUDA equivalent PMCHWT surface currents\n";
    os << "ASCII\n";
    os << "DATASET POLYDATA\n";
    os << "POINTS " << mesh.nv() << " double\n";
    for (const Vec3& v : mesh.verts)
        os << v.x << " " << v.y << " " << v.z << "\n";
    os << "POLYGONS " << mesh.nt() << " " << 4 * mesh.nt() << "\n";
    for (int t = 0; t < mesh.nt(); t++)
        os << "3 " << mesh.tris[3*t] << " " << mesh.tris[3*t+1] << " " << mesh.tris[3*t+2] << "\n";

    os << "CELL_DATA " << mesh.nt() << "\n";
    os << "SCALARS J_par_abs double 1\nLOOKUP_TABLE default\n";
    for (const auto& ci : par) os << vec_complex_abs(ci.J_re, ci.J_im) << "\n";
    os << "SCALARS M_par_abs double 1\nLOOKUP_TABLE default\n";
    for (const auto& ci : par) os << vec_complex_abs(ci.M_re, ci.M_im) << "\n";
    os << "SCALARS J_perp_abs double 1\nLOOKUP_TABLE default\n";
    for (const auto& ci : perp) os << vec_complex_abs(ci.J_re, ci.J_im) << "\n";
    os << "SCALARS M_perp_abs double 1\nLOOKUP_TABLE default\n";
    for (const auto& ci : perp) os << vec_complex_abs(ci.M_re, ci.M_im) << "\n";
    write_vec_array(os, "J_par_vector", par, &CellCurrent::J_re, &CellCurrent::J_im);
    write_vec_array(os, "M_par_vector", par, &CellCurrent::M_re, &CellCurrent::M_im);
    write_vec_array(os, "J_perp_vector", perp, &CellCurrent::J_re, &CellCurrent::J_im);
    write_vec_array(os, "M_perp_vector", perp, &CellCurrent::M_re, &CellCurrent::M_im);
    return true;
}

static bool export_coefficients_json(const char* path, int N,
                                     const cdouble* x_par,
                                     const cdouble* x_perp)
{
    if (!path || !*path)
        return false;
    std::ofstream os(path);
    if (!os) {
        fprintf(stderr, "Error: cannot write coefficient JSON: %s\n", path);
        return false;
    }
    os << std::setprecision(17);
    os << "{\n";
    os << "  \"basis_count\": " << N << ",\n";
    auto write_vec = [&](const char* name, const cdouble* x) {
        os << "  \"" << name << "\": [";
        for (int i = 0; i < 2 * N; i++) {
            if (i) os << ", ";
            os << "[" << x[i].real() << ", " << x[i].imag() << "]";
        }
        os << "]";
    };
    write_vec("x_par", x_par);
    os << ",\n";
    write_vec("x_perp", x_perp);
    os << "\n}\n";
    if (!os) {
        fprintf(stderr, "Error: failed while writing coefficient JSON: %s\n", path);
        return false;
    }
    printf("  Exported solve coefficients: %s\n", path);
    return true;
}

static double complex_vec_norm(const cdouble* x, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += std::norm(x[i]);
    return std::sqrt(s);
}

static bool load_coefficients_json(const char* path, int N,
                                   std::vector<cdouble>& x_par,
                                   std::vector<cdouble>& x_perp)
{
    if (!path || !*path)
        return false;
    std::ifstream is(path);
    if (!is) {
        fprintf(stderr, "Error: cannot read coefficient JSON: %s\n", path);
        return false;
    }
    std::string text((std::istreambuf_iterator<char>(is)),
                     std::istreambuf_iterator<char>());
    std::vector<double> values;
    const char* p = text.c_str();
    char* end = nullptr;
    while (*p) {
        double v = std::strtod(p, &end);
        if (end != p) {
            values.push_back(v);
            p = end;
        } else {
            ++p;
        }
    }
    const size_t need = 1 + (size_t)8 * (size_t)N;
    if (values.size() < need) {
        fprintf(stderr, "Error: coefficient JSON %s has %zu numeric values, need at least %zu\n",
                path, values.size(), need);
        return false;
    }
    int file_N = (int)std::llround(values[0]);
    if (file_N != N) {
        fprintf(stderr, "Error: coefficient JSON basis_count=%d but current N=%d\n", file_N, N);
        return false;
    }
    x_par.assign(2 * N, cdouble(0));
    x_perp.assign(2 * N, cdouble(0));
    size_t k = 1;
    for (int i = 0; i < 2 * N; i++, k += 2)
        x_par[i] = cdouble(values[k], values[k + 1]);
    for (int i = 0; i < 2 * N; i++, k += 2)
        x_perp[i] = cdouble(values[k], values[k + 1]);
    return true;
}

static double relative_residual(BemFmmOperator& op, const cdouble* x, const cdouble* b)
{
    const int n = op.system_size;
    std::vector<cdouble> y(n);
    op.matvec(x, y.data());
    for (int i = 0; i < n; i++)
        y[i] = b[i] - y[i];
    double bn = complex_vec_norm(b, n);
    if (bn < 1e-300)
        bn = 1.0;
    return complex_vec_norm(y.data(), n) / bn;
}

static bool solve_small_linear(std::vector<cdouble>& A, std::vector<cdouble>& b, int n) {
    for (int k = 0; k < n; k++) {
        int piv = k;
        double best = std::abs(A[k * n + k]);
        for (int i = k + 1; i < n; i++) {
            double v = std::abs(A[i * n + k]);
            if (v > best) {
                best = v;
                piv = i;
            }
        }
        if (best < 1e-24)
            return false;
        if (piv != k) {
            for (int j = k; j < n; j++)
                std::swap(A[k * n + j], A[piv * n + j]);
            std::swap(b[k], b[piv]);
        }
        cdouble diag = A[k * n + k];
        for (int i = k + 1; i < n; i++) {
            cdouble f = A[i * n + k] / diag;
            A[i * n + k] = 0;
            for (int j = k + 1; j < n; j++)
                A[i * n + j] -= f * A[k * n + j];
            b[i] -= f * b[k];
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        cdouble s = b[i];
        for (int j = i + 1; j < n; j++)
            s -= A[i * n + j] * b[j];
        b[i] = s / A[i * n + i];
    }
    return true;
}

static bool parse_cli_int_arg(const char* opt, const char* value, int& out)
{
    if (!bem_parse_int_value(value, &out)) {
        fprintf(stderr, "Error: %s expects an integer, got '%s'\n", opt, value ? value : "");
        return false;
    }
    return true;
}

static bool parse_cli_double_arg(const char* opt, const char* value, double& out)
{
    if (!bem_parse_double_value(value, &out)) {
        fprintf(stderr, "Error: %s expects a finite number, got '%s'\n", opt, value ? value : "");
        return false;
    }
    return true;
}

static bool parse_cli_string_arg(int argc, char** argv, int& i, const char* opt, const char*& out)
{
    if (i + 1 >= argc || std::strncmp(argv[i + 1], "--", 2) == 0 || std::strcmp(argv[i + 1], "-h") == 0) {
        fprintf(stderr, "Error: %s expects a value\n", opt);
        return false;
    }
    out = argv[++i];
    return true;
}

static void recycle_initial_guess_pair(const std::vector<std::vector<cdouble>>& hist_b,
                                       const std::vector<std::vector<cdouble>>& hist_x,
                                       const cdouble* b1, const cdouble* b2, int n,
                                       cdouble* x1, cdouble* x2)
{
    int m = (int)hist_b.size();
    if (m == 0) {
        std::fill(x1, x1 + n, cdouble(0));
        std::fill(x2, x2 + n, cdouble(0));
        return;
    }

    std::vector<cdouble> G(m * m), rhs1(m), rhs2(m);
    for (int i = 0; i < m; i++) {
        cdouble bi1(0), bi2(0);
        for (int k = 0; k < n; k++) {
            cdouble hb = std::conj(hist_b[i][k]);
            bi1 += hb * b1[k];
            bi2 += hb * b2[k];
        }
        rhs1[i] = bi1;
        rhs2[i] = bi2;
        for (int j = 0; j < m; j++) {
            cdouble gij(0);
            for (int k = 0; k < n; k++)
                gij += std::conj(hist_b[i][k]) * hist_b[j][k];
            G[i * m + j] = gij;
        }
    }

    double trace = 0.0;
    for (int i = 0; i < m; i++)
        trace += std::abs(G[i * m + i]);
    double lambda = trace * 1e-12 / m;
    for (int i = 0; i < m; i++)
        G[i * m + i] += lambda;

    std::vector<cdouble> G2 = G;
    if (!solve_small_linear(G, rhs1, m) || !solve_small_linear(G2, rhs2, m)) {
        std::fill(x1, x1 + n, cdouble(0));
        std::fill(x2, x2 + n, cdouble(0));
        return;
    }

    std::fill(x1, x1 + n, cdouble(0));
    std::fill(x2, x2 + n, cdouble(0));
    for (int i = 0; i < m; i++) {
        const std::vector<cdouble>& xi = hist_x[i];
        cdouble c1 = rhs1[i];
        cdouble c2 = rhs2[i];
        for (int k = 0; k < n; k++) {
            x1[k] += c1 * xi[k];
            x2[k] += c2 * xi[k];
        }
    }
}

static void push_history(std::vector<std::vector<cdouble>>& hist_b,
                         std::vector<std::vector<cdouble>>& hist_x,
                         const cdouble* b, const cdouble* x,
                         int n, int max_hist)
{
    if ((int)hist_b.size() == max_hist) {
        hist_b.erase(hist_b.begin());
        hist_x.erase(hist_x.begin());
    }
    hist_b.emplace_back(b, b + n);
    hist_x.emplace_back(x, x + n);
}

enum class OrientWarmStart {
    Zero,
    Previous,
    Recycle
};

static OrientWarmStart parse_orient_warm_start()
{
    const char* env = std::getenv("BEM_ORIENT_WARM_START");
    if (!env || env[0] == '\0')
        return OrientWarmStart::Zero;
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "zero") == 0 || std::strcmp(env, "none") == 0)
        return OrientWarmStart::Zero;
    if (std::strcmp(env, "recycle") == 0 || std::strcmp(env, "history") == 0)
        return OrientWarmStart::Recycle;
    if (std::strcmp(env, "prev") == 0 || std::strcmp(env, "previous") == 0 || std::strcmp(env, "1") == 0)
        return OrientWarmStart::Previous;
    fprintf(stderr, "Warning: unknown BEM_ORIENT_WARM_START=%s; using zero\n", env);
    return OrientWarmStart::Zero;
}

static const char* orient_warm_start_name(OrientWarmStart mode)
{
    switch (mode) {
    case OrientWarmStart::Zero: return "zero";
    case OrientWarmStart::Recycle: return "recycle";
    case OrientWarmStart::Previous:
    default: return "previous";
    }
}

static int hex_auto_refinement(double ka)
{
    if (ka < 7.0)
        return 2;
    if (ka < 14.0)
        return 3;
    if (ka < 28.0)
        return 4;
    return 5;
}

static double orient_distance2(const Orientation& a, const Orientation& b)
{
    double s = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double d = a.RT.m[i][j] - b.RT.m[i][j];
            s += d * d;
        }
    }
    return s;
}

static void reorder_orientations_nearest(std::vector<Orientation>& orients)
{
    int n = (int)orients.size();
    if (n <= 2)
        return;

    std::vector<Orientation> ordered;
    ordered.reserve(n);
    std::vector<char> used(n, 0);

    int cur = 0;
    for (int step = 0; step < n; step++) {
        ordered.push_back(orients[cur]);
        used[cur] = 1;

        if (step == n - 1)
            break;

        int best = -1;
        double best_d = 1e300;
        for (int i = 0; i < n; i++) {
            if (used[i])
                continue;
            double d = orient_distance2(orients[cur], orients[i]);
            if (d < best_d) {
                best_d = d;
                best = i;
            }
        }
        cur = best;
    }

    orients.swap(ordered);
}

static void project_random_orientation_mueller(double* M, int ntheta)
{
    for (int t = 0; t < ntheta; t++) {
        auto at = [&](int i, int j) -> double& {
            return M[(i * 4 + j) * ntheta + t];
        };

        double s12 = 0.5 * (at(1,0) - at(0,1));
        at(0,1) = s12;
        at(1,0) = s12;

        double s34 = -0.5 * (at(2,3) + at(3,2));
        at(2,3) = s34;
        at(3,2) = -s34;

        at(1,1) = -at(1,1);
        at(3,3) = -at(3,3);

        at(0,2) = 0.0; at(0,3) = 0.0;
        at(1,2) = 0.0; at(1,3) = 0.0;
        at(2,0) = 0.0; at(2,1) = 0.0;
        at(3,0) = 0.0; at(3,1) = 0.0;
    }
}

static void transform_rhs_to_n_form(cdouble* b, int N, int mode = 0)
{
    for (int i = 0; i < N; i++) {
        cdouble e = b[i];
        cdouble h = b[N + i];
        switch (mode) {
        case 1: b[i] = h;  b[N + i] = e;  break;
        case 2: b[i] = -h; b[N + i] = e;  break;
        case 3: b[i] = -h; b[N + i] = -e; break;
        case 4: b[i] = e;  b[N + i] = h;  break;
        case 5: b[i] = e;  b[N + i] = -h; break;
        case 6: b[i] = -e; b[N + i] = h;  break;
        case 7: b[i] = -e; b[N + i] = -h; break;
        default:
            b[i] = h;
            b[N + i] = -e;
            break;
        }
    }
}

static void recover_n_form_farfield_currents(cdouble& j, cdouble& m, int mode)
{
    cdouble aj = j;
    cdouble am = m;
    switch (mode) {
    case 1:
        j = am;  m = aj;  break;
    case 2:
        j = -am; m = aj;  break;
    case 3:
        j = am;  m = -aj; break;
    case 4:
        m = -m;          break;
    case 5:
        j = -j;          break;
    default:
        break;
    }
}

static void recover_n_form_farfield_pair(const cdouble* J_in, const cdouble* M_in,
                                         int N, int mode,
                                         std::vector<cdouble>& J_out,
                                         std::vector<cdouble>& M_out)
{
    J_out.resize(N);
    M_out.resize(N);
    for (int i = 0; i < N; i++) {
        cdouble j = J_in[i];
        cdouble m = M_in[i];
        recover_n_form_farfield_currents(j, m, mode);
        J_out[i] = j;
        M_out[i] = m;
    }
}

static bool load_orientation_file(const char* filename, std::vector<Orientation>& orients)
{
    std::ifstream in(filename);
    if (!in) {
        fprintf(stderr, "Error: cannot open orientation file %s\n", filename);
        return false;
    }

    struct Row {
        double alpha;
        double beta;
        double gamma;
        double weight;
    };
    std::vector<Row> rows;
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        lineno++;
        size_t hash = line.find('#');
        if (hash != std::string::npos)
            line.erase(hash);
        std::istringstream iss(line);
        Row row;
        row.weight = 1.0;
        if (!(iss >> row.alpha >> row.beta >> row.gamma))
            continue;
        if (iss >> row.weight) {
            if (row.weight < 0.0) {
                fprintf(stderr, "Error: negative orientation weight at %s:%d\n", filename, lineno);
                return false;
            }
        }
        rows.push_back(row);
    }
    if (rows.empty()) {
        fprintf(stderr, "Error: no orientations found in %s\n", filename);
        return false;
    }

    double wsum = 0.0;
    for (const Row& row : rows)
        wsum += row.weight;
    if (wsum <= 0.0) {
        fprintf(stderr, "Error: orientation weights sum to zero in %s\n", filename);
        return false;
    }

    orients.clear();
    orients.reserve(rows.size());
    for (const Row& row : rows) {
        double alpha = row.alpha * M_PI / 180.0;
        double beta = row.beta * M_PI / 180.0;
        double gamma = row.gamma * M_PI / 180.0;
        Mat3 R = euler_rotation(alpha, beta, gamma);
        Orientation o;
        o.RT = R.T();
        o.weight = row.weight / wsum;
        orients.push_back(o);
    }
    return true;
}

static bool load_beta_gamma_orientation_file(const char* filename, std::vector<Orientation>& orients)
{
    std::ifstream in(filename);
    if (!in) {
        fprintf(stderr, "Error: cannot open beta/gamma orientation file %s\n", filename);
        return false;
    }

    struct Row {
        double beta;
        double gamma;
        double weight;
    };
    std::vector<Row> rows;
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        lineno++;
        size_t hash = line.find('#');
        if (hash != std::string::npos)
            line.erase(hash);
        std::istringstream iss(line);
        Row row;
        row.weight = 1.0;
        if (!(iss >> row.beta >> row.gamma))
            continue;
        if (iss >> row.weight) {
            if (row.weight < 0.0) {
                fprintf(stderr, "Error: negative beta/gamma orientation weight at %s:%d\n", filename, lineno);
                return false;
            }
        }
        rows.push_back(row);
    }
    if (rows.empty()) {
        fprintf(stderr, "Error: no beta/gamma orientations found in %s\n", filename);
        return false;
    }

    double wsum = 0.0;
    for (const Row& row : rows)
        wsum += row.weight;
    if (wsum <= 0.0) {
        fprintf(stderr, "Error: beta/gamma orientation weights sum to zero in %s\n", filename);
        return false;
    }

    orients.clear();
    orients.reserve(rows.size());
    for (const Row& row : rows) {
        double beta = row.beta * M_PI / 180.0;
        double gamma = row.gamma * M_PI / 180.0;
        Mat3 R = euler_rotation(0.0, beta, gamma);
        Orientation o;
        o.RT = R.T();
        o.weight = row.weight / wsum;
        orients.push_back(o);
    }
    return true;
}

static bool load_int_index_file(const char* filename, std::vector<int>& indices)
{
    std::ifstream in(filename);
    if (!in) {
        fprintf(stderr, "Error: cannot open index file %s\n", filename);
        return false;
    }
    indices.clear();
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        line_no++;
        size_t hash = line.find('#');
        if (hash != std::string::npos)
            line.resize(hash);
        std::istringstream iss(line);
        int idx = -1;
        if (!(iss >> idx))
            continue;
        if (idx < 0) {
            fprintf(stderr, "Error: negative index in %s:%d\n", filename, line_no);
            return false;
        }
        indices.push_back(idx);
    }
    if (indices.empty()) {
        fprintf(stderr, "Error: index file %s is empty\n", filename);
        return false;
    }
    return true;
}

static std::string orient_part_path(const char* dir, int index)
{
    char buf[4096];
    std::snprintf(buf, sizeof(buf), "%s/part_%04d.json", dir, index);
    return std::string(buf);
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
#ifdef _OPENMP
    omp_set_dynamic(0);
    if (!std::getenv("OMP_NUM_THREADS")) {
        int threads = std::min(16, omp_get_max_threads());
        omp_set_num_threads(threads);
    }
#endif
    // Default parameters
    double ka = 0;
    double n_re = 1.3116, n_im = 0.0;
    const char* shape = "sphere";
    const char* obj_file = nullptr;
    int obj_subdiv = 0;
    double prism_aspect = 1.0;
    bool prism_aspect_set = false;
    int edge_refine = -1;
    int refinements = 3;
    bool refinements_set = false;
    int n_alpha = 8, n_beta = 8, n_gamma = 1;
    const char* orient_file = nullptr;
    const char* orient_bg_file = nullptr;
    const char* orient_split_dir = nullptr;
    const char* orient_split_indices_file = nullptr;
    int orient_split_total = -1;
    int alpha_avg = 1;
    int orient_start = 0, orient_count = -1;
    int ntheta = 181;
    const char* scat_plane = "yz";
    int quad_order = 7;
    bool quad_order_set = false;
    const char* outfile = "result.json";
    bool single_orient = false;
    bool force_orient = false;
    bool accurate_mode = false;
    bool fast_obj_mode = false;
    bool adda_compare_mode = false;
    SolverKind solver = SOLVER_AUTO;
    bool solver_explicit = false;
    bool use_fmm = false;
    bool use_pfft = false;
    bool use_spfft = false;
    bool use_prec = false;
    bool no_prec = false;
    bool fmm_test = false;
    int fmm_digits = 3;
    bool fmm_digits_set = false;
    double gmres_tol = 1e-4;
    bool gmres_tol_set = false;
    int gmres_restart = 150;
    bool gmres_restart_set = false;
    int gmres_max_cycles_cli = 0;
    bool gmres_max_cycles_set = false;
    const char* krylov_kind = "gmres";
    bool krylov_kind_set = false;
    bool force_gpu_gmres = false;
    bool requested_gpu_adaptive = false;
    int max_leaf = 128;
    bool max_leaf_set = false;
    const char* system_kind = "pmchwt";
    bool system_kind_set = false;
    const char* export_currents_file = nullptr;
    const char* mesh_quality_report_file = nullptr;
    bool mesh_quality_strict = false;
    bool mesh_quality_only = false;

    // Parse CLI
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ka") == 0) {
            if (i + 1 >= argc || !parse_cli_double_arg("--ka", argv[++i], ka))
                return 1;
        } else if (strcmp(argv[i], "--ri") == 0) {
            if (i + 2 >= argc) {
                fprintf(stderr, "Error: --ri expects two finite numbers\n");
                return 1;
            }
            if (!parse_cli_double_arg("--ri real", argv[++i], n_re) ||
                !parse_cli_double_arg("--ri imag", argv[++i], n_im))
                return 1;
        } else if (strcmp(argv[i], "--shape") == 0) {
            if (!parse_cli_string_arg(argc, argv, i, "--shape", shape))
                return 1;
        } else if (strcmp(argv[i], "--obj") == 0) {
            if (!parse_cli_string_arg(argc, argv, i, "--obj", obj_file))
                return 1;
            shape = "obj";
        } else if (strcmp(argv[i], "--subdiv") == 0) {
            if (i + 1 >= argc || !parse_cli_int_arg("--subdiv", argv[++i], obj_subdiv))
                return 1;
        } else if (strcmp(argv[i], "--prism-aspect") == 0) {
            if (i + 1 >= argc || !parse_cli_double_arg("--prism-aspect", argv[++i], prism_aspect))
                return 1;
            prism_aspect_set = true;
        } else if (strcmp(argv[i], "--edge-refine") == 0) {
            if (i + 1 >= argc || !parse_cli_int_arg("--edge-refine", argv[++i], edge_refine))
                return 1;
        } else if (strcmp(argv[i], "--ref") == 0) {
            if (i + 1 >= argc || !parse_cli_int_arg("--ref", argv[++i], refinements))
                return 1;
            refinements_set = true;
        } else if (strcmp(argv[i], "--orient") == 0) {
            if (i + 3 >= argc) {
                fprintf(stderr, "Error: --orient expects three integer counts\n");
                return 1;
            }
            if (!parse_cli_int_arg("--orient alpha", argv[++i], n_alpha) ||
                !parse_cli_int_arg("--orient beta", argv[++i], n_beta) ||
                !parse_cli_int_arg("--orient gamma", argv[++i], n_gamma))
                return 1;
        } else if (strcmp(argv[i], "--orient-file") == 0) {
            if (!parse_cli_string_arg(argc, argv, i, "--orient-file", orient_file))
                return 1;
        } else if (strcmp(argv[i], "--orient-bg-file") == 0) {
            if (!parse_cli_string_arg(argc, argv, i, "--orient-bg-file", orient_bg_file))
                return 1;
        } else if (strcmp(argv[i], "--orient-split-dir") == 0) {
            if (!parse_cli_string_arg(argc, argv, i, "--orient-split-dir", orient_split_dir))
                return 1;
        } else if (strcmp(argv[i], "--orient-split-indices") == 0) {
            if (!parse_cli_string_arg(argc, argv, i, "--orient-split-indices", orient_split_indices_file))
                return 1;
        } else if (strcmp(argv[i], "--orient-split-total") == 0) {
            if (i + 1 >= argc || !parse_cli_int_arg("--orient-split-total", argv[++i], orient_split_total))
                return 1;
        } else if (strcmp(argv[i], "--alpha-avg") == 0) {
            if (i + 1 >= argc || !parse_cli_int_arg("--alpha-avg", argv[++i], alpha_avg))
                return 1;
        } else if (strcmp(argv[i], "--orient-start") == 0) {
            if (i + 1 >= argc || !parse_cli_int_arg("--orient-start", argv[++i], orient_start))
                return 1;
        } else if (strcmp(argv[i], "--orient-count") == 0) {
            if (i + 1 >= argc || !parse_cli_int_arg("--orient-count", argv[++i], orient_count))
                return 1;
        } else if (strcmp(argv[i], "--ntheta") == 0) {
            if (i + 1 >= argc || !parse_cli_int_arg("--ntheta", argv[++i], ntheta))
                return 1;
        } else if (strcmp(argv[i], "--scat-plane") == 0) {
            if (!parse_cli_string_arg(argc, argv, i, "--scat-plane", scat_plane))
                return 1;
        } else if (strcmp(argv[i], "--quad") == 0) {
            if (i + 1 >= argc || !parse_cli_int_arg("--quad", argv[++i], quad_order))
                return 1;
            quad_order_set = true;
        } else if (strcmp(argv[i], "--out") == 0) {
            if (!parse_cli_string_arg(argc, argv, i, "--out", outfile))
                return 1;
        } else if (strcmp(argv[i], "--single") == 0) {
            single_orient = true;
        } else if (strcmp(argv[i], "--force-orient") == 0) {
            force_orient = true;
        } else if (strcmp(argv[i], "--accurate") == 0) {
            accurate_mode = true;
        } else if (strcmp(argv[i], "--fast-obj") == 0) {
            fast_obj_mode = true;
        } else if (strcmp(argv[i], "--adda-compare") == 0) {
            adda_compare_mode = true;
        } else if (strcmp(argv[i], "--fmm") == 0) {
            solver = SOLVER_FMM;
            solver_explicit = true;
        } else if (strcmp(argv[i], "--solver") == 0) {
            const char* solver_arg = nullptr;
            if (!parse_cli_string_arg(argc, argv, i, "--solver", solver_arg))
                return 1;
            solver_explicit = true;
            if (strcmp(solver_arg, "auto") == 0) {
                solver = SOLVER_AUTO;
            } else if (strcmp(solver_arg, "dense") == 0) {
                solver = SOLVER_DENSE;
            } else if (strcmp(solver_arg, "fmm") == 0) {
                solver = SOLVER_FMM;
            } else if (strcmp(solver_arg, "pfft") == 0) {
                solver = SOLVER_PFFT;
            } else if (strcmp(solver_arg, "spfft") == 0) {
                solver = SOLVER_SPFFT;
            } else {
                fprintf(stderr, "Error: --solver must be auto, dense, fmm, pfft, or spfft\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--system") == 0) {
            if (!parse_cli_string_arg(argc, argv, i, "--system", system_kind))
                return 1;
            system_kind_set = true;
            if (strcmp(system_kind, "pmchwt") != 0 &&
                strcmp(system_kind, "balanced") != 0 &&
                strcmp(system_kind, "muller") != 0 &&
                strcmp(system_kind, "muller-balanced") != 0 &&
                strcmp(system_kind, "muller2") != 0 &&
                strcmp(system_kind, "muller2-balanced") != 0) {
                fprintf(stderr, "Error: --system must be pmchwt, balanced, muller, muller-balanced, muller2, or muller2-balanced\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--no-prec") == 0) {
            no_prec = true;
        } else if (strcmp(argv[i], "--fmm-digits") == 0) {
            if (i + 1 >= argc || !parse_cli_int_arg("--fmm-digits", argv[++i], fmm_digits))
                return 1;
            fmm_digits_set = true;
        } else if (strcmp(argv[i], "--gmres-tol") == 0) {
            if (i + 1 >= argc || !parse_cli_double_arg("--gmres-tol", argv[++i], gmres_tol))
                return 1;
            gmres_tol_set = true;
        } else if (strcmp(argv[i], "--gmres-restart") == 0) {
            if (i + 1 >= argc || !parse_cli_int_arg("--gmres-restart", argv[++i], gmres_restart))
                return 1;
            gmres_restart_set = true;
        } else if (strcmp(argv[i], "--gmres-max-cycles") == 0) {
            if (i + 1 >= argc || !parse_cli_int_arg("--gmres-max-cycles", argv[++i], gmres_max_cycles_cli))
                return 1;
            gmres_max_cycles_set = true;
        } else if (strcmp(argv[i], "--krylov") == 0) {
            if (!parse_cli_string_arg(argc, argv, i, "--krylov", krylov_kind))
                return 1;
            krylov_kind_set = true;
            if (strcmp(krylov_kind, "gmres") != 0 &&
                strcmp(krylov_kind, "bicgstab") != 0 &&
                strcmp(krylov_kind, "bcgstab") != 0 &&
                strcmp(krylov_kind, "bicgstab-rr") != 0 &&
                strcmp(krylov_kind, "bicgstab_rr") != 0 &&
                strcmp(krylov_kind, "bcgstab-rr") != 0 &&
                strcmp(krylov_kind, "bcgstab_rr") != 0 &&
                strcmp(krylov_kind, "cgs") != 0 &&
                strcmp(krylov_kind, "cgs-rr") != 0 &&
                strcmp(krylov_kind, "cgs_rr") != 0 &&
                strcmp(krylov_kind, "gpu-gmres") != 0 &&
                strcmp(krylov_kind, "gpu_gmres") != 0 &&
                strcmp(krylov_kind, "hybrid") != 0 &&
                strcmp(krylov_kind, "gpu-hybrid") != 0 &&
                strcmp(krylov_kind, "gpu_hybrid") != 0 &&
                strcmp(krylov_kind, "gpu-adaptive") != 0 &&
                strcmp(krylov_kind, "gpu_adaptive") != 0 &&
                strcmp(krylov_kind, "gpu-native") != 0 &&
                strcmp(krylov_kind, "gpu_native") != 0 &&
                strcmp(krylov_kind, "auto") != 0) {
                fprintf(stderr, "Error: --krylov must be gmres, gpu-gmres, bicgstab, bicgstab-rr, cgs-rr, gpu-adaptive, gpu-native, hybrid, or auto\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--max-leaf") == 0) {
            if (i + 1 >= argc || !parse_cli_int_arg("--max-leaf", argv[++i], max_leaf))
                return 1;
            max_leaf_set = true;
        } else if (strcmp(argv[i], "--export-currents") == 0) {
            if (!parse_cli_string_arg(argc, argv, i, "--export-currents", export_currents_file))
                return 1;
        } else if (strcmp(argv[i], "--mesh-quality-report") == 0) {
            if (!parse_cli_string_arg(argc, argv, i, "--mesh-quality-report", mesh_quality_report_file))
                return 1;
        } else if (strcmp(argv[i], "--mesh-quality-strict") == 0) {
            mesh_quality_strict = true;
        } else if (strcmp(argv[i], "--mesh-quality-only") == 0) {
            mesh_quality_only = true;
        } else if (strcmp(argv[i], "--fmm-test") == 0) {
            fmm_test = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (ka <= 0) {
        fprintf(stderr, "Error: --ka must be positive\n");
        print_usage(argv[0]);
        return 1;
    }
    if (obj_subdiv < 0) {
        fprintf(stderr, "Error: --subdiv must be non-negative\n");
        return 1;
    }
    if (prism_aspect <= 0.0) {
        fprintf(stderr, "Error: --prism-aspect must be positive\n");
        return 1;
    }
    if (edge_refine < -1) {
        fprintf(stderr, "Error: --edge-refine must be non-negative, or omitted for auto\n");
        return 1;
    }
    if (refinements < 0) {
        fprintf(stderr, "Error: --ref must be non-negative\n");
        return 1;
    }
    if (n_alpha < 1 || n_beta < 1 || n_gamma < 1) {
        fprintf(stderr, "Error: --orient counts must be positive\n");
        return 1;
    }
    if (orient_start < 0) {
        fprintf(stderr, "Error: --orient-start must be non-negative\n");
        return 1;
    }
    if (orient_count < -1) {
        fprintf(stderr, "Error: --orient-count must be positive, or omitted for all remaining orientations\n");
        return 1;
    }
    if (ntheta < 2) {
        fprintf(stderr, "Error: --ntheta must be at least 2\n");
        return 1;
    }
    if (fmm_digits < 1) {
        fprintf(stderr, "Error: --fmm-digits must be positive\n");
        return 1;
    }
    if (gmres_tol <= 0.0) {
        fprintf(stderr, "Error: --gmres-tol must be positive\n");
        return 1;
    }
    if (gmres_restart < 1) {
        fprintf(stderr, "Error: --gmres-restart must be positive\n");
        return 1;
    }
    if (gmres_max_cycles_set && gmres_max_cycles_cli < 1) {
        fprintf(stderr, "Error: --gmres-max-cycles must be positive\n");
        return 1;
    }
    if (max_leaf < 1) {
        fprintf(stderr, "Error: --max-leaf must be positive\n");
        return 1;
    }
    if (strcmp(krylov_kind, "bcgstab") == 0)
        krylov_kind = "bicgstab";
    if (strcmp(krylov_kind, "bcgstab-rr") == 0 ||
        strcmp(krylov_kind, "bcgstab_rr") == 0 ||
        strcmp(krylov_kind, "bicgstab_rr") == 0)
        krylov_kind = "bicgstab-rr";
    if (strcmp(krylov_kind, "cgs") == 0 ||
        strcmp(krylov_kind, "cgs_rr") == 0)
        krylov_kind = "cgs-rr";
    if (strcmp(krylov_kind, "gpu-gmres") == 0 ||
        strcmp(krylov_kind, "gpu_gmres") == 0) {
        krylov_kind = "gmres";
        force_gpu_gmres = true;
    }
    if (strcmp(krylov_kind, "gpu-hybrid") == 0 ||
        strcmp(krylov_kind, "gpu_hybrid") == 0)
        krylov_kind = "hybrid";
    if (strcmp(krylov_kind, "gpu-adaptive") == 0 ||
        strcmp(krylov_kind, "gpu_adaptive") == 0) {
        krylov_kind = "hybrid";
        requested_gpu_adaptive = true;
    }
    if (strcmp(krylov_kind, "gpu-native") == 0 ||
        strcmp(krylov_kind, "gpu_native") == 0)
        krylov_kind = "hybrid";
    if (orient_file && orient_bg_file) {
        fprintf(stderr, "Error: use either --orient-file or --orient-bg-file, not both\n");
        return 1;
    }
    if (orient_file && alpha_avg != 1) {
        fprintf(stderr, "Error: --orient-file contains explicit alpha/gamma; do not combine it with --alpha-avg\n");
        return 1;
    }
    if (orient_bg_file) {
        n_alpha = 1;
    }
    if (alpha_avg < 1) {
        fprintf(stderr, "Error: --alpha-avg must be positive\n");
        return 1;
    }
    if (alpha_avg > 1 && !orient_bg_file && n_alpha != 1) {
        fprintf(stderr, "Error: --alpha-avg N averages alpha in far-field only; run with --orient 1 NB NG or --orient-bg-file FILE\n");
        return 1;
    }
    if ((orient_split_dir == nullptr) != (orient_split_indices_file == nullptr)) {
        fprintf(stderr, "Error: use --orient-split-dir and --orient-split-indices together\n");
        return 1;
    }
    if (orient_split_total == 0 || orient_split_total < -1) {
        fprintf(stderr, "Error: --orient-split-total must be positive when provided\n");
        return 1;
    }
    bool scattering_plane_yz = true;
    if (strcmp(scat_plane, "yz") == 0) {
        scattering_plane_yz = true;
    } else if (strcmp(scat_plane, "xz") == 0) {
        scattering_plane_yz = false;
    } else {
        fprintf(stderr, "Error: --scat-plane must be yz or xz\n");
        return 1;
    }
    bool is_obj = (strcmp(shape, "obj") == 0);
    bool is_hex_prism = (strcmp(shape, "hex_prism") == 0 || strcmp(shape, "prism6") == 0);
    bool is_sphere = (strcmp(shape, "sphere") == 0);
    if (is_hex_prism && adda_compare_mode && !prism_aspect_set) {
        prism_aspect = 1.5;
        printf("  [ADDA compare] Hex prism aspect defaulted to h/Dx=1.5; pass --prism-aspect to override.\n");
    }
    if (is_obj && fast_obj_mode && accurate_mode) {
        fprintf(stderr, "Error: --fast-obj conflicts with --accurate\n");
        return 1;
    }
    if (is_obj && !fast_obj_mode)
        accurate_mode = true;
    std::complex<double> m_probe(n_re, n_im);
    if (!system_kind_set) {
        BemSystemKind auto_system = choose_default_bem_system(
            m_probe, !bem_env_flag_enabled("BEM_NO_AUTO_BALANCED"),
            is_obj && accurate_mode);
        system_kind = bem_system_kind_name(auto_system);
    }

    if (fmm_test) {
        // Standalone FMM accuracy test: random charges, FMM vs brute-force
        printf("=== FMM Accuracy Test ===\n");
        cdouble k_test(ka, 0.0);
        int Npts = 500;
        std::vector<double> pts(Npts * 3);
        srand(42);
        for (int i = 0; i < Npts * 3; i++)
            pts[i] = 2.0 * ((double)rand() / RAND_MAX - 0.5);

        std::vector<cdouble> charges(Npts);
        for (int i = 0; i < Npts; i++)
            charges[i] = cdouble((double)rand()/RAND_MAX - 0.5,
                                 (double)rand()/RAND_MAX - 0.5);

        printf("  k = %.4f, N = %d, digits = %d, max_leaf = %d\n",
               ka, Npts, fmm_digits, max_leaf);

        HelmholtzFMM fmm;
        fmm.init(pts.data(), Npts, pts.data(), Npts, k_test, fmm_digits, max_leaf);

        std::vector<cdouble> result_fmm(Npts);
        fmm.evaluate(charges.data(), result_fmm.data());

        // Brute-force for first 20 targets
        int n_check = std::min(20, Npts);
        double max_rel_err = 0, avg_rel_err = 0;
        double inv4pi = 1.0 / (4.0 * M_PI);
        for (int i = 0; i < n_check; i++) {
            cdouble exact(0);
            for (int j = 0; j < Npts; j++) {
                double dx = pts[i*3] - pts[j*3];
                double dy = pts[i*3+1] - pts[j*3+1];
                double dz = pts[i*3+2] - pts[j*3+2];
                double R = sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-12) continue;
                cdouble G = std::exp(cdouble(0,1) * k_test * R) * inv4pi / R;
                exact += G * charges[j];
            }
            double err = std::abs(result_fmm[i] - exact);
            double rel = err / std::abs(exact);
            if (rel > max_rel_err) max_rel_err = rel;
            avg_rel_err += rel;
            printf("  [%3d] FMM=(%.6e,%.6e) exact=(%.6e,%.6e) rel_err=%.3e\n",
                   i, result_fmm[i].real(), result_fmm[i].imag(),
                   exact.real(), exact.imag(), rel);
        }
        avg_rel_err /= n_check;
        printf("  Max relative error: %.3e\n", max_rel_err);
        printf("  Avg relative error: %.3e\n", avg_rel_err);

        // Also test gradient
        printf("\n  Testing gradient...\n");
        std::vector<cdouble> grad_fmm(Npts * 3);
        fmm.evaluate_gradient(charges.data(), grad_fmm.data());

        max_rel_err = 0; avg_rel_err = 0;
        for (int i = 0; i < n_check; i++) {
            cdouble exact_gx(0), exact_gy(0), exact_gz(0);
            for (int j = 0; j < Npts; j++) {
                double dx = pts[i*3] - pts[j*3];
                double dy = pts[i*3+1] - pts[j*3+1];
                double dz = pts[i*3+2] - pts[j*3+2];
                double R = sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-12) continue;
                cdouble G = std::exp(cdouble(0,1) * k_test * R) * inv4pi / R;
                cdouble factor = G * (cdouble(0,1) * k_test - 1.0/R) / R;
                exact_gx += factor * dx * charges[j];
                exact_gy += factor * dy * charges[j];
                exact_gz += factor * dz * charges[j];
            }
            double norm_exact = std::sqrt(std::norm(exact_gx) + std::norm(exact_gy) + std::norm(exact_gz));
            double norm_err = std::sqrt(
                std::norm(grad_fmm[i*3] - exact_gx) +
                std::norm(grad_fmm[i*3+1] - exact_gy) +
                std::norm(grad_fmm[i*3+2] - exact_gz));
            double rel = norm_err / norm_exact;
            if (rel > max_rel_err) max_rel_err = rel;
            avg_rel_err += rel;
            if (i < 5) printf("  [%3d] grad rel_err=%.3e\n", i, rel);
        }
        avg_rel_err /= n_check;
        printf("  Grad max relative error: %.3e\n", max_rel_err);
        printf("  Grad avg relative error: %.3e\n", avg_rel_err);

        fmm.cleanup();
        return 0;
    }

    Timer total_timer;

    // Physical parameters
    std::complex<double> m(n_re, n_im);
    std::complex<double> k_ext(ka, 0.0);
    std::complex<double> k_int = k_ext * m;
    double eta_ext = 1.0;
    std::complex<double> eta_ext_c(eta_ext, 0.0);
    std::complex<double> eta_int_c = 1.0 / m;
    const char* requested_system_kind = system_kind;
    BemSystemKind requested_system_enum = BemSystemKind::PMCHWT;
    parse_bem_system_kind(system_kind, requested_system_enum);
    const bool requested_n_form =
        (requested_system_enum == BemSystemKind::Muller2 ||
         requested_system_enum == BemSystemKind::Muller2Balanced);
    const bool enable_experimental_n_form =
        requested_n_form && bem_env_flag_enabled("BEM_EXPERIMENTAL_NFORM");
    BemSystemKind actual_system_enum =
        canonicalize_bem_system_kind(requested_system_enum,
                                     enable_experimental_n_form);
    system_kind = bem_system_kind_name(actual_system_enum);
    BemBlockScales block_scales =
        bem_block_scales_for_system(actual_system_enum, m, eta_int_c,
                                    enable_experimental_n_form);
    double unknown_m_scale = block_scales.unknown_m_scale;
    std::complex<double> row_h_scale = block_scales.row_h_scale;
    double int_op_sign = block_scales.int_op_sign;
    double k_identity = block_scales.k_identity;
    bool n_form = block_scales.n_form;
    double n_form_eps_int = block_scales.n_form_eps_int;
    double n_form_m_identity = block_scales.n_form_m_identity;
    int_op_sign = bem_env_double("BEM_SYSTEM_INT_SIGN", int_op_sign);
    k_identity = bem_env_double("BEM_SYSTEM_K_IDENTITY", k_identity);
    unknown_m_scale = bem_env_double("BEM_SYSTEM_M_SCALE", unknown_m_scale);
    row_h_scale = std::complex<double>(
        bem_env_double("BEM_SYSTEM_H_ROW_SCALE", row_h_scale.real()),
        bem_env_double("BEM_SYSTEM_H_ROW_SCALE_IMAG", row_h_scale.imag()));
    n_form_eps_int = bem_env_double("BEM_SYSTEM_NFORM_EPS_INT", n_form_eps_int);
    n_form_m_identity = bem_env_double("BEM_SYSTEM_NFORM_M_IDENTITY", n_form_m_identity);

    printf("=== BEM-CUDA Solver ===\n");
    printf("  ka = %.4f, m = %.4f + %.4fi\n", ka, n_re, n_im);
    printf("  k_ext = %.4f, k_int = %.4f + %.4fi\n",
           k_ext.real(), k_int.real(), k_int.imag());
    printf("  eta_ext = %.4f, eta_int = %.6f %+.6fi\n",
           eta_ext, eta_int_c.real(), eta_int_c.imag());
    printf("  System: %s", system_kind);
    if (requested_n_form && !enable_experimental_n_form)
        printf(" (requested %s; canonicalized to %s because experimental N-form is disabled)",
               requested_system_kind, system_kind);
    if (unknown_m_scale != 1.0 || std::abs(row_h_scale - std::complex<double>(1.0, 0.0)) > 0.0)
        printf(" (M_scaled=%.4g*M, H-row scale=%.6g%+.6gi)",
               unknown_m_scale, row_h_scale.real(), row_h_scale.imag());
    if (int_op_sign < 0.0)
        printf(" (interior operator sign=-1)");
    if (k_identity != 0.0)
        printf(" (K identity jump=%.3g)", k_identity);
    if (n_form)
        printf(" (N-form RHS, eps_int=%.4g, M jump=%.4g)", n_form_eps_int, n_form_m_identity);
    printf("\n");
    if (is_obj && !obj_file) {
        fprintf(stderr, "Error: --shape obj requires --obj FILE\n");
        return 1;
    }
    if (quad_order != 4 && quad_order != 7 && quad_order != 13) {
        fprintf(stderr, "Error: unsupported --quad %d; supported values are 4, 7, 13\n",
                quad_order);
        return 1;
    }
    if (is_hex_prism && (adda_compare_mode || accurate_mode) && !refinements_set)
        refinements = hex_auto_refinement(ka);
    const bool hex_guarded_accuracy =
        is_hex_prism && adda_compare_mode &&
        hex_needs_guarded_accuracy(ka, refinements) &&
        !bem_env_flag_enabled("BEM_HEX_UNSAFE_FAST");
    if ((is_hex_prism || is_obj) && accurate_mode && !quad_order_set)
        quad_order = 7;
    else if (hex_guarded_accuracy && !quad_order_set)
        quad_order = 7;
    else if (is_hex_prism && !quad_order_set)
        quad_order = 4;
    if (is_hex_prism && edge_refine < 0) {
        // The current sharp-edge refinement is useful for diagnostics, but it
        // can create skinny transition triangles on coarse prism meshes. Keep
        // it opt-in until the conforming edge scheme has a quality-preserving
        // transition layer.
        edge_refine = 0;
    }
    if (!is_hex_prism)
        edge_refine = 0;
    printf("  Shape: %s", shape);
    if (strcmp(shape, "hex_prism") == 0)
        printf(" (h/Dx=%.3f, edge_refine=%d)", prism_aspect, edge_refine);
    if (is_obj)
        printf(" (%s, subdiv=%d)", obj_file, obj_subdiv);
    printf("\n");
    printf("  Refinements: %d, Quad order: %d\n", is_obj ? obj_subdiv : refinements, quad_order);
    if (hex_guarded_accuracy)
        printf("  Hex guarded accuracy: enabled for ka=%.4g/ref%d (set BEM_HEX_UNSAFE_FAST=1 only for speed diagnostics)\n",
               ka, refinements);
    if (is_obj && accurate_mode)
        printf("  OBJ accurate profile: quad%d, controlled FMM/GMRES defaults%s\n",
               quad_order, system_kind_set ? "" : ", system=balanced");
    if (single_orient)
        printf("  Single orientation, scattering plane: %s\n", scattering_plane_yz ? "yz" : "xz");
    else {
        if (orient_file)
            printf("  Orientations: explicit file %s\n", orient_file);
        else if (orient_bg_file)
            printf("  Orientations: explicit beta/gamma file %s\n", orient_bg_file);
        else
            printf("  Orientations: %d x %d x %d = %d\n",
                   n_alpha, n_beta, n_gamma, n_alpha * n_beta * n_gamma);
        if (alpha_avg > 1)
            printf("  Alpha far-field average: %d samples (no extra GMRES solves)\n", alpha_avg);
    }

    // 1. Generate mesh
    Timer mesh_timer;
    double radius = 1.0;
    Mesh mesh;
    if (strcmp(shape, "sphere") == 0) {
        mesh = icosphere(radius, refinements);
    } else if (strcmp(shape, "hex_prism") == 0 || strcmp(shape, "prism6") == 0) {
        mesh = regular_prism(6, prism_aspect, refinements, radius, edge_refine);
    } else if (is_obj) {
        mesh = load_obj(obj_file);
        double a_eq0 = normalize_mesh(mesh);
        for (int s = 0; s < obj_subdiv; s++)
            mesh = subdivide_flat(mesh);
        printf("  OBJ normalized: a_eq=%.6g, Dmax=%.4f\n", a_eq0, mesh_dmax(mesh));
        refinements = obj_subdiv;
    } else {
        fprintf(stderr, "Error: unsupported --shape %s\n", shape);
        return 1;
    }
    printf("  Mesh: %d vertices, %d triangles (%.1fms)\n",
           mesh.nv(), mesh.nt(), mesh_timer.elapsed_ms());
    MeshQualityReport mesh_quality = analyze_mesh_quality(mesh);
    print_mesh_quality_report(mesh_quality);
    if (is_obj && !fast_obj_mode && !quad_order_set &&
        mesh_quality.recommended_min_quad_order > quad_order) {
        if (mesh_quality.recommended_min_quad_order <= 7)
            quad_order = 7;
        else
            quad_order = 13;
        printf("  OBJ mesh guard: raised quadrature to quad%d from mesh-quality recommendation (%s)\n",
               quad_order, mesh_quality.recommended_mesh_strategy.c_str());
    }
    if (mesh_quality_report_file) {
        if (!write_mesh_quality_json(mesh_quality_report_file, mesh_quality, shape, ka,
                                     is_obj ? obj_subdiv : refinements, quad_order)) {
            fprintf(stderr, "Error: cannot write mesh quality report: %s\n",
                    mesh_quality_report_file);
            return 1;
        }
        printf("  Mesh quality report: %s\n", mesh_quality_report_file);
    }
    if (mesh_quality_only)
        return mesh_quality.pass_default_gate ? 0 : 2;
    if (mesh_quality_strict && !mesh_quality.pass_default_gate) {
        fprintf(stderr,
                "Error: mesh quality gate failed; rerun without --mesh-quality-strict only for diagnostics.\n");
        return 2;
    }
    bool sphere_orientation_shortcut = (!single_orient && !force_orient && strcmp(shape, "sphere") == 0);
    if (sphere_orientation_shortcut)
        printf("  Sphere orientation shortcut: using one solve for exact sphere average (--force-orient to disable)\n");

    // 2. Build RWG basis
    Timer rwg_timer;
    RWG rwg = build_rwg(mesh);
    printf("  RWG: %d basis functions (%.1fms)\n", rwg.N, rwg_timer.elapsed_ms());

    int N = rwg.N;
    int N2 = 2 * N;

    if (solver == SOLVER_AUTO) {
        if (single_orient && N < 384) {
            solver = SOLVER_DENSE;
        } else {
            solver = SOLVER_FMM;
        }
        printf("  [Auto] Solver selected: %s%s\n",
               solver_name(solver),
               solver_explicit ? " (explicit auto)" : "");
    }

    if (solver == SOLVER_SPFFT && !is_hex_prism) {
        fprintf(stderr, "Error: --solver spfft is only valid for hex_prism/prism6 flat-face meshes\n");
        return 1;
    }
    if (solver == SOLVER_SPFFT && !bem_env_flag_enabled("BEM_SPFFT_FORCE")) {
        printf("  [SurfPFFT] Disabled: cross-face direct P2P is slower than FMM on current meshes; using FMM backend\n");
        solver = SOLVER_FMM;
    }

    use_pfft = (solver == SOLVER_PFFT);
    use_spfft = (solver == SOLVER_SPFFT);
    use_fmm = (solver == SOLVER_FMM || solver == SOLVER_PFFT || solver == SOLVER_SPFFT);
    if (n_form && !use_fmm) {
        fprintf(stderr, "Error: %s is implemented only for FMM/GMRES path\n", system_kind);
        return 1;
    }

    const int requested_fmm_digits = fmm_digits;
    const double requested_gmres_tol = gmres_tol;
    SolverAccuracyInput acc_in;
    acc_in.use_fmm = use_fmm;
    acc_in.fmm_backend = (solver == SOLVER_FMM);
    acc_in.sphere = is_sphere;
    acc_in.hex_prism = is_hex_prism;
    acc_in.obj = is_obj;
    acc_in.obj_fast = fast_obj_mode;
    acc_in.accurate = accurate_mode;
    acc_in.adda_compare = adda_compare_mode;
    acc_in.hex_unsafe_fast = bem_env_flag_enabled("BEM_HEX_UNSAFE_FAST");
    acc_in.fmm_digits_set = fmm_digits_set;
    acc_in.max_leaf_set = max_leaf_set;
    acc_in.gmres_tol_set = gmres_tol_set;
    acc_in.gmres_restart_set = gmres_restart_set;
    acc_in.refinements = refinements;
    acc_in.mesh_requires_remesh = mesh_quality.requires_remesh;
    acc_in.mesh_recommended_min_quad_order = mesh_quality.recommended_min_quad_order;
    acc_in.mesh_min_angle_deg = mesh_quality.min_angle_deg;
    acc_in.mesh_max_aspect_ratio = mesh_quality.max_aspect_ratio;
    acc_in.ka = ka;
    acc_in.fmm_digits = fmm_digits;
    acc_in.max_leaf = max_leaf;
    acc_in.gmres_restart = gmres_restart;
    acc_in.gmres_tol = gmres_tol;
    SolverAccuracyPolicy acc_policy = choose_solver_accuracy_policy(acc_in);
    fmm_digits = acc_policy.fmm_digits;
    max_leaf = acc_policy.max_leaf;
    gmres_restart = acc_policy.gmres_restart;
    gmres_tol = acc_policy.gmres_tol;
    if (use_fmm && is_obj && accurate_mode && !fast_obj_mode &&
        gmres_tol_set && gmres_tol > 5e-5 &&
        !bem_env_flag_enabled("BEM_ALLOW_LOOSE_OBJ_GMRES")) {
        printf("  [Accuracy guard] OBJ --accurate with loose --gmres-tol %.1e is not reliable "
               "for Mueller amplitudes; using 5e-5. Set --fast-obj or "
               "BEM_ALLOW_LOOSE_OBJ_GMRES=1 for diagnostics.\n", gmres_tol);
        gmres_tol = 5e-5;
    }
    if (use_fmm && (is_hex_prism || is_obj) &&
        !bem_env_flag_present("BEM_GMRES_REORTH") &&
        bem_env_flag_enabled("BEM_FAST_REORTH_OFF", false))
        setenv("BEM_GMRES_REORTH", "0", 0);
    const bool accuracy_policy_adjusted =
        fmm_digits != requested_fmm_digits || gmres_tol != requested_gmres_tol;
    int batch4_max_n = 120000;
    if (bem_env_has_value("BEM_FMM_BATCH4_MAX_N")) {
        batch4_max_n = std::max(0, bem_env_int("BEM_FMM_BATCH4_MAX_N", batch4_max_n));
    } else if (use_fmm && solver == SOLVER_FMM && !bem_env_flag_enabled("BEM_FMM_NO_BATCH4")) {
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            double free_gb = (double)free_bytes / (1024.0 * 1024.0 * 1024.0);
            if (free_gb >= 24.0)
                batch4_max_n = 400000;
            else if (free_gb >= 12.0) {
                batch4_max_n = 220000;
                if (is_hex_prism && accurate_mode && fmm_digits >= 5)
                    batch4_max_n = 70000;
            }
        } else {
            cudaGetLastError();
        }
    }
    if (use_fmm && solver == SOLVER_FMM && N < batch4_max_n &&
        !bem_env_flag_present("BEM_FMM_BATCH4") && !bem_env_flag_enabled("BEM_FMM_NO_BATCH4"))
        setenv("BEM_FMM_BATCH4", "1", 0);
    if (use_fmm && bem_env_flag_enabled("BEM_FMM_BATCH4"))
        setenv("BEM_FMM_ALLOC_BATCH4", "1", 0);

    if (use_fmm && force_gpu_gmres)
        setenv("BEM_GMRES_DEVICE", "1", 1);
    if (use_fmm && krylov_kind_set && !force_gpu_gmres)
        setenv("BEM_KRYLOV", krylov_kind, 1);
    const char* krylov_env = std::getenv("BEM_KRYLOV");
    const bool use_bicgstab =
        use_fmm && krylov_env &&
        (strcmp(krylov_env, "bicgstab") == 0 ||
         strcmp(krylov_env, "bcgstab") == 0 ||
         strcmp(krylov_env, "bicgstab-rr") == 0 ||
         strcmp(krylov_env, "bicgstab_rr") == 0 ||
         strcmp(krylov_env, "bcgstab-rr") == 0 ||
         strcmp(krylov_env, "bcgstab_rr") == 0 ||
         strcmp(krylov_env, "BiCGSTAB") == 0);
    const bool use_bicgstab_rr =
        use_fmm && krylov_env &&
        (strcmp(krylov_env, "bicgstab-rr") == 0 ||
         strcmp(krylov_env, "bicgstab_rr") == 0 ||
         strcmp(krylov_env, "bcgstab-rr") == 0 ||
         strcmp(krylov_env, "bcgstab_rr") == 0);
    const bool use_krylov_auto =
        use_fmm && krylov_env && strcmp(krylov_env, "auto") == 0;
    const bool use_krylov_hybrid =
        use_fmm && krylov_env && strcmp(krylov_env, "hybrid") == 0;
    const bool use_cgs_rr =
        use_fmm && krylov_env &&
        (strcmp(krylov_env, "cgs-rr") == 0 ||
         strcmp(krylov_env, "cgs_rr") == 0 ||
         strcmp(krylov_env, "cgs") == 0);
    if (use_bicgstab || use_cgs_rr || use_krylov_auto || use_krylov_hybrid)
        no_prec = true;
    const bool use_gpu_gmres =
        use_fmm && !use_bicgstab && !use_cgs_rr && !use_krylov_auto && !use_krylov_hybrid &&
        (force_gpu_gmres || bem_env_flag_enabled("BEM_GMRES_DEVICE"));
    const char* output_krylov_solver = use_krylov_auto ? "auto_best_short_recurrence_gmres_gpu" :
                                       (use_krylov_hybrid ?
                                        (requested_gpu_adaptive ? "gpu_adaptive_short_recurrence_gmres" :
                                         "gpu_native_short_recurrence_gmres") :
                                       (use_cgs_rr ? "cgs_rr_gpu" :
                                       (use_bicgstab_rr ? "bicgstab_rr_gpu" :
                                        (use_bicgstab ? "bicgstab_gpu" :
                                         (use_gpu_gmres ? "gmres_gpu_requested" : "gmres")))));

    PrecondPolicyInput prec_in;
    prec_in.use_fmm = use_fmm;
    prec_in.user_disabled = no_prec;
    prec_in.force = bem_env_flag_enabled("BEM_PREC_FORCE");
    prec_in.pfft_backend = (solver == SOLVER_PFFT);
    prec_in.sphere = is_sphere;
    prec_in.hex_prism = is_hex_prism;
    prec_in.obj_mesh = is_obj;
    prec_in.mesh_requires_remesh = mesh_quality.requires_remesh;
    prec_in.n_form = n_form;
    prec_in.strict_accuracy = accurate_mode || gmres_tol <= 1e-3;
    prec_in.basis_count = N;
    prec_in.ka = ka;
    prec_in.gmres_tol = gmres_tol;
    PrecondPolicy prec_policy = choose_precond_policy(prec_in);
    use_prec = prec_policy.enabled;

    if (use_fmm && use_prec && !bem_env_flag_present("BEM_PREC_BLOCK")) {
        setenv("BEM_PREC_BLOCK", prec_policy.schwarz ? "1" : "0", 0);
    }
    const bool strict_obj_prec = is_obj && gmres_tol <= 5e-4;
    if (use_fmm && use_prec && bem_env_flag_enabled("BEM_PREC_BLOCK") && !bem_env_has_value("BEM_PREC_BLOCK_SIZE"))
        setenv("BEM_PREC_BLOCK_SIZE", strict_obj_prec ? "10" : (is_obj ? "6" : (is_hex_prism ? "8" : "4")), 0);
    if (use_fmm && use_prec && bem_env_flag_enabled("BEM_PREC_BLOCK") && !bem_env_has_value("BEM_PREC_SWEEPS"))
        setenv("BEM_PREC_SWEEPS", strict_obj_prec ? "2" : ((is_obj || is_hex_prism) ? "1" : "2"), 0);
    if (use_fmm && use_prec && bem_env_flag_enabled("BEM_PREC_BLOCK") && !bem_env_has_value("BEM_PREC_NEAR"))
        setenv("BEM_PREC_NEAR", strict_obj_prec ? "16" : (is_obj ? "10" : (is_hex_prism ? "12" : "6")), 0);
    if (use_fmm && use_prec && bem_env_flag_enabled("BEM_PREC_BLOCK") && !bem_env_has_value("BEM_PREC_OMEGA"))
        setenv("BEM_PREC_OMEGA", "1.0", 0);
    if (use_fmm && use_prec &&
        !bem_env_flag_present("BEM_GMRES_STORE_Z") &&
        !bem_env_flag_enabled("BEM_GMRES_NO_STORE_Z")) {
        double store_z_max_mb = 512.0;
        store_z_max_mb = std::max(0.0, bem_env_double("BEM_GMRES_STORE_Z_MAX_MB", store_z_max_mb));
        const double n_system = 2.0 * (double)N;
        const double store_z_mb = 2.0 * n_system * (double)gmres_restart *
                                  (double)sizeof(cdouble) / (1024.0 * 1024.0);
        if (store_z_mb <= store_z_max_mb)
            setenv("BEM_GMRES_STORE_Z", "1", 0);
    }
    int gmres_max_cycles = gmres_max_cycles_set ? gmres_max_cycles_cli : acc_policy.gmres_max_cycles;
    if (bem_env_has_value("BEM_GMRES_MAX_CYCLES"))
        gmres_max_cycles = std::max(1, bem_env_int("BEM_GMRES_MAX_CYCLES", gmres_max_cycles));
    if (use_fmm && acc_policy.gmres_stagnation_cycles > 0 &&
        !bem_env_has_value("BEM_GMRES_STAGNATION_CYCLES")) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%d", acc_policy.gmres_stagnation_cycles);
        setenv("BEM_GMRES_STAGNATION_CYCLES", buf, 0);
    }
    if (use_fmm && acc_policy.gmres_stagnation_rel > 0.0 &&
        !bem_env_has_value("BEM_GMRES_STAGNATION_REL")) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", acc_policy.gmres_stagnation_rel);
        setenv("BEM_GMRES_STAGNATION_REL", buf, 0);
    }
    if (use_fmm && acc_policy.gmres_inner_stagnation_window > 0 &&
        !bem_env_has_value("BEM_GMRES_INNER_STAGNATION_WINDOW")) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%d", acc_policy.gmres_inner_stagnation_window);
        setenv("BEM_GMRES_INNER_STAGNATION_WINDOW", buf, 0);
    }
    if (use_fmm && acc_policy.gmres_inner_stagnation_rel > 0.0 &&
        !bem_env_has_value("BEM_GMRES_INNER_STAGNATION_REL")) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", acc_policy.gmres_inner_stagnation_rel);
        setenv("BEM_GMRES_INNER_STAGNATION_REL", buf, 0);
    }
    if (use_fmm && acc_policy.gmres_inner_stagnation_min_iter > 0 &&
        !bem_env_has_value("BEM_GMRES_INNER_STAGNATION_MIN_ITER")) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%d", acc_policy.gmres_inner_stagnation_min_iter);
        setenv("BEM_GMRES_INNER_STAGNATION_MIN_ITER", buf, 0);
    }
    const bool coeffs_farfield_only =
        use_fmm && bem_env_has_value("BEM_LOAD_COEFFS_FARFIELD_JSON");

    if (use_fmm) {
        const bool schwarz_prec = use_prec && bem_env_flag_enabled("BEM_PREC_BLOCK");
        const bool batch4_active = bem_env_flag_enabled("BEM_FMM_BATCH4");
        const bool store_z_active = use_prec && bem_env_flag_enabled("BEM_GMRES_STORE_Z");
        const char* prec_name = use_prec ? (schwarz_prec ? ", Schwarz prec" : ", block-Jacobi prec") : "";
        const char* batch4_name = batch4_active ? ", batch4" : "";
        const char* store_z_name = store_z_active ? ", store-Z" : "";
        const char* krylov_name = use_krylov_auto ? "Auto-best-GPU-Krylov" :
                                  (use_krylov_hybrid ?
                                   (requested_gpu_adaptive ? "GPU-adaptive Krylov" :
                                    "GPU-native Krylov") :
                                  (use_cgs_rr ? "CGS-RR-GPU" :
                                  (use_bicgstab_rr ? "BiCGSTAB-RR-GPU" :
                                   (use_bicgstab ? "BiCGSTAB-GPU" :
                                    (use_gpu_gmres ? "GMRES (GPU requested)" : "GMRES")))));
        printf("  Mode: %s+%s (profile=%s, digits=%d, tol=%.0e, restart=%d, cycles=%d, max_leaf=%d%s%s%s)\n",
               solver_name(solver), krylov_name, acc_policy.profile, fmm_digits, gmres_tol, gmres_restart, gmres_max_cycles, max_leaf,
               prec_name, batch4_name, store_z_name);
    } else {
        printf("  Mode: Dense LU\n");
    }

    // 4. Scattering angles
    std::vector<double> theta_arr(ntheta);
    for (int i = 0; i < ntheta; i++)
        theta_arr[i] = M_PI * i / (ntheta - 1);

    // 5. Precompute far-field quadrature cache (once!)
    FFCache ff_cache;
    ff_cache.init(rwg, mesh, quad_order);

    // Mueller matrix accumulator: [16 * ntheta]
    std::vector<double> M_avg(16 * ntheta, 0.0);
    int output_orient_start = single_orient ? 0 : orient_start;
    int output_orient_count = single_orient ? 1 : 1;
    int output_orient_total = single_orient ? 1 : 1;
    double output_orientation_weight_sum = 1.0;
    long long output_gmres_matvecs = 0;
    int output_gmres_converged_systems = 0;
    int output_gmres_nonconverged_systems = 0;
    int output_gmres_stagnation_stops = 0;
    int output_gmres_numerical_breakdowns = 0;
    int output_gmres_restored_best_iterates = 0;
    int output_gmres_max_cycle_exhaustions = 0;
    double output_gmres_max_final_relres = 0.0;
    const char* output_farfield_mode = "not_started";
    bool currents_exported = false;
    bool gmres_verbose_orient = bem_env_flag_enabled("BEM_GMRES_VERBOSE");
    if (gmres_verbose_orient)
        printf("  [GMRES] verbose residual logging enabled\n");
    int nform_ff_mode = 0;
    nform_ff_mode = bem_env_int("BEM_NFORM_FF_MODE", nform_ff_mode);
    if (n_form && nform_ff_mode != 0)
        printf("  [N-form] Far-field recovery mode: %d\n", nform_ff_mode);
    int nform_rhs_mode = 0;
    nform_rhs_mode = bem_env_int("BEM_NFORM_RHS_MODE", nform_rhs_mode);
    if (n_form && nform_rhs_mode != 0)
        printf("  [N-form] RHS transform mode: %d\n", nform_rhs_mode);

    double time_assembly = 0, time_solve = 0, time_farfield = 0;

    if (use_fmm) {
        // ============================================================
        // FMM + GMRES path
        // ============================================================
	    Timer asm_timer;
	    BemFmmOperator fmm_op;
	    fmm_op.unknown_m_scale = unknown_m_scale;
	    fmm_op.row_h_scale = row_h_scale;
	    fmm_op.int_op_sign = int_op_sign;
	    fmm_op.k_identity = k_identity;
	    fmm_op.n_form = n_form;
	    fmm_op.n_form_eps_int = n_form_eps_int;
	    fmm_op.n_form_m_identity = n_form_m_identity;
        if (coeffs_farfield_only) {
            printf("  [CoeffFarfield] Skipping FMM assembly/residual check; "
                   "using saved coefficients for far-field only.\n");
        } else {
	        fmm_op.init(rwg, mesh, k_ext, k_int, eta_ext_c, eta_int_c,
	                     quad_order, fmm_digits, max_leaf, use_pfft, use_spfft);
        }

	    // Build preconditioner if requested
        NearFieldPrecond* precond_ptr = nullptr;
        NearFieldPrecond precond;
        if (use_prec && !coeffs_farfield_only) {
            precond.build(fmm_op);
            precond_ptr = &precond;
        }

        time_assembly = asm_timer.elapsed_s();

        auto solve_pair_with_prec_fallback = [&](const cdouble* b1, const cdouble* b2,
                                                 cdouble* x1, cdouble* x2,
                                                 GmresPairedWorkspace& ws) -> int {
            int mv = gmres_solve_paired_ws(fmm_op, b1, b2, x1, x2,
                                           gmres_restart, gmres_tol, gmres_max_cycles,
                                           gmres_verbose_orient, precond_ptr, ws);
            if ((ws.converged1 && ws.converged2) || precond_ptr == nullptr ||
                bem_env_flag_enabled("BEM_GMRES_NO_PREC_FALLBACK")) {
                return mv;
            }

            double first_rel = std::max(ws.final_relres1, ws.final_relres2);
            std::vector<cdouble> saved_x1(x1, x1 + N2);
            std::vector<cdouble> saved_x2(x2, x2 + N2);
            GmresPairedWorkspace retry_ws;
            printf("  [GMRES-paired] preconditioned solve did not converge "
                   "(max true rel=%.2e); retrying without preconditioner\n", first_rel);
            fflush(stdout);
            int mv_retry = gmres_solve_paired_ws(fmm_op, b1, b2, x1, x2,
                                                 gmres_restart, gmres_tol, gmres_max_cycles,
                                                 gmres_verbose_orient, nullptr, retry_ws);
            double retry_rel = std::max(retry_ws.final_relres1, retry_ws.final_relres2);
            if ((retry_ws.converged1 && retry_ws.converged2) || retry_rel < first_rel) {
                ws = retry_ws;
                return mv + mv_retry;
            }
            std::copy(saved_x1.begin(), saved_x1.end(), x1);
            std::copy(saved_x2.begin(), saved_x2.end(), x2);
            printf("  [GMRES-paired] keeping preconditioned result; "
                   "fallback max true rel=%.2e was not better\n", retry_rel);
            fflush(stdout);
            return mv + mv_retry;
        };

        Timer solve_timer;

        if (single_orient || sphere_orientation_shortcut) {
            Vec3 k_hat(0, 0, 1);
            Vec3 E_par  = scattering_plane_yz ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
            Vec3 E_perp = scattering_plane_yz ? Vec3(1, 0, 0) : Vec3(0, 1, 0);

            // Solve for both polarizations
            std::vector<cdouble> b_par(N2), b_perp(N2);
            if (!coeffs_farfield_only) {
                compute_rhs_planewave_pair_cached(ff_cache, k_ext, eta_ext, E_par, E_perp,
                                                 k_hat, b_par.data(), b_perp.data());
                if (n_form) {
                    transform_rhs_to_n_form(b_par.data(), N, nform_rhs_mode);
                    transform_rhs_to_n_form(b_perp.data(), N, nform_rhs_mode);
                }
                if (std::abs(row_h_scale - std::complex<double>(1.0, 0.0)) > 0.0) {
                    for (int i = 0; i < N; i++) {
                        b_par[N + i] *= row_h_scale;
                        b_perp[N + i] *= row_h_scale;
                    }
                }
            }

            std::vector<cdouble> x_par(N2, cdouble(0)), x_perp(N2, cdouble(0));
            if (use_fmm && bem_env_has_value("BEM_CHECK_COEFFS_JSON")) {
                const char* coeff_path = std::getenv("BEM_CHECK_COEFFS_JSON");
                if (!load_coefficients_json(coeff_path, N, x_par, x_perp))
                    return 1;
                double rel_par = relative_residual(fmm_op, x_par.data(), b_par.data());
                double rel_perp = relative_residual(fmm_op, x_perp.data(), b_perp.data());
                printf("  [CoeffCheck] %s: relres_par=%.6e relres_perp=%.6e max=%.6e\n",
                       coeff_path, rel_par, rel_perp, std::max(rel_par, rel_perp));
                fflush(stdout);
                if (bem_env_flag_enabled("BEM_CHECK_COEFFS_EXIT"))
                    return 0;
            }
            GmresPairedWorkspace single_gmres_ws;
            if (coeffs_farfield_only) {
                const char* coeff_path = std::getenv("BEM_LOAD_COEFFS_FARFIELD_JSON");
                if (!load_coefficients_json(coeff_path, N, x_par, x_perp))
                    return 1;
                single_gmres_ws.converged1 = true;
                single_gmres_ws.converged2 = true;
                printf("  [CoeffFarfield] single %s: residual check skipped\n", coeff_path);
                fflush(stdout);
            } else {
                printf("\n  Solving both polarizations (paired %s)...\n", output_krylov_solver);
                fflush(stdout);
                output_gmres_matvecs = solve_pair_with_prec_fallback(
                    b_par.data(), b_perp.data(),
                    x_par.data(), x_perp.data(),
                    single_gmres_ws);
            }
            output_gmres_converged_systems = (single_gmres_ws.converged1 ? 1 : 0) +
                                             (single_gmres_ws.converged2 ? 1 : 0);
            output_gmres_nonconverged_systems = 2 - output_gmres_converged_systems;
            output_gmres_stagnation_stops = single_gmres_ws.stopped_stagnant ? 1 : 0;
            output_gmres_numerical_breakdowns = single_gmres_ws.numerical_breakdown ? 1 : 0;
            output_gmres_restored_best_iterates = single_gmres_ws.restored_best_iterate ? 1 : 0;
            output_gmres_max_cycle_exhaustions = single_gmres_ws.reached_max_cycles ? 1 : 0;
            output_gmres_max_final_relres = std::max(single_gmres_ws.final_relres1,
                                                     single_gmres_ws.final_relres2);
            if (!coeffs_farfield_only && use_gpu_gmres &&
                !bem_env_flag_enabled("BEM_SKIP_HOST_RESIDUAL_CHECK")) {
                double host_rel_par = relative_residual(fmm_op, x_par.data(), b_par.data());
                double host_rel_perp = relative_residual(fmm_op, x_perp.data(), b_perp.data());
                double host_rel = std::max(host_rel_par, host_rel_perp);
                printf("  [PostSolveResidualCheck] relres_par=%.6e relres_perp=%.6e max=%.6e\n",
                       host_rel_par, host_rel_perp, host_rel);
                fflush(stdout);
                output_gmres_max_final_relres = std::max(output_gmres_max_final_relres, host_rel);
                if (host_rel > 2.0 * gmres_tol) {
                    output_gmres_converged_systems = 0;
                    output_gmres_nonconverged_systems = 2;
                    printf("  [PostSolveResidualCheck] device GMRES result rejected: residual exceeds %.3e\n",
                           2.0 * gmres_tol);
                    fflush(stdout);
                }
            }

            time_solve = solve_timer.elapsed_s();

            if (bem_env_has_value("BEM_EXPORT_COEFFS_JSON"))
                export_coefficients_json(std::getenv("BEM_EXPORT_COEFFS_JSON"),
                                         N, x_par.data(), x_perp.data());

            // Far-field
            Timer ff_timer;
            cdouble* J_par  = x_par.data();
            cdouble* M_par  = x_par.data() + N;
            cdouble* J_perp = x_perp.data();
            cdouble* M_perp = x_perp.data() + N;
            std::vector<cdouble> M_par_phys, M_perp_phys;
            if (unknown_m_scale != 1.0) {
                M_par_phys.resize(N);
                M_perp_phys.resize(N);
                double inv_s = 1.0 / unknown_m_scale;
                for (int i = 0; i < N; i++) {
                    M_par_phys[i] = M_par[i] * inv_s;
                    M_perp_phys[i] = M_perp[i] * inv_s;
                }
                M_par = M_par_phys.data();
                M_perp = M_perp_phys.data();
            }
            std::vector<cdouble> J_par_nform, M_par_nform;
            std::vector<cdouble> J_perp_nform, M_perp_nform;
            if (n_form) {
                recover_n_form_farfield_pair(J_par, M_par, N, nform_ff_mode,
                                             J_par_nform, M_par_nform);
                recover_n_form_farfield_pair(J_perp, M_perp, N, nform_ff_mode,
                                             J_perp_nform, M_perp_nform);
                J_par = J_par_nform.data();
                M_par = M_par_nform.data();
                J_perp = J_perp_nform.data();
                M_perp = M_perp_nform.data();
            }

            std::vector<cdouble> Fth_par(ntheta), Fph_par(ntheta);
            std::vector<cdouble> Fth_perp(ntheta), Fph_perp(ntheta);

            compute_far_field(ff_cache, J_par, M_par, k_ext, eta_ext,
                             theta_arr.data(), ntheta,
                             scattering_plane_yz,
                             Fth_par.data(), Fph_par.data());
            compute_far_field(ff_cache, J_perp, M_perp, k_ext, eta_ext,
                             theta_arr.data(), ntheta,
                             scattering_plane_yz,
                             Fth_perp.data(), Fph_perp.data());

            cdouble ik(0, -1);
            ik *= k_ext;
            std::vector<cdouble> S1(ntheta), S2(ntheta), S3(ntheta), S4(ntheta);
            for (int t = 0; t < ntheta; t++) {
                S2[t] = ik * Fth_par[t];
                S4[t] = ik * Fph_par[t];
                S3[t] = ik * Fth_perp[t];
                S1[t] = ik * Fph_perp[t];
            }

            amplitude_to_mueller(S1.data(), S2.data(), S3.data(), S4.data(),
                                ntheta, M_avg.data());
            output_farfield_mode = use_fmm ? "single_orientation_cpu_farfield_fmm_solve"
                                           : "single_orientation_cpu_farfield_dense_solve";

            if (export_currents_file) {
                currents_exported = export_currents_vtk(export_currents_file, mesh, rwg,
                                                        J_par, M_par, J_perp, M_perp);
                if (currents_exported)
                    printf("  Exported equivalent currents: %s\n", export_currents_file);
            }

            time_farfield = ff_timer.elapsed_s();
        } else {
            // Orientation averaging with GMRES
            std::vector<Orientation> orients;
            if (orient_file) {
                if (!load_orientation_file(orient_file, orients))
                    return 1;
            } else if (orient_bg_file) {
                if (!load_beta_gamma_orientation_file(orient_bg_file, orients))
                    return 1;
            } else {
                orients = generate_orientations(n_alpha, n_beta, n_gamma);
                reorder_orientations_nearest(orients);
            }
            int n_all = (int)orients.size();
            if (orient_start < 0) orient_start = 0;
            if (orient_start > n_all) orient_start = n_all;
            int n_keep = (orient_count < 0) ? (n_all - orient_start)
                                            : std::min(orient_count, n_all - orient_start);
            if (orient_start > 0 || n_keep < n_all) {
                orients = std::vector<Orientation>(orients.begin() + orient_start,
                                                   orients.begin() + orient_start + n_keep);
                printf("  Orientation chunk: start=%d count=%d of %d\n", orient_start, n_keep, n_all);
            }
            int n_total = (int)orients.size();
            output_orient_start = orient_start;
            output_orient_count = n_total;
            output_orient_total = n_all;
            output_orientation_weight_sum = 0.0;
            for (const auto& o : orients)
                output_orientation_weight_sum += o.weight;
            std::vector<int> split_indices;
            const bool split_orientation_outputs = orient_split_dir != nullptr;
            int split_output_total = n_all;
            if (split_orientation_outputs) {
                if (!load_int_index_file(orient_split_indices_file, split_indices))
                    return 1;
                if ((int)split_indices.size() != n_total) {
                    fprintf(stderr, "Error: --orient-split-indices has %zu entries, but chunk has %d orientations\n",
                            split_indices.size(), n_total);
                    return 1;
                }
                if (orient_split_total > 0) {
                    split_output_total = orient_split_total;
                } else {
                    for (int idx : split_indices)
                        split_output_total = std::max(split_output_total, idx + 1);
                }
                if (split_output_total < n_all) {
                    fprintf(stderr, "Error: --orient-split-total=%d is smaller than chunk orientation count %d\n",
                            split_output_total, n_all);
                    return 1;
                }
                output_orient_total = split_output_total;
            }

            // Far-field GPU cache
            FFCacheGPU ff_gpu;
            ff_gpu.upload(ff_cache);

	            // Lab-frame scattering vectors
	            std::vector<Vec3> r_hat_lab(ntheta), e_theta_lab(ntheta);
	            std::vector<double> r_hat_lab_flat((size_t)ntheta * 3);
	            std::vector<double> e_theta_lab_flat((size_t)ntheta * 3);
	            Vec3 e_phi_lab = scattering_plane_yz ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
	            double e_phi_lab_flat[3] = {e_phi_lab.x, e_phi_lab.y, e_phi_lab.z};
	            for (int it = 0; it < ntheta; it++) {
	                double ct = cos(theta_arr[it]), st = sin(theta_arr[it]);
	                if (scattering_plane_yz) {
	                    r_hat_lab[it] = Vec3(0, st, ct);
	                    e_theta_lab[it] = Vec3(0, ct, -st);
	                } else {
	                    r_hat_lab[it] = Vec3(st, 0, ct);
	                    e_theta_lab[it] = Vec3(ct, 0, -st);
	                }
	                r_hat_lab_flat[(size_t)it * 3]     = r_hat_lab[it].x;
	                r_hat_lab_flat[(size_t)it * 3 + 1] = r_hat_lab[it].y;
	                r_hat_lab_flat[(size_t)it * 3 + 2] = r_hat_lab[it].z;
	                e_theta_lab_flat[(size_t)it * 3]     = e_theta_lab[it].x;
	                e_theta_lab_flat[(size_t)it * 3 + 1] = e_theta_lab[it].y;
	                e_theta_lab_flat[(size_t)it * 3 + 2] = e_theta_lab[it].z;
	            }

            printf("\n  Solving %d orientations x 2 polarizations with GMRES...\n", n_total);
            int orient_progress_step = std::max(10, n_total / 10);
            orient_progress_step = std::max(1, bem_env_int("BEM_ORIENT_PROGRESS", orient_progress_step));

            std::vector<cdouble> x_par(N2, cdouble(0)), x_perp(N2, cdouble(0));
            std::vector<cdouble> b_par(N2), b_perp(N2);
            GmresPairedWorkspace gmres_ws;
            std::vector<std::vector<cdouble>> hist_b, hist_x;
            OrientWarmStart warm_start = parse_orient_warm_start();
            int max_recycle = 12;
            max_recycle = std::max(1, bem_env_int("BEM_ORIENT_RECYCLE", max_recycle));
            printf("  Orientation GMRES initial guess: %s", orient_warm_start_name(warm_start));
            if (warm_start == OrientWarmStart::Recycle)
                printf(" (history=%d)", max_recycle);
            printf(" (BEM_ORIENT_WARM_START=zero|previous|recycle)\n");
            long long orient_matvecs = 0;

            const bool ff_batch_set = bem_env_has_value("BEM_FF_BATCH");
            int ff_batch_orient = 64;
            if (ff_batch_set) {
                ff_batch_orient = std::max(1, bem_env_int("BEM_FF_BATCH", ff_batch_orient));
            } else {
                double target_mb = bem_env_double("BEM_FF_TARGET_MB", 512.0);
                int max_auto_batch = bem_env_int("BEM_FF_MAX_BATCH", 512);
                double per_orient_bytes = 64.0 * (double)N + 120.0 * (double)ntheta + 8.0;
                ff_batch_orient = (int)((target_mb * 1024.0 * 1024.0) / std::max(1.0, per_orient_bytes));
                ff_batch_orient = std::max(1, std::min(ff_batch_orient, std::max(1, max_auto_batch)));
            }
            long long n_farfield_samples = (long long)n_total * (long long)alpha_avg;
            ff_batch_orient = std::min(ff_batch_orient, std::max(1, (int)std::min<long long>(n_farfield_samples, INT_MAX)));
            bool ff_gpu_accum = !bem_env_flag_enabled("BEM_FF_CPU_ACCUM");
            bool ff_alpha_direct = ff_gpu_accum &&
                                   !bem_env_flag_enabled("BEM_FF_SEPARATE") &&
                                   !bem_env_flag_enabled("BEM_FF_NO_ALPHA_DIRECT") &&
                                   !bem_env_flag_enabled("BEM_FF_NO_ALPHA_GEOM");
            output_farfield_mode = ff_gpu_accum
                ? (ff_alpha_direct ? "gpu_geometry_direct" : "gpu_host_geometry_mueller_accum")
                : "gpu_farfield_cpu_mueller_accum";
            int ff_base_batch_orient = ff_batch_orient;
            if (ff_alpha_direct) {
                if (!ff_batch_set) {
                    double target_mb = bem_env_double("BEM_FF_TARGET_MB", 512.0);
                    int max_auto_base = bem_env_int("BEM_FF_MAX_BASE_BATCH", 4096);
                    double per_base_bytes = 64.0 * (double)N + 9.0 * sizeof(double) +
                                            (double)alpha_avg * sizeof(double);
                    ff_base_batch_orient = (int)((target_mb * 1024.0 * 1024.0) / std::max(1.0, per_base_bytes));
                    ff_base_batch_orient = std::max(1, std::min(ff_base_batch_orient, std::max(1, max_auto_base)));
                    ff_base_batch_orient = std::min(ff_base_batch_orient, std::max(1, n_total));
                } else {
                    ff_base_batch_orient = std::max(1, ff_batch_orient / alpha_avg);
                }
                ff_batch_orient = ff_base_batch_orient * alpha_avg;
            }
            printf("  Streaming far-field batch: %d orientations (BEM_FF_BATCH overrides; BEM_FF_TARGET_MB tunes auto)\n",
                   ff_batch_orient);
            if (ff_alpha_direct)
                printf("  FMM geometry-direct GPU far-field enabled: %d base orientations x %d alpha\n",
                       ff_base_batch_orient, alpha_avg);

            int batch_count = 0;
            int coeff_batch_orient = ff_alpha_direct ? ff_base_batch_orient : ff_batch_orient;
            PinnedHostBuffer<cdouble> batch_coeffs_J;
            PinnedHostBuffer<cdouble> batch_coeffs_M;
            batch_coeffs_J.resize((size_t)coeff_batch_orient * 2 * N);
            batch_coeffs_M.resize((size_t)coeff_batch_orient * 2 * N);
            PinnedHostBuffer<double> batch_r_hats;
            PinnedHostBuffer<Vec3> batch_e_par;
            PinnedHostBuffer<Vec3> batch_e_perp;
            if (!ff_alpha_direct) {
                batch_r_hats.resize((size_t)ff_batch_orient * ntheta * 3);
                batch_e_par.resize((size_t)ff_batch_orient * ntheta);
                batch_e_perp.resize((size_t)ff_batch_orient * ntheta);
            }
            PinnedHostBuffer<double> batch_RT;
            batch_RT.resize((size_t)ff_base_batch_orient * 9);
            FFBatchWorkspace ff_workspace;
            PinnedHostBuffer<double> batch_weights;
            batch_weights.resize(ff_alpha_direct ? ff_base_batch_orient : ff_batch_orient);
            PinnedHostBuffer<cdouble> batch_Fv;
            std::vector<cdouble> S1, S2, S3, S4;
            std::vector<double> M_orient;
            if (!ff_gpu_accum) {
                batch_Fv.resize((size_t)ff_batch_orient * 2 * ntheta * 3);
                S1.resize(ntheta);
                S2.resize(ntheta);
                S3.resize(ntheta);
                S4.resize(ntheta);
                M_orient.resize(16 * ntheta);
            }
            if (ff_gpu_accum) {
                if (!ff_alpha_direct)
                    ff_workspace.reserve_mueller(ff_batch_orient, ntheta);
                ff_workspace.zero_mueller(ntheta);
                printf("  GPU Mueller accumulation enabled (set BEM_FF_CPU_ACCUM=1 for CPU fallback)\n");
            }
            if (split_orientation_outputs && !ff_alpha_direct) {
                fprintf(stderr, "Error: --orient-split-dir currently requires GPU alpha-direct far-field accumulation\n");
                return 1;
            }
            std::vector<double> M_split;
            if (split_orientation_outputs) {
                M_split.assign(16 * ntheta, 0.0);
                std::fill(M_avg.begin(), M_avg.end(), 0.0);
                printf("  Split orientation output enabled: %s\n", orient_split_dir);
            }
            cdouble ik_val = cdouble(0, -1) * k_ext;
            auto flush_farfield_batch = [&]() {
                if (batch_count == 0)
                    return;

                Timer ff_timer;
                int batch_calls = batch_count * 2;
                if (ff_gpu_accum) {
                    accumulate_farfield_mueller_batch_cuda_ws(
                        ff_gpu, ff_workspace,
                        batch_coeffs_J.data(), batch_coeffs_M.data(),
                        batch_r_hats.data(),
                        reinterpret_cast<const double*>(batch_e_par.data()),
                        reinterpret_cast<const double*>(batch_e_perp.data()),
                        batch_weights.data(),
                        k_ext, eta_ext,
                        batch_calls, batch_count, ntheta);
                    time_farfield += ff_timer.elapsed_s();
                    batch_count = 0;
                    return;
                }

                compute_farfield_batch_cuda_ws(ff_gpu, ff_workspace,
                                               batch_coeffs_J.data(), batch_coeffs_M.data(),
                                               batch_r_hats.data(),
                                               k_ext, eta_ext,
                                               batch_calls, batch_count, ntheta,
                                               batch_Fv.data());

                for (int bi = 0; bi < batch_count; bi++) {
                    double weight = batch_weights[bi];
                    cdouble* Fv_par  = &batch_Fv[(2*bi) * ntheta * 3];
                    cdouble* Fv_perp = &batch_Fv[(2*bi+1) * ntheta * 3];

                    for (int it = 0; it < ntheta; it++) {
                        Vec3& ep = batch_e_par[bi * ntheta + it];
                        Vec3& epp = batch_e_perp[bi * ntheta + it];

                        cdouble F_par_p  = Fv_par[it*3]*ep.x  + Fv_par[it*3+1]*ep.y  + Fv_par[it*3+2]*ep.z;
                        cdouble F_perp_p = Fv_par[it*3]*epp.x + Fv_par[it*3+1]*epp.y + Fv_par[it*3+2]*epp.z;
                        cdouble F_par_pp  = Fv_perp[it*3]*ep.x  + Fv_perp[it*3+1]*ep.y  + Fv_perp[it*3+2]*ep.z;
                        cdouble F_perp_pp = Fv_perp[it*3]*epp.x + Fv_perp[it*3+1]*epp.y + Fv_perp[it*3+2]*epp.z;

                        S2[it] = ik_val * F_par_p;
                        S4[it] = ik_val * F_perp_p;
                        S3[it] = ik_val * F_par_pp;
                        S1[it] = ik_val * F_perp_pp;
                    }

                    amplitude_to_mueller(S1.data(), S2.data(), S3.data(), S4.data(),
                                        ntheta, M_orient.data());

                    for (int i = 0; i < 16 * ntheta; i++)
                        M_avg[i] += weight * M_orient[i];
                }

                time_farfield += ff_timer.elapsed_s();
                batch_count = 0;
            };

            auto rotate_z_minus = [](const Vec3& v, double ca, double sa) {
                return Vec3(ca * v.x + sa * v.y,
                           -sa * v.x + ca * v.y,
                            v.z);
            };

            std::vector<double> alpha_cos(alpha_avg), alpha_sin(alpha_avg);
            for (int ia = 0; ia < alpha_avg; ia++) {
                double alpha = 2.0 * M_PI * (double)ia / (double)alpha_avg;
                alpha_cos[ia] = cos(alpha);
                alpha_sin[ia] = sin(alpha);
            }
            bool orient_pack_omp = !bem_env_flag_enabled("BEM_ORIENT_PACK_SERIAL");

            auto append_farfield_sample = [&](int oi, const Mat3& RT, double ca, double sa, double weight) {
                if (batch_count == ff_batch_orient)
                    flush_farfield_batch();

                double inv_s = (unknown_m_scale == 1.0) ? 1.0 : (1.0 / unknown_m_scale);
                int bi = batch_count++;
                batch_weights[bi] = weight;

                cdouble* J0 = &batch_coeffs_J[(2*bi) * N];
                cdouble* M0 = &batch_coeffs_M[(2*bi) * N];
                cdouble* J1 = &batch_coeffs_J[(2*bi+1) * N];
                cdouble* M1 = &batch_coeffs_M[(2*bi+1) * N];
                #pragma omp parallel for schedule(static) if(orient_pack_omp && N > 2048)
                for (int i = 0; i < N; i++) {
                    cdouble jp = x_par[i];
                    cdouble mp = x_par[N + i] * inv_s;
                    cdouble ju = x_perp[i];
                    cdouble mu = x_perp[N + i] * inv_s;
                    if (n_form) {
                        cdouble aj = jp, am = mp;
                        cdouble bj = ju, bm = mu;
                        switch (nform_ff_mode) {
                        case 1: // swap
                            jp = am; mp = aj; ju = bm; mu = bj; break;
                        case 2: // J=-M, M=J
                            jp = -am; mp = aj; ju = -bm; mu = bj; break;
                        case 3: // J=M, M=-J
                            jp = am; mp = -aj; ju = bm; mu = -bj; break;
                        case 4: // flip magnetic current
                            mp = -mp; mu = -mu; break;
                        case 5: // flip electric current
                            jp = -jp; ju = -ju; break;
                        default:
                            break;
                        }
                    }
                    if (scattering_plane_yz) {
                        J0[i] = ca * jp + sa * ju;
                        M0[i] = ca * mp + sa * mu;
                        J1[i] = -sa * jp + ca * ju;
                        M1[i] = -sa * mp + ca * mu;
                    } else {
                        J0[i] = ca * jp - sa * ju;
                        M0[i] = ca * mp - sa * mu;
                        J1[i] = -sa * jp - ca * ju;
                        M1[i] = -sa * mp - ca * mu;
                    }
                }

	                for (int it = 0; it < ntheta; it++) {
	                    Vec3 rh = RT * rotate_z_minus(r_hat_lab[it], ca, sa);
	                    int base = (bi * ntheta + it) * 3;
                    batch_r_hats[base]   = rh.x;
                    batch_r_hats[base+1] = rh.y;
                    batch_r_hats[base+2] = rh.z;
                    batch_e_par[bi * ntheta + it]  = RT * rotate_z_minus(e_theta_lab[it], ca, sa);
                    batch_e_perp[bi * ntheta + it] = RT * rotate_z_minus(e_phi_lab, ca, sa);
                }
            };

            int base_batch_count = 0;
            auto flush_alpha_direct_batch = [&]() {
                if (base_batch_count == 0)
                    return;
                Timer ff_timer;
                accumulate_farfield_mueller_alpha_geom_cuda_ws(
                    ff_gpu, ff_workspace,
                    batch_coeffs_J.data(), batch_coeffs_M.data(),
                    batch_RT.data(),
                    r_hat_lab_flat.data(), e_theta_lab_flat.data(),
                    e_phi_lab_flat,
                    batch_weights.data(),
                    alpha_cos.data(), alpha_sin.data(),
                    k_ext, eta_ext,
                    base_batch_count, alpha_avg, ntheta);
                time_farfield += ff_timer.elapsed_s();
                base_batch_count = 0;
            };

            auto append_alpha_direct_orientation = [&](int oi, const Mat3& RT, double weight) {
                if (base_batch_count == ff_base_batch_orient)
                    flush_alpha_direct_batch();

                double inv_s = (unknown_m_scale == 1.0) ? 1.0 : (1.0 / unknown_m_scale);
                int bi = base_batch_count++;
                batch_weights[bi] = weight;
                double* RT_out = &batch_RT[(size_t)bi * 9];
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        RT_out[r * 3 + c] = RT.m[r][c];

                cdouble* J0 = &batch_coeffs_J[(2*bi) * N];
                cdouble* M0 = &batch_coeffs_M[(2*bi) * N];
                cdouble* J1 = &batch_coeffs_J[(2*bi+1) * N];
                cdouble* M1 = &batch_coeffs_M[(2*bi+1) * N];
                #pragma omp parallel for schedule(static) if(orient_pack_omp && N > 2048)
                for (int i = 0; i < N; i++) {
                    cdouble jp = x_par[i];
                    cdouble mp = x_par[N + i] * inv_s;
                    cdouble ju = x_perp[i];
                    cdouble mu = x_perp[N + i] * inv_s;
                    if (n_form) {
                        cdouble aj = jp, am = mp;
                        cdouble bj = ju, bm = mu;
                        switch (nform_ff_mode) {
                        case 1:
                            jp = am; mp = aj; ju = bm; mu = bj; break;
                        case 2:
                            jp = -am; mp = aj; ju = -bm; mu = bj; break;
                        case 3:
                            jp = am; mp = -aj; ju = bm; mu = -bj; break;
                        case 4:
                            mp = -mp; mu = -mu; break;
                        case 5:
                            jp = -jp; ju = -ju; break;
                        default:
                            break;
                        }
                    }
                    J0[i] = jp;
                    M0[i] = mp;
                    J1[i] = ju;
                    M1[i] = mu;
                }
            };

            bool use_gmres_gpu_rhs = !bem_env_flag_enabled("BEM_NO_GPU_RHS");
            int rhs_batch_orient = 1;
            std::vector<Vec3> rhs_batch_k, rhs_batch_par, rhs_batch_perp;
            RHSBatchWorkspace rhs_workspace;
            bool rhs_can_use_workspace_direct = !n_form;
            if (use_gmres_gpu_rhs) {
                double target_mb = bem_env_double("BEM_RHS_TARGET_MB", 256.0);
                int max_auto_rhs = bem_env_int("BEM_RHS_MAX_BATCH", 512);
                double per_orient_bytes = 4.0 * (double)N * sizeof(cdouble);
                rhs_batch_orient = (int)((target_mb * 1024.0 * 1024.0) / std::max(1.0, per_orient_bytes));
                rhs_batch_orient = std::max(1, std::min(rhs_batch_orient, std::max(1, max_auto_rhs)));
                if (bem_env_has_value("BEM_RHS_BATCH"))
                    rhs_batch_orient = std::max(1, bem_env_int("BEM_RHS_BATCH", rhs_batch_orient));
                rhs_batch_orient = std::min(rhs_batch_orient, std::max(1, n_total));
                rhs_batch_k.resize(rhs_batch_orient);
                rhs_batch_par.resize(rhs_batch_orient);
                rhs_batch_perp.resize(rhs_batch_orient);
                printf("  GPU RHS streaming batch: %d orientations (BEM_RHS_BATCH overrides; BEM_RHS_TARGET_MB tunes auto)\n",
                       rhs_batch_orient);
            }

            for (int rhs_start = 0; rhs_start < n_total; rhs_start += rhs_batch_orient) {
                int rhs_count = std::min(rhs_batch_orient, n_total - rhs_start);
                if (use_gmres_gpu_rhs) {
                    for (int bi = 0; bi < rhs_count; bi++) {
                        Mat3& RT = orients[rhs_start + bi].RT;
                        rhs_batch_k[bi] = RT * Vec3(0, 0, 1);
                        if (scattering_plane_yz) {
                            // ADDA yz-plane convention: alpha=0 has par=Y and per=X.
                            rhs_batch_par[bi] = RT * Vec3(0, 1, 0);
                            rhs_batch_perp[bi] = RT * Vec3(1, 0, 0);
                        } else {
                            rhs_batch_par[bi] = RT * Vec3(1, 0, 0);
                            rhs_batch_perp[bi] = RT * Vec3(0, -1, 0);
                        }
                    }
                    compute_rhs_planewave_pairs_cached_cuda_ws_scaled(
                        ff_gpu, rhs_workspace, k_ext, eta_ext,
                        n_form ? std::complex<double>(1.0, 0.0) : row_h_scale,
                        rhs_batch_par.data(), rhs_batch_perp.data(), rhs_batch_k.data(),
                        rhs_count, nullptr);
                }

                for (int local = 0; local < rhs_count; local++) {
                    int oi = rhs_start + local;
                    Mat3& RT = orients[oi].RT;
                    const cdouble* solve_b_par = b_par.data();
                    const cdouble* solve_b_perp = b_perp.data();
                    if (use_gmres_gpu_rhs) {
                        const cdouble* rhs_batch_B = rhs_workspace.host_B();
                        const cdouble* rhs_par = &rhs_batch_B[(size_t)local * 2 * N2];
                        const cdouble* rhs_perp = &rhs_batch_B[((size_t)local * 2 + 1) * N2];
                        if (rhs_can_use_workspace_direct) {
                            solve_b_par = rhs_par;
                            solve_b_perp = rhs_perp;
                        } else {
                            std::memcpy(b_par.data(), rhs_par, (size_t)N2 * sizeof(cdouble));
                            std::memcpy(b_perp.data(), rhs_perp, (size_t)N2 * sizeof(cdouble));
                        }
                    } else {
                        Vec3 k_hat = RT * Vec3(0, 0, 1);
                        Vec3 e_par = scattering_plane_yz ? (RT * Vec3(0, 1, 0)) : (RT * Vec3(1, 0, 0));
                        Vec3 e_perp = scattering_plane_yz ? (RT * Vec3(1, 0, 0)) : (RT * Vec3(0, -1, 0));
                        compute_rhs_planewave_pair_cached(ff_cache, k_ext, eta_ext, e_par, e_perp,
                                                         k_hat, b_par.data(), b_perp.data());
                    }
                    if (!rhs_can_use_workspace_direct && n_form) {
                        transform_rhs_to_n_form(b_par.data(), N, nform_rhs_mode);
                        transform_rhs_to_n_form(b_perp.data(), N, nform_rhs_mode);
                    }
                    if ((!use_gmres_gpu_rhs || n_form) &&
                        std::abs(row_h_scale - std::complex<double>(1.0, 0.0)) > 0.0) {
                        #pragma omp parallel for schedule(static) if(N > 2048)
                        for (int i = 0; i < N; i++) {
                            b_par[N + i] *= row_h_scale;
                            b_perp[N + i] *= row_h_scale;
                        }
                    }

                    if (bem_env_has_value("BEM_CHECK_COEFFS_JSON")) {
                        const char* coeff_path = std::getenv("BEM_CHECK_COEFFS_JSON");
                        if (!load_coefficients_json(coeff_path, N, x_par, x_perp))
                            return 1;
                        double rel_par = relative_residual(fmm_op, x_par.data(), solve_b_par);
                        double rel_perp = relative_residual(fmm_op, x_perp.data(), solve_b_perp);
                        printf("  [CoeffCheck] orient %d %s: relres_par=%.6e relres_perp=%.6e max=%.6e\n",
                               oi + 1, coeff_path, rel_par, rel_perp, std::max(rel_par, rel_perp));
                        fflush(stdout);
                        if (bem_env_flag_enabled("BEM_CHECK_COEFFS_EXIT"))
                            return 0;
                    }

                    bool loaded_coeffs_for_farfield = false;
                    int mv = 0;
                    double one_solve_s = 0.0;
                    if (bem_env_has_value("BEM_LOAD_COEFFS_FARFIELD_JSON")) {
                        const char* coeff_path = std::getenv("BEM_LOAD_COEFFS_FARFIELD_JSON");
                        if (!load_coefficients_json(coeff_path, N, x_par, x_perp))
                            return 1;
                        double rel_par = 0.0;
                        double rel_perp = 0.0;
                        if (!coeffs_farfield_only) {
                            rel_par = relative_residual(fmm_op, x_par.data(), solve_b_par);
                            rel_perp = relative_residual(fmm_op, x_perp.data(), solve_b_perp);
                        }
                        gmres_ws.final_relres1 = rel_par;
                        gmres_ws.final_relres2 = rel_perp;
                        gmres_ws.converged1 = true;
                        gmres_ws.converged2 = true;
                        loaded_coeffs_for_farfield = true;
                        if (coeffs_farfield_only) {
                            printf("  [CoeffFarfield] orient %d %s: residual check skipped\n",
                                   oi + 1, coeff_path);
                        } else {
                            printf("  [CoeffFarfield] orient %d %s: relres_par=%.6e relres_perp=%.6e max=%.6e\n",
                                   oi + 1, coeff_path, rel_par, rel_perp, std::max(rel_par, rel_perp));
                        }
                        fflush(stdout);
                    } else {
                        if (warm_start == OrientWarmStart::Zero) {
                            #pragma omp parallel for schedule(static) if(N2 > 4096)
                            for (int i = 0; i < N2; i++) {
                                x_par[i] = cdouble(0);
                                x_perp[i] = cdouble(0);
                            }
                        } else if (warm_start == OrientWarmStart::Recycle) {
                            recycle_initial_guess_pair(hist_b, hist_x,
                                                       solve_b_par, solve_b_perp, N2,
                                                       x_par.data(), x_perp.data());
                        }

                        Timer orient_solve_timer;
                        mv = solve_pair_with_prec_fallback(
                            solve_b_par, solve_b_perp,
                            x_par.data(), x_perp.data(),
                            gmres_ws);
                        one_solve_s = orient_solve_timer.elapsed_s();
                    }
                    time_solve += one_solve_s;
                    orient_matvecs += mv;
                    output_gmres_matvecs = orient_matvecs;
                    if (!loaded_coeffs_for_farfield &&
                        use_gpu_gmres && !bem_env_flag_enabled("BEM_SKIP_HOST_RESIDUAL_CHECK")) {
                        double host_rel_par = relative_residual(fmm_op, x_par.data(), solve_b_par);
                        double host_rel_perp = relative_residual(fmm_op, x_perp.data(), solve_b_perp);
                        double host_rel = std::max(host_rel_par, host_rel_perp);
                        if (host_rel > 2.0 * gmres_tol) {
                            printf("  [PostSolveResidualCheck] orient %d rejected: relres_par=%.6e relres_perp=%.6e max=%.6e > %.3e\n",
                                   oi + 1, host_rel_par, host_rel_perp, host_rel, 2.0 * gmres_tol);
                            fflush(stdout);
                            gmres_ws.converged1 = false;
                            gmres_ws.converged2 = false;
                        }
                        gmres_ws.final_relres1 = std::max(gmres_ws.final_relres1, host_rel_par);
                        gmres_ws.final_relres2 = std::max(gmres_ws.final_relres2, host_rel_perp);
                    }
                    output_gmres_converged_systems += (gmres_ws.converged1 ? 1 : 0) +
                                                      (gmres_ws.converged2 ? 1 : 0);
                    output_gmres_nonconverged_systems += (gmres_ws.converged1 ? 0 : 1) +
                                                         (gmres_ws.converged2 ? 0 : 1);
                    if (gmres_ws.stopped_stagnant)
                        output_gmres_stagnation_stops++;
                    if (gmres_ws.numerical_breakdown)
                        output_gmres_numerical_breakdowns++;
                    if (gmres_ws.restored_best_iterate)
                        output_gmres_restored_best_iterates++;
                    if (gmres_ws.reached_max_cycles)
                        output_gmres_max_cycle_exhaustions++;
                    output_gmres_max_final_relres = std::max(output_gmres_max_final_relres,
                        std::max(gmres_ws.final_relres1, gmres_ws.final_relres2));

                    if (warm_start == OrientWarmStart::Recycle) {
                        push_history(hist_b, hist_x, solve_b_par, x_par.data(), N2, max_recycle);
                        push_history(hist_b, hist_x, solve_b_perp, x_perp.data(), N2, max_recycle);
                    }

                    if (bem_env_has_value("BEM_EXPORT_COEFFS_JSON")) {
                        if (n_total != 1) {
                            fprintf(stderr,
                                    "Error: BEM_EXPORT_COEFFS_JSON in FMM orientation-loop currently requires exactly one orientation, got %d\n",
                                    n_total);
                            return 1;
                        }
                        export_coefficients_json(std::getenv("BEM_EXPORT_COEFFS_JSON"),
                                                 N, x_par.data(), x_perp.data());
                    }

                    if (ff_alpha_direct) {
                        if (split_orientation_outputs) {
                            Timer ff_one_timer;
                            ff_workspace.zero_mueller(ntheta);
                            append_alpha_direct_orientation(oi, RT, orients[oi].weight);
                            flush_alpha_direct_batch();
                            ff_workspace.download_mueller(M_split.data(), ntheta);
                            double one_farfield_s = ff_one_timer.elapsed_s();
                            for (int mi = 0; mi < 16 * ntheta; mi++)
                                M_avg[mi] += M_split[mi];
                            std::string split_path = orient_part_path(orient_split_dir, split_indices[(size_t)oi]);
                            write_json(split_path.c_str(), M_split.data(), theta_arr.data(), ntheta,
                                       ka, n_re, n_im, refinements,
                                       shape, obj_file, prism_aspect, edge_refine,
                                       n_alpha, n_beta, n_gamma, alpha_avg,
                                       split_indices[(size_t)oi], 1, split_output_total,
                                       orients[oi].weight, mv,
                                       (gmres_ws.converged1 ? 1 : 0) + (gmres_ws.converged2 ? 1 : 0),
                                       (gmres_ws.converged1 ? 0 : 1) + (gmres_ws.converged2 ? 0 : 1),
                                       gmres_ws.stopped_stagnant ? 1 : 0,
                                       gmres_ws.numerical_breakdown ? 1 : 0,
                                       gmres_ws.restored_best_iterate ? 1 : 0,
                                       gmres_ws.reached_max_cycles ? 1 : 0,
                                       std::max(gmres_ws.final_relres1, gmres_ws.final_relres2),
                                       fmm_digits, max_leaf, gmres_restart, gmres_tol, gmres_max_cycles,
                                       requested_fmm_digits, requested_gmres_tol,
                                       fmm_digits_set, gmres_tol_set, accuracy_policy_adjusted,
                                       "disabled",
                                       output_farfield_mode,
                                       solver_name(solver), acc_policy.profile,
                                       output_krylov_solver,
                                       requested_system_kind, system_kind,
                                       std::strcmp(requested_system_kind, system_kind) != 0,
                                       quad_order, unknown_m_scale,
                                       row_h_scale, int_op_sign, k_identity,
                                       use_prec,
                                       use_prec && bem_env_flag_enabled("BEM_PREC_BLOCK"),
                                       !use_prec && (bem_env_flag_enabled("BEM_GMRES_DEVICE") || use_bicgstab || use_cgs_rr || use_krylov_auto || use_krylov_hybrid),
                                       prec_policy.reason,
                                       mesh_quality.vertices, mesh_quality.triangles,
                                       mesh_quality.skinny_triangles,
                                       mesh_quality.min_angle_deg, mesh_quality.max_aspect_ratio,
                                       mesh_quality.feature_edges_30deg,
                                       mesh_quality.feature_edge_fraction,
                                       mesh_quality.max_dihedral_deg,
                                       mesh_quality.mean_feature_dihedral_deg,
                                       mesh_quality.max_adjacent_area_ratio,
                                       mesh_quality.near_touch_checked,
                                       mesh_quality.near_touch_ratio,
                                       mesh_quality.near_touch_pairs,
                                       mesh_quality.self_panel_count,
                                       mesh_quality.edge_adjacent_pair_count,
                                       mesh_quality.vertex_adjacent_pair_count,
                                       mesh_quality.near_disjoint_pair_count,
                                       mesh_quality.taylor_duffy_candidate_count,
                                       mesh_quality.recommended_min_quad_order,
                                       mesh_quality.recommended_mesh_strategy.c_str(),
                                       mesh_quality.recommended_mesh_action.c_str(),
                                       mesh_quality.voxel_surface_like,
                                       mesh_quality.requires_remesh,
                                       mesh_quality.edge_refine_requested, mesh_quality.edge_refine_applied,
                                       mesh_quality.edge_refine_uniform_fallback,
                                       mesh_quality.pass_default_gate,
                                       time_assembly / std::max(1, n_total),
                                       one_solve_s,
                                       one_farfield_s,
                                       time_assembly / std::max(1, n_total) + one_solve_s + one_farfield_s);
                        } else {
                            append_alpha_direct_orientation(oi, RT, orients[oi].weight);
                        }
                    } else if (alpha_avg == 1) {
                        append_farfield_sample(oi, RT, 1.0, 0.0, orients[oi].weight);
                    } else {
                        double sample_weight = orients[oi].weight / (double)alpha_avg;
                        for (int ia = 0; ia < alpha_avg; ia++) {
                            append_farfield_sample(oi, RT, alpha_cos[ia], alpha_sin[ia], sample_weight);
                        }
                    }

                    if ((oi + 1) % orient_progress_step == 0 || oi == n_total - 1)
                        printf("    Orient %d/%d done (avg %.1f matvec/orient)\n",
                               oi + 1, n_total, (double)orient_matvecs / (oi + 1));
                }
            }
		            if (!split_orientation_outputs) {
		                if (ff_alpha_direct)
		                    flush_alpha_direct_batch();
		                else
		                    flush_farfield_batch();
		                if (ff_gpu_accum)
		                    ff_workspace.download_mueller(M_avg.data(), ntheta);
		            }
            if (alpha_avg > 1)
                printf("  Averaged over %d solved orientations x %d alpha samples.\n", n_total, alpha_avg);
            else
                printf("  Averaged over %d orientations.\n", n_total);
        }

        if (!coeffs_farfield_only)
            fmm_op.cleanup();

    } else {
        // ============================================================
        // Dense LU path (original code)
        // ============================================================
        Timer asm_timer;
        std::vector<std::complex<double>> Z(N2 * N2);
        assemble_pmchwt(rwg, mesh, k_ext, k_int, eta_ext_c, eta_int_c,
                        quad_order, unknown_m_scale, row_h_scale,
                        int_op_sign, k_identity, Z.data(), NULL, NULL);
        time_assembly = asm_timer.elapsed_s();

        Timer solve_timer;

        if (single_orient || sphere_orientation_shortcut) {
            Vec3 k_hat(0, 0, 1);
            Vec3 E_par  = scattering_plane_yz ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
            Vec3 E_perp = scattering_plane_yz ? Vec3(1, 0, 0) : Vec3(0, 1, 0);

            std::vector<int> ipiv(N2);
            lu_factorize_cuda(Z.data(), N2, ipiv.data());

            std::vector<std::complex<double>> B(N2 * 2);
	            compute_rhs_planewave_pair_cached(ff_cache, k_ext, eta_ext, E_par, E_perp,
	                                             k_hat, &B[0], &B[N2]);
            if (n_form) {
                transform_rhs_to_n_form(&B[0], N, nform_rhs_mode);
                transform_rhs_to_n_form(&B[N2], N, nform_rhs_mode);
            }
            if (std::abs(row_h_scale - std::complex<double>(1.0, 0.0)) > 0.0) {
                for (int i = 0; i < N; i++) {
                    B[N + i] *= row_h_scale;
                    B[N2 + N + i] *= row_h_scale;
                }
            }

	            lu_solve_cuda(Z.data(), ipiv.data(), N2, B.data(), 2);
	            time_solve = solve_timer.elapsed_s();

            if (bem_env_has_value("BEM_EXPORT_COEFFS_JSON"))
                export_coefficients_json(std::getenv("BEM_EXPORT_COEFFS_JSON"),
                                         N, &B[0], &B[N2]);

            Timer ff_timer;
            std::complex<double>* J_par  = &B[0];
            std::complex<double>* M_par  = &B[N];
            std::complex<double>* J_perp = &B[N2];
            std::complex<double>* M_perp = &B[N2 + N];
            std::vector<std::complex<double>> M_par_phys, M_perp_phys;
            if (unknown_m_scale != 1.0) {
                M_par_phys.resize(N);
                M_perp_phys.resize(N);
                double inv_s = 1.0 / unknown_m_scale;
                for (int i = 0; i < N; i++) {
                    M_par_phys[i] = M_par[i] * inv_s;
                    M_perp_phys[i] = M_perp[i] * inv_s;
                }
                M_par = M_par_phys.data();
                M_perp = M_perp_phys.data();
            }
            std::vector<std::complex<double>> J_par_nform, M_par_nform;
            std::vector<std::complex<double>> J_perp_nform, M_perp_nform;
            if (n_form) {
                recover_n_form_farfield_pair(J_par, M_par, N, nform_ff_mode,
                                             J_par_nform, M_par_nform);
                recover_n_form_farfield_pair(J_perp, M_perp, N, nform_ff_mode,
                                             J_perp_nform, M_perp_nform);
                J_par = J_par_nform.data();
                M_par = M_par_nform.data();
                J_perp = J_perp_nform.data();
                M_perp = M_perp_nform.data();
            }

            std::vector<std::complex<double>> Fth_par(ntheta), Fph_par(ntheta);
            std::vector<std::complex<double>> Fth_perp(ntheta), Fph_perp(ntheta);

            compute_far_field(ff_cache, J_par, M_par, k_ext, eta_ext,
                             theta_arr.data(), ntheta,
                             scattering_plane_yz,
                             Fth_par.data(), Fph_par.data());
            compute_far_field(ff_cache, J_perp, M_perp, k_ext, eta_ext,
                             theta_arr.data(), ntheta,
                             scattering_plane_yz,
                             Fth_perp.data(), Fph_perp.data());

            std::complex<double> ik(0, -1);
            ik *= k_ext;
            std::vector<std::complex<double>> S1(ntheta), S2(ntheta), S3(ntheta), S4(ntheta);
            for (int t = 0; t < ntheta; t++) {
                S2[t] = ik * Fth_par[t];
                S4[t] = ik * Fph_par[t];
                S3[t] = ik * Fth_perp[t];
                S1[t] = ik * Fph_perp[t];
            }

            amplitude_to_mueller(S1.data(), S2.data(), S3.data(), S4.data(),
                                ntheta, M_avg.data());
            output_farfield_mode = "single_orientation_cpu_farfield_dense_solve";

            if (export_currents_file) {
                currents_exported = export_currents_vtk(export_currents_file, mesh, rwg,
                                                        J_par, M_par, J_perp, M_perp);
                if (currents_exported)
                    printf("  Exported equivalent currents: %s\n", export_currents_file);
            }

            time_farfield = ff_timer.elapsed_s();

        } else {
            // Orientation averaging (batched)
            std::vector<Orientation> orients;
            if (orient_file) {
                if (!load_orientation_file(orient_file, orients))
                    return 1;
            } else if (orient_bg_file) {
                if (!load_beta_gamma_orientation_file(orient_bg_file, orients))
                    return 1;
            } else {
                orients = generate_orientations(n_alpha, n_beta, n_gamma);
                reorder_orientations_nearest(orients);
            }
            int n_all = (int)orients.size();
            if (orient_start < 0) orient_start = 0;
            if (orient_start > n_all) orient_start = n_all;
            int n_keep = (orient_count < 0) ? (n_all - orient_start)
                                            : std::min(orient_count, n_all - orient_start);
            if (orient_start > 0 || n_keep < n_all) {
                orients = std::vector<Orientation>(orients.begin() + orient_start,
                                                   orients.begin() + orient_start + n_keep);
                printf("  Orientation chunk: start=%d count=%d of %d\n", orient_start, n_keep, n_all);
            }
            int n_total = (int)orients.size();
            output_orient_start = orient_start;
            output_orient_count = n_total;
            output_orient_total = n_all;
            output_orientation_weight_sum = 0.0;
            for (const auto& o : orients)
                output_orientation_weight_sum += o.weight;

            printf("\n  Building %d RHS vectors...\n", n_total * 2);

            FFCacheGPU ff_gpu;
            bool ff_gpu_uploaded = false;
	            // Phase 1: Build all RHS
	            std::vector<std::complex<double>> B_storage;
	            RHSBatchWorkspace rhs_workspace;
	            cdouble* B = nullptr;
	            std::vector<Vec3> rhs_k_hat(n_total), rhs_e_par(n_total), rhs_e_perp(n_total);
	            for (int oi = 0; oi < n_total; oi++) {
	                Mat3& RT = orients[oi].RT;
	                rhs_k_hat[oi] = RT * Vec3(0, 0, 1);
	                if (scattering_plane_yz) {
		                rhs_e_par[oi] = RT * Vec3(0, 1, 0);
		                rhs_e_perp[oi] = RT * Vec3(1, 0, 0);
                    } else {
		                rhs_e_par[oi] = RT * Vec3(1, 0, 0);
		                rhs_e_perp[oi] = RT * Vec3(0, -1, 0);
                    }
	            }
	            bool use_gpu_rhs = !bem_env_flag_enabled("BEM_NO_GPU_RHS");
	            if (use_gpu_rhs) {
	                printf("  GPU RHS batch enabled (set BEM_NO_GPU_RHS=1 for CPU fallback)\n");
                    ff_gpu.upload(ff_cache);
                    ff_gpu_uploaded = true;
	                compute_rhs_planewave_pairs_cached_cuda_ws_scaled(
	                    ff_gpu, rhs_workspace, k_ext, eta_ext, row_h_scale,
	                    rhs_e_par.data(), rhs_e_perp.data(), rhs_k_hat.data(),
	                    n_total, nullptr);
                    B = rhs_workspace.host_B();
	            } else {
                    B_storage.assign((size_t)N2 * n_total * 2, cdouble(0));
                    B = B_storage.data();
	                #ifdef _OPENMP
	                #pragma omp parallel for schedule(static)
	                #endif
	                for (int oi = 0; oi < n_total; oi++) {
	                    compute_rhs_planewave_pair_cached(
	                        ff_cache, k_ext, eta_ext,
	                        rhs_e_par[oi], rhs_e_perp[oi], rhs_k_hat[oi],
	                        &B[oi * 2 * N2], &B[(oi * 2 + 1) * N2]);
	                }
                    if (std::abs(row_h_scale - std::complex<double>(1.0, 0.0)) > 0.0) {
                        #ifdef _OPENMP
                        #pragma omp parallel for schedule(static)
                        #endif
                        for (int oi = 0; oi < n_total; oi++) {
                            cdouble* bp = &B[(size_t)oi * 2 * N2];
                            cdouble* bu = &B[((size_t)oi * 2 + 1) * N2];
                            for (int i = 0; i < N; i++) {
                                bp[N + i] *= row_h_scale;
                                bu[N + i] *= row_h_scale;
                            }
                        }
                    }
	            }

            // Phase 2: LU solve all at once
	            printf("  Solving %d RHS with LU...\n", n_total * 2);
	            lu_solve_full(Z.data(), N2, B, n_total * 2);
	            time_solve = solve_timer.elapsed_s();

            if (bem_env_has_value("BEM_EXPORT_COEFFS_JSON")) {
                if (n_total != 1) {
                    fprintf(stderr, "Error: BEM_EXPORT_COEFFS_JSON in orientation-loop currently requires exactly one orientation, got %d\n",
                            n_total);
                    return 1;
                }
                export_coefficients_json(std::getenv("BEM_EXPORT_COEFFS_JSON"),
                                         N, &B[0], &B[N2]);
            }

            // Phase 3: Far-field + Mueller accumulation (GPU batched)
            Timer ff_timer;
            int n_calls = n_total * 2;
            printf("  Computing GPU far-field: %d calls x %d dirs...\n", n_calls, ntheta);

	            std::vector<Vec3> r_hat_lab(ntheta), e_theta_lab(ntheta);
	            std::vector<double> r_hat_lab_flat((size_t)ntheta * 3);
	            std::vector<double> e_theta_lab_flat((size_t)ntheta * 3);
	            Vec3 e_phi_lab = scattering_plane_yz ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
	            double e_phi_lab_flat[3] = {e_phi_lab.x, e_phi_lab.y, e_phi_lab.z};
	            for (int it = 0; it < ntheta; it++) {
	                double ct = cos(theta_arr[it]), st = sin(theta_arr[it]);
	                if (scattering_plane_yz) {
                    r_hat_lab[it] = Vec3(0, st, ct);
                    e_theta_lab[it] = Vec3(0, ct, -st);
                } else {
	                    r_hat_lab[it] = Vec3(st, 0, ct);
	                    e_theta_lab[it] = Vec3(ct, 0, -st);
	                }
	                r_hat_lab_flat[(size_t)it * 3]     = r_hat_lab[it].x;
	                r_hat_lab_flat[(size_t)it * 3 + 1] = r_hat_lab[it].y;
	                r_hat_lab_flat[(size_t)it * 3 + 2] = r_hat_lab[it].z;
	                e_theta_lab_flat[(size_t)it * 3]     = e_theta_lab[it].x;
	                e_theta_lab_flat[(size_t)it * 3 + 1] = e_theta_lab[it].y;
	                e_theta_lab_flat[(size_t)it * 3 + 2] = e_theta_lab[it].z;
	            }

	            long long n_farfield_samples = (long long)n_total * (long long)alpha_avg;
	            int ff_batch_orient = 64;
	            if (bem_env_has_value("BEM_FF_BATCH")) {
	                ff_batch_orient = std::max(1, bem_env_int("BEM_FF_BATCH", ff_batch_orient));
	            } else {
	                double target_mb = bem_env_double("BEM_FF_TARGET_MB", 512.0);
	                int default_max_batch = (n_farfield_samples > 8192) ? 8192 : 512;
	                int max_auto_batch = bem_env_int("BEM_FF_MAX_BATCH", default_max_batch);
	                double per_orient_bytes = 64.0 * (double)N + 120.0 * (double)ntheta + 8.0;
	                ff_batch_orient = (int)((target_mb * 1024.0 * 1024.0) / std::max(1.0, per_orient_bytes));
	                ff_batch_orient = std::max(1, std::min(ff_batch_orient, std::max(1, max_auto_batch)));
	            }
		            ff_batch_orient = std::min(ff_batch_orient, std::max(1, (int)std::min<long long>(n_farfield_samples, INT_MAX)));
	            bool ff_gpu_accum = !bem_env_flag_enabled("BEM_FF_CPU_ACCUM");
	            bool ff_alpha_direct = ff_gpu_accum &&
	                                   !bem_env_flag_enabled("BEM_FF_SEPARATE") &&
	                                   !bem_env_flag_enabled("BEM_FF_NO_ALPHA_DIRECT");
	            bool ff_alpha_geom = ff_alpha_direct && !bem_env_flag_enabled("BEM_FF_NO_ALPHA_GEOM");
		            int ff_mgpu = 1;
		            int cuda_device_count = 1;
		            bool have_cuda_device_count = false;
		            std::vector<int> ff_devices;
		            bool ff_gpu_list_explicit = false;
		            int ff_original_device = 0;
		            cudaGetDevice(&ff_original_device);
		            if (const char* env = std::getenv("BEM_FF_GPU_LIST")) {
		                ff_gpu_list_explicit = true;
		                cudaError_t dev_err = cudaGetDeviceCount(&cuda_device_count);
		                have_cuda_device_count = (dev_err == cudaSuccess);
		                if (!have_cuda_device_count) {
		                    fprintf(stderr, "Warning: cudaGetDeviceCount failed: %s; disabling BEM_FF_GPU_LIST\n",
		                            cudaGetErrorString(dev_err));
		                    ff_gpu_list_explicit = false;
		                } else {
		                    ff_devices = bem_parse_gpu_list_env(env);
		                    if (!bem_validate_gpu_list(ff_devices, cuda_device_count)) {
		                        fprintf(stderr, "Warning: invalid BEM_FF_GPU_LIST for %d CUDA devices; disabling far-field multi-GPU\n",
		                                cuda_device_count);
		                        ff_devices.clear();
		                        ff_gpu_list_explicit = false;
		                    } else {
		                        ff_mgpu = (int)ff_devices.size();
		                    }
		                }
            } else if (bem_env_has_value("BEM_FF_MGPU")) {
		                ff_mgpu = std::max(1, bem_env_int("BEM_FF_MGPU", ff_mgpu));
            } else if (ff_alpha_geom && !bem_env_flag_enabled("BEM_NO_AUTO_MGPU")) {
                int mgpu_min_samples = 4096;
                mgpu_min_samples = std::max(1, bem_env_int("BEM_FF_MGPU_MIN_SAMPLES", mgpu_min_samples));
                cudaError_t dev_err = cudaGetDeviceCount(&cuda_device_count);
                have_cuda_device_count = (dev_err == cudaSuccess);
                if (have_cuda_device_count && n_farfield_samples >= (long long)mgpu_min_samples) {
                    ff_mgpu = std::max(1, cuda_device_count);
                } else if (!have_cuda_device_count) {
                    fprintf(stderr, "Warning: cudaGetDeviceCount failed: %s; disabling auto multi-GPU far-field\n",
                            cudaGetErrorString(dev_err));
                }
            }
		            if (ff_mgpu > 1) {
		                if (!have_cuda_device_count) {
		                    cudaError_t dev_err = cudaGetDeviceCount(&cuda_device_count);
		                    if (dev_err != cudaSuccess) {
		                        fprintf(stderr, "Warning: cudaGetDeviceCount failed: %s; disabling BEM_FF_MGPU\n",
		                                cudaGetErrorString(dev_err));
		                        ff_mgpu = 1;
		                    } else {
		                        have_cuda_device_count = true;
		                    }
		                }
		                if (have_cuda_device_count)
		                    ff_mgpu = std::min(ff_mgpu, cuda_device_count);
		            }
		            if (ff_devices.empty()) {
		                for (int gd = 0; gd < ff_mgpu; gd++)
		                    ff_devices.push_back(gd);
		            } else {
		                ff_mgpu = (int)ff_devices.size();
		            }
	            bool ff_alpha_mgpu = ff_alpha_geom && ff_mgpu > 1;
            if (ff_alpha_mgpu)
                output_farfield_mode = "gpu_geometry_direct_multi_gpu";
            else if (!ff_gpu_accum)
                output_farfield_mode = "gpu_farfield_cpu_mueller_accum";
            else if (ff_alpha_geom)
                output_farfield_mode = "gpu_geometry_direct";
            else if (ff_alpha_direct)
                output_farfield_mode = "gpu_alpha_direct_host_geometry";
            else
                output_farfield_mode = "gpu_host_geometry_mueller_accum";
	            int ff_base_batch_orient = ff_batch_orient;
	            if (ff_alpha_direct) {
	                if (ff_alpha_geom && !bem_env_has_value("BEM_FF_BATCH")) {
	                    double target_mb = bem_env_double("BEM_FF_TARGET_MB", 512.0);
	                    int max_auto_base = bem_env_int("BEM_FF_MAX_BASE_BATCH", 4096);
	                    double per_base_bytes = 64.0 * (double)N + 9.0 * sizeof(double) +
	                                            (double)alpha_avg * sizeof(double);
	                    ff_base_batch_orient = (int)((target_mb * 1024.0 * 1024.0) / std::max(1.0, per_base_bytes));
	                    ff_base_batch_orient = std::max(1, std::min(ff_base_batch_orient, std::max(1, max_auto_base)));
	                    ff_base_batch_orient = std::min(ff_base_batch_orient, std::max(1, n_total));
	                } else {
	                    ff_base_batch_orient = std::max(1, ff_batch_orient / alpha_avg);
	                }
	                ff_batch_orient = ff_base_batch_orient * alpha_avg;
	            }
	            printf("  Streaming GPU Mueller batch: %d orientations\n", ff_batch_orient);
	            if (ff_alpha_direct)
	                printf("  Geometry-direct GPU coefficient mixing enabled: %d base orientations x %d alpha\n",
	                       ff_base_batch_orient, alpha_avg);
	            if (ff_alpha_geom)
	                printf("  Alpha-direct GPU geometry enabled\n");
	            if (ff_alpha_mgpu)
	                printf("  In-process multi-GPU far-field enabled: %d GPUs\n", ff_mgpu);

	            int batch_count = 0;
            int coeff_batch_orient = ff_alpha_direct ? ff_base_batch_orient : ff_batch_orient;
	            PinnedHostBuffer<cdouble> batch_coeffs_J;
            PinnedHostBuffer<cdouble> batch_coeffs_M;
            batch_coeffs_J.resize((size_t)coeff_batch_orient * 2 * N);
            batch_coeffs_M.resize((size_t)coeff_batch_orient * 2 * N);
            bool need_host_farfield_geom = !ff_alpha_direct || !ff_alpha_geom;
	            PinnedHostBuffer<double> batch_r_hats;
	            PinnedHostBuffer<Vec3> batch_e_par;
	            PinnedHostBuffer<Vec3> batch_e_perp;
            if (need_host_farfield_geom) {
                batch_r_hats.resize((size_t)ff_batch_orient * ntheta * 3);
                batch_e_par.resize((size_t)ff_batch_orient * ntheta);
                batch_e_perp.resize((size_t)ff_batch_orient * ntheta);
            }
	            PinnedHostBuffer<double> batch_weights;
            batch_weights.resize(ff_batch_orient);
	            PinnedHostBuffer<double> batch_RT;
            batch_RT.resize((size_t)ff_base_batch_orient * 9);
	            FFBatchWorkspace ff_workspace;
	            std::vector<double> mgpu_M_accum;
	            std::vector<std::vector<double>> mgpu_partial;
	            std::vector<FFCacheGPU*> mgpu_ff;
	            std::vector<FFBatchWorkspace*> mgpu_ws;
            PinnedHostBuffer<cdouble> batch_Fv;
            std::vector<cdouble> S1, S2, S3, S4;
            std::vector<double> M_orient;
            cdouble ik_val = cdouble(0, -1) * k_ext;
            bool ff_single_gpu_pinned = false;
            if (ff_gpu_list_explicit && !ff_alpha_mgpu && !ff_devices.empty()) {
                CUDA_CHECK(cudaSetDevice(ff_devices[0]));
                ff_single_gpu_pinned = (ff_devices[0] != ff_original_device);
                printf("  Far-field GPU work pinned to GPU %d\n", ff_devices[0]);
            }
            if (ff_gpu_uploaded && (ff_single_gpu_pinned || ff_alpha_mgpu)) {
                ff_gpu.free();
                ff_gpu_uploaded = false;
            }
	            if (ff_gpu_accum && !ff_alpha_mgpu) {
	                if (!ff_gpu_uploaded) {
	                    ff_gpu.upload(ff_cache);
	                    ff_gpu_uploaded = true;
	                }
	                if (!ff_alpha_geom)
	                    ff_workspace.reserve_mueller(ff_batch_orient, ntheta);
	                ff_workspace.zero_mueller(ntheta);
	                printf("  GPU Mueller accumulation enabled (set BEM_FF_CPU_ACCUM=1 for CPU fallback)\n");
	            } else if (ff_alpha_mgpu) {
	                mgpu_M_accum.assign(16 * ntheta, 0.0);
	                mgpu_partial.assign((size_t)ff_mgpu, std::vector<double>(16 * ntheta, 0.0));
	                mgpu_ff.resize(ff_mgpu, 0);
	                mgpu_ws.resize(ff_mgpu, 0);
	                for (int gd = 0; gd < ff_mgpu; gd++) {
	                    CUDA_CHECK(cudaSetDevice(ff_devices[(size_t)gd]));
	                    mgpu_ff[gd] = new FFCacheGPU();
	                    mgpu_ff[gd]->upload(ff_cache);
	                    mgpu_ws[gd] = new FFBatchWorkspace();
	                    mgpu_ws[gd]->zero_mueller(ntheta);
	                }
	                CUDA_CHECK(cudaSetDevice(ff_original_device));
	                printf("  Multi-GPU Mueller accumulation enabled (%d GPUs)\n", ff_mgpu);
	            } else {
                if (!ff_gpu_uploaded) {
                    ff_gpu.upload(ff_cache);
                    ff_gpu_uploaded = true;
                }
                batch_Fv.resize((size_t)ff_batch_orient * 2 * ntheta * 3);
                S1.resize(ntheta);
                S2.resize(ntheta);
                S3.resize(ntheta);
                S4.resize(ntheta);
                M_orient.resize(16 * ntheta);
                printf("  CPU Mueller accumulation enabled\n");
            }

            auto rotate_z_minus = [](const Vec3& v, double ca, double sa) {
                return Vec3(ca * v.x + sa * v.y,
                           -sa * v.x + ca * v.y,
                            v.z);
            };

            std::vector<double> alpha_cos(alpha_avg), alpha_sin(alpha_avg);
            for (int ia = 0; ia < alpha_avg; ia++) {
                double alpha = 2.0 * M_PI * (double)ia / (double)alpha_avg;
                alpha_cos[ia] = cos(alpha);
                alpha_sin[ia] = sin(alpha);
            }
            bool orient_pack_omp = !bem_env_flag_enabled("BEM_ORIENT_PACK_SERIAL");

            auto flush_mueller_batch = [&]() {
                if (batch_count == 0)
                    return;
                if (ff_gpu_accum) {
                    if (bem_env_flag_enabled("BEM_FF_SEPARATE")) {
                        accumulate_farfield_mueller_batch_cuda_ws(
                            ff_gpu, ff_workspace,
                            batch_coeffs_J.data(), batch_coeffs_M.data(),
                            batch_r_hats.data(),
                            reinterpret_cast<const double*>(batch_e_par.data()),
                            reinterpret_cast<const double*>(batch_e_perp.data()),
                            batch_weights.data(),
                            k_ext, eta_ext,
                            batch_count * 2, batch_count, ntheta);
                    } else {
                        accumulate_farfield_mueller_direct_cuda_ws(
                            ff_gpu, ff_workspace,
                            batch_coeffs_J.data(), batch_coeffs_M.data(),
                            batch_r_hats.data(),
                            reinterpret_cast<const double*>(batch_e_par.data()),
                            reinterpret_cast<const double*>(batch_e_perp.data()),
                            batch_weights.data(),
                            k_ext, eta_ext,
                            batch_count * 2, batch_count, ntheta);
                    }
                } else {
                    compute_farfield_batch_cuda_ws(
                        ff_gpu, ff_workspace,
                        batch_coeffs_J.data(), batch_coeffs_M.data(),
                        batch_r_hats.data(),
                        k_ext, eta_ext,
                        batch_count * 2, batch_count, ntheta,
                        batch_Fv.data());

                    for (int bi = 0; bi < batch_count; bi++) {
                        double weight = batch_weights[bi];
                        cdouble* Fv_par  = &batch_Fv[(2*bi) * ntheta * 3];
                        cdouble* Fv_perp = &batch_Fv[(2*bi+1) * ntheta * 3];

                        for (int it = 0; it < ntheta; it++) {
                            Vec3& ep = batch_e_par[bi * ntheta + it];
                            Vec3& epp = batch_e_perp[bi * ntheta + it];

                            cdouble F_par_p  = Fv_par[it*3]*ep.x  + Fv_par[it*3+1]*ep.y  + Fv_par[it*3+2]*ep.z;
                            cdouble F_perp_p = Fv_par[it*3]*epp.x + Fv_par[it*3+1]*epp.y + Fv_par[it*3+2]*epp.z;
                            cdouble F_par_pp  = Fv_perp[it*3]*ep.x  + Fv_perp[it*3+1]*ep.y  + Fv_perp[it*3+2]*ep.z;
                            cdouble F_perp_pp = Fv_perp[it*3]*epp.x + Fv_perp[it*3+1]*epp.y + Fv_perp[it*3+2]*epp.z;

                            S2[it] = ik_val * F_par_p;
                            S4[it] = ik_val * F_perp_p;
                            S3[it] = ik_val * F_par_pp;
                            S1[it] = ik_val * F_perp_pp;
                        }

                        amplitude_to_mueller(S1.data(), S2.data(), S3.data(), S4.data(),
                                            ntheta, M_orient.data());

                        for (int i = 0; i < 16 * ntheta; i++)
                            M_avg[i] += weight * M_orient[i];
                    }
                }
                batch_count = 0;
            };

	            auto append_dense_farfield_sample = [&](int oi, const Mat3& RT, double ca, double sa, double weight) {
	                if (batch_count == ff_batch_orient)
	                    flush_mueller_batch();
	                std::complex<double>* X_par  = &B[oi * 2 * N2];
	                std::complex<double>* X_perp = &B[(oi * 2 + 1) * N2];
	                double inv_s = (unknown_m_scale == 1.0) ? 1.0 : (1.0 / unknown_m_scale);
	                int bi = batch_count++;
	                batch_weights[bi] = weight;

	                cdouble* J0 = &batch_coeffs_J[(2*bi) * N];
	                cdouble* M0 = &batch_coeffs_M[(2*bi) * N];
                cdouble* J1 = &batch_coeffs_J[(2*bi+1) * N];
                cdouble* M1 = &batch_coeffs_M[(2*bi+1) * N];
	                #pragma omp parallel for schedule(static) if(orient_pack_omp && N > 2048)
	                for (int i = 0; i < N; i++) {
	                    cdouble jp = X_par[i];
	                    cdouble mp = X_par[N + i] * inv_s;
	                    cdouble ju = X_perp[i];
	                    cdouble mu = X_perp[N + i] * inv_s;
	                    if (scattering_plane_yz) {
	                        J0[i] = ca * jp + sa * ju;
	                        M0[i] = ca * mp + sa * mu;
                        J1[i] = -sa * jp + ca * ju;
                        M1[i] = -sa * mp + ca * mu;
                    } else {
                        J0[i] = ca * jp - sa * ju;
                        M0[i] = ca * mp - sa * mu;
                        J1[i] = -sa * jp - ca * ju;
                        M1[i] = -sa * mp - ca * mu;
                    }
                }

                for (int it = 0; it < ntheta; it++) {
                    Vec3 rh = RT * rotate_z_minus(r_hat_lab[it], ca, sa);
                    int base = (bi * ntheta + it) * 3;
                    batch_r_hats[base]   = rh.x;
                    batch_r_hats[base+1] = rh.y;
                    batch_r_hats[base+2] = rh.z;
                    batch_e_par[bi * ntheta + it]  = RT * rotate_z_minus(e_theta_lab[it], ca, sa);
	                    batch_e_perp[bi * ntheta + it] = RT * rotate_z_minus(e_phi_lab, ca, sa);
	                }
	            };

	            int base_batch_count = 0;
	            auto flush_alpha_direct_batch = [&]() {
	                if (base_batch_count == 0)
	                    return;
	                if (ff_alpha_mgpu) {
	                    #ifdef _OPENMP
	                    #pragma omp parallel for schedule(static)
	                    #endif
	                    for (int gd = 0; gd < ff_mgpu; gd++) {
	                        int start = (base_batch_count * gd) / ff_mgpu;
	                        int end = (base_batch_count * (gd + 1)) / ff_mgpu;
	                        int count = end - start;
	                        if (count <= 0)
	                            continue;
	                        CUDA_CHECK(cudaSetDevice(ff_devices[(size_t)gd]));
	                        FFBatchWorkspace& local_ws = *mgpu_ws[(size_t)gd];
                        accumulate_farfield_mueller_alpha_geom_cuda_ws(
                            *mgpu_ff[(size_t)gd], local_ws,
                            batch_coeffs_J.data() + (size_t)start * 2 * N,
                            batch_coeffs_M.data() + (size_t)start * 2 * N,
                            batch_RT.data() + (size_t)start * 9,
                            r_hat_lab_flat.data(), e_theta_lab_flat.data(),
                            e_phi_lab_flat,
                            batch_weights.data() + (size_t)start,
                            alpha_cos.data(), alpha_sin.data(),
                            k_ext, eta_ext,
                            count, alpha_avg, ntheta, false);
	                    }
	                    for (int gd = 0; gd < ff_mgpu; gd++) {
	                        CUDA_CHECK(cudaSetDevice(ff_devices[(size_t)gd]));
	                        CUDA_CHECK(cudaStreamSynchronize(mgpu_ws[(size_t)gd]->stream));
	                    }
	                    CUDA_CHECK(cudaSetDevice(ff_original_device));
	                } else if (ff_alpha_geom) {
                    accumulate_farfield_mueller_alpha_geom_cuda_ws(
                        ff_gpu, ff_workspace,
                        batch_coeffs_J.data(), batch_coeffs_M.data(),
                        batch_RT.data(),
                        r_hat_lab_flat.data(), e_theta_lab_flat.data(),
                        e_phi_lab_flat,
                        batch_weights.data(),
                        alpha_cos.data(), alpha_sin.data(),
                        k_ext, eta_ext,
                        base_batch_count, alpha_avg, ntheta);
	                } else {
	                    accumulate_farfield_mueller_alpha_cuda_ws(
	                        ff_gpu, ff_workspace,
	                        batch_coeffs_J.data(), batch_coeffs_M.data(),
	                        batch_r_hats.data(),
	                        reinterpret_cast<const double*>(batch_e_par.data()),
	                        reinterpret_cast<const double*>(batch_e_perp.data()),
	                        batch_weights.data(),
	                        alpha_cos.data(), alpha_sin.data(),
	                        k_ext, eta_ext,
	                        base_batch_count, alpha_avg, ntheta);
	                }
	                base_batch_count = 0;
	            };

	            auto append_dense_alpha_orientation = [&](int oi, const Mat3& RT, double weight) {
	                if (base_batch_count == ff_base_batch_orient)
	                    flush_alpha_direct_batch();
	                std::complex<double>* X_par  = &B[oi * 2 * N2];
	                std::complex<double>* X_perp = &B[(oi * 2 + 1) * N2];
	                double inv_s = (unknown_m_scale == 1.0) ? 1.0 : (1.0 / unknown_m_scale);
	                int bi = base_batch_count++;
	                double* RT_out = &batch_RT[(size_t)bi * 9];
	                for (int r = 0; r < 3; r++)
	                    for (int c = 0; c < 3; c++)
	                        RT_out[r * 3 + c] = RT.m[r][c];

	                cdouble* J0 = &batch_coeffs_J[(2*bi) * N];
	                cdouble* M0 = &batch_coeffs_M[(2*bi) * N];
	                cdouble* J1 = &batch_coeffs_J[(2*bi+1) * N];
	                cdouble* M1 = &batch_coeffs_M[(2*bi+1) * N];
	                #pragma omp parallel for schedule(static) if(orient_pack_omp && N > 2048)
	                for (int i = 0; i < N; i++) {
	                    J0[i] = X_par[i];
	                    M0[i] = X_par[N + i] * inv_s;
	                    J1[i] = X_perp[i];
	                    M1[i] = X_perp[N + i] * inv_s;
	                }

	                double sample_weight = weight / (double)alpha_avg;
	                if (ff_alpha_geom)
	                    batch_weights[bi] = weight;
	                for (int ia = 0; ia < alpha_avg; ia++) {
	                    double ca = alpha_cos[ia], sa = alpha_sin[ia];
	                    int si = bi * alpha_avg + ia;
	                    if (!ff_alpha_geom)
	                        batch_weights[si] = sample_weight;
	                    if (ff_alpha_geom)
	                        continue;
	                    for (int it = 0; it < ntheta; it++) {
	                        Vec3 rh = RT * rotate_z_minus(r_hat_lab[it], ca, sa);
	                        int base = (si * ntheta + it) * 3;
	                        batch_r_hats[base]   = rh.x;
	                        batch_r_hats[base+1] = rh.y;
	                        batch_r_hats[base+2] = rh.z;
	                        batch_e_par[si * ntheta + it]  = RT * rotate_z_minus(e_theta_lab[it], ca, sa);
	                        batch_e_perp[si * ntheta + it] = RT * rotate_z_minus(e_phi_lab, ca, sa);
	                    }
	                }
	            };

	            int orient_progress_step = std::max(10, n_total / 10);
	            orient_progress_step = std::max(1, bem_env_int("BEM_ORIENT_PROGRESS", orient_progress_step));
	            for (int oi = 0; oi < n_total; oi++) {
	                Mat3& RT = orients[oi].RT;
	                if (ff_alpha_direct) {
	                    append_dense_alpha_orientation(oi, RT, orients[oi].weight);
	                } else if (alpha_avg == 1) {
	                    append_dense_farfield_sample(oi, RT, 1.0, 0.0, orients[oi].weight);
	                } else {
	                    double sample_weight = orients[oi].weight / (double)alpha_avg;
                    for (int ia = 0; ia < alpha_avg; ia++) {
                        append_dense_farfield_sample(oi, RT, alpha_cos[ia], alpha_sin[ia], sample_weight);
                    }
                }
	                if ((oi + 1) % orient_progress_step == 0 || oi == n_total - 1)
	                    printf("    Far-field orient %d/%d\n", oi + 1, n_total);
	            }
	            if (ff_alpha_direct)
	                flush_alpha_direct_batch();
	            else
	                flush_mueller_batch();
	            if (ff_alpha_mgpu) {
	                std::fill(mgpu_M_accum.begin(), mgpu_M_accum.end(), 0.0);
	                for (int gd = 0; gd < ff_mgpu; gd++) {
	                    CUDA_CHECK(cudaSetDevice(ff_devices[(size_t)gd]));
	                    mgpu_ws[(size_t)gd]->download_mueller(mgpu_partial[(size_t)gd].data(), ntheta);
	                    for (int i = 0; i < 16 * ntheta; i++)
	                        mgpu_M_accum[i] += mgpu_partial[(size_t)gd][i];
	                }
	                for (int i = 0; i < 16 * ntheta; i++)
	                    M_avg[i] = mgpu_M_accum[i];
	                for (int gd = 0; gd < ff_mgpu; gd++) {
	                    CUDA_CHECK(cudaSetDevice(ff_devices[(size_t)gd]));
	                    delete mgpu_ws[(size_t)gd];
	                    delete mgpu_ff[(size_t)gd];
	                }
	                CUDA_CHECK(cudaSetDevice(ff_original_device));
	            } else if (ff_gpu_accum) {
	                ff_workspace.download_mueller(M_avg.data(), ntheta);
	            }
	            if (ff_single_gpu_pinned)
	                CUDA_CHECK(cudaSetDevice(ff_original_device));

            time_farfield = ff_timer.elapsed_s();
            if (alpha_avg > 1)
                printf("  Averaged over %d solved orientations x %d alpha samples.\n", n_total, alpha_avg);
            else
                printf("  Averaged over %d orientations.\n", n_total);
        }
    }

    if (export_currents_file && !currents_exported)
        printf("  Currents export skipped: use --single, or the sphere shortcut, to save one physical orientation.\n");

    const bool orient_project_requested =
        bem_env_flag_enabled("BEM_ORIENT_PROJECT", false) &&
        !bem_env_flag_enabled("BEM_NO_ORIENT_PROJECT");
    const char* random_orientation_projection = "not_applicable";
    if (!single_orient && orient_project_requested) {
        project_random_orientation_mueller(M_avg.data(), ntheta);
        random_orientation_projection = "applied";
        printf("  Random-orientation Mueller projection applied.\n");
    } else if (!single_orient) {
        random_orientation_projection = "disabled";
        printf("  Random-orientation Mueller projection disabled.\n");
    }

    double time_total = total_timer.elapsed_s();

    write_json(outfile, M_avg.data(), theta_arr.data(), ntheta,
               ka, n_re, n_im, refinements,
               shape, obj_file, prism_aspect, edge_refine,
               n_alpha, n_beta, n_gamma, alpha_avg,
               output_orient_start, output_orient_count, output_orient_total,
               output_orientation_weight_sum, output_gmres_matvecs,
               output_gmres_converged_systems, output_gmres_nonconverged_systems,
               output_gmres_stagnation_stops, output_gmres_numerical_breakdowns,
               output_gmres_restored_best_iterates,
               output_gmres_max_cycle_exhaustions,
               output_gmres_max_final_relres,
               fmm_digits, max_leaf, gmres_restart, gmres_tol, gmres_max_cycles,
               requested_fmm_digits, requested_gmres_tol,
               fmm_digits_set, gmres_tol_set, accuracy_policy_adjusted,
               random_orientation_projection,
               output_farfield_mode,
               solver_name(solver), acc_policy.profile,
               output_krylov_solver,
               requested_system_kind, system_kind,
               std::strcmp(requested_system_kind, system_kind) != 0,
               quad_order,
               unknown_m_scale, row_h_scale, int_op_sign, k_identity,
               use_prec,
               use_prec && bem_env_flag_enabled("BEM_PREC_BLOCK"),
               !use_prec && (bem_env_flag_enabled("BEM_GMRES_DEVICE") || use_bicgstab || use_cgs_rr || use_krylov_auto || use_krylov_hybrid),
               prec_policy.reason,
               mesh_quality.vertices, mesh_quality.triangles,
               mesh_quality.skinny_triangles, mesh_quality.min_angle_deg,
               mesh_quality.max_aspect_ratio,
               mesh_quality.feature_edges_30deg,
               mesh_quality.feature_edge_fraction,
               mesh_quality.max_dihedral_deg,
               mesh_quality.mean_feature_dihedral_deg,
               mesh_quality.max_adjacent_area_ratio,
               mesh_quality.near_touch_checked, mesh_quality.near_touch_ratio,
               mesh_quality.near_touch_pairs,
               mesh_quality.self_panel_count,
               mesh_quality.edge_adjacent_pair_count,
               mesh_quality.vertex_adjacent_pair_count,
               mesh_quality.near_disjoint_pair_count,
               mesh_quality.taylor_duffy_candidate_count,
               mesh_quality.recommended_min_quad_order,
               mesh_quality.recommended_mesh_strategy.c_str(),
               mesh_quality.recommended_mesh_action.c_str(),
               mesh_quality.voxel_surface_like,
               mesh_quality.requires_remesh,
               mesh_quality.edge_refine_requested, mesh_quality.edge_refine_applied,
               mesh_quality.edge_refine_uniform_fallback,
               mesh_quality.pass_default_gate,
               time_assembly, time_solve, time_farfield, time_total);

    if (use_fmm && output_gmres_nonconverged_systems > 0 &&
        !bem_env_flag_enabled("BEM_ALLOW_NONCONVERGED", false)) {
        fprintf(stderr,
                "Error: GMRES did not converge for %d solved system(s); "
                "wrote diagnostic JSON to %s. Set BEM_ALLOW_NONCONVERGED=1 "
                "only for explicit diagnostics.\n",
                output_gmres_nonconverged_systems, outfile);
        return 2;
    }

    printf("\n=== Done ===\n");
    printf("  Assembly: %.1fs\n", time_assembly);
    printf("  Solve:    %.1fs\n", time_solve);
    printf("  Farfield: %.1fs\n", time_farfield);
    printf("  Total:    %.1fs\n", time_total);

    return 0;
}
