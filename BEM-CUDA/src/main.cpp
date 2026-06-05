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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

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
    printf("  --prism-aspect F  Hex prism h/Dx, ADDA convention (default: 1)\n");
    printf("  --edge-refine N   Prism local edge-refinement passes (default: 0)\n");
    printf("  --ref N         Icosphere refinement level (default: 3)\n");
    printf("  --orient NA NB NG  Orientation quadrature (default: 8 8 1)\n");
    printf("  --orient-start I    First orientation index for chunked averaging (default: 0)\n");
    printf("  --orient-count N    Number of orientations in chunk (default: all)\n");
    printf("  --ntheta N      Number of scattering angles (default: 181)\n");
    printf("  --scat-plane P  Single-orient scattering plane: yz or xz (default: yz, ADDA convention)\n");
    printf("  --quad N        Quadrature order: 4, 7, 13 (default: 7; hex_prism fastest auto: 4)\n");
    printf("  --out FILE      Output JSON file (default: result.json)\n");
    printf("  --single        Single orientation (no averaging)\n");
    printf("  --force-orient  Force explicit orientation loop for sphere mesh\n");
    printf("  --accurate      Use conservative hex_prism defaults: quad7, digits3, tol1e-3, leaf128\n");
    printf("  --solver TYPE   Solver backend: auto, dense, fmm, pfft, spfft (spfft falls back to fmm; default: auto)\n");
    printf("  --system TYPE   Linear system: pmchwt, balanced, muller, muller-balanced, muller2, or muller2-balanced (default: pmchwt)\n");
    printf("  --fmm           Use FMM+GMRES instead of dense LU\n");
    printf("  --fmm-digits N  FMM accuracy digits (default: 3; hex_prism fastest auto: 2)\n");
    printf("  --gmres-tol F   GMRES relative tolerance (default: 1e-4; hex_prism fastest auto: 5e-2)\n");
    printf("  --gmres-restart N  GMRES restart (default: 150; hex_prism fast auto: 100)\n");
    printf("  --max-leaf N    FMM max particles per leaf (default: 128; hex_prism fastest auto: 96)\n");
    printf("  --no-prec       Disable automatic FMM block-Jacobi preconditioner\n");
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

static void recycle_initial_guess(const std::vector<std::vector<cdouble>>& hist_b,
                                  const std::vector<std::vector<cdouble>>& hist_x,
                                  const cdouble* b, int n,
                                  cdouble* x)
{
    int m = (int)hist_b.size();
    if (m == 0) {
        std::fill(x, x + n, cdouble(0));
        return;
    }

    std::vector<cdouble> G(m * m), rhs(m);
    for (int i = 0; i < m; i++) {
        cdouble bi(0);
        for (int k = 0; k < n; k++)
            bi += std::conj(hist_b[i][k]) * b[k];
        rhs[i] = bi;
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
    double lambda = (m > 0) ? trace * 1e-12 / m : 0.0;
    for (int i = 0; i < m; i++)
        G[i * m + i] += lambda;

    if (!solve_small_linear(G, rhs, m)) {
        std::fill(x, x + n, cdouble(0));
        return;
    }

    std::fill(x, x + n, cdouble(0));
    for (int i = 0; i < m; i++) {
        const std::vector<cdouble>& xi = hist_x[i];
        cdouble ci = rhs[i];
        for (int k = 0; k < n; k++)
            x[k] += ci * xi[k];
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

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
#ifdef _OPENMP
    if (!std::getenv("OMP_NUM_THREADS")) {
        int threads = std::min(8, omp_get_max_threads());
        omp_set_num_threads(threads);
    }
#endif
    // Default parameters
    double ka = 0;
    double n_re = 1.3116, n_im = 0.0;
    const char* shape = "sphere";
    double prism_aspect = 1.0;
    int edge_refine = 0;
    int refinements = 3;
    int n_alpha = 8, n_beta = 8, n_gamma = 1;
    int orient_start = 0, orient_count = -1;
    int ntheta = 181;
    const char* scat_plane = "yz";
    int quad_order = 7;
    bool quad_order_set = false;
    const char* outfile = "result.json";
    bool single_orient = false;
    bool force_orient = false;
    bool accurate_mode = false;
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
    int max_leaf = 128;
    bool max_leaf_set = false;
    const char* system_kind = "pmchwt";

    // Parse CLI
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ka") == 0 && i+1 < argc) {
            ka = atof(argv[++i]);
        } else if (strcmp(argv[i], "--ri") == 0 && i+2 < argc) {
            n_re = atof(argv[++i]);
            n_im = atof(argv[++i]);
        } else if (strcmp(argv[i], "--shape") == 0 && i+1 < argc) {
            shape = argv[++i];
        } else if (strcmp(argv[i], "--prism-aspect") == 0 && i+1 < argc) {
            prism_aspect = atof(argv[++i]);
        } else if (strcmp(argv[i], "--edge-refine") == 0 && i+1 < argc) {
            edge_refine = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ref") == 0 && i+1 < argc) {
            refinements = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--orient") == 0 && i+3 < argc) {
            n_alpha = atoi(argv[++i]);
            n_beta  = atoi(argv[++i]);
            n_gamma = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--orient-start") == 0 && i+1 < argc) {
            orient_start = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--orient-count") == 0 && i+1 < argc) {
            orient_count = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ntheta") == 0 && i+1 < argc) {
            ntheta = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--scat-plane") == 0 && i+1 < argc) {
            scat_plane = argv[++i];
        } else if (strcmp(argv[i], "--quad") == 0 && i+1 < argc) {
            quad_order = atoi(argv[++i]);
            quad_order_set = true;
        } else if (strcmp(argv[i], "--out") == 0 && i+1 < argc) {
            outfile = argv[++i];
        } else if (strcmp(argv[i], "--single") == 0) {
            single_orient = true;
        } else if (strcmp(argv[i], "--force-orient") == 0) {
            force_orient = true;
        } else if (strcmp(argv[i], "--accurate") == 0) {
            accurate_mode = true;
        } else if (strcmp(argv[i], "--fmm") == 0) {
            solver = SOLVER_FMM;
            solver_explicit = true;
        } else if (strcmp(argv[i], "--solver") == 0 && i+1 < argc) {
            const char* solver_arg = argv[++i];
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
        } else if (strcmp(argv[i], "--system") == 0 && i+1 < argc) {
            system_kind = argv[++i];
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
        } else if (strcmp(argv[i], "--fmm-digits") == 0 && i+1 < argc) {
            fmm_digits = atoi(argv[++i]);
            fmm_digits_set = true;
        } else if (strcmp(argv[i], "--gmres-tol") == 0 && i+1 < argc) {
            gmres_tol = atof(argv[++i]);
            gmres_tol_set = true;
        } else if (strcmp(argv[i], "--gmres-restart") == 0 && i+1 < argc) {
            gmres_restart = atoi(argv[++i]);
            gmres_restart_set = true;
        } else if (strcmp(argv[i], "--max-leaf") == 0 && i+1 < argc) {
            max_leaf = atoi(argv[++i]);
            max_leaf_set = true;
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
    bool scattering_plane_yz = true;
    if (strcmp(scat_plane, "yz") == 0) {
        scattering_plane_yz = true;
    } else if (strcmp(scat_plane, "xz") == 0) {
        scattering_plane_yz = false;
    } else {
        fprintf(stderr, "Error: --scat-plane must be yz or xz\n");
        return 1;
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
    double eta_int = 1.0 / std::abs(m);
    double unknown_m_scale = 1.0;
    double row_h_scale = 1.0;
    double int_op_sign = 1.0;
    double k_identity = 0.0;
    if (strcmp(system_kind, "muller") == 0 || strcmp(system_kind, "muller-balanced") == 0 ||
        strcmp(system_kind, "muller2") == 0 || strcmp(system_kind, "muller2-balanced") == 0)
        int_op_sign = -1.0;
    if (strcmp(system_kind, "muller2") == 0 || strcmp(system_kind, "muller2-balanced") == 0)
        k_identity = 1.0;
    if (strcmp(system_kind, "balanced") == 0 || strcmp(system_kind, "muller-balanced") == 0 ||
        strcmp(system_kind, "muller2-balanced") == 0) {
        unknown_m_scale = std::abs(m);
        row_h_scale = eta_int;
    }
    if (const char* env = std::getenv("BEM_SYSTEM_INT_SIGN"))
        int_op_sign = atof(env);
    if (const char* env = std::getenv("BEM_SYSTEM_K_IDENTITY"))
        k_identity = atof(env);
    if (const char* env = std::getenv("BEM_SYSTEM_M_SCALE"))
        unknown_m_scale = atof(env);
    if (const char* env = std::getenv("BEM_SYSTEM_H_ROW_SCALE"))
        row_h_scale = atof(env);

    printf("=== BEM-CUDA Solver ===\n");
    printf("  ka = %.4f, m = %.4f + %.4fi\n", ka, n_re, n_im);
    printf("  k_ext = %.4f, k_int = %.4f + %.4fi\n",
           k_ext.real(), k_int.real(), k_int.imag());
    printf("  eta_ext = %.4f, eta_int = %.4f\n", eta_ext, eta_int);
    printf("  System: %s", system_kind);
    if (unknown_m_scale != 1.0 || row_h_scale != 1.0)
        printf(" (M_scaled=%.4g*M, H-row scale=%.4g)", unknown_m_scale, row_h_scale);
    if (int_op_sign < 0.0)
        printf(" (interior operator sign=-1)");
    if (k_identity != 0.0)
        printf(" (K identity jump=%.3g)", k_identity);
    printf("\n");
    bool is_hex_prism = (strcmp(shape, "hex_prism") == 0 || strcmp(shape, "prism6") == 0);
    if (is_hex_prism && accurate_mode && !quad_order_set)
        quad_order = 7;
    else if (is_hex_prism && !quad_order_set)
        quad_order = 4;
    printf("  Shape: %s", shape);
    if (strcmp(shape, "hex_prism") == 0)
        printf(" (h/Dx=%.3f, edge_refine=%d)", prism_aspect, edge_refine);
    printf("\n");
    printf("  Refinements: %d, Quad order: %d\n", refinements, quad_order);
    if (single_orient)
        printf("  Single orientation, scattering plane: %s\n", scattering_plane_yz ? "yz" : "xz");
    else
        printf("  Orientations: %d x %d x %d = %d\n",
               n_alpha, n_beta, n_gamma, n_alpha * n_beta * n_gamma);

    // 1. Generate mesh
    Timer mesh_timer;
    double radius = 1.0;
    Mesh mesh;
    if (strcmp(shape, "sphere") == 0) {
        mesh = icosphere(radius, refinements);
    } else if (strcmp(shape, "hex_prism") == 0 || strcmp(shape, "prism6") == 0) {
        mesh = regular_prism(6, prism_aspect, refinements, radius, edge_refine);
    } else {
        fprintf(stderr, "Error: unsupported --shape %s\n", shape);
        return 1;
    }
    printf("  Mesh: %d vertices, %d triangles (%.1fms)\n",
           mesh.nv(), mesh.nt(), mesh_timer.elapsed_ms());
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
    if (solver == SOLVER_SPFFT) {
        printf("  [SurfPFFT] Disabled: cross-face direct P2P is slower than FMM on current meshes; using FMM backend\n");
        solver = SOLVER_FMM;
    }

    use_pfft = (solver == SOLVER_PFFT);
    use_spfft = (solver == SOLVER_SPFFT);
    use_fmm = (solver == SOLVER_FMM || solver == SOLVER_PFFT || solver == SOLVER_SPFFT);

    if (use_fmm && solver == SOLVER_FMM && is_hex_prism) {
        if (accurate_mode) {
            if (!fmm_digits_set)
                fmm_digits = 3;
            if (!max_leaf_set)
                max_leaf = 128;
            if (!gmres_tol_set)
                gmres_tol = 1e-3;
            if (!gmres_restart_set)
                gmres_restart = 150;
        } else {
        if (!fmm_digits_set)
            fmm_digits = 2;
        if (!max_leaf_set)
            max_leaf = 96;
        if (!gmres_tol_set)
            gmres_tol = 5e-2;
        if (!gmres_restart_set)
            gmres_restart = 100;
        }
    }

    if (use_fmm && is_hex_prism && refinements >= 4 && !gmres_restart_set)
        gmres_restart = 300;
    if (use_fmm && is_hex_prism && gmres_tol <= 1e-3 && !gmres_restart_set)
        gmres_restart = 200;
    if (use_fmm && is_hex_prism && !std::getenv("BEM_GMRES_REORTH"))
        setenv("BEM_GMRES_REORTH", "0", 0);
    if (use_fmm && is_hex_prism && !std::getenv("BEM_FMM_BATCH4") &&
        !std::getenv("BEM_FMM_NO_BATCH4"))
        setenv("BEM_FMM_BATCH4", "1", 0);
    if (use_fmm && std::getenv("BEM_FMM_BATCH4"))
        setenv("BEM_FMM_ALLOC_BATCH4", "1", 0);

    use_prec = false;
    if (use_fmm && !no_prec) {
        use_prec = true;
        if (solver == SOLVER_PFFT && !std::getenv("BEM_PREC_FORCE"))
            use_prec = false;
        if (is_hex_prism && !std::getenv("BEM_PREC_FORCE"))
            use_prec = false;
        if (N < 512 && !std::getenv("BEM_PREC_FORCE"))
            use_prec = false;
    }

    if (use_fmm && use_prec && is_hex_prism) {
        if (!std::getenv("BEM_PREC_BLOCK"))
            setenv("BEM_PREC_BLOCK", "1", 0);
        if (!std::getenv("BEM_PREC_BLOCK_SIZE"))
            setenv("BEM_PREC_BLOCK_SIZE", "6", 0);
        if (!std::getenv("BEM_PREC_SWEEPS"))
            setenv("BEM_PREC_SWEEPS", "2", 0);
        if (!std::getenv("BEM_PREC_OMEGA"))
            setenv("BEM_PREC_OMEGA", "1.0", 0);
    }

    if (use_fmm) {
        const char* prec_name = use_prec ? (is_hex_prism ? ", Schwarz prec" : ", block-Jacobi prec") : "";
        printf("  Mode: %s+GMRES (digits=%d, tol=%.0e, restart=%d, max_leaf=%d%s)\n",
               solver_name(solver), fmm_digits, gmres_tol, gmres_restart, max_leaf, prec_name);
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

    double time_assembly = 0, time_solve = 0, time_farfield = 0;

    if (use_fmm) {
        // ============================================================
        // FMM + GMRES path
        // ============================================================
        Timer asm_timer;
        BemFmmOperator fmm_op;
        fmm_op.init(rwg, mesh, k_ext, k_int, eta_ext, eta_int,
                     quad_order, fmm_digits, max_leaf, use_pfft, use_spfft);
        fmm_op.unknown_m_scale = unknown_m_scale;
        fmm_op.row_h_scale = row_h_scale;
        fmm_op.int_op_sign = int_op_sign;
        fmm_op.k_identity = k_identity;

        // Build preconditioner if requested
        NearFieldPrecond* precond_ptr = nullptr;
        NearFieldPrecond precond;
        if (use_prec) {
            precond.build(fmm_op);
            precond_ptr = &precond;
        }

        time_assembly = asm_timer.elapsed_s();

        Timer solve_timer;

        if (single_orient || sphere_orientation_shortcut) {
            Vec3 k_hat(0, 0, 1);
            Vec3 E_par  = scattering_plane_yz ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
            Vec3 E_perp = scattering_plane_yz ? Vec3(1, 0, 0) : Vec3(0, 1, 0);

            // Solve for both polarizations
            std::vector<cdouble> b_par(N2), b_perp(N2);
            compute_rhs_planewave(rwg, mesh, k_ext, eta_ext, E_par, k_hat, quad_order, b_par.data());
            compute_rhs_planewave(rwg, mesh, k_ext, eta_ext, E_perp, k_hat, quad_order, b_perp.data());
            if (row_h_scale != 1.0) {
                for (int i = 0; i < N; i++) {
                    b_par[N + i] *= row_h_scale;
                    b_perp[N + i] *= row_h_scale;
                }
            }

            std::vector<cdouble> x_par(N2, cdouble(0)), x_perp(N2, cdouble(0));
            printf("\n  Solving both polarizations (paired GMRES)...\n");
            gmres_solve_paired(fmm_op,
                b_par.data(), b_perp.data(),
                x_par.data(), x_perp.data(),
                gmres_restart, gmres_tol, 300, true, precond_ptr);

            time_solve = solve_timer.elapsed_s();

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

            double k2 = std::norm(k_ext);
            for (int i = 0; i < 16 * ntheta; i++)
                M_avg[i] /= k2;

            time_farfield = ff_timer.elapsed_s();
        } else {
            // Orientation averaging with GMRES
            std::vector<Orientation> orients = generate_orientations(n_alpha, n_beta, n_gamma);
            reorder_orientations_nearest(orients);
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

            // Far-field GPU cache
            FFCacheGPU ff_gpu;
            ff_gpu.upload(ff_cache);

            // Lab-frame scattering vectors
            std::vector<Vec3> r_hat_lab(ntheta), e_theta_lab(ntheta);
            Vec3 e_phi_lab = scattering_plane_yz ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
            for (int it = 0; it < ntheta; it++) {
                double ct = cos(theta_arr[it]), st = sin(theta_arr[it]);
                if (scattering_plane_yz) {
                    r_hat_lab[it] = Vec3(0, st, ct);
                    e_theta_lab[it] = Vec3(0, ct, -st);
                } else {
                    r_hat_lab[it] = Vec3(st, 0, ct);
                    e_theta_lab[it] = Vec3(ct, 0, -st);
                }
            }

            printf("\n  Solving %d orientations x 2 polarizations with GMRES...\n", n_total);

            std::vector<cdouble> x_par(N2, cdouble(0)), x_perp(N2, cdouble(0));
            std::vector<cdouble> b_par(N2), b_perp(N2);
            std::vector<std::vector<cdouble>> hist_b, hist_x;
            const int max_recycle = 12;
            long long orient_matvecs = 0;

            const char* ff_batch_env = std::getenv("BEM_FF_BATCH");
            int ff_batch_orient = ff_batch_env ? std::max(1, atoi(ff_batch_env)) : 64;
            ff_batch_orient = std::min(ff_batch_orient, std::max(1, n_total));
            printf("  Streaming far-field batch: %d orientations (set BEM_FF_BATCH to tune memory)\n",
                   ff_batch_orient);

            int batch_count = 0;
            std::vector<int> batch_orient_idx(ff_batch_orient);
            std::vector<cdouble> batch_coeffs_J((size_t)ff_batch_orient * 2 * N);
            std::vector<cdouble> batch_coeffs_M((size_t)ff_batch_orient * 2 * N);
            std::vector<double> batch_r_hats((size_t)ff_batch_orient * ntheta * 3);
            std::vector<Vec3> batch_e_par((size_t)ff_batch_orient * ntheta);
            std::vector<Vec3> batch_e_perp((size_t)ff_batch_orient * ntheta);
            std::vector<cdouble> batch_Fv((size_t)ff_batch_orient * 2 * ntheta * 3);
            std::vector<cdouble> S1(ntheta), S2(ntheta), S3(ntheta), S4(ntheta);
            std::vector<double> M_orient(16 * ntheta);
            cdouble ik_val = cdouble(0, -1) * k_ext;
            double k2 = std::norm(k_ext);

            auto flush_farfield_batch = [&]() {
                if (batch_count == 0)
                    return;

                Timer ff_timer;
                int batch_calls = batch_count * 2;
                compute_farfield_batch_cuda(ff_gpu,
                                            batch_coeffs_J.data(), batch_coeffs_M.data(),
                                            batch_r_hats.data(),
                                            k_ext, eta_ext,
                                            batch_calls, batch_count, ntheta,
                                            batch_Fv.data());

                for (int bi = 0; bi < batch_count; bi++) {
                    int oi = batch_orient_idx[bi];
                    double weight = orients[oi].weight;
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
                        M_avg[i] += weight * M_orient[i] / k2;
                }

                time_farfield += ff_timer.elapsed_s();
                batch_count = 0;
            };

            for (int oi = 0; oi < n_total; oi++) {
                Mat3& RT = orients[oi].RT;
                Vec3 k_hat = RT * Vec3(0, 0, 1);
                Vec3 e_par = RT * Vec3(1, 0, 0);
                Vec3 e_perp = RT * Vec3(0, 1, 0);

                compute_rhs_planewave(rwg, mesh, k_ext, eta_ext, e_par, k_hat, quad_order, b_par.data());
                compute_rhs_planewave(rwg, mesh, k_ext, eta_ext, e_perp, k_hat, quad_order, b_perp.data());
                if (row_h_scale != 1.0) {
                    for (int i = 0; i < N; i++) {
                        b_par[N + i] *= row_h_scale;
                        b_perp[N + i] *= row_h_scale;
                    }
                }

                recycle_initial_guess(hist_b, hist_x, b_par.data(), N2, x_par.data());
                recycle_initial_guess(hist_b, hist_x, b_perp.data(), N2, x_perp.data());

                Timer orient_solve_timer;
                int mv = gmres_solve_paired(fmm_op,
                    b_par.data(), b_perp.data(),
                    x_par.data(), x_perp.data(),
                    gmres_restart, gmres_tol, 300, false, precond_ptr);
                time_solve += orient_solve_timer.elapsed_s();
                orient_matvecs += mv;

                push_history(hist_b, hist_x, b_par.data(), x_par.data(), N2, max_recycle);
                push_history(hist_b, hist_x, b_perp.data(), x_perp.data(), N2, max_recycle);

                int bi = batch_count++;
                batch_orient_idx[bi] = oi;
                memcpy(&batch_coeffs_J[(2*bi) * N], x_par.data(), N * sizeof(cdouble));
                if (unknown_m_scale == 1.0) {
                    memcpy(&batch_coeffs_M[(2*bi) * N], x_par.data() + N, N * sizeof(cdouble));
                } else {
                    double inv_s = 1.0 / unknown_m_scale;
                    for (int i = 0; i < N; i++)
                        batch_coeffs_M[(2*bi) * N + i] = x_par[N + i] * inv_s;
                }
                memcpy(&batch_coeffs_J[(2*bi+1) * N], x_perp.data(), N * sizeof(cdouble));
                if (unknown_m_scale == 1.0) {
                    memcpy(&batch_coeffs_M[(2*bi+1) * N], x_perp.data() + N, N * sizeof(cdouble));
                } else {
                    double inv_s = 1.0 / unknown_m_scale;
                    for (int i = 0; i < N; i++)
                        batch_coeffs_M[(2*bi+1) * N + i] = x_perp[N + i] * inv_s;
                }

                for (int it = 0; it < ntheta; it++) {
                    Vec3 rh = RT * r_hat_lab[it];
                    int base = (bi * ntheta + it) * 3;
                    batch_r_hats[base]   = rh.x;
                    batch_r_hats[base+1] = rh.y;
                    batch_r_hats[base+2] = rh.z;
                    batch_e_par[bi * ntheta + it]  = RT * e_theta_lab[it];
                    batch_e_perp[bi * ntheta + it] = RT * e_phi_lab;
                }

                if (batch_count == ff_batch_orient)
                    flush_farfield_batch();

                if ((oi + 1) % 10 == 0 || oi == n_total - 1)
                    printf("    Orient %d/%d done (avg %.1f matvec/orient)\n",
                           oi + 1, n_total, (double)orient_matvecs / (oi + 1));
            }
            flush_farfield_batch();
            printf("  Averaged over %d orientations.\n", n_total);
        }

        fmm_op.cleanup();

    } else {
        // ============================================================
        // Dense LU path (original code)
        // ============================================================
        Timer asm_timer;
        std::vector<std::complex<double>> Z(N2 * N2);
        assemble_pmchwt(rwg, mesh, k_ext, k_int, eta_ext, eta_int,
                        quad_order, Z.data(), NULL, NULL);
        time_assembly = asm_timer.elapsed_s();

        Timer solve_timer;

        if (single_orient || sphere_orientation_shortcut) {
            Vec3 k_hat(0, 0, 1);
            Vec3 E_par  = scattering_plane_yz ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
            Vec3 E_perp = scattering_plane_yz ? Vec3(1, 0, 0) : Vec3(0, 1, 0);

            std::vector<int> ipiv(N2);
            lu_factorize_cuda(Z.data(), N2, ipiv.data());

            std::vector<std::complex<double>> B(N2 * 2);
            compute_rhs_planewave(rwg, mesh, k_ext, eta_ext, E_par, k_hat, quad_order, &B[0]);
            compute_rhs_planewave(rwg, mesh, k_ext, eta_ext, E_perp, k_hat, quad_order, &B[N2]);

            lu_solve_cuda(Z.data(), ipiv.data(), N2, B.data(), 2);
            time_solve = solve_timer.elapsed_s();

            Timer ff_timer;
            std::complex<double>* J_par  = &B[0];
            std::complex<double>* M_par  = &B[N];
            std::complex<double>* J_perp = &B[N2];
            std::complex<double>* M_perp = &B[N2 + N];

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

            double k2 = std::norm(k_ext);
            for (int i = 0; i < 16 * ntheta; i++)
                M_avg[i] /= k2;

            time_farfield = ff_timer.elapsed_s();

        } else {
            // Orientation averaging (batched)
            std::vector<Orientation> orients = generate_orientations(n_alpha, n_beta, n_gamma);
            reorder_orientations_nearest(orients);
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

            printf("\n  Building %d RHS vectors...\n", n_total * 2);

            // Phase 1: Build all RHS
            std::vector<std::complex<double>> B(N2 * n_total * 2, 0);
            for (int oi = 0; oi < n_total; oi++) {
                Mat3& RT = orients[oi].RT;
                Vec3 k_hat = RT * Vec3(0, 0, 1);
                Vec3 e_par = RT * Vec3(1, 0, 0);
                Vec3 e_perp = RT * Vec3(0, 1, 0);

                compute_rhs_planewave(rwg, mesh, k_ext, eta_ext, e_par, k_hat,
                                      quad_order, &B[oi * 2 * N2]);
                compute_rhs_planewave(rwg, mesh, k_ext, eta_ext, e_perp, k_hat,
                                      quad_order, &B[(oi * 2 + 1) * N2]);
            }

            // Phase 2: LU solve all at once
            printf("  Solving %d RHS with LU...\n", n_total * 2);
            lu_solve_full(Z.data(), N2, B.data(), n_total * 2);
            time_solve = solve_timer.elapsed_s();

            // Phase 3: Far-field + Mueller accumulation (GPU batched)
            Timer ff_timer;
            int n_calls = n_total * 2;
            printf("  Computing GPU far-field: %d calls x %d dirs...\n", n_calls, ntheta);

            FFCacheGPU ff_gpu;
            ff_gpu.upload(ff_cache);

            std::vector<Vec3> r_hat_lab(ntheta), e_theta_lab(ntheta);
            Vec3 e_phi_lab = scattering_plane_yz ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
            for (int it = 0; it < ntheta; it++) {
                double ct = cos(theta_arr[it]), st = sin(theta_arr[it]);
                if (scattering_plane_yz) {
                    r_hat_lab[it] = Vec3(0, st, ct);
                    e_theta_lab[it] = Vec3(0, ct, -st);
                } else {
                    r_hat_lab[it] = Vec3(st, 0, ct);
                    e_theta_lab[it] = Vec3(ct, 0, -st);
                }
            }

            std::vector<std::complex<double>> all_coeffs_J(n_calls * N);
            std::vector<std::complex<double>> all_coeffs_M(n_calls * N);
            for (int oi = 0; oi < n_total; oi++) {
                std::complex<double>* X_par  = &B[oi * 2 * N2];
                std::complex<double>* X_perp = &B[(oi * 2 + 1) * N2];
                memcpy(&all_coeffs_J[(2*oi) * N],     X_par,      N * sizeof(std::complex<double>));
                memcpy(&all_coeffs_M[(2*oi) * N],     X_par + N,  N * sizeof(std::complex<double>));
                memcpy(&all_coeffs_J[(2*oi+1) * N],   X_perp,     N * sizeof(std::complex<double>));
                memcpy(&all_coeffs_M[(2*oi+1) * N],   X_perp + N, N * sizeof(std::complex<double>));
            }

            std::vector<double> all_r_hats(n_total * ntheta * 3);
            std::vector<Vec3> all_e_par(n_total * ntheta), all_e_perp(n_total * ntheta);
            for (int oi = 0; oi < n_total; oi++) {
                Mat3& RT = orients[oi].RT;
                for (int it = 0; it < ntheta; it++) {
                    Vec3 rh = RT * r_hat_lab[it];
                    int base = (oi * ntheta + it) * 3;
                    all_r_hats[base]   = rh.x;
                    all_r_hats[base+1] = rh.y;
                    all_r_hats[base+2] = rh.z;
                    all_e_par[oi * ntheta + it]  = RT * e_theta_lab[it];
                    all_e_perp[oi * ntheta + it] = RT * e_phi_lab;
                }
            }

            std::vector<std::complex<double>> all_Fv(n_calls * ntheta * 3);
            compute_farfield_batch_cuda(ff_gpu,
                                        all_coeffs_J.data(), all_coeffs_M.data(),
                                        all_r_hats.data(),
                                        k_ext, eta_ext,
                                        n_calls, n_total, ntheta,
                                        all_Fv.data());

            std::complex<double> ik_val = std::complex<double>(0, -1) * k_ext;
            double k2 = std::norm(k_ext);
            for (int oi = 0; oi < n_total; oi++) {
                double weight = orients[oi].weight;
                std::complex<double>* Fv_par  = &all_Fv[(2*oi) * ntheta * 3];
                std::complex<double>* Fv_perp = &all_Fv[(2*oi+1) * ntheta * 3];

                std::vector<std::complex<double>> S1(ntheta), S2(ntheta), S3(ntheta), S4(ntheta);
                for (int it = 0; it < ntheta; it++) {
                    Vec3& ep = all_e_par[oi * ntheta + it];
                    Vec3& epp = all_e_perp[oi * ntheta + it];

                    std::complex<double> F_par_p  = Fv_par[it*3]*ep.x  + Fv_par[it*3+1]*ep.y  + Fv_par[it*3+2]*ep.z;
                    std::complex<double> F_perp_p = Fv_par[it*3]*epp.x + Fv_par[it*3+1]*epp.y + Fv_par[it*3+2]*epp.z;
                    std::complex<double> F_par_pp  = Fv_perp[it*3]*ep.x  + Fv_perp[it*3+1]*ep.y  + Fv_perp[it*3+2]*ep.z;
                    std::complex<double> F_perp_pp = Fv_perp[it*3]*epp.x + Fv_perp[it*3+1]*epp.y + Fv_perp[it*3+2]*epp.z;

                    S2[it] = ik_val * F_par_p;
                    S4[it] = ik_val * F_perp_p;
                    S3[it] = ik_val * F_par_pp;
                    S1[it] = ik_val * F_perp_pp;
                }

                std::vector<double> M_orient(16 * ntheta);
                amplitude_to_mueller(S1.data(), S2.data(), S3.data(), S4.data(),
                                    ntheta, M_orient.data());

                for (int i = 0; i < 16 * ntheta; i++)
                    M_avg[i] += weight * M_orient[i] / k2;
            }

            time_farfield = ff_timer.elapsed_s();
            printf("  Averaged over %d orientations.\n", n_total);
        }
    }

    double time_total = total_timer.elapsed_s();

    write_json(outfile, M_avg.data(), theta_arr.data(), ntheta,
               ka, n_re, n_im, refinements,
               n_alpha, n_beta, n_gamma,
               time_assembly, time_solve, time_farfield, time_total);

    printf("\n=== Done ===\n");
    printf("  Assembly: %.1fs\n", time_assembly);
    printf("  Solve:    %.1fs\n", time_solve);
    printf("  Farfield: %.1fs\n", time_farfield);
    printf("  Total:    %.1fs\n", time_total);

    return 0;
}
