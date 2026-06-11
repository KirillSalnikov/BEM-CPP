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
#include <climits>
#include <numeric>
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
    printf("  --obj FILE      Load OBJ mesh, normalize to unit equal-volume sphere radius\n");
    printf("  --subdiv N      Flat midpoint subdivisions for OBJ mesh (default: 0)\n");
    printf("  --prism-aspect F  Hex prism h/Dx, ADDA convention (default: 1)\n");
    printf("  --edge-refine N   Prism local edge-refinement passes (default: auto; use 0 to disable)\n");
    printf("  --ref N         Icosphere refinement level (default: 3)\n");
    printf("  --orient NA NB NG  Orientation quadrature (default: 8 8 1)\n");
    printf("  --alpha-avg N  Average alpha/phi in far-field only; use with --orient 1 NB NG\n");
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

static void transform_rhs_to_n_form(cdouble* b, int N)
{
    for (int i = 0; i < N; i++) {
        cdouble e = b[i];
        cdouble h = b[N + i];
        b[i] = h;
        b[N + i] = -e;
    }
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
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
    int edge_refine = -1;
    int refinements = 3;
    int n_alpha = 8, n_beta = 8, n_gamma = 1;
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
        } else if (strcmp(argv[i], "--obj") == 0 && i+1 < argc) {
            obj_file = argv[++i];
            shape = "obj";
        } else if (strcmp(argv[i], "--subdiv") == 0 && i+1 < argc) {
            obj_subdiv = atoi(argv[++i]);
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
        } else if (strcmp(argv[i], "--alpha-avg") == 0 && i+1 < argc) {
            alpha_avg = atoi(argv[++i]);
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
    bool auto_alpha_avg = false;
    int requested_n_alpha = n_alpha;
    if (!single_orient && alpha_avg == 1 && n_alpha > 1 &&
        !std::getenv("BEM_NO_AUTO_ALPHA_AVG")) {
        alpha_avg = n_alpha;
        n_alpha = 1;
        auto_alpha_avg = true;
    }
    if (alpha_avg < 1) {
        fprintf(stderr, "Error: --alpha-avg must be positive\n");
        return 1;
    }
    if (alpha_avg > 1 && n_alpha != 1) {
        fprintf(stderr, "Error: --alpha-avg N averages alpha in far-field only; run with --orient 1 NB NG\n");
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
    std::complex<double> eta_ext_c(eta_ext, 0.0);
    std::complex<double> eta_int_c = 1.0 / m;
    double unknown_m_scale = 1.0;
    double row_h_scale = 1.0;
    double int_op_sign = 1.0;
    double k_identity = 0.0;
    bool n_form = false;
    double n_form_eps_int = 1.0;
    if (strcmp(system_kind, "muller") == 0 || strcmp(system_kind, "muller-balanced") == 0 ||
        strcmp(system_kind, "muller2") == 0 || strcmp(system_kind, "muller2-balanced") == 0)
        int_op_sign = -1.0;
    if (strcmp(system_kind, "muller2") == 0 || strcmp(system_kind, "muller2-balanced") == 0) {
        n_form = true;
        k_identity = -1.0;
        n_form_eps_int = std::norm(m);
    }
    if (strcmp(system_kind, "balanced") == 0 || strcmp(system_kind, "muller-balanced") == 0 ||
        strcmp(system_kind, "muller2-balanced") == 0) {
        unknown_m_scale = std::abs(m);
        row_h_scale = eta_int;
    }
    if (n_form) {
        unknown_m_scale = 1.0;
        row_h_scale = 1.0;
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
    printf("  eta_ext = %.4f, eta_int = %.6f %+.6fi\n",
           eta_ext, eta_int_c.real(), eta_int_c.imag());
    printf("  System: %s", system_kind);
    if (unknown_m_scale != 1.0 || row_h_scale != 1.0)
        printf(" (M_scaled=%.4g*M, H-row scale=%.4g)", unknown_m_scale, row_h_scale);
    if (int_op_sign < 0.0)
        printf(" (interior operator sign=-1)");
    if (k_identity != 0.0)
        printf(" (K identity jump=%.3g)", k_identity);
    if (n_form)
        printf(" (N-form RHS, eps_int=%.4g)", n_form_eps_int);
    printf("\n");
    bool is_obj = (strcmp(shape, "obj") == 0);
    bool is_hex_prism = (strcmp(shape, "hex_prism") == 0 || strcmp(shape, "prism6") == 0);
    if (is_obj && !obj_file) {
        fprintf(stderr, "Error: --shape obj requires --obj FILE\n");
        return 1;
    }
    if (is_hex_prism && accurate_mode && !quad_order_set)
        quad_order = 7;
    else if (is_hex_prism && !quad_order_set)
        quad_order = 4;
    if (is_hex_prism && edge_refine < 0)
        edge_refine = 1;
    if (!is_hex_prism)
        edge_refine = 0;
    printf("  Shape: %s", shape);
    if (strcmp(shape, "hex_prism") == 0)
        printf(" (h/Dx=%.3f, edge_refine=%d)", prism_aspect, edge_refine);
    if (is_obj)
        printf(" (%s, subdiv=%d)", obj_file, obj_subdiv);
    printf("\n");
    printf("  Refinements: %d, Quad order: %d\n", is_obj ? obj_subdiv : refinements, quad_order);
    if (single_orient)
        printf("  Single orientation, scattering plane: %s\n", scattering_plane_yz ? "yz" : "xz");
    else {
        printf("  Orientations: %d x %d x %d = %d\n",
               n_alpha, n_beta, n_gamma, n_alpha * n_beta * n_gamma);
        if (auto_alpha_avg)
            printf("  Auto alpha far-field average: converted requested alpha=%d to --alpha-avg %d\n",
                   requested_n_alpha, alpha_avg);
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
    if (n_form && !use_fmm) {
        fprintf(stderr, "Error: %s is implemented only for FMM/GMRES path\n", system_kind);
        return 1;
    }

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
    if (use_fmm && solver == SOLVER_FMM && !std::getenv("BEM_FMM_BATCH4") &&
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
        if (n_form && !std::getenv("BEM_PREC_FORCE"))
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
        const bool schwarz_prec = use_prec && std::getenv("BEM_PREC_BLOCK") &&
                                  std::atoi(std::getenv("BEM_PREC_BLOCK")) != 0;
        const char* prec_name = use_prec ? (schwarz_prec ? ", Schwarz prec" : ", block-Jacobi prec") : "";
        const char* batch4_name = std::getenv("BEM_FMM_BATCH4") ? ", batch4" : "";
        printf("  Mode: %s+GMRES (digits=%d, tol=%.0e, restart=%d, max_leaf=%d%s%s)\n",
               solver_name(solver), fmm_digits, gmres_tol, gmres_restart, max_leaf, prec_name, batch4_name);
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
        fmm_op.n_form = n_form;
        fmm_op.n_form_eps_int = n_form_eps_int;

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
            compute_rhs_planewave_pair_cached(ff_cache, k_ext, eta_ext, E_par, E_perp,
                                             k_hat, b_par.data(), b_perp.data());
            if (n_form) {
                transform_rhs_to_n_form(b_par.data(), N);
                transform_rhs_to_n_form(b_perp.data(), N);
            }
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
            int orient_progress_step = std::max(10, n_total / 10);
            if (const char* env = std::getenv("BEM_ORIENT_PROGRESS"))
                orient_progress_step = std::max(1, atoi(env));

            std::vector<cdouble> x_par(N2, cdouble(0)), x_perp(N2, cdouble(0));
            std::vector<cdouble> b_par(N2), b_perp(N2);
            std::vector<std::vector<cdouble>> hist_b, hist_x;
            const int max_recycle = 12;
            long long orient_matvecs = 0;

            const char* ff_batch_env = std::getenv("BEM_FF_BATCH");
            int ff_batch_orient = 64;
            if (ff_batch_env) {
                ff_batch_orient = std::max(1, atoi(ff_batch_env));
            } else {
                double target_mb = std::getenv("BEM_FF_TARGET_MB") ? atof(std::getenv("BEM_FF_TARGET_MB")) : 512.0;
                int max_auto_batch = std::getenv("BEM_FF_MAX_BATCH") ? atoi(std::getenv("BEM_FF_MAX_BATCH")) : 512;
                double per_orient_bytes = 64.0 * (double)N + 120.0 * (double)ntheta + 8.0;
                ff_batch_orient = (int)((target_mb * 1024.0 * 1024.0) / std::max(1.0, per_orient_bytes));
                ff_batch_orient = std::max(1, std::min(ff_batch_orient, std::max(1, max_auto_batch)));
            }
            long long n_farfield_samples = (long long)n_total * (long long)alpha_avg;
            ff_batch_orient = std::min(ff_batch_orient, std::max(1, (int)std::min<long long>(n_farfield_samples, INT_MAX)));
            printf("  Streaming far-field batch: %d orientations (BEM_FF_BATCH overrides; BEM_FF_TARGET_MB tunes auto)\n",
                   ff_batch_orient);

            int batch_count = 0;
            std::vector<int> batch_orient_idx(ff_batch_orient);
            std::vector<cdouble> batch_coeffs_J((size_t)ff_batch_orient * 2 * N);
            std::vector<cdouble> batch_coeffs_M((size_t)ff_batch_orient * 2 * N);
            std::vector<double> batch_r_hats((size_t)ff_batch_orient * ntheta * 3);
            std::vector<Vec3> batch_e_par((size_t)ff_batch_orient * ntheta);
            std::vector<Vec3> batch_e_perp((size_t)ff_batch_orient * ntheta);
            FFBatchWorkspace ff_workspace;
            bool ff_gpu_accum = !std::getenv("BEM_FF_CPU_ACCUM");
            std::vector<double> batch_weights(ff_batch_orient);
            std::vector<cdouble> batch_Fv;
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
                ff_workspace.reserve_mueller(ff_batch_orient, ntheta);
                ff_workspace.zero_mueller(ntheta);
                printf("  GPU Mueller accumulation enabled (set BEM_FF_CPU_ACCUM=1 for CPU fallback)\n");
            }
            cdouble ik_val = cdouble(0, -1) * k_ext;
            double k2 = std::norm(k_ext);

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
                        M_avg[i] += weight * M_orient[i] / k2;
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

            auto append_farfield_sample = [&](int oi, const Mat3& RT, double alpha, double weight) {
                if (batch_count == ff_batch_orient)
                    flush_farfield_batch();

                double ca = cos(alpha), sa = sin(alpha);
                double inv_s = (unknown_m_scale == 1.0) ? 1.0 : (1.0 / unknown_m_scale);
                int bi = batch_count++;
                batch_orient_idx[bi] = oi;
                batch_weights[bi] = weight;

                cdouble* J0 = &batch_coeffs_J[(2*bi) * N];
                cdouble* M0 = &batch_coeffs_M[(2*bi) * N];
                cdouble* J1 = &batch_coeffs_J[(2*bi+1) * N];
                cdouble* M1 = &batch_coeffs_M[(2*bi+1) * N];
                for (int i = 0; i < N; i++) {
                    cdouble jp = x_par[i];
                    cdouble mp = x_par[N + i] * inv_s;
                    cdouble ju = x_perp[i];
                    cdouble mu = x_perp[N + i] * inv_s;
                    J0[i] = ca * jp - sa * ju;
                    M0[i] = ca * mp - sa * mu;
                    J1[i] = sa * jp + ca * ju;
                    M1[i] = sa * mp + ca * mu;
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

            for (int oi = 0; oi < n_total; oi++) {
                Mat3& RT = orients[oi].RT;
                Vec3 k_hat = RT * Vec3(0, 0, 1);
                Vec3 e_par = RT * Vec3(1, 0, 0);
                Vec3 e_perp = RT * Vec3(0, 1, 0);

                compute_rhs_planewave_pair_cached(ff_cache, k_ext, eta_ext, e_par, e_perp,
                                                 k_hat, b_par.data(), b_perp.data());
                if (n_form) {
                    transform_rhs_to_n_form(b_par.data(), N);
                    transform_rhs_to_n_form(b_perp.data(), N);
                }
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

                if (alpha_avg == 1) {
                    append_farfield_sample(oi, RT, 0.0, orients[oi].weight);
                } else {
                    double sample_weight = orients[oi].weight / (double)alpha_avg;
                    for (int ia = 0; ia < alpha_avg; ia++) {
                        double alpha = 2.0 * M_PI * (double)ia / (double)alpha_avg;
                        append_farfield_sample(oi, RT, alpha, sample_weight);
                    }
                }

                if ((oi + 1) % orient_progress_step == 0 || oi == n_total - 1)
                    printf("    Orient %d/%d done (avg %.1f matvec/orient)\n",
                           oi + 1, n_total, (double)orient_matvecs / (oi + 1));
	            }
	            flush_farfield_batch();
	            if (ff_gpu_accum)
	                ff_workspace.download_mueller(M_avg.data(), ntheta);
            if (alpha_avg > 1)
                printf("  Averaged over %d solved orientations x %d alpha samples.\n", n_total, alpha_avg);
            else
                printf("  Averaged over %d orientations.\n", n_total);
        }

        fmm_op.cleanup();

    } else {
        // ============================================================
        // Dense LU path (original code)
        // ============================================================
        Timer asm_timer;
        std::vector<std::complex<double>> Z(N2 * N2);
        assemble_pmchwt(rwg, mesh, k_ext, k_int, eta_ext_c, eta_int_c,
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
	            compute_rhs_planewave_pair_cached(ff_cache, k_ext, eta_ext, E_par, E_perp,
	                                             k_hat, &B[0], &B[N2]);

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
	            std::vector<Vec3> rhs_k_hat(n_total), rhs_e_par(n_total), rhs_e_perp(n_total);
	            for (int oi = 0; oi < n_total; oi++) {
	                Mat3& RT = orients[oi].RT;
	                rhs_k_hat[oi] = RT * Vec3(0, 0, 1);
		                rhs_e_par[oi] = RT * Vec3(1, 0, 0);
		                rhs_e_perp[oi] = RT * Vec3(0, 1, 0);
	            }
	            bool use_gpu_rhs = !std::getenv("BEM_NO_GPU_RHS");
	            if (use_gpu_rhs) {
	                printf("  GPU RHS batch enabled (set BEM_NO_GPU_RHS=1 for CPU fallback)\n");
	                compute_rhs_planewave_pairs_cached_cuda(
	                    ff_cache, k_ext, eta_ext,
	                    rhs_e_par.data(), rhs_e_perp.data(), rhs_k_hat.data(),
	                    n_total, B.data());
	            } else {
	                #ifdef _OPENMP
	                #pragma omp parallel for schedule(static)
	                #endif
	                for (int oi = 0; oi < n_total; oi++) {
	                    compute_rhs_planewave_pair_cached(
	                        ff_cache, k_ext, eta_ext,
	                        rhs_e_par[oi], rhs_e_perp[oi], rhs_k_hat[oi],
	                        &B[oi * 2 * N2], &B[(oi * 2 + 1) * N2]);
	                }
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
	            if (const char* ff_batch_env = std::getenv("BEM_FF_BATCH")) {
	                ff_batch_orient = std::max(1, atoi(ff_batch_env));
	            } else {
	                double target_mb = std::getenv("BEM_FF_TARGET_MB") ? atof(std::getenv("BEM_FF_TARGET_MB")) : 512.0;
	                int default_max_batch = (n_farfield_samples > 8192) ? 8192 : 512;
	                int max_auto_batch = std::getenv("BEM_FF_MAX_BATCH") ? atoi(std::getenv("BEM_FF_MAX_BATCH")) : default_max_batch;
	                double per_orient_bytes = 64.0 * (double)N + 120.0 * (double)ntheta + 8.0;
	                ff_batch_orient = (int)((target_mb * 1024.0 * 1024.0) / std::max(1.0, per_orient_bytes));
	                ff_batch_orient = std::max(1, std::min(ff_batch_orient, std::max(1, max_auto_batch)));
	            }
		            ff_batch_orient = std::min(ff_batch_orient, std::max(1, (int)std::min<long long>(n_farfield_samples, INT_MAX)));
	            bool ff_gpu_accum = !std::getenv("BEM_FF_CPU_ACCUM");
	            bool ff_alpha_direct = ff_gpu_accum && alpha_avg > 1 &&
	                                   !std::getenv("BEM_FF_SEPARATE") &&
	                                   !std::getenv("BEM_FF_NO_ALPHA_DIRECT");
	            bool ff_alpha_geom = ff_alpha_direct && !std::getenv("BEM_FF_NO_ALPHA_GEOM");
		            int ff_mgpu = 1;
		            int cuda_device_count = 1;
		            bool have_cuda_device_count = false;
		            if (const char* env = std::getenv("BEM_FF_MGPU")) {
		                ff_mgpu = std::max(1, atoi(env));
            } else if (ff_alpha_geom && !std::getenv("BEM_NO_AUTO_MGPU")) {
                int mgpu_min_samples = 4096;
                if (const char* min_env = std::getenv("BEM_FF_MGPU_MIN_SAMPLES"))
                    mgpu_min_samples = std::max(1, atoi(min_env));
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
	            bool ff_alpha_mgpu = ff_alpha_geom && ff_mgpu > 1;
	            int ff_base_batch_orient = ff_batch_orient;
	            if (ff_alpha_direct) {
	                if (ff_alpha_geom && !std::getenv("BEM_FF_BATCH")) {
	                    double target_mb = std::getenv("BEM_FF_TARGET_MB") ? atof(std::getenv("BEM_FF_TARGET_MB")) : 512.0;
	                    int max_auto_base = std::getenv("BEM_FF_MAX_BASE_BATCH") ? atoi(std::getenv("BEM_FF_MAX_BASE_BATCH")) : 4096;
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
	                printf("  Alpha-direct GPU coefficient mixing enabled: %d base orientations x %d alpha\n",
	                       ff_base_batch_orient, alpha_avg);
	            if (ff_alpha_geom)
	                printf("  Alpha-direct GPU geometry enabled\n");
	            if (ff_alpha_mgpu)
	                printf("  In-process multi-GPU far-field enabled: %d GPUs\n", ff_mgpu);

	            int batch_count = 0;
            int coeff_batch_orient = ff_alpha_direct ? ff_base_batch_orient : ff_batch_orient;
	            std::vector<cdouble> batch_coeffs_J((size_t)coeff_batch_orient * 2 * N);
            std::vector<cdouble> batch_coeffs_M((size_t)coeff_batch_orient * 2 * N);
            bool need_host_farfield_geom = !ff_alpha_direct || !ff_alpha_geom;
	            std::vector<double> batch_r_hats;
	            std::vector<Vec3> batch_e_par;
	            std::vector<Vec3> batch_e_perp;
            if (need_host_farfield_geom) {
                batch_r_hats.resize((size_t)ff_batch_orient * ntheta * 3);
                batch_e_par.resize((size_t)ff_batch_orient * ntheta);
                batch_e_perp.resize((size_t)ff_batch_orient * ntheta);
            }
	            std::vector<double> batch_weights(ff_batch_orient);
	            std::vector<double> batch_RT((size_t)ff_base_batch_orient * 9);
	            FFBatchWorkspace ff_workspace;
	            std::vector<double> mgpu_M_accum;
	            std::vector<std::vector<double>> mgpu_partial;
	            std::vector<FFCacheGPU*> mgpu_ff;
	            std::vector<FFBatchWorkspace*> mgpu_ws;
	            std::vector<cdouble> batch_Fv;
            std::vector<cdouble> S1, S2, S3, S4;
            std::vector<double> M_orient;
            cdouble ik_val = cdouble(0, -1) * k_ext;
            double k2 = std::norm(k_ext);
	            if (ff_gpu_accum && !ff_alpha_mgpu) {
	                ff_gpu.upload(ff_cache);
	                ff_workspace.reserve_mueller(ff_batch_orient, ntheta);
	                ff_workspace.zero_mueller(ntheta);
	                printf("  GPU Mueller accumulation enabled (set BEM_FF_CPU_ACCUM=1 for CPU fallback)\n");
	            } else if (ff_alpha_mgpu) {
	                mgpu_M_accum.assign(16 * ntheta, 0.0);
	                mgpu_partial.assign((size_t)ff_mgpu, std::vector<double>(16 * ntheta, 0.0));
	                mgpu_ff.resize(ff_mgpu, 0);
	                mgpu_ws.resize(ff_mgpu, 0);
	                for (int gd = 0; gd < ff_mgpu; gd++) {
	                    CUDA_CHECK(cudaSetDevice(gd));
	                    mgpu_ff[gd] = new FFCacheGPU();
	                    mgpu_ff[gd]->upload(ff_cache);
	                    mgpu_ws[gd] = new FFBatchWorkspace();
	                }
	                CUDA_CHECK(cudaSetDevice(0));
	                printf("  Multi-GPU Mueller accumulation enabled\n");
	            } else {
                ff_gpu.upload(ff_cache);
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

            auto flush_mueller_batch = [&]() {
                if (batch_count == 0)
                    return;
                if (ff_gpu_accum) {
                    if (std::getenv("BEM_FF_SEPARATE")) {
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
                            M_avg[i] += weight * M_orient[i] / k2;
                    }
                }
                batch_count = 0;
            };

            auto append_dense_farfield_sample = [&](int oi, const Mat3& RT, double ca, double sa, double weight) {
                if (batch_count == ff_batch_orient)
                    flush_mueller_batch();
                std::complex<double>* X_par  = &B[oi * 2 * N2];
                std::complex<double>* X_perp = &B[(oi * 2 + 1) * N2];
                int bi = batch_count++;
                batch_weights[bi] = weight;

                cdouble* J0 = &batch_coeffs_J[(2*bi) * N];
                cdouble* M0 = &batch_coeffs_M[(2*bi) * N];
                cdouble* J1 = &batch_coeffs_J[(2*bi+1) * N];
                cdouble* M1 = &batch_coeffs_M[(2*bi+1) * N];
                for (int i = 0; i < N; i++) {
                    cdouble jp = X_par[i];
                    cdouble mp = X_par[N + i];
                    cdouble ju = X_perp[i];
                    cdouble mu = X_perp[N + i];
                    J0[i] = ca * jp - sa * ju;
                    M0[i] = ca * mp - sa * mu;
                    J1[i] = sa * jp + ca * ju;
                    M1[i] = sa * mp + ca * mu;
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
	                    for (int gd = 0; gd < ff_mgpu; gd++)
	                        std::fill(mgpu_partial[(size_t)gd].begin(), mgpu_partial[(size_t)gd].end(), 0.0);
	                    #ifdef _OPENMP
	                    #pragma omp parallel for schedule(static)
	                    #endif
	                    for (int gd = 0; gd < ff_mgpu; gd++) {
	                        int start = (base_batch_count * gd) / ff_mgpu;
	                        int end = (base_batch_count * (gd + 1)) / ff_mgpu;
	                        int count = end - start;
	                        if (count <= 0)
	                            continue;
	                        CUDA_CHECK(cudaSetDevice(gd));
	                        FFBatchWorkspace& local_ws = *mgpu_ws[(size_t)gd];
	                        local_ws.reserve_mueller(count * alpha_avg, ntheta);
	                        local_ws.zero_mueller(ntheta);
                        accumulate_farfield_mueller_alpha_geom_cuda_ws(
                            *mgpu_ff[(size_t)gd], local_ws,
                            batch_coeffs_J.data() + (size_t)start * 2 * N,
                            batch_coeffs_M.data() + (size_t)start * 2 * N,
                            batch_RT.data() + (size_t)start * 9,
                            r_hat_lab_flat.data(), e_theta_lab_flat.data(),
                            e_phi_lab_flat,
                            batch_weights.data() + (size_t)start * alpha_avg,
                            alpha_cos.data(), alpha_sin.data(),
                            k_ext, eta_ext,
                            count, alpha_avg, ntheta);
	                        local_ws.download_mueller(mgpu_partial[(size_t)gd].data(), ntheta);
	                    }
	                    CUDA_CHECK(cudaSetDevice(0));
	                    for (int gd = 0; gd < ff_mgpu; gd++)
	                        for (int i = 0; i < 16 * ntheta; i++)
	                            mgpu_M_accum[i] += mgpu_partial[(size_t)gd][i];
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
	                int bi = base_batch_count++;
	                double* RT_out = &batch_RT[(size_t)bi * 9];
	                for (int r = 0; r < 3; r++)
	                    for (int c = 0; c < 3; c++)
	                        RT_out[r * 3 + c] = RT.m[r][c];

	                cdouble* J0 = &batch_coeffs_J[(2*bi) * N];
	                cdouble* M0 = &batch_coeffs_M[(2*bi) * N];
	                cdouble* J1 = &batch_coeffs_J[(2*bi+1) * N];
	                cdouble* M1 = &batch_coeffs_M[(2*bi+1) * N];
	                memcpy(J0, X_par, (size_t)N * sizeof(cdouble));
	                memcpy(M0, X_par + N, (size_t)N * sizeof(cdouble));
	                memcpy(J1, X_perp, (size_t)N * sizeof(cdouble));
	                memcpy(M1, X_perp + N, (size_t)N * sizeof(cdouble));

	                double sample_weight = weight / (double)alpha_avg;
	                for (int ia = 0; ia < alpha_avg; ia++) {
	                    double ca = alpha_cos[ia], sa = alpha_sin[ia];
	                    int si = bi * alpha_avg + ia;
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
	            if (const char* env = std::getenv("BEM_ORIENT_PROGRESS"))
	                orient_progress_step = std::max(1, atoi(env));
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
	                for (int i = 0; i < 16 * ntheta; i++)
	                    M_avg[i] = mgpu_M_accum[i];
	                for (int gd = 0; gd < ff_mgpu; gd++) {
	                    CUDA_CHECK(cudaSetDevice(gd));
	                    delete mgpu_ws[(size_t)gd];
	                    delete mgpu_ff[(size_t)gd];
	                }
	                CUDA_CHECK(cudaSetDevice(0));
	            } else if (ff_gpu_accum) {
	                ff_workspace.download_mueller(M_avg.data(), ntheta);
	            }

            time_farfield = ff_timer.elapsed_s();
            if (alpha_avg > 1)
                printf("  Averaged over %d solved orientations x %d alpha samples.\n", n_total, alpha_avg);
            else
                printf("  Averaged over %d orientations.\n", n_total);
        }
    }

    if (!single_orient && !std::getenv("BEM_NO_ORIENT_PROJECT")) {
        project_random_orientation_mueller(M_avg.data(), ntheta);
        printf("  Random-orientation Mueller projection applied.\n");
    } else if (!single_orient) {
        printf("  Random-orientation Mueller projection disabled by BEM_NO_ORIENT_PROJECT.\n");
    }

    double time_total = total_timer.elapsed_s();

    write_json(outfile, M_avg.data(), theta_arr.data(), ntheta,
               ka, n_re, n_im, refinements,
               n_alpha, n_beta, n_gamma, alpha_avg,
               time_assembly, time_solve, time_farfield, time_total);

    printf("\n=== Done ===\n");
    printf("  Assembly: %.1fs\n", time_assembly);
    printf("  Solve:    %.1fs\n", time_solve);
    printf("  Farfield: %.1fs\n", time_farfield);
    printf("  Total:    %.1fs\n", time_total);

    return 0;
}
