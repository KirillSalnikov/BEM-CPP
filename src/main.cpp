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
#include "gmres_dr.h"
#include "precond.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>
#include <string>

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --ka F          Size parameter (required)\n");
    printf("  --ri RE IM      Refractive index (default: 1.3116 0)\n");
    printf("  --ref N         Icosphere refinement level (default: 3)\n");
    printf("  --orient NA NB NG  Orientation quadrature (default: 8 8 1)\n");
    printf("  --ntheta N      Number of scattering angles (default: 181)\n");
    printf("  --quad N        Quadrature order: 4, 7, 13 (default: 7)\n");
    printf("  --out FILE      Output JSON file (default: result.json)\n");
    printf("  --single        Single orientation (no averaging)\n");
    printf("  --fmm           Use FMM+GMRES instead of dense LU\n");
    printf("  --pfft          Use pFFT+GMRES instead of dense LU (faster than FMM)\n");
    printf("  --spfft         Use surface pFFT (2D per face, hex only)\n");
    printf("  --fmm-digits N  FMM/pFFT accuracy digits (default: 3)\n");
    printf("  --gmres-tol F   GMRES relative tolerance (default: 1e-4)\n");
    printf("  --gmres-restart N  GMRES restart (default: 100)\n");
    printf("  --max-leaf N    FMM max particles per leaf (default: 64)\n");
    printf("  --prec TYPE     Preconditioner: diag, ilu0, nearlu, blockj (default: none)\n");
    printf("  --prec-r F      Block-Jacobi radius multiplier (default: 2.0)\n");
    printf("  --prec-bs N     Block-Jacobi max RWG per block (default: 1500)\n");
    printf("  --prec-overlap N  RAS overlap layers (default: 0 = standard BlockJ)\n");
    printf("  --gmres-dr      Use GMRES-DR (deflated restarting)\n");
    printf("  --gmres-k N     Deflation subspace size (default: 20)\n");
    printf("  --shape TYPE    Particle shape: sphere (default), hex\n");
    printf("  --ar F          Hex aspect ratio H/D (default: 1.0)\n");
    printf("  --obj FILE      Load mesh from OBJ file\n");
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    // Default parameters
    double ka = 0;
    double n_re = 1.3116, n_im = 0.0;
    int refinements = 3;
    int n_alpha = 8, n_beta = 8, n_gamma = 1;
    int ntheta = 181;
    int quad_order = 7;
    const char* outfile = "result.json";
    bool single_orient = false;
    bool use_fmm = false;
    bool use_pfft = false;
    bool use_spfft = false;
    PrecondMode prec_mode = PREC_NONE;
    double prec_radius = 2.0;
    int prec_block_size = 1500;
    int prec_overlap = 0;
    bool fmm_test = false;
    int fmm_digits = 3;
    double gmres_tol = 1e-4;
    int gmres_restart = 100;
    int max_leaf = 64;
    bool use_gmres_dr = false;
    int gmres_k = 20;
    std::string shape = "sphere";
    double hex_ar = 1.0;
    const char* obj_file = nullptr;

    // Parse CLI
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ka") == 0 && i+1 < argc) {
            ka = atof(argv[++i]);
        } else if (strcmp(argv[i], "--ri") == 0 && i+2 < argc) {
            n_re = atof(argv[++i]);
            n_im = atof(argv[++i]);
        } else if (strcmp(argv[i], "--ref") == 0 && i+1 < argc) {
            refinements = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--orient") == 0 && i+3 < argc) {
            n_alpha = atoi(argv[++i]);
            n_beta  = atoi(argv[++i]);
            n_gamma = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ntheta") == 0 && i+1 < argc) {
            ntheta = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--quad") == 0 && i+1 < argc) {
            quad_order = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && i+1 < argc) {
            outfile = argv[++i];
        } else if (strcmp(argv[i], "--single") == 0) {
            single_orient = true;
        } else if (strcmp(argv[i], "--fmm") == 0) {
            use_fmm = true;
        } else if (strcmp(argv[i], "--pfft") == 0) {
            use_fmm = true;
            use_pfft = true;
        } else if (strcmp(argv[i], "--spfft") == 0) {
            use_fmm = true;
            use_pfft = true;
            use_spfft = true;
        } else if (strcmp(argv[i], "--prec") == 0 && i+1 < argc) {
            const char* pt = argv[++i];
            if (strcmp(pt, "diag") == 0) prec_mode = PREC_DIAG;
            else if (strcmp(pt, "ilu0") == 0) prec_mode = PREC_ILU0;
            else if (strcmp(pt, "nearlu") == 0) prec_mode = PREC_NEARLU;
            else if (strcmp(pt, "blockj") == 0) prec_mode = PREC_BLOCKJ;
            else { fprintf(stderr, "Unknown prec type: %s\n", pt); return 1; }
        } else if (strcmp(argv[i], "--prec-r") == 0 && i+1 < argc) {
            prec_radius = atof(argv[++i]);
        } else if (strcmp(argv[i], "--prec-bs") == 0 && i+1 < argc) {
            prec_block_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prec-overlap") == 0 && i+1 < argc) {
            prec_overlap = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fmm-digits") == 0 && i+1 < argc) {
            fmm_digits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gmres-tol") == 0 && i+1 < argc) {
            gmres_tol = atof(argv[++i]);
        } else if (strcmp(argv[i], "--gmres-restart") == 0 && i+1 < argc) {
            gmres_restart = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-leaf") == 0 && i+1 < argc) {
            max_leaf = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gmres-dr") == 0) {
            use_gmres_dr = true;
        } else if (strcmp(argv[i], "--gmres-k") == 0 && i+1 < argc) {
            gmres_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--shape") == 0 && i+1 < argc) {
            shape = argv[++i];
        } else if (strcmp(argv[i], "--ar") == 0 && i+1 < argc) {
            hex_ar = atof(argv[++i]);
        } else if (strcmp(argv[i], "--obj") == 0 && i+1 < argc) {
            obj_file = argv[++i];
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

    printf("=== BEM-CUDA Solver ===\n");
    printf("  ka = %.4f, m = %.4f + %.4fi\n", ka, n_re, n_im);
    printf("  k_ext = %.4f, k_int = %.4f + %.4fi\n",
           k_ext.real(), k_int.real(), k_int.imag());
    printf("  eta_ext = %.4f, eta_int = %.4f\n", eta_ext, eta_int);
    printf("  Refinements: %d, Quad order: %d\n", refinements, quad_order);
    if (use_fmm)
        printf("  Mode: %s+%s (digits=%d, tol=%.0e, restart=%d, max_leaf=%d%s%s)\n",
               use_spfft ? "SurfPFFT" : use_pfft ? "pFFT" : "FMM",
               use_gmres_dr ? "GMRES-DR" : "GMRES",
               fmm_digits, gmres_tol, gmres_restart, max_leaf,
               prec_mode == PREC_DIAG ? ", DIAG prec" :
               prec_mode == PREC_ILU0 ? ", ILU(0) prec" :
               prec_mode == PREC_NEARLU ? ", NEARLU prec" :
               prec_mode == PREC_BLOCKJ ? ", BlockJ prec" : "",
               use_gmres_dr ? (", k=" + std::to_string(gmres_k)).c_str() : "");
    else
        printf("  Mode: Dense LU\n");
    if (single_orient)
        printf("  Single orientation\n");
    else
        printf("  Orientations: %d x %d x %d = %d\n",
               n_alpha, n_beta, n_gamma, n_alpha * n_beta * n_gamma);

    // 1. Generate mesh
    Timer mesh_timer;
    Mesh mesh;
    if (obj_file) {
        mesh = load_obj(obj_file);
    } else if (shape == "hex") {
        mesh = hex_prism(hex_ar, refinements);
    } else {
        double radius = 1.0;
        mesh = icosphere(radius, refinements);
    }
    printf("  Mesh: %d vertices, %d triangles (%.1fms)\n",
           mesh.nv(), mesh.nt(), mesh_timer.elapsed_ms());

    // 2. Build RWG basis
    Timer rwg_timer;
    RWG rwg = build_rwg(mesh);
    printf("  RWG: %d basis functions (%.1fms)\n", rwg.N, rwg_timer.elapsed_ms());

    int N = rwg.N;
    int N2 = 2 * N;

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

        // Build preconditioner if requested
        NearFieldPrecond* precond_ptr = nullptr;
        NearFieldPrecond precond;
        if (prec_mode != PREC_NONE) {
            precond.build(fmm_op, prec_mode, prec_radius, prec_block_size, prec_overlap);
            precond_ptr = &precond;
        }

        time_assembly = asm_timer.elapsed_s();

        Timer solve_timer;

        if (single_orient) {
            Vec3 k_hat(0, 0, 1);
            Vec3 E_par(1, 0, 0);
            Vec3 E_perp(0, 1, 0);

            // Solve for both polarizations
            std::vector<cdouble> b_par(N2), b_perp(N2);
            compute_rhs_planewave(rwg, mesh, k_ext, eta_ext, E_par, k_hat, quad_order, b_par.data());
            compute_rhs_planewave(rwg, mesh, k_ext, eta_ext, E_perp, k_hat, quad_order, b_perp.data());

            std::vector<cdouble> x_par(N2, cdouble(0)), x_perp(N2, cdouble(0));
            if (use_gmres_dr) {
                printf("\n  Solving both polarizations (GMRES-DR, k=%d)...\n", gmres_k);
                gmres_dr_paired(fmm_op,
                    b_par.data(), b_perp.data(),
                    x_par.data(), x_perp.data(),
                    gmres_restart, gmres_k, gmres_tol, 300, true, precond_ptr);
            } else {
                printf("\n  Solving both polarizations (paired GMRES)...\n");
                gmres_solve_paired(fmm_op,
                    b_par.data(), b_perp.data(),
                    x_par.data(), x_perp.data(),
                    gmres_restart, gmres_tol, 300, true, precond_ptr);
            }

            time_solve = solve_timer.elapsed_s();

            // Far-field
            Timer ff_timer;
            cdouble* J_par  = x_par.data();
            cdouble* M_par  = x_par.data() + N;
            cdouble* J_perp = x_perp.data();
            cdouble* M_perp = x_perp.data() + N;

            std::vector<cdouble> Fth_par(ntheta), Fph_par(ntheta);
            std::vector<cdouble> Fth_perp(ntheta), Fph_perp(ntheta);

            compute_far_field(ff_cache, J_par, M_par, k_ext, eta_ext,
                             theta_arr.data(), ntheta,
                             Fth_par.data(), Fph_par.data());
            compute_far_field(ff_cache, J_perp, M_perp, k_ext, eta_ext,
                             theta_arr.data(), ntheta,
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
            sort_orientations_nearest(orients);
            int n_total = (int)orients.size();

            // Far-field GPU cache
            FFCacheGPU ff_gpu;
            ff_gpu.upload(ff_cache);

            // Lab-frame scattering vectors
            std::vector<Vec3> r_hat_lab(ntheta), e_theta_lab(ntheta);
            Vec3 e_phi_lab(0, 1, 0);
            for (int it = 0; it < ntheta; it++) {
                double ct = cos(theta_arr[it]), st = sin(theta_arr[it]);
                r_hat_lab[it] = Vec3(st, 0, ct);
                e_theta_lab[it] = Vec3(ct, 0, -st);
            }

            // Storage for all solutions (for batched far-field)
            int n_calls = n_total * 2;
            std::vector<cdouble> all_coeffs_J(n_calls * N);
            std::vector<cdouble> all_coeffs_M(n_calls * N);

            printf("\n  Solving %d orientations x 2 polarizations with GMRES...\n", n_total);

            // Solution vectors — reused across orientations as initial guess
            std::vector<cdouble> x_par(N2, cdouble(0)), x_perp(N2, cdouble(0));

            // Persistent GCRO-DR context: recycles deflation vectors across orientations
            GcroDrContext* gcro_ctx = nullptr;
            if (use_gmres_dr && gmres_k > 0)
                gcro_ctx = gcro_dr_create(N2, gmres_k);

            for (int oi = 0; oi < n_total; oi++) {
                Mat3& RT = orients[oi].RT;
                Vec3 k_hat = RT * Vec3(0, 0, 1);
                Vec3 e_par = RT * Vec3(1, 0, 0);
                Vec3 e_perp = RT * Vec3(0, 1, 0);

                std::vector<cdouble> b_par(N2), b_perp(N2);

                compute_rhs_planewave(rwg, mesh, k_ext, eta_ext, e_par, k_hat, quad_order, b_par.data());
                compute_rhs_planewave(rwg, mesh, k_ext, eta_ext, e_perp, k_hat, quad_order, b_perp.data());

                Timer oi_timer;
                int mv;
                if (use_gmres_dr) {
                    mv = gmres_dr_paired(fmm_op,
                        b_par.data(), b_perp.data(),
                        x_par.data(), x_perp.data(),
                        gmres_restart, gmres_k, gmres_tol, 300, false, precond_ptr,
                        gcro_ctx);
                } else {
                    mv = gmres_solve_paired(fmm_op,
                        b_par.data(), b_perp.data(),
                        x_par.data(), x_perp.data(),
                        gmres_restart, gmres_tol, 300, false, precond_ptr);
                }
                printf("    Orient %d/%d: %d matvecs, %.1fs\n",
                       oi + 1, n_total, mv, oi_timer.elapsed_s());

                memcpy(&all_coeffs_J[(2*oi) * N], x_par.data(), N * sizeof(cdouble));
                memcpy(&all_coeffs_M[(2*oi) * N], x_par.data() + N, N * sizeof(cdouble));
                memcpy(&all_coeffs_J[(2*oi+1) * N], x_perp.data(), N * sizeof(cdouble));
                memcpy(&all_coeffs_M[(2*oi+1) * N], x_perp.data() + N, N * sizeof(cdouble));

                if (false && (oi + 1) % 10 == 0 || oi == n_total - 1)
                    printf("    Orient %d/%d done\n", oi + 1, n_total);
            }

            if (gcro_ctx) gcro_dr_destroy(gcro_ctx);

            time_solve = solve_timer.elapsed_s();

            // Far-field (batched GPU)
            Timer ff_timer;
            printf("  Computing GPU far-field: %d calls x %d dirs...\n", n_calls, ntheta);

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

            std::vector<cdouble> all_Fv(n_calls * ntheta * 3);
            compute_farfield_batch_cuda(ff_gpu,
                                        all_coeffs_J.data(), all_coeffs_M.data(),
                                        all_r_hats.data(),
                                        k_ext, eta_ext,
                                        n_calls, n_total, ntheta,
                                        all_Fv.data());

            // Post-process: Fv -> S-matrix -> Mueller
            cdouble ik_val = cdouble(0, -1) * k_ext;
            double k2 = std::norm(k_ext);
            for (int oi = 0; oi < n_total; oi++) {
                double weight = orients[oi].weight;
                cdouble* Fv_par  = &all_Fv[(2*oi) * ntheta * 3];
                cdouble* Fv_perp = &all_Fv[(2*oi+1) * ntheta * 3];

                std::vector<cdouble> S1(ntheta), S2(ntheta), S3(ntheta), S4(ntheta);
                for (int it = 0; it < ntheta; it++) {
                    Vec3& ep = all_e_par[oi * ntheta + it];
                    Vec3& epp = all_e_perp[oi * ntheta + it];

                    cdouble F_par_p  = Fv_par[it*3]*ep.x  + Fv_par[it*3+1]*ep.y  + Fv_par[it*3+2]*ep.z;
                    cdouble F_perp_p = Fv_par[it*3]*epp.x + Fv_par[it*3+1]*epp.y + Fv_par[it*3+2]*epp.z;
                    cdouble F_par_pp  = Fv_perp[it*3]*ep.x  + Fv_perp[it*3+1]*ep.y  + Fv_perp[it*3+2]*ep.z;
                    cdouble F_perp_pp = Fv_perp[it*3]*epp.x + Fv_perp[it*3+1]*epp.y + Fv_perp[it*3+2]*epp.z;

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

        if (prec_mode != PREC_NONE)
            precond.cleanup_gpu();
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

        if (single_orient) {
            Vec3 k_hat(0, 0, 1);
            Vec3 E_par(1, 0, 0);
            Vec3 E_perp(0, 1, 0);

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
                             Fth_par.data(), Fph_par.data());
            compute_far_field(ff_cache, J_perp, M_perp, k_ext, eta_ext,
                             theta_arr.data(), ntheta,
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
            sort_orientations_nearest(orients);
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
            Vec3 e_phi_lab(0, 1, 0);
            for (int it = 0; it < ntheta; it++) {
                double ct = cos(theta_arr[it]), st = sin(theta_arr[it]);
                r_hat_lab[it] = Vec3(st, 0, ct);
                e_theta_lab[it] = Vec3(ct, 0, -st);
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
