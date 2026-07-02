#include "block_gmres.h"
#include "bem_fmm.h"
#include "gpu_select.h"
#include "precond.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static cdouble orthogonalize_against_p(const cdouble* vi, cdouble* w, int n)
{
    double sr = 0.0, si = 0.0;
    cdouble hij = cdouble(0);
    #pragma omp parallel
    {
        #pragma omp for reduction(+:sr,si) schedule(static)
        for (int i = 0; i < n; i++) {
            cdouble v = std::conj(vi[i]) * w[i];
            sr += v.real();
            si += v.imag();
        }
        #pragma omp single
        {
            hij = cdouble(sr, si);
        }
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++)
            w[i] -= hij * vi[i];
    }
    return hij;
}

static void orthogonalize_against_pair_p(const cdouble* vi1, cdouble* w1,
                                         const cdouble* vi2, cdouble* w2,
                                         int n, cdouble& hij1, cdouble& hij2)
{
    double sr1 = 0.0, si1 = 0.0;
    double sr2 = 0.0, si2 = 0.0;
    hij1 = cdouble(0);
    hij2 = cdouble(0);
    #pragma omp parallel
    {
        #pragma omp for reduction(+:sr1,si1,sr2,si2) schedule(static)
        for (int i = 0; i < n; i++) {
            cdouble v1 = std::conj(vi1[i]) * w1[i];
            cdouble v2 = std::conj(vi2[i]) * w2[i];
            sr1 += v1.real();
            si1 += v1.imag();
            sr2 += v2.real();
            si2 += v2.imag();
        }
        #pragma omp single
        {
            hij1 = cdouble(sr1, si1);
            hij2 = cdouble(sr2, si2);
        }
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            w1[i] -= hij1 * vi1[i];
            w2[i] -= hij2 * vi2[i];
        }
    }
}

static double norm_p(const cdouble* a, int n) {
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int i = 0; i < n; i++)
        s += std::norm(a[i]);
    return std::sqrt(s);
}

static void norm_pair_p(const cdouble* a1, const cdouble* a2, int n,
                        double& out1, double& out2)
{
    double s1 = 0.0, s2 = 0.0;
    #pragma omp parallel for reduction(+:s1,s2) schedule(static)
    for (int i = 0; i < n; i++) {
        s1 += std::norm(a1[i]);
        s2 += std::norm(a2[i]);
    }
    out1 = std::sqrt(s1);
    out2 = std::sqrt(s2);
}

static bool finite_complex(cdouble z)
{
    return std::isfinite(z.real()) && std::isfinite(z.imag());
}

static bool solve_gmres_y(int restart, int m,
                          const std::vector<cdouble>& H,
                          const std::vector<cdouble>& s,
                          std::vector<cdouble>& y)
{
    y.resize(m);
    for (int i = m - 1; i >= 0; i--) {
        y[i] = s[i];
        for (int k = i + 1; k < m; k++)
            y[i] -= H[i * restart + k] * y[k];
        cdouble diag = H[i * restart + i];
        if (!finite_complex(diag) || std::abs(diag) <= 1e-300)
            return false;
        y[i] /= diag;
        if (!finite_complex(y[i]))
            return false;
    }
    return true;
}

static int gmres_step_update(int n, int restart, int m,
                             const std::vector<cdouble>& H,
                             const std::vector<cdouble>& s,
                             const std::vector<cdouble>& V,
                             const std::vector<cdouble>& Z,
                             bool has_precond,
                             bool store_z,
                             NearFieldPrecond* precond,
                             cdouble* x,
                             std::vector<cdouble>& y,
                             std::vector<cdouble>& ztmp)
{
    if (m <= 0)
        return 0;

    if (!solve_gmres_y(restart, m, H, s, y))
        return 1;

    if (!has_precond || store_z) {
        const std::vector<cdouble>& basis = (has_precond && store_z) ? Z : V;
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; k++) {
            cdouble acc = cdouble(0);
            for (int i = 0; i < m; i++)
                acc += y[i] * basis[(size_t)i * n + k];
            x[k] += acc;
        }
        return 0;
    }

    ztmp.resize(n);
    for (int i = 0; i < m; i++) {
        const cdouble* vi = &V[(size_t)i * n];
        precond->apply(vi, ztmp.data());
        vi = ztmp.data();
        cdouble yi = y[i];
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; k++)
            x[k] += yi * vi[k];
    }
    return 0;
}

static int gmres_step_update_pair(int n, int restart, int m1, int m2,
                                  const std::vector<cdouble>& H1,
                                  const std::vector<cdouble>& H2,
                                  const std::vector<cdouble>& s1,
                                  const std::vector<cdouble>& s2,
                                  const std::vector<cdouble>& V1,
                                  const std::vector<cdouble>& V2,
                                  const std::vector<cdouble>& Z1,
                                  const std::vector<cdouble>& Z2,
                                  bool has_precond,
                                  bool store_z,
                                  NearFieldPrecond* precond,
                                  cdouble* x1,
                                  cdouble* x2,
                                  std::vector<cdouble>& y1,
                                  std::vector<cdouble>& y2,
                                  std::vector<cdouble>& ztmp,
                                  std::vector<cdouble>& ztmp2)
{
    bool fused_update = bem_env_flag_enabled("BEM_GMRES_FUSED_UPDATE", true);
    if (!fused_update) {
        int rc1 = gmres_step_update(n, restart, m1, H1, s1, V1, Z1,
                                    has_precond, store_z, precond, x1, y1, ztmp);
        int rc2 = gmres_step_update(n, restart, m2, H2, s2, V2, Z2,
                                    has_precond, store_z, precond, x2, y2, ztmp2);
        if (rc1 != 0 || rc2 != 0)
            return 1;
        return 0;
    }

    if (m1 <= 0 && m2 <= 0)
        return 0;

    if (!solve_gmres_y(restart, m1, H1, s1, y1))
        return 1;
    if (!solve_gmres_y(restart, m2, H2, s2, y2))
        return 1;

    if (has_precond && !store_z) {
        ztmp.resize(n);
        ztmp2.resize(n);
        int mm = std::max(m1, m2);
        for (int i = 0; i < mm; i++) {
            const bool do1 = i < m1;
            const bool do2 = i < m2;
            if (do1)
                precond->apply(&V1[(size_t)i * n], ztmp.data());
            if (do2)
                precond->apply(&V2[(size_t)i * n], ztmp2.data());
            cdouble yi1 = do1 ? y1[i] : cdouble(0);
            cdouble yi2 = do2 ? y2[i] : cdouble(0);
            const cdouble* z1p = ztmp.data();
            const cdouble* z2p = ztmp2.data();
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < n; k++) {
                if (do1)
                    x1[k] += yi1 * z1p[k];
                if (do2)
                    x2[k] += yi2 * z2p[k];
            }
        }
        return 0;
    }

    const std::vector<cdouble>& basis1 = (has_precond && store_z) ? Z1 : V1;
    const std::vector<cdouble>& basis2 = (has_precond && store_z) ? Z2 : V2;
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        cdouble acc1 = cdouble(0);
        cdouble acc2 = cdouble(0);
        for (int i = 0; i < m1; i++)
            acc1 += y1[i] * basis1[(size_t)i * n + k];
        for (int i = 0; i < m2; i++)
            acc2 += y2[i] * basis2[(size_t)i * n + k];
        x1[k] += acc1;
        x2[k] += acc2;
    }
    return 0;
}

int gmres_solve_paired(BemFmmOperator& op,
                       const cdouble* b1, const cdouble* b2,
                       cdouble* x1, cdouble* x2,
                       int restart, double tol, int maxiter,
                       bool verbose, NearFieldPrecond* precond)
{
    GmresPairedWorkspace ws;
    return gmres_solve_paired_ws(op, b1, b2, x1, x2,
                                 restart, tol, maxiter, verbose, precond, ws);
}

int gmres_solve_paired_ws(BemFmmOperator& op,
                          const cdouble* b1, const cdouble* b2,
                          cdouble* x1, cdouble* x2,
                          int restart, double tol, int maxiter,
                          bool verbose, NearFieldPrecond* precond,
                          GmresPairedWorkspace& ws)
{
    int n = op.system_size;
    ws.final_relres1 = 0.0;
    ws.final_relres2 = 0.0;
    ws.converged1 = false;
    ws.converged2 = false;
    ws.stopped_stagnant = false;
    ws.numerical_breakdown = false;
    ws.restored_best_iterate = false;
    ws.reached_max_cycles = false;

    bool has_precond = (precond != nullptr);
    bool reorth = bem_env_flag_enabled("BEM_GMRES_REORTH", true);
    bool pair_arnoldi = bem_env_flag_enabled("BEM_GMRES_PAIR_ARNOLDI", true);
    bool store_z = false;
    if (has_precond) {
        store_z = bem_env_flag_enabled("BEM_GMRES_STORE_Z");
        if (verbose && store_z) {
            double store_z_mb = (2.0 * (double)n * (double)restart * sizeof(cdouble)) /
                                (1024.0 * 1024.0);
            printf("  [GMRES-paired] STORE_Z enabled (%.1f MB)\n", store_z_mb);
            fflush(stdout);
        }
    }
    int stagnation_cycles = 0;
    stagnation_cycles = std::max(0, bem_env_int("BEM_GMRES_STAGNATION_CYCLES", stagnation_cycles));
    double stagnation_rel = 0.01;
    stagnation_rel = std::max(0.0, bem_env_double("BEM_GMRES_STAGNATION_REL", stagnation_rel));
    int inner_stagnation_window = 0;
    inner_stagnation_window = std::max(0, bem_env_int("BEM_GMRES_INNER_STAGNATION_WINDOW", inner_stagnation_window));
    double inner_stagnation_rel = 0.05;
    inner_stagnation_rel = std::max(0.0, bem_env_double("BEM_GMRES_INNER_STAGNATION_REL", inner_stagnation_rel));
    int inner_stagnation_min_iter = 300;
    inner_stagnation_min_iter = std::max(0, bem_env_int("BEM_GMRES_INNER_STAGNATION_MIN_ITER", inner_stagnation_min_iter));

    auto& r1 = ws.r1; auto& r2 = ws.r2;
    auto& w1 = ws.w1; auto& w2 = ws.w2;
    auto& z1 = ws.z1; auto& z2 = ws.z2;
    auto& V1 = ws.V1; auto& V2 = ws.V2;
    auto& Z1 = ws.Z1; auto& Z2 = ws.Z2;
    auto& H1 = ws.H1; auto& H2 = ws.H2;
    auto& cs1 = ws.cs1; auto& sn1 = ws.sn1; auto& s1 = ws.s1;
    auto& cs2 = ws.cs2; auto& sn2 = ws.sn2; auto& s2 = ws.s2;
    auto& ytmp = ws.ytmp; auto& ytmp2 = ws.ytmp2; auto& ztmp = ws.ztmp; auto& ztmp2 = ws.ztmp2;

    r1.assign(b1, b1 + n);
    r2.assign(b2, b2 + n);
    w1.resize(n);
    w2.resize(n);
    if (has_precond) {
        z1.resize(n);
        z2.resize(n);
    } else {
        z1.clear();
        z2.clear();
    }
    V1.resize((size_t)n * (restart + 1));
    V2.resize((size_t)n * (restart + 1));
    if (has_precond && store_z) {
        Z1.resize((size_t)n * restart);
        Z2.resize((size_t)n * restart);
    } else {
        Z1.clear();
        Z2.clear();
    }

    H1.resize((restart + 1) * restart);
    H2.resize((restart + 1) * restart);
    cs1.resize(restart); sn1.resize(restart); s1.resize(restart + 1);
    cs2.resize(restart); sn2.resize(restart); s2.resize(restart + 1);

    double bnorm1, bnorm2;
    norm_pair_p(b1, b2, n, bnorm1, bnorm2);
    if (bnorm1 < 1e-30) bnorm1 = 1.0;
    if (bnorm2 < 1e-30) bnorm2 = 1.0;

    double xnorm1, xnorm2;
    norm_pair_p(x1, x2, n, xnorm1, xnorm2);
    bool warm1 = (xnorm1 > 1e-30);
    bool warm2 = (xnorm2 > 1e-30);
    int total_matvecs = 0;

    if (warm1 && warm2) {
        op.matvec_batch2(x1, x2, r1.data(), r2.data());
        total_matvecs++;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            r1[i] = b1[i] - r1[i];
            r2[i] = b2[i] - r2[i];
        }
    } else if (warm1) {
        op.matvec(x1, r1.data());
        total_matvecs++;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            r1[i] = b1[i] - r1[i];
    } else if (warm2) {
        op.matvec(x2, r2.data());
        total_matvecs++;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            r2[i] = b2[i] - r2[i];
    }

    double rnorm1, rnorm2;
    norm_pair_p(r1.data(), r2.data(), n, rnorm1, rnorm2);

    bool conv1 = (rnorm1 / bnorm1 < tol);
    bool conv2 = (rnorm2 / bnorm2 < tol);
    double last_rel1 = rnorm1 / bnorm1;
    double last_rel2 = rnorm2 / bnorm2;
    double best_rel = std::max(conv1 ? 0.0 : rnorm1 / bnorm1,
                               conv2 ? 0.0 : rnorm2 / bnorm2);
    double best_true_rel1 = last_rel1;
    double best_true_rel2 = last_rel2;
    double best_pair_rel = std::max(last_rel1, last_rel2);
    double best_inner_rel = best_pair_rel;
    int best_inner_iter = total_matvecs;
    std::vector<cdouble> best_x1(x1, x1 + n);
    std::vector<cdouble> best_x2(x2, x2 + n);
    int stagnant_restart_count = 0;
    bool stopped_stagnant = false;
    bool numerical_breakdown = false;

    if (verbose) {
        printf("  [GMRES-paired] start: res1=%.2e res2=%.2e%s\n",
               rnorm1 / bnorm1, rnorm2 / bnorm2,
               (warm1 || warm2) ? " (warm)" : "");
        fflush(stdout);
    }

    for (int cycle = 0; cycle < maxiter && !(conv1 && conv2); cycle++) {
        std::fill(H1.begin(), H1.end(), cdouble(0));
        std::fill(H2.begin(), H2.end(), cdouble(0));
        std::fill(s1.begin(), s1.end(), cdouble(0));
        std::fill(s2.begin(), s2.end(), cdouble(0));

        if (!conv1) {
            double inv = 1.0 / rnorm1;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++)
                V1[i] = r1[i] * inv;
            s1[0] = cdouble(rnorm1);
        }
        if (!conv2) {
            double inv = 1.0 / rnorm2;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++)
                V2[i] = r2[i] * inv;
            s2[0] = cdouble(rnorm2);
        }

        int m1 = 0, m2 = 0;
        for (int j = 0; j < restart && !(conv1 && conv2) && !stopped_stagnant; j++) {
            if (!conv1 && !conv2) {
                const cdouble* in1 = &V1[(size_t)j * n];
                const cdouble* in2 = &V2[(size_t)j * n];
                if (has_precond) {
                    precond->apply_pair(in1, in2, z1.data(), z2.data());
                    if (store_z) {
                        memcpy(&Z1[(size_t)j * n], z1.data(), n * sizeof(cdouble));
                        memcpy(&Z2[(size_t)j * n], z2.data(), n * sizeof(cdouble));
                    }
                    op.matvec_batch2(z1.data(), z2.data(), w1.data(), w2.data());
                } else {
                    op.matvec_batch2(in1, in2, w1.data(), w2.data());
                }
                total_matvecs++;
            } else if (!conv1) {
                const cdouble* in = &V1[(size_t)j * n];
                if (has_precond) {
                    precond->apply(in, z1.data());
                    if (store_z)
                        memcpy(&Z1[(size_t)j * n], z1.data(), n * sizeof(cdouble));
                    op.matvec(z1.data(), w1.data());
                } else {
                    op.matvec(in, w1.data());
                }
                total_matvecs++;
            } else if (!conv2) {
                const cdouble* in = &V2[(size_t)j * n];
                if (has_precond) {
                    precond->apply(in, z2.data());
                    if (store_z)
                        memcpy(&Z2[(size_t)j * n], z2.data(), n * sizeof(cdouble));
                    op.matvec(z2.data(), w2.data());
                } else {
                    op.matvec(in, w2.data());
                }
                total_matvecs++;
            }

            auto arnoldi = [&](std::vector<cdouble>& V, std::vector<cdouble>& H,
                               std::vector<cdouble>& cs, std::vector<cdouble>& sn,
                               std::vector<cdouble>& s, std::vector<cdouble>& w,
                               double bnorm, bool& conv, int& m, double& last_rel) {
                if (conv)
                    return;
                for (int i = 0; i <= j; i++) {
                    cdouble* vi = &V[(size_t)i * n];
                    cdouble hij = orthogonalize_against_p(vi, w.data(), n);
                    H[i * restart + j] = hij;
                }
                if (reorth) {
                    for (int i = 0; i <= j; i++) {
                        cdouble* vi = &V[(size_t)i * n];
                        cdouble hij = orthogonalize_against_p(vi, w.data(), n);
                        H[i * restart + j] += hij;
                    }
                }
                double wn = norm_p(w.data(), n);
                H[(j + 1) * restart + j] = cdouble(wn);
                if (wn > 1e-30) {
                    cdouble* vnext = &V[(size_t)(j + 1) * n];
                    double inv = 1.0 / wn;
                    #pragma omp parallel for schedule(static)
                    for (int k = 0; k < n; k++)
                        vnext[k] = w[k] * inv;
                }
                for (int i = 0; i < j; i++) {
                    cdouble h0 = H[i * restart + j];
                    cdouble h1 = H[(i + 1) * restart + j];
                    H[i * restart + j]       = std::conj(cs[i]) * h0 + std::conj(sn[i]) * h1;
                    H[(i + 1) * restart + j] = -sn[i] * h0 + cs[i] * h1;
                }
                cdouble h0 = H[j * restart + j];
                cdouble h1 = H[(j + 1) * restart + j];
                double den = std::sqrt(std::norm(h0) + std::norm(h1));
                cs[j] = (den > 1e-30) ? h0 / den : cdouble(1);
                sn[j] = (den > 1e-30) ? h1 / den : cdouble(0);
                H[j * restart + j] = std::conj(cs[j]) * h0 + std::conj(sn[j]) * h1;
                H[(j + 1) * restart + j] = cdouble(0);
                cdouble s0 = s[j];
                s[j] = std::conj(cs[j]) * s0;
                s[j + 1] = -sn[j] * s0;
                m = j + 1;
                last_rel = std::abs(s[j + 1]) / bnorm;
                if (last_rel < tol)
                    conv = true;
            };

            auto arnoldi_pair = [&]() {
                for (int i = 0; i <= j; i++) {
                    cdouble* vi1 = &V1[(size_t)i * n];
                    cdouble* vi2 = &V2[(size_t)i * n];
                    cdouble hij1, hij2;
                    orthogonalize_against_pair_p(vi1, w1.data(), vi2, w2.data(), n, hij1, hij2);
                    H1[i * restart + j] = hij1;
                    H2[i * restart + j] = hij2;
                }
                if (reorth) {
                    for (int i = 0; i <= j; i++) {
                        cdouble* vi1 = &V1[(size_t)i * n];
                        cdouble* vi2 = &V2[(size_t)i * n];
                        cdouble hij1, hij2;
                        orthogonalize_against_pair_p(vi1, w1.data(), vi2, w2.data(), n, hij1, hij2);
                        H1[i * restart + j] += hij1;
                        H2[i * restart + j] += hij2;
                    }
                }

                double wn1, wn2;
                norm_pair_p(w1.data(), w2.data(), n, wn1, wn2);
                H1[(j + 1) * restart + j] = cdouble(wn1);
                H2[(j + 1) * restart + j] = cdouble(wn2);
                if (wn1 > 1e-30 && wn2 > 1e-30) {
                    cdouble* vnext1 = &V1[(size_t)(j + 1) * n];
                    cdouble* vnext2 = &V2[(size_t)(j + 1) * n];
                    double inv1 = 1.0 / wn1;
                    double inv2 = 1.0 / wn2;
                    #pragma omp parallel for schedule(static)
                    for (int k = 0; k < n; k++) {
                        vnext1[k] = w1[k] * inv1;
                        vnext2[k] = w2[k] * inv2;
                    }
                } else {
                    if (wn1 > 1e-30) {
                        cdouble* vnext1 = &V1[(size_t)(j + 1) * n];
                        double inv1 = 1.0 / wn1;
                        #pragma omp parallel for schedule(static)
                        for (int k = 0; k < n; k++)
                            vnext1[k] = w1[k] * inv1;
                    }
                    if (wn2 > 1e-30) {
                        cdouble* vnext2 = &V2[(size_t)(j + 1) * n];
                        double inv2 = 1.0 / wn2;
                        #pragma omp parallel for schedule(static)
                        for (int k = 0; k < n; k++)
                            vnext2[k] = w2[k] * inv2;
                    }
                }

                for (int i = 0; i < j; i++) {
                    cdouble h10 = H1[i * restart + j];
                    cdouble h11 = H1[(i + 1) * restart + j];
                    H1[i * restart + j]       = std::conj(cs1[i]) * h10 + std::conj(sn1[i]) * h11;
                    H1[(i + 1) * restart + j] = -sn1[i] * h10 + cs1[i] * h11;

                    cdouble h20 = H2[i * restart + j];
                    cdouble h21 = H2[(i + 1) * restart + j];
                    H2[i * restart + j]       = std::conj(cs2[i]) * h20 + std::conj(sn2[i]) * h21;
                    H2[(i + 1) * restart + j] = -sn2[i] * h20 + cs2[i] * h21;
                }

                cdouble h10 = H1[j * restart + j];
                cdouble h11 = H1[(j + 1) * restart + j];
                double den1 = std::sqrt(std::norm(h10) + std::norm(h11));
                cs1[j] = (den1 > 1e-30) ? h10 / den1 : cdouble(1);
                sn1[j] = (den1 > 1e-30) ? h11 / den1 : cdouble(0);
                H1[j * restart + j] = std::conj(cs1[j]) * h10 + std::conj(sn1[j]) * h11;
                H1[(j + 1) * restart + j] = cdouble(0);
                cdouble s10 = s1[j];
                s1[j] = std::conj(cs1[j]) * s10;
                s1[j + 1] = -sn1[j] * s10;
                m1 = j + 1;
                last_rel1 = std::abs(s1[j + 1]) / bnorm1;
                if (last_rel1 < tol)
                    conv1 = true;

                cdouble h20 = H2[j * restart + j];
                cdouble h21 = H2[(j + 1) * restart + j];
                double den2 = std::sqrt(std::norm(h20) + std::norm(h21));
                cs2[j] = (den2 > 1e-30) ? h20 / den2 : cdouble(1);
                sn2[j] = (den2 > 1e-30) ? h21 / den2 : cdouble(0);
                H2[j * restart + j] = std::conj(cs2[j]) * h20 + std::conj(sn2[j]) * h21;
                H2[(j + 1) * restart + j] = cdouble(0);
                cdouble s20 = s2[j];
                s2[j] = std::conj(cs2[j]) * s20;
                s2[j + 1] = -sn2[j] * s20;
                m2 = j + 1;
                last_rel2 = std::abs(s2[j + 1]) / bnorm2;
                if (last_rel2 < tol)
                    conv2 = true;
            };

            if (pair_arnoldi && !conv1 && !conv2)
                arnoldi_pair();
            else {
                arnoldi(V1, H1, cs1, sn1, s1, w1, bnorm1, conv1, m1, last_rel1);
                arnoldi(V2, H2, cs2, sn2, s2, w2, bnorm2, conv2, m2, last_rel2);
            }

            if (verbose && (total_matvecs <= 3 || total_matvecs % 10 == 0)) {
                double rel1 = conv1 ? 0.0 : std::abs(s1[j + 1]) / bnorm1;
                double rel2 = conv2 ? 0.0 : std::abs(s2[j + 1]) / bnorm2;
                printf("    GMRES iter %d: rel1=%.2e rel2=%.2e%s%s\n",
                       total_matvecs, rel1, rel2,
                       conv1 ? " [1:done]" : "", conv2 ? " [2:done]" : "");
                fflush(stdout);
            }

            if (inner_stagnation_window > 0 && !(conv1 && conv2)) {
                double rel1 = conv1 ? 0.0 : last_rel1;
                double rel2 = conv2 ? 0.0 : last_rel2;
                double pair_est = std::max(rel1, rel2);
                double required = best_inner_rel * (1.0 - inner_stagnation_rel);
                if (pair_est < required) {
                    best_inner_rel = pair_est;
                    best_inner_iter = total_matvecs;
                } else if (total_matvecs >= inner_stagnation_min_iter &&
                           total_matvecs - best_inner_iter >= inner_stagnation_window) {
                    stopped_stagnant = true;
                    if (verbose) {
                        printf("  [GMRES-paired] Inner stagnation stop at iter %d: best_est=%.2e current_est=%.2e threshold=%.2g window=%d\n",
                               total_matvecs, best_inner_rel, pair_est,
                               inner_stagnation_rel, inner_stagnation_window);
                        fflush(stdout);
                    }
                    break;
                }
            }
        }

        if (gmres_step_update_pair(n, restart, m1, m2, H1, H2, s1, s2, V1, V2, Z1, Z2,
                                   has_precond, store_z, precond, x1, x2, ytmp, ytmp2, ztmp, ztmp2) != 0) {
            numerical_breakdown = true;
            if (verbose) {
                printf("  [GMRES-paired] numerical breakdown while solving Hessenberg least-squares update\n");
                fflush(stdout);
            }
            break;
        }

        // The Arnoldi/Givens residual is an estimate.  Recompute the true
        // residual after each update before accepting convergence; otherwise
        // restarted/preconditioned GMRES can report a solved system while
        // ||b - A*x|| is still above tolerance.
        op.matvec_batch2(x1, x2, r1.data(), r2.data());
        total_matvecs++;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            r1[i] = b1[i] - r1[i];
            r2[i] = b2[i] - r2[i];
        }

        rnorm1 = norm_p(r1.data(), n);
        rnorm2 = norm_p(r2.data(), n);
        last_rel1 = rnorm1 / bnorm1;
        last_rel2 = rnorm2 / bnorm2;
        conv1 = (last_rel1 < tol);
        conv2 = (last_rel2 < tol);
        double pair_rel = std::max(last_rel1, last_rel2);
        if (pair_rel < best_pair_rel) {
            best_pair_rel = pair_rel;
            best_true_rel1 = last_rel1;
            best_true_rel2 = last_rel2;
            std::copy(x1, x1 + n, best_x1.begin());
            std::copy(x2, x2 + n, best_x2.begin());
        }

        if (verbose) {
            printf("  [GMRES-paired] restart %d: true rel1=%.2e true rel2=%.2e\n",
                   cycle + 1, last_rel1, last_rel2);
            fflush(stdout);
        }

        if (stopped_stagnant)
            break;

        if (stagnation_cycles > 0 && !(conv1 && conv2)) {
            double rel_now = std::max(conv1 ? 0.0 : rnorm1 / bnorm1,
                                      conv2 ? 0.0 : rnorm2 / bnorm2);
            double required = best_rel * (1.0 - stagnation_rel);
            if (rel_now < required) {
                best_rel = rel_now;
                best_inner_rel = rel_now;
                best_inner_iter = total_matvecs;
                stagnant_restart_count = 0;
            } else {
                stagnant_restart_count++;
            }
            if (stagnant_restart_count >= stagnation_cycles) {
                stopped_stagnant = true;
                if (verbose) {
                    printf("  [GMRES-paired] Stagnation stop after %d restarts: best=%.2e current=%.2e threshold=%.2g\n",
                           cycle + 1, best_rel, rel_now, stagnation_rel);
                    fflush(stdout);
                }
                break;
            }
        }
    }

    double final_pair_rel = std::max(last_rel1, last_rel2);
    bool reached_max_cycles = !(conv1 && conv2) && !stopped_stagnant && !numerical_breakdown;
    if (!(conv1 && conv2) && best_pair_rel < final_pair_rel) {
        std::copy(best_x1.begin(), best_x1.end(), x1);
        std::copy(best_x2.begin(), best_x2.end(), x2);
        last_rel1 = best_true_rel1;
        last_rel2 = best_true_rel2;
        conv1 = (last_rel1 < tol);
        conv2 = (last_rel2 < tol);
        ws.restored_best_iterate = true;
        if (verbose) {
            printf("  [GMRES-paired] restored best true-residual iterate: res1=%.2e res2=%.2e\n",
                   last_rel1, last_rel2);
            fflush(stdout);
        }
    }

    if (verbose) {
        if (conv1 && conv2)
            printf("  [GMRES-paired] Both converged, %d matvec evaluations\n", total_matvecs);
        else if (stopped_stagnant)
            printf("  [GMRES-paired] STOPPED by stagnation guard, %d matvecs, res1=%.2e res2=%.2e\n",
                   total_matvecs,
                   conv1 ? 0.0 : rnorm1 / bnorm1,
                   conv2 ? 0.0 : rnorm2 / bnorm2);
        else if (numerical_breakdown)
            printf("  [GMRES-paired] NOT fully converged: numerical breakdown, %d matvecs, res1=%.2e res2=%.2e\n",
                   total_matvecs,
                   conv1 ? 0.0 : rnorm1 / bnorm1,
                   conv2 ? 0.0 : rnorm2 / bnorm2);
        else if (reached_max_cycles)
            printf("  [GMRES-paired] NOT fully converged: reached max cycles, %d matvecs, res1=%.2e res2=%.2e\n",
                   total_matvecs,
                   conv1 ? 0.0 : last_rel1,
                   conv2 ? 0.0 : last_rel2);
        else
            printf("  [GMRES-paired] NOT fully converged (%s%s), %d matvecs, res1=%.2e res2=%.2e\n",
                   conv1 ? "" : "sys1 ", conv2 ? "" : "sys2 ", total_matvecs,
                   conv1 ? 0.0 : rnorm1 / bnorm1,
                   conv2 ? 0.0 : rnorm2 / bnorm2);
        fflush(stdout);
    }

    ws.final_relres1 = last_rel1;
    ws.final_relres2 = last_rel2;
    ws.converged1 = conv1;
    ws.converged2 = conv2;
    ws.stopped_stagnant = stopped_stagnant;
    ws.numerical_breakdown = numerical_breakdown;
    ws.reached_max_cycles = reached_max_cycles;

    return total_matvecs;
}
