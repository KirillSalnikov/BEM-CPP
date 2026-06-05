#include "block_gmres.h"
#include "bem_fmm.h"
#include "precond.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static cdouble dot_conj_p(const cdouble* a, const cdouble* b, int n) {
    double sr = 0.0, si = 0.0;
    #pragma omp parallel for reduction(+:sr,si) schedule(static)
    for (int i = 0; i < n; i++) {
        cdouble v = std::conj(a[i]) * b[i];
        sr += v.real();
        si += v.imag();
    }
    return cdouble(sr, si);
}

static double norm_p(const cdouble* a, int n) {
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int i = 0; i < n; i++)
        s += std::norm(a[i]);
    return std::sqrt(s);
}

static int gmres_step_update(int n, int restart, int m,
                             const std::vector<cdouble>& H,
                             const std::vector<cdouble>& s,
                             const std::vector<cdouble>& V,
                             const std::vector<cdouble>& Z,
                             bool has_precond,
                             bool store_z,
                             NearFieldPrecond* precond,
                             std::vector<cdouble>& x)
{
    if (m <= 0)
        return 0;

    std::vector<cdouble> y(m);
    for (int i = m - 1; i >= 0; i--) {
        y[i] = s[i];
        for (int k = i + 1; k < m; k++)
            y[i] -= H[i * restart + k] * y[k];
        y[i] /= H[i * restart + i];
    }

    std::vector<cdouble> ztmp;
    if (has_precond && !store_z)
        ztmp.resize(n);
    for (int i = 0; i < m; i++) {
        const cdouble* vi = &V[(size_t)i * n];
        if (has_precond) {
            if (store_z) {
                vi = &Z[(size_t)i * n];
            } else {
                precond->apply(vi, ztmp.data());
                vi = ztmp.data();
            }
        }
        cdouble yi = y[i];
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; k++)
            x[k] += yi * vi[k];
    }
    return 0;
}

int gmres_solve_paired(BemFmmOperator& op,
                       const cdouble* b1, const cdouble* b2,
                       cdouble* x1, cdouble* x2,
                       int restart, double tol, int maxiter,
                       bool verbose, NearFieldPrecond* precond)
{
    int n = op.system_size;
    bool has_precond = (precond != nullptr);
    const char* env_reorth = std::getenv("BEM_GMRES_REORTH");
    bool reorth = !(env_reorth && std::atoi(env_reorth) == 0);
    bool store_z = false;
    if (has_precond) {
        const char* env_store_z = std::getenv("BEM_GMRES_STORE_Z");
        store_z = (env_store_z && std::atoi(env_store_z) != 0);
    }

    std::vector<cdouble> r1(b1, b1 + n), r2(b2, b2 + n);
    std::vector<cdouble> w1(n), w2(n), z1(n), z2(n);
    std::vector<cdouble> hx1(x1, x1 + n), hx2(x2, x2 + n);
    std::vector<cdouble> V1((size_t)n * (restart + 1)), V2((size_t)n * (restart + 1));
    std::vector<cdouble> Z1, Z2;
    if (has_precond && store_z) {
        Z1.resize((size_t)n * restart);
        Z2.resize((size_t)n * restart);
    }

    std::vector<cdouble> H1((restart + 1) * restart), H2((restart + 1) * restart);
    std::vector<cdouble> cs1(restart), sn1(restart), s1(restart + 1);
    std::vector<cdouble> cs2(restart), sn2(restart), s2(restart + 1);

    double bnorm1 = norm_p(b1, n), bnorm2 = norm_p(b2, n);
    if (bnorm1 < 1e-30) bnorm1 = 1.0;
    if (bnorm2 < 1e-30) bnorm2 = 1.0;

    bool warm1 = (norm_p(hx1.data(), n) > 1e-30);
    bool warm2 = (norm_p(hx2.data(), n) > 1e-30);
    int total_matvecs = 0;

    if (warm1 && warm2) {
        op.matvec_batch2(hx1.data(), hx2.data(), r1.data(), r2.data());
        total_matvecs++;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            r1[i] = b1[i] - r1[i];
            r2[i] = b2[i] - r2[i];
        }
    } else if (warm1) {
        op.matvec(hx1.data(), r1.data());
        total_matvecs++;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            r1[i] = b1[i] - r1[i];
    } else if (warm2) {
        op.matvec(hx2.data(), r2.data());
        total_matvecs++;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            r2[i] = b2[i] - r2[i];
    }

    double rnorm1 = norm_p(r1.data(), n), rnorm2 = norm_p(r2.data(), n);

    bool conv1 = (rnorm1 / bnorm1 < tol);
    bool conv2 = (rnorm2 / bnorm2 < tol);

    if (verbose)
        printf("  [GMRES-paired] start: res1=%.2e res2=%.2e%s\n",
               rnorm1 / bnorm1, rnorm2 / bnorm2,
               (warm1 || warm2) ? " (warm)" : "");

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
        for (int j = 0; j < restart && !(conv1 && conv2); j++) {
            if (!conv1 && !conv2) {
                const cdouble* in1 = &V1[(size_t)j * n];
                const cdouble* in2 = &V2[(size_t)j * n];
                if (has_precond) {
                    precond->apply(in1, z1.data());
                    precond->apply(in2, z2.data());
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
                               double bnorm, bool& conv, int& m) {
                if (conv)
                    return;
                for (int i = 0; i <= j; i++) {
                    cdouble* vi = &V[(size_t)i * n];
                    cdouble hij = dot_conj_p(vi, w.data(), n);
                    H[i * restart + j] = hij;
                    #pragma omp parallel for schedule(static)
                    for (int k = 0; k < n; k++)
                        w[k] -= hij * vi[k];
                }
                if (reorth) {
                    for (int i = 0; i <= j; i++) {
                        cdouble* vi = &V[(size_t)i * n];
                        cdouble hij = dot_conj_p(vi, w.data(), n);
                        H[i * restart + j] += hij;
                        #pragma omp parallel for schedule(static)
                        for (int k = 0; k < n; k++)
                            w[k] -= hij * vi[k];
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
                if (std::abs(s[j + 1]) / bnorm < tol)
                    conv = true;
            };

            arnoldi(V1, H1, cs1, sn1, s1, w1, bnorm1, conv1, m1);
            arnoldi(V2, H2, cs2, sn2, s2, w2, bnorm2, conv2, m2);

            if (verbose && (total_matvecs <= 3 || total_matvecs % 10 == 0)) {
                double rel1 = conv1 ? 0.0 : std::abs(s1[j + 1]) / bnorm1;
                double rel2 = conv2 ? 0.0 : std::abs(s2[j + 1]) / bnorm2;
                printf("    GMRES iter %d: rel1=%.2e rel2=%.2e%s%s\n",
                       total_matvecs, rel1, rel2,
                       conv1 ? " [1:done]" : "", conv2 ? " [2:done]" : "");
            }
        }

        gmres_step_update(n, restart, m1, H1, s1, V1, Z1, has_precond, store_z, precond, hx1);
        gmres_step_update(n, restart, m2, H2, s2, V2, Z2, has_precond, store_z, precond, hx2);

        if (conv1 && conv2)
            break;

        if (!conv1 && !conv2) {
            op.matvec_batch2(hx1.data(), hx2.data(), r1.data(), r2.data());
            total_matvecs++;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                r1[i] = b1[i] - r1[i];
                r2[i] = b2[i] - r2[i];
            }
        } else if (!conv1) {
            op.matvec(hx1.data(), r1.data());
            total_matvecs++;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++)
                r1[i] = b1[i] - r1[i];
        } else if (!conv2) {
            op.matvec(hx2.data(), r2.data());
            total_matvecs++;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++)
                r2[i] = b2[i] - r2[i];
        }

        if (!conv1) {
            rnorm1 = norm_p(r1.data(), n);
            conv1 = (rnorm1 / bnorm1 < tol);
        }
        if (!conv2) {
            rnorm2 = norm_p(r2.data(), n);
            conv2 = (rnorm2 / bnorm2 < tol);
        }

        if (verbose)
            printf("  [GMRES-paired] restart %d: rel1=%.2e rel2=%.2e\n",
                   cycle + 1, conv1 ? 0.0 : rnorm1 / bnorm1,
                   conv2 ? 0.0 : rnorm2 / bnorm2);
    }

    memcpy(x1, hx1.data(), n * sizeof(cdouble));
    memcpy(x2, hx2.data(), n * sizeof(cdouble));

    if (verbose) {
        if (conv1 && conv2)
            printf("  [GMRES-paired] Both converged, %d matvec evaluations\n", total_matvecs);
        else
            printf("  [GMRES-paired] NOT fully converged (%s%s), %d matvecs, res1=%.2e res2=%.2e\n",
                   conv1 ? "" : "sys1 ", conv2 ? "" : "sys2 ", total_matvecs,
                   conv1 ? 0.0 : rnorm1 / bnorm1,
                   conv2 ? 0.0 : rnorm2 / bnorm2);
    }

    return total_matvecs;
}
