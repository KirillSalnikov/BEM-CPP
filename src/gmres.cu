#include "gmres.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static cdouble dot_conj(const cdouble* a, const cdouble* b, int n) {
    cdouble s(0);
    for (int i = 0; i < n; i++)
        s += std::conj(a[i]) * b[i];
    return s;
}

static double norm2(const cdouble* a, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += std::norm(a[i]);
    return std::sqrt(s);
}

int gmres_solve(BemFmmOperator& op, const cdouble* b, cdouble* x,
                int restart, double tol, int maxiter, bool verbose,
                NearFieldPrecond* precond)
{
    int n = op.system_size;
    const char* env_reorth = std::getenv("BEM_GMRES_REORTH");
    bool reorth = !(env_reorth && std::atoi(env_reorth) == 0);
    bool store_z = false;
    if (precond) {
        const char* env_store_z = std::getenv("BEM_GMRES_STORE_Z");
        store_z = (env_store_z && std::atoi(env_store_z) != 0);
    }

    std::vector<cdouble> r(n), w(n), z(n), xh(n, cdouble(0));
    std::vector<cdouble> V((size_t)n * (restart + 1));
    std::vector<cdouble> Z;
    if (precond && store_z)
        Z.resize((size_t)n * restart);

    std::vector<cdouble> H((restart + 1) * restart, cdouble(0));
    std::vector<cdouble> s(restart + 1), cs(restart), sn(restart), y(restart);

    memcpy(r.data(), b, n * sizeof(cdouble));
    double bnorm = norm2(b, n);
    if (bnorm < 1e-30) bnorm = 1.0;
    double rnorm = norm2(r.data(), n);

    if (verbose)
        printf("  [GMRES] start: ||r||/||b|| = %.2e\n", rnorm / bnorm);

    int total_iters = 0;
    bool converged = false;

    for (int cycle = 0; cycle < maxiter && !converged; cycle++) {
        double inv_rnorm = 1.0 / rnorm;
        for (int i = 0; i < n; i++)
            V[i] = r[i] * inv_rnorm;

        std::fill(s.begin(), s.end(), cdouble(0));
        s[0] = cdouble(rnorm);
        std::fill(H.begin(), H.end(), cdouble(0));

        int j;
        for (j = 0; j < restart; j++) {
            total_iters++;
            cdouble* vj = &V[(size_t)j * n];

            if (precond) {
                precond->apply(vj, z.data());
                if (store_z)
                    memcpy(&Z[(size_t)j * n], z.data(), n * sizeof(cdouble));
                op.matvec(z.data(), w.data());
            } else {
                op.matvec(vj, w.data());
            }

            for (int i = 0; i <= j; i++) {
                cdouble* vi = &V[(size_t)i * n];
                cdouble hij = dot_conj(vi, w.data(), n);
                H[i * restart + j] = hij;
                for (int k = 0; k < n; k++)
                    w[k] -= hij * vi[k];
            }
            if (reorth) {
                for (int i = 0; i <= j; i++) {
                    cdouble* vi = &V[(size_t)i * n];
                    cdouble hij = dot_conj(vi, w.data(), n);
                    H[i * restart + j] += hij;
                    for (int k = 0; k < n; k++)
                        w[k] -= hij * vi[k];
                }
            }

            double w_norm = norm2(w.data(), n);
            H[(j + 1) * restart + j] = cdouble(w_norm);
            if (w_norm > 1e-30) {
                cdouble* vnext = &V[(size_t)(j + 1) * n];
                double inv = 1.0 / w_norm;
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
            double denom = std::sqrt(std::norm(h0) + std::norm(h1));
            if (denom > 1e-30) {
                cs[j] = h0 / denom;
                sn[j] = h1 / denom;
            } else {
                cs[j] = cdouble(1);
                sn[j] = cdouble(0);
            }

            H[j * restart + j] = std::conj(cs[j]) * h0 + std::conj(sn[j]) * h1;
            H[(j + 1) * restart + j] = cdouble(0);

            cdouble s0 = s[j];
            s[j]     = std::conj(cs[j]) * s0;
            s[j + 1] = -sn[j] * s0;

            double rel_res = std::abs(s[j + 1]) / bnorm;
            if (verbose && (total_iters <= 3 || total_iters % 10 == 0))
                printf("    GMRES iter %d: rel=%.2e\n", total_iters, rel_res);

            if (rel_res < tol) {
                j++;
                converged = true;
                break;
            }
        }

        int m = j;
        for (int i = m - 1; i >= 0; i--) {
            y[i] = s[i];
            for (int k = i + 1; k < m; k++)
                y[i] -= H[i * restart + k] * y[k];
            y[i] /= H[i * restart + i];
        }

        if (precond) {
            std::vector<cdouble> ztmp;
            if (!store_z)
                ztmp.resize(n);
            for (int i = 0; i < m; i++) {
                const cdouble* zi = nullptr;
                if (store_z) {
                    zi = &Z[(size_t)i * n];
                } else {
                    const cdouble* vi = &V[(size_t)i * n];
                    precond->apply(vi, ztmp.data());
                    zi = ztmp.data();
                }
                for (int k = 0; k < n; k++)
                    xh[k] += y[i] * zi[k];
            }
        } else {
            for (int i = 0; i < m; i++) {
                const cdouble* vi = &V[(size_t)i * n];
                for (int k = 0; k < n; k++)
                    xh[k] += y[i] * vi[k];
            }
        }

        if (converged)
            break;

        op.matvec(xh.data(), r.data());
        for (int i = 0; i < n; i++)
            r[i] = b[i] - r[i];
        rnorm = norm2(r.data(), n);

        if (verbose)
            printf("  [GMRES] restart %d: ||r||/||b|| = %.2e\n", cycle + 1, rnorm / bnorm);

        if (rnorm / bnorm < tol)
            converged = true;
    }

    memcpy(x, xh.data(), n * sizeof(cdouble));

    if (verbose) {
        if (converged)
            printf("  [GMRES] Converged in %d iterations\n", total_iters);
        else
            printf("  [GMRES] NOT converged after %d iterations, res=%.2e\n",
                   total_iters, rnorm / bnorm);
    }

    return converged ? 0 : 1;
}
