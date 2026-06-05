#ifndef BEM_SPHERE_QUAD_H
#define BEM_SPHERE_QUAD_H

#include <vector>
#include <cmath>
#include <cstdio>

// Truncation order for FMM plane-wave expansion
inline int fmm_truncation_order(double k_abs, double box_size, int digits = 3) {
    double ka = k_abs * box_size;
    double c;
    switch (digits) {
        case 2: c = 3.0; break;
        case 3: c = 5.0; break;
        case 4: c = 7.0; break;
        case 5: c = 9.0; break;
        default: c = 5.0; break;
    }
    int p = (int)std::ceil(ka + c * std::pow(std::max(ka, 1.0), 1.0/3.0));
    if (p < 3) p = 3;
    return p;
}

// Sphere quadrature: (p+1) Gauss-Legendre theta nodes x (2p+2) uniform phi nodes
// Returns L = (p+1)*(2p+2) directions + weights
struct SphereQuad {
    int p;           // truncation order
    int L;           // total number of directions
    int n_theta;     // p+1
    int n_phi;       // 2p+2

    std::vector<double> dirs;     // (L*3) unit direction vectors [x,y,z]
    std::vector<double> weights;  // (L) quadrature weights
    std::vector<double> theta;    // (L) theta angles
    std::vector<double> phi;      // (L) phi angles

    void init(int p_order) {
        p = p_order;
        n_theta = p + 1;
        n_phi = 2 * p + 2;
        L = n_theta * n_phi;

        // Gauss-Legendre nodes for cos(theta) on [-1, 1]
        std::vector<double> mu(n_theta), w_mu(n_theta);
        gauss_legendre_nodes(n_theta, mu.data(), w_mu.data());

        double d_phi = 2.0 * M_PI / n_phi;

        dirs.resize(L * 3);
        weights.resize(L);
        theta.resize(L);
        phi.resize(L);

        int idx = 0;
        for (int it = 0; it < n_theta; it++) {
            double th = std::acos(mu[it]);
            double st = std::sin(th);
            double ct = mu[it];  // cos(theta)
            for (int ip = 0; ip < n_phi; ip++) {
                double ph = ip * d_phi;
                double cp = std::cos(ph);
                double sp = std::sin(ph);
                dirs[idx*3]     = st * cp;
                dirs[idx*3 + 1] = st * sp;
                dirs[idx*3 + 2] = ct;
                weights[idx] = w_mu[it] * d_phi;
                theta[idx] = th;
                phi[idx] = ph;
                idx++;
            }
        }

        printf("  [SphereQuad] p=%d, L=%d (%d theta x %d phi)\n", p, L, n_theta, n_phi);
    }

private:
    // Gauss-Legendre nodes and weights via Newton iteration
    static void gauss_legendre_nodes(int n, double* nodes, double* wts) {
        for (int i = 0; i < n; i++) {
            // Initial guess
            double x = std::cos(M_PI * (i + 0.75) / (n + 0.5));
            for (int iter = 0; iter < 100; iter++) {
                double p0 = 1.0, p1 = x;
                for (int j = 2; j <= n; j++) {
                    double p2 = ((2*j - 1) * x * p1 - (j - 1) * p0) / j;
                    p0 = p1; p1 = p2;
                }
                double dp = n * (x * p1 - p0) / (x * x - 1.0);
                double dx = p1 / dp;
                x -= dx;
                if (std::fabs(dx) < 1e-15) break;
            }
            nodes[i] = x;
            double p0 = 1.0, p1 = x;
            for (int j = 2; j <= n; j++) {
                double p2 = ((2*j - 1) * x * p1 - (j - 1) * p0) / j;
                p0 = p1; p1 = p2;
            }
            double dp = n * (x * p1 - p0) / (x * x - 1.0);
            wts[i] = 2.0 / ((1.0 - x * x) * dp * dp);
        }
    }
};

#endif // BEM_SPHERE_QUAD_H
