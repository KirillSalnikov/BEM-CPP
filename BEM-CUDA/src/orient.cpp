#include "orient.h"
#include <cmath>

Mat3 euler_rotation(double alpha, double beta, double gamma) {
    double ca = cos(alpha), sa = sin(alpha);
    double cb = cos(beta),  sb = sin(beta);
    double cg = cos(gamma), sg = sin(gamma);

    // R = Rz(alpha) * Ry(beta) * Rz(gamma)
    Mat3 Rz1, Ry, Rz2;
    Rz1.m[0][0] = ca;  Rz1.m[0][1] = -sa; Rz1.m[0][2] = 0;
    Rz1.m[1][0] = sa;  Rz1.m[1][1] = ca;  Rz1.m[1][2] = 0;
    Rz1.m[2][0] = 0;   Rz1.m[2][1] = 0;   Rz1.m[2][2] = 1;

    Ry.m[0][0] = cb;   Ry.m[0][1] = 0;    Ry.m[0][2] = sb;
    Ry.m[1][0] = 0;    Ry.m[1][1] = 1;    Ry.m[1][2] = 0;
    Ry.m[2][0] = -sb;  Ry.m[2][1] = 0;    Ry.m[2][2] = cb;

    Rz2.m[0][0] = cg;  Rz2.m[0][1] = -sg; Rz2.m[0][2] = 0;
    Rz2.m[1][0] = sg;  Rz2.m[1][1] = cg;  Rz2.m[1][2] = 0;
    Rz2.m[2][0] = 0;   Rz2.m[2][1] = 0;   Rz2.m[2][2] = 1;

    // R = Rz1 * Ry * Rz2
    Mat3 R;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            R.m[i][j] = 0;
            for (int k = 0; k < 3; k++) {
                double tmp = 0;
                for (int l = 0; l < 3; l++)
                    tmp += Ry.m[k][l] * Rz2.m[l][j];
                R.m[i][j] += Rz1.m[i][k] * tmp;
            }
        }
    return R;
}


void gauss_legendre(int n, std::vector<double>& nodes, std::vector<double>& weights) {
    nodes.resize(n);
    weights.resize(n);

    // Newton's method for Legendre polynomial roots
    for (int i = 0; i < n; i++) {
        // Initial guess (Tricomi approximation)
        double x = cos(M_PI * (i + 0.75) / (n + 0.5));

        for (int iter = 0; iter < 100; iter++) {
            double p0 = 1.0, p1 = x;
            for (int j = 2; j <= n; j++) {
                double p2 = ((2*j - 1) * x * p1 - (j - 1) * p0) / j;
                p0 = p1; p1 = p2;
            }
            // p1 = P_n(x), derivative: P'_n = n*(x*P_n - P_{n-1})/(x^2-1)
            double dp = n * (x * p1 - p0) / (x * x - 1.0);
            double dx = -p1 / dp;
            x += dx;
            if (fabs(dx) < 1e-15) break;
        }
        nodes[i] = x;
        // Weight
        double p0 = 1.0, p1 = x;
        for (int j = 2; j <= n; j++) {
            double p2 = ((2*j - 1) * x * p1 - (j - 1) * p0) / j;
            p0 = p1; p1 = p2;
        }
        double dp = n * (x * p1 - p0) / (x * x - 1.0);
        weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
}


std::vector<Orientation> generate_orientations(int n_alpha, int n_beta, int n_gamma) {
    std::vector<double> mu_nodes, mu_weights;
    gauss_legendre(n_beta, mu_nodes, mu_weights);

    double d_alpha = 2.0 * M_PI / n_alpha;
    double d_gamma = 2.0 * M_PI / n_gamma;

    std::vector<Orientation> orients;
    orients.reserve(n_alpha * n_beta * n_gamma);

    for (int ia = 0; ia < n_alpha; ia++) {
        double alpha = ia * d_alpha;
        for (int ib = 0; ib < n_beta; ib++) {
            double beta = acos(mu_nodes[ib]);
            double w_beta = mu_weights[ib];
            for (int ig = 0; ig < n_gamma; ig++) {
                double gamma = ig * d_gamma;

                Mat3 R = euler_rotation(alpha, beta, gamma);
                Orientation o;
                o.RT = R.T();
                o.weight = d_alpha * w_beta * d_gamma / (8.0 * M_PI * M_PI);
                orients.push_back(o);
            }
        }
    }
    return orients;
}
