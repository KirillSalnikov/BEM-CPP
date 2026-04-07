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


std::vector<Orientation> generate_orientations(int n_alpha, int n_beta, int n_gamma,
                                               int beta_sym, int gamma_sym) {
    std::vector<double> mu_nodes, mu_weights;

    if (beta_sym == 2) {
        // Beta symmetry: generate GL(2*n_beta) on [-1,1], keep only positive nodes (cos>0 => beta<90)
        // This gives EXACT equivalence to full GL(2*n_beta) when f(beta) = f(pi-beta)
        std::vector<double> full_nodes, full_weights;
        gauss_legendre(2 * n_beta, full_nodes, full_weights);
        mu_nodes.resize(n_beta);
        mu_weights.resize(n_beta);
        int j = 0;
        for (int i = 0; i < 2 * n_beta; i++) {
            if (full_nodes[i] > 0) {  // cos(beta) > 0 => beta < 90
                mu_nodes[j] = full_nodes[i];
                mu_weights[j] = full_weights[i] * 2.0;  // double weight for symmetry
                j++;
            }
        }
    } else {
        gauss_legendre(n_beta, mu_nodes, mu_weights);
    }

    double d_alpha = 2.0 * M_PI / n_alpha;
    // Weight uses full 2pi/n_gamma (symmetry factor already accounted for)
    double d_gamma = 2.0 * M_PI / n_gamma;
    // Actual gamma range is [0, 2pi/gamma_sym)
    double gamma_step = 2.0 * M_PI / (gamma_sym * n_gamma);

    std::vector<Orientation> orients;
    orients.reserve(n_alpha * n_beta * n_gamma);

    for (int ia = 0; ia < n_alpha; ia++) {
        double alpha = ia * d_alpha;
        for (int ib = 0; ib < n_beta; ib++) {
            double beta = acos(mu_nodes[ib]);
            double w_beta = mu_weights[ib];
            for (int ig = 0; ig < n_gamma; ig++) {
                double gamma = ig * gamma_step;

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

void sort_orientations_nearest(std::vector<Orientation>& orients) {
    int n = (int)orients.size();
    if (n <= 2) return;

    std::vector<bool> used(n, false);
    std::vector<Orientation> sorted;
    sorted.reserve(n);

    // Start from first orientation
    sorted.push_back(orients[0]);
    used[0] = true;

    for (int step = 1; step < n; step++) {
        const Mat3& Rprev = sorted.back().RT;
        int best = -1;
        double best_d = 1e30;

        for (int j = 0; j < n; j++) {
            if (used[j]) continue;
            // Frobenius distance between rotation matrices
            double d = 0;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++) {
                    double diff = Rprev.m[r][c] - orients[j].RT.m[r][c];
                    d += diff * diff;
                }
            if (d < best_d) {
                best_d = d;
                best = j;
            }
        }
        sorted.push_back(orients[best]);
        used[best] = true;
    }

    orients = sorted;
}
