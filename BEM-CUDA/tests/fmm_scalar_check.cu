#include "fmm.h"
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double urand(unsigned& s)
{
    s = 1664525u * s + 1013904223u;
    return (double)(s & 0x00ffffffu) / (double)0x01000000u;
}

int main(int argc, char** argv)
{
    int n = argc > 1 ? std::atoi(argv[1]) : 1200;
    double k_re = argc > 2 ? std::atof(argv[2]) : 10.0;
    int digits = argc > 3 ? std::atoi(argv[3]) : 8;
    int leaf = argc > 4 ? std::atoi(argv[4]) : 128;

    std::vector<double> pts(3 * n);
    std::vector<cdouble> q(n), yf(n), yd(n), gf(3 * n), gd(3 * n);
    unsigned seed = 12345u;
    for (int i = 0; i < n; i++) {
        double z = 2.0 * urand(seed) - 1.0;
        double phi = 2.0 * M_PI * urand(seed);
        double r = std::cbrt(urand(seed));
        double st = std::sqrt(std::max(0.0, 1.0 - z * z));
        pts[3 * i + 0] = r * st * std::cos(phi);
        pts[3 * i + 1] = r * st * std::sin(phi);
        pts[3 * i + 2] = r * z;
        q[i] = cdouble(2.0 * urand(seed) - 1.0, 2.0 * urand(seed) - 1.0);
    }

    cdouble k(k_re, 0.0);
    const double inv4pi = 1.0 / (4.0 * M_PI);
    for (int i = 0; i < n; i++) {
        cdouble acc(0.0, 0.0);
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            double dx = pts[3 * i + 0] - pts[3 * j + 0];
            double dy = pts[3 * i + 1] - pts[3 * j + 1];
            double dz = pts[3 * i + 2] - pts[3 * j + 2];
            double R = std::sqrt(dx * dx + dy * dy + dz * dz);
            cdouble G = std::exp(cdouble(0.0, 1.0) * k * R) * inv4pi / R;
            acc += G * q[j];
            cdouble grad_scalar = G * (cdouble(0.0, 1.0) * k - 1.0 / R) / R * q[j];
            gd[3 * i + 0] += grad_scalar * dx;
            gd[3 * i + 1] += grad_scalar * dy;
            gd[3 * i + 2] += grad_scalar * dz;
        }
        yd[i] = acc;
    }

    HelmholtzFMM fmm;
    fmm.init(pts.data(), n, pts.data(), n, k, digits, leaf);
    fmm.evaluate(q.data(), yf.data());
    fmm.evaluate_gradient(q.data(), gf.data());

    double nd = 0.0, ne = 0.0, dot_re = 0.0;
    double gnd = 0.0, gne = 0.0, gdot_re = 0.0;
    for (int i = 0; i < n; i++) {
        nd += std::norm(yd[i]);
        ne += std::norm(yf[i] - yd[i]);
        dot_re += (yf[i] * std::conj(yd[i])).real();
        for (int d = 0; d < 3; d++) {
            int idx = 3 * i + d;
            gnd += std::norm(gd[idx]);
            gne += std::norm(gf[idx] - gd[idx]);
            gdot_re += (gf[idx] * std::conj(gd[idx])).real();
        }
    }
    printf("n=%d k=%.6g digits=%d leaf=%d pot_rel_l2=%.17g pot_scale_re=%.17g grad_rel_l2=%.17g grad_scale_re=%.17g\n",
           n, k_re, digits, leaf, std::sqrt(ne / nd), dot_re / nd,
           std::sqrt(gne / gnd), gdot_re / gnd);
    return 0;
}
