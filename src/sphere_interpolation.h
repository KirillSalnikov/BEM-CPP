#ifndef BEM_SPHERE_INTERPOLATION_H
#define BEM_SPHERE_INTERPOLATION_H

#include "sphere_quad.h"
#include <stdexcept>

// Exact on spherical polynomials of degree <= fine.p. Fourier modes of even
// order are polynomials in cos(theta); odd modes are sin(theta) times a
// polynomial. Keeping these two parities separate makes the global spherical
// interpolant a sum of two tensor products, without a dense L_coarse*L_fine
// matrix. The Nyquist mode is deliberately omitted (it lies above fine.p).
struct SphereInterpolation {
    int ntf, npf, ntc, npc;
    std::vector<double> theta_even, theta_odd, phi_even, phi_odd;
    SphereInterpolation(const SphereQuad& fine, const SphereQuad& coarse)
        : ntf(fine.n_theta), npf(fine.n_phi),
          ntc(coarse.n_theta), npc(coarse.n_phi) {
        if (fine.p > coarse.p) throw std::runtime_error("invalid angular interpolation orders");
        theta_even.resize(ntc * ntf); theta_odd.resize(ntc * ntf);
        phi_even.resize(npc * npf); phi_odd.resize(npc * npf);
        std::vector<double> bary(ntf);
        for (int j = 0; j < ntf; ++j) {
            const double x = fine.dirs[3 * j * npf + 2];
            // At a GL root, 1/P'_n(x) is a barycentric weight.
            double a = 1., b = x;
            for (int l = 2; l <= ntf; ++l) {
                double c = ((2*l-1)*x*b-(l-1)*a)/l; a=b; b=c;
            }
            bary[j] = (x*x-1.)/(ntf*(x*b-a));
        }
        for (int i = 0; i < ntc; ++i) {
            const double x = coarse.dirs[3*i*npc+2];
            int exact = -1; double sum = 0.;
            for (int j = 0; j < ntf; ++j) {
                const double y = fine.dirs[3*j*npf+2];
                if (std::abs(x-y) < 1e-14) exact = j;
                else sum += bary[j]/(x-y);
            }
            for (int j = 0; j < ntf; ++j) {
                const double y = fine.dirs[3*j*npf+2];
                const double v = exact >= 0 ? double(j==exact) : bary[j]/((x-y)*sum);
                theta_even[i*ntf+j] = v;
                theta_odd[i*ntf+j] = v*std::sqrt((1.-x*x)/(1.-y*y));
            }
        }
        for (int i = 0; i < npc; ++i) for (int j = 0; j < npf; ++j) {
            const double delta = 2*M_PI*(double(i)/npc-double(j)/npf);
            double even = 1., odd = 0.;
            for (int m = 1; m <= fine.p; ++m) {
                if (m%2) odd += 2*std::cos(m*delta);
                else even += 2*std::cos(m*delta);
            }
            phi_even[i*npf+j] = even/npf;
            phi_odd[i*npf+j] = odd/npf;
        }
    }
};
#endif
