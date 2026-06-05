#include "rhs.h"
#include "quadrature.h"
#include <cstring>
#include <cmath>

void compute_rhs_planewave(const RWG& rwg, const Mesh& mesh,
                           std::complex<double> k_ext, double eta_ext,
                           const Vec3& E0, const Vec3& k_hat,
                           int quad_order,
                           std::complex<double>* b)
{
    int N = rwg.N;
    TriQuad quad = tri_quadrature(quad_order);
    int Nq = quad.npts;

    // H0 = k_hat x E0 / eta_ext
    Vec3 H0 = k_hat.cross(E0) * (1.0 / eta_ext);

    std::vector<double> lam0(Nq);
    for (int q = 0; q < Nq; q++)
        lam0[q] = 1.0 - quad.pts[q][0] - quad.pts[q][1];

    memset(b, 0, 2 * N * sizeof(std::complex<double>));

    // Two halves: plus (+1) and minus (-1)
    for (int half = 0; half < 2; half++) {
        int sign = (half == 0) ? +1 : -1;

        for (int n = 0; n < N; n++) {
            int ti = (sign > 0) ? rwg.tri_p[n] : rwg.tri_m[n];
            Vec3 free_v = (sign > 0) ? rwg.free_p[n] : rwg.free_m[n];
            double area = (sign > 0) ? rwg.area_p[n] : rwg.area_m[n];
            double len = rwg.length[n];
            double coeff = sign * len / (2.0 * area);

            Vec3 v0, v1, v2;
            mesh.tri_verts(ti, v0, v1, v2);

            std::complex<double> bE(0), bH(0);

            for (int q = 0; q < Nq; q++) {
                double l0 = lam0[q], l1 = quad.pts[q][0], l2 = quad.pts[q][1];
                Vec3 rr = v0 * l0 + v1 * l1 + v2 * l2;

                // Basis function value
                Vec3 fv = (rr - free_v) * coeff;

                // Phase: exp(i * k_ext * k_hat . r)
                double phase_arg = k_ext.real() * k_hat.dot(rr);
                double phase_arg_im = k_ext.imag() * k_hat.dot(rr);
                std::complex<double> phase = std::exp(
                    std::complex<double>(-phase_arg_im, phase_arg));

                double jw = area * quad.wts[q];

                bE += fv.dot(E0) * phase * jw;
                bH += fv.dot(H0) * phase * jw;
            }

            b[n]     += bE;
            b[N + n] += bH;
        }
    }
}
