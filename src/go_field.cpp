#include "go_field.h"
#include "quadrature.h"
#include <cstring>
#include <cmath>

void compute_go_initial_guess(
    const Mesh& mesh, const RWG& rwg,
    cdouble k_ext, cdouble m_ri,
    double eta_ext,
    const Vec3& E0, const Vec3& k_hat,
    int quad_order,
    cdouble* x_go)
{
    int N = rwg.N;
    memset(x_go, 0, 2 * N * sizeof(cdouble));

    TriQuad quad = tri_quadrature(quad_order);
    int Nq = quad.npts;

    Vec3 H0 = k_hat.cross(E0) * (1.0 / eta_ext);

    // Precompute lambda0
    std::vector<double> lam0(Nq);
    for (int q = 0; q < Nq; q++)
        lam0[q] = 1.0 - quad.pts[q][0] - quad.pts[q][1];

    // Process plus and minus halves (same structure as rhs.cpp)
    for (int half = 0; half < 2; half++) {
        int sign = (half == 0) ? +1 : -1;

        for (int n = 0; n < N; n++) {
            int ti    = (sign > 0) ? rwg.tri_p[n] : rwg.tri_m[n];
            Vec3 free_v = (sign > 0) ? rwg.free_p[n] : rwg.free_m[n];
            double area = (sign > 0) ? rwg.area_p[n] : rwg.area_m[n];
            double len  = rwg.length[n];
            double coeff = sign * len / (2.0 * area);

            Vec3 v0, v1, v2;
            mesh.tri_verts(ti, v0, v1, v2);

            // Outward normal from vertex ordering
            Vec3 n_hat = (v1 - v0).cross(v2 - v0).normalized();

            // Illumination check
            double cos_nk = n_hat.dot(k_hat);
            if (cos_nk >= 0) continue;  // shadow side — skip
            double cosI = -cos_nk;

            // Fresnel: complex Snell
            double sinIsq = 1.0 - cosI * cosI;
            cdouble gamma = std::sqrt(m_ri * m_ri - cdouble(sinIsq, 0));
            if (gamma.imag() < 0) gamma = -gamma;

            cdouble Rs = (cdouble(cosI) - gamma) / (cdouble(cosI) + gamma);
            cdouble Rp = (m_ri * m_ri * cdouble(cosI) - gamma) /
                         (m_ri * m_ri * cdouble(cosI) + gamma);

            // s/p decomposition basis
            Vec3 s_hat = k_hat.cross(n_hat);
            double slen = s_hat.norm();
            if (slen < 1e-12) {
                // Normal incidence — pick arbitrary perpendicular
                Vec3 tmp = (fabs(k_hat.x) < 0.9) ? Vec3(1,0,0) : Vec3(0,1,0);
                s_hat = k_hat.cross(tmp);
                slen = s_hat.norm();
            }
            s_hat = s_hat * (1.0 / slen);

            Vec3 p_inc = s_hat.cross(k_hat);   // p-direction for incident wave

            // Reflected direction: k_ref = k_hat - 2*(k_hat·n̂)*n̂
            Vec3 k_ref = k_hat - n_hat * (2.0 * cos_nk);
            Vec3 p_ref = s_hat.cross(k_ref);   // p-direction for reflected wave

            // E0 decomposition into s,p (real vectors, phase-independent)
            double E0_s = E0.dot(s_hat);
            double E0_p = E0.dot(p_inc);

            // Reflected E0 amplitudes (complex from Fresnel)
            cdouble Er_s = Rs * E0_s;
            cdouble Er_p = Rp * E0_p;

            Vec3 kxs  = k_ref.cross(s_hat);
            Vec3 kxp  = k_ref.cross(p_ref);
            double inv_eta = 1.0 / eta_ext;

            cdouble bJ(0), bM(0);

            for (int q = 0; q < Nq; q++) {
                Vec3 rr = v0 * lam0[q] + v1 * quad.pts[q][0] + v2 * quad.pts[q][1];

                // RWG basis function
                Vec3 fv = (rr - free_v) * coeff;

                // Phase at this point (same convention as rhs.cpp)
                double kr = k_hat.dot(rr);
                cdouble phase = std::exp(cdouble(-k_ext.imag() * kr,
                                                  k_ext.real() * kr));
                double jw = area * quad.wts[q];

                // Total external E = E_inc + E_ref
                cdouble Ex = (E0.x + Er_s * s_hat.x + Er_p * p_ref.x) * phase;
                cdouble Ey = (E0.y + Er_s * s_hat.y + Er_p * p_ref.y) * phase;
                cdouble Ez = (E0.z + Er_s * s_hat.z + Er_p * p_ref.z) * phase;

                // Total external H = H_inc + H_ref
                cdouble Hx = (H0.x + (Er_s * kxs.x + Er_p * kxp.x) * inv_eta) * phase;
                cdouble Hy = (H0.y + (Er_s * kxs.y + Er_p * kxp.y) * inv_eta) * phase;
                cdouble Hz = (H0.z + (Er_s * kxs.z + Er_p * kxp.z) * inv_eta) * phase;

                // J_GO = n̂ × H_ext
                cdouble Jx = n_hat.y * Hz - n_hat.z * Hy;
                cdouble Jy = n_hat.z * Hx - n_hat.x * Hz;
                cdouble Jz = n_hat.x * Hy - n_hat.y * Hx;

                // M_GO = -n̂ × E_ext
                cdouble Mx = -(n_hat.y * Ez - n_hat.z * Ey);
                cdouble My = -(n_hat.z * Ex - n_hat.x * Ez);
                cdouble Mz = -(n_hat.x * Ey - n_hat.y * Ex);

                // Project: x[n] += ∫ f_n · J_GO dS
                bJ += (fv.x * Jx + fv.y * Jy + fv.z * Jz) * jw;
                bM += (fv.x * Mx + fv.y * My + fv.z * Mz) * jw;
            }

            x_go[n]     += bJ;
            x_go[N + n] += bM;
        }
    }
}
