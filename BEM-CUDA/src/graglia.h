#ifndef BEM_GRAGLIA_H
#define BEM_GRAGLIA_H

#include "types.h"
#include "quadrature.h"
#include <cmath>
#include <cstring>

// Analytical potential integral: ∫_T 1/|r_obs - r'| dS'
// Graglia 1993 formula.
inline double potential_integral_triangle(const Vec3& r_obs,
                                          const Vec3& v0, const Vec3& v1, const Vec3& v2)
{
    Vec3 vertices[3] = {v0, v1, v2};

    // Triangle normal
    Vec3 e10 = v1 - v0;
    Vec3 e20 = v2 - v0;
    Vec3 n_tri = e10.cross(e20);
    double n_norm = n_tri.norm();
    if (n_norm < 1e-15) return 0.0;
    Vec3 n_hat = n_tri * (1.0 / n_norm);

    double result = 0.0;

    for (int i = 0; i < 3; i++) {
        Vec3 vi = vertices[i];
        Vec3 vj = vertices[(i + 1) % 3];

        Vec3 edge = vj - vi;
        double l_edge = edge.norm();
        if (l_edge < 1e-15) continue;
        Vec3 t_hat = edge * (1.0 / l_edge);

        // Inward normal to edge (in triangle plane)
        Vec3 m_hat = n_hat.cross(t_hat);

        Vec3 diff_i = r_obs - vi;
        double d = diff_i.dot(m_hat);
        double h = diff_i.dot(n_hat);

        Vec3 diff_j = vj - r_obs;
        Vec3 diff_ii = vi - r_obs;
        double s_plus = diff_j.dot(t_hat);
        double s_minus = diff_ii.dot(t_hat);
        double R_plus = diff_j.norm();
        double R_minus = diff_ii.norm();

        // ln term
        if (R_plus + s_plus > 1e-15 && R_minus + s_minus > 1e-15) {
            double log_arg = (R_plus + s_plus) / (R_minus + s_minus);
            if (log_arg > 0)
                result += d * log(log_arg);
        }

        // arctan term
        if (fabs(h) > 1e-15) {
            double R0_sq = d * d + h * h;
            double t1 = atan2(d * s_plus, R0_sq + fabs(h) * R_plus);
            double t2 = atan2(d * s_minus, R0_sq + fabs(h) * R_minus);
            result -= fabs(h) * (t1 - t2);
        }
    }

    return result;
}

// ∫_T |r_obs - r'| dS' by numerical quadrature
inline double integral_R_triangle(const Vec3& r_obs,
                                   const Vec3& v0, const Vec3& v1, const Vec3& v2,
                                   const TriQuad& quad)
{
    Vec3 e10 = v1 - v0;
    Vec3 e20 = v2 - v0;
    double area = 0.5 * e10.cross(e20).norm();

    double result = 0.0;
    for (int q = 0; q < quad.npts; q++) {
        double l0 = 1.0 - quad.pts[q][0] - quad.pts[q][1];
        Vec3 rr = v0 * l0 + v1 * quad.pts[q][0] + v2 * quad.pts[q][1];
        double R = (rr - r_obs).norm();
        result += quad.wts[q] * R;
    }
    return area * result;
}

// ∫_T r'/|r_obs - r'| dS' using:
// ∫ r'/R dS' = r_obs * P - ∇_r W
// where P = ∫ 1/R dS', W = ∫ R dS'
inline Vec3 vector_potential_integral_triangle(const Vec3& r_obs,
                                                const Vec3& v0, const Vec3& v1, const Vec3& v2,
                                                const TriQuad& quad)
{
    double P = potential_integral_triangle(r_obs, v0, v1, v2);

    // ∇_r W by finite differences
    double eps = 1e-6;
    Vec3 grad_W;
    for (int d = 0; d < 3; d++) {
        Vec3 rp = r_obs, rm = r_obs;
        if (d == 0) { rp.x += eps; rm.x -= eps; }
        else if (d == 1) { rp.y += eps; rm.y -= eps; }
        else { rp.z += eps; rm.z -= eps; }
        double Wp = integral_R_triangle(rp, v0, v1, v2, quad);
        double Wm = integral_R_triangle(rm, v0, v1, v2, quad);
        if (d == 0) grad_W.x = (Wp - Wm) / (2.0 * eps);
        else if (d == 1) grad_W.y = (Wp - Wm) / (2.0 * eps);
        else grad_W.z = (Wp - Wm) / (2.0 * eps);
    }

    return r_obs * P - grad_W;
}

#endif // BEM_GRAGLIA_H
