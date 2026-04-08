#ifndef GO_FIELD_H
#define GO_FIELD_H

#include "types.h"
#include "mesh.h"
#include "rwg.h"

// Compute GO (Geometric Optics / Physical Optics) initial guess for PMCHWT.
// On illuminated triangles: Fresnel reflection → J = n×H_ext, M = -n×E_ext
// On shadowed triangles: J = M = 0
// Projects GO surface currents onto RWG basis via quadrature.
//
// x_go: output array of size 2*N, [J_0..J_{N-1}, M_0..M_{N-1}]
void compute_go_initial_guess(
    const Mesh& mesh, const RWG& rwg,
    cdouble k_ext, cdouble m_ri,
    double eta_ext,
    const Vec3& E0, const Vec3& k_hat,
    int quad_order,
    cdouble* x_go);

#endif // GO_FIELD_H
