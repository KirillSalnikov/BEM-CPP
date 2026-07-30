#include "solver_policy.h"

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>

static void require_policy(const SolverAccuracyInput& in,
                           int digits, int leaf, int restart, double tol,
                           const char* profile, bool guarded,
                           int max_cycles = 300)
{
    SolverAccuracyPolicy p = choose_solver_accuracy_policy(in);
    if (p.fmm_digits != digits || p.max_leaf != leaf ||
        p.gmres_restart != restart || std::abs(p.gmres_tol - tol) > 1e-14 ||
        std::strcmp(p.profile, profile) != 0 ||
        p.hex_guarded_accuracy != guarded ||
        p.gmres_max_cycles != max_cycles) {
        std::cerr << "unexpected solver policy: digits=" << p.fmm_digits
                  << " leaf=" << p.max_leaf
                  << " restart=" << p.gmres_restart
                  << " cycles=" << p.gmres_max_cycles
                  << " tol=" << p.gmres_tol
                  << " profile=" << p.profile
                  << " guarded=" << p.hex_guarded_accuracy << "\n";
        std::exit(1);
    }
}

int main()
{
    if (!symmetry_reconstruction_meets_tolerance(1e-5, 1e-5) ||
        symmetry_reconstruction_meets_tolerance(0.302, 1e-5) ||
        symmetry_reconstruction_meets_tolerance(-1.0, 1e-5)) {
        std::cerr << "unexpected symmetry reconstruction policy\n";
        return 1;
    }

    SolverAccuracyInput in;
    in.use_fmm = true;
    in.fmm_backend = true;
    in.hex_prism = true;
    in.ka = 5.0;
    in.refinements = 2;
    require_policy(in, 5, 128, 300, 1e-3, "hex_default_accurate", false, 80);

    in.adda_compare = true;
    require_policy(in, 5, 128, 300, 1e-3, "hex_adda_compare", false, 80);

    in.ka = 20.0;
    in.refinements = 4;
    require_policy(in, 5, 128, 300, 1e-3, "hex_guarded", true, 80);

    in.accurate = true;
    require_policy(in, 6, 128, 400, 5e-4, "hex_accurate", true, 60);

    in = SolverAccuracyInput();
    in.use_fmm = true;
    in.fmm_backend = true;
    in.obj = true;
    require_policy(in, 7, 128, 1000, 1e-5, "obj_strict", false, 80);

    in.gmres_restart = 500;
    in.gmres_restart_set = true;
    require_policy(in, 7, 128, 500, 1e-5, "obj_strict", false, 80);

    in.gmres_tol = 1e-5;
    in.gmres_tol_set = true;
    in.gmres_restart = 150;
    in.gmres_restart_set = false;
    require_policy(in, 7, 128, 1000, 1e-5, "obj_strict", false, 80);

    in.gmres_restart = 500;
    in.gmres_restart_set = true;
    require_policy(in, 7, 128, 500, 1e-5, "obj_strict", false, 80);

    in = SolverAccuracyInput();
    in.use_fmm = true;
    in.fmm_backend = true;
    in.obj = true;
    in.mesh_requires_remesh = true;
    in.mesh_recommended_min_quad_order = 13;
    in.mesh_min_angle_deg = 2.0;
    in.mesh_max_aspect_ratio = 28.0;
    require_policy(in, 8, 128, 1400, 1e-5, "obj_mesh_guard", false, 80);

    in.gmres_tol = 1e-4;
    in.gmres_tol_set = true;
    in.gmres_restart = 900;
    in.gmres_restart_set = true;
    require_policy(in, 8, 128, 900, 1e-4, "obj_mesh_guard", false, 80);

    in.gmres_tol = 1e-4;
    in.gmres_tol_set = false;
    in.gmres_restart = 150;
    in.gmres_restart_set = false;
    in.obj_fast = true;
    require_policy(in, 3, 128, 150, 1e-4, "obj_fast", false);

    in.accurate = true;
    require_policy(in, 3, 128, 150, 1e-4, "obj_fast", false);

    in = SolverAccuracyInput();
    in.use_fmm = true;
    in.fmm_backend = true;
    in.sphere = true;
    in.ka = 25.0;
    require_policy(in, 5, 128, 300, 1e-4, "sphere_large", false);

    in.fmm_digits = 7;
    in.fmm_digits_set = true;
    require_policy(in, 7, 128, 300, 1e-4, "sphere_large", false);

    std::cout << "solver accuracy policy: ok\n";
    return 0;
}
