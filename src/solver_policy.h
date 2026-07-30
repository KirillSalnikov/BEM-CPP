#ifndef BEM_SOLVER_POLICY_H
#define BEM_SOLVER_POLICY_H

struct SolverAccuracyInput {
    bool use_fmm = false;
    bool fmm_backend = false;
    bool sphere = false;
    bool hex_prism = false;
    bool obj = false;
    bool obj_fast = false;
    bool accurate = false;
    bool adda_compare = false;
    bool hex_unsafe_fast = false;
    bool fmm_digits_set = false;
    bool max_leaf_set = false;
    bool gmres_tol_set = false;
    bool gmres_restart_set = false;
    int refinements = 0;
    bool mesh_requires_remesh = false;
    int mesh_recommended_min_quad_order = 4;
    double mesh_min_angle_deg = 60.0;
    double mesh_max_aspect_ratio = 1.0;
    double ka = 0.0;
    int fmm_digits = 3;
    int max_leaf = 128;
    int gmres_restart = 150;
    double gmres_tol = 1e-4;
};

struct SolverAccuracyPolicy {
    int fmm_digits = 3;
    int max_leaf = 128;
    int gmres_restart = 150;
    int gmres_max_cycles = 300;
    double gmres_tol = 1e-4;
    int gmres_stagnation_cycles = 0;
    double gmres_stagnation_rel = 0.01;
    int gmres_inner_stagnation_window = 0;
    double gmres_inner_stagnation_rel = 0.05;
    int gmres_inner_stagnation_min_iter = 300;
    const char* profile = "user";
    bool hex_guarded_accuracy = false;
};

inline bool hex_needs_guarded_accuracy(double ka, int refinements)
{
    return ka >= 18.0 || refinements >= 5;
}

inline bool obj_needs_mesh_guard(const SolverAccuracyInput& in)
{
    return in.mesh_requires_remesh ||
           in.mesh_recommended_min_quad_order >= 13 ||
           in.mesh_min_angle_deg < 10.0 ||
           in.mesh_max_aspect_ratio > 20.0;
}

inline int bem_policy_max_int(int a, int b)
{
    return (a > b) ? a : b;
}

inline bool symmetry_reconstruction_meets_tolerance(
    double operator_residual,
    double tolerance)
{
    return operator_residual >= 0.0 &&
           operator_residual <= tolerance;
}

inline SolverAccuracyPolicy choose_solver_accuracy_policy(const SolverAccuracyInput& in)
{
    SolverAccuracyPolicy out;
    out.fmm_digits = in.fmm_digits;
    out.max_leaf = in.max_leaf;
    out.gmres_restart = in.gmres_restart;
    out.gmres_tol = in.gmres_tol;

    if (!in.use_fmm || !in.fmm_backend) {
        out.profile = "non_fmm";
        return out;
    }

    out.hex_guarded_accuracy =
        in.hex_prism && in.adda_compare &&
        hex_needs_guarded_accuracy(in.ka, in.refinements) &&
        !in.hex_unsafe_fast;

    if (in.hex_prism) {
        if (in.accurate) {
            if (!in.fmm_digits_set) out.fmm_digits = 6;
            if (!in.max_leaf_set) out.max_leaf = 128;
            if (!in.gmres_tol_set) out.gmres_tol = 5e-4;
            if (!in.gmres_restart_set) out.gmres_restart = 400;
            out.gmres_max_cycles = 60;
            out.profile = "hex_accurate";
        } else if (out.hex_guarded_accuracy) {
            if (!in.fmm_digits_set) out.fmm_digits = 5;
            if (!in.max_leaf_set) out.max_leaf = 128;
            if (!in.gmres_tol_set) out.gmres_tol = 1e-3;
            if (!in.gmres_restart_set) out.gmres_restart = 300;
            out.gmres_max_cycles = 80;
            out.profile = "hex_guarded";
        } else if (in.adda_compare) {
            if (!in.fmm_digits_set) out.fmm_digits = 5;
            if (!in.max_leaf_set) out.max_leaf = 128;
            if (!in.gmres_tol_set) out.gmres_tol = 1e-3;
            if (!in.gmres_restart_set) out.gmres_restart = 300;
            out.gmres_max_cycles = 80;
            out.profile = "hex_adda_compare";
        } else {
            if (!in.fmm_digits_set) out.fmm_digits = 5;
            if (!in.max_leaf_set) out.max_leaf = 128;
            if (!in.gmres_tol_set) out.gmres_tol = 1e-3;
            if (!in.gmres_restart_set) out.gmres_restart = 300;
            out.gmres_max_cycles = 80;
            out.profile = "hex_default_accurate";
        }
    } else if (in.obj && !in.obj_fast) {
        if (!in.fmm_digits_set) out.fmm_digits = 7;
        if (!in.max_leaf_set) out.max_leaf = 128;
        if (!in.gmres_tol_set) out.gmres_tol = 1e-5;
        if (!in.gmres_restart_set) out.gmres_restart = 1000;
        out.gmres_max_cycles = 80;
        out.profile = "obj_accurate";
        if (obj_needs_mesh_guard(in)) {
            if (!in.fmm_digits_set) out.fmm_digits = 8;
            if (!in.gmres_tol_set) out.gmres_tol = 1e-5;
            if (!in.gmres_restart_set) out.gmres_restart = 1400;
            out.gmres_max_cycles = 80;
            out.profile = "obj_mesh_guard";
        }
        if (out.gmres_tol <= 5e-4 && !obj_needs_mesh_guard(in)) {
            if (!in.gmres_restart_set) out.gmres_restart = 1000;
            out.profile = "obj_strict";
        }
    } else if (in.obj) {
        out.profile = "obj_fast";
    } else if (in.sphere && in.ka >= 20.0) {
        if (!in.fmm_digits_set) out.fmm_digits = 5;
        if (!in.gmres_restart_set) out.gmres_restart = 300;
        if (!in.gmres_tol_set) out.gmres_tol = 1e-4;
        out.profile = "sphere_large";
    } else {
        out.profile = "default";
    }

    if (in.hex_prism && in.refinements >= 4 && !in.gmres_restart_set)
        out.gmres_restart = bem_policy_max_int(out.gmres_restart, 300);
    if (in.hex_prism && out.gmres_tol <= 1e-3 && !in.gmres_restart_set)
        out.gmres_restart = bem_policy_max_int(out.gmres_restart, 300);

    return out;
}

#endif // BEM_SOLVER_POLICY_H
