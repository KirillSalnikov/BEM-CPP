#ifndef BEM_PRECOND_POLICY_H
#define BEM_PRECOND_POLICY_H

struct PrecondPolicyInput {
    bool use_fmm = false;
    bool user_disabled = false;
    bool force = false;
    bool pfft_backend = false;
    bool sphere = false;
    bool hex_prism = false;
    bool obj_mesh = false;
    bool mesh_requires_remesh = false;
    bool n_form = false;
    bool strict_accuracy = false;
    int basis_count = 0;
    double ka = 0.0;
    double gmres_tol = 1e-4;
};

struct PrecondPolicy {
    bool enabled = false;
    bool schwarz = false;
    const char* reason = "disabled";
};

inline PrecondPolicy choose_precond_policy(const PrecondPolicyInput& in)
{
    PrecondPolicy out;
    if (!in.use_fmm) {
        out.reason = "dense_solver";
        return out;
    }
    if (in.user_disabled) {
        out.reason = "user_disabled";
        return out;
    }
    if (!in.force) {
        if (in.pfft_backend) {
            out.reason = "pfft_backend";
            return out;
        }
        if (in.basis_count < 512) {
            out.reason = "small_system";
            return out;
        }
        if (!in.sphere && in.basis_count < 1500) {
            out.reason = "small_nonsphere";
            return out;
        }
        if (in.n_form) {
            out.reason = "n_form";
            return out;
        }
        if (in.sphere) {
            out.reason = "sphere_unpreconditioned_measured";
            return out;
        }
        if (in.hex_prism) {
            out.reason = in.strict_accuracy ?
                "hex_strict_unpreconditioned_measured" :
                "hex_unpreconditioned_faster";
            return out;
        }
        if (in.obj_mesh && in.mesh_requires_remesh) {
            if (in.gmres_tol >= 1e-3) {
                out.reason = "obj_quality_loose_unpreconditioned_measured";
                return out;
            }
            if (in.strict_accuracy) {
                out.reason = "obj_quality_strict_unpreconditioned_measured";
                return out;
            }
            out.reason = "obj_quality_remesh_unpreconditioned";
            return out;
        }
        if (in.obj_mesh && in.ka >= 4.0) {
            if (in.strict_accuracy) {
                out.enabled = true;
                out.schwarz = false;
                out.reason = "obj_strict_block_jacobi";
                return out;
            }
            out.reason = "obj_ka_ge_4_unpreconditioned_measured";
            return out;
        }
    }
    out.enabled = true;
    out.schwarz = (!in.sphere || in.ka <= 10.0);
    out.reason = in.force ? "forced" : "auto";
    return out;
}

#endif
