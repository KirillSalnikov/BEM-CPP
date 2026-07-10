#include "precond_policy.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

static void require_policy(const PrecondPolicyInput& in, bool enabled,
                           bool schwarz, const char* reason)
{
    PrecondPolicy p = choose_precond_policy(in);
    if (p.enabled != enabled || p.schwarz != schwarz ||
        std::strcmp(p.reason, reason) != 0) {
        std::cerr << "unexpected policy: enabled=" << p.enabled
                  << " schwarz=" << p.schwarz
                  << " reason=" << p.reason << "\n";
        std::exit(1);
    }
}

int main()
{
    PrecondPolicyInput in;

    require_policy(in, false, false, "dense_solver");

    in.use_fmm = true;
    in.basis_count = 300;
    require_policy(in, false, false, "small_system");

    in.basis_count = 792;
    in.sphere = false;
    in.ka = 5.0;
    require_policy(in, false, false, "small_nonsphere");

    in.force = true;
    require_policy(in, true, true, "forced");

    in.force = false;
    in.basis_count = 2400;
    require_policy(in, true, true, "auto");

    in.pfft_backend = true;
    require_policy(in, false, false, "pfft_backend");

    in.force = true;
    require_policy(in, true, true, "forced");

    in.force = false;
    in.pfft_backend = false;

    in.n_form = true;
    require_policy(in, false, false, "n_form");

    in.force = true;
    require_policy(in, true, true, "forced");

    in.force = false;
    in.n_form = false;

    in.obj_mesh = true;
    in.ka = 2.0;
    require_policy(in, true, true, "auto");

    in.ka = 5.0;
    require_policy(in, false, false, "obj_ka_ge_4_unpreconditioned_measured");

    in.strict_accuracy = true;
    require_policy(in, false, false, "obj_strict_unpreconditioned_measured");

    in.mesh_requires_remesh = true;
    in.strict_accuracy = false;
    in.gmres_tol = 1e-3;
    require_policy(in, false, false, "obj_quality_loose_unpreconditioned_measured");

    in.strict_accuracy = true;
    in.gmres_tol = 1e-5;
    require_policy(in, false, false, "obj_quality_strict_unpreconditioned_measured");

    in.strict_accuracy = false;
    in.gmres_tol = 1e-4;
    require_policy(in, false, false, "obj_quality_remesh_unpreconditioned");

    in.mesh_requires_remesh = false;
    in.force = true;
    require_policy(in, true, true, "forced");

    in.force = false;
    in.obj_mesh = false;

    in.hex_prism = true;
    require_policy(in, false, false, "hex_unpreconditioned_faster");

    in.strict_accuracy = true;
    require_policy(in, false, false, "hex_strict_unpreconditioned_measured");

    in.strict_accuracy = false;
    in.hex_prism = false;
    in.sphere = true;
    in.ka = 20.0;
    require_policy(in, false, false, "sphere_unpreconditioned_measured");

    in.force = true;
    require_policy(in, true, false, "forced");
    in.force = false;

    in.user_disabled = true;
    require_policy(in, false, false, "user_disabled");

    std::cout << "preconditioner policy: ok\n";
    return 0;
}
