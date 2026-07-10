#!/usr/bin/env python3

import argparse
import importlib.util
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_orient_queue", ROOT / "run_orient_queue.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def options(exe, obj, orient, ntheta=181):
    return argparse.Namespace(
        exe=str(exe), ka="30.25", ri=["1.6", "0"], shape="obj", obj=str(obj),
        subdiv="0", prism_aspect="1", edge_refine="auto", ref="3", quad="7",
        ntheta=str(ntheta), scat_plane="yz", alpha_avg="256", solver="fmm",
        accurate=True, fast_obj=False, system="balanced", fmm_digits="7",
        gmres_tol="0.001", gmres_restart="200", krylov="gpu-gmres",
        max_leaf="128", no_prec=True,
    )


def legacy_part(args, ntheta=181):
    return {
        "ka": 30.25, "ri": [1.6, 0.0], "shape": "obj", "obj_file": args.obj,
        "ntheta": ntheta, "alpha_avg": 256, "refinements": 0, "prism_aspect": 1,
        "orient_total": 994, "method": {"quad_order": 7},
        "requested_system": "balanced", "fmm_digits": 7, "gmres_tol": 0.001,
        "gmres_restart": 200, "max_leaf": 128, "preconditioner_enabled": False,
        "gmres_nonconverged_systems": 0, "gmres_numerical_breakdowns": 0,
        "gmres_max_final_relres": 0.00099,
    }


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exe, obj, orient = root / "bem", root / "dust.obj", root / "grid.txt"
        exe.write_bytes(b"binary-v1")
        obj.write_text("mesh-v1\n")
        orient.write_text("0 0 1\n")
        args = options(exe, obj, orient)
        original = MODULE.build_resume_signature(args, str(orient))

        assert MODULE.build_resume_signature(options(exe, obj, orient, 1801), str(orient)) != original
        orient.write_text("0 10 1\n")
        assert MODULE.build_resume_signature(args, str(orient)) != original
        orient.write_text("0 0 1\n")
        obj.write_text("mesh-v2\n")
        assert MODULE.build_resume_signature(args, str(orient)) != original

        good = legacy_part(args)
        assert MODULE.legacy_part_matches(good, args, 994)
        assert not MODULE.legacy_part_matches(legacy_part(args, ntheta=1801), args, 994)
        bad_solver = legacy_part(args)
        bad_solver["gmres_nonconverged_systems"] = 1
        assert not MODULE.legacy_part_matches(bad_solver, args, 994)

    print("PASS: orientation resume signatures reject incompatible results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
