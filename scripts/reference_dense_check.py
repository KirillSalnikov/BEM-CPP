#!/usr/bin/env python3
"""Run a small dense-vs-FMM reference check for the same BEM problem."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bemcuda import BemJob, MeshQuality  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from mueller_audit import load_bem_mueller, relative_l2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "reference_dense_check")
    parser.add_argument("--ka", type=float, default=2.0)
    parser.add_argument("--shape", default="sphere")
    parser.add_argument("--ref", type=int, default=1)
    parser.add_argument("--ri", nargs=2, type=float, default=(1.3116, 0.0))
    parser.add_argument("--ntheta", type=int, default=37)
    parser.add_argument("--max-l2", type=float, default=5e-3)
    parser.add_argument("--relative-floor", type=float, default=1e-8,
                        help="Only use elementwise relative L2 when dense norm exceeds this S11 fraction")
    parser.add_argument("--system", default=None,
                        help="Pass an explicit --system; default leaves executable auto system enabled")
    parser.add_argument("--require-complex-operator", action="store_true",
                        help="Require row_h_scale_complex metadata in dense and FMM JSON outputs")
    parser.add_argument("--binary", type=Path, default=ROOT / "bin" / "bem_cuda_fmm")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        ka=args.ka,
        shape=args.shape,
        ref=args.ref,
        ri=(args.ri[0], args.ri[1]),
        ntheta=args.ntheta,
        single=True,
        quad=7,
        system=args.system,
        mesh_quality=MeshQuality(strict=False),
        binary=args.binary,
    )
    dense = BemJob(**common, solver="dense", out=args.out_dir / "dense.json")
    fmm = BemJob(**common, solver="fmm", out=args.out_dir / "fmm.json",
                 fmm_digits=6, gmres_tol=1e-6, gmres_restart=250)

    commands = {"dense": dense.command(), "fmm": fmm.command()}
    (args.out_dir / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")

    for job in (dense, fmm):
        subprocess.run(job.command(), cwd=str(ROOT), check=True)

    theta_d, md, _ = load_bem_mueller(dense.out)
    theta_f, mf, _ = load_bem_mueller(fmm.out)
    if not np.allclose(theta_d, theta_f):
        raise SystemExit("dense and FMM theta grids differ")

    relative_errors = {}
    absolute_errors = {}
    active_relative_errors = {}
    s11_norm = float(np.linalg.norm(md[0, 0]))
    if s11_norm <= 0:
        raise SystemExit("dense S11 norm is zero")
    for i in range(4):
        for j in range(4):
            name = f"S{i + 1}{j + 1}"
            diff_norm = float(np.linalg.norm(np.asarray(mf[i, j]) - np.asarray(md[i, j])))
            dense_norm = float(np.linalg.norm(md[i, j]))
            absolute_errors[name] = diff_norm / s11_norm
            if dense_norm > args.relative_floor * s11_norm:
                relative_errors[name] = diff_norm / dense_norm
                active_relative_errors[name] = relative_errors[name]
            else:
                relative_errors[name] = None
    max_abs = max(absolute_errors.values())
    max_active_rel = max(active_relative_errors.values()) if active_relative_errors else 0.0
    report = {
        "dense": str(dense.out),
        "fmm": str(fmm.out),
        "relative_l2_fmm_vs_dense": relative_errors,
        "absolute_l2_over_s11": absolute_errors,
        "relative_floor_over_s11": args.relative_floor,
        "max_active_relative_l2": max_active_rel,
        "max_absolute_l2_over_s11": max_abs,
        "max_l2": max(max_abs, max_active_rel),
    }
    complex_operator = {}
    for label, path in (("dense", dense.out), ("fmm", fmm.out)):
        method = json.loads(path.read_text()).get("method", {})
        row = method.get("row_h_scale_complex")
        ok = (
            isinstance(row, list) and
            len(row) == 2 and
            all(isinstance(x, (int, float)) for x in row)
        )
        complex_operator[label] = {
            "row_h_scale_complex": row,
            "present": ok,
        }
    report["complex_operator"] = complex_operator
    report["complex_operator_verified"] = all(
        item["present"] for item in complex_operator.values()
    )
    report["pass"] = (
        max_abs <= args.max_l2 and
        max_active_rel <= args.max_l2 and
        (report["complex_operator_verified"] or not args.require_complex_operator)
    )
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
