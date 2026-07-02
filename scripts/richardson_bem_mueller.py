#!/usr/bin/env python3
"""Richardson-extrapolate two BEM Mueller JSON files and optionally compare/plot.

The intended use is mesh extrapolation from a coarse and fine surface mesh with
the same orientation/theta sampling:

    M_ext = (r**p * M_fine - M_coarse) / (r**p - 1)

where r is the mesh spacing ratio h_coarse / h_fine. For the current prism
ref2/ref3 pair, r=2 and p=2 is the validated default.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def timing_sum(*items):
    out = {"assembly_s": 0.0, "solve_s": 0.0, "farfield_s": 0.0, "total_s": 0.0}
    for data in items:
        timing = data.get("timing", {})
        for key in out:
            out[key] += float(timing.get(key, 0.0))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse", required=True, help="Coarser BEM JSON")
    parser.add_argument("--fine", required=True, help="Finer BEM JSON")
    parser.add_argument("--out", required=True, help="Output extrapolated BEM JSON")
    parser.add_argument("--ratio", type=float, default=2.0,
                        help="Mesh spacing ratio h_coarse/h_fine")
    parser.add_argument("--order", type=float, default=2.0,
                        help="Assumed convergence order")
    parser.add_argument("--adda", help="Raw ADDA directory for comparison")
    parser.add_argument("--beta-order", type=int, default=8)
    parser.add_argument("--plot", help="Output comparison PNG; requires --adda")
    parser.add_argument("--title", default="BEM Richardson extrapolation")
    parser.add_argument("--log-big", action="store_true")
    args = parser.parse_args()

    coarse = load_json(args.coarse)
    fine = load_json(args.fine)
    theta_c = np.asarray(coarse["theta"], dtype=float)
    theta_f = np.asarray(fine["theta"], dtype=float)
    if len(theta_c) != len(theta_f) or np.max(np.abs(theta_c - theta_f)) > 1e-12:
        raise ValueError("coarse/fine theta grids differ")

    mu_c = np.asarray(coarse["mueller"], dtype=float)
    mu_f = np.asarray(fine["mueller"], dtype=float)
    if mu_c.shape != mu_f.shape:
        raise ValueError(f"coarse/fine Mueller shapes differ: {mu_c.shape} vs {mu_f.shape}")

    q = args.ratio ** args.order
    if abs(q - 1.0) < 1e-15:
        raise ValueError("ratio**order must differ from 1")
    w_fine = q / (q - 1.0)
    w_coarse = -1.0 / (q - 1.0)
    mu_ext = w_fine * mu_f + w_coarse * mu_c

    out = {
        "ka": fine.get("ka"),
        "ri": fine.get("ri"),
        "refinements": fine.get("refinements"),
        "orient": fine.get("orient"),
        "alpha_avg": fine.get("alpha_avg"),
        "ntheta": fine.get("ntheta"),
        "timing": timing_sum(coarse, fine),
        "theta": theta_f.tolist(),
        "mueller": mu_ext.tolist(),
        "richardson": {
            "ratio": args.ratio,
            "order": args.order,
            "coarse_weight": w_coarse,
            "fine_weight": w_fine,
            "coarse": str(args.coarse),
            "fine": str(args.fine),
        },
        "orientation_weight_sum": fine.get("orientation_weight_sum"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"weights: fine={w_fine:.12g} coarse={w_coarse:.12g}")

    if args.adda:
        subprocess.check_call([
            "python3", "scripts/compare_mueller.py",
            "--bem", str(out_path),
            "--adda", args.adda,
            "--beta-order", str(args.beta_order),
        ])

    if args.plot:
        if not args.adda:
            raise ValueError("--plot requires --adda")
        cmd = [
            "python3", "scripts/plot_bem_raw_adda.py",
            "--bem", str(out_path),
            "--adda", args.adda,
            "--beta-order", str(args.beta_order),
            "--title", args.title,
            "--out", args.plot,
        ]
        if args.log_big:
            cmd.append("--log-big")
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
