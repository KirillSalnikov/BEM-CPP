#!/usr/bin/env python3
"""Linearly combine BEM Mueller JSON files.

Useful for mesh-bias cancellation experiments: all inputs must have identical
theta grids. Weights are applied directly to Mueller matrices.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", nargs=2, metavar=("WEIGHT", "JSON"),
                        required=True, help="Input weight and BEM JSON path")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    total_weight = 0.0
    combined = None
    theta = None
    first = None
    timing = {"assembly_s": 0.0, "solve_s": 0.0, "farfield_s": 0.0, "total_s": 0.0}
    sources = []

    for weight_s, path_s in args.input:
        weight = float(weight_s)
        path = Path(path_s)
        with path.open("r") as f:
            data = json.load(f)
        mueller = np.asarray(data["mueller"], dtype=float)
        this_theta = np.asarray(data["theta"], dtype=float)
        if theta is None:
            theta = this_theta
            combined = np.zeros_like(mueller)
            first = data
        elif len(theta) != len(this_theta) or np.max(np.abs(theta - this_theta)) > 1e-12:
            raise ValueError(f"theta grid mismatch in {path}")
        combined += weight * mueller
        total_weight += weight
        for key in timing:
            timing[key] += float(data.get("timing", {}).get(key, 0.0))
        sources.append({"weight": weight, "path": str(path)})

    if abs(total_weight) < 1e-300:
        raise ValueError("sum of weights is zero")

    out = {
        "ka": first.get("ka"),
        "ri": first.get("ri"),
        "refinements": first.get("refinements"),
        "orient": first.get("orient"),
        "alpha_avg": first.get("alpha_avg"),
        "ntheta": first.get("ntheta"),
        "timing": timing,
        "theta": theta.tolist(),
        "mueller": combined.tolist(),
        "combined_sources": sources,
        "combined_weight_sum": total_weight,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
