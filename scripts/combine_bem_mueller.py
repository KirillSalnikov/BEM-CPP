#!/usr/bin/env python3
"""Linearly combine BEM Mueller JSON files.

For chunked orientation runs, pass ``--input orient chunk.json``. Each chunk
stores a weighted partial Mueller sum using the global orientation weights, so
the physically correct global average is the direct sum of chunks divided by
the included orientation_weight_sum. Numeric weights are still supported for
extrapolation experiments.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def project_random_orientation_mueller(mueller):
    out = np.array(mueller, dtype=float, copy=True)
    if out.ndim != 3 or out.shape[0] != 4 or out.shape[1] != 4:
        raise ValueError(f"project_random_orientation_mueller expects (4,4,ntheta), got {out.shape}")
    s12 = 0.5 * (out[1, 0, :] - out[0, 1, :])
    out[0, 1, :] = s12
    out[1, 0, :] = s12

    s34 = -0.5 * (out[2, 3, :] + out[3, 2, :])
    out[2, 3, :] = s34
    out[3, 2, :] = -s34

    out[1, 1, :] = -out[1, 1, :]
    out[3, 3, :] = -out[3, 3, :]

    for i, j in ((0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (2, 1), (3, 0), (3, 1)):
        out[i, j, :] = 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", nargs=2, metavar=("WEIGHT", "JSON"),
                        required=True,
                        help="Input weight and BEM JSON path; use WEIGHT=orient for chunked orientation averages")
    parser.add_argument("--out", required=True)
    parser.add_argument("--project-random", action="store_true",
                        help="Apply random-orientation Mueller projection after all inputs are combined")
    parser.add_argument("--timing-mode", choices=("sum", "max"), default="sum",
                        help="Combine timing by sum for serial chunks or max for chunks run in parallel")
    args = parser.parse_args()

    total_weight = 0.0
    combined = None
    theta = None
    first = None
    timing_values = {"assembly_s": [], "solve_s": [], "farfield_s": [], "total_s": []}
    summed_counters = {
        "orient_count": 0,
        "gmres_matvecs": 0,
        "gmres_converged_systems": 0,
        "gmres_nonconverged_systems": 0,
        "gmres_stagnation_stops": 0,
        "gmres_numerical_breakdowns": 0,
        "gmres_restored_best_iterates": 0,
        "gmres_max_cycle_exhaustions": 0,
    }
    max_counters = {
        "gmres_max_final_relres": 0.0,
    }
    sources = []
    orientation_weight_sum = 0.0
    normalize_by_total_weight = False
    projection_states = []

    for weight_s, path_s in args.input:
        path = Path(path_s)
        with path.open("r") as f:
            data = json.load(f)
        orient_weight = float(data.get("orientation_weight_sum", 0.0))
        if weight_s == "orient":
            if orient_weight <= 0.0:
                raise ValueError(f"{path} lacks positive orientation_weight_sum")
            # BEM writes orientation chunks as weighted partial sums using the
            # global quadrature weights, not as chunk-local averages. Combine
            # those partial sums directly, then normalize by the included
            # orientation weight sum below.
            weight = 1.0
            normalize_by_total_weight = True
        else:
            weight = float(weight_s)
        mueller = np.asarray(data["mueller"], dtype=float)
        this_theta = np.asarray(data["theta"], dtype=float)
        if theta is None:
            theta = this_theta
            combined = np.zeros_like(mueller)
            first = data
        elif len(theta) != len(this_theta) or np.max(np.abs(theta - this_theta)) > 1e-12:
            raise ValueError(f"theta grid mismatch in {path}")
        combined += weight * mueller
        total_weight += orient_weight if weight_s == "orient" else weight
        for key in timing_values:
            timing_values[key].append(float(data.get("timing", {}).get(key, 0.0)))
        for key in summed_counters:
            value = data.get(key)
            if value is not None:
                summed_counters[key] += int(value)
        for key in max_counters:
            value = data.get(key)
            if value is not None:
                max_counters[key] = max(max_counters[key], float(value))
        orientation_weight_sum += orient_weight if weight_s == "orient" else weight * orient_weight
        projection_state = data.get("random_orientation_projection", "unknown")
        projection_states.append(projection_state)
        sources.append({
            "weight": weight,
            "weight_mode": weight_s,
            "path": str(path),
            "orientation_weight_sum": orient_weight,
            "random_orientation_projection": projection_state,
        })

    if abs(total_weight) < 1e-300:
        raise ValueError("sum of weights is zero")
    if normalize_by_total_weight:
        combined /= total_weight
    output_projection = (
        projection_states[0]
        if projection_states and all(x == projection_states[0] for x in projection_states)
        else "mixed"
    )
    if args.project_random:
        combined = project_random_orientation_mueller(combined)
        output_projection = "applied_after_combine"

    if args.timing_mode == "max":
        timing = {key: max(values) if values else 0.0 for key, values in timing_values.items()}
    else:
        timing = {key: sum(values) for key, values in timing_values.items()}

    passthrough_keys = (
        "ka",
        "ri",
        "refinements",
        "shape",
        "obj_file",
        "prism_aspect",
        "edge_refine",
        "orient",
        "alpha_avg",
        "orient_total",
        "fmm_digits",
        "max_leaf",
        "gmres_restart",
        "gmres_tol",
        "gmres_max_cycles",
        "solver_backend",
        "solver_profile",
        "krylov_solver",
        "requested_system",
        "system",
        "device_gmres",
        "preconditioner_enabled",
        "method",
        "mesh",
        "ntheta",
        "mgpu",
    )
    out = {key: first.get(key) for key in passthrough_keys if key in first}
    out.update({
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
        "combined_timing_mode": args.timing_mode,
        "combined_input_weight_sum": total_weight,
        "orientation_weight_sum": orientation_weight_sum,
        "random_orientation_projection": output_projection,
    })
    out.update({key: value for key, value in summed_counters.items() if value})
    out.update(max_counters)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
