#!/usr/bin/env python3
"""Search Mueller basis/convention transforms between BEM and ADDA.

This is a diagnostic, not a production correction.  It applies signed
permutations of the polarization Stokes axes (Q,U,V) independently on output
and input sides:

    M_ref ~= T_out @ M_bem @ T_in

This covers common basis flips, handedness flips, and Q/U/V ordering mistakes
without allowing arbitrary element-by-element sign fitting.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import compare_mueller


ALL_COMPONENTS = {
    f"S{i + 1}{j + 1}": (i, j)
    for i in range(4)
    for j in range(4)
}

DEFAULT_COMPONENT_NAMES = ("S11", "S12", "S22", "S33", "S34", "S44")
COMPONENTS = {name: ALL_COMPONENTS[name] for name in DEFAULT_COMPONENT_NAMES}


def load_bem_matrix(path):
    theta, mueller, timing = compare_mueller.load_bem(path)
    data = np.asarray(mueller, dtype=float)
    n = len(theta)
    if data.shape == (4, 4, n):
        pass
    elif data.shape == (n, 4, 4):
        data = np.moveaxis(data, 0, -1)
    elif data.shape == (16, n):
        data = data.reshape(4, 4, n)
    elif data.shape == (n, 16):
        data = np.moveaxis(data.reshape(n, 4, 4), 0, -1)
    else:
        raise ValueError(f"unknown BEM Mueller shape: {data.shape}")
    return np.asarray(theta, dtype=float), data, timing


def load_ref_matrix(path, theta, beta_order):
    ref, count = compare_mueller.load_adda_average(path, beta_order)
    out = np.zeros((4, 4, len(theta)), dtype=float)
    for name, (i, j, col, _) in compare_mueller.ALL_COMPONENTS.items():
        out[i, j, :] = np.interp(theta, ref[:, 0], ref[:, col])
    return out, count


def load_mbs_ref_matrix(path, theta):
    ref = compare_mueller.load_mbs_table(path)
    out = np.zeros((4, 4, len(theta)), dtype=float)
    for name, (i, j, _, col) in compare_mueller.ALL_COMPONENTS.items():
        out[i, j, :] = np.interp(theta, ref[:, 0], ref[:, col])
    return out, 1


def signed_permutation_transforms(include_swaps):
    transforms = []
    axes = "QUV"
    perms = itertools.permutations(range(3)) if include_swaps else [(0, 1, 2)]
    for perm in perms:
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mat = np.zeros((4, 4), dtype=float)
            mat[0, 0] = 1.0
            for dst, src in enumerate(perm):
                mat[1 + dst, 1 + src] = signs[dst]
            name = ",".join(
                f"{axes[dst]}={int(signs[dst]):+d}{axes[src]}"
                for dst, src in enumerate(perm)
            )
            transforms.append((name, mat))
    return transforms


def normalize(mu, raw):
    if raw:
        return mu.copy()
    denom = mu[0, 0, 0]
    if denom == 0.0:
        raise ValueError("S11(0) is zero")
    return mu / denom


def rel_l2(y, r):
    return float(np.linalg.norm(y - r) / max(np.linalg.norm(r), 1e-300))


def score(mu, ref, elements):
    total = 0.0
    parts = {}
    for name in elements:
        i, j = ALL_COMPONENTS[name]
        err = rel_l2(mu[i, j, :], ref[i, j, :])
        parts[name] = err
        total += err
    parts["score6"] = total
    return parts


def transform_mueller(mu, tout, tin, transpose):
    if transpose:
        mu = np.swapaxes(mu, 0, 1)
    return np.einsum("ab,bcT,cd->adT", tout, mu, tin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bem", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--adda")
    group.add_argument("--mbs")
    parser.add_argument("--beta-order", type=int, default=0)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--allow-axis-swaps", action="store_true")
    parser.add_argument("--allow-transpose", action="store_true")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--elements", default=",".join(DEFAULT_COMPONENT_NAMES))
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.elements.strip().lower() == "all":
        elements = list(ALL_COMPONENTS)
    else:
        elements = [x.strip().upper() for x in args.elements.split(",") if x.strip()]
    unknown = [x for x in elements if x not in ALL_COMPONENTS]
    if unknown:
        raise SystemExit("unknown elements: " + ", ".join(unknown))

    theta, bem, timing = load_bem_matrix(args.bem)
    if args.adda:
        ref, ref_count = load_ref_matrix(args.adda, theta, args.beta_order)
        ref_label = args.adda
    else:
        ref, ref_count = load_mbs_ref_matrix(args.mbs, theta)
        ref_label = args.mbs
    transforms = signed_permutation_transforms(args.allow_axis_swaps)
    transpose_modes = [False, True] if args.allow_transpose else [False]

    rows = []
    for transpose in transpose_modes:
        for out_name, tout in transforms:
            for in_name, tin in transforms:
                trial = transform_mueller(bem, tout, tin, transpose)
                trial_n = normalize(trial, args.raw)
                ref_n = normalize(ref, args.raw)
                parts = score(trial_n, ref_n, elements)
                rows.append({
                    "score6": parts["score6"],
                    "out": out_name,
                    "in": in_name,
                    "transpose": transpose,
                    "parts": parts,
                    "scale_bem_ref": float(trial[0, 0, 0] / ref[0, 0, 0]),
                })
    rows.sort(key=lambda item: item["score6"])

    print(f"BEM: {args.bem}")
    print(f"Reference: {ref_label} ({ref_count} file{'s' if ref_count != 1 else ''})")
    if timing:
        print("Timing:", " ".join(f"{k}={v}" for k, v in timing.items()))
    print(f"Transforms tried: {len(rows)}")
    for idx, row in enumerate(rows[:args.top], start=1):
        parts = " ".join(f"{name}={row['parts'][name]:.6g}" for name in elements)
        print(
            f"{idx:2d} score6={row['score6']:.6g} scale={row['scale_bem_ref']:.8g} "
            f"transpose={int(row['transpose'])} out=[{row['out']}] in=[{row['in']}] {parts}"
        )

    if args.out:
        payload = {
            "bem": args.bem,
            "adda": args.adda,
            "mbs": args.mbs,
            "beta_order": args.beta_order,
            "raw": args.raw,
            "allow_axis_swaps": args.allow_axis_swaps,
            "allow_transpose": args.allow_transpose,
            "rows": rows,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
