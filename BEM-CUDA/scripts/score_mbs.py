#!/usr/bin/env python3
"""Score BEM/ADDA Mueller curves against an MBS reference table."""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


COMPONENTS = {
    "S11": (0, 0, 1, 2),
    "S12": (0, 1, 2, 3),
    "S22": (1, 1, 6, 7),
    "S33": (2, 2, 11, 12),
    "S34": (2, 3, 12, 13),
    "S44": (3, 3, 16, 17),
}


def bem_component(mueller, i, j):
    m = np.asarray(mueller)
    if m.ndim == 3 and m.shape[0:2] == (4, 4):
        return m[i, j]
    if m.ndim == 3 and m.shape[1:3] == (4, 4):
        return m[:, i, j]
    if m.ndim == 2 and m.shape[0] == 16:
        return m[4 * i + j]
    if m.ndim == 2 and m.shape[1] == 16:
        return m[:, 4 * i + j]
    raise ValueError(f"unknown BEM mueller shape: {m.shape}")


def load_bem(path):
    with open(path, "r") as f:
        data = json.load(f)
    theta = np.asarray(data["theta"], dtype=float)
    mueller = np.asarray(data["mueller"], dtype=float)
    table = np.zeros((len(theta), 17), dtype=float)
    table[:, 0] = theta
    for _, (i, j, table_col, _) in COMPONENTS.items():
        table[:, table_col] = bem_component(mueller, i, j)
    return table, data.get("timing", {})


def load_adda(path, beta_order):
    files = sorted(glob.glob(str(Path(path) / "*" / "mueller")))
    if Path(path).is_file():
        files = [str(path)]
    if not files:
        raise FileNotFoundError(f"no ADDA mueller files under {path}")

    beta_weights = None
    if beta_order:
        nodes, weights = np.polynomial.legendre.leggauss(beta_order)
        betas = np.degrees(np.arccos(nodes))
        beta_weights = list(zip(betas, weights))

    acc = None
    wsum = 0.0
    for file_name in files:
        data = np.loadtxt(file_name, skiprows=1)
        weight = 1.0
        if beta_weights is not None:
            match = re.search(r"_b([0-9p]+)_g", file_name)
            if not match:
                raise ValueError(f"cannot parse beta from {file_name}")
            beta = float(match.group(1).replace("p", "."))
            weight = min(beta_weights, key=lambda item: abs(item[0] - beta))[1]
        if acc is None:
            acc = np.zeros_like(data)
        acc += weight * data
        wsum += weight
    return acc / wsum, {"files": len(files)}


def restrict_theta(table, theta_max):
    if theta_max is None:
        return table
    return table[table[:, 0] <= theta_max]


def score_against_mbs(table, mbs, raw):
    theta = table[:, 0]
    data_norm = 1.0 if raw else table[0, COMPONENTS["S11"][2]]
    ref_s11 = np.interp(theta, mbs[:, 0], mbs[:, COMPONENTS["S11"][3]])
    ref_norm = 1.0 if raw else ref_s11[0]

    parts = {}
    total = 0.0
    for name, (_, _, data_col, mbs_col) in COMPONENTS.items():
        y = table[:, data_col] / data_norm
        r = np.interp(theta, mbs[:, 0], mbs[:, mbs_col]) / ref_norm
        err = np.linalg.norm(y - r) / max(np.linalg.norm(r), 1e-300)
        parts[name] = float(err)
        total += err
    return float(total), parts, float(data_norm / ref_norm)


def score_against_mbs_s11_weighted(table, mbs, raw):
    theta = table[:, 0]
    data_norm = 1.0 if raw else table[0, COMPONENTS["S11"][2]]
    ref_s11 = np.interp(theta, mbs[:, 0], mbs[:, COMPONENTS["S11"][3]])
    ref_norm = 1.0 if raw else ref_s11[0]
    ref_s11_norm = ref_s11 / ref_norm

    parts = {}
    total = 0.0
    for name, (_, _, data_col, mbs_col) in COMPONENTS.items():
        y = table[:, data_col] / data_norm
        r = np.interp(theta, mbs[:, 0], mbs[:, mbs_col]) / ref_norm
        err = y - r
        denom = max(np.linalg.norm(ref_s11_norm), 1e-300)
        val = np.linalg.norm(err) / denom
        parts[name] = float(val)
        total += val
    return float(total), parts


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--bem", help="BEM JSON output")
    src.add_argument("--adda", help="ADDA mueller file or directory with */mueller files")
    parser.add_argument("--mbs", required=True, help="MBS/converted reference table")
    parser.add_argument("--beta-order", type=int, default=0)
    parser.add_argument("--theta-max", type=float, default=180.0)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    if args.bem:
        table, meta = load_bem(args.bem)
        label = args.bem
    else:
        table, meta = load_adda(args.adda, args.beta_order)
        label = args.adda
    table = restrict_theta(table, args.theta_max)
    mbs = np.loadtxt(args.mbs, skiprows=1)
    total, parts, scale = score_against_mbs(table, mbs, args.raw)
    weighted_total, weighted_parts = score_against_mbs_s11_weighted(table, mbs, args.raw)

    print(f"Source: {label}")
    print(f"Theta points: {len(table)}")
    if meta:
        print("Meta:", " ".join(f"{k}={v}" for k, v in meta.items()))
    if not args.raw:
        print(f"Scale source/MBS S11(0): {scale:.8g}")
    for name in COMPONENTS:
        print(f"{name}: {parts[name]:.6g}")
    print(f"score6: {total:.6g}")
    for name in COMPONENTS:
        print(f"{name}_s11w: {weighted_parts[name]:.6g}")
    print(f"score6_s11w: {weighted_total:.6g}")


if __name__ == "__main__":
    main()
