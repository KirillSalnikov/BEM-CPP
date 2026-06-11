#!/usr/bin/env python3
"""Compare BEM JSON Mueller output against ADDA mueller files or MBS tables."""

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


def bem_component(mueller, theta, i, j):
    m = np.asarray(mueller)
    n = len(theta)
    if m.shape == (4, 4, n):
        return m[i, j]
    if m.shape == (n, 4, 4):
        return m[:, i, j]
    if m.shape == (16, n):
        return m[4 * i + j]
    if m.shape == (n, 16):
        return m[:, 4 * i + j]
    raise ValueError(f"unknown BEM mueller shape: {m.shape}")


def load_bem(path):
    with open(path, "r") as f:
        data = json.load(f)
    return np.asarray(data["theta"]), np.asarray(data["mueller"]), data.get("timing", {})


def load_adda_average(path, beta_order):
    files = sorted(glob.glob(str(Path(path) / "*" / "mueller")))
    if Path(path).is_file():
        files = [str(path)]
    if not files:
        raise FileNotFoundError(f"no ADDA mueller files under {path}")

    beta_weights = None
    if beta_order:
        nodes, weights = np.polynomial.legendre.leggauss(beta_order)
        betas = np.degrees(np.arccos(nodes))
        beta_weights = [(float(b), float(w)) for b, w in zip(betas, weights)]

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
    return acc / wsum, len(files)


def load_mbs_table(path):
    return np.loadtxt(path, skiprows=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bem", required=True)
    parser.add_argument("--adda", help="ADDA mueller file or directory with */mueller files")
    parser.add_argument("--mbs", help="MBS/converted table")
    parser.add_argument("--beta-order", type=int, default=0,
                        help="Use Gauss-Legendre beta weights for ADDA directory averaging")
    parser.add_argument("--raw", action="store_true",
                        help="Compare raw Mueller values instead of normalizing by S11(theta=0)")
    args = parser.parse_args()

    if bool(args.adda) == bool(args.mbs):
        raise SystemExit("provide exactly one of --adda or --mbs")

    theta, bem_mueller, timing = load_bem(args.bem)
    if args.adda:
        ref, ref_count = load_adda_average(args.adda, args.beta_order)
        ref_kind = f"ADDA ({ref_count} files)"
        ref_col = lambda name: COMPONENTS[name][2]
    else:
        ref = load_mbs_table(args.mbs)
        ref_kind = "MBS table"
        ref_col = lambda name: COMPONENTS[name][3]

    bem_s11 = bem_component(bem_mueller, theta, 0, 0)
    ref_s11 = np.interp(theta, ref[:, 0], ref[:, ref_col("S11")])
    bem_norm = 1.0 if args.raw else bem_s11[0]
    ref_norm = 1.0 if args.raw else ref_s11[0]

    print(f"BEM: {args.bem}")
    print(f"Reference: {ref_kind}")
    if timing:
        print("Timing:", " ".join(f"{k}={v}" for k, v in timing.items()))
    if not args.raw:
        print(f"Scale BEM/REF S11(0): {bem_s11[0] / ref_s11[0]:.8g}")

    score = 0.0
    for name, (i, j, _, _) in COMPONENTS.items():
        y = bem_component(bem_mueller, theta, i, j) / bem_norm
        r = np.interp(theta, ref[:, 0], ref[:, ref_col(name)]) / ref_norm
        err = np.linalg.norm(y - r) / max(np.linalg.norm(r), 1e-300)
        score += err
        print(f"{name}: {err:.6g}")
    print(f"score6: {score:.6g}")


if __name__ == "__main__":
    main()
