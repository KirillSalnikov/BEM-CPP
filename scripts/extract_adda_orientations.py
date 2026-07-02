#!/usr/bin/env python3
"""Extract ADDA -orient alpha beta gamma triples from raw ADDA run logs."""

import argparse
import re
from pathlib import Path

import numpy as np


ORIENT_RE = re.compile(
    r"-orient\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
)


def nested_axis_order(n):
    def vdc2(i):
        x = 0.0
        scale = 0.5
        while i:
            if i & 1:
                x += scale
            i >>= 1
            scale *= 0.5
        return x
    return sorted(range(n), key=lambda i: (vdc2(i), i))


def value_index(values, value):
    return min(range(len(values)), key=lambda i: abs(values[i] - value))


def nested_sort_key(row, alpha_values, beta_values, gamma_values, axis_rank):
    _, alpha, beta, gamma, _ = row
    ia = value_index(alpha_values, alpha)
    ib = value_index(beta_values, beta)
    ig = value_index(gamma_values, gamma)
    ra = axis_rank["alpha"][ia]
    rb = axis_rank["beta"][ib]
    rg = axis_rank["gamma"][ig]
    # Low max-rank first gives balanced prefixes across all axes.
    return (max(ra, rb, rg), ra + rb + rg, rg, rb, ra)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("adda_dir", help="Directory containing raw ADDA */log files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--weight", type=float, default=1.0,
                        help="Weight written for each orientation before BEM normalization")
    parser.add_argument("--beta-order", type=int, default=0,
                        help="Use Gauss-Legendre beta weights instead of a constant weight")
    parser.add_argument("--euler-transform", default="identity",
                        choices=["identity", "inverse", "swap-ag", "neg-ag"],
                        help="Transform ADDA alpha,beta,gamma before writing BEM orientation file")
    parser.add_argument("--order", default="path", choices=["path", "nested"],
                        help="Orientation output order. nested makes prefixes more representative for adaptive runs")
    args = parser.parse_args()

    root = Path(args.adda_dir)
    logs = sorted(root.glob("*/log"))
    if not logs and (root / "log").exists():
        logs = [root / "log"]
    if not logs:
        raise SystemExit("no ADDA log files found under %s" % root)

    beta_weights = None
    if args.beta_order:
        nodes, weights = np.polynomial.legendre.leggauss(args.beta_order)
        betas = np.degrees(np.arccos(nodes))
        beta_weights = [(float(b), float(w)) for b, w in zip(betas, weights)]

    rows = []
    for log in logs:
        text = log.read_text(errors="replace")
        match = ORIENT_RE.search(text)
        if not match:
            raise SystemExit("cannot find -orient in %s" % log)
        alpha, beta, gamma = (float(x) for x in match.groups())
        if args.euler_transform == "inverse":
            alpha, beta, gamma = -gamma, -beta, -alpha
        elif args.euler_transform == "swap-ag":
            alpha, gamma = gamma, alpha
        elif args.euler_transform == "neg-ag":
            alpha, gamma = -alpha, -gamma
        weight = args.weight
        if beta_weights is not None:
            weight = min(beta_weights, key=lambda item: abs(item[0] - beta))[1]
        rows.append((str(log.parent), alpha, beta, gamma, weight))

    if args.order == "nested":
        alpha_values = sorted({round(row[1], 10) for row in rows})
        beta_values = sorted({round(row[2], 10) for row in rows})
        gamma_values = sorted({round(row[3], 10) for row in rows})
        axis_rank = {
            "alpha": {idx: rank for rank, idx in enumerate(nested_axis_order(len(alpha_values)))},
            "beta": {idx: rank for rank, idx in enumerate(nested_axis_order(len(beta_values)))},
            "gamma": {idx: rank for rank, idx in enumerate(nested_axis_order(len(gamma_values)))},
        }
        rows.sort(key=lambda row: nested_sort_key(row, alpha_values, beta_values, gamma_values, axis_rank))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# alpha_deg beta_deg gamma_deg weight source\n")
        for source, alpha, beta, gamma, weight in rows:
            f.write("%.12g %.12g %.12g %.12g # %s\n" %
                    (alpha, beta, gamma, weight, source))
    print("Wrote %d orientations to %s" % (len(rows), out))


if __name__ == "__main__":
    main()
