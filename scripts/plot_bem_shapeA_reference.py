#!/usr/bin/env python3
"""Plot BEM-CUDA Mueller JSON against converted Shape-A reference tables."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import compare_mueller


SELECTED = [(0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, 3)]
BIG_POSITIVE = {"S11", "S22", "S33", "S44"}


def load_bem(path):
    with open(path, "r") as f:
        data = json.load(f)
    theta = np.asarray(data["theta"], dtype=float)
    mueller = np.asarray(data["mueller"], dtype=float)
    n = len(theta)
    if mueller.shape == (4, 4, n):
        pass
    elif mueller.shape == (n, 4, 4):
        mueller = np.transpose(mueller, (1, 2, 0))
    elif mueller.shape == (16, n):
        mueller = mueller.reshape(4, 4, n)
    elif mueller.shape == (n, 16):
        mueller = np.transpose(mueller.reshape(n, 4, 4), (1, 2, 0))
    else:
        raise ValueError(f"unknown BEM mueller shape: {mueller.shape}")
    return theta, mueller, data


def rel_l2(y, ref):
    return float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-300))


def ref_component(ref, theta, i, j):
    return np.interp(theta, ref["theta"], ref[f"S{i + 1}{j + 1}"])


def parse_stokes_transform(text):
    return compare_mueller.parse_stokes_signs(text)


def transform_bem(theta, bem, stokes_out, stokes_in):
    if stokes_out == (1.0, 1.0, 1.0) and stokes_in == (1.0, 1.0, 1.0):
        return bem
    return compare_mueller.apply_stokes_signs(
        bem, theta, stokes_out, stokes_in)


def plot_selected(theta, bem, ref, out, title, log_big=False, ref_label="ADDA"):
    bem_norm = float(bem[0, 0, 0])
    ref_norm = float(ref["S11"][0])

    fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, (i, j) in zip(axs.ravel(), SELECTED):
        name = f"S{i + 1}{j + 1}"
        y = bem[i, j] / bem_norm
        r = ref_component(ref, theta, i, j) / ref_norm
        ax.plot(theta, r, color="#111111", lw=2.0, label=ref_label)
        ax.plot(theta, y, color="#d62728", lw=1.5, ls="--", label="BEM-CUDA")
        ax.set_title(f"{name}  rel={rel_l2(y, r):.3g}")
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
        ax.set_xlabel("theta, deg")
        ax.set_ylabel("Sij / S11(0)")
        if log_big and name in BIG_POSITIVE:
            ymin = min(np.min(y[y > 0]), np.min(r[r > 0]))
            ymax = max(np.max(y), np.max(r))
            ax.set_yscale("log")
            ax.set_ylim(max(ymin * 0.7, 1e-8), ymax * 1.3)
    axs[0, 0].legend(fontsize=9, loc="best")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bem", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Shape-A reference vs BEM-CUDA")
    parser.add_argument("--ref-label", default="ADDA")
    parser.add_argument("--log-big", action="store_true",
                        help="Use log y-scale for S11/S22/S33/S44")
    parser.add_argument("--bem-stokes-out", type=parse_stokes_transform,
                        default=(1.0, 1.0, 1.0), metavar="Q,U,V")
    parser.add_argument("--bem-stokes-in", type=parse_stokes_transform,
                        default=(1.0, 1.0, 1.0), metavar="Q,U,V")
    args = parser.parse_args()

    theta, bem, _ = load_bem(Path(args.bem))
    bem = transform_bem(theta, bem, args.bem_stokes_out, args.bem_stokes_in)
    ref = np.genfromtxt(args.ref, names=True)
    plot_selected(theta, bem, ref, Path(args.out), args.title, args.log_big,
                  args.ref_label)
    print(args.out)


if __name__ == "__main__":
    main()
