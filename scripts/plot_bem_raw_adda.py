#!/usr/bin/env python3
"""Plot BEM-CUDA Mueller JSON against verified raw ADDA mueller directories."""

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SELECTED = [("S11", 0, 0), ("S12", 0, 1), ("S22", 1, 1),
            ("S33", 2, 2), ("S34", 2, 3), ("S44", 3, 3)]

ORIENT_RE = re.compile(
    r"-orient\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
)


def bem_component(mueller, theta, i, j):
    data = np.asarray(mueller)
    ntheta = len(theta)
    if data.shape == (4, 4, ntheta):
        return data[i, j]
    if data.shape == (ntheta, 4, 4):
        return data[:, i, j]
    if data.shape == (16, ntheta):
        return data[4 * i + j]
    if data.shape == (ntheta, 16):
        return data[:, 4 * i + j]
    raise ValueError(f"unknown BEM mueller shape: {data.shape}")


def load_bem(path):
    with open(path, "r") as f:
        data = json.load(f)
    theta = np.asarray(data["theta"], dtype=float)
    return theta, np.asarray(data["mueller"], dtype=float), data


def adda_files(path):
    files = sorted(glob.glob(str(Path(path) / "*" / "mueller")))
    if Path(path).is_file():
        files = [str(path)]
    if not files:
        raise FileNotFoundError(f"no ADDA mueller files under {path}")
    missing_logs = [name for name in files if not (Path(name).parent / "log").exists()]
    if missing_logs:
        sample = ", ".join(missing_logs[:3])
        raise FileNotFoundError(
            "raw ADDA plot requires mueller and adjacent log for every orientation; "
            f"missing {len(missing_logs)} logs, e.g. {sample}"
        )
    return files


def beta_weight(file_name, beta_order):
    if beta_order <= 0:
        return 1.0
    nodes, weights = np.polynomial.legendre.leggauss(beta_order)
    betas = np.degrees(np.arccos(nodes))
    match = re.search(r"_b([0-9p]+)_g", file_name)
    if match:
        beta = float(match.group(1).replace("p", "."))
    else:
        log_text = (Path(file_name).parent / "log").read_text(errors="replace")
        orient_match = ORIENT_RE.search(log_text)
        if not orient_match:
            raise ValueError(f"cannot parse beta from {file_name}")
        beta = float(orient_match.group(2))
    return float(weights[int(np.argmin(np.abs(betas - beta)))])


def load_adda_average(path, beta_order):
    files = adda_files(path)
    samples = []
    theta_union = []
    for file_name in files:
        data = np.loadtxt(file_name, skiprows=1)
        weight = beta_weight(file_name, beta_order)
        samples.append((data, weight))
        theta_union.append(data[:, 0])
    theta = np.unique(np.concatenate(theta_union))
    acc = np.zeros((len(theta), samples[0][0].shape[1]), dtype=float)
    acc[:, 0] = theta
    wsum = 0.0
    for data, weight in samples:
        acc[:, 1:] += weight * np.column_stack([
            np.interp(theta, data[:, 0], data[:, col])
            for col in range(1, data.shape[1])
        ])
        wsum += weight
    acc[:, 1:] /= wsum
    return acc, len(files)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bem", required=True)
    parser.add_argument("--adda", required=True,
                        help="Verified raw ADDA directory containing */mueller and */log")
    parser.add_argument("--out", required=True)
    parser.add_argument("--beta-order", type=int, default=0)
    parser.add_argument("--title", default="BEM-CUDA vs raw ADDA")
    parser.add_argument("--raw", action="store_true",
                        help="Plot raw Mueller values instead of normalizing by S11(0)")
    parser.add_argument("--log-big", action="store_true")
    args = parser.parse_args()

    theta, bem, meta = load_bem(Path(args.bem))
    ref, count = load_adda_average(Path(args.adda), args.beta_order)
    bem_s11 = bem_component(bem, theta, 0, 0)
    ref_s11 = np.interp(theta, ref[:, 0], ref[:, 1])
    bem_norm = 1.0 if args.raw else bem_s11[0]
    ref_norm = 1.0 if args.raw else ref_s11[0]

    fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    metrics = []
    for ax, (name, i, j) in zip(axs.ravel(), SELECTED):
        col = 1 + 4 * i + j
        y = bem_component(bem, theta, i, j) / bem_norm
        r = np.interp(theta, ref[:, 0], ref[:, col]) / ref_norm
        err = rel_l2(y, r)
        metrics.append((name, err))
        ax.plot(theta, r, color="#111111", lw=2.0, label=f"raw ADDA ({count})")
        ax.plot(theta, y, color="#d62728", lw=1.5, ls="--", label="BEM-CUDA")
        ax.set_title(f"{name} rel={err:.3g}")
        ax.set_xlabel("theta, deg")
        ax.set_ylabel("Sij" if args.raw else "Sij / S11(0)")
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
        if args.log_big and name in {"S11", "S22", "S33", "S44"}:
            pos = np.concatenate([y[y > 0], r[r > 0]])
            if len(pos):
                ax.set_yscale("log")
                ax.set_ylim(max(pos.min() * 0.7, 1e-9), pos.max() * 1.3)
    axs[0, 0].legend(fontsize=9)
    timing = meta.get("timing", {})
    time_text = f", BEM total={timing.get('total_s', 'n/a')}s" if timing else ""
    mode = "raw" if args.raw else "normalized"
    fig.suptitle(f"{args.title}; {mode}; scale={bem_s11[0] / ref_s11[0]:.6g}{time_text}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.out)
    fig.savefig(out, dpi=180)
    plt.close(fig)

    metric_path = out.with_suffix(".metrics.txt")
    with open(metric_path, "w") as f:
        f.write(f"BEM: {args.bem}\n")
        f.write(f"Raw ADDA: {args.adda}\n")
        f.write(f"ADDA files: {count}\n")
        f.write(f"Mode: {'raw' if args.raw else 'normalized'}\n")
        f.write(f"Scale BEM/ADDA S11(0): {bem_s11[0] / ref_s11[0]:.12g}\n")
        for name, err in metrics:
            f.write(f"{name}: {err:.8g}\n")
    print(out)
    print(metric_path)


if __name__ == "__main__":
    main()
