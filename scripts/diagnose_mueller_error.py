#!/usr/bin/env python3
"""Report localized Mueller-curve errors against an ADDA/MBS table."""

import argparse
import json
from pathlib import Path

import numpy as np

from score_mbs import COMPONENTS, bem_component


def load_bem_table(path):
    with open(path, "r") as f:
        data = json.load(f)
    theta = np.asarray(data["theta"], dtype=float)
    mueller = np.asarray(data["mueller"], dtype=float)
    out = {"theta": theta}
    for name, (i, j, _, _) in COMPONENTS.items():
        out[name] = bem_component(mueller, i, j).astype(float)
    return out, data.get("timing", {})


def window_rms(theta, err, width_deg):
    if width_deg <= 0:
        idx = int(np.argmax(np.abs(err)))
        return idx, abs(err[idx])
    best_i = 0
    best = -1.0
    for i, t0 in enumerate(theta):
        mask = (theta >= t0) & (theta <= t0 + width_deg)
        if not np.any(mask):
            continue
        val = float(np.sqrt(np.mean(err[mask] * err[mask])))
        if val > best:
            best = val
            best_i = i
    return best_i, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bem", required=True)
    parser.add_argument("--mbs", required=True)
    parser.add_argument("--theta-max", type=float, default=180.0)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    bem, timing = load_bem_table(args.bem)
    mbs = np.loadtxt(args.mbs, skiprows=1)
    theta = bem["theta"]
    keep = theta <= args.theta_max
    theta = theta[keep]

    print(f"Source: {args.bem}")
    if timing:
        print("Timing:", " ".join(f"{k}={v}" for k, v in timing.items()))
    print(f"Theta points: {len(theta)}")

    s11_ref = np.interp(theta, mbs[:, 0], mbs[:, COMPONENTS["S11"][3]])
    scale_bem = bem["S11"][keep][0]
    scale_ref = s11_ref[0]
    print(f"Scale source/MBS S11(0): {scale_bem / scale_ref:.8g}")

    for name, (_, _, _, mbs_col) in COMPONENTS.items():
        y = bem[name][keep] / scale_bem
        r = np.interp(theta, mbs[:, 0], mbs[:, mbs_col]) / scale_ref
        err = y - r
        rel = np.linalg.norm(err) / max(np.linalg.norm(r), 1e-300)
        corr = np.corrcoef(y, r)[0, 1] if np.std(y) > 0 and np.std(r) > 0 else np.nan
        peak_idx = np.argsort(np.abs(err))[-args.top:][::-1]
        widx, wrms = window_rms(theta, err, args.window)
        print(f"\n{name}: rel={rel:.6g} corr={corr:.6g} "
              f"max_abs={np.max(np.abs(err)):.6g} "
              f"worst_window={theta[widx]:.1f}-{theta[widx] + args.window:.1f} rms={wrms:.6g}")
        for idx in peak_idx:
            print(f"  theta={theta[idx]:6.2f} bem={y[idx]: .8g} ref={r[idx]: .8g} err={err[idx]: .8g}")


if __name__ == "__main__":
    main()
