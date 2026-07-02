#!/usr/bin/env python3
"""Summarize BEM-CUDA JSON accuracy against converted Shape-A reference tables."""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


COMPONENTS = [
    ("S11", 0, 0),
    ("S12", 0, 1),
    ("S22", 1, 1),
    ("S33", 2, 2),
    ("S34", 2, 3),
    ("S44", 3, 3),
]


def ax_from_name(name):
    match = re.search(r"Ax([0-9]+(?:\.[0-9]+)?)", name)
    return float(match.group(1)) if match else None


def load_bem(path):
    with open(path, "r") as f:
        data = json.load(f)
    theta = np.asarray(data["theta"], dtype=float)
    mueller = np.asarray(data["mueller"], dtype=float)
    if mueller.shape == (len(theta), 4, 4):
        mueller = np.transpose(mueller, (1, 2, 0))
    if mueller.shape != (4, 4, len(theta)):
        raise ValueError(f"unknown BEM mueller shape: {mueller.shape}")
    return theta, mueller, data


def find_ref_table(db_dir, ax):
    candidates = [
        db_dir / f"A_x={ax:.2f}_refr_1_6__0_002.dat",
        db_dir / f"A_x={ax:g}_refr_1_6__0_002.dat",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(db_dir.glob(f"A_x={ax:g}*_refr_1_6__0_002.dat"))
    return matches[0] if matches else None


def rel_l2(y, ref):
    return float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-300))


def log_rel_l2(y, ref, floor=1e-8):
    y = np.maximum(np.abs(y), floor)
    ref = np.maximum(np.abs(ref), floor)
    return float(np.linalg.norm(np.log10(y) - np.log10(ref)) /
                 max(np.linalg.norm(np.log10(ref)), 1e-300))


def score(bem_path, ref_path):
    theta, mueller, meta = load_bem(bem_path)
    ref = np.genfromtxt(ref_path, names=True)
    bem_norm = float(mueller[0, 0, 0])
    ref_norm = float(ref["S11"][0])

    result = {}
    absmax = {}
    logerr = {}
    for name, i, j in COMPONENTS:
        y = mueller[i, j] / bem_norm
        r = np.interp(theta, ref["theta"], ref[name]) / ref_norm
        result[name] = rel_l2(y, r)
        absmax[name] = float(np.max(np.abs(y - r)))
        if name in ("S11", "S22", "S33", "S44"):
            logerr[name] = log_rel_l2(y, r)

    timing = meta.get("timing", {})
    diag_names = ("S11", "S22", "S33", "S44")
    result["diag_mean"] = sum(result[name] for name in diag_names) / len(diag_names)
    result["diag_log_mean"] = sum(logerr[name] for name in diag_names) / len(diag_names)
    result["pol_abs_mean"] = 0.5 * (absmax["S12"] + absmax["S34"])
    result["score6"] = sum(result[name] for name, _, _ in COMPONENTS)
    result["time_s"] = float(timing.get("total_s", math.nan))
    result["solve_s"] = float(timing.get("solve_s", math.nan))
    result["farfield_s"] = float(timing.get("farfield_s", math.nan))
    result["S12_absmax"] = absmax["S12"]
    result["S34_absmax"] = absmax["S34"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/greek_larger_valid")
    parser.add_argument("--db", default="/home/user/BEM-CPP/greek/ADDA_for_PO_comparison/refr_1_6__0_002")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    runs = Path(args.runs)
    db_dir = Path(args.db)
    rows = []
    for path in sorted(runs.glob("bem*Ax*.json")):
        ax = ax_from_name(path.name)
        if ax is None:
            continue
        ref = find_ref_table(db_dir, ax)
        if ref is None:
            continue
        rows.append((ax, path.name, score(path, ref)))

    best = {}
    for ax, name, values in rows:
        if ax not in best or values["score6"] < best[ax][1]["score6"]:
            best[ax] = (name, values)

    header = (
        "Ax,file,time_s,diag_mean,diag_log_mean,S12_rel,S12_absmax,"
        "S34_rel,S34_absmax,pol_abs_mean,score6"
    )
    lines = [header]
    print("Ax file time_s diag diag_log S12_rel S12_abs S34_rel S34_abs pol_abs score6")
    for ax in sorted(best):
        name, v = best[ax]
        print(f"{ax:5.2f} {name} {v['time_s']:8.1f} {v['diag_mean']:8.4g} "
              f"{v['diag_log_mean']:8.4g} {v['S12']:8.3g} {v['S12_absmax']:9.3g} "
              f"{v['S34']:8.3g} {v['S34_absmax']:9.3g} {v['pol_abs_mean']:9.3g} "
              f"{v['score6']:8.3g}")
        lines.append(
            f"{ax},{name},{v['time_s']},{v['diag_mean']},{v['diag_log_mean']},"
            f"{v['S12']},{v['S12_absmax']},{v['S34']},{v['S34_absmax']},"
            f"{v['pol_abs_mean']},{v['score6']}"
        )

    if args.out:
        out = Path(args.out)
        out.write_text("\n".join(lines) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
