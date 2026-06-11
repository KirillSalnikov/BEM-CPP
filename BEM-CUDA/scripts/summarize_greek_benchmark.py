#!/usr/bin/env python3
"""Summarize Greek-particle BEM runs against the ADDA/MBS database."""

import argparse
import re
import subprocess
from pathlib import Path


DEFAULT_RUNS = Path("runs/greek_larger_valid")
DEFAULT_MBS_DIR = Path("/home/user/cluster/BEM-CPP/greek/ADDA_for_PO_comparison/refr_1_6__0_002")


def parse_float_line(text, key):
    match = re.search(rf"^{re.escape(key)}:\s*([-+0-9.eE]+)", text, re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_meta_time(text):
    match = re.search(r"total_s=([-+0-9.eE]+)", text)
    return float(match.group(1)) if match else None


def ax_from_name(path):
    match = re.search(r"Ax([0-9.]+)", path.name)
    return float(match.group(1)) if match else None


def mesh_from_name(path, ax):
    stem = path.stem
    marker = f"_Ax{ax:g}"
    if marker not in stem:
        marker = f"_Ax{ax:.2f}".rstrip("0").rstrip(".")
    prefix = stem.split(marker)[0] if marker in stem else stem
    return prefix.removeprefix("bem_")


def score_file(path, mbs_dir, theta_max):
    ax = ax_from_name(path)
    if ax is None:
        return None
    ref = mbs_dir / f"A_x={ax:g}_refr_1_6__0_002.dat"
    if not ref.is_file():
        ref = mbs_dir / f"A_x={ax:.2f}_refr_1_6__0_002.dat"
    if not ref.is_file():
        return None
    out = subprocess.check_output(
        [
            "python3",
            "scripts/score_mbs.py",
            "--bem",
            str(path),
            "--mbs",
            str(ref),
            "--theta-max",
            str(theta_max),
        ],
        text=True,
    )
    return {
        "ax": ax,
        "mesh": mesh_from_name(path, ax),
        "path": path,
        "time": parse_meta_time(out),
        "score6": parse_float_line(out, "score6"),
        "score6_s11w": parse_float_line(out, "score6_s11w"),
        "s12": parse_float_line(out, "S12"),
        "s34": parse_float_line(out, "S34"),
    }


def print_table(rows):
    print("A_x, time_s, score6, score6_s11w, S12, S34, mesh")
    for r in rows:
        print(
            f"{r['ax']:g}, "
            f"{r['time'] if r['time'] is not None else float('nan'):.2f}, "
            f"{r['score6']:.6g}, {r['score6_s11w']:.6g}, "
            f"{r['s12']:.6g}, {r['s34']:.6g}, {r['mesh']}"
        )


def print_best(rows):
    by_ax = {}
    for r in rows:
        by_ax.setdefault(r["ax"], []).append(r)
    print("\nBest per A_x by strict score6:")
    for ax in sorted(by_ax):
        best = min(by_ax[ax], key=lambda r: r["score6"])
        fastest_good = min(
            (r for r in by_ax[ax] if r["score6"] <= 1.15 * best["score6"]),
            key=lambda r: r["time"] if r["time"] is not None else float("inf"),
        )
        note = "best" if fastest_good is best else "fast within 15% of best"
        print(
            f"A_x={ax:g}: {fastest_good['mesh']} "
            f"time={fastest_good['time']:.2f}s score6={fastest_good['score6']:.6g} "
            f"score6_s11w={fastest_good['score6_s11w']:.6g} ({note})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--mbs-dir", type=Path, default=DEFAULT_MBS_DIR)
    parser.add_argument("--theta-max", type=float, default=180.0)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.runs.glob("bem_*Ax*.json")):
        row = score_file(path, args.mbs_dir, args.theta_max)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (r["ax"], r["score6"], r["time"] or 0.0))
    print_table(rows)
    print_best(rows)


if __name__ == "__main__":
    main()
