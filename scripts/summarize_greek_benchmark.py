#!/usr/bin/env python3
"""Summarize Greek-particle BEM runs against the ADDA/MBS database."""

import argparse
import re
import subprocess
from pathlib import Path

from greek_profiles import select_greek_profile


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


def adda_sizes(mbs_dir):
    values = []
    for path in mbs_dir.glob("A_x=*_refr_1_6__0_002.dat"):
        match = re.search(r"A_x=([0-9.]+)_", path.name)
        if match:
            values.append(float(match.group(1)))
    return sorted(values)


def fastest_within(rows, metric, tolerance):
    best = min(rows, key=lambda r: r[metric])
    cutoff = (1.0 + tolerance) * best[metric]
    candidates = [r for r in rows if r[metric] <= cutoff]
    fastest = min(candidates, key=lambda r: r["time"] if r["time"] is not None else float("inf"))
    return best, fastest


def weak_component_note(row):
    if row["score6_s11w"] < 0.05 and (row["s12"] > 0.7 or row["s34"] > 0.3):
        return " weak-component dominated"
    return ""


def print_best(rows, tolerance):
    by_ax = {}
    for r in rows:
        by_ax.setdefault(r["ax"], []).append(r)
    pct = 100.0 * tolerance
    print(f"\nFastest profiles within {pct:.0f}% of best strict score6:")
    for ax in sorted(by_ax):
        best, fastest_good = fastest_within(by_ax[ax], "score6", tolerance)
        note = "best" if fastest_good is best else f"fast within {pct:.0f}% of best"
        print(
            f"A_x={ax:g}: {fastest_good['mesh']} "
            f"time={fastest_good['time']:.2f}s score6={fastest_good['score6']:.6g} "
            f"score6_s11w={fastest_good['score6_s11w']:.6g} ({note})"
            f"{weak_component_note(fastest_good)}"
        )

    print(f"\nFastest profiles within {pct:.0f}% of best S11-weighted score:")
    for ax in sorted(by_ax):
        best, fastest_good = fastest_within(by_ax[ax], "score6_s11w", tolerance)
        note = "best" if fastest_good is best else f"fast within {pct:.0f}% of best"
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
    parser.add_argument("--within", type=float, default=0.15,
                        help="relative tolerance for fastest-within-best summaries")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.runs.glob("bem_*Ax*.json")):
        row = score_file(path, args.mbs_dir, args.theta_max)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (r["ax"], r["score6"], r["time"] or 0.0))
    print_table(rows)
    print_best(rows, args.within)

    sizes = adda_sizes(args.mbs_dir)
    if rows and sizes:
        max_done = max(r["ax"] for r in rows)
        next_sizes = [v for v in sizes if v > max_done + 1e-12]
        if next_sizes:
            ax = next_sizes[0]
            profile, extrapolated = select_greek_profile(ax)
            status = "extrapolated" if extrapolated else "validated"
            print(
                f"\nNext Shape-A reference size: A_x={ax:g}; "
                f"profile={profile.mesh} ({status})"
            )


if __name__ == "__main__":
    main()
