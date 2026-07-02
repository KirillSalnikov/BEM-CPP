#!/usr/bin/env python3
"""Summarize per-case GPU monitor CSV files from production queues."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def read_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({
                    "timestamp_s": float(row["timestamp_s"]),
                    "gpu": float(row["gpu"]),
                    "temp_c": float(row["temp_c"]),
                    "util_pct": float(row["util_pct"]),
                    "mem_mib": float(row["mem_mib"]),
                    "power_w": float(row["power_w"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def summarize_file(path: Path) -> Dict[str, object]:
    rows = read_rows(path)
    name = path.name
    if name.endswith(".gpu.csv"):
        name = name[:-len(".gpu.csv")]
    if not rows:
        return {
            "case": name,
            "path": str(path),
            "samples": 0,
            "valid": False,
        }
    power = [r["power_w"] for r in rows]
    temp = [r["temp_c"] for r in rows]
    util = [r["util_pct"] for r in rows]
    mem = [r["mem_mib"] for r in rows]
    duration_s = max(0.0, rows[-1]["timestamp_s"] - rows[0]["timestamp_s"])
    return {
        "case": name,
        "path": str(path),
        "samples": len(rows),
        "valid": True,
        "gpu": int(rows[0]["gpu"]),
        "duration_s": duration_s,
        "power_w": {
            "mean": mean(power),
            "p95": percentile(power, 0.95),
            "max": max(power),
        },
        "temp_c": {
            "mean": mean(temp),
            "p95": percentile(temp, 0.95),
            "max": max(temp),
        },
        "util_pct": {
            "mean": mean(util),
            "p95": percentile(util, 0.95),
            "max": max(util),
        },
        "mem_mib": {
            "mean": mean(mem),
            "p95": percentile(mem, 0.95),
            "max": max(mem),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path,
                        help="CSV files or directories with *.gpu.csv")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    args = parser.parse_args()

    inputs = args.paths or [Path("runs/production_matrix_15/logs")]
    files: List[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.glob("*.gpu.csv")))
        else:
            files.append(path)

    summaries = [summarize_file(path) for path in sorted(files)]
    if args.json:
        print(json.dumps({"files": len(summaries), "cases": summaries},
                         indent=2, ensure_ascii=False))
        return 0

    print("case,samples,gpu,duration_s,power_mean_w,power_p95_w,power_max_w,temp_max_c,mem_max_mib")
    for item in summaries:
        if not item.get("valid"):
            print(f"{item['case']},0,,,,,,,")
            continue
        print(
            f"{item['case']},{item['samples']},{item['gpu']},"
            f"{item['duration_s']:.0f},"
            f"{item['power_w']['mean']:.1f},"
            f"{item['power_w']['p95']:.1f},"
            f"{item['power_w']['max']:.1f},"
            f"{item['temp_c']['max']:.0f},"
            f"{item['mem_mib']['max']:.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
