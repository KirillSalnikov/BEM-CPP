#!/usr/bin/env python3
"""Record ADDA M11 metrics as adaptive BEM levels become available."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def read_metrics(command):
    output = subprocess.check_output(command, universal_newlines=True)
    metrics = {}
    for line in output.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--adda-dir", type=Path, required=True)
    ap.add_argument("--summary-script", default="scripts/summarize_bem_adda_m11.py")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--max-l2", type=float, default=0.10)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    while True:
        for case in sorted(args.run_root.glob("ka*")):
            changed = False
            ka = case.name[2:].replace("p", ".")
            refs = list(args.adda_dir.glob("A_x={}_refr_*.dat".format(ka)))
            if len(refs) != 1:
                continue
            report_path = case / "level_m11_vs_adda.json"
            try:
                report = json.load(report_path.open()) if report_path.exists() else {}
            except (ValueError, OSError):
                report = {}
            levels = report.setdefault("levels", {})
            for bem in sorted(case.glob("level*_Jb*_Jg*/bem.json")):
                key = bem.parent.name
                stamp = bem.stat().st_mtime
                if key in levels and levels[key].get("mtime") == stamp:
                    continue
                metrics = read_metrics([
                    "python3", args.summary_script,
                    "--bem", str(bem), "--adda", str(refs[0]),
                ])
                l2 = metrics.get("m11_integral_rel_l2")
                grid_resolves = metrics.get("angular_grid_resolves_reference") == 1.0
                levels[key] = {
                    "bem": str(bem),
                    "adda": str(refs[0]),
                    "mtime": stamp,
                    "metrics": metrics,
                    "spatial_profile_l2_pass": isinstance(l2, float) and l2 <= args.max_l2,
                    "angular_grid_resolves_reference": grid_resolves,
                    "m11_l2_pass": (
                        isinstance(l2, float) and l2 <= args.max_l2 and grid_resolves
                    ),
                }
                changed = True
            if changed:
                report["max_l2"] = args.max_l2
                report["updated_unix"] = time.time()
                tmp = report_path.with_suffix(".json.tmp")
                with tmp.open("w") as stream:
                    json.dump(report, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                tmp.replace(report_path)
        if args.once:
            return 0
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
