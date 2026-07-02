#!/usr/bin/env python3
"""Combine completed orientation chunks from a level and run the dust ADDA gate."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level-dir", required=True, type=Path)
    ap.add_argument("--adda-dir", required=True, type=Path)
    ap.add_argument("--ka", required=True, type=float)
    ap.add_argument("--min-parts", type=int, default=1)
    ap.add_argument("--max-parts", type=int, default=0,
                    help="Use only the first N completed parts; 0 means all completed parts")
    ap.add_argument("--component-floor", type=float, default=1e-3)
    ap.add_argument("--bem-stokes-out", default="1,-1,-1")
    ap.add_argument("--bem-stokes-in", default="-1,-1,1")
    args = ap.parse_args()

    parts_dir = args.level_dir / "parts"
    parts = sorted(parts_dir.glob("part_*.json"))
    if args.max_parts > 0:
        parts = parts[:args.max_parts]
    if len(parts) < args.min_parts:
        print(f"not enough parts: {len(parts)} < {args.min_parts}")
        return 2

    out = args.level_dir / f"bem_partial_{len(parts)}parts.json"
    cmd = [sys.executable, "scripts/combine_bem_mueller.py", "--out", str(out)]
    for part in parts:
        cmd += ["--input", "orient", str(part)]
    subprocess.run(cmd, check=True)

    gate_json = args.level_dir / f"adda_gate_partial_{len(parts)}parts.json"
    gate_cmd = [
        sys.executable, "scripts/check_dust_adda_gate.py",
        "--bem", str(out),
        "--adda-dir", str(args.adda_dir),
        "--ka", str(args.ka),
        "--component-floor", str(args.component_floor),
        f"--bem-stokes-out={args.bem_stokes_out}",
        f"--bem-stokes-in={args.bem_stokes_in}",
        "--json-out", str(gate_json),
    ]
    proc = subprocess.run(
        gate_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    try:
        data = json.loads(proc.stdout)
        print(
            "partial_parts={parts} tail_l2={tail:.6g} tail_ratio={ratio:.6g} "
            "back_ratio={back:.6g} max_error={maxerr:.6g} passed={passed}".format(
                parts=len(parts),
                tail=data.get("s11_tail_l2_30_180"),
                ratio=data.get("s11_tail_ratio_median_30_180"),
                back=data.get("s11_back_ratio_median_90_180"),
                maxerr=data.get("max_error"),
                passed=data.get("passed"),
            )
        )
    except Exception:
        pass
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
