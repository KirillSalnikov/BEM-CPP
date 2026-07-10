#!/usr/bin/env python3
"""Audit adaptive BEM dust runs against the ready ADDA refr_1_6__0 database.

The goal gate is intentionally strict about evidence:
  * ka must be at least the requested threshold;
  * an accepted adaptive BEM JSON must exist;
  * the matching ADDA database file must exist;
  * M11 integral-normalized relative L2 must be below the threshold;
  * speedup is reported only when an ADDA timing log is explicitly provided.
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from summarize_bem_adda_m11 import load_adda, load_bem, m11_metrics  # noqa: E402


def parse_ka(text):
    try:
        return float(text)
    except Exception:
        pass
    m = re.search(r"ka([0-9]+(?:p[0-9]+)?)", text)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def find_adda(adda_dir: Path, ka: float):
    candidates = [
        adda_dir / f"A_x={ka:g}_refr_1_6__0.dat",
        adda_dir / f"A_x={ka:.2f}_refr_1_6__0.dat",
    ]
    for path in candidates:
        if path.exists():
            return path
    pattern = re.compile(r"A_x=([^_]+)_refr_1_6__0\.dat$")
    best = None
    best_delta = float("inf")
    for path in adda_dir.glob("A_x=*_refr_1_6__0.dat"):
        m = pattern.match(path.name)
        if not m:
            continue
        try:
            value = float(m.group(1))
        except ValueError:
            continue
        delta = abs(value - ka)
        if delta < best_delta:
            best = path
            best_delta = delta
    if best is not None and best_delta < 5e-3:
        return best
    return None


def load_manifest(run_dir: Path):
    path = run_dir / "adaptive_nested_bg_manifest.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def accepted_bem_path(run_dir: Path, manifest):
    if not isinstance(manifest, dict):
        return None
    accepted = manifest.get("accepted")
    if not accepted:
        return None
    path = Path(accepted)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, run_dir / path, run_dir.parent / path]
    return next((p for p in candidates if p.exists()), candidates[0])


def bem_time(meta):
    timing = meta.get("timing", {})
    value = timing.get("total_s")
    try:
        return float(value)
    except Exception:
        return math.nan


def audit_run(run_dir: Path, adda_dir: Path, min_ka: float, m11_threshold: float):
    ka = parse_ka(run_dir.name)
    row = {
        "run_dir": str(run_dir),
        "ka": ka if ka is not None else "",
        "status": "missing_ka" if ka is None else "unknown",
        "accepted": "",
        "adda": "",
        "m11_integral_rel_l2": "",
        "raw_integral_ratio": "",
        "raw_band_ratio_120_180": "",
        "bem_total_s": "",
        "orient_count": "",
        "alpha_avg": "",
        "ka_gate": False,
        "m11_gate": False,
        "goal_gate_without_speed": False,
    }
    if ka is None:
        return row
    row["ka_gate"] = ka >= min_ka
    manifest = load_manifest(run_dir)
    if manifest is None:
        row["status"] = "missing_manifest"
        return row
    row["status"] = manifest.get("status") or "running"
    bem_path = accepted_bem_path(run_dir, manifest)
    if bem_path is None or not bem_path.exists():
        row["status"] = "missing_accepted_bem" if row["status"] == "complete" else row["status"]
        return row
    row["accepted"] = str(bem_path)
    adda_path = find_adda(adda_dir, ka)
    if adda_path is None or not adda_path.exists():
        row["status"] = "missing_adda"
        return row
    row["adda"] = str(adda_path)
    theta, mueller, meta = load_bem(bem_path)
    ref_m11 = load_adda(adda_path, theta)
    metrics = m11_metrics(theta, mueller[0, 0], ref_m11)
    row["m11_integral_rel_l2"] = metrics["m11_integral_rel_l2"]
    row["raw_integral_ratio"] = metrics["raw_integral_ratio"]
    row["raw_band_ratio_120_180"] = metrics.get("raw_band_ratio_120_180", math.nan)
    row["bem_total_s"] = bem_time(meta)
    row["orient_count"] = int(meta.get("orient_count", meta.get("orient_total", 0)) or 0)
    row["alpha_avg"] = int(meta.get("alpha_avg", 1) or 1)
    row["m11_gate"] = row["m11_integral_rel_l2"] <= m11_threshold
    row["goal_gate_without_speed"] = bool(row["ka_gate"] and row["m11_gate"])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/dust_ka10_20_bem_vs_adda_db_20260710"))
    ap.add_argument("--adda-dir", type=Path,
                    default=Path("/home/user/cluster/BEM-CPP/greek/ADDA_for_PO_comparison/refr_1_6__0"))
    ap.add_argument("--min-ka", type=float, default=30.0)
    ap.add_argument("--m11-threshold", type=float, default=0.10)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    run_dirs = sorted(
        [p for p in args.runs_dir.glob("ka*_adaptive_nested_*") if p.is_dir()],
        key=lambda p: parse_ka(p.name) if parse_ka(p.name) is not None else -1.0,
    )
    rows = [audit_run(p, args.adda_dir, args.min_ka, args.m11_threshold) for p in run_dirs]
    fields = [
        "ka", "status", "ka_gate", "m11_gate", "goal_gate_without_speed",
        "m11_integral_rel_l2", "raw_integral_ratio", "raw_band_ratio_120_180",
        "bem_total_s", "orient_count", "alpha_avg", "accepted", "adda", "run_dir",
    ]
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})

    print("ka,status,ka_gate,m11_gate,goal_gate_without_speed,m11_l2,raw_integral_ratio,bem_total_s")
    for row in rows:
        print(
            f"{row.get('ka')},{row.get('status')},{row.get('ka_gate')},"
            f"{row.get('m11_gate')},{row.get('goal_gate_without_speed')},"
            f"{row.get('m11_integral_rel_l2')},{row.get('raw_integral_ratio')},"
            f"{row.get('bem_total_s')}"
        )
    passed = [r for r in rows if r.get("goal_gate_without_speed")]
    print(f"passed_without_speed={len(passed)} / total_runs={len(rows)}")
    print("speed_gate=unverified_without_explicit_adda_timing")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
