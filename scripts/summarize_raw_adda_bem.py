#!/usr/bin/env python3
"""Summarize BEM JSON comparisons against raw ADDA orientation directories."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

import compare_mueller


WALL_RE = re.compile(r"Total wall time:\s*([-+0-9.eE]+)")


def parse_ka(path):
    match = re.search(r"ka([0-9]+(?:\.[0-9]+)?)", str(path))
    if match:
        return float(match.group(1))
    return None


def adda_wall_stats(adda_dir):
    vals = []
    for log in sorted(Path(adda_dir).glob("*/log")):
        match = WALL_RE.search(log.read_text(errors="replace"))
        if match:
            vals.append(float(match.group(1)))
    if not vals:
        return {"adda_files": 0, "adda_sum_s": np.nan, "adda_max_s": np.nan}
    return {
        "adda_files": len(vals),
        "adda_sum_s": float(np.sum(vals)),
        "adda_max_s": float(np.max(vals)),
    }


def bem_wall_from_chunks(bem_path):
    parent = Path(bem_path).parent
    vals = []
    for log in sorted(parent.glob("chunk_*/run.log")):
        matches = re.findall(r"Total:\s*([0-9.]+)s", log.read_text(errors="replace"))
        if matches:
            vals.append(float(matches[-1]))
    return float(max(vals)) if vals else np.nan


def bem_wall_estimate(bem_path, meta=None):
    direct = bem_wall_from_chunks(bem_path)
    if np.isfinite(direct):
        return direct
    if meta is None:
        with Path(bem_path).open("r") as f:
            meta = json.load(f)
    richardson = meta.get("richardson")
    if richardson:
        coarse = richardson.get("coarse")
        fine = richardson.get("fine")
        if coarse and fine:
            return bem_wall_estimate(coarse) + bem_wall_estimate(fine)
    timing = meta.get("timing", {})
    return float(timing.get("total_s", np.nan))


def score_components(bem_path, adda_dir, beta_order):
    theta, bem_mueller, timing = compare_mueller.load_bem(bem_path)
    ref, ref_count = compare_mueller.load_adda_average(adda_dir, beta_order)
    bem_s11 = compare_mueller.bem_component(bem_mueller, theta, 0, 0)
    ref_s11 = np.interp(theta, ref[:, 0], ref[:, compare_mueller.COMPONENTS["S11"][2]])
    bem_norm = bem_s11[0]
    ref_norm = ref_s11[0]
    out = {}
    score = 0.0
    for name, (i, j, col, _) in compare_mueller.COMPONENTS.items():
        y = compare_mueller.bem_component(bem_mueller, theta, i, j) / bem_norm
        r = np.interp(theta, ref[:, 0], ref[:, col]) / ref_norm
        err = float(np.linalg.norm(y - r) / max(np.linalg.norm(r), 1e-300))
        out[name] = err
        score += err
    out["score6"] = score
    out["scale_s11_0"] = float(bem_s11[0] / ref_s11[0])
    out["ref_count"] = ref_count
    out["timing_total_s"] = float(timing.get("total_s", np.nan)) if timing else np.nan
    return out


def load_bem_meta(bem_path):
    with Path(bem_path).open("r") as f:
        data = json.load(f)
    return data, {
        "ka": data.get("ka"),
        "refinements": data.get("refinements"),
        "rwg_orient_count": data.get("orient_count"),
        "richardson": bool(data.get("richardson")),
    }


def format_float(value, digits=4):
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.{digits}g}"


def write_markdown(rows, out_path):
    cols = [
        ("ka", "ka"),
        ("label", "profile"),
        ("refinements", "ref"),
        ("bem_wall_s", "BEM wall"),
        ("adda_sum_s", "ADDA sum"),
        ("speedup_vs_adda_sum", "sum speedup"),
        ("score6", "score6"),
        ("S11", "S11"),
        ("S12", "S12"),
        ("S34", "S34"),
    ]
    with Path(out_path).open("w") as f:
        f.write("| " + " | ".join(title for _, title in cols) + " |\n")
        f.write("|" + "|".join("---:" if key != "label" else "---" for key, _ in cols) + "|\n")
        for row in rows:
            vals = []
            for key, _ in cols:
                val = row.get(key)
                if key == "label":
                    vals.append(str(val))
                elif key in {"ka", "refinements"}:
                    vals.append(format_float(float(val), 4) if val is not None else "")
                elif key.endswith("_s") or key == "bem_wall_s":
                    vals.append(format_float(float(val), 4))
                else:
                    vals.append(format_float(float(val), 4))
            f.write("| " + " | ".join(vals) + " |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adda-root", required=True,
                        help="Directory containing ka*/ raw ADDA directories")
    parser.add_argument("--case", action="append", nargs=3,
                        metavar=("KA", "LABEL", "BEM_JSON"), required=True)
    parser.add_argument("--beta-order", type=int, default=8)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--markdown")
    args = parser.parse_args()

    rows = []
    for ka_s, label, bem_s in args.case:
        ka = float(ka_s)
        ka_label = f"{ka:g}"
        adda_dir = Path(args.adda_root) / f"ka{ka_label}"
        if not adda_dir.exists():
            # Preserve integer-looking ka labels from CLI but allow ka10.0 fallback.
            candidates = sorted(Path(args.adda_root).glob(f"ka{int(ka)}"))
            if candidates:
                adda_dir = candidates[0]
        if not adda_dir.exists():
            raise FileNotFoundError(f"raw ADDA directory not found for ka={ka_s}: {adda_dir}")

        bem_path = Path(bem_s)
        raw_meta, meta = load_bem_meta(bem_path)
        row = {"ka": ka, "label": label, "bem_json": str(bem_path), "adda_dir": str(adda_dir)}
        row.update(meta)
        row.update(adda_wall_stats(adda_dir))
        row.update(score_components(bem_path, adda_dir, args.beta_order))
        row["bem_wall_s"] = bem_wall_estimate(bem_path, raw_meta)
        row["speedup_vs_adda_sum"] = row["adda_sum_s"] / row["bem_wall_s"]
        rows.append(row)

    rows.sort(key=lambda r: (r["ka"], r["label"]))
    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")

    if args.markdown:
        write_markdown(rows, args.markdown)
        print(f"Wrote {args.markdown}")

    for row in rows:
        print(
            f"ka={row['ka']:g} {row['label']} wall={row['bem_wall_s']:.4g}s "
            f"ADDA_sum={row['adda_sum_s']:.4g}s speedup={row['speedup_vs_adda_sum']:.4g} "
            f"score6={row['score6']:.6g} S12={row['S12']:.6g} S34={row['S34']:.6g}"
        )


if __name__ == "__main__":
    main()
