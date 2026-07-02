#!/usr/bin/env python3
"""Gate a dust-particle BEM result against an ADDA_for_PO_comparison table."""

import argparse
import json
import re
from pathlib import Path

import numpy as np

import compare_mueller


KA_RE = re.compile(r"A_x=([-+0-9.]+)_")
DEFAULT_ELEMENTS = ("S11", "S12", "S22", "S33", "S34", "S44")


def rel_l2(y, ref):
    return float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-300))


def floored_rel_rms(y, ref, floor):
    den = np.sqrt(np.mean(np.maximum(np.abs(ref), floor) ** 2))
    return float(np.sqrt(np.mean((y - ref) ** 2)) / max(den, 1e-300))


def find_adda_table(root: Path, ka: float):
    tables = []
    for path in root.glob("A_x=*_refr_*.dat"):
        match = KA_RE.search(path.name)
        if match:
            tables.append((float(match.group(1)), path))
    if not tables:
        raise FileNotFoundError(f"no ADDA tables under {root}")
    return min(tables, key=lambda item: abs(item[0] - ka))


def parse_elements(text: str):
    if text.strip().lower() == "all":
        return list(compare_mueller.ALL_COMPONENTS)
    names = [x.strip().upper() for x in text.split(",") if x.strip()]
    unknown = [name for name in names if name not in compare_mueller.ALL_COMPONENTS]
    if unknown:
        raise argparse.ArgumentTypeError("unknown Mueller elements: " + ",".join(unknown))
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bem", required=True, type=Path)
    ap.add_argument("--adda-dir", required=True, type=Path)
    ap.add_argument("--ka", required=True, type=float)
    ap.add_argument("--elements", type=parse_elements, default=list(DEFAULT_ELEMENTS))
    ap.add_argument("--component-floor", type=float, default=1e-3)
    ap.add_argument("--max-element-error", type=float, default=0.10)
    ap.add_argument("--max-score", type=float, default=0.60,
                    help="Maximum sum of selected component errors")
    ap.add_argument("--max-tail-l2", type=float, default=0.10)
    ap.add_argument("--min-tail-ratio", type=float, default=0.90)
    ap.add_argument("--max-tail-ratio", type=float, default=1.10)
    ap.add_argument("--bem-stokes-out", type=compare_mueller.parse_stokes_signs,
                    default=(1.0, 1.0, 1.0), metavar="Q,U,V")
    ap.add_argument("--bem-stokes-in", type=compare_mueller.parse_stokes_signs,
                    default=(1.0, 1.0, 1.0), metavar="Q,U,V")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    adda_ka, table_path = find_adda_table(args.adda_dir, args.ka)
    theta, bem_mueller, timing = compare_mueller.load_bem(args.bem)
    if args.bem_stokes_out != (1.0, 1.0, 1.0) or args.bem_stokes_in != (1.0, 1.0, 1.0):
        bem_mueller = compare_mueller.apply_stokes_signs(
            bem_mueller, theta, args.bem_stokes_out, args.bem_stokes_in)
    ref = compare_mueller.load_named_table(table_path)

    bem_s11 = compare_mueller.bem_component(bem_mueller, theta, 0, 0)
    ref_s11 = np.interp(theta, ref["theta"], ref["S11"])
    bem_norm = bem_s11[0]
    ref_norm = ref_s11[0]

    errors = {}
    for name in args.elements:
        i, j, _, _ = compare_mueller.ALL_COMPONENTS[name]
        y = compare_mueller.bem_component(bem_mueller, theta, i, j) / bem_norm
        r = np.interp(theta, ref["theta"], ref[name]) / ref_norm
        errors[name] = floored_rel_rms(y, r, args.component_floor)

    s11_bem = compare_mueller.bem_component(bem_mueller, theta, 0, 0) / bem_norm
    s11_ref = ref_s11 / ref_norm
    tail = (theta >= 30.0) & (theta <= 180.0) & (s11_ref > 0.0)
    back = (theta >= 90.0) & (theta <= 180.0) & (s11_ref > 0.0)
    tail_l2 = rel_l2(s11_bem[tail], s11_ref[tail]) if np.any(tail) else float("nan")
    tail_ratio = float(np.median(s11_bem[tail] / s11_ref[tail])) if np.any(tail) else float("nan")
    back_ratio = float(np.median(s11_bem[back] / s11_ref[back])) if np.any(back) else float("nan")
    score = float(sum(errors.values()))
    max_error = float(max(errors.values())) if errors else float("nan")

    passed = (
        max_error <= args.max_element_error
        and score <= args.max_score
        and tail_l2 <= args.max_tail_l2
        and args.min_tail_ratio <= tail_ratio <= args.max_tail_ratio
    )
    out = {
        "passed": bool(passed),
        "bem": str(args.bem),
        "ka": args.ka,
        "adda_ka": adda_ka,
        "adda_table": str(table_path),
        "timing": timing,
        "component_floor": args.component_floor,
        "errors": errors,
        "score": score,
        "max_error": max_error,
        "s11_tail_l2_30_180": tail_l2,
        "s11_tail_ratio_median_30_180": tail_ratio,
        "s11_back_ratio_median_90_180": back_ratio,
        "limits": {
            "max_element_error": args.max_element_error,
            "max_score": args.max_score,
            "max_tail_l2": args.max_tail_l2,
            "min_tail_ratio": args.min_tail_ratio,
            "max_tail_ratio": args.max_tail_ratio,
        },
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    print(text)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
