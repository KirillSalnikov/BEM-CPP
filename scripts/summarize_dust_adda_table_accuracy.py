#!/usr/bin/env python3
"""Summarize final dust BEM runs against ADDA_for_PO_comparison tables."""

import argparse
import json
import re
from pathlib import Path

import numpy as np

import compare_mueller


DEFAULT_ELEMENTS = ("S11", "S12", "S22", "S33", "S34", "S44")
KA_RE = re.compile(r"A_x=([-+0-9.]+)_")


def ka_from_case(case: Path) -> float:
    manifest = case / "adaptive_final_manifest.json"
    if manifest.exists():
        with manifest.open() as f:
            return float(json.load(f)["ka"])
    name = case.name
    if name.startswith("ka"):
        name = name[2:]
    return float(name.replace("p", "."))


def index_adda_tables(path: Path):
    out = []
    for file in path.glob("A_x=*_refr_*.dat"):
        match = KA_RE.search(file.name)
        if match:
            out.append((float(match.group(1)), file))
    if not out:
        raise FileNotFoundError(f"no A_x=*_refr_*.dat files under {path}")
    return sorted(out)


def nearest_table(ka: float, tables):
    return min(tables, key=lambda item: abs(item[0] - ka))


def rel_l2(y, ref):
    return float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-300))


def floored_rel_rms(y, ref, floor):
    if floor <= 0.0:
        return rel_l2(y, ref)
    den = np.sqrt(np.mean(np.maximum(np.abs(ref), floor) ** 2))
    return float(np.sqrt(np.mean((y - ref) ** 2)) / max(den, 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--adda-dir", required=True)
    ap.add_argument("--elements", default=",".join(DEFAULT_ELEMENTS))
    ap.add_argument("--component-floor", type=float, default=0.0)
    ap.add_argument("--bem-stokes-out", type=compare_mueller.parse_stokes_signs,
                    default=(1.0, 1.0, 1.0), metavar="Q,U,V")
    ap.add_argument("--bem-stokes-in", type=compare_mueller.parse_stokes_signs,
                    default=(1.0, 1.0, 1.0), metavar="Q,U,V")
    args = ap.parse_args()

    root = Path(args.run_root)
    tables = index_adda_tables(Path(args.adda_dir))
    if args.elements.strip().lower() == "all":
        names = list(compare_mueller.ALL_COMPONENTS)
    else:
        names = [x.strip().upper() for x in args.elements.split(",") if x.strip()]

    header = [
        "case", "status", "ka", "adda_ka", "ka_delta", "J", "N",
        "time_s", "matvec_per_orientation", "nonconv", "max_relres",
        "s11_tail_l2_30_180", "s11_tail_ratio_median_30_180",
        "s11_back_ratio_median_90_180",
        *[f"err_{name}" for name in names], "score",
    ]
    print(",".join(header))
    for case in sorted(root.glob("ka*")):
        if not case.is_dir():
            continue
        bem_path = case / "final_quality" / "bem.json"
        ka = ka_from_case(case)
        adda_ka, table_path = nearest_table(ka, tables)
        final_manifest = case / "adaptive_final_manifest.json"
        accepted_j = {}
        accepted_n = {}
        if final_manifest.exists():
            with final_manifest.open() as f:
                meta = json.load(f)
            accepted_j = meta.get("accepted_J", {})
            accepted_n = meta.get("accepted_N", {})
        if not bem_path.exists():
            print(",".join([
                case.name, "missing_final", f"{ka:.8g}", f"{adda_ka:.8g}",
                f"{adda_ka - ka:.4g}", str(accepted_j).replace(",", ";"),
                str(accepted_n).replace(",", ";"), *["-"] * (9 + len(names)),
            ]))
            continue

        theta, bem_mueller, timing = compare_mueller.load_bem(bem_path)
        if args.bem_stokes_out != (1.0, 1.0, 1.0) or args.bem_stokes_in != (1.0, 1.0, 1.0):
            bem_mueller = compare_mueller.apply_stokes_signs(
                bem_mueller, theta, args.bem_stokes_out, args.bem_stokes_in)
        ref = compare_mueller.load_named_table(table_path)
        bem_s11 = compare_mueller.bem_component(bem_mueller, theta, 0, 0)
        ref_s11 = np.interp(theta, ref["theta"], ref["S11"])
        bem_norm = bem_s11[0]
        ref_norm = ref_s11[0]
        errs = []
        s11_y = None
        s11_r = None
        for name in names:
            i, j, _, _ = compare_mueller.ALL_COMPONENTS[name]
            y = compare_mueller.bem_component(bem_mueller, theta, i, j) / bem_norm
            r = np.interp(theta, ref["theta"], ref[name]) / ref_norm
            if name == "S11":
                s11_y = y
                s11_r = r
            errs.append(floored_rel_rms(y, r, args.component_floor))
        score = float(sum(errs))
        if s11_y is None:
            i, j, _, _ = compare_mueller.ALL_COMPONENTS["S11"]
            s11_y = compare_mueller.bem_component(bem_mueller, theta, i, j) / bem_norm
            s11_r = np.interp(theta, ref["theta"], ref["S11"]) / ref_norm
        tail = (theta >= 30.0) & (theta <= 180.0) & (s11_r > 0.0)
        back = (theta >= 90.0) & (theta <= 180.0) & (s11_r > 0.0)
        tail_l2 = rel_l2(s11_y[tail], s11_r[tail]) if np.any(tail) else float("nan")
        tail_ratio = float(np.median(s11_y[tail] / s11_r[tail])) if np.any(tail) else float("nan")
        back_ratio = float(np.median(s11_y[back] / s11_r[back])) if np.any(back) else float("nan")

        with bem_path.open() as f:
            bem_data = json.load(f)
        row = [
            case.name,
            "final_done",
            f"{ka:.8g}",
            f"{adda_ka:.8g}",
            f"{adda_ka - ka:.4g}",
            str(accepted_j).replace(",", ";"),
            str(accepted_n).replace(",", ";"),
            f"{float((timing or {}).get('total_s', 0.0)):.6g}",
            f"{float(bem_data.get('gmres_matvecs_per_orientation', 0.0)):.6g}",
            str(bem_data.get("gmres_nonconverged_systems", "-")),
            f"{float(bem_data.get('gmres_max_final_relres', 0.0)):.6g}",
            f"{tail_l2:.6g}",
            f"{tail_ratio:.6g}",
            f"{back_ratio:.6g}",
            *[f"{x:.6g}" for x in errs],
            f"{score:.6g}",
        ]
        print(",".join(row))


if __name__ == "__main__":
    main()
