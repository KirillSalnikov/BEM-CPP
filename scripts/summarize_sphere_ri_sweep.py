#!/usr/bin/env python3
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify_mie import (  # noqa: E402
    ALL_MUELLER,
    MAIN_GATE_MUELLER,
    component_floor2_errors,
    mie_mueller,
    mueller_component,
)


def phase_shape_metrics(bem, mie):
    bem = np.asarray(bem, dtype=float)
    mie = np.asarray(mie, dtype=float)
    scale = float(np.dot(bem, mie) / max(np.dot(mie, mie), 1e-300))
    mie_s = scale * mie
    bem_n = bem / bem[0]
    mie_n = mie_s / mie_s[0]
    shape_l2 = float(np.linalg.norm(bem_n - mie_n) / max(np.linalg.norm(mie_n), 1e-300))
    floor = 1e-3 * max(float(np.max(np.abs(mie_n))), 1e-300)
    floor_rel = np.abs(bem_n - mie_n) / np.maximum(np.abs(mie_n), floor)
    return scale, bem_n, mie_n, shape_l2, float(np.mean(floor_rel)), float(np.max(floor_rel))


def summarize_mueller_errors(errors):
    vals = [float(errors[name]) for name, _, _ in ALL_MUELLER]
    main_vals = [float(errors[name]) for name, _, _ in MAIN_GATE_MUELLER]
    worst_name, worst_value = max(
        ((name, float(errors[name])) for name, _, _ in ALL_MUELLER),
        key=lambda item: item[1],
    )
    failed_main_10 = [
        name for name, _, _ in MAIN_GATE_MUELLER
        if float(errors[name]) > 0.10
    ]
    failed_all_20 = [
        name for name, _, _ in ALL_MUELLER
        if float(errors[name]) > 0.20
    ]
    return {
        "max_main_floor2": max(main_vals),
        "max16_floor2": max(vals),
        "worst_component": worst_name,
        "worst_component_error": worst_value,
        "failed_main_10pct": ",".join(failed_main_10),
        "failed_all_20pct": ",".join(failed_all_20),
    }


def rc_for(path: Path) -> str:
    rc = path.parent / "logs" / f"{path.stem}.rc"
    return rc.read_text().strip() if rc.exists() else ""


def log_tail(path: Path) -> str:
    log = path.parent / "logs" / f"{path.stem}.log"
    if not log.exists():
        return ""
    lines = log.read_text(errors="ignore").splitlines()
    return " | ".join(lines[-8:])


def rel_source(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/sphere_ri_sweep_20260622"
    rows = []
    for path in sorted(out_dir.glob("*.json")):
        data = json.load(open(path))
        theta = np.asarray(data["theta"], dtype=float)
        mu = np.asarray(data["mueller"], dtype=float)
        bem = np.asarray(mueller_component(data["mueller"], 0, 0), dtype=float)
        ka = float(data["ka"])
        ri = data.get("ri", [math.nan, 0.0])
        n_re = float(ri[0] if isinstance(ri, list) else ri)
        mie = mie_mueller(theta, complex(n_re, 0.0), ka)
        mie_m11 = np.asarray(mie[0][0], dtype=float)
        scale, _, _, shape_l2, mean_floor_rel, max_floor_rel = phase_shape_metrics(bem, mie_m11)
        errors = component_floor2_errors(theta, data["mueller"], mie)
        error_summary = summarize_mueller_errors(errors)
        name = path.stem
        m = re.search(r"_ref([0-9]+)$", name)
        ref = int(m.group(1)) if m else int(data.get("refinements", -1))
        rows.append({
            "case": name,
            "ka": ka,
            "n": n_re,
            "ref": ref,
            "rc": rc_for(path),
            "time_s": data.get("timing", {}).get("total_s", math.nan),
            "solve_s": data.get("timing", {}).get("solve_s", math.nan),
            "matvecs": data.get("gmres_matvecs", ""),
            "nonconv": data.get("gmres_nonconverged_systems", ""),
            "final_relres": data.get("gmres_max_final_relres", ""),
            "scale": scale,
            "shape_l2": shape_l2,
            "mean_floor_rel": mean_floor_rel,
            "max_floor_rel": max_floor_rel,
            "pass10_shape_l2": shape_l2 <= 0.10,
            "pass10_full_mueller": (
                error_summary["max_main_floor2"] <= 0.10 and
                not error_summary["failed_all_20pct"]
            ),
            **error_summary,
            "source": rel_source(path),
        })
    csv_path = out_dir / "summary_mie.csv"
    fields = ["case", "ka", "n", "ref", "rc", "time_s", "solve_s", "matvecs", "nonconv",
              "final_relres", "scale", "shape_l2", "mean_floor_rel", "max_floor_rel",
              "pass10_shape_l2", "pass10_full_mueller", "max_main_floor2", "max16_floor2",
              "worst_component", "worst_component_error", "failed_main_10pct",
              "failed_all_20pct", "source"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} rows={len(rows)}")
    for row in rows:
        print(f"ka={row['ka']:g} n={row['n']:g} ref{row['ref']} "
              f"time={float(row['time_s']):.1f}s mv={row['matvecs']} "
              f"L2={100*float(row['shape_l2']):.2f}% "
              f"full={100*float(row['max_main_floor2']):.2f}% "
              f"worst={row['worst_component']} pass={row['pass10_full_mueller']}")
    for rc in sorted((out_dir / "logs").glob("*.rc")):
        if rc.read_text().strip() != "0":
            print(f"NONZERO {rc.name}: rc={rc.read_text().strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
