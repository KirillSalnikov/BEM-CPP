#!/usr/bin/env python3
"""Update poster work-copy sphere ka=30 rows from measured BEM/Mie candidates."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_accuracy_matrix_15 import score_mie  # noqa: E402


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "pass"}
    return bool(value)


def candidate_rows(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        if not path.exists():
            rows.append({
                "case": path.stem,
                "status": "missing",
                "bem_file": rel(path),
            })
            continue
        scored = score_mie(path, 30.0, 1.3116, 0.0)
        if scored is None:
            rows.append({
                "case": path.stem,
                "status": "unreadable",
                "bem_file": rel(path),
            })
            continue
        gate = float(scored["gate_error"])
        rows.append({
            "case": path.stem,
            "status": "PASS" if gate <= 0.10 else "FAIL",
            "pass10": gate <= 0.10,
            "pass5": gate <= 0.05,
            "mesh_label": "ref6" if "ref6" in path.stem else "ref5",
            "gate_error": gate,
            "m11": float(scored["m11"]),
            "m12": float(scored["m12"]) if math.isfinite(float(scored.get("m12", math.nan))) else math.nan,
            "m34": float(scored["m34"]) if math.isfinite(float(scored.get("m34", math.nan))) else math.nan,
            "worst_component": scored["worst_component"],
            "worst_component_error": float(scored["worst_component_error"]),
            "mie_mean_floor2": float(scored["mie_mean_floor2"]),
            "mie_max_floor2": float(scored["mie_max_floor2"]),
            "time_s": float(scored["time_s"]),
            "bem_file": rel(path),
            "reference_file": "verify_mie.py",
            "reference": "Mie",
        })
    return pd.DataFrame(rows)


def choose_best(df: pd.DataFrame) -> pd.Series:
    available = df[df["status"].isin(["PASS", "FAIL"])].copy()
    if available.empty:
        raise SystemExit("no available sphere ka30 candidates")
    available["rank_pass5"] = available["pass5"].map(lambda v: 0 if truthy(v) else 1)
    available["rank_pass10"] = available["pass10"].map(lambda v: 0 if truthy(v) else 1)
    available["rank_ref"] = available["mesh_label"].map(lambda v: 0 if str(v) == "ref6" else 1)
    available = available.sort_values(
        ["rank_pass5", "rank_pass10", "gate_error", "rank_ref", "time_s"],
        ascending=[True, True, True, True, True],
    )
    return available.iloc[0]


def update_accuracy_table(path: Path, best: pd.Series) -> None:
    df = pd.read_csv(path)
    for col in ("metadata_errors", "metadata_warnings", "operator_warnings"):
        if col in df.columns:
            df[col] = df[col].astype("object")
    mask = (df["shape"].astype(str) == "сфера") & (df["ka"].astype(float) == 30.0)
    if not mask.any():
        raise SystemExit(f"sphere ka30 row not found in {path}")
    idx = df.index[mask][0]
    df.loc[idx, "mesh_label"] = best["mesh_label"]
    df.loc[idx, "status"] = best["status"]
    df.loc[idx, "pass10"] = bool(best["pass10"])
    df.loc[idx, "mean16_floor2"] = math.nan
    df.loc[idx, "max16_floor2"] = math.nan
    df.loc[idx, "mean_pol15_floor2"] = math.nan
    df.loc[idx, "max_pol15_floor2"] = math.nan
    df.loc[idx, "mean_main_floor2"] = best["gate_error"]
    df.loc[idx, "max_main_floor2"] = best["gate_error"]
    df.loc[idx, "main_floor2_pass_5pct"] = bool(best["pass5"])
    df.loc[idx, "metadata_status"] = "ok"
    df.loc[idx, "metadata_rank"] = 0
    df.loc[idx, "metadata_errors"] = ""
    df.loc[idx, "metadata_warnings"] = "Mie-only sphere row; no ADDA full16 metrics"
    df.loc[idx, "operator_status"] = "not_required"
    df.loc[idx, "operator_rank"] = 0
    df.loc[idx, "requested_system"] = "balanced"
    df.loc[idx, "actual_system"] = "balanced"
    df.loc[idx, "system_canonicalized"] = False
    df.loc[idx, "solver_profile"] = "sphere_large"
    df.loc[idx, "preconditioner_reason"] = "auto"
    df.loc[idx, "gate_error"] = best["gate_error"]
    df.loc[idx, "mie_mean_floor2"] = best["mie_mean_floor2"]
    df.loc[idx, "mie_max_floor2"] = best["mie_max_floor2"]
    df.loc[idx, "m11"] = best["m11"]
    df.loc[idx, "m12"] = best["m12"]
    df.loc[idx, "m34"] = best["m34"]
    df.loc[idx, "time_s"] = best["time_s"]
    df.loc[idx, "bem_file"] = best["bem_file"]
    df.loc[idx, "reference_file"] = "verify_mie.py"
    df.loc[idx, "reference"] = "Mie"
    df.to_csv(path, index=False)


def update_speed_pair(path: Path, best: pd.Series) -> None:
    df = pd.read_csv(path)
    mask = (df["shape"].astype(str) == "сфера") & (df["ka"].astype(float) == 30.0)
    df = df[~mask].copy()
    rows = [
        ("M11", best["m11"]),
        ("M12", best["m12"]),
        ("M34", best["m34"]),
        ("gate", best["gate_error"]),
        ("max main floor", best["gate_error"]),
    ]
    add = pd.DataFrame([{
        "shape": "сфера",
        "ka": 30.0,
        "mesh_label": best["mesh_label"],
        "component": component,
        "rel_l2": float(value),
        "rel_l2_percent": 100.0 * float(value),
        "pass10": bool(float(value) <= 0.10),
        "status": "PASS" if float(value) <= 0.10 else "FAIL",
        "bem_file": best["bem_file"],
        "reference": "Mie",
    } for component, value in rows if math.isfinite(float(value))])
    pd.concat([df, add], ignore_index=True).to_csv(path, index=False)


def update_shape_time(path: Path, best: pd.Series) -> None:
    df = pd.read_csv(path)
    mask = (df["shape"].astype(str) == "сфера") & (df["ka"].astype(float) == 30.0)
    if not mask.any():
        raise SystemExit(f"sphere ka30 row not found in {path}")
    idx = df.index[mask][0]
    df.loc[idx, "mesh_ref"] = int(str(best["mesh_label"]).replace("ref", ""))
    df.loc[idx, "time_s"] = best["time_s"]
    df.loc[idx, "backend"] = "BEM-CUDA measured run"
    df.loc[idx, "source"] = best["bem_file"]
    df.loc[idx, "ntheta"] = 181
    df.loc[idx, "angle_grid"] = "full181"
    df.to_csv(path, index=False)


def regenerate_dependent_figures(poster: Path) -> None:
    make_assets_path = poster / "make_assets.py"
    if not make_assets_path.exists():
        return
    spec = importlib.util.spec_from_file_location("poster_work_make_assets", make_assets_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.make_accuracy_matrix_15_figures()
    module.make_production_accuracy_figures()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poster", default="poster_a0_work")
    ap.add_argument("--candidate", action="append", type=Path, default=[])
    args = ap.parse_args()

    poster = Path(args.poster)
    assets = poster / "assets"
    default_candidates = [
        ROOT / "runs/sphere30_ref6_rerun/sphere_ka30_ref5_q13_d7_tol1e3.json",
        ROOT / "runs/sphere30_ref6_rerun/sphere_ka30_ref6_q7_d7_tol3e3.json",
        ROOT / "runs/sphere30_ref6_rerun/sphere_ka30_ref6_q7_d7_tol1e3.json",
        ROOT / "runs/sphere30_ref6_rerun/sphere_ka30_ref6_q13_d7_tol3e3_leaf256.json",
    ]
    paths = args.candidate or default_candidates
    df = candidate_rows(paths)
    assets.mkdir(parents=True, exist_ok=True)
    df.to_csv(assets / "table_sphere_ka30_candidates.csv", index=False)
    best = choose_best(df)
    update_accuracy_table(assets / "table_accuracy_matrix_15.csv", best)
    update_speed_pair(assets / "table_speed_pair_accuracy_by_shape.csv", best)
    update_shape_time(assets / "table_shape_single_time.csv", best)
    regenerate_dependent_figures(poster)
    print("best", best["case"], "gate", best["gate_error"], "time", best["time_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
