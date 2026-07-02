#!/usr/bin/env python3
"""Update one poster work-copy production row only when a candidate improves accuracy."""

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

from scripts.audit_accuracy_matrix_15 import score_adda, score_mie  # noqa: E402


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "pass"}
    return bool(value)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def score_candidate(args: argparse.Namespace) -> dict:
    bem = Path(args.bem)
    if args.reference == "mie":
        scored = score_mie(bem, args.ka, args.n_re, args.n_im)
    else:
        scored = score_adda(bem, Path(args.adda))
    if scored is None:
        raise SystemExit(f"cannot score candidate: {bem}")
    return dict(scored)


def update_accuracy_table(path: Path, args: argparse.Namespace, scored: dict) -> tuple[bool, float]:
    df = pd.read_csv(path)
    for col in ("metadata_errors", "metadata_warnings", "operator_warnings", "row_h_scale_complex"):
        if col in df.columns:
            df[col] = df[col].astype("object")
    mask = (df["shape"].astype(str) == args.shape) & (df["ka"].astype(float) == float(args.ka))
    if not mask.any():
        raise SystemExit(f"row not found: shape={args.shape!r} ka={args.ka}")
    idx = df.index[mask][0]
    old_gate = float(pd.to_numeric(pd.Series([df.loc[idx, "gate_error"]]), errors="coerce").iloc[0])
    new_gate = float(scored["gate_error"])
    if not args.force and math.isfinite(old_gate) and new_gate >= old_gate:
        return False, old_gate

    df.loc[idx, "mesh_label"] = args.mesh_label
    df.loc[idx, "status"] = "PASS" if new_gate <= args.pass_gate else "FAIL"
    df.loc[idx, "pass10"] = bool(new_gate <= 0.10)
    df.loc[idx, "mean16_floor2"] = scored.get("mean16_floor2", math.nan)
    df.loc[idx, "max16_floor2"] = scored.get("max16_floor2", math.nan)
    df.loc[idx, "mean_pol15_floor2"] = scored.get("mean_pol15_floor2", math.nan)
    df.loc[idx, "max_pol15_floor2"] = scored.get("max_pol15_floor2", math.nan)
    df.loc[idx, "mean_main_floor2"] = scored.get("mean_main_floor2", new_gate)
    df.loc[idx, "max_main_floor2"] = scored.get("max_main_floor2", new_gate)
    df.loc[idx, "main_floor2_pass_5pct"] = bool(new_gate <= 0.05)
    df.loc[idx, "gate_error"] = new_gate
    df.loc[idx, "mie_mean_floor2"] = scored.get("mie_mean_floor2", math.nan)
    df.loc[idx, "mie_max_floor2"] = scored.get("mie_max_floor2", math.nan)
    df.loc[idx, "m11"] = scored.get("m11", math.nan)
    df.loc[idx, "m12"] = scored.get("m12", math.nan)
    df.loc[idx, "m34"] = scored.get("m34", math.nan)
    df.loc[idx, "time_s"] = scored.get("time_s", math.nan)
    df.loc[idx, "bem_file"] = scored.get("bem_file", rel(Path(args.bem)))
    df.loc[idx, "reference_file"] = scored.get("reference_file", args.adda or "verify_mie.py")
    df.loc[idx, "reference"] = scored.get("reference", "Mie" if args.reference == "mie" else "ADDA-OCL")

    for col in [
        "metadata_status", "metadata_rank", "metadata_errors", "metadata_warnings",
        "operator_status", "operator_rank", "operator_warnings", "requested_system",
        "actual_system", "system_canonicalized", "solver_profile",
        "preconditioner_reason", "row_h_scale_complex",
    ]:
        if col in df.columns and col in scored:
            df.loc[idx, col] = scored[col]
    df.to_csv(path, index=False)
    return True, old_gate


def update_shape_time(path: Path, args: argparse.Namespace, scored: dict) -> None:
    df = pd.read_csv(path)
    mask = (df["shape"].astype(str) == args.shape) & (df["ka"].astype(float) == float(args.ka))
    if not mask.any():
        return
    idx = df.index[mask][0]
    df.loc[idx, "mesh_label"] = args.mesh_label
    if args.mesh_ref is not None:
        df.loc[idx, "mesh_ref"] = args.mesh_ref
    df.loc[idx, "time_s"] = scored.get("time_s", math.nan)
    df.loc[idx, "backend"] = args.backend
    df.loc[idx, "source"] = scored.get("bem_file", rel(Path(args.bem)))
    df.loc[idx, "ntheta"] = args.ntheta
    df.loc[idx, "angle_grid"] = "full181" if args.ntheta >= 181 else "probe"
    df.to_csv(path, index=False)


def regenerate_figures(poster: Path) -> None:
    make_assets_path = poster / "make_assets.py"
    spec = importlib.util.spec_from_file_location("poster_work_make_assets", make_assets_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.make_accuracy_matrix_15_figures()
    module.make_production_accuracy_figures()
    module.make_vram_forecast()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poster", default="poster_a0_work")
    ap.add_argument("--shape", required=True)
    ap.add_argument("--ka", type=float, required=True)
    ap.add_argument("--mesh-label", required=True)
    ap.add_argument("--mesh-ref", type=float)
    ap.add_argument("--bem", required=True)
    ap.add_argument("--reference", choices=["adda", "mie"], required=True)
    ap.add_argument("--adda", default="")
    ap.add_argument("--n-re", type=float, default=1.3116)
    ap.add_argument("--n-im", type=float, default=0.0)
    ap.add_argument("--pass-gate", type=float, default=0.10)
    ap.add_argument("--ntheta", type=int, default=181)
    ap.add_argument("--backend", default="BEM-CUDA measured run")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    poster = Path(args.poster)
    assets = poster / "assets"
    scored = score_candidate(args)
    updated, old_gate = update_accuracy_table(assets / "table_accuracy_matrix_15.csv", args, scored)
    if updated:
        update_shape_time(assets / "table_shape_single_time.csv", args, scored)
        regenerate_figures(poster)
        print(f"updated {args.shape} ka={args.ka:g}: gate {old_gate:.6g} -> {float(scored['gate_error']):.6g}")
    else:
        print(f"kept {args.shape} ka={args.ka:g}: old gate {old_gate:.6g} <= candidate {float(scored['gate_error']):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
