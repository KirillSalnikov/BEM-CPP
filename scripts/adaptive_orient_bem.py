#!/usr/bin/env python3
"""Run BEM-CUDA orientation averaging with ADDA-like adaptive refinement.

The solver itself uses a fixed tensor quadrature:
  alpha: uniform, usually handled by --alpha-avg without extra GMRES solves
  beta: Gauss-Legendre in cos(beta)
  gamma: uniform

This driver makes the orientation average adaptive by running a sequence of
increasing beta/gamma grids and stopping when the averaged Mueller matrix is
stable by a normalized curve metric.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


PHYSICAL = {
    "S11": (0, 0),
    "S12": (0, 1),
    "S22": (1, 1),
    "S33": (2, 2),
    "S34": (2, 3),
    "S44": (3, 3),
}


def parse_levels(text: str) -> List[int]:
    out = [int(x) for x in text.replace(":", ",").split(",") if x.strip()]
    if not out or any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("levels must be positive integers")
    return out


def load_bem(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    with path.open() as f:
        data = json.load(f)
    theta = np.asarray(data["theta"], dtype=float)
    mueller = np.asarray(data["mueller"], dtype=float)
    if mueller.shape == (4, 4, len(theta)):
        pass
    elif mueller.shape == (len(theta), 4, 4):
        mueller = np.moveaxis(mueller, 0, -1)
    else:
        raise ValueError(f"unsupported mueller shape in {path}: {mueller.shape}")
    return theta, mueller, data


def component(mueller: np.ndarray, i: int, j: int) -> np.ndarray:
    return np.asarray(mueller[i, j, :], dtype=float)


def curve_error(prev_path: Path, curr_path: Path, names: Iterable[str]) -> Dict[str, float]:
    theta_a, mu_a, _ = load_bem(prev_path)
    theta_b, mu_b, _ = load_bem(curr_path)
    if len(theta_a) != len(theta_b) or np.max(np.abs(theta_a - theta_b)) > 1e-9:
        raise ValueError("adaptive comparison requires identical theta grids")

    s11_a0 = component(mu_a, 0, 0)[0]
    s11_b0 = component(mu_b, 0, 0)[0]
    if s11_a0 == 0.0 or s11_b0 == 0.0:
        raise ValueError("S11(0) is zero; cannot normalize orientation convergence")

    result = {}  # type: Dict[str, float]
    total = 0.0
    count = 0
    for name in names:
        i, j = PHYSICAL[name]
        a = component(mu_a, i, j) / s11_a0
        b = component(mu_b, i, j) / s11_b0
        scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)
        err = float(np.sqrt(np.mean(((b - a) / scale) ** 2)))
        result[name] = err
        total += err
        count += 1
    result["score"] = total / max(1, count)
    result["max"] = max(result[name] for name in names)
    result["scale_change"] = float(abs(s11_b0 / s11_a0 - 1.0))
    return result


def run_command(cmd: List[str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adaptive ADDA-like orientation averaging driver for BEM-CUDA")
    parser.add_argument("--exe", default="./bin/bem_cuda_fmm")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alpha", type=int, default=360,
                        help="alpha samples handled through --alpha-avg")
    parser.add_argument("--beta-levels", type=parse_levels,
                        default=parse_levels("4,6,8,12,16,24,32"))
    parser.add_argument("--gamma-levels", type=parse_levels,
                        default=parse_levels("1,2,4,8,12,16,24,32"))
    parser.add_argument("--tol", type=float, default=0.02,
                        help="stop when average selected-component change is below this")
    parser.add_argument("--max-tol", type=float, default=0.06,
                        help="also require every selected component below this")
    parser.add_argument("--scale-tol", type=float, default=0.02,
                        help="also require absolute S11(0) scale change below this")
    parser.add_argument("--min-levels", type=int, default=2)
    parser.add_argument("--elements", default="S11,S12,S22,S33,S34,S44")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("bem_args", nargs=argparse.REMAINDER,
                        help="arguments passed to BEM after optional -- separator")
    args = parser.parse_args()

    if args.bem_args and args.bem_args[0] == "--":
        args.bem_args = args.bem_args[1:]
    if not args.bem_args:
        parser.error("pass BEM arguments after --, for example -- --ka 10 --shape hex_prism")

    names = [x.strip() for x in args.elements.split(",") if x.strip()]
    unknown = [x for x in names if x not in PHYSICAL]
    if unknown:
        parser.error(f"unknown Mueller elements: {', '.join(unknown)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    levels = list(zip(args.beta_levels, args.gamma_levels))
    if len(args.beta_levels) != len(args.gamma_levels):
        if len(args.gamma_levels) == 1:
            levels = [(b, args.gamma_levels[0]) for b in args.beta_levels]
        else:
            parser.error("--beta-levels and --gamma-levels must have same length, or one gamma level")

    manifest = {  # type: Dict[str, object]
        "alpha": args.alpha,
        "tol": args.tol,
        "max_tol": args.max_tol,
        "scale_tol": args.scale_tol,
        "elements": names,
        "exe": args.exe,
        "bem_args": args.bem_args,
        "levels": [],
    }

    prev_json = None  # type: Optional[Path]
    accepted = None  # type: Optional[Path]
    for li, (nb, ng) in enumerate(levels, start=1):
        level_dir = out_dir / f"level{li:02d}_b{nb}_g{ng}_a{args.alpha}"
        out_json = level_dir / "bem.json"
        cmd = [
            args.exe,
            *args.bem_args,
            "--orient", "1", str(nb), str(ng),
            "--alpha-avg", str(args.alpha),
            "--out", str(out_json),
        ]
        if not out_json.exists():
            run_command(cmd, args.dry_run)
        else:
            print(f"SKIP existing {out_json}", flush=True)

        rec = {  # type: Dict[str, object]
            "level": li,
            "beta": nb,
            "gamma": ng,
            "alpha_avg": args.alpha,
            "out": str(out_json),
            "command": cmd,
        }
        if not args.dry_run and prev_json is not None:
            err = curve_error(prev_json, out_json, names)
            rec["change_from_previous"] = err
            print(
                f"level {li}: beta={nb} gamma={ng} "
                f"score={err['score']:.4g} max={err['max']:.4g} "
                f"scale={err['scale_change']:.4g}",
                flush=True,
            )
            converged = (
                li >= args.min_levels
                and err["score"] <= args.tol
                and err["max"] <= args.max_tol
                and err["scale_change"] <= args.scale_tol
            )
            if converged:
                accepted = out_json
                rec["accepted"] = True
                manifest["levels"].append(rec)
                break
        manifest["levels"].append(rec)
        prev_json = out_json

    if accepted is None and not args.dry_run:
        accepted = prev_json
    manifest["accepted"] = str(accepted) if accepted is not None else None

    manifest_path = out_dir / "adaptive_orient_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Manifest written to {manifest_path}", flush=True)
    if accepted is not None:
        print(f"Accepted BEM average: {accepted}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
