#!/usr/bin/env python3
"""ADDA-like Jmax adaptive orientation averaging through run_orient_queue.py.

ADDA defines orientation integration limits through refinement stages J, with
N(J) = 2^J + 1 nodes.  This driver evaluates a sequence of tensor grids up to
the requested Jmax values and stops when the orientation-averaged Mueller
matrix changes little between two consecutive levels.

The expensive BEM solves are dispatched through run_orient_queue.py, so each
level can be split across multiple GPUs while preserving the existing BEM
physics and output format.
"""

import argparse
import json
import math
import os
import subprocess
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


def j_to_n(j: int) -> int:
    if j < 0:
        raise ValueError("J must be non-negative")
    return (1 << j) + 1


def parse_names(text: str) -> List[str]:
    names = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in names if x not in PHYSICAL]
    if unknown:
        raise argparse.ArgumentTypeError("unknown Mueller elements: " + ",".join(unknown))
    return names


def load_mueller(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    with path.open() as f:
        data = json.load(f)
    theta = np.asarray(data["theta"], dtype=float)
    mu = np.asarray(data["mueller"], dtype=float)
    if mu.shape == (4, 4, len(theta)):
        pass
    elif mu.shape == (len(theta), 4, 4):
        mu = np.moveaxis(mu, 0, -1)
    else:
        raise ValueError(f"unsupported Mueller shape in {path}: {mu.shape}")
    return theta, mu, data


def comp(mu: np.ndarray, i: int, j: int) -> np.ndarray:
    return np.asarray(mu[i, j, :], dtype=float)


def curve_error(
    prev_path: Path,
    curr_path: Path,
    names: Iterable[str],
    component_floor: float = 1e-8,
) -> Dict[str, float]:
    theta_a, mu_a, _ = load_mueller(prev_path)
    theta_b, mu_b, _ = load_mueller(curr_path)
    if len(theta_a) != len(theta_b) or np.max(np.abs(theta_a - theta_b)) > 1e-9:
        raise ValueError("adaptive comparison requires identical theta grids")

    s11_a0 = comp(mu_a, 0, 0)[0]
    s11_b0 = comp(mu_b, 0, 0)[0]
    if abs(s11_a0) <= 1e-300 or abs(s11_b0) <= 1e-300:
        raise ValueError("S11(0) is zero; cannot normalize orientation convergence")

    out = {}  # type: Dict[str, float]
    total = 0.0
    for name in names:
        i, j = PHYSICAL[name]
        a = comp(mu_a, i, j) / s11_a0
        b = comp(mu_b, i, j) / s11_b0
        scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), component_floor)
        err = float(np.sqrt(np.mean(((b - a) / scale) ** 2)))
        out[name] = err
        total += err
    out["score"] = total / max(1, len(list(names)))
    out["max"] = max(out[name] for name in out if name in PHYSICAL)
    out["scale_change"] = float(abs(s11_b0 / s11_a0 - 1.0))
    return out


def run_checked(cmd: List[str], log_path: Path, env: Dict[str, str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    with log_path.open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", default="./run_orient_queue.py")
    ap.add_argument("--exe", default="./bin/bem_cuda_fmm")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpus", default=os.environ.get("BEM_ORIENT_GPUS", "0,1"))
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--chunk-order", choices=["sequential", "spread"], default="sequential")
    ap.add_argument("--omp-threads", type=int, default=8)
    ap.add_argument("--jmin-alpha", type=int, default=2)
    ap.add_argument("--jmin-beta", type=int, default=2)
    ap.add_argument("--jmin-gamma", type=int, default=2)
    ap.add_argument("--jmax-alpha", type=int, default=8)
    ap.add_argument("--jmax-beta", type=int, default=8)
    ap.add_argument("--jmax-gamma", type=int, default=8)
    ap.add_argument("--fixed-alpha-avg", type=int, default=0,
                    help="Use this many far-field alpha samples at every level; "
                         "then only beta/gamma change the number of BEM solves")
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--max-tol", type=float, default=0.08)
    ap.add_argument("--scale-tol", type=float, default=0.03)
    ap.add_argument("--component-floor", type=float, default=1e-8,
                    help="Minimum denominator for component-wise relative changes after S11(0) normalization")
    ap.add_argument("--min-levels", type=int, default=2)
    ap.add_argument("--elements", type=parse_names, default=parse_names("S11,S12,S22,S33,S34,S44"))
    ap.add_argument("--orient-warm-start", choices=["zero", "previous", "recycle"],
                    default=os.environ.get("BEM_ORIENT_WARM_START", "previous"),
                    help="GMRES initial guess policy inside each orientation chunk")
    ap.add_argument("--orient-recycle", type=int, default=None,
                    help="History length for --orient-warm-start recycle")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("bem_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if args.bem_args and args.bem_args[0] == "--":
        args.bem_args = args.bem_args[1:]
    if not args.bem_args:
        ap.error("pass BEM arguments after --")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_steps = 0 if args.fixed_alpha_avg > 0 else args.jmax_alpha - args.jmin_alpha
    max_steps = max(
        alpha_steps,
        args.jmax_beta - args.jmin_beta,
        args.jmax_gamma - args.jmin_gamma,
    )
    if max_steps < 0:
        ap.error("Jmax must be >= Jmin for every angle")

    env = os.environ.copy()
    env.setdefault("BEM_FAST_REORTH_OFF", "1")

    manifest = {  # type: Dict[str, object]
        "mode": "adda_like_jmax_queue",
        "jmin": {"alpha": args.jmin_alpha, "beta": args.jmin_beta, "gamma": args.jmin_gamma},
        "jmax": {"alpha": args.jmax_alpha, "beta": args.jmax_beta, "gamma": args.jmax_gamma},
        "tol": args.tol,
        "max_tol": args.max_tol,
        "scale_tol": args.scale_tol,
        "component_floor": args.component_floor,
        "fixed_alpha_avg": args.fixed_alpha_avg,
        "elements": args.elements,
        "gpus": args.gpus,
        "chunk_size": args.chunk_size,
        "chunk_order": args.chunk_order,
        "orient_warm_start": args.orient_warm_start,
        "orient_recycle": args.orient_recycle,
        "queue": args.queue,
        "exe": args.exe,
        "bem_args": args.bem_args,
        "levels": [],
    }

    prev_json = None  # type: Optional[Path]
    accepted = None  # type: Optional[Path]
    for step in range(max_steps + 1):
        ja = args.jmax_alpha if args.fixed_alpha_avg > 0 else min(args.jmax_alpha, args.jmin_alpha + step)
        jb = min(args.jmax_beta, args.jmin_beta + step)
        jg = min(args.jmax_gamma, args.jmin_gamma + step)
        na = args.fixed_alpha_avg if args.fixed_alpha_avg > 0 else j_to_n(ja)
        nb, ng = j_to_n(jb), j_to_n(jg)
        level_name = f"level{step + 1:02d}_Ja{ja}_Jb{jb}_Jg{jg}_a{na}_b{nb}_g{ng}"
        level_dir = out_dir / level_name
        out_json = level_dir / "bem.json"
        work_dir = level_dir / "parts"
        log_path = out_dir / "logs" / f"{level_name}.log"
        cmd = [
            "python3", args.queue,
            "--exe", args.exe,
            "--out", str(out_json),
            "--work-dir", str(work_dir),
            "--gpus", args.gpus,
            "--chunk-size", str(args.chunk_size),
            "--chunk-order", args.chunk_order,
            "--omp-threads", str(args.omp_threads),
            *args.bem_args,
            "--orient", "1", str(nb), str(ng),
            "--alpha-avg", str(na),
            "--orient-warm-start", args.orient_warm_start,
        ]
        if args.orient_recycle is not None:
            cmd += ["--orient-recycle", str(args.orient_recycle)]
        if not out_json.exists():
            run_checked(cmd, log_path, env, args.dry_run)
        else:
            print(f"SKIP existing {out_json}", flush=True)

        rec = {  # type: Dict[str, object]
            "step": step + 1,
            "J": {"alpha": ja, "beta": jb, "gamma": jg},
            "N": {"alpha": na, "beta": nb, "gamma": ng},
            "orient_solve_count": nb * ng,
            "alpha_avg": na,
            "out": str(out_json),
            "log": str(log_path),
            "command": cmd,
        }
        if not args.dry_run and prev_json is not None:
            err = curve_error(prev_json, out_json, args.elements, args.component_floor)
            rec["change_from_previous"] = err
            print(
                f"{level_name}: score={err['score']:.4g} max={err['max']:.4g} "
                f"scale={err['scale_change']:.4g}",
                flush=True,
            )
            if (
                step + 1 >= args.min_levels
                and err["score"] <= args.tol
                and err["max"] <= args.max_tol
                and err["scale_change"] <= args.scale_tol
            ):
                rec["accepted"] = True
                accepted = out_json
                manifest["levels"].append(rec)
                break
        manifest["levels"].append(rec)
        prev_json = out_json

    if accepted is None and prev_json is not None:
        accepted = prev_json
    manifest["accepted"] = str(accepted) if accepted is not None else None
    manifest_path = out_dir / "adaptive_jmax_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Manifest written to {manifest_path}", flush=True)
    if accepted is not None:
        print(f"Accepted BEM average: {accepted}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
