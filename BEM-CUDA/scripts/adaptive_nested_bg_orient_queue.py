#!/usr/bin/env python3
"""Adaptive reusable beta/gamma orientation averaging.

This driver uses a nested beta/gamma master grid.  Each level computes only
active master indices, stores one orientation per part_XXXX.json, recombines the
level with its own quadrature weights, and stops when Mueller curves stabilize.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


PHYSICAL = {
    "S11": (0, 0),
    "S12": (0, 1),
    "S22": (1, 1),
    "S33": (2, 2),
    "S34": (2, 3),
    "S44": (3, 3),
}


def parse_names(text):
    names = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in names if x not in PHYSICAL]
    if bad:
        raise argparse.ArgumentTypeError("unknown Mueller elements: " + ",".join(bad))
    return names


def load_mueller(path):
    with Path(path).open() as f:
        data = json.load(f)
    theta = np.asarray(data["theta"], dtype=float)
    mu = np.asarray(data["mueller"], dtype=float)
    if mu.shape == (4, 4, len(theta)):
        pass
    elif mu.shape == (len(theta), 4, 4):
        mu = np.moveaxis(mu, 0, -1)
    else:
        raise ValueError(f"unsupported Mueller shape in {path}: {mu.shape}")
    return theta, mu


def curve_error(prev_path, curr_path, names, component_floor):
    ta, ma = load_mueller(prev_path)
    tb, mb = load_mueller(curr_path)
    if len(ta) != len(tb) or np.max(np.abs(ta - tb)) > 1e-9:
        raise ValueError("adaptive comparison requires identical theta grids")
    s11a = ma[0, 0, 0]
    s11b = mb[0, 0, 0]
    if abs(s11a) <= 1e-300 or abs(s11b) <= 1e-300:
        raise ValueError("S11(0) is zero; cannot normalize orientation convergence")
    out = {}
    total = 0.0
    for name in names:
        i, j = PHYSICAL[name]
        a = ma[i, j, :] / s11a
        b = mb[i, j, :] / s11b
        scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), component_floor)
        err = float(np.sqrt(np.mean(((b - a) / scale) ** 2)))
        out[name] = err
        total += err
    out["score"] = total / max(1, len(names))
    out["max"] = max(out[name] for name in names)
    out["scale_change"] = float(abs(s11b / s11a - 1.0))
    return out


def write_manifest(out_dir, manifest):
    tmp = out_dir / "adaptive_nested_bg_manifest.json.tmp"
    final = out_dir / "adaptive_nested_bg_manifest.json"
    with tmp.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(final)


def run(cmd, dry_run, env=None):
    print("+ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nested-manifest", required=True)
    ap.add_argument("--queue", default="./run_orient_queue.py")
    ap.add_argument("--recombine", default="scripts/recombine_orient_parts.py")
    ap.add_argument("--exe", default="./bin/bem_cuda_fmm")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--chunk-size", type=int, default=8,
                    help="Number of active beta/gamma master points per BEM process")
    ap.add_argument("--tail-chunk-size", type=int, default=0,
                    help="Use smaller chunks near the end of a level to keep more GPUs busy")
    ap.add_argument("--tail-threshold-chunks", type=int, default=2,
                    help="Activate --tail-chunk-size when remaining chunks fit this many GPU waves")
    ap.add_argument("--chunk-order", choices=["sequential", "spread"], default="sequential",
                    help="Order used by run_orient_queue for active beta/gamma chunks")
    ap.add_argument("--omp-threads", type=int, default=8)
    ap.add_argument("--alpha-avg", type=int, default=256)
    ap.add_argument("--orient-warm-start", choices=["zero", "previous", "recycle"], default=None)
    ap.add_argument("--orient-warm-history", type=int, default=4)
    ap.add_argument("--tol", type=float, default=0.025)
    ap.add_argument("--max-tol", type=float, default=0.07)
    ap.add_argument("--scale-tol", type=float, default=0.025)
    ap.add_argument("--component-floor", type=float, default=1e-4)
    ap.add_argument("--min-levels", type=int, default=2)
    ap.add_argument("--elements", type=parse_names, default=parse_names("S11,S12,S22,S33,S34,S44"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("bem_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if args.bem_args and args.bem_args[0] == "--":
        args.bem_args = args.bem_args[1:]
    if not args.bem_args:
        ap.error("pass BEM arguments after --")

    out_dir = Path(args.out_dir)
    parts_dir = out_dir / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.nested_manifest).open() as f:
        nested = json.load(f)
    master_bg = nested["master_file"]

    manifest = {
        "mode": "adaptive_nested_bg_queue",
        "nested_manifest": args.nested_manifest,
        "master_bg_file": master_bg,
        "alpha_avg": args.alpha_avg,
        "J_alpha": int(round(np.log2(args.alpha_avg))) if args.alpha_avg > 0 else None,
        "N_alpha": args.alpha_avg,
        "tol": args.tol,
        "max_tol": args.max_tol,
        "scale_tol": args.scale_tol,
        "min_levels": args.min_levels,
        "component_floor": args.component_floor,
        "elements": args.elements,
        "parts_dir": str(parts_dir),
        "levels": [],
        "bem_args": args.bem_args,
        "status": "running",
    }
    write_manifest(out_dir, manifest)

    prev = None
    accepted = None
    for level_no, level in enumerate(nested["levels"], start=1):
        jb = level["J"]["beta"]
        jg = level["J"]["gamma"]
        level_dir = out_dir / f"level{level_no:02d}_Jb{jb}_Jg{jg}"
        level_dir.mkdir(parents=True, exist_ok=True)
        out_json = level_dir / "bem.json"
        queue_out = level_dir / "_queue_weighted_sum.json"
        cmd = [
            "python3", args.queue,
            "--exe", args.exe,
            "--out", str(queue_out),
            "--work-dir", str(parts_dir),
            "--gpus", args.gpus,
            "--chunk-size", str(args.chunk_size),
            "--chunk-order", args.chunk_order,
            "--omp-threads", str(args.omp_threads),
            *args.bem_args,
            "--orient-bg-file", master_bg,
            "--active-indices-file", level["active_indices_file"],
            "--alpha-avg", str(args.alpha_avg),
        ]
        if args.orient_warm_start is not None:
            cmd.extend(["--orient-warm-start", args.orient_warm_start])
            if args.orient_warm_start == "recycle":
                cmd.extend(["--orient-recycle", str(args.orient_warm_history)])
        if not out_json.exists():
            env = os.environ.copy()
            if args.tail_chunk_size > 0:
                env["BEM_ORIENT_TAIL_CHUNK_SIZE"] = str(args.tail_chunk_size)
                env["BEM_ORIENT_TAIL_THRESHOLD_CHUNKS"] = str(args.tail_threshold_chunks)
            run(cmd, args.dry_run, env=env)
            rec_cmd = [
                "python3", args.recombine,
                "--parts-dir", str(parts_dir),
                "--weights-file", level["file"],
                "--out", str(out_json),
            ]
            run(rec_cmd, args.dry_run)
        else:
            print(f"SKIP existing {out_json}", flush=True)

        rec = {
            "step": level_no,
            "J": level["J"],
            "N": level["N"],
            "active_count": level["active_count"],
            "out": str(out_json),
            "queue_command": cmd,
        }
        if not args.dry_run and prev is not None:
            err = curve_error(prev, out_json, args.elements, args.component_floor)
            rec["change_from_previous"] = err
            gate_passed = (
                level_no >= args.min_levels and
                err["score"] <= args.tol and
                err["max"] <= args.max_tol and
                err["scale_change"] <= args.scale_tol
            )
            reasons = []
            if level_no < args.min_levels:
                reasons.append(f"level<{args.min_levels}")
            if err["score"] > args.tol:
                reasons.append(f"score>{args.tol:g}")
            if err["max"] > args.max_tol:
                reasons.append(f"max>{args.max_tol:g}")
            if err["scale_change"] > args.scale_tol:
                reasons.append(f"scale>{args.scale_tol:g}")
            rec["gate_passed"] = gate_passed
            rec["gate_reason"] = "pass" if gate_passed else ",".join(reasons)
            print(
                f"level{level_no:02d}: score={err['score']:.4g} "
                f"max={err['max']:.4g} scale={err['scale_change']:.4g} "
                f"gate={rec['gate_reason']}",
                flush=True,
            )
            if gate_passed:
                rec["accepted"] = True
                accepted = out_json
                manifest["levels"].append(rec)
                break
        manifest["levels"].append(rec)
        write_manifest(out_dir, manifest)
        prev = out_json

    converged = accepted is not None
    if accepted is None and prev is not None:
        # Keep the finest result available for diagnostics, but do not label an
        # exhausted refinement ladder as converged.
        accepted = prev
    manifest["accepted"] = str(accepted) if accepted is not None else None
    manifest["converged"] = converged
    manifest["status"] = "complete" if converged else "not_converged"
    if accepted is not None:
        for rec in manifest["levels"]:
            if rec.get("out") == str(accepted):
                manifest["accepted_level"] = rec.get("step")
                manifest["accepted_J"] = {
                    "alpha": manifest["J_alpha"],
                    "beta": rec.get("J", {}).get("beta"),
                    "gamma": rec.get("J", {}).get("gamma"),
                }
                manifest["accepted_N"] = {
                    "alpha": manifest["N_alpha"],
                    "beta": rec.get("N", {}).get("beta"),
                    "gamma": rec.get("N", {}).get("gamma"),
                }
                manifest["accepted_active_count"] = rec.get("active_count")
                break
    write_manifest(out_dir, manifest)
    print(f"accepted={manifest['accepted']}", flush=True)


if __name__ == "__main__":
    main()
