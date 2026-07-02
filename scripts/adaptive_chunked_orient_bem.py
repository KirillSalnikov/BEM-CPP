#!/usr/bin/env python3
"""Chunked ADDA-like adaptive orientation averaging for BEM-CUDA.

Unlike scripts/adaptive_orient_bem.py, this does not rerun whole quadrature
levels. It chooses one beta/gamma grid, solves it in chunks, accumulates the
weighted Mueller sum, and stops when the cumulative average changes little.
"""

import argparse
import json
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
        raise ValueError("unsupported Mueller shape in %s: %s" % (path, mu.shape))
    return theta, mu, data


def write_combined(path, theta, mu, first_meta, sources, weight_sum, timing):
    out = {
        "ka": first_meta.get("ka"),
        "ri": first_meta.get("ri"),
        "refinements": first_meta.get("refinements"),
        "orient": first_meta.get("orient"),
        "alpha_avg": first_meta.get("alpha_avg"),
        "ntheta": first_meta.get("ntheta"),
        "orient_start": 0,
        "orient_count": int(sum(s["orient_count"] for s in sources)),
        "orient_total": first_meta.get("orient_total"),
        "orientation_weight_sum": weight_sum,
        "timing": timing,
        "theta": theta.tolist(),
        "mueller": mu.tolist(),
        "combined_sources": sources,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


def component(mu, i, j):
    return np.asarray(mu[i, j, :], dtype=float)


def curve_error(prev_path, curr_path, names):
    theta_a, mu_a, _ = load_mueller(prev_path)
    theta_b, mu_b, _ = load_mueller(curr_path)
    if len(theta_a) != len(theta_b) or np.max(np.abs(theta_a - theta_b)) > 1e-9:
        raise ValueError("theta mismatch")
    s11_a0 = component(mu_a, 0, 0)[0]
    s11_b0 = component(mu_b, 0, 0)[0]
    result = {}  # type: Dict[str, float]
    total = 0.0
    for name in names:
        i, j = PHYSICAL[name]
        a = component(mu_a, i, j) / s11_a0
        b = component(mu_b, i, j) / s11_b0
        scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)
        err = float(np.sqrt(np.mean(((b - a) / scale) ** 2)))
        result[name] = err
        total += err
    result["score"] = total / max(1, len(names))
    result["max"] = max(result[name] for name in names)
    result["scale_change"] = float(abs(s11_b0 / s11_a0 - 1.0))
    return result


def run(cmd, dry_run):
    print("+ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", default="./bin/bem_cuda_fmm")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--beta", type=int, required=True)
    parser.add_argument("--gamma", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--min-chunks", type=int, default=2)
    parser.add_argument("--tol", type=float, default=0.03)
    parser.add_argument("--max-tol", type=float, default=0.08)
    parser.add_argument("--scale-tol", type=float, default=0.03)
    parser.add_argument("--elements", default="S11,S12,S22,S33,S34,S44")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("bem_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.bem_args and args.bem_args[0] == "--":
        args.bem_args = args.bem_args[1:]
    if not args.bem_args:
        parser.error("pass BEM arguments after --")
    names = [x.strip() for x in args.elements.split(",") if x.strip()]
    for name in names:
        if name not in PHYSICAL:
            parser.error("unknown Mueller element: %s" % name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_orient = args.beta * args.gamma

    accum = None
    theta = None
    first_meta = None
    weight_sum = 0.0
    timing = {"assembly_s": 0.0, "solve_s": 0.0, "farfield_s": 0.0, "total_s": 0.0}
    sources = []
    prev_combined = None  # type: Optional[Path]
    accepted = None  # type: Optional[Path]
    levels = []

    for start in range(0, total_orient, args.chunk):
        count = min(args.chunk, total_orient - start)
        chunk_dir = out_dir / ("chunk_%04d_%04d" % (start, count))
        chunk_json = chunk_dir / "bem.json"
        cmd = [
            args.exe,
            *args.bem_args,
            "--orient", "1", str(args.beta), str(args.gamma),
            "--alpha-avg", str(args.alpha),
            "--orient-start", str(start),
            "--orient-count", str(count),
            "--out", str(chunk_json),
        ]
        if not chunk_json.exists():
            run(cmd, args.dry_run)
        else:
            print("SKIP existing %s" % chunk_json, flush=True)
        if args.dry_run:
            continue

        this_theta, this_mu, meta = load_mueller(chunk_json)
        w = float(meta.get("orientation_weight_sum", 0.0))
        if w <= 0.0:
            raise ValueError("%s lacks positive orientation_weight_sum; rebuild BEM-CUDA" % chunk_json)
        if theta is None:
            theta = this_theta
            accum = np.zeros_like(this_mu)
            first_meta = meta
        elif len(theta) != len(this_theta) or np.max(np.abs(theta - this_theta)) > 1e-9:
            raise ValueError("theta mismatch in %s" % chunk_json)
        accum += this_mu * w
        weight_sum += w
        for key in timing:
            timing[key] += float(meta.get("timing", {}).get(key, 0.0))
        sources.append({
            "path": str(chunk_json),
            "orient_start": int(meta.get("orient_start", start)),
            "orient_count": int(meta.get("orient_count", count)),
            "orientation_weight_sum": w,
        })

        combined_mu = accum / weight_sum
        combined_json = out_dir / ("combined_%04d.json" % (start + count))
        write_combined(combined_json, theta, combined_mu, first_meta, sources,
                       weight_sum, timing)
        rec = {
            "included_orient": start + count,
            "combined": str(combined_json),
            "weight_sum": weight_sum,
            "timing": dict(timing),
        }
        if prev_combined is not None:
            err = curve_error(prev_combined, combined_json, names)
            rec["change_from_previous"] = err
            print(
                "included %d/%d: score=%.4g max=%.4g scale=%.4g total=%.2fs" %
                (start + count, total_orient, err["score"], err["max"],
                 err["scale_change"], timing["total_s"]),
                flush=True,
            )
            chunks_done = len(sources)
            if (chunks_done >= args.min_chunks and err["score"] <= args.tol and
                    err["max"] <= args.max_tol and err["scale_change"] <= args.scale_tol):
                rec["accepted"] = True
                accepted = combined_json
                levels.append(rec)
                break
        levels.append(rec)
        prev_combined = combined_json

    if accepted is None and prev_combined is not None:
        accepted = prev_combined
    manifest = {
        "mode": "chunked_adaptive",
        "exe": args.exe,
        "bem_args": args.bem_args,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "chunk": args.chunk,
        "tol": args.tol,
        "max_tol": args.max_tol,
        "scale_tol": args.scale_tol,
        "elements": names,
        "levels": levels,
        "accepted": str(accepted) if accepted is not None else None,
    }
    manifest_path = out_dir / "adaptive_chunked_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print("Manifest written to %s" % manifest_path, flush=True)
    if accepted is not None:
        print("Accepted BEM average: %s" % accepted, flush=True)


if __name__ == "__main__":
    main()
