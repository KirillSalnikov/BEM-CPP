#!/usr/bin/env python3
"""Recombine completed single-orientation parts as a renormalized subset.

This is a diagnostic companion to recombine_orient_parts.py.  It uses the same
per-part rescaling by the new quadrature weight, but normalizes only over the
completed indices.  The result is therefore a partial quadrature estimate, not a
replacement for the full level unless an external accuracy gate accepts it.
"""

import argparse
import copy
import json
from pathlib import Path


def read_weights(path):
    weights = []
    with Path(path).open() as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            weight = float(fields[2]) if len(fields) >= 3 else 1.0
            if weight < 0.0:
                raise SystemExit(f"negative weight in {path}: {line}")
            weights.append(weight)
    total = sum(weights)
    if total <= 0.0:
        raise SystemExit(f"weights sum to zero in {path}")
    return [w / total for w in weights]


def read_indices(path):
    out = []
    with Path(path).open() as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(int(line))
    return out


def zero_like(mu):
    return [[[0.0 for _ in mu[0][0]] for _ in range(4)] for _ in range(4)]


def add_scaled(dst, src, scale):
    for i in range(4):
        for j in range(4):
            d = dst[i][j]
            s = src[i][j]
            for k in range(len(d)):
                d[k] += scale * s[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts-dir", required=True, type=Path)
    ap.add_argument("--weights-file", required=True, type=Path)
    ap.add_argument("--active-indices-file", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-used", type=int, default=1)
    args = ap.parse_args()

    weights = read_weights(args.weights_file)
    if args.active_indices_file:
        candidates = read_indices(args.active_indices_file)
    else:
        candidates = [i for i, w in enumerate(weights) if w > 0.0]

    used = []
    missing = []
    raw_weight_sum = 0.0
    result = None
    total_timing = 0.0
    total_matvecs = 0
    max_relres = 0.0
    nonconv = 0

    for idx in candidates:
        if idx < 0 or idx >= len(weights) or weights[idx] <= 0.0:
            continue
        path = args.parts_dir / f"part_{idx:04d}.json"
        if not path.exists():
            missing.append(idx)
            continue
        with path.open() as f:
            part = json.load(f)
        start = int(part.get("orient_start", idx))
        count = int(part.get("orient_count", 0))
        if start != idx or count != 1:
            raise SystemExit(
                f"{path} is not a reusable single-orientation chunk "
                f"(orient_start={start}, orient_count={count})"
            )
        old_w = float(part.get("orientation_weight_sum", 0.0))
        if old_w <= 0.0:
            raise SystemExit(f"{path} has non-positive orientation_weight_sum={old_w}")
        if result is None:
            result = copy.deepcopy(part)
            result["mueller"] = zero_like(part["mueller"])
        raw_weight_sum += weights[idx]
        used.append((idx, part, old_w))

    if len(used) < args.min_used:
        raise SystemExit(f"not enough completed chunks: {len(used)} < {args.min_used}")
    if raw_weight_sum <= 0.0:
        raise SystemExit("completed subset has zero quadrature weight")

    for idx, part, old_w in used:
        add_scaled(result["mueller"], part["mueller"], (weights[idx] / raw_weight_sum) / old_w)
        timing = part.get("timing", {})
        total_timing += float(timing.get("total_s", 0.0))
        total_matvecs += int(part.get("gmres_matvecs", 0))
        max_relres = max(max_relres, float(part.get("gmres_max_final_relres", 0.0)))
        nonconv += int(part.get("gmres_nonconverged_systems", 0))

    used_indices = [idx for idx, _, _ in used]
    result["orient_start"] = 0
    result["orient_count"] = len(used_indices)
    result["orient_total"] = len(weights)
    result["orientation_weight_sum"] = 1.0
    result["active_orient_count"] = len(used_indices)
    result["partial_orientation_subset"] = {
        "weights_file": str(args.weights_file),
        "active_indices_file": str(args.active_indices_file) if args.active_indices_file else None,
        "parts_dir": str(args.parts_dir),
        "candidate_orientations": len(candidates),
        "used_orientations": len(used_indices),
        "missing_orientations": missing,
        "raw_completed_weight_sum": raw_weight_sum,
        "renormalized_to_completed_subset": True,
    }
    result["timing"] = {
        **result.get("timing", {}),
        "total_s": total_timing,
    }
    result["gmres_matvecs"] = total_matvecs
    result["gmres_nonconverged_systems"] = nonconv
    result["gmres_max_final_relres"] = max_relres

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        f"wrote {args.out} from {len(used_indices)}/{len(candidates)} completed "
        f"orientations, raw_weight_sum={raw_weight_sum:.8g}"
    )


if __name__ == "__main__":
    main()
