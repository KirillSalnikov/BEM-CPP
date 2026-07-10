#!/usr/bin/env python3
"""Recombine single-orientation BEM chunks with a new beta/gamma weight file.

This is meant for reusable adaptive orientation grids.  Run BEM chunks with
chunk_size=1, then rebuild a level average by changing only the quadrature
weights instead of recomputing the solved surface currents.
"""
import argparse
import copy
import json
from pathlib import Path


def read_bg_weights(path):
    rows = []
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
            rows.append(weight)
    total = sum(rows)
    if total <= 0.0:
        raise SystemExit(f"weights sum to zero in {path}")
    return [w / total for w in rows]


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
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--weights-file", required=True,
                    help="beta gamma [weight] file whose line order matches orient_start")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()

    parts_dir = Path(args.parts_dir)
    weights = read_bg_weights(args.weights_file)
    parts = sorted(parts_dir.glob("part_*.json"))
    if not parts:
        raise SystemExit(f"no part_*.json files under {parts_dir}")

    result = None
    used = []
    missing = []
    for idx in range(len(weights)):
        if weights[idx] == 0.0:
            continue
        expected = parts_dir / f"part_{idx:04d}.json"
        if not expected.exists():
            missing.append(idx)
            continue
        with expected.open() as f:
            part = json.load(f)
        start = int(part.get("orient_start", idx))
        count = int(part.get("orient_count", 0))
        if start != idx or count != 1:
            raise SystemExit(
                f"{expected} is not a reusable single-orientation chunk "
                f"(orient_start={start}, orient_count={count})"
            )
        old_w = float(part.get("orientation_weight_sum", 0.0))
        if old_w <= 0.0:
            raise SystemExit(f"{expected} has non-positive orientation_weight_sum={old_w}")
        mu = part["mueller"]
        if result is None:
            result = copy.deepcopy(part)
            result["mueller"] = zero_like(mu)
        add_scaled(result["mueller"], mu, weights[idx] / old_w)
        used.append(idx)

    if missing and not args.allow_missing:
        raise SystemExit(
            f"missing {len(missing)} required chunks, first missing indices: {missing[:20]}"
        )
    if result is None:
        raise SystemExit("no chunks were usable")

    result["orient_start"] = 0
    result["orient_count"] = len(used)
    result["orient_total"] = len(weights)
    result["orientation_weight_sum"] = sum(weights[i] for i in used)
    result["recombined_orientation_average"] = {
        "weights_file": str(args.weights_file),
        "parts_dir": str(parts_dir),
        "required_orientations": len(weights),
        "used_orientations": len(used),
        "missing_orientations": missing,
    }
    timing = result.get("timing", {})
    timing["total_s"] = sum(
        float(json.load(open(parts_dir / f"part_{i:04d}.json")).get("timing", {}).get("total_s", 0.0))
        for i in used
    )
    result["timing"] = timing

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"wrote {out} from {len(used)}/{len(weights)} orientations")


if __name__ == "__main__":
    main()
