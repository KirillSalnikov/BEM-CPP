#!/usr/bin/env python3
"""Scale a DDSCAT/ADDA read-shape file to a target ka at fixed dpl."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def read_shape(path: Path):
    lines = path.read_text().splitlines()
    if len(lines) < 6:
        raise ValueError(f"shape file too short: {path}")
    nat = int(lines[1].split()[0])
    coords = []
    comps = []
    for line in lines[6:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        coords.append(tuple(int(x) for x in parts[1:4]))
        comps.append(tuple(int(x) for x in parts[4:7]))
    if len(coords) != nat:
        raise ValueError(f"NAT header {nat} but parsed {len(coords)} dipoles")
    return lines[:6], coords, comps


def scaled_set(coords, scale: float):
    occ = set(coords)
    center = [sum(p[d] for p in coords) / len(coords) for d in range(3)]
    mins = [min(p[d] for p in coords) for d in range(3)]
    maxs = [max(p[d] for p in coords) for d in range(3)]
    out_mins = [math.floor((mins[d] - center[d] - 0.5) * scale) for d in range(3)]
    out_maxs = [math.ceil((maxs[d] - center[d] + 0.5) * scale) for d in range(3)]
    out = set()
    for x in range(out_mins[0], out_maxs[0] + 1):
        px = int(round(center[0] + x / scale))
        for y in range(out_mins[1], out_maxs[1] + 1):
            py = int(round(center[1] + y / scale))
            for z in range(out_mins[2], out_maxs[2] + 1):
                pz = int(round(center[2] + z / scale))
                if (px, py, pz) in occ:
                    out.add((x, y, z))
    norm_mins = [min(p[d] for p in out) for d in range(3)]
    return {tuple(p[d] - norm_mins[d] + 1 for d in range(3)) for p in out}


def choose_scale(coords, target_nat: int, initial: float):
    lo = initial * 0.90
    hi = initial * 1.10
    for _ in range(80):
        if len(scaled_set(coords, lo)) > target_nat:
            hi = lo
            lo *= 0.95
        elif len(scaled_set(coords, hi)) < target_nat:
            lo = hi
            hi *= 1.05
        else:
            break
    best_scale = initial
    best_set = scaled_set(coords, initial)
    best_err = abs(len(best_set) - target_nat)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cand = scaled_set(coords, mid)
        err = abs(len(cand) - target_nat)
        if err < best_err:
            best_scale, best_set, best_err = mid, cand, err
        if len(cand) < target_nat:
            lo = mid
        else:
            hi = mid
    return best_scale, best_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--target-ka", type=float, required=True)
    ap.add_argument("--dpl", type=float, required=True)
    args = ap.parse_args()

    header, coords, comps = read_shape(args.input)
    source_nat = len(coords)
    source_a_over_d = (3.0 * source_nat / (4.0 * math.pi)) ** (1.0 / 3.0)
    target_a_over_d = args.target_ka * args.dpl / (2.0 * math.pi)
    target_nat = int(round((4.0 * math.pi / 3.0) * target_a_over_d ** 3))
    initial = target_a_over_d / source_a_over_d
    scale, out_coords = choose_scale(coords, target_nat, initial)
    out_coords = sorted(out_coords)
    out_nat = len(out_coords)
    out_a_over_d = (3.0 * out_nat / (4.0 * math.pi)) ** (1.0 / 3.0)
    out_ka = 2.0 * math.pi * out_a_over_d / args.dpl

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.write(
            f"BEM-CUDA scaled read-shape; target_ka={args.target_ka:g}; "
            f"dpl={args.dpl:g}; source={args.input}; scale={scale:.12g}; "
            f"a_over_d={out_a_over_d:.12g}; actual_ka={out_ka:.12g}\n"
        )
        f.write(f"{out_nat} = NAT\n")
        for line in header[2:6]:
            f.write(line + "\n")
        comp = comps[0] if comps else (1, 1, 1)
        for ja, p in enumerate(out_coords, start=1):
            f.write(f"{ja} {p[0]} {p[1]} {p[2]} {comp[0]} {comp[1]} {comp[2]}\n")

    print(f"source_nat={source_nat}")
    print(f"target_nat={target_nat}")
    print(f"out_nat={out_nat}")
    print(f"scale={scale:.12g}")
    print(f"actual_ka_at_dpl={out_ka:.12g}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
