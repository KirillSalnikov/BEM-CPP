#!/usr/bin/env python3
"""Audit triangle meshes for singular and near-singular BEM integration risk.

The report intentionally separates topological panel pairs:

* self panels need a singular Galerkin rule;
* edge/vertex adjacent panels are Duffy/Taylor-Duffy candidates;
* disjoint but geometrically close panels need a near-singular rule;
* far disjoint panels can use the normal quadrature path.

This is an audit and routing tool.  It does not evaluate singular integrals.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


def load_obj(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    verts = []
    tris = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("v "):
            _, x, y, z, *rest = line.split()
            verts.append((float(x), float(y), float(z)))
        elif line.startswith("f "):
            idx = []
            for tok in line.split()[1:]:
                idx.append(int(tok.split("/")[0]) - 1)
            if len(idx) == 3:
                tris.append(tuple(idx))
            elif len(idx) > 3:
                for i in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[i], idx[i + 1]))
    if not verts or not tris:
        raise ValueError(f"no triangular mesh found in {path}")
    return np.asarray(verts, dtype=float), np.asarray(tris, dtype=int)


def tri_stats(verts: np.ndarray, tris: np.ndarray):
    p = verts[tris]
    e01 = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
    e12 = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
    e20 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
    max_edge = np.maximum(e01, np.maximum(e12, e20))
    min_edge = np.minimum(e01, np.minimum(e12, e20))
    ctr = p.mean(axis=1)
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    return ctr, area, min_edge, max_edge


def classify_pair(shared_vertices: int, ratio: float, threshold: float) -> str:
    if shared_vertices >= 3:
        return "duplicate"
    if shared_vertices == 2:
        return "edge_adjacent"
    if shared_vertices == 1:
        return "vertex_adjacent"
    if ratio < threshold:
        return "near_disjoint"
    return "far_disjoint"


def audit_obj(path: Path, *, threshold: float, block: int = 1024) -> dict:
    verts, tris = load_obj(path)
    ctr, area, min_edge, max_edge = tri_stats(verts, tris)
    n = len(tris)
    near_pairs = 0
    edge_adjacent_pairs = 0
    vertex_adjacent_pairs = 0
    duplicate_pairs = 0
    near_disjoint_pairs = 0
    adjacent_near_pairs = 0
    min_gap_ratio = float("inf")
    worst_pair = None
    worst_pair_class = None
    for i0 in range(0, n, block):
        c0 = ctr[i0:i0 + block]
        h0 = max_edge[i0:i0 + block]
        d = np.linalg.norm(c0[:, None, :] - ctr[None, :, :], axis=2)
        scale = np.maximum(h0[:, None], max_edge[None, :])
        ratio = d / np.maximum(scale, 1e-300)
        tri_block = tris[i0:i0 + block]
        shared = np.zeros(ratio.shape, dtype=np.int8)
        for a in range(3):
            for b in range(3):
                shared += (tri_block[:, None, a] == tris[None, :, b])
        rows = np.arange(i0, min(i0 + block, n))[:, None]
        cols = np.arange(n)[None, :]
        mask = cols > rows
        local = ratio[mask]
        if local.size:
            j = int(np.argmin(local))
            local_min = float(local[j])
            if local_min < min_gap_ratio:
                pair_rows, pair_cols = np.where(mask)
                worst_pair = (int(i0 + pair_rows[j]), int(pair_cols[j]))
                min_gap_ratio = local_min
                worst_pair_class = classify_pair(
                    int(shared[mask][j]),
                    local_min,
                    threshold,
                )
        near_mask = (ratio < threshold) & mask
        shared_masked = shared[mask]
        near_shared = shared[near_mask]
        edge_adjacent_pairs += int(np.count_nonzero(shared_masked == 2))
        vertex_adjacent_pairs += int(np.count_nonzero(shared_masked == 1))
        duplicate_pairs += int(np.count_nonzero(shared_masked >= 3))
        near_disjoint_pairs += int(np.count_nonzero(near_shared == 0))
        adjacent_near_pairs += int(np.count_nonzero(near_shared > 0))
        near_pairs += int(np.count_nonzero(near_mask))

    taylor_duffy_candidates = (
        n + edge_adjacent_pairs + vertex_adjacent_pairs + near_disjoint_pairs
    )
    if duplicate_pairs:
        recommendation = "fix duplicate triangles before BEM assembly"
    elif taylor_duffy_candidates > n:
        recommendation = "route self/edge/vertex/near pairs to singular or near-singular quadrature"
    elif near_pairs:
        recommendation = "increase quadrature / remesh local gaps"
    else:
        recommendation = "ok"

    report = {
        "path": str(path),
        "triangles": int(n),
        "area_min": float(np.min(area)),
        "area_median": float(np.median(area)),
        "edge_min": float(np.min(min_edge)),
        "edge_max": float(np.max(max_edge)),
        "near_pair_threshold_edge_lengths": threshold,
        "near_pair_count": near_pairs,
        "self_panel_count": int(n),
        "edge_adjacent_pair_count": edge_adjacent_pairs,
        "vertex_adjacent_pair_count": vertex_adjacent_pairs,
        "duplicate_pair_count": duplicate_pairs,
        "near_disjoint_pair_count": near_disjoint_pairs,
        "adjacent_near_pair_count": adjacent_near_pairs,
        "taylor_duffy_candidate_count": int(taylor_duffy_candidates),
        "min_centroid_gap_over_edge": min_gap_ratio,
        "worst_pair": worst_pair,
        "worst_pair_class": worst_pair_class,
        "integration_policy": {
            "self": "singular Galerkin/Duffy",
            "edge_adjacent": "edge Duffy or Taylor-Duffy",
            "vertex_adjacent": "vertex Duffy or Taylor-Duffy",
            "near_disjoint": "adaptive near-singular high-order rule",
            "far_disjoint": "standard quadrature",
        },
        "recommendation": recommendation,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("obj", type=Path)
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="centroid distance divided by local max edge")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = audit_obj(args.obj, threshold=args.threshold)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
