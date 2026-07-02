#!/usr/bin/env python3
"""Estimate H-matrix/ACA memory feasibility for a surface mesh.

This is a routing and sizing audit, not a solver backend.  It answers a narrow
question needed before implementing an ACA fallback: for the current geometry,
how many cluster pairs are geometrically admissible, and what memory reduction
would a capped-rank block representation provide compared with a dense complex
PMCHWT matrix?
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from near_singular_audit import load_obj, tri_stats


@dataclass
class Cluster:
    indices: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    level: int
    left: int = -1
    right: int = -1

    @property
    def is_leaf(self) -> bool:
        return self.left < 0 and self.right < 0

    @property
    def count(self) -> int:
        return int(self.indices.size)

    @property
    def diameter(self) -> float:
        return float(np.linalg.norm(self.bbox_max - self.bbox_min))


def bbox_distance(a: Cluster, b: Cluster) -> float:
    sep = np.maximum(0.0, np.maximum(a.bbox_min - b.bbox_max, b.bbox_min - a.bbox_max))
    return float(np.linalg.norm(sep))


def admissible(a: Cluster, b: Cluster, eta: float) -> bool:
    if a is b:
        return False
    dist = bbox_distance(a, b)
    return dist > eta * max(a.diameter, b.diameter)


def build_cluster_tree(points: np.ndarray, max_leaf: int) -> Tuple[List[Cluster], int]:
    nodes: List[Cluster] = []

    def add(indices: np.ndarray, level: int) -> int:
        pts = points[indices]
        node = Cluster(
            indices=indices,
            bbox_min=np.min(pts, axis=0),
            bbox_max=np.max(pts, axis=0),
            level=level,
        )
        idx = len(nodes)
        nodes.append(node)
        if indices.size > max_leaf:
            extent = node.bbox_max - node.bbox_min
            axis = int(np.argmax(extent))
            order = np.argsort(points[indices, axis], kind="mergesort")
            mid = indices.size // 2
            left_indices = indices[order[:mid]]
            right_indices = indices[order[mid:]]
            if left_indices.size and right_indices.size:
                nodes[idx].left = add(left_indices, level + 1)
                nodes[idx].right = add(right_indices, level + 1)
        return idx

    root = add(np.arange(points.shape[0], dtype=np.int64), 0)
    return nodes, root


def dofs_for_cluster(node: Cluster, dofs_per_triangle: float) -> int:
    return max(1, int(round(node.count * dofs_per_triangle)))


def block_memory_bytes(m: int, n: int, rank: int, complex_bytes: int) -> Tuple[int, int]:
    dense = m * n * complex_bytes
    low_rank = rank * (m + n) * complex_bytes
    return dense, min(dense, low_rank)


def audit_obj(path: Path, *, max_leaf: int, eta: float, rank: int,
              dofs_per_triangle: float, complex_bytes: int) -> dict:
    verts, tris = load_obj(path)
    centers, area, _min_edge, _max_edge = tri_stats(verts, tris)
    nodes, root = build_cluster_tree(centers, max_leaf=max_leaf)

    dense_block_count = 0
    admissible_block_count = 0
    near_block_count = 0
    dense_bytes_partitioned = 0
    hmatrix_bytes = 0
    max_level = max(node.level for node in nodes)

    stack = [(root, root)]
    while stack:
        ia, ib = stack.pop()
        a = nodes[ia]
        b = nodes[ib]
        m = dofs_for_cluster(a, dofs_per_triangle)
        n = dofs_for_cluster(b, dofs_per_triangle)
        dense, low_rank = block_memory_bytes(m, n, rank, complex_bytes)

        if admissible(a, b, eta):
            admissible_block_count += 1
            dense_bytes_partitioned += dense
            hmatrix_bytes += low_rank
            continue

        if a.is_leaf and b.is_leaf:
            dense_block_count += 1
            near_block_count += 1
            dense_bytes_partitioned += dense
            hmatrix_bytes += dense
            continue

        if b.is_leaf or (not a.is_leaf and a.count >= b.count):
            stack.append((a.left, ib))
            stack.append((a.right, ib))
        else:
            stack.append((ia, b.left))
            stack.append((ia, b.right))

    total_dofs = int(round(len(tris) * dofs_per_triangle))
    dense_full_bytes = total_dofs * total_dofs * complex_bytes
    compression = dense_full_bytes / max(hmatrix_bytes, 1)
    report = {
        "path": str(path),
        "triangles": int(len(tris)),
        "area_min": float(np.min(area)),
        "area_median": float(np.median(area)),
        "estimated_total_dofs": total_dofs,
        "dofs_per_triangle": dofs_per_triangle,
        "complex_bytes": complex_bytes,
        "max_leaf_triangles": int(max_leaf),
        "eta": float(eta),
        "rank": int(rank),
        "clusters": int(len(nodes)),
        "max_level": int(max_level),
        "dense_block_count": int(dense_block_count),
        "near_block_count": int(near_block_count),
        "admissible_block_count": int(admissible_block_count),
        "dense_full_bytes": int(dense_full_bytes),
        "dense_partitioned_bytes": int(dense_bytes_partitioned),
        "hmatrix_estimated_bytes": int(hmatrix_bytes),
        "dense_full_gb": dense_full_bytes / 1e9,
        "hmatrix_estimated_gb": hmatrix_bytes / 1e9,
        "compression_vs_dense": float(compression),
        "admissible_fraction": (
            admissible_block_count / max(admissible_block_count + dense_block_count, 1)
        ),
        "recommendation": (
            "H-matrix/ACA memory fallback is promising"
            if compression >= 2.0 and admissible_block_count > 0
            else "H-matrix/ACA fallback needs lower rank, larger clusters, or is not useful for this mesh"
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("obj", type=Path)
    parser.add_argument("--max-leaf-triangles", type=int, default=64)
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--dofs-per-triangle", type=float, default=3.0,
                        help="PMCHWT unknown estimate; closed RWG surfaces are roughly 3 matrix dofs per triangle")
    parser.add_argument("--complex-bytes", type=int, default=16)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = audit_obj(
        args.obj,
        max_leaf=args.max_leaf_triangles,
        eta=args.eta,
        rank=args.rank,
        dofs_per_triangle=args.dofs_per_triangle,
        complex_bytes=args.complex_bytes,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
