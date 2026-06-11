#!/usr/bin/env python3
"""Print basic quality metrics for OBJ/STL surface meshes."""

import argparse
from pathlib import Path

import numpy as np
import trimesh


def triangle_angles_deg(mesh):
    tri = np.asarray(mesh.triangles, dtype=np.float64)
    if len(tri) == 0:
        return np.empty(0, dtype=np.float64)
    a = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
    b = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    c = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
    eps = 1e-300
    cos_a = (b * b + c * c - a * a) / np.maximum(2.0 * b * c, eps)
    cos_b = (a * a + c * c - b * b) / np.maximum(2.0 * a * c, eps)
    cos_c = (a * a + b * b - c * c) / np.maximum(2.0 * a * b, eps)
    cosines = np.clip(np.column_stack((cos_a, cos_b, cos_c)), -1.0, 1.0)
    return np.degrees(np.arccos(cosines)).reshape(-1)


def edge_counts(mesh):
    if len(mesh.edges_unique) == 0:
        return 0, 0
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    return int(np.sum(counts == 1)), int(np.sum(counts > 2))


def quality(path):
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path}: not a single mesh")
    angles = triangle_angles_deg(mesh)
    boundary, nonmanifold = edge_counts(mesh)
    return {
        "path": str(path),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "boundary": boundary,
        "nonmanifold": nonmanifold,
        "min": float(np.min(angles)) if len(angles) else 0.0,
        "p1": float(np.percentile(angles, 1)) if len(angles) else 0.0,
        "p5": float(np.percentile(angles, 5)) if len(angles) else 0.0,
        "median": float(np.median(angles)) if len(angles) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", nargs="+", type=Path)
    args = parser.parse_args()

    print("mesh,vertices,faces,watertight,boundary,nonmanifold,angle_min,angle_p1,angle_p5,angle_median")
    for path in args.mesh:
        row = quality(path)
        print(
            f"{row['path']},{row['vertices']},{row['faces']},{int(row['watertight'])},"
            f"{row['boundary']},{row['nonmanifold']},"
            f"{row['min']:.6g},{row['p1']:.6g},{row['p5']:.6g},{row['median']:.6g}"
        )


if __name__ == "__main__":
    main()
