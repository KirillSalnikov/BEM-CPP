#!/usr/bin/env python3
"""Project a good-quality template mesh toward a target surface.

This is useful for ADDA voxel shapes: marching-cubes meshes match the voxel
surface but can contain skinny triangles, while a previously cleaned mesh can
have much better connectivity.  The script keeps the template connectivity and
moves its normalized vertices toward nearest points on the target surface.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


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


def clean_mesh(mesh):
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def mesh_stats(mesh):
    edges = mesh.edges_unique
    if len(edges):
        counts = np.bincount(mesh.edges_unique_inverse, minlength=len(edges))
        boundary_edges = int(np.sum(counts == 1))
        nonmanifold_edges = int(np.sum(counts > 2))
    else:
        boundary_edges = 0
        nonmanifold_edges = 0
    angles = triangle_angles_deg(mesh)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "volume": float(mesh.volume),
        "angle_min_deg": float(np.min(angles)) if len(angles) else 0.0,
        "angle_p01_deg": float(np.percentile(angles, 1)) if len(angles) else 0.0,
        "angle_p05_deg": float(np.percentile(angles, 5)) if len(angles) else 0.0,
    }


def normalized_vertices(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = vertices.mean(axis=0)
    shifted = vertices - center
    scale = np.max(np.linalg.norm(shifted, axis=1))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("mesh has invalid normalization scale")
    return shifted / scale, center, scale


def project_vertices(template_vertices, target_mesh, target_vertices, alpha):
    try:
        matched, _, _ = trimesh.proximity.closest_point(target_mesh, template_vertices)
    except BaseException as exc:
        print(f"warning: closest-point surface projection failed ({exc}); using nearest vertices")
        tree = cKDTree(target_vertices)
        _, nearest = tree.query(template_vertices, k=1)
        matched = target_vertices[np.asarray(nearest, dtype=np.int64)]
    return (1.0 - alpha) * template_vertices + alpha * matched


def load_mesh(path):
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path} did not load as a triangular mesh")
    return clean_mesh(mesh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True,
                        help="Quality mesh whose connectivity is preserved")
    parser.add_argument("--target", required=True,
                        help="Target surface mesh, e.g. marching-cubes ADDA surface")
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=0.75,
                        help="Projection strength: 0 keeps template, 1 uses nearest target vertices")
    parser.add_argument("--stats-json", default=None)
    parser.add_argument("--min-angle-deg", type=float, default=10.0,
                        help="Reject the mesh if the minimum triangle angle is lower")
    parser.add_argument("--min-angle-p05-deg", type=float, default=20.0,
                        help="Reject the mesh if the 5th-percentile triangle angle is lower")
    parser.add_argument("--allow-poor-quality", action="store_true",
                        help="Write the mesh even if quality gates fail")
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be between 0 and 1")

    template = load_mesh(Path(args.template))
    target = load_mesh(Path(args.target))
    template_v, _, _ = normalized_vertices(template)
    target_v, _, _ = normalized_vertices(target)
    target_norm = trimesh.Trimesh(
        vertices=target_v,
        faces=np.asarray(target.faces, dtype=np.int64),
        process=False,
    )
    projected_v = project_vertices(template_v, target_norm, target_v, args.alpha)

    out_mesh = clean_mesh(trimesh.Trimesh(
        vertices=projected_v,
        faces=np.asarray(template.faces, dtype=np.int64),
        process=False,
    ))

    stats = mesh_stats(out_mesh)
    stats.update({
        "template": str(Path(args.template)),
        "target": str(Path(args.target)),
        "out": str(Path(args.out)),
        "alpha": float(args.alpha),
        "min_angle_gate_deg": float(args.min_angle_deg),
        "min_angle_p05_gate_deg": float(args.min_angle_p05_deg),
    })
    quality_ok = (
        stats["watertight"]
        and stats["boundary_edges"] == 0
        and stats["nonmanifold_edges"] == 0
        and stats["angle_min_deg"] >= args.min_angle_deg
        and stats["angle_p05_deg"] >= args.min_angle_p05_deg
    )
    stats["quality_gate"] = bool(quality_ok)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if not quality_ok and not args.allow_poor_quality:
        raise SystemExit("mesh quality gate failed; use --allow-poor-quality to write it anyway")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_mesh.export(out_path)
    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
