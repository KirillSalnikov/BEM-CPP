#!/usr/bin/env python3
"""Build BEM surface meshes directly from an ADDA `-shape read` voxel file."""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from trimesh import smoothing
from scipy import ndimage
from skimage import measure


def load_shape_points(path):
    points = []
    with open(path, "r") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) >= 7 and all(x.lstrip("-").isdigit() for x in fields[:4]):
                points.append((int(fields[1]), int(fields[2]), int(fields[3])))
    if not points:
        raise ValueError(f"no ADDA dipoles found in {path}")
    return np.asarray(points, dtype=np.int32)


def mesh_stats(mesh):
    edges = mesh.edges_unique
    if len(edges):
        counts = np.bincount(mesh.edges_unique_inverse, minlength=len(edges))
        boundary = int(np.sum(counts == 1))
        nonmanifold = int(np.sum(counts > 2))
    else:
        boundary = nonmanifold = 0
    angles = triangle_angles_deg(mesh)
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "volume": float(mesh.volume),
        "angle_min_deg": float(np.min(angles)) if len(angles) else 0.0,
        "angle_p01_deg": float(np.percentile(angles, 1)) if len(angles) else 0.0,
        "angle_p05_deg": float(np.percentile(angles, 5)) if len(angles) else 0.0,
    }


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


def decimate(mesh, face_count, aggression):
    if face_count <= 0 or len(mesh.faces) <= face_count:
        return clean_mesh(mesh.copy())
    return clean_mesh(mesh.simplify_quadric_decimation(
        face_count=face_count, aggression=aggression))


def smooth_taubin(mesh, iterations, lamb, nu):
    if iterations <= 0:
        return mesh
    out = mesh.copy()
    smoothing.filter_taubin(out, lamb=lamb, nu=nu, iterations=iterations)
    return clean_mesh(out)


def occupancy_grid(points, pad):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    dims = maxs - mins + 1
    grid = np.zeros(tuple(dims + 2 * pad), dtype=np.float32)
    idx = points - mins + pad
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return grid


def marching_mesh(grid, sigma, level):
    field = ndimage.gaussian_filter(grid, sigma=sigma) if sigma > 0 else grid
    vertices, faces, _, _ = measure.marching_cubes(
        field, level=level, spacing=(1.0, 1.0, 1.0), gradient_direction="descent")
    vertices -= vertices.mean(axis=0)
    return clean_mesh(trimesh.Trimesh(vertices, faces, process=True))


def cubical_mesh(points):
    occ = set(map(tuple, points))
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    vertices = []
    vertex_id = {}
    faces = []

    def add_vertex(coord):
        key = tuple(coord)
        if key not in vertex_id:
            vertex_id[key] = len(vertices)
            vertices.append(key)
        return vertex_id[key]

    face_defs = [
        ((1, 0, 0), [(0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)]),
        ((-1, 0, 0), [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5)]),
        ((0, 1, 0), [(-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5)]),
        ((0, -1, 0), [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)]),
        ((0, 0, 1), [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]),
        ((0, 0, -1), [(-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5)]),
    ]

    for point in points:
        base = point.astype(np.float64) - center
        p = tuple(point)
        for normal, corners in face_defs:
            neighbor = (p[0] + normal[0], p[1] + normal[1], p[2] + normal[2])
            if neighbor in occ:
                continue
            ids = [add_vertex(base + np.asarray(corner)) for corner in corners]
            faces.append([ids[0], ids[1], ids[2]])
            faces.append([ids[0], ids[2], ids[3]])

    faces = np.asarray(faces, dtype=np.int64)
    keys = np.sort(faces, axis=1)
    _, keep = np.unique(keys, axis=0, return_index=True)
    mesh = trimesh.Trimesh(np.asarray(vertices), faces[np.sort(keep)], process=False)
    return clean_mesh(mesh)


def write_mesh(mesh, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    stats = mesh_stats(mesh)
    print(f"{path} V={stats['vertices']} F={stats['faces']} "
          f"watertight={stats['watertight']} boundary={stats['boundary_edges']} "
          f"nonmanifold={stats['nonmanifold_edges']} volume={stats['volume']:.8g} "
          f"angle_min={stats['angle_min_deg']:.2f} "
          f"angle_p1={stats['angle_p01_deg']:.2f} "
          f"angle_p5={stats['angle_p05_deg']:.2f}")


def tag(value):
    return f"{value:g}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--faces", type=int, default=1200)
    parser.add_argument("--aggression", type=int, default=8)
    parser.add_argument("--pad", type=int, default=3)
    parser.add_argument("--sigmas", nargs="*", type=float, default=[0.0, 0.5, 0.65])
    parser.add_argument("--levels", nargs="*", type=float, default=[0.42, 0.5])
    parser.add_argument("--smooth-iters", nargs="*", type=int, default=[0],
                        help="Taubin smoothing iterations before decimation")
    parser.add_argument("--smooth-lambda", type=float, default=0.5)
    parser.add_argument("--smooth-nu", type=float, default=0.53)
    parser.add_argument("--cubical", action="store_true")
    parser.add_argument("--only-cubical", action="store_true",
                        help="Skip marching-cubes variants and write only cubical meshes")
    args = parser.parse_args()

    points = load_shape_points(args.shape)
    out_dir = Path(args.out_dir)
    grid = occupancy_grid(points, args.pad)

    if not args.only_cubical:
        for sigma in args.sigmas:
            for level in args.levels:
                base = marching_mesh(grid, sigma, level)
                for smooth_iters in args.smooth_iters:
                    mesh = smooth_taubin(base, smooth_iters, args.smooth_lambda, args.smooth_nu)
                    mesh = decimate(mesh, args.faces, args.aggression)
                    smooth_tag = "" if smooth_iters <= 0 else f"_t{smooth_iters}"
                    name = f"adda_mc_s{tag(sigma)}_l{tag(level)}{smooth_tag}_f{args.faces}.obj"
                    write_mesh(mesh, out_dir / name)

    if args.cubical or args.only_cubical:
        mesh = cubical_mesh(points)
        write_mesh(mesh, out_dir / "adda_cubical_raw.obj")
        mesh = decimate(mesh, args.faces, args.aggression)
        write_mesh(mesh, out_dir / f"adda_cubical_f{args.faces}_ag{args.aggression}.obj")


if __name__ == "__main__":
    main()
