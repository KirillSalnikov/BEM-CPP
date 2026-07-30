#!/usr/bin/env python3
"""Generate one deterministic symmetry-free particle for BEM and ADDA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


TARGET_VOLUME = 4.0 * math.pi / 3.0


def raw_vertices() -> np.ndarray:
    angles = np.array([0.00, 0.74, 1.69, 2.55, 3.36, 4.31, 5.37])
    bottom_radii = np.array([1.00, 0.76, 1.13, 0.83, 1.07, 0.71, 0.94])
    top_radii = np.array([0.81, 1.02, 0.73, 0.91, 0.68, 0.88, 0.77])

    bottom = np.column_stack(
        (
            bottom_radii * np.cos(angles),
            bottom_radii * np.sin(angles),
            np.full(angles.size, -0.62),
        )
    )
    top_angles = angles + 0.19
    top = np.column_stack(
        (
            0.18 + top_radii * np.cos(top_angles),
            -0.13 + top_radii * np.sin(top_angles),
            np.full(angles.size, 0.79),
        )
    )
    return np.vstack((bottom, top))


def normalized_hull() -> tuple[np.ndarray, ConvexHull]:
    vertices = raw_vertices()
    vertices -= vertices.mean(axis=0)
    hull = ConvexHull(vertices)
    vertices *= (TARGET_VOLUME / hull.volume) ** (1.0 / 3.0)
    hull = ConvexHull(vertices)
    return vertices, hull


def oriented_faces(vertices: np.ndarray, hull: ConvexHull) -> np.ndarray:
    faces = hull.simplices.copy()
    for index, (a, b, c) in enumerate(faces):
        normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
        outward = hull.equations[index, :3]
        if np.dot(normal, outward) < 0.0:
            faces[index, 1], faces[index, 2] = faces[index, 2], faces[index, 1]
    return faces


def signed_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write(
            "# Deterministic oblique irregular heptagonal particle; "
            "volume-equivalent radius = 1\n"
        )
        for x, y, z in vertices:
            stream.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for a, b, c in faces:
            stream.write(f"f {a + 1} {b + 1} {c + 1}\n")


def write_adda_geometry(
    path: Path,
    vertices: np.ndarray,
    ka: float,
    dpl: float,
) -> dict[str, float | int | list[int]]:
    spacing = 2.0 * math.pi / (ka * dpl)
    hull = ConvexHull(vertices)
    normals = hull.equations[:, :3]
    offsets = hull.equations[:, 3]
    target_occupied = TARGET_VOLUME / spacing**3
    occupied = int(round(target_occupied))
    lower = np.floor(vertices.min(axis=0) / spacing).astype(int) - 2
    upper = np.ceil(vertices.max(axis=0) / spacing).astype(int) + 2
    x_values = np.arange(lower[0], upper[0] + 1, dtype=np.int32)
    y_values = np.arange(lower[1], upper[1] + 1, dtype=np.int32)
    xy = np.stack(
        np.meshgrid(x_values, y_values, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    scores = []
    for z_index in range(int(lower[2]), int(upper[2]) + 1):
        points = np.empty((xy.shape[0], 3), dtype=np.float64)
        points[:, :2] = xy
        points[:, 2] = z_index
        points *= spacing
        scores.append(
            np.max(points @ normals.T + offsets[None, :], axis=1)
        )
    all_scores = np.concatenate(scores)
    if occupied >= all_scores.size:
        raise RuntimeError("ADDA voxel candidate box is too small")
    selected_flat = np.argpartition(all_scores, occupied - 1)[:occupied]
    selected_mask = np.zeros(all_scores.size, dtype=bool)
    selected_mask[selected_flat] = True
    level_set_offset = float(np.max(all_scores[selected_flat]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", buffering=4 * 1024 * 1024) as stream:
        stream.write(
            "# Oblique irregular heptagonal particle; no rotational or mirror symmetry\n"
        )
        stream.write(f"# target ka={ka:.17g}, target dpl={dpl:.17g}\n")
        xy_count = xy.shape[0]
        for z_offset, z_index in enumerate(
            range(int(lower[2]), int(upper[2]) + 1)
        ):
            first = z_offset * xy_count
            selected = xy[selected_mask[first:first + xy_count]]
            stream.writelines(
                f"{int(x)} {int(y)} {z_index}\n" for x, y in selected
            )

    realized_volume = occupied * spacing**3
    realized_radius = (3.0 * realized_volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    realized_ka_if_dpl_fixed = (
        2.0
        * math.pi
        / dpl
        * (3.0 * occupied / (4.0 * math.pi)) ** (1.0 / 3.0)
    )
    return {
        "ka": ka,
        "requested_dpl": dpl,
        "normalized_lattice_spacing": spacing,
        "target_occupied_voxels": target_occupied,
        "occupied_voxels": occupied,
        "voxelization_level_set_offset": level_set_offset,
        "grid_lower": lower.tolist(),
        "grid_upper": upper.tolist(),
        "voxelized_equivalent_radius_before_adda_volume_correction": realized_radius,
        "voxelized_volume_relative_error": realized_volume / TARGET_VOLUME - 1.0,
        "realized_ka_if_dpl_fixed": realized_ka_if_dpl_fixed,
        "realized_ka_relative_error_if_dpl_fixed":
            realized_ka_if_dpl_fixed / ka - 1.0,
        "estimated_dpl_with_exact_eq_rad": dpl * realized_radius,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--geom", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--ka", type=float)
    parser.add_argument("--dpl", type=float, default=15.0)
    args = parser.parse_args()
    if (args.geom is not None and args.ka is None):
        parser.error("--geom requires --ka")
    return args


def main() -> None:
    args = parse_args()
    vertices, hull = normalized_hull()
    faces = oriented_faces(vertices, hull)
    volume = signed_mesh_volume(vertices, faces)
    if volume <= 0.0 or abs(volume / TARGET_VOLUME - 1.0) > 1.0e-12:
        raise RuntimeError(f"invalid oriented mesh volume: {volume}")
    write_obj(args.obj, vertices, faces)

    metadata: dict[str, object] = {
        "description": "oblique irregular heptagonal convex particle",
        "rotational_symmetry_order": 1,
        "mirror_symmetry": False,
        "equivalent_radius": 1.0,
        "mesh_volume": volume,
        "vertices": int(vertices.shape[0]),
        "triangles": int(faces.shape[0]),
        "obj": str(args.obj.resolve()),
    }
    if args.geom is not None:
        metadata["adda"] = write_adda_geometry(
            args.geom, vertices, args.ka, args.dpl
        )
        metadata["geom"] = str(args.geom.resolve())
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
            encoding="ascii",
        )
    print(json.dumps(metadata, ensure_ascii=True))


if __name__ == "__main__":
    main()
