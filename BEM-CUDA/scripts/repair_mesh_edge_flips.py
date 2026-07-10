#!/usr/bin/env python3
"""Improve triangle quality by flipping internal edges on a fixed surface mesh."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh


def triangle_angles(vertices, faces):
    tri = vertices[np.asarray(faces, dtype=np.int64)]
    if len(tri) == 0:
        return np.empty((0, 3), dtype=np.float64)
    a = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
    b = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    c = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
    eps = 1e-300
    cos_a = (b * b + c * c - a * a) / np.maximum(2.0 * b * c, eps)
    cos_b = (a * a + c * c - b * b) / np.maximum(2.0 * a * c, eps)
    cos_c = (a * a + b * b - c * c) / np.maximum(2.0 * a * b, eps)
    return np.degrees(np.arccos(np.clip(np.column_stack((cos_a, cos_b, cos_c)), -1.0, 1.0)))


def face_normal(vertices, face):
    p = vertices[np.asarray(face, dtype=np.int64)]
    n = np.cross(p[1] - p[0], p[2] - p[0])
    norm = np.linalg.norm(n)
    return n / norm if norm > 0.0 else n


def oriented(face, vertices, ref_normal):
    face = list(face)
    if np.dot(face_normal(vertices, face), ref_normal) < 0.0:
        face[1], face[2] = face[2], face[1]
    return face


def directed_edge_sign(face, a, b):
    for i in range(3):
        u = face[i]
        v = face[(i + 1) % 3]
        if u == a and v == b:
            return 1
        if u == b and v == a:
            return -1
    return 0


def edge_faces(faces):
    mapping = defaultdict(list)
    for idx, face in enumerate(faces):
        for i in range(3):
            a = int(face[i])
            b = int(face[(i + 1) % 3])
            mapping[tuple(sorted((a, b)))].append(idx)
    return mapping


def try_flip(vertices, faces, edge, adjacent, existing_edges):
    f0, f1 = adjacent
    face0 = list(map(int, faces[f0]))
    face1 = list(map(int, faces[f1]))
    a, b = edge
    if directed_edge_sign(face0, a, b) < 0:
        a, b = b, a
    cands0 = [v for v in face0 if v not in edge]
    cands1 = [v for v in face1 if v not in edge]
    if len(cands0) != 1 or len(cands1) != 1:
        return None
    c, d = cands0[0], cands1[0]
    if c == d:
        return None
    # The replacement diagonal must not already exist elsewhere. Otherwise
    # the flip creates duplicate/non-manifold connectivity.
    if tuple(sorted((c, d))) in existing_edges:
        return None

    before = triangle_angles(vertices, [face0, face1]).min()
    ref0 = face_normal(vertices, face0)
    ref1 = face_normal(vertices, face1)
    # With face0 containing a->b, manifold winding makes face1 contain b->a.
    # Preserve the oriented outer cycle b->c->a->d->b after replacing a-b
    # with c-d.
    new0 = [c, a, d]
    new1 = [c, d, b]
    if directed_edge_sign(new0, c, d) * directed_edge_sign(new1, c, d) != -1:
        return None
    after_angles = triangle_angles(vertices, [new0, new1])
    after = after_angles.min()
    if not np.isfinite(after) or after <= before + 1.0e-9:
        return None
    # Reject flips that create extremely inverted local normals.
    if np.dot(face_normal(vertices, new0), ref0) < 0.25:
        return None
    if np.dot(face_normal(vertices, new1), ref1) < 0.25:
        return None
    return after - before, new0, new1, before, after


def repair(mesh, target_angle, max_passes):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    total_flips = 0
    for _ in range(max_passes):
        angles = triangle_angles(vertices, faces)
        bad_faces = np.where(angles.min(axis=1) < target_angle)[0]
        if len(bad_faces) == 0:
            break
        mapping = edge_faces(faces)
        best = None
        bad_set = set(map(int, bad_faces))
        candidate_edges = set()
        for fi in bad_set:
            face = faces[fi]
            for i in range(3):
                candidate_edges.add(tuple(sorted((int(face[i]), int(face[(i + 1) % 3])))))
        for edge in candidate_edges:
            adjacent = mapping.get(edge, [])
            if len(adjacent) != 2:
                continue
            result = try_flip(vertices, faces, edge, adjacent, mapping)
            if result is None:
                continue
            gain, new0, new1, before, after = result
            if best is None or (after, gain) > (best[0], best[1]):
                best = (after, gain, edge, adjacent, new0, new1, before)
        if best is None:
            break
        _, _, _, adjacent, new0, new1, _ = best
        faces[adjacent[0]] = new0
        faces[adjacent[1]] = new1
        total_flips += 1
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False), total_flips


def stats(mesh):
    angles = triangle_angles(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": int(np.sum(counts == 1)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "angle_min": float(np.min(angles)),
        "angle_p1": float(np.percentile(angles, 1)),
        "angle_p5": float(np.percentile(angles, 5)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-angle", type=float, default=20.0)
    parser.add_argument("--max-passes", type=int, default=2000)
    args = parser.parse_args()

    mesh = trimesh.load_mesh(args.src, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{args.src}: not a triangular mesh")
    repaired, flips = repair(mesh, args.target_angle, args.max_passes)
    report = stats(repaired)
    if not report["watertight"] or not report["winding_consistent"] or report["nonmanifold_edges"]:
        raise RuntimeError("edge flips violated closed manifold topology: {}".format(report))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    repaired.export(args.out)
    print({"flips": flips, **report})


if __name__ == "__main__":
    main()
