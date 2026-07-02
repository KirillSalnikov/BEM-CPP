#!/usr/bin/env python3
"""Render BEM-CUDA equivalent surface currents from a legacy VTK export."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_legacy_vtk(path: Path):
    lines = path.read_text().splitlines()
    points = None
    tris = None
    scalars = {}
    vectors = {}
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if not parts:
            i += 1
            continue
        key = parts[0].upper()
        if key == "POINTS":
            n = int(parts[1])
            vals = []
            i += 1
            while len(vals) < 3 * n:
                vals.extend(float(x) for x in lines[i].split())
                i += 1
            points = np.asarray(vals, dtype=float).reshape(n, 3)
            continue
        if key == "POLYGONS":
            n = int(parts[1])
            tri = []
            i += 1
            for _ in range(n):
                row = [int(x) for x in lines[i].split()]
                if row[0] != 3:
                    raise ValueError("only triangular VTK polygons are supported")
                tri.append(row[1:4])
                i += 1
            tris = np.asarray(tri, dtype=int)
            continue
        if key == "SCALARS":
            name = parts[1]
            if i + 1 < len(lines) and lines[i + 1].upper().startswith("LOOKUP_TABLE"):
                i += 2
            else:
                i += 1
            vals = []
            n = len(tris)
            while len(vals) < n:
                vals.extend(float(x) for x in lines[i].split())
                i += 1
            scalars[name] = np.asarray(vals[:n], dtype=float)
            continue
        if key == "VECTORS":
            name = parts[1]
            vals = []
            n = len(tris)
            i += 1
            while len(vals) < 3 * n:
                vals.extend(float(x) for x in lines[i].split())
                i += 1
            vectors[name] = np.asarray(vals[: 3 * n], dtype=float).reshape(n, 3)
            continue
        i += 1
    if points is None or tris is None:
        raise ValueError(f"{path} is not a supported legacy POLYDATA VTK file")
    return points, tris, scalars, vectors


def set_axes_equal(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * np.max(maxs - mins)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vtk", type=Path)
    ap.add_argument("--out", type=Path, default=Path("currents_3d.png"))
    ap.add_argument("--field", default="J_par", choices=["J_par", "M_par", "J_perp", "M_perp"])
    ap.add_argument("--arrows", type=int, default=600)
    ap.add_argument("--arrow-mode", choices=["strong", "uniform"], default="strong")
    ap.add_argument("--elev", type=float, default=24.0)
    ap.add_argument("--azim", type=float, default=-52.0)
    ap.add_argument("--title", default=None)
    ap.add_argument("--incident", action="store_true", help="draw incident k and E arrows for the default single-orientation run")
    ap.add_argument("--incident-e", choices=["x", "y"], default="y")
    args = ap.parse_args()

    pts, tris, scalars, vectors = read_legacy_vtk(args.vtk)
    mag = scalars[f"{args.field}_abs"]
    vec = vectors[f"{args.field}_vector"]
    centers = pts[tris].mean(axis=1)
    faces = pts[tris]

    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(vmin=np.percentile(mag, 2), vmax=np.percentile(mag, 98))
    colors = plt.cm.viridis(norm(mag))
    coll = Poly3DCollection(faces, facecolors=colors, edgecolors=(0, 0, 0, 0.10), linewidths=0.12)
    ax.add_collection3d(coll)

    if args.arrows > 0 and len(tris) > 0:
        if args.arrow_mode == "uniform":
            idx = np.linspace(0, len(tris) - 1, min(args.arrows, len(tris)), dtype=int)
        else:
            order = np.argsort(mag)[::-1]
            step = max(1, len(order) // args.arrows)
            idx = order[::step][: args.arrows]
        q = vec[idx]
        qn = np.linalg.norm(q, axis=1)
        keep = qn > 0
        q = q[keep] / qn[keep, None]
        c = centers[idx][keep]
        scale = 0.10 * np.max(pts.max(axis=0) - pts.min(axis=0))
        ax.quiver(c[:, 0], c[:, 1], c[:, 2], q[:, 0], q[:, 1], q[:, 2],
                  length=scale, normalize=False, color="black", linewidth=0.75, alpha=0.9)

    axis_pts = pts
    if args.incident:
        extent = np.max(pts.max(axis=0) - pts.min(axis=0))
        origin = np.array([pts[:, 0].min() - 0.35 * extent,
                           pts[:, 1].min() - 0.18 * extent,
                           pts[:, 2].min() - 0.05 * extent])
        k0 = origin
        e0 = origin + np.array([0.0, 0.0, 0.18 * extent])
        ax.quiver(k0[0], k0[1], k0[2], 0.0, 0.0, 1.0,
                  length=0.46 * extent, normalize=True, color="#d62728",
                  linewidth=2.2, arrow_length_ratio=0.18)
        ax.text(k0[0], k0[1], k0[2] + 0.52 * extent, r"$\mathbf{k}_{inc}$",
                color="#d62728", fontsize=11)
        e_dir = np.array([1.0, 0.0, 0.0]) if args.incident_e == "x" else np.array([0.0, 1.0, 0.0])
        ax.quiver(e0[0], e0[1], e0[2], e_dir[0], e_dir[1], e_dir[2],
                  length=0.46 * extent, normalize=True, color="#1f77b4",
                  linewidth=2.2, arrow_length_ratio=0.18)
        label_pos = e0 + 0.50 * extent * e_dir
        ax.text(label_pos[0], label_pos[1], label_pos[2], r"$\mathbf{E}_{inc}$",
                color="#1f77b4", fontsize=11)
        axis_pts = np.vstack([
            pts,
            k0,
            k0 + np.array([0.0, 0.0, 0.55 * extent]),
            e0,
            e0 + 0.55 * extent * e_dir,
        ])

    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array(mag)
    base_label = "J" if args.field.startswith("J_") else "M"
    fig.colorbar(sm, ax=ax, shrink=0.48, fraction=0.030, aspect=18,
                 pad=0.012, label=f"|{base_label}|")
    ax.set_proj_type("ortho")
    ax.view_init(elev=args.elev, azim=args.azim)
    set_axes_equal(ax, axis_pts)
    ax.set_axis_off()
    if args.title:
        ax.set_title(args.title, pad=8)
    fig.savefig(args.out, dpi=220, bbox_inches="tight", pad_inches=0.015)
    print(args.out)


if __name__ == "__main__":
    main()
