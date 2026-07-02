#!/usr/bin/env python3
"""Voxelize an OBJ mesh into ADDA `-shape read` format.

The OBJ is normalized to equal-volume sphere radius 1, matching BEM-CUDA's
`--obj` normalization. The ADDA lattice spacing is one dipole, so coordinates
are scaled by a/d = ka*dpl/(2*pi).
"""

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh


def load_normalized_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise SystemExit(f"{path} did not load as a single mesh")
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    else:
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if mesh.volume < 0:
        mesh.invert()
    volume = abs(float(mesh.volume))
    if volume <= 0:
        raise SystemExit("mesh volume is zero; cannot equal-volume normalize")
    a_eq = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    mesh.apply_scale(1.0 / a_eq)
    return mesh


def voxel_centers(bounds: np.ndarray, pad: int = 1) -> tuple[np.ndarray, np.ndarray]:
    lo = np.floor(bounds[0]).astype(int) - pad
    hi = np.ceil(bounds[1]).astype(int) + pad
    xs = np.arange(lo[0], hi[0] + 1, dtype=int)
    ys = np.arange(lo[1], hi[1] + 1, dtype=int)
    zs = np.arange(lo[2], hi[2] + 1, dtype=int)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid.astype(float), grid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obj", required=True, type=Path)
    ap.add_argument("--ka", required=True, type=float)
    ap.add_argument("--dpl", required=True, type=float)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    args = ap.parse_args()

    mesh = load_normalized_mesh(args.obj)
    scale = args.ka * args.dpl / (2.0 * math.pi)
    mesh.apply_scale(scale)

    centers_f, centers_i = voxel_centers(mesh.bounds)
    inside_parts = []
    for start in range(0, len(centers_f), args.chunk):
        pts = centers_f[start:start + args.chunk]
        inside_parts.append(mesh.contains(pts))
    inside = np.concatenate(inside_parts)
    occ = centers_i[inside]
    if len(occ) == 0:
        raise SystemExit("voxelization produced no occupied dipoles")

    mins = occ.min(axis=0)
    occ = occ - mins + 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(f"BEM-CUDA OBJ voxelization; ka={args.ka:g}; dpl={args.dpl:g}; a_over_d={scale:.12g}\n")
        f.write(f"{len(occ)} = NAT\n")
        f.write("1 0 0 = A_1 vector\n")
        f.write("0 1 0 = A_2 vector\n")
        f.write("1.0 1.0 1.0 = lattice spacings (d_x,d_y,d_z)/d\n")
        f.write("JA  IX  IY  IZ ICOMP1 ICOMP2 ICOMP3\n")
        for ja, (ix, iy, iz) in enumerate(occ, start=1):
            f.write(f"{ja} {ix} {iy} {iz} 1 1 1\n")

    ext = occ.max(axis=0) - occ.min(axis=0) + 1
    print(f"wrote {args.out}")
    print(f"NAT={len(occ)} grid={int(ext[0])}x{int(ext[1])}x{int(ext[2])} a/d={scale:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
