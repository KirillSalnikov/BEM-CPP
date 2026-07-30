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


def lattice_axes(bounds: np.ndarray, pad: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = np.floor(bounds[0]).astype(int) - pad
    hi = np.ceil(bounds[1]).astype(int) + pad
    xs = np.arange(lo[0], hi[0] + 1, dtype=int)
    ys = np.arange(lo[1], hi[1] + 1, dtype=int)
    zs = np.arange(lo[2], hi[2] + 1, dtype=int)
    return xs, ys, zs


def occupied_centers(
    mesh: trimesh.Trimesh,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    chunk: int,
) -> np.ndarray:
    """Return occupied integer centers without materializing the full 3-D grid."""
    xs, ys, zs = axes
    ny, nz = len(ys), len(zs)
    yz = ny * nz
    total = len(xs) * yz
    occupied_parts = []

    for start in range(0, total, chunk):
        flat = np.arange(start, min(start + chunk, total), dtype=np.int64)
        ix = flat // yz
        rem = flat % yz
        iy = rem // nz
        iz = rem % nz
        centers = np.column_stack((xs[ix], ys[iy], zs[iz]))
        inside = mesh.contains(centers.astype(float, copy=False))
        if np.any(inside):
            occupied_parts.append(centers[inside])

    if not occupied_parts:
        return np.empty((0, 3), dtype=np.int64)
    return np.concatenate(occupied_parts, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obj", required=True, type=Path)
    ap.add_argument("--ka", required=True, type=float)
    ap.add_argument("--dpl", required=True, type=float)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--chunk",
        type=int,
        default=20_000,
        help="points per mesh.contains call (default: 20000; keeps memory bounded)",
    )
    args = ap.parse_args()
    if args.chunk <= 0:
        ap.error("--chunk must be positive")

    mesh = load_normalized_mesh(args.obj)
    scale = args.ka * args.dpl / (2.0 * math.pi)
    mesh.apply_scale(scale)

    occ = occupied_centers(mesh, lattice_axes(mesh.bounds), args.chunk)
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
