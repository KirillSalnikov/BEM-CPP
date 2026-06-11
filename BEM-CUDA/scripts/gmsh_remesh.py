#!/usr/bin/env python3
"""Remesh an OBJ/STL surface with Gmsh at an approximate target face count."""

import argparse
import math
from pathlib import Path

import gmsh
import trimesh


def target_edge_length(mesh, faces):
    if faces <= 0:
        raise ValueError("faces must be positive")
    # Equilateral triangle area = sqrt(3) / 4 * h^2.
    return math.sqrt(max(float(mesh.area), 1e-300) * 4.0 / (math.sqrt(3.0) * faces))


def remesh(src, dst, faces, angle, force_parametrizable_patches):
    src_mesh = trimesh.load_mesh(src, process=False)
    if not isinstance(src_mesh, trimesh.Trimesh):
        raise ValueError(f"{src}: not a single mesh")
    h = target_edge_length(src_mesh, faces)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.75 * h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1.25 * h)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.merge(str(src))
        gmsh.model.mesh.classifySurfaces(
            angle * math.pi / 180.0,
            True,
            force_parametrizable_patches,
            math.pi,
        )
        gmsh.model.mesh.createGeometry()
        surfaces = gmsh.model.getEntities(2)
        gmsh.model.addPhysicalGroup(2, [tag for _, tag in surfaces], 1)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")
        dst.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(dst))
    finally:
        gmsh.finalize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--faces", required=True, type=int)
    parser.add_argument("--angle", type=float, default=35.0)
    parser.add_argument("--force-parametrizable-patches", action="store_true")
    args = parser.parse_args()
    remesh(args.src, args.out, args.faces, args.angle, args.force_parametrizable_patches)


if __name__ == "__main__":
    main()
