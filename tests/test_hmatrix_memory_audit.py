#!/usr/bin/env python3
"""Regression checks for H-matrix/ACA memory feasibility audit."""

from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hmatrix_memory_audit import audit_obj  # noqa: E402


def write_two_panel_clouds(path: Path) -> None:
    # Four triangles near x=0 and four near x=10.  With leaf size 2 the two
    # groups produce far admissible cluster pairs and dense local leaf blocks.
    verts = []
    faces = []
    for base_x in (0.0, 10.0):
        for iy in (0.0, 1.0):
            for iz in (0.0, 1.0):
                i0 = len(verts) + 1
                verts.extend([
                    (base_x, iy, iz),
                    (base_x + 0.4, iy, iz),
                    (base_x, iy + 0.3, iz + 0.2),
                ])
                faces.append((i0, i0 + 1, i0 + 2))
    lines = [f"v {x} {y} {z}" for x, y, z in verts]
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        obj = Path(td) / "clouds.obj"
        write_two_panel_clouds(obj)
        report = audit_obj(
            obj,
            max_leaf=2,
            eta=1.0,
            rank=1,
            dofs_per_triangle=1.0,
            complex_bytes=16,
        )

    assert report["triangles"] == 8, report
    assert report["estimated_total_dofs"] == 8, report
    assert report["clusters"] >= 7, report
    assert report["dense_block_count"] > 0, report
    assert report["admissible_block_count"] > 0, report
    assert report["hmatrix_estimated_bytes"] < report["dense_full_bytes"], report
    assert report["compression_vs_dense"] > 1.0, report
    assert "H-matrix/ACA" in report["recommendation"], report
    print("hmatrix memory audit: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
