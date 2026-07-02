#!/usr/bin/env python3
"""Regression checks for near-singular panel routing."""

from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from near_singular_audit import audit_obj  # noqa: E402


def write_obj(path: Path) -> None:
    # Triangles 1-2 share an edge.  Triangles 3-4 are geometrically close
    # but use distinct vertices, so they are a near-disjoint pair.
    path.write_text(
        "\n".join([
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "v 1 1 0",
            "v 2 0 0",
            "v 3 0 0",
            "v 2 1 0",
            "v 2 0 0.05",
            "v 3 0 0.05",
            "v 2 1 0.05",
            "f 1 2 3",
            "f 2 4 3",
            "f 5 6 7",
            "f 8 9 10",
            "",
        ])
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        obj = Path(td) / "near.obj"
        write_obj(obj)
        report = audit_obj(obj, threshold=0.2)

    assert report["triangles"] == 4, report
    assert report["self_panel_count"] == 4, report
    assert report["edge_adjacent_pair_count"] == 1, report
    assert report["vertex_adjacent_pair_count"] == 0, report
    assert report["duplicate_pair_count"] == 0, report
    assert report["near_disjoint_pair_count"] == 1, report
    assert report["taylor_duffy_candidate_count"] == 6, report
    assert report["worst_pair_class"] == "near_disjoint", report
    assert "near-singular quadrature" in report["recommendation"], report
    assert report["integration_policy"]["edge_adjacent"].startswith("edge Duffy"), report
    print("near singular audit: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
