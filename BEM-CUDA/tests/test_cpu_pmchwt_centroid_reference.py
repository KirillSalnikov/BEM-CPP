#!/usr/bin/env python3
"""Regression checks for the tiny CPU PMCHWT/general-system reference."""

import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cpu_pmchwt_centroid_reference.py"
SYSTEMS = {"pmchwt", "balanced", "muller", "muller-balanced", "muller2-balanced"}


def run_reference(system: str, *, ri=("1.3116", "0.0")) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / f"{system}.json"
        proc = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--system",
                system,
                "--ri",
                *ri,
                "--json-out",
                str(out),
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        from_file = json.loads(out.read_text())
        from_stdout = json.loads(proc.stdout)
        assert from_file == from_stdout
        return from_file


def assert_selected_consistent(report: dict, system: str) -> None:
    selected = report["systems"][system]
    assert report["selected_system"] == system, report
    for key in ("condition_estimate", "min_singular_value", "max_singular_value"):
        assert report[key] == selected[key], (key, report, selected)


def main() -> int:
    pmchwt = run_reference("pmchwt")
    balanced = run_reference("balanced")
    muller2 = run_reference("muller2-balanced")
    lossy_balanced = run_reference("balanced", ri=("1.6", "0.2"))

    assert set(pmchwt["systems"]) == SYSTEMS, pmchwt
    assert pmchwt["matrix_shape"] == [2 * pmchwt["rwg"], 2 * pmchwt["rwg"]]
    assert pmchwt["centroid_rule"] is True
    assert pmchwt["singular_terms_skipped"] is True
    assert pmchwt["offdiag_antisymmetry_l2"] < 1e-12, pmchwt

    for name, system in pmchwt["systems"].items():
        assert system["matrix_shape"] == pmchwt["matrix_shape"], (name, system)
        assert system["condition_estimate"] > 0.0, (name, system)
        assert system["min_singular_value"] > 0.0, (name, system)
        assert system["max_singular_value"] >= system["min_singular_value"], (name, system)
        assert system["coupling_antisymmetry_l2"] < 1e-12, (name, system)
        if name == "muller2-balanced":
            assert system["transpose_defect_rel_l2"] > 1e-6, (name, system)
        else:
            assert system["transpose_defect_rel_l2"] < 1e-12, (name, system)

    assert_selected_consistent(pmchwt, "pmchwt")
    assert_selected_consistent(balanced, "balanced")
    assert_selected_consistent(muller2, "muller2-balanced")

    assert pmchwt["systems"]["pmchwt"]["unknown_m_scale"] == 1.0
    assert pmchwt["systems"]["pmchwt"]["row_h_scale"] == [1.0, 0.0]
    assert pmchwt["systems"]["pmchwt"]["int_op_sign"] == 1.0
    assert pmchwt["systems"]["pmchwt"]["k_identity"] == 0.0
    assert pmchwt["systems"]["pmchwt"]["n_form"] is False

    assert pmchwt["systems"]["balanced"]["unknown_m_scale"] > 1.0
    assert pmchwt["systems"]["balanced"]["row_h_scale"][0] != 1.0
    assert pmchwt["systems"]["balanced"]["int_op_sign"] == 1.0
    assert pmchwt["systems"]["balanced"]["n_form"] is False

    assert pmchwt["systems"]["muller"]["unknown_m_scale"] == 1.0
    assert pmchwt["systems"]["muller"]["int_op_sign"] == -1.0
    assert pmchwt["systems"]["muller"]["k_identity"] == 0.0
    assert pmchwt["systems"]["muller"]["n_form"] is False

    assert pmchwt["systems"]["muller-balanced"]["unknown_m_scale"] > 1.0
    assert pmchwt["systems"]["muller-balanced"]["int_op_sign"] == -1.0
    assert pmchwt["systems"]["muller-balanced"]["n_form"] is False

    n_form = pmchwt["systems"]["muller2-balanced"]
    assert n_form["unknown_m_scale"] == 1.0
    assert n_form["row_h_scale"] == [1.0, 0.0]
    assert n_form["int_op_sign"] == -1.0
    assert n_form["k_identity"] == -1.0
    assert n_form["n_form"] is True

    # Complex refractive index must be carried into the H-row scaling, not
    # silently truncated to a real impedance.
    row_h = lossy_balanced["systems"]["balanced"]["row_h_scale"]
    assert row_h[1] != 0.0, row_h

    print("cpu pmchwt centroid reference: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
