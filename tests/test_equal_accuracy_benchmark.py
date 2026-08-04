#!/usr/bin/env python3
"""Keep the published ten-case benchmark complete and unambiguous."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "equal_accuracy_10_20260804"
EXPECTED = [
    f"{shape}_ka{ka}"
    for shape in ("sphere", "prism")
    for ka in (2, 4, 6, 8, 10)
]


def main() -> None:
    rows = json.loads((BENCHMARK / "equal_accuracy_10.json").read_text())
    provenance = json.loads((BENCHMARK / "provenance.json").read_text())
    report = (BENCHMARK / "README.md").read_text()

    assert [row["case"] for row in rows] == EXPECTED
    assert len(rows) == 10
    assert all(row["claimable_equal_accuracy_timing"] for row in rows)
    assert all(row["independent_polarizations_ok"] for row in rows)
    assert all(row["residual_ok"] for row in rows)
    assert all(row["convergence_ok"] for row in rows)
    assert all(row["agreement_ok"] for row in rows)
    assert all(row["mie_ok"] for row in rows)
    assert all(len(row["bem_wall_samples_s"]) == 3 for row in rows)
    assert all(len(row["adda_wall_samples_s"]) == 3 for row in rows)
    assert all(row["bem_max_true_residual"] <= 1.05e-5 for row in rows)
    assert all(row["adda_max_recalculated_residual"] <= 1.05e-5 for row in rows)
    assert all(row["adda_wall_over_bem_wall"] < 1.0 for row in rows)

    policy = provenance["benchmark_policy"]
    assert policy["cases"] == EXPECTED
    assert policy["residual_target"] == 1e-5
    assert policy["independent_polarizations"] is True
    assert policy["scattering_angles"] == 181
    assert policy["replicates"] == 3
    assert provenance["adda"]["source"]["commit"].startswith("8f550a7")
    assert provenance["adda"]["source"]["dirty"] is False

    assert "A value below one is a BEM slowdown, not an acceleration." in report
    assert "All ten cases were declared before execution" in report
    assert "locally modified FP32 experimental build" in report
    print("equal-accuracy ten-case benchmark: ok")


if __name__ == "__main__":
    main()
