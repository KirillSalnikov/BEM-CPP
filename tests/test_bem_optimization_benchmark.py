#!/usr/bin/env python3
"""Audit the published strict before/after BEM optimization benchmark."""

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "bem_optimization_10_20260804"


def main() -> int:
    data = json.loads(
        (BENCHMARK / "bem_optimization_10.json").read_text(encoding="utf-8")
    )
    cases = data["cases"]
    assert len(cases) == 10
    assert len({case["case"] for case in cases}) == 10
    assert all(case["optimized_max_true_residual"] <= 1e-5 for case in cases)
    assert 1.61 < data["cold_speedup_median"] < 1.63
    assert 2.32 < data["cold_speedup_maximum"] < 2.33

    cache = data["shared_cache_validation"]
    assert cache["shared_cache_wall_s"] < cache["cold_wall_s"]
    assert cache["old_to_shared_cache_speedup"] > 3.6
    assert cache["maximum_true_residual"] <= 1e-5
    assert cache["cold_warm_mueller_relative_l2"] < 1e-8
    assert len(data["binary_sha256"]) == 3
    assert all(len(value) == 64 for value in data["binary_sha256"].values())

    assert (BENCHMARK / "bem_optimization_10.csv").is_file()
    assert (BENCHMARK / "bem_optimization_10.png").stat().st_size > 100_000
    assert "ADDA remains faster" in (BENCHMARK / "README.md").read_text(
        encoding="utf-8"
    )
    print("strict BEM optimization benchmark: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
