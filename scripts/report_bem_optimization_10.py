#!/usr/bin/env python3
"""Compare the original and optimized equal-accuracy BEM benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GATES = (
    "independent_polarizations_ok",
    "residual_ok",
    "convergence_ok",
    "agreement_ok",
    "mie_ok",
    "claimable_equal_accuracy_timing",
)


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["case"]: row for row in csv.DictReader(stream)}


def wall_time(path: Path) -> float:
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("wall_s="):
            return float(line.split("=", 1)[1])
    raise ValueError(f"missing wall_s in {path}")


def mueller_values(path: Path) -> tuple[list[float], float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = [
        float(value)
        for row in data["physical"]["mueller"]
        for component in row
        for value in component
    ]
    residuals = (
        float(data["mbj"]["fmm_residual"]),
        float(data["physical"]["parallel_fmm_residual"]),
    )
    return values, max(residuals), float(data["physical"]["mueller"][0][0][0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--warm-root", type=Path, required=True)
    parser.add_argument("--shared-cache-binary-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    old = read_rows(args.old / "equal_accuracy_10.csv")
    new = read_rows(args.new / "equal_accuracy_10.csv")
    old_provenance = json.loads(
        (args.old / "provenance.json").read_text(encoding="utf-8")
    )
    new_provenance = json.loads(
        (args.new / "provenance.json").read_text(encoding="utf-8")
    )
    if old.keys() != new.keys() or len(old) != 10:
        raise SystemExit("the two benchmark case sets do not match")

    rows = []
    for case in old:
        before = float(old[case]["bem_wall_s"])
        after = float(new[case]["bem_wall_s"])
        if not all(new[case][gate].lower() == "true" for gate in GATES):
            raise SystemExit(f"optimized benchmark gate failed for {case}")
        rows.append(
            {
                "case": case,
                "old_bem_cold_wall_s": before,
                "optimized_bem_cold_wall_s": after,
                "cold_speedup": before / after,
                "optimized_max_true_residual": float(
                    new[case]["bem_max_true_residual"]
                ),
                "optimized_bem_vs_adda": float(
                    new[case]["bem_vs_adda_shape_difference"]
                ),
            }
        )

    cold_time = wall_time(args.warm_root / "ref5_cold.time.txt")
    warm_time = wall_time(args.warm_root / "ref5_warm.time.txt")
    cold_values, cold_residual, cold_m11 = mueller_values(
        args.warm_root / "ref5_cold" / "result.json"
    )
    warm_values, warm_residual, _ = mueller_values(
        args.warm_root / "ref5_warm" / "result.json"
    )
    mueller_relative_l2 = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(cold_values, warm_values))
        / sum(a * a for a in cold_values)
    )
    mueller_max_over_m11 = max(
        abs(a - b) for a, b in zip(cold_values, warm_values)
    ) / abs(cold_m11)
    warm = {
        "case": "prism_ka6_ref5",
        "cold_wall_s": cold_time,
        "shared_cache_wall_s": warm_time,
        "shared_cache_speedup": cold_time / warm_time,
        "old_to_shared_cache_speedup": (
            float(old["prism_ka6"]["bem_wall_s"]) / warm_time
        ),
        "maximum_true_residual": max(cold_residual, warm_residual),
        "cold_warm_mueller_relative_l2": mueller_relative_l2,
        "cold_warm_max_abs_over_m11_0": mueller_max_over_m11,
    }

    speedups = [row["cold_speedup"] for row in rows]
    summary = {
        "cases": rows,
        "cold_speedup_median": statistics.median(speedups),
        "cold_speedup_geometric_mean": math.exp(
            sum(math.log(value) for value in speedups) / len(speedups)
        ),
        "cold_speedup_minimum": min(speedups),
        "cold_speedup_maximum": max(speedups),
        "shared_cache_validation": warm,
        "binary_sha256": {
            "before_cold_benchmark": old_provenance["bem"]["binary_sha256"],
            "optimized_cold_benchmark": new_provenance["bem"]["binary_sha256"],
            "shared_cache_measurement": args.shared_cache_binary_sha256,
        },
        "benchmark_policy": {
            "residual_target": 1e-5,
            "independent_polarizations": True,
            "scattering_angles": 181,
            "cold_replicates_per_case": 3,
            "discretization_and_physics_gates": "inherited from equal_accuracy_10",
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bem_optimization_10.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out / "bem_optimization_10.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    labels = [
        row["case"].replace("sphere_", "sphere\n").replace("prism_", "prism\n")
        for row in rows
    ]
    positions = list(range(len(rows)))
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    width = 0.38
    axes[0].bar(
        [x - width / 2 for x in positions],
        [row["old_bem_cold_wall_s"] for row in rows],
        width, label="before", color="#8a8f98",
    )
    axes[0].bar(
        [x + width / 2 for x in positions],
        [row["optimized_bem_cold_wall_s"] for row in rows],
        width, label="optimized", color="#16865c",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("complete cold wall time, s")
    axes[0].set_title("Same strict ten-case protocol")
    axes[0].legend()

    colors = ["#16865c" if value >= 1.0 else "#d47a1f" for value in speedups]
    bars = axes[1].bar(positions, speedups, color=colors)
    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].set_ylim(0, max(speedups) * 1.2)
    axes[1].set_ylabel("BEM speedup, before / optimized")
    axes[1].set_title("Cold full-process speedup")
    for bar, value in zip(bars, speedups):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.04,
            f"{value:.2f}x",
            ha="center", va="bottom", fontsize=8,
        )

    cache_values = [
        float(old["prism_ka6"]["bem_wall_s"]), cold_time, warm_time
    ]
    cache_bars = axes[2].bar(
        range(3), cache_values, color=["#8a8f98", "#16865c", "#2a69b8"]
    )
    axes[2].set_xticks(range(3), ["before", "optimized\ncold", "optimized\ncache hit"])
    axes[2].set_ylabel("complete wall time, s")
    axes[2].set_title("Prism ka=6, ref=5")
    for bar, value in zip(cache_bars, cache_values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.2f} s",
            ha="center", va="bottom", fontsize=9,
        )

    for axis in axes[:2]:
        axis.set_xticks(positions, labels, rotation=45, ha="right")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "BEM optimization at residual 1e-5, two independent polarizations",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(args.out / "bem_optimization_10.png", dpi=180)
    plt.close(figure)

    table = "\n".join(
        "| {case} | {old_bem_cold_wall_s:.2f} | "
        "{optimized_bem_cold_wall_s:.2f} | {cold_speedup:.3f}x |".format(**row)
        for row in rows
    )
    readme = f"""# Strict BEM optimization benchmark

This compares the BEM implementation before and after mixed iterative
refinement. Both source benchmarks use the same predeclared ten cases, three
independent cold complete-process runs per case, a `1e-5` true FMM residual,
two independently solved polarizations, 181 angles, mesh-convergence gates,
ADDA agreement, and Mie checks for spheres. No unfavorable row was removed.

| case | before, s | optimized cold, s | BEM speedup |
|---|---:|---:|---:|
{table}

Median cold speedup: **{summary['cold_speedup_median']:.3f}x**. Geometric mean:
**{summary['cold_speedup_geometric_mean']:.3f}x**. The range is
**{summary['cold_speedup_minimum']:.3f}x--{summary['cold_speedup_maximum']:.3f}x**.
The `ka=10` pFFT cases barely change because this optimization targets the
direct FMM+MBJ Krylov path.

The content-addressed setup cache was tested separately on the production
`prism_ka6, ref=5` case. Complete wall time changed from **{cold_time:.2f} s**
with an empty cache to **{warm_time:.2f} s** in a new output directory with a
validated cache hit: another **{warm['shared_cache_speedup']:.3f}x**. Relative
L2 change of the full Mueller matrix was **{mueller_relative_l2:.3e}**, and the
maximum true residual was **{warm['maximum_true_residual']:.3e}**.

Relative to the pre-optimization BEM time of
`{float(old['prism_ka6']['bem_wall_s']):.2f} s`, the repeated calculation now
takes `{warm_time:.2f} s`, a total BEM speedup of
**{warm['old_to_shared_cache_speedup']:.3f}x**.

This is an implementation speedup claim for BEM. It is not a claim that BEM
beats ADDA; ADDA remains faster in all ten equal-accuracy cases.
"""
    (args.out / "README.md").write_text(readme, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
