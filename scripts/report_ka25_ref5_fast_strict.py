#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_wall_seconds(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", text
    )
    if not match:
        match = re.search(r"WALL(?:_S)?=([0-9.]+)", text)
    if not match:
        raise ValueError(f"wall time not found in {path}")
    fields = [float(value) for value in match.group(1).split(":")]
    seconds = 0.0
    for value in fields:
        seconds = 60.0 * seconds + value
    return seconds


def weighted_relative_l2(
    reference: np.ndarray, candidate: np.ndarray, weights: np.ndarray
) -> float:
    extra_axes = reference.ndim - 1
    shaped_weights = weights.reshape((1,) * extra_axes + (-1,))
    numerator = np.sum(np.square(reference - candidate) * shaped_weights)
    denominator = np.sum(np.square(reference) * shaped_weights)
    return math.sqrt(numerator / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/goal_ka25_ref5"),
    )
    args = parser.parse_args()
    run_dir = args.run_dir

    legacy = json.loads(
        (run_dir / "baseline_fp64.json").read_text(encoding="utf-8")
    )
    result_dir = run_dir / "mirror_hdiv"
    gpu_result_dir = run_dir / "gpu_assembly"
    optimized_json = (
        gpu_result_dir / "new_farfield.json"
        if (gpu_result_dir / "new_farfield.json").exists()
        else result_dir / "optimized_batch3_inner004_mirror.json"
    )
    optimized = json.loads(
        optimized_json.read_text(encoding="utf-8")
    )
    physics_reference = json.loads(
        (result_dir / "independent_same_geometry.json").read_text(
            encoding="utf-8"
        )
    )
    legacy_wall = parse_wall_seconds(run_dir / "baseline_fp64.time")
    optimized_time = (
        gpu_result_dir / "new_farfield.log"
        if optimized_json.parent == gpu_result_dir
        else result_dir / "optimized_batch3_inner004_mirror.time"
    )
    optimized_repeated_wall = parse_wall_seconds(optimized_time)

    reference_mueller = np.asarray(
        physics_reference["physical"]["mueller"], dtype=np.float64
    )
    optimized_mueller = np.asarray(
        optimized["physical"]["mueller"], dtype=np.float64
    )
    theta = np.deg2rad(
        np.asarray(
            physics_reference["physical"]["theta_degrees"],
            dtype=np.float64,
        )
    )
    weights = np.sin(theta)
    weights[0] = 0.0
    weights[-1] = 0.0
    peak_m11 = float(np.max(np.abs(reference_mueller[0, 0])))

    summary = {
        "case": {
            "shape": "regular_hexagonal_prism",
            "aspect_h_over_d": 1.0,
            "ka": optimized["ka"],
            "refractive_index": optimized["ri"],
            "refinement": optimized["refinements"],
            "system_dofs": optimized["system_dofs"],
            "quadrature_points": optimized["quadrature_points"],
            "relative_residual_tolerance": optimized["tolerance"],
        },
        "legacy_strict_fp64": {
            "wall_s": legacy_wall,
            "iterations": [
                legacy["mbj"]["iterations"],
                legacy["physical"]["parallel_iterations"],
            ],
            "residuals": [
                legacy["mbj"]["fmm_residual"],
                legacy["physical"]["parallel_fmm_residual"],
            ],
        },
        "optimized": {
            "result": str(optimized_json),
            "repeated_wall_s": optimized_repeated_wall,
            "outer_iterations": [
                optimized["pfft_fgmres"]["outer_iterations"],
                optimized["physical"]["parallel_iterations"],
            ],
            "residuals": [
                optimized["pfft_fgmres"]["fmm_residual"],
                optimized["physical"]["parallel_fmm_residual"],
            ],
            "inner_pfft_iterations": [
                optimized["pfft_fgmres"]["inner_iterations"],
                optimized["physical"]["parallel_pfft_inner_iterations"],
            ],
        },
        "speedup": {
            "legacy_to_current_repeated":
                legacy_wall / optimized_repeated_wall,
        },
        "physics_difference": {
            "reference": "independent_two_polarization_same_geometry",
            "all_mueller_weighted_relative_l2": weighted_relative_l2(
                reference_mueller, optimized_mueller, weights
            ),
            "m11_weighted_relative_l2": weighted_relative_l2(
                reference_mueller[0, 0],
                optimized_mueller[0, 0],
                weights,
            ),
            "maximum_absolute_difference_over_peak_m11": float(
                np.max(np.abs(reference_mueller - optimized_mueller))
                / peak_m11
            ),
        },
    }

    output_json = result_dir / "ka25_ref5_fast_strict_summary.json"
    output_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    names = [
        "Original strict\nFP64",
        "Current strict\nrepeated",
    ]
    times = [
        legacy_wall,
        optimized_repeated_wall,
    ]
    bars = axes[0].bar(
        names, times, color=["#5b6472", "#148f55"]
    )
    axes[0].set_ylabel("Complete wall time, s")
    axes[0].set_title(
        "ka=25, ref=5: "
        f"{legacy_wall / optimized_repeated_wall:.2f}x faster"
    )
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, times):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f} s",
            ha="center",
            va="bottom",
        )

    theta_degrees = np.asarray(
        physics_reference["physical"]["theta_degrees"],
        dtype=np.float64,
    )
    axes[1].semilogy(
        theta_degrees,
        np.maximum(
            np.abs(reference_mueller[0, 0] - optimized_mueller[0, 0])
            / peak_m11,
            1.0e-16,
        ),
        color="#c64f24",
        linewidth=2,
    )
    axes[1].set_xlabel("Scattering angle, degrees")
    axes[1].set_ylabel(r"$|\Delta M_{11}| / \max |M_{11}^{ref}|$")
    axes[1].set_title("Physical-output difference")
    axes[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(
        result_dir / "ka25_ref5_fast_strict_summary.png", dpi=180
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
