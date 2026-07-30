#!/usr/bin/env python3
"""Compare matched BEM and ADDA orientation-averaging runs."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_wall(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"ACTUAL_WALL_S=([0-9.]+)", text)
    if not match:
        raise ValueError(f"ACTUAL_WALL_S is missing in {path}")
    return float(match.group(1))


def weighted_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    theta = np.linspace(0.0, np.pi, reference.shape[-1])
    weights = np.sin(theta)
    weights[[0, -1]] = 0.0
    weight_shape = (1,) * (reference.ndim - 1) + (-1,)
    weights = weights.reshape(weight_shape)
    numerator = np.sum(weights * np.square(candidate - reference))
    denominator = np.sum(weights * np.square(reference))
    return float(np.sqrt(numerator / denominator))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs/goal_ka25_ref5/orientation_gpu"),
    )
    args = parser.parse_args()
    root = args.root

    bem_cold_path = root / "avg_a4_b1_g1.json"
    bem_path = root / "avg_a4_b1_g1_repeated.json"
    adda_dir = root / "adda_dpl15_a4_b1_g1"
    bem = json.loads(bem_path.read_text(encoding="utf-8"))
    bem_cold = json.loads(bem_cold_path.read_text(encoding="utf-8"))

    adda_table = np.loadtxt(adda_dir / "mueller", skiprows=1)
    adda_theta = adda_table[:, 0]
    adda_mueller = adda_table[:, 1:].reshape((-1, 4, 4)).transpose(1, 2, 0)
    bem_theta = np.asarray(bem["theta_degrees"], dtype=np.float64)
    bem_mueller = np.asarray(bem["mueller"], dtype=np.float64)
    if not np.allclose(bem_theta, adda_theta, rtol=0.0, atol=1.0e-12):
        raise ValueError("BEM and ADDA scattering-angle grids differ")

    bem_wall = read_wall(root / "avg_a4_b1_g1_repeated.time")
    adda_wall = read_wall(adda_dir / "time.txt")
    bem_cold_wall = float(bem_cold["timing"]["total_with_setup_s"])

    bem_scale = float(bem_mueller[0, 0, 0])
    adda_scale = float(adda_mueller[0, 0, 0])
    bem_normalized = bem_mueller / bem_scale
    adda_normalized = adda_mueller / adda_scale
    component_peak_errors = np.max(
        np.abs(bem_normalized - adda_normalized), axis=2
    )

    summary = {
        "case": {
            "shape": "regular_hexagonal_prism",
            "aspect_h_over_d": 1.0,
            "ka": 25.0,
            "refractive_index": 1.3,
            "linear_residual_tolerance": 1.0e-5,
            "alpha_degrees": [0.0, 90.0, 180.0, 270.0],
            "beta_degrees": [90.0],
            "gamma_degrees": [30.0],
            "scattering_angles": int(bem_theta.size),
            "bem_refinement": int(bem["refinements"]),
            "adda_dpl": 15,
        },
        "timing": {
            "bem_cold_s": bem_cold_wall,
            "bem_cached_external_wall_s": bem_wall,
            "bem_cached_reported_s": float(
                bem["timing"]["total_with_setup_s"]
            ),
            "bem_cached_solve_s": float(bem["timing"]["solve_s"]),
            "bem_cached_farfield_s": float(bem["timing"]["farfield_s"]),
            "adda_external_wall_s": adda_wall,
            "adda_reported_wall_s": 43.2471,
            "adda_initialization_s": 1.1288,
            "adda_internal_fields_s": 43.0089,
            "adda_scattered_fields_s": 1.1288,
            "adda_speedup_over_bem_cached": bem_wall / adda_wall,
            "adda_speedup_over_bem_cold": bem_cold_wall / adda_wall,
        },
        "linear_solves": {
            "bem": {
                "base_orientations": int(
                    bem["orientation"]["solved_base_orientations"]
                ),
                "polarizations": 2,
                "outer_iterations_total": int(bem["iterations"]["total"]),
                "maximum_verified_residual": float(
                    bem["iterations"]["maximum_residual"]
                ),
            },
            "adda": {
                "single_particle_evaluations": 1,
                "polarizations": 2,
                "qmr2_iterations_total": 955,
            },
        },
        "physics": {
            "forward_m11": {
                "bem": bem_scale,
                "adda": adda_scale,
                "relative_difference": abs(bem_scale - adda_scale)
                / abs(adda_scale),
            },
            "raw_all_mueller_weighted_relative_l2": weighted_l2(
                adda_mueller, bem_mueller
            ),
            "normalized_all_mueller_weighted_relative_l2": weighted_l2(
                adda_normalized, bem_normalized
            ),
            "normalized_m11_weighted_relative_l2": weighted_l2(
                adda_normalized[0, 0], bem_normalized[0, 0]
            ),
            "maximum_normalized_component_difference":
                float(np.max(component_peak_errors)),
            "maximum_normalized_difference_by_component":
                component_peak_errors.tolist(),
        },
        "interpretation": (
            "The timing is matched, but ref=5 BEM and dpl=15 ADDA are "
            "different spatial discretizations. Their physical difference "
            "is therefore a discretization comparison, not a solver error."
        ),
    }

    output_json = root / "bem_vs_adda_orientation_a4_b1_g1.json"
    output_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    labels = ["BEM\nпервый", "BEM\nс кэшами", "ADDA-OCL\ndpl=15"]
    times = [bem_cold_wall, bem_wall, adda_wall]
    bars = axes[0, 0].bar(
        labels, times, color=["#777777", "#16865c", "#2676b8"]
    )
    axes[0, 0].set_ylabel("Полное стеночное время, с")
    axes[0, 0].set_title("Время одинакового усреднения")
    axes[0, 0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, times):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    axes[0, 1].semilogy(
        bem_theta,
        np.maximum(np.abs(bem_normalized[0, 0]), 1.0e-12),
        linewidth=2.0,
        label="BEM ref=5",
    )
    axes[0, 1].semilogy(
        adda_theta,
        np.maximum(np.abs(adda_normalized[0, 0]), 1.0e-12),
        linewidth=1.8,
        linestyle="--",
        label="ADDA dpl=15",
    )
    axes[0, 1].set_xlabel("Угол рассеяния, град.")
    axes[0, 1].set_ylabel(r"$|M_{11}(\theta)|/M_{11}(0)$")
    axes[0, 1].set_title("Нормированная угловая зависимость")
    axes[0, 1].grid(which="both", alpha=0.25)
    axes[0, 1].legend()

    selected = [(0, 1), (1, 1), (2, 2), (2, 3)]
    colors = ["#2676b8", "#c34f2c", "#16865c", "#8f5daa"]
    for (row, column), color in zip(selected, colors):
        label = rf"$M_{{{row + 1}{column + 1}}}/M_{{11}}(0)$"
        axes[1, 0].plot(
            bem_theta,
            bem_normalized[row, column],
            color=color,
            linewidth=1.8,
            label=f"BEM {label}",
        )
        axes[1, 0].plot(
            adda_theta,
            adda_normalized[row, column],
            color=color,
            linewidth=1.4,
            linestyle="--",
            label=f"ADDA {label}",
        )
    axes[1, 0].set_xlabel("Угол рассеяния, град.")
    axes[1, 0].set_ylabel(r"$M_{ij}(\theta)/M_{11}(0)$")
    axes[1, 0].set_title("Выбранные элементы: BEM и ADDA")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8, ncol=2)

    image = axes[1, 1].imshow(
        component_peak_errors,
        cmap="magma",
        origin="upper",
    )
    for row in range(4):
        for column in range(4):
            value = component_peak_errors[row, column]
            color = (
                "black"
                if value > 0.45 * np.max(component_peak_errors)
                else "white"
            )
            axes[1, 1].text(
                column,
                row,
                f"{value:.2e}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    axes[1, 1].set_xticks(range(4), [f"j={index}" for index in range(1, 5)])
    axes[1, 1].set_yticks(range(4), [f"i={index}" for index in range(1, 5)])
    axes[1, 1].set_title(
        r"$\max_\theta|\Delta(M_{ij}/M_{11}(0))|$"
    )
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        r"Шестигранная призма: $ka=25$, $m=1.3$, "
        r"$N_\alpha=4$, $\beta=90^\circ$, $\gamma=30^\circ$",
        fontsize=15,
    )
    fig.tight_layout()
    output_png = root / "bem_vs_adda_orientation_a4_b1_g1.png"
    fig.savefig(output_png, dpi=180)

    all_figure, all_axes = plt.subplots(
        4, 4, figsize=(16.0, 12.5), sharex=True
    )
    for row in range(4):
        for column in range(4):
            axis = all_axes[row, column]
            axis.plot(
                bem_theta,
                bem_normalized[row, column],
                color="#16865c",
                linewidth=1.8,
                label="BEM ref=5",
            )
            axis.plot(
                adda_theta,
                adda_normalized[row, column],
                color="#2676b8",
                linewidth=1.5,
                linestyle="--",
                label="ADDA dpl=15",
            )
            axis.set_yscale("symlog", linthresh=1.0e-6, linscale=0.6)
            axis.set_title(rf"$M_{{{row + 1}{column + 1}}}$")
            axis.grid(which="both", alpha=0.22)
            if row == 3:
                axis.set_xlabel(r"$\theta$, град.")
            if column == 0:
                axis.set_ylabel(r"$M_{ij}/M_{11}(0)$")
    handles, labels = all_axes[0, 0].get_legend_handles_labels()
    all_figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=2,
        frameon=True,
    )
    all_figure.suptitle(
        r"Все элементы матрицы Мюллера: $ka=25$, $m=1.3$, "
        r"$N_\alpha=4$, $\beta=90^\circ$, $\gamma=30^\circ$"
        "\nСимметричная логарифмическая шкала, нормировка каждого метода "
        r"на собственное $M_{11}(0)$",
        fontsize=15,
        y=0.995,
    )
    all_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    all_output_png = (
        root / "bem_vs_adda_orientation_a4_b1_g1_all_mueller.png"
    )
    all_figure.savefig(all_output_png, dpi=180)

    print(json.dumps(summary, indent=2))
    print(output_png)
    print(all_output_png)


if __name__ == "__main__":
    main()
