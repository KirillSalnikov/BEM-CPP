#!/usr/bin/env python3
"""Report the matched ka=30, 256-alpha BEM/ADDA comparison."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("runs/orientation_bem_adda_crossover/ka30")
BEM_DIR = ROOT / "bem_ref5_pfft01_alpha256"
ADDA_DIR = ROOT / "adda_dpl15_alpha256"


def wall_seconds(path: Path) -> float:
    match = re.search(
        r"ACTUAL_WALL_S=([0-9.]+)",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    if not match:
        raise ValueError(f"wall time is missing in {path}")
    return float(match.group(1))


def weighted_relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    theta = np.linspace(0.0, np.pi, reference.shape[-1])
    weights = np.sin(theta)
    weights[[0, -1]] = 0.0
    shaped = weights.reshape((1,) * (reference.ndim - 1) + (-1,))
    return float(
        np.sqrt(
            np.sum(shaped * np.square(candidate - reference))
            / np.sum(shaped * np.square(reference))
        )
    )


def main() -> None:
    bem = json.loads((BEM_DIR / "average.json").read_text(encoding="utf-8"))
    adda_table = np.loadtxt(ADDA_DIR / "mueller", skiprows=1)
    theta = np.asarray(bem["theta_degrees"], dtype=np.float64)
    adda_theta = adda_table[:, 0]
    if not np.allclose(theta, adda_theta, rtol=0.0, atol=1.0e-12):
        raise ValueError("BEM and ADDA scattering grids differ")

    bem_mueller = np.asarray(bem["mueller"], dtype=np.float64)
    adda_mueller = adda_table[:, 1:].reshape((-1, 4, 4)).transpose(1, 2, 0)
    bem_scale = float(bem_mueller[0, 0, 0])
    adda_scale = float(adda_mueller[0, 0, 0])
    bem_norm = bem_mueller / bem_scale
    adda_norm = adda_mueller / adda_scale
    component_error = np.max(np.abs(bem_norm - adda_norm), axis=2)
    bem_wall = wall_seconds(BEM_DIR / "time.txt")
    adda_wall = wall_seconds(ADDA_DIR / "time.txt")

    summary = {
        "case": {
            "shape": "regular_hexagonal_prism",
            "aspect_h_over_d": 1.0,
            "ka": 30.0,
            "refractive_index": 1.3,
            "alpha_samples": 256,
            "beta_degrees": 90.0,
            "gamma_degrees": 30.0,
            "scattering_angles": int(theta.size),
            "tolerance": 1.0e-5,
            "bem_refinement": 5,
            "adda_dpl": 15,
        },
        "timing": {
            "bem_wall_s": bem_wall,
            "bem_solve_s": float(bem["timing"]["solve_s"]),
            "bem_farfield_s": float(bem["timing"]["farfield_s"]),
            "adda_wall_s": adda_wall,
            "adda_internal_fields_s": 181.7027,
            "adda_farfield_s": 129.4135,
            "bem_speedup": adda_wall / bem_wall,
            "farfield_speedup": 129.4135 / float(bem["timing"]["farfield_s"]),
        },
        "solver": {
            "bem_outer_iterations": int(bem["iterations"]["total"]),
            "bem_inner_pfft_iterations": int(
                bem["pfft_inner"]["iterations"]
            ),
            "bem_verified_residual": float(
                bem["iterations"]["maximum_residual"]
            ),
            "adda_qmr2_iterations": 1919,
        },
        "physics": {
            "forward_m11": {
                "bem": bem_scale,
                "adda": adda_scale,
                "relative_difference": abs(bem_scale - adda_scale)
                / abs(adda_scale),
            },
            "raw_all_mueller_weighted_relative_l2":
                weighted_relative_l2(adda_mueller, bem_mueller),
            "normalized_all_mueller_weighted_relative_l2":
                weighted_relative_l2(adda_norm, bem_norm),
            "normalized_m11_weighted_relative_l2":
                weighted_relative_l2(adda_norm[0, 0], bem_norm[0, 0]),
            "maximum_normalized_difference":
                float(np.max(component_error)),
            "maximum_normalized_difference_by_component":
                component_error.tolist(),
        },
    }
    (ROOT / "bem_vs_adda_alpha256_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    bars = axes[0].bar(
        ["BEM ref=5", "ADDA dpl=15"],
        [bem_wall, adda_wall],
        color=["#16865c", "#2676b8"],
    )
    axes[0].set_ylabel("Полное стеночное время, с")
    axes[0].set_title(f"BEM быстрее в {adda_wall / bem_wall:.2f} раза")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, (bem_wall, adda_wall)):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    axes[1].semilogy(
        theta,
        np.maximum(np.abs(bem_norm[0, 0]), 1.0e-12),
        color="#16865c",
        linewidth=2,
        label="BEM ref=5",
    )
    axes[1].semilogy(
        theta,
        np.maximum(np.abs(adda_norm[0, 0]), 1.0e-12),
        color="#2676b8",
        linewidth=1.6,
        linestyle="--",
        label="ADDA dpl=15",
    )
    axes[1].set_xlabel("Угол рассеяния, град.")
    axes[1].set_ylabel(r"$|M_{11}(\theta)|/M_{11}(0)$")
    axes[1].set_title("Нормированная угловая зависимость")
    axes[1].grid(which="both", alpha=0.25)
    axes[1].legend()

    image = axes[2].imshow(component_error, cmap="magma", origin="upper")
    maximum = float(np.max(component_error))
    for row in range(4):
        for column in range(4):
            value = component_error[row, column]
            axes[2].text(
                column,
                row,
                f"{value:.2e}",
                ha="center",
                va="center",
                color="black" if value > 0.45 * maximum else "white",
                fontsize=9,
            )
    axes[2].set_xticks(range(4), [f"j={index}" for index in range(1, 5)])
    axes[2].set_yticks(range(4), [f"i={index}" for index in range(1, 5)])
    axes[2].set_title(
        r"$\max_\theta|\Delta(M_{ij}/M_{11}(0))|$"
    )
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    figure.suptitle(
        r"$ka=30$, $m=1.3$, $N_\alpha=256$, "
        r"$\beta=90^\circ$, $\gamma=30^\circ$, невязка $10^{-5}$",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(ROOT / "bem_vs_adda_alpha256.png", dpi=180)

    all_figure, all_axes = plt.subplots(
        4, 4, figsize=(16.0, 12.5), sharex=True
    )
    for row in range(4):
        for column in range(4):
            axis = all_axes[row, column]
            axis.plot(
                theta,
                bem_norm[row, column],
                color="#16865c",
                linewidth=1.8,
                label="BEM ref=5",
            )
            axis.plot(
                theta,
                adda_norm[row, column],
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
        r"Все элементы матрицы Мюллера: $ka=30$, $m=1.3$, "
        r"$N_\alpha=256$, $\beta=90^\circ$, $\gamma=30^\circ$"
        "\nСимметричная логарифмическая шкала, нормировка каждого метода "
        r"на собственное $M_{11}(0)$",
        fontsize=15,
        y=0.995,
    )
    all_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    all_figure.savefig(
        ROOT / "bem_vs_adda_alpha256_all_mueller.png", dpi=180
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
