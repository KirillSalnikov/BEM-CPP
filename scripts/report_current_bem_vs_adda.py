#!/usr/bin/env python3
"""Plot the current strict Muller pFFT-FGMRES BEM against matching ADDA-OCL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path(
            "runs/adda_exact/compare_ref5_ka20_n1p5_alpha15/"
            "comparison_summary.json"
        ),
    )
    parser.add_argument(
        "--bem",
        type=Path,
        default=Path(
            "runs/muller_pfft/"
            "ref5_ka20_pfft_fgmres_auto_cached.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "runs/adda_exact/compare_ref5_ka20_n1p5/"
            "current_fastest_bem_vs_adda.png"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = json.loads(args.comparison.read_text())
    bem = json.loads(args.bem.read_text())

    bem_summary = comparison["bem"]
    adda_summary = comparison["adda"][0]["run"]

    bem_wall = float(bem_summary["wall_s"])
    adda_wall = float(adda_summary["wall_s"])
    adda_setup = float(adda_summary.get("initialization_s", 0.6894))
    adda_per_orientation = adda_wall - adda_setup

    bem_setup = (
        float(bem["fmm_setup_s"])
        + float(bem["mbj_local_setup_s"])
        + float(bem["pfft_fgmres"]["fmm_switch_setup_s"])
    )
    bem_solve = float(bem["mbj"]["solve_s"])
    bem_symmetry_check = float(bem["physical"]["parallel_s"])
    bem_farfield = float(bem["physical"]["farfield_s"])
    bem_other = max(
        0.0,
        bem_wall
        - bem_setup
        - bem_solve
        - bem_symmetry_check
        - bem_farfield,
    )
    bem_axis_per_orientation = (
        bem_solve + bem_symmetry_check + bem_farfield + bem_other
    )
    bem_general_per_orientation = 2.0 * bem_solve + bem_farfield + bem_other

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))
    fig.suptitle(
        "Предварительное сравнение времени BEM и ADDA-OCL",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        (
            r"$ka=20$, $m=1.5$, $h/D=1$, "
            r"$\varepsilon=10^{-5}$, около 15 узлов на длину волны"
        ),
        ha="center",
        fontsize=14,
    )

    ax = axes[0]
    labels = ["ADDA-OCL", "BEM pFFT-\nFGMRES"]
    x = np.arange(2)
    setup = [adda_setup, bem_setup]
    solve = [adda_per_orientation, bem_solve]
    symmetry = [0.0, bem_symmetry_check]
    farfield_other = [
        0.0,
        bem_farfield + bem_other,
    ]
    colors = ["#4f8f5b", "#dd8a32", "#4f75b5", "#7d6a91"]
    bottoms = np.zeros(2)
    for values, label, color in (
        (setup, "Одноразовая подготовка", colors[0]),
        (solve, "Решение системы", colors[1]),
        (symmetry, "Проверка симметрии", colors[2]),
        (farfield_other, "Дальнее поле и прочее", colors[3]),
    ):
        ax.bar(x, values, bottom=bottoms, width=0.62, label=label, color=color)
        bottoms += np.asarray(values)
    for index, value in enumerate((adda_wall, bem_wall)):
        ax.text(
            index,
            value + 3.0,
            f"{value:.2f} с",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )
    ax.text(
        0.5,
        max(adda_wall, bem_wall) * 0.56,
        f"На этих сетках ADDA быстрее в {bem_wall / adda_wall:.2f} раза",
        ha="center",
        bbox={"facecolor": "white", "edgecolor": "#555555", "alpha": 0.92},
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Полное время, с")
    ax.set_title("Одна осевая ориентация: измерено")
    ax.set_ylim(0.0, bem_wall * 1.2)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=10.5)

    ax = axes[1]
    orientation_count = np.unique(
        np.rint(np.geomspace(1, 1000, 200)).astype(int)
    )
    adda_total = adda_setup + orientation_count * adda_per_orientation
    bem_axis_total = bem_setup + orientation_count * bem_axis_per_orientation
    bem_general_total = (
        bem_setup + orientation_count * bem_general_per_orientation
    )
    ax.plot(
        orientation_count,
        adda_total / 3600.0,
        color="#3d7f4a",
        linewidth=2.8,
        label="ADDA-OCL",
    )
    ax.plot(
        orientation_count,
        bem_axis_total / 3600.0,
        color="#d9781e",
        linewidth=2.8,
        label="BEM: ось симметрии",
    )
    ax.plot(
        orientation_count,
        bem_general_total / 3600.0,
        color="#8b3f52",
        linewidth=2.5,
        linestyle="--",
        label="BEM: произвольные ориентации, оценка",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Число ориентаций")
    ax.set_ylabel("Полное время, ч")
    ax.set_title("Усреднение: подготовка выполняется один раз")
    ax.grid(which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=10.5)
    ax.text(
        0.98,
        0.04,
        (
            "При большом числе ориентаций ADDA быстрее:\n"
            f"в {bem_axis_per_orientation / adda_per_orientation:.2f} раза "
            "на оси симметрии;\n"
            f"в {bem_general_per_orientation / adda_per_orientation:.2f} раза "
            "для двух независимых поляризаций."
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.92},
    )

    fig.text(
        0.5,
        0.015,
        (
            "Не итоговый рейтинг при одинаковой физической точности: "
            r"ADDA при dpl=15$\to$20 меняет $M_{11}$ на 0.46%, "
            r"BEM при ref=4$\to$5 — на 3.62%."
        ),
        ha="center",
        fontsize=12,
        color="#8b1e2d",
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.105, wspace=0.25)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)

    summary_path = args.out.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "case": comparison["particle"],
                "measured_one_orientation_s": {
                    "adda_ocl": adda_wall,
                    "bem_pfft_fgmres": bem_wall,
                    "adda_speedup": bem_wall / adda_wall,
                },
                "amortized_per_orientation_s": {
                    "adda_ocl": adda_per_orientation,
                    "bem_axis_symmetry": bem_axis_per_orientation,
                    "bem_general_two_polarizations_estimate": (
                        bem_general_per_orientation
                    ),
                },
                "large_sweep_adda_speedup": {
                    "axis_symmetry": (
                        bem_axis_per_orientation / adda_per_orientation
                    ),
                    "general_two_polarizations_estimate": (
                        bem_general_per_orientation / adda_per_orientation
                    ),
                },
                "accuracy_matched_comparison": False,
                "grid_convergence": {
                    "adda_dpl15_to_dpl20_m11_relative": 0.004628810654641212,
                    "bem_ref4_to_ref5_m11_relative": 0.03620698541249845,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(args.out)
    print(summary_path)


if __name__ == "__main__":
    main()
