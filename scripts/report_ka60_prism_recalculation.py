#!/usr/bin/env python3
"""Build a compact audit page for the strict ka=60 prism recalculation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bem", type=Path, required=True)
    parser.add_argument("--bem-time", type=Path, required=True)
    parser.add_argument("--adda-dir", type=Path, required=True)
    parser.add_argument("--strict-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def wall_seconds(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"Elapsed \(wall clock\) time .*?:\s*"
        r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)",
        text,
    )
    if match:
        return (
            3600.0 * int(match.group(1) or 0)
            + 60.0 * int(match.group(2))
            + float(match.group(3))
        )
    match = re.search(r"(?:ACTUAL_)?WALL_S=([0-9.eE+-]+)", text)
    if match:
        return float(match.group(1))
    raise ValueError(f"cannot parse wall time from {path}")


def load_adda(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, skiprows=1)
    return table[:, 0], table[:, 1:].reshape(-1, 4, 4).transpose(1, 2, 0)


def common_data(
    theta_adda: np.ndarray,
    adda: np.ndarray,
    theta_bem: np.ndarray,
    bem: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.intersect1d(theta_adda, theta_bem)
    adda_indices = np.asarray(
        [np.flatnonzero(np.isclose(theta_adda, value))[0] for value in theta]
    )
    bem_indices = np.asarray(
        [np.flatnonzero(np.isclose(theta_bem, value))[0] for value in theta]
    )
    return theta, adda[:, :, adda_indices], bem[:, :, bem_indices]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bem_data = json.loads(args.bem.read_text(encoding="utf-8"))
    strict = json.loads(args.strict_metrics.read_text(encoding="utf-8"))
    theta_bem = np.asarray(bem_data["physical"]["theta_degrees"], dtype=float)
    bem = np.asarray(bem_data["physical"]["mueller"], dtype=float)
    theta_adda, adda = load_adda(args.adda_dir / "mueller")
    theta, adda_common, bem_common = common_data(
        theta_adda, adda, theta_bem, bem
    )
    scale = float(adda[0, 0, 0])
    maximum = np.max(np.abs(bem_common - adda_common), axis=2) / abs(scale)

    bem_wall_s = wall_seconds(args.bem_time)
    adda_wall_s = wall_seconds(args.adda_dir / "time.txt")
    resumed_first = int(
        bem_data["pfft_fgmres"].get("resumed_outer_iterations", 0)
    )
    resumed_second = int(
        bem_data["physical"].get("parallel_resumed_iterations", 0)
    )
    full_cold_run = resumed_first == 0 and resumed_second == 0
    speedup = adda_wall_s / bem_wall_s if full_cold_run else None
    first_residual = float(bem_data["pfft_fgmres"]["fmm_residual"])
    second_residual = float(bem_data["physical"]["parallel_fmm_residual"])
    shape_error = float(
        strict["shape_only_full_relative_l2_common_angles"]
    )
    extinction_error = float(
        strict["physical_extinction_audit"]["relative_difference"]
    )
    forward_ratio = float(strict["forward_M11_bem_over_adda"])

    dofs_text = f"{int(bem_data['system_dofs']):,}".replace(",", " ")
    quadrature_text = (
        f"{int(bem_data['quadrature_points']):,}".replace(",", " ")
    )

    figure = plt.figure(figsize=(17.0, 10.0))
    grid = figure.add_gridspec(
        2, 3, height_ratios=(0.72, 1.28), width_ratios=(1.15, 1.15, 1.0)
    )
    metadata_axis = figure.add_subplot(grid[0, :2])
    metadata_axis.axis("off")
    if full_cold_run:
        timing_line = (
            f"Полный холодный BEM: {bem_wall_s:.2f} с; "
            f"ускорение относительно ADDA: {speedup:.3f}×."
        )
    else:
        timing_line = (
            f"Контрольный пересчёт из чекпоинта: {bem_wall_s:.2f} с. "
            "Это не полное холодное время BEM."
        )
    metadata = (
        r"$ka=60$, $m=1.3$, шестигранная призма $h/D_x=1$, "
        r"$\varepsilon=10^{-5}$"
        "\n"
        f"ADDA: dpl=15.047, 12 426 777 диполей, QMR2, "
        f"полный запуск {adda_wall_s:.2f} с"
        "\n"
        f"BEM: ref=6 + сгущение у рёбер, "
        f"{dofs_text} неизвестных, "
        f"{quadrature_text} квадратурных точек"
        "\n"
        f"Две поляризации: {bem_data['pfft_fgmres']['outer_iterations']} и "
        f"{bem_data['physical']['parallel_iterations']} итераций; "
        f"невязки {first_residual:.3e} и {second_residual:.3e}"
        "\n"
        f"{timing_line}"
    )
    metadata_axis.text(
        0.0,
        1.0,
        metadata,
        va="top",
        ha="left",
        fontsize=14,
        linespacing=1.5,
        bbox={
            "facecolor": "#f4f6f7",
            "edgecolor": "#9aa4aa",
            "boxstyle": "round,pad=0.7",
        },
    )

    metric_axis = figure.add_subplot(grid[0, 2])
    labels = [
        r"Форма всех $M_{ij}$",
        r"$C_{\rm ext}$",
        r"$M_{11}(0)$",
    ]
    values = np.asarray(
        [100.0 * shape_error, 100.0 * extinction_error,
         100.0 * abs(forward_ratio - 1.0)]
    )
    bars = metric_axis.bar(
        np.arange(3), values, color=["#16865c", "#2676b8", "#c47a12"]
    )
    metric_axis.set_yscale("log")
    metric_axis.set_xticks(np.arange(3), labels)
    metric_axis.set_ylabel("Относительное расхождение, %")
    metric_axis.set_title("Согласие BEM и ADDA")
    metric_axis.grid(axis="y", which="both", alpha=0.25)
    for bar, value in zip(bars, values):
        metric_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.08,
            f"{value:.3g}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    curve_axis = figure.add_subplot(grid[1, :2])
    curve_axis.semilogy(
        theta_adda,
        np.maximum(adda[0, 0] / scale, 1.0e-12),
        "o-",
        color="#2676b8",
        markersize=3,
        linewidth=1.7,
        label="ADDA, шаг 2,5°",
    )
    curve_axis.semilogy(
        theta_bem,
        np.maximum(bem[0, 0] / scale, 1.0e-12),
        color="#c47a12",
        linewidth=1.5,
        label="BEM, шаг 1°",
    )
    curve_axis.set_xlabel(r"Угол рассеяния $\theta$, град.")
    curve_axis.set_ylabel(r"$M_{11}(\theta)/M_{11}^{ADDA}(0)$")
    curve_axis.set_title("Фазовая функция на исходных угловых сетках")
    curve_axis.set_xlim(0.0, 180.0)
    curve_axis.grid(which="both", alpha=0.25)
    curve_axis.legend()

    map_axis = figure.add_subplot(grid[1, 2])
    image = map_axis.imshow(
        np.maximum(maximum, 1.0e-16),
        cmap="magma",
        norm=matplotlib.colors.LogNorm(),
        aspect="equal",
    )
    threshold = np.sqrt(
        np.maximum(maximum.min(), 1.0e-16)
        * np.maximum(maximum.max(), 1.0e-16)
    )
    for row in range(4):
        for column in range(4):
            map_axis.text(
                column,
                row,
                f"{maximum[row, column]:.1e}",
                ha="center",
                va="center",
                color="black" if maximum[row, column] > threshold else "white",
                fontsize=8,
            )
    map_axis.set_xticks(range(4), [f"j={index}" for index in range(1, 5)])
    map_axis.set_yticks(range(4), [f"i={index}" for index in range(1, 5)])
    map_axis.set_title(r"Максимум $|\Delta M_{ij}|$")
    figure.colorbar(
        image,
        ax=map_axis,
        orientation="horizontal",
        pad=0.12,
        fraction=0.07,
        label=r"$\max_\theta|\Delta M_{ij}|/M_{11}^{ADDA}(0)$",
    )

    figure.suptitle(
        "Контрольный пересчёт: призма, ka=60, одна ориентация",
        fontsize=20,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    output = args.out_dir / "ka60_prism_recalculation_summary.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)

    summary = {
        "case": {
            "shape": "regular hexagonal prism",
            "aspect_h_over_Dx": 1.0,
            "ka": 60.0,
            "refractive_index": 1.3,
            "orientation": "default, single orientation",
            "residual_tolerance": 1.0e-5,
        },
        "adda": {
            "dpl": 15.0471,
            "occupied_dipoles": 12426777,
            "full_wall_s": adda_wall_s,
        },
        "bem": {
            "refinement": int(bem_data["refinements"]),
            "edge_refinement": int(bem_data["edge_refine_applied"]),
            "system_dofs": int(bem_data["system_dofs"]),
            "quadrature_points": int(bem_data["quadrature_points"]),
            "first_residual": first_residual,
            "second_residual": second_residual,
            "wall_s": bem_wall_s,
            "full_cold_run": full_cold_run,
            "resumed_first_iterations": resumed_first,
            "resumed_second_iterations": resumed_second,
        },
        "comparison": {
            "common_angle_count": int(theta.size),
            "shape_only_full_relative_l2": shape_error,
            "extinction_relative_difference": extinction_error,
            "forward_M11_bem_over_adda": forward_ratio,
        },
        "speedup_adda_over_bem": speedup,
        "timing_warning": (
            None
            if full_cold_run
            else (
                "The BEM wall time is a checkpoint recalculation, not a "
                "cold solve, and must not be used as an ADDA/BEM speedup."
            )
        ),
    }
    (args.out_dir / "ka60_prism_recalculation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
