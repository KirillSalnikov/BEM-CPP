#!/usr/bin/env python3
"""Validate exact Cn polarization reconstruction against a two-solve result."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--symmetric", type=Path, required=True)
    parser.add_argument("--independent-time", type=Path, required=True)
    parser.add_argument("--symmetric-time", type=Path)
    parser.add_argument("--adda-time", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def wall_seconds(path: Path) -> float:
    match = re.search(r"ACTUAL_WALL_S=([0-9.eE+-]+)", path.read_text())
    if not match:
        raise ValueError(f"wall time is absent from {path}")
    return float(match.group(1))


def load(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    physical = data["physical"]
    return (
        data,
        np.asarray(physical["theta_degrees"], dtype=float),
        np.asarray(physical["mueller"], dtype=float),
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    independent_data, theta, independent = load(args.independent)
    symmetric_data, symmetric_theta, symmetric = load(args.symmetric)
    if independent.shape != symmetric.shape or not np.allclose(theta, symmetric_theta):
        raise ValueError("the two Mueller grids differ")

    radians = np.deg2rad(theta)
    weights = np.sin(radians)
    forward = independent[0, 0, 0]
    scale = np.sqrt(np.sum(independent[0, 0] ** 2 * weights))
    component_error = np.empty((4, 4))
    for row in range(4):
        for column in range(4):
            component_error[row, column] = (
                np.sqrt(
                    np.sum(
                        (symmetric[row, column] - independent[row, column]) ** 2
                        * weights
                    )
                )
                / scale
            )
    full_weighted_error = np.sqrt(
        np.sum((symmetric - independent) ** 2 * weights)
        / np.sum(independent**2 * weights)
    )
    forward_error = abs(symmetric[0, 0, 0] / forward - 1.0)
    point_error = np.max(np.abs(symmetric - independent), axis=(0, 1)) / abs(forward)

    independent_wall = wall_seconds(args.independent_time)
    adda_wall = wall_seconds(args.adda_time)
    second_solve = float(independent_data["physical"]["parallel_s"])
    reconstructed_seconds = float(symmetric_data["physical"]["parallel_s"])
    symmetric_full_estimate = independent_wall - second_solve + reconstructed_seconds
    symmetric_wall = (
        wall_seconds(args.symmetric_time)
        if args.symmetric_time is not None
        else symmetric_full_estimate
    )
    measured_symmetric_wall = args.symmetric_time is not None

    metrics = {
        "ka": independent_data["ka"],
        "ref": independent_data["refinements"],
        "refractive_index": independent_data["ri"],
        "independent_bem_wall_s": independent_wall,
        "symmetric_bem_wall_s": symmetric_wall,
        "symmetric_bem_wall_is_measured": measured_symmetric_wall,
        "legacy_symmetry_only_estimated_wall_s": symmetric_full_estimate,
        "adda_wall_s": adda_wall,
        "bem_speedup": independent_wall / symmetric_wall,
        "speedup_vs_adda": adda_wall / symmetric_wall,
        "strict_fmm_residual": symmetric_data["pfft_fgmres"]["fmm_residual"],
        "outer_iterations": symmetric_data["pfft_fgmres"]["outer_iterations"],
        "inner_iterations": symmetric_data["pfft_fgmres"]["inner_iterations"],
        "solid_angle_weighted_full_mueller_relative_error": full_weighted_error,
        "forward_m11_relative_error": forward_error,
        "maximum_point_error_over_forward_m11": float(np.max(point_error)),
        "component_error_over_m11_weighted_norm": component_error.tolist(),
        "symmetry_rhs_relative_error": symmetric_data["physical"][
            "cyclic_rhs_relative_error"
        ],
    }
    timing_note = (
        "Время ускоренного BEM измерено полным запуском: смешанный FMM\n"
        "используется в итерациях, FP64 FMM проверяет невязку после цикла,\n"
        "вторая поляризация восстанавливается по точной C6-симметрии."
        if measured_symmetric_wall
        else
        "Время C6 оценено для полного запуска: из измеренного времени\n"
        "двух решений исключено второе решение и добавлено дешёвое\n"
        "геометрическое восстановление. Подготовка оператора сохранена."
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )

    with PdfPages(args.output) as pdf:
        fig, axes = plt.subplots(4, 4, figsize=(15, 11), sharex=True)
        for row in range(4):
            for column in range(4):
                ax = axes[row, column]
                first = independent[row, column] / forward
                second = symmetric[row, column] / forward
                if row == 0 and column == 0:
                    ax.semilogy(theta, np.maximum(np.abs(first), 1.0e-12), label="два решения")
                    ax.semilogy(theta, np.maximum(np.abs(second), 1.0e-12), "--", label="симметрия C6")
                else:
                    ax.plot(theta, first, label="два решения")
                    ax.plot(theta, second, "--", label="симметрия C6")
                    ax.axhline(0.0, color="0.75", linewidth=0.6)
                ax.set_title(rf"$M_{{{row + 1}{column + 1}}}/M_{{11}}(0)$")
                ax.grid(alpha=0.25)
                if row == 3:
                    ax.set_xlabel(r"Угол рассеяния $\theta$, град.")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            "Шестигранная призма: все элементы матрицы Мюллера\n"
            rf"$ka=80$, $m=1.3$, ref=6; интегральное расхождение {100 * full_weighted_error:.3f}%",
            fontsize=16,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig)
        fig.savefig(args.output.with_name(args.output.stem + "_mueller.png"), dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        labels = [
            "ADDA FP32\ndpl=15, eps=1e-4",
            "BEM:\n2 решения",
            "BEM:\nсмешанный + C6",
        ]
        times = [adda_wall, independent_wall, symmetric_wall]
        colors = ["#6b7280", "#2563eb", "#16a34a"]
        bars = axes[0, 0].bar(labels, times, color=colors)
        axes[0, 0].bar_label(bars, labels=[f"{value:.0f} с" for value in times])
        axes[0, 0].set_ylabel("Полное время, с")
        axes[0, 0].set_title(
            f"Ускорение: {independent_wall / symmetric_wall:.2f}× к BEM, "
            f"{adda_wall / symmetric_wall:.2f}× к ADDA"
        )
        axes[0, 0].grid(axis="y", alpha=0.25)

        image = axes[0, 1].imshow(100 * component_error, cmap="magma")
        for row in range(4):
            for column in range(4):
                value = 100 * component_error[row, column]
                axes[0, 1].text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="black" if value > 0.34 else "white",
                )
        axes[0, 1].set_xticks(range(4), [f"j={value}" for value in range(1, 5)])
        axes[0, 1].set_yticks(range(4), [f"i={value}" for value in range(1, 5)])
        axes[0, 1].set_title(r"Ошибка $M_{ij}$ относительно нормы $M_{11}$, %")
        fig.colorbar(image, ax=axes[0, 1], fraction=0.046)

        axes[1, 0].plot(theta, 100 * point_error, color="#dc2626")
        axes[1, 0].set_xlabel(r"Угол рассеяния $\theta$, град.")
        axes[1, 0].set_ylabel(r"Макс. $|\Delta M_{ij}|/|M_{11}(0)|$, %")
        axes[1, 0].set_title(f"Максимум по углу: {100 * np.max(point_error):.3f}%")
        axes[1, 0].grid(alpha=0.25)

        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.0,
            1.0,
            "Условия проверки\n\n"
            "Геометрия: правильная шестигранная призма, h/D=1\n"
            "Размер: ka=80; показатель преломления: m=1.3\n"
            "Сетка: ref=6; 806 400 комплексных неизвестных\n"
            "Требуемая невязка первой поляризации: 10⁻⁵\n"
            f"Полученная строгая невязка: {metrics['strict_fmm_residual']:.2e}\n"
            f"Итерации: {metrics['outer_iterations']} внешних, "
            f"{metrics['inner_iterations']} внутренних\n"
            "Контроль времени ADDA: dpl=15, невязка 10⁻⁴\n"
            "Угловая сетка: 361 точка, 0…180°\n\n"
            f"Проверка преобразования правой части: "
            f"{metrics['symmetry_rhs_relative_error']:.2e}\n"
            f"Полная интегральная ошибка матрицы: {100 * full_weighted_error:.3f}%\n"
            f"Ошибка M11(0): {100 * forward_error:.3f}%\n\n"
            + timing_note,
            va="top",
            fontsize=12,
        )
        fig.suptitle("Проверка ускорения второй поляризации по точной C6-симметрии", fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig)
        fig.savefig(args.output.with_name(args.output.stem + "_summary.png"), dpi=180)
        plt.close(fig)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
