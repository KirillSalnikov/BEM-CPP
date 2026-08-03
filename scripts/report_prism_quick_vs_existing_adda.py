#!/usr/bin/env python3
"""Report BEM quick timings against existing ADDA prism calculations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_nodal_bem_adda import comparison_metrics  # noqa: E402


KAS = (10, 15, 20, 25, 30, 60)
DEFAULT_BEM_ROOT = ROOT / "runs" / "prism_quick_vs_existing_adda_ka10_60"
DEFAULT_ADDA_ROOT = ROOT / "runs" / "hdiv_bem_vs_adda_sweep_n1p3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bem-root", type=Path, default=DEFAULT_BEM_ROOT)
    parser.add_argument("--adda-root", type=Path, default=DEFAULT_ADDA_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bem(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = load_json(path)
    physical = data["physical"]
    theta = np.asarray(physical["theta_degrees"], dtype=float)
    mueller = np.asarray(physical["mueller"], dtype=float)
    if mueller.shape != (4, 4, theta.size):
        raise ValueError(f"{path}: unexpected Mueller shape {mueller.shape}")
    return theta, mueller, data


def load_adda(directory: Path, theta_target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    table = np.loadtxt(directory / "mueller", skiprows=1)
    theta = table[:, 0]
    if theta.shape != theta_target.shape or not np.allclose(
        theta, theta_target, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(f"{directory}: ADDA and BEM angular grids differ")
    mueller = table[:, 1:].reshape(-1, 4, 4).transpose(1, 2, 0)
    log = (directory / "log").read_text(errors="replace")
    time_text = (directory / "time.txt").read_text(errors="replace")
    wall_match = re.search(r"ACTUAL_WALL_S=([0-9.eE+-]+)", time_text)
    residual_match = re.search(r"Required relative residual norm:\s*([0-9.eE+-]+)", log)
    iterations_match = re.search(r"Total number of iterations:\s*(\d+)", log)
    dipoles_match = re.search(r"Total number of occupied dipoles:\s*(\d+)", log)
    if not wall_match:
        raise ValueError(f"{directory}: no external ADDA wall time")
    return mueller, {
        "wall_s": float(wall_match.group(1)),
        "tolerance": float(residual_match.group(1)) if residual_match else None,
        "iterations": int(iterations_match.group(1)) if iterations_match else None,
        "occupied_dipoles": int(dipoles_match.group(1)) if dipoles_match else None,
    }


def collect(bem_root: Path, adda_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ka in KAS:
        bem_dir = bem_root / f"ka{ka}" / "bem_quick"
        adda_dir = adda_root / f"ka{ka}" / "adda_dpl15"
        theta, bem_mueller, bem = load_bem(bem_dir / "result.json")
        adda_mueller, adda = load_adda(adda_dir, theta)
        validation = load_json(bem_dir / "validation.json")
        if validation.get("errors"):
            raise ValueError(f"ka={ka}: BEM validation failed: {validation['errors']}")
        bem_wall = float(validation["wall_time_s"])
        metrics = comparison_metrics(theta, bem_mueller, adda_mueller)
        row = {
                "ka": ka,
                "bem_ref": int(bem["refinements"]),
                "bem_unknowns": int(bem["system_dofs"]),
                "bem_iterations_first": int(bem["mbj"]["iterations"]),
                "bem_iterations_second": int(bem["physical"]["parallel_iterations"]),
                "bem_maximum_residual": max(
                    float(bem["mbj"]["fmm_residual"]),
                    float(bem["physical"]["parallel_fmm_residual"]),
                ),
                "bem_wall_s": bem_wall,
                "adda_dipoles": adda["occupied_dipoles"],
                "adda_iterations_total": adda["iterations"],
                "adda_wall_s": adda["wall_s"],
                "adda_over_bem_speedup": adda["wall_s"] / bem_wall,
                "quick_vs_strict_normalized_full_l2": None,
                "quick_m11_over_strict_m11": None,
                "strict_vs_adda_weighted_full_l2": None,
                **metrics,
                "theta": theta,
                "bem_mueller": bem_mueller,
                "adda_mueller": adda_mueller,
            }
        if ka == 60:
            strict_path = (
                adda_root / "ka60" / "max_dpl15"
                / "bem_ref6_pfft_fgmres_r40_strict2pol.json"
            )
            if strict_path.is_file():
                strict_theta, strict_mueller, _ = load_bem(strict_path)
                if not np.allclose(theta, strict_theta, rtol=0.0, atol=1.0e-12):
                    raise ValueError("ka=60: quick and strict BEM angular grids differ")
                quick_normalized = bem_mueller / bem_mueller[0, 0, 0]
                strict_normalized = strict_mueller / strict_mueller[0, 0, 0]
                row["quick_vs_strict_normalized_full_l2"] = float(
                    np.linalg.norm(quick_normalized - strict_normalized)
                    / np.linalg.norm(strict_normalized)
                )
                row["quick_m11_over_strict_m11"] = float(
                    bem_mueller[0, 0, 0] / strict_mueller[0, 0, 0]
                )
                row["strict_vs_adda_weighted_full_l2"] = comparison_metrics(
                    theta, strict_mueller, adda_mueller
                )["solid_angle_weighted_full_relative_l2"]
        rows.append(row)
    return rows


def serializable(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"theta", "bem_mueller", "adda_mueller"}
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    clean = [serializable(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(clean[0]))
        writer.writeheader()
        writer.writerows(clean)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Быстрый BEM против сохранённых расчётов ADDA",
        "",
        (
            "Шестигранная призма h/D=1, m=1.3, ориентация (0,0,0), две "
            "поляризации и 73 угла. BEM использует профиль quick с невязкой "
            "1e-3 и автоматическим ref; ADDA-OCL использует dpl=15 и невязку 1e-5."
        ),
        "",
        "ADDA не пересчитывалась: использованы сохранённые внешние времена `ACTUAL_WALL_S`.",
        "",
        "| ka | BEM ref | BEM неизвестных | BEM, с | ADDA диполей | ADDA, с | ADDA/BEM | отличие полной матрицы, % | отличие M11, % |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['ka']} | {row['bem_ref']} | {row['bem_unknowns']} | "
            f"{row['bem_wall_s']:.2f} | {row['adda_dipoles']} | "
            f"{row['adda_wall_s']:.2f} | {row['adda_over_bem_speedup']:.2f} | "
            f"{100 * row['solid_angle_weighted_full_relative_l2']:.3f} | "
            f"{100 * row['solid_angle_weighted_M11_relative_l2']:.3f} |"
        )
    lines += [
        "",
        (
            "Значение ADDA/BEM больше единицы означает, что быстрый BEM быстрее. "
            "Физические отличия вычислены после независимой нормировки каждой "
            "матрицы на её M11(0) и с весом sin(theta)."
        ),
    ]
    ka60 = next((row for row in rows if row["ka"] == 60), None)
    if ka60 and ka60["quick_vs_strict_normalized_full_l2"] is not None:
        lines += [
            "",
            "## Контроль быстрого режима при ka=60",
            "",
            (
                "На той же сетке ref=6 быстрый и сохранённый строгий BEM "
                f"различаются по форме нормированной полной матрицы на "
                f"{100 * ka60['quick_vs_strict_normalized_full_l2']:.3f}%, "
                f"но M11(0) быстрого режима составляет только "
                f"{100 * ka60['quick_m11_over_strict_m11']:.2f}% строгого значения."
            ),
            (
                "Сохранённый строгий BEM и ADDA dpl=15 различаются по "
                f"взвешенной полной матрице на "
                f"{100 * ka60['strict_vs_adda_weighted_full_l2']:.3f}%. "
                "Поэтому ускорение 55.4 раза относится к практическому quick-режиму, "
                "а не к расчёту с одинаковой строгой точностью."
            ),
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_overview(path: Path, rows: list[dict[str, Any]]) -> None:
    ka = np.asarray([row["ka"] for row in rows], dtype=float)
    bem_time = np.asarray([row["bem_wall_s"] for row in rows])
    adda_time = np.asarray([row["adda_wall_s"] for row in rows])
    speedup = adda_time / bem_time
    full_error = 100 * np.asarray(
        [row["solid_angle_weighted_full_relative_l2"] for row in rows]
    )
    m11_error = 100 * np.asarray(
        [row["solid_angle_weighted_M11_relative_l2"] for row in rows]
    )
    forward_error = 100 * np.asarray(
        [abs(row["forward_M11_ratio_adda_over_bem"] - 1.0) for row in rows]
    )
    shape_error = 100 * np.asarray(
        [row["after_best_scalar_relative_l2"] for row in rows]
    )

    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10), constrained_layout=True)
    axes[0, 0].plot(ka, adda_time, "o-", lw=2, label="ADDA-OCL, dpl=15")
    axes[0, 0].plot(ka, bem_time, "s-", lw=2, label="BEM, быстрый режим")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Полное время, с")
    axes[0, 0].set_title("Холодный запуск двух поляризаций")
    axes[0, 0].legend()

    axes[0, 1].plot(ka, speedup, "o-", color="#16803c", lw=2)
    axes[0, 1].axhline(1.0, color="black", lw=1, ls="--")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Отношение времени ADDA/BEM")
    axes[0, 1].set_title("Отношение полного времени ADDA/BEM")
    for x, y in zip(ka, speedup):
        axes[0, 1].annotate(f"{y:.2f}×", (x, y), xytext=(0, 7),
                            textcoords="offset points", ha="center")

    axes[1, 0].plot(ka, full_error, "o-", lw=2, label="полная матрица")
    axes[1, 0].plot(ka, m11_error, "s-", lw=2, label=r"$M_{11}$")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Взвешенное отличие, %")
    axes[1, 0].set_title("BEM против сохранённой ADDA")
    axes[1, 0].legend()

    axes[1, 1].plot(ka, forward_error, "o-", lw=2,
                    label=r"масштаб $M_{11}(0)$")
    axes[1, 1].plot(ka, shape_error, "s-", lw=2,
                    label="форма после подгонки масштаба")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("Отличие, %")
    axes[1, 1].set_title("Масштаб и форма угловой зависимости")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Размерный параметр ka")
        axis.set_xticks(ka)
        axis.grid(True, which="both", alpha=0.25)
    figure.suptitle(
        "Шестигранная призма h/D=1, m=1,3: быстрый BEM и ADDA-OCL",
        fontsize=16,
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def element_figure(row: dict[str, Any]) -> plt.Figure:
    theta = row["theta"]
    bem = row["bem_mueller"]
    adda = row["adda_mueller"]
    scale = max(abs(float(adda[0, 0, 0])), 1.0e-300)
    figure, axes = plt.subplots(4, 4, figsize=(16.5, 12), sharex=True)
    for i in range(4):
        for j in range(4):
            axis = axes[i, j]
            axis.plot(theta, adda[i, j] / scale, color="#0072b2", lw=1.7,
                      label="ADDA-OCL, dpl=15")
            axis.plot(theta, bem[i, j] / scale, color="#16803c", lw=1.5,
                      ls="--", label="BEM, быстрый режим")
            axis.set_yscale("symlog", linthresh=1.0e-8, linscale=0.7)
            axis.set_title(rf"$M_{{{i + 1}{j + 1}}}/M_{{11}}^{{ADDA}}(0)$")
            axis.set_xlim(0, 180)
            axis.set_xticks((0, 45, 90, 135, 180))
            axis.grid(True, which="both", alpha=0.22)
            axis.tick_params(labelsize=8)
            if i == 3:
                axis.set_xlabel(r"Угол рассеяния $\theta$, град.", fontsize=9)
            if j == 0:
                axis.set_ylabel("Общая нормировка", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.922),
                  ncol=2, frameon=False)
    figure.suptitle(
        f"Шестигранная призма, ka={row['ka']}, m=1,3, h/D=1\n"
        f"BEM {row['bem_wall_s']:.2f} с; ADDA {row['adda_wall_s']:.2f} с; "
        f"ADDA/BEM={row['adda_over_bem_speedup']:.2f}×; "
        f"полная матрица: {100 * row['solid_angle_weighted_full_relative_l2']:.3f}%",
        fontsize=14.5,
        y=0.986,
    )
    figure.text(
        0.5,
        0.012,
        (
            r"Общая нормировка на $M_{11}^{ADDA}(0)$ сохраняет отличие масштаба. "
            r"BEM quick: невязка $10^{-3}$; ADDA: невязка $10^{-5}$."
        ),
        ha="center",
        fontsize=9.5,
    )
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.065, top=0.88,
                           hspace=0.34, wspace=0.24)
    return figure


def write_element_plots(output: Path, rows: list[dict[str, Any]]) -> None:
    pdf_path = output / "all_mueller_quick_bem_vs_adda.pdf"
    with PdfPages(pdf_path) as pdf:
        for row in rows:
            figure = element_figure(row)
            figure.savefig(output / f"ka{row['ka']}_all_mueller.png", dpi=180)
            pdf.savefig(figure)
            plt.close(figure)


def main() -> int:
    args = parse_args()
    bem_root = args.bem_root.expanduser().resolve()
    adda_root = args.adda_root.expanduser().resolve()
    output = (args.output or bem_root / "report").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = collect(bem_root, adda_root)
    clean = [serializable(row) for row in rows]
    (output / "summary.json").write_text(
        json.dumps(clean, indent=2), encoding="utf-8"
    )
    write_csv(output / "summary.csv", rows)
    write_markdown(output / "report.md", rows)
    write_overview(output / "prism_quick_bem_vs_adda.png", rows)
    write_element_plots(output, rows)
    print(f"Wrote {output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
