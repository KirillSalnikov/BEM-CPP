#!/usr/bin/env python3
"""Plot cold wall time for BEM quick, BEM standard, and saved ADDA runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "prism_quick_standard_vs_existing_adda_ka10_60_ntheta181"
ADDA_ROOT = ROOT / "runs" / "hdiv_bem_vs_adda_sweep_n1p3"
KAS = (10, 15, 20, 25, 30, 60)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def adda_wall_time(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"ACTUAL_WALL_S=([0-9.eE+-]+)", text)
    if not match:
        raise ValueError(f"missing ACTUAL_WALL_S in {path}")
    return float(match.group(1))


def result_summary(directory: Path) -> dict:
    result = load_json(directory / "result.json")
    validation = load_json(directory / "validation.json")
    pfft = result.get("pfft_fgmres") or {}
    if validation.get("errors"):
        raise ValueError(f"invalid result in {directory}: {validation['errors']}")
    return {
        "wall_s": float(validation["wall_time_s"]),
        "ref": int(result["refinements"]),
        "unknowns": int(result["system_dofs"]),
        "iterations_first": int(result["mbj"]["iterations"]),
        "iterations_second": int(result["physical"]["parallel_iterations"]),
        "maximum_residual": max(
            float(result["mbj"]["fmm_residual"]),
            float(result["physical"]["parallel_fmm_residual"]),
        ),
        "theta_points": len(result["physical"]["theta_degrees"]),
        "gpu_memory_mb": float(
            pfft.get(
                "combined_gpu_memory_delta_mb",
                result.get("gpu_memory_delta_mb", 0.0),
            )
        ),
    }


def collect() -> list[dict]:
    rows = []
    for ka in KAS:
        quick = result_summary(RUN_ROOT / f"ka{ka}" / "bem_quick")
        standard_name = (
            "bem_standard_ref6_memory_cap"
            if ka == 60 else "bem_standard_strict_outer"
        )
        standard = result_summary(RUN_ROOT / f"ka{ka}" / standard_name)
        adda = adda_wall_time(
            ADDA_ROOT / f"ka{ka}" / "adda_dpl15" / "time.txt"
        )
        rows.append(
            {
                "ka": ka,
                "quick_wall_s": quick["wall_s"],
                "quick_ref": quick["ref"],
                "quick_unknowns": quick["unknowns"],
                "quick_maximum_residual": quick["maximum_residual"],
                "quick_theta_points": quick["theta_points"],
                "standard_wall_s": standard["wall_s"],
                "standard_ref": standard["ref"],
                "standard_unknowns": standard["unknowns"],
                "standard_maximum_residual": standard["maximum_residual"],
                "standard_theta_points": standard["theta_points"],
                "standard_iterations_first": standard["iterations_first"],
                "standard_iterations_second": standard["iterations_second"],
                "standard_gpu_memory_mb": standard["gpu_memory_mb"],
                "standard_mesh_limited": ka == 60,
                "adda_wall_s": adda,
                "adda_over_quick": adda / quick["wall_s"],
                "adda_over_standard": adda / standard["wall_s"],
                "standard_over_quick": standard["wall_s"] / quick["wall_s"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# BEM quick, BEM standard и ADDA-OCL",
        "",
        (
            "Шестигранная призма, h/D=1, m=1.3, одна ориентация, две "
            "поляризации. BEM quick и standard выводят 181 угол рассеяния. "
            "ADDA-OCL не пересчитывалась: использованы сохраненные расчеты "
            "dpl=15 с 73 углами и внешним временем ACTUAL_WALL_S."
        ),
        "",
        "| ka | quick, с | standard, с | ADDA, с | ADDA/quick | ADDA/standard | standard ref | standard невязка |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        marker = "*" if row["standard_mesh_limited"] else ""
        lines.append(
            f"| {row['ka']} | {row['quick_wall_s']:.2f} | "
            f"{row['standard_wall_s']:.2f} | {row['adda_wall_s']:.2f} | "
            f"{row['adda_over_quick']:.2f} | "
            f"{row['adda_over_standard']:.2f} | "
            f"{row['standard_ref']}{marker} | "
            f"{row['standard_maximum_residual']:.2e} |"
        )
    lines += [
        "",
        (
            "`ADDA/BEM > 1` означает преимущество BEM. Звездочка у ka=60 "
            "обозначает ручной ref=6, ограниченный памятью RTX 3090 Ti. "
            "Автоматический standard требует ref=7, около 3.23 млн "
            "неизвестных и по оценке 88 ГиБ GPU-памяти."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path: Path, rows: list[dict]) -> None:
    ka = [row["ka"] for row in rows]
    quick = [row["quick_wall_s"] for row in rows]
    standard = [row["standard_wall_s"] for row in rows]
    adda = [row["adda_wall_s"] for row in rows]

    figure, axis = plt.subplots(figsize=(14.2, 8.0), constrained_layout=True)
    axis.plot(
        ka, adda, "o-", color="#d97706", linewidth=2.6, markersize=8,
        label=r"ADDA-OCL: dpl=15, невязка $10^{-5}$",
    )
    axis.plot(
        ka, quick, "s-", color="#15803d", linewidth=2.6, markersize=8,
        label=r"BEM quick: 181 угол, невязка $10^{-3}$",
    )
    axis.plot(
        ka, standard, "D-", color="#2563eb", linewidth=2.6, markersize=8,
        label=r"BEM standard: 181 угол, невязка $10^{-5}$",
    )
    axis.plot(
        [60], [standard[-1]], marker="D", markersize=12,
        markerfacecolor="white", markeredgecolor="#2563eb",
        markeredgewidth=2.4, linestyle="none",
    )

    for x, value in zip(ka, quick):
        axis.annotate(
            f"{value:.1f}", (x, value), xytext=(0, -17),
            textcoords="offset points", ha="center", color="#166534",
            fontsize=10,
        )
    for x, value in zip(ka, standard):
        axis.annotate(
            f"{value:.1f}", (x, value), xytext=(0, 9),
            textcoords="offset points", ha="center", color="#1d4ed8",
            fontsize=10,
        )
    for x, value in zip(ka, adda):
        axis.annotate(
            f"{value:.1f}", (x, value), xytext=(9, 0),
            textcoords="offset points", va="center", color="#b45309",
            fontsize=10,
        )

    axis.set_yscale("log")
    axis.set_xticks(ka)
    axis.set_xlabel("Размерный параметр ka", fontsize=13)
    axis.set_ylabel("Полное холодное время, с", fontsize=13)
    axis.set_title(
        "Шестигранная призма: время BEM quick, BEM standard и ADDA-OCL",
        fontsize=17,
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="upper left", fontsize=11)
    axis.text(
        0.99, 0.02,
        (
            "Пустой маркер: standard ka=60 рассчитан на ref=6 из-за 24 ГиБ; "
            "автоматический ref=7 требует около 88 ГиБ.\n"
            "ADDA: сохраненные 73 угла; BEM: 181 угол."
        ),
        transform=axis.transAxes, ha="right", va="bottom", fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#9ca3af", "alpha": 0.92},
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main() -> int:
    output = RUN_ROOT / "report_three_modes"
    output.mkdir(parents=True, exist_ok=True)
    rows = collect()
    (output / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "summary.csv", rows)
    write_markdown(output / "report.md", rows)
    plot(output / "wall_time_three_modes.png", rows)
    print(output / "wall_time_three_modes.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
