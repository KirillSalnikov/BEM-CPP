from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "work_report_assets"
RESULTS = ROOT / "results" / "reference"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def muller_benchmark() -> None:
    data = load(RESULTS / "muller_nodal_fmm_ref2_ka6_n1p5_local_mbj_tol1e-6.json")
    labels = ["Без предобуславливателя", "Morton MBJ"]
    iterations = [data["baseline"]["iterations"], data["mbj"]["iterations"]]
    times = [data["baseline"]["solve_s"], data["mbj"]["solve_s"]]
    colors = ["#376996", "#2f855a"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for axis, values, title, ylabel in (
        (axes[0], iterations, "Число итераций GMRES", "Итерации"),
        (axes[1], times, "Время итерационного решения", "Время, с"),
    ):
        bars = axis.bar(labels, values, color=colors, width=0.62)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=8)
        for bar, value in zip(bars, values):
            text = f"{value:.2f}" if isinstance(value, float) else str(value)
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.015,
                text,
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    figure.suptitle(
        "P2-Мюллер, сфера: ka=6, n=1.5, 2568 комплексных неизвестных, ε=10⁻⁶",
        fontweight="bold",
    )
    figure.savefig(ASSETS / "muller_mbj_benchmark.png", dpi=190)
    plt.close(figure)


def pmchwt_comparison() -> None:
    methods = ["baseline", "local", "ilu0"]
    labels = ["Без", "Локальный", "ILU(0)"]
    ref1 = [load(RESULTS / "pmchwt_ref1" / f"{method}.json") for method in methods]
    ref2 = [load(RESULTS / "pmchwt_ref2" / f"{method}.json") for method in methods]
    actions = np.array(
        [
            [item["gmres_matvecs"] for item in ref1],
            [item["gmres_matvecs"] for item in ref2],
        ]
    )
    times = np.array(
        [
            [item["timing"]["solve_s"] for item in ref1],
            [item["timing"]["solve_s"] for item in ref2],
        ]
    )
    colors = ["#376996", "#d18c2f", "#2f855a"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    x = np.arange(2)
    width = 0.24
    for index, (label, color) in enumerate(zip(labels, colors)):
        offset = (index - 1) * width
        bars = axes[0].bar(x + offset, actions[:, index], width, label=label, color=color)
        axes[1].bar(x + offset, times[:, index], width, label=label, color=color)
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.015,
                f"{bar.get_height():.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    for axis, title, ylabel in (
        (axes[0], "Полные действия FMM-оператора", "Число действий"),
        (axes[1], "Время решения", "Время, с"),
    ):
        axis.set_xticks(x, ["ref=1\n216 RWG", "ref=2\n792 RWG"])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, ncol=3, loc="upper left")
    figure.suptitle(
        "PMCHWT, гексагональная призма: ka=4.6087, n=2.3, ε=10⁻³",
        fontweight="bold",
    )
    figure.savefig(ASSETS / "pmchwt_preconditioners.png", dpi=190)
    plt.close(figure)


def validation_errors() -> None:
    names = [
        "P2-геометрия\nи ядро",
        "Плотный MBJ",
        "FMM против\nпрямой квадратуры",
        "Прямая квадратура\nпротив плотной",
        "Локальный блок\nMBJ",
    ]
    values = [9.445e-6, 3.761e-16, 1.165e-15, 5.268e-16, 1.0e-18]
    colors = ["#d18c2f", "#2f855a", "#376996", "#6b46a5", "#3f8c8c"]
    figure, axis = plt.subplots(figsize=(11.5, 4.3), constrained_layout=True)
    bars = axis.bar(names, values, color=colors, width=0.68)
    axis.set_yscale("log")
    axis.set_ylim(1.0e-19, 1.0e-4)
    axis.set_ylabel("Относительная ошибка")
    axis.set_title("Независимые уровни проверки реализации", fontweight="bold")
    for bar, value in zip(bars, values):
        label = "0 (машинная точность)" if value == 1.0e-18 else f"{value:.2e}"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.7,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=5,
        )
    figure.savefig(ASSETS / "validation_errors.png", dpi=190)
    plt.close(figure)


def pipeline() -> None:
    figure, axis = plt.subplots(figsize=(11.5, 4.5), constrained_layout=True)
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 5)
    axis.axis("off")
    blocks = [
        (0.2, 2.0, 2.0, 1.15, "#dbeafe", "Поверхность\nи параметры\nka, n"),
        (2.7, 2.0, 2.0, 1.15, "#e8f5e9", "P2-узлы,\nкасательный\nбазис"),
        (5.2, 2.0, 2.0, 1.15, "#fff2cc", "FMM:\n∇G и ∇∇G\n+ ближняя зона"),
        (7.7, 2.0, 2.0, 1.15, "#f3e8ff", "Оператор A\nбез полной\nматрицы"),
        (10.2, 2.0, 1.6, 1.15, "#d1fae5", "GMRES\n+ MBJ"),
    ]
    for x, y, width, height, color, text in blocks:
        patch = plt.Rectangle(
            (x, y), width, height, facecolor=color, edgecolor="#334155", linewidth=1.4
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center")
    for left, right in ((2.2, 2.7), (4.7, 5.2), (7.2, 7.7), (9.7, 10.2)):
        axis.annotate(
            "",
            xy=(right, 2.58),
            xytext=(left, 2.58),
            arrowprops={"arrowstyle": "->", "color": "#c2410c", "lw": 2.0},
        )
    axis.text(
        6,
        4.25,
        "Матрично-свободный путь решения уравнения Мюллера второго рода",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    axis.text(
        6,
        0.7,
        "Точная сингулярная квадратура заменяет приближение FMM для совпадающих "
        "и соседних элементов",
        ha="center",
        va="center",
        fontsize=11,
    )
    figure.savefig(ASSETS / "muller_pipeline.png", dpi=190)
    plt.close(figure)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    configure()
    muller_benchmark()
    pmchwt_comparison()
    validation_errors()
    pipeline()
    print(f"Report assets written to {ASSETS}")


if __name__ == "__main__":
    main()
