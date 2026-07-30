#!/usr/bin/env python3
"""Compare BEM and ADDA after each method passes the same self-convergence gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bem-ref4",
        type=Path,
        default=Path("runs/adda_exact/ref4_ka20_n1p5_pfft_auto.json"),
    )
    parser.add_argument(
        "--bem-ref5",
        type=Path,
        default=Path(
            "runs/muller_pfft/ref5_ka20_pfft_fgmres_auto_cached.json"
        ),
    )
    parser.add_argument(
        "--bem-ref6",
        type=Path,
        default=Path(
            "runs/adda_exact/ref6_ka20_n1p5_pfft_auto_warm.json"
        ),
    )
    parser.add_argument(
        "--bem-ref6-warm-log",
        type=Path,
        default=Path(
            "runs/adda_exact/ref6_ka20_n1p5_pfft_auto_warm.log"
        ),
    )
    parser.add_argument(
        "--bem-ref6-cold-log",
        type=Path,
        default=Path("runs/adda_exact/ref6_ka20_n1p5_pfft_auto.log"),
    )
    parser.add_argument(
        "--adda-dpl15",
        type=Path,
        default=Path(
            "runs/adda_exact/ka20_n1p5_aspect1_dpl15_alpha15"
        ),
    )
    parser.add_argument(
        "--adda-dpl20",
        type=Path,
        default=Path(
            "runs/adda_exact/ka20_n1p5_aspect1_dpl20_alpha15_qmr2"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/adda_exact/self_converged_1pct_report"),
    )
    return parser.parse_args()


def load_bem(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = json.loads(path.read_text())
    theta = np.asarray(data["physical"]["theta_degrees"], dtype=float)
    mueller = np.asarray(data["physical"]["mueller"], dtype=float)
    return theta, mueller, data


def load_adda(directory: Path) -> tuple[np.ndarray, np.ndarray, str]:
    table = np.loadtxt(directory / "mueller", skiprows=1)
    log = (directory / "log").read_text(errors="replace")
    return (
        table[:, 0],
        table[:, 1:].reshape(-1, 4, 4).transpose(1, 2, 0),
        log,
    )


def wall_from_log(path: Path) -> float:
    text = path.read_text(errors="replace")
    match = re.search(r"^ACTUAL_WALL_S=(\d+(?:\.\d+)?)$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"ACTUAL_WALL_S is missing from {path}")
    return float(match.group(1))


def adda_wall(log: str) -> float:
    match = re.search(r"Total wall time:\s*([0-9.eE+-]+)", log)
    if not match:
        raise ValueError("ADDA wall time is missing")
    return float(match.group(1))


def weighted_relative(
    theta: np.ndarray, coarse: np.ndarray, fine: np.ndarray
) -> float:
    weights = np.sin(np.deg2rad(theta))
    while weights.ndim < fine.ndim:
        weights = weights[None, ...]
    numerator = np.sum(weights * np.abs(coarse - fine) ** 2)
    denominator = np.sum(weights * np.abs(fine) ** 2)
    return float(np.sqrt(numerator / denominator))


def convergence_pair(
    theta: np.ndarray, coarse: np.ndarray, fine: np.ndarray
) -> dict:
    return {
        "m11": weighted_relative(theta, coarse[0, 0], fine[0, 0]),
        "full_mueller": weighted_relative(theta, coarse, fine),
        "forward_m11": float(
            abs(coarse[0, 0, 0] - fine[0, 0, 0])
            / abs(fine[0, 0, 0])
        ),
    }


def main() -> None:
    args = parse_args()
    theta4, bem4, data4 = load_bem(args.bem_ref4)
    theta5, bem5, data5 = load_bem(args.bem_ref5)
    theta6, bem6, data6 = load_bem(args.bem_ref6)
    theta15, adda15, log15 = load_adda(args.adda_dpl15)
    theta20, adda20, log20 = load_adda(args.adda_dpl20)

    for label, theta in (
        ("BEM ref4", theta4),
        ("BEM ref5", theta5),
        ("ADDA dpl15", theta15),
        ("ADDA dpl20", theta20),
    ):
        if len(theta) != len(theta6) or not np.allclose(theta, theta6):
            raise ValueError(f"{label} angle grid does not match BEM ref6")

    bem45 = convergence_pair(theta6, bem4, bem5)
    bem56 = convergence_pair(theta6, bem5, bem6)
    adda1520 = convergence_pair(theta6, adda15, adda20)

    bem_warm_wall = wall_from_log(args.bem_ref6_warm_log)
    bem_cold_wall = wall_from_log(args.bem_ref6_cold_log)
    adda_selected_wall = adda_wall(log20)
    speedup_warm = bem_warm_wall / adda_selected_wall
    speedup_cold = bem_cold_wall / adda_selected_wall

    bem_setup = (
        float(data6["fmm_setup_s"])
        + float(data6["mbj_local_setup_s"])
        + float(data6["pfft_fgmres"]["fmm_switch_setup_s"])
    )
    bem_axis_per_orientation = bem_warm_wall - bem_setup
    adda_setup_match = re.search(
        r"Initialization time:\s*([0-9.eE+-]+)", log20
    )
    adda_setup = float(adda_setup_match.group(1)) if adda_setup_match else 0.0
    adda_per_orientation = adda_selected_wall - adda_setup

    cross_m11 = weighted_relative(
        theta6,
        bem6[0, 0] / bem6[0, 0, 0],
        adda20[0, 0] / adda20[0, 0, 0],
    )
    cross_full = weighted_relative(
        theta6,
        bem6 / bem6[0, 0, 0],
        adda20 / adda20[0, 0, 0],
    )
    forward_ratio = float(bem6[0, 0, 0] / adda20[0, 0, 0])

    summary = {
        "criterion": {
            "name": "solid-angle-weighted relative L2 change to previous grid",
            "threshold": args.threshold,
            "formula": "sqrt(sum(sin(theta)*|fine-coarse|^2)/sum(sin(theta)*|fine|^2))",
        },
        "case": {
            "shape": "regular hexagonal prism",
            "aspect_h_over_D": 1.0,
            "ka": 20.0,
            "refractive_index": 1.5,
            "solver_tolerance": 1.0e-5,
            "azimuth_degrees": 15.0,
        },
        "convergence": {
            "bem_ref4_to_ref5": bem45,
            "bem_ref5_to_ref6": bem56,
            "adda_dpl15_to_dpl20": adda1520,
        },
        "selected": {
            "bem": {
                "grid": "ref=6",
                "unknowns": data6["system_dofs"],
                "passes": max(bem56["m11"], bem56["full_mueller"])
                <= args.threshold,
                "warm_wall_s": bem_warm_wall,
                "cold_wall_s": bem_cold_wall,
            },
            "adda": {
                "grid": "dpl=20",
                "occupied_dipoles": int(
                    re.search(
                        r"Total number of occupied dipoles:\s*(\d+)", log20
                    ).group(1)
                ),
                "passes": max(adda1520["m11"], adda1520["full_mueller"])
                <= args.threshold,
                "wall_s": adda_selected_wall,
            },
        },
        "timing": {
            "adda_speedup_vs_bem_warm": speedup_warm,
            "adda_speedup_vs_bem_cold": speedup_cold,
            "amortized_axis_orientation_s": {
                "bem": bem_axis_per_orientation,
                "adda": adda_per_orientation,
                "adda_speedup": bem_axis_per_orientation
                / adda_per_orientation,
            },
        },
        "cross_method_check_not_used_for_selection": {
            "forward_normalized_m11_relative": cross_m11,
            "forward_normalized_full_mueller_relative": cross_full,
            "forward_m11_bem_over_adda": forward_ratio,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "self_converged_comparison.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5))
    fig.suptitle(
        "BEM и ADDA при одинаковом критерии собственной сходимости",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        (
            r"$ka=20$, $m=1.5$, $h/D=1$, "
            r"невязка решателя $10^{-5}$; порог изменения сетки 1%"
        ),
        ha="center",
        fontsize=14,
    )

    ax = axes[0]
    pair_labels = ["BEM\nref 4→5", "BEM\nref 5→6", "ADDA\ndpl 15→20"]
    m11_values = np.array(
        [bem45["m11"], bem56["m11"], adda1520["m11"]]
    ) * 100.0
    full_values = np.array(
        [
            bem45["full_mueller"],
            bem56["full_mueller"],
            adda1520["full_mueller"],
        ]
    ) * 100.0
    x = np.arange(3)
    width = 0.36
    ax.bar(
        x - width / 2,
        m11_values,
        width,
        color="#4f8f5b",
        label=r"$M_{11}$",
    )
    ax.bar(
        x + width / 2,
        full_values,
        width,
        color="#4f75b5",
        label="Вся матрица",
    )
    ax.axhline(
        100.0 * args.threshold,
        color="#a12830",
        linewidth=2,
        linestyle="--",
        label="Порог 1%",
    )
    for xpos, value in zip(x - width / 2, m11_values):
        ax.text(xpos, value + 0.08, f"{value:.3f}%", ha="center", fontsize=10)
    for xpos, value in zip(x + width / 2, full_values):
        ax.text(xpos, value + 0.08, f"{value:.3f}%", ha="center", fontsize=10)
    ax.set_xticks(x, pair_labels)
    ax.set_ylabel("Изменение относительно следующей сетки, %")
    ax.set_title("Самостоятельная сходимость")
    ax.set_ylim(0.0, max(full_values) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=10)

    ax = axes[1]
    labels = ["ADDA\nQMR2, dpl=20", "BEM\nref=6, кэш", "BEM\nref=6, первый запуск"]
    times = [adda_selected_wall, bem_warm_wall, bem_cold_wall]
    colors = ["#4f8f5b", "#dd8a32", "#a6533d"]
    bars = ax.bar(np.arange(3), times, color=colors, width=0.66)
    bars[2].set_hatch("//")
    for index, value in enumerate(times):
        ax.text(
            index,
            value + 25,
            f"{value:.1f} с",
            ha="center",
            fontweight="bold",
        )
    ax.text(
        0.5,
        0.80,
        f"ADDA быстрее тёплого BEM в {speedup_warm:.2f} раза",
        transform=ax.transAxes,
        ha="center",
        bbox={"facecolor": "white", "edgecolor": "#555555", "alpha": 0.92},
    )
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Полное время одной осевой ориентации, с")
    ax.set_title("Первые сетки, прошедшие порог 1%")
    ax.set_ylim(0.0, max(times) * 1.18)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    ax.semilogy(
        theta6,
        bem6[0, 0] / bem6[0, 0, 0],
        color="#d9781e",
        linewidth=2.5,
        label="BEM ref=6",
    )
    ax.semilogy(
        theta6,
        adda20[0, 0] / adda20[0, 0, 0],
        color="#3d7f4a",
        linewidth=2.5,
        label="ADDA dpl=20",
    )
    ax.set_xlabel(r"Угол рассеяния $\theta$, град.")
    ax.set_ylabel(r"$M_{11}(\theta)/M_{11}(0)$")
    ax.set_title("Проверка результата после сходимости")
    ax.grid(which="both", alpha=0.25)
    ax.legend()
    ax.text(
        0.04,
        0.05,
        (
            f"Различие нормированных $M_{{11}}$: {100*cross_m11:.2f}%\n"
            f"$M_{{11}}(0)$ BEM/ADDA: {forward_ratio:.4f}"
        ),
        transform=ax.transAxes,
        bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.92},
    )

    fig.text(
        0.5,
        0.015,
        (
            "Выбор сетки основан только на сходимости каждого метода "
            "относительно самого себя; межметодное различие показано отдельно."
        ),
        ha="center",
        fontsize=12,
    )
    fig.subplots_adjust(
        left=0.06, right=0.985, top=0.86, bottom=0.13, wspace=0.25
    )
    png_path = args.out_dir / "self_converged_bem_vs_adda.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    report_path = args.out_dir / "README.md"
    report_path.write_text(
        "\n".join(
            [
                "# BEM и ADDA при одинаковом критерии собственной сходимости",
                "",
                (
                    "Критерий: взвешенная по `sin(theta)` относительная "
                    "L2-разность результата на соседних сетках не более 1%."
                ),
                "",
                "| Метод | Проверка сетки | M11 | Вся матрица | Выбранная сетка | Wall time |",
                "|---|---|---:|---:|---|---:|",
                (
                    f"| BEM pFFT-FGMRES | ref=5 -> 6 | "
                    f"{100*bem56['m11']:.3f}% | "
                    f"{100*bem56['full_mueller']:.3f}% | ref=6 | "
                    f"{bem_warm_wall:.2f} s (cache), {bem_cold_wall:.2f} s cold |"
                ),
                (
                    f"| ADDA-OCL QMR2 | dpl=15 -> 20 | "
                    f"{100*adda1520['m11']:.3f}% | "
                    f"{100*adda1520['full_mueller']:.3f}% | dpl=20 | "
                    f"{adda_selected_wall:.2f} s |"
                ),
                "",
                (
                    f"При готовом кэше ADDA быстрее BEM в "
                    f"**{speedup_warm:.2f} раза**. Если включить первичное "
                    f"построение кэша BEM, разница составляет "
                    f"**{speedup_cold:.2f} раза**."
                ),
                "",
                (
                    "Это сравнение при одинаковом пороге самостоятельной "
                    "сходимости, а не доказательство совпадения методов. "
                    f"После нормировки на M11(0) межметодная разность M11 "
                    f"равна {100*cross_m11:.2f}%."
                ),
                "",
            ]
        )
    )
    print(png_path)
    print(report_path)
    print(json_path)


if __name__ == "__main__":
    main()
