#!/usr/bin/env python3
"""Build an incremental report for the strict BEM/ADDA size sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_nodal_bem_adda import comparison_metrics, load_adda, load_bem


FINE_REFINEMENT = {10: 4, 15: 5, 20: 5, 25: 5, 30: 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs/hdiv_bem_vs_adda_sweep_n1p3"),
    )
    parser.add_argument("--variant", default="batch3_fused")
    parser.add_argument("--output-prefix", default="size_sweep_batch3_fused")
    return parser.parse_args()


def wall_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    match = re.search(
        r"Elapsed \(wall clock\) time .*?:\s*"
        r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)",
        text,
    )
    if not match:
        return None
    return (
        3600.0 * int(match.group(1) or 0)
        + 60.0 * int(match.group(2))
        + float(match.group(3))
    )


def iteration_statistics(path: Path) -> dict[str, float | int | None]:
    if not path.exists():
        return {
            "logged_iterations": 0,
            "median_matvec_s": None,
            "mean_matvec_s": None,
            "total_matvec_s": None,
            "total_preconditioner_s": None,
            "total_orthogonalization_s": None,
        }
    rows = list(csv.DictReader(path.open()))
    iterations = [row for row in rows if row["event"] == "iteration"]
    matvec = np.asarray([float(row["matvec_s"]) for row in iterations])
    preconditioner = np.asarray(
        [float(row["preconditioner_s"]) for row in iterations]
    )
    orthogonalization = np.asarray(
        [float(row["orthogonalization_s"]) for row in iterations]
    )
    return {
        "logged_iterations": len(iterations),
        "median_matvec_s": float(np.median(matvec)) if len(matvec) else None,
        "mean_matvec_s": float(np.mean(matvec)) if len(matvec) else None,
        "total_matvec_s": float(np.sum(matvec)),
        "total_preconditioner_s": float(np.sum(preconditioner)),
        "total_orthogonalization_s": float(np.sum(orthogonalization)),
    }


def collect(root: Path, variant: str) -> list[dict]:
    rows = []
    for ka, fine_ref in FINE_REFINEMENT.items():
        coarse_ref = fine_ref - 1
        directory = root / f"ka{ka}"
        stem = f"bem_ref{fine_ref}_sparse_c6_{variant}"
        coarse_stem = f"bem_ref{coarse_ref}_sparse_c6_{variant}"
        fine_path = directory / f"{stem}.json"
        coarse_path = directory / f"{coarse_stem}.json"
        adda15_path = directory / "adda_dpl15"
        adda20_path = directory / "adda_dpl20"
        if not (
            fine_path.exists()
            and coarse_path.exists()
            and (adda15_path / "mueller").exists()
            and (adda20_path / "mueller").exists()
        ):
            continue

        theta, fine, fine_info = load_bem(
            fine_path, directory / f"{stem}.time"
        )
        _, coarse, _ = load_bem(coarse_path, None)
        adda15, adda15_info = load_adda(adda15_path, theta)
        adda20, adda20_info = load_adda(adda20_path, theta)
        self_metrics = comparison_metrics(theta, fine, coarse)
        adda_self_metrics = comparison_metrics(theta, adda20, adda15)
        adda15_metrics = comparison_metrics(theta, fine, adda15)
        adda20_metrics = comparison_metrics(theta, fine, adda20)
        statistics = iteration_statistics(
            directory / f"{stem}.iterations.csv"
        )

        old_stem = f"bem_ref{fine_ref}_sparse_c6"
        old_wall = wall_seconds(directory / f"{old_stem}.time")
        bem_wall = fine_info["wall_s"]
        adda15_wall = adda15_info.get("process_wall_s")
        adda20_wall = adda20_info.get("process_wall_s")
        rows.append(
            {
                "ka": ka,
                "coarse_ref": coarse_ref,
                "fine_ref": fine_ref,
                "bem_unknowns": fine_info["unknowns"],
                "bem_iterations_first": fine_info[
                    "iterations_first_polarization"
                ],
                "bem_iterations_second": fine_info[
                    "iterations_second_polarization"
                ],
                "bem_wall_s": bem_wall,
                "adda_dpl15_wall_s": adda15_wall,
                "adda_dpl20_wall_s": adda20_wall,
                "bem_over_adda_dpl20_wall": (
                    bem_wall / adda20_wall
                    if bem_wall is not None and adda20_wall
                    else None
                ),
                "old_bem_wall_s": old_wall,
                "batch3_speedup": (
                    old_wall / bem_wall
                    if old_wall is not None and bem_wall
                    else None
                ),
                "bem_refinement_full_percent": 100.0
                * self_metrics["solid_angle_weighted_full_relative_l2"],
                "bem_refinement_M11_percent": 100.0
                * self_metrics["solid_angle_weighted_M11_relative_l2"],
                "adda_refinement_full_percent": 100.0
                * adda_self_metrics["solid_angle_weighted_full_relative_l2"],
                "adda_refinement_M11_percent": 100.0
                * adda_self_metrics["solid_angle_weighted_M11_relative_l2"],
                "bem_adda_dpl15_full_percent": 100.0
                * adda15_metrics["solid_angle_weighted_full_relative_l2"],
                "bem_adda_dpl20_full_percent": 100.0
                * adda20_metrics["solid_angle_weighted_full_relative_l2"],
                "bem_adda_dpl20_M11_percent": 100.0
                * adda20_metrics["solid_angle_weighted_M11_relative_l2"],
                **statistics,
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    ka = np.asarray([row["ka"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), constrained_layout=True)

    axes[0].plot(ka, [row["bem_wall_s"] for row in rows], "o-", label="BEM")
    axes[0].plot(
        ka,
        [row["adda_dpl15_wall_s"] for row in rows],
        "s-",
        label="ADDA, dpl=15",
    )
    axes[0].plot(
        ka,
        [row["adda_dpl20_wall_s"] for row in rows],
        "^-",
        label="ADDA, dpl=20",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Размерный параметр $ka$")
    axes[0].set_ylabel("Полное время, с")
    axes[0].set_title("Время решения")
    axes[0].legend()

    axes[1].plot(
        ka,
        [row["bem_refinement_full_percent"] for row in rows],
        "o-",
        label=r"BEM: $ref_c\to ref_f$",
    )
    axes[1].plot(
        ka,
        [row["bem_adda_dpl20_full_percent"] for row in rows],
        "s-",
        label="BEM и ADDA, dpl=20",
    )
    axes[1].plot(
        ka,
        [row["adda_refinement_full_percent"] for row in rows],
        "^-",
        label="ADDA: dpl=15→20",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Размерный параметр $ka$")
    axes[1].set_ylabel("Относительное отличие, %")
    axes[1].set_title("Сходимость физических результатов")
    axes[1].legend()

    axes[2].plot(
        ka,
        [row["bem_over_adda_dpl20_wall"] for row in rows],
        "o-",
        label="Время BEM / время ADDA",
    )
    speedups = np.asarray(
        [
            np.nan if row["batch3_speedup"] is None else row["batch3_speedup"]
            for row in rows
        ]
    )
    if np.isfinite(speedups).any():
        axes[2].plot(
            ka, speedups, "s-", label="Ускорение пакетного FMM"
        )
    axes[2].axhline(1.0, color="black", lw=1.0, ls="--")
    axes[2].set_xlabel(r"Размерный параметр $ka$")
    axes[2].set_ylabel("Отношение времени")
    axes[2].set_title("Относительная стоимость")
    axes[2].legend()

    for axis in axes:
        axis.grid(True, which="both", alpha=0.28)
        axis.set_xticks(ka)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def load_physical_rows(root: Path, variant: str) -> list[dict]:
    physical_rows = []
    for ka, fine_ref in FINE_REFINEMENT.items():
        directory = root / f"ka{ka}"
        stem = f"bem_ref{fine_ref}_sparse_c6_{variant}"
        bem_path = directory / f"{stem}.json"
        if not bem_path.exists():
            continue
        theta, bem, _ = load_bem(bem_path, None)
        adda15, _ = load_adda(directory / "adda_dpl15", theta)
        adda20, _ = load_adda(directory / "adda_dpl20", theta)
        physical_rows.append(
            {
                "ka": ka,
                "theta": theta,
                "bem": bem,
                "adda15": adda15,
                "adda20": adda20,
            }
        )
    return physical_rows


def plot_all_m11(physical_rows: list[dict], path: Path) -> None:
    if not physical_rows:
        return
    fig, axes = plt.subplots(
        2, 3, figsize=(16.2, 9.4), sharex=True, constrained_layout=True
    )
    for axis, row in zip(axes.flat, physical_rows):
        theta = row["theta"]
        axis.plot(theta, row["bem"][0, 0], color="black", lw=2.1, label="BEM")
        axis.plot(
            theta,
            row["adda15"][0, 0],
            color="#d97706",
            lw=1.4,
            ls="--",
            label="ADDA, dpl=15",
        )
        axis.plot(
            theta,
            row["adda20"][0, 0],
            color="#15803d",
            lw=1.4,
            label="ADDA, dpl=20",
        )
        axis.set_yscale("log")
        axis.set_xlim(0, 180)
        axis.set_title(rf"$ka={row['ka']}$")
        axis.set_xlabel(r"Угол рассеяния $\theta$, град.")
        axis.set_ylabel(r"$M_{11}$")
        axis.grid(True, which="both", alpha=0.24)
    for axis in axes.flat[len(physical_rows):]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=10)
    fig.suptitle(
        "Сравнение BEM и ADDA для всех рассчитанных размеров",
        fontsize=17,
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_all_selected(physical_rows: list[dict], path: Path) -> None:
    if not physical_rows:
        return
    specifications = [
        (0, 0, r"$M_{11}$", True),
        (0, 1, r"$M_{12}/M_{11}$", False),
        (2, 3, r"$M_{34}/M_{11}$", False),
    ]
    fig, axes = plt.subplots(
        len(physical_rows),
        len(specifications),
        figsize=(16.5, 3.25 * len(physical_rows)),
        sharex=True,
        constrained_layout=True,
    )
    for row_index, row in enumerate(physical_rows):
        theta = row["theta"]
        for column, (i, j, title, logarithmic) in enumerate(specifications):
            axis = axes[row_index, column]
            denominator_bem = 1.0 if logarithmic else row["bem"][0, 0]
            denominator_adda15 = (
                1.0 if logarithmic else row["adda15"][0, 0]
            )
            denominator_adda20 = (
                1.0 if logarithmic else row["adda20"][0, 0]
            )
            axis.plot(
                theta,
                row["bem"][i, j] / denominator_bem,
                color="black",
                lw=2.0,
                label="BEM",
            )
            axis.plot(
                theta,
                row["adda15"][i, j] / denominator_adda15,
                color="#d97706",
                lw=1.3,
                ls="--",
                label="ADDA, dpl=15",
            )
            axis.plot(
                theta,
                row["adda20"][i, j] / denominator_adda20,
                color="#15803d",
                lw=1.3,
                label="ADDA, dpl=20",
            )
            if logarithmic:
                axis.set_yscale("log")
            if row_index == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(rf"$ka={row['ka']}$")
            axis.set_xlim(0, 180)
            axis.grid(True, which="both", alpha=0.24)
    for axis in axes[-1]:
        axis.set_xlabel(r"Угол рассеяния $\theta$, град.")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "Элементы матрицы Мюллера: BEM и ADDA для всех размеров",
        fontsize=17,
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = collect(args.root, args.variant)
    output_base = args.root / args.output_prefix
    write_csv(rows, output_base.with_suffix(".csv"))
    output_base.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    )
    plot(rows, output_base.with_suffix(".png"))
    physical_rows = load_physical_rows(args.root, args.variant)
    plot_all_m11(
        physical_rows,
        args.root / "bem_vs_adda_all_ka_M11.png",
    )
    plot_all_selected(
        physical_rows,
        args.root / "bem_vs_adda_all_ka_selected.png",
    )
    print(f"Collected {len(rows)} completed ka points")
    for row in rows:
        print(
            f"ka={row['ka']}: BEM {row['bem_wall_s']:.2f}s, "
            f"ADDA dpl20 {row['adda_dpl20_wall_s']:.2f}s, "
            f"BEM self {row['bem_refinement_full_percent']:.3f}%"
        )


if __name__ == "__main__":
    main()
