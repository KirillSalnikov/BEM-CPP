#!/usr/bin/env python3
"""Build matched BEM/ADDA orientation-sweep reports, optionally with Mie."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_mie import mie_mueller


def wall_seconds(path: Path) -> float:
    match = re.search(
        r"ACTUAL_WALL_S=([0-9.]+)",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    if not match:
        raise ValueError(f"wall time is missing in {path}")
    return float(match.group(1))


def adda_timing(path: Path) -> tuple[float, float, int]:
    text = path.read_text(encoding="utf-8", errors="replace")

    def number(pattern: str) -> float:
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"{pattern!r} is missing in {path}")
        return float(match.group(1))

    return (
        number(r"Internal fields:\s+([0-9.]+)"),
        number(r"Scattered fields:\s+([0-9.]+)"),
        int(number(r"Total number of iterations:\s+([0-9]+)")),
    )


def weighted_relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    theta = np.linspace(0.0, np.pi, reference.shape[-1])
    weights = np.sin(theta)
    weights[[0, -1]] = 0.0
    shaped = weights.reshape((1,) * (reference.ndim - 1) + (-1,))
    denominator = np.sum(shaped * np.square(reference))
    if denominator <= 1.0e-300:
        return float("nan")
    return float(
        np.sqrt(
            np.sum(shaped * np.square(candidate - reference))
            / denominator
        )
    )


def normalized_forward_error(
    reference: np.ndarray, candidate: np.ndarray
) -> float:
    scale = abs(float(reference[0, 0, 0]))
    theta = np.linspace(0.0, np.pi, reference.shape[-1])
    weights = np.sin(theta)
    weights[[0, -1]] = 0.0
    difference = (candidate - reference) / max(scale, 1.0e-300)
    return float(
        np.sqrt(
            np.sum(weights[None, None, :] * difference**2)
            / np.sum(weights)
        )
    )


def load_case(root: Path, ka: int, with_mie: bool) -> dict[str, object]:
    case_root = root / f"ka{ka}"
    bem_candidates = [
        case_root / "bem_ref5_alpha256",
        case_root / "bem_ref5_pfft01_alpha256_pairff_warm",
        case_root / "bem_ref5_pfft01_alpha256_pairff",
    ]
    bem_dir = next(
        (
            candidate
            for candidate in bem_candidates
            if (candidate / "average.json").is_file()
        ),
        bem_candidates[0],
    )
    adda_dir = case_root / "adda_dpl15_alpha256"
    bem = json.loads((bem_dir / "average.json").read_text(encoding="utf-8"))
    adda_table = np.loadtxt(adda_dir / "mueller", skiprows=1)
    theta = np.asarray(bem["theta_degrees"], dtype=np.float64)
    if not np.allclose(theta, adda_table[:, 0], atol=1.0e-12, rtol=0.0):
        raise ValueError(f"scattering grids differ for ka={ka}")
    bem_mueller = np.asarray(bem["mueller"], dtype=np.float64)
    adda_mueller = adda_table[:, 1:].reshape((-1, 4, 4)).transpose(1, 2, 0)
    mie = (
        np.asarray(mie_mueller(theta, complex(1.3, 0.0), float(ka)))
        if with_mie
        else None
    )
    internal_s, scattered_s, adda_iterations = adda_timing(adda_dir / "log")
    return {
        "ka": ka,
        "theta": theta,
        "bem": bem_mueller,
        "adda": adda_mueller,
        "mie": mie,
        "bem_json": bem,
        "bem_wall_s": wall_seconds(bem_dir / "time.txt"),
        "adda_wall_s": wall_seconds(adda_dir / "time.txt"),
        "adda_internal_s": internal_s,
        "adda_scattered_s": scattered_s,
        "adda_iterations": adda_iterations,
        "root": case_root,
    }


def plot_all_mueller(case: dict[str, object], shape_label: str) -> None:
    ka = int(case["ka"])
    theta = np.asarray(case["theta"])
    bem = np.asarray(case["bem"])
    adda = np.asarray(case["adda"])
    mie = case["mie"]
    curves = [
        ("BEM ref=5", bem / bem[0, 0, 0], "#16865c", "-"),
        ("ADDA dpl=15", adda / adda[0, 0, 0], "#2676b8", "--"),
    ]
    if mie is not None:
        mie_array = np.asarray(mie)
        curves.append(
            ("теория Ми", mie_array / mie_array[0, 0, 0], "#111111", ":")
        )

    figure, axes = plt.subplots(4, 4, figsize=(16.0, 12.5), sharex=True)
    for row in range(4):
        for column in range(4):
            axis = axes[row, column]
            for label, matrix, color, linestyle in curves:
                axis.plot(
                    theta,
                    matrix[row, column],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=label,
                )
            axis.set_yscale("symlog", linthresh=1.0e-7, linscale=0.6)
            axis.set_title(rf"$M_{{{row + 1}{column + 1}}}$")
            axis.grid(which="both", alpha=0.22)
            if row == 3:
                axis.set_xlabel(r"$\theta$, град.")
            if column == 0:
                axis.set_ylabel(r"$M_{ij}/M_{11}(0)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(curves),
    )
    figure.suptitle(
        f"{shape_label}: все элементы матрицы Мюллера, "
        rf"$ka={ka}$, $m=1.3$, $N_\alpha=256$, невязка $10^{{-5}}$"
        "\nКаждый метод нормирован на собственное "
        r"$M_{11}(0)$; симметричная логарифмическая шкала",
        fontsize=15,
        y=0.997,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    filename = (
        "all_mueller_bem_adda_mie.png"
        if mie is not None
        else "all_mueller_bem_adda.png"
    )
    figure.savefig(
        Path(case["root"]) / filename,
        dpi=180,
    )
    plt.close(figure)


def summarize(case: dict[str, object]) -> dict[str, float | int]:
    bem = np.asarray(case["bem"])
    adda = np.asarray(case["adda"])
    mie = case["mie"]
    bem_wall = float(case["bem_wall_s"])
    adda_wall = float(case["adda_wall_s"])
    bem_json = case["bem_json"]
    row: dict[str, float | int] = {
        "ka": int(case["ka"]),
        "bem_wall_s": bem_wall,
        "adda_wall_s": adda_wall,
        "speedup_adda_over_bem": adda_wall / bem_wall,
        "bem_setup_s": float(bem_json["timing"]["operator_setup_s"])
        + float(bem_json["timing"]["mbj_setup_s"])
        + float(bem_json["timing"]["fmm_switch_s"]),
        "bem_solve_s": float(bem_json["timing"]["solve_s"]),
        "bem_farfield_s": float(bem_json["timing"]["farfield_s"]),
        "adda_solve_s": float(case["adda_internal_s"]),
        "adda_farfield_s": float(case["adda_scattered_s"]),
        "bem_outer_iterations": int(bem_json["iterations"]["total"]),
        "bem_inner_iterations": int(bem_json["pfft_inner"]["iterations"]),
        "bem_residual": float(bem_json["iterations"]["maximum_residual"]),
        "adda_iterations": int(case["adda_iterations"]),
        "bem_vs_adda_normalized_l2": weighted_relative_l2(
            adda / adda[0, 0, 0], bem / bem[0, 0, 0]
        ),
        "bem_vs_adda_forward_scaled_rms": normalized_forward_error(adda, bem),
    }
    if mie is not None:
        mie_array = np.asarray(mie)
        row.update(
            {
                "bem_vs_mie_full_relative_l2": weighted_relative_l2(
                    mie_array, bem
                ),
                "adda_vs_mie_full_relative_l2": weighted_relative_l2(
                    mie_array, adda
                ),
                "bem_vs_mie_forward_scaled_rms": normalized_forward_error(
                    mie_array, bem
                ),
                "adda_vs_mie_forward_scaled_rms": normalized_forward_error(
                    mie_array, adda
                ),
                "bem_forward_m11_ratio_to_mie": float(
                    bem[0, 0, 0] / mie_array[0, 0, 0]
                ),
                "adda_forward_m11_ratio_to_mie": float(
                    adda[0, 0, 0] / mie_array[0, 0, 0]
                ),
            }
        )
    return row


def plot_sweep(
    output: Path,
    rows: list[dict[str, float | int]],
    shape_label: str,
    with_mie: bool,
    gamma_degrees: float,
) -> None:
    ka = np.array([row["ka"] for row in rows])
    bem_time = np.array([row["bem_wall_s"] for row in rows])
    adda_time = np.array([row["adda_wall_s"] for row in rows])
    speedup = adda_time / bem_time
    columns = 3 if with_mie else 2
    figure, axes = plt.subplots(1, columns, figsize=(6.0 * columns, 5.0))
    axes = np.atleast_1d(axes)
    axes[0].plot(ka, bem_time, "o-", color="#16865c", label="BEM ref=5")
    axes[0].plot(ka, adda_time, "s--", color="#2676b8", label="ADDA dpl=15")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Размерный параметр $ka$")
    axes[0].set_ylabel("Полное стеночное время, с")
    axes[0].set_title("Время одинакового усреднения")
    axes[0].legend()
    axes[0].grid(which="both", alpha=0.25)

    axes[1].plot(ka, speedup, "o-", color="#9a5b13")
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.2)
    axes[1].set_xlabel(r"Размерный параметр $ka$")
    axes[1].set_ylabel(r"$T_{\mathrm{ADDA}}/T_{\mathrm{BEM}}$")
    axes[1].set_title("Ускорение BEM относительно ADDA")
    axes[1].grid(alpha=0.25)
    for x, value in zip(ka, speedup):
        axes[1].annotate(
            f"{value:.2f}",
            (x, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )

    if with_mie:
        bem_error = 100.0 * np.array(
            [row["bem_vs_mie_forward_scaled_rms"] for row in rows]
        )
        adda_error = 100.0 * np.array(
            [row["adda_vs_mie_forward_scaled_rms"] for row in rows]
        )
        axes[2].semilogy(
            ka, bem_error, "o-", color="#16865c", label="BEM ref=5"
        )
        axes[2].semilogy(
            ka, adda_error, "s--", color="#2676b8", label="ADDA dpl=15"
        )
        axes[2].set_xlabel(r"Размерный параметр $ka$")
        axes[2].set_ylabel(
            r"СКО всех элементов относительно Ми / $M_{11}^{\rm Ми}(0)$, %"
        )
        axes[2].set_title("Физическая точность")
        axes[2].legend()
        axes[2].grid(which="both", alpha=0.25)
    figure.suptitle(
        f"{shape_label}: $m=1.3$, $N_\\alpha=256$, "
        rf"$\beta=90^\circ$, $\gamma={gamma_degrees:g}^\circ$, "
        r"невязка $10^{-5}$",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(output / "sweep_time_speedup_accuracy.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shape-label", required=True)
    parser.add_argument("--ka", type=int, nargs="+", required=True)
    parser.add_argument("--with-mie", action="store_true")
    parser.add_argument("--gamma", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = [load_case(args.root, ka, args.with_mie) for ka in args.ka]
    rows = [summarize(case) for case in cases]
    for case in cases:
        plot_all_mueller(case, args.shape_label)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    with (args.root / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_sweep(
        args.root, rows, args.shape_label, args.with_mie, args.gamma
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
