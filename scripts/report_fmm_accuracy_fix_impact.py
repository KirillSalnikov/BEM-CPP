#!/usr/bin/env python3
"""Summarize the effect of the strict FMM accuracy fix against ADDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_adda_bem_all_mueller import (
    adda_extinction,
    bem_optical_theorem_extinction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adda", type=Path, required=True)
    parser.add_argument("--old-bem", type=Path, required=True)
    parser.add_argument("--new-bem", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_adda(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, skiprows=1)
    return table[:, 0], table[:, 1:].reshape(-1, 4, 4).transpose(1, 2, 0)


def load_bem(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = json.loads(path.read_text())
    physical = data["physical"]
    return (
        np.asarray(physical["theta_degrees"], dtype=float),
        np.asarray(physical["mueller"], dtype=float),
        data,
    )


def common_metrics(
    theta_adda: np.ndarray,
    adda: np.ndarray,
    theta_bem: np.ndarray,
    bem: np.ndarray,
) -> dict:
    common = np.intersect1d(theta_adda, theta_bem)
    adda_indices = np.asarray(
        [np.flatnonzero(np.isclose(theta_adda, value))[0] for value in common]
    )
    bem_indices = np.asarray(
        [np.flatnonzero(np.isclose(theta_bem, value))[0] for value in common]
    )
    adda_common = adda[:, :, adda_indices]
    bem_common = bem[:, :, bem_indices]
    adda_forward = float(adda[0, 0, 0])
    bem_forward = float(bem[0, 0, 0])
    return {
        "common_angle_count": int(common.size),
        "forward_ratio_bem_over_adda": bem_forward / adda_forward,
        "forward_scale_error": abs(bem_forward / adda_forward - 1.0),
        "raw_full_relative_l2": float(
            np.linalg.norm(bem_common - adda_common) / np.linalg.norm(adda_common)
        ),
        "shape_only_full_relative_l2": float(
            np.linalg.norm(
                bem_common / bem_forward - adda_common / adda_forward
            )
            / np.linalg.norm(adda_common / adda_forward)
        ),
    }


def normalized_stokes_factor(ka: float, csca_over_radius_squared: float) -> float:
    return 4.0 * np.pi / (ka**2 * csca_over_radius_squared)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    theta_adda, adda = load_adda(args.adda)
    theta_old, old_bem, old_data = load_bem(args.old_bem)
    theta_new, new_bem, new_data = load_bem(args.new_bem)
    old_metrics = common_metrics(theta_adda, adda, theta_old, old_bem)
    new_metrics = common_metrics(theta_adda, adda, theta_new, new_bem)

    scale = float(adda[0, 0, 0])
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    axes[0].semilogy(
        theta_adda,
        np.maximum(adda[0, 0] / scale, 1.0e-12),
        "o-",
        markersize=3,
        label="ADDA, dpl=15",
    )
    axes[0].semilogy(
        theta_old,
        np.maximum(old_bem[0, 0] / scale, 1.0e-12),
        linewidth=1.4,
        label="BEM до исправления",
    )
    axes[0].semilogy(
        theta_new,
        np.maximum(new_bem[0, 0] / scale, 1.0e-12),
        linewidth=1.4,
        label="BEM после исправления",
    )
    axes[0].set(
        xlabel=r"Угол рассеяния $\theta$, град.",
        ylabel=r"$M_{11}(\theta)/M_{11}^{ADDA}(0)$",
        title="Абсолютная нормировка",
        xlim=(0.0, 180.0),
    )
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    axes[1].semilogy(
        theta_adda,
        np.maximum(adda[0, 0] / adda[0, 0, 0], 1.0e-12),
        "o-",
        markersize=3,
        label="ADDA, dpl=15",
    )
    axes[1].semilogy(
        theta_old,
        np.maximum(old_bem[0, 0] / old_bem[0, 0, 0], 1.0e-12),
        linewidth=1.4,
        label="BEM до исправления",
    )
    axes[1].semilogy(
        theta_new,
        np.maximum(new_bem[0, 0] / new_bem[0, 0, 0], 1.0e-12),
        linewidth=1.4,
        label="BEM после исправления",
    )
    axes[1].set(
        xlabel=r"Угол рассеяния $\theta$, град.",
        ylabel=r"$M_{11}(\theta)/M_{11}(0)$",
        title="Форма угловой зависимости",
        xlim=(0.0, 180.0),
    )
    axes[1].grid(True, which="both", alpha=0.25)

    labels = [
        "Ошибка\nпрямого $M_{11}$",
        "Все элементы,\nобщий масштаб",
        "Все элементы,\nтолько форма",
    ]
    old_values = 100.0 * np.asarray(
        [
            old_metrics["forward_scale_error"],
            old_metrics["raw_full_relative_l2"],
            old_metrics["shape_only_full_relative_l2"],
        ]
    )
    new_values = 100.0 * np.asarray(
        [
            new_metrics["forward_scale_error"],
            new_metrics["raw_full_relative_l2"],
            new_metrics["shape_only_full_relative_l2"],
        ]
    )
    positions = np.arange(len(labels))
    width = 0.36
    axes[2].bar(positions - width / 2, old_values, width, label="До исправления")
    axes[2].bar(positions + width / 2, new_values, width, label="После исправления")
    axes[2].set_yscale("log")
    axes[2].set_xticks(positions, labels)
    axes[2].set_ylabel("Относительное расхождение, %")
    axes[2].set_title("Ошибка на 37 общих углах")
    axes[2].grid(True, axis="y", which="both", alpha=0.25)
    axes[2].legend()
    for position, values, offset in (
        (positions, old_values, -width / 2),
        (positions, new_values, width / 2),
    ):
        for index, value in enumerate(values):
            axes[2].text(
                position[index] + offset,
                value * 1.08,
                f"{value:.3g}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    figure.suptitle(
        "Шестигранная призма: влияние исправления точности FMM",
        fontsize=17,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(args.out_dir / "fmm_accuracy_fix_impact.png", dpi=190)
    plt.close(figure)

    adda_ext = adda_extinction(args.adda.parent, float(new_data["ka"]))
    old_ext = bem_optical_theorem_extinction(old_data)
    new_ext = bem_optical_theorem_extinction(new_data)
    if adda_ext is None or old_ext is None or new_ext is None:
        raise ValueError("normalized Stokes comparison requires extinction data")
    ka = float(new_data["ka"])
    adda_csca = adda_ext["Cext_average_over_equivalent_radius_squared"]
    old_csca = old_ext["Cext_average_over_equivalent_radius_squared"]
    new_csca = new_ext["Cext_average_over_equivalent_radius_squared"]
    adda_f = adda * normalized_stokes_factor(ka, adda_csca)
    old_f = old_bem * normalized_stokes_factor(ka, old_csca)
    new_f = new_bem * normalized_stokes_factor(ka, new_csca)
    old_f_metrics = common_metrics(theta_adda, adda_f, theta_old, old_f)
    new_f_metrics = common_metrics(theta_adda, adda_f, theta_new, new_f)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for axis, element, name in (
        (axes[0], (0, 0), r"$F_{11}$"),
        (axes[1], (1, 1), r"$F_{22}$"),
    ):
        row, column = element
        axis.semilogy(
            theta_adda,
            np.maximum(adda_f[row, column], 1.0e-12),
            "o-",
            markersize=3,
            label="ADDA, dpl=15",
        )
        axis.semilogy(
            theta_old,
            np.maximum(old_f[row, column], 1.0e-12),
            linewidth=1.4,
            label="BEM до исправления",
        )
        axis.semilogy(
            theta_new,
            np.maximum(new_f[row, column], 1.0e-12),
            linewidth=1.4,
            label="BEM после исправления",
        )
        axis.set(
            xlabel=r"Угол рассеяния $\theta$, град.",
            ylabel="Безразмерная величина",
            title=name,
            xlim=(0.0, 180.0),
        )
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend()

    labels = [
        "Прямое\n$F_{11}$",
        "Все элементы\nна общих углах",
        "$C_{sca}$",
    ]
    old_values = 100.0 * np.asarray(
        [
            old_f_metrics["forward_scale_error"],
            old_f_metrics["raw_full_relative_l2"],
            abs(old_csca / adda_csca - 1.0),
        ]
    )
    new_values = 100.0 * np.asarray(
        [
            new_f_metrics["forward_scale_error"],
            new_f_metrics["raw_full_relative_l2"],
            abs(new_csca / adda_csca - 1.0),
        ]
    )
    positions = np.arange(len(labels))
    width = 0.36
    axes[2].bar(positions - width / 2, old_values, width, label="До исправления")
    axes[2].bar(positions + width / 2, new_values, width, label="После исправления")
    axes[2].set_yscale("log")
    axes[2].set_xticks(positions, labels)
    axes[2].set_ylabel("Относительное расхождение, %")
    axes[2].set_title("Стандартная безразмерная нормировка")
    axes[2].grid(True, axis="y", which="both", alpha=0.25)
    axes[2].legend()
    for values, offset in ((old_values, -width / 2), (new_values, width / 2)):
        for index, value in enumerate(values):
            axes[2].text(
                positions[index] + offset,
                value * 1.08,
                f"{value:.3g}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle(
        r"$F_{ij}=4\pi S_{ij}/(k^2C_{\mathrm{sca}})$, "
        r"$\langle F_{11}\rangle_{4\pi}=1$",
        fontsize=17,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(
        args.out_dir / "fmm_accuracy_fix_dimensionless_stokes_F.png",
        dpi=190,
    )
    plt.close(figure)

    report = {
        "case": {"ka": new_data["ka"], "ri": new_data["ri"]},
        "old_fmm": {
            "max_leaf_points": old_data["fmm_max_leaf_points"],
            "metrics": old_metrics,
        },
        "strict_fmm": {
            "max_leaf_points": new_data["fmm_max_leaf_points"],
            "metrics": new_metrics,
            "residuals": {
                "first": new_data["pfft_fgmres"]["fmm_residual"],
                "second": new_data["physical"]["parallel_fmm_residual"],
            },
        },
        "dimensionless_normalized_stokes": {
            "definition": "F_ij = 4*pi*S_ij/(k^2*Csca)",
            "nonabsorbing_assumption": "Csca = Cext",
            "old_metrics": old_f_metrics,
            "strict_metrics": new_f_metrics,
            "csca_over_equivalent_radius_squared": {
                "adda": adda_csca,
                "old_bem": old_csca,
                "strict_bem": new_csca,
            },
        },
    }
    (args.out_dir / "fmm_accuracy_fix_impact.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
