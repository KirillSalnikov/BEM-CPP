#!/usr/bin/env python3
"""Compare strict BEM and ADDA Mueller data without angular interpolation."""

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
    parser.add_argument("--bem", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title", required=True)
    return parser.parse_args()


def load_adda(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, skiprows=1)
    theta = table[:, 0]
    mueller = table[:, 1:].reshape(-1, 4, 4).transpose(1, 2, 0)
    return theta, mueller


def load_bem(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = json.loads(path.read_text())
    theta = np.asarray(data["physical"]["theta_degrees"], dtype=float)
    mueller = np.asarray(data["physical"]["mueller"], dtype=float)
    return theta, mueller, data


def polar_plane_integral(theta_degrees: np.ndarray, values: np.ndarray) -> float:
    theta_radians = np.deg2rad(theta_degrees)
    return float(
        2.0
        * np.pi
        * np.trapezoid(values * np.sin(theta_radians), theta_radians)
    )


def common_indices(
    theta_adda: np.ndarray,
    theta_bem: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = np.intersect1d(theta_adda, theta_bem)
    adda_indices = np.asarray(
        [np.flatnonzero(np.isclose(theta_adda, value))[0] for value in common]
    )
    bem_indices = np.asarray(
        [np.flatnonzero(np.isclose(theta_bem, value))[0] for value in common]
    )
    return common, adda_indices, bem_indices


def plot_all_elements(
    theta_adda: np.ndarray,
    adda: np.ndarray,
    theta_bem: np.ndarray,
    bem: np.ndarray,
    scale: float,
    title: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 15), sharex=True)
    for row in range(4):
        for column in range(4):
            axis = axes[row, column]
            axis.plot(
                theta_adda,
                adda[row, column] / scale,
                color="#1769aa",
                linewidth=2.0,
                marker="o",
                markersize=2.5,
                label="ADDA, шаг 2,5°",
            )
            axis.plot(
                theta_bem,
                bem[row, column] / scale,
                color="#d95f02",
                linewidth=1.4,
                label="BEM, шаг 1°",
            )
            axis.set_yscale("symlog", linthresh=1.0e-8)
            axis.grid(True, alpha=0.25)
            axis.set_title(
                rf"$M_{{{row + 1}{column + 1}}}/M_{{11}}^{{ADDA}}(0)$"
            )
            if row == 3:
                axis.set_xlabel(r"Угол рассеяния $\theta$, град.")
            if column == 0:
                axis.set_ylabel("Общая нормировка")
    axes[0, 0].legend(loc="best")
    fig.suptitle(title, fontsize=20)
    fig.text(
        0.5,
        0.012,
        (
            "Линии показаны на исходных угловых сетках; "
            "интерполяция не применялась."
        ),
        ha="center",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_normalized_stokes_matrix(
    theta_adda: np.ndarray,
    adda: np.ndarray,
    theta_bem: np.ndarray,
    bem: np.ndarray,
    title: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 15), sharex=True)
    for row in range(4):
        for column in range(4):
            axis = axes[row, column]
            axis.plot(
                theta_adda,
                adda[row, column],
                color="#1769aa",
                linewidth=2.0,
                marker="o",
                markersize=2.5,
                label="ADDA, шаг 2,5°",
            )
            axis.plot(
                theta_bem,
                bem[row, column],
                color="#d95f02",
                linewidth=1.4,
                label="BEM, шаг 1°",
            )
            axis.set_yscale("symlog", linthresh=1.0e-8)
            axis.grid(True, alpha=0.25)
            axis.set_title(rf"$F_{{{row + 1}{column + 1}}}$")
            if row == 3:
                axis.set_xlabel(r"Угол рассеяния $\theta$, град.")
            if column == 0:
                axis.set_ylabel("Безразмерная величина")
    axes[0, 0].legend(loc="best")
    fig.suptitle(title, fontsize=20)
    fig.text(
        0.5,
        0.012,
        (
            r"$F_{ij}=4\pi S_{ij}/(k^2 C_{\mathrm{sca}})$; "
            r"$\langle F_{11}\rangle_{4\pi}=1$. "
            "Интерполяция углов не применялась."
        ),
        ha="center",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_common_error(
    theta: np.ndarray,
    adda: np.ndarray,
    bem: np.ndarray,
    scale: float,
    output: Path,
) -> np.ndarray:
    maximum = np.max(np.abs(bem - adda), axis=2) / abs(scale)
    figure, (axis_curve, axis_map) = plt.subplots(
        1,
        2,
        figsize=(14, 5.8),
        gridspec_kw={"width_ratios": [1.55, 1]},
    )
    axis_curve.semilogy(
        theta,
        np.maximum(adda[0, 0] / scale, 1.0e-300),
        "o-",
        color="#1769aa",
        label="ADDA",
    )
    axis_curve.semilogy(
        theta,
        np.maximum(bem[0, 0] / scale, 1.0e-300),
        "s--",
        color="#d95f02",
        label="BEM",
    )
    axis_curve.set_xlabel(r"Общий угол $\theta$, град.")
    axis_curve.set_ylabel(r"$M_{11}(\theta)/M_{11}^{ADDA}(0)$")
    axis_curve.set_title("Сравнение в точно совпадающих углах")
    axis_curve.grid(True, which="both", alpha=0.25)
    axis_curve.legend()

    image = axis_map.imshow(
        np.maximum(maximum, 1.0e-16),
        cmap="magma",
        norm=matplotlib.colors.LogNorm(),
    )
    for row in range(4):
        for column in range(4):
            value = maximum[row, column]
            color = "black" if value > np.sqrt(maximum.min() * maximum.max()) else "white"
            axis_map.text(
                column,
                row,
                f"{value:.2e}",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )
    axis_map.set_xticks(range(4), [f"столбец {index}" for index in range(1, 5)])
    axis_map.set_yticks(range(4), [f"строка {index}" for index in range(1, 5)])
    axis_map.set_title("Максимальная разность в общих углах")
    figure.colorbar(
        image,
        ax=axis_map,
        label=r"$\max_\theta|M_{ij}^{BEM}-M_{ij}^{ADDA}|/M_{11}^{ADDA}(0)$",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)
    return maximum


def main() -> None:
    args = parse_args()
    theta_adda, adda = load_adda(args.adda)
    theta_bem, bem, bem_data = load_bem(args.bem)
    common, adda_indices, bem_indices = common_indices(theta_adda, theta_bem)
    if common.size < 2:
        raise ValueError("the ADDA and BEM angular grids have no useful overlap")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scale = float(adda[0, 0, 0])
    adda_common = adda[:, :, adda_indices]
    bem_common = bem[:, :, bem_indices]
    maximum = plot_common_error(
        common,
        adda_common,
        bem_common,
        scale,
        args.out_dir / "strict_bem_vs_adda_common_angles.png",
    )
    plot_all_elements(
        theta_adda,
        adda,
        theta_bem,
        bem,
        scale,
        args.title,
        args.out_dir / "strict_bem_vs_adda_all_mueller_native_grids.png",
    )

    adda_ext = adda_extinction(args.adda.parent, float(bem_data["ka"]))
    bem_ext = bem_optical_theorem_extinction(bem_data)
    extinction_audit = {
        "adda": adda_ext,
        "bem_optical_theorem": bem_ext,
    }
    if adda_ext is not None and bem_ext is not None:
        adda_value = adda_ext["Cext_average_over_equivalent_radius_squared"]
        bem_value = bem_ext["Cext_average_over_equivalent_radius_squared"]
        extinction_audit["bem_over_adda"] = bem_value / adda_value
        extinction_audit["relative_difference"] = (
            abs(bem_value - adda_value) / abs(adda_value)
        )
        ka = float(bem_data["ka"])
        adda_stokes_factor = 4.0 * np.pi / (ka**2 * adda_value)
        bem_stokes_factor = 4.0 * np.pi / (ka**2 * bem_value)
        adda_stokes = adda * adda_stokes_factor
        bem_stokes = bem * bem_stokes_factor
        plot_normalized_stokes_matrix(
            theta_adda,
            adda_stokes,
            theta_bem,
            bem_stokes,
            (
                "Нормированная безразмерная матрица рассеяния: "
                "строгий BEM и ADDA"
            ),
            (
                args.out_dir
                / "strict_bem_vs_adda_dimensionless_normalized_stokes_F.png"
            ),
        )
        stokes_common_adda = adda_stokes[:, :, adda_indices]
        stokes_common_bem = bem_stokes[:, :, bem_indices]
        normalized_stokes_audit = {
            "definition": "F_ij = 4*pi*S_ij/(k^2*Csca)",
            "normalization": "solid-angle average of F11 equals one",
            "nonabsorbing_assumption": "Csca = Cext",
            "adda_factor": adda_stokes_factor,
            "bem_factor": bem_stokes_factor,
            "forward_F11_adda": float(adda_stokes[0, 0, 0]),
            "forward_F11_bem": float(bem_stokes[0, 0, 0]),
            "forward_ratio_bem_over_adda": float(
                bem_stokes[0, 0, 0] / adda_stokes[0, 0, 0]
            ),
            "full_relative_l2_common_angles": float(
                np.linalg.norm(stokes_common_bem - stokes_common_adda)
                / np.linalg.norm(stokes_common_adda)
            ),
        }
    else:
        normalized_stokes_audit = None

    adda_plane_integral = polar_plane_integral(theta_adda, adda[0, 0])
    bem_plane_integral = polar_plane_integral(theta_bem, bem[0, 0])
    metrics = {
        "normalization": "ADDA M11(theta=0)",
        "normalization_value": scale,
        "adda_angular_step_degrees": float(theta_adda[1] - theta_adda[0]),
        "bem_angular_step_degrees": float(theta_bem[1] - theta_bem[0]),
        "common_angles_degrees": common.tolist(),
        "common_angle_count": int(common.size),
        "interpolation_used": False,
        "bem_residuals": {
            "first": bem_data["pfft_fgmres"]["fmm_residual"],
            "second": bem_data["physical"]["parallel_fmm_residual"],
        },
        "forward_M11_bem_over_adda": float(bem[0, 0, 0] / scale),
        "raw_full_relative_l2_common_angles": float(
            np.linalg.norm(bem_common - adda_common)
            / np.linalg.norm(adda_common)
        ),
        "shape_only_full_relative_l2_common_angles": float(
            np.linalg.norm(bem_common / bem[0, 0, 0] - adda_common / scale)
            / np.linalg.norm(adda_common / scale)
        ),
        "native_grid_M11_polar_plane_integral": {
            "warning": (
                "This is a polar-plane integral, not a full solid-angle "
                "integral for a non-axisymmetric prism."
            ),
            "adda": adda_plane_integral,
            "bem": bem_plane_integral,
            "bem_over_adda": bem_plane_integral / adda_plane_integral,
            "relative_difference": (
                abs(bem_plane_integral - adda_plane_integral)
                / abs(adda_plane_integral)
            ),
        },
        "physical_extinction_audit": extinction_audit,
        "dimensionless_normalized_stokes_matrix": normalized_stokes_audit,
        "maximum_normalized_component_error": maximum.tolist(),
    }
    (args.out_dir / "strict_bem_vs_adda_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
