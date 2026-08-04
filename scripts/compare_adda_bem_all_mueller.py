#!/usr/bin/env python3
"""Compare all Mueller elements from matching ADDA and Muller-BEM runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adda", type=Path, required=True)
    parser.add_argument("--bem", type=Path, required=True)
    parser.add_argument(
        "--bem-coarse",
        type=Path,
        help="Optional coarser BEM result for a self-convergence audit.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title", default="ADDA и BEM: все элементы матрицы Мюллера")
    return parser.parse_args()


def load_adda(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, skiprows=1)
    if table.ndim != 2 or table.shape[1] != 17:
        raise ValueError(f"{path}: expected theta and 16 Mueller columns")
    theta = table[:, 0]
    mueller = table[:, 1:].reshape(-1, 4, 4).transpose(1, 2, 0)
    return theta, mueller


def load_bem(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = json.loads(path.read_text())
    physical = data["physical"]
    theta = np.asarray(physical["theta_degrees"], dtype=float)
    mueller = np.asarray(physical["mueller"], dtype=float)
    if mueller.shape != (4, 4, len(theta)):
        raise ValueError(f"{path}: unsupported Mueller shape {mueller.shape}")
    return theta, mueller, data


def interpolate_mueller(
    theta_source: np.ndarray,
    mueller_source: np.ndarray,
    theta_target: np.ndarray,
) -> np.ndarray:
    if (
        theta_target[0] < theta_source[0] - 1.0e-12
        or theta_target[-1] > theta_source[-1] + 1.0e-12
    ):
        raise ValueError("target angles extend beyond the Mueller data")
    return np.asarray(
        [
            [
                np.interp(theta_target, theta_source, mueller_source[row, column])
                for column in range(4)
            ]
            for row in range(4)
        ]
    )


def adda_extinction(directory: Path, ka: float) -> dict | None:
    values = []
    for name in ("CrossSec-X", "CrossSec-Y"):
        path = directory / name
        if not path.exists():
            return None
        match = re.search(
            r"^Cext\s*=\s*([0-9.eE+-]+)",
            path.read_text(),
            re.MULTILINE,
        )
        if not match:
            return None
        values.append(float(match.group(1)))

    log_path = directory / "log"
    if not log_path.exists():
        return None
    log_text = log_path.read_text(errors="replace")
    wavelength_match = re.search(r"^lambda:\s*([0-9.eE+-]+)", log_text, re.MULTILINE)
    if not wavelength_match:
        return None
    wavelength = float(wavelength_match.group(1))
    equivalent_radius = ka * wavelength / (2.0 * np.pi)
    average = 0.5 * sum(values)
    return {
        "Cext_X": values[0],
        "Cext_Y": values[1],
        "Cext_average": average,
        "equivalent_radius": equivalent_radius,
        "Cext_average_over_equivalent_radius_squared": (
            average / equivalent_radius**2
        ),
    }


def bem_optical_theorem_extinction(data: dict) -> dict | None:
    amplitudes = data.get("physical", {}).get("amplitudes")
    if not amplitudes:
        return None
    try:
        s1 = complex(*amplitudes["S1"][0])
        s2 = complex(*amplitudes["S2"][0])
        ka = float(data["ka"])
    except (KeyError, TypeError, ValueError):
        return None
    value = -2.0 * np.pi * (s1.real + s2.real) / ka**2
    return {
        "formula": "-2*pi*Re(S1(0)+S2(0))/ka^2",
        "Cext_average_over_equivalent_radius_squared": float(value),
        "S1_forward": [s1.real, s1.imag],
        "S2_forward": [s2.real, s2.imag],
    }


def trapz(values: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(values, x))


def plane_weighted_relative(
    theta_rad: np.ndarray, candidate: np.ndarray, reference: np.ndarray
) -> float:
    weight = np.sin(theta_rad)
    numerator = trapz(
        np.sum((candidate - reference) ** 2, axis=(0, 1)) * weight,
        theta_rad,
    )
    denominator = trapz(
        np.sum(reference**2, axis=(0, 1)) * weight,
        theta_rad,
    )
    return float(np.sqrt(max(numerator, 0.0) / max(denominator, 1.0e-300)))


def component_metrics(
    adda: np.ndarray, bem: np.ndarray, forward_scale: float
) -> list[dict]:
    m11_norm = max(float(np.linalg.norm(adda[0, 0])), 1.0e-300)
    rows = []
    for row in range(4):
        for column in range(4):
            reference = adda[row, column]
            candidate = bem[row, column]
            reference_norm = float(np.linalg.norm(reference))
            strength = reference_norm / m11_norm
            rows.append(
                {
                    "element": f"M{row + 1}{column + 1}",
                    "reference_l2_over_M11_l2": strength,
                    "relative_l2": (
                        float(np.linalg.norm(candidate - reference) / reference_norm)
                        if reference_norm > 1.0e-14 * m11_norm
                        else None
                    ),
                    "difference_l2_over_M11_l2": float(
                        np.linalg.norm(candidate - reference) / m11_norm
                    ),
                    "max_abs_difference_over_forward_M11": float(
                        np.max(np.abs(candidate - reference)) / forward_scale
                    ),
                    "near_zero_reference": strength < 1.0e-3,
                }
            )
    return rows


def angular_bands(
    theta_degrees: np.ndarray,
    theta_rad: np.ndarray,
    adda_m11: np.ndarray,
    bem_m11: np.ndarray,
) -> list[dict]:
    rows = []
    for lower, upper in (
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 20.0),
        (20.0, 40.0),
        (40.0, 80.0),
        (80.0, 120.0),
        (120.0, 160.0),
        (160.0, 180.0),
    ):
        mask = (theta_degrees >= lower) & (theta_degrees <= upper)
        adda_integral = 2.0 * np.pi * trapz(
            adda_m11[mask] * np.sin(theta_rad[mask]), theta_rad[mask]
        )
        bem_integral = 2.0 * np.pi * trapz(
            bem_m11[mask] * np.sin(theta_rad[mask]), theta_rad[mask]
        )
        rows.append(
            {
                "theta_min_degrees": lower,
                "theta_max_degrees": upper,
                "adda_plane_integral": adda_integral,
                "bem_plane_integral": bem_integral,
                "bem_over_adda": (
                    bem_integral / adda_integral
                    if abs(adda_integral) > 1.0e-300
                    else None
                ),
            }
        )
    return rows


def plot_elements(
    theta: np.ndarray,
    adda: np.ndarray,
    bem: np.ndarray,
    scale: float,
    title: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    for row in range(4):
        for column in range(4):
            ax = axes[row, column]
            is_m11 = row == 0 and column == 0
            if is_m11:
                adda_values = adda[row, column] / scale
                bem_values = bem[row, column] / scale
                panel_title = r"$M_{11}/M_{11}^{ADDA}(0)$"
            else:
                adda_values = adda[row, column] / adda[0, 0]
                bem_values = bem[row, column] / bem[0, 0]
                panel_title = rf"$M_{{{row + 1}{column + 1}}}(\theta)/M_{{11}}(\theta)$"
            ax.plot(
                theta,
                adda_values,
                color="#1769aa",
                linewidth=1.8,
                label="ADDA",
            )
            ax.plot(
                theta,
                bem_values,
                color="#d95f02",
                linewidth=1.5,
                linestyle="--",
                label="BEM",
            )
            if is_m11:
                ax.set_yscale("log")
            ax.set_title(panel_title)
            ax.set_xlim(0.0, 180.0)
            ax.grid(True, which="both", alpha=0.22)
            if row == 3:
                ax.set_xlabel(r"Угол рассеяния $\theta$, град.")
            if column == 0:
                ax.set_ylabel("Общая нормировка")
    axes[0, 0].legend(loc="best")
    fig.suptitle(title, fontsize=17)
    fig.text(
        0.5,
        0.008,
        (
            r"$M_{11}$ дан в общей нормировке на $M_{11}^{ADDA}(0)$; "
            r"остальные элементы каждого метода нормированы на его $M_{11}(\theta)$."
        ),
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.02, 0.025, 1.0, 0.965))
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_differences(
    theta: np.ndarray,
    adda: np.ndarray,
    bem: np.ndarray,
    scale: float,
    output: Path,
) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(16, 11.5), sharex=True)
    for row in range(4):
        for column in range(4):
            ax = axes[row, column]
            difference = (bem[row, column] - adda[row, column]) / scale
            ax.plot(theta, difference, color="#6a3d9a", linewidth=1.5)
            ax.axhline(0.0, color="black", linewidth=0.7)
            ax.set_title(rf"$\Delta M_{{{row + 1}{column + 1}}}/M_{{11}}^{{ADDA}}(0)$")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.set_xlim(0.0, 180.0)
            ax.grid(True, alpha=0.22)
            if row == 3:
                ax.set_xlabel(r"Угол рассеяния $\theta$, град.")
            if column == 0:
                ax.set_ylabel("BEM минус ADDA")
    fig.suptitle("Абсолютное расхождение всех элементов в общей нормировке", fontsize=17)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.965))
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_error_heatmap(
    rows: list[dict],
    output: Path,
    title: str,
    colorbar_label: str,
) -> None:
    matrix = np.asarray(
        [row["max_abs_difference_over_forward_M11"] for row in rows],
        dtype=float,
    ).reshape(4, 4)
    fig, ax = plt.subplots(figsize=(7.8, 6.5))
    image = ax.imshow(matrix, cmap="magma", norm=matplotlib.colors.LogNorm(
        vmin=max(float(np.min(matrix[matrix > 0])), 1.0e-9),
        vmax=max(float(np.max(matrix)), 1.0e-8),
    ))
    for row in range(4):
        for column in range(4):
            value = matrix[row, column]
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "black" if luminance > 0.55 else "white"
            ax.text(column, row, f"{value:.2e}", ha="center", va="center", color=color)
    ax.set_xticks(range(4), [f"столбец {index}" for index in range(1, 5)])
    ax.set_yticks(range(4), [f"строка {index}" for index in range(1, 5)])
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    theta_adda, adda = load_adda(args.adda)
    theta_bem, bem, bem_data = load_bem(args.bem)
    if len(theta_adda) != len(theta_bem) or not np.allclose(theta_adda, theta_bem):
        # Compare on the coarser native grid.  Interpolating a sparse angular
        # curve onto a denser grid invents structure between samples and can
        # dominate error norms for high-ka oscillatory scattering patterns.
        if len(theta_adda) <= len(theta_bem):
            bem = interpolate_mueller(theta_bem, bem, theta_adda)
        else:
            adda = interpolate_mueller(theta_adda, adda, theta_bem)
            theta_adda = theta_bem

    args.out_dir.mkdir(parents=True, exist_ok=True)
    theta_rad = np.deg2rad(theta_adda)
    scale = float(adda[0, 0, 0])
    rows = component_metrics(adda, bem, scale)
    bands = angular_bands(theta_adda, theta_rad, adda[0, 0], bem[0, 0])

    adda_plane_integral = 2.0 * np.pi * trapz(
        adda[0, 0] * np.sin(theta_rad), theta_rad
    )
    bem_plane_integral = 2.0 * np.pi * trapz(
        bem[0, 0] * np.sin(theta_rad), theta_rad
    )
    pfft_result = bem_data.get("pfft_fgmres") or {}
    mbj_result = bem_data.get("mbj") or {}
    main_residual = pfft_result.get(
        "fmm_residual", mbj_result.get("fmm_residual")
    )
    parallel_residual = bem_data["physical"].get("parallel_fmm_residual")
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
        extinction_audit["relative_difference"] = abs(bem_value - adda_value) / adda_value
    summary = {
        "normalization": {
            "common_denominator": "ADDA M11(theta=0)",
            "value": scale,
            "per_element_normalization_used": False,
        },
        "angular_grid": {
            "points": int(len(theta_adda)),
            "step_degrees": float(theta_adda[1] - theta_adda[0]),
            "warning": (
                "The 2*pi polar-plane integral is not a full solid-angle "
                "integral for a non-axisymmetric prism."
            ),
        },
        "solver_residuals": {
            "bem_first_polarization": main_residual,
            "bem_second_polarization": parallel_residual,
        },
        "physical_extinction_audit": extinction_audit,
        "global_metrics": {
            "forward_M11_adda": scale,
            "forward_M11_bem": float(bem[0, 0, 0]),
            "forward_ratio_bem_over_adda": float(bem[0, 0, 0] / scale),
            "raw_full_relative_l2": float(
                np.linalg.norm(bem - adda) / np.linalg.norm(adda)
            ),
            "shape_only_full_relative_l2": float(
                np.linalg.norm(bem / bem[0, 0, 0] - adda / scale)
                / np.linalg.norm(adda / scale)
            ),
            "polar_plane_sin_weighted_full_relative_l2": (
                plane_weighted_relative(theta_rad, bem, adda)
            ),
            "shape_only_polar_plane_sin_weighted_full_relative_l2": (
                plane_weighted_relative(
                    theta_rad,
                    bem / bem[0, 0, 0],
                    adda / scale,
                )
            ),
            "shape_only_polar_plane_sin_weighted_M11_relative_l2": float(
                np.sqrt(
                    trapz(
                        (
                            bem[0, 0] / bem[0, 0, 0]
                            - adda[0, 0] / scale
                        )
                        ** 2
                        * np.sin(theta_rad),
                        theta_rad,
                    )
                    / max(
                        trapz(
                            (adda[0, 0] / scale) ** 2
                            * np.sin(theta_rad),
                            theta_rad,
                        ),
                        1.0e-300,
                    )
                )
            ),
            "adda_M11_polar_plane_integral": adda_plane_integral,
            "bem_M11_polar_plane_integral": bem_plane_integral,
            "bem_over_adda_M11_polar_plane_integral": (
                bem_plane_integral / adda_plane_integral
            ),
        },
        "components": rows,
        "angular_bands_M11": bands,
    }
    if args.bem_coarse is not None:
        theta_coarse, bem_coarse, coarse_data = load_bem(args.bem_coarse)
        if (
            len(theta_coarse) != len(theta_adda)
            or not np.allclose(theta_coarse, theta_adda)
        ):
            bem_coarse = interpolate_mueller(theta_coarse, bem_coarse, theta_adda)
        refinement_rows = component_metrics(bem_coarse, bem, scale)
        summary["bem_self_convergence"] = {
            "coarse_file": str(args.bem_coarse),
            "fine_file": str(args.bem),
            "coarse_refinement": coarse_data["refinements"],
            "fine_refinement": bem_data["refinements"],
            "raw_full_relative_l2": float(
                np.linalg.norm(bem - bem_coarse) / np.linalg.norm(bem_coarse)
            ),
            "polar_plane_sin_weighted_full_relative_l2": (
                plane_weighted_relative(theta_rad, bem, bem_coarse)
            ),
            "M11_relative_l2": float(
                np.linalg.norm(bem[0, 0] - bem_coarse[0, 0])
                / np.linalg.norm(bem_coarse[0, 0])
            ),
            "forward_M11_fine_over_coarse": float(
                bem[0, 0, 0] / bem_coarse[0, 0, 0]
            ),
            "components": refinement_rows,
        }
        write_csv(args.out_dir / "bem_self_convergence_components.csv", refinement_rows)
        plot_error_heatmap(
            refinement_rows,
            args.out_dir / "bem_self_convergence_error_heatmap.png",
            (
                r"$\max_\theta|M_{ij}^{ref=6}-M_{ij}^{ref=5}|"
                r"/M_{11}^{ADDA}(0)$"
            ),
            "Изменение при сгущении BEM-сетки",
        )
    (args.out_dir / "comparison_all_mueller.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    write_csv(args.out_dir / "comparison_all_mueller.csv", rows)
    with (args.out_dir / "m11_angular_band_integrals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bands[0]))
        writer.writeheader()
        writer.writerows(bands)

    plot_elements(
        theta_adda,
        adda,
        bem,
        scale,
        args.title,
        args.out_dir / "adda_vs_bem_all_mueller.png",
    )
    plot_differences(
        theta_adda,
        adda,
        bem,
        scale,
        args.out_dir / "adda_vs_bem_all_mueller_difference.png",
    )
    plot_error_heatmap(
        rows,
        args.out_dir / "adda_vs_bem_all_mueller_error_heatmap.png",
        r"$\max_\theta|M_{ij}^{BEM}-M_{ij}^{ADDA}|/M_{11}^{ADDA}(0)$",
        "Максимальное абсолютное расхождение",
    )
    print(json.dumps(summary["global_metrics"], indent=2))


if __name__ == "__main__":
    main()
