#!/usr/bin/env python3
"""Compare the ka=60 strict BEM result with ADDA dpl=15 and MBS-fast PO."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("runs/hdiv_bem_vs_adda_sweep_n1p3/ka60")
CASE = ROOT / "max_dpl15"
BEM_JSON = CASE / "bem_ref5_edge1_pfft_fgmres_exactC6.json"
BEM_TIME = CASE / "bem_ref5_edge1_pfft_fgmres_exactC6.time"
ADDA_DIR = ROOT / "adda_dpl15"
MBS_ROOT = Path("/home/kirill/MBS-fast-github-current")
MBS_DATA = (
    MBS_ROOT
    / "plots/bem_adda_mbs_ka60_n1p3/"
    "matched_po_shadowfix/matched_po_shadowfix.dat"
)
MBS_TIME = (
    MBS_ROOT / "plots/bem_adda_mbs_ka60_n1p3/matched_po_shadowfix.time"
)
PREFIX = CASE / "ka60_bem_adda_mbs"
PHYSICAL_WAVE_NUMBER = 2.0 * np.pi


def wall_seconds(path: Path) -> float:
    text = path.read_text(errors="replace")
    match = re.search(
        r"Elapsed \(wall clock\) time .*?:\s*"
        r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)",
        text,
    )
    if match:
        return (
            3600.0 * int(match.group(1) or 0)
            + 60.0 * int(match.group(2))
            + float(match.group(3))
        )
    match = re.search(r"(?:ACTUAL_)?WALL_S=([0-9.eE+-]+)", text)
    if match:
        return float(match.group(1))
    raise ValueError(f"cannot parse wall time from {path}")


def maxrss_mb(path: Path) -> float:
    text = path.read_text(errors="replace")
    match = re.search(
        r"(?:Maximum resident set size \(kbytes\):|MAXRSS_KB=)\s*"
        r"([0-9.eE+-]+)",
        text,
    )
    if not match:
        raise ValueError(f"cannot parse peak RSS from {path}")
    return float(match.group(1)) / 1024.0


def adda_log_metadata(path: Path) -> dict[str, float | int]:
    text = path.read_text(errors="replace")

    def value(pattern: str, cast: type[float] | type[int]) -> float | int:
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"cannot parse ADDA metadata from {path}: {pattern}")
        return cast(match.group(1))

    return {
        "iterations": value(r"Total number of iterations:\s*(\d+)", int),
        "gpu_memory_mb": value(
            r"OpenCL memory usage: peak total -\s*([0-9.eE+-]+)\s*MB",
            float,
        ),
        "actual_dpl": value(r"Dipoles/lambda:\s*([0-9.eE+-]+)", float),
        "occupied_dipoles": value(
            r"Total number of occupied dipoles:\s*(\d+)", int
        ),
    }


def normalized_metrics(
    theta: np.ndarray, reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    ref = reference / reference[0]
    other = candidate / candidate[0]
    weights = np.sin(np.deg2rad(theta))
    return {
        "forward_normalized_relative_l2": float(
            np.linalg.norm(other - ref) / np.linalg.norm(ref)
        ),
        "solid_angle_weighted_relative_l2": float(
            np.sqrt(
                np.sum(weights * (other - ref) ** 2)
                / np.sum(weights * ref**2)
            )
        ),
        "backscatter_ratio_to_reference": float(other[-1] / ref[-1]),
    }


def load_mbs(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(MBS_DATA, skiprows=1)
    plane = table[np.isclose(np.mod(table[:, 1], 360.0), 90.0)]
    if plane.shape[0] != len(theta) or not np.allclose(plane[:, 0], theta):
        raise ValueError("MBS-fast phi=90 theta grid does not match BEM")
    return plane[:, 3], plane[:, 3:].reshape(-1, 4, 4)


def load_adda(theta: np.ndarray) -> np.ndarray | None:
    path = ADDA_DIR / "mueller"
    if not path.exists():
        return None
    table = np.loadtxt(path, skiprows=1)
    if table.shape[0] != len(theta) or not np.allclose(table[:, 0], theta):
        raise ValueError("ADDA theta grid does not match BEM")
    return table[:, 1:].reshape(-1, 4, 4)


def main() -> None:
    bem_data = json.loads(BEM_JSON.read_text())
    theta = np.asarray(bem_data["physical"]["theta_degrees"], dtype=float)
    bem_matrix = np.asarray(bem_data["physical"]["mueller"], dtype=float)
    bem_m11 = bem_matrix[0, 0]
    mbs_m11_physical, mbs_matrix_physical = load_mbs(theta)
    # ADDA and this BEM implementation store |S|^2. MBS-fast stores the
    # dimensional differential-scattering matrix, so convert it with k^2.
    mbs_matrix = mbs_matrix_physical * PHYSICAL_WAVE_NUMBER**2
    mbs_m11 = mbs_m11_physical * PHYSICAL_WAVE_NUMBER**2
    adda_matrix = load_adda(theta)
    adda_m11 = None if adda_matrix is None else adda_matrix[:, 0, 0]
    adda_metadata = (
        None
        if adda_matrix is None
        else adda_log_metadata(ADDA_DIR / "log")
    )

    methods = {
        "BEM": {
            "model": "full-wave Muller BEM; pFFT preconditioner; FMM residual",
            "wall_s": wall_seconds(BEM_TIME),
            "residual": bem_data["pfft_fgmres"]["fmm_residual"],
            "gpu_memory_mb": bem_data["pfft_fgmres"][
                "combined_gpu_memory_delta_mb"
            ],
            "host_memory_mb": maxrss_mb(BEM_TIME),
        },
        "MBS-fast PO": {
            "model": (
                "geometrical ray tracing plus Physical Optics; fixed-mode "
                "shadow beam enabled"
            ),
            "wall_s_upper_bound": max(wall_seconds(MBS_TIME), 0.01),
            "residual": None,
            "gpu_memory_mb": 0,
            "host_memory_mb": maxrss_mb(MBS_TIME),
        },
    }
    metrics = {"MBS-fast_PO_vs_BEM_M11": normalized_metrics(
        theta, bem_m11, mbs_m11
    )}
    metrics["MBS-fast_PO_vs_BEM_M11"].update({
        "forward_BEM_dimensionless": float(bem_m11[0]),
        "forward_MBS_dimensionless": float(mbs_m11[0]),
        "forward_BEM_physical": float(
            bem_m11[0] / PHYSICAL_WAVE_NUMBER**2
        ),
        "forward_MBS_physical": float(mbs_m11_physical[0]),
        "forward_MBS_over_BEM": float(mbs_m11[0] / bem_m11[0]),
    })
    if adda_m11 is not None:
        methods["ADDA dpl=15"] = {
            "model": "full-wave DDA, QMR2",
            "wall_s": wall_seconds(ADDA_DIR / "time.txt"),
            "residual": 1.0e-5,
            "host_memory_mb": maxrss_mb(ADDA_DIR / "time.txt"),
            **adda_metadata,
        }
        metrics["ADDA_vs_BEM_M11"] = normalized_metrics(
            theta, bem_m11, adda_m11
        )
        metrics["ADDA_vs_BEM_M11"].update({
            "forward_BEM_dimensionless": float(bem_m11[0]),
            "forward_ADDA_dimensionless": float(adda_m11[0]),
            "forward_ADDA_over_BEM": float(adda_m11[0] / bem_m11[0]),
            "forward_relative_difference": float(
                abs(adda_m11[0] - bem_m11[0]) / abs(bem_m11[0])
            ),
        })
        bem_normalized = bem_matrix / bem_m11[0]
        adda_for_bem = np.moveaxis(adda_matrix, 0, -1)
        adda_normalized = adda_for_bem / adda_m11[0]
        solid_angle_weights = np.sin(np.deg2rad(theta))[None, None, :]
        metrics["ADDA_vs_BEM_full_Mueller"] = {
            "raw_relative_l2": float(
                np.linalg.norm(adda_for_bem - bem_matrix)
                / np.linalg.norm(bem_matrix)
            ),
            "forward_normalized_relative_l2": float(
                np.linalg.norm(adda_normalized - bem_normalized)
                / np.linalg.norm(bem_normalized)
            ),
            "solid_angle_weighted_relative_l2": float(
                np.sqrt(
                    np.sum(
                        solid_angle_weights
                        * (adda_normalized - bem_normalized) ** 2
                    )
                    / np.sum(solid_angle_weights * bem_normalized**2)
                )
            ),
        }

    summary = {
        "case": {
            "shape": "regular hexagonal prism",
            "aspect_h_over_Dx": 1.0,
            "ka": 60.0,
            "refractive_index": 1.3,
            "orientation": "axis incidence",
            "theta_points": len(theta),
            "mbs_geometry": (
                "built-in hex: height=sqrt(3), vertex_diameter=2; "
                "equivalent to h/flat_to_flat_diameter=1"
            ),
        },
        "methods": methods,
        "metrics": metrics,
        "warning": (
            "MBS-fast PO is an asymptotic model and has no iterative "
            "full-wave residual. Its forward value is reported without "
            "fitting, so its wall time is not an equal-accuracy solve."
        ),
    }
    PREFIX.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    fig, axes = plt.subplots(
        2, 2, figsize=(13.8, 9.0), constrained_layout=True
    )
    bem_norm = bem_m11 / bem_m11[0]
    mbs_norm = mbs_m11 / mbs_m11[0]

    axis = axes[0, 0]
    axis.plot(theta, bem_norm, color="black", lw=2.2, label="строгий BEM")
    axis.plot(theta, mbs_norm, color="#d97706", lw=1.7, label="MBS-fast PO")
    if adda_m11 is not None:
        axis.plot(
            theta,
            adda_m11 / adda_m11[0],
            color="#2563eb",
            lw=1.5,
            label="ADDA dpl=15",
        )
        axis.text(
            0.02,
            0.04,
            (
                "Абсолютно вперёд:\n"
                f"ADDA/BEM = {adda_m11[0] / bem_m11[0]:.3f}; "
                f"MBS/BEM = {mbs_m11[0] / bem_m11[0]:.3f}"
            ),
            transform=axis.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
    axis.set_yscale("log")
    axis.set_title(r"Нормированная угловая зависимость $M_{11}$")
    axis.set_xlabel(r"Угол рассеяния $\theta$, град.")
    axis.set_ylabel(r"$M_{11}(\theta)/M_{11}(0)$")
    axis.legend()

    axis = axes[0, 1]
    axis.plot(
        theta,
        mbs_norm / bem_norm,
        color="#d97706",
        lw=1.8,
        label="MBS-fast / BEM",
    )
    if adda_m11 is not None:
        axis.plot(
            theta,
            (adda_m11 / adda_m11[0]) / bem_norm,
            color="#2563eb",
            lw=1.5,
            label="ADDA / BEM",
        )
    axis.axhline(1.0, color="black", ls="--", lw=1.2)
    axis.set_yscale("log")
    axis.set_title(r"Отношение методов к BEM для $M_{11}$")
    axis.set_xlabel(r"Угол рассеяния $\theta$, град.")
    axis.set_ylabel("Отношение нормированных значений")
    axis.legend()

    axis = axes[1, 0]
    labels = ["MBS-fast\nPO", "строгий\nBEM"]
    values = [max(wall_seconds(MBS_TIME), 0.01), wall_seconds(BEM_TIME)]
    colors = ["#d97706", "#16a34a"]
    if adda_m11 is not None:
        labels.append("ADDA\ndpl=15")
        values.append(wall_seconds(ADDA_DIR / "time.txt"))
        colors.append("#2563eb")
    bars = axis.bar(labels, values, color=colors)
    axis.set_yscale("log")
    axis.set_title("Полное время одного расчета")
    axis.set_ylabel("Время, с")
    axis.bar_label(
        bars,
        labels=[
            "<0,01 с" if value <= 0.01 else f"{value:.1f} с"
            for value in values
        ],
        padding=3,
    )

    axis = axes[1, 1]
    angles = np.asarray([2.5, 5.0, 30.0, 90.0, 150.0, 180.0])
    indices = [int(np.argmin(np.abs(theta - value))) for value in angles]
    ratios = mbs_norm[indices] / bem_norm[indices]
    bars = axis.bar([f"{value:g}°" for value in angles], ratios, color="#d97706")
    axis.set_yscale("log")
    axis.axhline(1.0, color="black", ls="--", lw=1.2)
    axis.set_title(r"Различие $M_{11}$ в характерных направлениях")
    axis.set_xlabel(r"Угол рассеяния $\theta$")
    axis.set_ylabel("MBS-fast / BEM")
    axis.bar_label(bars, labels=[f"{value:.1f}×" for value in ratios], padding=3)

    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.25)
    fig.suptitle(
        r"Шестигранная призма: $h/D_x=1$, $ka=60$, $m=1{,}3$",
        fontsize=15,
    )
    fig.savefig(PREFIX.with_suffix(".png"), dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
