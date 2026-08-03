#!/usr/bin/env python3
"""Report the ref=6 prism accuracy boundary against mixed-precision ADDA."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def read_time(path: Path) -> float | None:
    if not path.exists():
        return None
    match = re.search(r"ACTUAL_WALL_S=([0-9.eE+-]+)", path.read_text())
    return float(match.group(1)) if match else None


def read_log_number(path: Path, pattern: str) -> float | None:
    if not path.exists():
        return None
    match = re.search(pattern, path.read_text(errors="replace"), re.MULTILINE)
    return float(match.group(1)) if match else None


def read_log_max(path: Path, pattern: str) -> float | None:
    if not path.exists():
        return None
    matches = re.findall(pattern, path.read_text(errors="replace"), re.MULTILINE)
    return max(map(float, matches)) if matches else None


def load_adda(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(directory / "mueller", skiprows=1)
    theta = table[:, 0]
    matrix = table[:, 1:].reshape(-1, 4, 4).transpose(1, 2, 0)
    return theta, matrix


def load_bem(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = json.loads(path.read_text())
    theta = np.asarray(data["physical"]["theta_degrees"], dtype=float)
    matrix = np.asarray(data["physical"]["mueller"], dtype=float)
    return theta, matrix, data


def load_mbs_fast(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, skiprows=1)
    if table.ndim != 2 or table.shape[1] != 19:
        raise ValueError(f"unexpected MBS-fast fixed-plane table: {path}")
    theta = table[:, 0]
    matrix = table[:, 3:].reshape(-1, 4, 4).transpose(1, 2, 0)

    # MBS-fast and ADDA use opposite U,V Stokes signs. D is its own inverse.
    signs = np.asarray((1.0, 1.0, -1.0, -1.0))
    matrix = matrix * signs[:, None, None] * signs[None, :, None]
    return theta, matrix


def interpolate_matrix(
    source_theta: np.ndarray,
    source: np.ndarray,
    target_theta: np.ndarray,
) -> np.ndarray:
    if len(source_theta) == len(target_theta) and np.allclose(
        source_theta, target_theta
    ):
        return source
    result = np.empty((4, 4, len(target_theta)))
    for row in range(4):
        for column in range(4):
            result[row, column] = np.interp(
                target_theta, source_theta, source[row, column]
            )
    return result


def adda_extinction(directory: Path) -> float | None:
    values = []
    for name in ("CrossSec-X", "CrossSec-Y"):
        path = directory / name
        if not path.exists():
            return None
        match = re.search(r"^Cext\s*=\s*([0-9.eE+-]+)", path.read_text(), re.M)
        if not match:
            return None
        values.append(float(match.group(1)))
    log = (directory / "log").read_text(errors="replace")
    ka_match = re.search(r"^Volume-equivalent size parameter:\s*([0-9.eE+-]+)", log, re.M)
    wavelength_match = re.search(r"^lambda:\s*([0-9.eE+-]+)", log, re.M)
    if not ka_match or not wavelength_match:
        return None
    radius = float(ka_match.group(1)) * float(wavelength_match.group(1)) / (2 * math.pi)
    return (sum(values) / 2) / radius**2


def bem_extinction(data: dict) -> float | None:
    amplitudes = data.get("physical", {}).get("amplitudes")
    if not amplitudes:
        return None
    s1 = complex(*amplitudes["S1"][0])
    s2 = complex(*amplitudes["S2"][0])
    return -2 * math.pi * (s1.real + s2.real) / float(data["ka"]) ** 2


def result_row(root: Path, ka: int) -> dict | None:
    adda_dir = root / f"adda_fp32_ka{ka}_dpl15_e4"
    bem_file = root / f"bem_ka{ka}_ref6_pfft" / "result.json"
    mbs_file = root / f"mbs_fast_po_ka{ka}" / "mbs" / "mbs.dat"
    if not (adda_dir / "mueller").exists() or not bem_file.exists():
        return None

    theta_adda, adda = load_adda(adda_dir)
    theta_bem, bem, data = load_bem(bem_file)
    bem = interpolate_matrix(theta_bem, bem, theta_adda)
    adda_forward = float(adda[0, 0, 0])
    bem_forward = float(bem[0, 0, 0])
    adda_shape = adda / adda_forward
    bem_shape = bem / bem_forward
    component_max = np.max(np.abs(bem_shape - adda_shape), axis=2)
    significant = np.linalg.norm(adda_shape, axis=2) >= 1.0e-3 * np.linalg.norm(
        adda_shape[0, 0]
    )

    mbs_metrics = {
        "mbs_forward_m11_relative_error": None,
        "mbs_raw_full_relative_l2": None,
        "mbs_shape_full_relative_l2": None,
        "mbs_shape_max_significant_component": None,
        "mbs_wall_s": None,
    }
    if mbs_file.exists():
        theta_mbs, mbs_phase = load_mbs_fast(mbs_file)
        mbs_phase = interpolate_matrix(theta_mbs, mbs_phase, theta_adda)
        # MBS-fast writes the phase/cross-section matrix; ADDA stores k^2 times it.
        mbs = mbs_phase * (2.0 * math.pi) ** 2
        mbs_forward = float(mbs[0, 0, 0])
        mbs_shape = mbs / mbs_forward
        mbs_component_max = np.max(np.abs(mbs_shape - adda_shape), axis=2)
        mbs_metrics = {
            "mbs_forward_m11_relative_error": abs(
                mbs_forward / adda_forward - 1.0
            ),
            "mbs_raw_full_relative_l2": float(
                np.linalg.norm(mbs - adda) / np.linalg.norm(adda)
            ),
            "mbs_shape_full_relative_l2": float(
                np.linalg.norm(mbs_shape - adda_shape)
                / np.linalg.norm(adda_shape)
            ),
            "mbs_shape_max_significant_component": float(
                np.max(mbs_component_max[significant])
            ),
            "mbs_wall_s": read_time(root / f"mbs_fast_po_ka{ka}.time"),
        }

    adda_cext = adda_extinction(adda_dir)
    bem_cext = bem_extinction(data)
    extinction_error = (
        abs(bem_cext / adda_cext - 1.0)
        if adda_cext is not None and bem_cext is not None
        else None
    )
    physical = data.get("physical", {})
    residuals = [
        data.get("mbj", {}).get("fmm_residual"),
        physical.get("parallel_fmm_residual"),
    ]
    residuals = [value for value in residuals if value is not None]
    bem_memory = data.get("pfft_fgmres", {}).get("combined_gpu_memory_delta_mb")
    if bem_memory is None:
        bem_memory = data.get("gpu_memory_delta_mb")
    if bem_memory is None:
        bem_memory = data.get("mbj", {}).get("storage_mb")

    return {
        "ka": ka,
        "ref": int(data.get("refinements", 6)),
        "bem_dofs": int(data.get("system_dofs", 0)),
        "adda_dipoles": int(
            read_log_number(
                adda_dir / "log", r"^Total number of occupied dipoles:\s*([0-9]+)"
            )
            or 0
        ),
        "bem_points_per_internal_wavelength": data.get(
            "p2_nodes_per_wavelength_min"
        ),
        "forward_m11_bem_over_adda": bem_forward / adda_forward,
        "forward_m11_relative_error": abs(bem_forward / adda_forward - 1.0),
        "raw_full_relative_l2": float(np.linalg.norm(bem - adda) / np.linalg.norm(adda)),
        "shape_full_relative_l2": float(
            np.linalg.norm(bem_shape - adda_shape) / np.linalg.norm(adda_shape)
        ),
        "shape_max_abs_over_m11_forward": float(np.max(component_max)),
        "shape_max_significant_component": float(np.max(component_max[significant])),
        "extinction_relative_error": extinction_error,
        "adda_residual": read_log_max(
            adda_dir / "log", r"^Final \(recalculated\) residual norm:\s*([0-9.eE+-]+)"
        ),
        "bem_max_residual": max(residuals) if residuals else None,
        "adda_wall_s": read_time(root / f"adda_fp32_ka{ka}_dpl15_e4.time")
        or read_log_number(adda_dir / "log", r"^Total wall time:\s*([0-9.eE+-]+)"),
        "bem_wall_s": read_time(root / f"bem_ka{ka}_ref6_pfft.time"),
        "adda_gpu_memory_gib": (
            read_log_number(
                adda_dir / "log",
                r"^OpenCL memory usage: peak total -\s*([0-9.eE+-]+) MB",
            )
            or 0.0
        )
        / 1024.0,
        "bem_gpu_memory_gib": (float(bem_memory) / 1024.0 if bem_memory else None),
        **mbs_metrics,
    }


def percent(values: list[float | None]) -> np.ndarray:
    return np.asarray([np.nan if value is None else 100 * value for value in values])


def plot(rows: list[dict], output: Path, pdf: PdfPages | None = None) -> None:
    ka = np.asarray([row["ka"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))

    ax = axes[0, 0]
    ax.plot(ka, percent([row["raw_full_relative_l2"] for row in rows]), "o-", label="полная матрица, включая масштаб")
    ax.plot(ka, percent([row["forward_m11_relative_error"] for row in rows]), "s-", label=r"амплитуда $M_{11}(0)$")
    ax.plot(ka, percent([row["shape_full_relative_l2"] for row in rows]), "^-", label="угловая форма полной матрицы")
    ax.plot(ka, percent([row["shape_max_significant_component"] for row in rows]), "D-", label="максимум значимого элемента")
    ax.plot(ka, percent([row["extinction_relative_error"] for row in rows]), "v-", label="сечение экстинкции")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="порог 1 %")
    ax.set_ylabel("Расхождение BEM и ADDA, %")
    ax.set_title("Физическая точность BEM ref=6")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(ka, [row["bem_points_per_internal_wavelength"] for row in rows], "o-", color="#d95f02")
    ax.axhline(8.0, color="black", linestyle="--", linewidth=1, label="целевое значение 8")
    ax.set_ylabel("Узлов BEM на внутреннюю длину волны")
    ax.set_title("Разрешение поверхностной сетки")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(ka, [row["adda_wall_s"] / 3600 if row["adda_wall_s"] else np.nan for row in rows], "o-", label="ADDA: FP32-оператор, FP64-решатель")
    ax.plot(ka, [row["bem_wall_s"] / 3600 if row["bem_wall_s"] else np.nan for row in rows], "s-", label="BEM ref=6")
    for row in rows:
        if row["adda_wall_s"] and row["bem_wall_s"]:
            ax.annotate(
                f"{row['adda_wall_s'] / row['bem_wall_s']:.2f} раза",
                (row["ka"], row["bem_wall_s"] / 3600),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                color="#d95f02",
                fontsize=9,
            )
    ax.set_yscale("log")
    ax.set_ylabel("Полное время, ч")
    ax.set_title("Время одного расчёта")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(ka, [row["adda_gpu_memory_gib"] for row in rows], "o-", label="ADDA: FP32-оператор, FP64-решатель")
    ax.plot(ka, [row["bem_gpu_memory_gib"] for row in rows], "s-", label="BEM ref=6")
    ax.axhline(24.0, color="black", linestyle="--", linewidth=1, label="24 ГБ")
    ax.set_ylabel("Память видеокарты, ГБ")
    ax.set_title("Память видеокарты")
    ax.text(
        0.02,
        0.08,
        "BEM при ka=60: ранний некомпактный запуск",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.legend()

    for ax in axes.flat:
        ax.set_xlabel(r"Размерный параметр $ka$")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Шестигранная призма h/D=1, m=1,3: граница BEM ref=6", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=200)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)


def plot_mueller_comparison(
    root: Path,
    ka: int,
    output: Path,
    pdf: PdfPages | None = None,
) -> None:
    adda_dir = root / f"adda_fp32_ka{ka}_dpl15_e4"
    bem_file = root / f"bem_ka{ka}_ref6_pfft" / "result.json"
    theta_adda, adda = load_adda(adda_dir)
    theta_bem, bem, _ = load_bem(bem_file)
    bem = interpolate_matrix(theta_bem, bem, theta_adda)
    mbs_file = root / f"mbs_fast_po_ka{ka}" / "mbs" / "mbs.dat"
    mbs = None
    if mbs_file.exists():
        theta_mbs, mbs = load_mbs_fast(mbs_file)
        mbs = interpolate_matrix(theta_mbs, mbs, theta_adda)
    metrics = result_row(root, ka)
    if metrics is None:
        raise RuntimeError(f"missing completed comparison for ka={ka}")
    speedup = metrics["adda_wall_s"] / metrics["bem_wall_s"]

    fig, axes = plt.subplots(4, 4, figsize=(16, 13), sharex=True)
    for row in range(4):
        for column in range(4):
            ax = axes[row, column]
            if row == 0 and column == 0:
                adda_curve = np.maximum(
                    adda[0, 0] / adda[0, 0, 0], np.finfo(float).tiny
                )
                bem_curve = np.maximum(
                    bem[0, 0] / bem[0, 0, 0], np.finfo(float).tiny
                )
            else:
                adda_curve = np.divide(
                    adda[row, column],
                    adda[0, 0],
                    out=np.full_like(adda[row, column], np.nan),
                    where=np.abs(adda[0, 0]) > np.finfo(float).tiny,
                )
                bem_curve = np.divide(
                    bem[row, column],
                    bem[0, 0],
                    out=np.full_like(bem[row, column], np.nan),
                    where=np.abs(bem[0, 0]) > np.finfo(float).tiny,
                )
            mbs_curve = None
            if mbs is not None:
                if row == 0 and column == 0:
                    mbs_curve = np.maximum(
                        mbs[0, 0] / mbs[0, 0, 0], np.finfo(float).tiny
                    )
                else:
                    mbs_curve = np.divide(
                        mbs[row, column],
                        mbs[0, 0],
                        out=np.full_like(mbs[row, column], np.nan),
                        where=np.abs(mbs[0, 0]) > np.finfo(float).tiny,
                    )
            ax.plot(
                theta_adda,
                adda_curve,
                color="#1f77b4",
                linewidth=1.4,
                label="ADDA" if row == 0 and column == 0 else None,
            )
            ax.plot(
                theta_adda,
                bem_curve,
                color="#d95f02",
                linewidth=1.1,
                linestyle="--",
                label="BEM ref=6" if row == 0 and column == 0 else None,
            )
            if mbs_curve is not None:
                ax.plot(
                    theta_adda,
                    mbs_curve,
                    color="#1b9e77",
                    linewidth=1.1,
                    linestyle=":",
                    label="MBS-fast (ФО)" if row == 0 and column == 0 else None,
                )
            if row == 0 and column == 0:
                ax.set_yscale("log")
                ax.set_title(r"$M_{11}(\theta)/M_{11}(0)$, лог. шкала")
            else:
                ax.set_title(
                    rf"$M_{{{row + 1}{column + 1}}}(\theta)/M_{{11}}(\theta)$"
                )
            ax.grid(True, alpha=0.22)
            if row == 3:
                ax.set_xlabel(r"Угол рассеяния $\theta$, град.")
    axes[0, 0].legend(loc="best")
    mbs_title = ""
    if metrics["mbs_wall_s"]:
        mbs_speedup = metrics["adda_wall_s"] / metrics["mbs_wall_s"]
        mbs_title = (
            "\n"
            rf"MBS-fast (физическая оптика) "
            rf"{1000 * metrics['mbs_wall_s']:.2f} мс, быстрее ADDA "
            rf"в {mbs_speedup:.0f} раза; полное расхождение "
            rf"{100 * metrics['mbs_raw_full_relative_l2']:.2f} %"
        )
    fig.suptitle(
        rf"Шестигранная призма, $ka={ka}$, $m=1{{,}}3$: все элементы матрицы Мюллера"
        "\n"
        rf"ADDA {metrics['adda_wall_s'] / 60:.2f} мин, BEM ref=6 "
        rf"{metrics['bem_wall_s'] / 60:.2f} мин, BEM быстрее в {speedup:.2f} раза"
        "\n"
        rf"полное расхождение {100 * metrics['raw_full_relative_l2']:.2f} %, "
        rf"расхождение нормированной угловой формы "
        rf"{100 * metrics['shape_full_relative_l2']:.2f} %"
        + mbs_title,
        fontsize=13.5,
        y=0.97,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.85 if mbs_title else 0.89))
    fig.savefig(output, dpi=180)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.root.glob("adda_fp32_ka*_dpl15_e4")):
        match = re.fullmatch(r"adda_fp32_ka(\d+)_dpl15_e4", path.name)
        if not match:
            continue
        row = result_row(args.root, int(match.group(1)))
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda row: row["ka"])
    if not rows:
        raise SystemExit("no matching completed ADDA/BEM pairs")

    with (args.root / "ref6_adda_fp32_boundary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.root / "ref6_adda_fp32_boundary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    )
    pdf_path = args.root / "ref6_adda_fp32_all_sizes.pdf"
    with PdfPages(
        pdf_path,
        metadata={
            "Title": "BEM ref=6 and mixed-precision ADDA comparison",
            "Subject": "Hexagonal prism, ka=60, 80, 111",
        },
    ) as pdf:
        plot(rows, args.root / "ref6_adda_fp32_boundary.png", pdf)
        for row in rows:
            plot_mueller_comparison(
                args.root,
                row["ka"],
                args.root / f"ref6_adda_fp32_mueller_ka{row['ka']}.png",
                pdf,
            )
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
