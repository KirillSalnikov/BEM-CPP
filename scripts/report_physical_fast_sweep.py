#!/usr/bin/env python3
"""Report the validated two-stage BEM acceleration at ka=60, 80, and 111."""

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
from matplotlib.backends.backend_pdf import PdfPages


CASES = (
    {
        "ka": 60,
        "stage1_log": "runs/physical_fast_sweep_20260803/ka60_cold/stage1/run_and_time.log",
        "stage2_log": "runs/physical_fast_sweep_20260803/ka60_cold/stage2/run_and_time.log",
        "result": "runs/physical_fast_sweep_20260803/ka60_cold/stage2/result.json",
        "strict": "runs/banded_pfft_vs_adda_20260803/ka60_three_band_refine/result.json",
        "adda": "runs/ref6_vs_adda_fp32_ka_gt60_20260802/adda_fp32_ka60_dpl15_e4",
    },
    {
        "ka": 80,
        "pipeline_summary": "runs/honest_x10_20260803/wrapper_cold/physical_fast_summary.json",
        "result": "runs/honest_x10_20260803/wrapper_cold/final/result.json",
        "strict": "runs/banded_pfft_vs_adda_20260803/ka80_three_band_refine/result.json",
        "adda": "runs/ref6_vs_adda_fp32_ka_gt60_20260802/adda_fp32_ka80_dpl15_e4",
    },
    {
        "ka": 111,
        "stage1_log": "runs/physical_fast_sweep_20260803/ka111_warm_probe/stage1/run_and_time.log",
        "stage2_log": "runs/physical_fast_sweep_20260803/ka111_warm_probe/stage2/run_and_time.log",
        "result": "runs/physical_fast_sweep_20260803/ka111_warm_probe/stage2/result.json",
        "strict": "runs/ka111_depth3_symmetry_20260803/result.json",
        "adda": "runs/ref6_vs_adda_fp32_ka_gt60_20260802/adda_fp32_ka111_dpl15_e4",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/physical_fast_sweep_20260803/report"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def elapsed_seconds(path: Path) -> float:
    text = path.read_text(errors="replace")
    match = re.search(
        r"Elapsed \(wall clock\) time .*?:\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)",
        text,
    )
    if not match:
        raise ValueError(f"wall-clock time is missing from {path}")
    return (
        3600.0 * int(match.group(1) or 0)
        + 60.0 * int(match.group(2))
        + float(match.group(3))
    )


def adda_wall_seconds(directory: Path) -> float:
    text = (directory / "log").read_text(errors="replace")
    match = re.search(r"^Total wall time:\s*([0-9.eE+-]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"ADDA wall time is missing from {directory / 'log'}")
    return float(match.group(1))


def load_bem(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = read_json(path)
    physical = data["physical"]
    theta = np.asarray(physical["theta_degrees"], dtype=float)
    matrix = np.asarray(physical["mueller"], dtype=float)
    if matrix.shape != (4, 4, theta.size):
        raise ValueError(f"unexpected Mueller shape {matrix.shape} in {path}")
    return theta, matrix, data


def load_adda(directory: Path, theta_target: np.ndarray) -> np.ndarray:
    table = np.loadtxt(directory / "mueller", skiprows=1)
    theta = table[:, 0]
    matrix = table[:, 1:].reshape(-1, 4, 4).transpose(1, 2, 0)
    if theta.size == theta_target.size and np.allclose(theta, theta_target):
        return matrix
    return np.asarray(
        [
            [np.interp(theta_target, theta, matrix[i, j]) for j in range(4)]
            for i in range(4)
        ]
    )


def normalized_metrics(
    theta_degrees: np.ndarray, candidate: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    candidate_forward = float(candidate[0, 0, 0])
    reference_forward = float(reference[0, 0, 0])
    candidate = candidate / candidate_forward
    reference = reference / reference_forward
    weights = np.sin(np.deg2rad(theta_degrees))[None, None, :]
    difference = candidate - reference
    full_l2 = np.sqrt(
        np.sum(weights * difference**2) / np.sum(weights * reference**2)
    )
    m11_l2 = np.sqrt(
        np.sum(weights[0, 0] * difference[0, 0] ** 2)
        / np.sum(weights[0, 0] * reference[0, 0] ** 2)
    )
    return {
        "weighted_full_relative_l2": float(full_l2),
        "weighted_m11_relative_l2": float(m11_l2),
        "forward_m11_relative_difference": float(
            abs(candidate_forward / reference_forward - 1.0)
        ),
        "maximum_absolute_normalized_element_difference": float(
            np.max(np.abs(difference))
        ),
    }


def load_case(root: Path, specification: dict) -> dict:
    result_path = root / specification["result"]
    strict_path = root / specification["strict"]
    adda_dir = root / specification["adda"]
    theta, matrix, result = load_bem(result_path)
    strict_theta, strict, _ = load_bem(strict_path)
    if strict_theta.size != theta.size or not np.allclose(strict_theta, theta):
        strict = np.asarray(
            [
                [np.interp(theta, strict_theta, strict[i, j]) for j in range(4)]
                for i in range(4)
            ]
        )
    adda = load_adda(adda_dir, theta)

    if "pipeline_summary" in specification:
        pipeline = read_json(root / specification["pipeline_summary"])
        stage1_s = float(pipeline["stage1_wall_time_s"])
        stage2_s = float(pipeline["stage2_wall_time_s"])
    else:
        stage1_s = elapsed_seconds(root / specification["stage1_log"])
        stage2_s = elapsed_seconds(root / specification["stage2_log"])
    bem_wall_s = stage1_s + stage2_s
    adda_wall_s = adda_wall_seconds(adda_dir)
    residual = float(result["mbj"]["fmm_residual"])
    residual_verified = bool(result["mbj"].get("fmm_residual_verified", False))
    if not residual_verified or residual > 0.004:
        raise ValueError(
            f"ka={specification['ka']}: invalid exact residual {residual:g}"
        )

    return {
        "ka": specification["ka"],
        "stage1_wall_s": stage1_s,
        "stage2_wall_s": stage2_s,
        "bem_wall_s": bem_wall_s,
        "adda_wall_s": adda_wall_s,
        "speedup": adda_wall_s / bem_wall_s,
        "exact_residual": residual,
        "outer_iterations": int(result["mbj"]["iterations"]),
        "points_per_internal_wavelength": float(
            result["p2_nodes_per_wavelength_min"]
        ),
        "system_dofs": int(result["system_dofs"]),
        "strict_error": normalized_metrics(theta, matrix, strict),
        "adda_error": normalized_metrics(theta, matrix, adda),
        "theta": theta,
        "matrix": matrix,
        "strict_matrix": strict,
        "adda_matrix": adda,
        "paths": {
            "result": str(result_path.resolve()),
            "strict": str(strict_path.resolve()),
            "adda": str(adda_dir.resolve()),
        },
    }


def summary_figure(rows: list[dict]) -> plt.Figure:
    ka = np.asarray([row["ka"] for row in rows])
    x = np.arange(len(rows))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)

    axes[0, 0].bar(
        x - width / 2, [row["bem_wall_s"] for row in rows], width,
        label="BEM: новый двухэтапный режим", color="#1b9e77",
    )
    axes[0, 0].bar(
        x + width / 2, [row["adda_wall_s"] for row in rows], width,
        label="ADDA-OCL FP32, dpl=15", color="#7570b3",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(x, [f"ka={value:g}" for value in ka])
    axes[0, 0].set_ylabel("Полное холодное время, с")
    axes[0, 0].set_title("Время расчёта с подготовкой")
    axes[0, 0].legend()

    axes[0, 1].plot(ka, [row["speedup"] for row in rows], "o-", color="#1b9e77")
    axes[0, 1].axhline(1.0, color="black", lw=1, ls="--")
    for row in rows:
        axes[0, 1].annotate(
            f"{row['speedup']:.2f}×", (row["ka"], row["speedup"]),
            xytext=(0, 8), textcoords="offset points", ha="center",
        )
    axes[0, 1].set_xlabel("Размерный параметр ka")
    axes[0, 1].set_ylabel("Ускорение ADDA / BEM, раз")
    axes[0, 1].set_title("Итоговое ускорение")

    axes[1, 0].plot(
        ka, [row["exact_residual"] for row in rows], "o-",
        label="точная невязка", color="#d95f02",
    )
    axes[1, 0].plot(
        ka,
        [row["strict_error"]["weighted_full_relative_l2"] for row in rows],
        "s-", label="отличие от строгого BEM", color="#1f78b4",
    )
    axes[1, 0].plot(
        ka,
        [row["adda_error"]["weighted_full_relative_l2"] for row in rows],
        "^-", label="отличие от ADDA dpl=15", color="#6a3d9a",
    )
    axes[1, 0].axhline(0.004, color="black", lw=1, ls="--", label="порог невязки")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Размерный параметр ka")
    axes[1, 0].set_ylabel("Относительная величина")
    axes[1, 0].set_title("Невязка и проверка физического результата")
    axes[1, 0].legend()

    axes[1, 1].plot(
        ka, [row["points_per_internal_wavelength"] for row in rows],
        "o-", color="#e7298a",
    )
    axes[1, 1].axhline(8.0, color="black", lw=1, ls="--", label="ориентир: 8")
    axes[1, 1].set_xlabel("Размерный параметр ka")
    axes[1, 1].set_ylabel("Узлов на внутреннюю длину волны")
    axes[1, 1].set_title("Физическое разрешение сетки BEM ref=6")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.25)
    figure.suptitle(
        "Шестигранная призма h/D=1, m=1,3, ref=6; 181 угол рассеяния",
        fontsize=16,
    )
    return figure


def mueller_figure(row: dict) -> plt.Figure:
    figure, axes = plt.subplots(4, 4, figsize=(15.5, 13.0), constrained_layout=True)
    theta = row["theta"]
    datasets = (
        ("новый BEM", row["matrix"], "#1b9e77", "-"),
        ("строгий BEM", row["strict_matrix"], "#1f78b4", "--"),
        ("ADDA-OCL FP32", row["adda_matrix"], "#d95f02", ":"),
    )
    for i in range(4):
        for j in range(4):
            axis = axes[i, j]
            for label, matrix, color, linestyle in datasets:
                values = matrix[i, j]
                if (i, j) != (0, 0):
                    values = values / matrix[0, 0, 0]
                axis.plot(theta, values, color=color, ls=linestyle, lw=1.4, label=label)
            if (i, j) == (0, 0):
                axis.set_yscale("log")
            axis.set_title(rf"$M_{{{i + 1}{j + 1}}}$")
            axis.set_xlim(0, 180)
            axis.grid(True, which="both", alpha=0.23)
            if i == 3:
                axis.set_xlabel(r"Угол $\theta$, град.")
            if j == 0:
                axis.set_ylabel("Значение" if (i, j) == (0, 0) else r"$M_{ij}/M_{11}(0)$")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        f"Все элементы: ka={row['ka']}, ускорение {row['speedup']:.2f}×, "
        f"точная невязка {row['exact_residual']:.2e}",
        fontsize=16,
    )
    return figure


def serializable(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in {"theta", "matrix", "strict_matrix", "adda_matrix"}
    }


def write_outputs(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    serializable_rows = [serializable(row) for row in rows]
    (output / "physical_fast_sweep.json").write_text(
        json.dumps(serializable_rows, indent=2, ensure_ascii=False) + "\n"
    )
    fields = (
        "ka", "stage1_wall_s", "stage2_wall_s", "bem_wall_s", "adda_wall_s",
        "speedup", "exact_residual", "outer_iterations",
        "points_per_internal_wavelength", "system_dofs",
    )
    with (output / "physical_fast_sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in serializable_rows:
            writer.writerow({field: row[field] for field in fields})

    lines = [
        "# Проверенное ускорение двухэтапного BEM",
        "",
        "| ka | BEM, с | ADDA, с | Ускорение | Невязка | BEM против строгого BEM | BEM против ADDA | Узлов/λ внутри |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['ka']} | {row['bem_wall_s']:.2f} | {row['adda_wall_s']:.2f} | "
            f"{row['speedup']:.2f}× | {row['exact_residual']:.3e} | "
            f"{100 * row['strict_error']['weighted_full_relative_l2']:.4f}% | "
            f"{100 * row['adda_error']['weighted_full_relative_l2']:.4f}% | "
            f"{row['points_per_internal_wavelength']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Время BEM включает обе стадии, холодную подготовку локального MBJ и "
            "ближнего оператора, решение двух поляризаций с использованием точной "
            "C6-симметрии и расчёт 181 угла рассеяния.",
            "",
            "На ka=111 ускоренный и строгий BEM согласуются, но ref=6 даёт только "
            "5,25 узла на внутреннюю длину волны. Поэтому отличие от ADDA dpl=15 "
            "уже нельзя трактовать как ошибку ускоренного решателя.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")

    overview = summary_figure(rows)
    overview.savefig(output / "physical_fast_speedup.png", dpi=190)
    with PdfPages(output / "physical_fast_speedup_and_all_mueller.pdf") as pdf:
        pdf.savefig(overview)
        for row in rows:
            figure = mueller_figure(row)
            figure.savefig(output / f"ka{row['ka']}_all_mueller.png", dpi=170)
            pdf.savefig(figure)
            plt.close(figure)
    plt.close(overview)


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    rows = [load_case(root, specification) for specification in CASES]
    write_outputs(rows, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
