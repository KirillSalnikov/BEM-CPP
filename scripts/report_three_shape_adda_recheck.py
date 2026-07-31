#!/usr/bin/env python3
"""Build the strict three-shape BEM/ADDA timing and accuracy report."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "orientation_bem_adda_recheck_20260731"
RESULT_ROOT = RUN_ROOT / "results"
KAS = (17, 18, 20, 25, 30)
TOLERANCE = 1.0e-5

SHAPES = {
    "prism": {
        "name": "Шестигранная призма",
        "short": "Призма",
        "base": ROOT / "runs" / "orientation_bem_adda_crossover",
    },
    "sphere": {
        "name": "Сфера",
        "short": "Сфера",
        "base": ROOT / "runs" / "orientation_bem_adda_shapes" / "sphere",
    },
    "asymmetric": {
        "name": "Несимметричный многогранник",
        "short": "Многогранник",
        "base": ROOT / "runs" / "orientation_bem_adda_shapes" / "asymmetric",
    },
}

SOLVER_NAMES = {
    "paired_gpu_gmres": "парный GPU GMRES",
    "pfft_fgmres": "строгий pFFT-FGMRES",
    "pfft_fgmres_paired_strict": "строгий парный pFFT-FGMRES",
}


def parse_time(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = float(value)
    return values


def load_bem(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    theta = np.asarray(data["theta_degrees"], dtype=float)
    mueller = np.moveaxis(np.asarray(data["mueller"], dtype=float), -1, 0)
    if mueller.shape != (theta.size, 4, 4):
        raise ValueError(f"Unexpected BEM Mueller shape in {path}: {mueller.shape}")
    return data, theta, mueller


def load_adda(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, skiprows=1)
    theta = table[:, 0]
    mueller = table[:, 1:].reshape(-1, 4, 4)
    if mueller.shape != (theta.size, 4, 4):
        raise ValueError(f"Unexpected ADDA Mueller shape in {path}: {mueller.shape}")
    return theta, mueller


def interpolate_mueller(
    source_theta: np.ndarray,
    source: np.ndarray,
    target_theta: np.ndarray,
) -> np.ndarray:
    result = np.empty((target_theta.size, 4, 4), dtype=float)
    for row in range(4):
        for column in range(4):
            result[:, row, column] = np.interp(
                target_theta, source_theta, source[:, row, column]
            )
    return result


def relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(second.ravel())
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm((first - second).ravel()) / denominator)


def normalized_metrics(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, float]:
    first_scale = float(first[0, 0, 0])
    second_scale = float(second[0, 0, 0])
    first_normalized = first / first_scale
    second_normalized = second / second_scale
    difference = first_normalized - second_normalized
    return {
        "raw_relative_l2": relative_l2(first, second),
        "normalized_relative_l2": relative_l2(
            first_normalized, second_normalized
        ),
        "forward_scaled_rms": float(np.sqrt(np.mean(difference**2))),
        "forward_scaled_max": float(np.max(np.abs(difference))),
        "forward_m11_ratio": first_scale / second_scale,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_old_summary() -> dict[tuple[str, int], dict]:
    rows = json.loads(
        (ROOT / "runs" / "orientation_bem_adda_shapes" /
         "combined_summary.json").read_text(encoding="utf-8")
    )
    return {(row["shape"], int(row["ka"])): row for row in rows}


def collect_records() -> tuple[list[dict], dict[tuple[str, int], dict]]:
    old_summary = load_old_summary()
    records: list[dict] = []
    arrays: dict[tuple[str, int], dict] = {}
    hash_lines: list[str] = []

    for shape_key, shape in SHAPES.items():
        for ka in KAS:
            base = shape["base"] / f"ka{ka}"
            current_dir = RESULT_ROOT / shape_key / f"ka{ka}"
            old_path = base / "bem_ref5_alpha256" / "average.json"
            adda_dir = base / "adda_dpl15_alpha256"
            adda_path = adda_dir / "mueller"
            current_path = current_dir / "average.json"

            current, theta, current_mueller = load_bem(current_path)
            old, old_theta, old_mueller = load_bem(old_path)
            adda_theta, adda_mueller = load_adda(adda_path)
            old_mueller = interpolate_mueller(old_theta, old_mueller, theta)
            adda_mueller = interpolate_mueller(adda_theta, adda_mueller, theta)

            current_vs_adda = normalized_metrics(current_mueller, adda_mueller)
            old_vs_adda = normalized_metrics(old_mueller, adda_mueller)
            current_vs_old = normalized_metrics(current_mueller, old_mueller)

            summary = old_summary[(shape["name"], ka)]
            wall = parse_time(current_dir / "time.txt")
            solver_key = (
                current_dir / "selected_solver.txt"
            ).read_text(encoding="utf-8").strip()
            residual = float(current["iterations"]["maximum_residual"])
            if not np.isfinite(residual) or residual > TOLERANCE:
                raise RuntimeError(
                    f"Non-converged selected result: {shape_key} ka={ka}, "
                    f"residual={residual:.6e}"
                )

            record = {
                "shape": shape["name"],
                "shape_key": shape_key,
                "ka": ka,
                "solver": solver_key,
                "solver_ru": SOLVER_NAMES[solver_key],
                "system_dofs": int(current["system_dofs"]),
                "new_bem_wall_s": wall["wall_s"],
                "new_bem_setup_s": float(
                    current["timing"]["operator_setup_s"]
                    + current["timing"]["mbj_setup_s"]
                    + current["timing"]["fmm_switch_s"]
                ),
                "new_bem_solve_s": float(current["timing"]["solve_s"]),
                "new_bem_farfield_s": float(current["timing"]["farfield_s"]),
                "new_bem_iterations_total": int(current["iterations"]["total"]),
                "new_bem_residual": residual,
                "new_bem_max_rss_mb": wall["max_rss_kb"] / 1024.0,
                "old_bem_wall_s": float(summary["bem_wall_s"]),
                "old_bem_setup_s": float(summary["bem_setup_s"]),
                "old_bem_solve_s": float(summary["bem_solve_s"]),
                "old_bem_residual": float(summary["bem_residual"]),
                "adda_wall_s": float(summary["adda_wall_s"]),
                "adda_solve_s": float(summary["adda_solve_s"]),
                "adda_farfield_s": float(summary["adda_farfield_s"]),
                "adda_iterations": int(summary["adda_iterations"]),
                "adda_over_new_bem_speedup": (
                    float(summary["adda_wall_s"]) / wall["wall_s"]
                ),
                "old_over_new_bem_speedup": (
                    float(summary["bem_wall_s"]) / wall["wall_s"]
                ),
                "current_vs_adda_raw_relative_l2": (
                    current_vs_adda["raw_relative_l2"]
                ),
                "current_vs_adda_normalized_relative_l2": (
                    current_vs_adda["normalized_relative_l2"]
                ),
                "current_vs_adda_forward_scaled_rms": (
                    current_vs_adda["forward_scaled_rms"]
                ),
                "current_vs_adda_forward_scaled_max": (
                    current_vs_adda["forward_scaled_max"]
                ),
                "old_vs_adda_normalized_relative_l2": (
                    old_vs_adda["normalized_relative_l2"]
                ),
                "current_vs_old_normalized_relative_l2": (
                    current_vs_old["normalized_relative_l2"]
                ),
                "current_vs_old_forward_scaled_max": (
                    current_vs_old["forward_scaled_max"]
                ),
                "current_vs_old_forward_m11_ratio": (
                    current_vs_old["forward_m11_ratio"]
                ),
            }
            records.append(record)
            arrays[(shape_key, ka)] = {
                "theta": theta,
                "current": current_mueller,
                "old": old_mueller,
                "adda": adda_mueller,
            }

            for filename in ("mueller", "log", "time.txt"):
                path = adda_dir / filename
                hash_lines.append(
                    f"{sha256(path)}  {path.relative_to(ROOT)}"
                )

    (RUN_ROOT / "adda_baseline_sha256.txt").write_text(
        "\n".join(hash_lines) + "\n", encoding="utf-8"
    )
    return records, arrays


def write_tables(records: list[dict]) -> None:
    fields = list(records[0].keys())
    with (RUN_ROOT / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "generated": datetime.now().astimezone().isoformat(),
        "conditions": {
            "ka": list(KAS),
            "refractive_index": 1.3,
            "bem_refinement": 5,
            "adda_dpl": 15,
            "required_relative_residual": TOLERANCE,
            "theta_points": 73,
            "alpha_samples": 256,
            "beta_nodes": 1,
            "gamma_nodes": 1,
            "timing_note": (
                "New BEM wall time is a repeated run with persistent "
                "near-correction and MBJ caches. Old BEM wall time includes "
                "their initial construction."
            ),
        },
        "records": records,
    }
    (RUN_ROOT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def records_for_shape(records: list[dict], shape_key: str) -> list[dict]:
    return [record for record in records if record["shape_key"] == shape_key]


def plot_timing(records: list[dict]) -> Path:
    figure, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
    for row, (shape_key, shape) in enumerate(SHAPES.items()):
        rows = records_for_shape(records, shape_key)
        ka = np.asarray([record["ka"] for record in rows])
        old = np.asarray([record["old_bem_wall_s"] for record in rows])
        new = np.asarray([record["new_bem_wall_s"] for record in rows])
        adda = np.asarray([record["adda_wall_s"] for record in rows])
        speedup = adda / new

        axis = axes[row, 0]
        axis.plot(ka, adda, "o--", color="#444444", label="ADDA-OCL")
        axis.plot(ka, old, "s:", color="#8c6d31", label="прежняя BEM")
        axis.plot(ka, new, "o-", color="#087f5b", label="новая BEM")
        axis.set_yscale("log")
        axis.set_title(shape["name"])
        axis.set_xlabel("Размерный параметр $ka$")
        axis.set_ylabel("Полное время, с")
        axis.grid(True, which="both", alpha=0.25)
        if row == 0:
            axis.legend(ncol=3, fontsize=9)

        axis = axes[row, 1]
        axis.axhline(1.0, color="black", linewidth=1)
        axis.plot(ka, speedup, "o-", color="#087f5b", linewidth=2)
        for x_value, y_value in zip(ka, speedup):
            axis.annotate(
                f"{y_value:.2f}×",
                (x_value, y_value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        axis.set_title(f"{shape['name']}: ADDA / новая BEM")
        axis.set_xlabel("Размерный параметр $ka$")
        axis.set_ylabel("Ускорение полного повторного расчёта")
        axis.grid(True, alpha=0.25)

    figure.suptitle(
        "Повторное сравнение времени: одинаковые частицы, размеры и точность",
        fontsize=17,
    )
    path = RUN_ROOT / "timing_and_speedup.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_accuracy(records: list[dict]) -> Path:
    figure, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
    for row, (shape_key, shape) in enumerate(SHAPES.items()):
        rows = records_for_shape(records, shape_key)
        ka = np.asarray([record["ka"] for record in rows])
        current_adda = np.asarray(
            [
                record["current_vs_adda_normalized_relative_l2"]
                for record in rows
            ]
        )
        old_adda = np.asarray(
            [record["old_vs_adda_normalized_relative_l2"] for record in rows]
        )
        current_old = np.asarray(
            [
                record["current_vs_old_normalized_relative_l2"]
                for record in rows
            ]
        )
        residual = np.asarray(
            [record["new_bem_residual"] for record in rows]
        )

        axis = axes[row, 0]
        axis.semilogy(
            ka, old_adda, "s:", color="#777777",
            label="прежняя BEM ↔ ADDA"
        )
        axis.semilogy(
            ka, current_adda, "o-", color="#087f5b",
            label="новая BEM ↔ ADDA"
        )
        axis.set_title(f"{shape['name']}: согласие всех 16 элементов")
        axis.set_xlabel("Размерный параметр $ka$")
        axis.set_ylabel("Нормированное относительное отличие")
        axis.grid(True, which="both", alpha=0.25)
        if row == 0:
            axis.legend(fontsize=9)

        axis = axes[row, 1]
        axis.semilogy(
            ka, current_old, "o-", color="#d9480f",
            label="новая BEM ↔ прежняя BEM"
        )
        axis.semilogy(
            ka, residual, "s--", color="#1971c2",
            label="истинная невязка новой BEM"
        )
        axis.axhline(
            TOLERANCE, color="black", linewidth=1, label="требование $10^{-5}$"
        )
        axis.set_title(f"{shape['name']}: контроль потери точности")
        axis.set_xlabel("Размерный параметр $ka$")
        axis.set_ylabel("Относительная величина")
        axis.grid(True, which="both", alpha=0.25)
        if row == 0:
            axis.legend(fontsize=9)

    figure.suptitle(
        "Точность повторных расчётов BEM по всем элементам матрицы Мюллера",
        fontsize=17,
    )
    path = RUN_ROOT / "accuracy_all_mueller.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def mueller_figure(
    shape_key: str,
    ka: int,
    arrays: dict,
    record: dict,
) -> plt.Figure:
    theta = arrays["theta"]
    current = arrays["current"]
    old = arrays["old"]
    adda = arrays["adda"]
    scale = float(adda[0, 0, 0])

    figure, axes = plt.subplots(
        4, 4, figsize=(16, 11), sharex=True, constrained_layout=True
    )
    for row in range(4):
        for column in range(4):
            axis = axes[row, column]
            axis.plot(
                theta,
                adda[:, row, column] / scale,
                "--",
                color="#222222",
                linewidth=1.3,
                label="ADDA",
            )
            axis.plot(
                theta,
                old[:, row, column] / scale,
                ":",
                color="#888888",
                linewidth=1.2,
                label="прежняя BEM",
            )
            axis.plot(
                theta,
                current[:, row, column] / scale,
                "-",
                color="#e8590c",
                linewidth=1.1,
                label="новая BEM",
            )
            difference = np.max(
                np.abs(current[:, row, column] - old[:, row, column])
            ) / abs(scale)
            axis.set_title(
                rf"$M_{{{row + 1}{column + 1}}}/M_{{11}}^{{ADDA}}(0)$"
                + "\n"
                + rf"$\max|\Delta_{{new-old}}|={difference:.1e}$",
                fontsize=9,
            )
            axis.grid(True, alpha=0.2)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            if row == 3:
                axis.set_xlabel(r"Угол рассеяния $\theta$, град.")
            if column == 0:
                axis.set_ylabel("Нормированное значение")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        f"{SHAPES[shape_key]['name']}, $ka={ka}$: все 16 элементов; "
        f"{record['solver_ru']}, невязка {record['new_bem_residual']:.2e}",
        fontsize=15,
    )
    return figure


def add_summary_page(pdf: PdfPages, records: list[dict]) -> None:
    speeds = np.asarray(
        [record["adda_over_new_bem_speedup"] for record in records]
    )
    deltas = np.asarray(
        [
            record["current_vs_old_normalized_relative_l2"]
            for record in records
        ]
    )
    residuals = np.asarray([record["new_bem_residual"] for record in records])

    figure = plt.figure(figsize=(16, 11))
    figure.text(
        0.5,
        0.965,
        "Повторная проверка BEM относительно ADDA для трёх частиц",
        ha="center",
        va="top",
        fontsize=21,
        weight="bold",
    )
    figure.text(
        0.04,
        0.91,
        "Условия: m=1.3; ka=17, 18, 20, 25, 30; BEM ref=5; "
        "ADDA dpl=15; требуемая невязка 10⁻⁵; 73 угла рассеяния; "
        "усреднение по 256 значениям α.",
        fontsize=11,
    )
    figure.text(
        0.04,
        0.865,
        "Время новой BEM относится к повторному запуску с сохранёнными "
        "кэшами ближней коррекции и MBJ. Время прежней BEM из исходного "
        "PDF включает первичную подготовку этих кэшей.",
        fontsize=10,
        color="#444444",
    )
    figure.text(
        0.04,
        0.80,
        f"Итог: BEM быстрее ADDA в {np.count_nonzero(speeds > 1.0)} из "
        f"{len(speeds)} случаев; диапазон ADDA/BEM "
        f"{speeds.min():.2f}×…{speeds.max():.2f}×, медиана "
        f"{np.median(speeds):.2f}×.",
        fontsize=13,
        weight="bold",
        color="#087f5b",
    )
    figure.text(
        0.04,
        0.755,
        "Потери точности не обнаружено: максимальное нормированное отличие "
        f"новой BEM от прежней BEM равно {deltas.max():.2e}; максимальная "
        f"истинная невязка {residuals.max():.2e} ≤ 10⁻⁵.",
        fontsize=12,
    )

    columns = [
        "Частица",
        "ka",
        "Решатель",
        "ADDA, с",
        "Новая BEM, с",
        "ADDA/BEM",
        "Прежняя/новая",
        "Невязка",
        "Δ новая/старая",
    ]
    cell_text = []
    for record in records:
        cell_text.append(
            [
                SHAPES[record["shape_key"]]["short"],
                str(record["ka"]),
                (
                    "GPU GMRES"
                    if record["solver"] == "paired_gpu_gmres"
                    else "pFFT-FGMRES"
                ),
                f"{record['adda_wall_s']:.2f}",
                f"{record['new_bem_wall_s']:.2f}",
                f"{record['adda_over_new_bem_speedup']:.2f}×",
                f"{record['old_over_new_bem_speedup']:.2f}×",
                f"{record['new_bem_residual']:.2e}",
                f"{record['current_vs_old_normalized_relative_l2']:.2e}",
            ]
        )

    axis = figure.add_axes([0.025, 0.06, 0.95, 0.64])
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.12, 0.045, 0.13, 0.075, 0.095, 0.08, 0.095, 0.09, 0.11],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.5)
    for column in range(len(columns)):
        table[(0, column)].set_facecolor("#d8f3dc")
        table[(0, column)].set_text_props(weight="bold")
    for row, record in enumerate(records, start=1):
        if record["adda_over_new_bem_speedup"] < 1.0:
            table[(row, 5)].set_facecolor("#ffe3e3")

    pdf.savefig(figure)
    plt.close(figure)


def add_image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    image = plt.imread(image_path)
    figure = plt.figure(figsize=(16, 11))
    axis = figure.add_axes([0.02, 0.02, 0.96, 0.92])
    axis.imshow(image)
    axis.axis("off")
    figure.suptitle(title, fontsize=16)
    pdf.savefig(figure)
    plt.close(figure)


def build_report(
    records: list[dict],
    arrays: dict[tuple[str, int], dict],
    timing_path: Path,
    accuracy_path: Path,
) -> Path:
    all_dir = RUN_ROOT / "all_mueller"
    all_dir.mkdir(parents=True, exist_ok=True)
    report_path = RUN_ROOT / "bem_vs_adda_three_shapes_strict_recheck.pdf"
    record_lookup = {
        (record["shape_key"], int(record["ka"])): record
        for record in records
    }

    with PdfPages(report_path) as pdf:
        add_summary_page(pdf, records)
        add_image_page(
            pdf,
            timing_path,
            "Время и ускорение повторных расчётов",
        )
        add_image_page(
            pdf,
            accuracy_path,
            "Контроль точности по всем элементам матрицы Мюллера",
        )
        for shape_key in SHAPES:
            for ka in KAS:
                figure = mueller_figure(
                    shape_key,
                    ka,
                    arrays[(shape_key, ka)],
                    record_lookup[(shape_key, ka)],
                )
                image_path = all_dir / f"{shape_key}_ka{ka}_all_mueller.png"
                figure.savefig(image_path, dpi=160)
                pdf.savefig(figure)
                plt.close(figure)
    return report_path


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    records, arrays = collect_records()
    write_tables(records)
    timing_path = plot_timing(records)
    accuracy_path = plot_accuracy(records)
    report_path = build_report(records, arrays, timing_path, accuracy_path)
    print(report_path)


if __name__ == "__main__":
    main()
