#!/usr/bin/env python3
"""Plot every Mueller element for quick/standard orientation validation runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_orientation_profiles import CASES, output_for  # noqa: E402
from verify_mie import mie_mueller  # noqa: E402


DEFAULT_INPUT = ROOT / "runs" / "orientation_profile_validation_v4"
DEFAULT_OUTPUT = DEFAULT_INPUT / "all_mueller_elements"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_result(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = load_json(path)
    theta = np.asarray(data["theta_degrees"], dtype=float)
    mueller = np.asarray(data["mueller"], dtype=float)
    expected = (4, 4, theta.size)
    if mueller.shape != expected:
        raise ValueError(f"{path}: expected Mueller shape {expected}, got {mueller.shape}")
    return theta, mueller, data


def case_title(case: dict[str, Any]) -> str:
    if case["shape"] == "sphere":
        shape = "Сфера"
    elif case["shape"] == "cube":
        shape = "Куб"
    elif case["shape"] == "prism":
        shape = (
            f"Призма, граней: {case['sides']}, "
            f"отношение высоты к диаметру: {case['aspect']:g}"
        )
    else:
        shape = "Несимметричная частица из поверхностной сетки"
    return f"{shape}; ka = {case['ka']:g}; m = {case['ri']:g}"


def decimal(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}g}".replace(".", ",")


def seconds(value: float) -> str:
    if abs(value) >= 1000.0:
        return f"{value:,.0f}".replace(",", " ")
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    if abs(value) >= 10.0:
        return f"{value:.1f}".replace(".", ",")
    return f"{value:.2f}".replace(".", ",")


def timing_index(summary_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not summary_path.is_file():
        return {}
    summary = load_json(summary_path)
    return {
        (record["case"], record["profile"]): record
        for record in summary.get("runs", [])
    }


def timing_line(
    case_name: str,
    quick_result: dict[str, Any],
    standard_result: dict[str, Any],
    timings: dict[tuple[str, str], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    quick_record = timings.get((case_name, "quick"), {})
    standard_record = timings.get((case_name, "standard"), {})
    quick_time = quick_record.get(
        "wall_time_s", quick_result.get("timing", {}).get("total_with_setup_s")
    )
    standard_time = standard_record.get(
        "wall_time_s", standard_result.get("timing", {}).get("total_with_setup_s")
    )
    quick_time = float(quick_time)
    standard_time = float(standard_time)
    delta = standard_time - quick_time
    ratio = standard_time / max(quick_time, 1.0e-300)
    scope = standard_record.get("timing_scope", "complete")
    marker = "*" if scope != "complete" else ""
    line = (
        f"Полное время: быстрый {seconds(quick_time)} с; "
        f"стандартный {seconds(standard_time)} с{marker}; "
        f"разность {seconds(delta)} с; отношение {decimal(ratio)} раза"
    )
    return line, {
        "case": case_name,
        "quick_wall_s": quick_time,
        "standard_wall_s": standard_time,
        "standard_minus_quick_s": delta,
        "standard_over_quick": ratio,
        "standard_timing_scope": scope,
    }


def build_figure(
    case: dict[str, Any],
    quick_theta: np.ndarray,
    standard_theta: np.ndarray,
    quick: np.ndarray,
    standard: np.ndarray,
    timing_text: str,
) -> plt.Figure:
    scale = max(abs(float(standard[0, 0, 0])), 1.0e-300)
    mie = None
    if case["shape"] == "sphere":
        mie = np.asarray(
            mie_mueller(
                standard_theta.tolist(), complex(case["ri"], 0.0), case["ka"]
            ),
            dtype=float,
        )

    figure, axes = plt.subplots(
        4, 4, figsize=(16.5, 12.0), sharex=True, constrained_layout=False
    )
    for row in range(4):
        for column in range(4):
            axis = axes[row, column]
            axis.plot(
                quick_theta,
                quick[row, column] / scale,
                color="#e69f00",
                linewidth=1.7,
                linestyle="--",
                label="Быстрый режим",
            )
            axis.plot(
                standard_theta,
                standard[row, column] / scale,
                color="#0072b2",
                linewidth=1.65,
                label="Стандартный режим",
            )
            if mie is not None:
                axis.plot(
                    standard_theta,
                    mie[row, column] / scale,
                    color="#111111",
                    linewidth=1.25,
                    linestyle=":",
                    label="Точное решение Ми",
                )
            axis.set_yscale("symlog", linthresh=1.0e-8, linscale=0.7)
            axis.set_title(
                rf"$M_{{{row + 1}{column + 1}}}/M_{{11}}^{{ст}}(0)$",
                fontsize=11,
            )
            axis.set_xlim(0.0, 180.0)
            axis.set_xticks((0, 45, 90, 135, 180))
            axis.grid(True, which="both", alpha=0.22)
            axis.tick_params(labelsize=8)
            if row == 3:
                axis.set_xlabel(r"Угол рассеяния $\theta$, град.", fontsize=9)
            if column == 0:
                axis.set_ylabel("Общая нормировка", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(labels),
        frameon=False,
        fontsize=11,
    )
    figure.suptitle(
        f"{case_title(case)}\n{timing_text}",
        fontsize=15,
        y=0.988,
    )
    note = (
        r"Все элементы разделены на одно число $M_{11}^{ст}(0)$; "
        "масштаб между режимами сохранён.\nУгловые сетки: "
        f"быстрый режим — {quick_theta.size} точек, стандартный — "
        f"{standard_theta.size} точек. "
        "Звёздочка означает продолжение расчёта с контрольной точки."
    )
    if case["shape"] != "sphere":
        note += " Решение Ми существует только для сферы."
    figure.text(0.5, 0.012, note, ha="center", fontsize=9.5)
    figure.subplots_adjust(
        left=0.055, right=0.985, bottom=0.065, top=0.885, hspace=0.34, wspace=0.24
    )
    return figure


def write_timing_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case",
        "quick_wall_s",
        "standard_wall_s",
        "standard_minus_quick_s",
        "standard_over_quick",
        "standard_timing_scope",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timings = timing_index(input_root / "summary.json")
    timing_rows: list[dict[str, Any]] = []
    pdf_path = output_root / "all_particles_all_mueller_quick_standard_mie.pdf"

    with PdfPages(pdf_path) as pdf:
        for case_name, case in CASES.items():
            quick_path = output_for(input_root, case_name, "quick") / "result.json"
            standard_path = output_for(input_root, case_name, "standard") / "result.json"
            if not quick_path.is_file() or not standard_path.is_file():
                raise FileNotFoundError(
                    f"missing completed quick/standard result for {case_name}"
                )
            quick_theta, quick, quick_result = load_result(quick_path)
            standard_theta, standard, standard_result = load_result(standard_path)
            timing_text, timing_row = timing_line(
                case_name, quick_result, standard_result, timings
            )
            timing_rows.append(timing_row)
            figure = build_figure(
                case,
                quick_theta,
                standard_theta,
                quick,
                standard,
                timing_text,
            )
            png_path = output_root / f"{case_name}_all_mueller.png"
            figure.savefig(png_path, dpi=args.dpi)
            pdf.savefig(figure)
            plt.close(figure)
            print(f"Wrote {png_path}")

    write_timing_csv(output_root / "timing_comparison.csv", timing_rows)
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
