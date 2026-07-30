#!/usr/bin/env python3
"""Build the strict ka=30 H(div)-BEM versus ADDA comparison."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_nodal_bem_adda import comparison_metrics, load_adda, load_bem


ROOT = Path("runs/hdiv_bem_vs_adda_sweep_n1p3/ka30")
EDGE = ROOT / "edge_refinement"
PREFIX = EDGE / "ka30_strict_bem_vs_adda"


def wall_seconds(path: Path) -> float:
    text = path.read_text(errors="replace")
    match = re.search(
        r"Elapsed \(wall clock\) time .*?:\s*"
        r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)",
        text,
    )
    if not match:
        raise ValueError(f"cannot parse wall time from {path}")
    return (
        3600.0 * int(match.group(1) or 0)
        + 60.0 * int(match.group(2))
        + float(match.group(3))
    )


def percent(metrics: dict, key: str) -> float:
    return 100.0 * metrics[key]


def main() -> None:
    fmm_path = EDGE / "bem_ref4_edge1_exactC6.json"
    pfft_path = EDGE / "bem_ref4_edge1_pfft_fgmres_exactC6.json"
    old_ref4_path = ROOT / "bem_ref4_sparse_c6_batch3_fused.json"
    old_ref5_path = ROOT / "bem_ref5_sparse_c6_batch3_fused.json"

    theta, strict_bem, strict_info = load_bem(
        pfft_path, EDGE / "bem_ref4_edge1_pfft_fgmres_exactC6.time"
    )
    _, edge_fmm, edge_info = load_bem(
        fmm_path, EDGE / "bem_ref4_edge1_exactC6.time"
    )
    _, old_ref4, old_ref4_info = load_bem(
        old_ref4_path, ROOT / "bem_ref4_sparse_c6_batch3_fused.time"
    )
    _, old_ref5, old_ref5_info = load_bem(
        old_ref5_path, ROOT / "bem_ref5_sparse_c6_batch3_fused.time"
    )

    adda = {}
    adda_info = {}
    for dpl in (15, 20, 25):
        directory = ROOT / f"adda_dpl{dpl}"
        if not (directory / "mueller").exists():
            continue
        adda[dpl], adda_info[dpl] = load_adda(directory, theta)

    strict_json = json.loads(pfft_path.read_text())
    strict_vs_edge = comparison_metrics(theta, strict_bem, edge_fmm)
    strict_vs_ref5 = comparison_metrics(theta, strict_bem, old_ref5)
    ref4_vs_ref5 = comparison_metrics(theta, old_ref5, old_ref4)
    ref4_vs_edge = comparison_metrics(theta, edge_fmm, old_ref4)

    comparison_rows = []
    for dpl, matrix in sorted(adda.items()):
        metrics = comparison_metrics(theta, strict_bem, matrix)
        comparison_rows.append(
            {
                "dpl": dpl,
                "adda_wall_s": adda_info[dpl]["process_wall_s"],
                "bem_adda_full_percent": percent(
                    metrics, "solid_angle_weighted_full_relative_l2"
                ),
                "bem_adda_M11_percent": percent(
                    metrics, "solid_angle_weighted_M11_relative_l2"
                ),
                "bem_adda_forward_normalized_full_percent": percent(
                    metrics, "forward_normalized_full_relative_l2"
                ),
            }
        )

    adda_convergence = []
    dpls = sorted(adda)
    for coarse, fine in zip(dpls, dpls[1:]):
        metrics = comparison_metrics(theta, adda[fine], adda[coarse])
        adda_convergence.append(
            {
                "coarse_dpl": coarse,
                "fine_dpl": fine,
                "full_percent": percent(
                    metrics, "solid_angle_weighted_full_relative_l2"
                ),
                "M11_percent": percent(
                    metrics, "solid_angle_weighted_M11_relative_l2"
                ),
            }
        )

    summary = {
        "case": {
            "shape": "regular hexagonal prism",
            "aspect_h_over_Dx": 1.0,
            "ka": 30.0,
            "refractive_index": 1.3,
            "relative_solver_tolerance": 1.0e-5,
        },
        "strict_bem": {
            "method": "H(div)-BDM1 edge-refined Muller BEM, pFFT-FGMRES",
            "unknowns": strict_info["unknowns"],
            "wall_s": strict_info["wall_s"],
            "outer_fmm_iterations": strict_json["pfft_fgmres"][
                "outer_iterations"
            ],
            "outer_fmm_residual": strict_json["pfft_fgmres"][
                "fmm_residual"
            ],
            "inner_pfft_iterations": strict_json["pfft_fgmres"][
                "inner_iterations"
            ],
            "gpu_memory_mb": strict_json["pfft_fgmres"][
                "combined_gpu_memory_delta_mb"
            ],
            "speedup_over_adda_dpl20": (
                adda_info[20]["process_wall_s"] / strict_info["wall_s"]
                if 20 in adda_info
                else None
            ),
            "speedup_over_adda_dpl25": (
                adda_info[25]["process_wall_s"] / strict_info["wall_s"]
                if 25 in adda_info
                else None
            ),
            "difference_from_edge_fmm_full_percent": percent(
                strict_vs_edge, "solid_angle_weighted_full_relative_l2"
            ),
            "difference_from_ref5_full_percent": percent(
                strict_vs_ref5, "solid_angle_weighted_full_relative_l2"
            ),
        },
        "bem_grid_convergence": {
            "uniform_ref4_to_uniform_ref5_full_percent": percent(
                ref4_vs_ref5, "solid_angle_weighted_full_relative_l2"
            ),
            "uniform_ref4_to_edge_refined_full_percent": percent(
                ref4_vs_edge, "solid_angle_weighted_full_relative_l2"
            ),
            "edge_refined_to_uniform_ref5_full_percent": percent(
                strict_vs_ref5, "solid_angle_weighted_full_relative_l2"
            ),
        },
        "timings_s": {
            "bem_uniform_ref4": old_ref4_info["wall_s"],
            "bem_uniform_ref5": old_ref5_info["wall_s"],
            "bem_edge_fmm_cold": edge_info["wall_s"],
            "bem_strict_pfft_fgmres": strict_info["wall_s"],
            **{
                f"adda_dpl{dpl}": info["process_wall_s"]
                for dpl, info in sorted(adda_info.items())
            },
        },
        "adda_grid_convergence": adda_convergence,
        "bem_vs_adda": comparison_rows,
    }

    EDGE.mkdir(parents=True, exist_ok=True)
    PREFIX.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with PREFIX.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(comparison_rows[0])
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    fig, axes = plt.subplots(
        2, 2, figsize=(13.8, 9.2), constrained_layout=True
    )

    axis = axes[0, 0]
    axis.plot(
        theta,
        strict_bem[0, 0] / strict_bem[0, 0, 0],
        color="black",
        lw=2.3,
        label="BEM: строгий pFFT-FGMRES",
    )
    adda_colors = {15: "#93c5fd", 20: "#2563eb", 25: "#1e3a8a"}
    for dpl, matrix in sorted(adda.items()):
        axis.plot(
            theta,
            matrix[0, 0] / matrix[0, 0, 0],
            color=adda_colors[dpl],
            lw=1.5,
            label=f"ADDA dpl={dpl}",
        )
    axis.set_yscale("log")
    axis.set_title(r"Угловая зависимость $M_{11}$")
    axis.set_xlabel(r"Угол рассеяния $\theta$, град.")
    axis.set_ylabel(r"$M_{11}(\theta)/M_{11}(0)$")
    axis.legend()

    axis = axes[0, 1]
    dpl_values = [row["dpl"] for row in comparison_rows]
    axis.plot(
        dpl_values,
        [row["bem_adda_full_percent"] for row in comparison_rows],
        "o-",
        lw=2.2,
        label="полная матрица Мюллера",
    )
    axis.plot(
        dpl_values,
        [row["bem_adda_M11_percent"] for row in comparison_rows],
        "s-",
        lw=2.2,
        label=r"$M_{11}$",
    )
    axis.axhline(
        percent(strict_vs_ref5, "solid_angle_weighted_full_relative_l2"),
        color="black",
        ls="--",
        lw=1.5,
        label="BEM edge ↔ BEM ref5",
    )
    axis.set_xticks(dpl_values)
    axis.set_title("Согласие BEM с ADDA")
    axis.set_xlabel("Число диполей на длину волны, dpl")
    axis.set_ylabel("Взвешенное относительное расхождение, %")
    axis.legend()

    axis = axes[1, 0]
    time_labels = [
        "BEM\nref4",
        "BEM\nref5",
        "BEM edge\nFMM",
        "BEM edge\npFFT-FGMRES",
    ]
    time_values = [
        old_ref4_info["wall_s"],
        old_ref5_info["wall_s"],
        edge_info["wall_s"],
        strict_info["wall_s"],
    ]
    time_colors = ["#9ca3af", "#6b7280", "#d97706", "#16a34a"]
    for dpl in dpl_values:
        time_labels.append(f"ADDA\ndpl={dpl}")
        time_values.append(adda_info[dpl]["process_wall_s"])
        time_colors.append(adda_colors[dpl])
    bars = axis.bar(time_labels, time_values, color=time_colors)
    axis.bar_label(bars, fmt="%.1f с", padding=3, fontsize=9)
    axis.set_yscale("log")
    speedup_text = []
    for dpl in (20, 25):
        if dpl in adda_info:
            speedup = adda_info[dpl]["process_wall_s"] / strict_info["wall_s"]
            speedup_text.append(f"{speedup:.2f}× при dpl={dpl}")
    axis.set_title(
        "Полное время расчета\n"
        + "Строгий BEM быстрее ADDA: "
        + "; ".join(speedup_text)
    )
    axis.set_ylabel("Время, с")

    axis = axes[1, 1]
    convergence_labels = [
        "BEM\nref4→edge",
        "BEM\nedge→ref5",
    ]
    convergence_full = [
        percent(ref4_vs_edge, "solid_angle_weighted_full_relative_l2"),
        percent(strict_vs_ref5, "solid_angle_weighted_full_relative_l2"),
    ]
    convergence_m11 = [
        percent(ref4_vs_edge, "solid_angle_weighted_M11_relative_l2"),
        percent(strict_vs_ref5, "solid_angle_weighted_M11_relative_l2"),
    ]
    for row in adda_convergence:
        convergence_labels.append(
            f"ADDA\n{row['coarse_dpl']}→{row['fine_dpl']}"
        )
        convergence_full.append(row["full_percent"])
        convergence_m11.append(row["M11_percent"])
    x = np.arange(len(convergence_labels))
    width = 0.35
    axis.bar(
        x - width / 2,
        convergence_full,
        width,
        label="полная матрица",
    )
    axis.bar(
        x + width / 2,
        convergence_m11,
        width,
        label=r"$M_{11}$",
    )
    axis.set_yscale("log")
    axis.set_xticks(x, convergence_labels)
    axis.set_title("Собственная сходимость сеток")
    axis.set_ylabel("Изменение результата, %")
    axis.legend()

    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.25)
    fig.suptitle(
        r"Шестигранная призма: $h/D_x=1$, $ka=30$, $m=1{,}3$, "
        r"невязка $10^{-5}$",
        fontsize=15,
    )
    fig.savefig(PREFIX.with_suffix(".png"), dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
