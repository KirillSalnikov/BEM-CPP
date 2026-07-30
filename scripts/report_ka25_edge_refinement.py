#!/usr/bin/env python3
"""Report the ka=25 edge-refined BEM and ADDA convergence study."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_nodal_bem_adda import comparison_metrics, load_adda, load_bem


ROOT = Path("runs/hdiv_bem_vs_adda_sweep_n1p3/ka25")
EDGE = ROOT / "edge_refinement"
PREFIX = EDGE / "ka25_edge_refinement_vs_adda"


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


def main() -> None:
    edge_json = EDGE / "bem_ref4_edge1_exactC6.json"
    warm_json = EDGE / "bem_ref4_edge1_exactC6_warm.json"
    old_ref4_json = ROOT / "bem_ref4_sparse_c6_batch3_fused.json"
    old_ref5_json = ROOT / "bem_ref5_sparse_c6_batch3_fused.json"

    theta, edge_bem, edge_info = load_bem(
        edge_json, EDGE / "bem_ref4_edge1_exactC6.time"
    )
    _, warm_bem, warm_info = load_bem(
        warm_json, EDGE / "bem_ref4_edge1_exactC6_warm.time"
    )
    _, old_ref4, _ = load_bem(old_ref4_json, None)
    _, old_ref5, old_ref5_info = load_bem(
        old_ref5_json,
        ROOT / "bem_ref5_sparse_c6_batch3_fused.time",
    )

    adda = {}
    adda_info = {}
    for dpl in (15, 20, 25):
        adda[dpl], adda_info[dpl] = load_adda(
            ROOT / f"adda_dpl{dpl}", theta
        )

    rows = []
    for dpl in (15, 20, 25):
        metrics = comparison_metrics(theta, edge_bem, adda[dpl])
        rows.append(
            {
                "dpl": dpl,
                "adda_wall_s": adda_info[dpl]["process_wall_s"],
                "bem_adda_full_percent": 100.0
                * metrics["solid_angle_weighted_full_relative_l2"],
                "bem_adda_M11_percent": 100.0
                * metrics["solid_angle_weighted_M11_relative_l2"],
                "bem_adda_forward_normalized_full_percent": 100.0
                * metrics["forward_normalized_full_relative_l2"],
            }
        )

    adda_convergence = []
    for coarse, fine in ((15, 20), (20, 25)):
        metrics = comparison_metrics(theta, adda[fine], adda[coarse])
        adda_convergence.append(
            {
                "coarse_dpl": coarse,
                "fine_dpl": fine,
                "full_percent": 100.0
                * metrics["solid_angle_weighted_full_relative_l2"],
                "M11_percent": 100.0
                * metrics["solid_angle_weighted_M11_relative_l2"],
            }
        )

    edge_vs_ref5 = comparison_metrics(theta, edge_bem, old_ref5)
    old_refinement = comparison_metrics(theta, old_ref5, old_ref4)
    warm_reproducibility = comparison_metrics(theta, warm_bem, edge_bem)
    summary = {
        "case": {
            "shape": "regular hexagonal prism",
            "aspect_h_over_Dx": 1.0,
            "ka": 25.0,
            "refractive_index": 1.3,
            "relative_solver_tolerance": 1.0e-5,
        },
        "edge_refined_bem": {
            "refinement": 4,
            "edge_refinement_passes": 1,
            "unknowns": edge_info["unknowns"],
            "iterations_first_polarization": edge_info[
                "iterations_first_polarization"
            ],
            "iterations_second_polarization": edge_info[
                "iterations_second_polarization"
            ],
            "cold_wall_s": edge_info["wall_s"],
            "warm_wall_s": warm_info["wall_s"],
            "cold_to_warm_speedup": edge_info["wall_s"]
            / warm_info["wall_s"],
            "old_ref5_wall_s": old_ref5_info["wall_s"],
            "old_ref5_to_warm_speedup": old_ref5_info["wall_s"]
            / warm_info["wall_s"],
            "difference_from_old_ref5_full_percent": 100.0
            * edge_vs_ref5["solid_angle_weighted_full_relative_l2"],
            "warm_reproducibility_full_percent": 100.0
            * warm_reproducibility[
                "solid_angle_weighted_full_relative_l2"
            ],
        },
        "old_bem_ref4_to_ref5_full_percent": 100.0
        * old_refinement["solid_angle_weighted_full_relative_l2"],
        "adda_grid_convergence": adda_convergence,
        "bem_vs_adda": rows,
    }

    EDGE.mkdir(parents=True, exist_ok=True)
    (PREFIX.with_suffix(".json")).write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with PREFIX.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(
        2, 2, figsize=(13.5, 9.0), constrained_layout=True
    )
    dpl = np.asarray([row["dpl"] for row in rows])
    axes[0, 0].plot(
        dpl,
        [row["bem_adda_full_percent"] for row in rows],
        "o-",
        lw=2.2,
        label="полная матрица Мюллера",
    )
    axes[0, 0].plot(
        dpl,
        [row["bem_adda_M11_percent"] for row in rows],
        "s-",
        lw=2.2,
        label=r"$M_{11}$",
    )
    axes[0, 0].set_title("Расхождение BEM и ADDA уменьшается")
    axes[0, 0].set_xlabel("Число диполей ADDA на длину волны, dpl")
    axes[0, 0].set_ylabel("Взвешенное относительное расхождение, %")
    axes[0, 0].set_xticks(dpl)
    axes[0, 0].legend()

    convergence_labels = [
        f"{row['coarse_dpl']}→{row['fine_dpl']}"
        for row in adda_convergence
    ]
    x = np.arange(len(convergence_labels))
    width = 0.34
    axes[0, 1].bar(
        x - width / 2,
        [row["full_percent"] for row in adda_convergence],
        width,
        label="полная матрица",
    )
    axes[0, 1].bar(
        x + width / 2,
        [row["M11_percent"] for row in adda_convergence],
        width,
        label=r"$M_{11}$",
    )
    axes[0, 1].axhline(
        100.0 * edge_vs_ref5["solid_angle_weighted_full_relative_l2"],
        color="black",
        ls="--",
        lw=1.8,
        label="новая BEM ↔ прежняя BEM ref5",
    )
    axes[0, 1].set_xticks(x, convergence_labels)
    axes[0, 1].set_title("Собственная сходимость сеток")
    axes[0, 1].set_xlabel("Сгущение сетки ADDA")
    axes[0, 1].set_ylabel("Изменение результата, %")
    axes[0, 1].legend()

    time_labels = [
        "BEM ref5\nпрежняя",
        "BEM edge\nхолодный",
        "BEM edge\nс кешем",
        "ADDA\ndpl=20",
        "ADDA\ndpl=25",
    ]
    time_values = [
        old_ref5_info["wall_s"],
        edge_info["wall_s"],
        warm_info["wall_s"],
        adda_info[20]["process_wall_s"],
        adda_info[25]["process_wall_s"],
    ]
    colors = ["#8c8c8c", "#d97706", "#16a34a", "#2563eb", "#1d4ed8"]
    bars = axes[1, 0].bar(time_labels, time_values, color=colors)
    axes[1, 0].bar_label(bars, fmt="%.1f с", padding=3)
    axes[1, 0].set_title("Полное время одного расчета")
    axes[1, 0].set_ylabel("Время, с")
    axes[1, 0].set_ylim(0, max(time_values) * 1.16)

    scale = edge_bem[0, 0, 0]
    axes[1, 1].plot(
        theta,
        edge_bem[0, 0] / scale,
        color="black",
        lw=2.2,
        label="BEM edge",
    )
    for dpl_value, color in ((20, "#60a5fa"), (25, "#1d4ed8")):
        axes[1, 1].plot(
            theta,
            adda[dpl_value][0, 0] / adda[dpl_value][0, 0, 0],
            color=color,
            lw=1.5,
            label=f"ADDA dpl={dpl_value}",
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title(r"Форма угловой зависимости $M_{11}$")
    axes[1, 1].set_xlabel(r"Угол рассеяния $\theta$, град.")
    axes[1, 1].set_ylabel(r"$M_{11}(\theta)/M_{11}(0)$")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    fig.suptitle(
        r"Шестигранная призма: $h/D_x=1$, $ka=25$, $m=1{,}3$, "
        r"невязка $10^{-5}$",
        fontsize=15,
    )
    fig.savefig(PREFIX.with_suffix(".png"), dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
