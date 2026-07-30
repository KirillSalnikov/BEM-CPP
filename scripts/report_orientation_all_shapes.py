#!/usr/bin/env python3
"""Combine the prism, sphere, and asymmetric-particle sweep reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image


ROOT = Path("runs/orientation_bem_adda_shapes")
KA60_REPORT = Path(
    "runs/hdiv_bem_vs_adda_sweep_n1p3/ka60/recalc_depth5_edge1/"
    "current_pairff_20260730/report"
)
CASES = [
    (
        "Шестигранная призма",
        Path("runs/orientation_bem_adda_crossover"),
        "#c47a12",
        "all_mueller_bem_adda.png",
    ),
    (
        "Сфера",
        ROOT / "sphere",
        "#2676b8",
        "all_mueller_bem_adda_mie.png",
    ),
    (
        "Несимметричный многогранник",
        ROOT / "asymmetric",
        "#16865c",
        "all_mueller_bem_adda.png",
    ),
]
KA_VALUES = [17, 18, 20, 25, 30]


def plot_asymmetric_geometry() -> Path:
    obj = ROOT / "asymmetric_oblique_heptagon.obj"
    vertices = []
    faces = []
    for line in obj.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f":
            faces.append([int(value) - 1 for value in fields[1:4]])
    vertices_array = np.asarray(vertices)
    polygons = [vertices_array[face] for face in faces]
    figure = plt.figure(figsize=(8.0, 7.0))
    axis = figure.add_subplot(111, projection="3d")
    collection = Poly3DCollection(
        polygons,
        facecolor="#27a36f",
        edgecolor="#174b3a",
        linewidth=0.8,
        alpha=0.88,
    )
    axis.add_collection3d(collection)
    axis.scatter(
        vertices_array[:, 0],
        vertices_array[:, 1],
        vertices_array[:, 2],
        color="#8d3f12",
        s=22,
        depthshade=False,
    )
    minima = vertices_array.min(axis=0)
    maxima = vertices_array.max(axis=0)
    center = 0.5 * (minima + maxima)
    radius = 0.55 * float(np.max(maxima - minima))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=24, azim=34)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title(
        "Несимметричный контрольный многогранник\n"
        "14 вершин, 24 треугольные грани, порядок вращательной "
        "симметрии 1",
        fontsize=14,
    )
    figure.tight_layout()
    output = ROOT / "asymmetric_particle_geometry.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    loaded = []
    combined_rows = []
    for label, root, color, image_name in CASES:
        rows = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        loaded.append((label, root, color, image_name, rows))
        for row in rows:
            combined_rows.append({"shape": label, **row})

    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))
    for label, _, color, _, rows in loaded:
        ka = np.asarray([row["ka"] for row in rows])
        bem = np.asarray([row["bem_wall_s"] for row in rows])
        adda = np.asarray([row["adda_wall_s"] for row in rows])
        axes[0, 0].plot(
            ka, bem, "o-", color=color, label=f"BEM: {label}"
        )
        axes[0, 0].plot(
            ka, adda, "s--", color=color, alpha=0.72, label=f"ADDA: {label}"
        )
        axes[0, 1].plot(
            ka,
            adda / bem,
            "o-",
            color=color,
            linewidth=2,
            label=label,
        )
        axes[1, 0].semilogy(
            ka,
            100.0
            * np.asarray(
                [row["bem_vs_adda_normalized_l2"] for row in rows]
            ),
            "o-",
            color=color,
            label=label,
        )
        axes[1, 1].plot(
            ka,
            np.asarray([row["adda_farfield_s"] for row in rows])
            / np.asarray([row["bem_farfield_s"] for row in rows]),
            "o-",
            color=color,
            label=label,
        )

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Полное стеночное время")
    axes[0, 0].set_ylabel("Время, с")
    axes[0, 0].legend(ncol=2, fontsize=9)
    axes[0, 1].axhline(1.0, color="black", linestyle=":", linewidth=1.2)
    axes[0, 1].set_title("Ускорение BEM относительно ADDA")
    axes[0, 1].set_ylabel(r"$T_{\mathrm{ADDA}}/T_{\mathrm{BEM}}$")
    axes[0, 1].legend()
    axes[1, 0].set_title("Расхождение BEM и ADDA")
    axes[1, 0].set_ylabel(
        "Взвешенная относительная ошибка\n"
        "нормированной полной матрицы, %"
    )
    axes[1, 0].legend()
    axes[1, 1].set_title("Ускорение вычисления дальнего поля")
    axes[1, 1].set_ylabel(r"$T_{\mathrm{ADDA,\,far}}/T_{\mathrm{BEM,\,far}}$")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel(r"Размерный параметр $ka$")
        axis.grid(which="both", alpha=0.25)
    figure.suptitle(
        r"$m=1.3$, $N_\alpha=256$, $\beta=90^\circ$, "
        r"невязка $10^{-5}$, BEM ref=5, ADDA dpl=15",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(ROOT / "combined_shapes_comparison.png", dpi=180)
    plt.close(figure)

    fieldnames = list(
        dict.fromkeys(
            key for row in combined_rows for key in row
        )
    )
    with (ROOT / "combined_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)
    (ROOT / "combined_summary.json").write_text(
        json.dumps(combined_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    geometry_page = plot_asymmetric_geometry()
    pages = [ROOT / "combined_shapes_comparison.png", geometry_page]
    for _, root, _, image_name, _ in loaded:
        pages.append(root / "sweep_time_speedup_accuracy.png")
        pages.extend(root / f"ka{ka}" / image_name for ka in KA_VALUES)
    ka60_pages = [
        KA60_REPORT / "ka60_prism_recalculation_summary.png",
        KA60_REPORT / "strict_bem_vs_adda_all_mueller_native_grids.png",
        KA60_REPORT
        / "strict_bem_vs_adda_dimensionless_normalized_stokes_F.png",
        KA60_REPORT / "strict_bem_vs_adda_common_angles.png",
    ]
    pages.extend(path for path in ka60_pages if path.is_file())
    images = [Image.open(path).convert("RGB") for path in pages]
    images[0].save(
        ROOT / "orientation_bem_adda_all_shapes_all_mueller.pdf",
        save_all=True,
        append_images=images[1:],
        resolution=180.0,
    )
    for image in images:
        image.close()
    print(ROOT / "orientation_bem_adda_all_shapes_all_mueller.pdf")


if __name__ == "__main__":
    main()
