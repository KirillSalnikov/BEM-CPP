#!/usr/bin/env python3
"""Report strict ref=6 -> ref=7 sphere convergence against Mie theory."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_mie import mie_mueller


ROOT = Path("runs/sphere_ka60_ref6_ref7_c5_audit")
CASES = {
    6: ROOT / "ref6_hdiv.json",
    7: ROOT / "ref7_hdiv_leaf256.json",
}
PREFIX = ROOT / "sphere_ka60_ref6_ref7_convergence"
NONZERO_ELEMENTS = (
    ("M11", 0, 0),
    ("M12", 0, 1),
    ("M21", 1, 0),
    ("M22", 1, 1),
    ("M33", 2, 2),
    ("M34", 2, 3),
    ("M43", 3, 2),
    ("M44", 3, 3),
)


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def relative_frobenius_by_angle(
    candidate: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    numerator = np.linalg.norm(candidate - reference, axis=(0, 1))
    denominator = np.maximum(np.linalg.norm(reference, axis=(0, 1)), 1.0e-30)
    return numerator / denominator


def load_case(path: Path) -> dict:
    data = json.loads(path.read_text())
    theta = np.asarray(data["physical"]["theta_degrees"], dtype=float)
    mueller = np.asarray(data["physical"]["mueller"], dtype=float)
    if mueller.shape != (4, 4, theta.size):
        raise ValueError(f"unexpected Mueller shape in {path}: {mueller.shape}")
    return {"raw": data, "theta": theta, "mueller": mueller}


def load_solver_history(path: Path) -> dict[str, float | int | np.ndarray]:
    iterations: list[int] = []
    residuals: list[float] = []
    segment_elapsed: list[float] = []
    current_segment_elapsed = 0.0
    first_iteration: int | None = None
    last_iteration: int | None = None
    inner_iterations = 0
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row["solver"] == "pFFT-inner" and row["event"] == "iteration":
                inner_iterations += 1
                continue
            if row["solver"] != "FMM-pFFT-FGMRES":
                continue
            iteration = int(row["iteration"])
            elapsed = float(row["elapsed_s"])
            if row["event"] == "initial":
                if first_iteration is None:
                    first_iteration = iteration
                if current_segment_elapsed > 0.0:
                    segment_elapsed.append(current_segment_elapsed)
                current_segment_elapsed = elapsed
            else:
                current_segment_elapsed = max(current_segment_elapsed, elapsed)
            last_iteration = iteration
            if row["event"] == "iteration":
                iterations.append(int(row["iteration"]))
                residuals.append(float(row["projected_residual"]))
    if current_segment_elapsed > 0.0:
        segment_elapsed.append(current_segment_elapsed)
    if first_iteration is None or last_iteration is None:
        raise ValueError(f"no FMM-pFFT-FGMRES history in {path}")
    return {
        "iterations": np.asarray(iterations),
        "residuals": np.asarray(residuals),
        "first_checkpoint_iteration": first_iteration,
        "last_checkpoint_iteration": last_iteration,
        "correct_operator_iterations": last_iteration - first_iteration,
        "inner_iterations": inner_iterations,
        "accumulated_solve_s": float(sum(segment_elapsed)),
        "solve_segments": len(segment_elapsed),
    }


def case_metrics(
    case: dict, mie: np.ndarray, history: dict[str, float | int | np.ndarray]
) -> dict[str, float | int]:
    data = case["raw"]
    matrix = case["mueller"]
    m11 = matrix[0, 0]
    mie_m11 = mie[0, 0]
    normalized = matrix / m11[0]
    mie_normalized = mie / mie_m11[0]
    weights = np.sin(np.deg2rad(case["theta"]))[None, None, :]
    weighted_error = np.sqrt(
        np.sum(weights * (normalized - mie_normalized) ** 2)
        / np.sum(weights * mie_normalized**2)
    )
    scale = float(np.dot(m11, mie_m11) / np.dot(mie_m11, mie_m11))
    pfft = data["pfft_fgmres"]
    return {
        "refinement": int(data["refinements"]),
        "system_dofs": int(data["system_dofs"]),
        "quadrature_points": int(data["quadrature_points"]),
        "max_element_edge": float(data["max_element_edge"]),
        "ka_h_element": float(data["ka_h_element"]),
        "outer_iterations": int(history["correct_operator_iterations"]),
        "checkpoint_iteration": int(history["last_checkpoint_iteration"]),
        "solve_segments": int(history["solve_segments"]),
        "inner_iterations": int(history["inner_iterations"]),
        "operator_residual": float(pfft["fmm_residual"]),
        "m11_relative_l2_vs_mie": relative_l2(m11, mie_m11),
        "m11_scale_vs_mie": scale,
        "m11_shape_relative_l2_vs_mie": relative_l2(m11 / scale, mie_m11),
        "full_mueller_relative_l2_vs_mie": relative_l2(matrix, mie),
        "full_mueller_solid_angle_weighted_relative_l2_vs_mie": float(
            weighted_error
        ),
        "sphere_cross_polarization_relative": float(
            data["physical"]["sphere_cross_polarization_relative"]
        ),
        "fmm_setup_s": float(data["fmm_setup_s"]),
        "mbj_setup_s": float(data["mbj_local_setup_s"]),
        "solve_s": float(history["accumulated_solve_s"]),
        "farfield_s": float(data["physical"]["farfield_s"]),
    }


def main() -> None:
    cases = {ref: load_case(path) for ref, path in CASES.items()}
    histories = {
        ref: load_solver_history(path.with_suffix(".iterations.csv"))
        for ref, path in CASES.items()
    }
    theta = cases[6]["theta"]
    if not np.array_equal(theta, cases[7]["theta"]):
        raise ValueError("ref=6 and ref=7 theta grids differ")
    raw6 = cases[6]["raw"]
    raw7 = cases[7]["raw"]
    if raw6["ka"] != raw7["ka"] or raw6["ri"] != raw7["ri"]:
        raise ValueError("ref=6 and ref=7 physical parameters differ")

    refractive_index = complex(raw6["ri"])
    mie = np.asarray(mie_mueller(theta, refractive_index, raw6["ka"]))
    metrics = {
        ref: case_metrics(case, mie, histories[ref])
        for ref, case in cases.items()
    }
    matrix6 = cases[6]["mueller"]
    matrix7 = cases[7]["mueller"]
    self_metrics = {
        "full_mueller_relative_l2_ref7_vs_ref6": relative_l2(matrix7, matrix6),
        "m11_relative_l2_ref7_vs_ref6": relative_l2(
            matrix7[0, 0], matrix6[0, 0]
        ),
        "m11_normalized_shape_relative_l2_ref7_vs_ref6": relative_l2(
            matrix7[0, 0] / matrix7[0, 0, 0],
            matrix6[0, 0] / matrix6[0, 0, 0],
        ),
    }
    element_metrics = {}
    for name, row, column in NONZERO_ELEMENTS:
        mie_element = mie[row, column]
        ref6_element = matrix6[row, column]
        ref7_element = matrix7[row, column]
        forward_m11 = mie[0, 0, 0]
        sample_count = mie_element.size
        element_metrics[name] = {
            "mie_element_l2_relative_to_m11_l2": float(
                np.linalg.norm(mie_element) / np.linalg.norm(mie[0, 0])
            ),
            "ref6_relative_l2_vs_mie": relative_l2(
                ref6_element, mie_element
            ),
            "ref7_relative_l2_vs_mie": relative_l2(
                ref7_element, mie_element
            ),
            "ref7_relative_l2_vs_ref6": relative_l2(
                ref7_element, ref6_element
            ),
            "ref7_max_absolute_difference_normalized_by_forward_m11": float(
                np.max(np.abs(ref7_element - ref6_element)) / forward_m11
            ),
            "ref6_rms_error_normalized_by_forward_m11": float(
                np.linalg.norm(ref6_element - mie_element)
                / np.sqrt(sample_count)
                / forward_m11
            ),
            "ref7_rms_error_normalized_by_forward_m11": float(
                np.linalg.norm(ref7_element - mie_element)
                / np.sqrt(sample_count)
                / forward_m11
            ),
            "ref7_max_error_vs_mie_normalized_by_forward_m11": float(
                np.max(np.abs(ref7_element - mie_element)) / forward_m11
            ),
        }

    summary = {
        "case": {
            "shape": "sphere",
            "ka": raw6["ka"],
            "refractive_index": raw6["ri"],
            "basis": "H(div)-conforming BDM1",
            "polarization_mode": raw6["physical"]["polarization_mode"],
            "operator_tolerance": 1.0e-5,
        },
        "refinements": metrics,
        "self_convergence": self_metrics,
        "nonzero_element_metrics": element_metrics,
    }
    PREFIX.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")

    with PREFIX.with_suffix(".csv").open("w", newline="") as stream:
        fieldnames = list(next(iter(metrics.values())).keys())
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics.values())
    with PREFIX.with_name(PREFIX.name + "_elements.csv").open(
        "w", newline=""
    ) as stream:
        fieldnames = ["element", *next(iter(element_metrics.values())).keys()]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for element, values in element_metrics.items():
            writer.writerow({"element": element, **values})

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.semilogy(theta, mie[0, 0] / mie[0, 0, 0], "k--", label="теория Ми")
    for ref, color in ((6, "#268bd2"), (7, "#2ca25f")):
        m11 = cases[ref]["mueller"][0, 0]
        ax.semilogy(theta, m11 / m11[0], color=color, label=f"BEM ref={ref}")
    ax.set(
        xlabel=r"Угол рассеяния $\theta$, град.",
        ylabel=r"$M_{11}(\theta)/M_{11}(0)$",
        title="Индикатриса рассеяния",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for ref, color in ((6, "#268bd2"), (7, "#2ca25f")):
        error = relative_frobenius_by_angle(cases[ref]["mueller"], mie)
        ax.semilogy(theta, error, color=color, label=f"ref={ref}")
    ax.set(
        xlabel=r"Угол рассеяния $\theta$, град.",
        ylabel="Относительное отличие",
        title="Матрица Мюллера относительно теории Ми",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    for ref, color in ((6, "#268bd2"), (7, "#2ca25f")):
        iterations = histories[ref]["iterations"]
        residuals = histories[ref]["residuals"]
        iterations = iterations - int(
            histories[ref]["first_checkpoint_iteration"]
        )
        ax.semilogy(
            iterations, residuals, marker=".", color=color, label=f"ref={ref}"
        )
    ax.axhline(1.0e-5, color="black", linestyle="--", label=r"цель $10^{-5}$")
    ax.set(
        xlabel="Внешняя итерация FGMRES",
        ylabel="Относительная невязка",
        title="Сходимость решения линейной системы",
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1, 1]
    labels = ["вся\nматрица", *[item[0] for item in NONZERO_ELEMENTS]]
    x = np.arange(len(labels))
    width = 0.38
    ref6_errors = [100.0 * metrics[6]["full_mueller_relative_l2_vs_mie"]]
    ref7_errors = [100.0 * metrics[7]["full_mueller_relative_l2_vs_mie"]]
    ref6_errors.extend(
        100.0 * element_metrics[name]["ref6_relative_l2_vs_mie"]
        for name, _, _ in NONZERO_ELEMENTS
    )
    ref7_errors.extend(
        100.0 * element_metrics[name]["ref7_relative_l2_vs_mie"]
        for name, _, _ in NONZERO_ELEMENTS
    )
    ax.bar(x - width / 2.0, ref6_errors, width, label="ref=6")
    ax.bar(x + width / 2.0, ref7_errors, width, label="ref=7")
    ax.set_xticks(x, labels)
    ax.set(
        ylabel="Относительное отличие от теории Ми, %",
        title="Сходимость всех ненулевых элементов",
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    ax.set_yscale("log")

    fig.suptitle(
        r"Сфера: $ka=60$, $m=1{,}3$, H(div)-BEM, невязка $10^{-5}$",
        fontsize=16,
    )
    fig.savefig(PREFIX.with_suffix(".png"), dpi=180)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    forward_m11 = mie[0, 0, 0]
    for ax, (name, row, column) in zip(axes.flat, NONZERO_ELEMENTS):
        ax.plot(
            theta,
            mie[row, column] / forward_m11,
            "k--",
            linewidth=1.5,
            label="теория Ми",
        )
        ax.plot(
            theta,
            matrix6[row, column] / forward_m11,
            color="#268bd2",
            linewidth=1.0,
            label="ref=6",
        )
        ax.plot(
            theta,
            matrix7[row, column] / forward_m11,
            color="#2ca25f",
            linewidth=1.0,
            label="ref=7",
        )
        ax.set_yscale("symlog", linthresh=1.0e-5)
        ax.set_title(rf"$M_{{{name[1:]}}}$")
        ax.set_xlabel(r"Угол $\theta$, град.")
        ax.set_ylabel(rf"$M_{{{name[1:]}}}(\theta)/M_{{11}}^{{Mie}}(0)$")
        ax.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        r"Все ненулевые элементы матрицы Мюллера: "
        r"сфера, $ka=60$, $m=1{,}3$",
        fontsize=16,
    )
    fig.savefig(
        PREFIX.with_name(PREFIX.name + "_nonzero_elements.png"), dpi=180
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
