#!/usr/bin/env python3
"""Compare nodal Muller-BEM physical output with matching ADDA runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bem", type=Path, required=True)
    parser.add_argument("--bem-log", type=Path)
    parser.add_argument(
        "--bem-coarse",
        action="append",
        default=[],
        metavar="LABEL=FILE",
        help="Coarser BEM result used for a self-convergence check.",
    )
    parser.add_argument(
        "--adda",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="Label and ADDA result directory; may be repeated.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected LABEL=DIR, got {value!r}")
    label, path = value.split("=", 1)
    return label, Path(path)


def parse_number(text: str, pattern: str, cast=float):
    match = re.search(pattern, text)
    return cast(match.group(1)) if match else None


def load_bem(path: Path, log_path: Path | None) -> tuple[np.ndarray, np.ndarray, dict]:
    data = json.loads(path.read_text())
    physical = data["physical"]
    theta = np.asarray(physical["theta_degrees"], dtype=float)
    mueller = np.asarray(physical["mueller"], dtype=float)
    if mueller.shape != (4, 4, len(theta)):
        raise ValueError(f"unexpected BEM Mueller shape {mueller.shape}")

    wall_s = None
    if log_path and log_path.exists():
        text = log_path.read_text(errors="replace")
        match = re.search(
            r"Elapsed \(wall clock\) time .*?:\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)",
            text,
        )
        if match:
            hours = int(match.group(1) or 0)
            wall_s = 3600 * hours + 60 * int(match.group(2)) + float(match.group(3))
        else:
            match = re.search(r"^ACTUAL_WALL_S=(\d+(?:\.\d+)?)$", text, re.MULTILINE)
            if match:
                wall_s = float(match.group(1))

    info = {
        "method": (
            "surface Muller BEM, H(div)-BDM1, FMM, MBJ"
            if data.get("hdiv_conforming")
            else "surface Muller BEM, nodal P2, FMM, MBJ"
        ),
        "ka": data["ka"],
        "refractive_index": data["ri"],
        "shape": data["shape"],
        "refinements": data["refinements"],
        "azimuth_degrees": data["prism_azimuth_degrees"],
        "unknowns": data["system_dofs"],
        "nodes_per_wavelength_min": data["p2_nodes_per_wavelength_min"],
        "tolerance": data["tolerance"],
        "iterations_first_polarization": data["mbj"]["iterations"],
        "iterations_second_polarization": physical["parallel_iterations"],
        "solve_first_s": data["mbj"]["solve_s"],
        "solve_second_s": physical["parallel_s"],
        "fmm_setup_s": data["fmm_setup_s"],
        "preconditioner_setup_s": data["mbj_local_setup_s"],
        "farfield_s": physical["farfield_s"],
        "wall_s": wall_s,
        "second_polarization": physical["polarization_mode"],
        "edge_mode": data.get("edge_mode"),
        "hdiv_conforming": bool(data.get("hdiv_conforming", False)),
    }
    return theta, mueller, info


def load_adda(directory: Path, theta_target: np.ndarray) -> tuple[np.ndarray, dict]:
    table = np.loadtxt(directory / "mueller", skiprows=1)
    theta = table[:, 0]
    keep = (theta >= theta_target[0] - 1e-12) & (theta <= theta_target[-1] + 1e-12)
    table = table[keep]
    theta = table[:, 0]
    if len(theta) != len(theta_target) or not np.allclose(theta, theta_target):
        raise ValueError(f"ADDA angles in {directory} do not match BEM angles")
    mueller = table[:, 1:].reshape(-1, 4, 4).transpose(1, 2, 0)

    log_text = (directory / "log").read_text(errors="replace")
    solver_match = re.search(r"Iterative Method:\s*(.+)", log_text)
    solver = solver_match.group(1).strip() if solver_match else "unknown"
    info = {
        "method": f"volume DDA, OpenCL, {solver}",
        "occupied_dipoles": parse_number(
            log_text, r"Total number of occupied dipoles:\s*(\d+)", int
        ),
        "dipoles_per_wavelength": parse_number(
            log_text, r"Dipoles/lambda:\s*([0-9.eE+-]+)"
        ),
        "iterations_total": parse_number(
            log_text, r"Total number of iterations:\s*(\d+)", int
        ),
        "matvec_total": parse_number(
            log_text, r"Total number of matrix-vector products:\s*(\d+)", int
        ),
        "wall_s": parse_number(log_text, r"Total wall time:\s*([0-9.eE+-]+)"),
        "internal_fields_s": parse_number(
            log_text, r"Internal fields:\s*([0-9.eE+-]+)"
        ),
        "scattered_fields_s": parse_number(
            log_text, r"Scattered fields:\s*([0-9.eE+-]+)"
        ),
    }
    time_path = directory / "time.txt"
    if time_path.exists():
        info["process_wall_s"] = parse_number(
            time_path.read_text(errors="replace"),
            r"ACTUAL_WALL_S=([0-9.eE+-]+)",
        )
    else:
        info["process_wall_s"] = None
    return mueller, info


def comparison_metrics(
    theta_degrees: np.ndarray, bem: np.ndarray, adda: np.ndarray
) -> dict:
    bem_scale = bem[0, 0, 0]
    adda_scale = adda[0, 0, 0]
    bem_normalized = bem / bem_scale
    adda_normalized = adda / adda_scale
    bem_local = bem / bem[0, 0][None, None, :]
    adda_local = adda / adda[0, 0][None, None, :]
    solid_angle_weights = np.sin(np.deg2rad(theta_degrees))[None, None, :]
    fitted_scale = float(np.vdot(bem, adda).real / np.vdot(bem, bem).real)
    return {
        "forward_M11_bem": float(bem_scale),
        "forward_M11_adda": float(adda_scale),
        "forward_M11_ratio_adda_over_bem": float(adda_scale / bem_scale),
        "raw_full_relative_l2": float(np.linalg.norm(adda - bem) / np.linalg.norm(bem)),
        "best_scalar_adda_over_bem": fitted_scale,
        "after_best_scalar_relative_l2": float(
            np.linalg.norm(adda - fitted_scale * bem) / np.linalg.norm(adda)
        ),
        "forward_normalized_full_relative_l2": float(
            np.linalg.norm(adda_normalized - bem_normalized)
            / np.linalg.norm(bem_normalized)
        ),
        "forward_normalized_M11_relative_l2": float(
            np.linalg.norm(adda_normalized[0, 0] - bem_normalized[0, 0])
            / np.linalg.norm(bem_normalized[0, 0])
        ),
        "solid_angle_weighted_full_relative_l2": float(
            np.sqrt(
                np.sum(
                    solid_angle_weights
                    * (adda_normalized - bem_normalized) ** 2
                )
                / np.sum(solid_angle_weights * bem_normalized**2)
            )
        ),
        "solid_angle_weighted_M11_relative_l2": float(
            np.sqrt(
                np.sum(
                    solid_angle_weights[0, 0]
                    * (adda_normalized[0, 0] - bem_normalized[0, 0]) ** 2
                )
                / np.sum(
                    solid_angle_weights[0, 0] * bem_normalized[0, 0] ** 2
                )
            )
        ),
        "locally_M11_normalized_mueller_rms_absolute": float(
            np.sqrt(np.mean((adda_local - bem_local) ** 2))
        ),
        "locally_M11_normalized_mueller_relative_l2": float(
            np.linalg.norm(adda_local - bem_local) / np.linalg.norm(bem_local)
        ),
    }


def plot_comparison(
    theta: np.ndarray,
    bem: np.ndarray,
    adda_runs: list[tuple[str, np.ndarray]],
    output: Path,
    particle: dict,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.8), constrained_layout=True)
    specs = [
        (0, 0, r"$M_{11}$", True),
        (0, 1, r"$M_{12}/M_{11}$", False),
        (0, 2, r"$M_{13}/M_{11}$", False),
        (1, 1, r"$M_{22}/M_{11}$", False),
        (2, 2, r"$M_{33}/M_{11}$", False),
        (2, 3, r"$M_{34}/M_{11}$", False),
    ]
    colors = plt.cm.viridis(
        np.linspace(0.12, 0.88, max(len(adda_runs), 1))
    )

    for ax, (i, j, title, use_log) in zip(axes.flat, specs):
        y_bem = bem[i, j] if use_log else bem[i, j] / bem[0, 0]
        ax.plot(
            theta, y_bem, color="black", lw=2.2,
            label="BEM: Мюллер, H(div)-BDM1 + MBJ",
        )
        for color, (label, mueller) in zip(colors, adda_runs):
            y_adda = mueller[i, j] if use_log else mueller[i, j] / mueller[0, 0]
            ax.plot(theta, y_adda, color=color, lw=1.5, label=f"ADDA: {label}")
        if use_log:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlim(0, 180)
        ax.set_xlabel(r"Угол рассеяния $\theta$, град.")
        ax.grid(True, which="both", alpha=0.25)

    axes[0, 0].set_ylabel("Значение")
    axes[1, 0].set_ylabel("Нормированное значение")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "Одинаковая частица: шестигранная призма, "
        f"h/D={particle['aspect_h_over_D']:g}, "
        f"ka={particle['ka']:g}, "
        f"m={particle['refractive_index']:g}, "
        f"невязка={particle['relative_residual_tolerance']:.0e}",
        fontsize=15,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_study(summary: dict, output: Path) -> None:
    convergence = summary.get("adda_grid_convergence", [])
    adda = summary["adda"]
    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 4.8), constrained_layout=True
    )

    if convergence:
        labels = [f"{row['from']}→{row['to']}" for row in convergence]
        x = np.arange(len(labels))
        axes[0].plot(
            x,
            [100 * row["solid_angle_weighted_full_relative_l2"]
             for row in convergence],
            "o-", label="полная матрица",
        )
        axes[0].plot(
            x,
            [100 * row["solid_angle_weighted_M11_relative_l2"]
             for row in convergence],
            "s-", label=r"$M_{11}$",
        )
        axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Изменение, %")
    axes[0].set_title("Сходимость сетки ADDA")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    labels = [row["label"] for row in adda]
    x = np.arange(len(labels))
    axes[1].plot(
        x,
        [100 * row["comparison"]["solid_angle_weighted_full_relative_l2"]
         for row in adda],
        "o-", label="полная матрица",
    )
    axes[1].plot(
        x,
        [100 * row["comparison"]["solid_angle_weighted_M11_relative_l2"]
         for row in adda],
        "s-", label=r"$M_{11}$",
    )
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Отличие от BEM, %")
    axes[1].set_title("H(div)-BEM против ADDA")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    adda_wall = [
        row["run"].get("process_wall_s") or row["run"]["wall_s"]
        for row in adda
    ]
    axes[2].plot(x, adda_wall, "o-", label="ADDA-OCL")
    if summary["bem"]["wall_s"] is not None:
        axes[2].axhline(
            summary["bem"]["wall_s"], color="black", lw=2,
            label="H(div)-BEM",
        )
    axes[2].set_xticks(x, labels, rotation=35, ha="right")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Полное wall time, с")
    axes[2].set_title("Время двух поляризаций")
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend()

    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: dict) -> None:
    bem = summary["bem"]
    lines = [
        "# Сравнение BEM и ADDA на одинаковой частице",
        "",
        (
            f"Шестигранная призма h/D=1, ka={bem['ka']}, m={bem['refractive_index']}, "
            f"азимут {bem['azimuth_degrees']}°, относительная невязка {bem['tolerance']:.0e}."
        ),
        "",
        "## Время и дискретизация",
        "",
        (
            f"- BEM: {bem['method']}; {bem['unknowns']} неизвестных, "
            f"{bem['nodes_per_wavelength_min']:.2f} узла/длину волны, "
            f"{bem['iterations_first_polarization']} итераций первой "
            f"поляризации; полное wall time "
            + (
                f"{bem['wall_s']:.2f} с."
                if bem["wall_s"] is not None
                else "не записано."
            )
        ),
    ]
    bem_refinement = summary.get("bem_grid_refinement")
    if bem_refinement:
        lines.extend(
            [
                (
                    f"- Сходимость BEM {bem_refinement['from']} -> "
                    f"{bem_refinement['to']}: изменение полной матрицы "
                    f"{100 * bem_refinement['solid_angle_weighted_full_relative_l2']:.3f}%, "
                    f"M11 {100 * bem_refinement['solid_angle_weighted_M11_relative_l2']:.3f}%."
                ),
            ]
        )
    for run in summary["adda"]:
        info = run["run"]
        metrics = run["comparison"]
        lines.extend(
            [
                (
                    f"- ADDA {run['label']}: {info['occupied_dipoles']} диполей, "
                    f"{info['dipoles_per_wavelength']:.3f} диполя/длину волны, "
                    f"{info['iterations_total']} итераций на две поляризации; "
                    f"wall time процесса "
                    f"{(info['process_wall_s'] or info['wall_s']):.2f} с; "
                    f"ADDA быстрее BEM по полному "
                    f"времени в {metrics['wall_speedup_adda_vs_bem']:.2f} раза."
                ),
                (
                    f"- ADDA {run['label']} против BEM: взвешенная по телесному углу "
                    f"ошибка полной матрицы {100 * metrics['solid_angle_weighted_full_relative_l2']:.3f}%, "
                    f"ошибка M11 {100 * metrics['solid_angle_weighted_M11_relative_l2']:.3f}%."
                ),
                (
                    f"- Вперёд: M11(BEM)={metrics['forward_M11_bem']:.6g}, "
                    f"M11(ADDA)={metrics['forward_M11_adda']:.6g}; отношение "
                    f"ADDA/BEM={metrics['forward_M11_ratio_adda_over_bem']:.4f}."
                ),
                (
                    f"- Более строгая локальная проверка Mij(theta)/M11(theta): "
                    f"среднеквадратичное абсолютное отличие "
                    f"{metrics['locally_M11_normalized_mueller_rms_absolute']:.3f}. "
                    "Эта мера особенно чувствительна в глубоких минимумах M11."
                ),
            ]
        )
    refinement = summary.get("adda_grid_refinement")
    if refinement:
        lines.extend(
            [
                "",
                "## Сходимость сетки ADDA",
                "",
                (
                    f"Переход {refinement['from']} -> {refinement['to']} меняет "
                    f"нормированную матрицу на "
                    f"{100 * refinement['solid_angle_weighted_full_relative_l2']:.3f}% "
                    f"в норме с весом по телесному углу; M11 меняется на "
                    f"{100 * refinement['solid_angle_weighted_M11_relative_l2']:.3f}%."
                ),
            ]
        )
    lines.append("")
    lines.append(
        "Время разных методов нельзя трактовать как чистое сравнение предобуславливателей: "
        "BEM использует CPU+GPU FMM и поверхностную сетку, ADDA использует объёмную FFT-сетку "
        "на GPU. Сравнение физики при этом выполняется для одной геометрии и одних параметров."
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    theta, bem_mueller, bem_info = load_bem(args.bem, args.bem_log)
    bem_coarse_summary = []
    for item in args.bem_coarse:
        label, path = parse_labeled_path(item)
        coarse_theta, coarse_mueller, coarse_info = load_bem(path, None)
        if (
            len(coarse_theta) != len(theta)
            or not np.allclose(coarse_theta, theta)
        ):
            raise ValueError(f"BEM angles in {path} do not match fine BEM")
        metrics = comparison_metrics(theta, bem_mueller, coarse_mueller)
        bem_coarse_summary.append(
            {
                "label": label,
                "path": str(path),
                "refinements": coarse_info["refinements"],
                "solid_angle_weighted_full_relative_l2":
                    metrics["solid_angle_weighted_full_relative_l2"],
                "solid_angle_weighted_M11_relative_l2":
                    metrics["solid_angle_weighted_M11_relative_l2"],
            }
        )

    adda_arrays = []
    adda_summary = []
    for item in args.adda:
        label, path = parse_labeled_path(item)
        mueller, info = load_adda(path, theta)
        adda_arrays.append((label, mueller))
        metrics = comparison_metrics(theta, bem_mueller, mueller)
        adda_process_wall = info.get("process_wall_s") or info["wall_s"]
        if bem_info["wall_s"] is not None and adda_process_wall is not None:
            metrics["wall_speedup_adda_vs_bem"] = (
                bem_info["wall_s"] / adda_process_wall
            )
        else:
            metrics["wall_speedup_adda_vs_bem"] = None
        adda_summary.append(
            {
                "label": label,
                "directory": str(path),
                "run": info,
                "comparison": metrics,
            }
        )

    summary = {
        "particle": {
            "shape": "regular hexagonal prism",
            "aspect_h_over_D": 1.0,
            "ka": bem_info["ka"],
            "refractive_index": bem_info["refractive_index"],
            "azimuth_degrees": bem_info["azimuth_degrees"],
            "relative_residual_tolerance": bem_info["tolerance"],
            "theta_degrees": theta.tolist(),
        },
        "bem": bem_info,
        "adda": adda_summary,
    }
    if bem_coarse_summary:
        coarse = bem_coarse_summary[-1]
        summary["bem_grid_refinement"] = {
            "from": coarse["label"],
            "to": f"ref={bem_info['refinements']}",
            "solid_angle_weighted_full_relative_l2":
                coarse["solid_angle_weighted_full_relative_l2"],
            "solid_angle_weighted_M11_relative_l2":
                coarse["solid_angle_weighted_M11_relative_l2"],
        }
    if len(adda_arrays) >= 2:
        convergence_history = []
        for (
            (coarse_label, coarse_mueller),
            (fine_label, fine_mueller),
        ) in zip(adda_arrays[:-1], adda_arrays[1:]):
            metrics = comparison_metrics(
                theta, fine_mueller, coarse_mueller
            )
            convergence_history.append(
                {
                    "from": coarse_label,
                    "to": fine_label,
                    "solid_angle_weighted_full_relative_l2":
                        metrics["solid_angle_weighted_full_relative_l2"],
                    "solid_angle_weighted_M11_relative_l2":
                        metrics["solid_angle_weighted_M11_relative_l2"],
                }
            )
        summary["adda_grid_convergence"] = convergence_history
        first_label, first_mueller = adda_arrays[-2]
        last_label, last_mueller = adda_arrays[-1]
        refinement = comparison_metrics(
            theta, last_mueller, first_mueller
        )
        summary["adda_grid_refinement"] = {
            "from": first_label,
            "to": last_label,
            "solid_angle_weighted_full_relative_l2":
                refinement["solid_angle_weighted_full_relative_l2"],
            "solid_angle_weighted_M11_relative_l2":
                refinement["solid_angle_weighted_M11_relative_l2"],
        }
    (args.out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_report(args.out_dir / "comparison_report.md", summary)
    plot_comparison(
        theta, bem_mueller, adda_arrays,
        args.out_dir / "bem_vs_adda_mueller.png",
        summary["particle"],
    )
    plot_comparison(
        theta, bem_mueller, [adda_arrays[-1]],
        args.out_dir / "bem_vs_adda_selected.png",
        summary["particle"],
    )
    plot_study(
        summary, args.out_dir / "convergence_and_timing.png"
    )


if __name__ == "__main__":
    main()
