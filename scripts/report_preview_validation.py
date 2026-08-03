#!/usr/bin/env python3
"""Compare a fast Muller preview with a converged BEM reference."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--reference-time", type=Path, required=True)
    parser.add_argument("--preview-time", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    physical = data["physical"]
    theta = np.asarray(physical["theta_degrees"], dtype=float)
    mueller = np.asarray(physical["mueller"], dtype=float)
    if mueller.shape != (4, 4, theta.size):
        raise ValueError(f"{path}: expected 4x4xN Mueller data")
    return data, theta, mueller


def interpolate(
    source_theta: np.ndarray,
    source: np.ndarray,
    target_theta: np.ndarray,
) -> np.ndarray:
    if source_theta.shape == target_theta.shape and np.allclose(
        source_theta, target_theta
    ):
        return source
    return np.asarray(
        [
            [np.interp(target_theta, source_theta, source[i, j]) for j in range(4)]
            for i in range(4)
        ]
    )


def elapsed_seconds(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    direct = re.search(r"ACTUAL_WALL_S=([0-9.eE+-]+)", text)
    if direct:
        return float(direct.group(1))
    match = re.search(r"Elapsed \(wall clock\) time.*: ([0-9:.]+)", text)
    if not match:
        raise ValueError(f"{path}: GNU time elapsed value is absent")
    fields = [float(value) for value in match.group(1).split(":")]
    if len(fields) == 2:
        return 60.0 * fields[0] + fields[1]
    if len(fields) == 3:
        return 3600.0 * fields[0] + 60.0 * fields[1] + fields[2]
    raise ValueError(f"{path}: unsupported elapsed value")


def main() -> None:
    args = arguments()
    reference_data, reference_theta, reference = load_result(args.reference)
    preview_data, theta, preview = load_result(args.preview)
    reference = interpolate(reference_theta, reference, theta)

    weights = np.sin(np.deg2rad(theta))
    difference = preview - reference
    weighted_m11_norm = np.sqrt(np.sum(reference[0, 0] ** 2 * weights))
    component_error = np.sqrt(
        np.sum(difference**2 * weights[None, None, :], axis=2)
    ) / weighted_m11_norm
    full_error = np.sqrt(
        np.sum(difference**2 * weights[None, None, :])
        / np.sum(reference**2 * weights[None, None, :])
    )
    m11_error = np.sqrt(
        np.sum(difference[0, 0] ** 2 * weights)
        / np.sum(reference[0, 0] ** 2 * weights)
    )
    forward = reference[0, 0, 0]
    forward_error = abs(preview[0, 0, 0] / forward - 1.0)
    maximum_error = np.max(np.abs(difference)) / abs(forward)
    reference_wall = elapsed_seconds(args.reference_time)
    preview_wall = elapsed_seconds(args.preview_time)

    metrics = {
        "case": {
            "shape": preview_data["shape"],
            "ka": preview_data["ka"],
            "refractive_index": preview_data["ri"],
            "refinement": preview_data["refinements"],
            "system_dofs": preview_data["system_dofs"],
            "theta_samples": int(theta.size),
        },
        "reference": {
            "path": str(args.reference.resolve()),
            "wall_s": reference_wall,
            "outer_iterations": reference_data["pfft_fgmres"]["outer_iterations"],
            "verified_fmm_residual": reference_data["pfft_fgmres"]["fmm_residual"],
        },
        "preview": {
            "path": str(args.preview.resolve()),
            "wall_s": preview_wall,
            "outer_iterations": preview_data["pfft_fgmres"]["outer_iterations"],
            "inner_iterations": preview_data["pfft_fgmres"]["inner_iterations"],
            "projected_residual": preview_data["pfft_fgmres"]["projected_residual"],
            "residual_verified_in_run": preview_data["pfft_fgmres"][
                "fmm_residual_verified"
            ],
        },
        "wall_speedup": reference_wall / preview_wall,
        "solid_angle_weighted_full_mueller_relative_l2": float(full_error),
        "solid_angle_weighted_m11_relative_l2": float(m11_error),
        "forward_m11_relative_error": float(forward_error),
        "maximum_absolute_error_over_reference_forward_m11": float(maximum_error),
        "component_error_over_weighted_reference_m11_norm": component_error.tolist(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    scale = abs(forward)
    figure, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    for row in range(4):
        for column in range(4):
            axis = axes[row, column]
            reference_curve = reference[row, column] / scale
            preview_curve = preview[row, column] / scale
            if row == 0 and column == 0:
                axis.semilogy(theta, np.maximum(np.abs(reference_curve), 1.0e-12),
                              color="black", linewidth=1.5, label="converged")
                axis.semilogy(theta, np.maximum(np.abs(preview_curve), 1.0e-12),
                              color="#e66101", linestyle="--", linewidth=1.2,
                              label="preview")
            else:
                axis.plot(theta, reference_curve, color="black", linewidth=1.3)
                axis.plot(theta, preview_curve, color="#e66101", linestyle="--",
                          linewidth=1.1)
                axis.axhline(0.0, color="0.8", linewidth=0.6)
            axis.set_title(
                rf"$M_{{{row + 1}{column + 1}}}$; error "
                f"{100.0 * component_error[row, column]:.3f}%"
            )
            axis.grid(alpha=0.2)
            if row == 3:
                axis.set_xlabel(r"Scattering angle $\theta$, deg")
            if column == 0:
                axis.set_ylabel(r"$M_{ij}/M_{11}(0)$")
    axes[0, 0].legend(loc="best")
    figure.suptitle(
        "ka=80, m=1.3, ref=6: converged BEM vs three-step preview\n"
        f"wall speedup {reference_wall / preview_wall:.2f}x; "
        f"weighted full-Mueller error {100.0 * full_error:.3f}%",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(args.output.with_suffix(".png"), dpi=180)
    figure.savefig(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
