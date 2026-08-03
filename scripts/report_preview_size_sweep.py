#!/usr/bin/env python3
"""Report the validated fixed-orientation prism preview size sweep."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = {
    20: (
        "runs/prism_quick_standard_vs_existing_adda_ka10_60_ntheta181/"
        "ka20/bem_standard_strict_outer/result.json",
        "runs/preview_size_sweep_20260803/"
        "ka20_ref6_inner012_outer4_c6verified",
    ),
    30: (
        "runs/prism_quick_standard_vs_existing_adda_ka10_60_ntheta181/"
        "ka30/bem_standard_strict_outer/result.json",
        "runs/preview_size_sweep_20260803/"
        "ka30_ref6_inner012_outer4_c6verified",
    ),
    60: (
        "runs/prism_quick_standard_vs_existing_adda_ka10_60_ntheta181/"
        "ka60/bem_standard_ref6_memory_cap/result.json",
        "runs/preview_size_sweep_20260803/"
        "ka60_ref6_inner012_outer4_c6verified",
    ),
    80: (
        "runs/ref6_vs_adda_fp32_ka_gt60_20260802/"
        "bem_ka80_ref6_pfft/result.json",
        "runs/preview_size_sweep_20260803/ka80_preview_c6verified",
    ),
}

REJECTED_111 = (
    "runs/ref6_vs_adda_fp32_ka_gt60_20260802/"
    "bem_ka111_ref6_pfft/result.json",
    "runs/preview_size_sweep_20260803/"
    "ka111_ref6_inner012_outer3_fp16phase_1g",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/preview_size_sweep_20260803/preview_size_sweep"
        ),
    )
    return parser.parse_args()


def elapsed_seconds(path: Path) -> float:
    match = re.search(
        r"Elapsed \(wall clock\) time.*: ([0-9:.]+)",
        path.read_text(encoding="utf-8"),
    )
    if not match:
        raise ValueError(f"elapsed time is absent from {path}")
    fields = [float(value) for value in match.group(1).split(":")]
    return fields[-1] + 60.0 * fields[-2] + (
        3600.0 * fields[-3] if len(fields) == 3 else 0.0
    )


def load_physical(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))["physical"]
    return (
        np.asarray(data["theta_degrees"], dtype=float),
        np.asarray(data["mueller"], dtype=float),
    )


def errors(reference_path: Path, result_path: Path) -> dict[str, float]:
    reference_theta, reference = load_physical(reference_path)
    theta, result = load_physical(result_path)
    if reference_theta.shape != theta.shape or not np.allclose(
        reference_theta, theta
    ):
        reference = np.asarray(
            [
                [
                    np.interp(theta, reference_theta, reference[i, j])
                    for j in range(4)
                ]
                for i in range(4)
            ]
        )
    weights = np.sin(np.deg2rad(theta))
    difference = result - reference
    return {
        "full_mueller_percent": float(
            100.0
            * np.sqrt(
                np.sum(difference**2 * weights)
                / np.sum(reference**2 * weights)
            )
        ),
        "m11_percent": float(
            100.0
            * np.sqrt(
                np.sum(difference[0, 0] ** 2 * weights)
                / np.sum(reference[0, 0] ** 2 * weights)
            )
        ),
        "forward_m11_percent": float(
            100.0
            * abs(result[0, 0, 0] / reference[0, 0, 0] - 1.0)
        ),
    }


def record(root: Path, ka: int, paths: tuple[str, str]) -> dict:
    reference = root / paths[0]
    directory = root / paths[1]
    result_path = directory / "result.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "ka": ka,
        "accepted": ka != 111,
        "wall_s": elapsed_seconds(directory / "time.txt"),
        "primary_iterations": data["pfft_fgmres"]["outer_iterations"],
        "primary_residual": data["pfft_fgmres"]["fmm_residual"],
        "polarization_correction_iterations": data["physical"][
            "parallel_iterations"
        ],
        "polarization_residual": data["physical"][
            "parallel_fmm_residual"
        ],
        "polarization_mode": data["physical"]["polarization_mode"],
        "reference": str(reference.resolve()),
        "result": str(result_path.resolve()),
        **errors(reference, result_path),
    }


def main() -> None:
    args = arguments()
    root = args.root.resolve()
    records = [record(root, ka, paths) for ka, paths in CASES.items()]
    records.append(record(root, 111, REJECTED_111))
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps({"cases": records}, indent=2) + "\n",
        encoding="utf-8",
    )

    accepted = [item for item in records if item["accepted"]]
    sizes = np.asarray([item["ka"] for item in accepted])
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(
        sizes, [item["wall_s"] for item in accepted], "o-",
        color="#2166ac", label="accepted preview",
    )
    rejected = records[-1]
    axes[0, 0].scatter(
        [111], [rejected["wall_s"]], marker="x", s=100,
        color="#b2182b", label="rejected by physics",
    )
    axes[0, 0].set_ylabel("Complete wall time, s")
    axes[0, 0].legend()

    for key, label, color in (
        ("full_mueller_percent", "full Mueller", "#1b9e77"),
        ("m11_percent", "M11", "#d95f02"),
        ("forward_m11_percent", "forward M11", "#7570b3"),
    ):
        axes[0, 1].semilogy(
            sizes, [item[key] for item in accepted], "o-",
            label=label, color=color,
        )
    axes[0, 1].scatter(
        [111], [rejected["full_mueller_percent"]], marker="x", s=100,
        color="#b2182b",
    )
    axes[0, 1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0, 1].set_ylabel("Difference from converged result, %")
    axes[0, 1].legend()

    axes[1, 0].semilogy(
        sizes, [item["primary_residual"] for item in accepted], "o-",
        label="primary polarization",
    )
    axes[1, 0].semilogy(
        sizes, [item["polarization_residual"] for item in accepted], "s-",
        label="second polarization",
    )
    axes[1, 0].set_ylabel("Reported operator/projected residual")
    axes[1, 0].legend()

    axes[1, 1].bar(
        sizes,
        [item["polarization_correction_iterations"] for item in accepted],
        width=5.0,
        color="#4daf4a",
    )
    axes[1, 1].set_ylabel("Second-polarization correction steps")
    axes[1, 1].set_ylim(0, 3)

    for axis in axes.flat:
        axis.set_xlabel("Size parameter ka")
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Regular hexagonal prism, m=1.3, ref=6: "
        "validated fast fixed-orientation solves"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
