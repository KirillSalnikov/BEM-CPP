#!/usr/bin/env python3
"""Plot the strict pFFT-FGMRES inner-iteration sweep."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="LABEL=GLOB",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_case(value: str) -> dict:
    label, patterns = value.split("=", 1)
    points = []
    paths = []
    for pattern in patterns.split(","):
        paths.extend(glob.glob(pattern))
    for path in sorted(set(paths)):
        data = json.loads(Path(path).read_text())
        if data["pfft_inner_tolerance"] != 0.1:
            continue
        if data.get("pfft_inner_iterations_auto", False):
            continue
        result = data["pfft_fgmres"]
        points.append(
            {
                "limit": data["pfft_inner_max_iterations"],
                "outer_iterations": result["outer_iterations"],
                "inner_iterations": result["inner_iterations"],
                "solve_s": result["outer_solve_s"],
                "inner_solve_s": result["inner_solve_s"],
                "residual": result["fmm_residual"],
            }
        )
    points.sort(key=lambda item: item["limit"])
    if not points:
        raise ValueError(f"no sweep points for {label}")
    return {"label": label, "points": points}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = [load_case(value) for value in args.case]
    figure, axes = plt.subplots(
        1, 3, figsize=(14.5, 5.1)
    )

    for case in cases:
        limits = [point["limit"] for point in case["points"]]
        axes[0].plot(
            limits,
            [point["solve_s"] for point in case["points"]],
            "o-",
            label=case["label"],
        )
        axes[1].plot(
            limits,
            [point["outer_iterations"] for point in case["points"]],
            "o-",
            label=case["label"],
        )
        axes[2].plot(
            limits,
            [point["inner_iterations"] for point in case["points"]],
            "o-",
            label=case["label"],
        )

    axes[0].set_ylabel("Время строгого решения, с")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Внешние FMM-итерации")
    axes[2].set_ylabel("Суммарные внутренние pFFT-итерации")
    for axis in axes:
        axis.set_xlabel("Максимум внутренних итераций")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(
        "Выбор силы внутреннего pFFT-GMRES при допуске 0.10"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    figure.savefig(
        args.out_dir / "pfft_inner_iteration_sweep.png", dpi=180
    )
    plt.close(figure)
    (args.out_dir / "pfft_inner_iteration_sweep.json").write_text(
        json.dumps(cases, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
