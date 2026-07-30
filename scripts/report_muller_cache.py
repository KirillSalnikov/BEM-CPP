#!/usr/bin/env python3
"""Report cold-versus-cached Muller near-correction setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="LABEL=COLD_SETUP_JSON=WARM_SETUP_JSON",
    )
    parser.add_argument(
        "--physical",
        required=True,
        metavar="LABEL=FMM_JSON=COLD_PFFT_JSON=WARM_PFFT_JSON",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def setup_total(data: dict) -> float:
    return data["fmm_setup_s"] + data["mbj_local_setup_s"]


def physical_total(data: dict) -> float:
    pfft = data.get("pfft_fgmres")
    solve = (
        pfft["outer_solve_s"]
        if pfft is not None
        else data["mbj"]["solve_s"]
    )
    switch = pfft["fmm_switch_setup_s"] if pfft is not None else 0.0
    physical = data["physical"]
    return (
        setup_total(data)
        + switch
        + solve
        + physical["parallel_s"]
        + physical["farfield_s"]
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for value in args.case:
        label, cold_path, warm_path = value.split("=", 2)
        cold = load(cold_path)
        warm = load(warm_path)
        cases.append(
            {
                "label": label,
                "cold_near_s": cold["fmm_setup_breakdown"][
                    "near_correction_s"
                ],
                "warm_near_s": warm["fmm_setup_breakdown"][
                    "near_correction_s"
                ],
                "cold_setup_s": setup_total(cold),
                "warm_setup_s": setup_total(warm),
                "entries": warm["near_correction_cache"]["entries"],
            }
        )

    physical_label, baseline_path, cold_path, warm_path = (
        args.physical.split("=", 3)
    )
    baseline = load(baseline_path)
    cold = load(cold_path)
    warm = load(warm_path)
    baseline_total = physical_total(baseline)
    cold_total = physical_total(cold)
    warm_total = physical_total(warm)

    labels = [case["label"] for case in cases]
    x = np.arange(len(cases))
    width = 0.36
    figure, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.6), constrained_layout=True
    )

    axes[0].bar(
        x - width / 2,
        [case["cold_near_s"] for case in cases],
        width,
        label="Первый запуск",
    )
    axes[0].bar(
        x + width / 2,
        [case["warm_near_s"] for case in cases],
        width,
        label="Загрузка кэша",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Поправка ближнего поля, с")
    axes[0].set_xticks(x, labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(
        x - width / 2,
        [case["cold_setup_s"] for case in cases],
        width,
        label="Первый запуск",
    )
    axes[1].bar(
        x + width / 2,
        [case["warm_setup_s"] for case in cases],
        width,
        label="Загрузка кэша",
    )
    axes[1].set_ylabel("Полная подготовка, с")
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", alpha=0.25)

    total_labels = ["FMM", "pFFT\nпервый", "pFFT\nповторный"]
    total_values = [baseline_total, cold_total, warm_total]
    axes[2].bar(
        total_labels,
        total_values,
        color=["#2878b5", "#e17c05", "#2ca02c"],
    )
    axes[2].set_ylabel("Полный физический расчёт, с")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].set_title(physical_label)
    for index, value in enumerate(total_values):
        axes[2].text(
            index,
            value,
            f"{value:.1f} с",
            ha="center",
            va="bottom",
        )

    figure.suptitle(
        "Повторное использование точной ближней поправки"
    )
    figure.savefig(
        args.out_dir / "muller_near_cache_benchmark.png", dpi=180
    )
    plt.close(figure)

    summary = {
        "setup_cases": cases,
        "physical": {
            "label": physical_label,
            "fmm_total_s": baseline_total,
            "cold_pfft_total_s": cold_total,
            "cached_pfft_total_s": warm_total,
            "cold_speedup": baseline_total / cold_total,
            "cached_speedup": baseline_total / warm_total,
        },
    }
    (args.out_dir / "muller_near_cache_benchmark.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
