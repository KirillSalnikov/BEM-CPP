#!/usr/bin/env python3
"""Build a compact FMM-versus-pFFT benchmark report."""

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
        metavar="LABEL=FMM_JSON=PFFT_JSON",
    )
    parser.add_argument("--adda", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_case(value: str) -> dict:
    label, fmm_path, pfft_path = value.split("=", 2)
    fmm = json.loads(Path(fmm_path).read_text())
    pfft = json.loads(Path(pfft_path).read_text())
    theta_fmm = np.asarray(fmm["physical"]["theta_degrees"])
    theta_pfft = np.asarray(pfft["physical"]["theta_degrees"])
    mueller_fmm = np.asarray(fmm["physical"]["mueller"])
    mueller_pfft = np.asarray(pfft["physical"]["mueller"])
    indices = np.asarray(
        [np.argmin(np.abs(theta_fmm - angle)) for angle in theta_pfft]
    )
    if not np.allclose(theta_fmm[indices], theta_pfft):
        raise ValueError(f"angle grids do not match for {label}")
    mueller_fmm = mueller_fmm[:, :, indices]

    strict = pfft.get("pfft_fgmres") is not None

    def first_system_total(data: dict) -> float:
        result = data["fmm_setup_s"] + data["mbj_local_setup_s"]
        if data.get("pfft_fgmres") is not None:
            result += data["pfft_fgmres"]["fmm_switch_setup_s"]
            result += data["pfft_fgmres"]["outer_solve_s"]
        else:
            result += data["mbj"]["solve_s"]
        return result

    accelerated_solve_s = (
        pfft["pfft_fgmres"]["outer_solve_s"]
        if strict else pfft["mbj"]["solve_s"]
    )

    return {
        "label": label,
        "fmm": fmm,
        "pfft": pfft,
        "theta": theta_pfft,
        "mueller_fmm": mueller_fmm,
        "mueller_pfft": mueller_pfft,
        "strict": strict,
        "near_cache_hit": pfft.get(
            "near_correction_cache", {}
        ).get("hit", False),
        "fmm_solve_s": fmm["mbj"]["solve_s"],
        "pfft_solve_s": accelerated_solve_s,
        "fmm_total_s": first_system_total(fmm),
        "pfft_total_s": first_system_total(pfft),
        "solve_speedup": fmm["mbj"]["solve_s"] / accelerated_solve_s,
        "total_speedup": first_system_total(fmm) / first_system_total(pfft),
        "outer_iterations": (
            pfft["pfft_fgmres"]["outer_iterations"] if strict
            else pfft["mbj"]["iterations"]
        ),
        "inner_iterations": (
            pfft["pfft_fgmres"]["inner_iterations"] if strict else 0
        ),
        "mueller_relative_l2": float(
            np.linalg.norm(mueller_pfft - mueller_fmm)
            / np.linalg.norm(mueller_fmm)
        ),
        "m11_relative_l2": float(
            np.linalg.norm(mueller_pfft[0, 0] - mueller_fmm[0, 0])
            / np.linalg.norm(mueller_fmm[0, 0])
        ),
    }


def adda_mueller(path: Path, theta: np.ndarray) -> np.ndarray:
    table = np.loadtxt(path / "mueller", skiprows=1)
    indices = np.asarray(
        [np.argmin(np.abs(table[:, 0] - angle)) for angle in theta]
    )
    if not np.allclose(table[indices, 0], theta):
        raise ValueError("ADDA and BEM angle grids do not match")
    return table[indices, 1:].reshape(-1, 4, 4).transpose(1, 2, 0)


def make_plot(cases: list[dict], adda: np.ndarray | None, output: Path) -> None:
    labels = [case["label"] for case in cases]
    x = np.arange(len(cases))
    width = 0.36
    accelerated_label = (
        "pFFT-FGMRES" if all(c["strict"] for c in cases) else "pFFT"
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.bar(x - width / 2, [c["fmm_solve_s"] for c in cases], width, label="FMM")
    ax.bar(
        x + width / 2,
        [c["pfft_solve_s"] for c in cases],
        width,
        label=accelerated_label,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Время решения, с")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x, [c["solve_speedup"] for c in cases], "o-", label="Только решение")
    ax.plot(x, [c["total_speedup"] for c in cases], "s-", label="Подготовка + решение")
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_ylabel("Ускорение, раз")
    ax.set_xticks(x, labels)
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.bar(
        x,
        [100 * c["mueller_relative_l2"] for c in cases],
        color="#d55e00",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Отличие матрицы Мюллера от FMM, %")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)

    largest = cases[-1]
    theta = largest["theta"]
    fmm = largest["mueller_fmm"]
    pfft = largest["mueller_pfft"]
    ax = axes[1, 1]
    ax.plot(theta, fmm[0, 0] / fmm[0, 0, 0], lw=2.0, label="FMM")
    ax.plot(
        theta,
        pfft[0, 0] / pfft[0, 0, 0],
        lw=1.6,
        label=accelerated_label,
    )
    if adda is not None:
        ax.plot(theta, adda[0, 0] / adda[0, 0, 0], lw=1.3, label="ADDA dpl20")
    ax.set_yscale("log")
    ax.set_xlabel("Угол рассеяния, град.")
    ax.set_ylabel(r"$M_{11}(\theta)/M_{11}(0)$")
    ax.grid(which="both", alpha=0.25)
    ax.legend()

    fig.suptitle(
        "Строгий pFFT-FGMRES для P2-уравнения Мюллера"
        if all(c["strict"] for c in cases)
        else "Экспериментальный FFT-бэкенд для P2-уравнения Мюллера"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = [load_case(value) for value in args.case]
    adda = adda_mueller(args.adda, cases[-1]["theta"]) if args.adda else None
    make_plot(cases, adda, args.out_dir / "muller_pfft_benchmark.png")

    serializable = []
    for case in cases:
        serializable.append(
            {
                key: value
                for key, value in case.items()
                if key
                not in {"fmm", "pfft", "theta", "mueller_fmm", "mueller_pfft"}
            }
        )
    (args.out_dir / "muller_pfft_benchmark.json").write_text(
        json.dumps(serializable, indent=2) + "\n"
    )

    lines = [
        "# pFFT для P2-уравнения Мюллера",
        "",
        "| Случай | Кэш поправки | Внешние FMM итерации | Внутренние pFFT итерации | Ускорение решения | Полное ускорение первой системы | Отличие Мюллера от FMM |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case['label']} | "
            f"{'да' if case['near_cache_hit'] else 'нет'} | "
            f"{case['outer_iterations']} | "
            f"{case['inner_iterations']} | "
            f"{case['solve_speedup']:.2f}x | "
            f"{case['total_speedup']:.2f}x | "
            f"{case['mueller_relative_l2']:.2e} |"
        )
    lines.extend(
        [
            "",
            "pFFT использует общую регулярную сетку для внешнего и внутреннего "
            "ядер и кубическую интерполяцию. В строгом режиме pFFT приближённо "
            "обращается внутренним GMRES, а внешний FGMRES проверяет невязку "
            "исходным FMM-оператором.",
            "",
            "Полное ускорение включает фактическую подготовку конкретного "
            "запуска. Для строк с `near_cache_hit=true` это повторный запуск "
            "с проверенным кэшем точной ближней поправки.",
        ]
    )
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
