#!/usr/bin/env python3
"""Audit the archived ka=200 comparison without renormalizing away its error."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.archive / "bem_vs_mbs_fast_orientation_fixed_theta181/mueller_curves.csv"
    table = np.genfromtxt(source, delimiter=",", names=True)
    theta = table["theta_degrees"]
    matrices = {
        method: np.stack([table[f"{method}_M{i}{j}"] for i in range(1, 5)
                          for j in range(1, 5)], axis=1).reshape(-1, 4, 4)
        for method in ("bem", "mbs")
    }
    bem, mbs = matrices["bem"], matrices["mbs"]
    assert theta[0] == 0 and np.all(bem[:, 0, 0] > 0)
    signed = 100 * (mbs[:, 0, 0] / bem[:, 0, 0] - 1)
    result = json.loads((args.archive / "full_host_gmres/result.json").read_text())
    np.testing.assert_allclose(theta, result["theta_degrees"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(bem, np.asarray(result["mueller"]).transpose(2, 0, 1), rtol=1e-13, atol=1e-12)
    raw_mbs_path = (args.archive / "mbs_fast_orientation_fixed_theta181"
                    / "mbs_corrected_beta90_gamma60_theta181_n12"
                    / "mbs_corrected_beta90_gamma60_theta181_n12.dat")
    raw_mbs = np.loadtxt(raw_mbs_path, skiprows=1)
    signs = np.array([1, 1, -1, -1])
    raw_mbs_matrix = raw_mbs[:, 2:].reshape(-1, 4, 4) * (2 * np.pi) ** 2
    raw_mbs_matrix *= signs[None, :, None] * signs[None, None, :]
    np.testing.assert_allclose(mbs, raw_mbs_matrix, rtol=1e-13, atol=1e-12)
    report = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "archive_csv_checked_against_original_BEM_and_MBS_outputs": True,
        "case": {
            "ka": result["ka"], "ref": result["refinements"],
            "system_dofs": result["system_dofs"],
            "orientation": "BEM beta=90 deg, gamma=30 deg; MBS beta=90 deg, gamma=60 deg",
            "averaging": "alpha only, 3600 samples; not full SO(3)",
            "maximum_solver_residual": result["iterations"]["maximum_residual"],
            "mesh_convergence_checked_by_this_audit": False,
        },
        "units": "dimensionless k^2 dC/dOmega; archived MBS wavelength=1, conversion=(2*pi)^2",
        "forward": {
            "BEM_M11": float(bem[0, 0, 0]), "MBS_M11": float(mbs[0, 0, 0]),
            "MBS_over_BEM_minus1_percent": float(signed[0]),
            "all16_signed_delta_over_BEM_M11_percent": (100 * (mbs[0] - bem[0]) / bem[0, 0, 0]).tolist(),
        },
        "selected_angles": [
            {"theta_deg": float(theta[i]), "BEM_M11": float(bem[i, 0, 0]),
             "MBS_M11": float(mbs[i, 0, 0]), "signed_difference_percent": float(signed[i])}
            for i in [0, 1, 2, 3, 5, 10, 30, 90, 180]
        ],
        "raw_all16_relative_l2": float(np.linalg.norm(mbs - bem) / np.linalg.norm(bem)),
        "pointwise_M11_median_absolute_relative_difference": float(np.median(np.abs(signed)) / 100),
        "old_plot_normalization": {
            "M11": "each curve divided by its own M11(0); both forward values identically one",
            "other_elements": "each method's Mij(theta) divided by that method's M11(theta)",
            "caption": "old caption did report M11(0): MBS/BEM=0.979",
        },
        "caveats": [
            "The 1-degree grid is coarse relative to the ka=200 forward lobe; no interpolated fine-angle accuracy is claimed.",
            "The solver residual is not a bound on mesh-discretization error.",
            "Changing both ka and orientation averaging does not isolate either effect.",
        ],
    }
    check = args.out / "current_binary_forward_check.json"
    if check.exists():
        report["current_binary_forward_check"] = json.loads(check.read_text())
    rerun = args.out / "current_alpha3600_depth12/result/result.dat"
    if rerun.exists() and (rerun.parent.parent / "complete.json").exists():
        completion = json.loads((rerun.parent.parent / "complete.json").read_text())
        if completion["returncode"] == 0:
            new_table = np.loadtxt(rerun, skiprows=1)
            assert new_table.shape == (181, 18)
            np.testing.assert_allclose(theta, new_table[:, 0], atol=1e-10)
            new = new_table[:, 2:].reshape(-1, 4, 4) * (2 * np.pi) ** 2
            signs = np.array([1, 1, -1, -1])
            new *= signs[None, :, None] * signs[None, None, :]
            report["current_binary_full_curve_check"] = {
                "elapsed_s": completion["elapsed_s"],
                "forward_difference_vs_archived_MBS_percent": float(100 * (new[0, 0, 0] / mbs[0, 0, 0] - 1)),
                "all16_l2_vs_archived_MBS": float(np.linalg.norm(new - mbs) / np.linalg.norm(mbs)),
                "max_all16_abs_difference_over_archived_local_M11": float(np.max(np.abs(new - mbs) / mbs[:, 0, 0, None, None])),
                "selected_M11_difference_vs_BEM_percent": [float(100 * (new[i, 0, 0] / bem[i, 0, 0] - 1)) for i in [0, 1, 2, 5, 10]],
            }
    (args.out / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    np.savetxt(args.out / "M11_common_units.csv", np.column_stack((theta, bem[:, 0, 0], mbs[:, 0, 0], signed)),
               delimiter=",", header="theta_deg,BEM_M11,MBS_M11,MBS_over_BEM_minus1_percent", comments="")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    colors = ["#2166ac", "#d6604d"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    values = [1., mbs[0, 0, 0] / bem[0, 0, 0]]
    ax.bar(["BEM", "MBS-fast"], values, color=colors, width=0.5)
    ax.set_ylim(0.94, 1.02)
    for i, value in enumerate(values):
        ax.text(i, value + 0.002, f"{value:.6f}", ha="center")
    ax.set_ylabel(r"$M_{11}(0)/M_{11}^{BEM}(0)$")
    ax.set_title(f"Строго вперёд: разница {signed[0]:+.3f}% (ось Y увеличена)")
    for ax, stop in [(axes[0, 1], 5), (axes[1, 0], 180)]:
        mask = theta <= stop
        ax.semilogy(theta[mask], bem[mask, 0, 0] / bem[0, 0, 0], "o-" if stop == 5 else "-", color=colors[0], label="BEM")
        ax.semilogy(theta[mask], mbs[mask, 0, 0] / bem[0, 0, 0], "s--" if stop == 5 else "--", color=colors[1], label="MBS-fast")
        ax.set_xlabel("Угол рассеяния θ, град.")
        ax.set_ylabel(r"$M_{11}(\theta)/M_{11}^{BEM}(0)$")
        ax.set_title("Общий знаменатель для обеих кривых")
        ax.legend()
    ax = axes[1, 1]
    mask = theta <= 10
    ax.plot(theta[mask], signed[mask], "o-", color=colors[1])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Угол рассеяния θ, град.")
    ax.set_ylabel("100 × (MBS/BEM − 1), %")
    ax.set_title("Поточечная разница M11 около прямого направления")
    fig.suptitle("Архивный расчёт: ka=200, ref=7, 3 225 600 неизвестных\n"
                 "BEM β=90°, γ=30°; усреднение только по α (3600 поворотов)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for suffix in ["png", "pdf"]:
        fig.savefig(args.out / f"legacy_forward_common_normalization.{suffix}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            divisor = bem[0, 0, 0] if (i, j) == (0, 0) else bem[:, 0, 0]
            for matrix, label, style, color in [(bem, "BEM", "-", colors[0]), (mbs, "MBS-fast", "--", colors[1])]:
                if (i, j) == (0, 0):
                    ax.semilogy(theta, matrix[:, i, j] / divisor, style, label=label, color=color)
                else:
                    ax.plot(theta, matrix[:, i, j] / divisor, style, color=color)
            ax.set_title(f"M{i+1}{j+1}")
            ax.set_ylabel(r"$M_{11}/M_{11}^{BEM}(0)$" if (i, j) == (0, 0)
                          else rf"$M_{{{i+1}{j+1}}}/M_{{11}}^{{BEM}}(\theta)$")
            if i == 3:
                ax.set_xlabel("θ, град.")
    axes[0, 0].legend()
    fig.suptitle("ka=200, ref=7; фиксированные β, γ, усреднение по α\n"
                 "У обеих кривых общий знаменатель BEM; без подгонки масштаба")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ["png", "pdf"]:
        fig.savefig(args.out / f"all16_common_BEM_normalization.{suffix}", dpi=170)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ["forward", "selected_angles", "current_binary_full_curve_check"] if k in report}, indent=2))


if __name__ == "__main__":
    main()
