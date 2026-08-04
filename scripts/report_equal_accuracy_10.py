#!/usr/bin/env python3
"""Audit and report ten predeclared equal-residual BEM/ADDA benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from verify_mie import mie_mueller


CASES = [(shape, ka) for shape in ("sphere", "prism") for ka in (2, 4, 6, 8, 10)]
RESIDUAL_LIMIT = 1.05e-5
SELF_CONVERGENCE_LIMIT = 0.02
CROSS_METHOD_LIMIT = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bem-bin", type=Path, required=True)
    parser.add_argument("--adda-bin", type=Path, required=True)
    return parser.parse_args()


def read_wall(path: Path) -> float:
    match = re.search(r"^wall_s=([0-9.]+)$", path.read_text(), re.MULTILINE)
    if not match:
        raise ValueError(f"{path}: wall_s is missing")
    return float(match.group(1))


def load_bem(path: Path) -> tuple[np.ndarray, np.ndarray, list[float], dict]:
    data = json.loads(path.read_text())
    physical = data["physical"]
    theta = np.asarray(physical["theta_degrees"], dtype=float)
    matrix = np.asarray(physical["mueller"], dtype=float)
    residuals = []
    section = data.get("pfft_fgmres") or data.get("mbj") or {}
    value = section.get("fmm_residual")
    if value is not None:
        residuals.append(float(value))
    value = physical.get("parallel_fmm_residual")
    if value is not None:
        residuals.append(float(value))
    if matrix.shape != (4, 4, len(theta)):
        raise ValueError(f"{path}: invalid Mueller shape {matrix.shape}")
    if len(residuals) != 2:
        raise ValueError(f"{path}: expected two independently verified residuals")
    return theta, matrix, residuals, data


def load_adda(directory: Path) -> tuple[np.ndarray, np.ndarray, list[float]]:
    table = np.loadtxt(directory / "mueller_scatgrid", skiprows=1)
    if table.ndim != 2 or table.shape[1] != 18:
        raise ValueError(f"{directory}/mueller_scatgrid: expected theta, phi, and 16 columns")
    theta = table[:, 0]
    matrix = table[:, 2:].reshape(-1, 4, 4).transpose(1, 2, 0)
    text = (directory / "log").read_text(errors="replace")
    residuals = [
        float(value)
        for value in re.findall(
            r"Final \(recalculated\) residual norm:\s*([0-9.eE+-]+)", text
        )
    ]
    if len(residuals) != 2:
        raise ValueError(f"{directory}/log: expected two recalculated residuals")
    return theta, matrix, residuals


def weighted_relative(theta: np.ndarray, candidate: np.ndarray, reference: np.ndarray) -> float:
    radians = np.deg2rad(theta)
    weight = np.sin(radians)
    numerator = np.trapezoid(np.sum((candidate - reference) ** 2, axis=(0, 1)) * weight, radians)
    denominator = np.trapezoid(np.sum(reference**2, axis=(0, 1)) * weight, radians)
    return float(math.sqrt(max(numerator, 0.0) / max(denominator, 1e-300)))


def normalized(matrix: np.ndarray) -> np.ndarray:
    scale = float(matrix[0, 0, 0])
    if not math.isfinite(scale) or abs(scale) < 1e-300:
        raise ValueError("invalid forward M11 normalization")
    return matrix / scale


def adda_metadata(directory: Path) -> dict:
    text = (directory / "log").read_text(errors="replace")
    dpl = re.search(r"Dipoles/lambda:\s*([0-9.]+)", text)
    dipoles = re.search(r"Total number of occupied dipoles:\s*(\d+)", text)
    if not dpl or not dipoles:
        raise ValueError(f"{directory}/log: discretization metadata is missing")
    return {
        "actual_dipoles_per_wavelength": float(dpl.group(1)),
        "occupied_dipoles": int(dipoles.group(1)),
    }


def audit_case(root: Path, shape: str, ka: int) -> dict:
    case = root / f"{shape}_ka{ka}"
    theta_bem, bem, bem_residuals, bem_data = load_bem(case / "bem_production/result.json")
    theta_bem_coarse, bem_coarse, bem_coarse_residuals, _ = load_bem(case / "bem_control/result.json")
    theta_adda, adda, adda_residuals = load_adda(case / "adda_official_production_dpl20")
    theta_adda_coarse, adda_coarse, adda_coarse_residuals = load_adda(case / "adda_official_control_dpl15")

    grids = (theta_bem, theta_bem_coarse, theta_adda, theta_adda_coarse)
    common_grid = all(len(grid) == 181 for grid in grids)
    common_grid = common_grid and all(np.allclose(theta_bem, grid, atol=1e-12, rtol=0) for grid in grids[1:])
    if not common_grid:
        raise ValueError(f"{case}: angular grids are not identical")

    bem_self = weighted_relative(theta_bem, normalized(bem), normalized(bem_coarse))
    adda_self = weighted_relative(theta_bem, normalized(adda), normalized(adda_coarse))
    bem_self_forward_ratio = float(bem[0, 0, 0] / bem_coarse[0, 0, 0])
    adda_self_forward_ratio = float(adda[0, 0, 0] / adda_coarse[0, 0, 0])
    cross_shape = weighted_relative(theta_bem, normalized(bem), normalized(adda))
    forward_ratio = float(bem[0, 0, 0] / adda[0, 0, 0])

    bem_wall_samples = [read_wall(case / "bem_production.time.txt")]
    adda_wall_samples = [read_wall(case / "adda_official_production_dpl20/time.txt")]
    all_bem_residuals = list(bem_residuals)
    all_adda_residuals = list(adda_residuals)
    independent_polarizations_ok = (
        str(bem_data["physical"].get("polarization_mode", "")).startswith("independent")
        and not bem_data["physical"].get("trusted_cyclic_exact_geometry_used", False)
    )
    for replicate in (2, 3):
        _, _, residuals, repeat_data = load_bem(case / f"bem_production_r{replicate}/result.json")
        all_bem_residuals.extend(residuals)
        independent_polarizations_ok = independent_polarizations_ok and (
            str(repeat_data["physical"].get("polarization_mode", "")).startswith("independent")
            and not repeat_data["physical"].get("trusted_cyclic_exact_geometry_used", False)
        )
        _, _, residuals = load_adda(case / f"adda_official_production_dpl20_r{replicate}")
        all_adda_residuals.extend(residuals)
        bem_wall_samples.append(read_wall(case / f"bem_production_r{replicate}.time.txt"))
        adda_wall_samples.append(read_wall(case / f"adda_official_production_dpl20_r{replicate}/time.txt"))
    bem_wall = statistics.median(bem_wall_samples)
    adda_wall = statistics.median(adda_wall_samples)
    ratio = adda_wall / bem_wall
    residual_ok = max(all_bem_residuals + bem_coarse_residuals + all_adda_residuals + adda_coarse_residuals) <= RESIDUAL_LIMIT
    convergence_ok = (
        bem_self <= SELF_CONVERGENCE_LIMIT
        and adda_self <= SELF_CONVERGENCE_LIMIT
        and abs(bem_self_forward_ratio - 1.0) <= SELF_CONVERGENCE_LIMIT
        and abs(adda_self_forward_ratio - 1.0) <= SELF_CONVERGENCE_LIMIT
    )
    agreement_ok = cross_shape <= CROSS_METHOD_LIMIT and abs(forward_ratio - 1.0) <= CROSS_METHOD_LIMIT
    mie_metrics = None
    mie_ok = True
    if shape == "sphere":
        mie = np.asarray(mie_mueller(theta_bem, complex(1.3, 0.0), ka), dtype=float)
        mie_metrics = {
            "bem_shape_difference": weighted_relative(theta_bem, normalized(bem), normalized(mie)),
            "adda_shape_difference": weighted_relative(theta_bem, normalized(adda), normalized(mie)),
            "bem_forward_m11_ratio": float(bem[0, 0, 0] / mie[0, 0, 0]),
            "adda_forward_m11_ratio": float(adda[0, 0, 0] / mie[0, 0, 0]),
        }
        mie_ok = (
            mie_metrics["bem_shape_difference"] <= SELF_CONVERGENCE_LIMIT
            and mie_metrics["adda_shape_difference"] <= SELF_CONVERGENCE_LIMIT
            and abs(mie_metrics["bem_forward_m11_ratio"] - 1.0) <= SELF_CONVERGENCE_LIMIT
            and abs(mie_metrics["adda_forward_m11_ratio"] - 1.0) <= SELF_CONVERGENCE_LIMIT
        )
    claimable = (
        common_grid
        and independent_polarizations_ok
        and residual_ok
        and convergence_ok
        and agreement_ok
        and mie_ok
    )

    return {
        "case": f"{shape}_ka{ka}",
        "shape": shape,
        "ka": ka,
        "ri": 1.3,
        "bem_ref": int(bem_data["refinements"]),
        "bem_system_dofs": int(bem_data["system_dofs"]),
        "bem_points_per_shortest_wavelength": float(bem_data["p2_nodes_per_wavelength_min"]),
        "adda_dpl": 20,
        **{
            f"adda_{key}": value
            for key, value in adda_metadata(case / "adda_official_production_dpl20").items()
        },
        "angular_points": len(theta_bem),
        "independent_polarizations_ok": independent_polarizations_ok,
        "bem_wall_samples_s": bem_wall_samples,
        "adda_wall_samples_s": adda_wall_samples,
        "bem_wall_s": bem_wall,
        "adda_wall_s": adda_wall,
        "adda_wall_over_bem_wall": ratio,
        "bem_max_true_residual": max(all_bem_residuals),
        "adda_max_recalculated_residual": max(all_adda_residuals),
        "bem_self_convergence": bem_self,
        "adda_self_convergence": adda_self,
        "bem_self_forward_m11_ratio": bem_self_forward_ratio,
        "adda_self_forward_m11_ratio": adda_self_forward_ratio,
        "bem_vs_adda_shape_difference": cross_shape,
        "bem_forward_m11_over_adda": forward_ratio,
        "mie_metrics": mie_metrics,
        "residual_ok": residual_ok,
        "convergence_ok": convergence_ok,
        "agreement_ok": agreement_ok,
        "mie_ok": mie_ok,
        "claimable_equal_accuracy_timing": claimable,
    }


def ratio_text(value: float) -> str:
    if value >= 1.0:
        return f"{value:.3f}x BEM speedup"
    return f"{value:.3f}x ({1.0 / value:.2f}x BEM slowdown)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def git_metadata(directory: Path) -> dict:
    status = output(["git", "status", "--short"], directory)
    return {
        "root": str(directory.resolve()),
        "commit": output(["git", "rev-parse", "HEAD"], directory),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def build_provenance(bem_bin: Path, adda_bin: Path) -> dict:
    adda_root = adda_bin.resolve().parents[2]
    gpu = output([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,power.limit",
        "--format=csv,noheader",
    ])
    return {
        "benchmark_policy": {
            "cases": [f"{shape}_ka{ka}" for shape, ka in CASES],
            "relative_refractive_index": 1.3,
            "residual_target": 1e-5,
            "independent_polarizations": True,
            "scattering_angles": 181,
            "replicates": 3,
            "reported_statistic": "median complete process wall time",
            "bem_minimum_points_per_shortest_wavelength": 15,
            "adda_requested_dpl": 20,
            "self_convergence_limit": SELF_CONVERGENCE_LIMIT,
            "cross_method_limit": CROSS_METHOD_LIMIT,
        },
        "bem": {
            "binary": str(bem_bin.resolve()),
            "binary_sha256": sha256(bem_bin),
            "version": output([str(bem_bin), "--version"]),
            "source": git_metadata(REPO_ROOT),
        },
        "adda": {
            "binary": str(adda_bin.resolve()),
            "binary_sha256": sha256(adda_bin),
            "version": output([str(adda_bin), "-V"]),
            "source": git_metadata(adda_root),
        },
        "hardware": {
            "gpu": gpu,
            "cpu": output(["lscpu"]).splitlines(),
        },
    }


def write_report(root: Path, rows: list[dict], provenance: dict) -> None:
    csv_path = root / "equal_accuracy_10.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    (root / "equal_accuracy_10.json").write_text(json.dumps(rows, indent=2) + "\n")
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    lines = [
        "# Ten predeclared BEM versus ADDA timing ratios",
        "",
        "The ratio is `median ADDA complete wall time / median BEM complete wall time` "
        "from three independent runs with fresh application-cache directories. CUDA/OpenCL "
        "driver and compiler caches were not flushed, so this is a warm-system benchmark. "
        "A value below one is a BEM slowdown, not an acceleration. All ten cases were "
        "declared before execution; failed or unfavorable rows are retained.",
        "",
        "Both programs use a `1e-5` linear residual target, two independently solved "
        "polarizations, and the same 181 scattering angles. ADDA reports a final "
        "recalculated residual. BEM reports a final exact-FMM-operator residual. "
        "The production discretizations are BEM >=15 points per shortest wavelength "
        "and ADDA dpl=20; BEM one-level-coarser and ADDA dpl=15 controls must change "
        "the normalized complete Mueller matrix by no more than 2%. The two production "
        "matrices must agree within 5%, including forward M11.",
        "For spheres, both production matrices must also agree with exact Mie theory "
        "within 2% in normalized shape and forward M11.",
        "The ADDA baseline is the clean official adda-team/adda commit 8f550a7, not "
        "the locally modified FP32 experimental build.",
        f"BEM binary SHA-256: `{provenance['bem']['binary_sha256']}`. ADDA binary "
        f"SHA-256: `{provenance['adda']['binary_sha256']}`.",
        "",
        "| case | BEM wall, s | ADDA wall, s | ADDA/BEM | BEM residual | ADDA residual | BEM grid change | ADDA grid change | BEM-ADDA | valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['bem_wall_s']:.2f} | {row['adda_wall_s']:.2f} | "
            f"{ratio_text(row['adda_wall_over_bem_wall'])} | "
            f"{row['bem_max_true_residual']:.2e} | {row['adda_max_recalculated_residual']:.2e} | "
            f"{100 * row['bem_self_convergence']:.2f}% | {100 * row['adda_self_convergence']:.2f}% | "
            f"{100 * row['bem_vs_adda_shape_difference']:.2f}% | "
            f"{'yes' if row['claimable_equal_accuracy_timing'] else 'no'} |"
        )
    lines.extend([
        "",
        "A row marked `no` has no publishable speedup claim. Inspect the CSV/JSON gate "
        "fields instead of quoting its wall-time ratio.",
    ])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")

    labels = [row["case"].replace("sphere", "sphere ").replace("prism", "prism ") for row in rows]
    values = [row["adda_wall_over_bem_wall"] for row in rows]
    colors = ["#1b9e77" if row["claimable_equal_accuracy_timing"] and value >= 1 else "#d95f02" for row, value in zip(rows, values)]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(1.0, color="black", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("ADDA wall time / BEM wall time")
    ax.set_title("Ten predeclared equal-residual timing ratios; below 1 means BEM is slower")
    ax.tick_params(axis="x", rotation=35)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}x", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(root / "equal_accuracy_10.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = [audit_case(args.root, shape, ka) for shape, ka in CASES]
    provenance = build_provenance(args.bem_bin, args.adda_bin)
    write_report(args.root, rows, provenance)
    valid = sum(row["claimable_equal_accuracy_timing"] for row in rows)
    print(f"Audited {len(rows)} predeclared cases; {valid} pass every equal-accuracy gate.")
    print(args.root / "REPORT.md")
    if valid != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
