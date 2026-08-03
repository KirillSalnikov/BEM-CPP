#!/usr/bin/env python3
"""Run and summarize the reviewed quick/standard orientation validation matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BEM = ROOT / "bem"
DEFAULT_ROOT = ROOT / "runs" / "orientation_profile_validation_v4"
PROFILES = ("quick", "standard")
CASES = {
    "sphere_ka1_m1p3": {
        "shape": "sphere", "ka": 1.0, "ri": 1.3,
    },
    "sphere_ka3_m1p5": {
        "shape": "sphere", "ka": 3.0, "ri": 1.5,
    },
    "cube_ka2_m1p3": {
        "shape": "cube", "ka": 2.0, "ri": 1.3,
    },
    "prism6_ka3_m1p5": {
        "shape": "prism", "ka": 3.0, "ri": 1.5,
        "sides": 6, "aspect": 1.0,
    },
    "prism7_ka5_m1p3": {
        "shape": "prism", "ka": 5.0, "ri": 1.3,
        "sides": 7, "aspect": 0.65,
    },
    "prism5_ka2_m1p8": {
        "shape": "prism", "ka": 2.0, "ri": 1.8,
        "sides": 5, "aspect": 1.4,
    },
    "prism8_ka4_m2p5": {
        "shape": "prism", "ka": 4.0, "ri": 2.5,
        "sides": 8, "aspect": 0.5,
        "output_name": {"standard": "standard_mbj100_j5_a64_dihedral_run"},
    },
    "cube_ka5_m2p0": {
        "shape": "cube", "ka": 5.0, "ri": 2.0,
        "output_name": {"standard": "standard_j5"},
    },
    "sphere_ka10_m1p1": {
        "shape": "sphere", "ka": 10.0, "ri": 1.1,
    },
    "asymmetric_ka3_m1p5": {
        "shape": "obj", "ka": 3.0, "ri": 1.5,
        "obj": "model_repaired.obj",
        "refinement": {"quick": 0, "standard": 1},
    },
}


def output_for(
    output_root: Path, case_name: str, profile: str,
) -> Path:
    case = CASES[case_name]
    name = case.get("output_name", {}).get(profile, profile)
    return output_root / case_name / name


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def calculation_command(
    case: dict[str, Any], profile: str, output: Path, alpha: int,
) -> list[str]:
    command = [
        str(BEM), "average",
        "--shape", str(case["shape"]),
        "--ka", f"{case['ka']:g}",
        "--ri", f"{case['ri']:g}",
        "--quality", profile,
        "--alpha", str(alpha),
        "--out", str(output),
    ]
    if case["shape"] == "prism":
        command += [
            "--sides", str(case["sides"]),
            "--aspect", f"{case['aspect']:g}",
        ]
    elif case["shape"] == "obj":
        command += [
            "--obj", str((ROOT / case["obj"]).resolve()),
            "--ref", str(case["refinement"][profile]),
        ]
    command += case.get("profile_arguments", {}).get(profile, [])
    return command


def result_is_valid(output: Path) -> bool:
    validation = output / "validation.json"
    if not validation.is_file():
        return False
    data = load_json(validation)
    return not data.get("errors")


def run_matrix(args: argparse.Namespace) -> int:
    output_root = Path(args.out_root).expanduser().resolve()
    case_names = args.case or list(CASES)
    profiles = args.profile or list(PROFILES)
    failures = 0
    for case_name in case_names:
        case = CASES[case_name]
        for profile in profiles:
            output = output_for(output_root, case_name, profile)
            if result_is_valid(output) and not args.force:
                print(f"SKIP {case_name}/{profile}: validated result exists", flush=True)
                continue
            if (output / "effective_config.json").is_file() and not args.dry_run:
                command = [str(BEM), "resume", str(output)]
            else:
                command = calculation_command(case, profile, output, args.alpha)
            if args.dry_run:
                command.append("--dry-run")
            print(f"RUN  {case_name}/{profile}", flush=True)
            print("     " + " ".join(command), flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                failures += 1
                print(
                    f"FAIL {case_name}/{profile}: exit {completed.returncode}",
                    file=sys.stderr,
                    flush=True,
                )
                if not args.keep_going:
                    return completed.returncode
    return 1 if failures else 0


def comparison(candidate: Path, reference: Path) -> dict[str, Any] | None:
    completed = subprocess.run(
        [str(BEM), "validate", str(candidate), "--reference", str(reference), "--json"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        return json.loads(completed.stdout).get("comparison")
    except json.JSONDecodeError:
        return None


def mueller_arrays(data: dict[str, Any]) -> tuple[list[float], list[list[list[float]]]]:
    theta = data.get("theta_degrees")
    mueller = data.get("mueller")
    if theta is None or mueller is None:
        physical = data["physical"]
        theta = physical["theta_degrees"]
        mueller = physical["mueller"]
    return theta, mueller


def normalized_weighted_l2(
    theta: list[float], candidate: list[list[list[float]]],
    reference: list[list[list[float]]],
) -> float:
    candidate_scale = max(abs(candidate[0][0][0]), 1.0e-300)
    reference_scale = max(abs(reference[0][0][0]), 1.0e-300)
    numerator = 0.0
    denominator = 0.0
    for angle_index, angle in enumerate(theta):
        weight = max(math.sin(math.radians(angle)), 0.0)
        for i in range(4):
            for j in range(4):
                value = candidate[i][j][angle_index] / candidate_scale
                target = reference[i][j][angle_index] / reference_scale
                numerator += weight * (value - target) ** 2
                denominator += weight * target ** 2
    return math.sqrt(numerator / max(denominator, 1.0e-300))


def mie_metrics(data: dict[str, Any], case: dict[str, Any]) -> dict[str, float] | None:
    if case["shape"] != "sphere":
        return None
    sys.path.insert(0, str(ROOT))
    from verify_mie import mie_mueller  # pylint: disable=import-outside-toplevel

    theta, mueller = mueller_arrays(data)
    reference = mie_mueller(theta, complex(case["ri"], 0.0), case["ka"])
    return {
        "normalized_mueller_l2": normalized_weighted_l2(theta, mueller, reference),
        "forward_m11_ratio": mueller[0][0][0] / reference[0][0][0],
    }


def record_for(case_name: str, profile: str, output: Path) -> dict[str, Any] | None:
    result_path = output / "result.json"
    config_path = output / "effective_config.json"
    validation_path = output / "validation.json"
    if not result_path.is_file() or not config_path.is_file():
        return None
    result = load_json(result_path)
    config = load_json(config_path)
    validation = load_json(validation_path) if validation_path.is_file() else {}
    provenance_path = output / "validation_provenance.json"
    provenance = load_json(provenance_path) if provenance_path.is_file() else {}
    validation_errors = validation.get("errors", []) if validation else [
        "solver exited before wrapper validation"
    ]
    orientation = config["effective_parameters"]["orientation"]
    adaptive = result.get("adaptive", {})
    runtime_orientation = result.get("orientation", {})
    maximum_base = orientation.get("maximum_base_orientations")
    maximum_solved_base = orientation.get(
        "maximum_solved_base_orientations"
    )
    if maximum_solved_base is None and maximum_base is not None:
        maximum_solved_base = (
            (maximum_base + 2) // 2
            if orientation.get("dihedral_symmetry_reuse") else maximum_base
        )
    solved_base = runtime_orientation.get("solved_base_orientations")
    savings = None
    if maximum_base and solved_base is not None:
        savings = 1.0 - solved_base / maximum_base
    record = {
        "case": case_name,
        "profile": profile,
        "shape": config["inputs"]["shape"],
        "ka": config["inputs"]["ka"],
        "ri": config["inputs"]["relative_refractive_index"],
        "ref": config["inputs"]["refinement"],
        "points_per_shortest_wavelength": config["inputs"].get(
            "estimated_points_per_shortest_wavelength"
        ),
        "system_dofs": result.get("system_dofs", config["estimate"]["system_dofs"]),
        "solver": config["effective_parameters"]["solver"],
        "adaptive_minimum_level": adaptive.get("minimum_level"),
        "adaptive_maximum_level": adaptive.get("maximum_level"),
        "adaptive_accepted_level": adaptive.get("accepted_level"),
        "adaptive_converged": adaptive.get("converged"),
        "solved_base_orientations": solved_base,
        "maximum_base_orientations": maximum_base,
        "maximum_solved_base_orientations": maximum_solved_base,
        "orientation_savings_fraction": savings,
        "total_iterations": result.get("iterations", {}).get("total"),
        "maximum_residual": result.get("iterations", {}).get("maximum_residual"),
        "solve_s": result.get("timing", {}).get("solve_s"),
        "total_with_setup_s": result.get("timing", {}).get("total_with_setup_s"),
        "wall_time_s": validation.get("wall_time_s"),
        "timing_scope": provenance.get("timing_scope", "complete"),
        "reused_base_orientations": provenance.get("reused_base_orientations", 0),
        "validation_passed": not validation_errors,
        "validation_errors": "; ".join(validation_errors),
    }
    mie = mie_metrics(result, CASES[case_name])
    if mie:
        record["mie_normalized_mueller_l2"] = mie["normalized_mueller_l2"]
        record["mie_forward_m11_ratio"] = mie["forward_m11_ratio"]
    return record


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def format_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def write_markdown(
    path: Path, records: list[dict[str, Any]], comparisons: list[dict[str, Any]],
) -> None:
    lines = [
        "# Adaptive orientation profile validation",
        "",
        "Built-in shapes use automatic refractive-index-aware surface refinement; "
        "the OBJ control uses two explicit refinements. All cases use 64 alpha samples.",
        "`quick` is the fast exploratory profile; `standard` is the normal accuracy profile.",
        "",
        "## Runs",
        "",
        "| Case | Profile | status | ref | shortest PPW | DOFs | J | unique solves/full nodes | residual | wall, s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        wall = format_value(record["wall_time_s"])
        if record["timing_scope"] != "complete":
            wall += "*"
        lines.append(
            f"| {record['case']} | {record['profile']} | "
            f"{'PASS' if record['validation_passed'] else 'FAIL'} | {record['ref']} | "
            f"{format_value(record['points_per_shortest_wavelength'])} | "
            f"{record['system_dofs']} | {record['adaptive_accepted_level']} | "
            f"{record['solved_base_orientations']}/{record['maximum_base_orientations']} | "
            f"{format_value(record['maximum_residual'])} | "
            f"{wall} |"
        )
    if any(record["timing_scope"] != "complete" for record in records):
        lines += [
            "",
            "`*` Incremental continuation time after validated nested orientation "
            "samples were reused; excluded from runtime plots.",
        ]
    lines += [
        "",
        "## Quick versus standard",
        "",
        "| Case | normalized Mueller L2 | maximum normalized difference | forward M11 difference |",
        "|---|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['case']} | {format_value(item.get('normalized_relative_l2'))} | "
            f"{format_value(item.get('maximum_normalized_absolute_difference'))} | "
            f"{format_value(item.get('forward_relative_difference'))} |"
        )
    lines += [
        "",
        "## Independent Mie controls",
        "",
        "| Case | Profile | normalized full-Mueller L2 | forward M11 ratio BEM/Mie |",
        "|---|---|---:|---:|",
    ]
    for record in records:
        if "mie_normalized_mueller_l2" not in record:
            continue
        lines.append(
            f"| {record['case']} | {record['profile']} | "
            f"{format_value(record['mie_normalized_mueller_l2'])} | "
            f"{format_value(record['mie_forward_m11_ratio'], 6)} |"
        )
    failed = [record for record in records if not record["validation_passed"]]
    completed_keys = {(record["case"], record["profile"]) for record in records}
    missing = [
        (case_name, profile)
        for case_name in CASES
        for profile in PROFILES
        if (case_name, profile) not in completed_keys
    ]
    if failed or missing:
        lines += ["", "## Failed stress controls", ""]
        for record in failed:
            reason = record["validation_errors"] or "profile acceptance criterion failed"
            lines.append(
                f"- `{record['case']}/{record['profile']}`: {reason}; "
                f"residual={format_value(record['maximum_residual'])}, "
                f"accepted J={record['adaptive_accepted_level']}."
            )
        for case_name, profile in missing:
            lines.append(
                f"- `{case_name}/{profile}`: no completed result; any partial "
                "checkpoint is diagnostic only and is not used in comparisons."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(
    path: Path, records: list[dict[str, Any]], comparisons: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy unavailable; report PNG was not written", file=sys.stderr)
        return

    case_names = [name for name in CASES if any(r["case"] == name for r in records)]
    positions = np.arange(len(case_names))
    width = 0.36
    figure, axes_grid = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    axes = axes_grid.ravel()
    for offset, profile, color in ((-width / 2, "quick", "#4c78a8"),
                                   (width / 2, "standard", "#f58518")):
        values = []
        savings = []
        for name in case_names:
            record = next(
                (r for r in records if r["case"] == name and r["profile"] == profile),
                None,
            )
            values.append(
                record["wall_time_s"]
                if (record and record["wall_time_s"] is not None
                    and record["timing_scope"] == "complete")
                else math.nan
            )
            savings.append(
                100.0 * record["orientation_savings_fraction"]
                if record and record["orientation_savings_fraction"] is not None
                else math.nan
            )
        axes[0].bar(positions + offset, values, width, label=profile, color=color)
        axes[2].bar(positions + offset, savings, width, label=profile, color=color)

    comparison_by_case = {item["case"]: item for item in comparisons}
    differences = [
        100.0 * comparison_by_case[name]["normalized_relative_l2"]
        if name in comparison_by_case else math.nan
        for name in case_names
    ]
    axes[1].bar(positions, differences, color="#54a24b")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Complete wall time, s")
    axes[0].set_title("Adaptive orientation averaging: runtime")
    axes[0].legend()
    axes[1].set_ylabel("Quick vs standard Mueller L2, %")
    axes[1].set_title("Cross-profile numerical difference")
    axes[2].set_ylabel("Avoided maximum-grid orientations, %")
    axes[2].set_title("Savings from adaptive stopping and exact symmetry")
    axes[2].legend()
    axes[2].set_ylim(0.0, 100.0)
    for axis in axes[:3]:
        axis.set_xticks(positions)
        axis.set_xticklabels(case_names, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)

    for profile, marker, color in (
        ("quick", "o", "#4c78a8"),
        ("standard", "s", "#f58518"),
    ):
        selected = [
            record for record in records
            if record["profile"] == profile and record["shape"] != "obj"
        ]
        axes[3].scatter(
            [record["ka"] * max(1.0, abs(record["ri"])) for record in selected],
            [record["ref"] for record in selected],
            marker=marker, color=color, s=55, label=profile,
        )
    axes[3].set_xlabel(r"$ka\,\max(1, |m|)$")
    axes[3].set_ylabel("Automatically selected ref")
    axes[3].set_title("Mesh refinement versus size and refractive index")
    axes[3].grid(alpha=0.25)
    axes[3].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def report_matrix(args: argparse.Namespace) -> int:
    output_root = Path(args.out_root).expanduser().resolve()
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for case_name in CASES:
        for profile in PROFILES:
            record = record_for(
                case_name, profile, output_for(output_root, case_name, profile)
            )
            if record:
                records.append(record)
        quick = output_for(output_root, case_name, "quick") / "result.json"
        standard = output_for(output_root, case_name, "standard") / "result.json"
        if (
            quick.is_file() and standard.is_file()
            and result_is_valid(quick.parent)
            and result_is_valid(standard.parent)
        ):
            item = comparison(quick, standard)
            if item:
                item["case"] = case_name
                comparisons.append(item)
    output_root.mkdir(parents=True, exist_ok=True)
    document = {"cases": CASES, "runs": records, "comparisons": comparisons}
    (output_root / "summary.json").write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )
    write_csv(output_root / "summary.csv", records)
    write_markdown(output_root / "report.md", records, comparisons)
    write_plot(output_root / "orientation_profile_validation.png", records, comparisons)
    print(f"Wrote {output_root / 'report.md'}")
    passed = sum(record["validation_passed"] for record in records)
    print(
        f"Collected {len(records)} runs ({passed} passed) and "
        f"{len(comparisons)} comparisons"
    )
    complete = len(records) == len(CASES) * len(PROFILES)
    valid = complete and all(record["validation_passed"] for record in records)
    return 0 if valid else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run or resume the validation matrix")
    run.add_argument("--out-root", default=str(DEFAULT_ROOT))
    run.add_argument("--case", action="append", choices=tuple(CASES))
    run.add_argument("--profile", action="append", choices=PROFILES)
    run.add_argument("--alpha", type=int, default=64)
    run.add_argument("--force", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--keep-going", action="store_true")
    run.set_defaults(handler=run_matrix)
    report = subparsers.add_parser("report", help="build JSON, CSV, Markdown, and PNG reports")
    report.add_argument("--out-root", default=str(DEFAULT_ROOT))
    report.set_defaults(handler=report_matrix)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
