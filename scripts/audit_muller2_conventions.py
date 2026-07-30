#!/usr/bin/env python3
"""Audit experimental RWG Muller sign/order conventions against PMCHWT."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any


def flatten(values: Any) -> list[float]:
    if isinstance(values, list):
        result: list[float] = []
        for value in values:
            result.extend(flatten(value))
        return result
    return [float(values)]


def relative_l2(reference: list[float], candidate: list[float]) -> float:
    numerator = sum((a - b) ** 2 for a, b in zip(reference, candidate))
    denominator = sum(a * a for a in reference)
    return math.sqrt(numerator / max(denominator, 1.0e-300))


def run_case(
    binary: Path,
    output: Path,
    system: str,
    rhs_mode: int | None,
    farfield_mode: int | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command = [
        str(binary),
        "--shape",
        args.shape,
        "--ref",
        str(args.ref),
        "--ka",
        str(args.ka),
        "--ri",
        str(args.n_re),
        str(args.n_im),
        "--single",
        "--ntheta",
        str(args.ntheta),
        "--fmm",
        "--fmm-digits",
        str(args.fmm_digits),
        "--gmres-tol",
        str(args.tol),
        "--gmres-restart",
        str(args.restart),
        "--system",
        system,
        "--no-prec",
        "--out",
        str(output),
    ]
    env = os.environ.copy()
    if rhs_mode is not None:
        env["BEM_EXPERIMENTAL_NFORM"] = "1"
        env["BEM_NFORM_RHS_MODE"] = str(rhs_mode)
    if farfield_mode is not None:
        env["BEM_NFORM_FF_MODE"] = str(farfield_mode)

    completed = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=args.timeout,
        check=False,
    )
    log_path = output.with_suffix(".log")
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or not output.exists():
        raise RuntimeError(
            f"solver failed for RHS={rhs_mode}, FF={farfield_mode}; "
            f"see {log_path}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=Path, default=Path("bin/bem_cuda"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/muller2_audit"))
    parser.add_argument("--shape", default="hex_prism")
    parser.add_argument("--ref", type=int, default=2)
    parser.add_argument("--ka", type=float, default=3.0)
    parser.add_argument("--n-re", type=float, default=1.5)
    parser.add_argument("--n-im", type=float, default=0.0)
    parser.add_argument("--ntheta", type=int, default=19)
    parser.add_argument("--fmm-digits", type=int, default=6)
    parser.add_argument("--tol", type=float, default=1.0e-5)
    parser.add_argument("--restart", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    binary = args.bin.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = out_dir / "baseline_balanced.json"
    baseline = run_case(binary, baseline_path, "balanced", None, None, args)
    baseline_mueller = flatten(baseline["mueller"])

    rows: list[dict[str, Any]] = []
    for rhs_mode in range(8):
        for farfield_mode in range(6):
            output = out_dir / f"muller2_rhs{rhs_mode}_ff{farfield_mode}.json"
            candidate = run_case(
                binary,
                output,
                "muller2-balanced",
                rhs_mode,
                farfield_mode,
                args,
            )
            candidate_mueller = flatten(candidate["mueller"])
            rows.append(
                {
                    "rhs_mode": rhs_mode,
                    "farfield_mode": farfield_mode,
                    "relative_l2_all_mueller": relative_l2(
                        baseline_mueller, candidate_mueller
                    ),
                    "max_final_relative_residual": candidate[
                        "gmres_max_final_relres"
                    ],
                    "full_operator_actions": candidate["gmres_matvecs"],
                    "solve_s": candidate["timing"]["solve_s"],
                    "total_s": candidate["timing"]["total_s"],
                    "output": str(output),
                }
            )

    rows.sort(key=lambda row: row["relative_l2_all_mueller"])
    csv_path = out_dir / "convention_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "configuration": {
            "shape": args.shape,
            "ref": args.ref,
            "ka": args.ka,
            "refractive_index": [args.n_re, args.n_im],
            "ntheta": args.ntheta,
            "fmm_digits": args.fmm_digits,
            "gmres_tolerance": args.tol,
        },
        "baseline": str(baseline_path),
        "best": rows[0],
        "top_five": rows[:5],
        "all_sign_and_order_conventions_fail_5e-3": (
            rows[0]["relative_l2_all_mueller"] > 5.0e-3
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
