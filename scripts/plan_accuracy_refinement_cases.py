#!/usr/bin/env python3
"""Plan BEM reruns for accuracy-matrix rows that are not yet audit-clean.

The script consumes the CSV written by ``audit_accuracy_matrix_15.py`` and
prints parameterized case names accepted by ``run_accuracy_matrix_case.sh``.
It is deliberately conservative: if a row is numerically within the gate but
is marked STALE/legacy/invalid by provenance checks, it reruns the same
numerical level. It raises numerical parameters only when the accuracy gate
itself failed.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "poster_a0/assets/table_accuracy_matrix_15.csv"


SHAPE_MAP = {
    "сфера": "sphere",
    "sphere": "sphere",
    "гексагональная призма": "hex",
    "hex": "hex",
    "hex_prism": "hex",
    "пылевая частица": "dust",
    "dust": "dust",
}

DUST_MESH_ORDER = ["gmsh3400", "gmsh4200", "gmsh5200", "gmsh6000", "gmsh7000"]
DUST_ACCURATE_SUFFIX = "balanced_q7_d6_tol5e4"
DUST_STRICT_SUFFIX = "balanced_q13_d6_tol5e4"


def finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def truthy(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "pass"}


def has_explicit_value(row: Dict[str, str], key: str) -> bool:
    value = row.get(key)
    return value is not None and str(value).strip() != ""


def parse_ref(mesh_label: str, default: int) -> int:
    match = re.search(r"ref([0-9]+)", mesh_label or "")
    if match:
        return int(match.group(1))
    return default


def parse_gmsh(mesh_label: str) -> Optional[str]:
    match = re.search(r"gmsh[0-9]+", mesh_label or "")
    if match:
        return match.group(0)
    return None


def next_dust_mesh(mesh_label: str, ka: int) -> str:
    current = parse_gmsh(mesh_label)
    if current in DUST_MESH_ORDER:
        idx = DUST_MESH_ORDER.index(current)
        return DUST_MESH_ORDER[min(idx + 1, len(DUST_MESH_ORDER) - 1)]
    if ka <= 5:
        return "gmsh5200"
    if ka <= 10:
        return "gmsh6000"
    return "gmsh7000"


def accuracy_bad(row: Dict[str, str], threshold: float) -> bool:
    status = (row.get("status") or "").upper()
    if status in {"MISSING", "FAIL"}:
        return True
    if (row.get("failed_all_20pct") or "").strip():
        return True
    gate = finite_float(row.get("gate_error"))
    if gate is None or gate > threshold:
        return True
    if has_explicit_value(row, "raw_pass10"):
        return not truthy(row.get("raw_pass10"))
    if has_explicit_value(row, "pass10"):
        return not truthy(row.get("pass10"))
    return True


def metadata_bad(row: Dict[str, str]) -> bool:
    if (row.get("metadata_status") or "missing") != "ok":
        return True
    if (row.get("operator_status") or "missing") not in {"complex_operator", "not_required"}:
        return True
    return False


def needs_rerun(row: Dict[str, str], threshold: float) -> bool:
    return accuracy_bad(row, threshold) or metadata_bad(row)


def same_level_candidate_name(row: Dict[str, str]) -> Optional[str]:
    shape = SHAPE_MAP.get((row.get("shape") or "").strip().lower())
    ka_f = finite_float(row.get("ka"))
    if shape is None or ka_f is None:
        return None
    ka = int(round(ka_f))
    label = row.get("mesh_label") or ""

    if shape == "sphere":
        current_ref = parse_ref(label, 4 if ka <= 20 else 6)
        return f"sphere_ka{ka}_ref{current_ref}_current_q7_d6_tol3e3"

    if shape == "hex":
        current_ref = parse_ref(label, 2 if ka <= 5 else 3 if ka <= 10 else 4 if ka <= 20 else 5)
        return f"hex_ka{ka}_ref{current_ref}_balanced_q7_d5_tol1e3"

    if shape == "dust":
        mesh = parse_gmsh(label) or next_dust_mesh(label, ka)
        return f"dust_ka{ka}_{mesh}_{DUST_ACCURATE_SUFFIX}"

    return None


def stricter_candidate_name(row: Dict[str, str]) -> Optional[str]:
    shape = SHAPE_MAP.get((row.get("shape") or "").strip().lower())
    ka_f = finite_float(row.get("ka"))
    if shape is None or ka_f is None:
        return None
    ka = int(round(ka_f))
    label = row.get("mesh_label") or ""

    if shape == "sphere":
        current_ref = parse_ref(label, 4 if ka <= 20 else 6)
        ref = min(current_ref + 1, 8)
        return f"sphere_ka{ka}_ref{ref}_current_q13_d7_tol1e3"

    if shape == "hex":
        current_ref = parse_ref(label, 2 if ka <= 5 else 3 if ka <= 10 else 4 if ka <= 20 else 5)
        ref = min(current_ref + 1, 7)
        return f"hex_ka{ka}_ref{ref}_balanced_q13_d6_tol5e4"

    if shape == "dust":
        mesh = next_dust_mesh(label, ka)
        return f"dust_ka{ka}_{mesh}_{DUST_STRICT_SUFFIX}"

    return None


def candidate_name(row: Dict[str, str], threshold: float) -> Optional[str]:
    if accuracy_bad(row, threshold):
        return stricter_candidate_name(row)
    if metadata_bad(row):
        return same_level_candidate_name(row)
    return None


def rerun_reason(row: Dict[str, str], threshold: float) -> str:
    reasons = []
    if accuracy_bad(row, threshold):
        reasons.append("accuracy")
    if metadata_bad(row):
        reasons.append("metadata")
    return "+".join(reasons) or "none"


def reason_allowed(reason: str, only_reason: str) -> bool:
    if only_reason == "all":
        return reason != "none"
    parts = {part for part in reason.split("+") if part}
    if only_reason == "accuracy":
        return "accuracy" in parts
    if only_reason == "metadata":
        return parts == {"metadata"}
    return False


def priority_key(row: Dict[str, str], threshold: float) -> tuple:
    gate = finite_float(row.get("gate_error"))
    ka = finite_float(row.get("ka")) or 0.0
    bad_accuracy = accuracy_bad(row, threshold)
    bad_metadata = metadata_bad(row)
    gate_score = -1.0 if gate is None else gate
    return (
        0 if bad_accuracy else 1 if bad_metadata else 2,
        -gate_score,
        -ka,
    )


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def gpu_compute_apps(nvidia_smi: str, gpu: str) -> str:
    proc = subprocess.run([
        nvidia_smi,
        "-i",
        gpu,
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return ""
    return "\n".join(line.strip() for line in proc.stdout.splitlines() if line.strip())


def parse_gpu_count(gpus: str, *, health_check: bool = True,
                    allow_compute_share: bool = False) -> int:
    if gpus.strip().lower() == "auto":
        nvidia_smi = os.environ.get("BEM_NVIDIA_SMI", "nvidia-smi")
        if shutil.which(nvidia_smi) is None and not Path(nvidia_smi).is_file():
            raise RuntimeError(f"--gpus auto requires {nvidia_smi}")
        proc = subprocess.run([
            nvidia_smi,
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"--gpus auto failed: {proc.stderr.strip() or proc.stdout.strip()}")
        gpu_ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not health_check or allow_compute_share:
            return len(gpu_ids)
        return sum(1 for gpu in gpu_ids if not gpu_compute_apps(nvidia_smi, gpu))
    normalized = gpus.replace(",", " ")
    items = [item for item in normalized.split() if item]
    return len(items)


def build_resume_command(cases: Iterable[str], args: argparse.Namespace,
                         case_limit: int) -> List[str]:
    case_list = list(cases)
    cmd = [
        "scripts/resume_accuracy_matrix_cases.sh",
        "--gpus",
        args.gpus,
        "--cases",
        ",".join(case_list),
        "--case-max-power",
        str(args.case_max_power),
        "--case-max-bad-samples",
        str(args.case_max_bad_samples),
    ]
    if case_limit > 0:
        cmd.extend(["--max-jobs", str(case_limit)])
    elif len(case_list) > 0:
        cmd.extend(["--max-jobs", str(len(case_list))])
    if args.run:
        cmd.insert(1, "--run")
    if args.out:
        cmd.extend(["--out", args.out])
    if args.no_health_check:
        cmd.append("--no-health-check")
    if args.allow_compute_share:
        cmd.append("--allow-compute-share")
    return cmd


def validate_case_names(cases: Sequence[str], out_dir: str) -> List[str]:
    errors: List[str] = []
    for name in cases:
        proc = subprocess.run([
            str(ROOT / "scripts/run_accuracy_matrix_case.sh"),
            "--gpu", "0",
            "--case", name,
            "--out", out_dir,
            "--print",
        ], cwd=str(ROOT), text=True, stdout=subprocess.PIPE,
           stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            errors.append(f"{name}: {proc.stdout.strip()}")
    return errors


def write_plan_csv(path: Path, planned_rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "case_name",
        "reason",
        "shape",
        "ka",
        "mesh_label",
        "status",
        "gate_error",
        "worst_component",
        "worst_component_error",
        "failed_main_10pct",
        "failed_all_20pct",
        "metadata_status",
        "operator_status",
        "source_bem_file",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in planned_rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--refresh-audit", action="store_true",
                        help="run audit_accuracy_matrix_15.py before reading the CSV")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--only-reason", choices=["all", "accuracy", "metadata"], default="all",
                        help="restrict planned reruns: all, accuracy failures, or metadata-only refreshes")
    parser.add_argument("--gpus", default="0 1 2")
    parser.add_argument("--out", default="runs/production_matrix_refinement")
    parser.add_argument("--plan-csv", type=Path,
                        help="write selected rerun cases and reasons to this CSV "
                             "(default: OUT/plan.csv)")
    parser.add_argument("--no-plan-csv", action="store_true",
                        help="do not write a plan CSV")
    parser.add_argument("--case-max-power", type=int, default=290)
    parser.add_argument("--case-max-bad-samples", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=None,
                        help="limit printed cases; default is the number of GPUs in --gpus")
    parser.add_argument("--all-cases", action="store_true",
                        help="print every planned case instead of limiting to available GPUs")
    parser.add_argument("--no-health-check", action="store_true")
    parser.add_argument("--allow-compute-share", action="store_true",
                        help="allow refinement resume command to use GPUs with existing CUDA compute apps")
    parser.add_argument("--no-validate-cases", action="store_true",
                        help="do not verify planned names with run_accuracy_matrix_case.sh --print")
    parser.add_argument("--run", action="store_true",
                        help="print a --run resume command instead of a dry-run command")
    parser.add_argument("--execute", action="store_true",
                        help="execute the generated resume command")
    args = parser.parse_args()

    if args.refresh_audit:
        subprocess.run([
            str(ROOT / "scripts/audit_accuracy_matrix_15.py"),
            "--out",
            str(args.csv),
        ], cwd=str(ROOT), check=False)

    if not args.csv.is_file():
        raise SystemExit(f"accuracy CSV not found: {args.csv}")

    if args.max_cases is not None and args.max_cases < 0:
        raise SystemExit("--max-cases must be non-negative")
    if args.all_cases:
        case_limit = 0
    elif args.max_cases is not None:
        case_limit = args.max_cases
    else:
        try:
            case_limit = parse_gpu_count(
                args.gpus,
                health_check=not args.no_health_check,
                allow_compute_share=args.allow_compute_share,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        if case_limit <= 0:
            raise SystemExit("--gpus does not contain any GPU ids")

    rows = read_rows(args.csv)
    planned: List[str] = []
    planned_rows: List[Dict[str, str]] = []
    seen = set()
    selected_rows = [
        row for row in rows
        if needs_rerun(row, args.threshold)
        and reason_allowed(rerun_reason(row, args.threshold), args.only_reason)
    ]
    selected_rows.sort(key=lambda row: priority_key(row, args.threshold))
    for row in selected_rows:
        reason = rerun_reason(row, args.threshold)
        if not needs_rerun(row, args.threshold) or not reason_allowed(reason, args.only_reason):
            continue
        name = candidate_name(row, args.threshold)
        if name is None or name in seen:
            continue
        planned.append(name)
        planned_rows.append({
            "case_name": name,
            "reason": reason,
            "shape": row.get("shape", ""),
            "ka": row.get("ka", ""),
            "mesh_label": row.get("mesh_label", ""),
            "status": row.get("status", ""),
            "gate_error": row.get("gate_error", ""),
            "worst_component": row.get("worst_component", ""),
            "worst_component_error": row.get("worst_component_error", ""),
            "failed_main_10pct": row.get("failed_main_10pct", ""),
            "failed_all_20pct": row.get("failed_all_20pct", ""),
            "metadata_status": row.get("metadata_status", ""),
            "operator_status": row.get("operator_status", ""),
            "source_bem_file": row.get("bem_file", ""),
        })
        seen.add(name)
        if case_limit > 0 and len(planned) >= case_limit:
            break

    limit_label = "all" if case_limit == 0 else str(case_limit)
    print(f"REFINE threshold={args.threshold:g} reason={args.only_reason} planned={len(planned)} limit={limit_label}")
    for name in planned:
        print(name)
    if not planned:
        return 0
    if not args.no_validate_cases:
        validation_errors = validate_case_names(planned, args.out)
        if validation_errors:
            print("\ncase validation failed:")
            for error in validation_errors:
                print(error)
            return 4

    plan_csv_path = None
    if not args.no_plan_csv:
        plan_csv_path = args.plan_csv or (Path(args.out) / "plan.csv")
        write_plan_csv(plan_csv_path, planned_rows)
        print(f"plan_csv={plan_csv_path}")

    cmd = build_resume_command(planned, args, case_limit)
    print("\ncommand:")
    print(" ".join(repr(part) if " " in part else part for part in cmd))
    if args.execute:
        return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
