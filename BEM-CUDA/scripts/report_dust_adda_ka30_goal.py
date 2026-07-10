#!/usr/bin/env python3
"""Build an auditable status table for the ka>=30 ADDA comparison goal."""

import argparse
import csv
import json
from pathlib import Path


def load_json(path, default=None):
    try:
        with path.open() as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return default


def level_metrics(case_dir, accepted):
    report = load_json(case_dir / "level_m11_vs_adda.json", {}) or {}
    accepted_path = Path(accepted).resolve() if accepted else None
    for level_name, record in report.get("levels", {}).items():
        bem = Path(record.get("bem", "")).resolve()
        if accepted_path is not None and bem == accepted_path:
            return level_name, record
    return None, None


def case_record(run_root, material, case, max_l2):
    tag = str(case["ka"]).replace(".", "p")
    case_dir = run_root / material["directory"] / ("ka" + tag)
    manifest = load_json(case_dir / "adaptive_nested_bg_manifest.json", {}) or {}
    accepted = manifest.get("accepted")
    level_name, metric_record = level_metrics(case_dir, accepted)
    metrics = metric_record.get("metrics", {}) if metric_record else {}
    l2 = metrics.get("m11_integral_rel_l2")
    grid_ok = bool(metric_record and metric_record.get("angular_grid_resolves_reference"))
    converged = manifest.get("converged") is True
    l2_ok = isinstance(l2, (int, float)) and l2 <= max_l2
    passed = converged and grid_ok and l2_ok
    if passed:
        status = "pass"
    elif not case_dir.exists():
        status = "missing"
    elif not accepted:
        status = "running_or_incomplete"
    elif not converged:
        status = "orientation_not_converged"
    elif not grid_ok:
        status = "angular_grid_not_final"
    elif not l2_ok:
        status = "m11_l2_fail"
    else:
        status = "metric_missing"
    return {
        "material": material["directory"],
        "ri_real": material["ri"][0],
        "ri_imag": material["ri"][1],
        "ka": case["ka"],
        "mesh": case["mesh"],
        "triangles": case["triangles"],
        "case_dir": str(case_dir),
        "status": status,
        "passed": passed,
        "orientation_converged": converged,
        "accepted_level": manifest.get("accepted_level"),
        "accepted_active_count": manifest.get("accepted_active_count"),
        "metric_level": level_name,
        "angular_grid_final": grid_ok,
        "m11_weighted_l2": l2,
        "raw_integral_ratio": metrics.get("raw_integral_ratio"),
        "total_s": metrics.get("total_s"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()
    matrix = load_json(args.matrix)
    if not matrix:
        raise SystemExit("cannot read case matrix: {}".format(args.matrix))
    max_l2 = float(matrix["metric_limit"])
    records = [
        case_record(args.run_root, material, case, max_l2)
        for material in matrix["materials"]
        for case in matrix["cases"]
    ]
    summary = {
        "objective": "all cases converged with final angular grid and weighted M11 L2 <= {}".format(max_l2),
        "expected_cases": len(records),
        "passed_cases": sum(record["passed"] for record in records),
        "complete": bool(records) and all(record["passed"] for record in records),
        "records": records,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_json.with_suffix(args.out_json.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    tmp.replace(args.out_json)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("passed={}/{} complete={}".format(
        summary["passed_cases"], summary["expected_cases"], summary["complete"]
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
