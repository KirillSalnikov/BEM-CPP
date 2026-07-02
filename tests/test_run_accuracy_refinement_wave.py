#!/usr/bin/env python3
"""Smoke-tests for the accuracy refinement wave wrapper."""

import csv
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_accuracy_refinement_wave.sh"
PRODUCTION_CSV = ROOT / "poster_a0" / "assets" / "table_accuracy_matrix_15.csv"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(SCRIPT), *args],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        plan_csv = Path(tmp) / "plan.csv"
        proc = run(
            "--csv", str(PRODUCTION_CSV),
            "--gpus", "0 1 2 3",
            "--max-cases", "4",
            "--no-health-check",
            "--out", str(out_dir),
            "--plan-csv", str(plan_csv),
        )
        assert proc.returncode == 0, proc.stdout
        assert "REFINEMENT_WAVE mode=dry-run gpus=0 1 2 3" in proc.stdout, proc.stdout
        assert "REFINE threshold=0.1 reason=all planned=4 limit=4" in proc.stdout, proc.stdout
        assert "--max-jobs 4" in proc.stdout, proc.stdout
        assert "STARTED" not in proc.stdout, proc.stdout
        rows = list(csv.DictReader(plan_csv.open()))
        assert len(rows) == 4, rows
        assert len({row["case_name"] for row in rows}) == 4, rows
        assert rows[0]["case_name"] == "dust_ka30_gmsh7000_balanced_q9_d6_tol5e4", rows

        oversub = run(
            "--csv", str(PRODUCTION_CSV),
            "--gpus", "0 1",
            "--max-cases", "2",
            "--no-health-check",
            "--allow-oversubscribe",
            "--out", str(out_dir),
        )
        assert oversub.returncode == 0, oversub.stdout
        assert "REFINEMENT_WAVE mode=dry-run gpus=0 1" in oversub.stdout, oversub.stdout
        assert "REFINE threshold=0.1 reason=all planned=2 limit=2" in oversub.stdout, oversub.stdout
        assert "oversubscribe_command:" in oversub.stdout, oversub.stdout
        assert "--allow-oversubscribe" in oversub.stdout, oversub.stdout
        assert "--run --execute" not in oversub.stdout, oversub.stdout
        assert "STARTED" not in oversub.stdout, oversub.stdout

    print("accuracy refinement wave: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
