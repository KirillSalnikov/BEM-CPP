#!/usr/bin/env python3
"""Smoke-test full-Mueller Mie metrics in sphere RI sweep summary."""

import csv
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    import sys
    sys.path.insert(0, str(ROOT))
    from verify_mie import mie_mueller  # noqa: E402

    theta = [0.0, 30.0, 90.0, 150.0, 180.0]
    ka = 5.0
    n_re = 1.3116
    mu = mie_mueller(theta, complex(n_re, 0.0), ka)
    norm = max(abs(mu[0][0][0]), 1.0)
    for t in range(len(theta)):
        mu[2][3][t] += 0.3 * norm

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        result = {
            "theta": theta,
            "mueller": mu,
            "ka": ka,
            "ri": [n_re, 0.0],
            "refinements": 3,
            "timing": {"total_s": 1.0, "solve_s": 0.5},
            "gmres_matvecs": 10,
            "gmres_nonconverged_systems": 0,
            "gmres_max_final_relres": 1e-4,
        }
        (out_dir / "sphere_n1p3116_ref3.json").write_text(json.dumps(result))
        proc = subprocess.run(
            ["python3", "scripts/summarize_sphere_ri_sweep.py", str(out_dir)],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout
        rows = list(csv.DictReader((out_dir / "summary_mie.csv").open()))
        assert len(rows) == 1, rows
        row = rows[0]
        assert row["worst_component"] == "M34", row
        assert row["pass10_shape_l2"] == "True", row
        assert row["pass10_full_mueller"] == "False", row
        assert "M34" in row["failed_all_20pct"], row

    print("summarize sphere ri sweep: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
