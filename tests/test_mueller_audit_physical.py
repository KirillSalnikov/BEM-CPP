#!/usr/bin/env python3
"""Regression tests for physical Mueller-matrix audit gates."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def write_mueller(path: Path, m: np.ndarray) -> None:
    path.write_text(json.dumps({"theta": [0.0], "mueller": m.tolist()}))


def run_audit(path: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "mueller_audit.py"), "--bem", str(path), *extra],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        valid = np.zeros((4, 4, 1), dtype=float)
        for i in range(4):
            valid[i, i, 0] = 1.0
        valid_path = tmp / "valid.json"
        write_mueller(valid_path, valid)

        ok = run_audit(valid_path, "--require-cloude-physical")
        assert ok.returncode == 0, ok.stderr + ok.stdout
        report = json.loads(ok.stdout)
        assert report["max_polarizance"] == 0.0
        assert report["max_diattenuation"] == 0.0
        assert report["min_cloude_eigenvalue"] >= -1e-8

        bad_pol = valid.copy()
        bad_pol[1, 0, 0] = 1.5
        bad_pol_path = tmp / "bad_pol.json"
        write_mueller(bad_pol_path, bad_pol)
        failed = run_audit(bad_pol_path)
        assert failed.returncode == 2, failed.stdout
        report = json.loads(failed.stdout)
        assert report["max_polarizance"] > 1.0

        bad_diattenuation = valid.copy()
        bad_diattenuation[0, 1, 0] = 1.5
        bad_diattenuation_path = tmp / "bad_diattenuation.json"
        write_mueller(bad_diattenuation_path, bad_diattenuation)
        failed = run_audit(bad_diattenuation_path)
        assert failed.returncode == 2, failed.stdout
        report = json.loads(failed.stdout)
        assert report["max_diattenuation"] > 1.0

        bad_bound = valid.copy()
        bad_bound[2, 2, 0] = 1.1
        bad_bound_path = tmp / "bad_bound.json"
        write_mueller(bad_bound_path, bad_bound)
        failed = run_audit(bad_bound_path)
        assert failed.returncode == 2, failed.stdout
        report = json.loads(failed.stdout)
        assert report["max_abs_over_m11_all"] > 1.0

    print("mueller audit physical gates: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
